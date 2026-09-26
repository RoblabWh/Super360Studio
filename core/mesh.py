"""Dreiecksnetz (Mesh) aus der Punktwolke: Geometrie, Farben je Ebene, CloudCompare.

Die Geometrie haengt nur an der Punktwolke, nicht an ihrer Einfaerbung. Sie
wird je Aufzeichnung einmal gerechnet und im Projekt abgelegt
(:func:`build_geometry`, :func:`save_geometry`); jede Farbebene (Onboard,
Maeander, Fusion …) und die Intensitaet kommen danach in Sekunden dazu
(:func:`vertex_colors`, :func:`vertex_scalar`).

Weg:
  1. Voxelraster ueber ALLE Punkte (auch ungefaerbte — die Form soll
     vollstaendig sein). Jeder Punkt merkt sich seine Zelle
     (``point_voxel``); darueber mittelt spaeter jede Farbebene je Zelle.
  2. Normalen je Zelle aus den Nachbarzellen im Radius ``_NORMAL_R`` Zellen
     (Hauptachsen der Kovarianz, kleinste = Normale). Auf der Grafikkarte per
     OpenCL (:data:`_KERNEL`), sonst mit Open3D. Danach zur naechsten
     Position der Flugbahn hin ausgerichtet: der Lidar hat jede Flaeche von
     der Seite gesehen, auf der die Drohne war. Ohne das zeigen Normalen
     zufaellig nach innen oder aussen und Poisson schliesst Fassaden zu Blasen.
  3. Poisson-Rekonstruktion (Open3D, CPU); Tiefe d = 2^d Zellen ueber die
     groesste Ausdehnung.
  4. Beschnitt: Poisson schliesst die Flaeche auch dort, wo nichts gemessen
     wurde. Alles weiter als ``max_gap`` Zellen von der naechsten Zelle faellt
     weg. Wahlweise auch die Ecken mit der geringsten Dichte (``trim``); das
     ist Standard 0, denn die duennsten Ecken liegen bei gleichmaessiger
     Abtastung verstreut ueber alle Flaechen und stanzen dort kleine Loecher
     (an Waenden des Nachtflugs sichtbar), waehrend der Abstand die
     ueberstehenden Poisson-Flaechen schon fast allein entfernt (am ganzen
     Flug 5 % Dichte-Beschnitt: nur ~1000 von 4,56 Mio. Dreiecken mehr weg).
  5. Jede Ecke merkt sich ihre ``_K_VOX`` naechsten Zellen; ihre Farbe ist
     das Mittel dieser Zellen, gewichtet mit der Zahl eingefaerbter Punkte.

CloudCompare wird gesucht als Programm im PATH (``CloudCompare``, Snap
``cloudcompare.CloudCompare``) oder als Flatpak ``org.cloudcompare.CloudCompare``;
``SUPER360_CLOUDCOMPARE`` ueberschreibt die Suche mit einem eigenen Befehl.

Qt-frei, kein print (ausser Selbsttest).
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import time
from typing import Callable, Optional

import numpy as np

__all__ = ["build_geometry", "save_geometry", "load_geometry", "geometry_dir",
           "vertex_colors", "vertex_scalar", "to_open3d", "build_mesh", "write_mesh",
           "find_cloudcompare", "open_in_cloudcompare", "opencl_available",
           "INSTALL_HINT"]

ProgressCb = Optional[Callable[[float, str], None]]

_FLATPAK_ID = "org.cloudcompare.CloudCompare"
INSTALL_HINT = ("CloudCompare nicht gefunden. Installieren ohne Root-Rechte:\n"
                f"  flatpak install --user flathub {_FLATPAK_ID}\n"
                "oder systemweit:\n"
                "  sudo snap install cloudcompare")

_NORMAL_R = 3      # Nachbarzellen je Richtung fuer die Normale (3 x 5 cm = 15 cm)
_K_VOX = 4         # Zellen je Ecke fuer die Farbe
_GEOM_VERSION = 1  # erhoehen, wenn sich das Verfahren aendert (alter Cache gilt nicht)
_GRAU = 90         # Ecken ohne eingefaerbte Nachbarn, wie ungefaerbte Punkte


def _p(cb: ProgressCb, f: float, msg: str) -> None:
    if cb is not None:
        cb(float(f), msg)


def _check(cancel) -> None:
    if cancel is not None and cancel():
        raise RuntimeError("Abgebrochen")


# ================================================================ Voxelraster

def _voxelize(points: np.ndarray, voxel: float):
    """(point_voxel int32 (N,), Gitterzellen int32 (M,3), Schluessel int64 (M,)
    aufsteigend, Mittelpunkte float64 (M,3), Anzahl je Zelle (M,), dims)."""
    P = np.asarray(points, dtype=np.float32)
    g = np.floor(P / np.float32(voxel)).astype(np.int64)
    g -= g.min(axis=0)
    dims = g.max(axis=0) + 1
    key = (g[:, 0] * dims[1] + g[:, 1]) * dims[2] + g[:, 2]
    order = np.argsort(key, kind="stable")
    ks = key[order]
    neu = np.empty(len(ks), bool)
    neu[0] = True
    np.not_equal(ks[1:], ks[:-1], out=neu[1:])
    inv_sorted = (np.cumsum(neu) - 1).astype(np.int32)
    point_voxel = np.empty(len(P), np.int32)
    point_voxel[order] = inv_sorted
    keys = ks[neu]
    cnt = np.bincount(point_voxel).astype(np.float64)
    cen = np.stack([np.bincount(point_voxel, weights=P[:, k]) / cnt for k in range(3)], axis=1)
    gz = keys % dims[2]
    gy = (keys // dims[2]) % dims[1]
    gx = keys // (dims[1] * dims[2])
    cells = np.stack([gx, gy, gz], axis=1).astype(np.int32)
    return point_voxel, cells, keys, cen, cnt, dims


# ================================================================ Normalen

_KERNEL = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

inline void jacobi3(double a[3][3], double v[3][3])
{
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) v[i][j] = (i == j);
    for (int sweep = 0; sweep < 12; ++sweep) {
        double off = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
        if (off < 1e-30) break;
        for (int p = 0; p < 2; ++p) for (int q = p + 1; q < 3; ++q) {
            if (fabs(a[p][q]) < 1e-300) continue;
            double th = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
            double t = (th >= 0 ? 1.0 : -1.0) / (fabs(th) + sqrt(th * th + 1.0));
            double c = 1.0 / sqrt(t * t + 1.0), s = t * c;
            for (int k = 0; k < 3; ++k) {
                double akp = a[k][p], akq = a[k][q];
                a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq;
            }
            for (int k = 0; k < 3; ++k) {
                double apk = a[p][k], aqk = a[q][k];
                a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk;
            }
            for (int k = 0; k < 3; ++k) {
                double vkp = v[k][p], vkq = v[k][q];
                v[k][p] = c * vkp - s * vkq; v[k][q] = s * vkp + c * vkq;
            }
        }
    }
}

__kernel void normals(__global const long* keys, const int m,
                      __global const int* cells, __global const double* cen,
                      const long dx_, const long dy_, const long dz_,
                      const int r, const double rad2,
                      __global float* out_n, __global int* out_k)
{
    const int i = get_global_id(0);
    if (i >= m) return;
    const int x = cells[3 * i], y = cells[3 * i + 1], z = cells[3 * i + 2];
    const double cx = cen[3 * i], cy = cen[3 * i + 1], cz = cen[3 * i + 2];
    double s[3] = {0, 0, 0}, ss[6] = {0, 0, 0, 0, 0, 0};
    int n = 0;
    for (int ax = -r; ax <= r; ++ax) {
        const long nx = x + ax;
        if (nx < 0 || nx >= dx_) continue;
        for (int ay = -r; ay <= r; ++ay) {
            const long ny = y + ay;
            if (ny < 0 || ny >= dy_) continue;
            const long base = (nx * dy_ + ny) * dz_;
            const long klo = base + max((long)z - r, 0L);
            const long khi = base + min((long)z + r, dz_ - 1);
            int lo = 0, hi = m;
            while (lo < hi) {
                const int mid = lo + (hi - lo) / 2;
                if (keys[mid] < klo) lo = mid + 1; else hi = mid;
            }
            for (int j = lo; j < m && keys[j] <= khi; ++j) {
                const double ex = cen[3 * j] - cx, ey = cen[3 * j + 1] - cy,
                             ez = cen[3 * j + 2] - cz;
                if (ex * ex + ey * ey + ez * ez > rad2) continue;
                s[0] += ex; s[1] += ey; s[2] += ez;
                ss[0] += ex * ex; ss[1] += ex * ey; ss[2] += ex * ez;
                ss[3] += ey * ey; ss[4] += ey * ez; ss[5] += ez * ez;
                ++n;
            }
        }
    }
    out_k[i] = n;
    if (n < 3) {
        out_n[3 * i] = 0.0f; out_n[3 * i + 1] = 0.0f; out_n[3 * i + 2] = 1.0f;
        return;
    }
    const double mx = s[0] / n, my = s[1] / n, mz = s[2] / n;
    double a[3][3], v[3][3];
    a[0][0] = ss[0] / n - mx * mx; a[0][1] = ss[1] / n - mx * my; a[0][2] = ss[2] / n - mx * mz;
    a[1][1] = ss[3] / n - my * my; a[1][2] = ss[4] / n - my * mz; a[2][2] = ss[5] / n - mz * mz;
    a[1][0] = a[0][1]; a[2][0] = a[0][2]; a[2][1] = a[1][2];
    jacobi3(a, v);
    int k = 0;
    if (a[1][1] < a[k][k]) k = 1;
    if (a[2][2] < a[k][k]) k = 2;
    out_n[3 * i] = (float)v[0][k]; out_n[3 * i + 1] = (float)v[1][k];
    out_n[3 * i + 2] = (float)v[2][k];
}
"""


def opencl_available() -> str | None:
    """Name der OpenCL-GPU (dieselbe wie fuer die Einfaerbung) oder None."""
    try:
        from . import colorizer_gpu
    except ImportError:  # pragma: no cover - Direktstart
        import colorizer_gpu  # type: ignore
    return colorizer_gpu.available()


def _normals_opencl(keys, cells, cen, dims, voxel) -> np.ndarray:
    import pyopencl as cl
    try:
        from . import colorizer_gpu
    except ImportError:  # pragma: no cover
        import colorizer_gpu  # type: ignore
    ctx, queue, _dev = colorizer_gpu._context()
    prg = cl.Program(ctx, _KERNEL).build()
    k = cl.Kernel(prg, "normals")
    m = len(keys)
    mf = cl.mem_flags
    bufs = [cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(a))
            for a in (keys.astype(np.int64), cells.astype(np.int32), cen.astype(np.float64))]
    out_n = np.empty((m, 3), np.float32)
    out_k = np.empty(m, np.int32)
    d_n = cl.Buffer(ctx, mf.WRITE_ONLY, out_n.nbytes)
    d_k = cl.Buffer(ctx, mf.WRITE_ONLY, out_k.nbytes)
    rad = (_NORMAL_R + 0.5) * voxel
    k(queue, (((m + 255) // 256) * 256,), (256,), bufs[0], np.int32(m), bufs[1], bufs[2],
      np.int64(dims[0]), np.int64(dims[1]), np.int64(dims[2]), np.int32(_NORMAL_R),
      np.float64(rad * rad), d_n, d_k)
    cl.enqueue_copy(queue, out_n, d_n)
    cl.enqueue_copy(queue, out_k, d_k)
    queue.finish()
    for b in bufs + [d_n, d_k]:
        b.release()
    return out_n


def _normals_cpu(cen, voxel) -> np.ndarray:
    import open3d as o3d
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cen))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=(_NORMAL_R + 0.5) * voxel, max_nn=60))
    return np.asarray(pc.normals, dtype=np.float32)


# ================================================================ Geometrie

def build_geometry(points: np.ndarray, sensor_path: np.ndarray | None,
                   voxel: float = 0.05, depth: int = 11, trim: float = 0.0,
                   max_gap: float = 3.0, backend: str = "auto",
                   progress: ProgressCb = None, cancel=None, log=None) -> dict:
    """Mesh-Geometrie einer Punktwolke (alle Punkte, ohne Farben).

    ``backend`` fuer die Normalen: "auto" (OpenCL, wenn da), "gpu", "cpu".
    Rueckgabe: dict mit vertices (V,3 f32), triangles (F,3 i32), normals
    (V,3 f32), vertex_voxel (V,K i32), point_voxel (N i32), stats.
    """
    import open3d as o3d
    from scipy.spatial import cKDTree

    t0 = time.time()
    say = log or (lambda _m: None)
    pts = np.asarray(points)
    if len(pts) < 100:
        raise RuntimeError(f"Zu wenige Punkte für ein Mesh ({len(pts)}).")

    _p(progress, 0.02, f"Mesh: Voxelraster {voxel * 100:g} cm …")
    point_voxel, cells, keys, cen, cnt, dims = _voxelize(pts, voxel)
    m = len(keys)
    say(f"Mesh: {len(pts):,} Punkte → {m:,} Zellen im {voxel * 100:g}-cm-Raster.")
    _check(cancel)

    gpu = None
    if backend != "cpu":
        gpu = opencl_available()
        if gpu is None and backend == "gpu":
            raise RuntimeError("Keine OpenCL-GPU für die Mesh-Normalen gefunden.")
    _p(progress, 0.15, f"Mesh: Normalen ({'GPU' if gpu else 'CPU'}) …")
    t1 = time.time()
    N = _normals_opencl(keys, cells, cen, dims, voxel) if gpu else _normals_cpu(cen, voxel)
    t_norm = time.time() - t1
    if sensor_path is not None and len(sensor_path):
        bahn = np.asarray(sensor_path, np.float64)
        _, k = cKDTree(bahn).query(cen, workers=-1)
        zum_sensor = bahn[k] - cen
    else:
        zum_sensor = cen - cen.mean(axis=0)
    flip = np.einsum("ij,ij->i", N, zum_sensor) < 0.0
    N[flip] *= -1.0
    _check(cancel)

    _p(progress, 0.3, f"Mesh: Poisson-Rekonstruktion (Tiefe {depth}) …")
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cen))
    pc.normals = o3d.utility.Vector3dVector(N.astype(np.float64))
    t1 = time.time()
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
        mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pc, depth=int(depth), n_threads=-1)
    t_poisson = time.time() - t1
    del pc
    dens = np.asarray(dens)
    n_roh = len(mesh.triangles)
    _check(cancel)

    _p(progress, 0.8, "Mesh: beschneiden, Ecken den Zellen zuordnen …")
    if trim > 0.0:
        mesh.remove_vertices_by_mask(dens < np.quantile(dens, float(trim)))
    V = np.asarray(mesh.vertices)
    dist, vv = cKDTree(cen).query(V, k=_K_VOX, workers=-1)
    weg = dist[:, 0] > float(max_gap) * float(voxel)
    mesh.remove_vertices_by_mask(weg)
    vv = vv[~weg]
    # remove_vertices_by_mask laesst die Reihenfolge der uebrigen Ecken
    # stehen, vv passt also weiter zu den Ecken
    mesh.compute_vertex_normals()
    V = np.asarray(mesh.vertices, dtype=np.float32)
    if len(V) != len(vv):
        # Sicherheitsnetz: falls Open3D Ecken doch umsortiert hat, neu zuordnen
        _, vv = cKDTree(cen).query(V, k=_K_VOX, workers=-1)
    geom = {
        "vertices": np.ascontiguousarray(V),
        "triangles": np.ascontiguousarray(np.asarray(mesh.triangles, dtype=np.int32)),
        "normals": np.ascontiguousarray(np.asarray(mesh.vertex_normals, dtype=np.float32)),
        "vertex_voxel": np.ascontiguousarray(vv.astype(np.int32)),
        "point_voxel": point_voxel,
    }
    geom["stats"] = {
        "version": _GEOM_VERSION, "punkte": int(len(pts)), "zellen": int(m),
        "voxel_m": float(voxel), "tiefe": int(depth), "trim": float(trim),
        "max_gap_voxel": float(max_gap), "dreiecke_roh": int(n_roh),
        "dreiecke": int(len(geom["triangles"])), "ecken": int(len(V)),
        "normalen": "gpu" if gpu else "cpu", "gpu": gpu,
        "normalen_s": round(t_norm, 1), "poisson_s": round(t_poisson, 1),
        "laufzeit_s": round(time.time() - t0, 1)}
    say(f"Mesh: {len(geom['triangles']):,} Dreiecke (roh {n_roh:,}); Normalen "
        f"{'GPU' if gpu else 'CPU'} {t_norm:.1f} s, Poisson {t_poisson:.1f} s, "
        f"gesamt {time.time() - t0:.1f} s.")
    _p(progress, 0.95, "Mesh fertig")
    return geom


def geometry_dir(project_dir: str, fingerprint: str, voxel: float, depth: int,
                 trim: float) -> str:
    """Ablage einer Geometrie im Projekt, eindeutig je Aufzeichnung und Parameter."""
    name = (f"v{_GEOM_VERSION}_{fingerprint}_{int(round(voxel * 1000))}mm_t{int(depth)}"
            f"_r{int(round(trim * 100))}")
    return os.path.join(project_dir, "mesh", "geometrie", name.replace("/", "_"))


def save_geometry(geom: dict, path: str) -> str:
    """Geometrie als .npy je Feld plus meta.json; atomar ueber ein .tmp-Verzeichnis."""
    tmp = path + ".tmp"
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    for k in ("vertices", "triangles", "normals", "vertex_voxel", "point_voxel"):
        np.save(os.path.join(tmp, k + ".npy"), geom[k])
    with open(os.path.join(tmp, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(geom["stats"], created=time.strftime("%Y-%m-%dT%H:%M:%S")), fh,
                  indent=2)
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.replace(tmp, path)
    return path


def load_geometry(path: str) -> dict | None:
    """Gespeicherte Geometrie laden (point_voxel als Memmap), sonst None."""
    try:
        with open(os.path.join(path, "meta.json"), encoding="utf-8") as fh:
            stats = json.load(fh)
        if int(stats.get("version", 0)) != _GEOM_VERSION:
            return None
        geom = {k: np.load(os.path.join(path, k + ".npy"),
                           mmap_mode="r" if k == "point_voxel" else None)
                for k in ("vertices", "triangles", "normals", "vertex_voxel", "point_voxel")}
    except (OSError, ValueError):
        return None
    geom["stats"] = stats
    return geom


# ================================================================ Farben

def _voxel_mittel(geom: dict, werte: np.ndarray, gewicht: np.ndarray | None):
    """Summe und Gewicht je Zelle fuer (N,C)-Werte."""
    pv = np.asarray(geom["point_voxel"])
    m = int(geom["stats"]["zellen"])
    w = None if gewicht is None else gewicht.astype(np.float64)
    wsum = np.bincount(pv, weights=w, minlength=m)
    sums = np.stack([np.bincount(pv, weights=werte[:, c] if w is None else werte[:, c] * w,
                                 minlength=m) for c in range(werte.shape[1])], axis=1)
    return sums, wsum


def _auf_ecken(geom: dict, sums: np.ndarray, wsum: np.ndarray):
    vv = geom["vertex_voxel"]
    s = sums[vv].sum(axis=1)          # (V,C)
    w = wsum[vv].sum(axis=1)          # (V,)
    ok = w > 0
    out = np.zeros_like(s)
    out[ok] = s[ok] / w[ok, None]
    return out, ok


def vertex_colors(geom: dict, colors_rgb: np.ndarray, valid: np.ndarray | None
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Farben einer Ebene auf die Ecken: (rgb uint8 (V,3), gueltig bool (V,)).

    Mittel der eingefaerbten Punkte in den naechsten Zellen jeder Ecke; Ecken
    ohne eingefaerbte Punkte in der Naehe sind grau und ungueltig.
    """
    c = np.asarray(colors_rgb, dtype=np.float32)
    sums, wsum = _voxel_mittel(geom, c, None if valid is None else np.asarray(valid, bool))
    rgb, ok = _auf_ecken(geom, sums, wsum)
    out = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    out[~ok] = _GRAU
    return out, ok


def vertex_scalar(geom: dict, values: np.ndarray) -> np.ndarray:
    """Mittel eines Werts je Punkt (z. B. Intensitaet) auf die Ecken, float32 (V,)."""
    v = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    sums, wsum = _voxel_mittel(geom, v, None)
    out, _ = _auf_ecken(geom, sums, wsum)
    return out[:, 0].astype(np.float32)


# ================================================================ Export

def to_open3d(geom: dict, rgb: np.ndarray | None = None):
    """Open3D-TriangleMesh aus Geometrie und optionalen Eckfarben (uint8)."""
    import open3d as o3d
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(geom["vertices"], np.float64)),
        o3d.utility.Vector3iVector(np.asarray(geom["triangles"], np.int32)))
    mesh.vertex_normals = o3d.utility.Vector3dVector(np.asarray(geom["normals"], np.float64))
    if rgb is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(rgb, np.float64) / 255.0)
    return mesh


def build_mesh(points: np.ndarray, colors_rgb: np.ndarray | None,
               sensor_path: np.ndarray | None, voxel: float = 0.05, depth: int = 11,
               trim: float = 0.0, max_gap: float = 3.0, progress: ProgressCb = None,
               cancel=None, log=None, backend: str = "auto"):
    """Geometrie plus Farben in einem Schritt: (open3d.TriangleMesh, stats)."""
    geom = build_geometry(points, sensor_path, voxel=voxel, depth=depth, trim=trim,
                          max_gap=max_gap, backend=backend, progress=progress,
                          cancel=cancel, log=log)
    rgb = None
    if colors_rgb is not None:
        rgb, _ = vertex_colors(geom, colors_rgb, None)
    return to_open3d(geom, rgb), geom["stats"]


def write_mesh(mesh, path: str, stats: dict | None = None) -> str:
    """Mesh als binaeres PLY (mit Farben) schreiben; Statistik daneben als JSON."""
    import open3d as o3d
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp.ply"
    if not o3d.io.write_triangle_mesh(tmp, mesh, write_ascii=False, compressed=False):
        raise RuntimeError(f"Mesh konnte nicht geschrieben werden: {path}")
    os.replace(tmp, path)
    if stats is not None:
        with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as fh:
            json.dump(dict(stats, created=time.strftime("%Y-%m-%dT%H:%M:%S")), fh, indent=2)
    return path


# ================================================================ CloudCompare

def find_cloudcompare() -> list[str] | None:
    """Befehl (als Liste) zum Start von CloudCompare, oder None."""
    eigen = os.environ.get("SUPER360_CLOUDCOMPARE", "").strip()
    if eigen:
        return shlex.split(eigen)
    for name in ("CloudCompare", "cloudcompare.CloudCompare", "cloudcompare"):
        p = shutil.which(name)
        if p:
            return [p]
    if shutil.which("flatpak"):
        try:
            r = subprocess.run(["flatpak", "info", _FLATPAK_ID], capture_output=True,
                               timeout=10)
            if r.returncode == 0:
                return ["flatpak", "run", _FLATPAK_ID]
        except (OSError, subprocess.TimeoutExpired):
            pass
    return None


def open_in_cloudcompare(paths: list[str]) -> list[str]:
    """CloudCompare mit den Dateien starten (eigene Sitzung, laeuft weiter).

    Rueckgabe: der verwendete Befehl. RuntimeError mit Installationshinweis,
    wenn CloudCompare fehlt.
    """
    cmd = find_cloudcompare()
    if cmd is None:
        raise RuntimeError(INSTALL_HINT)
    full = cmd + [os.path.abspath(p) for p in paths]
    subprocess.Popen(full, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True)
    return full


# ================================================================ Selbsttest

if __name__ == "__main__":
    import sys

    OUT = "/tmp/super360_modtests/mesh"
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)

    # ---------- Test 1: Kugel mit Loch, CPU und GPU ----------
    # Normalen muessen zur Flugbahn (Mittelpunkt) zeigen, das Netz darf das
    # Loch nicht schliessen, die Farben muessen ankommen
    v = rng.normal(size=(200_000, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    v = (v[v[:, 2] < 0.8] * 5.0).astype(np.float32)          # Kappe oben offen
    farbe = np.where(v[:, :1] > 0, [[200, 40, 30]], [[30, 60, 200]]).astype(np.uint8)
    gueltig = np.ones(len(v), bool)
    gueltig[v[:, 1] > 3.0] = False                           # ein Streifen ungefaerbt
    for be in ("cpu", "gpu") if opencl_available() else ("cpu",):
        geom = build_geometry(v, np.zeros((1, 3)), voxel=0.05, depth=8, backend=be,
                              log=lambda m: print("  " + m))
        V, Nn = geom["vertices"], geom["normals"]
        r = np.linalg.norm(V, axis=1)
        assert geom["stats"]["dreiecke"] > 1000, geom["stats"]
        assert np.abs(r - 5.0).max() < 0.3, f"Mesh weicht ab: {np.abs(r - 5).max()}"
        assert V[:, 2].max() < 5.0 * 0.8 + 0.2, "Loch oben wurde zugeflickt"
        innen = (np.einsum("ij,ij->i", Nn, -V) > 0).mean()
        assert innen > 0.95, f"Normalen zeigen nicht zur Flugbahn ({innen:.2f})"
        rgb, ok = vertex_colors(geom, farbe, gueltig)
        rechts = V[:, 0] > 1.0
        assert rgb[rechts & ok].mean(axis=0)[0] > 150, "Farben gehen verloren"
        assert not ok[V[:, 1] > 3.3].any(), "ungefaerbter Streifen bekommt Farbe"
        assert (rgb[~ok] == _GRAU).all()
        # keine eingestanzten Loecher: offene Kanten nur am Rand der Kappe
        # (Umfang 2*pi*3 m ~ 19 m bei ~5-7 cm Kantenlaenge)
        T = geom["triangles"]
        kanten = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1)
        _, anz = np.unique(kanten, axis=0, return_counts=True)
        offen = int((anz == 1).sum())
        assert offen < 1500, f"Mesh hat Loecher: {offen} offene Kanten"
        hoehe = vertex_scalar(geom, v[:, 2])
        assert np.abs(hoehe - V[:, 2]).max() < 0.2, "Skalar weicht von der Lage ab"
        print(f"== Kugel {be}: {geom['stats']['dreiecke']:,} Dreiecke, Radiusfehler "
              f"{np.abs(r - 5).max() * 100:.1f} cm, Normalen innen {innen * 100:.0f} %, "
              f"offene Kanten {offen}, ungefaerbt {100 * (~ok).mean():.1f} % der Ecken")

    # ---------- Test 2: Speichern und Laden ----------
    d = save_geometry(geom, os.path.join(OUT, "geom_test"))
    g2 = load_geometry(d)
    assert g2 is not None and np.array_equal(g2["triangles"], geom["triangles"])
    assert np.array_equal(np.asarray(g2["point_voxel"]), geom["point_voxel"])
    path = write_mesh(to_open3d(g2, rgb), os.path.join(OUT, "kugel.ply"), g2["stats"])
    print(f"== Speichern/Laden gleich, PLY -> {path}")
    print(f"== CloudCompare: {find_cloudcompare() or 'nicht gefunden'}")
    if "--oeffnen" in sys.argv:
        print("   gestartet:", open_in_cloudcompare([path]))
    print("mesh SELFTEST OK")
