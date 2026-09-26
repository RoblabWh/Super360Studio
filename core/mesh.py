"""Dreiecksnetz (Mesh) aus der Punktwolke und Anzeige in CloudCompare.

Weg (Open3D):
  1. Voxelraster: je Zelle ein Punkt, Farbe gemittelt. 57 Mio. Punkte waeren
     fuer die Poisson-Rekonstruktion zu viel und bringen unter ~5 cm nichts,
     was das Netz noch aufloesen koennte.
  2. Normalen aus den Nachbarn, dann zur naechsten Position der Flugbahn hin
     ausgerichtet: der Lidar hat jede Flaeche von der Seite gesehen, auf der die
     Drohne war. Ohne das zeigen Normalen zufaellig nach innen oder aussen und
     Poisson schliesst Fassaden zu Blasen.
  3. Poisson-Rekonstruktion; Tiefe d heisst ein Raster von 2^d Zellen ueber die
     groesste Ausdehnung.
  4. Beschnitt: Poisson schliesst die Flaeche auch dort, wo keine Messung ist.
     Ecken mit der geringsten Punktdichte (``trim``) und alles weiter als
     ``max_gap`` Voxel vom naechsten Punkt fallen weg.
  5. Farben kommen aus der Punktwolke (Poisson interpoliert sie mit).

Gemessen am ganzen Flug rosbag_2026-09-19_02-52-40 (57,6 Mio. eingefaerbte
Punkte, Raster 5 cm, Tiefe 11): 10,9 Mio. Voxelpunkte, 4,1 Mio. Dreiecke nach
dem Beschnitt, ~80 s.

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

__all__ = ["build_mesh", "write_mesh", "find_cloudcompare", "open_in_cloudcompare",
           "INSTALL_HINT"]

ProgressCb = Optional[Callable[[float, str], None]]

_FLATPAK_ID = "org.cloudcompare.CloudCompare"
INSTALL_HINT = ("CloudCompare nicht gefunden. Installieren ohne Root-Rechte:\n"
                f"  flatpak install --user flathub {_FLATPAK_ID}\n"
                "oder systemweit:\n"
                "  sudo snap install cloudcompare")


def _p(cb: ProgressCb, f: float, msg: str) -> None:
    if cb is not None:
        cb(float(f), msg)


def _cancelled(cancel) -> bool:
    return cancel is not None and cancel()


def build_mesh(points: np.ndarray, colors_rgb: np.ndarray | None,
               sensor_path: np.ndarray | None, voxel: float = 0.05, depth: int = 11,
               trim: float = 0.05, max_gap: float = 3.0, progress: ProgressCb = None,
               cancel=None, log=None):
    """Poisson-Mesh aus Punkten (N,3) mit Farben (N,3 uint8 RGB oder None).

    ``sensor_path`` (M,3): Positionen der Flugbahn fuer die Ausrichtung der
    Normalen; None richtet sie nach aussen vom Schwerpunkt aus (schlechter).
    Rueckgabe: (open3d.geometry.TriangleMesh, Statistik dict).
    """
    import open3d as o3d
    from scipy.spatial import cKDTree

    t0 = time.time()
    say = log or (lambda _m: None)
    pts = np.asarray(points)
    if len(pts) < 100:
        raise RuntimeError(f"Zu wenige Punkte für ein Mesh ({len(pts)}).")
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64)))
    if colors_rgb is not None:
        pc.colors = o3d.utility.Vector3dVector(
            np.asarray(colors_rgb, np.float64) / 255.0)

    _p(progress, 0.02, f"Voxelraster {voxel * 100:.0f} cm …")
    d = pc.voxel_down_sample(float(voxel))
    del pc
    n_vox = len(d.points)
    say(f"Mesh: {len(pts):,} Punkte → {n_vox:,} im {voxel * 100:.0f}-cm-Raster.")
    if _cancelled(cancel):
        raise RuntimeError("Abgebrochen")

    _p(progress, 0.25, "Normalen schätzen …")
    d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4.0 * voxel, max_nn=30))
    X = np.asarray(d.points)
    N = np.asarray(d.normals)
    if sensor_path is not None and len(sensor_path):
        _, k = cKDTree(np.asarray(sensor_path, np.float64)).query(X, workers=-1)
        zum_sensor = np.asarray(sensor_path, np.float64)[k] - X
    else:
        zum_sensor = X - X.mean(axis=0)
    flip = np.einsum("ij,ij->i", N, zum_sensor) < 0.0
    N[flip] *= -1.0
    d.normals = o3d.utility.Vector3dVector(N)
    if _cancelled(cancel):
        raise RuntimeError("Abgebrochen")

    _p(progress, 0.45, f"Poisson-Rekonstruktion (Tiefe {depth}) …")
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
        mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            d, depth=int(depth), n_threads=-1)
    dens = np.asarray(dens)
    n_roh = len(mesh.triangles)
    if _cancelled(cancel):
        raise RuntimeError("Abgebrochen")

    _p(progress, 0.85, "Mesh beschneiden …")
    if trim > 0.0:
        mesh.remove_vertices_by_mask(dens < np.quantile(dens, float(trim)))
    if max_gap > 0.0:
        V = np.asarray(mesh.vertices)
        dist, _ = cKDTree(X).query(V, workers=-1)
        mesh.remove_vertices_by_mask(dist > float(max_gap) * float(voxel))
    mesh.remove_unreferenced_vertices()
    if mesh.has_vertex_colors():
        # Poisson interpoliert Farben leicht ueber 0..1 hinaus
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.clip(np.asarray(mesh.vertex_colors), 0.0, 1.0))
    mesh.compute_vertex_normals()
    stats = {"punkte": int(len(pts)), "voxelpunkte": int(n_vox), "voxel_m": float(voxel),
             "tiefe": int(depth), "trim": float(trim), "max_gap_voxel": float(max_gap),
             "dreiecke_roh": int(n_roh), "dreiecke": int(len(mesh.triangles)),
             "ecken": int(len(mesh.vertices)), "farbig": bool(mesh.has_vertex_colors()),
             "laufzeit_s": round(time.time() - t0, 1)}
    say(f"Mesh: {stats['dreiecke']:,} Dreiecke (roh {n_roh:,}), {stats['laufzeit_s']} s.")
    _p(progress, 0.95, "Mesh fertig")
    return mesh, stats


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
    # Kugel mit Loch: Normalen muessen zur Flugbahn (Mittelpunkt) zeigen,
    # das Netz darf das Loch nicht schliessen
    rng = np.random.default_rng(0)
    v = rng.normal(size=(200_000, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    v = v[v[:, 2] < 0.8] * 5.0                        # Kappe oben offen
    farbe = np.where(v[:, :1] > 0, [[200, 40, 30]], [[30, 60, 200]]).astype(np.uint8)
    mesh, st = build_mesh(v, farbe, np.zeros((1, 3)), voxel=0.05, depth=8,
                          log=lambda m: print("  " + m))
    V = np.asarray(mesh.vertices)
    r = np.linalg.norm(V, axis=1)
    assert st["dreiecke"] > 1000, st
    assert np.abs(r - 5.0).max() < 0.3, f"Mesh weicht von der Kugel ab: {np.abs(r - 5).max()}"
    assert V[:, 2].max() < 5.0 * 0.8 + 0.2, "Loch oben wurde zugeflickt"
    Nn = np.asarray(mesh.vertex_normals)
    innen = np.einsum("ij,ij->i", Nn, -V) > 0
    assert innen.mean() > 0.95, f"Normalen zeigen nicht zur Flugbahn ({innen.mean():.2f})"
    C = np.asarray(mesh.vertex_colors)
    rot = C[V[:, 0] > 1.0].mean(axis=0)
    assert rot[0] > 0.6 and rot[2] < 0.3, f"Farben gehen verloren: {rot}"
    path = write_mesh(mesh, os.path.join(OUT, "kugel.ply"), st)
    print(f"== Kugel: {st['dreiecke']:,} Dreiecke, max. Radiusfehler "
          f"{np.abs(r - 5).max() * 100:.1f} cm, Normalen innen {innen.mean() * 100:.0f} %, "
          f"-> {path}")
    print(f"== CloudCompare: {find_cloudcompare() or 'nicht gefunden'}")
    if "--oeffnen" in sys.argv:
        print("   gestartet:", open_in_cloudcompare([path]))
    print("mesh SELFTEST OK")
