"""Einfaerben ueber ein Gaussian Splat, das auf der Lidar-Karte sitzt.

Die Frage dahinter: faerbt ein Splat die Karte besser als die direkte
Projektion? Die direkte Projektion nimmt je Punkt **ein** Bild (Maeander: das
frontalste, das ihn sieht) oder den Median weniger Bilder (Onboard). Sie kann
nichts dafuer, dass Bilder unterschiedlich belichtet sind, dass eine Pose um
ein halbes Pixel danebenliegt oder dass ein Punkt in einem Bild spiegelt —
all das landet in der Farbe, und an den Grenzen zwischen zwei Bildern wird es
zur Naht.

Ein Splat rechnet andersherum: Farben werden so gesucht, dass **alle** Bilder
zugleich erklaert sind, mit Belichtung je Bild, Verdeckung beim Rendern und
optional nachgefuehrten Posen. Das Training selbst laeuft in
``scripts/splat_train.py`` im Interpreter mit torch und gsplat; hier steht, was
im System-Python passiert:

1. **Anker** (:func:`anker`): die Karte auf ein Voxelraster, der Schwerpunkt
   jeder Zelle wird eine Gaussian. Das Raster waechst, bis die Zahl passt
   (Vorgabe 1,5 Mio. — was eine 8-GB-Karte traegt). Jeder Kartenpunkt kennt
   seine Zelle und bekommt am Ende deren Farbe.
2. **Datensatz**: Kamerabilder als verzeichnungsfreie Lochkameras im Rahmen der
   Karte, dazu je Bild eine Maske — nur Pixel, hinter denen die Karte etwas
   hat (:func:`abdeckung`), und beim Onboard-Flug ohne Himmel, Ueberbelichtung
   und Drohnenteile.
   * Maeander (:func:`datensatz_maeander`): die COLMAP-Kameras ueber die Kette
     COLMAP -> Karte, die das Einfaerben auch nimmt.
   * Onboard (:func:`datensatz_onboard`): je Fisheye fuenf Wuerfelseiten aus
     dem Double-Sphere-Modell, jede mit dem Zentrum ihrer eigenen Linse — kein
     Stitching, keine Parallaxe zwischen den Linsen.
   Dazu die Kartenpunkte selbst (``punkte.npy``) und ihr Zellindex: gefaerbt
   wird am Ende jeder Punkt, nicht jede Zelle.
3. **Training** (:func:`trainieren`) als Unterprozess. Danach rendert der
   Trainer jede Trainingsansicht ohne Belichtung und gibt jedem Punkt den
   Wert des Renderings an seinem Pixel, nach den Sichtregeln der direkten
   Projektion (s. ``scripts/splat_train.py``, ``abtasten``).
4. **Werte auf die Punkte** (:func:`punkt_farben`) und die **Gegenprobe**
   (:func:`vergleich`): zurueckgehaltene Bilder, die weder das Splat noch die
   direkte Projektion gesehen hat, gegen beide Einfaerbungen.

Qt-frei.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

from core.gemeinsam import (
    GRAU_EBENE, kamera_modell as _modell, melde as _p, pruefe_abbruch as _abbruch,
)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKRIPT = os.path.join(_REPO, "scripts", "splat_train.py")

#: Interpreter mit torch und gsplat. Der erste mit CUDA gewinnt.
PYTHON_KANDIDATEN = (
    os.environ.get("SUPER360_SPLAT_PYTHON", ""),
    os.path.expanduser("~/.venvs/splat/bin/python"),
    "/home/lena/_Data/26.05.21_DRZ_Avata360_Super360Pointcloud/"
    "gaussian_splat_avata360/.venv/bin/python",
)

#: Obergrenze der Gaussians. Mit festen Ankern (keine Verdichtung) und SH
#: ersten Grades belegt eine Gaussian samt Adam-Zustand rund 400 Byte, 4 Mio.
#: also 1,6 GB — geschaetzt fuer eine 8-GB-Karte. Die DRZ-Karte (24 Mio.
#: Punkte) landet damit bei 7,6 cm Raster, bei 5 cm sind es 7,2 Mio. Zellen.
MAX_ANKER = 4_000_000
ORDNER = "splat"          # Arbeitsordner im Projekt
_FALLBACK = (GRAU_EBENE,) * 3

#: Wuerfelseiten je Fisheye: Drehung Seite -> Linse (dritte Spalte = Blickachse)
_SEITEN = {
    "+z": np.eye(3),
    "+x": np.array([[0.0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
    "-x": np.array([[0.0, 0, -1], [0, 1, 0], [1, 0, 0]]),
    "+y": np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]]),
    "-y": np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]]),
}


# ------------------------------------------------------------ Interpreter

_PRUEF_CODE = r"""
import json
r = {}
try:
    import torch
    r["torch"] = torch.__version__
    r["cuda"] = bool(torch.cuda.is_available())
    if r["cuda"]:
        p = torch.cuda.get_device_properties(0)
        r["gpu"] = p.name
        r["vram_gb"] = round(p.total_memory / 1e9, 1)
        r["faehigkeit"] = "%d.%d" % torch.cuda.get_device_capability(0)
        r["architekturen"] = torch.cuda.get_arch_list()
        try:
            x = torch.ones(8, 8, device="cuda")
            r["rechnet"] = float((x @ x).sum()) == 512.0
        except Exception as e:
            r["rechnet"] = False
            r["cuda_fehler"] = (str(e).strip().splitlines() or [type(e).__name__])[0][:200]
        r["cuda"] = r["rechnet"]
    import importlib.util
    r["gsplat"] = importlib.util.find_spec("gsplat") is not None
    if r["gsplat"]:
        from importlib.metadata import version
        r["gsplat_version"] = version("gsplat")
        import glob, os
        ort = importlib.util.find_spec("gsplat").submodule_search_locations[0]
        r["gsplat_kompiliert"] = bool(glob.glob(os.path.join(ort, "csrc*.so")))
        if r.get("cuda") and r["gsplat_kompiliert"]:
            # Die Bibliothek kann fuer eine andere Karte gebaut sein (am
            # 2026-09-22: nur sm_120, gelaufen auf einer RTX 3060 Ti mit sm_86)
            # — erst ein Kernel sagt, ob sie hier rechnet.
            try:
                from gsplat.rendering import rasterization
                d = "cuda"
                m = torch.zeros(4, 3, device=d); m[:, 2] = 3.0
                q = torch.zeros(4, 4, device=d); q[:, 0] = 1.0
                rasterization(means=m, quats=q, scales=torch.full((4, 3), 0.1, device=d),
                              opacities=torch.full((4,), 0.5, device=d),
                              colors=torch.ones(4, 3, device=d),
                              viewmats=torch.eye(4, device=d)[None],
                              Ks=torch.tensor([[[16.0, 0, 8], [0, 16.0, 8], [0, 0, 1]]], device=d),
                              width=16, height=16)
                torch.cuda.synchronize()
                r["gsplat_rechnet"] = True
            except Exception as e:
                r["gsplat_rechnet"] = False
                r["gsplat_fehler"] = (str(e).strip().splitlines() or [type(e).__name__])[0][:200]
except Exception as e:
    r["fehler"] = f"{type(e).__name__}: {e}"
print(json.dumps(r))
"""


def pruefe_python(python: str) -> dict | None:
    """Was der Interpreter kann; None, wenn es ihn nicht gibt.

    gsplat wird bewusst nicht importiert: ohne GPU stoesst der Import eine
    CUDA-Kompilierung an, die mit einem IndexError abbricht.

    ``cuda`` heisst: auf der Karte laeuft wirklich eine Rechnung.
    ``torch.cuda.is_available()`` allein sagt das nicht — torch 2.4 mit CUDA
    12.1 meldet eine RTX 5090 als verfuegbar und scheitert beim ersten Kernel
    ("no kernel image is available"), weil es fuer deren Architektur (12.0)
    nicht gebaut ist.
    """
    if not python or not os.path.isfile(python):
        return None
    try:
        r = subprocess.run([python, "-c", _PRUEF_CODE], capture_output=True,
                           text=True, timeout=180)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"fehler": str(exc)}
    zeilen = [z for z in r.stdout.splitlines() if z.startswith("{")]
    if not zeilen:
        return {"fehler": (r.stderr or "").strip()[-300:]}
    try:
        return json.loads(zeilen[-1])
    except ValueError as exc:
        return {"fehler": str(exc)}


def find_splat_python() -> tuple[str | None, dict]:
    """(Interpreter, Auskunft). Mit CUDA vor ohne; ohne torch+gsplat (None, …)."""
    ersatz = None
    for p in PYTHON_KANDIDATEN:
        info = pruefe_python(p)
        if not info or not info.get("torch") or not info.get("gsplat"):
            continue
        info["python"] = p
        if info.get("cuda"):
            return p, info
        ersatz = ersatz or (p, info)
    if ersatz:
        return ersatz
    return None, {"fehler": "kein Interpreter mit torch und gsplat, gesucht: "
                            + ", ".join(p for p in PYTHON_KANDIDATEN if p)}


def hinweis(info: dict) -> str | None:
    """Warum das Training nicht laufen kann, als Satz — oder None."""
    if not info.get("torch") or not info.get("gsplat"):
        return ("Für das Splat-Training braucht es einen Interpreter mit torch und "
                "gsplat (" + str(info.get("fehler", "")) + "). Anlegen: python3 -m "
                "venv ~/.venvs/splat && ~/.venvs/splat/bin/pip install torch gsplat, "
                "oder SUPER360_SPLAT_PYTHON setzen.")
    if info.get("cuda") and info.get("gsplat_rechnet") is False:
        f = info.get("faehigkeit", "")
        return (f"gsplat ist nicht für diese Karte gebaut ({info.get('gpu')}, "
                f"sm_{f.replace('.', '')}: {info.get('gsplat_fehler', '')}). Neu bauen, "
                f"für mehrere Rechner mit allen Architekturen: TORCH_CUDA_ARCH_LIST=\"{f};12.0\" "
                f"CUDA_HOME=~/.venvs/splat/cuda ~/.venvs/splat/bin/pip install "
                f"--no-build-isolation --no-deps --force-reinstall --no-binary gsplat "
                f"gsplat=={info.get('gsplat_version', '')}")
    if info.get("cuda"):
        return None
    if info.get("rechnet") is False:
        return (f"PyTorch {info.get('torch')} sieht die {info.get('gpu')}, kann auf ihr aber "
                f"nicht rechnen: gebaut für {', '.join(info.get('architekturen') or [])}, die "
                f"Karte braucht sm_{info.get('faehigkeit', '').replace('.', '')} "
                f"({info.get('cuda_fehler', '')}). Für eine RTX 50xx: torch ab 2.7 mit "
                "cu128 und gsplat dafür gebaut, s. README, Abschnitt Gaussian Splat.")
    if not os.path.exists("/dev/nvidiactl"):
        return ("PyTorch sieht keine NVIDIA-GPU, der Treiber ist nicht geladen (kein "
                "/dev/nvidiactl). Bei eingeschaltetem Secure Boot lädt ein per DKMS "
                "gebautes Modul nur mit eingeschriebenem Schlüssel: sudo mokutil "
                "--import /var/lib/shim-signed/mok/MOK.der, neu starten und im "
                "MOK-Manager bestätigen — oder Secure Boot abschalten.")
    return "PyTorch sieht keine CUDA-GPU, obwohl der Treiber geladen ist."


def interpreter() -> tuple[str, dict]:
    """(Interpreter, Auskunft) fuer das Training; RuntimeError mit dem Hinweis,
    wenn es nicht laufen kann."""
    py, info = find_splat_python()
    if hinweis(info):
        raise RuntimeError(hinweis(info))
    return py, info


# ------------------------------------------------------------------ Anker

def anker(punkte: np.ndarray, voxel: float = 0.05, max_anker: int = MAX_ANKER,
          progress=None, cancel=None, log=None) -> dict:
    """Karte auf ein Voxelraster: Schwerpunkt und Normale je belegter Zelle.

    Rueckgabe ``pos`` float32 (M, 3), ``normal`` float32 (M, 3), ``voxel`` (m),
    ``index`` int32 (N,) — die Zelle jedes Kartenpunkts —, ``anzahl`` (M,).
    Das Raster waechst, bis hoechstens ``max_anker`` Zellen belegt sind. Reine
    Flaechen skalierten quadratisch mit dem Raster; an der DRZ-Karte mit
    Bewuchs waren es gemessen Exponent 1,34 (3 -> 9,2 cm) und 1,69
    (9,2 -> 13,4 cm), also wird mit 1,5 gerechnet und auf 95 % gezielt.
    """
    P = np.asarray(punkte)
    n = len(P)
    if n == 0:
        raise RuntimeError("Leere Karte — nichts zu verankern.")
    lo = P.min(axis=0).astype(np.float64) - 1e-3
    voxel = float(voxel)
    for _ in range(8):
        _abbruch(cancel)
        _p(progress, 0.05, f"Ankerraster {voxel * 100:.1f} cm …")
        keys = np.empty(n, np.int64)
        for s in range(0, n, 4_000_000):
            q = ((P[s:s + 4_000_000] - lo) / voxel).astype(np.int64)
            if q.max() >= (1 << 21):
                voxel *= 2.0
                break
            keys[s:s + len(q)] = (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]
        else:
            uniq, inv = np.unique(keys, return_inverse=True)
            if len(uniq) <= max_anker:
                break
            neu = voxel * float((len(uniq) / (0.95 * max_anker)) ** (1.0 / 1.5))
            if log is not None:
                log(f"{len(uniq):,} Zellen bei {voxel * 100:.1f} cm sind zu viele — "
                    f"Raster auf {neu * 100:.1f} cm.")
            voxel = neu
    else:
        raise RuntimeError("Ankerraster findet keine passende Zellgröße.")
    del keys
    M = len(uniq)
    cnt = np.bincount(inv, minlength=M)
    pos = np.stack([np.bincount(inv, weights=P[:, k], minlength=M) for k in range(3)],
                   1) / cnt[:, None]
    _p(progress, 0.6, f"Normalen für {M:,} Anker …")
    import open3d as o3d  # noqa: PLC0415
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pos))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3.0 * voxel, max_nn=20))
    nrm = np.asarray(pc.normals, np.float32)
    leer = ~np.isfinite(nrm).all(1) | (np.linalg.norm(nrm, axis=1) < 0.5)
    nrm[leer] = (0.0, 0.0, 1.0)
    _p(progress, 1.0, f"{M:,} Anker, Raster {voxel * 100:.1f} cm")
    return {"pos": pos.astype(np.float32), "normal": nrm, "voxel": float(voxel),
            "index": inv.astype(np.int32 if M < 2 ** 31 else np.int64),
            "anzahl": cnt.astype(np.int32)}


# ----------------------------------------------------------- Kameras

def ansichten_aus_colmap(cams, A, b) -> tuple[np.ndarray, dict]:
    """COLMAP-Kameras als Welt->Kamera-Matrizen im Rahmen der Karte.

    Die Einfaerbung rechnet ``p_cam = Rcw · A⁻¹ (p − b) + tcw``. Mit
    ``A = s·Q`` (Q orthogonal) ist das ``(Rcw Qᵀ (p − b)) / s + tcw``; die
    Kamerakoordinaten mit ``s`` gestreckt aendern kein Pixel, und uebrig bleibt
    eine starre Kamera in Metern: ``R = Rcw Qᵀ``, ``t = s·tcw − R b``. Die
    Feinausrichtung macht A nicht exakt aehnlich — wie weit, steht in
    ``abweichung`` (groesster Eintrag von A/s − Q).
    """
    A = np.asarray(A, float)
    b = np.asarray(b, float).ravel()
    s = float(np.cbrt(abs(np.linalg.det(A))))
    U, _, Vt = np.linalg.svd(A / s)
    Q = U @ Vt
    if np.linalg.det(Q) < 0:
        raise RuntimeError("Die Ausrichtung spiegelt — so eine Lage lässt sich nicht rendern.")
    Rcw = np.asarray(cams["Rcw"], float)
    tcw = np.asarray(cams["tcw"], float)
    vm = np.tile(np.eye(4), (len(Rcw), 1, 1))
    R = Rcw @ Q.T
    vm[:, :3, :3] = R
    vm[:, :3, 3] = s * tcw - np.einsum("nij,j->ni", R, b)
    return vm, {"massstab": s, "abweichung": float(np.abs(A / s - Q).max())}


def _lochkamera(model: str, params) -> tuple:
    p = np.asarray(params, float)
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        return p[0], p[0], p[1], p[2]
    if model in ("PINHOLE", "OPENCV"):
        return p[0], p[1], p[2], p[3]
    raise ValueError(f"Kameramodell {model} wird nicht unterstützt")


def entzerrung(model: str, params, size, bild_groesse, skala: float = 1.0) -> tuple:
    """Karten fuer ``cv2.remap``: verzeichnungsfreies Bild -> Originalpixel.

    Rueckgabe ``(map_x, map_y, K, gueltig)``. Konventionen: ``params`` gelten
    fuer ``size`` (W, H) mit Pixelmitten bei k+0,5, wie COLMAP und die
    Einfaerbung (``int(u · iw/W)``); gsplat rendert Pixel k ebenfalls bei k+0,5,
    cv2 tastet Pixel k bei k ab — daher die −0,5. Die Verzeichnung selbst kommt
    aus ``colorize_pipeline.colorize._project``, damit es keine zweite, leicht
    andere Formel gibt.
    """
    from core.meander import find_pipeline  # noqa: PLC0415
    find_pipeline()
    from colorize_pipeline import colorize as cz  # noqa: PLC0415
    W, H = float(size[0]), float(size[1])
    bw, bh = int(bild_groesse[0]), int(bild_groesse[1])
    aw, ah = max(1, int(round(W * skala))), max(1, int(round(H * skala)))
    sx, sy = aw / W, ah / H
    fx, fy, cx, cy = _lochkamera(model, params)
    K = np.array([[fx * sx, 0.0, cx * sx], [0.0, fy * sy, cy * sy], [0.0, 0.0, 1.0]])
    ii, jj = np.meshgrid(np.arange(aw) + 0.5, np.arange(ah) + 0.5)
    x = (ii / sx - cx) / fx
    y = (jj / sy - cy) / fy
    pc = np.stack([x.ravel(), y.ravel(), np.ones(x.size)], 1)
    u, v, _, _ = cz._project(pc, (W, H), np.asarray(params, float), model)
    mx = (u * (bw / W) - 0.5).reshape(ah, aw)
    my = (v * (bh / H) - 0.5).reshape(ah, aw)
    gueltig = (mx >= 0) & (mx <= bw - 1) & (my >= 0) & (my <= bh - 1)
    # Wo sich eine starke Tonnenverzeichnung zurueckfaltet, laeuft die Karte
    # rueckwaerts — dort gibt es kein eindeutiges Pixel.
    gueltig[:, 1:] &= np.diff(mx, axis=1) > 0
    gueltig[1:, :] &= np.diff(my, axis=0) > 0
    return mx.astype(np.float32), my.astype(np.float32), K, gueltig


def abdeckung(pos: np.ndarray, voxel: float, viewmat: np.ndarray, K: np.ndarray,
              W: int, H: int, min_weite: float = 0.0) -> np.ndarray:
    """Pixel, hinter denen die Karte eine Oberflaeche hat (bool H x W).

    Nur dort darf ein Bild das Splat belehren: der Himmel, alles jenseits der
    Karte und Luecken im Scan haben keine Gaussian, die die Farbe tragen koennte
    — ihr Pixel wuerde sonst die naechstbeste Gaussian daneben einfaerben. Jeder
    Anker wird mit seinem Fussabdruck gezeichnet (drei Viertel Voxel, in Pixel
    umgerechnet, auf 1/2/4/8/16 px gerundet).
    """
    import cv2  # noqa: PLC0415
    R = np.asarray(viewmat[:3, :3], np.float32)
    t = np.asarray(viewmat[:3, 3], np.float32)
    pc = np.asarray(pos, np.float32) @ R.T + t
    z = pc[:, 2]
    ok = z > 0.05
    nah = np.zeros(len(pc), bool)
    if min_weite > 0:
        nah = ok & (np.einsum("ni,ni->n", pc, pc) < min_weite * min_weite)
        ok &= ~nah
    zs = np.where(z > 0.05, z, 1.0)
    u = K[0, 0] * pc[:, 0] / zs + K[0, 2]
    v = K[1, 1] * pc[:, 1] / zs + K[1, 2]
    im_bild = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    ok &= im_bild
    nah &= im_bild
    r = 0.75 * voxel * K[0, 0] / zs
    klasse = np.clip(np.ceil(np.log2(np.maximum(r, 1.0))), 0, 4).astype(np.int8)

    def zeichne(auswahl):
        m = np.zeros((H, W), bool)
        for k in range(5):
            sel = auswahl & (klasse == k)
            if not sel.any():
                continue
            bild = np.zeros((H, W), np.uint8)
            bild[v[sel].astype(np.int64), u[sel].astype(np.int64)] = 1
            rad = 1 << k
            # Bei 1–2 px ist die "Ellipse" von cv2 ein Kreuz; fehlen durch
            # Rundung eine Spalte und eine Zeile zugleich, bliebe am Schnitt
            # ein Loch.
            form = cv2.MORPH_RECT if rad <= 2 else cv2.MORPH_ELLIPSE
            ker = cv2.getStructuringElement(form, (2 * rad + 1, 2 * rad + 1))
            m |= cv2.dilate(bild, ker) > 0
        return m

    maske = zeichne(ok)
    if nah.any():
        # Was naeher steht als min_weite, verdeckt sein Pixel trotzdem — beim
        # Start liegt der Boden direkt unter der Drohne. Solche Pixel duerfen
        # das Splat nicht belehren, sonst faerbt der nahe Boden alles dahinter.
        maske &= ~zeichne(nah)
    return maske


# ------------------------------------------------------------ Datensatz

def _schreibe(ordner: str, ak: dict, punkte: np.ndarray, liste: dict, extra: dict,
              param: dict | None) -> None:
    """Datensatz ablegen. ``param`` ueberschreibt die Vorgaben des Trainers;
    die Sichtregeln der direkten Projektion kommen immer mit, damit beide
    Einfaerbungen dieselben Punkte fuer sichtbar halten."""
    from core import sichtbar  # noqa: PLC0415
    if len(punkte) != len(ak["index"]):
        raise RuntimeError("Karte und Anker passen nicht zusammen.")
    np.savez(os.path.join(ordner, "anker.npz"), pos=ak["pos"], normal=ak["normal"],
             voxel=np.float32(ak["voxel"]), index=np.asarray(ak["index"], np.int32))
    np.save(os.path.join(ordner, "punkte.npy"), np.asarray(punkte, np.float32))
    param = dict({"min_cos": sichtbar.MIN_COS, "tiefe_skala": sichtbar.TIEFE_SKALA,
                  "toleranz_m": sichtbar.TOLERANZ_M, "toleranz_rel": sichtbar.TOLERANZ_REL},
                 **(param or {}))
    np.savez(os.path.join(ordner, "ansichten.npz"),
             viewmat=np.asarray(liste["viewmat"], np.float32).reshape(-1, 4, 4),
             K=np.asarray(liste["K"], np.float32).reshape(-1, 3, 3),
             breite=np.asarray(liste["breite"], np.int32),
             hoehe=np.asarray(liste["hoehe"], np.int32),
             bild=np.asarray(liste["bild"]), maske=np.asarray(liste["maske"]),
             holdout=np.asarray(liste["holdout"], bool), quelle=np.asarray(liste["quelle"]),
             **extra)
    with open(os.path.join(ordner, "param.json"), "w", encoding="utf-8") as fh:
        json.dump(param, fh, indent=2)


def _leeren(ordner: str) -> None:
    """Alter Datensatz und alte Ergebnisse weg — nie ein Ergebnis zum falschen Satz."""
    import shutil  # noqa: PLC0415
    if os.path.isdir(ordner):
        shutil.rmtree(ordner)
    os.makedirs(os.path.join(ordner, "bilder"))


def _pruefbild(i: int, halte_jedes: int) -> bool:
    return halte_jedes > 0 and i % halte_jedes == halte_jedes // 2


def datensatz_maeander(ordner: str, ak: dict, punkte: np.ndarray, cams, bild_ordner: str,
                       A, b, temperatur=None, halte_jedes: int = 0, skala: float = 1.0,
                       param: dict | None = None, progress=None, cancel=None,
                       log=None) -> dict:
    """Datensatz aus den Bildern eines Maeanderfluges.

    ``punkte`` ist die Karte, zu der ``ak`` gehoert. ``cams`` wie beim Einfaerben (``optik.rgb_cams`` bzw. ``thermal_cams``),
    ``A, b`` die Lage dazu. Mit ``temperatur`` (s. ``core.temperatur.quelle``)
    wird statt der Palette die Temperatur gelernt — ein Kanal, auf 0,5./99,5.
    Perzentil normiert (``t_lo``/``t_hi`` im Datensatz). Jedes ``halte_jedes``-te
    Bild wird fuer die Gegenprobe zurueckgehalten.
    """
    import cv2  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    _leeren(ordner)
    vm_alle, info = ansichten_aus_colmap(cams, A, b)
    if log is not None and info["abweichung"] > 1e-3:
        log(f"Hinweis: die Lage ist nicht ganz ähnlich (Abweichung "
            f"{info['abweichung']:.4f}); das Splat nimmt den starren Anteil.")
    model = _modell(cams)
    names = [str(n) for n in cams["names"]]
    size = np.asarray(cams["size"], float)
    params = np.asarray(cams["params"], float)
    t_lo = t_hi = float("nan")
    if temperatur is not None:
        proben = [tb[::8, ::8].ravel() for i, n in enumerate(names)
                  if (tb := temperatur(i, n)) is not None]
        if not proben:
            raise RuntimeError("Keine Temperaturbilder — die Thermalbilder tragen keine Rohwerte.")
        t_lo, t_hi = (float(x) for x in np.percentile(np.concatenate(proben), [0.5, 99.5]))
        if t_hi - t_lo < 1e-3:
            t_hi = t_lo + 1.0
    liste = {k: [] for k in ("viewmat", "K", "breite", "hoehe", "bild", "maske",
                             "holdout", "quelle")}
    karten: dict = {}
    anteile = []
    for i, name in enumerate(names):
        _abbruch(cancel)
        _p(progress, i / len(names), f"Splat-Datensatz: Bild {i + 1}/{len(names)}")
        if temperatur is not None:
            bild = temperatur(i, name)
            if bild is None:
                continue
            bild = np.asarray(bild, np.float32)
        else:
            with Image.open(os.path.join(bild_ordner, name)) as im:
                bild = np.asarray(im.convert("RGB"))
        bh, bw = bild.shape[:2]
        schluessel = (model, tuple(params[i]), tuple(size[i]), bw, bh)
        if schluessel not in karten:
            karten[schluessel] = entzerrung(model, params[i], size[i], (bw, bh), skala)
        mx, my, K, gueltig = karten[schluessel]
        ah, aw = mx.shape
        aus = cv2.remap(bild, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        m = gueltig & abdeckung(ak["pos"], ak["voxel"], vm_alle[i], K, aw, ah)
        if m.mean() < 0.02:
            if log is not None:
                log(f"{name}: sieht kaum etwas von der Karte ({m.mean() * 100:.1f} %) — weggelassen.")
            continue
        k = len(liste["bild"])
        if temperatur is not None:
            datei = f"bilder/{k:04d}.npy"
            np.save(os.path.join(ordner, datei), ((aus - t_lo) / (t_hi - t_lo)).astype(np.float16))
        else:
            datei = f"bilder/{k:04d}.jpg"
            Image.fromarray(aus).save(os.path.join(ordner, datei), quality=97)
        mdatei = f"bilder/{k:04d}_m.png"
        Image.fromarray((m * 255).astype(np.uint8)).save(os.path.join(ordner, mdatei))
        for key, wert in (("viewmat", vm_alle[i]), ("K", K), ("breite", aw), ("hoehe", ah),
                          ("bild", datei), ("maske", mdatei), ("quelle", name),
                          ("holdout", _pruefbild(i, halte_jedes))):
            liste[key].append(wert)
        anteile.append(float(m.mean()))
    if not liste["bild"]:
        raise RuntimeError("Kein Bild sieht die Karte — stimmt die Ausrichtung?")
    _schreibe(ordner, ak, punkte, liste, {"t_lo": np.float32(t_lo), "t_hi": np.float32(t_hi),
                                          "art": np.array("maeander")}, param)
    _p(progress, 1.0, "Splat-Datensatz fertig")
    return {"ansichten": len(liste["bild"]), "pruef": int(sum(liste["holdout"])),
            "abdeckung": float(np.mean(anteile)), "t_lo": t_lo, "t_hi": t_hi, **info}


def _starre_pixel(proben: list, kreis: np.ndarray) -> np.ndarray | None:
    """Pixel, die sich ueber den Flug nicht aendern: Arme, Motoren, Kabel.

    Gemessen an 09-08 12-53-52 ueber 40 Frames: die Drohnenteile liegen fest
    im Bild und streuen zeitlich kaum (Motoren: Streuung 0,6–3,4 bei mittlerer
    Helligkeit 2,5–4 — schwarz, eine untere Helligkeitsgrenze verliert sie),
    der Boden darunter streut um 40. Der ausgebrannte Himmel streut auch nicht,
    ist aber hell und faellt ueber die obere Grenze heraus. So bleiben 2,0 %
    (Linse 0) und 2,6 % (Linse 1) des Bildkreises. Aus hoechstens 40 Bildern
    auf Viertelgroesse.
    """
    import cv2  # noqa: PLC0415
    if len(proben) < 10:
        return None
    g = np.stack(proben).astype(np.float32)
    sd, mw = g.std(0), g.mean(0)
    starr = (sd < 6.0) & (mw < 235.0) & kreis
    if starr.mean() > 0.3 * max(kreis.mean(), 1e-6):
        return None            # zu viel: die Drohne stand wohl, keine Aussage
    starr = cv2.dilate(starr.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return starr


def datensatz_onboard(ordner: str, ak: dict, punkte: np.ndarray, rec, teile,
                      calib_json: str, T_imu_cam0: np.ndarray, bmin: int = 20,
                      bmax: int = 235, sky_clip: int = 250, sky_grow: int = 4,
                      max_bilder: int = 300, seite: int = 640, min_weg: float = 0.3,
                      min_dreh_grad: float = 8.0, max_weite: float = 60.0,
                      min_weite: float = 0.5, halte_jedes: int = 0,
                      param: dict | None = None, progress=None, cancel=None,
                      log=None) -> dict:
    """Datensatz aus der 360°-Kamera an Bord.

    ``teile`` wie bei ``colorizer.colorize``: ``[(bag, scan_von, scan_bis)]``.
    Ausgewaehlt werden Frames, zwischen denen sich die Drohne mindestens
    ``min_weg`` bewegt oder ``min_dreh_grad`` gedreht hat — 20 Bilder je
    Sekunde im Schwebeflug waeren dasselbe Bild zwanzigmal —, hoechstens
    ``max_bilder``. Je Fisheye fuenf Wuerfelseiten zu 90°, jede aus dem Zentrum
    ihrer Linse, maskiert wie beim Einfaerben: Bildkreis mit Randzonen-Guard,
    Helligkeitsfenster, Himmelssaum, dazu starre Drohnenteile.
    """
    import cv2  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from scipy.spatial.transform import Rotation  # noqa: PLC0415
    from core.colorizer import _blown_mask, _in_circle  # noqa: PLC0415
    from core.stitcher import SPLIT_X, DoubleSphereCamera  # noqa: PLC0415

    _leeren(ordner)
    cam0, cam1, T01 = DoubleSphereCamera.from_calib(calib_json)
    T_imu_cam0 = np.asarray(T_imu_cam0, float)
    linsen = ((cam0, np.eye(4)), (cam1, np.asarray(T01, float)))

    # --- Frames waehlen
    wahl = []
    for ti, (bag, von, bis) in enumerate(teile):
        st = bag.camera_stamps()
        if len(st) == 0 or bis <= von:
            continue
        j0, j1 = np.searchsorted(st, [float(rec.stamps[von]), float(rec.stamps[bis - 1])])
        letzte = None
        for fidx in range(int(j0), int(j1)):
            T = rec.interpolate_pose(float(st[fidx]))
            if T is None:
                continue
            if letzte is not None:
                weg = np.linalg.norm(T[:3, 3] - letzte[:3, 3])
                dreh = np.degrees(Rotation.from_matrix(letzte[:3, :3].T @ T[:3, :3]).magnitude())
                if weg < min_weg and dreh < min_dreh_grad:
                    continue
            wahl.append((ti, fidx, T))
            letzte = T
    if not wahl:
        raise RuntimeError("Keine Kamera-Frames mit Pose — passt das Bag zur Karte?")
    if len(wahl) > max_bilder:
        wahl = [wahl[int(k)] for k in np.linspace(0, len(wahl) - 1, max_bilder).round()]
    if log is not None:
        log(f"Onboard-Splat: {len(wahl)} Frames ausgewählt (Abstand ≥ {min_weg} m "
            f"oder ≥ {min_dreh_grad:.0f}°).")

    # --- Wuerfelseiten je Linse (Geometrie haengt nicht vom Frame ab)
    f = seite / 2.0
    ii, jj = np.meshgrid(np.arange(seite) + 0.5, np.arange(seite) + 0.5)
    strahl = np.stack([(ii - seite / 2) / f, (jj - seite / 2) / f, np.ones_like(ii)], -1)
    K = np.array([[f, 0.0, seite / 2], [0.0, f, seite / 2], [0.0, 0.0, 1.0]])
    seiten = []           # (linse, name, R4, map_x, map_y, geo)
    for li, (cam, _) in enumerate(linsen):
        for name, Rf in _SEITEN.items():
            uv, geo = cam.project(strahl @ Rf.T)
            geo &= _in_circle(uv.reshape(-1, 2)).reshape(seite, seite)
            if geo.mean() < 0.05:
                continue
            R4 = np.eye(4)
            R4[:3, :3] = Rf
            seiten.append((li, name, R4, np.ascontiguousarray(uv[..., 0]),
                           np.ascontiguousarray(uv[..., 1]), geo))

    # --- starre Drohnenteile, je Linse auf Viertelgroesse
    proben = ([], [])
    for k in np.linspace(0, len(wahl) - 1, min(40, len(wahl))).round().astype(int):
        ti, fidx, _ = wahl[k]
        roh = teile[ti][0].read_camera(fidx)
        for li in range(2):
            h = roh[:, li * SPLIT_X:(li + 1) * SPLIT_X]
            proben[li].append(cv2.cvtColor(cv2.resize(h, (SPLIT_X // 4, h.shape[0] // 4)),
                                           cv2.COLOR_BGR2GRAY))
    ky, kx = np.mgrid[0:proben[0][0].shape[0], 0:proben[0][0].shape[1]] * 4.0 + 2.0
    kreis = _in_circle(np.stack([kx.ravel(), ky.ravel()], 1)).reshape(kx.shape)
    starr = []
    for li in range(2):
        s = _starre_pixel(proben[li], kreis)
        if s is not None:
            s = cv2.resize(s.astype(np.uint8), (SPLIT_X, proben[li][0].shape[0] * 4),
                           interpolation=cv2.INTER_NEAREST) > 0
            if log is not None:
                log(f"Linse {li}: {s.mean() * 100:.1f} % des Bildes als Drohnenteil maskiert.")
        starr.append(s)

    # --- Anker nach x sortiert: je Frame nur die Umgebung projizieren
    pos = ak["pos"]
    ordnung = np.argsort(pos[:, 0], kind="stable")
    xs = pos[ordnung, 0]

    liste = {k: [] for k in ("viewmat", "K", "breite", "hoehe", "bild", "maske",
                             "holdout", "quelle")}
    anteile = []
    for k, (ti, fidx, T_wi) in enumerate(wahl):
        _abbruch(cancel)
        _p(progress, k / len(wahl), f"Onboard-Splat-Datensatz: Frame {k + 1}/{len(wahl)}")
        roh = teile[ti][0].read_camera(fidx)
        haelften = [np.ascontiguousarray(roh[:, li * SPLIT_X:(li + 1) * SPLIT_X]) for li in range(2)]
        himmel = [_blown_mask(h, sky_clip, sky_grow) for h in haelften]
        c = (T_wi @ T_imu_cam0)[:3, 3]
        lo, hi = np.searchsorted(xs, [c[0] - max_weite, c[0] + max_weite])
        nah = ordnung[lo:hi]
        nah = pos[nah[np.abs(pos[nah, 1] - c[1]) < max_weite]]
        for li, name, R4, mx, my, geo in seiten:
            h = haelften[li]
            bild = cv2.remap(h, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            luma = 0.299 * bild[..., 2] + 0.587 * bild[..., 1] + 0.114 * bild[..., 0]
            m = geo & (luma >= bmin) & (luma <= bmax)
            if himmel[li] is not None:
                m &= cv2.remap(himmel[li].astype(np.uint8), mx, my, cv2.INTER_NEAREST) == 0
            if starr[li] is not None:
                m &= cv2.remap(starr[li].astype(np.uint8), mx, my, cv2.INTER_NEAREST) == 0
            T_wf = T_wi @ T_imu_cam0 @ linsen[li][1] @ R4
            vm = np.linalg.inv(T_wf)
            if m.mean() < 0.02:
                continue
            m &= abdeckung(nah, ak["voxel"], vm, K, seite, seite, min_weite=min_weite)
            if m.mean() < 0.02:
                continue
            j = len(liste["bild"])
            datei, mdatei = f"bilder/{j:05d}.jpg", f"bilder/{j:05d}_m.png"
            Image.fromarray(bild[..., ::-1]).save(os.path.join(ordner, datei), quality=95)
            Image.fromarray((m * 255).astype(np.uint8)).save(os.path.join(ordner, mdatei))
            for key, wert in (("viewmat", vm), ("K", K), ("breite", seite), ("hoehe", seite),
                              ("bild", datei), ("maske", mdatei),
                              ("quelle", f"teil{ti}_frame{fidx}_linse{li}_{name}"),
                              ("holdout", _pruefbild(k, halte_jedes))):
                liste[key].append(wert)
            anteile.append(float(m.mean()))
    if not liste["bild"]:
        raise RuntimeError("Keine brauchbare Ansicht — alles Himmel, überbelichtet oder "
                           "außerhalb der Karte?")
    # Weiter als max_weite hat der Datensatz keine Anker gezeichnet, dort ist
    # jedes Pixel maskiert — der Trainer braucht diese Punkte nicht anzufassen.
    _schreibe(ordner, ak, punkte, liste,
              {"t_lo": np.float32(np.nan), "t_hi": np.float32(np.nan),
               "art": np.array("onboard")},
              dict({"reichweite": float(max_weite)}, **(param or {})))
    _p(progress, 1.0, "Onboard-Splat-Datensatz fertig")
    return {"ansichten": len(liste["bild"]), "frames": len(wahl),
            "pruef": int(sum(liste["holdout"])), "abdeckung": float(np.mean(anteile))}


def datensatz_gemeinsam(ordner: str, ak: dict, punkte: np.ndarray, maeander: dict,
                        onboard: dict, abbildung: tuple | None = None,
                        param: dict | None = None, progress=None, cancel=None,
                        log=None) -> dict:
    """Ein Datensatz aus Maeander- und Onboard-Ansichten, Farbbezug Maeander.

    ``maeander`` und ``onboard`` sind die Schluesselwortargumente fuer
    :func:`datensatz_maeander` bzw. :func:`datensatz_onboard` (ohne ``ordner``,
    ``ak``, ``punkte``). Beide Teile entstehen in Unterordnern und werden zu
    einer Ansichtenliste verbunden. Je Ansicht kommt dazu:

    * ``bezug`` — Maeander ja: der Trainer belichtet sie nicht;
    * ``herkunft`` — gezogen wird abwechselnd aus jeder;
    * ``reichweite`` — Onboard wie im eigenen Datensatz, Maeander 0 (ein
      Nadirbild aus 80 m Hoehe sieht weiter als 60 m zur Seite);
    * ``bel0_D``/``bel0_e`` — mit ``abbildung = (M, t)`` aus
      ``core.fusion`` startet jede Onboard-Ansicht bei dieser Farbmatrix.
    """
    _leeren(ordner)
    teile = {}
    for name, fn, kw, von, bis in (("maeander", datensatz_maeander, maeander, 0.0, 0.3),
                                   ("onboard", datensatz_onboard, onboard, 0.3, 1.0)):
        unter = os.path.join(ordner, name)
        info = fn(unter, ak, punkte, progress=lambda f, m, a=von, b=bis:
                  _p(progress, a + (b - a) * f, m), cancel=cancel, log=log, **kw)
        ans = dict(np.load(os.path.join(unter, "ansichten.npz")))
        with open(os.path.join(unter, "param.json"), encoding="utf-8") as fh:
            reich = float(json.load(fh).get("reichweite", 0.0))
        teile[name] = (ans, reich, info)
        for datei in ("anker.npz", "punkte.npy"):     # stehen gleich im Hauptordner
            os.remove(os.path.join(unter, datei))

    liste = {k: [] for k in ("viewmat", "K", "breite", "hoehe", "bild", "maske",
                             "holdout", "quelle")}
    bezug, herkunft, reichweite = [], [], []
    for name, (ans, reich, _) in teile.items():
        V = len(ans["bild"])
        for k in ("viewmat", "K", "breite", "hoehe", "holdout", "quelle"):
            liste[k].extend(list(ans[k]))
        liste["bild"].extend(f"{name}/{b}" for b in ans["bild"])
        liste["maske"].extend(f"{name}/{m}" for m in ans["maske"])
        bezug += [name == "maeander"] * V
        herkunft += [name] * V
        reichweite += [reich] * V
    bezug = np.asarray(bezug, bool)
    V = len(bezug)
    extra = {"t_lo": np.float32(np.nan), "t_hi": np.float32(np.nan),
             "art": np.array("gemeinsam"), "bezug": bezug,
             "herkunft": np.asarray(herkunft), "reichweite": np.asarray(reichweite, np.float32)}
    if abbildung is not None:
        M, t = (np.asarray(x, np.float32) for x in abbildung)
        extra["bel0_D"] = np.where(bezug[:, None, None], 0.0,
                                   (M - np.eye(3))[None]).astype(np.float32)
        extra["bel0_e"] = np.where(bezug[:, None], 0.0, t[None]).astype(np.float32)
    # Maeander viermal so oft ziehen wie Onboard: am is7-Projekt (3000 Schritte)
    # lag der Maeander-Pruefbildfehler bei 1:1 / 2:1 / 4:1 bei 0,0866 / 0,0852 /
    # 0,0839, nur Maeander 0,0828 — bei gleicher Abdeckung von 96 % statt 67 %.
    param = dict({"ziehen": {"maeander": 4}}, **(param or {}))
    _schreibe(ordner, ak, punkte, liste, extra, param)
    _p(progress, 1.0, "Gemeinsamer Splat-Datensatz fertig")
    im, io = teile["maeander"][2], teile["onboard"][2]
    return {"ansichten": V, "maeander": im["ansichten"], "onboard": io["ansichten"],
            "frames": io["frames"], "pruef": int(np.sum(liste["holdout"])),
            "startabbildung": abbildung is not None}


# ------------------------------------------------------------- Training

def trainieren(python: str, ordner: str, progress=None, cancel=None, log=None,
               cpu: bool = False, alle_bilder: bool = False) -> dict:
    """``scripts/splat_train.py`` als Unterprozess; Protokoll s. dort.

    ``alle_bilder`` trainiert auch die Pruefbilder mit — der Lauf fuer die
    Ebene nach der Gegenprobe, auf demselben Datensatz.
    """
    cmd = ([python, "-u", SKRIPT, ordner] + (["--cpu"] if cpu else [])
           + (["--alle"] if alle_bilder else []))
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    if log is not None:
        log("Splat: Training gestartet. Beim allerersten Lauf baut gsplat seine "
            "CUDA-Kernel — das dauert Minuten, in denen keine Zeile kommt.")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, env=env)
    pruefung, rest = [], []
    try:
        for zeile in proc.stdout:
            zeile = zeile.rstrip()
            if not zeile:
                continue
            teile_ = zeile.split()
            if teile_[0] == "STEP" and len(teile_) >= 5:
                i, n = int(teile_[1]), int(teile_[2])
                _p(progress, i / max(n, 1),
                   f"Splat: Schritt {i}/{n}, PSNR {float(teile_[4]):.1f} dB")
            elif teile_[0] == "INFO":
                if log is not None:
                    log("Splat: " + zeile[5:])
            elif teile_[0] == "EVAL" and len(teile_) >= 6:
                pruefung.append({"bild": teile_[1], "l1_roh": float(teile_[2]),
                                 "l1_angepasst": float(teile_[3]),
                                 "psnr_angepasst": float(teile_[4]), "pixel": int(teile_[5])})
            elif teile_[0] != "DONE":
                rest.append(zeile)
                del rest[:-30]
            if cancel is not None and (cancel() if callable(cancel) else cancel.is_set()):
                proc.terminate()
                raise RuntimeError("Abgebrochen")
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
    if proc.returncode != 0:
        raise RuntimeError("Splat-Training fehlgeschlagen:\n" + "\n".join(rest[-12:]))
    with open(os.path.join(ordner, "bericht.json"), encoding="utf-8") as fh:
        bericht = json.load(fh)
    return {"bericht": bericht, "pruefung": pruefung}


def lauf(py, ordner, cfg, progress, cancel, log, von, bis, alle) -> dict:
    """Ein Training (:func:`trainieren`) mit Fortschritt von ``von`` bis ``bis``.

    ``cancel`` ist das Ereignis des Arbeiters, ``cfg`` die Einstellungen des
    Laufs. Nachgefuehrte Posen und die Pruefbilder kommen ins Protokoll.
    """
    erg = trainieren(
        py, ordner, progress=lambda f, m: progress(von + (bis - von) * f, m),
        cancel=lambda: cancel.is_set(), log=log, alle_bilder=alle)
    p = erg["bericht"].get("posen") or {}
    if cfg["param"]["posen"] and p:
        log(f"Splat-Posen nachgeführt: Drehung Median {p['dreh_grad_median']:.3f}° "
            f"(max {p['dreh_grad_max']:.3f}°), Weg Median "
            f"{p['weg_m_median'] * 100:.1f} cm (max {p['weg_m_max'] * 100:.1f} cm).")
    if erg["pruefung"]:
        psnr = np.mean([e["psnr_angepasst"] for e in erg["pruefung"]])
        log(f"Splat an {len(erg['pruefung'])} Prüfbildern gerendert: "
            f"PSNR {psnr:.1f} dB nach Belichtungsangleich.")
    return erg


def trainingsfolge(py, ordner, cfg, spannen, gegenprobe_cb, progress, cancel,
                   log) -> tuple:
    """Training mit oder ohne Gegenprobe: (Ergebnis des letzten Laufs, Gegenprobe).

    ``spannen = (von, mitte, weiter, bis)`` sind Fortschrittswerte. Mit
    ``cfg["pruefen"]`` erst ein Lauf ohne die Pruefbilder (``von`` bis
    ``mitte``), dann ``gegenprobe_cb(erg)`` mit dessen Ergebnis, dann ein Lauf
    mit allen Bildern fuer die Ebene (``weiter`` bis ``bis``). Sonst ein Lauf
    von ``von`` bis ``bis``, die Gegenprobe ist dann None.
    """
    von, mitte, weiter, bis = spannen
    if not cfg["pruefen"]:
        return lauf(py, ordner, cfg, progress, cancel, log, von, bis, False), None
    erg = lauf(py, ordner, cfg, progress, cancel, log, von, mitte, False)
    stats = gegenprobe_cb(erg) if gegenprobe_cb is not None else None
    return lauf(py, ordner, cfg, progress, cancel, log, weiter, bis, True), stats


def splat_meta(ak: dict, cfg: dict, posen) -> dict:
    """Eintrag ``splat`` in der meta.json einer Splat-Ebene."""
    return {"anker": int(len(ak["pos"])), "raster_m": ak["voxel"], **cfg["param"],
            "posen": posen}


# -------------------------------------------------------- auf die Punkte

def temperatur_farben(temp: np.ndarray, maske: np.ndarray) -> np.ndarray:
    """Temperatur -> RGB ueber eine feste Palette (Inferno, 1.–99. Perzentil).

    Bewusst nicht die Palette der Kamera: die DJI-Bilder skalieren sie je Bild
    selbst, dieselbe Farbe heisst in zwei Bildern zwei Temperaturen.
    """
    import cv2  # noqa: PLC0415
    rgb = np.empty((len(temp), 3), np.uint8)
    rgb[:] = _FALLBACK
    if not maske.any():
        return rgb
    lo, hi = np.percentile(temp[maske], [1, 99])
    g = np.clip((temp[maske] - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    rgb[maske] = cv2.applyColorMap(g.reshape(-1, 1), cv2.COLORMAP_INFERNO).reshape(-1, 3)[:, ::-1]
    return rgb


def punkt_farben(ordner: str, n_punkte: int) -> dict:
    """Ergebnis des Trainings je Kartenpunkt: ``rgb``, ``maske``, bei
    Temperatur zusaetzlich ``temperatur`` (°C, NaN wo ungesehen).

    Der Trainer hat jeden Punkt schon aus dem Rendering abgetastet; ungesehen
    ist, was in keiner Trainingsansicht sichtbar war (Guete 0).
    """
    z = np.load(os.path.join(ordner, "ergebnis.npz"))
    ans = np.load(os.path.join(ordner, "ansichten.npz"))
    if "punkt_wert" not in z:
        raise RuntimeError("Splat-Ergebnis aus einer älteren Version — neu trainieren.")
    wert, maske = z["punkt_wert"], z["punkt_guete"] > 0
    if len(wert) != int(n_punkte):
        raise RuntimeError("Splat-Ergebnis passt nicht zur Karte — neu trainieren.")
    if wert.shape[1] == 3:
        rgb = np.empty((len(wert), 3), np.uint8)
        for a in range(0, len(wert), 4_000_000):
            rgb[a:a + 4_000_000] = np.clip(
                np.rint(wert[a:a + 4_000_000].astype(np.float32) * 255.0), 0, 255)
        rgb[~maske] = _FALLBACK
        return {"rgb": rgb, "maske": maske}
    t_lo, t_hi = float(ans["t_lo"]), float(ans["t_hi"])
    temp = (wert[:, 0].astype(np.float32) * (t_hi - t_lo) + t_lo).astype(np.float32)
    temp[~maske] = np.nan
    return {"rgb": temperatur_farben(temp, maske), "maske": maske, "temperatur": temp}


# ------------------------------------------------------------ Gegenprobe

def probe(ak: dict, zellen: int = 60_000, seed: int = 0) -> np.ndarray:
    """Punktindizes ganzer Ankerzellen fuer die Gegenprobe — gefaerbt wird
    direkt nur diese Probe, nicht die ganze Karte."""
    M = len(ak["pos"])
    wahl = np.zeros(M, bool)
    wahl[np.random.default_rng(seed).choice(M, min(int(zellen), M), replace=False)] = True
    return np.flatnonzero(wahl[ak["index"]])


def pruef_namen(ordner: str) -> list:
    """Originalnamen der zurueckgehaltenen Bilder (Maeander: Dateinamen)."""
    ans = np.load(os.path.join(ordner, "ansichten.npz"))
    return [str(q) for q, h in zip(ans["quelle"], ans["holdout"]) if h]


def cams_ohne(cams, weg) -> dict:
    """Kameras ohne die genannten Bilder — fuer die direkte Projektion der Gegenprobe."""
    weg = set(weg)
    keep = np.array([str(n) not in weg for n in cams["names"]])
    out = {k: np.asarray(cams[k])[keep] for k in ("names", "Rcw", "tcw", "size", "params")}
    out["model"] = np.array(_modell(cams))
    return out


def _zpuffer(pos, viewmat, K, W, H, skala=0.25):
    R = np.asarray(viewmat[:3, :3], np.float32)
    t = np.asarray(viewmat[:3, 3], np.float32)
    pc = pos @ R.T + t
    z = pc[:, 2]
    ok = z > 0.05
    zs = np.where(ok, z, 1.0)
    u = (K[0, 0] * pc[:, 0] / zs + K[0, 2]) * skala
    v = (K[1, 1] * pc[:, 1] / zs + K[1, 2]) * skala
    w, h = max(1, int(W * skala)), max(1, int(H * skala))
    ok &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
    puffer = np.full(w * h, np.inf, np.float32)
    np.minimum.at(puffer, v[ok].astype(np.int64) * w + u[ok].astype(np.int64), z[ok])
    return puffer, w, h


def vergleich(ordner: str, ak: dict, punkte: np.ndarray, normalen: np.ndarray,
              werte: dict, min_cos: float = 0.3, progress=None, cancel=None) -> dict:
    """Einfaerbungen gegen die zurueckgehaltenen Bilder des Datensatzes.

    ``werte``: Name -> (Werte (N, C) in Bildeinheiten — RGB 0..255 bzw. °C —,
    Maske (N,)) fuer dieselbe Punktprobe ``punkte``. Verglichen wird nur, wo
    der Punkt im Pruefbild sichtbar ist (Tiefe aus den Ankern), frontal genug
    (``min_cos``), im gueltigen Bereich der Maske und **in allen** Einfaerbungen
    eine Farbe hat — sonst gewinnt, wer weniger faerbt.

    Je Einfaerbung ``roh`` (mittlerer Betrag der Differenz je Kanal) und
    ``angepasst`` (nach Verstaerkung und Versatz je Kanal und Bild, kleinste
    Quadrate): das Pruefbild hat seine eigene Belichtung, die keine der
    Einfaerbungen kennen kann.
    """
    from PIL import Image  # noqa: PLC0415
    ans = np.load(os.path.join(ordner, "ansichten.npz"))
    halt = np.flatnonzero(ans["holdout"])
    t_lo, t_hi = float(ans["t_lo"]), float(ans["t_hi"])
    P = np.asarray(punkte, np.float32)
    nrm = np.asarray(normalen, np.float32)
    namen = list(werte)
    alle = np.ones(len(P), bool)
    for nm in namen:
        alle &= np.asarray(werte[nm][1], bool)
    summe = {nm: [0.0, 0.0] for nm in namen}
    n_ges, n_bilder = 0, 0
    tol = 1.5 * float(ak["voxel"])
    for k, v in enumerate(halt):
        _abbruch(cancel)
        _p(progress, k / max(len(halt), 1), f"Gegenprobe: Bild {k + 1}/{len(halt)}")
        vm, K = ans["viewmat"][v], ans["K"][v]
        W, H = int(ans["breite"][v]), int(ans["hoehe"][v])
        pc = P @ vm[:3, :3].T + vm[:3, 3]
        z = pc[:, 2]
        zs = np.where(z > 0.05, z, 1.0)
        u = K[0, 0] * pc[:, 0] / zs + K[0, 2]
        vv = K[1, 1] * pc[:, 1] / zs + K[1, 2]
        sel = alle & (z > 0.05) & (u >= 0) & (u < W) & (vv >= 0) & (vv < H)
        puffer, w, h = _zpuffer(ak["pos"], vm, K, W, H)
        # Zeile mit h/H, Spalte mit w/W: bei einem Bild, dessen Hoehe die
        # Viertelung nicht glatt trifft, liefe die letzte Zeile sonst aus dem
        # Puffer heraus (IndexError mitten in der Gegenprobe).
        zelle = (np.minimum((vv[sel] * h / H).astype(np.int64), h - 1) * w
                 + np.minimum((u[sel] * w / W).astype(np.int64), w - 1))
        idx = np.flatnonzero(sel)
        idx = idx[z[idx] <= puffer[zelle] + tol + 0.01 * z[idx]]
        campos = -vm[:3, :3].T @ vm[:3, 3]
        blick = campos - P[idx]
        blick /= np.linalg.norm(blick, axis=1, keepdims=True)
        idx = idx[np.abs((blick * nrm[idx]).sum(1)) >= min_cos]
        with Image.open(os.path.join(ordner, str(ans["maske"][v]))) as im:
            maske = np.asarray(im.convert("L")) > 127
        ui, vi = u[idx].astype(np.int64), vv[idx].astype(np.int64)
        idx, ui, vi = idx[maske[vi, ui]], ui[maske[vi, ui]], vi[maske[vi, ui]]
        if len(idx) < 50:
            continue
        datei = os.path.join(ordner, str(ans["bild"][v]))
        if datei.endswith(".npy"):
            gt = (np.load(datei).astype(np.float32) * (t_hi - t_lo) + t_lo)[vi, ui][:, None]
        else:
            with Image.open(datei) as im:
                gt = np.asarray(im.convert("RGB"), np.float32)[vi, ui]
        for nm in namen:
            x = np.asarray(werte[nm][0], np.float32).reshape(len(P), -1)[idx]
            summe[nm][0] += float(np.abs(x - gt).mean()) * len(idx)
            angepasst = np.empty_like(x)
            for c in range(x.shape[1]):
                # zentriert und geschlossen: polyfit ist bei kleiner Streuung um
                # grosse Werte (Temperaturen um 30 °C) schlecht konditioniert
                xc = x[:, c].astype(np.float64) - x[:, c].mean()
                ym = float(gt[:, c].mean())
                var = float((xc * xc).mean())
                g = float((xc * (gt[:, c] - ym)).mean()) / var if var > 1e-6 else 0.0
                angepasst[:, c] = g * xc + ym
            summe[nm][1] += float(np.abs(angepasst - gt).mean()) * len(idx)
        n_ges += len(idx)
        n_bilder += 1
    out = {"bilder": n_bilder, "punkte": int(n_ges), "methoden": {}}
    for nm in namen:
        out["methoden"][nm] = {"roh": summe[nm][0] / max(n_ges, 1),
                               "angepasst": summe[nm][1] / max(n_ges, 1)}
    _p(progress, 1.0, "Gegenprobe fertig")
    return out


def urteil(stats: dict, splat: str, direkt: str, einheit: str) -> str:
    """Die Gegenprobe als Satz fuers Protokoll."""
    if stats.get("punkte", 0) < 100:
        return ("Gegenprobe ohne Aussage: in den zurückgehaltenen Bildern waren zu wenige "
                "Punkte in beiden Einfärbungen sichtbar.")
    s = stats["methoden"][splat]["angepasst"]
    d = stats["methoden"][direkt]["angepasst"]
    rel = (d - s) / max(d, 1e-9) * 100
    wer = "Splat besser" if rel > 0 else "direkte Projektion besser"
    punkte = f"{stats['punkte']:,}".replace(",", " ")
    return (f"Gegenprobe an {stats['bilder']} zurückgehaltenen Bildern, "
            f"{punkte} Punkte: Splat {s:.2f}, direkt {d:.2f} {einheit} "
            f"Abweichung nach Belichtungsangleich (roh {stats['methoden'][splat]['roh']:.2f} "
            f"/ {stats['methoden'][direkt]['roh']:.2f}) — {wer} um {abs(rel):.1f} %.")


def gegenprobe(ordner, ak, welt, pf, direkt_w, temp, cancel, log) -> dict:
    """Splat und direkte Projektion an den Pruefbildern messen.

    ``pf`` aus :func:`punkt_farben`, ``direkt_w(idx, P, nrm)`` liefert (Werte,
    Maske) der direkten Projektion fuer die Punktprobe, ``temp``: verglichen
    wird die Temperatur statt RGB.
    """
    idx = probe(ak)
    P = welt[idx]
    nrm = ak["normal"][ak["index"][idx]]
    if temp:
        s_w, einheit = pf["temperatur"][idx][:, None], "°C"
    else:
        s_w, einheit = pf["rgb"][idx], "(0–255)"
    werte = {"splat": (s_w, pf["maske"][idx]), "direkt": direkt_w(idx, P, nrm)}
    stats = vergleich(ordner, ak, P, nrm, werte,
                      cancel=lambda: cancel.is_set())
    log(urteil(stats, "splat", "direkt", einheit))
    return stats


if __name__ == "__main__":
    import tempfile

    sys.path.insert(0, _REPO)
    from PIL import Image

    rng = np.random.default_rng(0)

    print("== Test 1: Anker, Index, wachsendes Raster ==")
    P = np.c_[rng.uniform(0, 10, (400_000, 2)), rng.normal(0, 0.005, 400_000)].astype(np.float32)
    ak = anker(P, voxel=0.05)
    # 10 m / 5 cm = 200 Zellen je Achse, eine Randzelle mehr durch den Versatz
    assert abs(ak["voxel"] - 0.05) < 1e-12 and 38_000 < len(ak["pos"]) <= 201 * 201, len(ak["pos"])
    assert np.abs(ak["pos"][ak["index"]] - P).max() < 0.05 * 1.8
    assert np.abs(ak["normal"][:, 2]).mean() > 0.98, "Normalen der Ebene nicht senkrecht"
    ak2 = anker(P, voxel=0.05, max_anker=10_000)
    assert len(ak2["pos"]) <= 10_000 and ak2["voxel"] > 0.09, (len(ak2["pos"]), ak2["voxel"])
    print(f"  {len(ak['pos'])} Anker bei 5 cm; auf 10.000 begrenzt: "
          f"{len(ak2['pos'])} bei {ak2['voxel'] * 100:.1f} cm")

    print("== Test 2: Kamerakette und Entzerrung = Projektion der Einfaerbung ==")
    from core.meander import find_pipeline
    from scipy.spatial.transform import Rotation
    find_pipeline()
    from colorize_pipeline import colorize as cz
    Q = Rotation.from_euler("zyx", [80, 3, -2], degrees=True).as_matrix()
    A = 1.7 * Q
    b = np.array([12.0, -4.0, 3.0])
    Rcw = Rotation.from_euler("xyz", [178, 4, 30], degrees=True).as_matrix()[None]
    Cc = np.array([[1.0, 2.0, 35.0]])
    cams = {"names": np.array(["a.jpg"]), "Rcw": Rcw, "tcw": -np.einsum("nij,nj->ni", Rcw, Cc),
            "size": np.array([[1600.0, 1200.0]]),
            "params": np.array([[1150.0, 790.0, 610.0, -0.08]]), "model": np.array("SIMPLE_RADIAL")}
    vm, info = ansichten_aus_colmap(cams, A, b)
    mx, my, K, gueltig = entzerrung("SIMPLE_RADIAL", cams["params"][0], cams["size"][0],
                                    (800, 600), skala=0.5)
    Pk = A @ (Cc[0] + rng.uniform(-15, 15, (500, 3)) * [1, 1, 0] - [0, 0, 30]).T
    Pk = Pk.T + b                                        # Punkte im Kartenrahmen
    pc = (Rcw[0] @ (np.linalg.inv(A) @ (Pk - b).T)).T + cams["tcw"][0]
    u, v, vorn, _ = cz._project(pc, cams["size"][0], cams["params"][0], "SIMPLE_RADIAL")
    pm = Pk @ vm[0, :3, :3].T + vm[0, :3, 3]
    uo = K[0, 0] * pm[:, 0] / pm[:, 2] + K[0, 2]
    vo = K[1, 1] * pm[:, 1] / pm[:, 2] + K[1, 2]
    ok = vorn & (uo > 2) & (uo < 798) & (vo > 2) & (vo < 598)
    import cv2
    mxs = cv2.remap(mx, uo[ok].astype(np.float32).reshape(1, -1) - 0.5,
                    vo[ok].astype(np.float32).reshape(1, -1) - 0.5, cv2.INTER_LINEAR)[0]
    fehler = np.abs((mxs + 0.5) - u[ok] * 0.5).max()
    print(f"  Massstab {info['massstab']:.3f}, {ok.sum()} Punkte, groesster Pixelfehler {fehler:.4f}")
    assert abs(info["massstab"] - 1.7) < 1e-9 and info["abweichung"] < 1e-9
    assert fehler < 0.02, fehler

    print("== Test 3: Abdeckung ==")
    g = np.mgrid[-5:5:0.05, -5:5:0.05].reshape(2, -1).T
    pos = np.c_[g, np.zeros(len(g))].astype(np.float32)
    R = np.diag([1.0, -1.0, -1.0])
    vmd = np.eye(4)
    vmd[:3, :3] = R
    vmd[:3, 3] = -R @ [0, 0, 10.0]
    Kd = np.array([[200.0, 0, 100], [0, 200.0, 100], [0, 0, 1]])
    m = abdeckung(pos, 0.05, vmd, Kd, 200, 200)
    assert m[50:150, 50:150].all() and m.mean() > 0.95, m.mean()
    pos_halb = pos[pos[:, 0] < 0]
    m2 = abdeckung(pos_halb, 0.05, vmd, Kd, 200, 200)
    assert m2[:, :95].all() and not m2[:, 110:].any(), "Rand der Karte falsch"
    print(f"  volle Ebene {m.mean() * 100:.0f} %, halbe Ebene endet an der Kante")

    print("== Test 4: Wuerfelseiten sind Drehungen mit der richtigen Blickachse ==")
    for name, Rf in _SEITEN.items():
        assert np.allclose(Rf @ Rf.T, np.eye(3)) and abs(np.linalg.det(Rf) - 1) < 1e-12, name
        achse = {"+z": [0, 0, 1], "+x": [1, 0, 0], "-x": [-1, 0, 0], "+y": [0, 1, 0],
                 "-y": [0, -1, 0]}[name]
        assert np.allclose(Rf[:, 2], achse), name
    print("  fünf Seiten, alle det +1")

    print("== Test 5: Gegenprobe erkennt die bessere Einfaerbung ==")
    tmp = tempfile.mkdtemp(prefix="splatvergleich_")
    os.makedirs(os.path.join(tmp, "bilder"))
    def muster(x, y):
        return np.stack([128 + 80 * np.sin(0.8 * x), 128 + 80 * np.cos(0.7 * y),
                         100 + 60 * np.sin(x + y)], -1).astype(np.float32)

    wahr = muster(g[:, 0], g[:, 1])
    ak3 = anker(pos, voxel=0.05)
    stichprobe = pos
    # Das Pruefbild dicht aus dem Bodenmuster, nicht aus den Punkten gerastert:
    # gerastert blieben durch Rundung Spalten leer, und wer ein Pixel daneben
    # abtastet, liest schwarz — eine echte Kamera hat keine Loecher.
    jj, ii = np.mgrid[0:200, 0:200] + 0.5
    farbe_akt = muster((ii - 100) / 20.0, -(jj - 100) / 20.0) * 0.8 + 10   # andere Belichtung
    Image.fromarray(farbe_akt.clip(0, 255).astype(np.uint8)).save(os.path.join(tmp, "bilder/0.png"))
    Image.fromarray((m * 255).astype(np.uint8)).save(os.path.join(tmp, "bilder/0_m.png"))
    _schreibe(tmp, ak3, pos, {"viewmat": [vmd], "K": [Kd], "breite": [200], "hoehe": [200],
                              "bild": ["bilder/0.png"], "maske": ["bilder/0_m.png"],
                              "holdout": [True], "quelle": ["0"]},
              {"t_lo": np.float32(np.nan), "t_hi": np.float32(np.nan)}, {"schritte": 7})
    with open(os.path.join(tmp, "param.json"), encoding="utf-8") as fh:
        pj = json.load(fh)
    assert pj["schritte"] == 7 and pj["toleranz_m"] == 0.30, "Sichtregeln fehlen im Datensatz"
    assert len(np.load(os.path.join(tmp, "punkte.npy"))) == len(pos)
    nrm = np.tile([0, 0, 1.0], (len(stichprobe), 1))
    gut = (wahr, np.ones(len(stichprobe), bool))
    schlecht = (np.clip(wahr + rng.normal(0, 25, wahr.shape), 0, 255), np.ones(len(stichprobe), bool))
    st = vergleich(tmp, ak3, stichprobe, nrm, {"splat": gut, "direkt": schlecht})
    print("  " + urteil(st, "splat", "direkt", "(0–255)"))
    assert st["methoden"]["splat"]["angepasst"] < 2.0 < st["methoden"]["direkt"]["angepasst"], st
    assert st["methoden"]["splat"]["roh"] > 5.0, "Belichtung des Pruefbilds nicht bemerkt"

    print("== Test 5b: Werte je Punkt aus dem Trainer ==")
    n5 = len(pos)
    wert = np.zeros((n5, 3), np.float16)
    wert[:, 0] = 0.5
    wert[:, 1] = 1.2                              # ueberbelichtet gerendert
    guete = np.zeros(n5, np.float16)
    guete[::2] = 0.9
    np.savez(os.path.join(tmp, "ergebnis.npz"), punkt_wert=wert, punkt_guete=guete)
    pf = punkt_farben(tmp, n5)
    assert pf["maske"].sum() == (n5 + 1) // 2 and tuple(pf["rgb"][0]) == (128, 255, 0)
    assert tuple(pf["rgb"][1]) == _FALLBACK, "ungesehener Punkt nicht grau"
    try:
        punkt_farben(tmp, n5 + 1)
        raise AssertionError("falsche Punktzahl nicht bemerkt")
    except RuntimeError:
        pass
    print("  RGB geklemmt, ungesehen grau, falsche Karte abgelehnt")

    print("== Test 6: Temperatur-Palette ==")
    t = np.array([20.0, 25.0, 30.0, np.nan], np.float32)
    rgb = temperatur_farben(t, np.isfinite(t))
    assert tuple(rgb[3]) == _FALLBACK and rgb[2].sum() > rgb[0].sum()
    print("  kalt dunkel, warm hell, ungesehen grau")

    print("== Test 6b: Trainingsfolge mit den Fortschrittsspannen der drei Jobs ==")
    import threading
    aufrufe = []

    def trainieren_ersatz(python, ordner, progress=None, cancel=None, log=None,
                          cpu=False, alle_bilder=False):
        aufrufe.append(("trainieren", python, ordner, alle_bilder, cancel()))
        for f in (0.0, 0.5, 1.0):
            progress(f, f"Splat: {f}")
        return {"bericht": {"posen": {"dreh_grad_median": 0.1, "dreh_grad_max": 0.5,
                                      "weg_m_median": 0.02, "weg_m_max": 0.07}},
                "pruefung": [] if alle_bilder else [{"psnr_angepasst": 30.0},
                                                    {"psnr_angepasst": 32.0}]}

    def stufen(von, bis):
        # so rechnet lauf den Fortschritt des Trainers um
        return [von + (bis - von) * f for f in (0.0, 0.5, 1.0)]

    trainieren_echt = trainieren
    trainieren = trainieren_ersatz
    try:
        # Maeander (eine Ebene), Onboard, gemeinsam
        for spannen in ((0.1, 0.45, 0.6, 0.97), (0.25, 0.55, 0.6, 0.97),
                        (0.3, 0.6, 0.62, 0.97)):
            a, b, c, d = spannen
            for pruefen in (True, False):
                aufrufe.clear()
                werte, zeilen = [], []
                cfg_t = {"pruefen": pruefen,
                         "param": {"schritte": 7, "sh_grad": 1, "posen": True}}

                def mit_gegenprobe(erg):
                    aufrufe.append(("gegenprobe", len(erg["pruefung"]), len(werte)))
                    return {"bilder": 2}

                erg, st6 = trainingsfolge("py", "ordner", cfg_t, spannen, mit_gegenprobe,
                                          lambda f, m: werte.append(f), threading.Event(),
                                          zeilen.append)
                if pruefen:
                    soll = stufen(a, b) + stufen(c, d)
                    soll_aufrufe = [("trainieren", "py", "ordner", False, False),
                                    ("gegenprobe", 2, 3),
                                    ("trainieren", "py", "ordner", True, False)]
                    assert st6 == {"bilder": 2} and erg["pruefung"] == [], st6
                    assert len(zeilen) == 3 and "PSNR 31.0 dB" in zeilen[1], zeilen
                else:
                    soll = stufen(a, d)
                    soll_aufrufe = [("trainieren", "py", "ordner", False, False)]
                    assert st6 is None and len(zeilen) == 2, (st6, zeilen)
                assert werte == soll, (spannen, pruefen, werte, soll)
                assert np.allclose(werte[::3] + werte[2::3],
                                   [a, c, b, d] if pruefen else [a, d], rtol=0, atol=1e-12)
                assert aufrufe == soll_aufrufe, (spannen, pruefen, aufrufe)
                assert zeilen[0].startswith("Splat-Posen nachgeführt: Drehung Median 0.100°")
            print(f"  {a}/{b}/{c}/{d}: mit Gegenprobe "
                  + " ".join(f"{w:.4f}" for w in stufen(a, b) + stufen(c, d)))
        cfg_t = {"pruefen": True, "param": {"schritte": 7, "posen": False}}
        zeilen = []
        erg, st6 = trainingsfolge("py", "ordner", cfg_t, (0.0, 0.5, 0.5, 1.0), None,
                                  lambda f, m: None, threading.Event(), zeilen.append)
        assert st6 is None and len(zeilen) == 1 and "Prüfbildern" in zeilen[0], zeilen
    finally:
        trainieren = trainieren_echt

    meta6 = splat_meta(ak3, {"param": {"schritte": 7, "sh_grad": 1, "posen": True}},
                       {"weg_m_max": 0.07})
    assert list(meta6) == ["anker", "raster_m", "schritte", "sh_grad", "posen"], meta6
    assert meta6["anker"] == len(ak3["pos"]) and meta6["posen"] == {"weg_m_max": 0.07}

    zeilen = []
    pf6 = {"rgb": wahr, "maske": np.ones(len(pos), bool)}
    st6 = gegenprobe(tmp, ak3, pos, pf6, lambda idx, P, nrm: (schlecht[0][idx],
                                                               schlecht[1][idx]),
                     False, threading.Event(), zeilen.append)
    idx6 = probe(ak3)
    st_direkt = vergleich(tmp, ak3, pos[idx6], ak3["normal"][ak3["index"][idx6]],
                          {"splat": (wahr[idx6], pf6["maske"][idx6]),
                           "direkt": (schlecht[0][idx6], schlecht[1][idx6])})
    assert st6 == st_direkt and zeilen == [urteil(st_direkt, "splat", "direkt", "(0–255)")]

    finde_echt = find_splat_python
    try:
        find_splat_python = lambda: (None, {"fehler": "keiner da"})  # noqa: E731
        try:
            interpreter()
            raise AssertionError("fehlender Interpreter nicht bemerkt")
        except RuntimeError as exc:
            assert str(exc) == hinweis({"fehler": "keiner da"}), exc
        bereit = {"torch": "2.7.0", "gsplat": True, "cuda": True, "gsplat_rechnet": True}
        find_splat_python = lambda: ("py", bereit)  # noqa: E731
        assert interpreter() == ("py", bereit)
    finally:
        find_splat_python = finde_echt
    print("  Aufrufe und Fortschritt wie in den Jobs, Gegenprobe und Interpreter gleich")

    if "--mit-training" in sys.argv:
        print("== Test 7: Training im Splat-Interpreter (CPU-Selbsttest) ==")
        py, info = find_splat_python()
        print(f"  {py}: {info}")
        if py:
            r = subprocess.run([py, SKRIPT, "--selbsttest"], capture_output=True, text=True)
            print(r.stdout[-800:])
            assert r.returncode == 0, r.stderr[-2000:]
    print("splat SELFTEST OK")
