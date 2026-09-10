"""Punktwolken-Einfaerbung aus Dual-Fisheye-Frames + Extrinsik-Werkzeuge.

- :func:`colorize`         faerbt die FAST-LIO-Punktwolke aus den zeitnaechsten
                           Kamera-Frames ein (Helligkeitsfilter, k Kandidaten).
- :func:`overlay_preview`  Pano + projizierte Lidar-Punkte (Tiefe->Turbo) zur
                           visuellen Extrinsik-Justage.
- :func:`auto_calibrate`   grobe Rotationssuche fuer T_imu_cam0 ueber
                           Gradienten-Korrelation Pano <-> Punktprojektion.

Konventionen (ARCHITECTURE.md):
  T_imu_cam0 = Pose von cam0 im IMU/Body-Frame  =>  p_imu = T_imu_cam0 @ p_cam0.
  T_cam0_cam1 aus calib (value0.T_imu_cam[1])   =>  p_cam1 = R01^T (p_cam0 - t01)
  (identisch zur Verwendung in core.stitcher::_build_luts).
  Pano-Strahl (cam0-Frame): [-sin(y)cos(x), -cos(y), sin(y)sin(x)],
  x = u/W*2pi - pi/2, y = v/H*pi  (wie stitcher::_pano_rays / refine_extrinsic).
Qt-frei, kein print (ausser Selbsttest).
"""

from __future__ import annotations

import functools
import itertools
import json
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

try:  # Paket-Import (App) vs. Direktstart des Selbsttests
    from .stitcher import SPLIT_X, DoubleSphereCamera, EquirectStitcher
except ImportError:  # pragma: no cover - nur "python3 core/colorizer.py"
    from stitcher import SPLIT_X, DoubleSphereCamera, EquirectStitcher  # type: ignore

# Fisheye-Bildkreis-Pruefung (ARCHITECTURE: Radius <= 740 px um Zentrum 760/760)
_CIRCLE_C = 760.0
_CIRCLE_R = 740.0
_FRAME_LRU = 8          # dekodierte Frame-Haelften je colorize-Lauf
_PROGRESS_EVERY = 10    # Scans zwischen progress_cb-Aufrufen

ProgressCb = Optional[Callable[[float, str], None]]


# ---------------------------------------------------------------- Basics

def _check_cancel(cancel) -> None:
    if cancel is not None and cancel.is_set():
        raise RuntimeError("Abgebrochen")


_EXPECTED_FRAME_SHAPE = (1520, 3040, 3)


def _check_camera_resolution(bag) -> None:
    """Frame 0 muss 3040x1520 BGR sein (Dual-Fisheye, s. ARCHITECTURE.md).

    Andere Aufloesungen passen nicht zur hinterlegten Kalibrierung
    (SPLIT_X/Bildkreis) und wuerden stillschweigend falsche Farben liefern.
    """
    img = bag.read_camera(0)
    shape = getattr(img, "shape", None)
    if shape == _EXPECTED_FRAME_SHAPE:
        return
    if shape is not None and len(shape) >= 2:
        got = f"{shape[1]}x{shape[0]}"
        if len(shape) != 3 or shape[2] != 3:
            got += " (kein 3-Kanal-Bild)"
    else:
        got = "unbekannt"
    raise RuntimeError(
        f"Unerwartete Kamera-Aufloesung {got} — erwartet 3040x1520 "
        "(Dual-Fisheye). Dieses Bag passt nicht zur hinterlegten "
        "Kalibrierung; Einfaerbung/Vorschau/Auto-Kalibrierung sind "
        "nicht moeglich.")


def rec_fingerprint(rec) -> str:
    """Identitaets-Fingerprint einer Aufzeichnung (fuer den Farb-Cache).

    Format: "<n_scans>_<n_points>_<stamps[0]:.6f>_<stamps[-1]:.6f>".
    colorize() schreibt den Wert als "rec_fingerprint" in colors/meta.json;
    die UI vergleicht ihn beim Laden mit der aktuellen Aufzeichnung, damit
    Farben einer alten/anderen Aufzeichnung nicht wiederverwendet werden.
    """
    n_scans = int(rec.n_scans)
    n_points = int(rec.n_points)
    s0 = float(rec.stamps[0]) if n_scans > 0 else 0.0
    s1 = float(rec.stamps[-1]) if n_scans > 0 else 0.0
    return f"{n_scans}_{n_points}_{s0:.6f}_{s1:.6f}"


def _inv_rigid(T: np.ndarray) -> np.ndarray:
    """Inverse einer starren 4x4-Transformation."""
    R = T[:3, :3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ T[:3, 3]
    return out


def _pose_matrix(pose_row: np.ndarray) -> np.ndarray:
    """poses.npy-Zeile (x y z qx qy qz qw) -> 4x4 T_world_imu."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_quat(pose_row[3:7]).as_matrix()
    T[:3, 3] = pose_row[0:3]
    return T


@functools.lru_cache(maxsize=4)
def _get_stitcher(calib_json: str, width: int) -> EquirectStitcher:
    return EquirectStitcher(calib_json, width=width)


def _candidate_frames(cam_stamps: np.ndarray, t: float,
                      max_dt: float, k: int) -> list[int]:
    """Frame-Indizes nach |stamp - t| sortiert; nur |dt| <= max_dt, max. k."""
    j = int(np.searchsorted(cam_stamps, t))
    lo = max(0, j - k - 2)
    hi = min(len(cam_stamps), j + k + 2)
    idx = np.arange(lo, hi)
    dt = np.abs(cam_stamps[idx] - t)
    keep = dt <= max_dt
    idx, dt = idx[keep], dt[keep]
    order = np.argsort(dt, kind="stable")
    return idx[order][:k].tolist()


# cv2.remap verlangt Zielbreite/-hoehe < SHRT_MAX (32767); Punktlisten werden
# daher als 2D-Karte fester Breite angeordnet (auffuellen, danach abschneiden).
_REMAP_MAX_W = 16384


def _sample_bgr(img_half: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilineares Farbsampling: (N,2) float32-Pixel -> (N,3) float32 BGR."""
    n = int(uv.shape[0])
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)
    w = min(n, _REMAP_MAX_W)
    rows = -(-n // w)                       # ceil(n / w)
    pad = rows * w - n
    mapx = np.ascontiguousarray(uv[:, 0], dtype=np.float32)
    mapy = np.ascontiguousarray(uv[:, 1], dtype=np.float32)
    if pad:
        mapx = np.pad(mapx, (0, pad))
        mapy = np.pad(mapy, (0, pad))
    s = cv2.remap(img_half, mapx.reshape(rows, w), mapy.reshape(rows, w),
                  cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return s.reshape(rows * w, -1)[:n].astype(np.float32)


# Randzonen-Guard (2026-07-26): Punkte jenseits ~86 Grad Blickwinkel (>700 px
# vom Zentrum) nicht einfaerben — am Bildkreisrand sind die Intrinsik-
# Restfehler am groessten, dort sampeln Silhouettenpunkte Himmel.
_COLOR_RMAX = 700.0
# zeitlicher Versatz der zusaetzlichen Konsens-Frames (andere Drohnenpose)
_CONSENSUS_SPREAD_S = 0.8


def _in_circle(uv: np.ndarray) -> np.ndarray:
    """Innerhalb des Fisheye-Bildkreises (und damit im Bild)."""
    du = uv[:, 0] - _CIRCLE_C
    dv = uv[:, 1] - _CIRCLE_C
    r = min(_CIRCLE_R, _COLOR_RMAX)
    return du * du + dv * dv <= r * r


def _equirect_uv(dirs_cam0: np.ndarray, norms: np.ndarray,
                 width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Richtungen im cam0-Frame -> Equirect-Pixel (u float, v float).

    Exakte Umkehrung von stitcher::_pano_rays:
    ray = [-sin(y)cos(x), -cos(y), sin(y)sin(x)], x = u/W*2pi - pi/2, y = v/H*pi.
    """
    n = np.where(norms > 0.0, norms, 1.0)
    y = np.arccos(np.clip(-dirs_cam0[:, 1] / n, -1.0, 1.0))
    x = np.arctan2(dirs_cam0[:, 2], -dirs_cam0[:, 0])
    u = (x + np.pi / 2.0) / (2.0 * np.pi) * width
    v = y / np.pi * height
    return np.mod(u, width), np.clip(v, 0.0, height - 1e-3)


def _strided_world_points(rec, stride: int,
                          scan_lo: int = 0, scan_hi: int | None = None
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Jeden stride-ten Punkt der Scans [scan_lo, scan_hi) ins Weltsystem.

    Rueckgabe: (Punkte float32 (M,3), globale Punktindizes int64 (M,)).
    """
    if scan_hi is None:
        scan_hi = rec.n_scans
    start = int(rec.offsets[scan_lo])
    stop = int(rec.offsets[scan_hi])
    stride = max(1, int(stride))
    gidx = np.arange(start, stop, stride, dtype=np.int64)
    if gidx.size == 0:
        return np.zeros((0, 3), dtype=np.float32), gidx
    pts = np.asarray(rec.points[gidx], dtype=np.float32)
    scan_of = np.searchsorted(rec.offsets, gidx, side="right") - 1
    R_all = Rotation.from_quat(rec.poses[:, 3:7]).as_matrix().astype(np.float32)
    t_all = rec.poses[:, 0:3].astype(np.float32)
    pw = np.einsum("nij,nj->ni", R_all[scan_of], pts) + t_all[scan_of]
    return pw, gidx


# ================================================================ colorize

@dataclass
class ColorizeParams:
    brightness_min: int = 20      # Graustufen-Schwelle: dunkler => ungueltig
    brightness_max: int = 235     # heller => ungueltig (ueberbelichtet)
    k_frames: int = 3             # bis zu K zeitnaechste Frames je Scan
    max_dt: float = 0.08          # s; Frames weiter weg ignorieren
    min_range: float = 0.5        # m; naeher an der Kamera nicht einfaerben
    T_imu_cam0: np.ndarray = field(default_factory=lambda: np.eye(4))


def colorize(rec, bag, calib_json: str, params: ColorizeParams,
             out_dir: str, progress_cb: ProgressCb = None, cancel=None) -> dict:
    """Faerbt alle Punkte der Aufzeichnung aus den Kamera-Frames ein.

    Je Scan werden bis zu k zeitnaechste Frames plus zwei zeitversetzte
    Konsens-Frames (~+-0,8 s, andere Drohnenpose) gesampelt; je Punkt gewinnt
    der Farb-MEDIAN aller gueltigen Samples (Double-Sphere gueltig, im
    Bildkreis inkl. Randzonen-Guard, Distanz >= min_range, Grauwert im
    Helligkeitsfenster). Der Median ueberstimmt einzelne Silhouetten-
    Fehlgriffe (Baumkrone/Dachkante sampelt Himmel durch Luecken), die beim
    frueheren Erster-Treffer-Verfahren dauerhaft in der Wolke landeten.
    Schreibt colors.bin (uint8 N x 3, RGB!), valid.bin (uint8 N), meta.json.
    """
    t_start = time.time()
    T = np.asarray(params.T_imu_cam0, dtype=np.float64)
    if T.shape != (4, 4):
        raise RuntimeError(f"T_imu_cam0 muss 4x4 sein, erhalten {T.shape}.")
    _check_camera_resolution(bag)
    cam0, cam1, T_cam0_cam1 = DoubleSphereCamera.from_calib(calib_json)
    R01 = np.ascontiguousarray(T_cam0_cam1[:3, :3], dtype=np.float32)
    t01 = T_cam0_cam1[:3, 3].astype(np.float32)
    T_cam0_imu = _inv_rigid(T)

    cam_stamps = bag.camera_stamps()
    N = rec.n_points
    S = rec.n_scans
    colors = np.zeros((N, 3), dtype=np.uint8)
    valid = np.zeros(N, dtype=np.uint8)

    bmin = float(params.brightness_min)
    bmax = float(params.brightness_max)
    min_r2 = float(params.min_range) ** 2

    # kleiner LRU fuer kontiguierliche Frame-Haelften (Dekodierung cacht BagReader)
    halves: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def get_halves(fidx: int) -> tuple[np.ndarray, np.ndarray]:
        if fidx in halves:
            halves.move_to_end(fidx)
            return halves[fidx]
        img = bag.read_camera(fidx)
        pair = (np.ascontiguousarray(img[:, :SPLIT_X]),
                np.ascontiguousarray(img[:, SPLIT_X:2 * SPLIT_X]))
        halves[fidx] = pair
        if len(halves) > _FRAME_LRU:
            halves.popitem(last=False)
        return pair

    R_all = Rotation.from_quat(rec.poses[:, 3:7]).as_matrix()
    n_valid = 0

    for i in range(S):
        _check_cancel(cancel)
        s, e = int(rec.offsets[i]), int(rec.offsets[i + 1])
        if e <= s:
            continue
        t_scan = float(rec.stamps[i])
        cands = _candidate_frames(cam_stamps, t_scan, params.max_dt,
                                  params.k_frames)
        if not cands:
            continue
        # Konsens-Frames mit anderer Drohnenpose (~+-0,8 s) fuer den Median
        for t_off in (-_CONSENSUS_SPREAD_S, _CONSENSUS_SPREAD_S):
            for f in _candidate_frames(cam_stamps, t_scan + t_off, 0.25, 1):
                if f not in cands:
                    cands.append(f)
        p_body = np.asarray(rec.points[s:e], dtype=np.float32)
        p_world = p_body @ R_all[i].T.astype(np.float32) \
            + rec.poses[i, 0:3].astype(np.float32)

        sample_ok: list[np.ndarray] = []
        sample_col: list[np.ndarray] = []
        for fidx in cands:
            T_wi = rec.interpolate_pose(float(cam_stamps[fidx]))
            if T_wi is None:
                continue
            M = (T_cam0_imu @ _inv_rigid(T_wi)).astype(np.float32)
            pc0 = p_world @ M[:3, :3].T + M[:3, 3]
            r2 = np.einsum("ni,ni->n", pc0, pc0)
            rng_ok = r2 >= min_r2

            uv0, geo0 = cam0.project(pc0)
            geo0 &= rng_ok
            geo0 &= _in_circle(uv0)

            need1 = ~geo0 & rng_ok
            pc1 = (pc0[need1] - t01) @ R01     # p_cam1 = R01^T (p_cam0 - t01)
            uv1, geo1 = cam1.project(pc1)
            geo1 &= _in_circle(uv1)

            half0, half1 = get_halves(fidx)
            ok = np.zeros(len(p_world), dtype=bool)
            col = np.zeros((len(p_world), 3), dtype=np.float32)

            if geo0.any():
                bgr = _sample_bgr(half0, uv0[geo0])
                g = 0.299 * bgr[:, 2] + 0.587 * bgr[:, 1] + 0.114 * bgr[:, 0]
                bright = (g >= bmin) & (g <= bmax)
                sel = np.flatnonzero(geo0)[bright]
                ok[sel] = True
                col[sel] = bgr[bright]
            if geo1.any():
                bgr = _sample_bgr(half1, uv1[geo1])
                g = 0.299 * bgr[:, 2] + 0.587 * bgr[:, 1] + 0.114 * bgr[:, 0]
                bright = (g >= bmin) & (g <= bmax)
                sel = np.flatnonzero(need1)[geo1][bright]
                ok[sel] = True
                col[sel] = bgr[bright]
            if ok.any():
                sample_ok.append(ok)
                sample_col.append(col)

        if sample_ok:
            O = np.stack(sample_ok)                     # (k,n)
            C = np.stack(sample_col)                    # (k,n,3)
            C[~O] = np.nan
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(C, axis=0)           # (n,3) BGR
            any_ok = O.any(axis=0)
            gsel = np.flatnonzero(any_ok) + s
            colors[gsel] = np.nan_to_num(
                med[any_ok])[:, ::-1].astype(np.uint8)  # BGR -> RGB
            valid[gsel] = 1
            n_valid += int(any_ok.sum())

        if progress_cb is not None and (i % _PROGRESS_EVERY == 0 or i == S - 1):
            progress_cb((i + 1) / S, f"Faerbe Scan {i + 1}/{S}")

    os.makedirs(out_dir, exist_ok=True)
    colors.tofile(os.path.join(out_dir, "colors.bin"))
    valid.tofile(os.path.join(out_dir, "valid.bin"))
    frac = float(n_valid) / max(N, 1)
    meta = {
        "extrinsic": T.tolist(),
        "brightness_min": params.brightness_min,
        "brightness_max": params.brightness_max,
        "k_frames": params.k_frames,
        "max_dt": params.max_dt,
        "min_range": params.min_range,
        "calib_json": str(calib_json),
        "bag": getattr(bag, "bag_path", None),
        "rec_fingerprint": rec_fingerprint(rec),
        "n_points": int(N),
        "n_valid": int(n_valid),
        "frac_valid": frac,
        "runtime_s": round(time.time() - t_start, 2),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    tmp = os.path.join(out_dir, "meta.json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    os.replace(tmp, os.path.join(out_dir, "meta.json"))
    if progress_cb is not None:
        progress_cb(1.0, f"Einfaerbung fertig: {n_valid}/{N} Punkte gueltig")
    return {"n_valid": int(n_valid), "frac_valid": frac, "out_dir": out_dir}


# ========================================================== overlay_preview

def overlay_preview(rec, bag, calib_json: str, T_imu_cam0: np.ndarray,
                    frame_idx: int, stride: int = 50) -> np.ndarray:
    """Pano (1920x960) + projizierte Karte (Tiefe->Turbo, alpha 0.7), BGR."""
    T = np.asarray(T_imu_cam0, dtype=np.float64)
    if T.shape != (4, 4):
        raise RuntimeError(f"T_imu_cam0 muss 4x4 sein, erhalten {T.shape}.")
    _check_camera_resolution(bag)
    st = _get_stitcher(str(calib_json), 1920)
    W, H = st.width, st.height

    cam_stamps = bag.camera_stamps()
    if not 0 <= int(frame_idx) < len(cam_stamps):
        raise RuntimeError(f"Frame {frame_idx} ausserhalb 0..{len(cam_stamps) - 1}.")
    t_frame = float(cam_stamps[int(frame_idx)])
    T_wi = rec.interpolate_pose(t_frame)
    if T_wi is None:
        raise RuntimeError(
            f"Keine LIO-Pose zum Frame-Zeitpunkt t={t_frame:.3f} s "
            "(Frame liegt ausserhalb der Aufzeichnung).")

    pano = st.stitch(bag.read_camera(int(frame_idx)))

    pw, _ = _strided_world_points(rec, stride)
    M = (_inv_rigid(T) @ _inv_rigid(T_wi)).astype(np.float32)
    pc = pw @ M[:3, :3].T + M[:3, 3]
    d = np.linalg.norm(pc, axis=1)
    keep = d > 0.3
    pc, d = pc[keep], d[keep]
    if pc.shape[0] == 0:
        return pano

    u, v = _equirect_uv(pc, d, W, H)
    order = np.argsort(-d)                     # fern zuerst, nah uebermalt
    u = u[order].astype(np.int32)
    v = v[order].astype(np.int32)
    d = d[order]

    lo, hi = np.percentile(d, [2.0, 98.0])
    dn = np.clip((d - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    turbo = cv2.applyColorMap((dn * 255).astype(np.uint8).reshape(-1, 1),
                              cv2.COLORMAP_TURBO).reshape(-1, 3)

    canvas = np.zeros_like(pano)
    mask = np.zeros((H, W), dtype=bool)
    for du, dv in ((0, 0), (1, 0), (0, 1), (1, 1)):    # 2x2-px-Punkte
        uu = (u + du) % W
        vv = np.minimum(v + dv, H - 1)
        canvas[vv, uu] = turbo
        mask[vv, uu] = True

    out = pano.copy()
    out[mask] = (0.7 * canvas[mask] + 0.3 * pano[mask]).astype(np.uint8)
    return out


# =========================================================== auto_calibrate

_AC_WIDTH = 640                 # Equirect-Aufloesung fuer den Kantenscore
_AC_PTS_PER_FRAME = 30000       # Zielpunktzahl je Bewertungs-Frame
_AC_TIME_WINDOW = 4.0           # s; nur Scans nahe am Frame (weniger Verdeckung)
_AC_MAX_RANGE = 40.0            # m


class _EdgeScoreContext:
    """Kantenkorrelations-Score (wie in ARCHITECTURE.md beschrieben).

    Je Frame: Gradientenbild des Panos (640x320, grau) und Lidar-Punkte
    (Scans im Zeitfenster) im IMU-Frame des Frame-Zeitpunkts. score(R) misst
    die Pearson-Korrelation der Gradientenbilder von Pano und projizierter
    inverser-Tiefe-Dichte.

    HINWEIS: Auf den seg0-Referenzdaten ist diese Metrik flach und mehrdeutig
    (siehe Abschlussbericht); auto_calibrate nutzt daher primaer den
    Foto-Konsistenz-Score (:class:`_PhotoScoreContext`). Diese Klasse bleibt
    fuer Vergleich/Diagnose erhalten.
    """

    def __init__(self, rec, bag, calib_json: str, frame_indices: Sequence[int],
                 cancel=None):
        st = _get_stitcher(str(calib_json), _AC_WIDTH)
        self.W, self.H = st.width, st.height
        cam_stamps = bag.camera_stamps()
        self.frames: list[dict] = []
        for fidx in frame_indices:
            _check_cancel(cancel)
            t_f = float(cam_stamps[int(fidx)])
            T_wi = rec.interpolate_pose(t_f)
            if T_wi is None:
                continue
            pano = st.stitch(bag.read_camera(int(fidx)))
            gray = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gray = cv2.GaussianBlur(gray, (0, 0), 1.0)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
            grad = cv2.magnitude(gx, gy)

            a = int(np.searchsorted(rec.stamps, t_f - _AC_TIME_WINDOW))
            b = int(np.searchsorted(rec.stamps, t_f + _AC_TIME_WINDOW))
            b = max(b, a + 1)
            n_total = int(rec.offsets[b] - rec.offsets[a])
            stride = max(1, n_total // _AC_PTS_PER_FRAME)
            pw, _ = _strided_world_points(rec, stride, a, b)
            Mi = _inv_rigid(T_wi).astype(np.float32)
            p_imu = pw @ Mi[:3, :3].T + Mi[:3, 3]
            d = np.linalg.norm(p_imu, axis=1)
            keep = (d > 0.5) & (d < _AC_MAX_RANGE)
            p_imu, d = p_imu[keep], d[keep]
            self.frames.append({
                "idx": int(fidx),
                "grad_pano": grad,
                "p_imu": np.ascontiguousarray(p_imu, dtype=np.float32),
                "d": d.astype(np.float32),
                "w": (1.0 / d).astype(np.float64),
            })
        if not self.frames:
            raise RuntimeError(
                "Auto-Kalibrierung: keine Frames mit gueltiger LIO-Pose gefunden.")

    def score(self, R_imu_cam0: np.ndarray) -> float:
        """Mittlere Pearson-Korrelation der Gradientenbilder ueber alle Frames."""
        W, H = self.W, self.H
        R = np.ascontiguousarray(R_imu_cam0, dtype=np.float32)
        total = 0.0
        for fr in self.frames:
            pc0 = fr["p_imu"] @ R                 # == R^T @ p (zeilenweise)
            u, v = _equirect_uv(pc0, fr["d"], W, H)
            flat = v.astype(np.int32) * W + u.astype(np.int32)
            img = np.bincount(flat, weights=fr["w"], minlength=H * W)
            img = img.reshape(H, W).astype(np.float32)
            nz = img > 0.0
            if nz.any():
                img = np.log1p(img / float(img[nz].mean()))
            img = cv2.GaussianBlur(img, (0, 0), 1.5)
            gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
            grad = cv2.magnitude(gx, gy)
            total += _pearson(grad, fr["grad_pano"])
        return total / len(self.frames)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    x = a.ravel().astype(np.float64)
    y = b.ravel().astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    den = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / den) if den > 0.0 else 0.0


def _rot_distance_deg(Ra: Rotation, Rb: Rotation) -> float:
    return float(np.degrees((Ra.inv() * Rb).magnitude()))


def _top_distinct(scored: list[tuple[float, Rotation]], k: int,
                  min_sep_deg: float) -> list[tuple[float, Rotation]]:
    """Beste k Kandidaten mit paarweisem Winkelabstand >= min_sep_deg."""
    out: list[tuple[float, Rotation]] = []
    for sc, rot in sorted(scored, key=lambda x: -x[0]):
        if all(_rot_distance_deg(rot, r) >= min_sep_deg for _, r in out):
            out.append((sc, rot))
            if len(out) >= k:
                break
    return out


def _default_score_frames(rec, bag, n: int = 5) -> list[int]:
    """~n gleichverteilte Kamera-Frames innerhalb der Aufzeichnung (mit Rand)."""
    cs = bag.camera_stamps()
    t0 = float(rec.stamps[0]) + 2.0
    t1 = float(rec.stamps[-1]) - 2.0
    if t1 <= t0:
        t0, t1 = float(rec.stamps[0]), float(rec.stamps[-1])
    targets = np.linspace(t0, t1, n)
    idx = sorted({int(np.clip(np.searchsorted(cs, t), 0, len(cs) - 1))
                  for t in targets})
    return idx


_PC_DOWNSCALE = 2               # Fisheye-Graubilder 1/2-aufgeloest
_PC_PTS_PER_PAIR = 20000
_PC_MIN_SAMPLES = 3000          # Mindestpunkte je Paar fuer gueltige ZNCC
_PC_GRAY_LO, _PC_GRAY_HI = 15.0, 245.0
_PC_PARTNER_DT = (0.8, 2.6)     # s; Partner-Frame-Suchfenster


class _PhotoScoreContext:
    """Foto-Konsistenz-Score fuer die Rotationssuche (primaere Metrik).

    Fuer Frame-Paare (a, b) mit deutlicher Relativbewegung werden dieselben
    Lidar-Weltpunkte in beide Kamerabilder projiziert (cam0, sonst cam1) und
    die Grauwerte verglichen. Nur die korrekte Extrinsik trifft in beiden
    Frames dieselbe Oberflaeche -> hohe ZNCC. Score = Mittel ueber Paare.
    """

    def __init__(self, rec, bag, calib_json: str,
                 anchor_frames: Sequence[int], cancel=None):
        self.cam0, self.cam1, T01 = DoubleSphereCamera.from_calib(calib_json)
        self._R01 = np.ascontiguousarray(T01[:3, :3], dtype=np.float32)
        self._t01 = T01[:3, 3].astype(np.float32)
        cs = bag.camera_stamps()

        def gray_halves(fidx: int) -> tuple[np.ndarray, np.ndarray]:
            img = bag.read_camera(fidx)
            out = []
            for sl in (img[:, :SPLIT_X], img[:, SPLIT_X:2 * SPLIT_X]):
                g = cv2.cvtColor(sl, cv2.COLOR_BGR2GRAY)
                out.append(cv2.resize(
                    g, None, fx=1.0 / _PC_DOWNSCALE, fy=1.0 / _PC_DOWNSCALE,
                    interpolation=cv2.INTER_AREA).astype(np.float32))
            return out[0], out[1]

        self.pairs: list[dict] = []
        self.pair_frames: list[tuple[int, int]] = []
        for fa in anchor_frames:
            _check_cancel(cancel)
            t_a = float(cs[int(fa)])
            T_a = rec.interpolate_pose(t_a)
            if T_a is None:
                continue
            # Partner mit maximaler Relativbewegung (Rotation bevorzugt)
            rot_a = Rotation.from_matrix(T_a[:3, :3])
            best: tuple[float, float] | None = None
            for dt in np.arange(*_PC_PARTNER_DT, 0.2):
                T_b = rec.interpolate_pose(t_a + dt)
                if T_b is None:
                    continue
                rel = np.degrees((rot_a.inv()
                                  * Rotation.from_matrix(T_b[:3, :3])).magnitude())
                q = rel + 20.0 * float(np.linalg.norm(T_b[:3, 3] - T_a[:3, 3]))
                if best is None or q > best[0]:
                    best = (q, t_a + dt)
            if best is None:
                continue
            fb = int(np.clip(np.searchsorted(cs, best[1]), 0, len(cs) - 1))
            t_b = float(cs[fb])
            T_b = rec.interpolate_pose(t_b)
            if T_b is None or fb == int(fa):
                continue

            a = int(np.searchsorted(rec.stamps, min(t_a, t_b) - 0.5))
            b = max(int(np.searchsorted(rec.stamps, max(t_a, t_b) + 0.5)), a + 1)
            stride = max(1, int(rec.offsets[b] - rec.offsets[a]) // _PC_PTS_PER_PAIR)
            pw, _ = _strided_world_points(rec, stride, a, b)
            Ma = _inv_rigid(T_a).astype(np.float32)
            Mb = _inv_rigid(T_b).astype(np.float32)
            qa = pw @ Ma[:3, :3].T + Ma[:3, 3]
            qb = pw @ Mb[:3, :3].T + Mb[:3, 3]
            keep = ((np.linalg.norm(qa, axis=1) > 0.7)
                    & (np.linalg.norm(qb, axis=1) > 0.7)
                    & (np.linalg.norm(qa, axis=1) < _AC_MAX_RANGE))
            ga = gray_halves(int(fa))
            gb = gray_halves(fb)
            self.pairs.append(dict(
                qa=np.ascontiguousarray(qa[keep]),
                qb=np.ascontiguousarray(qb[keep]), ga=ga, gb=gb))
            self.pair_frames.append((int(fa), fb))
        if len(self.pairs) < 3:
            raise RuntimeError(
                "Auto-Kalibrierung: zu wenige Frame-Paare mit gueltiger LIO-Pose.")

    @staticmethod
    def _bilinear(g: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.clip(u, 0.0, g.shape[1] - 1.001)
        v = np.clip(v, 0.0, g.shape[0] - 1.001)
        u0 = u.astype(np.int32)
        v0 = v.astype(np.int32)
        fu = u - u0
        fv = v - v0
        return (g[v0, u0] * (1 - fu) * (1 - fv) + g[v0, u0 + 1] * fu * (1 - fv)
                + g[v0 + 1, u0] * (1 - fu) * fv + g[v0 + 1, u0 + 1] * fu * fv)

    def _sample(self, gs: tuple[np.ndarray, np.ndarray],
                pc0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Grauwerte fuer Punkte im cam0-Frame; cam0 zuerst, sonst cam1."""
        uv0, val0 = self.cam0.project(pc0)
        val0 &= _in_circle(uv0)
        s = np.zeros(pc0.shape[0], dtype=np.float32)
        val = val0.copy()
        if val0.any():
            s[val0] = self._bilinear(gs[0], uv0[val0, 0] / _PC_DOWNSCALE,
                                     uv0[val0, 1] / _PC_DOWNSCALE)
        rest = ~val0
        pc1 = (pc0[rest] - self._t01) @ self._R01
        uv1, val1 = self.cam1.project(pc1)
        val1 &= _in_circle(uv1)
        idx = np.flatnonzero(rest)[val1]
        if idx.size:
            s[idx] = self._bilinear(gs[1], uv1[val1, 0] / _PC_DOWNSCALE,
                                    uv1[val1, 1] / _PC_DOWNSCALE)
        val[idx] = True
        val &= (s > _PC_GRAY_LO) & (s < _PC_GRAY_HI)
        return s, val

    def score(self, R_imu_cam0: np.ndarray) -> float:
        R = np.ascontiguousarray(R_imu_cam0, dtype=np.float32)
        total = 0.0
        n_ok = 0
        for pd in self.pairs:
            sa, va = self._sample(pd["ga"], pd["qa"] @ R)
            sb, vb = self._sample(pd["gb"], pd["qb"] @ R)
            m = va & vb
            if int(m.sum()) < _PC_MIN_SAMPLES:
                continue
            a = sa[m] - sa[m].mean()
            b = sb[m] - sb[m].mean()
            den = np.sqrt(float((a * a).sum()) * float((b * b).sum()))
            if den <= 0.0:
                continue
            total += float((a * b).sum()) / den
            n_ok += 1
        if n_ok < max(3, len(self.pairs) // 2):
            return -1.0      # zu wenig Abdeckung: Kandidat nicht bewertbar
        return total / n_ok


def auto_calibrate(rec, bag, calib_json: str, T_init: np.ndarray | None = None,
                   frames: list[int] | None = None,
                   progress_cb: ProgressCb = None, cancel=None
                   ) -> tuple[np.ndarray, float]:
    """Grobe Rotationssuche fuer T_imu_cam0 (Translation = 0).

    Stufen: volles SO(3)-Gitter (30 deg, plus Prior-Seeds fuer die kopfueber
    montierte Kamera) -> Top-5 -> Koordinaten-Hillclimb mit Schrittweiten
    10/3/1 deg. Score = Foto-Konsistenz (ZNCC der Grauwerte derselben
    Lidar-Punkte, reprojiziert in Frame-Paare mit Relativbewegung); ``frames``
    dient als Anker-Frames der Paare. Die verfeinerten Finalisten werden auf
    einem unabhaengigen Validierungs-Paarsatz bewertet (gegen Overfitting);
    zurueckgegeben wird (T_imu_cam0 4x4, Validierungs-Score). Score < ~0.3
    bedeutet schwache/unsichere Ausrichtung -> UI warnt.
    """
    _check_camera_resolution(bag)
    if frames is None:
        frames = _default_score_frames(rec, bag, 10)
    if progress_cb is not None:
        progress_cb(0.0, "Auto-Kalibrierung: bereite Frame-Paare vor")
    ctx = _PhotoScoreContext(rec, bag, calib_json, frames, cancel=cancel)
    # Validierungs-Anker: zeitliche Zwischenpunkte der Optimierungs-Anker
    fr = sorted(int(f) for f in frames)
    val_frames = sorted({(a + b) // 2 for a, b in zip(fr[:-1], fr[1:])
                         if (a + b) // 2 not in fr})
    try:
        vctx: _PhotoScoreContext | None = _PhotoScoreContext(
            rec, bag, calib_json, val_frames, cancel=cancel)
    except RuntimeError:
        vctx = None

    def evaluate(rotations: list[Rotation], stage: str, frac0: float,
                 frac1: float) -> list[tuple[float, Rotation]]:
        scored: list[tuple[float, Rotation]] = []
        n = len(rotations)
        for i, rot in enumerate(rotations):
            if i % 16 == 0:
                _check_cancel(cancel)
                if progress_cb is not None:
                    progress_cb(frac0 + (frac1 - frac0) * i / max(n, 1),
                                f"Auto-Kalibrierung: {stage} ({i + 1}/{n})")
            scored.append((ctx.score(rot.as_matrix()), rot))
        return scored

    # ---- Stufe 1: grobes SO(3)-Gitter (30 deg) + Seeds ----------------------
    grid: list[Rotation] = []
    for yaw in range(-180, 180, 30):
        for pitch in range(-90, 91, 30):
            for roll in range(-180, 180, 30):
                grid.append(Rotation.from_euler(
                    "ZYX", [yaw, pitch, roll], degrees=True))
    # Prioren: Kamera kopfueber montiert -> Identitaet, Roll 180 und die
    # Seitwaerts-Familien Roll +-90, jeweils mit Yaw 0/90/180/270.
    for yaw in (0, 90, 180, 270):
        for roll in (0, 90, 180, -90):
            grid.append(Rotation.from_euler(
                "ZYX", [yaw, 0, roll], degrees=True))
    if T_init is not None:
        Ti = np.asarray(T_init, dtype=np.float64)
        if Ti.shape != (4, 4):
            raise RuntimeError(f"T_init muss 4x4 sein, erhalten {Ti.shape}.")
        grid.append(Rotation.from_matrix(Ti[:3, :3]))

    scored = evaluate(grid, "Grobsuche 30 deg", 0.02, 0.5)
    top = _top_distinct(scored, 5, 20.0)

    # ---- Stufe 2-4: Koordinaten-Hillclimb (10/3/1 deg) um die Top-5 ----------
    finalists: list[tuple[float, Rotation]] = []
    n_top = len(top)
    for ci, (sc, rot) in enumerate(top):
        _check_cancel(cancel)
        cur_sc, cur_rot = sc, rot
        for step_i, step in enumerate((10.0, 3.0, 1.0)):
            if progress_cb is not None:
                frac = 0.5 + 0.45 * (ci * 3 + step_i) / (n_top * 3)
                progress_cb(frac, (f"Auto-Kalibrierung: Verfeinerung "
                                   f"Kandidat {ci + 1}/{n_top}, {step:.0f} deg"))
            for _ in range(40):          # Sicherheitslimit
                _check_cancel(cancel)
                improved = False
                for dy, dp, dr in itertools.product((-step, 0.0, step), repeat=3):
                    if dy == dp == dr == 0.0:
                        continue
                    cand = cur_rot * Rotation.from_euler(
                        "ZYX", [dy, dp, dr], degrees=True)
                    s = ctx.score(cand.as_matrix())
                    if s > cur_sc:
                        cur_sc, cur_rot = s, cand
                        improved = True
                if not improved:
                    break
        finalists.append((cur_sc, cur_rot))

    # ---- Auswahl auf unabhaengigen Validierungs-Paaren (gegen Overfitting) ---
    if progress_cb is not None:
        progress_cb(0.96, "Auto-Kalibrierung: Validierung der Finalisten")
    best_sc = -np.inf
    best_rot = finalists[0][1]
    for opt_sc, rot in finalists:
        _check_cancel(cancel)
        sc = vctx.score(rot.as_matrix()) if vctx is not None else opt_sc
        if sc > best_sc:
            best_sc, best_rot = sc, rot

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = best_rot.as_matrix()
    if progress_cb is not None:
        ypr = best_rot.as_euler("ZYX", degrees=True)
        progress_cb(1.0, (f"Auto-Kalibrierung fertig: Yaw={ypr[0]:.1f} "
                          f"Pitch={ypr[1]:.1f} Roll={ypr[2]:.1f} deg, "
                          f"Score={best_sc:.3f}"))
    return T, float(best_sc)


# ======================================================================
# Selbsttest (gegen seg0, siehe ARCHITECTURE.md / Teststrategie)
# ======================================================================
if __name__ == "__main__":
    import sys

    if __package__ in (None, ""):
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.bag_reader import BagReader
    from core.project import Project
    from core.recording import Recording

    EVID = ("/tmp/super360_modtests/"
            "colorizer")
    BAG = "/home/lena/RosBagSuper_Gui/rosbag_2026-07-11_15-37-07_seg0"
    CALIB = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "calib", "calib_new_vign.json")
    CACHE = "/home/lena/RosBagSuper_Gui/rosbag_suite/cache/rosbag_2026-07-11_15-37-07_seg0"
    os.makedirs(EVID, exist_ok=True)

    rec = Recording.load(os.path.join(CACHE, "recording"))
    bag = BagReader(BAG)
    print(f"recording: {rec.n_scans} scans, {rec.n_points} points; "
          f"bag: {len(bag.camera_stamps())} frames")

    # ---------- Test 1: auto_calibrate ----------
    print("== Test 1: auto_calibrate ==")
    last_msg = [""]

    def prog(f, m):
        if m != last_msg[0]:
            last_msg[0] = m
            print(f"  [{f * 100:5.1f}%] {m}")

    t0 = time.perf_counter()
    T_found, score = auto_calibrate(rec, bag, CALIB, progress_cb=prog)
    rt_calib = time.perf_counter() - t0
    ypr = Rotation.from_matrix(T_found[:3, :3]).as_euler("ZYX", degrees=True)
    print(f"  found: yaw={ypr[0]:.2f} pitch={ypr[1]:.2f} roll={ypr[2]:.2f} deg  "
          f"score={score:.4f}  runtime={rt_calib:.1f} s")
    print(f"  T_imu_cam0=\n{np.array_str(T_found, precision=4, suppress_small=True)}")

    # Vergleichs-Scores: Foto-Konsistenz (primaer) + Kantenscore (ARCHITECTURE)
    frames6 = _default_score_frames(rec, bag, 6)
    pctx = _PhotoScoreContext(rec, bag, CALIB, frames6)
    s_id = pctx.score(np.eye(3))
    s_roll180 = pctx.score(Rotation.from_euler("ZYX", [0, 0, 180], degrees=True).as_matrix())
    s_found = pctx.score(T_found[:3, :3])
    print(f"  Foto-ZNCC: identity={s_id:.4f}  roll180={s_roll180:.4f}  "
          f"found={s_found:.4f}")
    frames5 = _default_score_frames(rec, bag, 5)
    ectx = _EdgeScoreContext(rec, bag, CALIB, frames5)
    print(f"  Kantenscore (ARCHITECTURE-Metrik): identity={ectx.score(np.eye(3)):.4f}  "
          f"found={ectx.score(T_found[:3, :3]):.4f}")

    # Overlays: 3 Frames, gefundene Extrinsik vs. Identitaet
    ov_frames = [frames5[0], frames5[len(frames5) // 2], frames5[-1]]
    for f in ov_frames:
        ov = overlay_preview(rec, bag, CALIB, T_found, f, stride=50)
        cv2.imwrite(os.path.join(EVID, f"overlay_found_f{f:06d}.png"), ov)
        ov_id = overlay_preview(rec, bag, CALIB, np.eye(4), f, stride=50)
        cv2.imwrite(os.path.join(EVID, f"overlay_identity_f{f:06d}.png"), ov_id)
    print(f"  overlays -> {EVID}/overlay_{{found,identity}}_f*.png")

    # ---------- Test 2: colorize (voll) ----------
    print("== Test 2: colorize seg0 (voll, k=3, Helligkeit 20..235) ==")
    params = ColorizeParams(T_imu_cam0=T_found)
    colors_dir = os.path.join(CACHE, "colors")
    t0 = time.perf_counter()
    res = colorize(rec, bag, CALIB, params, colors_dir, progress_cb=None)
    rt_col = time.perf_counter() - t0
    print(f"  n_valid={res['n_valid']}  frac_valid={res['frac_valid']:.4f}  "
          f"runtime={rt_col:.1f} s  out={res['out_dir']}")

    # ---------- Test 3: Beweis-Render ----------
    print("== Test 3: kolorierte Wolke (Voxel 5 cm) ==")
    import open3d as o3d

    colors_rgb = np.fromfile(os.path.join(colors_dir, "colors.bin"),
                             dtype=np.uint8).reshape(-1, 3)
    valid_mask = np.fromfile(os.path.join(colors_dir, "valid.bin"),
                             dtype=np.uint8).astype(bool)
    world = rec.world_points()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(world[valid_mask].astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors_rgb[valid_mask] / 255.0)
    pcd = pcd.voxel_down_sample(0.05)
    ply_path = os.path.join(EVID, "colored_voxel5cm.ply")
    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"  PLY: {ply_path}  ({len(pcd.points)} Punkte)")

    P = np.asarray(pcd.points)
    C = (np.asarray(pcd.colors) * 255).astype(np.uint8)

    rendered = False
    try:
        from open3d.visualization import rendering
        r = rendering.OffscreenRenderer(1280, 960)
        r.scene.set_background([0.12, 0.12, 0.14, 1.0])
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = 3.0
        # Decke ausblenden, sonst verdeckt sie die Innenansicht
        zcut = np.percentile(P[:, 2], 88.0)
        keep = P[:, 2] < zcut
        pcd_view = o3d.geometry.PointCloud()
        pcd_view.points = o3d.utility.Vector3dVector(P[keep])
        pcd_view.colors = o3d.utility.Vector3dVector(C[keep] / 255.0)
        r.scene.add_geometry("pcd", pcd_view, mat)
        med = np.median(P[keep], axis=0)
        views = [  # Uebersicht schraeg von oben + Innenansicht (Halle)
            ("persp1", med + np.array([8.0, -10.0, 7.0]), med),
            ("persp2", med + np.array([1.4, -2.2, 3.3]),
             med + np.array([-4.6, 2.8, -0.2]))]
        for name, eye, ctr in views:
            r.setup_camera(70.0, ctr.astype(np.float32), eye.astype(np.float32),
                           np.array([0.0, 0.0, 1.0], dtype=np.float32))
            img = np.asarray(r.render_to_image())
            cv2.imwrite(os.path.join(EVID, f"render_{name}.png"),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        rendered = True
        print(f"  OffscreenRenderer: render_persp1/2.png -> {EVID}")
    except Exception as exc:  # noqa: BLE001 - Renderer optional
        print(f"  OffscreenRenderer nicht verfuegbar ({exc}); nutze Ortho-Fallback")

    def ortho_png(P, C, au, av, aheight, flip_v, path, res=0.03, drop_top=None):
        pp, cc = P, C
        if drop_top is not None:
            zmax = np.percentile(pp[:, drop_top], 88.0)
            keep = pp[:, drop_top] < zmax
            pp, cc = pp[keep], cc[keep]
        order = np.argsort(pp[:, aheight])       # hoechste zuletzt gemalt
        pp, cc = pp[order], cc[order]
        u = ((pp[:, au] - pp[:, au].min()) / res).astype(np.int32)
        v = ((pp[:, av] - pp[:, av].min()) / res).astype(np.int32)
        img = np.zeros((v.max() + 1, u.max() + 1, 3), dtype=np.uint8)
        img[v, u] = cc[:, ::-1]                  # RGB -> BGR
        if flip_v:
            img = img[::-1]
        cv2.imwrite(path, img)
        return img.shape

    sh1 = ortho_png(P, C, 0, 1, 2, True,
                    os.path.join(EVID, "ortho_top.png"), drop_top=2)
    sh2 = ortho_png(P, C, 0, 2, 1, True,
                    os.path.join(EVID, "ortho_side.png"))
    print(f"  Ortho-Fallback: top {sh1}, side {sh2} -> {EVID}")

    # ---------- Test 4: Extrinsik speichern ----------
    prj = Project(BAG)
    prj.save_extrinsic(T_found)
    T_back = prj.load_extrinsic()
    assert T_back is not None and np.abs(T_back - T_found).max() < 1e-12
    print(f"== Test 4: extrinsic.json gespeichert: {prj.extrinsic_json()}")

    with open(os.path.join(EVID, "selftest_metrics.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"auto_calibrate: ypr={np.round(ypr, 2).tolist()} score={score:.4f} "
                 f"runtime={rt_calib:.1f}s\n")
        fh.write(f"photo_zncc identity={s_id:.4f} roll180={s_roll180:.4f} "
                 f"found={s_found:.4f}\n")
        fh.write(f"colorize: n_valid={res['n_valid']} frac={res['frac_valid']:.4f} "
                 f"runtime={rt_col:.1f}s\n")
        fh.write(f"offscreen_render={'ok' if rendered else 'fallback'}\n")
    bag.close()
    print("colorizer SELFTEST OK")
