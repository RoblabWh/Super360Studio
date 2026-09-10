#!/usr/bin/env python3
"""Dual-Fisheye -> Equirectangular-Stitcher (Double-Sphere-Modell, cv2.remap-LUTs).

Geometrie exakt wie die Referenz-Implementierung
``Super360_Stitcher_rosbag/work/refine_extrinsic.py`` / ``panoviewer.py``:
Pano-Pixel -> Einheitsstrahl (:func:`_pano_rays`) -> ``* depth_m`` ->
``inv(T_cam0_cami)`` -> Double-Sphere-Projektion -> Fisheye-Pixel.
Dadurch bleibt die bestehende Basalt-Kalibrierung 1:1 gueltig.

Qt-frei. Keine prints ausserhalb des Selbsttests.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

SPLIT_X = 1520                 # Rohbild 3040x1520: [:, :1520]=cam0, [:, 1520:]=cam1
# Beleuchteter Bildkreis je Fisheye (cx, cy, R in px der jeweiligen Haelfte).
# Aus seg0 vermessen (Mittel ueber 30 Frames, Kreisfit an die Dunkelgrenze,
# Residuum < 8 px). Der Kreis ist NICHT im Bild zentriert und wird oben/in den
# Ecken vom Sensor beschnitten.
_CIRCLES = ((811.5, 818.4, 813.3), (816.0, 828.8, 818.0))
_RHO_MAX = 0.995               # relativer Kreisrand, bis zu dem Samples gelten
_BORDER_FEATHER_PX = 50.0      # Feather zum Sensorrand (Kreis ist beschnitten)
_W_FLOOR = 1e-3                # Mindestgewicht gueltiger Pixel (gegen 0/0 am Rand)
_VIGN_VALID_MIN = 0.32         # Vignette-Schwelle: dunkler => Pixel ungueltig
                               # (knapp ueber Spline-Floor 0.30, sonst Loecher)
_VIGN_GAIN_FLOOR = 0.40        # Gain-Deckel 1/0.40 = 2.5
_VIGN_KNOT_PX = 10.0           # Basalt-Spline: Knotenabstand 1e10 bei Radius*1e9
_OVERLAP_W_MIN = 0.03          # Rohgewicht-Schwelle fuer Ueberlappband (expo_norm)
_MAX_OVERLAP_SAMPLES = 20000
_LUT_VERSION = 3


def _pano_rays(width: int, height: int) -> np.ndarray:
    """Einheitsstrahlen je Pano-Pixel im cam0-Frame ("imu"==cam0).

    EXAKT wie refine_extrinsic.py::rays — bindend fuer Kalibrier-Kompatibilitaet.
    Zeile 0..H = Polarwinkel 0..pi, Spalte = Azimut (Offset -pi/2 wie Referenz).
    """
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    x = u / width * (2.0 * np.pi) - np.pi / 2.0
    y = v / height * np.pi
    return np.stack([-np.sin(y) * np.cos(x), -np.cos(y), np.sin(y) * np.sin(x)], -1)


def _bspline3_eval(knots: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Uniforme kubische B-Spline (Basalt RdSpline<1,4>); s in Knoteneinheiten."""
    n = knots.shape[0]
    i = np.clip(np.floor(s).astype(np.int64), 0, n - 4)
    u = np.clip(s - i, 0.0, 1.0)
    u2 = u * u
    u3 = u2 * u
    b0 = (1.0 - 3.0 * u + 3.0 * u2 - u3) / 6.0
    b1 = (4.0 - 6.0 * u2 + 3.0 * u3) / 6.0
    b2 = (1.0 + 3.0 * u + 3.0 * u2 - 3.0 * u3) / 6.0
    b3 = u3 / 6.0
    return b0 * knots[i] + b1 * knots[i + 1] + b2 * knots[i + 2] + b3 * knots[i + 3]


class DoubleSphereCamera:
    """Double-Sphere-Kameramodell (Usenko et al. 2018), Konventionen wie Basalt."""

    def __init__(self, fx: float, fy: float, cx: float, cy: float,
                 xi: float, alpha: float):
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.xi = float(xi)
        self.alpha = float(alpha)

    @staticmethod
    def from_calib(calib_json_path: str) -> tuple["DoubleSphereCamera", "DoubleSphereCamera", np.ndarray]:
        """(cam0, cam1, T_cam0_cam1 4x4) aus Basalt-Kalibrier-JSON.

        value0.T_imu_cam[0] ist Identitaet ("imu"-Frame == cam0), [1] ist die
        Pose von cam1 im cam0-Frame (T_cam0_cam1).
        """
        try:
            with open(calib_json_path, encoding="utf-8") as f:
                data = json.load(f)["value0"]
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            raise RuntimeError(f"Kalibrierdatei nicht lesbar: {calib_json_path} ({exc})") from exc
        intr = data.get("intrinsics", [])
        if len(intr) < 2:
            raise RuntimeError(f"Kalibrierung enthaelt keine zwei Kameras: {calib_json_path}")
        cams = []
        for c in intr[:2]:
            p = c["intrinsics"]
            cams.append(DoubleSphereCamera(p["fx"], p["fy"], p["cx"], p["cy"],
                                           p["xi"], p["alpha"]))
        q = data["T_imu_cam"][1]
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = Rotation.from_quat([q["qx"], q["qy"], q["qz"], q["qw"]]).as_matrix()
        T[:3, 3] = (q["px"], q["py"], q["pz"])
        return cams[0], cams[1], T

    def project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Punkte (...,3) im Kameraframe -> (uv float32 (...,2), valid bool (...,)).

        Formeln identisch zu refine_extrinsic.py::ds_project.
        """
        p = np.asarray(pts, dtype=np.float64)
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        r2 = x * x + y * y
        d1 = np.sqrt(r2 + z * z)
        a, xi = self.alpha, self.xi
        w1 = (1.0 - a) / a if a > 0.5 else a / (1.0 - a)
        w2 = (w1 + xi) / np.sqrt(2.0 * w1 * xi + xi * xi + 1.0)
        valid = z > -w2 * d1
        k = xi * d1 + z
        d2 = np.sqrt(r2 + k * k)
        den = a * d2 + (1.0 - a) * k
        valid &= den > 1e-6
        den = np.where(den == 0.0, 1e-9, den)
        uv = np.empty(p.shape[:-1] + (2,), dtype=np.float32)
        uv[..., 0] = self.fx * x / den + self.cx
        uv[..., 1] = self.fy * y / den + self.cy
        return uv, valid

    def unproject(self, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pixel (...,2) -> (Einheitsrichtungen float64 (...,3), valid bool)."""
        q = np.asarray(uv, dtype=np.float64)
        mx = (q[..., 0] - self.cx) / self.fx
        my = (q[..., 1] - self.cy) / self.fy
        r2 = mx * mx + my * my
        a, xi = self.alpha, self.xi
        valid = np.ones(r2.shape, dtype=bool)
        if a > 0.5:
            valid &= r2 <= 1.0 / (2.0 * a - 1.0)
        tmp = np.clip(1.0 - (2.0 * a - 1.0) * r2, 0.0, None)
        mz = (1.0 - a * a * r2) / (a * np.sqrt(tmp) + 1.0 - a)
        mz2 = mz * mz
        num = mz * xi + np.sqrt(np.clip(mz2 + (1.0 - xi * xi) * r2, 0.0, None))
        den = np.where(mz2 + r2 > 0.0, mz2 + r2, 1e-12)
        k = num / den
        d = np.empty(q.shape[:-1] + (3,), dtype=np.float64)
        d[..., 0] = k * mx
        d[..., 1] = k * my
        d[..., 2] = k * mz - xi
        n = np.linalg.norm(d, axis=-1, keepdims=True)
        valid &= n[..., 0] > 1e-9
        d /= np.where(n > 0.0, n, 1.0)
        return d, valid


def _load_vignette_knots(calib_json_path: str) -> list[Optional[np.ndarray]]:
    """Basalt-Vignette-Splines je Kamera; None wenn trivial (alles 1.0)."""
    try:
        with open(calib_json_path, encoding="utf-8") as f:
            vign = json.load(f)["value0"].get("vignette", [])
    except (OSError, json.JSONDecodeError, KeyError):
        return [None, None]
    out: list[Optional[np.ndarray]] = []
    for i in range(2):
        knots = None
        if i < len(vign):
            try:
                k = np.asarray(vign[i]["value2"], dtype=np.float64).ravel()
                if k.size >= 4 and np.abs(k - 1.0).max() > 1e-6:
                    knots = k
            except (KeyError, TypeError, ValueError):
                knots = None
        out.append(knots)
    return out


class EquirectStitcher:
    """Dual-Fisheye (3040x1520 BGR) -> Equirect-Pano (width x width//2 BGR).

    Beim Bau werden pro Pano-Pixel zwei remap-Mappaare plus Blend-/Vignette-
    Gewichte vorberechnet; :meth:`stitch` ist danach reine remap+gewichtete
    Addition. LUTs werden optional als npz gecacht.
    """

    def __init__(self, calib_json_path: str, width: int = 1920,
                 depth_m: float = 10.0, vignette_softness: float = 0.15,
                 expo_norm: bool = True, lut_cache_dir: Optional[str] = None,
                 fill_holes: bool = True):
        self.calib_json_path = str(calib_json_path)
        self.width = int(width)
        self.height = self.width // 2
        self.depth_m = float(depth_m)
        self.vignette_softness = float(vignette_softness)
        self.expo_norm = bool(expo_norm)
        self.fill_holes = bool(fill_holes)
        if self.width < 32 or self.width % 2:
            raise ValueError(f"Ungueltige Pano-Breite {width} (gerade Zahl >= 32 erwartet)")

        self.cam0, self.cam1, self.T_cam0_cam1 = DoubleSphereCamera.from_calib(self.calib_json_path)
        self._vign_knots = _load_vignette_knots(self.calib_json_path)

        with open(self.calib_json_path, "rb") as f:
            calib_bytes = f.read()
        key_src = calib_bytes + (
            f"|w={self.width}|d={self.depth_m!r}|s={self.vignette_softness!r}"
            f"|v={_LUT_VERSION}".encode()
        )
        self._lut_key = hashlib.md5(key_src).hexdigest()

        lut = None
        cache_path = None
        if lut_cache_dir:
            cache_path = os.path.join(lut_cache_dir, f"stitch_lut_{self._lut_key}.npz")
            if os.path.isfile(cache_path):
                try:
                    with np.load(cache_path) as z:
                        lut = {k: z[k] for k in z.files}
                except (OSError, ValueError, KeyError):
                    lut = None  # defekter Cache -> neu bauen
        if lut is None:
            lut = self._build_luts()
            if cache_path is not None:
                os.makedirs(lut_cache_dir, exist_ok=True)
                tmp = cache_path + ".tmp.npz"   # savez haengt sonst ".npz" an
                np.savez_compressed(tmp, **lut)
                os.replace(tmp, cache_path)
        self._finalize(lut)

    # ------------------------------------------------------------- LUT-Bau

    def _cam_lut(self, uv: np.ndarray, ds_valid: np.ndarray,
                 cam: DoubleSphereCamera, knots: Optional[np.ndarray],
                 circle: tuple[float, float, float]
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(mapx, mapy, w_raw, vign_gain) fuer eine Kamera, alles (H,W) float32."""
        h, w = self.height, self.width
        u = uv[:, 0].astype(np.float64)
        v = uv[:, 1].astype(np.float64)
        inb = (u >= 0.0) & (u <= SPLIT_X - 1.0) & (v >= 0.0) & (v <= SPLIT_X - 1.0)
        ccx, ccy, cr = circle
        rho = np.hypot(u - ccx, v - ccy) / cr
        valid = ds_valid & inb & (rho <= _RHO_MAX)

        # 1) Kosinus-Feather zum Bildkreisrand (Ringbreite = vignette_softness)
        s = max(self.vignette_softness, 1e-3)
        t0 = 1.0 - s
        tt = np.clip((rho - t0) / max(_RHO_MAX - t0, 1e-6), 0.0, 1.0)
        w_raw = 0.5 * (1.0 + np.cos(np.pi * tt))

        # 2) Feather zum Sensorrand (der Bildkreis ist oben/in Ecken beschnitten)
        b = np.minimum(np.minimum(u, SPLIT_X - 1.0 - u),
                       np.minimum(v, SPLIT_X - 1.0 - v))
        tb = np.clip(b / _BORDER_FEATHER_PX, 0.0, 1.0)
        w_raw *= 0.5 * (1.0 - np.cos(np.pi * tb))

        # 3) Vignette: Gain-Korrektur (gedeckelt) + Feather/Cut wo zu dunkel
        if knots is not None:
            r_pp = np.hypot(u - cam.cx, v - cam.cy)
            vg = _bspline3_eval(knots, r_pp / _VIGN_KNOT_PX)
            gain = 1.0 / np.clip(vg, _VIGN_GAIN_FLOOR, 1.0)
            valid &= vg >= _VIGN_VALID_MIN
            v_hi = _VIGN_VALID_MIN + 2.0 * s
            tv = np.clip((vg - _VIGN_VALID_MIN) / max(v_hi - _VIGN_VALID_MIN, 1e-6),
                         0.0, 1.0)
            w_raw *= 0.5 * (1.0 - np.cos(np.pi * tv))
        else:
            gain = np.ones_like(u)

        w_raw = np.maximum(w_raw, _W_FLOOR)
        w_raw[~valid] = 0.0
        mapx = np.where(valid, u, -10.0).astype(np.float32).reshape(h, w)
        mapy = np.where(valid, v, -10.0).astype(np.float32).reshape(h, w)
        return (mapx, mapy,
                w_raw.astype(np.float32).reshape(h, w),
                gain.astype(np.float32).reshape(h, w))

    def _build_luts(self) -> dict[str, np.ndarray]:
        pts = _pano_rays(self.width, self.height).reshape(-1, 3) * self.depth_m
        uv0, val0 = self.cam0.project(pts)
        r = self.T_cam0_cam1[:3, :3]
        t = self.T_cam0_cam1[:3, 3]
        uv1, val1 = self.cam1.project((pts - t) @ r)   # p_cam1 = R^T (p_cam0 - t)
        m0x, m0y, w0, g0 = self._cam_lut(uv0, val0, self.cam0,
                                         self._vign_knots[0], _CIRCLES[0])
        m1x, m1y, w1, g1 = self._cam_lut(uv1, val1, self.cam1,
                                         self._vign_knots[1], _CIRCLES[1])
        return {"map0x": m0x, "map0y": m0y, "map1x": m1x, "map1y": m1y,
                "w0": w0, "w1": w1, "gain0": g0, "gain1": g1}

    def _finalize(self, lut: dict[str, np.ndarray]) -> None:
        self._w0 = lut["w0"]
        self._w1 = lut["w1"]
        self._gain0 = lut["gain0"]
        self._gain1 = lut["gain1"]
        wsum = self._w0 + self._w1
        covered = wsum > 0.0
        self.coverage_frac = float(covered.mean())
        # Loch-Fuellung: Pixel ohne Kameraabdeckung (Zenit-Kappe wegen
        # Sensor-Crop, Taschen an den Naehten) bekommen den Wert des naechsten
        # abgedeckten Pixels. Zuordnung ist je LUT konstant -> einmal hier
        # berechnet, im stitch() nur ein einziges Gather (O(Lochpixel)).
        self._fill_dst_idx = None
        self._fill_src_idx = None
        if self.fill_holes and not covered.all() and covered.any():
            cov_u8 = covered.astype(np.uint8)          # 0 = Loch
            _, labels = cv2.distanceTransformWithLabels(
                1 - cov_u8, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
            # DIST_LABEL_PIXEL: jedes Null-Pixel (=abgedeckt) erhaelt ein
            # eigenes Label; Loch-Pixel tragen das Label des naechsten.
            flat_cov = np.flatnonzero(covered.ravel())
            label_to_flat = np.zeros(int(labels.max()) + 1, dtype=np.int64)
            label_to_flat[labels.ravel()[flat_cov]] = flat_cov
            holes = np.flatnonzero(~covered.ravel())
            self._fill_dst_idx = holes
            self._fill_src_idx = label_to_flat[labels.ravel()[holes]]
        safe = np.where(covered, wsum, 1.0)
        wn0 = np.where(covered, self._w0 / safe, 0.0).astype(np.float32)
        wn1 = np.where(covered, self._w1 / safe, 0.0).astype(np.float32)
        a0 = (wn0 * self._gain0)[:, :, None]
        a1 = (wn1 * self._gain1)[:, :, None]
        self._A0 = np.ascontiguousarray(np.broadcast_to(a0, a0.shape[:2] + (3,)), dtype=np.float32)
        self._A1 = np.ascontiguousarray(np.broadcast_to(a1, a1.shape[:2] + (3,)), dtype=np.float32)
        # Fixpunkt-Maps: schnellster remap-Pfad
        self._m0a, self._m0b = cv2.convertMaps(lut["map0x"], lut["map0y"], cv2.CV_16SC2)
        self._m1a, self._m1b = cv2.convertMaps(lut["map1x"], lut["map1y"], cv2.CV_16SC2)
        # Ueberlappband fuer expo_norm (vignettekorrigierte Mittelwerte)
        ov = (self._w0 > _OVERLAP_W_MIN) & (self._w1 > _OVERLAP_W_MIN)
        idx = np.flatnonzero(ov.ravel())
        if idx.size > _MAX_OVERLAP_SAMPLES:
            idx = idx[:: idx.size // _MAX_OVERLAP_SAMPLES + 1]
        self._ov_idx = idx
        self._ov_v0 = self._gain0.ravel()[idx].astype(np.float64)
        self._ov_v1 = self._gain1.ravel()[idx].astype(np.float64)

    # ------------------------------------------------------------- Stitchen

    def _expo_gains(self, im0: np.ndarray, im1: np.ndarray) -> tuple[float, float]:
        """Gains, die die mittlere Helligkeit beider Fisheyes im Ueberlappband angleichen."""
        if not self.expo_norm or self._ov_idx.size == 0:
            return 1.0, 1.0
        p0 = im0.reshape(-1, 3)[self._ov_idx].mean(axis=1)
        p1 = im1.reshape(-1, 3)[self._ov_idx].mean(axis=1)
        m0 = float((p0 * self._ov_v0).mean())
        m1 = float((p1 * self._ov_v1).mean())
        if m0 < 1.0 or m1 < 1.0:
            return 1.0, 1.0
        target = 0.5 * (m0 + m1)
        g0 = float(np.clip(target / m0, 0.5, 2.0))
        g1 = float(np.clip(target / m1, 0.5, 2.0))
        return g0, g1

    def stitch(self, raw_3040: np.ndarray) -> np.ndarray:
        """Rohbild BGR (1520,3040,3) -> Pano BGR (height,width,3)."""
        img = raw_3040
        if img is None or img.ndim != 3 or img.shape[0] != SPLIT_X \
                or img.shape[1] != 2 * SPLIT_X or img.shape[2] != 3:
            shape = None if img is None else img.shape
            raise ValueError(f"Unerwartete Rohbildform {shape}, erwartet (1520, 3040, 3)")
        im0 = cv2.remap(img[:, :SPLIT_X], self._m0a, self._m0b, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT)
        im1 = cv2.remap(img[:, SPLIT_X:2 * SPLIT_X], self._m1a, self._m1b,
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        g0, g1 = self._expo_gains(im0, im1)
        f0 = im0 * self._A0   # uint8 * float32 -> float32, enthaelt Gewicht+Vignette
        f1 = im1 * self._A1
        pano = cv2.addWeighted(f0, g0, f1, g1, 0.0, dtype=cv2.CV_8U)
        if self._fill_dst_idx is not None:
            flat = pano.reshape(-1, pano.shape[2])
            flat[self._fill_dst_idx] = flat[self._fill_src_idx]
        return pano


# ======================================================================
# Selbsttest (siehe ARCHITECTURE.md / Teststrategie)
# ======================================================================
if __name__ == "__main__":
    import time
    from pathlib import Path

    EVID = ("/tmp/super360_modtests/"
            "stitcher")
    BAG = "/home/lena/RosBagSuper_Gui/rosbag_2026-07-11_15-37-07_seg0"
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CALIB_VIGN = os.path.join(_ROOT, "calib", "calib_new_vign.json")
    CALIB_REFINED = os.path.join(_ROOT, "calib", "calib_new_refined.json")
    os.makedirs(EVID, exist_ok=True)
    rng = np.random.default_rng(42)

    # ---------- Test 1: project/unproject-Roundtrip ----------
    print("== Test 1: DS project/unproject Roundtrip ==")
    cam0, cam1, T01 = DoubleSphereCamera.from_calib(CALIB_VIGN)
    print(f"T_cam0_cam1 t={np.round(T01[:3, 3], 4)}  "
          f"R diag={np.round(np.diag(T01[:3, :3]), 4)}")
    for name, cam in (("cam0", cam0), ("cam1", cam1)):
        n = 20000
        theta = rng.uniform(0.0, np.radians(100.0), n)
        phi = rng.uniform(-np.pi, np.pi, n)
        dirs = np.stack([np.sin(theta) * np.cos(phi),
                         np.sin(theta) * np.sin(phi),
                         np.cos(theta)], -1)
        uv, val = cam.project(dirs * rng.uniform(0.5, 30.0, n)[:, None])
        r_pp = np.hypot(uv[:, 0] - cam.cx, uv[:, 1] - cam.cy)
        m = val & (r_pp < 900)          # Unproject-Domaene (r2 <= 1/(2a-1))
        d2, val2 = cam.unproject(uv[m])
        uv2, val3 = cam.project(d2)
        assert val2.all() and val3.all(), "Roundtrip: valid-Flags verloren"
        err = np.abs(uv2 - uv[m]).max()
        ang = np.degrees(np.arccos(np.clip(
            (d2 * dirs[m]).sum(-1), -1.0, 1.0))).max()
        print(f"  {name}: n={int(m.sum())}  max|d_uv|={err:.5f} px  "
              f"max Winkelfehler={ang:.5f} deg")
        assert err < 0.1, f"Roundtrip-Fehler {err} px >= 0.1 px"

    # ---------- Bag-Frames inline lesen (kein core.bag_reader!) ----------
    print("== Lese Kameraframes aus seg0 ==")
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    frames: list[tuple[int, np.ndarray]] = []
    with AnyReader([Path(BAG)], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader:
        conns = [c for c in reader.connections
                 if c.topic == "/paycam/image_raw/compressed"]
        for conn, ts, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            frames.append((ts, np.frombuffer(bytes(msg.data), np.uint8)))
    frames.sort(key=lambda x: x[0])
    print(f"  {len(frames)} Frames")

    def decode(i: int) -> np.ndarray:
        return cv2.imdecode(frames[i][1], cv2.IMREAD_COLOR)

    # ---------- Test 2: echte Frames stitchen + Naht messen ----------
    print("== Test 2: Stitching (1920x960 und 3200x1600) ==")
    t0 = time.perf_counter()
    st = EquirectStitcher(CALIB_VIGN, width=1920, lut_cache_dir=EVID)
    print(f"  LUT-Bau 1920: {time.perf_counter() - t0:.2f} s  "
          f"coverage={st.coverage_frac:.6f}")

    def seam_metrics(st: EquirectStitcher, raw: np.ndarray) -> dict:
        im0 = cv2.remap(raw[:, :SPLIT_X], st._m0a, st._m0b, cv2.INTER_LINEAR)
        im1 = cv2.remap(raw[:, SPLIT_X:], st._m1a, st._m1b, cv2.INTER_LINEAR)
        g0, g1 = st._expo_gains(im0, im1)
        gr0, gr1 = im0.mean(2), im1.mean(2)
        c0 = np.minimum(gr0 * st._gain0 * g0, 255.0)   # wie im Blend: saturiert
        c1 = np.minimum(gr1 * st._gain1 * g1, 255.0)
        ov = (st._w0 > _OVERLAP_W_MIN) & (st._w1 > _OVERLAP_W_MIN)
        unsat = ov & (gr0 > 10) & (gr0 < 245) & (gr1 > 10) & (gr1 < 245)
        d = np.abs(c0 - c1)
        out = {"gains": (round(g0, 4), round(g1, 4)),
               "overlap_mean_absdiff": float(d[ov].mean()),
               "overlap_mean_absdiff_unsat": float(d[unsat].mean()),
               "n_ov": int(ov.sum()), "n_unsat": int(unsat.sum())}
        w = st.width
        for name, col in (("seam_links", w // 4), ("seam_rechts", 3 * w // 4)):
            band = np.zeros_like(ov)
            band[:, max(col - 15, 0):col + 15] = True
            band &= ov
            out[name] = float(d[band].mean()) if band.any() else float("nan")
        return out

    for idx in (0, 472, 943):
        raw = decode(idx)
        pano = st.stitch(raw)
        p = os.path.join(EVID, f"pano_1920_f{idx:06d}.png")
        cv2.imwrite(p, pano)
        m = seam_metrics(st, raw)
        print(f"  Frame {idx}: {p}")
        print(f"    gains={m['gains']}  |d|_overlap={m['overlap_mean_absdiff']:.2f} "
              f"(n={m['n_ov']})  |d|_unsat={m['overlap_mean_absdiff_unsat']:.2f} "
              f"(n={m['n_unsat']})  |d|_seamL={m['seam_links']:.2f}  "
              f"|d|_seamR={m['seam_rechts']:.2f}")

    t0 = time.perf_counter()
    st_hi = EquirectStitcher(CALIB_VIGN, width=3200, lut_cache_dir=EVID)
    print(f"  LUT-Bau 3200: {time.perf_counter() - t0:.2f} s  "
          f"coverage={st_hi.coverage_frac:.6f}")
    raw472 = decode(472)
    pano_hi = st_hi.stitch(raw472)
    p = os.path.join(EVID, "pano_3200_f000472.png")
    cv2.imwrite(p, pano_hi)
    m = seam_metrics(st_hi, raw472)
    print(f"  Frame 472 @3200: {p}")
    print(f"    gains={m['gains']}  |d|_overlap={m['overlap_mean_absdiff']:.2f}  "
          f"|d|_unsat={m['overlap_mean_absdiff_unsat']:.2f}  "
          f"|d|_seamL={m['seam_links']:.2f}  |d|_seamR={m['seam_rechts']:.2f}")

    # LUT-Cache-Wiederverwendung pruefen
    t0 = time.perf_counter()
    EquirectStitcher(CALIB_VIGN, width=1920, lut_cache_dir=EVID)
    print(f"  LUT-Cache-Load 1920: {time.perf_counter() - t0:.2f} s")

    # ---------- Test 3: Benchmark ----------
    print("== Test 3: Benchmark stitch() 1920x960 ==")
    raws = [decode(460 + i) for i in range(20)]
    st.stitch(raws[0])  # warmup
    times = []
    for r in raws:
        t0 = time.perf_counter()
        st.stitch(r)
        times.append((time.perf_counter() - t0) * 1e3)
    times = np.array(times)
    print(f"  {len(times)} Frames: mean={times.mean():.2f} ms  "
          f"median={np.median(times):.2f} ms  min={times.min():.2f} ms  "
          f"max={times.max():.2f} ms")

    # ---------- Test 4: PanoWeave-Referenzvergleich (falls Docker geht) ----------
    print("== Test 4: PanoWeave-Referenz (Docker) ==")
    import subprocess
    ok = False
    try:
        r = subprocess.run(["docker", "image", "inspect", "panoweave"],
                           capture_output=True, timeout=30)
        ok = r.returncode == 0
        if not ok:
            print(f"  uebersprungen: docker/Image nicht verfuegbar "
                  f"({r.stderr.decode(errors='replace').strip()[:120]})")
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"  uebersprungen: {exc}")
    if ok:
        tmpd = os.path.join(EVID, "panoweave_tmp")
        os.makedirs(tmpd, exist_ok=True)
        np.save(os.path.join(tmpd, "cam0.npy"), raw472[:, :SPLIT_X])
        np.save(os.path.join(tmpd, "cam1.npy"), raw472[:, SPLIT_X:])
        ref_py = r'''
import sys
import numpy as np
sys.path.insert(0, "/panoweave/build")
import panoweave
st = panoweave.Stitcher("/result/calibration.json")
st.resolution(1920, 960); st.fovX(2 * np.pi); st.fovY(np.pi)
try:
    st.vignetteThreshold(0.6)
except Exception:
    pass
c0 = np.load("/data/cam0.npy"); c1 = np.load("/data/cam1.npy")
expo = [float(c0.mean()), float(c1.mean())]
try:
    st.setDepth(10.0)
    pano = np.asarray(st.stitch([c0, c1], expo))
except Exception:
    pano = np.asarray(st.stitch([c0, c1], expo, 10.0))
np.save("/data/ref_pano.npy", np.ascontiguousarray(pano))
print("OK", pano.shape, pano.dtype)
'''
        with open(os.path.join(tmpd, "ref.py"), "w") as f:
            f.write(ref_py)
        calib_dir = os.path.dirname(CALIB_REFINED)
        r = subprocess.run(
            ["docker", "run", "--rm",
             "-v", f"{calib_dir}:/result:ro", "-v", f"{tmpd}:/data",
             "panoweave", "python3", "/data/ref.py"],
            capture_output=True, timeout=600)
        ref_path = os.path.join(tmpd, "ref_pano.npy")
        if r.returncode == 0 and os.path.isfile(ref_path):
            ref = np.load(ref_path)
            if ref.dtype != np.uint8:
                ref = np.clip(ref, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(EVID, "panoweave_ref_f000472.png"), ref)
            st_ref = EquirectStitcher(CALIB_REFINED, width=1920, lut_cache_dir=EVID)
            ours = st_ref.stitch(raw472)
            # Vergleich nur in Ein-Kamera-Zonen (kein Blend-Unterschied)
            wsum = st_ref._w0 + st_ref._w1
            wn0 = np.where(wsum > 0, st_ref._w0 / np.where(wsum > 0, wsum, 1), 0)
            solo = ((wn0 > 0.99) | (wn0 < 0.01)) & (wsum > 0)
            solo &= (ref.sum(2) > 0)
            diff = np.abs(ours.astype(np.float32) - ref.astype(np.float32)).mean(2)
            print(f"  mean|d| Ein-Kamera-Zonen={float(diff[solo].mean()):.2f}  "
                  f"gesamt={float(diff[wsum > 0].mean()):.2f}  "
                  f"(n_solo={int(solo.sum())})")
            side = np.concatenate([ours, ref], axis=0)
            cv2.putText(side, "oben: stitcher.py / unten: PanoWeave", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cmp_path = os.path.join(EVID, "cmp_panoweave_f000472.png")
            cv2.imwrite(cmp_path, side)
            print(f"  Vergleichsbild: {cmp_path}")
        else:
            print(f"  uebersprungen: Container-Lauf fehlgeschlagen "
                  f"(rc={r.returncode}) {r.stderr.decode(errors='replace').strip()[-300:]}")

    print("SELFTEST OK")
