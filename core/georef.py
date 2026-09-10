"""GPS-Qualitätsprüfung + Georeferenzierung (LIO-Welt <-> ENU/UTM).

Qt-frei. Eingabe-Fixe sind Objekte mit den Attributen von bag_reader.GpsFix
(Duck-Typing, kein Import von bag_reader nötig). ``align`` akzeptiert jedes
Objekt mit ``interpolate_pose(t)`` (Recording-Duck-Typing).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from pyproj import CRS, Transformer

# Sentinels (s. ARCHITECTURE.md / mavros)
_EPH_SENTINEL_CM = 9999          # GPSRAW eph unbekannt (bzw. 65535)
_COV_SENTINEL_M = 4294967.0      # UINT32_MAX mm -> 4294967.295 m
_SATS_SENTINEL = 255             # satellites_visible unbekannt

_MIN_GOOD_FIXES = 20
_MIN_BASELINE_M = 5.0
_MAX_EPH_CM = 500
_MAX_COV_M = 10.0
_MIN_SATS = 6
_MIN_FIX_TYPE = 3

_FIX_TYPE_NAMES = {
    0: "kein GPS-Gerät erkannt",
    1: "kein Fix",
    2: "nur 2D-Fix",
    3: "3D-Fix",
    4: "DGPS",
    5: "RTK Float",
    6: "RTK Fixed",
    7: "statisch",
    8: "PPP",
}

_T_LLH_ECEF: Transformer | None = None
_T_ECEF_LLH: Transformer | None = None


def _llh_ecef() -> Transformer:
    global _T_LLH_ECEF
    if _T_LLH_ECEF is None:
        _T_LLH_ECEF = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
    return _T_LLH_ECEF


def _ecef_llh() -> Transformer:
    global _T_ECEF_LLH
    if _T_ECEF_LLH is None:
        _T_ECEF_LLH = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)
    return _T_ECEF_LLH


def _enu_rotation(lat_deg: float, lon_deg: float) -> np.ndarray:
    """R so dass enu = R @ (ecef - ecef_origin)."""
    lam = math.radians(lon_deg)
    phi = math.radians(lat_deg)
    sl, cl = math.sin(lam), math.cos(lam)
    sp, cp = math.sin(phi), math.cos(phi)
    return np.array([
        [-sl, cl, 0.0],
        [-sp * cl, -sp * sl, cp],
        [cp * cl, cp * sl, sp],
    ])


def llh_to_enu(lat: np.ndarray, lon: np.ndarray, alt: np.ndarray,
               origin_llh: tuple[float, float, float]) -> np.ndarray:
    """WGS84 (Grad, Meter) -> lokales ENU (N,3) um origin_llh=(lat,lon,alt)."""
    ex, ey, ez = _llh_ecef().transform(np.asarray(lon), np.asarray(lat), np.asarray(alt))
    ox, oy, oz = _llh_ecef().transform(origin_llh[1], origin_llh[0], origin_llh[2])
    d = np.column_stack([ex - ox, ey - oy, ez - oz]).astype(np.float64)
    return d @ _enu_rotation(origin_llh[0], origin_llh[1]).T


def enu_to_llh(enu: np.ndarray,
               origin_llh: tuple[float, float, float]) -> np.ndarray:
    """Umkehrung von llh_to_enu; liefert (N,3) [lat, lon, alt]."""
    enu = np.atleast_2d(np.asarray(enu, dtype=np.float64))
    ox, oy, oz = _llh_ecef().transform(origin_llh[1], origin_llh[0], origin_llh[2])
    ecef = enu @ _enu_rotation(origin_llh[0], origin_llh[1]) + np.array([ox, oy, oz])
    lon, lat, alt = _ecef_llh().transform(ecef[:, 0], ecef[:, 1], ecef[:, 2])
    return np.column_stack([lat, lon, alt])


def utm_epsg_from_lonlat(lon: float, lat: float) -> int:
    zone = min(60, max(1, int((lon + 180.0) // 6) + 1))
    return (32600 if lat >= 0 else 32700) + zone


def _eph_m(fix) -> float:
    """Horizontale Unsicherheit in m; Fallback über Kovarianz, sonst 1 m."""
    eph_cm = getattr(fix, "eph_cm", None)
    if eph_cm is not None and 0 < eph_cm < _EPH_SENTINEL_CM:
        return float(eph_cm) / 100.0
    ce = getattr(fix, "cov_east_m", float("nan"))
    cn = getattr(fix, "cov_north_m", float("nan"))
    c = max(ce, cn)
    if np.isfinite(c) and 0.0 < c < _COV_SENTINEL_M:
        return float(c)
    return 1.0


@dataclass
class GpsQuality:
    usable: bool
    reasons: list[str]                     # DEUTSCH
    n_total: int
    n_good: int
    fix_type_hist: dict[int, int]
    median_eph_cm: float | None
    median_sats: float | None
    baseline_m: float
    per_fix_good: np.ndarray = field(repr=False)  # bool (n_total,)


def assess(fixes: Sequence) -> GpsQuality:
    """Bewertet GPS-Fixe (bag_reader.GpsFix-artige Objekte)."""
    n = len(fixes)
    per_good = np.zeros(n, dtype=bool)
    fix_type_hist: dict[int, int] = {}
    eph_vals: list[float] = []
    sat_vals: list[float] = []
    cnt_status = cnt_latlon = cnt_sats = cnt_eph = cnt_cov = 0
    bad_fix_types: dict[int, int] = {}

    for i, f in enumerate(fixes):
        good = True
        if getattr(f, "status", -1) < 0:
            cnt_status += 1
            good = False
        lat, lon = float(f.lat), float(f.lon)
        if lat == 0.0 and lon == 0.0:
            cnt_latlon += 1
            good = False
        ft = getattr(f, "fix_type", None)
        if ft is not None:
            fix_type_hist[int(ft)] = fix_type_hist.get(int(ft), 0) + 1
            if ft < _MIN_FIX_TYPE:
                bad_fix_types[int(ft)] = bad_fix_types.get(int(ft), 0) + 1
                good = False
        sats = getattr(f, "satellites", None)
        if sats is not None and sats != _SATS_SENTINEL:
            sat_vals.append(float(sats))
            if sats < _MIN_SATS:
                cnt_sats += 1
                good = False
        eph = getattr(f, "eph_cm", None)
        if eph is not None:
            if eph >= _EPH_SENTINEL_CM or eph > _MAX_EPH_CM:
                cnt_eph += 1
                good = False
            else:
                eph_vals.append(float(eph))
        ce = getattr(f, "cov_east_m", 0.0)
        cn = getattr(f, "cov_north_m", 0.0)
        if max(ce, cn) > _MAX_COV_M:  # Sentinel 4294967.295 m fällt hier mit rein
            cnt_cov += 1
            good = False
        per_good[i] = good

    n_good = int(per_good.sum())

    baseline = 0.0
    if n_good >= 2:
        gf = [f for f, g in zip(fixes, per_good) if g]
        origin = (float(gf[0].lat), float(gf[0].lon), float(gf[0].alt))
        enu = llh_to_enu(np.array([f.lat for f in gf]), np.array([f.lon for f in gf]),
                         np.array([f.alt for f in gf]), origin)
        span = enu[:, :2].max(axis=0) - enu[:, :2].min(axis=0)
        baseline = float(np.hypot(span[0], span[1]))

    median_eph = float(np.median(eph_vals)) if eph_vals else None
    median_sats = float(np.median(sat_vals)) if sat_vals else None

    reasons: list[str] = []
    if n == 0:
        reasons.append("keine GPS-Fixe im Bag gefunden")
    if cnt_status:
        reasons.append(f"NavSatFix-Status < 0 (kein Fix) bei {cnt_status}/{n} Fixen")
    if cnt_latlon:
        reasons.append(f"lat=lon=0 (keine gültige Position) bei {cnt_latlon}/{n} Fixen")
    for ft in sorted(bad_fix_types):
        name = _FIX_TYPE_NAMES.get(ft, "unbekannt")
        reasons.append(f"fix_type={ft} ({name}) bei {bad_fix_types[ft]}/{n} Fixen")
    if cnt_sats:
        med = int(median_sats) if median_sats is not None else 0
        reasons.append(f"nur {med} Satelliten (Median, benötigt ≥{_MIN_SATS}) — "
                       f"{cnt_sats}/{n} Fixe betroffen")
    if cnt_eph:
        reasons.append(f"eph > {_MAX_EPH_CM} cm oder unbekannt (Sentinel 9999) "
                       f"bei {cnt_eph}/{n} Fixen")
    if cnt_cov:
        reasons.append(f"Positions-Kovarianz > {_MAX_COV_M:.0f} m bei {cnt_cov}/{n} Fixen")

    usable = n_good >= _MIN_GOOD_FIXES and baseline >= _MIN_BASELINE_M
    if n_good < _MIN_GOOD_FIXES:
        reasons.append(f"nur {n_good} brauchbare Fixe (benötigt ≥{_MIN_GOOD_FIXES})")
    elif baseline < _MIN_BASELINE_M:
        reasons.append(f"Baseline nur {baseline:.1f} m (benötigt ≥{_MIN_BASELINE_M:.0f} m)")

    return GpsQuality(usable=usable, reasons=reasons, n_total=n, n_good=n_good,
                      fix_type_hist=fix_type_hist, median_eph_cm=median_eph,
                      median_sats=median_sats, baseline_m=baseline,
                      per_fix_good=per_good)


@dataclass
class GeorefResult:
    T_enu_world: np.ndarray                 # 4x4: LIO-Welt -> lokales ENU
    origin_llh: tuple[float, float, float]  # Ursprung = erster guter Fix
    utm_epsg: int
    rms_m: float
    n_used: int
    utm_offset: tuple[float, float]         # ENU-Ursprung in UTM (E, N)


def align(rec, fixes: Sequence, quality: GpsQuality,
          progress_cb: Callable[[float, str], None] | None = None) -> GeorefResult:
    """4-DOF-Ausrichtung (Yaw + Translation) LIO-Trajektorie -> ENU.

    ``rec`` braucht nur ``interpolate_pose(t) -> 4x4 | None`` (Duck-Typing).
    Gewichte 1/max(eph_m, 0.5)^2; geschlossene Lösung über gewichtete
    horizontale Kreuzkovarianz (2D-Umeyama ohne Skalierung) + z-Offset.
    """
    def _prog(frac: float, msg: str) -> None:
        if progress_cb is not None:
            progress_cb(frac, msg)

    if not quality.usable:
        raise RuntimeError("GPS-Qualität unzureichend — Georeferenzierung nicht möglich "
                           f"({'; '.join(quality.reasons[:3]) or 'keine guten Fixe'})")
    if len(fixes) != len(quality.per_fix_good):
        raise RuntimeError("GPS-Fixe passen nicht zur Qualitätsbewertung "
                           "(unterschiedliche Anzahl)")

    good = [f for f, g in zip(fixes, quality.per_fix_good) if g]
    _prog(0.1, "GPS-Fixe nach ENU umrechnen …")
    origin_llh = (float(good[0].lat), float(good[0].lon), float(good[0].alt))
    enu = llh_to_enu(np.array([f.lat for f in good]), np.array([f.lon for f in good]),
                     np.array([f.alt for f in good]), origin_llh)

    _prog(0.4, "LIO-Posen zu Fix-Zeitstempeln interpolieren …")
    p_list, q_list, w_list = [], [], []
    for f, q in zip(good, enu):
        T = rec.interpolate_pose(float(f.stamp))
        if T is None:
            continue
        p_list.append(np.asarray(T, dtype=np.float64)[:3, 3])
        q_list.append(q)
        w_list.append(1.0 / max(_eph_m(f), 0.5) ** 2)
    n_used = len(p_list)
    if n_used < 5:
        raise RuntimeError(f"Zu wenige GPS-Fixe innerhalb der Trajektorienzeit "
                           f"({n_used}, benötigt ≥5)")

    p = np.array(p_list)                    # LIO-Welt
    q = np.array(q_list)                    # ENU
    w = np.array(w_list)
    w /= w.sum()

    _prog(0.7, "4-DOF-Ausrichtung berechnen …")
    p_mean = w @ p
    q_mean = w @ q
    pc = (p - p_mean)[:, :2]
    qc = (q - q_mean)[:, :2]
    # maximiere sum w * qc^T R(yaw) pc  ->  yaw = atan2(B, A)
    a = float(np.sum(w * (pc[:, 0] * qc[:, 0] + pc[:, 1] * qc[:, 1])))
    b = float(np.sum(w * (pc[:, 0] * qc[:, 1] - pc[:, 1] * qc[:, 0])))
    yaw = math.atan2(b, a)
    c, s = math.cos(yaw), math.sin(yaw)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    T[0, 3] = q_mean[0] - (c * p_mean[0] - s * p_mean[1])
    T[1, 3] = q_mean[1] - (s * p_mean[0] + c * p_mean[1])
    T[2, 3] = q_mean[2] - p_mean[2]

    res = (p @ T[:3, :3].T + T[:3, 3]) - q
    rms = float(np.sqrt(np.mean(np.sum(res ** 2, axis=1))))

    median_eph_m = (quality.median_eph_cm / 100.0) if quality.median_eph_cm else 0.0
    rms_limit = max(3.0, 2.0 * median_eph_m)
    if rms > rms_limit:
        raise RuntimeError(f"Georeferenzierung fehlgeschlagen: RMS {rms:.2f} m > "
                           f"{rms_limit:.2f} m — GPS und Trajektorie passen nicht zusammen")

    epsg = utm_epsg_from_lonlat(origin_llh[1], origin_llh[0])
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    utm_e, utm_n = to_utm.transform(origin_llh[1], origin_llh[0])
    _prog(1.0, "Georeferenzierung abgeschlossen")

    return GeorefResult(T_enu_world=T, origin_llh=origin_llh, utm_epsg=epsg,
                        rms_m=rms, n_used=n_used, utm_offset=(float(utm_e), float(utm_n)))


def _colors_to_uint8(colors_rgb: np.ndarray | None, n: int) -> np.ndarray:
    if colors_rgb is None:
        return np.zeros((n, 3), dtype=np.uint8)
    c = np.asarray(colors_rgb)
    if c.dtype != np.uint8:
        c = np.clip(np.asarray(c, dtype=np.float64) * 255.0, 0, 255).astype(np.uint8)
    return c


def export_las(points_xyz: np.ndarray, colors_rgb: np.ndarray | None,
               georef: GeorefResult | None, path: str) -> None:
    """LAS 1.2, Punktformat 2 (RGB), Auflösung 1 mm.

    Mit Georef: jeder Punkt wird exakt projiziert (ENU → Lat/Lon/Höhe →
    UTM-Zone des Ursprungs). Das berücksichtigt Meridiankonvergenz und
    Maßstabsfaktor der Projektion — ENU-Offsets einfach auf den UTM-Ursprung
    zu addieren wäre abseits des Ursprungs um Meter falsch. z = ellipsoidische
    Höhe; EPSG landet als CRS-VLR im Header (laspy add_crs).
    """
    import laspy

    pts = np.asarray(points_xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
        raise RuntimeError("LAS-Export: Punktwolke ist leer oder hat falsche Form")
    if georef is not None:
        R = georef.T_enu_world[:3, :3]
        t = georef.T_enu_world[:3, 3]
        enu = pts @ R.T + t
        llh = enu_to_llh(enu, georef.origin_llh)          # (N,3) lat, lon, alt
        to_utm = Transformer.from_crs("EPSG:4979", f"EPSG:{georef.utm_epsg}",
                                      always_xy=True)
        e, n, z = to_utm.transform(llh[:, 1], llh[:, 0], llh[:, 2])
        pts = np.column_stack([e, n, z])

    header = laspy.LasHeader(version="1.2", point_format=2)
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.floor(pts.min(axis=0))
    if georef is not None:
        try:
            header.add_crs(CRS.from_epsg(georef.utm_epsg))
        except Exception:
            header.system_identifier = f"EPSG:{georef.utm_epsg}"[:31]

    las = laspy.LasData(header)
    las.x, las.y, las.z = pts[:, 0], pts[:, 1], pts[:, 2]
    c = _colors_to_uint8(colors_rgb, len(pts)).astype(np.uint16) * 257  # 8 -> 16 bit
    las.red, las.green, las.blue = c[:, 0], c[:, 1], c[:, 2]
    try:
        las.write(path)
    except OSError as e:
        raise RuntimeError(f"LAS-Datei konnte nicht geschrieben werden: {e}") from e


def export_ply_pcd(points_xyz: np.ndarray, colors_rgb: np.ndarray | None,
                   path: str) -> None:
    """PLY/PCD via open3d; Dateiendung entscheidet. Farben werden als 0..1 gesetzt."""
    import open3d as o3d

    ext = path.lower().rsplit(".", 1)[-1] if "." in path else ""
    if ext not in ("ply", "pcd"):
        raise RuntimeError(f"Nicht unterstütztes Exportformat: .{ext} (nur .ply/.pcd)")
    pts = np.asarray(points_xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
        raise RuntimeError("Export: Punktwolke ist leer oder hat falsche Form")
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    if colors_rgb is not None:
        pc.colors = o3d.utility.Vector3dVector(
            _colors_to_uint8(colors_rgb, len(pts)).astype(np.float64) / 255.0)
    if not o3d.io.write_point_cloud(path, pc):
        raise RuntimeError(f"Punktwolke konnte nicht geschrieben werden: {path}")


# --------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    from types import SimpleNamespace

    EVID = ("/tmp/super360_modtests/"
            "georef")
    os.makedirs(EVID, exist_ok=True)
    log_lines: list[str] = []

    def log(*a):
        line = " ".join(str(x) for x in a)
        print(line)
        log_lines.append(line)

    rng = np.random.default_rng(42)

    # ---------- Test 1: synthetisch ----------
    log("=== Test 1: Synthetik (Bogen 100 m, Yaw+Translation, Rauschen 0.5 m) ===")

    class StubRec:
        """Duck-typed Recording: nur interpolate_pose/path_positions."""

        def __init__(self, stamps: np.ndarray, positions: np.ndarray):
            self.stamps = stamps
            self.positions = positions

        def interpolate_pose(self, t: float) -> np.ndarray | None:
            if t < self.stamps[0] - 0.15 or t > self.stamps[-1] + 0.15:
                return None
            t = float(np.clip(t, self.stamps[0], self.stamps[-1]))
            T = np.eye(4)
            T[:3, 3] = [float(np.interp(t, self.stamps, self.positions[:, k]))
                        for k in range(3)]
            return T

        def path_positions(self) -> np.ndarray:
            return self.positions

    # Bogen: Radius 50 m, 2 rad -> 100 m Bogenlänge
    S = 400
    t0 = 1000.0
    stamps = t0 + np.linspace(0.0, 45.0, S)
    ang = np.linspace(0.0, 2.0, S)
    traj = np.column_stack([50.0 * np.sin(ang), 50.0 * (1.0 - np.cos(ang)),
                            0.5 * np.sin(3 * ang)])
    rec = StubRec(stamps, traj)

    yaw_true = math.radians(37.0)
    c, s = math.cos(yaw_true), math.sin(yaw_true)
    R_true = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    t_true = np.array([12.0, -5.0, 3.0])
    origin0 = (51.5740, 7.0270, 60.0)

    n_fix = 100
    fix_t = t0 + np.linspace(0.5, 44.5, n_fix)
    p_at_fix = np.column_stack([np.interp(fix_t, stamps, traj[:, k]) for k in range(3)])
    noise = rng.normal(0.0, 0.5, size=(n_fix, 3))
    q_enu = p_at_fix @ R_true.T + t_true + noise
    llh = enu_to_llh(q_enu, origin0)

    bad_idx = set(rng.choice(n_fix, size=10, replace=False).tolist())
    fixes_syn = []
    for i in range(n_fix):
        bad = i in bad_idx
        fixes_syn.append(SimpleNamespace(
            stamp=float(fix_t[i]), lat=float(llh[i, 0]), lon=float(llh[i, 1]),
            alt=float(llh[i, 2]), status=0, service=1,
            cov_east_m=0.8, cov_north_m=0.8, cov_up_m=1.5, cov_type=2,
            fix_type=4, eph_cm=9999 if bad else 80, epv_cm=120,
            satellites=14))

    qual = assess(fixes_syn)
    log(f"assess: usable={qual.usable} n_total={qual.n_total} n_good={qual.n_good} "
        f"baseline={qual.baseline_m:.1f} m median_eph={qual.median_eph_cm} cm "
        f"median_sats={qual.median_sats}")
    assert qual.usable, "Synthetik muss usable sein"
    assert qual.n_good == n_fix - 10, f"n_good {qual.n_good} != {n_fix - 10}"

    result = align(rec, fixes_syn, qual,
                   progress_cb=lambda f, m: log(f"  progress {f:.1f}: {m}"))
    yaw_est = math.atan2(result.T_enu_world[1, 0], result.T_enu_world[0, 0])
    # ENU-Ursprung von align = erster guter (verrauschter) Fix -> erwartete
    # Translation im align-Frame: t_true minus dessen ENU-Position.
    first_good = int(np.flatnonzero(qual.per_fix_good)[0])
    t_expected = t_true - q_enu[first_good]
    yaw_err_deg = abs(math.degrees(yaw_est - yaw_true))
    t_err = float(np.linalg.norm(result.T_enu_world[:3, 3] - t_expected))
    log(f"align: yaw_est={math.degrees(yaw_est):.3f}° (wahr 37°), Fehler={yaw_err_deg:.3f}°")
    log(f"align: |t_err|={t_err:.3f} m (Toleranz 0.3 m)")
    log(f"align: rms={result.rms_m:.3f} m (Erwartung ~ sqrt(3)*0.5 = 0.866 m)")
    log(f"align: epsg={result.utm_epsg} n_used={result.n_used} "
        f"utm_offset=({result.utm_offset[0]:.1f}, {result.utm_offset[1]:.1f})")
    assert yaw_err_deg < 1.0, f"Yaw-Fehler {yaw_err_deg}° >= 1°"
    assert t_err < 0.3, f"Translationsfehler {t_err:.3f} m >= 0.3 m"
    assert abs(result.rms_m - math.sqrt(3) * 0.5) < 0.25, "RMS weicht stark vom Rauschen ab"
    assert result.utm_epsg == 32632, "EPSG für lon=7.03/lat=51.57 muss 32632 sein"

    # ---------- Test 2: seg0 (GPS tot) ----------
    log("\n=== Test 2: seg0 real (Inline-Reader, GPS tot) ===")
    from pathlib import Path
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_types_from_msg, get_typestore

    ts_store = get_typestore(Stores.ROS2_HUMBLE)
    msg_txt = Path("/opt/ros/humble/share/mavros_msgs/msg/GPSRAW.msg").read_text()
    ts_store.register(get_types_from_msg(msg_txt, "mavros_msgs/msg/GPSRAW"))

    nav, raws = [], []
    bagp = Path("/home/lena/RosBagSuper_Gui/rosbag_2026-07-11_15-37-07_seg0")
    with AnyReader([bagp], default_typestore=ts_store) as reader:
        conns = [cn for cn in reader.connections
                 if cn.topic in ("/mavros/global_position/raw/fix",
                                 "/mavros/gpsstatus/gps1/raw")]
        for conn, _, raw in reader.messages(connections=conns):
            m = reader.deserialize(raw, conn.msgtype)
            st = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            if conn.msgtype.endswith("NavSatFix"):
                nav.append((st, m))
            else:
                raws.append((st, m))
    raw_stamps = np.array([r[0] for r in raws])
    fixes_real = []
    for st, m in nav:
        gr = None
        if len(raws):
            j = int(np.argmin(np.abs(raw_stamps - st)))
            if abs(raw_stamps[j] - st) <= 0.3:
                gr = raws[j][1]
        cov = np.asarray(m.position_covariance, dtype=np.float64)
        fixes_real.append(SimpleNamespace(
            stamp=st, lat=float(m.latitude), lon=float(m.longitude),
            alt=float(m.altitude), status=int(m.status.status),
            service=int(m.status.service),
            cov_east_m=float(np.sqrt(max(cov[0], 0.0))),
            cov_north_m=float(np.sqrt(max(cov[4], 0.0))),
            cov_up_m=float(np.sqrt(max(cov[8], 0.0))),
            cov_type=int(m.position_covariance_type),
            fix_type=int(gr.fix_type) if gr else None,
            eph_cm=int(gr.eph) if gr else None,
            epv_cm=int(gr.epv) if gr else None,
            satellites=int(gr.satellites_visible) if gr else None))
    qual_real = assess(fixes_real)
    log(f"seg0: n_total={qual_real.n_total} usable={qual_real.usable} "
        f"n_good={qual_real.n_good}")
    log("GpsQuality(seg0):")
    log(f"  usable={qual_real.usable}")
    log(f"  n_total={qual_real.n_total} n_good={qual_real.n_good}")
    log(f"  fix_type_hist={qual_real.fix_type_hist}")
    log(f"  median_eph_cm={qual_real.median_eph_cm} median_sats={qual_real.median_sats}")
    log(f"  baseline_m={qual_real.baseline_m}")
    for r in qual_real.reasons:
        log(f"  Grund: {r}")
    assert not qual_real.usable
    joined = " | ".join(qual_real.reasons)
    assert "fix_type=0" in joined and "lat=lon=0" in joined and "0 Satelliten" in joined, \
        f"Gründe unvollständig: {joined}"

    # ---------- Test 3: Export-Roundtrip ----------
    log("\n=== Test 3: Export PLY/PCD/LAS (1000 Punkte) ===")
    pts = rng.uniform(-20, 20, size=(1000, 3))
    cols = rng.integers(0, 256, size=(1000, 3), dtype=np.uint8)
    ply = os.path.join(EVID, "export_test.ply")
    pcd = os.path.join(EVID, "export_test.pcd")
    las_plain = os.path.join(EVID, "export_test_local.las")
    las_geo = os.path.join(EVID, "export_test_utm.las")
    export_ply_pcd(pts, cols, ply)
    export_ply_pcd(pts, cols, pcd)
    export_las(pts, cols, None, las_plain)
    export_las(pts, cols, result, las_geo)

    import open3d as o3d
    back = o3d.io.read_point_cloud(ply)
    bp = np.asarray(back.points)
    bc = np.rint(np.asarray(back.colors) * 255.0).astype(np.int64)
    log(f"PLY roundtrip: max|dp|={np.abs(bp - pts).max():.2e} m, "
        f"max|dc|={np.abs(bc - cols.astype(np.int64)).max()} (0..255)")
    assert np.abs(bp - pts).max() < 1e-6 and np.abs(bc - cols).max() <= 1

    import laspy
    lr = laspy.read(las_plain)
    lp = np.column_stack([lr.x, lr.y, lr.z])
    lc = np.column_stack([lr.red, lr.green, lr.blue]) // 257
    log(f"LAS roundtrip (lokal): max|dp|={np.abs(lp - pts).max():.2e} m "
        f"(Skala 0.001), max|dc|={np.abs(lc - cols).max()}")
    assert np.abs(lp - pts).max() <= 0.0006 and np.abs(lc - cols).max() == 0

    lg = laspy.read(las_geo)
    gp = np.column_stack([lg.x, lg.y, lg.z])
    # pyproj-Wahrheit: ENU -> LLH (Modul-Helfer) -> UTM über EPSG:4326 (2D),
    # z = ellipsoidische Höhe — unabhängiger Rechenweg zum Export (EPSG:4979).
    enu_chk = pts @ result.T_enu_world[:3, :3].T + result.T_enu_world[:3, 3]
    llh_chk = enu_to_llh(enu_chk, result.origin_llh)
    t_truth = Transformer.from_crs("EPSG:4326", f"EPSG:{result.utm_epsg}",
                                   always_xy=True)
    te, tn = t_truth.transform(llh_chk[:, 1], llh_chk[:, 0])
    exp = np.column_stack([te, tn, llh_chk[:, 2]])
    exp_old = enu_chk.copy()                # alte Formel: ENU + UTM-Offset
    exp_old[:, 0] += result.utm_offset[0]
    exp_old[:, 1] += result.utm_offset[1]
    exp_old[:, 2] += result.origin_llh[2]
    d_old = float(np.abs(gp[:, :2] - exp_old[:, :2]).max())
    crs_read = None
    try:
        crs_read = lg.header.parse_crs()
    except Exception:
        pass
    log(f"LAS (UTM): max|dp|={np.abs(gp - exp).max():.2e} m zur pyproj-Wahrheit, "
        f"max {d_old:.3f} m zur alten ENU+Offset-Formel, "
        f"CRS={crs_read}, mean_E={gp[:, 0].mean():.1f} mean_N={gp[:, 1].mean():.1f}")
    assert np.abs(gp - exp).max() <= 0.0006, "UTM-Export weicht von pyproj-Wahrheit ab"
    assert d_old < 1.0, f"nahe des Ursprungs muss die alte Formel <1 m abweichen ({d_old})"
    assert crs_read is not None and crs_read.to_epsg() == result.utm_epsg

    # 1-km-Punkte: dort macht die Meridiankonvergenz die alte Formel um Meter
    # falsch (Ursprung lon 7.03 liegt am Westrand von Zone 32, gamma ~ -1.5°).
    R_ew = result.T_enu_world[:3, :3]
    t_ew = result.T_enu_world[:3, 3]
    enu_far = np.array([[1000.0, 0.0, 2.0], [0.0, 1000.0, 2.0]])
    pts_far = (enu_far - t_ew) @ R_ew                    # zurück in LIO-Welt
    las_far = os.path.join(EVID, "export_test_utm_far.las")
    export_las(pts_far, None, result, las_far)
    lf = laspy.read(las_far)
    gf = np.column_stack([lf.x, lf.y, lf.z])
    llh_far = enu_to_llh(enu_far, result.origin_llh)
    fe, fn = t_truth.transform(llh_far[:, 1], llh_far[:, 0])
    exp_far = np.column_stack([fe, fn, llh_far[:, 2]])
    old_far = enu_far.copy()
    old_far[:, 0] += result.utm_offset[0]
    old_far[:, 1] += result.utm_offset[1]
    err_truth = float(np.abs(gf - exp_far).max())
    err_old = np.linalg.norm((gf - old_far)[:, :2], axis=1)
    log(f"LAS (UTM, 1 km): max|dp|={err_truth:.2e} m zur pyproj-Wahrheit; "
        f"alte Formel läge {err_old.min():.1f}..{err_old.max():.1f} m daneben")
    assert err_truth <= 0.0006, "1-km-Punkt weicht von pyproj-Wahrheit ab"
    assert err_old.min() > 5.0, "Konvergenz-Korrektur wirkt am 1-km-Punkt nicht"

    log("\nALLE GEOREF-SELBSTTESTS BESTANDEN")
    with open(os.path.join(EVID, "metrics.txt"), "w") as fh:
        fh.write("\n".join(log_lines) + "\n")
