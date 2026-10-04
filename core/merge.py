"""Zwei FAST-LIO-Aufzeichnungen zu einer zusammenfuehren.

Jeder Flug bekommt von FAST-LIO ein eigenes Weltsystem, verankert in der
Sensorlage seines ersten Scans. Zwei Aufzeichnungen liegen deshalb beliebig
zueinander im Raum, auch wenn sie dasselbe Gebaeude zeigen. :func:`register`
sucht die starre Transformation dazwischen, :func:`merge_recordings` schreibt
daraus eine neue, vollwertige Aufzeichnung.

Der Trick beim Schreiben: die Punkte stehen in ``points.bin`` im **Body-Frame**
des jeweiligen Scans, sind also von der Pose unabhaengig. Zusammenfuehren heisst
darum, die Punktdateien aneinanderzuhaengen und nur die Posen der zweiten
Aufzeichnung umzurechnen (``T_neu = T_ab @ T_alt``). Nichts wird neu berechnet
und nichts verzerrt.

Reihenfolge: der zeitlich fruehere Flug kommt zuerst, damit ``stamps.npy``
aufsteigend bleibt — darauf verlaesst sich ``Recording.interpolate_pose`` per
``searchsorted``. Ueberlappen sich die Zeitraeume beider Bags, wird abgelehnt
statt still etwas Falsches zu schreiben.

Lotrecht: :func:`register` sucht nur um die Hochachse breit, Kippen verfeinert
ICP bloss. Beide Wolken muessen also gleich lotrecht stehen. Startet ein Flug
schon in der Luft, findet ``Recording.load`` kein Ruhefenster und laesst ihn
gekippt (bei schraeg montiertem Livox um 35-45 Grad); dann landet jede Suche
in einer falschen Lage. :func:`lotrechte_aus_flug` misst die Lotrechte deshalb
ueber den ganzen Flug, :func:`kippausgleich` dreht B so, dass seine Lotrechte
auf der von A steht.

Qt-frei. open3d wird erst in :func:`register` importiert, damit der Rest der
Anwendung ohne die Bibliothek startet.
"""

from __future__ import annotations

import json
import os
import shutil
import time

import numpy as np
from scipy.spatial.transform import Rotation

from core.gemeinsam import pruefe_abbruch as _check_cancel

_REG_TARGET_PTS = 400_000   # so viele Punkte gehen hoechstens in die Ausrichtung
_WRITE_CHUNK = 2_000_000    # Punkte je Schreibblock (~24 MB)
_YAW_CANDIDATES = tuple(range(30, 360, 30))
_FGR_TRIALS = 10
_FEIN_ZIEL = 0.10           # m, feinstes Voxel im Feinschliff
_LOT_MIN_S = 10.0           # s Flug, darunter ist das Mittel keine Lotrechte
_LOT_MAX_STREUUNG = 2.0     # Grad zwischen den Flughaelften, darueber unbrauchbar


def cloud_for_registration(rec, max_points: int = _REG_TARGET_PTS) -> np.ndarray:
    """Ausgeduennte Weltpunkte einer Aufzeichnung, float64 (N,3).

    Gleichmaessig ueber alle Scans gegriffen (Schrittweite), damit die Auswahl
    den ganzen Flug abdeckt und nicht nur seinen Anfang.
    """
    n = int(rec.n_points)
    if n == 0:
        return np.empty((0, 3), dtype=np.float64)
    stride = max(1, n // max(max_points, 1))
    R_all = Rotation.from_quat(rec.poses[:, 3:7]).as_matrix().astype(np.float32)
    out: list[np.ndarray] = []
    for i in range(rec.n_scans):
        s, e = int(rec.offsets[i]), int(rec.offsets[i + 1])
        if e <= s:
            continue
        # Startversatz mitfuehren, damit der Raster ueber Scangrenzen hinweg passt
        first = (-s) % stride
        if first >= e - s:
            continue
        p = np.asarray(rec.points[s + first:e:stride], dtype=np.float32)
        if len(p):
            out.append(p @ R_all[i].T + rec.poses[i, 0:3].astype(np.float32))
    if not out:
        return np.empty((0, 3), dtype=np.float64)
    return np.vstack(out).astype(np.float64)


def lotrechte_aus_flug(rec, teile) -> np.ndarray | None:
    """Lotrechte (Einheitsvektor nach oben) im aktuellen Weltsystem von ``rec``.

    Mittel der IMU-Beschleunigung ueber den ganzen Flug, je Sample mit der
    interpolierten Pose ins Weltsystem gedreht. Die Bewegungsbeschleunigung
    mittelt sich heraus (sie ist die Geschwindigkeitsaenderung durch die
    Flugdauer, also fast null), uebrig bleibt die Gegenkraft zur Schwerkraft.
    Anders als das Ruhefenster in ``Recording.load`` klappt das auch, wenn die
    Aufnahme erst in der Luft gestartet wurde.

    ``teile``: Liste ``(bag_pfad, scan_von, scan_bis)``, bei einer
    zusammengefuehrten Aufzeichnung je Quelle ein Eintrag. None, wenn es keine
    IMU gibt, der Flug kuerzer als _LOT_MIN_S ist oder die beiden Flughaelften
    mehr als _LOT_MAX_STREUUNG auseinanderliegen.
    """
    from scipy.spatial.transform import Slerp

    from core.bag_reader import BagReader

    stuecke: list[tuple[np.ndarray, np.ndarray]] = []
    for bag, s0, s1 in teile:
        s0, s1 = int(s0), min(int(s1), int(rec.n_scans))
        if s1 - s0 < 2:
            continue
        stamps = np.asarray(rec.stamps[s0:s1], dtype=np.float64)
        # Slerp braucht streng steigende Stempel; doppelte kommen vor
        steigt = np.concatenate([[True], np.diff(stamps) > 0])
        if np.count_nonzero(steigt) < 2:
            continue
        try:
            with BagReader(str(bag)) as reader:
                t, acc = reader.read_imu_accel(float(stamps[0]), float(stamps[-1]))
        except Exception:  # noqa: BLE001 — ohne IMU keine Messung, kein Fehler
            continue
        if len(t) < 2:
            continue
        drehung = Slerp(stamps[steigt],
                        Rotation.from_quat(rec.poses[s0:s1, 3:7][steigt]))(t)
        stuecke.append((t, drehung.apply(acc)))
    if not stuecke:
        return None
    t = np.concatenate([x[0] for x in stuecke])
    welt = np.concatenate([x[1] for x in stuecke])
    if float(t.max() - t.min()) < _LOT_MIN_S:
        return None
    oben = welt.mean(axis=0)
    if float(np.linalg.norm(oben)) < 1e-9:
        return None
    oben /= np.linalg.norm(oben)
    haelfte = len(welt) // 2
    for teil in (welt[:haelfte], welt[haelfte:]):
        v = teil.mean(axis=0)
        v /= max(float(np.linalg.norm(v)), 1e-12)
        if np.degrees(np.arccos(np.clip(v @ oben, -1.0, 1.0))) > _LOT_MAX_STREUUNG:
            return None
    return oben


def kippausgleich(oben_a: np.ndarray, oben_b: np.ndarray) -> tuple[np.ndarray, float]:
    """Kuerzeste Drehung (4x4, um den Ursprung), die ``oben_b`` auf ``oben_a`` legt.

    Die kuerzeste, damit die Gier unangetastet bleibt; die sucht
    :func:`register`. Rueckgabe ``(T, winkel_grad)``.
    """
    a = np.asarray(oben_a, dtype=np.float64) / np.linalg.norm(oben_a)
    b = np.asarray(oben_b, dtype=np.float64) / np.linalg.norm(oben_b)
    achse = np.cross(b, a)
    winkel = float(np.arctan2(np.linalg.norm(achse), float(a @ b)))
    T = np.eye(4)
    if np.linalg.norm(achse) > 1e-12:
        T[:3, :3] = Rotation.from_rotvec(achse / np.linalg.norm(achse) * winkel).as_matrix()
    elif a @ b < 0:  # genau auf dem Kopf: um eine beliebige waagrechte Achse
        T[:3, :3] = Rotation.from_rotvec([np.pi, 0.0, 0.0]).as_matrix()
    return T, float(np.degrees(winkel))


def _o3d_cloud(points: np.ndarray, voxel: float, normals: bool = True):
    import open3d as o3d
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.ascontiguousarray(points, np.float64))
    pc = pc.voxel_down_sample(voxel)
    if normals:
        pc.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30))
    return pc


def register(points_a: np.ndarray, points_b: np.ndarray,
             T_init: np.ndarray | None = None, mode: str = "auto",
             progress_cb=None, cancel=None) -> dict:
    """Starre Transformation, die ``points_b`` auf ``points_a`` legt.

    ``mode="auto"``  probiert Identitaet, Schwerpunktversatz, Drehungen um die
    Hochachse in 30-Grad-Schritten und mehrere Laeufe Fast Global Registration
    ueber FPFH-Merkmale durch und verfeinert jede Startlage mit ICP (Punkt zu
    Ebene) im groben und mittleren Raster. Bewertet wird dort, Trefferquote
    minus Restfehler: das grobe Raster sieht die Gesamtform (Halle samt Hof)
    statt sich wiederholender Einzelheiten wie Dachbinder, und sein weiter
    Fangradius zieht auch Startlagen mit einigen Metern Versatz noch heran.
    Die Drehungen setzen gleich lotrechte Wolken voraus, s. Modulkopf.

    Danach verfeinert ein Feinschliff die beste Lage in mehreren Stufen bis
    _FEIN_ZIEL. Erst der bringt die Genauigkeit; das mittlere Raster ist bei
    grossen Szenen ueber einen Meter grob.

    ``mode="icp"`` laesst die Suche weg und verfeinert nur ``T_init``. Das
    reicht nach einer Handjustage und dauert Sekunden statt Minuten.

    Rueckgabe ``{"T", "fitness", "rmse", "kandidat", "rangliste", "voxel"}``.
    ``fitness`` ist der Anteil der Punkte aus B, die im mittleren Raster in A
    einen Partner finden — unter etwa 0,3 ist die Ausrichtung nicht zu trauen.
    ``rmse`` ist der Restfehler nach dem Feinschliff. ``rangliste`` fuehrt
    ``(bewertung, name, fitness, rmse)`` aller Kandidaten, beste zuerst; liegt
    der zweite knapp hinter dem ersten, war die Wahl nicht eindeutig.
    """
    try:
        import open3d as o3d
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Zum Ausrichten wird open3d gebraucht (pip install open3d).") from exc
    if len(points_a) < 100 or len(points_b) < 100:
        raise RuntimeError("Zu wenige Punkte zum Ausrichten.")

    def prog(f, m):
        if progress_cb is not None:
            progress_cb(f, m)

    lo = np.minimum(points_a.min(axis=0), points_b.min(axis=0))
    hi = np.maximum(points_a.max(axis=0), points_b.max(axis=0))
    diag = float(np.linalg.norm(hi - lo))
    if not np.isfinite(diag) or diag <= 0.0:
        raise RuntimeError("Punktwolken haben keine brauchbare Ausdehnung.")
    v_grob = max(0.05, diag / 80.0)
    v_mittel = max(0.02, diag / 200.0)

    prog(0.05, f"Dünne aus ({v_grob:.2f}/{v_mittel:.2f} m) …")
    _check_cancel(cancel)
    a_grob, b_grob = _o3d_cloud(points_a, v_grob), _o3d_cloud(points_b, v_grob)
    a_mit, b_mit = _o3d_cloud(points_a, v_mittel), _o3d_cloud(points_b, v_mittel)

    def icp(src, dst, T, thresh, iters):
        return o3d.pipelines.registration.registration_icp(
            src, dst, thresh, T,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iters))

    kandidaten: list[tuple[str, np.ndarray]] = []
    if mode == "icp":
        if T_init is None:
            raise RuntimeError("Für 'Nur fein ausrichten (ICP)' wird eine Ausgangslage gebraucht.")
        kandidaten.append(("Handjustage", np.asarray(T_init, dtype=np.float64)))
    else:
        ca = np.asarray(a_grob.get_center())
        cb = np.asarray(b_grob.get_center())
        T_zentrum = np.eye(4)
        T_zentrum[:3, 3] = ca - cb
        kandidaten.append(("Identität", np.eye(4)))
        kandidaten.append(("Schwerpunkt", T_zentrum))
        if T_init is not None:
            kandidaten.append(("Handjustage", np.asarray(T_init, dtype=np.float64)))
        for yaw in _YAW_CANDIDATES:  # um die Hochachse, um den Zielschwerpunkt
            a_rad = np.radians(yaw)
            R = np.eye(4)
            R[0, 0] = np.cos(a_rad); R[0, 1] = -np.sin(a_rad)
            R[1, 0] = np.sin(a_rad); R[1, 1] = np.cos(a_rad)
            T_hin = np.eye(4); T_hin[:3, 3] = ca
            T_weg = np.eye(4); T_weg[:3, 3] = -cb
            kandidaten.append((f"Gier {yaw}°", T_hin @ R @ T_weg))

        prog(0.15, "Merkmale (FPFH) …")
        _check_cancel(cancel)
        rf = v_grob * 5.0
        fa = o3d.pipelines.registration.compute_fpfh_feature(
            a_grob, o3d.geometry.KDTreeSearchParamHybrid(radius=rf, max_nn=100))
        fb = o3d.pipelines.registration.compute_fpfh_feature(
            b_grob, o3d.geometry.KDTreeSearchParamHybrid(radius=rf, max_nn=100))
        for k in range(_FGR_TRIALS):
            _check_cancel(cancel)
            prog(0.15 + 0.15 * (k + 1) / _FGR_TRIALS,
                 f"Globale Suche {k + 1}/{_FGR_TRIALS} …")
            try:
                res = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
                    b_grob, a_grob, fb, fa,
                    o3d.pipelines.registration.FastGlobalRegistrationOption(
                        maximum_correspondence_distance=v_grob * 0.5))
                kandidaten.append((f"FGR {k + 1}", np.asarray(res.transformation)))
            except Exception:  # noqa: BLE001 — ein Fehlschlag ist nur ein Kandidat weniger
                pass

    def bewerten(res) -> float:
        if res.fitness < 0.01:
            return -1e9
        return res.fitness - res.inlier_rmse / max(v_mittel, 1e-6)

    bestes: tuple[float, str, np.ndarray, float, float] | None = None
    rangliste: list[tuple[float, str, float, float]] = []
    for i, (name, T0) in enumerate(kandidaten):
        _check_cancel(cancel)
        prog(0.3 + 0.5 * i / max(len(kandidaten), 1),
             f"Verfeinere {name} ({i + 1}/{len(kandidaten)}) …")
        T = np.asarray(T0, dtype=np.float64)
        res = None
        try:
            for wolken, voxel, iters in ((("grob", b_grob, a_grob), v_grob, 60),
                                         (("mittel", b_mit, a_mit), v_mittel, 80)):
                _, src, dst = wolken
                for faktor in (4.0, 2.0, 1.0):
                    res = icp(src, dst, T, voxel * faktor, iters)
                    T = res.transformation
        except Exception:  # noqa: BLE001 — Kandidat gescheitert, naechster
            continue
        if res is None:
            continue
        s = bewerten(res)
        rangliste.append((float(s), name, float(res.fitness), float(res.inlier_rmse)))
        if bestes is None or s > bestes[0]:
            bestes = (s, name, np.asarray(T, dtype=np.float64),
                      float(res.fitness), float(res.inlier_rmse))

    if bestes is None:
        raise RuntimeError(
            "Ausrichten fehlgeschlagen — die Wolken überlappen zu wenig. "
            "Erst von Hand grob zusammenschieben, dann 'Nur fein ausrichten (ICP)'.")

    _, name, T, fit, rmse = bestes
    # Feinschliff in Stufen: jede teilt das Voxel durch 2,5, bis _FEIN_ZIEL
    # erreicht ist (bei kleinen Szenen mindestens eine Stufe unter v_mittel).
    ziel = min(_FEIN_ZIEL, v_mittel / 2.0)
    stufen: list[float] = []
    v = v_mittel
    while v > ziel * 1.01:
        v = max(v / 2.5, ziel)
        stufen.append(v)
    a_f = b_f = None
    T_fein = np.asarray(T, dtype=np.float64)
    try:
        for k, v in enumerate(stufen):
            _check_cancel(cancel)
            prog(0.85 + 0.13 * k / len(stufen), f"Feinschliff {v:.2f} m …")
            a_f, b_f = _o3d_cloud(points_a, v), _o3d_cloud(points_b, v)
            for faktor in (2.0, 1.0):
                T_fein = np.asarray(icp(b_f, a_f, T_fein, v * faktor, 60).transformation)
        # Vorher und nachher bei DERSELBEN Schwelle messen: bei kleinerer Schwelle
        # faellt die Trefferquote von selbst, das sagt nichts ueber die Lage.
        schwelle = 2.0 * stufen[-1]
        reg = o3d.pipelines.registration
        vorher = reg.evaluate_registration(b_f, a_f, schwelle, T)
        nachher = reg.evaluate_registration(b_f, a_f, schwelle, T_fein)
        if nachher.fitness >= vorher.fitness:
            T, rmse = T_fein, float(nachher.inlier_rmse)
        else:
            rmse = float(vorher.inlier_rmse)
    except Exception:  # noqa: BLE001 — der Wert aus dem mittleren Raster steht schon
        _check_cancel(cancel)   # ein Abbruch soll nicht als Fehlschlag durchgehen
    prog(1.0, f"Ausgerichtet: Trefferquote {fit:.2f}, Restfehler {rmse:.3f} m")
    # beschreibbare Kopie: open3d-Transformationen sind read-only
    return {"T": np.array(T, dtype=np.float64), "fitness": fit, "rmse": rmse,
            "kandidat": name, "rangliste": sorted(rangliste, reverse=True),
            "voxel": (v_grob, v_mittel, stufen[-1] if stufen else v_mittel)}


def transform_poses(poses: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Posen in ein anderes Weltsystem drehen/schieben: T_neu = T @ T_alt.

    ``np.array`` statt ``asarray``: open3d gibt seine Transformationen
    schreibgeschuetzt zurueck, und scipy braucht einen beschreibbaren Puffer.
    """
    T = np.array(T, dtype=np.float64)
    R = Rotation.from_matrix(T[:3, :3])
    out = np.array(poses, dtype=np.float64, copy=True)
    out[:, 0:3] = R.apply(poses[:, 0:3]) + T[:3, 3]
    out[:, 3:7] = (R * Rotation.from_quat(poses[:, 3:7])).as_quat()
    return out


def lage_aus_reglern(yaw_deg: float, versatz_xyz, zentrum,
                     basis: np.ndarray | None = None) -> np.ndarray:
    """Lage aus der Handjustage: um die Mitte der zweiten Wolke gieren, dann schieben.

    Die Regler sind ein Versatz zu ``basis`` (der Lage der letzten Ausrichtung,
    ohne Angabe die Einheit). ``zentrum`` ist der Schwerpunkt der Wolke vor
    ``basis``; gegiert wird um seinen Ort nach ``basis``. Ergebnis: 4x4.
    """
    basis = np.eye(4) if basis is None else np.asarray(basis, dtype=np.float64)
    mitte = basis[:3, :3] @ np.asarray(zentrum) + basis[:3, 3]
    yaw = np.radians(float(yaw_deg))
    R = np.eye(4)
    R[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
    hin = np.eye(4)
    hin[:3, 3] = mitte
    weg = np.eye(4)
    weg[:3, 3] = -mitte
    D = hin @ R @ weg
    D[:3, 3] += [float(v) for v in versatz_xyz]
    return D @ basis


def _append_binary(fh, arr, chunk: int = _WRITE_CHUNK) -> None:
    """Grosse memmap-Arrays blockweise anhaengen (haelt den Speicher klein)."""
    n = len(arr)
    for i in range(0, n, chunk):
        np.asarray(arr[i:i + chunk]).tofile(fh)


def merge_recordings(rec_a, rec_b, T_ab: np.ndarray, out_dir: str,
                     bag_a: str, bag_b: str, info: dict | None = None,
                     progress_cb=None, cancel=None) -> dict:
    """Schreibt eine neue Aufzeichnung aus A und B nach ``out_dir``.

    ``T_ab`` legt B in das Weltsystem von A. Beide Aufzeichnungen muessen
    bereits lotrecht geladen sein (``Recording.load``), die Kippkorrektur wird
    hier nicht noch einmal angewandt — die meta.json vermerkt das.

    Die Reihenfolge richtet sich nach den Zeitstempeln, nicht nach A/B.
    """
    def prog(f, m):
        if progress_cb is not None:
            progress_cb(f, m)

    if rec_a.n_scans == 0 or rec_b.n_scans == 0:
        raise RuntimeError("Eine der beiden Aufzeichnungen ist leer.")
    a_von, a_bis = float(rec_a.stamps[0]), float(rec_a.stamps[-1])
    b_von, b_bis = float(rec_b.stamps[0]), float(rec_b.stamps[-1])
    if a_von <= b_bis and b_von <= a_bis:
        raise RuntimeError(
            "Die beiden Aufnahmen überlappen sich zeitlich "
            f"({time.strftime('%H:%M:%S', time.localtime(max(a_von, b_von)))} bis "
            f"{time.strftime('%H:%M:%S', time.localtime(min(a_bis, b_bis)))}). "
            "Zusammenführen ist nur für nacheinander aufgenommene Flüge "
            "vorgesehen, sonst wäre die Scan-Reihenfolge nicht mehr eindeutig.")

    # frueherer Flug zuerst, damit stamps.npy aufsteigend bleibt
    if a_von <= b_von:
        teile = [(rec_a, bag_a, np.eye(4)), (rec_b, bag_b, np.asarray(T_ab, np.float64))]
    else:
        teile = [(rec_b, bag_b, np.asarray(T_ab, np.float64)), (rec_a, bag_a, np.eye(4))]

    os.makedirs(out_dir, exist_ok=True)
    tmp = out_dir + ".tmp"
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)

    quellen: list[dict] = []
    scan0 = 0
    pkt0 = 0
    offsets = [np.zeros(1, dtype=np.int64)]
    stamps: list[np.ndarray] = []
    poses: list[np.ndarray] = []
    try:
        with open(os.path.join(tmp, "points.bin"), "wb") as fp, \
                open(os.path.join(tmp, "intensity.bin"), "wb") as fi:
            for k, (rec, bag, T) in enumerate(teile):
                _check_cancel(cancel)
                prog(0.05 + 0.8 * k / len(teile),
                     f"Schreibe Teil {k + 1}/{len(teile)} "
                     f"({rec.n_points / 1e6:.1f} Mio. Punkte) …")
                _append_binary(fp, rec.points)
                _append_binary(fi, rec.intensity)
                offsets.append(np.asarray(rec.offsets[1:], dtype=np.int64) + pkt0)
                stamps.append(np.asarray(rec.stamps, dtype=np.float64))
                poses.append(rec.poses if np.allclose(T, np.eye(4))
                             else transform_poses(rec.poses, T))
                quellen.append({
                    "bag": str(bag),
                    "scan_range": [scan0, scan0 + int(rec.n_scans)],
                    "point_range": [pkt0, pkt0 + int(rec.n_points)],
                    "T_in_merged": np.asarray(T, dtype=float).tolist(),
                })
                scan0 += int(rec.n_scans)
                pkt0 += int(rec.n_points)

        prog(0.9, "Schreibe Posen und Stempel …")
        np.save(os.path.join(tmp, "offsets.npy"), np.concatenate(offsets))
        np.save(os.path.join(tmp, "stamps.npy"), np.concatenate(stamps))
        np.save(os.path.join(tmp, "poses.npy"), np.concatenate(poses))
        meta = {
            "bag": quellen[0]["bag"],
            "config": rec_a.meta.get("config"),
            "n_scans": scan0,
            "n_points": pkt0,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "merged": {
                "T_ab": np.asarray(T_ab, dtype=float).tolist(),
                **(info or {}),
            },
            "sources": quellen,
            # Beide Teile kamen lotrecht herein; nicht noch einmal drehen.
            "gravity_level": {
                "quat": [0.0, 0.0, 0.0, 1.0],
                "tilt_deg": 0.0,
                "threshold_deg": 0.0,
                "applied": False,
                "note": "beim Zusammenführen bereits angewandt",
            },
        }
        with open(os.path.join(tmp, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        prog(0.97, "Übernehme …")
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        os.rename(tmp, out_dir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    prog(1.0, f"Zusammengeführt: {scan0} Scans, {pkt0} Punkte")
    return meta


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Selbsttest: synthetische Wolke, bekannt verdreht -> zurueckfinden,
    # dann zwei Aufzeichnungen schreiben und wieder einlesen.
    # ------------------------------------------------------------------
    import tempfile
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.recording import Recording

    rng = np.random.default_rng(7)
    print("== Test 1: register findet eine bekannte Lage zurueck ==")
    # Raum mit Boden, zwei Waenden und einer Saeule: genug Struktur fuer ICP
    boden = rng.uniform([-8, -8, 0], [8, 8, 0.02], size=(30000, 3))
    wand1 = np.stack([np.full(12000, -8.0), rng.uniform(-8, 8, 12000),
                      rng.uniform(0, 4, 12000)], axis=1)
    wand2 = np.stack([rng.uniform(-8, 8, 12000), np.full(12000, 8.0),
                      rng.uniform(0, 4, 12000)], axis=1)
    saeule = np.stack([2 + 0.4 * np.cos(rng.uniform(0, 6.28, 8000)),
                       -3 + 0.4 * np.sin(rng.uniform(0, 6.28, 8000)),
                       rng.uniform(0, 4, 8000)], axis=1)
    A = np.vstack([boden, wand1, wand2, saeule])

    yaw = np.radians(35.0)
    T_wahr = np.eye(4)
    T_wahr[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
    T_wahr[:3, 3] = [3.0, -2.0, 0.4]
    B = (np.linalg.inv(T_wahr)[:3, :3] @ (A - T_wahr[:3, 3]).T).T  # B liegt verdreht

    res = register(A, B, progress_cb=lambda f, m: None)
    T_err = np.linalg.inv(T_wahr) @ res["T"]
    d_rot = np.degrees(np.linalg.norm(Rotation.from_matrix(T_err[:3, :3]).as_rotvec()))
    d_t = float(np.linalg.norm(T_err[:3, 3]))
    print(f"  Kandidat={res['kandidat']}  Trefferquote={res['fitness']:.3f}  "
          f"Restfehler={res['rmse']:.3f} m")
    print(f"  Restfehler gegen die wahre Lage: {d_rot:.2f} Grad, {d_t:.3f} m")
    assert res["fitness"] > 0.8, res["fitness"]
    assert d_rot < 1.0 and d_t < 0.10, (d_rot, d_t)

    print("== Test 2: merge_recordings schreibt eine lesbare Aufzeichnung ==")
    out = tempfile.mkdtemp(prefix="mergetest_")

    def bauen(n_scans, t0, punkte_je_scan=50):
        pts = rng.uniform(-1, 1, size=(n_scans * punkte_je_scan, 3)).astype(np.float32)
        off = np.arange(n_scans + 1, dtype=np.int64) * punkte_je_scan
        st = t0 + np.arange(n_scans, dtype=np.float64) * 0.1
        po = np.zeros((n_scans, 7), dtype=np.float64)
        po[:, 0] = np.arange(n_scans)          # Flugbahn laeuft in x
        po[:, 6] = 1.0                          # Einheitsquaternion
        return Recording(points=pts, intensity=np.ones(len(pts), np.float32),
                         offsets=off, stamps=st, poses=po,
                         meta={"bag": "/tmp/a", "config": "whs_dense.yaml"})

    ra = bauen(20, 1000.0)
    rb = bauen(15, 2000.0)
    T_ab = np.eye(4)
    T_ab[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    T_ab[:3, 3] = [10.0, 0.0, 0.0]
    meta = merge_recordings(ra, rb, T_ab, os.path.join(out, "recording"),
                            "/tmp/a", "/tmp/b", progress_cb=lambda f, m: None)
    m = Recording.load(os.path.join(out, "recording"))
    print(f"  {m.n_scans} Scans, {m.n_points} Punkte, "
          f"{len(m.meta['sources'])} Quellen")
    assert m.n_scans == 35 and m.n_points == 35 * 50
    assert np.all(np.diff(m.stamps) > 0), "Stempel nicht aufsteigend"
    assert m.gravity_level is not None and not m.gravity_level["applied"]
    # B-Posen muessen gedreht und verschoben sein: x -> 10, Flugbahn laeuft in y
    pb = m.poses[20:]
    assert abs(pb[0, 0] - 10.0) < 1e-9 and abs(pb[0, 1]) < 1e-9, pb[0]
    assert abs(pb[5, 1] - 5.0) < 1e-9, pb[5]
    print(f"  Posen von B gedreht: erste {np.round(pb[0, :3], 3)}, "
          f"sechste {np.round(pb[5, :3], 3)} (Gier 90 Grad, +10 m in x)")
    # Reihenfolge nach Zeit, nicht nach Argument
    meta2 = merge_recordings(rb, ra, np.eye(4), os.path.join(out, "rec2"),
                             "/tmp/b", "/tmp/a", progress_cb=lambda f, m: None)
    assert meta2["sources"][0]["bag"] == "/tmp/a", "frueherer Flug nicht zuerst"
    print("  Reihenfolge richtet sich nach den Zeitstempeln, nicht nach A/B")
    # Ueberlappende Zeitraeume muessen abgelehnt werden
    try:
        merge_recordings(ra, bauen(10, 1000.5), np.eye(4),
                         os.path.join(out, "rec3"), "/tmp/a", "/tmp/c",
                         progress_cb=lambda f, m: None)
    except RuntimeError as exc:
        print(f"  Ueberlappung abgelehnt: {str(exc)[:60]}…")
    else:
        raise AssertionError("Ueberlappung wurde nicht abgelehnt")
    shutil.rmtree(out, ignore_errors=True)

    print("== Test 3: lage_aus_reglern rechnet wie die Handjustage ==")

    def ui_formel(yaw_deg, versatz, zentrum, basis):
        # Rechnung der Handregler im Fenster Zusammenfuehren, Schritt fuer Schritt
        mitte = basis[:3, :3] @ zentrum + basis[:3, 3]
        yaw = np.radians(float(yaw_deg))
        R = np.eye(4)
        R[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
        hin = np.eye(4)
        hin[:3, 3] = mitte
        weg = np.eye(4)
        weg[:3, 3] = -mitte
        D = hin @ R @ weg
        D[:3, 3] += [float(v) for v in versatz]
        return D @ basis

    c = np.array([3.2, -1.7, 0.45])
    assert np.array_equal(lage_aus_reglern(0, (0, 0, 0), c), np.eye(4))
    basis_b = np.eye(4)
    basis_b[:3, :3] = Rotation.from_euler("zyx", [112.0, 1.5, -0.8],
                                          degrees=True).as_matrix()
    basis_b[:3, 3] = [-14.25, 6.5, 1.125]
    faelle = [(37.5, (1.25, -0.4, 0.1)), (-120.0, (0.0, 3.05, -0.02)),
              (359.9, (-7.5, 0.005, 2.0))]
    for yaw_deg, versatz in faelle:
        assert np.array_equal(lage_aus_reglern(yaw_deg, versatz, c),
                              ui_formel(yaw_deg, versatz, c, np.eye(4))), yaw_deg
        assert np.array_equal(lage_aus_reglern(yaw_deg, versatz, c, basis_b),
                              ui_formel(yaw_deg, versatz, c, basis_b)), yaw_deg
    print(f"  Einheit bei Nullreglern, {len(faelle)} Faelle je Basis bitgleich")
    print("merge SELFTEST OK")
