"""Datenformat der FAST-LIO-Aufzeichnung (recording/-Verzeichnis).

Format auf Platte:
    points.bin     float32, N x 3 konkateniert (Body/IMU-Frame, Scan-Reihenfolge)
    intensity.bin  float32, N
    offsets.npy    int64, (S+1,)   Scan i = points[offsets[i]:offsets[i+1]]
    stamps.npy     float64, (S,)   lidar_end_time je Scan
    poses.npy      float64, (S,7)  x y z qx qy qz qw  (T_world_imu)
    meta.json      {"bag": ..., "n_scans": S, "n_points": N, ...}

Kippkorrektur: FAST-LIO verankert sein Weltsystem in der IMU-Lage des ersten
Scans und richtet es NICHT an der Schwerkraft aus. Steht der Livox schraeg auf
der Drohne, erbt die ganze Karte diese Schraeglage — nicht nur der erste Scan,
der legt sie nur fest. Recording.load() misst die Lotrechte aus dem Ruhefenster
am Bag-Anfang und dreht sie heraus, aber erst ab LEVEL_MIN_TILT_DEG, damit
sauber montierte Fluege unveraendert bleiben. Gedreht wird nur das Weltsystem
(Posen); Punkte im Body-Frame und Kamera-Extrinsik bleiben unberuehrt, deshalb
aendern Einfaerbung und Farb-Cache sich dadurch nicht.

Qt-frei. points/intensity werden als np.memmap (read-only) geladen.
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from core.gemeinsam import write_json_atomic

_EDGE_TOL_S = 0.15  # Randtoleranz fuer interpolate_pose
_PROGRESS_EVERY = 50  # Scans zwischen zwei progress_cb-Aufrufen

# Ab dieser Schraeglage wird die Karte lotrecht gedreht. Darunter bleibt sie
# unangetastet: die sauber montierten Fluege liegen bei 0.3-5.6 Grad, das ist
# Montagetoleranz und keine Kippung, und ihre Karten sollen bitgleich bleiben.
LEVEL_MIN_TILT_DEG = 10.0


_write_json_atomic = write_json_atomic


def level_rotation(up_world: np.ndarray) -> Rotation:
    """Kuerzeste Drehung, die ``up_world`` auf die Welt-z-Achse legt.

    Bewusst die kuerzeste: sie kippt nur und dreht den Gierwinkel nicht mit,
    damit die Karte ihre Ausrichtung in der Ebene behaelt (die GPS-Georeferen-
    zierung bestimmt den Gierwinkel spaeter selbst).
    """
    v = np.asarray(up_world, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return Rotation.identity()
    v = v / n
    axis = np.cross(v, [0.0, 0.0, 1.0])
    sin_a = float(np.linalg.norm(axis))
    cos_a = float(v[2])
    if sin_a < 1e-12:
        # schon lotrecht, oder exakt auf dem Kopf (dann um x drehen)
        return Rotation.identity() if cos_a > 0 else Rotation.from_rotvec([np.pi, 0.0, 0.0])
    return Rotation.from_rotvec(axis / sin_a * np.arctan2(sin_a, cos_a))


def _rotate_world(poses: np.ndarray, rot: Rotation) -> np.ndarray:
    """Weltsystem drehen: T_neu = rot * T_alt (Body-Punkte bleiben, wie sie sind)."""
    out = np.array(poses, dtype=np.float64, copy=True)
    out[:, 0:3] = rot.apply(poses[:, 0:3])
    out[:, 3:7] = (rot * Rotation.from_quat(poses[:, 3:7])).as_quat()
    return out


def _measure_gravity_level(rec_dir: str, meta: dict, poses: np.ndarray,
                          stamps: np.ndarray, bag_path: str | None = None) -> dict | None:
    """Kippkorrektur bestimmen und in meta.json festschreiben.

    Einmal je Aufzeichnung: das Ergebnis landet unter "gravity_level" in der
    meta.json und wird beim naechsten Laden von dort gelesen. Schlaegt die
    Messung fehl (Bag verschoben, kein IMU, Drohne von Anfang an in Bewegung),
    wird nichts geschrieben und beim naechsten Mal erneut versucht.
    """
    cached = meta.get("gravity_level")
    if isinstance(cached, dict) and "quat" in cached:
        return cached

    # Der Pfad in der meta.json zeigt ins Leere, sobald das Bag nach der
    # Aufzeichnung umbenannt wurde; der Aufrufer kennt den aktuellen Ort.
    bag = bag_path or meta.get("bag")
    if not bag or not os.path.exists(str(bag)):
        return None
    try:
        # lazy: zieht cv2/rosbags nur nach, wenn wirklich gemessen wird
        from core.bag_reader import BagReader
        with BagReader(str(bag)) as reader:
            rest = reader.read_imu_up()
    except Exception:  # noqa: BLE001 — ohne Messung bleibt die Karte, wie sie ist
        return None
    if rest is None:
        return None

    # Lotrechte vom Sensor- ins Weltsystem: mit den Posen der Scans, die noch
    # ins Messfenster fallen. Gibt es keine, ist das Weltsystem laut FAST-LIO
    # die Anfangslage des Sensors und der Vektor gilt unveraendert.
    in_rest = stamps <= rest.t_end
    if len(poses) and bool(np.any(in_rest)):
        up_world = Rotation.from_quat(poses[in_rest, 3:7]).apply(rest.up_body).mean(axis=0)
    else:
        up_world = np.asarray(rest.up_body, dtype=np.float64)

    rot = level_rotation(up_world)
    nrm = float(np.linalg.norm(up_world))
    tilt = float(np.degrees(np.arccos(np.clip(up_world[2] / nrm if nrm else 1.0, -1.0, 1.0))))
    level = {
        "quat": [float(x) for x in rot.as_quat()],
        "tilt_deg": tilt,
        "threshold_deg": LEVEL_MIN_TILT_DEG,
        "applied": bool(tilt >= LEVEL_MIN_TILT_DEG),
        "up_body": [float(x) for x in rest.up_body],
        "mount_tilt_deg": float(rest.tilt_deg),
        "window_s": float(rest.window_s),
        "samples": int(rest.n_samples),
        "spread_deg": float(rest.spread_deg),
        "scans_in_window": int(np.count_nonzero(in_rest)),
    }
    meta["gravity_level"] = level
    try:
        _write_json_atomic(os.path.join(rec_dir, "meta.json"), meta)
    except OSError:
        pass  # nur ein Cache-Eintrag; die Korrektur gilt trotzdem
    return level


class Recording:
    """Geladene FAST-LIO-Aufzeichnung: Punkte im Body-Frame + Pose je Scan."""

    points: np.ndarray  # float32 (N,3), memmap read-only
    intensity: np.ndarray  # float32 (N,), memmap read-only
    offsets: np.ndarray  # int64 (S+1,)
    stamps: np.ndarray  # float64 (S,)
    poses: np.ndarray  # float64 (S,7): x y z qx qy qz qw (ggf. lotrecht gedreht)
    meta: dict
    gravity_level: dict | None  # Kippkorrektur, s. _measure_gravity_level

    def __init__(
        self,
        points: np.ndarray,
        intensity: np.ndarray,
        offsets: np.ndarray,
        stamps: np.ndarray,
        poses: np.ndarray,
        meta: dict,
        gravity_level: dict | None = None,
    ):
        self.points = points
        self.intensity = intensity
        self.offsets = offsets
        self.stamps = stamps
        self.poses = poses
        self.meta = meta
        self.gravity_level = gravity_level

    @property
    def n_scans(self) -> int:
        return len(self.stamps)

    @property
    def n_points(self) -> int:
        return self.points.shape[0]

    @staticmethod
    def load(dir_path: str, level: bool = True, bag_path: str | None = None) -> "Recording":
        """Laedt eine Aufzeichnung; points/intensity als read-only np.memmap.

        Mit ``level=True`` (Standard) wird die Karte lotrecht gedreht, sofern der
        Livox schraeger als LEVEL_MIN_TILT_DEG montiert war; s. Modulkopf. Dafuer
        wird das Bag noch einmal kurz gelesen — ``bag_path`` uebersteuert den in
        der meta.json gespeicherten Pfad, der nach einem Umbenennen ins Leere zeigt.
        """
        d = str(dir_path)
        if not os.path.isdir(d):
            raise RuntimeError(f"Aufzeichnungs-Verzeichnis nicht gefunden: {d}")
        needed = ["points.bin", "intensity.bin", "offsets.npy", "stamps.npy", "poses.npy", "meta.json"]
        missing = [f for f in needed if not os.path.isfile(os.path.join(d, f))]
        if missing:
            raise RuntimeError(f"Aufzeichnung unvollstaendig in {d}: {', '.join(missing)} fehlt.")

        with open(os.path.join(d, "meta.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        # Leere Aufzeichnung (abgebrochener/fehlgeschlagener FAST-LIO-Lauf):
        # klar deutsch melden statt ValueError('cannot mmap an empty file').
        n_scans_meta = meta.get("n_scans")
        if os.path.getsize(os.path.join(d, "points.bin")) == 0 or (
                n_scans_meta is not None and int(n_scans_meta) <= 0):
            raise RuntimeError(
                f"Aufzeichnung ist leer — FAST-LIO-Lauf war unvollständig oder wurde "
                f"abgebrochen ({d}). Bitte die Karte neu berechnen ('Karte berechnen')."
            )
        if os.path.getsize(os.path.join(d, "intensity.bin")) == 0:
            raise RuntimeError(
                f"Aufzeichnung inkonsistent in {d}: intensity.bin ist leer, points.bin nicht."
            )

        points = np.memmap(os.path.join(d, "points.bin"), dtype=np.float32, mode="r")
        if points.size % 3 != 0:
            raise RuntimeError(f"points.bin in {d} ist beschaedigt (Groesse nicht durch 3 teilbar).")
        points = points.reshape(-1, 3)
        intensity = np.memmap(os.path.join(d, "intensity.bin"), dtype=np.float32, mode="r")
        offsets = np.load(os.path.join(d, "offsets.npy"))
        stamps = np.load(os.path.join(d, "stamps.npy"))
        poses = np.load(os.path.join(d, "poses.npy"))

        S = len(stamps)
        ok = (
            offsets.ndim == 1
            and len(offsets) == S + 1
            and poses.shape == (S, 7)
            and int(offsets[0]) == 0
            and int(offsets[-1]) == points.shape[0]
            and intensity.shape[0] == points.shape[0]
            and bool(np.all(np.diff(offsets) >= 0))
        )
        if not ok:
            raise RuntimeError(
                f"Aufzeichnung inkonsistent in {d}: offsets/stamps/poses passen nicht zu den Punktdaten."
            )
        stamps = stamps.astype(np.float64, copy=False)
        poses = poses.astype(np.float64, copy=False)
        gravity_level = (_measure_gravity_level(d, meta, poses, stamps, bag_path)
                         if level else None)
        if gravity_level is not None and gravity_level.get("applied"):
            poses = _rotate_world(poses, Rotation.from_quat(gravity_level["quat"]))
        return Recording(
            points=points,
            intensity=intensity,
            offsets=offsets.astype(np.int64, copy=False),
            stamps=stamps,
            poses=poses,
            meta=meta,
            gravity_level=gravity_level,
        )

    # ---------------------------------------------------------------- world

    def world_points(self, progress_cb=None, cancel=None) -> np.ndarray:
        """Alle Punkte ins Weltsystem (camera_init) transformiert; float32 (N,3).

        p_w = R(q) @ p_body + t, vektorisiert je Scan.
        """
        S = self.n_scans
        out = np.empty((self.n_points, 3), dtype=np.float32)
        if S == 0:
            return out
        rot_mats = Rotation.from_quat(self.poses[:, 3:7]).as_matrix().astype(np.float32)
        trans = self.poses[:, 0:3].astype(np.float32)
        for i in range(S):
            if cancel is not None and cancel.is_set():
                raise RuntimeError("Abgebrochen")
            s, e = int(self.offsets[i]), int(self.offsets[i + 1])
            if e > s:
                np.matmul(self.points[s:e], rot_mats[i].T, out=out[s:e])
                out[s:e] += trans[i]
            if progress_cb is not None and (i % _PROGRESS_EVERY == 0 or i == S - 1):
                progress_cb((i + 1) / S, f"Transformiere Scan {i + 1}/{S}")
        return out

    # ----------------------------------------------------------------- pose

    def _pose_matrix(self, i: int) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = Rotation.from_quat(self.poses[i, 3:7]).as_matrix()
        T[:3, 3] = self.poses[i, 0:3]
        return T

    def interpolate_pose(self, t: float) -> np.ndarray | None:
        """4x4 T_world_imu zur Zeit t: SLERP(Rotation) + linear(Translation).

        Ausserhalb der Scan-Stempel: bis 0.15 s Randpose halten, sonst None.
        """
        S = self.n_scans
        if S == 0:
            return None
        stamps = self.stamps
        if t <= stamps[0]:
            return self._pose_matrix(0) if stamps[0] - t <= _EDGE_TOL_S else None
        if t >= stamps[-1]:
            return self._pose_matrix(S - 1) if t - stamps[-1] <= _EDGE_TOL_S else None

        i1 = int(np.searchsorted(stamps, t, side="left"))
        i0 = i1 - 1
        t0, t1 = float(stamps[i0]), float(stamps[i1])
        if t1 <= t0:  # doppelte Stempel
            return self._pose_matrix(i0)
        alpha = (t - t0) / (t1 - t0)
        trans = (1.0 - alpha) * self.poses[i0, 0:3] + alpha * self.poses[i1, 0:3]
        rots = Rotation.from_quat(self.poses[[i0, i1], 3:7])
        rot = Slerp([t0, t1], rots)([t])[0]
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rot.as_matrix()
        T[:3, 3] = trans
        return T

    def level_note(self) -> str | None:
        """Einzeiler fuers Protokoll, oder None wenn nichts zu melden ist."""
        lvl = self.gravity_level
        if not lvl:
            return None
        tilt = float(lvl.get("tilt_deg", 0.0))
        if lvl.get("applied"):
            return (f"Karte lotrecht gedreht: Livox war {tilt:.1f}° schräg montiert "
                    f"(gemessen über {lvl.get('window_s', 0.0):.2f} s, "
                    f"{lvl.get('samples', 0)} IMU-Samples, "
                    f"Streuung {lvl.get('spread_deg', 0.0):.1f}°).")
        return (f"Einbaulage {tilt:.1f}° — unter der Schwelle von "
                f"{lvl.get('threshold_deg', LEVEL_MIN_TILT_DEG):.0f}°, Karte unverändert.")

    def path_positions(self) -> np.ndarray:
        """(S,3) Trajektorie (Positionen der Scan-Posen)."""
        return np.array(self.poses[:, 0:3], dtype=np.float64, copy=True)


def lade_mit_hinweis(rec_dir: str, bag_path: str | None, log, praefix: str = "") -> Recording:
    """Recording.load und die Zeile zur Kippkorrektur an ``log``, sofern es eine gibt."""
    rec = Recording.load(rec_dir, bag_path=bag_path)
    note = rec.level_note()
    if note:
        log(f"{praefix}{note}")
    return rec


if __name__ == "__main__":
    import threading
    import time

    OUT = (
        "/tmp/super360_modtests/"
        "recording"
    )
    REC_DIR = os.path.join(OUT, "synthetic_recording")
    os.makedirs(REC_DIR, exist_ok=True)

    # --- synthetic recording: 3 scans, known poses incl. 90 deg yaw ---------
    pts0 = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    pts1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    pts2 = np.array([[2, 0, 0]], dtype=np.float32)
    points = np.concatenate([pts0, pts1, pts2], axis=0)
    intensity = np.arange(len(points), dtype=np.float32)
    offsets = np.array([0, 2, 5, 6], dtype=np.int64)
    stamps = np.array([0.0, 1.0, 2.0], dtype=np.float64)

    s2 = np.sqrt(0.5)
    poses = np.array(
        [
            [0, 0, 0, 0, 0, 0, 1],        # identity
            [1, 0, 0, 0, 0, s2, s2],      # +90 deg yaw, t=(1,0,0)
            [2, 0, 3, 0, 0, 1, 0],        # 180 deg yaw, t=(2,0,3)
        ],
        dtype=np.float64,
    )

    points.tofile(os.path.join(REC_DIR, "points.bin"))
    intensity.tofile(os.path.join(REC_DIR, "intensity.bin"))
    np.save(os.path.join(REC_DIR, "offsets.npy"), offsets)
    np.save(os.path.join(REC_DIR, "stamps.npy"), stamps)
    np.save(os.path.join(REC_DIR, "poses.npy"), poses)
    with open(os.path.join(REC_DIR, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {"bag": "synthetic", "config": "whs_dense.yaml", "n_scans": 3, "n_points": 6,
             "expected_scans": 3, "rate": 1.0, "created": time.time()},
            fh,
        )

    rec = Recording.load(REC_DIR)
    print(f"loaded: n_scans={rec.n_scans} n_points={rec.n_points} "
          f"points memmap={isinstance(rec.points, np.memmap)}")
    assert isinstance(rec.points, np.memmap) and isinstance(rec.intensity, np.memmap)
    assert rec.n_scans == 3 and rec.n_points == 6

    # --- world_points vs. hand-computed -------------------------------------
    progress_msgs: list[tuple[float, str]] = []
    w = rec.world_points(progress_cb=lambda f, m: progress_msgs.append((f, m)))
    expected = np.array(
        [
            [1, 0, 0], [0, 1, 0],              # scan0: identity
            [1, 1, 0], [0, 0, 0], [1, 0, 1],   # scan1: yaw90 + (1,0,0)
            [0, 0, 3],                          # scan2: yaw180 + (2,0,3)
        ],
        dtype=np.float32,
    )
    err = np.abs(w - expected).max()
    print(f"world_points: dtype={w.dtype} shape={w.shape} max|err|={err:.2e}")
    print(f"progress calls: {progress_msgs}")
    assert w.dtype == np.float32 and w.shape == (6, 3)
    assert np.allclose(w, expected, atol=1e-5), "world_points weicht ab"
    assert progress_msgs and progress_msgs[-1][0] == 1.0

    # --- cancel -------------------------------------------------------------
    ev = threading.Event()
    ev.set()
    try:
        rec.world_points(cancel=ev)
        raise AssertionError("cancel wurde ignoriert")
    except RuntimeError as exc:
        assert str(exc) == "Abgebrochen"
        print("cancel: RuntimeError('Abgebrochen') OK")

    # --- interpolate_pose ----------------------------------------------------
    T = rec.interpolate_pose(0.5)
    R_expect = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    assert T is not None
    assert np.allclose(T[:3, 3], [0.5, 0, 0], atol=1e-9)
    assert np.allclose(T[:3, :3], R_expect, atol=1e-9), "SLERP-Mittelpunkt falsch"
    print(f"interpolate_pose(0.5): t={T[:3, 3]} yaw="
          f"{Rotation.from_matrix(T[:3, :3]).as_euler('zyx', degrees=True)[0]:.3f} deg (soll 45)")

    T = rec.interpolate_pose(1.5)
    yaw = Rotation.from_matrix(T[:3, :3]).as_euler("zyx", degrees=True)[0]
    assert np.allclose(T[:3, 3], [1.5, 0, 1.5], atol=1e-9)
    assert abs(yaw - 135.0) < 1e-6, f"yaw={yaw}"
    print(f"interpolate_pose(1.5): t={T[:3, 3]} yaw={yaw:.3f} deg (soll 135)")

    # exact stamps + edges
    assert np.allclose(rec.interpolate_pose(1.0)[:3, 3], [1, 0, 0], atol=1e-9)
    assert np.allclose(rec.interpolate_pose(-0.1), rec.interpolate_pose(0.0), atol=1e-12)
    assert np.allclose(rec.interpolate_pose(2.14)[:3, 3], [2, 0, 3], atol=1e-9)
    assert rec.interpolate_pose(-0.2) is None
    assert rec.interpolate_pose(2.16) is None
    print("edges: -0.1/2.14 gehalten, -0.2/2.16 -> None OK")

    pp = rec.path_positions()
    assert pp.shape == (3, 3) and np.allclose(pp, poses[:, :3])
    print(f"path_positions: {pp.tolist()}")

    # ---------- Kippkorrektur: Drehung, nicht Verzerrung ----------------
    for tilt_deg, axis in ((40.0, [0, 1, 0]), (7.0, [1, 0, 0]), (0.0, [0, 1, 0])):
        tilt = Rotation.from_rotvec(np.radians(tilt_deg) * np.asarray(axis, float))
        up_world = tilt.apply([0.0, 0.0, 1.0])  # Lotrechte im schraegen Weltsystem
        lvl = level_rotation(up_world)
        back = lvl.apply(up_world)
        assert np.allclose(back, [0, 0, 1], atol=1e-9), f"{tilt_deg}: {back}"
        # kuerzeste Drehung: der Drehwinkel ist genau die Schraeglage
        ang = np.degrees(np.linalg.norm(lvl.as_rotvec()))
        assert abs(ang - tilt_deg) < 1e-6, f"{tilt_deg}: Drehwinkel {ang}"
    print("level_rotation: 40/7/0 Grad auf die Lotrechte gedreht, Winkel exakt")

    lvl = level_rotation(Rotation.from_rotvec(np.radians(40.0) * np.array([0, 1.0, 0]))
                         .apply([0.0, 0.0, 1.0]))
    rot_poses = _rotate_world(poses, lvl)
    d_alt = np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1)
    d_neu = np.linalg.norm(np.diff(rot_poses[:, :3], axis=0), axis=1)
    assert np.allclose(d_alt, d_neu, atol=1e-12), "Drehung veraendert Abstaende"
    # relative Lage zwischen zwei Posen muss erhalten bleiben
    rel_alt = (Rotation.from_quat(poses[0, 3:7]).inv()
               * Rotation.from_quat(poses[2, 3:7]))
    rel_neu = (Rotation.from_quat(rot_poses[0, 3:7]).inv()
               * Rotation.from_quat(rot_poses[2, 3:7]))
    assert np.allclose(rel_alt.as_matrix(), rel_neu.as_matrix(), atol=1e-12)
    print(f"_rotate_world: Abstaende und Relativlagen erhalten "
          f"(Trajektorie {d_alt.sum():.3f} m)")

    # ohne "bag" in der meta.json wird nichts gemessen und nichts geaendert
    assert rec.gravity_level is None and Recording.load(REC_DIR).gravity_level is None
    print("ohne Bag-Pfad: keine Messung, Karte unveraendert")

    # lade_mit_hinweis: ohne Kippkorrektur keine Zeile, mit gespeicherter eine mit Praefix
    zeilen: list[str] = []
    assert lade_mit_hinweis(REC_DIR, None, zeilen.append).n_scans == 3 and zeilen == []
    with open(os.path.join(REC_DIR, "meta.json"), encoding="utf-8") as fh:
        meta_alt = json.load(fh)
    _write_json_atomic(os.path.join(REC_DIR, "meta.json"), dict(
        meta_alt, gravity_level={"quat": [0.0, 0.0, 0.0, 1.0], "tilt_deg": 3.2,
                                 "threshold_deg": LEVEL_MIN_TILT_DEG, "applied": False}))
    rec_h = lade_mit_hinweis(REC_DIR, None, zeilen.append, praefix="Zweiter Flug — ")
    assert zeilen == [f"Zweiter Flug — {rec_h.level_note()}"], zeilen
    assert np.array_equal(rec_h.poses, poses)
    _write_json_atomic(os.path.join(REC_DIR, "meta.json"), meta_alt)
    print(f"lade_mit_hinweis OK: {zeilen[0]}")

    with open(os.path.join(OUT, "selftest_metrics.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"n_scans=3 n_points=6 world_points_max_err={err:.3e}\n")
        fh.write("interpolate_pose(0.5) yaw=45deg OK; (1.5) yaw=135deg OK; edges OK\n")
    print("recording SELFTEST OK")
