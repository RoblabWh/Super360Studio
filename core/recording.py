"""Datenformat der FAST-LIO-Aufzeichnung (recording/-Verzeichnis).

Format auf Platte:
    points.bin     float32, N x 3 konkateniert (Body/IMU-Frame, Scan-Reihenfolge)
    intensity.bin  float32, N
    offsets.npy    int64, (S+1,)   Scan i = points[offsets[i]:offsets[i+1]]
    stamps.npy     float64, (S,)   lidar_end_time je Scan
    poses.npy      float64, (S,7)  x y z qx qy qz qw  (T_world_imu)
    meta.json      {"bag": ..., "n_scans": S, "n_points": N, ...}

Qt-frei. points/intensity werden als np.memmap (read-only) geladen.
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

_EDGE_TOL_S = 0.15  # Randtoleranz fuer interpolate_pose
_PROGRESS_EVERY = 50  # Scans zwischen zwei progress_cb-Aufrufen


class Recording:
    """Geladene FAST-LIO-Aufzeichnung: Punkte im Body-Frame + Pose je Scan."""

    points: np.ndarray  # float32 (N,3), memmap read-only
    intensity: np.ndarray  # float32 (N,), memmap read-only
    offsets: np.ndarray  # int64 (S+1,)
    stamps: np.ndarray  # float64 (S,)
    poses: np.ndarray  # float64 (S,7): x y z qx qy qz qw
    meta: dict

    def __init__(
        self,
        points: np.ndarray,
        intensity: np.ndarray,
        offsets: np.ndarray,
        stamps: np.ndarray,
        poses: np.ndarray,
        meta: dict,
    ):
        self.points = points
        self.intensity = intensity
        self.offsets = offsets
        self.stamps = stamps
        self.poses = poses
        self.meta = meta

    @property
    def n_scans(self) -> int:
        return len(self.stamps)

    @property
    def n_points(self) -> int:
        return self.points.shape[0]

    @staticmethod
    def load(dir_path: str) -> "Recording":
        """Laedt eine Aufzeichnung; points/intensity als read-only np.memmap."""
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
        return Recording(
            points=points,
            intensity=intensity,
            offsets=offsets.astype(np.int64, copy=False),
            stamps=stamps.astype(np.float64, copy=False),
            poses=poses.astype(np.float64, copy=False),
            meta=meta,
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

    def path_positions(self) -> np.ndarray:
        """(S,3) Trajektorie (Positionen der Scan-Posen)."""
        return np.array(self.poses[:, 0:3], dtype=np.float64, copy=True)


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

    with open(os.path.join(OUT, "selftest_metrics.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"n_scans=3 n_points=6 world_points_max_err={err:.3e}\n")
        fh.write("interpolate_pose(0.5) yaw=45deg OK; (1.5) yaw=135deg OK; edges OK\n")
    print("recording SELFTEST OK")
