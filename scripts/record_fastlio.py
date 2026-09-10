#!/usr/bin/env python3
"""Standalone FAST-LIO2 recorder (runs inside a sourced ROS 2 Humble environment).

Spawned as a subprocess by core/fastlio_runner.py. Subscribes
/cloud_registered_body (sensor_msgs/PointCloud2, x,y,z,intensity float32,
IMU body frame) and /Odometry (nav_msgs/Odometry, T_world_imu, camera_init).
Both carry the identical header stamp (lidar_end_time) per scan -> exact
(sec, nanosec) matching.

Output format (see ARCHITECTURE.md, recording/):
  points.bin     float32 N x 3 (concatenated, scan order)
  intensity.bin  float32 N
  offsets.npy    int64 (S+1,)
  stamps.npy     float64 (S,)
  poses.npy      float64 (S, 7)  x y z qx qy qz qw
  meta.json

All files are written into <out>.tmp (fresh-created; a pre-existing tmp dir is
deleted). The final <out> directory is NEVER touched here — a previous good
recording survives any failed/cancelled run. Promotion (os.replace of
<out>.tmp -> <out>) is done by core/fastlio_runner.py on success only.

stdout protocol (line-buffered):
  READY                       node + subscriptions up
  SCAN <n> <total_points>     per matched scan (cumulative point count)
  DONE <n_scans> <n_points>   after finalize (on SIGINT/SIGTERM/'STOP'/stdin EOF)

Warnings/errors go to stderr (stdout is protocol only).
"""
import argparse
import json
import os
import shutil
import signal
import sys
import threading
import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2

PF_FLOAT32 = 7          # sensor_msgs/PointField.FLOAT32
WARN_UNMATCHED = 50
MAX_UNMATCHED = 500


def eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def parse_cloud(msg: PointCloud2) -> tuple[np.ndarray, np.ndarray]:
    """PointCloud2 with x,y,z,intensity float32 -> (xyz (N,3) f32, intensity (N,) f32)."""
    if msg.is_bigendian:
        raise RuntimeError("PointCloud2 ist big-endian — nicht unterstützt.")
    fields = {f.name: f for f in msg.fields}
    offs: list[int] = []
    for name in ("x", "y", "z", "intensity"):
        f = fields.get(name)
        if f is None or f.datatype != PF_FLOAT32 or f.count != 1:
            raise RuntimeError(
                f"PointCloud2-Layout nicht unterstützt: Feld '{name}' fehlt "
                "oder ist nicht float32 (erwartet x,y,z,intensity als float32).")
        offs.append(f.offset)
    n = msg.width * msg.height
    step = msg.point_step
    if msg.height > 1 and msg.row_step != msg.width * step:
        raise RuntimeError("PointCloud2 mit Zeilen-Padding nicht unterstützt.")
    if n == 0:
        return np.empty((0, 3), np.float32), np.empty(0, np.float32)
    try:
        buf = np.frombuffer(msg.data, dtype=np.uint8, count=n * step)
    except (TypeError, ValueError):
        buf = np.asarray(msg.data, dtype=np.uint8)[: n * step]
    ox, oy, oz, oi = offs
    if step == 16 and (ox, oy, oz, oi) == (0, 4, 8, 12):
        arr = buf.view(np.float32).reshape(n, 4)          # fast path, standard layout
        return np.ascontiguousarray(arr[:, :3]), arr[:, 3].copy()
    rows = buf.reshape(n, step)
    xyz = np.empty((n, 3), np.float32)
    for j, off in enumerate((ox, oy, oz)):
        xyz[:, j] = rows[:, off:off + 4].copy().view(np.float32)[:, 0]
    inten = rows[:, oi:oi + 4].copy().view(np.float32)[:, 0]
    return xyz, inten


class RecorderNode(Node):
    def __init__(self, out_dir: str, meta_base: dict):
        super().__init__("rosbag_suite_recorder")
        self.out_dir = out_dir
        self.meta_base = meta_base
        self.f_pts = open(os.path.join(out_dir, "points.bin"), "wb")
        self.f_int = open(os.path.join(out_dir, "intensity.bin"), "wb")
        self.offsets: list[int] = [0]
        self.stamps: list[float] = []
        self.poses: list[list[float]] = []
        self.pending_clouds: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self.pending_odoms: dict[tuple[int, int], list[float]] = {}
        self.n_scans = 0
        self.n_points = 0
        self._warned_unmatched = False
        self._finalized = False
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.VOLATILE,
                         history=HistoryPolicy.KEEP_LAST, depth=200)
        self.create_subscription(PointCloud2, "/cloud_registered_body", self._on_cloud, qos)
        self.create_subscription(Odometry, "/Odometry", self._on_odom, qos)

    def _on_cloud(self, msg: PointCloud2) -> None:
        key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        parsed = parse_cloud(msg)
        pose = self.pending_odoms.pop(key, None)
        if pose is not None:
            self._emit(key, parsed, pose)
        else:
            self.pending_clouds[key] = parsed
            self._check_pending(self.pending_clouds, "Punktwolken")

    def _on_odom(self, msg: Odometry) -> None:
        key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        pose = [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
        cloud = self.pending_clouds.pop(key, None)
        if cloud is not None:
            self._emit(key, cloud, pose)
        else:
            self.pending_odoms[key] = pose
            self._check_pending(self.pending_odoms, "Odometrien")

    def _check_pending(self, d: dict, label: str) -> None:
        if len(d) > WARN_UNMATCHED and not self._warned_unmatched:
            self._warned_unmatched = True
            eprint(f"WARNUNG: mehr als {WARN_UNMATCHED} ungematchte {label} — "
                   "Stempel-Matching prüfen.")
        while len(d) > MAX_UNMATCHED:          # bound memory, drop oldest
            d.pop(next(iter(d)))

    def _emit(self, key: tuple[int, int], cloud: tuple[np.ndarray, np.ndarray],
              pose: list[float]) -> None:
        xyz, inten = cloud
        stamp = key[0] + key[1] * 1e-9
        if self.stamps and stamp <= self.stamps[-1]:
            eprint(f"WARNUNG: Scan mit nicht-monotonem Stempel übersprungen (t={stamp:.6f}).")
            return
        self.f_pts.write(xyz.tobytes())
        self.f_pts.flush()
        self.f_int.write(inten.astype(np.float32, copy=False).tobytes())
        self.f_int.flush()
        self.n_points += int(xyz.shape[0])
        self.n_scans += 1
        self.offsets.append(self.n_points)
        self.stamps.append(stamp)
        self.poses.append(pose)
        print(f"SCAN {self.n_scans} {self.n_points}", flush=True)

    def finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        self.f_pts.close()
        self.f_int.close()
        np.save(os.path.join(self.out_dir, "offsets.npy"), np.asarray(self.offsets, np.int64))
        np.save(os.path.join(self.out_dir, "stamps.npy"), np.asarray(self.stamps, np.float64))
        np.save(os.path.join(self.out_dir, "poses.npy"),
                np.asarray(self.poses, np.float64).reshape(self.n_scans, 7))
        meta = dict(self.meta_base, n_scans=self.n_scans, n_points=self.n_points,
                    created=time.strftime("%Y-%m-%dT%H:%M:%S"))
        with open(os.path.join(self.out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"DONE {self.n_scans} {self.n_points}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="FAST-LIO2 Aufzeichnung (rclpy)")
    ap.add_argument("--out", required=True, help="Ausgabeverzeichnis (recording/)")
    ap.add_argument("--expected", type=int, default=None, help="erwartete Scan-Anzahl")
    ap.add_argument("--bag", default="", help="Bag-Pfad (nur für meta.json)")
    ap.add_argument("--config", default="whs_dense.yaml", help="fast_lio-Config (meta.json)")
    ap.add_argument("--rate", type=float, default=1.0, help="Bag-Abspielrate (meta.json)")
    args = ap.parse_args()
    # NEVER touch the final --out directory (it may hold a good previous
    # recording). Write into <out>.tmp; the runner promotes it on success.
    final_dir = os.path.normpath(os.path.abspath(args.out))
    tmp_dir = final_dir + ".tmp"
    if os.path.isdir(tmp_dir) and not os.path.islink(tmp_dir):
        shutil.rmtree(tmp_dir)
    elif os.path.lexists(tmp_dir):
        os.remove(tmp_dir)
    os.makedirs(tmp_dir)
    eprint(f"Recorder schreibt nach {tmp_dir} (Promotion nach {final_dir} "
           "übernimmt der Runner bei Erfolg).")

    stop = threading.Event()
    rclpy.init()
    node = RecorderNode(tmp_dir, {"bag": args.bag, "config": args.config,
                                  "expected_scans": args.expected, "rate": args.rate})
    # Own handlers (after rclpy.init) -> deterministic stop + finalize.
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    def stdin_watch() -> None:
        try:
            for line in sys.stdin:
                if line.strip().upper() == "STOP":
                    break
        except Exception:
            pass
        stop.set()                              # 'STOP' line or stdin EOF

    threading.Thread(target=stdin_watch, daemon=True).start()

    print("READY", flush=True)
    rc = 0
    try:
        while not stop.is_set() and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception as exc:                    # fatal (e.g. unsupported cloud layout)
        eprint(f"FEHLER: {exc}")
        rc = 2
    finally:
        node.finalize()
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass
    sys.exit(rc)


if __name__ == "__main__":
    main()
