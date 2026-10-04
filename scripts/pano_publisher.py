#!/usr/bin/env python3
"""Stitcht das Dual-Fisheye des Bags live zum 360°-Pano und sendet es fuer RViz.

Abonniert /paycam/image_raw/compressed (JPEG 3040x1520, zwei Fisheyes
nebeneinander), stitcht mit core.stitcher.EquirectStitcher zum Equirect-Pano und
veroeffentlicht es als sensor_msgs/Image auf /pano/image (RViz: Image-Display).

Laeuft als eigener Prozess in einer gesourcten ROS-Umgebung (die GUI selbst hat
kein ROS, s. ARCHITECTURE.md). stdout ist Protokoll.

  python3 pano_publisher.py [--calib PFAD] [--width 1920] [--rate 8] [--scale 0.5]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Header

from core.kalibrierung import default_calib
from core.stitcher import EquirectStitcher

LUT_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "super360studio_lut")


class PanoPublisher(Node):
    def __init__(self, calib: str, width: int, rate: float, scale: float,
                 rotate_deg: float = 180.0):
        super().__init__("pano_publisher")
        self.get_logger().info("Baue Stitcher-LUT (einmalig) …")
        self.stitcher = EquirectStitcher(calib, width=width, lut_cache_dir=LUT_CACHE)
        self.period = 1.0 / max(rate, 0.1)
        self.scale = float(scale)
        # Die Pano-Mitte zeigt roh nach HINTEN (die Kamera sitzt um 180° gedreht
        # am Rig). Azimut == Bildspalte, also genuegt ein zyklisches Verschieben
        # um rotate_deg, damit die Mitte nach VORNE zeigt.
        self.roll_px = int(round((rotate_deg % 360.0) / 360.0 * width))
        self.rotate_deg = rotate_deg
        self.last_t = -1e9
        self.n = 0

        # BEST_EFFORT: bei Ueberlast lieber Frames verwerfen als Rueckstau bilden.
        qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)
        self.pub = self.create_publisher(Image, "/pano/image", qos)
        self.create_subscription(CompressedImage, "/paycam/image_raw/compressed",
                                 self.on_img, qos)
        self.get_logger().info(
            f"Bereit — /pano/image, {width}x{width // 2} px, max {rate:.0f} Hz, "
            f"Drehung {self.rotate_deg:.0f}° ({self.roll_px} px)")

    def on_img(self, m: CompressedImage) -> None:
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        # Drosseln: ungedrosselt waeren das 20 Hz x 5,5 MB = 110 MB/s
        # unkomprimierte Bilder — das saettigt DDS und laesst RViz einbrechen.
        if t - self.last_t < self.period:
            return
        self.last_t = t
        raw = cv2.imdecode(np.frombuffer(m.data, np.uint8), cv2.IMREAD_COLOR)
        if raw is None or raw.shape[:2] != (1520, 3040):
            return
        pano = self.stitcher.stitch(raw)          # BGR (width//2, width, 3)
        if self.roll_px:
            # Vor dem Skalieren rollen: in voller Aufloesung ist die
            # Verschiebung exakt. Vertikales Spiegeln unten aendert die
            # Spaltenzuordnung nicht, die Reihenfolge ist also unkritisch.
            pano = np.roll(pano, self.roll_px, axis=1)
        # Pano-Zeile 0 ist der Boden (-Y_cam0, s. core.stitcher._pano_rays):
        # fuer die Anzeige NUR vertikal spiegeln (flipCode=0). flipCode=-1 wuerde
        # zusaetzlich horizontal spiegeln und die Szene seitenverkehrt zeigen.
        pano = cv2.flip(pano, 0)
        if self.scale != 1.0:
            pano = cv2.resize(pano, None, fx=self.scale, fy=self.scale,
                              interpolation=cv2.INTER_AREA)
        img = Image()
        img.header = Header(stamp=m.header.stamp, frame_id="world")
        img.height, img.width = pano.shape[:2]
        img.encoding = "bgr8"
        img.is_bigendian = 0
        img.step = 3 * img.width
        img.data = pano.tobytes()
        self.pub.publish(img)
        self.n += 1
        if self.n % 50 == 0:
            self.get_logger().info(f"{self.n} Panos gesendet")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default=None)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--rate", type=float, default=8.0)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--rotate-deg", type=float, default=180.0,
                    help="Pano horizontal drehen; 180 = Mitte zeigt nach vorn "
                         "statt nach hinten")
    a = ap.parse_args()
    rclpy.init()
    node = PanoPublisher(a.calib or default_calib(), a.width, a.rate, a.scale,
                         a.rotate_deg)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
