#!/usr/bin/env python3
"""Prueft, ob alles fuer Super360 Studio da ist, und sagt, was fehlt.

    python3 scripts/pruefe_installation.py

Getrennt nach PFLICHT (ohne startet die App nicht oder kann nichts laden) und
den Teilen, die nur einzelne Schritte brauchen. Aendert nichts.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys

HOME = os.path.expanduser("~")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ergebnis = {"pflicht": 0, "optional": 0}


def zeile(ok: bool, was: str, hinweis: str = "", pflicht: bool = True) -> None:
    zeichen = "OK  " if ok else ("FEHLT" if pflicht else "fehlt")
    print(f"  [{zeichen}] {was}" + (f"  —  {hinweis}" if (hinweis and not ok) else ""))
    if not ok:
        ergebnis["pflicht" if pflicht else "optional"] += 1


def modul(name: str, paket: str, pflicht: bool = True, wofuer: str = "") -> None:
    try:
        m = importlib.import_module(name)
        v = getattr(m, "__version__", "")
        if not v:
            try:
                from importlib.metadata import version
                v = version(paket.split()[-1] if paket.startswith("pip:") else name)
            except Exception:  # noqa: BLE001
                v = ""
        if name == "vtk":
            v = m.vtkVersion.GetVTKVersion()
        if name == "PyQt5":
            from PyQt5.QtCore import PYQT_VERSION_STR
            v = PYQT_VERSION_STR
        zeile(True, f"{name} {v}".strip())
    except Exception as exc:  # noqa: BLE001
        zeile(False, f"{name}{' (' + wofuer + ')' if wofuer else ''}",
              f"{paket}  [{type(exc).__name__}]", pflicht)


print(f"Python {sys.version.split()[0]} ({sys.executable})")
if sys.version_info[:2] != (3, 10):
    print("  Hinweis: getestet mit Python 3.10 (Ubuntu 22.04, ROS 2 Humble).")

print("\nPflicht — ohne startet die App nicht:")
modul("PyQt5", "sudo apt install python3-pyqt5 python3-pyqt5.qtopengl")
modul("vtk", "sudo apt install python3-vtk9")
for name, paket in (("numpy", "pip"), ("scipy", "pip"), ("cv2", "pip: opencv-python"),
                    ("open3d", "pip"), ("rosbags", "pip"), ("PIL", "pip: Pillow"),
                    ("laspy", "pip"), ("pyproj", "pip")):
    modul(name, f"pip install --user -r requirements.txt ({paket})")
try:
    from rosbags.typesys import Stores, get_typestore  # noqa: F401
    zeile(True, "rosbags mit get_typestore/Stores (ab 0.10)")
except Exception:  # noqa: BLE001
    zeile(False, "rosbags zu alt", "pip install --user 'rosbags>=0.10,<0.11'")

print("\nOptional:")
modul("qdarktheme", "pip install --user pyqtdarktheme", False, "dunkles Theme")

print("\nMäander-Einfärbung (DJI-Kartierungsflug):")
pcm = [os.path.join(HOME, "PointCloudMerger"), os.path.join(ROOT, "..", "PointCloudMerger")]
da = next((p for p in pcm if os.path.isdir(os.path.join(p, "colorize_pipeline"))), None)
zeile(bool(da), f"PointCloudMerger/colorize_pipeline{' in ' + os.path.abspath(da) if da else ''}",
      "git clone git@github.com:LenaKremer98/PointCloudMerger.git ~/PointCloudMerger")
zeile(bool(shutil.which("exiftool")), "exiftool (Geotags schneller; ohne liest Pillow)",
      "sudo apt install libimage-exiftool-perl", False)
colmap = None
if da:
    sys.path.insert(0, os.path.abspath(da))
    try:
        from colorize_pipeline import sfm
        colmap = sfm.find_python()
    except Exception:  # noqa: BLE001
        colmap = None
zeile(bool(colmap), f"Interpreter mit pycolmap{': ' + colmap if colmap else ''} "
      "(nur für eine neue COLMAP-Rekonstruktion)",
      "python3 -m venv ~/.venvs/colmap && ~/.venvs/colmap/bin/pip install pycolmap", False)

print("\nGaussian Splat (Einfärbung über alle Bilder zugleich, braucht eine NVIDIA-GPU):")
sys.path.insert(0, ROOT)
try:
    from core import splat as splat_mod
    py, info = splat_mod.find_splat_python()
except Exception as exc:  # noqa: BLE001
    py, info = None, {"fehler": f"{type(exc).__name__}: {exc}"}
zeile(bool(py), f"Interpreter mit torch und gsplat{': ' + py if py else ''}",
      "python3 -m venv ~/.venvs/splat && ~/.venvs/splat/bin/pip install torch gsplat "
      "(oder SUPER360_SPLAT_PYTHON setzen)", False)
if py:
    gpu = f" {info['gpu']} ({info['vram_gb']} GB)" if info.get("cuda") else ""
    zeile(bool(info.get("cuda")), f"CUDA-GPU{gpu}", splat_mod.hinweis(info) or "", False)

print("\nKarte berechnen (FAST-LIO2, braucht ROS 2 Humble):")
for pfad, hinweis in (
        ("/opt/ros/humble/setup.bash", "ROS 2 Humble: sudo apt install ros-humble-ros-base"),
        (f"{HOME}/ws_livox/install/setup.bash",
         "livox_ros_driver2 in ~/ws_livox bauen (s. README)"),
        (f"{HOME}/fastlio2_ws/install/setup.bash",
         "FAST_LIO_ROS2 in ~/fastlio2_ws bauen (s. README)"),
        (f"{HOME}/fastlio2_ws/install/fast_lio/share/fast_lio/config/whs_dense.yaml",
         "cp config/fastlio/whs_dense.yaml ~/fastlio2_ws/src/FAST_LIO_ROS2/config/ "
         "und neu bauen")):
    zeile(os.path.exists(pfad), pfad.replace(HOME, "~"), hinweis, False)
if os.path.exists("/opt/ros/humble/setup.bash"):
    r = subprocess.run(["bash", "-c", "source /opt/ros/humble/setup.bash && command -v ros2"],
                       capture_output=True, text=True)
    zeile(r.returncode == 0, "ros2 im ROS-Environment", "ROS-Installation prüfen", False)
zeile(os.path.exists(f"{HOME}/EPIC_ros2/install/setup.bash"),
      "~/EPIC_ros2 (nur RViz-Wiedergabe: traj_utils/quadrotor_msgs)", "", False)

print("\nTests (optional):")
zeile(bool(shutil.which("xvfb-run")), "xvfb-run (GUI-Tests ohne Bildschirm)",
      "sudo apt install xvfb", False)

print()
if ergebnis["pflicht"]:
    print(f"{ergebnis['pflicht']} Pflichtteil(e) fehlen — erst installieren, dann ./run_gui.sh")
    sys.exit(1)
print("Pflicht erfüllt — ./run_gui.sh startet die App."
      + (f" {ergebnis['optional']} optionale(r) Teil(e) fehlen, s. oben."
         if ergebnis["optional"] else ""))
