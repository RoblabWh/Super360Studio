# Basis für Super360 Studio auf Ubuntu 24.04: System-Python 3.12, PyQt5 und
# VTK aus apt, die Pakete aus requirements.txt in einer venv, xvfb für die GUI
# und OpenCL auf der NVIDIA-GPU. Ohne ROS, COLMAP und Splat; die bauen darauf
# auf (FROM super360-u2404-basis:latest).
#
# Bauen (Kontext ist die Repo-Wurzel, gebraucht wird nur requirements.txt):
#
#   docker build -f docker/ubuntu-24.04/basis.Dockerfile -t super360-u2404-basis:latest .
#
# Das Repo steckt nicht im Image, es kommt als Volume nach /super360:
#
#   docker run --rm --gpus all --user "$(id -u):$(id -g)" \
#       -v "$PWD":/super360:ro super360-u2404-basis:latest python3 scripts/pruefe_installation.py
#
# Warum eine venv mit --system-site-packages: Ubuntu 24.04 sperrt
# 'pip install --user' in das System-Python (PEP 668, externally-managed).
# '--break-system-packages' würde apt-Dateien überschreiben können, pipx ist für
# Programme statt Bibliotheken. Die venv sieht PyQt5 und VTK aus apt, ihre
# eigenen Pakete liegen davor im Suchpfad; damit gewinnt das numpy aus pip
# gegen python3-numpy aus apt, das python3-vtk9 mitzieht (VTK bindet numpy nur
# über Python, nicht über die C-ABI). Ihr bin/ steht vorn in PATH, 'python3'
# ist also die venv.

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# python3-pyqt5 5.15.10, python3-vtk9 9.1.0 (noble). xauth braucht xvfb-run,
# libgl1-mesa-dri liefert llvmpipe (OpenGL 4.5) für VTK unter xvfb.
# ocl-icd-libopencl1 und clinfo für OpenCL, exiftool für die Mäander-Geotags.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip \
        python3-pyqt5 python3-pyqt5.qtopengl python3-vtk9 \
        xvfb xauth libgl1-mesa-dri libegl1 libglx-mesa0 fonts-dejavu-core \
        ocl-icd-libopencl1 clinfo \
        libimage-exiftool-perl git ca-certificates procps \
    && rm -rf /var/lib/apt/lists/*

# OpenCL: Die nvidia-Runtime bindet libnvidia-opencl.so.1 ein (Fähigkeit
# 'compute'), die ICD-Datei dazu muss das Image selbst mitbringen.
RUN mkdir -p /etc/OpenCL/vendors \
    && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENV SUPER360_VENV=/opt/super360/venv
ENV PATH=$SUPER360_VENV/bin:$PATH

# opencv-python bringt eigene Qt-Plugins mit und setzt beim Import
# QT_QPA_PLATFORM_PLUGIN_PATH; wird cv2 vor der QApplication geladen (Skripte,
# Selbsttests), lädt das Qt aus apt dann das falsche xcb-Plugin. Die App braucht
# kein highgui, im Image steht deshalb die headless-Variante.
# pyqtdarktheme 2.1.0 verlangt Python <3.12, ist aber reines Python ohne
# entfernte Module; es kommt mit --ignore-requires-python dazu (s. requirements.txt).
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m venv --system-site-packages "$SUPER360_VENV" \
    && sed -e 's/^opencv-python\b/opencv-python-headless/' /tmp/requirements.txt > /tmp/req.txt \
    && pip install --no-cache-dir -r /tmp/req.txt \
    && pip install --no-cache-dir --ignore-requires-python "pyqtdarktheme==2.1.0" \
    && pip install --no-cache-dir pyflakes \
    && rm /tmp/requirements.txt /tmp/req.txt

# Was die pip-Pakete zur Laufzeit aus dem System brauchen: open3d lädt
# libgomp (bis 0.19) bzw. libusb-1.0 (ab 0.20), qdarktheme importiert
# PyQt5.QtSvg (fehlt es, startet die App ohne dunkles Theme).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libusb-1.0-0 python3-pyqt5.qtsvg \
    && rm -rf /var/lib/apt/lists/*

# Läuft der Container mit --user (damit Ausgaben auf dem Host nicht root
# gehören), braucht der Nutzer trotzdem ein beschreibbares HOME (Qt, open3d,
# matplotlib). XDG_RUNTIME_DIR bleibt ungesetzt: Qt legt sich dann
# /tmp/runtime-<nutzer> mit den verlangten Rechten selbst an.
RUN mkdir -p /home/super360 && chmod 1777 /home/super360
ENV HOME=/home/super360

WORKDIR /super360
CMD ["python3", "app.py"]
