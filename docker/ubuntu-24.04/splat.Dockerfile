# Gaussian Splat unter Ubuntu 24.04: der eigene Interpreter mit torch und gsplat
# (s. core/splat.py, find_splat_python) auf der App-Basis.
#
#   docker build -f docker/ubuntu-24.04/splat.Dockerfile -t super360-u2404-splat:latest \
#       docker/ubuntu-24.04
#
# Zwei Stufen: gsplat wird im CUDA-devel-Image gebaut (nvcc ab 12.8 kann
# sm_120), ins fertige Image kommt nur die venv — torch bringt seine
# CUDA-Bibliotheken als pip-Pakete mit, zum Rechnen braucht es kein Toolkit.
# gsplat muss fuer jede Kartenarchitektur gebaut sein, die die venv benutzt,
# sonst bricht das Training mit "no kernel image is available" ab. Vorgabe:
# 8.6 (RTX 3060 Ti), 8.9 (RTX 4090 Laptop), 12.0 (RTX 50xx).
#
# Nur die Bau-Stufe (ohne die App-Basis):
#   docker build -f docker/ubuntu-24.04/splat.Dockerfile --target gsplat-bau ...

ARG BASIS=super360-u2404-basis:latest
ARG CUDA_BILD=nvidia/cuda:12.8.1-devel-ubuntu24.04

# ------------------------------------------------------------ gsplat bauen
FROM ${CUDA_BILD} AS gsplat-bau

ARG TORCH_CUDA_ARCH_LIST="8.6;8.9;12.0"
ARG MAX_JOBS=8
ARG SPLAT_VENV=/opt/venvs/splat

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv python3-dev \
        build-essential git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Dieselben Versionen wie die venv unter 22.04 (torch 2.8.0+cu128, gsplat 1.5.3).
# Python 3.12 legt in einer venv kein setuptools mehr an; der Bau ohne
# Isolation braucht es aber, und gsplat seine Abhaengigkeiten (ninja, jaxtyping,
# rich, packaging).
RUN python3 -m venv ${SPLAT_VENV} \
 && ${SPLAT_VENV}/bin/pip install --no-cache-dir --upgrade pip setuptools wheel \
 && ${SPLAT_VENV}/bin/pip install --no-cache-dir torch==2.8.0 \
        --index-url https://download.pytorch.org/whl/cu128 \
 && ${SPLAT_VENV}/bin/pip install --no-cache-dir numpy==2.2.6 pillow==12.3.0 \
        ninja jaxtyping rich packaging

# Aus den Quellen, nicht das Wheel von PyPI: das baut seine Kernel erst beim
# ersten Aufruf, ohne nvcc also gar nicht.
RUN TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" MAX_JOBS=${MAX_JOBS} \
        CUDA_HOME=/usr/local/cuda \
        ${SPLAT_VENV}/bin/pip install --no-cache-dir --no-build-isolation --no-deps \
        --no-binary gsplat gsplat==1.5.3

# Jede verlangte Architektur muss als ELF in der Bibliothek stehen.
RUN so=$(ls ${SPLAT_VENV}/lib/python3*/site-packages/gsplat/csrc*.so) \
 && cuobjdump --list-elf "$so" | tee /tmp/elf.txt \
 && for a in $(echo "${TORCH_CUDA_ARCH_LIST}" | tr ';' ' '); do \
        grep -q "sm_$(echo "$a" | tr -d '.')\." /tmp/elf.txt \
            || { echo "gsplat ohne sm_$a gebaut" >&2; exit 1; }; \
    done

# ------------------------------------------------------------ App + venv
FROM ${BASIS}

ARG SPLAT_VENV=/opt/venvs/splat

# Die venv zeigt auf /usr/bin/python3 — in beiden Stufen das Python 3.12 von
# Ubuntu 24.04, darum laesst sie sich so kopieren.
COPY --from=gsplat-bau ${SPLAT_VENV} ${SPLAT_VENV}
# cuobjdump, um die Architekturen im fertigen Image nachzusehen
# (docker/ubuntu-24.04/pruefe_splat.sh).
COPY --from=gsplat-bau /usr/local/cuda/bin/cuobjdump /usr/local/bin/cuobjdump

# Erster Kandidat in core/splat.py; dazu ~/.venvs/splat im HOME der Basis,
# falls jemand die Variable ueberschreibt.
ENV SUPER360_SPLAT_PYTHON=${SPLAT_VENV}/bin/python
RUN mkdir -p "$HOME/.venvs" && ln -s ${SPLAT_VENV} "$HOME/.venvs/splat"
