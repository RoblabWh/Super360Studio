# Mäander-Einfärbung unter Ubuntu 24.04: der Interpreter mit pycolmap für die
# COLMAP-Rekonstruktion (s. core/meander.py, find_colmap_python) auf der
# App-Basis.
#
#   docker build -f docker/ubuntu-24.04/colmap.Dockerfile -t super360-u2404-colmap:latest \
#       docker/ubuntu-24.04
#
# Die Rekonstruktion laeuft als Unterprozess in einem eigenen Interpreter
# (colorize_pipeline/_colmap_worker.py im Nachbarrepo PointCloudMerger), der
# nur pycolmap, numpy und Pillow braucht. Gefunden wird er ueber
# colorize_pipeline/sfm.py:find_python, und zwar unter ~/.venvs/colmap — HOME
# ist in der Basis /home/super360, dort liegt ein Verweis auf die venv.
#
# Das Nachbarrepo steckt nicht im Image. Es kommt lesend nach ~/PointCloudMerger,
# den ersten Ort in PIPELINE_CANDIDATES (core/meander.py), wie in pruefe_app.sh:
#
#   docker run --rm --user "$(id -u):$(id -g)" \
#       -v "$PWD":/super360:ro -v <PointCloudMerger>:/home/super360/PointCloudMerger:ro \
#       super360-u2404-colmap:latest python3 -m core.meander
#
# exiftool fuer die Geotags bringt schon die Basis mit.
#
# Nur die venv (ohne die App-Basis), zum Ausprobieren:
#   docker build -f docker/ubuntu-24.04/colmap.Dockerfile --target colmap-bau ...

ARG BASIS=super360-u2404-basis:latest

# ------------------------------------------------------------ venv bauen
FROM ubuntu:24.04 AS colmap-bau

ARG COLMAP_VENV=/opt/venvs/colmap

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Dieselben Versionen wie die venv unter 22.04 (pycolmap 4.0.4, numpy 1.26.4,
# Pillow 12.2.0); alle drei gibt es als fertiges Wheel fuer Python 3.12.
# pycolmap von PyPI rechnet auf der CPU, wie unter 22.04.
RUN python3 -m venv ${COLMAP_VENV} \
 && ${COLMAP_VENV}/bin/pip install --no-cache-dir \
        pycolmap==4.0.4 numpy==1.26.4 pillow==12.2.0 \
 && ${COLMAP_VENV}/bin/python -c "import pycolmap; print('pycolmap', pycolmap.__version__)"

# ------------------------------------------------------------ App + venv
FROM ${BASIS}

ARG COLMAP_VENV=/opt/venvs/colmap

# Die venv zeigt auf /usr/bin/python3 — in beiden Stufen das Python 3.12 von
# Ubuntu 24.04, darum laesst sie sich so kopieren.
COPY --from=colmap-bau ${COLMAP_VENV} ${COLMAP_VENV}
# Der Ort, den sfm.py erwartet. Der Verweis haengt nicht am Benutzer: HOME ist
# fuer jede --user-ID dasselbe Verzeichnis.
RUN mkdir -p "$HOME/.venvs" \
 && ln -s ${COLMAP_VENV} "$HOME/.venvs/colmap" \
 && "$HOME/.venvs/colmap/bin/python" -c "import pycolmap"
