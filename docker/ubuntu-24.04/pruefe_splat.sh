#!/usr/bin/env bash
# Gaussian Splat im Image super360-u2404-splat pruefen (Ubuntu 24.04).
#
#   docker/ubuntu-24.04/pruefe_splat.sh [<bag-ordner> <cache-projekt>]
#
# 1. ohne GPU: Architekturen in gsplat, Selbsttest von core/splat.py, der
#    Selbsttest des Trainers (CPU) und die Auskunft ohne Karte
# 2. mit GPU:  find_splat_python mit echtem gsplat-Kernel, dieselbe
#    synthetische Szene mit gsplat statt des dichten CPU-Renderers
# 3. mit Bag und Projekt: ein kurzer Onboard-Splat ueber das Hauptfenster
#    (Bag oeffnen, Karte laden, "Onboard per Gaussian Splat") auf einer Kopie
#    des Projekts im Container — Laufzeit, L1 an den Pruefbildern, Farbebene.
#
# Bag und Projekt werden nur lesend eingebunden, die Kopie lebt und stirbt im
# Container. Einstellbar: IMAGE, MERGER (PointCloudMerger fuer den Selbsttest
# von core/splat.py, nur lesend), SCHRITTE (Vorgabe 500), FRAMES (60),
# ANKER_MIO (1.0), ZEITGRENZE in s (3600), OMP_NUM_THREADS (8: torch nimmt
# sonst alle Kerne und tritt sich auf einem belegten Rechner selbst auf die Fuesse).
set -euo pipefail

IMAGE=${IMAGE:-super360-u2404-splat:latest}
REPO_IN=/super360
SKRIPT_IN=$REPO_IN/docker/ubuntu-24.04/pruefe_splat.sh

# ------------------------------------------------------------ im Container
innen_cpu() {
    local py=$SUPER360_SPLAT_PYTHON
    echo "== gsplat: Architekturen"
    cuobjdump --list-elf "$(ls "$(dirname "$py")"/../lib/python3*/site-packages/gsplat/csrc*.so)"
    echo "== core.splat Selbsttest (App-Python $(python3 -V 2>&1))"
    if [ -d "$HOME/PointCloudMerger" ]; then
        python3 -m core.splat
    else
        echo "uebersprungen: braucht PointCloudMerger (MERGER=<pfad>)"
    fi
    echo "== splat_train --selbsttest (CPU, $("$py" -V 2>&1))"
    "$py" scripts/splat_train.py --selbsttest
    echo "== find_splat_python ohne GPU"
    python3 - <<'PY'
from core import splat
py, info = splat.find_splat_python()
print(py, info)
print("Hinweis:", splat.hinweis(info))
assert py and info.get("torch") and info.get("gsplat") and not info.get("cuda"), info
PY
}

innen_gpu() {
    echo "== find_splat_python mit GPU"
    python3 - <<'PY'
from core import splat
py, info = splat.find_splat_python()
print(py, info)
assert info.get("cuda") and info.get("gsplat_rechnet") is True, info
assert splat.hinweis(info) is None, splat.hinweis(info)
print("bereit:", info["gpu"], "sm_" + info["faehigkeit"].replace(".", ""))
PY
    echo "== Selbsttest-Szene mit gsplat auf der GPU"
    "$SUPER360_SPLAT_PYTHON" - <<'PY'
import sys
sys.path.insert(0, "scripts")
import splat_train as s
echt = s.trainieren
s.trainieren = lambda ds, aus, cpu=False, **k: echt(ds, aus, cpu=False, **k)
s.selbsttest()
PY
}

innen_app() {
    local bag=$1 quelle=$2
    export SUPER360_CACHE_ROOT=/tmp/cache
    local ziel
    ziel=$(python3 -c "from core.project import Project; print(Project('$bag').dir)")
    # Der Ordnername haengt am Pfad des Bags; der ist im Container ein anderer.
    cp -a "$quelle"/. "$ziel"/
    echo "== Projektkopie: $ziel"
    cat > /tmp/treiber.py <<'PY'
import json, os, sys, time
from PyQt5.QtWidgets import QApplication, QMessageBox

sys.path.insert(0, os.getcwd())     # die Repo-Wurzel, nicht /tmp
bag = sys.argv[1]
schritte, frames, anker = int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
grenze = time.monotonic() + float(sys.argv[5])
app = QApplication(sys.argv[:1])
for name in ("information", "warning", "critical"):
    setattr(QMessageBox, name, staticmethod(
        lambda *a, n=name: print(f"[Dialog {n}] {a[1:3]}", flush=True) or QMessageBox.Ok))
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)

from core import splat as splat_mod
from ui.main_window import MainWindow

# Was der Trainer an den Pruefbildern misst, steht nur in seinem Rueckgabewert.
laeufe = []
echt = splat_mod.trainieren
def mitschreiben(*a, **k):
    erg = echt(*a, **k)
    laeufe.append(erg)
    return erg
splat_mod.trainieren = mitschreiben

f = MainWindow()
f._autotest_path = bag            # Fehler melden statt Dialog
log = f._log
f._log = lambda text, *a, **k: (print("[Log]", text, flush=True), log(text, *a, **k))
f.show()

def warte(bedingung, was):
    while not bedingung():
        app.processEvents()
        time.sleep(0.05)
        if f._autotest_failed:
            sys.exit(f"FEHLER bei {was} (s. Protokoll)")
        if time.monotonic() > grenze:
            sys.exit(f"Zeitgrenze bei {was}")

f._open_bag(bag)
warte(lambda: not f._busy and f._worker is None and f._world is not None,
      "Bag und Karte laden")
print(f"Karte: {len(f._world)} Punkte, {f._rec.n_scans} Scans", flush=True)

# Die Oberflaeche beginnt bei 1000 Schritten; fuer die Probe reichen weniger.
f._spin_splat_schritte_onboard.setMinimum(1)
f._spin_splat_schritte_onboard.setValue(schritte)
f._spin_splat_frames.setValue(frames)
f._spin_splat_anker.setValue(anker)
t0 = time.monotonic()
f._on_splat_onboard()
warte(lambda: not f._busy and f._worker is None, "Onboard-Splat")
dauer = time.monotonic() - t0

ebene = f._project.layer_dir("onboard_splat")
with open(os.path.join(ebene, "meta.json"), encoding="utf-8") as fh:
    meta = json.load(fh)
pruef = [e for erg in laeufe for e in erg["pruefung"]]
mittel = lambda k: sum(e[k] for e in pruef) / max(len(pruef), 1)
print(f"Splat fertig in {dauer:.0f} s, {len(laeufe)} Trainingslauf/-läufe", flush=True)
print(f"Prüfbilder {len(pruef)}: L1 roh {mittel('l1_roh'):.4f}, angepasst "
      f"{mittel('l1_angepasst'):.4f}, PSNR {mittel('psnr_angepasst'):.1f} dB", flush=True)
print(f"Ebene {ebene}: {sorted(os.listdir(ebene))}, gefärbt {meta['anteil'] * 100:.1f} %, "
      f"Anker {meta['splat']['anker']}, Raster {meta['splat']['raster_m'] * 100:.1f} cm",
      flush=True)
print("Gegenprobe:", json.dumps(meta.get("gegenprobe"), ensure_ascii=False), flush=True)
bericht = laeufe[-1]["bericht"]
print(f"GPU-Speicher höchstens {bericht.get('gpu_speicher_gb', 0):.1f} GB", flush=True)
assert "onboard_splat" in f._layers, sorted(f._layers)
assert meta["anteil"] > 0 and meta["quelle"] == "onboard_splat", meta
print("APP-SPLAT OK", flush=True)
os._exit(0)
PY
    xvfb-run -a python3 /tmp/treiber.py "$bag" "$SCHRITTE" "$FRAMES" "$ANKER_MIO" "$ZEITGRENZE"
}

if [ "${1:-}" = "--innen" ]; then
    shift
    cd "$REPO_IN"
    teil=$1
    shift
    "innen_$teil" "$@"
    exit
fi

# ------------------------------------------------------------ auf dem Host
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
merger=()
if [ -n "${MERGER:-}" ]; then
    merger=(-v "$(cd "$MERGER" && pwd):/home/super360/PointCloudMerger:ro")
fi
lauf() {
    docker run "${merger[@]}" --rm --user "$(id -u):$(id -g)" -v "$REPO:$REPO_IN:ro" -w "$REPO_IN" \
        -e SCHRITTE="${SCHRITTE:-500}" -e FRAMES="${FRAMES:-60}" \
        -e ANKER_MIO="${ANKER_MIO:-1.0}" -e ZEITGRENZE="${ZEITGRENZE:-3600}" \
        -e OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" \
        "$@"
}

echo "##### 1. ohne GPU"
lauf "$IMAGE" bash "$SKRIPT_IN" --innen cpu
echo "##### 2. mit GPU"
lauf --gpus all "$IMAGE" bash "$SKRIPT_IN" --innen gpu
if [ $# -ge 2 ]; then
    bag=$(cd "$1" && pwd)
    projekt=$(cd "$2" && pwd)
    echo "##### 3. Onboard-Splat ueber das Hauptfenster"
    lauf --gpus all -v "$bag:/daten/$(basename "$bag"):ro" -v "$projekt:/daten/projekt:ro" \
        "$IMAGE" bash "$SKRIPT_IN" --innen app "/daten/$(basename "$bag")" /daten/projekt
fi
echo "SPLAT-PRUEFUNG OK"
