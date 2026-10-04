#!/usr/bin/env bash
# Funktionstest „Karte berechnen“ und RViz-Wiedergabe im ROS-Image (Jazzy).
#
#   docker/ubuntu-24.04/pruefe_karte.sh <bag-verzeichnis> <ausgabe> [<referenz-recording>]
#
# Läuft auf dem Host und startet das Image $IMAGE (Vorgabe
# super360-u2404-ros:latest) mit GPU. Das Bag und die Referenz werden nur
# lesend eingebunden; im Container wird das Bag kopiert, der Cache liegt unter
# <ausgabe>/cache (SUPER360_CACHE_ROOT), nie im echten Cache. Schritte, je über
# den Weg der App:
#   1. FAST-LIO: core.project.Project + core.fastlio_runner.FastLioRunner.run,
#      wie der Knopf „Karte berechnen“ (whs_dense.yaml, Rate 1).
#   2. Vergleich mit <referenz-recording> (meta.json, poses.npy): Scans,
#      Punkte, Bahnlänge.
#   3. RViz-Wiedergabe über core.rviz_player unter xvfb: starten, Abspielen
#      an /livox/imu nachweisen, beenden.
#   4. Projekt in der App laden (Autotest-Haken SUPER360_AUTOTEST).
# Exit-Code 0 nur, wenn alle Schritte bestehen.
set -euo pipefail

IMAGE=${IMAGE:-super360-u2404-ros:latest}
HIER=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HIER/../.." && pwd)

if [[ "${1:-}" != "--innen" ]]; then
    if [[ $# -lt 2 ]]; then
        sed -n '2,17p' "$0"
        exit 2
    fi
    BAG=$(realpath "$1")
    AUSGABE=$(realpath -m "$2")
    REFERENZ=${3:+$(realpath "$3")}
    mkdir -p "$AUSGABE"
    # HOME absichtlich nicht das des Images: die App muss die Workspaces
    # über SUPER360_ROS_WS finden, egal wohin HOME zeigt.
    args=(--rm --gpus all --user "$(id -u):$(id -g)"
          -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-73}" -e HOME=/arbeit/heim
          -v "$REPO":/super360:ro
          -v "$BAG":/daten/bag/"$(basename "$BAG")":ro
          -v "$AUSGABE":/arbeit)
    if [[ -n "$REFERENZ" ]]; then
        args+=(-v "$REFERENZ":/daten/referenz:ro)
    fi
    exec docker run "${args[@]}" "$IMAGE" \
        bash /super360/docker/ubuntu-24.04/pruefe_karte.sh --innen
fi

# ------------------------------------------------------------ im Container
cd /super360
export SUPER360_CACHE_ROOT=/arbeit/cache
export ROS_LOG_DIR=/arbeit/roslog
export PYTHONPATH=/super360${PYTHONPATH:+:$PYTHONPATH}
mkdir -p /arbeit/bags "$SUPER360_CACHE_ROOT" "$ROS_LOG_DIR" "$HOME"

QUELLE=$(ls -d /daten/bag/*)
BAG=/arbeit/bags/$(basename "$QUELLE")
if [[ ! -d "$BAG" ]]; then
    echo "== Kopiere Bag nach $BAG"
    cp -r "$QUELLE" "$BAG"
fi
fehler=0

echo "== 1. Karte berechnen (FAST-LIO2)"
python3 - "$BAG" <<'PY' | tee /arbeit/karte.txt || fehler=1
import json, os, sys, time
import numpy as np
from core import ros_umgebung
from core.fastlio_runner import FastLioRunner
from core.project import Project

bag = sys.argv[1]
print(f"ROS: {ros_umgebung.distro()} ({ros_umgebung.setup_bash()})", flush=True)
projekt = Project(bag)
log = open("/arbeit/fastlio.log", "w", encoding="utf-8")
zuletzt = [-1.0]

def fortschritt(frac, msg):
    if frac - zuletzt[0] >= 0.1 or frac in (0.0, 1.0):
        print(f"  [{frac * 100:5.1f} %] {msg}", flush=True)
        zuletzt[0] = frac

def protokoll(zeile):
    log.write(zeile + "\n")
    log.flush()

t0 = time.monotonic()
res = FastLioRunner().run(bag, projekt.recording_dir(), config="whs_dense.yaml",
                          rate=1.0, progress_cb=fortschritt, log_cb=protokoll)
posen = np.load(os.path.join(res.recording_dir, "poses.npy"))
bahn = float(np.linalg.norm(np.diff(posen[:, :3], axis=0), axis=1).sum())
ergebnis = dict(n_scans=res.n_scans, n_points=res.n_points,
                expected_scans=res.expected_scans, bahn_m=round(bahn, 3),
                dauer_s=round(time.monotonic() - t0, 1),
                recording_dir=res.recording_dir)
json.dump(ergebnis, open("/arbeit/karte.json", "w"), indent=2)
print("KARTE:", json.dumps(ergebnis), flush=True)
PY

if [[ -d /daten/referenz && -f /arbeit/karte.json ]]; then
    echo "== 2. Vergleich mit der Referenz (22.04/Humble)"
    python3 - <<'PY' | tee /arbeit/vergleich.txt || fehler=1
import json
import numpy as np
neu = json.load(open("/arbeit/karte.json"))
ref = json.load(open("/daten/referenz/meta.json"))
p = np.load("/daten/referenz/poses.npy")
ref_bahn = float(np.linalg.norm(np.diff(p[:, :3], axis=0), axis=1).sum())
zeilen = [("n_scans", ref["n_scans"], neu["n_scans"]),
          ("n_points", ref["n_points"], neu["n_points"]),
          ("bahn_m", round(ref_bahn, 3), neu["bahn_m"])]
ok = True
for name, r, n in zeilen:
    abw = (n - r) / r * 100.0 if r else float("nan")
    # FAST-LIO ist nicht bitgleich: verliert der Rechner unter Last Scans,
    # weicht alles ab. Zwei Laeufe auf 22.04 mit demselben Bag (einer unter
    # Last, 454 statt 460 Scans) lagen 1,3 % (Scans), 1,4 % (Punkte) und 14 %
    # (Bahnlaenge, das Zittern der Posen summiert sich) auseinander.
    grenze = {"n_scans": 3.0, "n_points": 3.0}.get(name, 20.0)
    gut = abs(abw) <= grenze
    ok &= gut
    print(f"  {name:9s} 22.04={r}  24.04={n}  Abweichung {abw:+.2f} % "
          f"({'ok' if gut else 'ZU GROSS'}, Grenze {grenze} %)")
print("VERGLEICH", "OK" if ok else "FEHLGESCHLAGEN")
raise SystemExit(0 if ok else 1)
PY
fi

echo "== 3. RViz-Wiedergabe (xvfb)"
xvfb-run -a -s "-screen 0 1600x1000x24" python3 - "$BAG" <<'PY' | tee /arbeit/rviz.txt || fehler=1
import os, re, subprocess, sys, time
from core.rviz_player import RvizPlayer

bag = sys.argv[1]
p = RvizPlayer(log_cb=lambda z: print(f"  [rviz_player] {z}", flush=True))
print(f"  Umgebung: {' '.join(p.setups)}", flush=True)
p.start(bag)
try:
    time.sleep(5.0)
    lauf = p.is_playing() and p.rviz_running()
    print(f"  nach 5 s: rviz={p.rviz_running()} play={p.is_playing()}", flush=True)
    # Nachweis, dass wirklich abgespielt wird: Rate von /livox/imu (200 Hz im
    # Bag) und /livox/lidar (10 Hz) aus derselben Umgebung messen.
    r = subprocess.run(p._cmd("timeout -s INT 8 ros2 topic hz /livox/imu"),
                       capture_output=True, text=True, env=p.child_env())
    raten = [float(x) for x in re.findall(r"average rate: ([0-9.]+)", r.stdout)]
    print(f"  /livox/imu: {raten[-1] if raten else 'keine Nachrichten'} Hz", flush=True)
    r = subprocess.run(p._cmd("timeout -s INT 8 ros2 node list"),
                       capture_output=True, text=True, env=p.child_env())
    knoten = r.stdout.split()
    print(f"  Knoten: {' '.join(knoten)}", flush=True)
    ok = lauf and bool(raten) and raten[-1] > 50.0 and any("rviz" in k for k in knoten)
finally:
    p.stop()
time.sleep(1.0)
ok = ok and not p.is_playing() and not p.rviz_running()
print(f"  nach stop(): rviz={p.rviz_running()} play={p.is_playing()}", flush=True)
print("RVIZ", "OK" if ok else "FEHLGESCHLAGEN", flush=True)
raise SystemExit(0 if ok else 1)
PY

echo "== 4. Projekt in der App laden (Autotest)"
SUPER360_AUTOTEST="$BAG" SUPER360_AUTOTEST_OUT=/arbeit/autotest \
    timeout 900 xvfb-run -a -s "-screen 0 1600x1000x24" python3 app.py \
    > /arbeit/autotest.txt 2>&1 || fehler=1
grep -E "AUTOTEST (OK|FEHLGESCHLAGEN)" /arbeit/autotest.txt || { tail -20 /arbeit/autotest.txt; fehler=1; }

echo "== Ergebnis: $([[ $fehler == 0 ]] && echo OK || echo FEHLGESCHLAGEN)"
exit $fehler
