#!/usr/bin/env bash
# Funktionstest der App im Image super360-u2404-basis (Ubuntu 24.04) und, mit
# --host, derselbe Lauf mit dem python3 des Rechners zum Vergleich.
#
# Aufrufe:
#
#   bash docker/ubuntu-24.04/pruefe_app.sh --daten DIR [Optionen] [Schritt …]
#   bash docker/ubuntu-24.04/pruefe_app.sh --host --daten DIR [Optionen] [Schritt …]
#
# Schritte (ohne Angabe alle, in dieser Reihenfolge):
#
#   versionen     Python, Qt, VTK und die Pakete aus requirements.txt, OpenCL-Geräte
#   installation  scripts/pruefe_installation.py
#   alles         scripts/umbau/pruefe_alles.sh mit den Stufen aus --stufen
#                 (Vorgabe: kompilieren pyflakes import namen threadregel
#                 selbsttests gpu ui autotest) und --erwartet
#                 scripts/umbau/erwartet_umbau.json. Der Vorher-Stand ist der
#                 Stand vor dem Umbau unter 22.04; die Datei nennt, was der Umbau
#                 gewollt geändert hat (28 statt 29 Job-Funktionen, weil die
#                 beiden FAST-LIO-Jobs in einer Fabrik liegen; ui.main_window
#                 hat keinen Selbsttest mehr). Die Stufe gpu braucht das Bag und
#                 das Projekt, die core/colorizer_gpu.py fest nennt; fehlt eins,
#                 wird sie mit Hinweis übersprungen (Bag auch per --gpu-bag)
#   numerik       scripts/umbau/numerik_probe.py --schreibe nach <aus>/numerik,
#                 danach Hash für Hash gegen --numerik-ref (Vorgabe: der
#                 Vorher-Stand scripts/umbau/vorher); bei Abweichung die größte
#                 Differenz aus den abgelegten Werten. Freigegeben sind die
#                 Teile aus scripts/umbau/erwartet_numerik.json (Grund und Hash
#                 des neuen Werts je Teil, wie numerik_probe.py --erwartet).
#                 Fehlen die abgelegten Werte (*.npz) in scripts/umbau/vorher,
#                 wird der Schritt mit Hinweis übersprungen
#   export        seg0-Projekt aus einer Kopie des Caches laden, als LAS (lokal
#                 und mit festem Georef-Ursprung) und PLY schreiben, zurücklesen
#                 und die Hashes der gelesenen Arrays ausgeben
#   autotest      app.py unter xvfb-run mit SUPER360_AUTOTEST auf das seg0-Bag;
#                 Screenshots und summary.json nach <aus>/autotest
#
# Optionen:
#
#   --daten DIR       Ordner mit Bags und rosbag_suite/cache (RosBagSuper_Gui).
#                     Im Container nur lesend unter seinem eigenen Pfad; HOME
#                     ist dort der Ordner darüber (Cache-Schlüssel, Selbsttests
#                     mit festen Pfaden)
#   --merger DIR      PointCloudMerger, nur lesend als ~/PointCloudMerger
#   --einbinden DIR   weiterer Ordner, nur lesend unter seinem eigenen Pfad
#                     (mehrfach möglich, etwa der Ordner eines Bags, das ein
#                     Selbsttest fest erwartet)
#   --aus DIR         Ausgaben (Vorgabe /tmp/super360_modtests/u2404/app);
#                     wird angelegt, der Container schreibt nur hierhin (dort
#                     als /aus)
#   --image NAME      Vorgabe super360-u2404-basis:latest
#   --stufen "…"      Stufen für 'alles'
#   --gpu-bag DIR     das Bag für die Stufe gpu, falls es nicht an dem Ort liegt,
#                     den core/colorizer_gpu.py nennt (im Container dort
#                     eingebunden; mit --host wirkungslos)
#   --numerik-ref DIR Vergleichsstand für 'numerik' (etwa <aus>/numerik eines
#                     --host-Laufs)
#   --host            ohne Docker auf dem Rechner laufen
#
# Der Container läuft mit --gpus all und der UID/GID des Aufrufers, damit die
# Ausgaben auf dem Rechner nicht root gehören. Das Repo ist nur lesend
# eingebunden. Ende: 0 = alle Schritte bestanden, sonst 1 (2 = Aufruffehler).

set -u

SKRIPT="$(readlink -f "${BASH_SOURCE[0]}")"
REPO="$(cd "$(dirname "$SKRIPT")/../.." && pwd)"
ALLE_SCHRITTE=(versionen installation alles numerik export autotest)
SEG0="rosbag_2026-07-11_15-37-07_seg0"

IMAGE="super360-u2404-basis:latest"
DATEN=""
MERGER=""
EINBINDEN=()
AUS="/tmp/super360_modtests/u2404/app"
STUFEN="kompilieren pyflakes import namen threadregel selbsttests gpu ui autotest"
NUMERIK_REF=""
GPU_BAG=""
ERWARTET_ALLES="scripts/umbau/erwartet_umbau.json"
ERWARTET_NUMERIK="scripts/umbau/erwartet_numerik.json"
HOST=0
DRINNEN=0
GEWAEHLT=()
while [ $# -gt 0 ]; do
    case "$1" in
        --daten) DATEN="${2:-}"; shift 2 || exit 2 ;;
        --merger) MERGER="${2:-}"; shift 2 || exit 2 ;;
        --einbinden) EINBINDEN+=("${2:-}"); shift 2 || exit 2 ;;
        --aus) AUS="${2:-}"; shift 2 || exit 2 ;;
        --image) IMAGE="${2:-}"; shift 2 || exit 2 ;;
        --stufen) STUFEN="${2:-}"; shift 2 || exit 2 ;;
        --numerik-ref) NUMERIK_REF="${2:-}"; shift 2 || exit 2 ;;
        --gpu-bag) GPU_BAG="${2:-}"; shift 2 || exit 2 ;;
        --host) HOST=1; shift ;;
        --drinnen) DRINNEN=1; shift ;;
        -h|--help) sed -n -e '2,/^$/{' -e 's/^# \{0,1\}//' -e 'p' -e '}' "$SKRIPT"; exit 0 ;;
        -*) echo "Unbekannte Option: $1" >&2; exit 2 ;;
        *) GEWAEHLT+=("$1"); shift ;;
    esac
done
for s in "${GEWAEHLT[@]+"${GEWAEHLT[@]}"}"; do
    case " ${ALLE_SCHRITTE[*]} " in
        *" $s "*) ;;
        *) echo "Unbekannter Schritt: $s (bekannt: ${ALLE_SCHRITTE[*]})" >&2; exit 2 ;;
    esac
done
[ ${#GEWAEHLT[@]} -gt 0 ] || GEWAEHLT=("${ALLE_SCHRITTE[@]}")

# ------------------------------------------------------- Stufe gpu vorbereiten

# Bag und Projekt, die der Selbsttest core.colorizer_gpu fest erwartet. Stehen
# sie nicht bereit, fällt die Stufe aus 'alles' heraus, statt zu scheitern.
gpu_pfade() {
    python3 - "$REPO/core/colorizer_gpu.py" <<'PY'
import ast, sys
werte = {}
for knoten in ast.walk(ast.parse(open(sys.argv[1], encoding="utf-8").read())):
    if isinstance(knoten, ast.Assign) and len(knoten.targets) == 1 \
            and getattr(knoten.targets[0], "id", None) in ("BAG", "CACHE"):
        werte[knoten.targets[0].id] = ast.literal_eval(knoten.value)
print(werte.get("BAG", ""))
print(werte.get("CACHE", ""))
PY
}

GPU_EINBINDEN=()
case " ${GEWAEHLT[*]} " in *" alles "*) mit_alles=1 ;; *) mit_alles=0 ;; esac
case " $STUFEN " in *" gpu "*) mit_gpu=1 ;; *) mit_gpu=0 ;; esac
if [ "$DRINNEN" = 0 ] && [ "$mit_alles" = 1 ] && [ "$mit_gpu" = 1 ]; then
    { read -r GPU_SOLL_BAG; read -r GPU_SOLL_CACHE; } < <(gpu_pfade)
    gpu_quelle="${GPU_BAG:-$GPU_SOLL_BAG}"
    grund=""
    if [ -z "$GPU_SOLL_BAG" ] || [ -z "$GPU_SOLL_CACHE" ]; then
        grund="core/colorizer_gpu.py nennt kein BAG/CACHE"
    elif [ ! -d "$gpu_quelle" ]; then
        grund="Bag $gpu_quelle fehlt (anderer Ort: --gpu-bag DIR)"
    elif [ ! -d "$GPU_SOLL_CACHE" ]; then
        grund="Projekt $GPU_SOLL_CACHE fehlt"
    elif [ "$HOST" = 1 ] && [ "$gpu_quelle" != "$GPU_SOLL_BAG" ]; then
        grund="--gpu-bag wirkt nur im Container, der Selbsttest liest $GPU_SOLL_BAG"
    fi
    if [ -n "$grund" ]; then
        echo "Hinweis: Stufe gpu übersprungen — $grund."
        STUFEN="$(echo " $STUFEN " | sed 's/ gpu / /; s/^ *//; s/ *$//')"
    else
        GPU_EINBINDEN=("$(cd "$gpu_quelle" && pwd):$GPU_SOLL_BAG" "$GPU_SOLL_CACHE:$GPU_SOLL_CACHE")
    fi
fi

# ------------------------------------------------------------ Docker starten

if [ "$HOST" = 0 ] && [ "$DRINNEN" = 0 ]; then
    [ -n "$DATEN" ] && [ -d "$DATEN" ] || { echo "--daten fehlt oder ist kein Ordner." >&2; exit 2; }
    DATEN="$(cd "$DATEN" && pwd)"
    mkdir -p "$AUS" && AUS="$(cd "$AUS" && pwd)"
    # HOME im Container ist der Ordner über --daten, wie auf dem Rechner: Der
    # Cache-Schlüssel eines Projekts ist ein Hash des absoluten Bagpfads, nur
    # so finden App und Prüfwerkzeuge die Projekte im Cache wieder. Das HOME
    # selbst ist ein leeres tmpfs, die Daten liegen nur lesend darin.
    HEIM="$(dirname "$DATEN")"
    # Ausgaben unter /aus, nicht unter ihrem Pfad: läge der unter
    # /tmp/super360_modtests, legte Docker die Zwischenordner im Container als
    # root an, und die Prüfwerkzeuge könnten dort ihre Arbeitsordner nicht anlegen.
    volumes=(-v "$REPO:/super360:ro" -v "$AUS:/aus" -e "HOME=$HEIM"
             --tmpfs "$HEIM:exec,mode=1777" -v "$DATEN:$DATEN:ro")
    [ -n "$MERGER" ] && volumes+=(-v "$(cd "$MERGER" && pwd):$HEIM/PointCloudMerger:ro")
    ziele=("$DATEN")
    for d in "${EINBINDEN[@]+"${EINBINDEN[@]}"}"; do
        d="$(cd "$d" && pwd)" || exit 2
        volumes+=(-v "$d:$d:ro")
        ziele+=("$d")
    done
    # Bag und Projekt der Stufe gpu, sofern nicht schon über --daten oder
    # --einbinden im Container.
    for paar in "${GPU_EINBINDEN[@]+"${GPU_EINBINDEN[@]}"}"; do
        quelle="${paar%%:*}" ziel="${paar#*:}" drin=0
        for z in "${ziele[@]}"; do
            case "$ziel/" in "$z"/*) [ "$quelle" = "$ziel" ] && drin=1 ;; esac
        done
        [ "$drin" = 1 ] || volumes+=(-v "$quelle:$ziel:ro")
    done
    argumente=(--drinnen --aus /aus --stufen "$STUFEN")
    if [ -n "$NUMERIK_REF" ]; then
        volumes+=(-v "$(cd "$NUMERIK_REF" && pwd):/numerik_ref:ro")
        argumente+=(--numerik-ref /numerik_ref)
    fi
    exec docker run --rm --gpus all --user "$(id -u):$(id -g)" --shm-size 2g \
        "${volumes[@]}" -w /super360 "$IMAGE" \
        bash /super360/docker/ubuntu-24.04/pruefe_app.sh "${argumente[@]}" "${GEWAEHLT[@]}"
fi

# ----------------------------------------------- Schritte (Container oder Host)

cd "$REPO"
mkdir -p "$AUS"
export PYTHONDONTWRITEBYTECODE=1
# Selbsttests und App schreiben nie in den echten Cache.
export SUPER360_CACHE_ROOT="$AUS/cache"
mkdir -p "$SUPER360_CACHE_ROOT"
ECHT_CACHE="$HOME/RosBagSuper_Gui/rosbag_suite/cache"
FEHLER=()

schritt() {
    local name="$1"; shift
    echo
    echo "=================== $name"
    local t0=$SECONDS
    "$@"
    local code=$?
    echo "=================== $name: Ende $code nach $((SECONDS - t0)) s"
    [ "$code" = 0 ] || FEHLER+=("$name")
}

seg0_kopie() {
    # Frische Kopie des seg0-Projekts in den Test-Cache; gibt ihren Ordner aus.
    local quelle
    quelle="$(ls -d "$ECHT_CACHE/$SEG0"-* 2>/dev/null | head -n1)"
    [ -n "$quelle" ] || { echo "Kein seg0-Projekt unter $ECHT_CACHE" >&2; return 1; }
    local ziel="$SUPER360_CACHE_ROOT/$(basename "$quelle")"
    rm -rf "$ziel" && cp -r "$quelle" "$ziel" && chmod -R u+w "$ziel" && echo "$ziel"
}

s_versionen() {
    python3 - <<'PY'
import importlib, platform, sys
from importlib import metadata
print("Python", sys.version.split()[0], "auf", platform.platform(), "|", sys.executable)
from PyQt5.QtCore import PYQT_VERSION_STR, QT_VERSION_STR
import vtk
print("PyQt5", PYQT_VERSION_STR, "Qt", QT_VERSION_STR, "VTK", vtk.vtkVersion.GetVTKVersion())
for p in ("numpy", "scipy", "opencv-python", "opencv-python-headless", "open3d", "rosbags",
          "Pillow", "laspy", "pyproj", "pyqtdarktheme", "pyopencl"):
    try:
        print(f"  {p:24s} {metadata.version(p)}")
    except metadata.PackageNotFoundError:
        print(f"  {p:24s} fehlt")
try:
    import pyopencl as cl
    for pf in cl.get_platforms():
        for d in pf.get_devices():
            print(f"OpenCL: {pf.name} / {d.name} ({cl.device_type.to_string(d.type)})")
except Exception as exc:  # noqa: BLE001
    print(f"OpenCL: keins ({exc})")
PY
}

s_installation() {
    python3 scripts/pruefe_installation.py
}

s_alles() {
    # Ohne Stufe liefe pruefe_alles.sh alle; bleibt nach dem Übergehen von gpu
    # keine übrig, gibt es nichts zu tun.
    [ -n "${STUFEN// /}" ] || { echo "Keine Stufe übrig."; return 0; }
    # shellcheck disable=SC2086
    bash scripts/umbau/pruefe_alles.sh --erwartet "$ERWARTET_ALLES" $STUFEN
}

s_numerik() {
    local ziel="$AUS/numerik" ref="${NUMERIK_REF:-scripts/umbau/vorher}"
    if ! ls "$ref"/numerik_*.npz >/dev/null 2>&1; then
        # Nur der Vorher-Stand im Repo darf fehlen; ein ausdrücklich genannter nicht.
        [ -z "$NUMERIK_REF" ] || { echo "--numerik-ref: in $ref liegen keine numerik_*.npz" >&2; return 1; }
        echo "Hinweis: numerik übersprungen — in $ref liegen keine numerik_*.npz" \
             "(abgelegte Werte des Vergleichsstands)."
        return 0
    fi
    rm -rf "$ziel"
    python3 scripts/umbau/numerik_probe.py --schreibe --ziel "$ziel" || return 1
    python3 - "$ziel" "$ref" "$ERWARTET_NUMERIK" <<'PY'
import glob, json, os, sys
import numpy as np
neu_dir, alt_dir, erwartet_pfad = sys.argv[1:4]
sys.path.insert(0, "scripts/umbau")
from numerik_probe import _eintrag_fuer, erwartet_laden, groesste_differenz
erwartet = erwartet_laden(erwartet_pfad) if os.path.isfile(erwartet_pfad) else {}
benutzt = set()


def freigegeben(familie, key, wert):
    """Grund aus erwartet_numerik.json, wenn der neue Wert genau so freigegeben ist."""
    e = _eintrag_fuer(erwartet, f"{familie}.{key}")
    if e is not None and erwartet[e]["hash"] == wert:
        benutzt.add(e)
        return erwartet[e]["grund"]
    return None


gleich = toleranz = frei = anders = 0
for js in sorted(glob.glob(os.path.join(neu_dir, "numerik_*.json"))):
    familie = os.path.basename(js)[8:-5]
    alt_js = os.path.join(alt_dir, os.path.basename(js))
    if not os.path.isfile(alt_js):
        print(f"{familie}: kein Vergleichsstand"); anders += 1; continue
    neu, alt = json.load(open(js)), json.load(open(alt_js))
    um_a, um_n = alt["umgebung"], neu["umgebung"]
    print(f"{familie}: Umgebung " + ", ".join(f"{k} {um_a.get(k)} -> {um_n.get(k)}"
          for k in sorted(set(um_a) | set(um_n)) if um_a.get(k) != um_n.get(k)))
    w_n = np.load(js[:-5] + ".npz"); w_a = np.load(alt_js[:-5] + ".npz")
    for probe in sorted(set(alt["proben"]) | set(neu["proben"])):
        ta = alt["proben"].get(probe, {}).get("teile", {})
        tn = neu["proben"].get(probe, {}).get("teile", {})
        for teil in sorted(set(ta) | set(tn)):
            a, n = ta.get(teil), tn.get(teil)
            if a and n and a["hash"] == n["hash"]:
                gleich += 1; continue
            anders += 1
            key = f"{probe}/{teil}"
            grund = freigegeben(familie, key, "neu" if a is None else "fehlt" if n is None
                                else n["hash"])
            if grund is not None:
                anders -= 1
                frei += 1
                print(f"  {key}: freigegeben ({grund[:70]}…)")
                continue
            if a is None or n is None:
                print(f"  {key}: nur {'neu' if a is None else 'alt'}"); continue
            if key in w_a.files and key in w_n.files:
                _, text = groesste_differenz(w_a[key], w_n[key])
            else:
                text = "Wert nicht abgelegt (> 2 MB)"
            unstet = neu["proben"][probe].get("unstet")
            if unstet is not None and key in w_a.files and key in w_n.files \
                    and groesste_differenz(w_a[key], w_n[key])[0] <= unstet:
                anders -= 1
                toleranz += 1
                continue
            print(f"  {key}: Hash anders{' (unstet, Toleranz %s)' % unstet if unstet else ''}; {text}")
for e in sorted(set(erwartet) - benutzt):
    print(f"  Hinweis: {e} ist freigegeben, weicht aber nicht (so) ab")
print(f"numerik: {gleich} Teile bitgleich, {toleranz} unstete in ihrer Toleranz, "
      f"{frei} freigegeben, {anders} anders")
sys.exit(1 if anders else 0)
PY
}

s_export() {
    local kopie
    kopie="$(seg0_kopie)" || return 1
    python3 - "$kopie" "$AUS/export" <<'PY'
import hashlib, os, sys
import numpy as np
sys.path.insert(0, os.getcwd())
from core import georef
from core.project import Project
from core.recording import Recording
kopie, ziel = sys.argv[1], sys.argv[2]
os.makedirs(ziel, exist_ok=True)
h = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]
p = Project.from_dir(kopie)
rec = Recording.load(p.recording_dir(), level=False)
pts = rec.world_points()
cols = np.fromfile(os.path.join(p.colors_dir(), "colors.bin"), np.uint8).reshape(-1, 3)
valid = np.fromfile(os.path.join(p.colors_dir(), "valid.bin"), np.uint8).astype(bool)
print(f"Projekt {p.bag_name}: {len(pts)} Punkte, {int(valid.sum())} eingefärbt, Welt {h(pts)}")
pts, cols = pts[valid], cols[valid]

import laspy, open3d as o3d
geo = georef.GeorefResult(T_enu_world=np.eye(4), origin_llh=(50.0, 8.0, 100.0),
                          utm_epsg=32632, rms_m=0.0, n_used=0, utm_offset=(0.0, 0.0))
for name, g in (("lokal", None), ("georef", geo)):
    pfad = os.path.join(ziel, f"wolke_{name}.las")
    georef.export_las(pts, cols, g, pfad)
    las = laspy.read(pfad)
    xyz = np.column_stack([las.x, las.y, las.z])
    rgb = np.column_stack([las.red, las.green, las.blue])
    crs = las.header.parse_crs()
    print(f"LAS {name}: {las.header.point_count} Punkte, xyz {h(np.asarray(las.X))}"
          f"{h(np.asarray(las.Y))}{h(np.asarray(las.Z))}, rgb {h(rgb)}, "
          f"CRS {crs.to_epsg() if crs else None}")
    assert las.header.point_count == len(pts)
    assert np.array_equal(rgb, cols.astype(np.uint16) * 257)
    if g is None:
        assert np.abs(xyz - pts).max() <= 0.0005 + 1e-9, np.abs(xyz - pts).max()
for endung in ("ply", "pcd"):
    pfad = os.path.join(ziel, f"wolke.{endung}")
    georef.export_ply_pcd(pts, cols, pfad)
    pc = o3d.io.read_point_cloud(pfad)
    xyz, rgb = np.asarray(pc.points), np.asarray(pc.colors)
    print(f"{endung.upper()}: {len(xyz)} Punkte, xyz {h(xyz)}, rgb {h(rgb)}, "
          f"Datei {hashlib.sha256(open(pfad, 'rb').read()).hexdigest()[:16]}")
    assert np.array_equal(xyz, pts.astype(np.float64))
    assert np.array_equal(np.rint(rgb * 255).astype(np.uint8), cols)
print("export: bestanden")
PY
}

s_autotest() {
    local kopie bag
    kopie="$(seg0_kopie)" || return 1
    bag="$HOME/RosBagSuper_Gui/$SEG0"
    [ -d "$bag" ] || { echo "Bag fehlt: $bag" >&2; return 1; }
    rm -rf "$AUS/autotest" && mkdir -p "$AUS/autotest"
    SUPER360_AUTOTEST="$bag" SUPER360_AUTOTEST_OUT="$AUS/autotest" \
        timeout 900 xvfb-run -a -s "-screen 0 1920x1080x24" python3 app.py
    local code=$?
    ls "$AUS/autotest"
    [ -f "$AUS/autotest/summary.json" ] && python3 - "$AUS/autotest/summary.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
for k, v in s.items():
    if k != "log_lines":
        print(f"  {k}: {v}")
PY
    return $code
}

for s in "${GEWAEHLT[@]}"; do
    schritt "$s" "s_$s"
done

echo
if [ ${#FEHLER[@]} -eq 0 ]; then
    echo "pruefe_app: alle Schritte bestanden (${GEWAEHLT[*]})"
    exit 0
fi
echo "pruefe_app: fehlgeschlagen: ${FEHLER[*]}"
exit 1
