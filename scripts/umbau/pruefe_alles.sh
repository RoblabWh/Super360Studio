#!/usr/bin/env bash
# Sammelprüfung für den Umbau: alle Stufen gegen den Vorher-Stand.
#
# Aufrufe:
#
#   bash scripts/umbau/pruefe_alles.sh [Optionen] [Stufe …]
#   bash scripts/umbau/pruefe_alles.sh --schreibe-vorher [--ersetzen] [--wurzel DIR | --tag TAG]
#
# Ohne Stufe laufen alle, immer in dieser Reihenfolge (die Reihenfolge der
# Argumente zählt nicht):
#
#   kompilieren  py_compile über app.py, core, ui, scripts (die .pyc landen im
#                Arbeitsordner, nicht im Baum)
#   pyflakes     python3 -m pyflakes über app.py, core, ui, scripts ohne
#                scripts/umbau (die Werkzeuge prüfen ihre eigenen Selbsttests);
#                gezählt wird je Meldung ohne Datei und Zeile, damit verschobener
#                Code dieselbe Meldung behält. Mehr als in vorher/pyflakes.txt
#                ist ein Fehler, 'undefined name' immer
#   import       python3 -c 'import app, ui.main_window'
#   namen        ast: kein Methoden- oder Klassenattributname in zwei Klassen
#                von MainWindow (ui/main_window.py) und den Mixins aus
#                ui/fenster/*.py (Klassen auf …Mixin und Basisklassen von
#                MainWindow, die dort liegen); solange MainWindow von keiner
#                Mixin-Klasse erbt, nur unter den Mixins, ein Name zusätzlich
#                in MainWindow ist bis zum Umschalten erlaubt
#   threadregel  ast: keine Job-Funktion in ui/ benutzt self. Job ist, was als
#                Job an _start_worker oder Worker geht (Funktion, Lambda,
#                Ausdruck), dazu jede verschachtelte Funktion job oder mit den
#                Parametern progress_cb, cancel, log_cb; geprüft werden Körper
#                und Vorgabewerte, Aliase wie 'fenster = self' und die von ihr
#                benutzten Nachbarfunktionen des Handlers. Stammt der Job aus
#                einer Fabrikmethode (self._x(…), direkt, über einen Index oder
#                über einen Namen, auch 'job, config, rate = self._x(…)'), muss
#                es _x in einer Klasse geben (bei MainWindow und den Mixins über
#                ui/main_window.py und ui/fenster/*.py, sonst in derselben Datei)
#                und _x eine Job-Funktion enthalten; die prüft die Regel dort.
#                Stammt er aus einer Modulfunktion der Datei, direkt
#                (_fabrik(…)) oder über einen Namen (job = _fabrik(…)), gilt
#                die Funktion samt ihrem Körper als Job.
#                Eine direkt übergebene gebundene Methode (self._x) bleibt ein
#                Fehler. Weniger Jobs als im Stand vor-umbau (29) ist ein Fehler.
#                Legt der Umbau Jobs zusammen (ab Welle 14 etwa die beiden
#                FAST-LIO-Jobs in einer Fabrik, dann 28), braucht die Stufe
#                --erwartet mit "threadregel_jobs".
#                Grenzen: eine gebundene Methode in einer lokalen Variable
#                ('melde = self._lbl.setText', im Job gerufen) und eine, die als
#                Argument in die Fabrik geht und dort im Job landet, fallen
#                nicht auf
#   selbsttests  python3 -m core.<modul> für jedes Modul in core/ mit
#                __main__-Block, selbst gefunden (core.stitcher startet dabei
#                einen Docker-Container, sofern das Image panoweave da ist)
#   gpu          python3 -m core.colorizer_gpu (liest ein Projekt aus dem echten
#                Cache; dessen meta.json muss gravity_level schon enthalten)
#   ui           QT_QPA_PLATFORM=offscreen python3 -m ui.<modul> für jedes
#                ui-Modul mit __main__-Block, selbst gefunden; dazu
#                xvfb-run -a python3 -m ui.cloud_view und
#                xvfb-run -a python3 scripts/test_cloud_view_steuerung.py
#   ui-zeit      ui.pano_view; prüft Wanduhrzeiten und gehört deshalb nur in
#                Prüfläufe, die allein auf dem Rechner laufen
#   autotest     zweimal app.py mit SUPER360_AUTOTEST auf derselben frischen
#                Kopie des seg0-Projekts: der erste Lauf rechnet Explorationsgrad
#                und Lotrechte, der zweite nimmt den Cache. summary.json ohne
#                log_lines und protokoll.txt ohne Zeitstempel, Dauern und
#                Ordnernamen gegen vorher/autotest
#   echtcache    Größe und Änderungszeit jeder Datei im echten Cache (darunter
#                alle recording/meta.json) und die Liste seiner Projekte sind am
#                Ende wie beim Start
#
# Ausgenommen von selbsttests und ui (mit Grund in vorher/selbsttests.txt):
# core.colorizer, core.fastlio_runner, ui.main_window, ui.explorationsgrad;
# core.colorizer_gpu und ui.pano_view haben eigene Stufen. Ein Selbsttest, der
# im Vorher-Stand bestand, muss bestehen; ein neu gefundener ebenso.
#
# Optionen:
#
#   --wurzel DIR      Repo-Wurzel, aus der geprüft wird (Vorgabe: der
#                     Arbeitsbaum), etwa ein mit 'git archive' entpackter Stand
#   --tag TAG         TAG per 'git archive' in den Arbeitsordner entpacken und
#                     als Wurzel nehmen
#   --erwartet JSON   erlaubte Abweichungen, Schlüssel (alle optional):
#                       "pyflakes":              [Regex über die Meldung, …]
#                       "selbsttests_entfallen": ["core.x", …]
#                       "autotest_summary":      ["schlüssel", …]
#                       "autotest_protokoll":    [Regex über die Zeile, …]
#                       "threadregel_jobs":      Zahl der Job-Funktionen, wenn
#                                                gewollt weniger als vorher (29)
#   --schreibe-vorher legt vorher/pyflakes.txt, vorher/selbsttests.txt und
#                     vorher/autotest/ an (Stufen pyflakes, selbsttests, gpu,
#                     ui, ui-zeit, autotest, echtcache); verweigert vorhandene
#                     Ziele ohne --ersetzen und aus dem Arbeitsbaum jeden Stand,
#                     der in app.py, core, ui oder scripts vom Tag vor-umbau
#                     abweicht
#   --behalten        Arbeitsordner und die Ausgabeordner der Selbsttests unter
#                     /tmp/super360_modtests stehen lassen (Fehlersuche)
#
# Jeder Aufruf arbeitet in einem eigenen Ordner von basis.arbeitsordner().
# Kinder bekommen SUPER360_CACHE_ROOT im Arbeitsordner (nie der echte Cache),
# TMPDIR im Arbeitsordner und PYTHONDONTWRITEBYTECODE. Was die Selbsttests
# nach ihrer Konvention unter /tmp/super360_modtests/<modul> ablegen, wird nach
# der Stufe wieder entfernt, sofern es vor ihr nicht da war; nur Ordner, deren
# Name zu einem in der Stufe gestarteten Modul passt. Die Stufen mit
# Selbsttests laufen dafür unter einer Sperre auf /tmp/super360_modtests/umbau:
# zwei gleichzeitige Aufrufe räumen sich so nicht gegenseitig die Ordner weg.
#
# Ende: 0 = alle Stufen bestanden, 1 = mindestens eine Stufe fehlgeschlagen,
# 2 = Aufruffehler.

set -u

WERKZEUG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARBEITSBAUM="$(cd "$WERKZEUG/../.." && pwd)"
VORHER="$WERKZEUG/vorher"
ALLE_STUFEN=(kompilieren pyflakes import namen threadregel selbsttests gpu ui ui-zeit autotest echtcache)
VORHER_STUFEN=(pyflakes selbsttests gpu ui ui-zeit autotest echtcache)

export PYTHONDONTWRITEBYTECODE=1
unset SUPER360_AUTOTEST SUPER360_AUTOTEST_OUT

hilfe() {
    sed -n -e '2,/^$/{' -e 's/^# \{0,1\}//' -e 'p' -e '}' "${BASH_SOURCE[0]}"
}

# ----------------------------------------------------------------- Argumente

WURZEL=""
TAG=""
ERWARTET=""
SCHREIBE=0
ERSETZEN=0
BEHALTEN=0
GEWAEHLT=()
while [ $# -gt 0 ]; do
    case "$1" in
        --wurzel) WURZEL="${2:-}"; shift 2 || { echo "--wurzel braucht einen Ordner." >&2; exit 2; } ;;
        --wurzel=*) WURZEL="${1#*=}"; shift ;;
        --tag) TAG="${2:-}"; shift 2 || { echo "--tag braucht einen Namen." >&2; exit 2; } ;;
        --tag=*) TAG="${1#*=}"; shift ;;
        --erwartet) ERWARTET="${2:-}"; shift 2 || { echo "--erwartet braucht eine Datei." >&2; exit 2; } ;;
        --erwartet=*) ERWARTET="${1#*=}"; shift ;;
        --schreibe-vorher) SCHREIBE=1; shift ;;
        --ersetzen) ERSETZEN=1; shift ;;
        --behalten) BEHALTEN=1; shift ;;
        -h|--help) hilfe; exit 0 ;;
        -*) echo "Unbekannte Option: $1" >&2; exit 2 ;;
        *) GEWAEHLT+=("$1"); shift ;;
    esac
done

for s in "${GEWAEHLT[@]+"${GEWAEHLT[@]}"}"; do
    case " ${ALLE_STUFEN[*]} " in
        *" $s "*) ;;
        *) echo "Unbekannte Stufe: $s (bekannt: ${ALLE_STUFEN[*]})" >&2; exit 2 ;;
    esac
done
if [ -n "$WURZEL" ] && [ -n "$TAG" ]; then
    echo "--wurzel und --tag schließen sich aus." >&2; exit 2
fi
if [ "$SCHREIBE" = 1 ]; then
    if [ ${#GEWAEHLT[@]} -gt 0 ]; then
        echo "--schreibe-vorher nimmt keine Stufen; es laufen: ${VORHER_STUFEN[*]}" >&2; exit 2
    fi
    if [ -n "$ERWARTET" ]; then
        echo "--schreibe-vorher und --erwartet schließen sich aus." >&2; exit 2
    fi
    GEWAEHLT=("${VORHER_STUFEN[@]}")
elif [ ${#GEWAEHLT[@]} -eq 0 ]; then
    GEWAEHLT=("${ALLE_STUFEN[@]}")
fi
if [ -n "$ERWARTET" ]; then
    [ -f "$ERWARTET" ] || { echo "--erwartet: $ERWARTET gibt es nicht." >&2; exit 2; }
    ERWARTET="$(cd "$(dirname "$ERWARTET")" && pwd)/$(basename "$ERWARTET")"
fi

# ----------------------------------------------------------- Python-Helfer

read -r -d '' HELFER <<'PY'
import ast
import collections
import difflib
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
WERKZEUG = os.environ["PRUEFE_WERKZEUG"]
sys.path.insert(0, WERKZEUG)
import basis  # noqa: E402

WURZEL = os.environ["PRUEFE_WURZEL"]
ARBEIT = os.environ["PRUEFE_ARBEIT"]
VORHER = os.environ["PRUEFE_VORHER"]
SCHREIBE = os.environ.get("PRUEFE_SCHREIBE") == "1"
BEHALTEN = os.environ.get("PRUEFE_BEHALTEN") == "1"
MODTESTS = "/tmp/super360_modtests"

#: Selbsttests, die nicht oder nicht in ihrer Stufe laufen, mit Grund.
AUSGENOMMEN = {
    "core.colorizer": ("selbsttests", "nicht lauffähig",
                       "ausgenommen: schreibt in den echten Cache "
                       "(core/colorizer.py:1234-1237, 1349-1351)"),
    "core.fastlio_runner": ("selbsttests", "nicht lauffähig",
                            "ausgenommen: startet FAST-LIO und schreibt ins Repo "
                            "(core/fastlio_runner.py:635)"),
    "core.colorizer_gpu": ("gpu", None, None),
    "ui.main_window": ("ui", "nicht lauffähig",
                       "ausgenommen: startet die ganze App und endet in app.exec_()"),
    "ui.explorationsgrad": ("ui", "nicht lauffähig",
                            "ausgenommen: endet in app.exec_() (ui/explorationsgrad.py:184) "
                            "und wartet auf das Schließen des Fensters"),
    "ui.pano_view": ("ui-zeit", None, None),
    "ui.cloud_view": ("ui", None, None),
}
#: Projekt im echten Cache, das ``python3 -m core.colorizer_gpu`` liest.
GPU_PROJEKT = "rosbag_2026-09-19_02-52-40-6e1e1c7b"
ZEITLIMIT = {"core.stitcher": 1500}


def neutral(text):
    """Arbeitsordner, Wurzel und Heimatordner durch feste Marken ersetzen."""
    text = str(text)
    marken = [(ARBEIT, "<arbeit>"), (WURZEL, "<wurzel>"),
              (basis.ARBEITSBAUM, "<wurzel>"),
              (basis.ARBEIT_WURZEL, "<arbeitswurzel>"),
              (os.path.expanduser("~"), "~")]
    for pfad, marke in sorted(marken, key=lambda m: -len(m[0])):
        for form in {pfad, os.path.realpath(pfad)}:
            text = text.replace(form, marke)
    return text


def lade_erwartet():
    pfad = os.environ.get("PRUEFE_ERWARTET")
    if not pfad:
        return {}
    with open(pfad, encoding="utf-8") as fh:
        d = json.load(fh)
    erlaubt = {"pyflakes", "selbsttests_entfallen", "autotest_summary",
               "autotest_protokoll", "threadregel_jobs"}
    falsch = sorted(set(d) - erlaubt)
    if falsch:
        raise SystemExit(f"--erwartet: unbekannte Schlüssel {falsch} "
                         f"(bekannt: {sorted(erlaubt)})")
    return d


def py_dateien(*teile, ohne=()):
    out = []
    for teil in teile:
        voll = os.path.join(WURZEL, teil)
        if os.path.isfile(voll):
            out.append(teil)
            continue
        for ordner, unter, dateien in os.walk(voll):
            rel = os.path.relpath(ordner, WURZEL)
            unter[:] = sorted(u for u in unter if u != "__pycache__"
                              and os.path.join(rel, u) not in ohne)
            out += [os.path.join(rel, d) for d in sorted(dateien) if d.endswith(".py")]
    return out


def kind_umgebung(**extra):
    env = dict(os.environ)
    for var in ("SUPER360_AUTOTEST", "SUPER360_AUTOTEST_OUT", "PYTHONPATH",
                "QT_QPA_PLATFORM"):
        env.pop(var, None)
    for var in [k for k in env if k.startswith("PRUEFE_")]:
        env.pop(var)
    tmp = os.path.join(ARBEIT, "tmp")
    cache = os.path.join(ARBEIT, "cache_selbsttests")
    os.makedirs(tmp, exist_ok=True)
    os.makedirs(cache, exist_ok=True)
    env.update(PYTHONDONTWRITEBYTECODE="1", TMPDIR=tmp, SUPER360_CACHE_ROOT=cache)
    env.update(extra)
    return env


# --------------------------------------------------------------- kompilieren

def stufe_kompilieren():
    import py_compile
    dateien = py_dateien("app.py", "core", "ui", "scripts")
    ziel = os.path.join(ARBEIT, "pyc")
    fehler = []
    for rel in dateien:
        try:
            py_compile.compile(os.path.join(WURZEL, rel),
                               cfile=os.path.join(ziel, rel + "c"), doraise=True)
        except py_compile.PyCompileError as exc:
            fehler.append(neutral(exc.msg.strip()))
    for f in fehler:
        print(f"  {f}")
    print(f"  {len(dateien)} Dateien übersetzt, {len(fehler)} mit Fehler")
    return not fehler


# ------------------------------------------------------------------ pyflakes

_PF = re.compile(r"^(?P<datei>.*?):(?P<zeile>\d+):(?:(?P<spalte>\d+):?)? (?P<text>.*)$")


def _pf_schluessel(text):
    return re.sub(r"\bline \d+", "line N", text)


def stufe_pyflakes():
    dateien = py_dateien("app.py", "core", "ui", "scripts", ohne=("scripts/umbau",))
    r = subprocess.run([sys.executable, "-m", "pyflakes"] + dateien, cwd=WURZEL,
                       env=kind_umgebung(), capture_output=True, text=True)
    zeilen, sonst = [], []
    for z in (r.stdout + r.stderr).splitlines():
        if not z.strip():
            continue
        (zeilen if _PF.match(z) else sonst).append(neutral(z))
    if sonst:
        for z in sonst:
            print(f"  pyflakes: {z}")
        print("  pyflakes lief nicht sauber durch.")
        return False
    zeilen.sort(key=lambda z: (_PF.match(z)["datei"], int(_PF.match(z)["zeile"]), z))
    ziel = os.path.join(VORHER, "pyflakes.txt")
    if SCHREIBE:
        with open(ziel, "w", encoding="utf-8") as fh:
            fh.write("# python3 -m pyflakes über app.py, core, ui, scripts (ohne "
                     "scripts/umbau), Stand vor dem Umbau\n")
            fh.write("".join(z + "\n" for z in zeilen))
        print(f"  {len(dateien)} Dateien, {len(zeilen)} Meldungen -> vorher/pyflakes.txt")
        return not any("undefined name" in z for z in zeilen)
    if not os.path.isfile(ziel):
        print("  vorher/pyflakes.txt fehlt — erst --schreibe-vorher.")
        return False
    with open(ziel, encoding="utf-8") as fh:
        alt = [z.rstrip("\n") for z in fh if z.strip() and not z.startswith("#")]
    erlaubt = [re.compile(m) for m in lade_erwartet().get("pyflakes", [])]
    vorher_n = collections.Counter(_pf_schluessel(_PF.match(z)["text"]) for z in alt)
    jetzt = collections.defaultdict(list)
    for z in zeilen:
        jetzt[_pf_schluessel(_PF.match(z)["text"])].append(z)
    ok = True
    for text, liste in sorted(jetzt.items()):
        zuviel = len(liste) - vorher_n.get(text, 0)
        frei = any(m.search(text) for m in erlaubt)
        if "undefined name" in text and not frei:
            ok = False
            for z in liste:
                print(f"  FEHLER (undefined name): {z}")
        elif zuviel > 0:
            art = "erlaubt" if frei else "NEU"
            ok = ok and frei
            for z in liste:
                print(f"  {art}: {z}" + (f"  ({zuviel} mehr als vorher)" if len(liste) > 1 else ""))
    weg = sum(max(0, n - len(jetzt.get(t, []))) for t, n in vorher_n.items())
    print(f"  {len(dateien)} Dateien, {len(zeilen)} Meldungen (vorher {len(alt)}, "
          f"{weg} entfallen)")
    return ok


# -------------------------------------------------------------------- import

def stufe_import():
    r = subprocess.run([sys.executable, "-c", "import app, ui.main_window"],
                       cwd=WURZEL, env=kind_umgebung(QT_QPA_PLATFORM="offscreen"),
                       capture_output=True, text=True)
    aus = neutral((r.stdout + r.stderr).strip())
    if aus:
        print("  " + aus.replace("\n", "\n  "))
    print(f"  import app, ui.main_window: Exit {r.returncode}")
    return r.returncode == 0


# --------------------------------------------------------------------- namen

def _klassen_namen(klasse):
    namen = set()
    for knoten in klasse.body:
        if isinstance(knoten, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            namen.add(knoten.name)
        elif isinstance(knoten, ast.Assign):
            for ziel in knoten.targets:
                for n in ast.walk(ziel):
                    if isinstance(n, ast.Name):
                        namen.add(n.id)
        elif isinstance(knoten, (ast.AnnAssign, ast.AugAssign)):
            if isinstance(knoten.target, ast.Name):
                namen.add(knoten.target.id)
    return namen


def _basisname(b):
    if isinstance(b, ast.Name):
        return b.id
    if isinstance(b, ast.Attribute):
        return b.attr
    return None


def stufe_namen():
    pfad_mw = os.path.join(WURZEL, "ui", "main_window.py")
    with open(pfad_mw, encoding="utf-8") as fh:
        baum = ast.parse(fh.read(), pfad_mw)
    mw = [k for k in baum.body if isinstance(k, ast.ClassDef) and k.name == "MainWindow"]
    if not mw:
        print("  ui/main_window.py: keine Klasse MainWindow")
        return False
    mw = mw[0]
    basen = {_basisname(b) for b in mw.bases}
    klassen = [("ui/main_window.py", mw)]
    uebergangen = []
    ordner = os.path.join(WURZEL, "ui", "fenster")
    if os.path.isdir(ordner):
        for datei in sorted(os.listdir(ordner)):
            if not datei.endswith(".py"):
                continue
            pfad = os.path.join(ordner, datei)
            with open(pfad, encoding="utf-8") as fh:
                b = ast.parse(fh.read(), pfad)
            for k in b.body:
                if not isinstance(k, ast.ClassDef):
                    continue
                if k.name.endswith("Mixin") or k.name in basen:
                    klassen.append((f"ui/fenster/{datei}", k))
                else:
                    uebergangen.append(f"ui/fenster/{datei}:{k.name}")
    # Vor dem Umschalten (MainWindow erbt von keiner Mixin-Klasse) liegen die
    # Mixins als Kopien neben dem unveränderten MainWindow: gezählt wird dann
    # nur unter den Mixins, ein Name in MainWindow und einem Mixin ist bis zum
    # Umschalten erlaubt. Danach überdeckt MainWindow still, jedes Doppel zählt.
    mixin_namen = {k.name for datei, k in klassen[1:]}
    umgeschaltet = bool(basen & mixin_namen)
    mw_ort = f"ui/main_window.py:{mw.name}"
    wo = collections.defaultdict(list)
    for datei, k in klassen:
        for name in sorted(_klassen_namen(k)):
            wo[name].append(f"{datei}:{k.name}")
    doppelt, bis_umschalten = {}, []
    for n, orte in wo.items():
        if len(orte) < 2:
            continue
        if umgeschaltet or len([o for o in orte if o != mw_ort]) > 1:
            doppelt[n] = orte
        else:
            bis_umschalten.append(n)
    for name, orte in sorted(doppelt.items()):
        print(f"  DOPPELT {name}: {', '.join(orte)}")
    if uebergangen:
        print(f"  nicht geprüft (keine Mixin-Klasse): {', '.join(uebergangen)}")
    print(f"  MainWindow erbt {'von den Mixins' if umgeschaltet else 'noch nicht von den Mixins'}")
    if bis_umschalten:
        print(f"  zusätzlich in MainWindow (bis zum Umschalten erlaubt): {len(bis_umschalten)}")
    print(f"  {len(klassen)} Klassen ({', '.join(k.name for _, k in klassen)}), "
          f"{len(wo)} Namen, {len(doppelt)} doppelt")
    return not doppelt


# --------------------------------------------------------------- threadregel

#: Job-Funktionen in ui/ im Stand vor-umbau (29 Aufrufe von _start_worker,
#: jeder mit einer verschachtelten Funktion job). Weniger ist ein Fehler, außer
#: --erwartet nennt die neue Zahl unter "threadregel_jobs".
JOBS_VORHER = 29
_FUNKTION = (ast.FunctionDef, ast.AsyncFunctionDef)
_JOB_PARAMETER = {"progress_cb", "cancel", "log_cb"}


def _parameter(fn):
    a = fn.args
    namen = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
    namen |= {x.arg for x in (a.vararg, a.kwarg) if x is not None}
    return namen


def _eigene_knoten(fn):
    """Knoten im Körper von ``fn`` ohne die verschachtelter Funktionen und Klassen."""
    stapel = list(fn.body) if isinstance(fn.body, list) else [fn.body]
    while stapel:
        k = stapel.pop()
        yield k
        if isinstance(k, _FUNKTION + (ast.Lambda, ast.ClassDef)):
            continue
        stapel.extend(ast.iter_child_nodes(k))


_BEREICHE = {}


def _bereich(fn):
    """Lokale Funktionen (def und ``name = lambda``) und Aliase von self in ``fn``."""
    if id(fn) not in _BEREICHE:
        _BEREICHE[id(fn)] = (fn, _bereich_rechnen(fn))
    return _BEREICHE[id(fn)][1]


def _bereich_rechnen(fn):
    """(defs, alias, fabriken); fabriken: Name -> Aufruf, dem er zugewiesen ist."""
    defs, alias, fabriken = {}, {"self"}, {}
    zuweisungen = []
    for k in _eigene_knoten(fn):
        if isinstance(k, _FUNKTION):
            defs[k.name] = k
        elif isinstance(k, ast.Assign):
            zuweisungen += [(z, k.value) for z in k.targets]
        elif isinstance(k, (ast.AnnAssign, ast.NamedExpr)) and k.value is not None:
            zuweisungen.append((k.target, k.value))
    paare = []
    for ziel, wert in zuweisungen:
        if isinstance(ziel, (ast.Tuple, ast.List)) and isinstance(wert, (ast.Tuple, ast.List)) \
                and len(ziel.elts) == len(wert.elts):
            paare += list(zip(ziel.elts, wert.elts))
        else:
            paare.append((ziel, wert))
    for ziel, wert in paare:
        if isinstance(ziel, ast.Name) and isinstance(wert, ast.Lambda):
            defs[ziel.id] = wert
        elif isinstance(wert, ast.Call):
            # job = self._x(…) und job, config, rate = self._x(…)
            ziele = ziel.elts if isinstance(ziel, (ast.Tuple, ast.List)) else [ziel]
            for z in ziele:
                if isinstance(z, ast.Name):
                    fabriken[z.id] = wert
    while True:
        neu = {z.id for z, w in paare if isinstance(z, ast.Name)
               and isinstance(w, ast.Name) and w.id in alias} - alias
        if not neu:
            break
        alias |= neu
    if "self" in _parameter(fn):
        alias.discard("self")
    return defs, alias - {"self"}, fabriken


def _hat_job(fn):
    """True, wenn ``fn`` eine verschachtelte Job-Funktion (Name job oder
    Parameter progress_cb, cancel, log_cb) enthält."""
    return any(isinstance(k, _FUNKTION)
               and (k.name == "job" or _JOB_PARAMETER <= _parameter(k))
               for k in _eigene_knoten(fn))


def _methoden(baeume):
    """Methodenname -> [(rel, Funktion)] über alle Klassen der Dateien."""
    out = collections.defaultdict(list)
    for rel, baum in baeume:
        for k in baum.body:
            if isinstance(k, ast.ClassDef):
                for m in k.body:
                    if isinstance(m, _FUNKTION):
                        out[m.name].append((rel, m))
    return out


def _job_ausdruck(aufruf):
    """Der Job-Ausdruck eines Aufrufs von _start_worker oder Worker, sonst None."""
    f = aufruf.func
    name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else None
    if name == "_start_worker":
        stelle, schluessel = 1, ("job",)
    elif name == "Worker":
        stelle, schluessel = 0, ("fn", "job")
    else:
        return None
    if len(aufruf.args) > stelle and not isinstance(aufruf.args[stelle], ast.Starred):
        return aufruf.args[stelle]
    for kw in aufruf.keywords:
        if kw.arg in schluessel:
            return kw.value
    return None


def _self_zeilen(fn, alias):
    """Zeilen, in denen ``fn`` (Körper und Vorgabewerte) self oder einen Alias benutzt."""
    if "self" in _parameter(fn):
        return [], set()
    a = fn.args
    teile = (list(fn.body) if isinstance(fn.body, list) else [fn.body]) \
        + list(a.defaults) + [d for d in a.kw_defaults if d is not None]
    eigene = _parameter(fn)
    namen = [n for t in teile for n in ast.walk(t) if isinstance(n, ast.Name)]
    zeilen = sorted({n.lineno for n in namen
                     if n.id == "self" or (n.id in alias and n.id not in eigene)})
    benutzt = {n.id for n in namen if isinstance(n.ctx, ast.Load) and n.id not in eigene}
    return zeilen, benutzt


def _jobs_einer_datei(rel, baum, methoden):
    """[(beschreibung, [treffer])] je Job-Funktion der Datei, dazu Aufruffehler.

    ``methoden`` (aus _methoden) löst Jobs aus einer Fabrikmethode auf."""
    oben = {k.name: k for k in baum.body if isinstance(k, _FUNKTION)}
    importiert = set()
    for k in baum.body:
        if isinstance(k, (ast.Import, ast.ImportFrom)):
            importiert |= {(a.asname or a.name).split(".")[0] for a in k.names}
    jobs = {}           # id(knoten) -> (knoten, kette, wie)
    fehler = []

    def merke(knoten, kette, wie):
        jobs.setdefault(id(knoten), (knoten, kette, wie))

    def aufloesen(name, kette):
        for fn in reversed(kette):
            if name in _parameter(fn):
                return "parameter", None
            defs = _bereich(fn)[0]
            if name in defs:
                return "lokal", defs[name]
        if name in oben:
            return "modul", oben[name]
        if name in importiert:
            return "import", None
        return None, None

    def besuche(knoten, kette):
        for kind in ast.iter_child_nodes(knoten):
            neue_kette = kette + [kind] if isinstance(kind, _FUNKTION + (ast.Lambda,)) else kette
            if isinstance(kind, _FUNKTION) and kette and _JOB_PARAMETER <= _parameter(kind):
                merke(kind, kette, "Parameter progress_cb, cancel, log_cb")
            if isinstance(kind, _FUNKTION) and kette and kind.name == "job":
                merke(kind, kette, "Name job")
            if isinstance(kind, ast.Call):
                ausdruck = _job_ausdruck(kind)
                if ausdruck is not None:
                    pruefe_aufruf(kind, ausdruck, kette)
            besuche(kind, neue_kette)

    def zugewiesen(name, kette):
        """Der Aufruf, dem der lokale Name zugewiesen ist (job = f(…), auch
        job, config, rate = f(…)); None, wenn der Name zuerst als Parameter oder
        lokale Funktion gilt oder keinem Aufruf zugewiesen ist."""
        for fn in reversed(kette):
            if name in _parameter(fn) or name in _bereich(fn)[0]:
                return None
            if name in _bereich(fn)[2]:
                return _bereich(fn)[2][name]
        return None

    def fabrik_aufruf(ausdruck, kette, alias):
        """Der Aufruf self._x(…), aus dem der Job stammt: direkt, über einen Index
        (self._x(…)[0]) oder über einen lokalen Namen (job = self._x(…), auch
        job, config, rate = self._x(…)); sonst None."""
        k = ausdruck
        while isinstance(k, ast.Subscript):
            k = k.value
        if isinstance(k, ast.Name):
            k = zugewiesen(k.id, kette)
        if (isinstance(k, ast.Call) and isinstance(k.func, ast.Attribute)
                and isinstance(k.func.value, ast.Name) and k.func.value.id in alias):
            return k
        return None

    def pruefe_aufruf(aufruf, ausdruck, kette):
        alias = set()
        for fn in kette:
            alias |= _bereich(fn)[1] | ({"self"} if "self" in _parameter(fn) else set())
        wo = f"{rel}:{aufruf.lineno}"
        fabrik = fabrik_aufruf(ausdruck, kette, alias)
        if fabrik is not None:
            # Die Fabrik läuft im GUI-Thread; ihre Job-Funktionen zählen und
            # prüft der Besuch der Klasse, in der sie steht.
            name = fabrik.func.attr
            kandidaten = methoden.get(name, [])
            if not kandidaten:
                fehler.append(f"{wo} Job aus {ast.unparse(fabrik.func)}(…): "
                              f"Methode {name} in keiner Klasse gefunden")
            elif not any(_hat_job(m) for _, m in kandidaten):
                orte = ", ".join(f"{r}:{m.lineno}" for r, m in kandidaten)
                fehler.append(f"{wo} Job aus {ast.unparse(fabrik.func)}(…): "
                              f"Methode {name} ({orte}) enthält keine Job-Funktion")
            return
        if isinstance(ausdruck, ast.Name):
            # job = _fabrik(…) mit einer Modulfunktion der Datei gilt wie
            # _start_worker(…, _fabrik(…), …): weiter unten als Ausdruck
            quelle = zugewiesen(ausdruck.id, kette)
            if (quelle is not None and isinstance(quelle.func, ast.Name)
                    and aufloesen(quelle.func.id, kette)[0] == "modul"):
                ausdruck = quelle
        if isinstance(ausdruck, ast.Lambda):
            merke(ausdruck, kette, f"Lambda an {wo}")
        elif isinstance(ausdruck, ast.Name):
            art, ziel = aufloesen(ausdruck.id, kette)
            if art == "parameter":
                return          # durchgereicht, etwa Worker(job, …) in _start_worker
            if ziel is not None:
                merke(ziel, kette if art == "lokal" else [], f"an {wo} übergeben")
            elif art != "import":
                fehler.append(f"{wo} Job {ausdruck.id} nicht aufzulösen")
        else:
            zeilen = sorted({n.lineno for n in ast.walk(ausdruck)
                             if isinstance(n, ast.Name) and n.id in alias})
            if zeilen:
                fehler.append(f"{wo} Job-Ausdruck {ast.unparse(ausdruck)[:60]} benutzt self "
                              f"(Zeile {', '.join(map(str, zeilen))})")
                return
            # etwa functools.partial(rechne, …): die Funktion darin ist der Job
            innen = False
            for n in ast.walk(ausdruck):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                    art, ziel = aufloesen(n.id, kette)
                    if ziel is not None:
                        innen = True
                        merke(ziel, kette if art == "lokal" else [], f"in {wo} übergeben")
            if not innen:
                merke(ausdruck, kette, f"Ausdruck an {wo}")

    besuche(baum, [])
    ergebnis = []
    for knoten, kette, wie in sorted(jobs.values(), key=lambda j: (j[0].lineno, j[0].col_offset)):
        name = getattr(knoten, "name", "lambda")
        if not isinstance(knoten, _FUNKTION + (ast.Lambda,)):
            ergebnis.append((f"{rel}:{knoten.lineno} {name}", []))
            continue
        alias = set()
        for fn in kette:
            alias |= _bereich(fn)[1]
        treffer = []
        # die Funktion selbst und jede Nachbarfunktion des Handlers, die sie
        # (auch über andere Nachbarn) benutzt; sie laufen alle im Arbeitsthread
        offen, gesehen = [(knoten, None)], set()
        while offen:
            fn, ueber = offen.pop()
            if id(fn) in gesehen:
                continue
            gesehen.add(id(fn))
            zeilen, benutzt = _self_zeilen(fn, alias)
            if zeilen:
                treffer.append(f"{'über ' + ueber + ' ' if ueber else ''}"
                               f"Zeile {', '.join(map(str, zeilen))}")
            for n in sorted(benutzt):
                for k in reversed(kette):
                    defs = _bereich(k)[0]
                    if n in defs:
                        if defs[n] is not knoten:
                            offen.append((defs[n], f"{n} ({rel}:{defs[n].lineno})"))
                        break
        ergebnis.append((f"{rel}:{knoten.lineno} {name}", treffer))
    return ergebnis, fehler


def stufe_threadregel():
    treffer, n_job = [], 0
    baeume = {}
    for rel in py_dateien("ui"):
        pfad = os.path.join(WURZEL, rel)
        with open(pfad, encoding="utf-8") as fh:
            baeume[rel] = ast.parse(fh.read(), pfad)
    # Fabrikmethoden sucht die Regel bei MainWindow und den Mixins über alle
    # ihre Dateien, sonst in der Klasse derselben Datei.
    fenster = [(r, b) for r, b in baeume.items()
               if r == os.path.join("ui", "main_window.py")
               or os.path.dirname(r) == os.path.join("ui", "fenster")]
    methoden_fenster = _methoden(fenster)
    for rel, baum in baeume.items():
        if any(r == rel for r, _ in fenster):
            methoden = methoden_fenster
        else:
            methoden = _methoden([(rel, baum)])
        jobs, fehler = _jobs_einer_datei(rel, baum, methoden)
        n_job += len(jobs)
        treffer += fehler
        for wo, t in jobs:
            if t:
                treffer.append(f"{wo} benutzt self ({'; '.join(t)})")
    for t in treffer:
        print(f"  {t}")
    erwartet = lade_erwartet().get("threadregel_jobs")
    soll = JOBS_VORHER if erwartet is None else int(erwartet)
    zu_wenig = n_job < soll
    print(f"  {n_job} Job-Funktionen (vorher {JOBS_VORHER}"
          + (f", erwartet {soll}" if erwartet is not None else "")
          + f"), {len(treffer)} Verstöße")
    if zu_wenig:
        print(f"  FEHLER: weniger Job-Funktionen als {soll} — ein Job wird nicht mehr "
              "erkannt oder ist entfallen (dann --erwartet threadregel_jobs)")
    return not treffer and not zu_wenig


# ------------------------------------------------------------ Selbsttests

def _hat_main(pfad):
    with open(pfad, encoding="utf-8") as fh:
        baum = ast.parse(fh.read(), pfad)
    for k in baum.body:
        if (isinstance(k, ast.If) and isinstance(k.test, ast.Compare)
                and isinstance(k.test.left, ast.Name) and k.test.left.id == "__name__"
                and len(k.test.comparators) == 1
                and isinstance(k.test.comparators[0], ast.Constant)
                and k.test.comparators[0].value == "__main__"):
            return True
    return False


def module_mit_main(paket):
    out = []
    for rel in py_dateien(paket):
        if os.path.basename(rel) == "__init__.py":
            continue
        if _hat_main(os.path.join(WURZEL, rel)):
            out.append(rel[:-3].replace(os.sep, "."))
    return out


def plan(stufe):
    """[(name, aufruf, umgebung_extra)] und [(name, status, grund)] einer Stufe."""
    laeufe, aus = [], []
    if stufe == "selbsttests":
        kandidaten = module_mit_main("core")
    elif stufe == "ui":
        kandidaten = module_mit_main("ui")
    else:
        kandidaten = []
    for name in kandidaten:
        if name in AUSGENOMMEN:
            st, status, grund = AUSGENOMMEN[name]
            if status is not None:
                aus.append((name, status, grund))
            if st != stufe or status is not None:
                continue
        if stufe == "selbsttests":
            laeufe.append((name, ["python3", "-m", name], {}))
        elif name == "ui.cloud_view":
            laeufe.append((name, ["xvfb-run", "-a", "python3", "-m", name], {}))
        else:
            laeufe.append((name, ["python3", "-m", name], {"QT_QPA_PLATFORM": "offscreen"}))
    if stufe == "ui":
        skript = "scripts/test_cloud_view_steuerung.py"
        if os.path.isfile(os.path.join(WURZEL, skript)):
            laeufe.append((skript, ["xvfb-run", "-a", "python3", skript], {}))
    if stufe == "gpu" and os.path.isfile(os.path.join(WURZEL, "core", "colorizer_gpu.py")):
        laeufe.append(("core.colorizer_gpu", ["python3", "-m", "core.colorizer_gpu"], {}))
    if stufe == "ui-zeit" and os.path.isfile(os.path.join(WURZEL, "ui", "pano_view.py")):
        laeufe.append(("ui.pano_view", ["python3", "-m", "ui.pano_view"],
                       {"QT_QPA_PLATFORM": "offscreen"}))
    return laeufe, aus


def aufruf_text(aufruf, extra):
    vorn = " ".join(f"{k}={v}" for k, v in sorted(extra.items()))
    return (vorn + " " if vorn else "") + " ".join(aufruf)


def lies_selbsttests():
    pfad = os.path.join(VORHER, "selbsttests.txt")
    if not os.path.isfile(pfad):
        return None
    out = {}
    with open(pfad, encoding="utf-8") as fh:
        for z in fh:
            if not z.strip() or z.startswith("#"):
                continue
            teile = z.rstrip("\n").split("\t")
            out[teile[0]] = {"stufe": teile[1], "status": teile[2],
                             "info": teile[3] if len(teile) > 3 else ""}
    return out


def _grund(ausgabe, code):
    zeilen = [z.strip() for z in neutral(ausgabe).splitlines() if z.strip()]
    kern = [z for z in zeilen if re.search(r"Error|Exception|assert|FEHL|Traceback", z)]
    letzte = (kern or zeilen or ["keine Ausgabe"])[-1]
    return f"Exit {code}: {letzte[:200]}"


def gpu_vorbedingung():
    meta = os.path.join(basis.ECHTER_CACHE, GPU_PROJEKT, "recording", "meta.json")
    if not os.path.isfile(meta):
        return f"{neutral(meta)} fehlt"
    with open(meta, encoding="utf-8") as fh:
        if "gravity_level" not in json.load(fh):
            return (f"{neutral(meta)} enthält kein gravity_level — Recording.load "
                    "schriebe es in den echten Cache")
    return None


def modtests_stand():
    return set(os.listdir(MODTESTS)) if os.path.isdir(MODTESTS) else set()


def stufe_laeufe(stufe):
    """Eine Selbsttest-Stufe unter der Sperre; danach neue Ausgabeordner räumen."""
    os.makedirs(basis.ARBEIT_WURZEL, exist_ok=True)
    sperre = os.open(basis.ARBEIT_WURZEL, os.O_RDONLY)
    try:
        try:
            fcntl.flock(sperre, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("  warte auf einen anderen Aufruf, der gerade Selbsttests fährt …",
                  flush=True)
            fcntl.flock(sperre, fcntl.LOCK_EX)
        vorher = modtests_stand()
        # nur Ordner der hier gestarteten Module (Konvention
        # /tmp/super360_modtests/<letzter Teil des Modulnamens>)
        eigene = {name.rsplit(".", 1)[-1] for name, _, _ in plan(stufe)[0]
                  if not name.endswith(".py")}
        try:
            return _stufe_laeufe(stufe)
        finally:
            neu = sorted((modtests_stand() - vorher - {"umbau"}) & eigene)
            if neu and not BEHALTEN:
                for n in neu:
                    pfad = os.path.join(MODTESTS, n)
                    if os.path.isdir(pfad) and not os.path.islink(pfad):
                        shutil.rmtree(pfad, ignore_errors=True)
                print(f"  Ausgaben der Selbsttests entfernt: {MODTESTS}/"
                      f"{{{','.join(neu)}}}")
    finally:
        os.close(sperre)


def _stufe_laeufe(stufe):
    if stufe == "gpu":
        hindernis = gpu_vorbedingung()
        if hindernis:
            print(f"  nicht gestartet: {hindernis}")
            return False
    laeufe, aus = plan(stufe)
    vorher = None if SCHREIBE else lies_selbsttests()
    erwartet = lade_erwartet()
    logs = os.path.join(ARBEIT, "logs")
    os.makedirs(logs, exist_ok=True)
    ergebnis = []           # (name, status, info)
    ok = True
    for name, aufruf, extra in laeufe:
        t0 = time.monotonic()
        try:
            r = subprocess.run(aufruf, cwd=WURZEL, env=kind_umgebung(**extra),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, errors="replace",
                               timeout=ZEITLIMIT.get(name, 900))
            code, ausgabe = r.returncode, r.stdout
        except subprocess.TimeoutExpired as exc:
            code = "Zeitlimit"
            ausgabe = (exc.stdout or b"").decode(errors="replace") if isinstance(
                exc.stdout, bytes) else (exc.stdout or "")
        dauer = time.monotonic() - t0
        with open(os.path.join(logs, name.replace("/", "_") + ".txt"), "w",
                  encoding="utf-8") as fh:
            fh.write(ausgabe)
        if code == 0:
            ergebnis.append((name, "bestanden", aufruf_text(aufruf, extra)))
            print(f"  bestanden   {name}  ({dauer:.0f} s)")
            continue
        grund = _grund(ausgabe, code)
        ergebnis.append((name, "nicht lauffähig", grund))
        war = (vorher or {}).get(name)
        if SCHREIBE:
            print(f"  NICHT LAUFFÄHIG {name}  ({dauer:.0f} s): {grund}")
        elif war and war["status"] == "nicht lauffähig":
            print(f"  wie vorher nicht lauffähig {name}: {grund}")
        else:
            ok = False
            print(f"  FEHLER      {name}  ({dauer:.0f} s): {grund}")
            for z in neutral(ausgabe).splitlines()[-12:]:
                print(f"      | {z}")
    for name, status, grund in aus:
        ergebnis.append((name, status, grund))
        print(f"  übergangen  {name}: {grund}")
    if SCHREIBE:
        with open(os.path.join(ARBEIT, f"selbsttests_{stufe}.tsv"), "w",
                  encoding="utf-8") as fh:
            for name, status, info in ergebnis:
                fh.write(f"{name}\t{stufe}\t{status}\t{info}\n")
    elif vorher is None:
        print("  vorher/selbsttests.txt fehlt — Vergleich entfällt, alle müssen bestehen.")
    else:
        jetzt = {n for n, _, _ in ergebnis}
        frei = set(erwartet.get("selbsttests_entfallen", []))
        for name, w in sorted(vorher.items()):
            if w["stufe"] != stufe or name in jetzt:
                continue
            if name in frei:
                print(f"  entfallen (erwartet) {name}")
            else:
                ok = False
                print(f"  FEHLER: {name} stand im Vorher-Stand ({w['status']}), "
                      "jetzt nicht mehr gefunden")
        neu = sorted(n for n in jetzt if n not in vorher)
        if neu:
            print(f"  neu gegenüber vorher: {', '.join(neu)}")
    n_ok = sum(1 for _, s, _ in ergebnis if s == "bestanden")
    print(f"  {len(laeufe)} gestartet, {n_ok} bestanden, {len(aus)} übergangen")
    return ok


def selbsttests_schreiben():
    teile = []
    for stufe in ("selbsttests", "gpu", "ui", "ui-zeit"):
        pfad = os.path.join(ARBEIT, f"selbsttests_{stufe}.tsv")
        if not os.path.isfile(pfad):
            raise SystemExit(f"Ergebnis der Stufe {stufe} fehlt.")
        with open(pfad, encoding="utf-8") as fh:
            teile += sorted(fh.readlines())
    with open(os.path.join(VORHER, "selbsttests.txt"), "w", encoding="utf-8") as fh:
        fh.write("# Selbsttests vor dem Umbau, je Zeile: Modul, Stufe von "
                 "pruefe_alles.sh, Ergebnis, Aufruf oder Grund (Tab-getrennt).\n"
                 "# bestanden = mit dem genannten Aufruf aus der Repo-Wurzel "
                 "lauffähig (Module über python3 -m), Exit 0.\n"
                 "# nicht lauffähig = ausgenommen oder fehlgeschlagen, mit Grund.\n")
        fh.writelines(teile)
    print(f"  {len(teile)} Selbsttests -> vorher/selbsttests.txt")


# ------------------------------------------------------------------ autotest

_ZEIT = re.compile(r"^\[\d{1,2}:\d{2}:\d{2}\]\s?")
_DAUER = re.compile(r"(?<!Dauer )(?<![\w.,])\d+(?:[.,]\d+)?\s?(?:ms|s|min)\b")


def protokoll_neutral(text, ausgabe):
    zeilen = []
    for z in text.splitlines():
        z = _ZEIT.sub("", z)
        z = z.replace(ausgabe, "<ausgabe>")
        z = neutral(z)
        z = _DAUER.sub("<dauer>", z)
        zeilen.append(z.rstrip())
    return zeilen


def summary_neutral(d):
    d = dict(d)
    d.pop("log_lines", None)
    return json.loads(neutral(json.dumps(d, ensure_ascii=False)))


def stufe_autotest():
    ordner = os.path.join(ARBEIT, "autotest")
    os.makedirs(ordner, exist_ok=True)
    kopie = basis.cache_kopie(ordner)
    meta_pfad = os.path.join(kopie.projekt, "recording", "meta.json")
    expl_pfad = os.path.join(kopie.projekt, "exploration.json")

    def im_cache():
        with open(meta_pfad, encoding="utf-8") as fh:
            meta = json.load(fh)
        return meta, "gravity_level" in meta, os.path.isfile(expl_pfad)

    meta, lot, expl = im_cache()
    n_scans_meta = meta.get("n_scans")
    print(f"  Kopie des seg0-Projekts: {neutral(kopie.projekt)} ({n_scans_meta} Scans "
          f"laut meta.json; Lotrechte {'schon' if lot else 'nicht'} darin, "
          f"exploration.json {'schon' if expl else 'nicht'} da)")
    ergebnisse, ok = [], True
    for lauf in (1, 2):
        aus = os.path.join(ordner, f"lauf{lauf}")
        env = kind_umgebung(SUPER360_CACHE_ROOT=kopie.cache_root,
                            SUPER360_AUTOTEST=kopie.bag, SUPER360_AUTOTEST_OUT=aus)
        t0 = time.monotonic()
        try:
            r = subprocess.run(["xvfb-run", "-a", "python3", "app.py"], cwd=WURZEL,
                               env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, errors="replace", timeout=900)
            code, ausgabe = r.returncode, r.stdout
        except subprocess.TimeoutExpired:
            code, ausgabe = "Zeitlimit", ""
        with open(os.path.join(ordner, f"lauf{lauf}.txt"), "w", encoding="utf-8") as fh:
            fh.write(ausgabe)
        summary_pfad = os.path.join(aus, "summary.json")
        if code != 0 or not os.path.isfile(summary_pfad):
            print(f"  FEHLER Lauf {lauf}: Exit {code}")
            for z in neutral(ausgabe).splitlines()[-12:]:
                print(f"      | {z}")
            return False
        with open(summary_pfad, encoding="utf-8") as fh:
            summary = summary_neutral(json.load(fh))
        with open(os.path.join(aus, "protokoll.txt"), encoding="utf-8") as fh:
            protokoll = protokoll_neutral(fh.read(), aus)
        print(f"  Lauf {lauf}: Exit 0 in {time.monotonic() - t0:.0f} s, "
              f"{summary.get('n_scans')} Scans, {summary.get('pano_frames')} Pano-Frames, "
              f"{len(protokoll)} Protokollzeilen")
        if lauf == 1:
            _, lot, expl = im_cache()
            if not (lot and expl):
                ok = False
                print("  FEHLER Lauf 1: hat "
                      + " und ".join(t for t, da in (("gravity_level", lot),
                                                     ("exploration.json", expl)) if not da)
                      + " nicht in die Kopie geschrieben — Lauf 2 nähme nicht den Cache")
        if summary.get("n_scans") != n_scans_meta:
            ok = False
            print(f"  FEHLER Lauf {lauf}: n_scans {summary.get('n_scans')} statt "
                  f"{n_scans_meta} aus meta.json der Kopie")
        ergebnisse.append((summary, protokoll))
    namen = {1: ("summary.json", "protokoll.txt"),
             2: ("summary_lauf2.json", "protokoll_lauf2.txt")}
    ziel = os.path.join(VORHER, "autotest")
    if SCHREIBE:
        os.makedirs(ziel, exist_ok=True)
        for lauf, (summary, protokoll) in zip((1, 2), ergebnisse):
            s_name, p_name = namen[lauf]
            with open(os.path.join(ziel, s_name), "w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2, ensure_ascii=False, sort_keys=True)
                fh.write("\n")
            with open(os.path.join(ziel, p_name), "w", encoding="utf-8") as fh:
                fh.write("".join(z + "\n" for z in protokoll))
        print("  -> vorher/autotest/ (Lauf 1: summary.json, protokoll.txt; "
              "Lauf 2: summary_lauf2.json, protokoll_lauf2.txt)")
        return ok
    erwartet = lade_erwartet()
    frei_s = set(erwartet.get("autotest_summary", []))
    frei_p = [re.compile(m) for m in erwartet.get("autotest_protokoll", [])]
    for lauf, (summary, protokoll) in zip((1, 2), ergebnisse):
        s_name, p_name = namen[lauf]
        try:
            with open(os.path.join(ziel, s_name), encoding="utf-8") as fh:
                alt_s = json.load(fh)
            with open(os.path.join(ziel, p_name), encoding="utf-8") as fh:
                alt_p = fh.read().splitlines()
        except FileNotFoundError as exc:
            print(f"  Vorher-Stand fehlt ({os.path.basename(exc.filename)}) — "
                  "erst --schreibe-vorher.")
            return False
        for k in sorted(set(alt_s) | set(summary)):
            if alt_s.get(k) == summary.get(k):
                continue
            art = "erlaubt" if k in frei_s else "ABWEICHUNG"
            ok = ok and k in frei_s
            print(f"  {art} Lauf {lauf} summary.{k}: vorher {alt_s.get(k)!r}, "
                  f"jetzt {summary.get(k)!r}")
        a = [z for z in alt_p if not any(m.search(z) for m in frei_p)]
        b = [z for z in protokoll if not any(m.search(z) for m in frei_p)]
        if a != b:
            ok = False
            print(f"  ABWEICHUNG Lauf {lauf} protokoll.txt:")
            for z in difflib.unified_diff(a, b, "vorher", "jetzt", lineterm="", n=1):
                print(f"      {z}")
        else:
            print(f"  Lauf {lauf}: summary und Protokoll wie vorher"
                  + (f" ({len(alt_p) - len(a)} Zeilen erlaubt abweichend)"
                     if len(alt_p) != len(a) or len(protokoll) != len(b) else ""))
    return ok


# ----------------------------------------------------------------- echtcache

def echtcache_stand():
    """Größe und Änderungszeit jeder Datei im echten Cache, dazu die Projektliste."""
    dateien = {}
    if os.path.isdir(basis.ECHTER_CACHE):
        for ordner, unter, namen in os.walk(basis.ECHTER_CACHE):
            unter.sort()
            for n in namen + [u for u in unter if os.path.islink(os.path.join(ordner, u))]:
                pfad = os.path.join(ordner, n)
                try:
                    st = os.lstat(pfad)
                except FileNotFoundError:
                    continue
                dateien[os.path.relpath(pfad, basis.ECHTER_CACHE)] = [st.st_size, st.st_mtime_ns]
    return {"dateien": dateien,
            "projekte": sorted(os.listdir(basis.ECHTER_CACHE))
            if os.path.isdir(basis.ECHTER_CACHE) else []}


def stufe_echtcache():
    with open(os.path.join(ARBEIT, "echtcache_start.json"), encoding="utf-8") as fh:
        alt = json.load(fh)
    neu = echtcache_stand()
    ok = True
    dazu = sorted(set(neu["projekte"]) - set(alt["projekte"]))
    weg = sorted(set(alt["projekte"]) - set(neu["projekte"]))
    if dazu or weg:
        ok = False
        print(f"  Projekte im echten Cache: neu {dazu}, weg {weg}")
    a, b = alt["dateien"], neu["dateien"]
    for art, liste in (("GEÄNDERT", [n for n in sorted(set(a) & set(b)) if a[n] != b[n]]),
                       ("NEU", sorted(set(b) - set(a))), ("WEG", sorted(set(a) - set(b)))):
        if liste:
            ok = False
            for n in liste[:20]:
                print(f"  {art}: {n}")
            if len(liste) > 20:
                print(f"  … und {len(liste) - 20} weitere ({art})")
    n_meta = sum(1 for n in b if n.endswith(os.path.join("recording", "meta.json")))
    print(f"  {len(b)} Dateien, darunter {n_meta} recording/meta.json, und {len(neu['projekte'])} "
          f"Projekte im echten Cache {'unverändert' if ok else 'VERÄNDERT'} seit dem Start "
          "dieses Aufrufs (Größe und Änderungszeit)")
    return ok


# ------------------------------------------------------------------- Rahmen

def main(argv):
    befehl = argv[0]
    if befehl == "start":
        with open(os.path.join(ARBEIT, "echtcache_start.json"), "w", encoding="utf-8") as fh:
            json.dump(echtcache_stand(), fh)
        lade_erwartet()
        return 0
    if befehl == "selbsttests-schreiben":
        selbsttests_schreiben()
        return 0
    stufen = {"kompilieren": stufe_kompilieren, "pyflakes": stufe_pyflakes,
              "import": stufe_import, "namen": stufe_namen,
              "threadregel": stufe_threadregel, "autotest": stufe_autotest,
              "echtcache": stufe_echtcache}
    if befehl in stufen:
        return 0 if stufen[befehl]() else 1
    if befehl in ("selbsttests", "gpu", "ui", "ui-zeit"):
        return 0 if stufe_laeufe(befehl) else 1
    print(f"unbekannter Befehl {befehl}", file=sys.stderr)
    return 2


sys.exit(main(sys.argv[1:]))
PY

helfer() {
    PRUEFE_WERKZEUG="$WERKZEUG" PRUEFE_WURZEL="$WURZEL" PRUEFE_ARBEIT="$ARBEIT" \
    PRUEFE_VORHER="$ZIEL_VORHER" PRUEFE_SCHREIBE="$SCHREIBE" PRUEFE_ERWARTET="$ERWARTET" \
    PRUEFE_BEHALTEN="$BEHALTEN" python3 "$ARBEIT/pruefe_helfer.py" "$@"
}

# ------------------------------------------------------------ Arbeitsordner

ARBEIT="$(python3 "$WERKZEUG/basis.py" --arbeitsordner)" || { echo "Kein Arbeitsordner." >&2; exit 2; }
ZIEL_VORHER="$VORHER"
aufraeumen() {
    if [ "$BEHALTEN" = 1 ]; then
        echo "Arbeitsordner bleibt stehen: $ARBEIT"
    else
        python3 "$WERKZEUG/basis.py" --raeume "$ARBEIT"
    fi
}
trap aufraeumen EXIT
trap 'exit 130' INT TERM
printf '%s\n' "$HELFER" > "$ARBEIT/pruefe_helfer.py"

if [ -n "$TAG" ]; then
    WURZEL="$ARBEIT/stand_$TAG"
    mkdir -p "$WURZEL"
    if ! git -C "$ARBEITSBAUM" archive "$TAG" | tar -x -C "$WURZEL"; then
        echo "git archive $TAG ist fehlgeschlagen." >&2; exit 2
    fi
fi
WURZEL="$(cd "${WURZEL:-$ARBEITSBAUM}" 2>/dev/null && pwd)" || { echo "--wurzel: Ordner fehlt." >&2; exit 2; }
for teil in core ui app.py; do
    [ -e "$WURZEL/$teil" ] || { echo "--wurzel $WURZEL: '$teil' fehlt darin." >&2; exit 2; }
done

if [ "$SCHREIBE" = 1 ]; then
    if [ "$WURZEL" = "$ARBEITSBAUM" ]; then
        if ! git -C "$ARBEITSBAUM" diff --quiet vor-umbau -- app.py core ui scripts ':(exclude)scripts/umbau' \
           || [ -n "$(git -C "$ARBEITSBAUM" ls-files --others --exclude-standard -- app.py core ui scripts ':(exclude)scripts/umbau')" ]; then
            echo "Der Arbeitsbaum weicht in app.py, core, ui oder scripts von vor-umbau ab;" >&2
            echo "den Vorher-Stand dann mit --tag vor-umbau schreiben." >&2
            exit 2
        fi
    fi
    if [ "$ERSETZEN" != 1 ]; then
        for ziel in "$VORHER/pyflakes.txt" "$VORHER/selbsttests.txt" "$VORHER/autotest"; do
            if [ -e "$ziel" ]; then
                echo "${ziel#"$ARBEITSBAUM"/} gibt es schon — nur mit --ersetzen." >&2
                exit 2
            fi
        done
    fi
    # erst im Arbeitsordner sammeln, übernommen wird nur ein vollständiger Stand
    ZIEL_VORHER="$ARBEIT/vorher_neu"
    mkdir -p "$ZIEL_VORHER"
fi

helfer start || exit 2

echo "Wurzel: $( [ "$WURZEL" = "$ARBEITSBAUM" ] && echo Arbeitsbaum || echo "${WURZEL/#$ARBEIT/<arbeit>}" )"
[ "$SCHREIBE" = 1 ] && echo "Modus: Vorher-Stand schreiben"

# ------------------------------------------------------------------- Stufen

FEHLER=()
GELAUFEN=0
for stufe in "${ALLE_STUFEN[@]}"; do
    case " ${GEWAEHLT[*]} " in *" $stufe "*) ;; *) continue ;; esac
    echo "== Stufe $stufe"
    t0=$SECONDS
    if helfer "$stufe"; then
        echo "== Stufe $stufe: OK ($((SECONDS - t0)) s)"
    else
        echo "== Stufe $stufe: FEHLER ($((SECONDS - t0)) s)"
        FEHLER+=("$stufe")
    fi
    GELAUFEN=$((GELAUFEN + 1))
done

if [ "$SCHREIBE" = 1 ] && [ ${#FEHLER[@]} -eq 0 ]; then
    if helfer selbsttests-schreiben; then
        mkdir -p "$VORHER"
        rm -rf "$VORHER/autotest"
        cp "$ZIEL_VORHER/pyflakes.txt" "$ZIEL_VORHER/selbsttests.txt" "$VORHER/"
        cp -r "$ZIEL_VORHER/autotest" "$VORHER/autotest"
        echo "Vorher-Stand übernommen: scripts/umbau/vorher/{pyflakes.txt,selbsttests.txt,autotest/}"
    else
        FEHLER+=("selbsttests-schreiben")
    fi
elif [ "$SCHREIBE" = 1 ]; then
    echo "Vorher-Stand NICHT übernommen."
fi

if [ ${#FEHLER[@]} -eq 0 ]; then
    echo "pruefe_alles: OK ($GELAUFEN Stufen)"
    exit 0
fi
echo "pruefe_alles: FEHLER in ${FEHLER[*]}"
exit 1
