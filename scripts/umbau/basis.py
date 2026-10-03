#!/usr/bin/env python3
"""Gemeinsame Grundlage der Prüfwerkzeuge für den Umbau.

Alle Werkzeuge in ``scripts/umbau`` holen sich hier dasselbe:

* die Repo-Wurzel (``--wurzel``, Vorgabe: der Arbeitsbaum). Damit laufen die
  Werkzeuge auch gegen einen mit ``git archive vor-umbau`` nach /tmp entpackten
  Stand; geladen werden ``core`` und ``ui`` dann von dort,
* einen eigenen Arbeitsordner je Aufruf unter ``/tmp/super360_modtests/umbau``,
  der am Ende geräumt wird. Ein Werkzeug, das Kindprozesse startet, gibt ihnen
  mit ``SUPER360_UMBAU_ORDNER`` seinen eigenen Ordner mit: deren Arbeitsordner
  entstehen dann darin und verschwinden mit ihm, auch wenn ein Kind abstürzt,
* eine frische Kopie des seg0-Projekts als Cache. ``SUPER360_CACHE_ROOT`` zeigt
  ab :func:`wurzel_setzen` immer in den Arbeitsordner, nie auf den echten
  Cache: ``Recording.load`` schreibt ``gravity_level`` in die meta.json,
* Hash, Umgebungsstempel, Fensteraufbau, abgefangene Dialoge und das Ersetzen
  eines Namens in allen geladenen Modulen.

Fensteraufbau: ``MainWindow()`` lässt sich unter ``xvfb-run -a`` bauen und
zeigen (samt VTK-Ansicht); ein Rückfall auf ``DISPLAY=:0`` ist nicht nötig.
Ohne xvfb-run erscheinen die Fenster auf dem Bildschirm.

Ende 5 unter xvfb-run: Meldet ein Aufruf über ``xvfb-run -a`` am Schluss
»xvfb-run: error: problem while cleaning up temporary directory« und endet mit
5, dann stammt die 5 von xvfb-run, nicht vom Werkzeug. Es gilt die
Ergebniszeile des Werkzeugs darüber; sein eigenes Ende war 1 oder 2, nie 0 —
ein bestandener Lauf wird so nicht rot. Danach läuft ein verwaister Xvfb
weiter (``pgrep -a Xvfb``, der mit dem gelöschten ``/tmp/xvfb-run.*``); ihn
beenden. Ursache: Endet der letzte X-Client, setzt sich Xvfb zurück und
schickt xvfb-run ein SIGUSR1; trifft es ein, während xvfb-run schon aufräumt,
hält dessen Shell das ``rm`` für gescheitert. Das betrifft jeden Aufruf über
``xvfb-run -a``, der nicht mit 0 endet. :func:`qt_app_schliessen` schließt die
X-Verbindung vor dem Prozessende und wartet, bis das Signal durch ist; die
Ablauf-Probe ruft es, die übrigen Werkzeuge nicht.

Aufrufe::

    xvfb-run -a python3 scripts/umbau/basis.py --selbsttest [--wurzel <dir>]
    python3 scripts/umbau/basis.py --arbeitsordner        # Pfad; bleibt stehen
    python3 scripts/umbau/basis.py --cache-kopie <ordner> # Cache-Wurzel
    python3 scripts/umbau/basis.py --raeume <ordner>
    python3 scripts/umbau/basis.py --stempel

Die drei mittleren sind für Shell-Skripte gedacht, die ihren Ordner über
mehrere Prozesse hinweg brauchen und ihn selbst wieder räumen.
"""
from __future__ import annotations

import argparse
import atexit
import collections
import gc
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

# Kein __pycache__ in den geprüften Bäumen hinterlassen.
sys.dont_write_bytecode = True

WERKZEUG_ORDNER = os.path.dirname(os.path.abspath(__file__))
ARBEITSBAUM = os.path.dirname(os.path.dirname(WERKZEUG_ORDNER))
VORHER_ORDNER = os.path.join(WERKZEUG_ORDNER, "vorher")
ARBEIT_WURZEL = "/tmp/super360_modtests/umbau"
VORHER_TAG = "vor-umbau"
ELTERN_VARIABLE = "SUPER360_UMBAU_ORDNER"
#: So lange wartet :func:`qt_app_schliessen` auf das Zurücksetzen von Xvfb;
#: gemessen sind 10 bis 25 ms nach dem Ende des letzten Clients.
XVFB_RUECKSETZEN_S = 0.5

#: Das Bag und der echte Cache, aus dem nur gelesen (kopiert) wird. Der Ort
#: steht fest und folgt bewusst NICHT der Umgebungsvariable.
SEG0_BAG = os.path.expanduser("~/RosBagSuper_Gui/rosbag_2026-07-11_15-37-07_seg0")
ECHTER_CACHE = os.path.expanduser("~/RosBagSuper_Gui/rosbag_suite/cache")

_wurzel: str | None = None
_cache: str | None = None
_ordner: list[str] = []          # alle Arbeitsordner dieses Prozesses
_raeumen: list[str] = []         # davon die, die am Ende verschwinden


# ------------------------------------------------------------------ Wurzel

def argumente(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Hängt ``--wurzel`` an den Parser eines Werkzeugs."""
    parser.add_argument("--wurzel", default=None, metavar="DIR",
                        help="Repo-Wurzel, aus der core und ui geladen werden "
                             "(Vorgabe: der Arbeitsbaum)")
    return parser


def wurzel_setzen(pfad: str | None = None) -> str:
    """Legt fest, von wo ``core`` und ``ui`` kommen — vor deren erstem Import.

    Setzt außerdem ``SUPER360_CACHE_ROOT`` auf einen leeren Cache im
    Arbeitsordner und wechselt in die Wurzel (die Selbsttests der Module
    erwarten das). Relative Pfade aus der Kommandozeile vorher absolut machen.
    """
    global _wurzel
    w = os.path.abspath(pfad or ARBEITSBAUM)
    for teil in ("core", "ui"):
        if not os.path.isdir(os.path.join(w, teil)):
            print(f"--wurzel {w}: kein Ordner '{teil}' darin.", file=sys.stderr)
            raise SystemExit(2)
    if _wurzel is not None and _wurzel != w:
        raise RuntimeError(f"Wurzel steht schon auf {_wurzel}; ein Prozess "
                           f"kann core und ui nur von einem Ort laden.")
    for name in ("core", "ui"):
        mod = sys.modules.get(name)
        ort = os.path.abspath(os.path.dirname(getattr(mod, "__file__", "") or
                                              next(iter(getattr(mod, "__path__", [""])), "")
                                              )) if mod is not None else ""
        if mod is not None and ort != os.path.join(w, name):
            raise RuntimeError(f"'{name}' ist schon von {ort} geladen, "
                               f"verlangt ist {w}.")
    _wurzel = w
    weg = {w, ARBEITSBAUM}
    sys.path[:] = [w] + [p for p in sys.path if os.path.abspath(p or ".") not in weg]
    if WERKZEUG_ORDNER not in sys.path:
        sys.path.append(WERKZEUG_ORDNER)
    cache_wurzel()
    os.chdir(w)
    return w


def wurzel() -> str:
    """Die gesetzte Repo-Wurzel (setzt die Vorgabe, falls noch keine steht)."""
    return _wurzel or wurzel_setzen()


def module_laden() -> dict:
    """Importiert jedes Modul aus ``core/`` und ``ui/`` der Wurzel.

    :func:`ueberall_ersetzen` sieht nur geladene Module; was die App erst in
    einer Funktion importiert, wäre sonst noch nicht da. Zurück kommt
    ``{modulname: fehlertext}`` für alles, was sich nicht laden ließ.
    """
    w = wurzel()
    fehler: dict = {}
    for paket in ("core", "ui"):
        for ordner, unter, dateien in os.walk(os.path.join(w, paket)):
            unter[:] = sorted(u for u in unter if u != "__pycache__")
            rel = os.path.relpath(ordner, w).replace(os.sep, ".")
            for datei in sorted(dateien):
                if not datei.endswith(".py") or datei == "__init__.py":
                    continue
                name = f"{rel}.{datei[:-3]}"
                try:
                    importlib.import_module(name)
                except Exception as exc:  # noqa: BLE001 — fehlende Zusatzpakete
                    fehler[name] = f"{exc.__class__.__name__}: {exc}"
    return fehler


def tag_entpacken(tag: str = VORHER_TAG) -> str:
    """Entpackt einen Stand per ``git archive`` in einen Arbeitsordner."""
    ziel = os.path.join(arbeitsordner("stand"), tag)
    os.makedirs(ziel)
    archiv = subprocess.Popen(["git", "-C", ARBEITSBAUM, "archive", tag],
                              stdout=subprocess.PIPE)
    tar = subprocess.run(["tar", "-x", "-C", ziel], stdin=archiv.stdout)
    archiv.stdout.close()
    if archiv.wait() != 0 or tar.returncode != 0:
        raise RuntimeError(f"git archive {tag} ist fehlgeschlagen.")
    return ziel


# ----------------------------------------------------------- Arbeitsordner

def arbeitsordner(name: str = "arbeit", behalten: bool = False) -> str:
    """Frischer Ordner unter ``/tmp/super360_modtests/umbau``.

    Je Aufruf ein eigener; er wird am Ende des Prozesses geräumt, außer mit
    ``behalten=True`` oder gesetztem ``SUPER360_UMBAU_BEHALTEN`` (Fehlersuche).
    Nennt ``SUPER360_UMBAU_ORDNER`` einen Ordner unter der Arbeitswurzel,
    entsteht er dort (Kindprozess eines Werkzeugs).
    """
    os.makedirs(ARBEIT_WURZEL, exist_ok=True)
    eltern = os.environ.get(ELTERN_VARIABLE) or ARBEIT_WURZEL
    if eltern != ARBEIT_WURZEL:
        innen = os.path.realpath(eltern).startswith(
            os.path.realpath(ARBEIT_WURZEL) + os.sep)
        if not innen or not os.path.isdir(eltern):
            raise RuntimeError(f"{ELTERN_VARIABLE}={eltern}: kein Ordner unter "
                               f"{ARBEIT_WURZEL}.")
    pfad = tempfile.mkdtemp(prefix=f"{name}_", dir=eltern)
    _ordner.append(pfad)
    if not behalten and not os.environ.get("SUPER360_UMBAU_BEHALTEN"):
        _raeumen.append(pfad)
    return pfad


def raeume(pfad: str) -> None:
    """Löscht einen Arbeitsordner — und nur etwas unterhalb der Arbeitswurzel."""
    echt = os.path.realpath(pfad)
    if not echt.startswith(os.path.realpath(ARBEIT_WURZEL) + os.sep):
        raise RuntimeError(f"{pfad} liegt nicht unter {ARBEIT_WURZEL}.")
    shutil.rmtree(echt, ignore_errors=True)


@atexit.register
def _am_ende_raeumen() -> None:
    for pfad in _raeumen:
        shutil.rmtree(pfad, ignore_errors=True)
    _raeumen.clear()


def pfad_neutral(text: str) -> str:
    """Ersetzt Arbeitsordner, Wurzel und Heimatordner durch feste Marken.

    So bleiben Protokolle zwischen zwei Läufen vergleichbar und es gelangt
    kein Sitzungs- oder Benutzerpfad in eine abgelegte Datei.
    """
    text = str(text)
    marken = [(p, "<arbeit>") for p in _ordner]
    if _wurzel:
        marken.append((_wurzel, "<wurzel>"))
    marken.append((ARBEITSBAUM, "<wurzel>"))
    marken.append((ARBEIT_WURZEL, "<arbeitswurzel>"))
    marken.append((os.path.expanduser("~"), "~"))
    for pfad, marke in sorted(marken, key=lambda m: -len(m[0])):
        for form in {pfad, os.path.realpath(pfad)}:
            text = text.replace(form, marke)
    return text


# ------------------------------------------------------------------- Cache

CacheKopie = collections.namedtuple("CacheKopie", "cache_root projekt bag")


def projekt_schluessel(bag: str = SEG0_BAG) -> str:
    """Ordnername eines Projekts im Cache, wie ``core.project.Project`` ihn bildet."""
    voll = os.path.abspath(bag)
    return f"{os.path.basename(voll)}-{hashlib.md5(voll.encode('utf-8')).hexdigest()[:8]}"


def cache_wurzel() -> str:
    """Der Cache dieses Prozesses im Arbeitsordner; setzt ``SUPER360_CACHE_ROOT``."""
    global _cache
    if _cache is None:
        _cache = os.path.join(arbeitsordner("cache"), "cache")
        os.makedirs(_cache)
    os.environ["SUPER360_CACHE_ROOT"] = _cache
    mod = sys.modules.get("core.project")
    if mod is not None and os.path.abspath(mod.DEFAULT_CACHE_ROOT) != _cache:
        raise RuntimeError(
            "core.project wurde vor basis.wurzel_setzen() geladen und zeigt auf "
            f"{mod.DEFAULT_CACHE_ROOT} — so liefe der Test gegen einen fremden Cache.")
    return _cache


def cache_kopie(ziel: str | None = None) -> CacheKopie:
    """Frische Kopie des seg0-Projekts für ``SUPER360_CACHE_ROOT``.

    Ohne ``ziel`` landet sie im Cache dieses Prozesses (eine frühere Kopie
    wird ersetzt), ``Project(kopie.bag)`` findet sie dann von selbst. Mit
    ``ziel`` wird dort ein Cache angelegt — für Aufrufe, die die Variable
    selbst an einen Kindprozess geben.
    """
    schluessel = projekt_schluessel()
    quelle = os.path.join(ECHTER_CACHE, schluessel)
    if not os.path.isdir(os.path.join(quelle, "recording")):
        raise RuntimeError(f"seg0-Projekt nicht gefunden: {quelle}")
    if ziel is None:
        root = cache_wurzel()
    else:
        root = os.path.join(os.path.abspath(ziel), "cache")
        os.makedirs(root, exist_ok=True)
    if os.path.realpath(root) == os.path.realpath(ECHTER_CACHE):
        raise RuntimeError("Ziel der Kopie ist der echte Cache.")
    projekt = os.path.join(root, schluessel)
    if os.path.isdir(projekt):
        shutil.rmtree(projekt)
    shutil.copytree(quelle, projekt)
    return CacheKopie(root, projekt, SEG0_BAG)


def cache_dateien() -> dict:
    """Größe und Änderungszeit (ns) jeder Datei im echten Cache.

    ``{projekt: {pfad im Projekt: [bytes, ns]}}``; was lose in der
    Cache-Wurzel liegt, steht unter ``"."``. Es wird nur gelistet, nichts
    geöffnet.
    """
    out: dict = {}
    if not os.path.isdir(ECHTER_CACHE):
        return out
    for name in sorted(os.listdir(ECHTER_CACHE)):
        voll = os.path.join(ECHTER_CACHE, name)
        if not os.path.isdir(voll):
            try:
                st = os.lstat(voll)
            except OSError:
                continue
            out.setdefault(".", {})[name] = [st.st_size, st.st_mtime_ns]
            continue
        dateien = out.setdefault(name, {})
        for ordner, unter, namen in os.walk(voll):
            unter.sort()
            for datei in sorted(namen):
                pfad = os.path.join(ordner, datei)
                try:
                    st = os.lstat(pfad)
                except OSError:
                    continue
                dateien[os.path.relpath(pfad, voll)] = [st.st_size, st.st_mtime_ns]
    return out


def meta_zeiten() -> dict:
    """Stand jedes Projekts im echten Cache als Kennung ``{projekt: hash}``.

    In die Kennung gehen Name, Größe und Änderungszeit (ns) aller Dateien des
    Projekts ein, nicht nur der ``recording/meta.json``. Vor und nach einem
    Lauf verglichen zeigt das, ob er den echten Cache angefasst hat; welche
    Datei es war, sagt :func:`cache_dateien`.
    """
    return {name: hash_wert(json.dumps(dateien, sort_keys=True))
            for name, dateien in cache_dateien().items()}


# -------------------------------------------------------------------- Hash

def hash_wert(wert) -> str:
    """sha256 über dtype, shape und Bytes eines Arrays, einer Zahl oder von Bytes."""
    import numpy as np

    if isinstance(wert, (bytes, bytearray, memoryview)):
        daten = bytes(wert)
        kopf = f"bytes|({len(daten)},)|"
    elif isinstance(wert, str):
        daten = wert.encode("utf-8")
        kopf = f"str|({len(daten)},)|"
    else:
        arr = np.asarray(wert)
        if arr.dtype.hasobject:
            raise TypeError(f"hash_wert: {type(wert).__name__} lässt sich nicht "
                            "als Zahl, Array oder Bytes fassen.")
        kopf = f"{arr.dtype.str}|{tuple(arr.shape)}|"
        daten = np.ascontiguousarray(arr).tobytes()
    h = hashlib.sha256()
    h.update(kopf.encode("ascii"))
    h.update(daten)
    return h.hexdigest()


def datei_hash(pfad: str) -> str:
    h = hashlib.sha256()
    with open(pfad, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------- Umgebungsstempel

def _version(*verteilungen: str, modul: str | None = None) -> str:
    from importlib import metadata
    for name in verteilungen:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    if modul:
        try:
            mod = importlib.import_module(modul)
            return str(getattr(mod, "__version__", "?"))
        except Exception:  # noqa: BLE001
            pass
    return "fehlt"


def _pointcloudmerger() -> dict:
    """git-Stand und Datei-Hashes von ``PointCloudMerger/colorize_pipeline``.

    Gesucht wird wie in ``core/meander.py``: im Heimatordner, dann neben dem
    Repo. Das Paket ist Teil der Rechenwege und liegt uncommittet da.
    """
    for basis_ordner in (os.path.expanduser("~/PointCloudMerger"),
                         os.path.join(ARBEITSBAUM, "..", "PointCloudMerger")):
        paket = os.path.join(basis_ordner, "colorize_pipeline")
        if os.path.isdir(paket):
            break
    else:
        return {"git": "fehlt", "dateien": {}}
    try:
        stand = subprocess.run(["git", "-C", basis_ordner, "rev-parse", "HEAD"],
                               capture_output=True, text=True).stdout.strip()
    except OSError:
        stand = ""
    dateien: dict = {}
    for ordner, unter, namen in os.walk(paket):
        unter[:] = sorted(u for u in unter if u != "__pycache__")
        for name in sorted(namen):
            voll = os.path.join(ordner, name)
            dateien[os.path.relpath(voll, paket)] = datei_hash(voll)
    return {"git": stand or "unbekannt", "dateien": dateien}


def umgebungsstempel() -> dict:
    """Worauf ein Vorher-Stand gerechnet wurde: Bibliotheken und PointCloudMerger.

    laspy und pyproj stehen wegen ``core/georef.py`` darin (LAS-Export, UTM).
    """
    try:
        from PyQt5.QtCore import QT_VERSION_STR as qt
    except Exception:  # noqa: BLE001
        qt = "fehlt"
    return {
        "python": ".".join(str(v) for v in sys.version_info[:3]),
        "numpy": _version("numpy", modul="numpy"),
        "scipy": _version("scipy", modul="scipy"),
        "cv2": _version("opencv-python", "opencv-contrib-python",
                        "opencv-python-headless",
                        "opencv-contrib-python-headless", modul="cv2"),
        "open3d": _version("open3d", modul="open3d"),
        "Pillow": _version("pillow", "Pillow", modul="PIL"),
        "PyQt5": _version("PyQt5"),
        "Qt": qt,
        "VTK": _version("vtk", modul="vtk"),
        "laspy": _version("laspy", modul="laspy"),
        "pyproj": _version("pyproj", modul="pyproj"),
        "pointcloudmerger": _pointcloudmerger(),
    }


# ----------------------------------------------------------------- Fenster

def unter_xvfb() -> bool:
    return "xvfb-run" in os.environ.get("XAUTHORITY", "")


def qt_app():
    """Die QApplication des Prozesses, bei Bedarf neu angelegt.

    cv2 biegt beim Import ``QT_QPA_PLATFORM_PLUGIN_PATH`` auf seine eigenen
    Qt-Plugins um; war es vor der QApplication da (etwa über ``core``), findet
    PyQt5 damit kein xcb und der Prozess bricht ab. Die App selbst legt die
    QApplication zuerst an, hier wird die Variable dafür zurückgenommen.
    """
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        for var in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR"):
            if "cv2" in os.environ.get(var, ""):
                del os.environ[var]
        app = QApplication(sys.argv[:1])
    return app


def qt_app_schliessen(warten: bool = True) -> bool:
    """Gibt die QApplication frei, damit die X-Verbindung vor dem Prozessende schließt.

    Für das Ende eines Prozesses, der selbst kein Fenster gebaut hat; der
    Rufer lässt vorher seine eigenen Verweise auf die QApplication fallen.
    Hält sie noch jemand, geschieht nichts. Unter xvfb-run wird mit ``warten``
    danach kurz gewartet: Xvfb setzt sich nach dem letzten Client zurück und
    meldet das xvfb-run mit einem Signal, und das soll ankommen, solange der
    Prozess noch läuft (s. »Ende 5 unter xvfb-run« im Kopf). Zurück kommt, ob
    die Verbindung geschlossen wurde.
    """
    if "PyQt5.QtWidgets" not in sys.modules:
        return False
    from PyQt5.QtWidgets import QApplication

    gc.collect()
    if QApplication.instance() is not None:
        return False
    if warten and unter_xvfb():
        time.sleep(XVFB_RUECKSETZEN_S)
    return True


def fenster_bauen():
    """Baut das echte Hauptfenster und zeigt es: ``(app, fenster)``.

    Ohne Autotest-Variable, damit kein Bag von selbst geöffnet wird.
    ``show()`` und ``processEvents`` laufen, damit ``isVisible()`` für die
    Kinder gilt. Das Farbschema von ``app.py`` wird nicht gesetzt.
    """
    wurzel()
    for var in ("SUPER360_AUTOTEST", "SUPER360_AUTOTEST_OUT"):
        os.environ.pop(var, None)
    if not os.environ.get("DISPLAY") and not os.environ.get("QT_QPA_PLATFORM"):
        raise SystemExit("Kein DISPLAY — bitte mit 'xvfb-run -a' starten.")
    if not unter_xvfb():
        print("Hinweis: läuft nicht unter xvfb-run, die Fenster erscheinen "
              "auf dem Bildschirm.", file=sys.stderr)
    app = qt_app()
    from ui.main_window import MainWindow

    fenster = MainWindow()
    fenster.show()
    for _ in range(3):
        app.processEvents()
    if not fenster.isVisible():
        raise RuntimeError("Das Hauptfenster ließ sich nicht zeigen.")
    return app, fenster


def fenster_schliessen(app, fenster) -> None:
    """Schließt das Fenster so, wie die App endet.

    Die Ereignisschleife läuft einmal bis ``quit``: an ``aboutToQuit`` hängt
    das Anhalten des Vorlade-Threads der 360°-Ansicht; ohne das bricht Qt beim
    Prozessende mit »QThread: Destroyed while thread is still running« ab.
    """
    from PyQt5.QtCore import QEvent, QTimer

    fenster.close()
    QTimer.singleShot(0, app.quit)
    app.exec_()
    # erst jetzt zerstören: vorher liefe der Thread noch
    fenster.deleteLater()
    app.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()


# ----------------------------------------------------------------- Dialoge

class Dialoge:
    """Ersetzt blockierende Dialoge durch protokollierende Stellvertreter.

    Erfasst werden die statischen Aufrufe von ``QMessageBox`` und
    ``QFileDialog`` sowie ``exec_``/``exec`` von ``QDialog`` und dessen
    Qt-Unterklassen. Jeder Aufruf landet als dict in ``protokoll``; die Antwort
    kommt aus :meth:`antworte` oder ist die Vorgabe (Frage: Ja, Meldung: Ok,
    Dateidialog: abgebrochen, ``exec_``: abgelehnt).
    """

    _MELDUNGEN = ("information", "warning", "critical", "question", "about")
    _DATEI = {"getOpenFileName": ("", ""), "getSaveFileName": ("", ""),
              "getOpenFileNames": ([], ""), "getExistingDirectory": ""}

    def __init__(self, melde=None):
        self.protokoll: list = []
        self._antworten: dict = {}
        self._alt: list = []
        self._melde = melde

    def antworte(self, art: str, *werte) -> "Dialoge":
        """Legt Antworten in die Schlange von ``art``.

        ``art`` ist etwa ``"QMessageBox.question"``, ``"QFileDialog.getSaveFileName"``,
        ``"QDialog.exec_"`` oder mit Klasse ``"QDialog.exec_:ExportDialog"``.
        Eine aufrufbare Antwort bekommt den Eintrag (bei ``exec_`` den Dialog)
        und liefert den Rückgabewert.
        """
        self._antworten.setdefault(art, []).extend(werte)
        return self

    def _antwort(self, arten, vorgabe, arg):
        for art in arten:
            schlange = self._antworten.get(art)
            if schlange:
                wert = schlange.pop(0)
                return wert(arg) if callable(wert) else wert
        return vorgabe

    def _eintrag(self, eintrag: dict) -> None:
        self.protokoll.append(eintrag)
        if self._melde is not None:
            self._melde(eintrag)

    def _setze(self, klasse, name, neu) -> None:
        self._alt.append((klasse, name, klasse.__dict__.get(name, _FEHLT),
                          getattr(klasse, name)))
        setattr(klasse, name, neu)

    def __enter__(self) -> "Dialoge":
        from PyQt5 import QtWidgets
        box, datei = QtWidgets.QMessageBox, QtWidgets.QFileDialog

        def meldung(name):
            def ersatz(parent=None, title="", text="", *a, **k):
                eintrag = {"art": f"QMessageBox.{name}",
                           "titel": pfad_neutral(k.get("title", title)),
                           "text": pfad_neutral(k.get("text", text))}
                vorgabe = (box.Yes if name == "question" else
                           None if name == "about" else box.Ok)
                antwort = self._antwort([eintrag["art"]], vorgabe, eintrag)
                eintrag["antwort"] = None if antwort is None else int(antwort)
                self._eintrag(eintrag)
                return antwort
            return staticmethod(ersatz)

        def dateidialog(name, vorgabe):
            def ersatz(parent=None, caption="", directory="", filter="", *a, **k):
                eintrag = {"art": f"QFileDialog.{name}",
                           "titel": pfad_neutral(k.get("caption", caption)),
                           "start": pfad_neutral(k.get("directory", directory)),
                           "filter": str(k.get("filter", filter))}
                antwort = self._antwort([eintrag["art"]], vorgabe, eintrag)
                eintrag["antwort"] = json.loads(pfad_neutral(json.dumps(antwort)))
                self._eintrag(eintrag)
                return antwort
            return staticmethod(ersatz)

        def ausfuehren(name):
            def ersatz(dialog, *a, **k):
                eintrag = {"art": f"QDialog.{name}",
                           "klasse": type(dialog).__name__,
                           "titel": pfad_neutral(dialog.windowTitle())}
                if isinstance(dialog, box):
                    eintrag["text"] = pfad_neutral(dialog.text())
                    if dialog.informativeText():
                        eintrag["zusatz"] = pfad_neutral(dialog.informativeText())
                antwort = self._antwort(
                    [f"{eintrag['art']}:{eintrag['klasse']}", eintrag["art"]],
                    QtWidgets.QDialog.Rejected, dialog)
                eintrag["antwort"] = int(antwort)
                self._eintrag(eintrag)
                return antwort
            return ersatz

        for name in self._MELDUNGEN:
            self._setze(box, name, meldung(name))
        for name, vorgabe in self._DATEI.items():
            self._setze(datei, name, dateidialog(name, vorgabe))
        klassen = [QtWidgets.QDialog] + [
            k for k in vars(QtWidgets).values()
            if isinstance(k, type) and issubclass(k, QtWidgets.QDialog)
            and k is not QtWidgets.QDialog]
        for klasse in klassen:
            for name in ("exec_", "exec"):
                if klasse is QtWidgets.QDialog or name in klasse.__dict__:
                    self._setze(klasse, name, ausfuehren(name))
        return self

    def __exit__(self, *exc) -> None:
        for klasse, name, eigen, geerbt in reversed(self._alt):
            if eigen is _FEHLT:
                try:
                    delattr(klasse, name)
                    continue
                except (AttributeError, TypeError):
                    pass
            setattr(klasse, name, geerbt)
        self._alt.clear()


_FEHLT = object()


def dialoge_abfangen(melde=None) -> Dialoge:
    """``with dialoge_abfangen() as d:`` — s. :class:`Dialoge`.

    ``melde(eintrag)`` wird zusätzlich je Dialog gerufen (Rekorder).
    """
    return Dialoge(melde)


# ---------------------------------------------------------------- Ersetzen

def ueberall_ersetzen(objekt, ersatz) -> list:
    """Bindet ``ersatz`` überall dort, wo ein Modulname an ``objekt`` hängt.

    Durchsucht jedes geladene ``core.*``- und ``ui.*``-Modul. Ein Stub greift
    damit am Ursprungsort und in jedem ``from … import``, vor und nach dem
    Verschieben einer Funktion. Vorher :func:`module_laden`, sonst fehlen die
    Module, die die App erst später importiert. Zurück kommt die Liste der
    Stellen ``(modul, name)``; rückgängig macht es der Aufruf mit vertauschten
    Argumenten. Methoden und Klassenmethoden sind keine Modulnamen: dafür
    ``setattr`` an der Klasse.
    """
    stellen = []
    for mname, mod in sorted(sys.modules.items()):
        if mod is None or not (mname in ("core", "ui")
                               or mname.startswith(("core.", "ui."))):
            continue
        for name, wert in list(vars(mod).items()):
            if wert is objekt:
                setattr(mod, name, ersatz)
                stellen.append((mname, name))
    return stellen


# -------------------------------------------------------------- Selbsttest

def _selbsttest() -> int:
    import numpy as np

    w = wurzel()
    print(f"Wurzel: {pfad_neutral(w)}")

    # Arbeitsordner: eigener Ordner je Aufruf, unter der Arbeitswurzel
    a, b = arbeitsordner("selbst"), arbeitsordner("selbst")
    assert a != b and os.path.isdir(a) and a.startswith(ARBEIT_WURZEL + os.sep)
    assert pfad_neutral(os.path.join(a, "x.json")) == "<arbeit>/x.json"
    assert pfad_neutral(os.path.expanduser("~/irgendwo")) == "~/irgendwo"
    raeume(b)
    assert not os.path.exists(b)
    # Kindprozess: sein Arbeitsordner entsteht im mitgegebenen Ordner
    kind = subprocess.run([sys.executable, os.path.abspath(__file__), "--arbeitsordner"],
                          capture_output=True, text=True,
                          env=dict(os.environ, **{ELTERN_VARIABLE: a}))
    assert kind.returncode == 0 and os.path.dirname(kind.stdout.strip()) == a, kind
    assert pfad_neutral(kind.stdout.strip()).startswith("<arbeit>/")
    fremd = subprocess.run([sys.executable, os.path.abspath(__file__), "--arbeitsordner"],
                           capture_output=True, text=True,
                           env=dict(os.environ, **{ELTERN_VARIABLE: os.path.expanduser("~")}))
    assert fremd.returncode != 0 and not fremd.stdout.strip(), fremd
    try:
        raeume(os.path.expanduser("~"))
        raise AssertionError("raeume() hat einen fremden Ordner angenommen")
    except RuntimeError:
        pass
    print("Arbeitsordner: eigener Ordner je Aufruf (im Kindprozess unter dem des "
          "Starters), räumt nur unter der Arbeitswurzel")

    # Hash: dtype, shape und Bytes zählen alle
    x = np.arange(6, dtype=np.float32)
    assert hash_wert(x) == hash_wert(x.copy())
    assert hash_wert(x) != hash_wert(x.astype(np.float64))
    assert hash_wert(x) != hash_wert(x.reshape(2, 3))
    assert hash_wert(x.reshape(2, 3)) == hash_wert(np.asfortranarray(x.reshape(2, 3)))
    assert hash_wert(1) != hash_wert(1.0) and hash_wert(b"ab") != hash_wert("ab")
    assert hash_wert(np.float64(0.5)) == hash_wert(0.5)
    try:
        hash_wert({"a": 1})
        raise AssertionError("hash_wert nimmt ein dict an")
    except TypeError:
        pass
    print("hash_wert: dtype, shape und Bytes gehen ein")

    # Cache: die Variable zeigt in den Arbeitsordner, bevor core geladen ist
    echt_vorher = meta_zeiten()
    root = cache_wurzel()
    assert os.environ["SUPER360_CACHE_ROOT"] == root and root.startswith(ARBEIT_WURZEL)
    from core import project as project_mod
    assert project_mod.DEFAULT_CACHE_ROOT == root, project_mod.DEFAULT_CACHE_ROOT
    kopie = cache_kopie()
    assert kopie.cache_root == root and os.path.isdir(kopie.projekt)
    projekt = project_mod.Project(kopie.bag)
    assert projekt.dir == kopie.projekt and projekt.has_recording(), projekt.dir
    meta_kopie = os.path.join(kopie.projekt, "recording", "meta.json")
    meta_echt = os.path.join(ECHTER_CACHE, projekt_schluessel(), "recording", "meta.json")
    assert datei_hash(meta_kopie) == datei_hash(meta_echt)
    if os.path.isdir(kopie.bag):
        from core.recording import Recording
        rec = Recording.load(projekt.recording_dir(), bag_path=kopie.bag)
        with open(meta_kopie, encoding="utf-8") as fh:
            geschrieben = "gravity_level" in json.load(fh)
        print(f"Cache-Kopie: {rec.n_scans} Scans geladen, gravity_level "
              f"{'in die Kopie geschrieben' if geschrieben else 'nicht gemessen'}")
        del rec
    else:
        print("Cache-Kopie: angelegt (Bag fehlt, Aufzeichnung nicht geladen)")
    with open(meta_echt, encoding="utf-8") as fh:
        echt_hat = "gravity_level" in json.load(fh)
    assert meta_zeiten() == echt_vorher, "echter Cache wurde angefasst"
    n_dateien = sum(len(d) for d in cache_dateien().values())
    print(f"Echter Cache unberührt ({len(echt_vorher)} Projekte, {n_dateien} Dateien "
          f"nach Größe und Änderungszeit; seg0 "
          f"{'mit' if echt_hat else 'ohne'} gravity_level)")
    frisch = cache_kopie()
    assert datei_hash(os.path.join(frisch.projekt, "recording", "meta.json")) \
        == datei_hash(meta_echt), "zweite Kopie ist nicht frisch"

    # Umgebungsstempel
    stempel = umgebungsstempel()
    assert stempel == umgebungsstempel()
    assert "fehlt" not in (stempel["numpy"], stempel["PyQt5"], stempel["VTK"]), stempel
    kurz = {k: v for k, v in stempel.items() if k != "pointcloudmerger"}
    pcm = stempel["pointcloudmerger"]
    print(f"Umgebungsstempel: {kurz}; PointCloudMerger {pcm['git'][:12]}, "
          f"{len(pcm['dateien'])} Dateien")
    assert "/home/" not in json.dumps(stempel), "Benutzerpfad im Stempel"

    # Fenster
    fehler = module_laden()
    for name, text in sorted(fehler.items()):
        print(f"  nicht ladbar: {name} ({text})")
    app, fenster = fenster_bauen()
    assert fenster.isVisible() and fenster._cloud_view.isVisible()
    n_abschnitte = len(fenster._SECTIONS)
    n_schluessel = len(fenster._collect_settings())
    n_actions = len(fenster._actions)
    print(f"Fenster gebaut und sichtbar ({'xvfb' if unter_xvfb() else 'Bildschirm'}, "
          f"DISPLAY {os.environ.get('DISPLAY')}): {n_abschnitte} Abschnitte, "
          f"{n_schluessel} Einstellungsschlüssel, {n_actions} Einträge in _actions")
    assert n_abschnitte and n_schluessel and n_actions

    # Dialoge
    from PyQt5.QtWidgets import QDialog, QFileDialog, QMessageBox
    alt = (QMessageBox.question, QDialog.exec_)
    with dialoge_abfangen() as dialoge:
        dialoge.antworte("QMessageBox.question", QMessageBox.No)
        dialoge.antworte("QFileDialog.getSaveFileName", (os.path.join(a, "x.ply"), "PLY"))
        dialoge.antworte("QDialog.exec_:QDialog", QDialog.Accepted)
        assert QMessageBox.question(fenster, "Titel", "Wirklich?") == QMessageBox.No
        assert QMessageBox.question(fenster, "Titel", "Wirklich?") == QMessageBox.Yes
        assert QMessageBox.information(fenster, "T", f"Liegt in {a}") == QMessageBox.Ok
        fenster._show_error("Testfehler", "nur zur Probe")
        assert QFileDialog.getSaveFileName(fenster, "Speichern", a)[1] == "PLY"
        assert QFileDialog.getExistingDirectory(fenster, "Ordner") == ""
        dlg = QDialog(fenster)
        dlg.setWindowTitle("Probe")
        assert dlg.exec_() == QDialog.Accepted and dlg.exec_() == QDialog.Rejected
        box = QMessageBox(fenster)
        box.setText("Instanz")
        assert box.exec_() == QDialog.Rejected
    arten = [e["art"] for e in dialoge.protokoll]
    assert arten == ["QMessageBox.question"] * 2 + ["QMessageBox.information",
                                                   "QMessageBox.critical",
                                                   "QFileDialog.getSaveFileName",
                                                   "QFileDialog.getExistingDirectory",
                                                   "QDialog.exec_", "QDialog.exec_",
                                                   "QDialog.exec_"], arten
    assert dialoge.protokoll[2]["text"] == "Liegt in <arbeit>"
    assert dialoge.protokoll[4]["antwort"] == ["<arbeit>/x.ply", "PLY"]
    assert dialoge.protokoll[8]["text"] == "Instanz"
    assert (QMessageBox.question, QDialog.exec_) == alt, "Dialoge nicht zurückgestellt"
    print(f"Dialoge: {len(arten)} abgefangen und protokolliert, danach zurückgestellt")

    # Ersetzen: am Ursprung und in jedem from-Import
    from core import project as p1
    import ui.main_window as mw
    original = mw.Project
    assert original is p1.Project

    class Ersatz:
        pass

    stellen = ueberall_ersetzen(original, Ersatz)
    assert ("core.project", "Project") in stellen and ("ui.main_window", "Project") in stellen
    assert mw.Project is Ersatz and p1.Project is Ersatz
    zurueck = ueberall_ersetzen(Ersatz, original)
    assert sorted(zurueck) == sorted(stellen) and mw.Project is original
    print(f"ueberall_ersetzen: {len(stellen)} Stellen ersetzt und zurückgestellt "
          f"({', '.join(m for m, _ in stellen)})")

    fenster_schliessen(app, fenster)
    assert meta_zeiten() == echt_vorher, "echter Cache wurde angefasst"
    print("basis SELFTEST OK")
    return 0


def main(argv=None) -> int:
    parser = argumente(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    parser.add_argument("--selbsttest", action="store_true")
    parser.add_argument("--arbeitsordner", action="store_true",
                        help="frischen Arbeitsordner anlegen und nennen (bleibt stehen)")
    parser.add_argument("--cache-kopie", metavar="ORDNER",
                        help="seg0-Projekt nach ORDNER/cache kopieren, Cache-Wurzel nennen")
    parser.add_argument("--raeume", metavar="ORDNER",
                        help="einen Arbeitsordner löschen")
    parser.add_argument("--stempel", action="store_true",
                        help="Umgebungsstempel als JSON ausgeben")
    args = parser.parse_args(argv)
    if args.arbeitsordner:
        print(arbeitsordner(behalten=True))
        return 0
    if args.cache_kopie:
        print(cache_kopie(args.cache_kopie).cache_root)
        return 0
    if args.raeume:
        raeume(args.raeume)
        return 0
    if args.stempel:
        print(json.dumps(umgebungsstempel(), indent=1, sort_keys=True))
        return 0
    if args.selbsttest:
        wurzel_setzen(args.wurzel)
        return _selbsttest()
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
