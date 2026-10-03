#!/usr/bin/env python3
"""Ablauf-Proben: Handler des Hauptfensters fahren und ihren Ablauf aufzeichnen.

Starter und Rekorder für alle ``scripts/umbau/ablaeufe_<familie>.py``. Vertrag::

    FAELLE = {name: funktion}      # funktion(sitzung) -> None

Je Fall baut die :class:`Sitzung` das echte Fenster neu und zeichnet auf:

* jeden Hintergrundschritt: ``_start_worker`` ist durch einen Rekorder ersetzt
  (Text, cancellable, ob schon ein Schritt lief). Der Job läuft nicht in einem
  Thread, sondern wartet, bis der Handler zurück ist — wie in der App — und
  wird dann von :meth:`Sitzung.laufe` im GUI-Thread gefahren, mit
  protokollierendem ``progress_cb``, ``cancel`` und ``log_cb``. Danach ruft der
  Rekorder ``_worker_done`` bzw. ``_worker_failed`` des Fensters; was diese an
  Folge-Arbeitern starten, kommt wieder in die Schlange. Das echte
  ``_start_worker`` läuft dabei nicht; wer es prüfen will (zweiter Start,
  während ein Schritt läuft), ruft es über :meth:`Sitzung.echter_start`,
* Aufrufe der ersetzten core-Blattfunktionen mit Argument-Kurzform
  (:meth:`Sitzung.ersetze`, :meth:`Sitzung.ersetze_attribut`),
* Fortschrittswerte auf 6 Stellen, Logzeilen, Dialoge (Texte wie Logzeilen
  ohne Sitzungspfade und Projektschlüssel),
* Aufrufe an die 3D-Ansicht (``set_cloud``, ``set_color_mode``,
  ``set_farbmodi``, ``set_preview_cloud``, ``set_color_preview``),
* am Ende die Texte aller Labels des Fensters und die im Arbeitsordner
  geschriebenen Dateien: JSON als dict, alles andere als Hash.

Fehlt dem Fenster, was ein Fall braucht — ein Handler, ein Widget, ein
Attribut, etwa weil der Umbau es entfernt hat —, endet nur dieser Fall mit
einem Fehler, der den fehlenden Namen nennt (``AttributeError: 'MainWindow'
object has no attribute '…'``); die übrigen Fälle laufen weiter. Das gilt
auch, wenn das Fenster sich gar nicht bauen lässt (dann scheitert jeder Fall
einzeln) und für eine Ausnahme, die in einem Slot unbehandelt bleibt (Signal,
Einmal-Timer): die Sitzung steht dafür als ``sys.excepthook``, der Fall läuft
zu Ende und gilt dann als gescheitert. ``--vergleiche`` meldet einen solchen
Fall als Abweichung; freigeben lässt er sich nicht, der Fall ist anzupassen.

Der Cache (``SUPER360_CACHE_ROOT``) liegt je Fall leer im Arbeitsordner;
Pfade erscheinen in der Aufzeichnung nur als ``<arbeit>/…``.

Jede Familie läuft in einem eigenen Kindprozess. Endet er durch ein Signal
(beim Aufbau vieler Fenster nacheinander kommt das vereinzelt vor), wird die
Familie einmal wiederholt; seine Arbeitsordner liegen im Ordner des Starters
und verschwinden mit ihm.

Aufrufe::

    xvfb-run -a python3 scripts/umbau/ablauf_probe.py --schreibe [--ersetzen] [--nur familie,…]
    xvfb-run -a python3 scripts/umbau/ablauf_probe.py --vergleiche [--nur familie,… | fall,…]
                                                      [--erwartet <json im Arbeitsordner>]
                                                      [--erwartet-vorlage <json im Arbeitsordner>]
    xvfb-run -a python3 scripts/umbau/ablauf_probe.py --selbsttest

``--schreibe`` legt je Familie ``vorher/ablaeufe_<familie>.json`` ab.
``--vergleiche`` nennt je Fall jede abweichende Stelle; die Ereignisfolgen
werden als Folgen abgeglichen, ein eingefügtes Ereignis verschiebt die übrigen
nicht. Verglichen wird typgenau: ``true`` gegen ``1`` und ``10`` gegen ``10.0``
sind Abweichungen.

``--erwartet`` nennt gewollte Abweichungen und bindet sie an den erwarteten
neuen Stand, damit eine weitere Änderung im selben Fall ein Fehler bleibt::

    {"familie.fall": {"grund": "Text", "hash": "<Hash der neuen Aufzeichnung>"},
     "familie.*":    {"grund": "Text", "zeilen": ["labels[_status_lbl]: alt … | neu …"]}}

Mit ``hash`` gilt der Fall nur, wenn die neue Aufzeichnung genau diesen Hash
trägt (``zeilen`` daneben dient dann nur dem Leser). Mit ``zeilen`` allein gilt
jede Abweichungszeile, die wörtlich in der Liste steht; jede andere bleibt
eine Abweichung. Der Schlüssel darf ``*`` enthalten. Ein Fall, der mit einem
Fehler endet, lässt sich nicht freigeben. ``--erwartet-vorlage`` schreibt für
alle Fälle, die abweichen, eine solche Datei mit Hash und Zeilen und leerem
``grund`` — ohne Grund wird ein Eintrag nicht angenommen.

Für Selbsttest und Gegenprüfung: ``--faelle-ordner`` und ``--ziel``.

Ende: 0 = alle Abläufe gleich (bis auf erwartete), 1 = Abweichung, 2 = Fehler.

Ende 5 mit »xvfb-run: error: problem while cleaning up temporary directory«
kommt von xvfb-run, nicht von hier: dann die Ergebniszeile darüber lesen (das
eigene Ende war 1 oder 2, nie 0) und den verwaisten Xvfb beenden; mehr dazu im
Kopf von ``basis.py``. Damit es nicht dazu kommt, schließt der Starter seine
X-Verbindung vor dem Ende und wartet bei einem Ende ungleich 0 eine halbe
Sekunde.
"""
from __future__ import annotations

import argparse
import dataclasses
import difflib
import fnmatch
import gc
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import basis  # noqa: E402

#: Der Projektordner im Cache trägt einen Hash des Bagpfads; der Pfad liegt im
#: Arbeitsordner und heißt je Lauf anders.
_SCHLUESSEL = re.compile(r"((?:^|/)cache/[^/\s\"']+)-[0-9a-f]{8}(?![0-9a-f])")

ANSICHT_METHODEN = ("set_cloud", "set_color_mode", "set_farbmodi",
                    "set_preview_cloud", "set_color_preview")

#: Die eine QApplication des Prozesses. Ohne diesen Verweis stirbt sie mit jeder
#: Sitzung, jeder Fall verbindet sich neu mit dem X-Server, und der setzt sich
#: zurück, sobald kein Client mehr da ist — unter Last scheitert dann die
#: nächste Verbindung.
_app_halter: list = []


def _qt_app():
    """``basis.qt_app()``, für die Dauer des Prozesses festgehalten."""
    if not _app_halter:
        _app_halter.append(basis.qt_app())
    return _app_halter[0]


# ------------------------------------------------------------------ Rekorder

class _Abbruch(threading.Event):
    """Das cancel-Event eines Jobs; ``set()`` wird einmal protokolliert."""

    def __init__(self, melde):
        super().__init__()
        self._melde = melde

    def set(self) -> None:
        if not self.is_set():
            self._melde("abbruch")
        super().set()


class _StubArbeiter:
    """Steht anstelle des QThread-Arbeiters in ``_worker`` und ``_retired``."""

    def __init__(self, cancellable: bool, melde):
        self.cancel = _Abbruch(melde)
        self.cancellable = bool(cancellable)

    def isFinished(self) -> bool:
        return True

    def isRunning(self) -> bool:
        return False

    def wait(self, *_a) -> bool:
        return True


class Sitzung:
    """Ein Fall: echtes Fenster, Rekorder, Stub-Objekte, Arbeitsordner."""

    def __init__(self, name: str):
        self.name = name
        self.ereignisse: list = []
        self.blattaufrufe = 0
        self._schlange: list = []
        self._ersetzt: list = []
        self._attribute: list = []
        self._muster: list = []
        self._json_fluechtig: set = set()
        self._abbruch_nach: int | None = None
        self._fortschritte = 0
        self._stand: dict = {}
        self._in_ansicht = 0
        self._haken = None
        self.ausnahmen: list = []
        self.dialoge = None
        self.app = None
        self.fenster = None
        # Cache und Fallordner liegen nebeneinander im selben Arbeitsordner
        self.cache_root = basis.cache_wurzel()
        self._arbeit = os.path.dirname(self.cache_root)
        self.ordner = os.path.join(self._arbeit, "fall")

    # ------------------------------------------------------------ Aufbau

    def __enter__(self) -> "Sitzung":
        try:
            self._aufbau()
        except BaseException:
            # lässt sich das Fenster nicht bauen, bleibt für den nächsten Fall
            # nichts ersetzt zurück
            self.__exit__()
            raise
        return self

    def _aufbau(self) -> None:
        self._haken = sys.excepthook
        sys.excepthook = self._ausnahme
        for pfad in (self.cache_root, self.ordner):
            shutil.rmtree(pfad, ignore_errors=True)
            os.makedirs(pfad)
        self.dialoge = basis.dialoge_abfangen(
            lambda eintrag: self.ereignisse.append(["dialog", self._texte(eintrag)]))
        self.dialoge.__enter__()
        self.app, f = basis.fenster_bauen()
        self.fenster = f
        f._start_worker = self._starte_arbeiter
        log_alt = f._log

        def log(line):
            self.ereignis("log", self.text(line))
            log_alt(line)
        f._log = log
        for name in ANSICHT_METHODEN:
            alt = getattr(f._cloud_view, name, None)
            if alt is not None:
                setattr(f._cloud_view, name, self._ansicht(name, alt))
        self.dateien_merken()

    def _ausnahme(self, typ, wert, spur) -> None:
        """Steht als ``sys.excepthook``: eine in einem Slot unbehandelte Ausnahme.

        Ohne einen eigenen Haken bräche PyQt5 den Prozess an dieser Stelle ab
        und mit ihm alle übrigen Fälle der Familie.
        """
        traceback.print_exception(typ, wert, spur)
        self.ausnahmen.append(f"{typ.__name__}: {wert}")

    def __exit__(self, *exc) -> None:
        for stub, original in reversed(self._ersetzt):
            basis.ueberall_ersetzen(stub, original)
        for besitzer, attribut, alt in reversed(self._attribute):
            setattr(besitzer, attribut, alt)
        try:
            if self.fenster is not None:
                self._schlange.clear()
                self.fenster._worker = None
                basis.fenster_schliessen(self.app, self.fenster)
        finally:
            if self.dialoge is not None:
                self.dialoge.__exit__()
            # Alle Verweise auf das Fenster fallen lassen und seine
            # Python-Hüllen sofort abräumen. Bleiben sie liegen, räumt die
            # Speicherbereinigung sie irgendwann, während das nächste Fenster
            # entsteht, und der Prozess stürzt dabei vereinzelt ab.
            self.fenster = None
            self._ersetzt.clear()
            self._attribute.clear()
            self._schlange.clear()
            gc.collect()
            if self.app is not None:
                self.app.processEvents()
            gc.collect()
            if self._haken is not None:
                sys.excepthook, self._haken = self._haken, None

    def _ansicht(self, name, alt):
        def ersatz(*a, **k):
            # nur Aufrufe von außen: was die Ansicht dabei selbst ruft, ist
            # ihre Sache und darf sich beim Aufräumen dort ändern
            if self._in_ansicht == 0:
                self.ereignis("ansicht", name, self.kurz(a), self.kurz(k))
            self._in_ansicht += 1
            try:
                return alt(*a, **k)
            finally:
                self._in_ansicht -= 1
        return ersatz

    # ------------------------------------------------------- Aufzeichnen

    def ereignis(self, art: str, *teile) -> None:
        self.ereignisse.append([art, *teile])

    def fluechtig(self, muster: str, ersatz: str = "<…>") -> None:
        """Regulärer Ausdruck, der in allen Texten ersetzt wird (Dauern, Uhrzeiten)."""
        self._muster.append((re.compile(muster), ersatz))

    def json_fluechtig(self, *schluessel: str) -> None:
        """Schlüssel, deren Wert in geschriebenen JSON-Dateien nicht zählt."""
        self._json_fluechtig.update(schluessel)

    def text(self, wert) -> str:
        out = _SCHLUESSEL.sub(r"\1-<schlüssel>", basis.pfad_neutral(str(wert)))
        for muster, ersatz in self._muster:
            out = muster.sub(ersatz, out)
        return out

    def _texte(self, wert):
        """Wie :meth:`text`, für jeden Text in einem Dialog-Eintrag."""
        if isinstance(wert, str):
            return self.text(wert)
        if isinstance(wert, dict):
            return {k: self._texte(v) for k, v in wert.items()}
        if isinstance(wert, (list, tuple)):
            return [self._texte(v) for v in wert]
        return wert

    def kurz(self, wert, tiefe: int = 0):
        """Argument-Kurzform: klein, vergleichbar, ohne Sitzungspfade."""
        import numpy as np

        if wert is None or isinstance(wert, (bool, int)):
            return wert
        if isinstance(wert, float):
            return float(f"{wert:.9g}") if wert == wert else "nan"
        if isinstance(wert, str):
            return self.text(wert)
        if isinstance(wert, os.PathLike):
            return self.text(os.fspath(wert))
        if isinstance(wert, np.generic):
            return self.kurz(wert.item(), tiefe)
        if isinstance(wert, np.ndarray):
            if wert.dtype.hasobject:
                return f"ndarray object {tuple(wert.shape)}"
            return (f"ndarray {wert.dtype.str} {tuple(wert.shape)} "
                    f"#{basis.hash_wert(np.asarray(wert))[:12]}")
        if isinstance(wert, (bytes, bytearray)):
            return f"bytes {len(wert)} #{basis.hash_wert(bytes(wert))[:12]}"
        eigen = getattr(wert, "_kurz", None)
        if isinstance(eigen, str):
            return f"<{eigen}>"
        if tiefe > 6:
            return f"<{type(wert).__name__}>"
        if isinstance(wert, dict):
            return {str(k): self.kurz(v, tiefe + 1)
                    for k, v in sorted(wert.items(), key=lambda kv: str(kv[0]))}
        if isinstance(wert, (list, tuple, set, frozenset)):
            folge = sorted(wert, key=repr) if isinstance(wert, (set, frozenset)) else list(wert)
            out = [self.kurz(v, tiefe + 1) for v in folge[:32]]
            if len(folge) > 32:
                out.append(f"… insgesamt {len(folge)}")
            return out
        if isinstance(wert, threading.Event):
            return "<Event>"
        if dataclasses.is_dataclass(wert) and not isinstance(wert, type):
            return {f"<{type(wert).__name__}>": {
                f.name: self.kurz(getattr(wert, f.name), tiefe + 1)
                for f in dataclasses.fields(wert)}}
        if isinstance(wert, type):
            return f"<Klasse {wert.__name__}>"
        if callable(wert):
            return "<aufrufbar>"
        return f"<{type(wert).__name__}>"

    # ----------------------------------------------------------- Ersetzen

    def _stub(self, name, ergebnis, ersatz):
        def stub(*a, **k):
            self.blattaufrufe += 1
            self.ereignis("aufruf", name, self.kurz(a), self.kurz(k))
            if ersatz is not None:
                return ersatz(*a, **k)
            return ergebnis
        stub._kurz = f"Stub {name}"
        return stub

    @staticmethod
    def _name(objekt, name):
        if name:
            return name
        modul = (getattr(objekt, "__module__", "") or "").split(".")[-1]
        eigen = getattr(objekt, "__qualname__", None) or getattr(objekt, "__name__", "?")
        return f"{modul}.{eigen}" if modul else eigen

    def ersetze(self, objekt, ergebnis=None, ersatz=None, name: str | None = None):
        """Ersetzt eine Blattfunktion (oder Klasse) aus core in allen Modulen.

        Der Stellvertreter zeichnet jeden Aufruf auf und liefert ``ergebnis``
        bzw. das, was ``ersatz(*a, **k)`` liefert. Greift am Ursprung und in
        jedem ``from … import``, also vor und nach dem Verschieben.
        """
        name = self._name(objekt, name)
        stub = self._stub(name, ergebnis, ersatz)
        if not basis.ueberall_ersetzen(objekt, stub):
            raise RuntimeError(f"{name}: in keinem geladenen core- oder ui-Modul "
                               f"an einen Namen gebunden.")
        self._ersetzt.append((stub, objekt))
        return stub

    def ersetze_attribut(self, besitzer, attribut: str, ergebnis=None, ersatz=None,
                         name: str | None = None):
        """Wie :meth:`ersetze`, für Methoden an einer Klasse (``Recording.load``)."""
        roh = besitzer.__dict__[attribut]
        name = name or f"{besitzer.__name__}.{attribut}"
        stub = self._stub(name, ergebnis, ersatz)
        if isinstance(roh, staticmethod):
            stub = staticmethod(stub)
        elif isinstance(roh, classmethod):
            stub = classmethod(stub)
        setattr(besitzer, attribut, stub)
        self._attribute.append((besitzer, attribut, roh))
        return stub

    def antworte(self, art: str, *werte) -> None:
        """Antworten für abgefangene Dialoge, s. ``basis.Dialoge.antworte``."""
        self.dialoge.antworte(art, *werte)

    # ----------------------------------------------------------- Arbeiter

    def _starte_arbeiter(self, text, job, on_done, extra_progress=None,
                         on_failed=None, cancellable=True) -> None:
        f = self.fenster
        self.ereignis("arbeiter", self.text(text), bool(cancellable),
                      "läuft schon einer" if f._busy else "frei")
        if f._closing:
            return
        arbeiter = _StubArbeiter(cancellable, self.ereignis)
        f._worker = arbeiter
        f._log(text)
        f._set_busy(True, text)
        self._schlange.append((arbeiter, text, job, on_done, extra_progress, on_failed))

    def echter_start(self, text, job, on_done, warten_s: float = 60.0, **k) -> bool:
        """Ruft das echte ``_start_worker`` des Fensters, nicht den Rekorder.

        Für das, was nur dort geschieht — etwa ein zweiter Start, während
        schon ein Schritt läuft. Aufgezeichnet wird, ob vorher einer lief und
        ob ein Arbeiter gestartet wurde. Ein gestarteter Arbeiter ist ein
        echter Thread; es wird gewartet, bis er fertig ist und das Fenster
        sein Ergebnis angenommen hat. Zurück kommt, ob gestartet wurde.
        """
        f = self.fenster
        vorher = f._worker
        lief = "lief schon einer" if f._busy else "frei"
        type(f)._start_worker(f, text, job, on_done, **k)
        arbeiter = f._worker
        gestartet = arbeiter is not None and arbeiter is not vorher
        self.ereignis("echter start", self.text(text), lief,
                      "gestartet" if gestartet else "abgelehnt")
        if gestartet:
            ende = time.monotonic() + warten_s
            while not arbeiter.isFinished():
                if time.monotonic() > ende:
                    raise RuntimeError(f"Der echte Arbeiter '{text}' wird nicht fertig.")
                self.app.processEvents()
                time.sleep(0.005)
            self.warte()
        return gestartet

    def abbruch_nach(self, n: int | None) -> None:
        """Setzt beim n-ten Fortschritt (ab jetzt gezählt) das cancel-Event."""
        self._abbruch_nach = n
        self._fortschritte = 0

    def wartend(self) -> int:
        return len(self._schlange)

    def laufe(self, hoechstens: int = 100) -> None:
        """Fährt die wartenden Jobs samt Folge-Arbeitern, bis keiner mehr wartet."""
        f = self.fenster
        while self._schlange:
            hoechstens -= 1
            if hoechstens < 0:
                raise RuntimeError("Arbeiter stoßen einander endlos an.")
            arbeiter, text, job, on_done, extra, on_failed = self._schlange.pop(0)

            def fortschritt(frac, msg, arbeiter=arbeiter, extra=extra):
                self.ereignis("fortschritt", round(float(frac), 6), self.text(msg))
                f._on_progress(float(frac), str(msg))
                if extra is not None:
                    extra(float(frac), str(msg))
                self._fortschritte += 1
                if self._abbruch_nach is not None \
                        and self._fortschritte >= self._abbruch_nach:
                    arbeiter.cancel.set()

            fehler = zeile = klasse = None
            ergebnis = None
            try:
                ergebnis = job(progress_cb=fortschritt, cancel=arbeiter.cancel,
                               log_cb=lambda line: f._log(str(line)))
            except Exception as exc:  # noqa: BLE001 — wie Worker.run
                fehler = str(exc) or exc.__class__.__name__
                klasse = exc.__class__.__name__
                zeile = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            if fehler is not None:
                f._log(zeile)
                self.ereignis("fehlgeschlagen", self.text(text), klasse, self.text(fehler))
                f._worker_failed(arbeiter, text, fehler, on_failed)
            else:
                self.ereignis("fertig", self.text(text), self.kurz(ergebnis))
                f._worker_done(arbeiter, on_done, ergebnis)
            self.warte()

    def warte(self, ms: int = 0) -> None:
        """Ereignisschleife laufen lassen (Einmal-Timer); ``ms`` für die Verzögerten."""
        from PyQt5.QtCore import QEventLoop, QTimer

        if ms > 0:
            schleife = QEventLoop()
            QTimer.singleShot(int(ms), schleife.quit)
            schleife.exec_()
        for _ in range(3):
            self.app.processEvents()

    # ------------------------------------------------------- Stub-Objekte

    def stub_projekt(self, name: str = "bag_demo"):
        """Echtes ``Project`` für ein gedachtes Bag; sein Ordner liegt im Fall-Cache."""
        from core.project import Project

        projekt = Project(os.path.join(self.ordner, name))
        os.makedirs(projekt.dir, exist_ok=True)
        return projekt

    def stub_aufzeichnung(self, n_scans: int = 4, punkte_je_scan: int = 50,
                          seed: int = 0, ordner: str | None = None):
        """Kleine echte ``Recording`` aus geseedeten Zufallspunkten.

        Mit ``ordner`` werden auch ihre sechs Dateien geschrieben, samt
        ``gravity_level`` in der meta.json — so liest ``Recording.load`` kein
        Bag nach.
        """
        import numpy as np
        from core.recording import Recording

        rng = np.random.default_rng(seed)
        n = n_scans * punkte_je_scan
        punkte = rng.uniform(-5.0, 5.0, size=(n, 3)).astype(np.float32)
        intensitaet = rng.uniform(0.0, 255.0, size=n).astype(np.float32)
        offsets = np.arange(0, n + 1, punkte_je_scan, dtype=np.int64)
        stamps = 1000.0 + 0.1 * np.arange(n_scans, dtype=np.float64)
        posen = np.zeros((n_scans, 7), dtype=np.float64)
        posen[:, 0] = 0.5 * np.arange(n_scans)
        posen[:, 6] = 1.0
        meta = {"bag": os.path.join(self.ordner, "bag_demo"), "config": "whs_dense.yaml",
                "expected_scans": n_scans, "rate": 1.0, "n_scans": n_scans,
                "n_points": n, "created": "2026-01-01T00:00:00",
                "gravity_level": {"quat": [0.0, 0.0, 0.0, 1.0], "tilt_deg": 0.0,
                                  "threshold_deg": 0.0, "applied": False}}
        if ordner is not None:
            os.makedirs(ordner, exist_ok=True)
            punkte.tofile(os.path.join(ordner, "points.bin"))
            intensitaet.tofile(os.path.join(ordner, "intensity.bin"))
            np.save(os.path.join(ordner, "offsets.npy"), offsets)
            np.save(os.path.join(ordner, "stamps.npy"), stamps)
            np.save(os.path.join(ordner, "poses.npy"), posen)
            with open(os.path.join(ordner, "meta.json"), "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=2)
        return Recording(punkte, intensitaet, offsets, stamps, posen, dict(meta),
                         meta["gravity_level"])

    def stub_bag(self, n_frames: int = 5, mit_kamera: bool = True,
                 name: str = "bag_demo"):
        """Stellvertreter für ``ThreadLocalBag``: feste Zeiten, kleine Bilder."""
        import numpy as np
        from core.bag_reader import BagInfo

        pfad = os.path.join(self.ordner, name)

        class StubBag:
            _kurz = f"StubBag {name}"
            bag_path = pfad

            def info(self):
                return BagInfo(
                    path=pfad, start=1000.0, end=1000.0 + n_frames, duration=float(n_frames),
                    topics={"/livox/lidar": ("livox_ros_driver2/msg/CustomMsg", 10)},
                    camera_topic="/kamera" if mit_kamera else None,
                    lidar_topic="/livox/lidar", gps_fix_topic=None,
                    gps_raw_topic=None, imu_topic="/livox/imu")

            def camera_stamps(self):
                n = n_frames if mit_kamera else 0
                return 1000.0 + np.arange(n, dtype=np.float64)

            def read_camera(self, idx):
                bild = np.zeros((48, 96, 3), dtype=np.uint8)
                bild[:, :, 1] = 10 * (int(idx) % 20)
                return bild

            def read_camera_jpeg(self, idx):
                return b"\xff\xd8stub%d\xff\xd9" % int(idx)

            def iter_camera(self, start=0, stop=None):
                for i in range(start, n_frames if stop is None else stop):
                    yield i, self.read_camera(i)

            def read_gps(self):
                return []

            def close(self):
                pass

        return StubBag()

    def stub_pipeline(self, thermal: bool = False, mit_lage: bool = True):
        """Stellvertreter für die Mäander-Pipeline: Lage und ein paar Kameras."""
        import numpy as np
        from scipy.spatial.transform import Rotation

        kameras = {
            "xyz": np.random.default_rng(0).uniform(-5, 5, size=(500, 3)),
            "C": np.random.default_rng(1).uniform(-5, 5, size=(20, 3)) + [0, 0, 50],
            "names": np.array(["a"]), "Rcw": np.eye(3)[None], "tcw": np.zeros((1, 3)),
            "size": np.array([[100.0, 100.0]]),
            "params": np.array([[50.0, 50.0, 50.0, 0.0]]),
            "model": np.array("SIMPLE_RADIAL")}

        class StubPipeline:
            _kurz = "StubPipeline"
            yaw = np.radians(30.0) if mit_lage else None
            t = np.array([1.0, 2.0, 0.5]) if mit_lage else None
            thermal_versatz = (0.0, 0.0)
            cams = kameras

            def affine(self):
                return (Rotation.from_euler("z", self.yaw).as_matrix(),
                        np.asarray(self.t, float))

            def rgb_cams(self):
                return kameras

            def thermal_cams(self):
                return kameras if thermal else None

        return StubPipeline()

    # ------------------------------------------------------------ Zustand

    def labels(self) -> dict:
        """Texte aller QLabel, die als Attribut am Fenster hängen."""
        from PyQt5.QtWidgets import QLabel

        return {name: self.text(wert.text())
                for name, wert in sorted(vars(self.fenster).items())
                if isinstance(wert, QLabel)}

    def merke_labels(self, marke: str) -> None:
        """Zwischenstand der Label-Texte als Ereignis festhalten."""
        self.ereignis("labels", marke, self.labels())

    def _dateiliste(self) -> dict:
        out = {}
        for ordner, unter, namen in os.walk(self._arbeit):
            unter.sort()
            for name in sorted(namen):
                voll = os.path.join(ordner, name)
                try:
                    st = os.stat(voll)
                except OSError:
                    continue
                out[os.path.relpath(voll, self._arbeit)] = (st.st_size, st.st_mtime_ns)
        return out

    def dateien_merken(self) -> None:
        """Ab hier zählen neue und geänderte Dateien als vom Ablauf geschrieben."""
        self._stand = self._dateiliste()

    def cache_kopie(self):
        """Frische Kopie des seg0-Projekts im Fall-Cache; zählt nicht als geschrieben."""
        kopie = basis.cache_kopie()
        self.dateien_merken()
        return kopie

    def _json_neutral(self, wert):
        if isinstance(wert, dict):
            return {k: ("<flüchtig>" if k in self._json_fluechtig else self._json_neutral(v))
                    for k, v in wert.items()}
        if isinstance(wert, list):
            return [self._json_neutral(v) for v in wert]
        if isinstance(wert, str):
            return self.text(wert)
        return wert

    def dateien(self) -> dict:
        """Seit :meth:`dateien_merken` geschriebene Dateien: JSON als dict, sonst Hash."""
        out = {}
        for rel, marke in self._dateiliste().items():
            if self._stand.get(rel) == marke:
                continue
            voll = os.path.join(self._arbeit, rel)
            name = _SCHLUESSEL.sub(r"\1-<schlüssel>", rel)
            if rel.endswith(".json"):
                try:
                    with open(voll, encoding="utf-8") as fh:
                        out[name] = {"json": self._json_neutral(json.load(fh))}
                    continue
                except (OSError, ValueError):
                    pass
            out[name] = {"hash": basis.datei_hash(voll), "bytes": marke[0]}
        for rel in sorted(set(self._stand) - set(self._dateiliste())):
            out[_SCHLUESSEL.sub(r"\1-<schlüssel>", rel)] = {"geloescht": True}
        return out

    def abschluss(self) -> dict:
        self.laufe()
        for art, schlange in sorted(self.dialoge._antworten.items()):
            if schlange:
                self.ereignis("antworten übrig", art, len(schlange))
        return {"ereignisse": self.ereignisse, "labels": self.labels(),
                "dateien": self.dateien(), "blattaufrufe": self.blattaufrufe}


# ------------------------------------------------------------------- Laden

def familien_finden(ordner: str) -> dict:
    out = {}
    for datei in sorted(os.listdir(ordner)):
        if datei.startswith("ablaeufe_") and datei.endswith(".py"):
            out[datei[len("ablaeufe_"):-3]] = os.path.join(ordner, datei)
    return out


def familie_laden(familie: str, pfad: str) -> dict:
    name = f"ablaeufe_{familie}"
    spec = importlib.util.spec_from_file_location(name, pfad)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    faelle = getattr(mod, "FAELLE", None)
    if not isinstance(faelle, dict) or not faelle:
        raise RuntimeError(f"{pfad}: FAELLE fehlt oder ist leer.")
    return faelle


def _wahl(nur: str | None, geladen: dict) -> dict:
    """``--nur``: Familien ganz (None) oder einzelne Fälle je Familie."""
    if not nur:
        return {f: None for f in geladen}
    wahl: dict = {}
    for teil in (t.strip() for t in nur.split(",")):
        if not teil:
            continue
        if teil in geladen:
            wahl[teil] = None
            continue
        fam, _, fall = teil.partition(".")
        treffer = ([fam] if fam in geladen and fall in geladen[fam] else
                   [f for f in geladen if teil in geladen[f]])
        if not treffer:
            print(f"--nur {teil}: weder Familie noch Fall "
                  f"(Familien: {', '.join(geladen) or 'keine'}).")
            raise SystemExit(2)
        for f in treffer:
            if wahl.get(f, ()) is not None:
                wahl.setdefault(f, set()).add(fall if f == fam and fall in geladen[f]
                                              else teil)
    return wahl


def fahre(faelle: dict, namen=None) -> dict:
    """Fährt die Fälle, jeden mit frischem Fenster: ``{name: aufzeichnung}``."""
    out = {}
    for name in faelle:
        if namen is not None and name not in namen:
            continue
        sitzung = None
        try:
            with Sitzung(name) as sitzung:
                faelle[name](sitzung)
                out[name] = sitzung.abschluss()
            if sitzung.ausnahmen:
                mehr = len(sitzung.ausnahmen) - 1
                out[name] = {"fehler": f"{sitzung.ausnahmen[0]} (unbehandelt in einem Slot"
                                       f"{f', {mehr} weitere' if mehr else ''})"}
        except Exception as exc:  # noqa: BLE001 — der Lauf nennt ihn als Fehler
            traceback.print_exc()
            out[name] = {"fehler": f"{exc.__class__.__name__}: {exc}"}
        # durch JSON schicken: Tupel werden Listen, wie beim Lesen des Vorher-Stands
        out[name] = json.loads(json.dumps(out[name], ensure_ascii=False))
        sitzung = None
        gc.collect()
    return out


def fahre_familie(args, familie: str, namen=None) -> dict:
    """Fährt eine Familie in einem Kindprozess: ``{name: aufzeichnung}``.

    Endet das Kind durch ein Signal, wird es einmal wiederholt. Seine
    Arbeitsordner entstehen im Ordner dieses Aufrufs und werden mit ihm
    geräumt, auch die eines abgestürzten Laufs.
    """
    ordner = basis.arbeitsordner("ablauf")
    auftrag = os.path.join(ordner, "auftrag.json")
    ergebnis = os.path.join(ordner, "ergebnis.json")
    with open(auftrag, "w", encoding="utf-8") as fh:
        json.dump({"familie": familie, "ergebnis": ergebnis,
                   "namen": None if namen is None else sorted(namen)}, fh)
    befehl = [sys.executable, os.path.abspath(__file__), "--wurzel", basis.wurzel(),
              "--faelle-ordner", args.faelle_ordner, "--kind", auftrag]
    umgebung = dict(os.environ, **{basis.ELTERN_VARIABLE: ordner})
    try:
        for versuch in (1, 2):
            sys.stdout.flush()
            code = subprocess.run(befehl, env=umgebung).returncode
            if code == 0 and os.path.isfile(ergebnis):
                with open(ergebnis, encoding="utf-8") as fh:
                    stand = json.load(fh)
                for name, text in sorted(stand["nicht_ladbar"].items()):
                    if name not in _gemeldet:
                        _gemeldet.add(name)
                        print(f"Hinweis: {name} nicht ladbar ({text})")
                return stand["faelle"]
            if code >= 0 or versuch == 2:
                raise RuntimeError(
                    f"Familie {familie}: der Kindprozess endete mit "
                    f"{f'Signal {-code}' if code < 0 else f'Ende {code}'}"
                    f"{' (auch beim zweiten Versuch)' if versuch == 2 else ''}.")
            print(f"Hinweis: Familie {familie}: der Kindprozess endete mit Signal "
                  f"{-code}; die Familie wird einmal wiederholt.")
            for rest in os.listdir(ordner):
                if rest != os.path.basename(auftrag):
                    basis.raeume(os.path.join(ordner, rest))
    finally:
        if ordner in basis._raeumen:
            basis.raeume(ordner)
    raise AssertionError("nicht erreichbar")


_gemeldet: set = set()


def _kind(args) -> int:
    """Der Kindprozess von :func:`fahre_familie`."""
    with open(args.kind, encoding="utf-8") as fh:
        auftrag = json.load(fh)
    basis.wurzel_setzen(args.wurzel)
    # QApplication vor den Modulen: cv2 verbiegt sonst die Plugin-Suche von Qt
    _qt_app()
    nicht_ladbar = basis.module_laden()
    familie = auftrag["familie"]
    faelle = familie_laden(familie, familien_finden(args.faelle_ordner)[familie])
    aufz = fahre(faelle, auftrag["namen"])
    with open(auftrag["ergebnis"] + ".tmp", "w", encoding="utf-8") as fh:
        json.dump({"faelle": aufz, "nicht_ladbar": nicht_ladbar}, fh, ensure_ascii=False)
    os.replace(auftrag["ergebnis"] + ".tmp", auftrag["ergebnis"])
    return 0


def _ziel(ziel: str, familie: str) -> str:
    return os.path.join(ziel, f"ablaeufe_{familie}.json")


def _zeile(name: str, aufz: dict) -> str:
    if "fehler" in aufz:
        return f"  {name}: FEHLER {aufz['fehler']}"
    return (f"  {name}: {aufz['blattaufrufe']} Blattaufrufe, "
            f"{len(aufz['ereignisse'])} Ereignisse, {len(aufz['dateien'])} Dateien")


# -------------------------------------------------- Schreiben, Vergleichen

def schreibe(args, geladen: dict, wahl: dict) -> int:
    for familie in wahl:
        pfad = _ziel(args.ziel, familie)
        if os.path.exists(pfad) and not args.ersetzen:
            print(f"{familie}: Ziel vorhanden ({os.path.basename(pfad)}) — nur mit --ersetzen.")
            return 2
    for familie in wahl:
        aufz = fahre_familie(args, familie)
        print(f"Familie {familie}: {len(aufz)} Fälle")
        for name, a in aufz.items():
            print(_zeile(name, a))
        if any("fehler" in a for a in aufz.values()):
            print(f"{familie}: nicht geschrieben, ein Fall scheitert.")
            return 2
        os.makedirs(args.ziel, exist_ok=True)
        pfad = _ziel(args.ziel, familie)
        with open(pfad + ".tmp", "w", encoding="utf-8") as fh:
            json.dump({"familie": familie, "faelle": aufz}, fh, indent=1,
                      sort_keys=True, ensure_ascii=False)
            fh.write("\n")
        os.replace(pfad + ".tmp", pfad)
        print(f"{familie}: geschrieben nach {basis.pfad_neutral(pfad)}")
    return 0


def _kurztext(wert, laenge: int = 220) -> str:
    """Wert als Text; ein gekürzter trägt den Hash des ganzen Texts."""
    text = json.dumps(wert, ensure_ascii=False, sort_keys=True)
    if len(text) <= laenge:
        return text
    return f"{text[:laenge]} … #{basis.hash_wert(text)[:12]}"


def aufzeichnung_hash(aufz: dict) -> str:
    """Hash einer Aufzeichnung, an den ``--erwartet`` einen Fall bindet."""
    return basis.hash_wert(json.dumps(aufz, ensure_ascii=False, sort_keys=True))


def _kanon(wert) -> str:
    """Kanonischer JSON-Text: trennt auch true von 1 und 10 von 10.0."""
    return json.dumps(wert, ensure_ascii=False, sort_keys=True)


def _stellen(pfad: str, alt, neu) -> list:
    """Die abweichenden Stellen zweier Ereignisse gleicher Art, bis ins dict."""
    if isinstance(alt, dict) and isinstance(neu, dict):
        out = []
        for k in sorted(set(alt) | set(neu)):
            if k not in alt or k not in neu:
                out.append(f"{pfad}[{k}]: alt {_kurztext(alt.get(k, '(fehlt)'))} | "
                           f"neu {_kurztext(neu.get(k, '(fehlt)'))}")
            else:
                out += _stellen(f"{pfad}[{k}]", alt[k], neu[k])
        return out
    if isinstance(alt, list) and isinstance(neu, list) and len(alt) == len(neu):
        out = []
        for k, (a, n) in enumerate(zip(alt, neu)):
            out += _stellen(f"{pfad}[{k}]", a, n)
        return out
    if _kanon(alt) != _kanon(neu):
        return [f"{pfad}: alt {_kurztext(alt)} | neu {_kurztext(neu)}"]
    return []


def unterschiede(alt: dict, neu: dict) -> list:
    """Unterschiede zweier Aufzeichnungen eines Falls als Textzeilen, alle."""
    out = []
    if "fehler" in alt or "fehler" in neu:
        if alt.get("fehler") != neu.get("fehler"):
            out.append(f"Fehler: alt {alt.get('fehler')!r}, neu {neu.get('fehler')!r}")
        return out
    # Verglichen wird überall der kanonische JSON-Text, nicht mit ``!=``: für
    # Python gilt True == 1 und 10 == 10.0, in einer meta.json nicht.
    ea, en = alt["ereignisse"], neu["ereignisse"]
    ta, tn = [_kanon(e) for e in ea], [_kanon(e) for e in en]
    if ta != tn:
        # als Folgen abgleichen: ein eingefügtes oder entferntes Ereignis
        # verschiebt die übrigen nicht; genannt wird jede abweichende Stelle
        folge = difflib.SequenceMatcher(None, ta, tn, autojunk=False)
        for art, i1, i2, j1, j2 in folge.get_opcodes():
            if art == "equal":
                continue
            if art == "replace" and i2 - i1 == j2 - j1:
                for d in range(i2 - i1):
                    i, j = i1 + d, j1 + d
                    ort = f"{i}" if i == j else f"{i}→{j}"
                    if ea[i][:1] == en[j][:1]:
                        out += _stellen(f"ereignisse[{ort}]", ea[i], en[j])
                    else:
                        out.append(f"ereignisse[{ort}]: alt {_kurztext(ea[i])} | "
                                   f"neu {_kurztext(en[j])}")
                continue
            for i in range(i1, i2):
                out.append(f"ereignisse[{i}]: alt {_kurztext(ea[i])} | neu (fehlt)")
            for j in range(j1, j2):
                out.append(f"ereignisse[→{j}]: alt (fehlt) | neu {_kurztext(en[j])}")
    for teil in ("labels", "dateien"):
        for key in sorted(set(alt[teil]) | set(neu[teil])):
            a, n = alt[teil].get(key, "(fehlt)"), neu[teil].get(key, "(fehlt)")
            if _kanon(a) != _kanon(n):
                if isinstance(a, dict) and isinstance(n, dict) and "json" in a and "json" in n \
                        and isinstance(a["json"], dict) and isinstance(n["json"], dict):
                    for k in sorted(set(a["json"]) | set(n["json"])):
                        va, vn = a["json"].get(k, "(fehlt)"), n["json"].get(k, "(fehlt)")
                        if _kanon(va) != _kanon(vn):
                            out.append(f"{teil}[{key}][{k}]: alt {_kurztext(va)} | "
                                       f"neu {_kurztext(vn)}")
                else:
                    out.append(f"{teil}[{key}]: alt {_kurztext(a)} | neu {_kurztext(n)}")
    if _kanon(alt["blattaufrufe"]) != _kanon(neu["blattaufrufe"]):
        out.append(f"blattaufrufe: alt {alt['blattaufrufe']} | neu {neu['blattaufrufe']}")
    return out


def erwartet_laden(pfad: str) -> dict:
    """Liest die Datei für ``--erwartet`` und prüft ihre Form."""
    with open(pfad, encoding="utf-8") as fh:
        erwartet = json.load(fh)
    if not isinstance(erwartet, dict):
        raise RuntimeError("--erwartet: die Datei muss ein Objekt {fall: eintrag} enthalten.")
    for key, eintrag in erwartet.items():
        if isinstance(eintrag, str):
            raise RuntimeError(
                f"--erwartet '{key}': ein bloßer Grund gibt den ganzen Fall frei und "
                f"wird nicht mehr angenommen. Verlangt ist {{\"grund\": …, \"hash\": …}} "
                f"oder {{\"grund\": …, \"zeilen\": […]}}; --erwartet-vorlage schreibt "
                f"die Einträge.")
        if not isinstance(eintrag, dict) or set(eintrag) - {"grund", "hash", "zeilen"}:
            raise RuntimeError(f"--erwartet '{key}': erlaubt sind grund, hash und zeilen.")
        grund = eintrag.get("grund")
        if not isinstance(grund, str) or not grund.strip():
            raise RuntimeError(f"--erwartet '{key}': der Grund fehlt.")
        zeilen = eintrag.get("zeilen")
        if "hash" not in eintrag and zeilen is None:
            raise RuntimeError(f"--erwartet '{key}': weder hash noch zeilen — so wäre "
                               f"der Eintrag an keinen Stand gebunden.")
        if "hash" in eintrag and not isinstance(eintrag["hash"], str):
            raise RuntimeError(f"--erwartet '{key}': hash muss ein Text sein.")
        if zeilen is not None and not (isinstance(zeilen, list)
                                       and all(isinstance(z, str) for z in zeilen)):
            raise RuntimeError(f"--erwartet '{key}': zeilen muss eine Liste von Texten sein.")
    return erwartet


def _eintrag_fuer(erwartet: dict, familie: str, name: str):
    """Schlüssel des Eintrags für einen Fall: der wörtliche vor einem Muster."""
    for key in (f"{familie}.{name}", name):
        if key in erwartet:
            return key
    return next((k for k in erwartet
                 if fnmatch.fnmatchcase(f"{familie}.{name}", k)
                 or fnmatch.fnmatchcase(name, k)), None)


def vergleiche(args, geladen: dict, wahl: dict) -> int:
    erwartet = erwartet_laden(args.erwartet) if args.erwartet else {}
    benutzt: set = set()
    getroffen: dict = {}         # Schlüssel -> eingetretene erwartete Zeilen
    vorlage: dict = {}
    n_faelle = n_ab = n_erwartet = 0
    for familie, namen in wahl.items():
        pfad = _ziel(args.ziel, familie)
        if not os.path.isfile(pfad):
            print(f"{familie}: kein Vorher-Stand in {basis.pfad_neutral(args.ziel)} "
                  f"— erst --schreibe.")
            return 2
        with open(pfad, encoding="utf-8") as fh:
            alt = json.load(fh)["faelle"]
        neu = fahre_familie(args, familie, namen)
        alle = [n for n in list(neu) + sorted(set(alt) - set(neu))
                if namen is None or n in namen]
        print(f"Familie {familie}: {len(alle)} Fälle")
        for name in alle:
            n_faelle += 1
            if name not in neu:
                zeilen = [f"fehlt in ablaeufe_{familie}.py, steht aber im Vorher-Stand"]
            elif name not in alt:
                zeilen = ["neu, steht nicht im Vorher-Stand"]
            else:
                zeilen = unterschiede(alt[name], neu[name])
            if name in neu:
                print(_zeile(name, neu[name]) + (" — gleich" if not zeilen else ""))
            if not zeilen:
                continue
            stand = aufzeichnung_hash(neu[name]) if name in neu else "fehlt"
            key = _eintrag_fuer(erwartet, familie, name)
            eintrag = erwartet.get(key, {})
            gescheitert = name in neu and "fehler" in neu[name]
            if not gescheitert:
                vorlage[f"{familie}.{name}"] = {"grund": eintrag.get("grund", ""),
                                                "hash": stand, "zeilen": zeilen}
            gilt: set = set()
            if key is not None:
                benutzt.add(key)
                if gescheitert:
                    # wie in numerik_probe: ein Fehler ist nie eine gewollte Abweichung
                    zeilen = zeilen + [
                        f"--erwartet '{key}' kann einen Fehler nicht freigeben"]
                elif "hash" in eintrag:
                    if eintrag["hash"] == stand:
                        gilt = set(zeilen)
                    else:
                        zeilen = zeilen + [
                            f"--erwartet '{key}' nennt einen anderen Stand "
                            f"(#{eintrag['hash'][:12]}, die neue Aufzeichnung trägt "
                            f"#{stand[:12]})"]
                else:
                    gilt = set(zeilen) & set(eintrag["zeilen"])
                    getroffen.setdefault(key, set()).update(gilt)
            offen = [z for z in zeilen if z not in gilt]
            if offen:
                n_ab += 1
            else:
                n_erwartet += 1
            for z in zeilen:
                marke = (f"ERWARTET ({eintrag['grund']})" if z in gilt else "ABWEICHUNG")
                print(f"{marke} {familie}: Fall {name}, {z}")
    for key in sorted(set(erwartet) - benutzt):
        print(f"Hinweis: erwartete Abweichung '{key}' ist nicht eingetreten.")
    for key in sorted(getroffen):
        for z in erwartet[key]["zeilen"]:
            if z not in getroffen[key]:
                print(f"Hinweis: erwartete Zeile von '{key}' ist nicht eingetreten: {z}")
    if args.erwartet_vorlage:
        with open(args.erwartet_vorlage, "w", encoding="utf-8") as fh:
            json.dump(vorlage, fh, indent=1, sort_keys=True, ensure_ascii=False)
            fh.write("\n")
        print(f"Vorlage für --erwartet geschrieben: {len(vorlage)} Fälle "
              f"({basis.pfad_neutral(args.erwartet_vorlage)}); der Grund ist je "
              f"Eintrag nachzutragen.")
    if n_ab:
        print(f"{n_ab} von {n_faelle} Fällen weichen ab"
              f"{f', {n_erwartet} weitere wie erwartet' if n_erwartet else ''}.")
        return 1
    print(f"alle Abläufe gleich ({n_faelle} Fälle"
          f"{f', davon {n_erwartet} mit erwarteter Abweichung' if n_erwartet else ''})")
    return 0


# -------------------------------------------------------------- Selbsttest

_DEMO_FAELLE = '''\
"""Demo-Fälle für den Selbsttest der Ablauf-Probe."""
import json
import os

import numpy as np
from PyQt5.QtWidgets import QMessageBox

from core import stitcher


def _kette(s):
    f = s.fenster
    s.ersetze(stitcher._pano_rays, ergebnis=np.arange(3.0))
    punkte = np.random.default_rng(3).uniform(-2, 2, size=(200, 3)).astype(np.float32)

    def job(progress_cb, cancel, log_cb):
        progress_cb(0.25, "erster Teil")
        log_cb("Demo rechnet")
        wert = stitcher._pano_rays(64, height=32)
        with open(os.path.join(s.ordner, "ergebnis.json"), "w", encoding="utf-8") as fh:
            json.dump({"summe": float(wert.sum()), "ordner": s.ordner}, fh)
        punkte.tofile(os.path.join(s.ordner, "punkte.bin"))
        progress_cb(1.0 / 3.0, "zweiter Teil")
        return wert

    def folge(progress_cb, cancel, log_cb):
        progress_cb(1.0, "Folge fertig")
        return {"n": 3, "pfad": s.ordner}

    def danach(wert):
        f._log(f"Demo fertig: {wert.sum():.1f}")
        QMessageBox.information(f, "Demo", f"Schritt beendet in {s.ordner}")
        f._cloud_view.set_cloud(punkte)
        f._start_worker("Demo-Folgeschritt", folge, lambda r: f._log("Folge da"),
                        cancellable=False)

    f._start_worker("Demo-Schritt", job, danach)
    assert f._busy and s.wartend() == 1      # der Job wartet, bis der Handler zurück ist
    f._log("Handler ist zurück")


def _fehlschlag(s):
    def job(progress_cb, cancel, log_cb):
        raise RuntimeError("geht nicht")

    s.fenster._start_worker("Demo-Fehlschlag", job, lambda r: None)


def _abbruch(s):
    def job(progress_cb, cancel, log_cb):
        for i in range(5):
            if cancel.is_set():
                raise RuntimeError("Abgebrochen")
            progress_cb(i / 5.0, f"Runde {i}")

    s.abbruch_nach(2)
    s.fenster._start_worker("Demo-Abbruch", job, lambda r: None)
    s.laufe()
    s.merke_labels("nach dem Abbruch")


def _stubs(s):
    f = s.fenster
    projekt = s.stub_projekt()
    rec = s.stub_aufzeichnung(ordner=projekt.recording_dir())
    bag = s.stub_bag()
    pipe = s.stub_pipeline(thermal=True)
    assert rec.n_scans == 4 and rec.n_points == 200 and projekt.has_recording()
    assert bag.info().duration == 5.0 and len(bag.camera_stamps()) == 5
    assert pipe.affine()[0].shape == (3, 3)
    s.dateien_merken()
    f._project = projekt
    f._save_settings()                      # schreibt settings.json in den Fall-Cache
    s.antworte("QMessageBox.question", QMessageBox.No)
    antwort = QMessageBox.question(f, "Demo", "Weiter?")
    s.ereignis("antwort", int(antwort == QMessageBox.No), s.kurz((rec.points, bag, pipe)))


def _zweitstart(s):
    f = s.fenster
    f._start_worker("Demo wartet", lambda progress_cb, cancel, log_cb: 1, lambda r: None)

    def job(progress_cb, cancel, log_cb):
        log_cb("Zweiter Job rechnet")
        return 2

    gestartet = s.echter_start("Demo-Zweitstart", job, lambda r: f._log(f"Zweiter da: {r}"))
    s.ereignis("zweitstart", bool(gestartet))
    QMessageBox.information(f, "Demo", f"Projekt in {s.cache_root}/bag_demo-0123abcd liegt da")


FAELLE = {"kette": _kette, "fehlschlag": _fehlschlag, "abbruch": _abbruch,
          "stubs": _stubs, "zweitstart": _zweitstart}
'''

#: Ein Fall, dessen Prozess durch ein Signal endet — beim ersten Mal oder immer.
_DEMO_ABSTURZ = '''\
"""Demo-Fall für den Selbsttest: der Prozess endet durch ein Signal."""
import os
import signal


def _absturz(s):
    marke = os.environ["ABLAUF_DEMO_MARKE"]
    with open(os.path.join(s.ordner, "liegengeblieben.bin"), "wb") as fh:
        fh.write(b"x")
    if os.environ.get("ABLAUF_DEMO_IMMER") or not os.path.exists(marke):
        open(marke, "w").close()
        os.kill(os.getpid(), signal.SIGKILL)
    s.fenster._log("zweiter Versuch läuft durch")


FAELLE = {"absturz": _absturz}
'''

#: Ein Fall, der auf Wunsch mit einem Fehler endet.
_DEMO_SCHEITERT = '''\
"""Demo-Fall für den Selbsttest: endet auf Wunsch mit einem Fehler."""
import os


def _scheitert(s):
    if os.environ.get("ABLAUF_DEMO_SCHEITERT"):
        raise RuntimeError("absichtlich")
    s.fenster._log("läuft durch")


FAELLE = {"scheitert": _scheitert}
'''

#: Dem Fenster fehlt auf Wunsch, was zwei der drei Fälle brauchen.
_DEMO_ENTFERNT = '''\
"""Demo-Fälle für den Selbsttest: dem Fenster fehlen Handler und Widget."""
import os

from PyQt5.QtCore import QTimer

_FEHLT = bool(os.environ.get("ABLAUF_DEMO_ENTFERNT"))


def _handler_fehlt(s):
    if _FEHLT:
        s.fenster._on_demo_gibt_es_nicht()
    s.fenster._log("Handler gerufen")


def _slot_scheitert(s):
    if _FEHLT:
        QTimer.singleShot(0, lambda: s.fenster._spin_demo_gibt_es_nicht.setValue(1))
    s.warte(20)
    s.fenster._log("Slot gelaufen")


def _danach(s):
    s.fenster._log("läuft weiter")


FAELLE = {"handler_fehlt": _handler_fehlt, "slot_scheitert": _slot_scheitert,
          "danach": _danach}
'''

#: Absichtliche Änderungen für die veränderte Kopie: (alt, neu) in ui/**/*.py
_MUTATIONEN = (('"Bereit")', '"Fertig")'),
               ('f"Abgebrochen — {title}"', 'f"Abbruch — {title}"'),
               ('setText("Abgebrochen.")', 'setText("Abbruch.")'))


def _veraenderte_kopie(ziel: str) -> int:
    quelle = basis.wurzel()
    shutil.copytree(quelle, ziel, ignore=shutil.ignore_patterns(
        ".git", "__pycache__", "vorher", "*.pyc"))
    n = 0
    for ordner, _unter, namen in os.walk(os.path.join(ziel, "ui")):
        for name in namen:
            if not name.endswith(".py"):
                continue
            pfad = os.path.join(ordner, name)
            with open(pfad, encoding="utf-8") as fh:
                text = fh.read()
            neu = text
            for alt, ersatz in _MUTATIONEN:
                n += neu.count(alt)
                neu = neu.replace(alt, ersatz)
            if neu != text:
                with open(pfad, "w", encoding="utf-8") as fh:
                    fh.write(neu)
    return n


def _selbsttest(args) -> int:
    basis.wurzel_setzen(args.wurzel)
    ordner = basis.arbeitsordner("ablauf_selbst")
    faelle = os.path.join(ordner, "faelle")
    os.makedirs(faelle)
    with open(os.path.join(faelle, "ablaeufe_demo.py"), "w", encoding="utf-8") as fh:
        fh.write(_DEMO_FAELLE)
    ziel = os.path.join(ordner, "vorher")
    kopie = os.path.join(ordner, "kopie")
    n_mut = _veraenderte_kopie(kopie)
    assert n_mut >= 1, "die Stellen für die veränderte Kopie gibt es in ui/ nicht mehr"

    # Arbeitsordner der Läufe und ihrer Kinder entstehen in diesem Ordner
    umgebung = dict(os.environ, **{basis.ELTERN_VARIABLE: ordner,
                                   "ABLAUF_DEMO_MARKE": os.path.join(ordner, "marke")})
    umgebung.pop("ABLAUF_DEMO_IMMER", None)
    umgebung.pop("SUPER360_UMBAU_BEHALTEN", None)

    def lauf(*argumente, wurzel=basis.wurzel(), faelle=faelle, **zusatz):
        befehl = [sys.executable, os.path.abspath(__file__), "--wurzel", wurzel,
                  "--faelle-ordner", faelle, "--ziel", ziel, *argumente]
        p = subprocess.run(befehl, capture_output=True, text=True,
                           env=dict(umgebung, **zusatz))
        return p.returncode, p.stdout + p.stderr

    def reste():
        return sorted(n for n in os.listdir(ordner)
                      if n not in ("faelle", "absturz", "scheitert", "entfernt", "vorher",
                                   "kopie", "marke")
                      and not n.endswith(".json"))

    code, aus = lauf("--schreibe")
    assert code == 0 and "Familie demo: 5 Fälle" in aus, aus
    assert "kette: 1 Blattaufrufe" in aus, aus
    with open(os.path.join(ziel, "ablaeufe_demo.json"), encoding="utf-8") as fh:
        roh = fh.read()
    assert "/tmp/" not in roh and "/home/" not in roh, "Pfad in der Aufzeichnung"
    stand = json.loads(roh)["faelle"]
    kette = stand["kette"]
    arten = [e[0] for e in kette["ereignisse"]]
    assert arten == ["arbeiter", "log", "log", "fortschritt", "log", "aufruf", "fortschritt",
                     "fertig", "log", "dialog", "ansicht", "arbeiter", "log", "fortschritt",
                     "fertig", "log"], arten
    assert kette["ereignisse"][0] == ["arbeiter", "Demo-Schritt", True, "frei"]
    assert kette["ereignisse"][2] == ["log", "Handler ist zurück"]
    assert kette["ereignisse"][5] == ["aufruf", "stitcher._pano_rays", [64], {"height": 32}]
    assert kette["ereignisse"][6] == ["fortschritt", 0.333333, "zweiter Teil"]
    assert kette["ereignisse"][9][1]["text"] == "Schritt beendet in <arbeit>/fall"
    assert kette["ereignisse"][10][1] == "set_cloud" \
        and kette["ereignisse"][10][2][0].startswith("ndarray <f4 (200, 3) #")
    assert kette["ereignisse"][11] == ["arbeiter", "Demo-Folgeschritt", False, "frei"]
    assert kette["ereignisse"][14][2] == {"n": 3, "pfad": "<arbeit>/fall"}
    assert kette["labels"]["_status_lbl"] == "Bereit", kette["labels"]
    assert kette["dateien"]["fall/ergebnis.json"] == {
        "json": {"summe": 3.0, "ordner": "<arbeit>/fall"}}
    assert kette["dateien"]["fall/punkte.bin"]["bytes"] == 2400
    fehl = stand["fehlschlag"]["ereignisse"]
    assert ["fehlgeschlagen", "Demo-Fehlschlag", "RuntimeError", "geht nicht"] in fehl
    assert fehl[-1][0] == "dialog" and fehl[-1][1]["art"] == "QMessageBox.critical"
    ab = stand["abbruch"]["ereignisse"]
    assert ["abbruch"] in ab and ["log", "Abgebrochen — Demo-Abbruch"] in ab
    assert ab[-1][0] == "labels" and ab[-1][2]["_status_lbl"] == "Abgebrochen."
    stubs = stand["stubs"]
    einst = [k for k in stubs["dateien"] if k.endswith("settings.json")]
    assert einst == ["cache/bag_demo-<schlüssel>/settings.json"], stubs["dateien"]
    assert len(stubs["dateien"]) == 1, "Stub-Eingaben zählen nicht als geschrieben"
    assert stubs["ereignisse"][-1][1] == 1
    zweit = stand["zweitstart"]["ereignisse"]
    echt = [e for e in zweit if e[0] == "echter start"]
    assert len(echt) == 1 and echt[0][1:3] == ["Demo-Zweitstart", "lief schon einer"] \
        and echt[0][3] in ("gestartet", "abgelehnt"), zweit
    assert ["zweitstart", echt[0][3] == "gestartet"] in zweit, zweit
    assert (["log", "Zweiter Job rechnet"] in zweit) == (echt[0][3] == "gestartet"), zweit
    dialog = [e[1] for e in zweit if e[0] == "dialog"][-1]
    assert dialog["text"] == "Projekt in <arbeit>/cache/bag_demo-<schlüssel> liegt da", dialog
    print(f"schreibe: 5 Demo-Fälle aufgezeichnet (Arbeiter samt Folge-Arbeiter, "
          f"Fortschritt, Log, Dialog, Ansicht, Labels, Dateien; "
          f"{len(stubs['dateien'][einst[0]]['json'])} Einstellungsschlüssel als dict)")
    print(f"schreibe: echtes _start_worker bei laufendem Schritt — zweiter Arbeiter "
          f"{echt[0][3]}; Dialogtext ohne Projektschlüssel")

    code, aus = lauf("--schreibe")
    assert code == 2 and "--ersetzen" in aus, aus
    print("schreibe: vorhandenes Ziel verweigert")

    for _ in range(2):
        code, aus = lauf("--vergleiche")
        assert code == 0 and "alle Abläufe gleich (5 Fälle)" in aus, aus
    print("vergleiche: zweimal gleich")

    code, aus = lauf("--vergleiche", "--nur", "fehlschlag")
    assert code == 0 and "alle Abläufe gleich (1 Fälle)" in aus, aus
    code, aus = lauf("--vergleiche", "--nur", "gibtsnicht")
    assert code == 2, aus
    print("vergleiche: --nur wählt Familie oder Fall")

    # typgenau: für Python ist True == 1 und 10 == 10.0, für den Vergleich nicht
    def typfall(ereignis, meta, label="1"):
        return {"ereignisse": [["log", "vorweg"], ereignis], "labels": {"_lbl": label},
                "dateien": {"fall/meta.json": {"json": meta},
                            "fall/roh.bin": {"hash": "00", "bytes": 10}},
                "blattaufrufe": 1}

    alt = typfall(["aufruf", "splat.lauf", [], {"halte_jedes": 10}],
                  {"sichtpruefung": True, "ebene": {"schritte": 10}})
    assert unterschiede(alt, json.loads(json.dumps(alt))) == []
    neu = typfall(["aufruf", "splat.lauf", [], {"halte_jedes": 10.0}],
                  {"sichtpruefung": 1, "ebene": {"schritte": 10.0}})
    assert alt == neu, "Python hält die beiden Aufzeichnungen für gleich"
    zeilen = unterschiede(alt, neu)
    assert zeilen == [
        "ereignisse[1][3][halte_jedes]: alt 10 | neu 10.0",
        'dateien[fall/meta.json][ebene]: alt {"schritte": 10} | neu {"schritte": 10.0}',
        "dateien[fall/meta.json][sichtpruefung]: alt true | neu 1"], zeilen
    neu = json.loads(json.dumps(alt))
    neu["dateien"]["fall/roh.bin"]["bytes"] = 10.0
    neu["blattaufrufe"] = True
    zeilen = unterschiede(alt, neu)
    assert len(zeilen) == 2 and zeilen[0].startswith("dateien[fall/roh.bin]: alt ") \
        and zeilen[1] == "blattaufrufe: alt 1 | neu True", zeilen
    print("vergleiche: typgenau — true gegen 1 und 10 gegen 10.0 sind Abweichungen "
          "(Blattaufruf, meta-dict, Dateieintrag, Zähler)")

    # Kindprozess endet durch ein Signal: einmal wiederholen, nichts bleibt liegen
    absturz = os.path.join(ordner, "absturz")
    os.makedirs(absturz)
    with open(os.path.join(absturz, "ablaeufe_absturz.py"), "w", encoding="utf-8") as fh:
        fh.write(_DEMO_ABSTURZ)
    code, aus = lauf("--schreibe", faelle=absturz)
    assert code == 0 and "Signal 9; die Familie wird einmal wiederholt" in aus \
        and "absturz: 0 Blattaufrufe" in aus, aus
    assert not reste(), reste()
    code, aus = lauf("--vergleiche", faelle=absturz, ABLAUF_DEMO_IMMER="1")
    assert code == 2 and "auch beim zweiten Versuch" in aus \
        and "alle Abläufe gleich" not in aus, aus
    assert not reste(), reste()
    print("Kindprozess: Ende durch Signal einmal wiederholt, beim zweiten Mal Fehler; "
          "kein Arbeitsordner bleibt liegen")

    vorlage = os.path.join(ordner, "vorlage.json")
    code, aus = lauf("--vergleiche", "--erwartet-vorlage", vorlage, wurzel=kopie)
    zeilen = [z for z in aus.splitlines() if z.startswith("ABWEICHUNG")]
    assert code == 1 and zeilen and "alle Abläufe gleich" not in aus, aus
    assert any("labels[_status_lbl]" in z and "Fertig" in z for z in zeilen), aus
    # beide Änderungen im selben Fall werden genannt, nicht nur die erste
    im_abbruch = [z for z in zeilen if "Fall abbruch, " in z]
    assert any("ereignisse[7][1]" in z and "Abbruch — Demo-Abbruch" in z for z in im_abbruch) \
        and any("ereignisse[8][2][_status_lbl]" in z for z in im_abbruch) \
        and len(im_abbruch) == 3, aus
    print(f"vergleiche: veränderte Kopie ({n_mut} Stellen in ui/) gemeldet, jede "
          f"Stelle je Fall ({len(im_abbruch)} im Fall abbruch) —")
    for z in zeilen:
        print("   " + z[:170])

    erwartet = os.path.join(ordner, "erwartet.json")

    def mit_erwartet(eintraege):
        with open(erwartet, "w", encoding="utf-8") as fh:
            json.dump(eintraege, fh)
        return lauf("--vergleiche", "--erwartet", erwartet, wurzel=kopie)

    with open(vorlage, encoding="utf-8") as fh:
        vorl = json.load(fh)
    assert "/tmp/" not in json.dumps(vorl) and "demo.abbruch" in vorl, vorl
    code, aus = mit_erwartet(vorl)
    assert code == 2 and "der Grund fehlt" in aus, aus
    code, aus = mit_erwartet({"demo.*": "Statustext absichtlich geändert"})
    assert code == 2 and "gibt den ganzen Fall frei" in aus, aus
    gebunden = {k: {"grund": "Statustext absichtlich geändert", "hash": v["hash"]}
                for k, v in vorl.items()}
    gebunden["anderes.fall"] = {"grund": "tritt nicht ein", "hash": "0" * 64}
    code, aus = mit_erwartet(gebunden)
    assert code == 0 and "ERWARTET (Statustext absichtlich geändert)" in aus \
        and "ABWEICHUNG" not in aus and f"davon {len(vorl)} mit erwarteter" in aus \
        and "'anderes.fall' ist nicht eingetreten" in aus, aus
    print("vergleiche: --erwartet lässt an den neuen Stand gebundene Abweichungen gelten")

    # eine zweite Änderung im selben Fall bleibt ein Fehler: über Zeilen …
    nur_status = sorted({z for v in vorl.values() for z in v["zeilen"]
                         if z.startswith("labels[")})
    assert len(nur_status) == 2 and len(vorl["demo.abbruch"]["zeilen"]) == 3, vorl
    code, aus = mit_erwartet({"demo.*": {"grund": "nur der Statustext",
                                         "zeilen": nur_status + ["tritt nicht ein"]}})
    zeilen = [z for z in aus.splitlines() if z.startswith("ABWEICHUNG")]
    assert code == 1 and zeilen and all("Fall abbruch, ereignisse[" in z for z in zeilen) \
        and "ERWARTET (nur der Statustext) demo: Fall abbruch, labels[" in aus \
        and "1 von 5 Fällen weichen ab" in aus \
        and "erwartete Zeile von 'demo.*' ist nicht eingetreten: tritt nicht ein" in aus, aus
    # … und über den Hash: ein anderer Stand als der erwartete
    falsch = dict(gebunden)
    falsch["demo.abbruch"] = {"grund": "alter Stand", "hash": "f" * 64}
    code, aus = mit_erwartet(falsch)
    assert code == 1 and "nennt einen anderen Stand" in aus \
        and "1 von 5 Fällen weichen ab" in aus, aus
    print("vergleiche: eine weitere Abweichung im selben Fall bleibt trotz --erwartet "
          "ein Fehler (über Zeilen und über den Hash)")
    # ein Fall, der mit einem Fehler endet, lässt sich nicht freigeben
    scheitert = os.path.join(ordner, "scheitert")
    os.makedirs(scheitert)
    with open(os.path.join(scheitert, "ablaeufe_scheitert.py"), "w", encoding="utf-8") as fh:
        fh.write(_DEMO_SCHEITERT)
    code, aus = lauf("--schreibe", faelle=scheitert)
    assert code == 0, aus
    code, aus = lauf("--vergleiche", "--erwartet-vorlage", vorlage, faelle=scheitert,
                     ABLAUF_DEMO_SCHEITERT="1")
    zeile = "Fehler: alt None, neu 'RuntimeError: absichtlich'"
    assert code == 1 and f"ABWEICHUNG scheitert: Fall scheitert, {zeile}" in aus, aus
    with open(vorlage, encoding="utf-8") as fh:
        assert json.load(fh) == {}, "ein Fehler gehört nicht in die Vorlage"
    for bindung in ({"hash": aufzeichnung_hash({"fehler": "RuntimeError: absichtlich"})},
                    {"zeilen": [zeile]}):
        with open(erwartet, "w", encoding="utf-8") as fh:
            json.dump({"scheitert.scheitert": dict(bindung, grund="soll so sein")}, fh)
        code, aus = lauf("--vergleiche", "--erwartet", erwartet, faelle=scheitert,
                         ABLAUF_DEMO_SCHEITERT="1")
        assert code == 1 and "ERWARTET" not in aus \
            and "kann einen Fehler nicht freigeben" in aus \
            and "1 von 1 Fällen weichen ab" in aus, aus
    print("vergleiche: ein Fall, der mit einem Fehler endet, bleibt trotz --erwartet "
          "ein Fehler (über Hash und über Zeilen)")
    # fehlt dem Fenster ein Handler oder ein Widget, scheitert nur dieser Fall
    entfernt = os.path.join(ordner, "entfernt")
    os.makedirs(entfernt)
    with open(os.path.join(entfernt, "ablaeufe_entfernt.py"), "w", encoding="utf-8") as fh:
        fh.write(_DEMO_ENTFERNT)
    code, aus = lauf("--schreibe", faelle=entfernt)
    assert code == 0 and "Familie entfernt: 3 Fälle" in aus, aus
    code, aus = lauf("--vergleiche", faelle=entfernt)
    assert code == 0 and "alle Abläufe gleich (3 Fälle)" in aus, aus
    code, aus = lauf("--vergleiche", faelle=entfernt, ABLAUF_DEMO_ENTFERNT="1")
    zeilen = [z for z in aus.splitlines() if z.startswith("ABWEICHUNG")]
    assert code == 1 and len(zeilen) == 2 and "2 von 3 Fällen weichen ab" in aus, aus
    assert "Fall handler_fehlt, Fehler: alt None, neu " in zeilen[0] \
        and "AttributeError: 'MainWindow' object has no attribute " \
            "'_on_demo_gibt_es_nicht'" in zeilen[0], aus
    assert "Fall slot_scheitert, Fehler: alt None, neu " in zeilen[1] \
        and "has no attribute '_spin_demo_gibt_es_nicht' (unbehandelt in einem Slot)" \
        in zeilen[1], aus
    assert "  danach: 0 Blattaufrufe, 1 Ereignisse, 0 Dateien — gleich" in aus, aus
    print("vergleiche: fehlt dem Fenster ein Handler oder Widget (auch erst in einem "
          "Slot), ist das eine Abweichung dieses Falls mit dem fehlenden Namen; die "
          "übrigen Fälle laufen weiter")
    assert not reste(), reste()
    print("ablauf_probe SELFTEST OK")
    return 0


# ------------------------------------------------------------------ Aufruf

def main(argv=None) -> int:
    parser = basis.argumente(argparse.ArgumentParser(
        description="Ablauf-Proben der Handler gegen den Vorher-Stand"))
    modus = parser.add_mutually_exclusive_group()
    modus.add_argument("--schreibe", action="store_true")
    modus.add_argument("--vergleiche", action="store_true")
    modus.add_argument("--selbsttest", action="store_true")
    parser.add_argument("--ersetzen", action="store_true")
    parser.add_argument("--nur", default=None, metavar="FAMILIE,… | FALL,…")
    parser.add_argument("--erwartet", default=None, metavar="JSON")
    parser.add_argument("--erwartet-vorlage", default=None, metavar="JSON")
    parser.add_argument("--kind", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--faelle-ordner", default=basis.WERKZEUG_ORDNER)
    parser.add_argument("--ziel", default=basis.VORHER_ORDNER)
    args = parser.parse_args(argv)
    args.faelle_ordner = os.path.abspath(args.faelle_ordner)
    args.ziel = os.path.abspath(args.ziel)
    for name in ("erwartet", "erwartet_vorlage"):
        if getattr(args, name):
            setattr(args, name, os.path.abspath(getattr(args, name)))
    if args.kind:
        return _kind(args)
    if not (args.schreibe or args.vergleiche or args.selbsttest):
        parser.error("einer der Modi --schreibe, --vergleiche, --selbsttest ist verlangt")
    if args.selbsttest:
        return _selbsttest(args)

    basis.wurzel_setzen(args.wurzel)
    familien = familien_finden(args.faelle_ordner)
    if not familien:
        print(f"Keine ablaeufe_*.py in {basis.pfad_neutral(args.faelle_ordner)}.")
        return 2
    # QApplication vor den Modulen: cv2 verbiegt sonst die Plugin-Suche von Qt.
    # Hier werden nur die Namen der Fälle gebraucht; gefahren wird im Kind.
    _qt_app()
    code = 2                     # gilt auch für SystemExit und jede Ausnahme
    try:
        code = _lauf(args, familien)
        return code
    finally:
        # X-Verbindung vor dem Prozessende schließen; bei einem Ende ungleich 0
        # warten, bis Xvfb sein Signal an xvfb-run los ist (s. Kopf, »Ende 5«)
        sys.stdout.flush()
        _app_halter.clear()
        basis.qt_app_schliessen(warten=code != 0)


def _lauf(args, familien: dict) -> int:
    try:
        geladen = {f: familie_laden(f, p) for f, p in familien.items()}
        wahl = _wahl(args.nur, geladen)
        if args.schreibe:
            if any(n is not None for n in wahl.values()):
                print("--schreibe nimmt nur ganze Familien.")
                return 2
            return schreibe(args, geladen, wahl)
        return vergleiche(args, geladen, wahl)
    except RuntimeError as exc:
        print(f"FEHLER: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
