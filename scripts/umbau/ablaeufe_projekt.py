"""Ablauf-Proben der Familie »projekt«.

Projekt und Karte (Rosbag öffnen, Aufzeichnung laden, Karte berechnen,
Explorationsgrad, Projekt öffnen und exportieren), Zusammenführen,
360°-Einfärbung samt Kalibrierung, Export, RViz-Wiedergabe und das Schließen
des Fensters. Vertrag und Rekorder: ``ablauf_probe.py``.

Jeder Fall fährt die echten Handler des Hauptfensters. Ersetzt sind nur die
Blätter in ``core``, die ein Bag, die Kamera, ROS oder ein fremdes Programm
brauchen; sie werden über ihren Ursprungsort gesucht (``modul:name``), damit
die Fälle vor und nach dem Verschieben eines Namens dieselben bleiben. Alles
im Projektordner (Einstellungen, Extrinsik, Farbdateien, Aufzeichnung) läuft
echt im Arbeitsordner des Falls.

Die meisten Fälle brauchen einen offenen Flug. Den stellt :func:`_offen` über
den echten Weg her; die Aufzeichnung dieses Vorlaufs wird verworfen
(:class:`_Vorlauf`): festgehalten wird nur der Ablauf, um den es im Fall geht.
Das Öffnen selbst steht in den Fällen ``bag_*`` und ``aufzeichnung_*``.
"""
from __future__ import annotations

import json
import os
import shutil
import sys

import numpy as np

#: Zeitstempel in Logzeilen, meta.json und Manifest (Uhrzeit des Laufs)
_ZEIT = r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d"

#: Beginn der Scan-Stempel je Flug: der zweite liegt zeitlich hinter dem
#: ersten, sonst lehnt merge_recordings die Zusammenführung ab.
_T0 = {"flug_a": 1000.0, "flug_b": 2000.0}


# ------------------------------------------------------------------ Blätter

def _orte(*orte: str) -> list:
    """Die Objekte hinter ``modul:name`` — jedes einmal.

    Mehrere Orte für einen Namen, der umzieht: solange es ihn an beiden Orten
    als eigene Kopie gibt, werden beide ersetzt.
    """
    gefunden: list = []
    for ort in orte:
        modul, _, name = ort.partition(":")
        mod = sys.modules.get(modul)
        obj = getattr(mod, name, None) if mod is not None else None
        if obj is not None and all(obj is not g for g in gefunden):
            gefunden.append(obj)
    if not gefunden:
        raise RuntimeError(f"{', '.join(orte)}: nirgends gefunden.")
    return gefunden


def _ersetze(s, name: str, *orte: str, ergebnis=None, ersatz=None,
             echt: bool = False) -> None:
    """Blatt ersetzen; mit ``echt`` läuft das Original, der Aufruf wird nur notiert."""
    for obj in _orte(*orte):
        if echt:
            s.ersetze(obj, ersatz=obj, name=name)
        else:
            s.ersetze(obj, ergebnis=ergebnis, ersatz=ersatz, name=name)


def _melde(progress, f: float, m: str) -> None:
    """Fortschritt eines Blatts. Jedes meldet 0.0 und 1.0: an zwei Anteilen
    zeigt sich die Spanne, auf die der Handler den Fortschritt umrechnet."""
    if progress is not None:
        progress(f, m)


class _Welt:
    """Was die Blätter eines Falls liefern. Die Fälle stellen es vor dem Handler ein."""

    def __init__(self, s):
        self.s = s
        self.ohne_kamera: set = set()     # Bag-Namen ohne Kamera-Topic
        self.pano_fehler = False
        self.grad = "wert"                # "wert" | "keine" | "fehler"
        self.fastlio_fehler = False
        self.extrinsik = "gut"            # "gut" | "verdreht" | "fehler"
        self.farbe = "normal"             # "normal" | "leer" | "teil1_duenn_teil2_leer"
        self.autocal_score = 0.8
        self.ausrichtung = {"fitness": 0.82, "rmse": 0.045, "kandidat": "fgr-2"}
        self.cloudcompare = True
        self.player = None
        s.fluechtig(_ZEIT, "<zeit>")

    # --------------------------------------------------------------- Bags

    def bag(self, pfad):
        name = os.path.basename(os.path.abspath(str(pfad)))
        bag = self.s.stub_bag(mit_kamera=name not in self.ohne_kamera, name=name)
        bag.bag_path = str(pfad)
        return bag

    def bags(self) -> None:
        """ThreadLocalBag, BagReader, GPS-Bewertung und Explorationsgrad."""
        s, welt = self.s, self

        class Leser:
            _kurz = "StubLeser"

            def __init__(self, bag):
                self._bag = bag

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def info(self):
                return self._bag.info()

        _ersetze(s, "ThreadLocalBag", "ui.main_window:ThreadLocalBag",
                 "core.bag_reader:ThreadLocalBag", ersatz=self.bag)
        _ersetze(s, "BagReader", "core.bag_reader:BagReader",
                 ersatz=lambda pfad: Leser(welt.bag(pfad)))
        _ersetze(s, "georef.assess", "core.georef:assess", echt=True)

        from core import exploration

        def berechne(bag_path, voxel=None, progress=None, cancel=None):
            _melde(progress, 0.0, "Explorationsgrad: Scan 1/4")
            _melde(progress, 1.0, "Explorationsgrad: Scan 4/4")
            if welt.grad == "keine":
                raise exploration.KeineExplorationsdaten(
                    "Keine Explorationsbox im Bag (Stub).")
            if welt.grad == "fehler":
                raise ValueError("Bag beschädigt (Stub)")
            return _grad(bag_path)

        _ersetze(s, "exploration.berechne", "core.exploration:berechne", ersatz=berechne)

    def pano(self) -> None:
        welt = self

        class Quelle:
            _kurz = "StubPano"
            count = 3
            fps = 1.0
            stamps = 1000.0 + np.arange(3.0)

            def get_pano(self, idx):
                return np.full((32, 64, 3), 40 * int(idx), dtype=np.uint8)

        def fabrik(bag, calib, width, cache_dir):
            if welt.pano_fehler:
                raise RuntimeError("Kalibrierung passt nicht zur Bildgröße (Stub).")
            return Quelle()

        _ersetze(self.s, "StitchingPanoSource", "ui.pano_view:StitchingPanoSource",
                 ersatz=fabrik)

    # ------------------------------------------------------- Aufzeichnung

    def aufzeichnung(self) -> None:
        """Recording.load läuft echt (notiert), FAST-LIO schreibt eine Mini-Aufzeichnung."""
        s, welt = self.s, self
        from core.fastlio_runner import FastLioResult, FastLioRunner
        from core.recording import Recording

        s.ersetze_attribut(Recording, "load", ersatz=Recording.__dict__["load"].__func__)

        def lauf(runner, bag_path, out_dir, config=None, rate=1.0, progress_cb=None,
                 cancel=None, log_cb=None):
            progress_cb(0.25, "FAST-LIO2: Scan 1/4")
            log_cb("fast_lio: Initialisierung abgeschlossen (Stub)")
            if welt.fastlio_fehler:
                raise RuntimeError("fast_lio hat sich beendet (Stub).")
            name = os.path.basename(os.path.abspath(bag_path))
            _schreibe_aufzeichnung(s, out_dir, bag_path, seed=3, t0=_T0.get(name, 1000.0))
            progress_cb(1.0, "FAST-LIO2: Scan 4/4")
            return FastLioResult(recording_dir=out_dir, n_scans=4, n_points=200,
                                 expected_scans=5, dropped_scans=1, duration_s=12.5,
                                 log_tail="")

        s.ersetze_attribut(FastLioRunner, "run", ersatz=lauf)

    def grund(self) -> "_Welt":
        """Die Blätter, die jedes Öffnen braucht."""
        self.bags()
        self.pano()
        self.aufzeichnung()
        return self

    # ------------------------------------------------------ Zusammenführen

    def merge(self) -> None:
        s, welt = self.s, self

        def register(points_a, points_b, T_init=None, mode="auto", progress_cb=None,
                     cancel=None):
            _melde(progress_cb, 0.0, "ICP 1/4")
            _melde(progress_cb, 1.0, "ICP 4/4")
            T = np.eye(4)
            T[:3, 3] = [0.5, -0.25, 0.125]
            return {"T": T, **welt.ausrichtung}

        _ersetze(s, "merge.cloud_for_registration", "core.merge:cloud_for_registration",
                 echt=True)
        _ersetze(s, "merge.register", "core.merge:register", ersatz=register)
        _ersetze(s, "merge.merge_recordings", "core.merge:merge_recordings", echt=True)

    # ---------------------------------------------------------- Einfärbung

    def colorizer(self) -> None:
        s, welt = self.s, self

        def pruefe(rec, bag, calib, T, frames=None, cancel=None):
            if welt.extrinsik == "fehler":
                raise RuntimeError("zu wenige Kamera-Frames für die Prüfung (Stub)")
            return {"score": 0.612, "best_score": 0.655, "dist_deg": 1.5,
                    "suspect": welt.extrinsik == "verdreht"}

        def faerbe(rec, bag, calib, params, out_dir, progress_cb=None, cancel=None,
                   parts=None, backend="auto"):
            _melde(progress_cb, 0.0, "Färbe Scan 1/4 …")
            n = int(rec.n_points)
            anteil = 0.0 if welt.farbe == "leer" else 0.75
            teile = welt.farbe == "teil1_duenn_teil2_leer"
            gueltig = _schreibe_farben(out_dir, n, seed=11, anteil=anteil,
                                       duenn_bis=n // 2 if teile else None,
                                       leer_ab=n // 2 if teile else None)
            _melde(progress_cb, 1.0, "Färbe Scan 4/4 …")
            n_valid = int(gueltig.sum())
            return {"n_valid": n_valid, "frac_valid": n_valid / n, "gpu": None,
                    "n_sky_blocked": 12, "n_edge_outvoted": 7, "n_blue_outvoted": 3,
                    "n_blue_kept": 1, "n_blue_neutral": 2}

        def kalibriere(rec, bag, calib, T_init=None, frames=None, progress_cb=None,
                       cancel=None):
            from scipy.spatial.transform import Rotation
            _melde(progress_cb, 0.0, "Stufe 1/3")
            _melde(progress_cb, 1.0, "Stufe 3/3")
            T = np.eye(4)
            T[:3, :3] = Rotation.from_euler("ZYX", [10.0, 5.0, -3.0],
                                            degrees=True).as_matrix()
            return T, welt.autocal_score

        _ersetze(s, "colorizer.check_extrinsic", "core.colorizer:check_extrinsic",
                 ersatz=pruefe)
        _ersetze(s, "colorizer.colorize", "core.colorizer:colorize", ersatz=faerbe)
        _ersetze(s, "colorizer.auto_calibrate", "core.colorizer:auto_calibrate",
                 ersatz=kalibriere)
        _ersetze(s, "colorizer.overlay_preview", "core.colorizer:overlay_preview",
                 ergebnis=np.full((20, 40, 3), 90, dtype=np.uint8))
        _ersetze(s, "colorizer.blue_preview", "core.colorizer:blue_preview",
                 ergebnis=(np.full((20, 40, 3), 120, dtype=np.uint8), 0.125))

    # -------------------------------------------------------------- Export

    def export(self) -> None:
        s, welt = self.s, self

        def cloudcompare(paths):
            if not welt.cloudcompare:
                raise RuntimeError("CloudCompare ist nicht installiert (Stub).")
            return ["CloudCompare", "-O", *paths]

        _ersetze(s, "georef.export_ply_pcd", "core.georef:export_ply_pcd")
        _ersetze(s, "georef.export_las", "core.georef:export_las")
        _ersetze(s, "mesh.open_in_cloudcompare", "core.mesh:open_in_cloudcompare",
                 ersatz=cloudcompare)

    def bundle(self) -> None:
        s = self.s

        def exportiere(project, dest, teile, bag_paths=None, calib_path=None,
                       meta_extra=None, progress=None, cancel=None):
            progress(0.5, "Kopiere points.bin — 1,0 kB von 2,0 kB")
            cancel()
            return {"format": "super360studio-projekt", "version": 1,
                    "inhalt": {"recording": True, "colors": bool(teile.get("colors")),
                               "panos": False, "meander": False, "bags": False},
                    "bytes": 123456, **(meta_extra or {})}

        _ersetze(s, "bundle.describe", "core.bundle:describe", echt=True)
        _ersetze(s, "bundle.export_project", "core.bundle:export_project",
                 ersatz=exportiere)

    # ---------------------------------------------------------------- RViz

    def rviz(self):
        """Stub-Player ans Fenster hängen; die Klasse liefert ihn ab jetzt auch."""
        s = self.s

        class Player:
            _kurz = "StubPlayer"
            spielt = False
            offen = False

            def is_playing(self):
                return self.spielt

            def rviz_running(self):
                return self.offen

            def start(self, bag_path, rate=1.0):
                self.spielt = self.offen = True

            def stop(self, close_rviz=True):
                self.spielt = self.offen = False

            def replay(self, rate=1.0):
                self.spielt = self.offen = True

        for name in ("start", "stop", "replay"):
            s.ersetze_attribut(Player, name, ersatz=Player.__dict__[name],
                               name=f"RvizPlayer.{name}")
        self.player = Player()
        _ersetze(s, "RvizPlayer", "core.rviz_player:RvizPlayer",
                 ersatz=lambda *a, **k: self.player)
        s.fenster._rviz_player = self.player
        return self.player


# ------------------------------------------------------------------ Vorlagen

def _grad(pfad: str):
    from core import exploration

    return exploration.Explorationsgrad.aus_dict({
        "prozent": 61.5, "prozent_flaeche": 72.25, "prozent_gesamt": 80.0,
        "prozent_flaeche_gesamt": 88.5, "stand_beginn": 10.0, "stand_ende": 71.5,
        "zuwachs": 61.5, "box_min": [-5.0, -5.0, 0.0], "box_max": [5.0, 5.0, 3.0],
        "volumen_m3": 300.0, "flaeche_m2": 100.0, "voxel_m": 0.5,
        "phasen": [[2.0, 4.0]], "quelle": "Stub", "dauer_s": 2.0, "strecke_m": 7.5,
        "scans": 4, "scans_phase": 2, "bag_dauer_s": 5.0, "bag": str(pfad)})


def _schreibe_aufzeichnung(s, ordner: str, bag: str, seed: int, t0: float = 1000.0):
    """Mini-Aufzeichnung (4 Scans, 200 Punkte) samt meta.json für ``bag``."""
    rec = s.stub_aufzeichnung(seed=seed, ordner=ordner)
    meta_p = os.path.join(ordner, "meta.json")
    with open(meta_p, encoding="utf-8") as fh:
        meta = json.load(fh)
    meta["bag"] = str(bag)
    with open(meta_p, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    stamps = t0 + 0.1 * np.arange(rec.n_scans, dtype=np.float64)
    np.save(os.path.join(ordner, "stamps.npy"), stamps)
    rec.stamps = stamps
    rec.meta["bag"] = str(bag)
    return rec


def _schreibe_farben(ordner: str, n: int, seed: int, anteil: float = 0.75,
                     fingerprint: str | None = None, leer_ab: int | None = None,
                     duenn_bis: int | None = None):
    """colors.bin, valid.bin und meta.json einer Farbebene; liefert die Maske.

    ``duenn_bis``: davor ist nur jeder zehnte Punkt gefärbt (10 %, zwischen den
    Schwellen 2 % und 20 %); ``leer_ab``: von dort an keiner.
    """
    rng = np.random.default_rng(seed)
    farben = rng.integers(0, 256, size=(n, 3), dtype=np.uint8)
    gueltig = (rng.uniform(size=n) < anteil).astype(np.uint8)
    if duenn_bis is not None:
        gueltig[:duenn_bis] = 0
        gueltig[:duenn_bis:10] = 1
    if leer_ab is not None:
        gueltig[leer_ab:] = 0
    os.makedirs(ordner, exist_ok=True)
    farben.tofile(os.path.join(ordner, "colors.bin"))
    gueltig.tofile(os.path.join(ordner, "valid.bin"))
    meta = {"n_valid": int(gueltig.sum())}
    if fingerprint is not None:
        meta["rec_fingerprint"] = fingerprint
    with open(os.path.join(ordner, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return gueltig


def _flug(s, name: str = "flug_a", karte: bool = True, farben: str | None = None,
          seed: int = 1) -> str:
    """Bag-Ordner und Projekt im Fall-Cache anlegen; liefert den Bagpfad.

    ``farben``: ``"passend"``, ``"fremd"`` (anderer Fingerprint) oder
    ``"alt"`` (meta.json ohne Fingerprint).
    """
    from core.colorizer import rec_fingerprint
    from core.project import Project

    pfad = os.path.join(s.ordner, name)
    os.makedirs(pfad, exist_ok=True)
    projekt = Project(pfad)
    if karte:
        rec = _schreibe_aufzeichnung(s, projekt.recording_dir(), pfad, seed,
                                     _T0.get(name, 1000.0))
        if farben:
            marke = {"passend": rec_fingerprint(rec), "fremd": "9_999_1.000000_2.000000",
                     "alt": None}[farben]
            _schreibe_farben(projekt.colors_dir(), rec.n_points, seed=seed + 20,
                             fingerprint=marke)
    return pfad


def _zusammengefuehrt(s, zweites_bag_da: bool = True, farben: bool = False):
    """Zwei Mini-Aufzeichnungen echt zusammenführen: das Projekt »flug_a+flug_b«."""
    from core import merge as merge_mod
    from core.colorizer import rec_fingerprint
    from core.project import Project
    from core.recording import Recording

    bag_a = os.path.join(s.ordner, "flug_a")
    bag_b = os.path.join(s.ordner, "flug_b")
    os.makedirs(bag_a, exist_ok=True)
    if zweites_bag_da:
        os.makedirs(bag_b, exist_ok=True)
    rec_a = s.stub_aufzeichnung(seed=1)
    rec_b = s.stub_aufzeichnung(seed=2)
    rec_b.stamps = rec_b.stamps + 1000.0
    T = np.eye(4)
    T[:3, 3] = [10.0, 0.0, 0.0]
    projekt = Project(os.path.join(s.ordner, "flug_a+flug_b"))
    merge_mod.merge_recordings(rec_a, rec_b, T, projekt.recording_dir(), bag_a, bag_b,
                               info={"fitness": None})
    if farben:
        rec = Recording.load(projekt.recording_dir())
        _schreibe_farben(projekt.colors_dir(), rec.n_points, seed=31,
                         fingerprint=rec_fingerprint(rec))
    return projekt


def _waehle_projekt(name: str):
    """Antwort auf den Projektdialog: die Zeile mit diesem Namen, dann Öffnen."""
    def antwort(dialog):
        from PyQt5.QtWidgets import QDialog, QTableWidget

        tabelle = dialog.findChild(QTableWidget)
        for zeile in range(tabelle.rowCount()):
            if tabelle.item(zeile, 0).text() == name:
                tabelle.selectRow(zeile)
                return QDialog.Accepted
        raise RuntimeError(f"Projekt '{name}' steht nicht in der Liste.")
    return antwort


def _exportordner(s, pfad: str) -> str:
    """Echter Export des Projekts zu ``pfad`` nach ``<fall>/export`` (ohne Bag)."""
    from core import bundle
    from core.project import Project

    ziel = os.path.join(s.ordner, "export")
    bundle.export_project(Project(pfad), ziel, {"colors": True}, bag_paths=[pfad])
    return ziel


# -------------------------------------------------------------------- Ablauf

def _laufe(s) -> None:
    """Wartende Schritte fahren; danach steht auch das erste Panorama."""
    from PyQt5.QtCore import QElapsedTimer

    s.laufe()
    f = s.fenster
    if f._pano_src is not None and getattr(f._pano_src, "count", 0) > 0:
        uhr = QElapsedTimer()
        uhr.start()
        while not f._frame_lbl.text() and uhr.elapsed() < 15000:
            s.warte(20)
    s.warte()


class _Vorlauf:
    """``with _Vorlauf(s):`` — was darin geschieht, stellt nur den Zustand her.

    Ereignisse, Blattaufrufe und geschriebene Dateien des Blocks zählen nicht;
    was der Fall davor aufgezeichnet hat, bleibt stehen.
    """

    def __init__(self, s):
        self.s = s

    def __enter__(self) -> "_Vorlauf":
        self.n = len(self.s.ereignisse)
        self.blatt = self.s.blattaufrufe
        return self

    def __exit__(self, typ, *exc) -> None:
        if typ is None:
            _laufe(self.s)
            del self.s.ereignisse[self.n:]
            self.s.blattaufrufe = self.blatt
            self.s.dateien_merken()


def _offen(s, welt: _Welt, **flug) -> str:
    """Flug über den echten Weg öffnen (samt Karte, wenn es eine gibt)."""
    with _Vorlauf(s):
        pfad = _flug(s, **flug)
        s.fenster._open_bag(pfad)
    return pfad


def _zweitflug(s, welt: _Welt, karte: bool = True) -> str:
    """Zweiten Flug »flug_b« anlegen und die Ordnerwahl darauf beantworten."""
    pfad = _flug(s, name="flug_b", karte=karte, seed=2)
    s.dateien_merken()
    s.antworte("QFileDialog.getExistingDirectory", pfad)
    return pfad


def _mit_zweitflug(s, welt: _Welt) -> None:
    """Offener Flug mit geladenem zweiten; der Vorlauf ist verworfen."""
    welt.merge()
    _offen(s, welt)
    with _Vorlauf(s):
        _zweitflug(s, welt)
        s.fenster._on_merge_pick()


def _langer_schritt(s) -> None:
    """Ein Schritt, der noch wartet: das Fenster ist beschäftigt."""
    def job(progress_cb, cancel, log_cb):
        progress_cb(0.5, "Langer Schritt: Hälfte")
        if cancel.is_set():
            raise RuntimeError("Abgebrochen")
        return None

    s.fenster._start_worker("Langer Schritt …", job, lambda _r: None)


def _bildfenster(s) -> None:
    """Titel der offenen Bildfenster festhalten (sie laufen nicht über exec_)."""
    from PyQt5.QtWidgets import QDialog

    s.warte()
    s.ereignis("bildfenster", [s.text(d.windowTitle())
                               for d in s.fenster.findChildren(QDialog) if d.isVisible()])


def _extrinsik(s, marke: str) -> None:
    werte = {k: round(float(sp.value()), 6) for k, sp in s.fenster._ext_spins.items()}
    s.ereignis("extrinsik", marke, werte)


def _stand_ablegen(pfad: str, exploration: dict) -> None:
    """settings.json, extrinsic.json und exploration.json ins Projekt des Flugs legen.

    Die Einstellungen weichen von der Vorgabe ab, die Extrinsik ist nicht die
    Einheit: nur so zeigt sich, ob das Öffnen den gespeicherten Stand übernimmt.
    """
    from scipy.spatial.transform import Rotation
    from core.project import Project

    projekt = Project(pfad)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("zyx", [12.0, -3.0, 5.0], degrees=True).as_matrix()
    T[:3, 3] = [0.05, -0.02, 0.1]
    projekt.save_extrinsic(T)
    projekt.save_settings({"k_frames": 5, "rate": 2.0, "only_colored": False,
                           "config": "mid360.yaml", "sky_grow": 7})
    projekt.save_exploration(exploration)


def _zustand(s, marke: str) -> None:
    """Regler der Extrinsik und die Einstellungen, wie das Fenster sie sammelt."""
    _extrinsik(s, marke)
    s.ereignis("einstellungen", marke, s.kurz(s.fenster._collect_settings()))


def _regler_fastlio(s) -> None:
    """Rate und Konfiguration weg von der Vorgabe (1.0, erster Eintrag)."""
    f = s.fenster
    f._spin_rate.setValue(2.0)
    f._combo_config.setCurrentIndex(1)


# =============================================================== Rosbag öffnen

def _bag_oeffnen_mit_kamera(s):
    _Welt(s).grund()
    pfad = _flug(s, karte=False)
    s.dateien_merken()
    s.antworte("QFileDialog.getExistingDirectory", pfad)
    s.fenster._on_open_clicked()
    _laufe(s)


def _bag_oeffnen_ohne_kamera(s):
    welt = _Welt(s).grund()
    welt.ohne_kamera.add("flug_a")
    welt.grad = "keine"
    pfad = _flug(s)
    s.dateien_merken()
    s.fenster._open_bag(pfad)
    _laufe(s)


def _bag_oeffnen_pano_fehler(s):
    welt = _Welt(s).grund()
    welt.pano_fehler = True
    welt.grad = "fehler"
    pfad = _flug(s)
    s.dateien_merken()
    s.fenster._open_bag(pfad)
    _laufe(s)
    s.merke_labels("nach dem Pano-Fehler")


def _bag_oeffnen_abbruch(s):
    _Welt(s).grund()
    pfad = _flug(s)
    s.dateien_merken()
    s.abbruch_nach(2)
    s.fenster._open_bag(pfad)
    _laufe(s)
    s.abbruch_nach(None)


def _bag_oeffnen_wahl_abgebrochen(s):
    _Welt(s).grund()
    s.fenster._on_open_clicked()          # Ordnerdialog: abgebrochen
    _laufe(s)


def _bag_oeffnen_mit_stand(s):
    _Welt(s).grund()
    pfad = _flug(s)
    _stand_ablegen(pfad, {"keine_daten": "Stub: keine EPIC-Daten im Bag"})
    s.dateien_merken()
    s.fenster._open_bag(pfad)             # gespeichertes »keine Daten« gilt, nichts wird gerechnet
    _laufe(s)
    _zustand(s, "nach dem Öffnen")


def _bag_oeffnen_grad_unbrauchbar(s):
    _Welt(s).grund()
    pfad = _flug(s)
    _stand_ablegen(pfad, {"wert": "kaputt"})
    s.dateien_merken()
    s.fenster._open_bag(pfad)             # unbrauchbarer Grad: wird neu gerechnet
    _laufe(s)
    _zustand(s, "nach dem Öffnen")


# ========================================================= Aufzeichnung laden

def _aufzeichnung_ohne_farben(s):
    _Welt(s).grund()
    pfad = _flug(s)
    s.dateien_merken()
    s.fenster._open_bag(pfad)
    _laufe(s)


def _aufzeichnung_mit_farben(s):
    _Welt(s).grund()
    pfad = _flug(s, farben="passend")
    s.dateien_merken()
    s.fenster._open_bag(pfad)
    _laufe(s)


def _aufzeichnung_fremder_fingerprint(s):
    _Welt(s).grund()
    pfad = _flug(s, farben="fremd")
    s.dateien_merken()
    s.fenster._open_bag(pfad)
    _laufe(s)


def _aufzeichnung_farben_ohne_fingerprint(s):
    _Welt(s).grund()
    pfad = _flug(s, farben="alt")
    s.dateien_merken()
    s.fenster._open_bag(pfad)
    _laufe(s)


def _aufzeichnung_zusammengefuehrt(s):
    _Welt(s).grund()
    projekt = _zusammengefuehrt(s, farben=True)
    s.dateien_merken()
    s.fenster._open_project_only(projekt)
    _laufe(s)


def _aufzeichnung_quellen_unbrauchbar(s):
    _Welt(s).grund()
    projekt = _zusammengefuehrt(s)
    meta_p = os.path.join(projekt.recording_dir(), "meta.json")
    with open(meta_p, encoding="utf-8") as fh:
        meta = json.load(fh)
    del meta["sources"][1]["scan_range"]
    with open(meta_p, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    s.dateien_merken()
    s.fenster._open_project_only(projekt)
    _laufe(s)


# ============================================================ Karte berechnen

def _karte_berechnen(s):
    welt = _Welt(s).grund()
    _offen(s, welt, karte=False)
    s.fenster._on_fastlio_clicked()
    s.merke_labels("Karte wird berechnet")
    _laufe(s)


def _karte_neu_berechnen(s):
    welt = _Welt(s).grund()
    _offen(s, welt, farben="passend")     # die alten Farben passen danach nicht mehr
    s.fenster._on_fastlio_clicked()
    _laufe(s)


def _karte_berechnen_fehler(s):
    welt = _Welt(s).grund()
    _offen(s, welt, karte=False)
    welt.fastlio_fehler = True
    s.fenster._on_fastlio_clicked()
    _laufe(s)


def _karte_berechnen_regler(s):
    welt = _Welt(s).grund()
    _offen(s, welt, karte=False)
    _regler_fastlio(s)
    s.fenster._on_fastlio_clicked()       # Rate 2.0 und die zweite Konfiguration
    _laufe(s)


# ============================================================ Explorationsgrad

def _exploration_neu(s):
    welt = _Welt(s).grund()
    welt.grad = "keine"
    _offen(s, welt)
    welt.grad = "wert"
    s.fenster._on_exploration_neu()
    _laufe(s)


def _exploration_neu_ohne_daten(s):
    welt = _Welt(s).grund()
    _offen(s, welt)
    welt.grad = "keine"
    s.fenster._on_exploration_neu()
    _laufe(s)


def _exploration_bericht(s):
    welt = _Welt(s).grund()
    _offen(s, welt)
    s.fenster._on_exploration_zeigen()


# ====================================================== Projekt öffnen, Export

def _projekt_cache_einzeln(s):
    _Welt(s).grund()
    _flug(s, farben="passend")
    s.dateien_merken()
    s.antworte("QDialog.exec_:ProjectOpenDialog", _waehle_projekt("flug_a"))
    s.fenster._on_open_project()          # Bag am Ort: der normale Weg über _open_bag
    _laufe(s)


def _projekt_cache_bag_fehlt(s):
    _Welt(s).grund()
    pfad = _flug(s)
    shutil.rmtree(pfad)
    s.dateien_merken()
    s.antworte("QDialog.exec_:ProjectOpenDialog", _waehle_projekt("flug_a"))
    s.fenster._on_open_project()          # ohne Bag: nur, was aus dem Cache lebt
    _laufe(s)


def _projekt_cache_bag_fehlt_mit_stand(s):
    _Welt(s).grund()
    pfad = _flug(s)
    _stand_ablegen(pfad, {"keine_daten": "Stub: keine EPIC-Daten im Bag"})
    shutil.rmtree(pfad)
    s.dateien_merken()
    s.antworte("QDialog.exec_:ProjectOpenDialog", _waehle_projekt("flug_a"))
    s.fenster._on_open_project()
    _laufe(s)
    _zustand(s, "nach dem Öffnen ohne Bag")


def _projekt_cache_bag_fehlt_grad_unbrauchbar(s):
    _Welt(s).grund()
    pfad = _flug(s)
    _stand_ablegen(pfad, {"wert": "kaputt"})
    shutil.rmtree(pfad)
    s.dateien_merken()
    s.antworte("QDialog.exec_:ProjectOpenDialog", _waehle_projekt("flug_a"))
    s.fenster._on_open_project()          # ohne Bag lässt sich der Grad nicht neu rechnen
    _laufe(s)
    _zustand(s, "nach dem Öffnen ohne Bag")


def _projekt_cache_zusammengefuehrt(s):
    _Welt(s).grund()
    _zusammengefuehrt(s, zweites_bag_da=False)
    s.dateien_merken()
    s.antworte("QDialog.exec_:ProjectOpenDialog", _waehle_projekt("flug_a+flug_b"))
    s.fenster._on_open_project()
    _laufe(s)


def _projekt_cache_leer(s):
    _Welt(s).grund()
    s.fenster._on_open_project()          # kein Projekt im Cache
    _flug(s)
    s.dateien_merken()
    s.fenster._on_open_project()          # Dialog abgelehnt
    _laufe(s)


def _projekt_import_mit_bag(s):
    _Welt(s).grund()
    pfad = _flug(s, farben="passend")
    ordner = _exportordner(s, pfad)
    s.dateien_merken()
    s.antworte("QFileDialog.getExistingDirectory", ordner)
    s.fenster._on_import_project()
    _laufe(s)


def _projekt_import_ohne_bag(s):
    _Welt(s).grund()
    pfad = _flug(s)
    ordner = _exportordner(s, pfad)
    shutil.rmtree(pfad)
    s.dateien_merken()
    s.antworte("QFileDialog.getExistingDirectory", ordner)
    s.fenster._on_import_project()
    _laufe(s)


def _projekt_import_kein_projekt(s):
    _Welt(s).grund()
    ordner = os.path.join(s.ordner, "irgendwas")
    os.makedirs(ordner)
    s.antworte("QFileDialog.getExistingDirectory", ordner, "")
    s.fenster._on_import_project()        # kein Manifest
    s.fenster._on_import_project()        # Ordnerwahl abgebrochen
    _laufe(s)


def _projekt_exportieren(s):
    welt = _Welt(s).grund()
    welt.bundle()
    _offen(s, welt, farben="passend")
    ziel = os.path.join(s.ordner, "mitnehmen")

    def antwort(dialog):
        from PyQt5.QtWidgets import QDialog, QLineEdit

        dialog.findChild(QLineEdit).setText(ziel)
        return QDialog.Accepted

    s.antworte("QDialog.exec_:ExportDialog", antwort)
    s.fenster._on_export_project()
    _laufe(s)
    s.fenster._on_export_project()        # Dialog abgelehnt
    _laufe(s)


def _projekt_exportieren_ohne_karte(s):
    welt = _Welt(s).grund()
    welt.bundle()
    s.fenster._on_export_project()        # kein Projekt offen
    _offen(s, welt, karte=False)
    s.fenster._on_export_project()        # Projekt ohne Karte
    _laufe(s)


# ======================================================== Beschäftigt-Abfragen

def _beschaeftigt_bag_waehlen(s):
    _Welt(s).grund()
    _langer_schritt(s)
    s.fenster._on_open_clicked()
    _laufe(s)


def _beschaeftigt_bag_oeffnen(s):
    _Welt(s).grund()
    pfad = _flug(s)
    s.dateien_merken()
    _langer_schritt(s)
    s.fenster._open_bag(pfad)
    _laufe(s)


def _beschaeftigt_projekt_oeffnen(s):
    _Welt(s).grund()
    _flug(s)
    s.dateien_merken()
    _langer_schritt(s)
    s.fenster._on_open_project()
    _laufe(s)


def _beschaeftigt_projekt_import(s):
    _Welt(s).grund()
    _langer_schritt(s)
    s.fenster._on_import_project()
    _laufe(s)


# ============================================================== Zusammenführen

def _zweitflug_laden_mit_karte(s):
    welt = _Welt(s).grund()
    welt.merge()
    _offen(s, welt)
    _zweitflug(s, welt)
    s.fenster._on_merge_pick()
    _laufe(s)


def _zweitflug_laden_ohne_karte(s):
    welt = _Welt(s).grund()
    welt.merge()
    _offen(s, welt)
    _zweitflug(s, welt, karte=False)
    s.fenster._on_merge_pick()            # Rückfrage: Karte jetzt berechnen? Ja
    _laufe(s)


def _zweitflug_laden_ohne_karte_regler(s):
    welt = _Welt(s).grund()
    welt.merge()
    _offen(s, welt)
    _regler_fastlio(s)
    _zweitflug(s, welt, karte=False)
    s.fenster._on_merge_pick()            # Karte mit Rate 2.0 und der zweiten Konfiguration
    _laufe(s)


def _zweitflug_abgewiesen(s):
    from PyQt5.QtWidgets import QMessageBox

    welt = _Welt(s).grund()
    welt.merge()
    s.fenster._on_merge_pick()            # kein Flug offen
    pfad_a = _offen(s, welt)
    pfad_b = _flug(s, name="flug_b", karte=False, seed=2)
    s.dateien_merken()
    s.antworte("QFileDialog.getExistingDirectory", "", pfad_a, pfad_b)
    s.antworte("QMessageBox.question", QMessageBox.No)
    s.fenster._on_merge_pick()            # Ordnerwahl abgebrochen
    s.fenster._on_merge_pick()            # derselbe Flug
    s.fenster._on_merge_pick()            # Karte fehlt, Berechnen abgelehnt
    s.fenster._on_merge_align("auto")     # ohne zweiten Flug
    s.fenster._on_merge_apply()
    s.fenster._on_merge_discard()
    _laufe(s)


def _ausrichten_auto(s):
    welt = _Welt(s).grund()
    _mit_zweitflug(s, welt)
    s.fenster._on_merge_align("auto")
    _laufe(s)


def _ausrichten_icp_schwach(s):
    welt = _Welt(s).grund()
    _mit_zweitflug(s, welt)
    welt.ausrichtung = {"fitness": 0.21, "rmse": 0.12, "kandidat": "start"}
    s.fenster._on_merge_align("icp")
    _laufe(s)


def _ausrichten_grober_rest(s):
    welt = _Welt(s).grund()
    _mit_zweitflug(s, welt)
    welt.ausrichtung = {"fitness": 0.64, "rmse": 0.41, "kandidat": "gier-90"}
    s.fenster._on_merge_align("auto")
    _laufe(s)


def _handjustage(s):
    welt = _Welt(s).grund()
    _mit_zweitflug(s, welt)
    f = s.fenster
    f._spin_merge["x"].setValue(1.5)
    f._spin_merge["z"].setValue(-0.25)
    s.warte(300)                          # der Regler wirkt nach kurzer Ruhe
    f._spin_merge["yaw"].setValue(20.0)
    s.warte(300)
    s.ereignis("lage", s.kurz(np.asarray(f._merge_T)))
    f._on_merge_align("icp")              # ICP startet von der Handlage
    _laufe(s)
    s.ereignis("regler", {k: float(sp.value()) for k, sp in f._spin_merge.items()})


def _zusammenfuehren(s):
    welt = _Welt(s).grund()
    _mit_zweitflug(s, welt)
    with _Vorlauf(s):
        s.fenster._on_merge_align("auto")
    s.fenster._on_merge_apply()
    _laufe(s)


def _zweitflug_verwerfen(s):
    welt = _Welt(s).grund()
    _mit_zweitflug(s, welt)
    s.fenster._on_merge_discard()
    _laufe(s)


# ============================================================ 360°-Einfärbung

def _einfaerben(s):
    welt = _Welt(s).grund()
    welt.colorizer()
    _offen(s, welt)
    s.fenster._on_colorize_clicked()
    _laufe(s)


def _einfaerben_mit_teilen(s):
    welt = _Welt(s).grund()
    welt.colorizer()
    welt.extrinsik = "verdreht"
    welt.farbe = "teil1_duenn_teil2_leer"  # 10 % und 0 %: nur der zweite Abschnitt warnt
    with _Vorlauf(s):
        s.fenster._open_project_only(_zusammengefuehrt(s))
    s.fenster._on_colorize_clicked()
    _laufe(s)


def _einfaerben_mit_zweitflug(s):
    from PyQt5.QtWidgets import QMessageBox

    welt = _Welt(s).grund()
    welt.colorizer()
    welt.extrinsik = "fehler"
    _mit_zweitflug(s, welt)
    s.antworte("QMessageBox.question", QMessageBox.No, QMessageBox.Yes)
    s.fenster._on_colorize_clicked()      # Rückfrage abgelehnt
    _laufe(s)
    s.fenster._on_colorize_clicked()
    _laufe(s)


def _einfaerben_kein_punkt(s):
    welt = _Welt(s).grund()
    welt.colorizer()
    welt.farbe = "leer"
    _offen(s, welt)
    s.fenster._on_colorize_clicked()
    _laufe(s)


def _einfaerben_ebene_maeander(s):
    from core.project import Project

    welt = _Welt(s).grund()
    welt.colorizer()
    with _Vorlauf(s):
        pfad = _flug(s, farben="passend")
        _schreibe_farben(Project(pfad).layer_dir("meander_rgb"), 200, seed=41)
        s.fenster._open_bag(pfad)
        _laufe(s)
        s.fenster._cloud_view.farbmodus_gewaehlt.emit("rgb:meander_rgb")
    s.ereignis("ebene", s.fenster._layer_key)
    s.fenster._on_colorize_clicked()
    _laufe(s)
    s.ereignis("ebene", s.fenster._layer_key)


def _blaumaske(s):
    welt = _Welt(s).grund()
    welt.colorizer()
    s.fenster._on_blue_preview_clicked()  # ohne Bag
    _offen(s, welt)
    s.fenster._on_blue_preview_clicked()
    _bildfenster(s)


def _ueberlagerung(s):
    welt = _Welt(s).grund()
    welt.colorizer()
    _offen(s, welt)
    s.fenster._on_overlay_clicked()
    _laufe(s)
    _bildfenster(s)


def _autokalibrierung_schwach(s):
    welt = _Welt(s).grund()
    welt.colorizer()
    welt.autocal_score = 0.12
    _offen(s, welt)
    s.fenster._on_autocal_clicked()
    _laufe(s)
    _extrinsik(s, "nach der Kalibrierung")


def _autokalibrierung_gut(s):
    welt = _Welt(s).grund()
    welt.colorizer()
    _offen(s, welt)
    s.fenster._on_autocal_clicked()
    _laufe(s)
    _extrinsik(s, "nach der Kalibrierung")


def _extrinsik_zuruecksetzen(s):
    welt = _Welt(s).grund()
    _offen(s, welt)
    f = s.fenster
    f._ext_spins["yaw"].setValue(12.5)    # von Hand: wird gespeichert
    f._ext_spins["x"].setValue(0.05)
    _extrinsik(s, "von Hand")
    f._on_extrinsic_reset()
    _laufe(s)
    _extrinsik(s, "zurückgesetzt")


def _einstellungen_vorgabe(s):
    from PyQt5.QtWidgets import QMessageBox

    welt = _Welt(s).grund()
    _offen(s, welt)
    f = s.fenster
    f._spin_kframes.setValue(5)
    f._spin_rate.setValue(2.0)
    f._chk_blue.setChecked(True)
    s.ereignis("einstellungen", "geändert", s.kurz(f._collect_settings()))
    s.antworte("QMessageBox.question", QMessageBox.No, QMessageBox.Yes)
    f._on_settings_reset()                # abgelehnt
    f._on_settings_reset()
    _laufe(s)
    s.ereignis("einstellungen", "Vorgabe", s.kurz(f._collect_settings()))


# ====================================================================== Export

def _export_ply(s):
    welt = _Welt(s).grund()
    welt.export()
    _offen(s, welt, farben="passend")
    s.antworte("QFileDialog.getSaveFileName",
               (os.path.join(s.ordner, "wolke.ply"), "PLY-Datei (*.ply)"), ("", ""))
    s.fenster._on_export_plypcd()
    _laufe(s)
    s.fenster._on_export_plypcd()         # Dateidialog abgebrochen
    _laufe(s)


def _export_pcd(s):
    welt = _Welt(s).grund()
    welt.export()
    _offen(s, welt)
    s.antworte("QFileDialog.getSaveFileName",
               (os.path.join(s.ordner, "wolke"), "PCD-Datei (*.pcd)"))
    s.fenster._on_export_plypcd()         # Endung kommt aus dem Filter
    _laufe(s)


def _export_las_lokal(s):
    welt = _Welt(s).grund()
    welt.export()
    _offen(s, welt, farben="passend")
    s.antworte("QFileDialog.getSaveFileName",
               (os.path.join(s.ordner, "wolke"), "LAS-Datei (*.las)"))
    s.fenster._on_export_las()
    _laufe(s)


def _georeferenz():
    from core import georef

    return georef.GeorefResult(T_enu_world=np.eye(4), origin_llh=(51.5, 7.1, 80.0),
                               utm_epsg=32632, rms_m=0.42, n_used=37,
                               utm_offset=(370000.0, 5700000.0))


def _export_las_georef(s):
    welt = _Welt(s).grund()
    welt.export()
    _offen(s, welt, farben="passend")
    with _Vorlauf(s):
        s.fenster._on_georef_ready(_georeferenz())
    s.antworte("QFileDialog.getSaveFileName",
               (os.path.join(s.ordner, "wolke.las"), "LAS-Datei (*.las)"))
    s.fenster._on_export_las()
    _laufe(s)


def _georef_bereit(s):
    welt = _Welt(s).grund()
    _offen(s, welt)
    s.fenster._on_georef_ready(_georeferenz())
    _laufe(s)


def _cloudcompare_punktwolke(s):
    welt = _Welt(s).grund()
    welt.export()
    _offen(s, welt, farben="passend")
    s.fenster._on_cloud_cloudcompare()
    _laufe(s)


def _cloudcompare_alle_punkte(s):
    welt = _Welt(s).grund()
    welt.export()
    _offen(s, welt, farben="passend")
    s.fenster._chk_only_colored.setChecked(False)
    s.fenster._on_cloud_cloudcompare()    # ungefärbte Punkte gehen grau mit
    _laufe(s)


def _cloudcompare_fehlt(s):
    welt = _Welt(s).grund()
    welt.export()
    welt.cloudcompare = False
    _offen(s, welt)
    s.fenster._on_cloud_cloudcompare()
    _laufe(s)


# ======================================================================== RViz

def _rviz_starten(s):
    welt = _Welt(s).grund()
    welt.rviz()
    s.fenster._on_rviz_start()            # ohne Bag: nichts
    _offen(s, welt)
    s.fenster._on_rviz_start()
    _laufe(s)
    s.merke_labels("läuft")


def _rviz_beenden(s):
    welt = _Welt(s).grund()
    welt.rviz()
    _offen(s, welt)
    with _Vorlauf(s):
        s.fenster._on_rviz_start()
    s.fenster._on_rviz_stop()
    _laufe(s)


def _rviz_wiederholen(s):
    welt = _Welt(s).grund()
    welt.rviz()
    _offen(s, welt)
    with _Vorlauf(s):
        s.fenster._on_rviz_start()
    s.fenster._on_rviz_replay()
    _laufe(s)


def _rviz_waehrend_schritt(s):
    welt = _Welt(s).grund()
    welt.rviz()
    _offen(s, welt)
    _langer_schritt(s)
    s.fenster._on_rviz_start()
    s._rviz_abwarten()                    # sonst Wettlauf mit dem RViz-Arbeiter
    s.fenster._on_rviz_replay()
    s._rviz_abwarten()
    s.fenster._on_rviz_stop()
    _laufe(s)


# ============================================================ Fenster schließen

def _fenster_schliessen(s):
    welt = _Welt(s).grund()
    welt.rviz()
    _offen(s, welt)
    f = s.fenster
    with _Vorlauf(s):
        f._spin_kframes.setValue(4)
    _langer_schritt(s)
    f.close()                             # bricht den Schritt ab, räumt RViz, speichert
    s.ereignis("zu", bool(f._closing), not f.isVisible())
    s.merke_labels("geschlossen")
    f._on_rviz_start()                    # nach dem Schließen startet nichts mehr


def _fall(funktion):
    """Hülle je Fall: am Ende die Dialogtexte neutral stellen.

    Dialoge nennen den Projektordner im Cache. Dessen Name trägt einen Hash
    des Bagpfads, der je Lauf anders ausfällt; der Rekorder ersetzt ihn in
    Logzeilen und Argumenten, in den Dialog-Einträgen aber nicht.
    """
    def fall(s):
        funktion(s)
        s.laufe()
        for eintrag in s.ereignisse:
            if eintrag[0] == "dialog":
                eintrag[1] = json.loads(s.text(json.dumps(eintrag[1], ensure_ascii=False)))
    return fall


_FAELLE = {
    "bag_oeffnen_mit_kamera": _bag_oeffnen_mit_kamera,
    "bag_oeffnen_ohne_kamera": _bag_oeffnen_ohne_kamera,
    "bag_oeffnen_pano_fehler": _bag_oeffnen_pano_fehler,
    "bag_oeffnen_abbruch": _bag_oeffnen_abbruch,
    "bag_oeffnen_wahl_abgebrochen": _bag_oeffnen_wahl_abgebrochen,
    "bag_oeffnen_mit_stand": _bag_oeffnen_mit_stand,
    "bag_oeffnen_grad_unbrauchbar": _bag_oeffnen_grad_unbrauchbar,
    "aufzeichnung_ohne_farben": _aufzeichnung_ohne_farben,
    "aufzeichnung_mit_farben": _aufzeichnung_mit_farben,
    "aufzeichnung_fremder_fingerprint": _aufzeichnung_fremder_fingerprint,
    "aufzeichnung_farben_ohne_fingerprint": _aufzeichnung_farben_ohne_fingerprint,
    "aufzeichnung_zusammengefuehrt": _aufzeichnung_zusammengefuehrt,
    "aufzeichnung_quellen_unbrauchbar": _aufzeichnung_quellen_unbrauchbar,
    "karte_berechnen": _karte_berechnen,
    "karte_neu_berechnen": _karte_neu_berechnen,
    "karte_berechnen_fehler": _karte_berechnen_fehler,
    "karte_berechnen_regler": _karte_berechnen_regler,
    "exploration_neu": _exploration_neu,
    "exploration_neu_ohne_daten": _exploration_neu_ohne_daten,
    "exploration_bericht": _exploration_bericht,
    "projekt_cache_einzeln": _projekt_cache_einzeln,
    "projekt_cache_bag_fehlt": _projekt_cache_bag_fehlt,
    "projekt_cache_bag_fehlt_mit_stand": _projekt_cache_bag_fehlt_mit_stand,
    "projekt_cache_bag_fehlt_grad_unbrauchbar": _projekt_cache_bag_fehlt_grad_unbrauchbar,
    "projekt_cache_zusammengefuehrt": _projekt_cache_zusammengefuehrt,
    "projekt_cache_leer": _projekt_cache_leer,
    "projekt_import_mit_bag": _projekt_import_mit_bag,
    "projekt_import_ohne_bag": _projekt_import_ohne_bag,
    "projekt_import_kein_projekt": _projekt_import_kein_projekt,
    "projekt_exportieren": _projekt_exportieren,
    "projekt_exportieren_ohne_karte": _projekt_exportieren_ohne_karte,
    "beschaeftigt_bag_waehlen": _beschaeftigt_bag_waehlen,
    "beschaeftigt_bag_oeffnen": _beschaeftigt_bag_oeffnen,
    "beschaeftigt_projekt_oeffnen": _beschaeftigt_projekt_oeffnen,
    "beschaeftigt_projekt_import": _beschaeftigt_projekt_import,
    "zweitflug_laden_mit_karte": _zweitflug_laden_mit_karte,
    "zweitflug_laden_ohne_karte": _zweitflug_laden_ohne_karte,
    "zweitflug_laden_ohne_karte_regler": _zweitflug_laden_ohne_karte_regler,
    "zweitflug_abgewiesen": _zweitflug_abgewiesen,
    "ausrichten_auto": _ausrichten_auto,
    "ausrichten_icp_schwach": _ausrichten_icp_schwach,
    "ausrichten_grober_rest": _ausrichten_grober_rest,
    "handjustage": _handjustage,
    "zusammenfuehren": _zusammenfuehren,
    "zweitflug_verwerfen": _zweitflug_verwerfen,
    "einfaerben": _einfaerben,
    "einfaerben_mit_teilen": _einfaerben_mit_teilen,
    "einfaerben_mit_zweitflug": _einfaerben_mit_zweitflug,
    "einfaerben_kein_punkt": _einfaerben_kein_punkt,
    "einfaerben_ebene_maeander": _einfaerben_ebene_maeander,
    "blaumaske": _blaumaske,
    "ueberlagerung": _ueberlagerung,
    "autokalibrierung_schwach": _autokalibrierung_schwach,
    "autokalibrierung_gut": _autokalibrierung_gut,
    "extrinsik_zuruecksetzen": _extrinsik_zuruecksetzen,
    "einstellungen_vorgabe": _einstellungen_vorgabe,
    "export_ply": _export_ply,
    "export_pcd": _export_pcd,
    "export_las_lokal": _export_las_lokal,
    "export_las_georef": _export_las_georef,
    "georef_bereit": _georef_bereit,
    "cloudcompare_punktwolke": _cloudcompare_punktwolke,
    "cloudcompare_alle_punkte": _cloudcompare_alle_punkte,
    "cloudcompare_fehlt": _cloudcompare_fehlt,
    "rviz_starten": _rviz_starten,
    "rviz_beenden": _rviz_beenden,
    "rviz_wiederholen": _rviz_wiederholen,
    "rviz_waehrend_schritt": _rviz_waehrend_schritt,
    "fenster_schliessen": _fenster_schliessen,
}

FAELLE = {name: _fall(funktion) for name, funktion in _FAELLE.items()}
