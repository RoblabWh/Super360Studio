#!/usr/bin/env python3
"""Ablauf-Proben der Familie ``maeander``.

Fälle für die Handler von Mäander-Einfärbung, Gaussian Splat, Fusion, Mesh und
Farbebenen des Hauptfensters; gefahren von ``ablauf_probe.py`` (Vertrag dort).

Die Rechenwege in ``core`` laufen nicht: jede Blattfunktion ist durch einen
Stellvertreter ersetzt, der seinen Aufruf aufzeichnet und ein kleines, festes
Ergebnis liefert (:class:`Stubs`). Echt bleiben die Dateien im Projektordner —
Farbebenen, ``rgb_zuschlag.json``, ``thermal_lage.json``,
``optik_kalibrierung.json`` und die Einstellungen landen im Arbeitsordner und
gehen als Hash bzw. dict in die Aufzeichnung.

Jeder Fall endet mit einem Ereignis ``zustand``: Lage der Pipeline, Optik,
Regler, vorhandene Ebenen, Auswahl der Farbquelle, Mesh-Schalter.
"""
from __future__ import annotations

import json
import os

import numpy as np
from PyQt5.QtWidgets import QMessageBox

from core import fusion as fusion_mod
from core import meander as meander_mod
from core import mesh as mesh_mod
from core import optik as optik_mod
from core import sichtbar as sichtbar_mod
from core import splat as splat_mod
from core import temperatur as temperatur_mod

FAELLE: dict = {}

_SPLAT_PYTHON = "/opt/splat/bin/python"
_SPLAT_INFO = {"torch": "2.7.0", "gsplat": True, "cuda": True, "gpu": "Prüf-GPU",
               "vram_gb": 8, "gsplat_version": "1.5.0", "python": _SPLAT_PYTHON}
_FRAGLICH = ("Die Ausrichtung sitzt nicht: nur 12.0 % der Fotopunkte liegen auf "
             "der Oberfläche der Wolke.")
_KORREKTUR = {"stufe": "neigung+versatz", "auf_flaeche_nachher": 0.78,
              "neigung_grad": [0.12, -0.05], "versatz_m": [0.1, -0.2],
              "M": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
              "v": [0.1, -0.2, 0.0]}
_THERMAL_KAL = {"f": 40.0, "cx": 50.0, "cy": 50.0, "k1": 0.0, "k2": 0.0,
                "breite": 100, "hoehe": 100, "drehung_grad": [0.0, 0.0, 0.0],
                "rgb_faktor": 1.0}
_FOTOS = ("a_V.JPG", "b_V.JPG", "c_V.JPG", "a_T.JPG", "b_T.JPG")


def fall(fn):
    """Trägt einen Fall ein; danach laufen die wartenden Jobs und der Zustand
    wird festgehalten."""
    def lauf(s):
        s.fluechtig(r"\(\d+ ms\)", "(… ms)")      # Dauer der Vorschau im Status
        fn(s)
        s.laufe()
        s.warte()
        # Dialogtexte nennen den Projektordner; sein Schlüssel hängt am
        # Arbeitsordner und heißt je Lauf anders
        for e in s.ereignisse:
            if e[0] == "dialog":
                e[1] = {k: s.text(v) if isinstance(v, str) else v
                        for k, v in e[1].items()}
        _zustand(s)
    FAELLE[fn.__name__] = lauf
    return fn


# ------------------------------------------------------------ Stellvertreter

class _Vorschau:
    """Steht für ``meander.LivePreview``: färbt einen festen Anteil der Punkte."""

    def __init__(self, stubs, art: str, n_bilder: int = 3):
        self._stubs = stubs
        self.art = art
        self._kurz = f"StubVorschau {art}"
        self.bilder = [None] * n_bilder

    def colorize(self, punkte, A, b, cams=None):
        s = self._stubs.s
        s.blattaufrufe += 1
        s.ereignis("aufruf", f"meander.LivePreview.colorize[{self.art}]",
                   s.kurz((punkte, A, b)), s.kurz({"cams": cams}))
        if "vorschau" in self._stubs.scheitert:
            raise RuntimeError(self._stubs.scheitert["vorschau"])
        return _farben(len(punkte), self._stubs.treffer)


class _Netz:
    _kurz = "StubNetz"


def _farben(n: int, anteil: float, temperatur: bool = False) -> tuple:
    rgb = np.empty((n, 3), dtype=np.uint8)
    rgb[:] = (np.arange(n) % 251)[:, None]
    maske = np.zeros(n, dtype=bool)
    maske[: int(round(anteil * n))] = True
    if not temperatur:
        return rgb, maske
    temp = np.where(maske, 20.0 + (np.arange(n) % 15), np.nan).astype(np.float32)
    return rgb, maske, temp


def _melde(progress, f: float, m: str) -> None:
    """Jeder Stellvertreter meldet zwei verschiedene Anteile: erst damit liegt
    eine Spanne ``a + b·f`` des Aufrufers an Anfang und Ende fest."""
    if progress is not None:
        progress(f, m)


def _abbruch(cancel) -> None:
    if cancel is not None and cancel():
        raise RuntimeError("Abgebrochen")


class Stubs:
    """Ersetzt alle Blattfunktionen der Familie; die Attribute sind die
    Stellschrauben eines Falls."""

    def __init__(self, s, **wahl):
        self.s = s
        self.thermal = False            # die Pipeline hat Thermalkameras
        self.aus_cache = True           # align fand eine align.json vor
        self.fraglich = None            # Meldung von pruefe_ausrichtung
        self.colmap = "/opt/colmap/bin/python"
        self.treffer = 0.6              # Anteil getroffener Punkte der Vorschau
        self.dz = 0.25                  # Höhenkorrektur aus dem Einmessen
        self.einmessen_thermal = False  # Einmessen liefert eine Thermaloptik
        self.fein = dict(_KORREKTUR)
        self.rohwerte = True            # die Thermalbilder tragen Temperaturen
        self.splat_python = _SPLAT_PYTHON
        self.splat_info = dict(_SPLAT_INFO)
        self.fusion_unplausibel = None
        self.geometrien: list = []      # Antworten von load_geometry der Reihe nach
        self.hybrid = True
        self.scheitert: dict = {}       # Blattfunktion -> Fehlermeldung
        fremd = set(wahl) - set(vars(self))
        assert not fremd, fremd
        vars(self).update(wahl)
        self._einbauen()

    def _einbauen(self) -> None:
        for objekt, ersatz, name in (
                (meander_mod.build_pipeline, self._build_pipeline, "meander.build_pipeline"),
                (meander_mod.prepare, self._prepare, "meander.prepare"),
                (meander_mod.align, self._align, "meander.align"),
                (meander_mod.pruefe_ausrichtung, lambda k: self.fraglich,
                 "meander.pruefe_ausrichtung"),
                (meander_mod.colorize_points, self._colorize, "meander.colorize_points"),
                (meander_mod.LivePreview, self._live_preview, "meander.LivePreview"),
                (meander_mod.set_manual, self._set_manual, "meander.set_manual"),
                (meander_mod.find_colmap_python, lambda: self.colmap,
                 "meander.find_colmap_python"),
                (optik_mod.einmessen, self._einmessen, "optik.einmessen"),
                (optik_mod.feinausrichten, self._feinausrichten, "optik.feinausrichten"),
                (optik_mod.rgb_cams, lambda pipe, faktor=1.0: pipe.rgb_cams(),
                 "optik.rgb_cams"),
                (optik_mod.thermal_cams,
                 lambda pipe, kal, faktor=1.0, rgb_faktor=1.0: pipe.thermal_cams(),
                 "optik.thermal_cams"),
                (sichtbar_mod.normalen, self._normalen, "sichtbar.normalen"),
                (sichtbar_mod.colorize_sichtbar, self._colorize,
                 "sichtbar.colorize_sichtbar"),
                (temperatur_mod.quelle, self._quelle, "temperatur.quelle"),
                (splat_mod.find_splat_python,
                 lambda: (self.splat_python, dict(self.splat_info)),
                 "splat.find_splat_python"),
                (splat_mod.anker, self._anker, "splat.anker"),
                (splat_mod.datensatz_maeander, self._datensatz_maeander,
                 "splat.datensatz_maeander"),
                (splat_mod.datensatz_onboard, self._datensatz_onboard,
                 "splat.datensatz_onboard"),
                (splat_mod.datensatz_gemeinsam, self._datensatz_gemeinsam,
                 "splat.datensatz_gemeinsam"),
                (splat_mod.trainieren, self._trainieren, "splat.trainieren"),
                (splat_mod.punkt_farben, self._punkt_farben, "splat.punkt_farben"),
                (splat_mod.pruef_namen, lambda ordner: ["a"], "splat.pruef_namen"),
                (splat_mod.vergleich, self._vergleich, "splat.vergleich"),
                (splat_mod.probe, lambda ak, *a, **k: np.arange(0, len(ak["index"]), 4),
                 "splat.probe"),
                (fusion_mod.fusioniere, self._fusioniere, "fusion.fusioniere"),
                (mesh_mod.build_geometry, self._build_geometry, "mesh.build_geometry"),
                (mesh_mod.load_geometry, self._load_geometry, "mesh.load_geometry"),
                (mesh_mod.save_geometry, self._save_geometry, "mesh.save_geometry"),
                (mesh_mod.vertex_colors,
                 lambda geom, farben, gueltig: (np.full((len(geom["vertices"]), 3), 120,
                                                        np.uint8),
                                                np.ones(len(geom["vertices"]), bool)),
                 "mesh.vertex_colors"),
                (mesh_mod.vertex_scalar,
                 lambda geom, werte: np.linspace(0.0, 255.0, len(geom["vertices"]),
                                                 dtype=np.float32),
                 "mesh.vertex_scalar"),
                (mesh_mod.rest_maske, self._rest_maske, "mesh.rest_maske"),
                (mesh_mod.write_mesh, self._write_mesh, "mesh.write_mesh"),
                (mesh_mod.to_open3d, lambda geom, rgb=None: _Netz(), "mesh.to_open3d"),
                (mesh_mod.open_in_cloudcompare, lambda paths: ["cloudcompare", *paths],
                 "mesh.open_in_cloudcompare")):
            self.s.ersetze(objekt, ersatz=ersatz, name=name)

    def _fehler(self, name: str) -> None:
        if name in self.scheitert:
            raise RuntimeError(self.scheitert[name])

    # ---------------------------------------------------------- Mäander

    def pipeline(self, mit_lage: bool = True, photo_dir=None, work_dir=None):
        f = self.s.fenster
        arbeit = work_dir or f._project.meander_work_dir()
        p = self.s.stub_pipeline(thermal=self.thermal, mit_lage=mit_lage)
        p.photo_dir = photo_dir or f._meander_dir
        p.work_dir = arbeit
        p._p = lambda name: os.path.join(arbeit, name)
        p.rgb_versatz = [0.0, 0.0]
        p.s360_korrektur = None
        p._cancel = lambda: False
        p.save_align = lambda: None
        return p

    def vorschau(self, art: str = "rgb") -> _Vorschau:
        return _Vorschau(self, art)

    def _build_pipeline(self, points, photo_dir, work_dir, **k):
        return self.pipeline(mit_lage=False, photo_dir=photo_dir, work_dir=work_dir)

    def _prepare(self, pipe, progress=None):
        _melde(progress, 0.3, "COLMAP-Modell …")
        _abbruch(pipe._cancel)
        _melde(progress, 0.9, "bereit zum Ausrichten")
        return {"kameras": 20, "massstab": 1.02, "gps_residuum": 0.4}

    def _align(self, pipe, progress=None):
        self._fehler("align")
        _melde(progress, 0.05, "Suche Gierwinkel und Verschiebung …")
        pipe.yaw = float(np.radians(30.0))
        pipe.t = np.array([1.0, 2.0, 0.5])
        _melde(progress, 1.0, "Ausgerichtet: 30.00°")
        return {"yaw_deg": 30.0, "t": [1.0, 2.0, 0.5], "aus_cache": self.aus_cache,
                "anteil_auf_flaeche": 0.12 if self.fraglich else 0.62,
                "median_abweichung": 0.21}

    def _live_preview(self, cams, image_dir, progress=None, cancel=None, **k):
        _melde(progress, 0.0, "Lade Vorschaubild 1/3 …")
        _abbruch(cancel)
        _melde(progress, 1.0, "Lade Vorschaubild 3/3 …")
        return _Vorschau(self, "thermal" if os.path.basename(image_dir) == "thermal"
                         else "rgb")

    @staticmethod
    def _set_manual(pipe, yaw_deg, t):
        pipe.yaw = float(np.radians(yaw_deg))
        pipe.t = meander_mod.as_t3(t)

    def _einmessen(self, pipe, punkte, yaw_deg, t, foto_ordner, thermal=True,
                   progress=None, cancel=None, log=None):
        _melde(progress, 0.2, "Höhe über den Entfernungsmesser …")
        _abbruch(cancel)
        self._fehler("einmessen")
        if log is not None:
            log("Einmessen: Stellvertreter.")
        _melde(progress, 1.0, "Optik eingemessen")
        return {"dz": self.dz, "hoehe": {"dz": self.dz, "bilder": 12},
                "rgb": {"faktor": 1.086, "auf_flaeche_vorher": 0.44,
                        "auf_flaeche_nachher": 0.71},
                "thermal": dict(_THERMAL_KAL) if self.einmessen_thermal and thermal
                else None}

    def _feinausrichten(self, pipe, punkte, yaw_deg, t, rgb_faktor, progress=None,
                        cancel=None, log=None):
        _melde(progress, 0.0, "Neigung und Höhe …")
        _abbruch(cancel)
        self._fehler("feinausrichten")
        _melde(progress, 1.0, "Feinausrichtung fertig")
        return dict(self.fein)

    def _normalen(self, punkte, raster=0.15, progress=None):
        _melde(progress, 0.0, "Normalen …")
        _melde(progress, 1.0, "Normalen fertig")
        normalen = np.zeros((len(punkte), 3), dtype=np.float32)
        normalen[:, 2] = 1.0
        return normalen

    def _colorize(self, points, cams, image_dir, A, b, progress=None, cancel=None,
                  temperatur=None, **k):
        _melde(progress, 0.0, "Bild 1/3 …")
        _abbruch(cancel)
        self._fehler("colorize")
        _melde(progress, 1.0, "Bild 3/3 …")
        thermal = os.path.basename(image_dir) == "thermal"
        return _farben(len(points), 0.5 if thermal else 0.8, temperatur is not None)

    def _quelle(self, pipe):
        return (lambda i, name: None) if self.rohwerte else None

    # ------------------------------------------------------------ Splat

    def _anker(self, punkte, voxel=0.05, max_anker=0, progress=None, cancel=None,
               log=None):
        _melde(progress, 0.0, "Anker …")
        _melde(progress, 1.0, "Anker gesetzt")
        normal = np.zeros((50, 3), dtype=np.float32)
        normal[:, 2] = 1.0
        return {"pos": np.zeros((50, 3), dtype=np.float32), "voxel": float(voxel),
                "index": np.arange(len(punkte)) % 50, "normal": normal}

    def _datensatz_maeander(self, ordner, ak, punkte, cams, bild_ordner, A, b,
                            progress=None, **k):
        os.makedirs(ordner, exist_ok=True)
        _melde(progress, 0.0, "Schreibe den Datensatz …")
        _melde(progress, 1.0, "Datensatz geschrieben")
        return {"ansichten": 3, "pruef": 1 if k.get("halte_jedes") else 0}

    def _datensatz_onboard(self, ordner, ak, punkte, rec, teile, calib, T,
                           progress=None, **k):
        os.makedirs(ordner, exist_ok=True)
        _melde(progress, 0.0, "Schreibe den Datensatz …")
        _melde(progress, 1.0, "Datensatz geschrieben")
        return {"ansichten": 30, "frames": 6, "abdeckung": 0.83}

    def _datensatz_gemeinsam(self, ordner, ak, punkte, maeander=None, onboard=None,
                             abbildung=None, progress=None, **k):
        os.makedirs(ordner, exist_ok=True)
        _melde(progress, 0.0, "Schreibe den Datensatz …")
        _melde(progress, 1.0, "Datensatz geschrieben")
        return {"ansichten": 33, "maeander": 3, "onboard": 30, "frames": 6,
                "startabbildung": None if abbildung is None else "fusion"}

    def _trainieren(self, python, ordner, progress=None, cancel=None, log=None,
                    alle_bilder=False, **k):
        _melde(progress, 0.0, "Splat: Schritt 0/10, PSNR 12.0 dB")
        _abbruch(cancel)
        self._fehler("trainieren")
        _melde(progress, 1.0, "Splat: Schritt 10/10, PSNR 24.0 dB")
        bericht = {
            "posen": {"dreh_grad_median": 0.05, "dreh_grad_max": 0.2,
                      "weg_m_median": 0.01, "weg_m_max": 0.04},
            "pruefung": [] if alle_bilder else [
                {"psnr_angepasst": 25.0, "herkunft": "maeander"},
                {"psnr_angepasst": 22.0, "herkunft": "onboard"}],
            "belichtung_frei": {"M": [[1.05, 0.0, 0.0], [0.0, 1.0, 0.0],
                                      [0.0, 0.0, 0.95]], "t": [0.01, 0.0, -0.01]}}
        with open(os.path.join(ordner, "bericht.json"), "w", encoding="utf-8") as fh:
            json.dump(bericht, fh)
        return {"bericht": bericht,
                "pruefung": [] if alle_bilder else [{"psnr_angepasst": 24.5},
                                                    {"psnr_angepasst": 23.5}]}

    @staticmethod
    def _punkt_farben(ordner, n_punkte):
        if "thermal" in os.path.basename(ordner):
            rgb, maske, temp = _farben(int(n_punkte), 0.75, True)
            return {"rgb": rgb, "maske": maske, "temperatur": temp}
        rgb, maske = _farben(int(n_punkte), 0.75)
        return {"rgb": rgb, "maske": maske}

    @staticmethod
    def _vergleich(ordner, ak, punkte, normalen, werte, **k):
        return {"bilder": 1, "punkte": 150,
                "methoden": {"splat": {"roh": 9.0, "angepasst": 6.0},
                             "direkt": {"roh": 10.0, "angepasst": 7.5}}}

    def _fusioniere(self, onboard, maeander, normalen, progress=None, cancel=None, **k):
        _melde(progress, 0.0, "Schätze die Farbabbildung …")
        _abbruch(cancel)
        self._fehler("fusioniere")
        rgb, maske = _farben(len(normalen), 0.9)
        _melde(progress, 1.0, "Fusion fertig")
        return {"rgb": rgb, "maske": maske, "M": np.diag([1.1, 1.0, 0.9]),
                "t": np.array([0.01, 0.0, -0.01]),
                "bericht": {"ueberlappung": 120, "abstand_vorher": 40.0,
                            "abstand_nachher": 12.0, "anteil": 0.9,
                            "unplausibel": self.fusion_unplausibel}}

    # ------------------------------------------------------------- Mesh

    def geometrie(self) -> dict:
        n = int(self.s.fenster._rec.n_points)
        normals = np.zeros((4, 3), dtype=np.float32)
        normals[:, 2] = 1.0
        return {"vertices": np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
                                     dtype=np.float32),
                "triangles": np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
                "normals": normals, "vertex_voxel": np.arange(4),
                "point_voxel": np.arange(n) % 4,
                "zelle_rest": np.array([False, False, False, True]) if self.hybrid
                else None,
                "stats": {"version": 2, "dreiecke": 2, "zellen": 4, "flaechenzellen": 3,
                          "restzellen": 1, "voxel_m": 0.04, "tiefe": 12,
                          "hybrid": self.hybrid, "normalen": "cpu"}}

    def _load_geometry(self, path):
        da = self.geometrien.pop(0) if self.geometrien else False
        if not da:
            return None
        os.makedirs(path, exist_ok=True)
        return self.geometrie()

    def _build_geometry(self, points, sensor_path, progress=None, cancel=None,
                        log=None, **k):
        _melde(progress, 0.0, "Mesh: Normalen …")
        _abbruch(cancel)
        self._fehler("build_geometry")
        _melde(progress, 1.0, "Mesh: Poisson …")
        if log is not None:
            log("Mesh: Stellvertreter.")
        return self.geometrie()

    @staticmethod
    def _save_geometry(geom, path):
        os.makedirs(path, exist_ok=True)
        return path

    def _rest_maske(self, geom):
        if geom.get("zelle_rest") is None:
            return None
        return np.asarray(geom["zelle_rest"])[np.asarray(geom["point_voxel"])]

    @staticmethod
    def _write_mesh(mesh, path, stats=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(b"ply\nStellvertreter\n")
        return path


# ------------------------------------------------------------------ Aufbau

def _fotos(s, namen=_FOTOS, ordner: str = "maeander_fotos") -> str:
    pfad = os.path.join(s.ordner, ordner)
    os.makedirs(pfad, exist_ok=True)
    for name in namen:
        open(os.path.join(pfad, name), "wb").close()
    s.dateien_merken()                    # die Fotos sind Eingabe, nicht Ergebnis
    return pfad


def _ebene(projekt, key: str, n: int, anteil: float = 0.8, meta: dict | None = None,
           temperatur: bool = False) -> None:
    """Farbebene von Hand ins Projekt legen (Format wie ``meander.save_layer``)."""
    ordner = projekt.layer_dir(key)
    rgb = np.random.default_rng(len(key)).integers(0, 256, size=(n, 3), dtype=np.uint8)
    maske = np.zeros(n, dtype=np.uint8)
    maske[: int(round(anteil * n))] = 1
    rgb.tofile(os.path.join(ordner, "colors.bin"))
    maske.tofile(os.path.join(ordner, "valid.bin"))
    if temperatur:
        np.where(maske > 0, 18.0 + (np.arange(n) % 20), np.nan).astype(
            np.float32).tofile(os.path.join(ordner, "temperatur.bin"))
    with open(os.path.join(ordner, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(dict({"quelle": key}, **(meta or {})), fh)


def _flug(s, bag: bool = True, flug: bool = True, modell: bool = True,
          ebenen=(), arbeit: dict | None = None):
    """Offener Flug mit Karte: Projekt, Aufzeichnung, Wolke, Mäanderordner.

    ``ebenen``: Schlüssel oder ``(Schlüssel, {anteil, meta, temperatur, punkte})``;
    ``arbeit``: JSON-Dateien für den Mäander-Arbeitsordner. Was hier entsteht,
    zählt nicht als vom Ablauf geschrieben.
    """
    from core.colorizer import rec_fingerprint

    f = s.fenster
    projekt = s.stub_projekt()
    rec = s.stub_aufzeichnung(ordner=projekt.recording_dir())
    f._project, f._rec, f._world = projekt, rec, rec.world_points()
    if bag:
        f._bag = s.stub_bag()
    f._calib = os.path.join(s.ordner, "kalibrierung.json")
    if flug:
        f._meander_dir = _fotos(s)
    werk = projekt.meander_work_dir()
    if modell:
        open(os.path.join(werk, "cameras.npz"), "wb").close()
    for name, inhalt in (arbeit or {}).items():
        with open(os.path.join(werk, name), "w", encoding="utf-8") as fh:
            json.dump(inhalt, fh)
    for eintrag in ebenen:
        key, wahl = (eintrag, {}) if isinstance(eintrag, str) else eintrag
        wahl = dict(wahl)
        meta = dict(wahl.pop("meta", {}))
        if key == "onboard":
            meta.setdefault("rec_fingerprint", rec_fingerprint(rec))
            meta.setdefault("extrinsic", f._extrinsic_from_spins().tolist())
        _ebene(projekt, key, wahl.pop("punkte", int(rec.n_points)), meta=meta, **wahl)
    s.dateien_merken()
    f._reload_layers()
    f._update_enabled()
    return projekt


def _lage(s, st: Stubs, bilder: bool = True) -> None:
    """Ausgerichtete Pipeline, auf Wunsch mit geladenen Vorschaubildern."""
    f = s.fenster
    f._meander_pipe = st.pipeline()
    if bilder:
        f._live = st.vorschau("rgb")
        f._live_th = st.vorschau("thermal") if st.thermal else None
        f._live_pts = np.ascontiguousarray(f._world, dtype=np.float64)
    f._update_enabled()


def _still(regler, wert) -> None:
    regler.blockSignals(True)
    regler.setValue(wert)
    regler.blockSignals(False)


def _mesh_an(s) -> None:
    """Schalter der Leiste umlegen, wie es der Klick täte."""
    f = s.fenster
    f._cloud_view.set_mesh_schalter(True)
    f._on_leiste_mesh(True)


def _zustand(s) -> None:
    f = s.fenster

    def hole(fn):
        try:
            return s.kurz(fn())
        except (AttributeError, KeyError, TypeError):
            return "<fehlt>"

    pipe = f._meander_pipe
    menue = f._actions.get("menu_farbquelle")
    s.ereignis("zustand", {
        "flug": hole(lambda: f._meander_dir),
        "pipeline": hole(lambda: pipe),
        "lage": hole(lambda: None if pipe is None or pipe.yaw is None else
                     [float(np.degrees(pipe.yaw)), [float(x) for x in pipe.t]]),
        "korrektur": hole(lambda: getattr(pipe, "s360_korrektur", None)),
        "optik": hole(lambda: f._optik),
        "automatik": hole(lambda: [f._auto_kette, f._optik_neu_messen]),
        "vorschaubilder": hole(lambda: [f._live, f._live_th, f._live_optik]),
        "vorschau_sichtbar": hole(lambda: f._cloud_view.has_color_preview()),
        "regler_rgb": hole(lambda: {k: r.value() for k, r in f._spin_meander.items()}),
        "regler_thermal": hole(lambda: {k: r.value()
                                        for k, r in f._spin_meander_th.items()}),
        "massstab": hole(lambda: {k: r.value() for k, r in f._massstab.items()}),
        "thermal_haken": hole(lambda: [f._chk_thermal.isEnabled(),
                                       f._chk_thermal.isChecked()]),
        "ebenen": hole(lambda: sorted(f._layers)),
        "ebene": hole(lambda: f._layer_key),
        "temperaturen": hole(lambda: sorted(f._temperaturen)),
        "temperatur": hole(lambda: f._temperatur),
        "farbmodus": hole(lambda: f._combo_colormode.currentData()),
        "farbquellen": hole(lambda: [[f._combo_layer.itemData(i),
                                      f._combo_layer.itemText(i)]
                                     for i in range(f._combo_layer.count())]),
        "farbquelle_frei": hole(lambda: f._combo_layer.isEnabled()),
        "menue": hole(lambda: [[a.data(), a.isChecked()] for a in menue.actions()]),
        "nur_eingefaerbt": hole(lambda: f._chk_only_colored.isChecked()),
        "mesh": hole(lambda: [f._cloud_view.mesh_an(), f._mesh_geom is not None,
                              f._mesh_wartet, f._mesh_laeuft]),
        "beschaeftigt": hole(lambda: f._busy),
    })


# -------------------------------------------------- Mäanderflug wählen

@fall
def waehlen_ohne_karte(s):
    Stubs(s)
    s.fenster._on_meander_pick()


@fall
def waehlen_abgebrochen(s):
    Stubs(s)
    _flug(s, flug=False)
    s.fenster._on_meander_pick()


@fall
def waehlen_mit_thermal(s):
    Stubs(s)
    _flug(s, flug=False)
    s.antworte("QFileDialog.getExistingDirectory", _fotos(s))
    s.fenster._on_meander_pick()


@fall
def waehlen_ohne_thermal(s):
    st = Stubs(s)
    _flug(s, flug=False)
    f = s.fenster
    f._chk_thermal.setChecked(True)
    _lage(s, st)                          # Pipeline und Vorschau des alten Fluges
    s.antworte("QFileDialog.getExistingDirectory",
               _fotos(s, ("a_V.JPG", "b_V.JPG"), "nur_rgb"))
    f._on_meander_pick()


@fall
def waehlen_ohne_v_namen(s):
    Stubs(s)
    _flug(s, flug=False)
    s.antworte("QFileDialog.getExistingDirectory",
               _fotos(s, ("x.jpg", "y.JPEG", "z_T.JPG", "notiz.txt"), "fremd"))
    s.fenster._on_meander_pick()


@fall
def waehlen_ohne_jpeg(s):
    Stubs(s)
    _flug(s, flug=False)
    s.antworte("QFileDialog.getExistingDirectory", _fotos(s, ("notiz.txt",), "leer"))
    s.fenster._on_meander_pick()


# ------------------------------------------- Rückfrage zur Rekonstruktion

@fall
def rueckfrage_nein(s):
    Stubs(s)
    _flug(s, modell=False)
    s.antworte("QMessageBox.question", QMessageBox.No)
    s.fenster._on_meander_align()


@fall
def rueckfrage_ohne_interpreter(s):
    Stubs(s, colmap=None)
    _flug(s, modell=False)
    s.fenster._on_meander_align()


# ------------------------------------------------------------ Ausrichten

@fall
def ausrichten_ohne_flug(s):
    Stubs(s)
    _flug(s, flug=False)
    s.fenster._on_meander_align()


@fall
def ausrichten_aus_cache(s):
    Stubs(s)
    _flug(s, arbeit={
        "rgb_zuschlag.json": {"yaw": 0.5, "x": 0.1, "y": -0.2, "z": 0.0},
        "thermal_lage.json": {"gier_grad": 0.3, "x": 0.05, "y": 0.0},
        "optik_kalibrierung.json": {"rgb_faktor": 1.05, "thermal_faktor": 1.0,
                                    "thermal": None, "korrektur": _KORREKTUR}})
    s.fenster._on_meander_align()


@fall
def ausrichten_frisch(s):
    # ohne Modell: Rückfrage, Antwort Ja. Neue Lage: Feinausrichtung gilt nicht
    # mehr, Höhe und Brennweite werden gleich neu gemessen.
    Stubs(s, aus_cache=False, thermal=True, einmessen_thermal=True)
    _flug(s, modell=False, arbeit={
        "rgb_zuschlag.json": {"yaw": 0.5, "x": 0.1, "y": -0.2, "z": 0.0},
        "optik_kalibrierung.json": {"rgb_faktor": 1.05, "thermal_faktor": 1.0,
                                    "thermal": None, "korrektur": _KORREKTUR}})
    s.fenster._chk_thermal.setChecked(True)
    s.fenster._on_meander_align()


@fall
def ausrichten_fraglich(s):
    Stubs(s, fraglich=_FRAGLICH)
    projekt = _flug(s, modell=False, arbeit={
        "optik_kalibrierung.json": {"rgb_faktor": 1.0, "thermal_faktor": 1.0,
                                    "thermal": None}})
    # fertiges COLMAP-Modell als sparse/0: keine Rückfrage
    sparse = os.path.join(projekt.meander_work_dir(), "sparse", "0")
    os.makedirs(sparse)
    open(os.path.join(sparse, "cameras.bin"), "wb").close()
    s.dateien_merken()
    s.fenster._on_meander_align()


@fall
def ausrichten_fehlschlag(s):
    Stubs(s, scheitert={"align": "Keine Geotags in den Bildern."})
    _flug(s)
    s.fenster._on_meander_align()


# --------------------------------------------------------- Live-Vorschau

@fall
def vorschau_laden_erstes_einmessen(s):
    # Optik noch nie eingemessen: läuft nach den Vorschaubildern von selbst
    st = Stubs(s, thermal=True)
    _flug(s)
    _lage(s, st, bilder=False)
    s.fenster._start_live_preview()


@fall
def vorschau_laden_abgebrochen(s):
    st = Stubs(s)
    _flug(s)
    _lage(s, st, bilder=False)
    s.fenster._start_live_preview()
    s.fenster._on_cancel()


# ------------------------------------------- Optik einmessen, Feinausrichten

@fall
def einmessen_ohne_lage(s):
    Stubs(s)
    _flug(s)
    s.fenster._on_meander_einmessen()


@fall
def einmessen(s):
    st = Stubs(s, thermal=True, einmessen_thermal=True)
    _flug(s)
    _lage(s, st)
    _still(s.fenster._spin_meander["x"], 0.4)       # Handzuschlag wird verrechnet
    s.fenster._on_meander_einmessen()


@fall
def einmessen_mit_vorschau(s):
    st = Stubs(s, dz=0.0)
    _flug(s)
    _lage(s, st)
    f = s.fenster
    f._live_update()                      # Vorschau steht in der Ansicht
    f._on_meander_einmessen()
    s.laufe()
    s.warte(400)                          # zieht die Vorschau nach


@fall
def feinausrichten_ohne_lage(s):
    Stubs(s)
    _flug(s)
    s.fenster._on_meander_fein()


@fall
def feinausrichten(s):
    st = Stubs(s)
    _flug(s)
    _lage(s, st)
    _still(s.fenster._massstab["rgb"], 5.0)
    s.fenster._on_meander_fein()


@fall
def feinausrichten_ohne_gewinn(s):
    st = Stubs(s, fein=dict(_KORREKTUR, stufe="nichts"))
    _flug(s, arbeit={"optik_kalibrierung.json": {
        "rgb_faktor": 1.0, "thermal_faktor": 1.0, "thermal": None,
        "korrektur": _KORREKTUR}})
    _lage(s, st)
    s.fenster._meander_lade_optik()
    s.fenster._on_meander_fein()


# -------------------------------------------------------- Automatik-Kette

@fall
def automatik_ohne_flug(s):
    Stubs(s)
    _flug(s, flug=False)
    s.fenster._on_meander_auto()


@fall
def automatik_bis_fertig(s):
    Stubs(s)
    _flug(s, modell=False)                # Rückfrage, Antwort Ja (zweimal gestellt)
    s.fenster._on_meander_auto()


@fall
def automatik_mit_lage(s):
    st = Stubs(s, thermal=True, einmessen_thermal=True)
    _flug(s)
    _lage(s, st)
    s.fenster._chk_thermal.setChecked(True)
    s.fenster._on_meander_auto()


@fall
def automatik_fehlschlag(s):
    st = Stubs(s, scheitert={"feinausrichten": "Zu wenige Fotopunkte auf der Karte."})
    _flug(s)
    _lage(s, st)
    s.fenster._on_meander_auto()


@fall
def automatik_abgebrochen(s):
    st = Stubs(s)
    _flug(s)
    _lage(s, st)
    s.abbruch_nach(1)
    s.fenster._on_meander_auto()


# ------------------------------------------------------ Mäander-Einfärben

@fall
def einfaerben_ohne_flug(s):
    Stubs(s)
    _flug(s, flug=False)
    s.fenster._on_meander_run()


@fall
def einfaerben_mit_pipeline(s):
    st = Stubs(s)
    _flug(s, arbeit={"optik_kalibrierung.json": {
        "rgb_faktor": 1.05, "thermal_faktor": 1.0, "thermal": None,
        "korrektur": _KORREKTUR}})
    _lage(s, st)
    f = s.fenster
    f._meander_lade_optik()
    _still(f._spin_meander["yaw"], 1.5)
    f._on_meander_run()


@fall
def einfaerben_ohne_sichtpruefung_thermal(s):
    st = Stubs(s, thermal=True)
    _flug(s, arbeit={"thermal_lage.json": {"gier_grad": 0.3, "x": 0.05, "y": 0.0}})
    _lage(s, st)
    f = s.fenster
    f._meander_lade_thermal()
    f._chk_thermal.setChecked(True)
    f._chk_sichtbar.setChecked(False)
    f._on_meander_run()


@fall
def einfaerben_sichtpruefung_thermal_ohne_rohwerte(s):
    st = Stubs(s, thermal=True, rohwerte=False)
    _flug(s, arbeit={"optik_kalibrierung.json": {
        "rgb_faktor": 1.0, "thermal_faktor": 1.02, "thermal": _THERMAL_KAL}})
    _lage(s, st)
    f = s.fenster
    f._meander_lade_optik()
    f._chk_thermal.setChecked(True)
    f._on_meander_run()


@fall
def einfaerben_thermal_ohne_optik(s):
    st = Stubs(s)                         # die Pipeline hat keine Thermalkameras
    _flug(s)
    _lage(s, st)
    s.fenster._chk_thermal.setChecked(True)
    s.fenster._on_meander_run()


@fall
def einfaerben_ohne_pipeline(s):
    # Projekt frisch geöffnet: Lage, Thermal-Zuschlag und Optik aus dem Arbeitsordner
    Stubs(s, thermal=True)
    _flug(s, arbeit={
        "thermal_lage.json": {"gier_grad": 0.3, "x": 0.05, "y": 0.0},
        "optik_kalibrierung.json": {"rgb_faktor": 1.05, "thermal_faktor": 1.02,
                                    "thermal": _THERMAL_KAL, "korrektur": _KORREKTUR}})
    f = s.fenster
    f._chk_thermal.setChecked(True)
    f._chk_sichtbar.setChecked(False)
    f._on_meander_run()


@fall
def einfaerben_ohne_pipeline_fraglich_eingemessen(s):
    Stubs(s, fraglich=_FRAGLICH)
    _flug(s, arbeit={"optik_kalibrierung.json": {
        "rgb_faktor": 1.086, "thermal_faktor": 1.0, "thermal": None,
        "rgb": {"faktor": 1.086, "auf_flaeche_nachher": 0.71}}})
    s.fenster._on_meander_run()


@fall
def einfaerben_ohne_pipeline_fraglich(s):
    Stubs(s, fraglich=_FRAGLICH)
    _flug(s)
    s.fenster._on_meander_run()


@fall
def einfaerben_abgebrochen(s):
    st = Stubs(s)
    _flug(s)
    _lage(s, st)
    s.fenster._live_update()              # Vorschau verdeckt die Karte nicht weiter
    s.abbruch_nach(2)
    s.fenster._on_meander_run()


# ------------------------------------------------------------ Handjustage

@fall
def handjustage_ohne_lage(s):
    Stubs(s)
    _flug(s)
    s.fenster._spin_meander["x"].setValue(0.4)
    s.warte(600)


@fall
def handjustage_ohne_bilder(s):
    st = Stubs(s)
    _flug(s)
    _lage(s, st, bilder=False)
    s.fenster._spin_meander["x"].setValue(0.4)
    s.warte(600)                          # der Zuschlag wird trotzdem gespeichert


@fall
def handjustage_rgb(s):
    st = Stubs(s)
    _flug(s)
    _lage(s, st)
    f = s.fenster
    f._spin_meander["x"].setValue(0.4)
    f._spin_meander["yaw"].setValue(1.5)
    s.warte(600)
    f._chk_solo.setChecked(True)
    f._spin_meander["z"].setValue(-0.25)
    s.warte(600)


@fall
def handjustage_thermal(s):
    st = Stubs(s, thermal=True)
    _flug(s)
    _lage(s, st)
    f = s.fenster
    f._spin_meander_th["x"].setValue(0.3)
    f._spin_meander_th["yaw"].setValue(-0.5)
    s.warte(400)


@fall
def handjustage_kaum_treffer(s):
    st = Stubs(s, treffer=0.02)
    _flug(s)
    _lage(s, st)
    f = s.fenster
    f._spin_meander["x"].setValue(0.4)
    s.warte(600)
    f._spin_meander["x"].setValue(0.8)    # die Warnung kommt nur einmal
    s.warte(600)


@fall
def massstab(s):
    st = Stubs(s, thermal=True)
    _flug(s)
    _lage(s, st)
    f = s.fenster
    f._massstab["rgb"].setValue(5.0)
    s.warte(600)
    f._massstab["thermal"].setValue(-2.0)
    s.warte(400)


@fall
def lage_uebernehmen(s):
    # _meander_apply_manual: Zuschlag in die Pipeline, Regler auf null, beides ins Projekt
    st = Stubs(s, thermal=True)
    _flug(s)
    _lage(s, st)
    f = s.fenster
    for key, wert in (("yaw", 1.5), ("x", 0.4), ("y", -0.3), ("z", 0.2)):
        _still(f._spin_meander[key], wert)
    _still(f._spin_meander_th["x"], 0.3)
    f._meander_apply_manual()


# --------------------------------------------------------- Ausrichtfenster

@fall
def ausrichtfenster_ohne_lage(s):
    Stubs(s)
    _flug(s)
    s.fenster._on_meander_fenster()


@fall
def ausrichtfenster_ohne_bilder(s):
    st = Stubs(s)
    _flug(s)
    _lage(s, st, bilder=False)
    f = s.fenster
    f._on_meander_fenster()
    s.warte(200)
    f._meander_fenster.close()


@fall
def ausrichtfenster_lage_uebernehmen(s):
    st = Stubs(s, thermal=True)
    _flug(s, arbeit={"optik_kalibrierung.json": {
        "rgb_faktor": 1.05, "thermal_faktor": 1.0, "thermal": None}})
    _lage(s, st)
    f = s.fenster
    f._meander_lade_optik()
    _still(f._spin_meander["x"], 0.4)     # wird vor dem Öffnen verrechnet
    _still(f._spin_meander_th["yaw"], 0.5)
    f._live_update()
    f._on_meander_fenster()
    s.warte(200)
    fenster = f._meander_fenster
    s.ereignis("ausrichtfenster", fenster.windowTitle(), s.kurz(fenster.ergebnis()))
    fenster.uebernommen.emit({"yaw": 33.5, "t": np.array([1.5, 2.25, 0.5]),
                              "thermal": (0.5, 0.1, -0.1), "rgb_faktor": 1.08,
                              "thermal_faktor": 0.98})
    fenster.close()
    s.warte(400)                          # die sichtbare Vorschau zieht nach


# -------------------------------------------------------- Gaussian Splat

@fall
def gpu_pruefen(s):
    Stubs(s)
    s.fenster._on_splat_pruefen()


@fall
def gpu_pruefen_ohne_interpreter(s):
    Stubs(s, splat_python=None,
          splat_info={"fehler": "kein Interpreter mit torch und gsplat, gesucht: python3"})
    s.fenster._on_splat_pruefen()


@fall
def splat_maeander_ohne_flug(s):
    Stubs(s)
    _flug(s, flug=False)
    s.fenster._on_splat_maeander()


@fall
def splat_maeander_ohne_interpreter(s):
    st = Stubs(s, splat_python=None,
               splat_info={"fehler": "kein Interpreter mit torch und gsplat, "
                                     "gesucht: python3"})
    _flug(s)
    _lage(s, st)
    s.fenster._on_splat_maeander()


@fall
def splat_maeander_gegenprobe(s):
    st = Stubs(s)
    _flug(s, arbeit={"optik_kalibrierung.json": {
        "rgb_faktor": 1.05, "thermal_faktor": 1.0, "thermal": None,
        "korrektur": _KORREKTUR}})
    _lage(s, st)
    f = s.fenster
    f._meander_lade_optik()
    f._chk_splat_thermal.setChecked(False)
    f._on_splat_maeander()


@fall
def splat_maeander_ohne_gegenprobe(s):
    # Temperaturen angehakt, aber die Thermalbilder tragen keine Rohwerte
    st = Stubs(s, thermal=True, rohwerte=False)
    _flug(s)
    _lage(s, st)
    f = s.fenster
    f._chk_splat_pruefen.setChecked(False)
    f._chk_splat_posen.setChecked(False)
    f._on_splat_maeander()


@fall
def splat_maeander_thermal(s):
    st = Stubs(s, thermal=True)
    _flug(s, arbeit={"thermal_lage.json": {"gier_grad": 0.3, "x": 0.05, "y": 0.0}})
    _lage(s, st)
    f = s.fenster
    f._meander_lade_thermal()
    f._on_splat_maeander()


@fall
def splat_maeander_ohne_pipeline(s):
    Stubs(s)                              # keine Thermalkameras: Thermal übersprungen
    _flug(s, arbeit={"optik_kalibrierung.json": {
        "rgb_faktor": 1.05, "thermal_faktor": 1.0, "thermal": None}})
    f = s.fenster
    f._chk_splat_pruefen.setChecked(False)
    f._on_splat_maeander()


@fall
def splat_maeander_training_scheitert(s):
    st = Stubs(s, scheitert={"trainieren": "Splat-Training fehlgeschlagen:\n"
                                           "CUDA out of memory"})
    _flug(s)
    _lage(s, st)
    s.fenster._chk_splat_thermal.setChecked(False)
    s.fenster._on_splat_maeander()


@fall
def splat_onboard_ohne_bag(s):
    Stubs(s)
    _flug(s, bag=False)
    s.fenster._on_splat_onboard()


@fall
def splat_onboard_mit_ebene(s):
    Stubs(s)
    _flug(s, ebenen=("onboard",))
    s.fenster._on_splat_onboard()


@fall
def splat_onboard_ohne_ebene(s):
    Stubs(s)
    _flug(s)
    s.fenster._on_splat_onboard()


@fall
def splat_onboard_ebene_fremde_extrinsik(s):
    Stubs(s)
    fremd = np.eye(4)
    fremd[0, 3] = 0.5
    _flug(s, ebenen=(("onboard", {"meta": {"extrinsic": fremd.tolist()}}),))
    s.fenster._on_splat_onboard()


@fall
def splat_onboard_ohne_gegenprobe(s):
    Stubs(s)
    _flug(s, ebenen=("onboard",))
    f = s.fenster
    f._chk_splat_pruefen.setChecked(False)
    f._spin_splat_frames.setValue(120)
    f._on_splat_onboard()


@fall
def splat_gemeinsam_ohne_flug(s):
    Stubs(s)
    _flug(s, flug=False)
    s.fenster._on_splat_gemeinsam()


@fall
def splat_gemeinsam_mit_fusion(s):
    st = Stubs(s)
    _flug(s, ebenen=("onboard", "meander_rgb", ("fusion", {"meta": {
        "M": [[1.1, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.9]],
        "t": [0.01, 0.0, -0.01], "abstand_vorher": 40.0, "abstand_nachher": 12.0}})))
    _lage(s, st)
    s.fenster._on_splat_gemeinsam()


@fall
def splat_gemeinsam_ohne_fusion(s):
    st = Stubs(s)
    _flug(s)
    _lage(s, st)
    s.fenster._chk_splat_pruefen.setChecked(False)
    s.fenster._on_splat_gemeinsam()


@fall
def splat_gemeinsam_fusion_unplausibel(s):
    Stubs(s)                              # ohne Pipeline: Lage aus dem Arbeitsordner
    _flug(s, ebenen=(("fusion", {"meta": {
        "M": [[0.2, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.9]],
        "t": [0.0, 0.0, 0.0], "abstand_vorher": 111.0, "abstand_nachher": 66.0}}),))
    s.fenster._chk_splat_pruefen.setChecked(False)
    s.fenster._on_splat_gemeinsam()


# ----------------------------------------------------------------- Fusion

@fall
def fusion_ohne_quellen(s):
    Stubs(s)
    _flug(s, ebenen=("onboard",))
    s.fenster._on_fusion()


@fall
def fusion(s):
    Stubs(s)
    _flug(s, ebenen=("onboard", "meander_rgb"))
    s.fenster._on_fusion()


@fall
def fusion_splat_vor_direkt_unplausibel(s):
    Stubs(s, fusion_unplausibel="die Angleichung senkt den Farbabstand kaum (111 → 66)")
    _flug(s, ebenen=("onboard", "onboard_splat", "meander_rgb", "meander_splat"))
    s.fenster._on_fusion()


# ------------------------------------------------------------------- Mesh

@fall
def mesh_laden(s):
    Stubs(s, geometrien=[True])
    _flug(s, ebenen=("onboard",))
    _mesh_an(s)


@fall
def mesh_bauen(s):
    Stubs(s, hybrid=False)
    _flug(s)
    s.fenster._chk_mesh_hybrid.setChecked(False)
    _mesh_an(s)


@fall
def mesh_wartet_bei_laufendem_schritt(s):
    Stubs(s)
    _flug(s)
    f = s.fenster
    f._on_splat_pruefen()                 # ein anderer Schritt läuft
    _mesh_an(s)
    s.merke_labels("Mesh wartet")
    s.ereignis("mesh", s.kurz([f._cloud_view.mesh_an(), f._mesh_wartet]))


@fall
def mesh_abgebrochen(s):
    Stubs(s)
    _flug(s)
    s.abbruch_nach(1)
    _mesh_an(s)


@fall
def mesh_fehlschlag(s):
    Stubs(s, scheitert={"build_geometry": "Zu wenige Flächenzellen für ein Mesh."})
    _flug(s)
    _mesh_an(s)


@fall
def mesh_parameter_geaendert(s):
    Stubs(s, geometrien=[True])
    _flug(s)
    _mesh_an(s)
    s.laufe()
    s.warte()
    s.fenster._spin_mesh_voxel.setValue(6)        # neues Ziel: wird neu gebaut
    s.warte(1800)


@fall
def mesh_parameter_geaendert_ohne_schalter(s):
    Stubs(s)
    _flug(s)
    s.fenster._spin_mesh_depth.setValue(11)
    s.warte(1800)


@fall
def mesh_cloudcompare_im_speicher(s):
    Stubs(s, geometrien=[True])
    _flug(s, ebenen=("onboard",))
    _mesh_an(s)
    s.laufe()
    s.warte()
    s.fenster._on_mesh_cloudcompare()


@fall
def mesh_cloudcompare_ohne_geometrie(s):
    Stubs(s)
    _flug(s)
    s.fenster._on_mesh_cloudcompare()


@fall
def mesh_cloudcompare_gespeicherte_geometrie(s):
    Stubs(s, geometrien=[True])
    _flug(s, ebenen=("meander_rgb",))
    s.fenster._on_mesh_cloudcompare()


# ------------------------------------------------------------- Farbebenen

@fall
def ebenen_keine(s):
    Stubs(s)
    _flug(s)


@fall
def ebenen_ohne_projekt(s):
    Stubs(s)
    s.fenster._reload_layers()


@fall
def ebenen_mit_temperaturen(s):
    Stubs(s)
    _flug(s, ebenen=("onboard", "meander_rgb",
                     ("meander_thermal", {"temperatur": True, "anteil": 0.5}),
                     ("meander_thermal_splat", {"temperatur": True, "anteil": 0.7})))
    f = s.fenster
    f._combo_layer.setCurrentIndex(f._combo_layer.findData("meander_thermal_splat"))


@fall
def ebenen_unpassend(s):
    # falsche Punktzahl, fremder Fingerprint, Thermal ohne Temperaturen
    Stubs(s)
    _flug(s, ebenen=(("onboard", {"meta": {"rec_fingerprint": "9_9_0.000000_1.000000"}}),
                     ("meander_rgb", {"punkte": 150}),
                     ("onboard_splat", {"punkte": 150}),
                     "meander_thermal", "fusion"))


@fall
def ebenen_onboard_ohne_fingerprint(s):
    from core.project import Project

    Stubs(s)
    projekt = _flug(s)
    _ebene(projekt, "onboard", int(s.fenster._rec.n_points))
    ordner = os.path.join(projekt.dir, Project.LAYERS["meander_splat"])
    os.makedirs(ordner)
    for name in ("colors.bin", "valid.bin"):          # unvollständig: keine meta.json
        open(os.path.join(ordner, name), "wb").close()
    s.dateien_merken()
    s.fenster._reload_layers()


@fall
def ebene_gespeicherte_fehlt(s):
    # die Einstellungen nennen eine Ebene, die es nicht gibt: erste vorhandene
    Stubs(s)
    f = s.fenster
    f._layer_key = "fusion"
    _flug(s, ebenen=("meander_rgb", "meander_splat"))


@fall
def ebene_fast_leer(s):
    Stubs(s)
    f = s.fenster
    f._chk_only_colored.setChecked(True)
    _flug(s, ebenen=(("meander_rgb", {"anteil": 0.005}), "meander_splat"))


@fall
def farbe_ueber_leiste(s):
    Stubs(s)
    _flug(s, ebenen=("onboard", "meander_rgb", ("meander_thermal", {"temperatur": True})))
    wahl = s.fenster._cloud_view.farbmodus_gewaehlt
    for key in ("rgb:meander_rgb", "hoehe", "rgb:meander_thermal", "intensitaet",
                "rgb:onboard", "rgb:fusion", "uniform"):
        wahl.emit(key)
        s.ereignis("gewählt", key, s.kurz([s.fenster._layer_key,
                                           s.fenster._settings.get("color_mode"),
                                           s.fenster._settings.get("layer")]))


@fall
def farbe_ueber_menue(s):
    Stubs(s)
    _flug(s, ebenen=("onboard", "meander_rgb", "fusion"))
    f = s.fenster
    for key in ("meander_rgb", "fusion", "onboard"):
        eintrag = next(a for a in f._actions["menu_farbquelle"].actions()
                       if a.data() == key)
        eintrag.trigger()
        s.ereignis("gewählt", key, s.kurz([f._layer_key, f._settings.get("layer")]))


@fall
def farbe_ueber_seitenleiste(s):
    Stubs(s)
    _flug(s, ebenen=("onboard", "meander_rgb"))
    f = s.fenster
    f._combo_layer.setCurrentIndex(f._combo_layer.findData("meander_rgb"))
    f._combo_colormode.setCurrentIndex(f._combo_colormode.findData("hoehe"))
    f._combo_colormode.setCurrentIndex(f._combo_colormode.findData("rgb"))
