#!/usr/bin/env python3
"""Numerik-Proben der Familie ``maeander``: meander, optik, sichtbar, temperatur.

Vertrag wie in ``numerik_probe.py``: ``PROBEN = {name: funktion}``, jede
Funktion liefert ``{teilname: ndarray | bytes | zahl}``. Alle Eingaben sind
synthetisch und fest geseedet; Bilder und Dateien entstehen in einem eigenen
Arbeitsordner unter ``/tmp/super360_modtests/umbau``, nie im echten Cache.

Die Kameras und die Lage kommen aus Stellvertretern für die Pipeline
(``pipe.cams``, ``pipe.affine()``, ``pipe.rgb_cams()``, ``pipe.thermal_cams()``);
COLMAP und die echte Pipeline laufen nicht. Aus ``colorize_pipeline`` werden
nur ``register.affine`` und die Projektion gebraucht, so wie ``core`` sie ruft.

Festgehalten wird auch, was auffällt, aber so bleiben soll, bis es jemand
bewusst ändert: ``meander.colorize_points`` rechnet u und v nicht auf die
Bildgröße um, wenn das Bild kleiner ist als die Kamera (die Probe
``meander.colorize_points`` trägt dafür die Teile ``einzeln_halb.*``);
``sichtbar.colorize_sichtbar`` und ``temperatur.abtasten`` rechnen um.

Direkt aufgerufen fährt die Datei jede Probe zweimal und nennt die, die nicht
bitgleich wiederkommen (Kandidaten für ``UNSTET``)::

    python3 scripts/umbau/proben_maeander.py [--wurzel <dir>]
"""
from __future__ import annotations

import importlib
import json
import os
import shutil
import struct
import sys

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import basis  # noqa: E402

_arbeit: str | None = None


# ------------------------------------------------------------------ Helfer

def _core(name: str):
    """``core.<name>`` aus der gesetzten Wurzel (erst beim Gebrauch laden)."""
    basis.wurzel()
    return importlib.import_module(f"core.{name}")


def _pipeline(name: str):
    """``colorize_pipeline.<name>``, gefunden wie in ``core.meander``."""
    _core("meander").find_pipeline()
    return importlib.import_module(f"colorize_pipeline.{name}")


def _ordner(name: str) -> str:
    """Frischer Unterordner je Probe im Arbeitsordner dieses Prozesses."""
    global _arbeit
    if _arbeit is None:
        _arbeit = basis.arbeitsordner("proben_maeander")
    pfad = os.path.join(_arbeit, name)
    if os.path.isdir(pfad):
        shutil.rmtree(pfad)
    os.makedirs(pfad)
    return pfad


def _bytes(pfad: str) -> bytes:
    with open(pfad, "rb") as fh:
        return fh.read()


def _teile(kopf: str, wert, out: dict | None = None) -> dict:
    """Beliebige Rückgabe flach auf ``{teilname: ndarray | bytes | zahl}`` legen."""
    out = {} if out is None else out
    if isinstance(wert, dict):
        for k in sorted(wert):
            _teile(f"{kopf}.{k}", wert[k], out)
    elif wert is None:
        out[kopf] = b"None"
    elif isinstance(wert, str):
        out[kopf] = wert.encode("utf-8")
    elif isinstance(wert, (bytes, bool, int, float, np.generic)):
        out[kopf] = wert
    else:
        arr = np.asarray(wert)
        if arr.dtype.kind in "US":
            out[kopf] = "\n".join(str(x) for x in arr.ravel()).encode("utf-8")
        elif arr.dtype.hasobject:
            for i, teil in enumerate(wert):
                _teile(f"{kopf}.{i}", teil, out)
        else:
            out[kopf] = arr
    return out


class _Fortschritt:
    """Zeichnet ``progress(f, m)`` auf."""

    def __init__(self):
        self.werte: list = []
        self.texte: list = []

    def __call__(self, f, m):
        self.werte.append(float(f))
        self.texte.append(str(m))

    def teile(self, kopf: str) -> dict:
        return {f"{kopf}.fortschritt": np.asarray(self.werte, np.float64),
                f"{kopf}.meldungen": "\n".join(self.texte).encode("utf-8")}


def _rot(ax: float, ay: float, az: float) -> np.ndarray:
    cx, sx, cy, sy, cz, sz = (np.cos(ax), np.sin(ax), np.cos(ay), np.sin(ay),
                              np.cos(az), np.sin(az))
    rx = np.array([[1.0, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1.0, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1.0]])
    return rz @ ry @ rx


_NADIR = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])   # blickt nach unten


def _kameras(zentren: np.ndarray, groesse, params, modell: str, rng, kipp: float,
             vorsilbe: str) -> dict:
    """Nadirkameras über ``zentren``, leicht gekippt; Rahmen wie die Zentren."""
    n = len(zentren)
    Rcw = np.stack([_NADIR @ _rot(*(rng.normal(0.0, kipp, 3))) for _ in range(n)])
    tcw = -np.einsum("nij,nj->ni", Rcw, zentren)
    return {"names": np.array([f"{vorsilbe}{i}.png" for i in range(n)]),
            "Rcw": Rcw, "tcw": tcw, "C": np.array(zentren, float),
            "size": np.tile(np.asarray(groesse, float), (n, 1)),
            "params": np.tile(np.asarray(params, float), (n, 1)),
            "model": np.array(modell)}


def _in_colmap(cams: dict, A: np.ndarray, b: np.ndarray) -> dict:
    """Kameras aus dem Kartenrahmen in den Rahmen vor dem Affin (A, b) legen."""
    s = float(np.cbrt(abs(np.linalg.det(A))))
    Ainv = np.linalg.inv(A)
    C = (cams["C"] - b) @ Ainv.T
    Rcw = cams["Rcw"] @ (A / s)
    out = dict(cams, Rcw=Rcw, C=C, tcw=-np.einsum("nij,nj->ni", Rcw, C))
    if "xyz" in cams:
        out["xyz"] = (cams["xyz"] - b) @ Ainv.T
    return out


def _bilder(ordner: str, namen, groessen, seed: int) -> list:
    """Rauschbilder als PNG; ``groessen`` je Bild (breite, hoehe)."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    out = []
    for name, (w, h) in zip(namen, groessen):
        bild = rng.integers(0, 256, size=(int(h), int(w), 3), dtype=np.uint8)
        Image.fromarray(bild).save(os.path.join(ordner, str(name)))
        out.append(bild)
    return out


def _temperaturquelle(n: int, form, seed: int, ohne=()):
    """``(index, name) -> Bild in °C``; für die Indizes in ``ohne`` None."""
    rng = np.random.default_rng(seed)
    bilder = [rng.uniform(10.0, 45.0, size=form).astype(np.float32) for _ in range(n)]
    for bild in bilder:
        bild[rng.integers(0, form[0], 5), rng.integers(0, form[1], 5)] = np.nan

    def hole(i, _name):
        return None if i in ohne else bilder[i]

    return hole


class _Pipe:
    """Stellvertreter für die Pipeline: Kameras, Lage und Affin, sonst nichts."""

    def __init__(self, cams=None, thermal=None):
        self._reg = _pipeline("register")
        self.yaw = 0.2
        self.t = np.array([3.0, -2.0, 0.5])
        self.enu = (2.0, _rot(0.0, 0.0, 0.4), np.array([1.0, 2.0, 3.0]))
        self.skala = 1.0
        self.zentrum = None
        self.cams = cams
        self.rgb_versatz = (0.0, 0.0)
        self._thermal = thermal

    def affine(self):
        return self._reg.affine(self.yaw, self.t, *self.enu, skala=self.skala,
                                zentrum=self.zentrum)

    def rgb_cams(self):
        du, dv = self.rgb_versatz
        if du == 0.0 and dv == 0.0:
            return self.cams
        params = np.array(self.cams["params"], float)
        params[:, 1] += du
        params[:, 2] += dv
        return dict(self.cams, params=params)

    def thermal_cams(self):
        return self._thermal


# ------------------------------------------------------------ Szenen: Farbe

_FARB_A = 1.02 * _rot(0.0, 0.0, 0.3)
_FARB_B = np.array([2.0, -1.0, 0.5])
_FARB_GROESSE = (96, 64)


def _farbszene(n_cams: int = 5, modell: str = "SIMPLE_RADIAL"):
    """Kameras 20 m über einem flachen Streifen; Punkte im Kartenrahmen."""
    rng = np.random.default_rng(101)
    zentren = np.c_[np.linspace(-10.0, 10.0, n_cams) if n_cams > 1 else [0.0],
                    rng.uniform(-1.0, 1.0, n_cams), np.full(n_cams, 20.0)]
    params = {"SIMPLE_RADIAL": [80.0, 48.0, 32.0, -0.05],
              "PINHOLE": [80.0, 82.0, 48.0, 32.0]}[modell]
    cams = _kameras(zentren, _FARB_GROESSE, params, modell, rng, 0.04, "b")
    P_col = np.c_[rng.uniform(-24.0, 24.0, 6000), rng.uniform(-12.0, 12.0, 6000),
                  rng.uniform(0.0, 3.0, 6000)]
    return cams, P_col @ _FARB_A.T + _FARB_B


def _farbbilder(name: str, cams: dict, halb=()) -> tuple:
    ordner = _ordner(name)
    w, h = _FARB_GROESSE
    groessen = [(w // 2, h // 2) if i in halb else (w, h)
                for i in range(len(cams["names"]))]
    return ordner, _bilder(ordner, cams["names"], groessen, 202)


# ---------------------------------------------------------------- meander

def _meander_thin():
    m = _core("meander")
    rng = np.random.default_rng(11)
    P = rng.uniform(-50.0, 50.0, size=(250_000, 3))
    P32 = rng.uniform(-50.0, 50.0, size=(400_001, 3)).astype(np.float32)
    return {"gross": m.thin(P, 100_000),
            "genau_grenze": m.thin(P[:100_000], 100_000),
            "eins_darueber": m.thin(P[:100_001], 100_000),
            "klein_float32": m.thin(P32[:50], 100_000),
            "vorgabe_float32": m.thin(P32),
            "liste": m.thin([[1, 2, 3], [4, 5, 6]], 10)}


def _meander_as_t3():
    m = _core("meander")
    return {"drei": m.as_t3([1.0, 2.0, 3.0]),
            "zwei": m.as_t3([1.5, -2.5]),
            "vier_2d": m.as_t3(np.array([[4.0, 5.0, 6.0, 7.0]])),
            "leer": m.as_t3([]),
            "zahl": m.as_t3(0.25),
            "ganzzahlen": m.as_t3(np.array([7, -8], np.int32)),
            "float32": m.as_t3(np.array([0.1, 0.2, 0.3], np.float32))}


def _korrektur():
    kipp = _rot(0.01, -0.007, 0.0)
    return {"M": kipp.tolist(), "v": [0.4, 0.35, -0.12]}


def _meander_korrektur_anwenden():
    m = _core("meander")
    rng = np.random.default_rng(12)
    A = rng.normal(size=(3, 3))
    b = rng.normal(size=3)
    out = {}
    for name, korr in (("ohne", None), ("leer", {}), ("mit", _korrektur())):
        A2, b2 = m.korrektur_anwenden(A, b, korr)
        out[f"{name}.A"], out[f"{name}.b"] = A2, b2
    A3, b3 = m.korrektur_anwenden(A.astype(np.float32).tolist(), b.tolist(), _korrektur())
    out["listen.A"], out["listen.b"] = A3, b3
    return out


def _meander_thermal_lage():
    m = _core("meander")
    out = {}
    for name, (yaw, t, zus) in {
            "voll": (80.0, [-18.0, 9.0, -2.0], (1.5, 0.5, -0.25)),
            "zweier": (80.0, [-18.0, 9.0], (0.0, 0.0, 0.0)),
            "krumm": (-123.456789, np.array([0.1, 0.2, 0.3]), (0.1, -0.7, 1e-3)),
            "ganzzahlen": (7, (1, 2, 3), (1, 2, 3))}.items():
        yaw_t, t_t = m.thermal_lage(yaw, t, zus)
        out[f"{name}.gier"], out[f"{name}.t"] = yaw_t, t_t
    return out


def _meander_lage_affine():
    m = _core("meander")
    out = {}
    pipe = _Pipe()
    for name in ("einfach", "skala", "korrektur"):
        if name == "skala":
            pipe.skala, pipe.zentrum = 1.03, np.array([4.0, -6.0, 1.5])
        if name == "korrektur":
            pipe.s360_korrektur = _korrektur()
        for lage, (yaw, t) in {"a": (90.0, [5.0, 6.0]),
                               "b": (-33.3, [0.25, -1.75, 2.5])}.items():
            A, b = m.lage_affine(pipe, yaw, t)
            out[f"{name}.{lage}.A"], out[f"{name}.{lage}.b"] = A, b
    # die Lage der Pipeline bleibt, was sie war
    out["danach.yaw"], out["danach.t"] = pipe.yaw, pipe.t
    return out


def _meander_thermal_zuschlag():
    m = _core("meander")
    ordner = _ordner("thermal_zuschlag")
    ziel = os.path.join(ordner, "flug")          # save legt den Ordner selbst an
    out = {"ohne_datei": np.asarray(m.load_thermal_zuschlag(ziel))}
    m.save_thermal_zuschlag(ziel, (1.5, 0.5, -0.25))
    out["datei"] = _bytes(os.path.join(ziel, m._THERMAL_LAGE))
    out["gelesen"] = np.asarray(m.load_thermal_zuschlag(ziel))
    m.save_thermal_zuschlag(ziel, (np.float32(0.1), 2, -1e-7))
    out["datei_krumm"] = _bytes(os.path.join(ziel, m._THERMAL_LAGE))
    out["gelesen_krumm"] = np.asarray(m.load_thermal_zuschlag(ziel))
    out["dateien"] = "\n".join(sorted(os.listdir(ziel))).encode("utf-8")
    with open(os.path.join(ziel, m._THERMAL_LAGE), "w", encoding="utf-8") as fh:
        fh.write("{kaputt")
    out["kaputt"] = np.asarray(m.load_thermal_zuschlag(ziel))
    with open(os.path.join(ziel, m._THERMAL_LAGE), "w", encoding="utf-8") as fh:
        json.dump({"gier_grad": 1.0, "x": 2.0}, fh)
    out["schluessel_fehlt"] = np.asarray(m.load_thermal_zuschlag(ziel))
    return out


def _meander_colorize_points():
    m = _core("meander")
    cz = _pipeline("colorize")
    cams, P = _farbszene()
    out = {}

    # alle Bilder in Kameragröße
    ordner, _ = _farbbilder("cp_voll", cams)
    fort = _Fortschritt()
    rgb, maske = m.colorize_points(P, cams, ordner, _FARB_A, _FARB_B, progress=fort)
    out.update({"voll.rgb": rgb, "voll.maske": maske}, **fort.teile("voll"))

    # ein Bild in halber Kameragröße: u und v werden nicht umgerechnet
    ordner_h, _ = _farbbilder("cp_halb", cams, halb=(2,))
    rgb, maske = m.colorize_points(P, cams, ordner_h, _FARB_A, _FARB_B)
    out.update({"halb.rgb": rgb, "halb.maske": maske})

    # mit Temperaturquelle (eine Kamera ohne Temperaturbild), Bilder gemischt
    quelle = _temperaturquelle(len(cams["names"]), (16, 24), 303, ohne=(1,))
    rgb, maske, temp = m.colorize_points(P, cams, ordner_h, _FARB_A, _FARB_B,
                                         temperatur=quelle)
    out.update({"temperatur.rgb": rgb, "temperatur.maske": maske,
                "temperatur.temp": temp})

    # ein anderes Kameramodell
    cams_p, P_p = _farbszene(modell="PINHOLE")
    rgb, maske = m.colorize_points(P_p, cams_p, ordner, _FARB_A, _FARB_B)
    out.update({"pinhole.rgb": rgb, "pinhole.maske": maske})

    # viele Kameras: welche Bilder den Fortschritt melden und mit welchem Text
    cams_v, P_v = _farbszene(n_cams=25)
    ordner_v, _ = _farbbilder("cp_viele", cams_v)
    fort = _Fortschritt()
    rgb, maske = m.colorize_points(P_v, cams_v, ordner_v, _FARB_A, _FARB_B, progress=fort)
    out.update({"viele.rgb": rgb, "viele.maske": maske}, **fort.teile("viele"))

    # eine Kamera, Bild halb so groß: gegen die Rechnung ohne und mit Umrechnung
    cams1, P1 = _farbszene(n_cams=1)
    ordner1, bilder1 = _farbbilder("cp_einzeln", cams1, halb=(0,))
    rgb, maske = m.colorize_points(P1, cams1, ordner1, _FARB_A, _FARB_B)
    bild = bilder1[0]
    ih, iw = bild.shape[:2]
    W, H = cams1["size"][0]
    pc = (P1 - _FARB_B) @ np.linalg.inv(_FARB_A).T @ cams1["Rcw"][0].T + cams1["tcw"][0]
    u, v, vorn, _ = cz._project(pc, cams1["size"][0], cams1["params"][0], "SIMPLE_RADIAL")
    drin = vorn & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    soll = {}
    for name, (su, sv) in {"ohne": (1.0, 1.0), "mit": (iw / W, ih / H)}.items():
        soll[name] = np.empty((len(P1), 3), np.uint8)
        soll[name][:] = m._FALLBACK
        soll[name][drin] = bild[np.clip((v[drin] * sv).astype(np.int32), 0, ih - 1),
                                np.clip((u[drin] * su).astype(np.int32), 0, iw - 1)]
    out.update({"einzeln_halb.rgb": rgb, "einzeln_halb.maske": maske,
                "einzeln_halb.wie_ohne_umrechnung": int(np.array_equal(rgb, soll["ohne"])),
                "einzeln_halb.wie_mit_umrechnung": int(np.array_equal(rgb, soll["mit"])),
                "einzeln_halb.am_rand_geklemmt": int(np.count_nonzero(
                    drin & ((u >= iw) | (v >= ih))))})
    return out


def _meander_live_preview():
    m = _core("meander")
    cams, P = _farbszene()
    ordner, _ = _farbbilder("lp_voll", cams)
    out = {}
    for name, scale in (("sechstel", 1.0 / 6.0), ("halb", 0.5), ("voll", 1.0)):
        fort = _Fortschritt()
        lp = m.LivePreview(cams, ordner, scale=scale, progress=fort)
        rgb, maske = lp.colorize(P, _FARB_A, _FARB_B)
        out.update({f"{name}.bilder": np.stack(lp.bilder), f"{name}.rgb": rgb,
                    f"{name}.maske": maske}, **fort.teile(name))
        # andere Optik für einen Durchlauf, gleiche Bilder
        andere = dict(cams, params=cams["params"] * [1.07, 1.0, 1.0, 1.0])
        rgb, maske = lp.colorize(P, _FARB_A, _FARB_B, cams=andere)
        out.update({f"{name}.andere_optik.rgb": rgb, f"{name}.andere_optik.maske": maske})
        rgb, maske = lp.colorize(P, _FARB_A, _FARB_B + [1000.0, 0.0, 0.0])
        out.update({f"{name}.verschoben.rgb": rgb, f"{name}.verschoben.maske": maske})
    # Vorgabe des Faktors und ein Bild, das kleiner ist als die Kamera
    ordner_h, _ = _farbbilder("lp_halb", cams, halb=(2,))
    lp = m.LivePreview(cams, ordner_h)
    rgb, maske = lp.colorize(P, _FARB_A, _FARB_B)
    out.update({"vorgabe.faktor": lp.scale, "vorgabe_halb.rgb": rgb,
                "vorgabe_halb.maske": maske,
                "vorgabe_halb.bildformen": np.array([b.shape for b in lp.bilder])})
    return out


def _meander_ebene():
    m = _core("meander")
    ordner = _ordner("ebene")
    lay = os.path.join(ordner, "farbe", "maeander")
    rng = np.random.default_rng(13)
    n = 500
    rgb = rng.integers(0, 256, size=(n, 3), dtype=np.uint8)
    maske = rng.random(n) > 0.3
    temp = rng.uniform(5.0, 60.0, n).astype(np.float32)
    temp[~maske] = np.nan
    meta = {"quelle": "Mäander", "punkte": n, "anteil": float(maske.mean()),
            "lage": {"yaw_deg": 12.5, "t": [1.0, -2.0, 0.25]}, "thermal": False}

    def datei(name):
        pfad = os.path.join(lay, name)
        return _bytes(pfad) if os.path.isfile(pfad) else b"fehlt"

    out = {"vorher.layer": int(m.load_layer(lay, n) is None),
           "vorher.temperatur": int(m.load_temperatur(lay, n) is None)}
    m.save_layer(lay, rgb, maske, meta)
    out.update({"ohne.colors": datei("colors.bin"), "ohne.valid": datei("valid.bin"),
                "ohne.meta": datei("meta.json"), "ohne.temperatur": datei("temperatur.bin"),
                "ohne.load_temperatur_none": int(m.load_temperatur(lay, n) is None)})
    gel = m.load_layer(lay, n)
    out.update({"ohne.load.rgb": gel[0], "ohne.load.maske": gel[1],
                "ohne.load_falsche_zahl": int(m.load_layer(lay, n + 1) is None)})

    m.save_layer(lay, rgb, maske, dict(meta, thermal=True), temperatur=temp)
    out.update({"mit.colors": datei("colors.bin"), "mit.valid": datei("valid.bin"),
                "mit.meta": datei("meta.json"), "mit.temperatur": datei("temperatur.bin"),
                "mit.load_temperatur": m.load_temperatur(lay, n),
                "mit.load_temperatur_falsche_zahl":
                    int(m.load_temperatur(lay, n - 1) is None)})

    # andere Eingangstypen: die Dateien tragen trotzdem uint8 bzw. float32
    m.save_layer(lay, rgb.astype(np.int64), maske.astype(np.uint8), {"leer": None},
                 temperatur=temp.astype(np.float64))
    out.update({"typen.colors": datei("colors.bin"), "typen.valid": datei("valid.bin"),
                "typen.meta": datei("meta.json"), "typen.temperatur": datei("temperatur.bin")})
    gel = m.load_layer(lay, n)
    out.update({"typen.load.rgb": gel[0], "typen.load.maske": gel[1]})

    # ohne Temperatur gespeichert: die alte verschwindet
    m.save_layer(lay, rgb[::-1], ~maske, meta)
    out.update({"danach.colors": datei("colors.bin"), "danach.valid": datei("valid.bin"),
                "danach.temperatur": datei("temperatur.bin"),
                "danach.dateien": "\n".join(sorted(os.listdir(lay))).encode("utf-8")})

    # nur eine der beiden Dateien passt zur Punktzahl: beide Längen zählen
    def mit_laengen(n_colors, n_valid):
        rgb[:n_colors].tofile(os.path.join(lay, "colors.bin"))
        maske[:n_valid].astype(np.uint8).tofile(os.path.join(lay, "valid.bin"))
        return int(m.load_layer(lay, n) is None)

    out.update({"laengen.valid_zu_kurz": mit_laengen(n, n - 1),
                "laengen.valid_leer": mit_laengen(n, 0),
                "laengen.colors_zu_kurz": mit_laengen(n - 1, n),
                "laengen.beide_kurz": mit_laengen(n - 1, n - 1),
                "laengen.beide_kurz_kleinere_wolke": int(m.load_layer(lay, n - 1) is None),
                "laengen.beide_passend": mit_laengen(n, n)})
    return out


def _meander_pruefe_ausrichtung():
    m = _core("meander")
    faelle = {"gut": {"anteil_auf_flaeche": 0.73},
              "ohne": {},
              "schlecht": {"anteil_auf_flaeche": 0.025, "median_abweichung": 5.84,
                           "yaw_deg": 154.53},
              "optik_gewinnt": {"anteil_auf_flaeche": 0.008,
                                "anteil_auf_flaeche_optik": 0.62},
              "optik_schlecht": {"anteil_auf_flaeche": 0.9,
                                 "anteil_auf_flaeche_optik": 0.3999},
              "grenze": {"anteil_auf_flaeche": m.MIN_AUF_FLAECHE}}
    return _teile("text", {k: m.pruefe_ausrichtung(v) for k, v in faelle.items()})


# ------------------------------------------------------------ Szene: Optik

_FAKTOR = 1.087


def _gelaende() -> np.ndarray:
    """Welliger Boden mit einem Haus, Raster 0,25 m über 60 x 60 m."""
    rng = np.random.default_rng(21)
    g = np.mgrid[-30:30.01:0.25, -30:30.01:0.25].reshape(2, -1).T
    z = 0.5 * np.sin(g[:, 0] / 7.0) + 0.3 * np.cos(g[:, 1] / 5.0)
    haus = (np.abs(g[:, 0] - 5.0) < 6.0) & (np.abs(g[:, 1] + 3.0) < 4.0)
    z[haus] = 6.0
    return np.c_[g, z + rng.normal(0.0, 0.02, len(g))]


def _optikszene(n_foto: int = 3000):
    """Stellvertreter-Pipeline: 12 Kameras 57 m über dem Gelände, Fotopunkte
    um den Faktor 1,087 zu nah an den Kameras. Gibt (pipe, punkte, kameras im
    Kartenrahmen)."""
    rng = np.random.default_rng(22)
    punkte = _gelaende()
    n = 12
    zentren = np.c_[np.linspace(-20.0, 20.0, n), np.where(np.arange(n) % 2, 8.0, -8.0),
                    57.0 + rng.normal(0.0, 0.3, n)]
    welt = _kameras(zentren, (4000, 3000), [3000.0, 2000.0, 1500.0, -0.02],
                    "SIMPLE_RADIAL", rng, 0.03, "DJI_")
    welt["names"] = np.array([f"DJI_{i:04d}_W.JPG" for i in range(n)])
    wahl = rng.choice(len(punkte), n_foto, replace=False)
    cz = float(np.mean(zentren[:, 2]))
    xyz = punkte[wahl].copy()
    xyz[:, 2] = cz - (cz - xyz[:, 2]) / _FAKTOR + rng.normal(0.0, 0.6, n_foto)
    welt["xyz"] = xyz
    pipe = _Pipe()
    A, b = pipe.affine()
    pipe.cams = _in_colmap(welt, A, b)
    pipe.points = punkte
    return pipe, punkte, welt


def _mit_loechern(punkte: np.ndarray, welt: dict) -> tuple:
    """Karte ohne zwei Blöcke mitten im Raster: dort trägt das Höhenraster
    keine Höhe. Dazu die Zahl der Fotopunkte, die über den Lücken liegen."""
    def in_luecke(p):
        return (((np.abs(p[:, 0] + 14.0) < 6.0) & (np.abs(p[:, 1] - 12.0) < 5.0))
                | ((np.abs(p[:, 0] - 16.0) < 4.0) & (np.abs(p[:, 1] + 18.0) < 3.0)))

    return punkte[~in_luecke(punkte)], int(np.count_nonzero(in_luecke(welt["xyz"])))


def _meander_guete_mit_optik():
    m = _core("meander")
    pipe, _, _ = _optikszene()
    out = _teile("voll", m._guete_mit_optik(pipe))
    wenig, _, _ = _optikszene(n_foto=120)
    _teile("zu_wenige", m._guete_mit_optik(wenig) or None, out)
    ohne = _Pipe(cams={"names": np.array([])})
    ohne.points = pipe.points
    _teile("ohne_fotopunkte", m._guete_mit_optik(ohne) or None, out)
    return out


# ------------------------------------------------------------------ optik

_KAL = {"f": 1340.0, "cx": 624.0, "cy": 509.0, "k1": -0.01, "k2": -0.39,
        "drehung_grad": [2.0, -0.7, 0.6], "breite": 1280, "hoehe": 1024,
        "rgb_faktor": 1.0}


def _optik_thermal_umrechnen():
    o = _core("optik")
    out = {}
    for name, s in (("eins", 1.0), ("laenger", 1.1), ("kuerzer", 0.93), ("ganz", 2)):
        _teile(name, o.thermal_umrechnen(_KAL, s), out)
    return out


def _optik_drehung():
    o = _core("optik")
    return {"null": o.drehung([0.0, 0.0, 0.0]),
            "schielwinkel": o.drehung([2.0, -0.7, 0.6]),
            "gross": o.drehung(np.array([120.0, -45.0, 270.0])),
            "ganzzahlen": o.drehung((90, 0, 0)),
            "winzig": o.drehung([1e-9, -1e-12, 0.0])}


def _optik_foto_hoehe():
    o = _core("optik")
    rng = np.random.default_rng(23)
    C = np.c_[rng.uniform(-20, 20, (9, 2)), 57.0 + rng.normal(0, 0.3, 9)]
    P = np.c_[rng.uniform(-30, 30, (400, 2)), rng.uniform(0.0, 9.0, 400)]
    leer = np.zeros((0, 3))
    return {"faktor_eins": o.foto_hoehe(P, C, 1.0),
            "laenger": o.foto_hoehe(P, C, _FAKTOR),
            "kuerzer": o.foto_hoehe(P, C, 0.9),
            "float32": o.foto_hoehe(P.astype(np.float32), C, 57.0 / 53.1),
            "ohne_punkte": o.foto_hoehe(leer, C, _FAKTOR),
            "ohne_kameras": o.foto_hoehe(P, leer, _FAKTOR),
            # die Eingabe bleibt unberührt
            "eingabe_danach": P}


def _optik_dsm():
    o = _core("optik")
    rng = np.random.default_rng(24)
    P = np.c_[rng.uniform(-12.3, 17.9, 4000), rng.uniform(3.1, 28.6, 4000),
              rng.normal(2.0, 1.5, 4000)]
    out = {}
    for name, args in (("vorgabe", ()), ("grob", (2.0,)), ("fein", (0.2,))):
        dsm, x0, y0, res = o._dsm(P, *args)
        out.update({f"{name}.dsm": dsm, f"{name}.x0": x0, f"{name}.y0": y0,
                    f"{name}.res": res})
    dsm, x0, y0, res = o._dsm(_gelaende())
    out.update({"gelaende.dsm": dsm, "gelaende.x0": x0, "gelaende.y0": y0})
    return out


def _optik_mi():
    o = _core("optik")
    rng = np.random.default_rng(25)
    a = rng.integers(0, 256, 5000).astype(float)
    rausch = np.clip(a + rng.normal(0.0, 20.0, 5000), 0, 255)
    return {"gleich": o._mi(a, a),
            "invertiert": o._mi(a, 255 - a),
            "zufall": o._mi(a, rng.permutation(a)),
            "verrauscht": o._mi(a, rausch),
            "float32": o._mi(a.astype(np.float32), rausch.astype(np.float32)),
            "bins_16": o._mi(a, rausch, bins=16),
            "ausserhalb": o._mi(a + 300.0, a),
            "leer": o._mi(np.zeros(0), np.zeros(0))}


def _optik_hoehe_aus_lrf():
    o = _core("optik")
    pipe, punkte, welt = _optikszene()
    A, b = pipe.affine()
    ordner = _ordner("lrf")
    rng = np.random.default_rng(26)
    einzeln = {}
    for i, name in enumerate(pipe.cams["names"]):
        if i == 7:
            continue                                     # Bild fehlt
        # Abstand entlang der Blickachse bis zur Ebene z = 0, dazu ein fester
        # Fehler: die Kameras stehen in der Karte 1,2 m zu tief
        achse = welt["Rcw"][i][2]
        lrf = welt["C"][i, 2] / -achse[2] + 1.2 + rng.normal(0.0, 0.05)
        status = b"TooFar" if i == 3 else b"Normal"
        if i == 5:
            lrf += 4.0                                   # Ausreißer
        with open(os.path.join(ordner, str(name)), "wb") as fh:
            fh.write(b"\xff\xd8junk<x drone-dji:LRFStatus=\"" + status + b"\" "
                     b"drone-dji:LRFTargetDistance=\"" + f"{lrf:.3f}".encode() + b"\"/>")
        einzeln[str(name)] = o.lies_lrf(os.path.join(ordner, str(name)))
    out = _teile("lies_lrf", einzeln)
    out["lies_lrf.fehlt"] = int(o.lies_lrf(os.path.join(ordner, "fehlt.jpg")) is None)
    _teile("voll", o.hoehe_aus_lrf(pipe, punkte, A, b, ordner), out)
    # dieselbe Lage 0,8 m höher: dz folgt
    _teile("hoeher", o.hoehe_aus_lrf(pipe, punkte, A, b + [0.0, 0.0, 0.8], ordner), out)
    _teile("float32", o.hoehe_aus_lrf(pipe, punkte.astype(np.float32), A.tolist(),
                                      b.tolist(), ordner), out)
    _teile("leerer_ordner", o.hoehe_aus_lrf(pipe, punkte, A, b, _ordner("lrf_leer")), out)
    # Karte links beschnitten: der Strahl einer Kamera trifft den Boden bis zu
    # einer Zelle links vom Ursprung des Höhenrasters, der der zweiten darunter
    for name, k, achse_xy in (("links", 2, 0), ("unten", 4, 1)):
        achse = welt["Rcw"][k][2]
        treffer = welt["C"][k] + welt["C"][k, 2] / -achse[2] * achse
        rest = punkte[punkte[:, achse_xy] >= treffer[achse_xy] + 0.25]
        _teile(f"beschnitten_{name}", o.hoehe_aus_lrf(pipe, rest, A, b, ordner), out)
        out[f"beschnitten_{name}.punkte"] = len(rest)
    return out


def _optik_schaetze_rgb_faktor_tiefe():
    o = _core("optik")
    pipe, punkte, welt = _optikszene()
    A, b = pipe.affine()
    out = _teile("voll", o.schaetze_rgb_faktor_tiefe(pipe, punkte, A, b))
    # Lücken in der Karte: Fotopunkte über Zellen ohne Höhe zählen nicht mit
    rest, n_ueber = _mit_loechern(punkte, welt)
    _teile("luecken", o.schaetze_rgb_faktor_tiefe(pipe, rest, A, b), out)
    out["luecken.fotopunkte_ueber_luecke"] = n_ueber
    _teile("hoeher", o.schaetze_rgb_faktor_tiefe(pipe, punkte, A, b + [0, 0, 2.0]), out)
    _teile("versetzt", o.schaetze_rgb_faktor_tiefe(pipe, punkte, A, b + [3.0, -2.0, 0]), out)
    # Fotopunkte bis zu einer Zelle links und unterhalb des Rasterursprungs
    _teile("knapp_daneben", o.schaetze_rgb_faktor_tiefe(pipe, punkte, A, b + [-0.3, -0.3, 0]),
           out)
    _teile("knapp_links", o.schaetze_rgb_faktor_tiefe(pipe, punkte, A, b + [-0.3, 0, 0]), out)
    _teile("listen", o.schaetze_rgb_faktor_tiefe(pipe, punkte, A.tolist(), b.tolist()), out)
    wenig, _, _ = _optikszene(n_foto=150)
    _teile("zu_wenige", o.schaetze_rgb_faktor_tiefe(wenig, punkte, A, b), out)
    _teile("weit_daneben", o.schaetze_rgb_faktor_tiefe(pipe, punkte, A, b + [500.0, 0, 0]), out)
    ohne = _Pipe(cams={"names": np.array([])})
    _teile("ohne_fotopunkte", o.schaetze_rgb_faktor_tiefe(ohne, punkte, A, b), out)
    return out


def _optik_anteil_auf_flaeche():
    o = _core("optik")
    pipe, punkte, welt = _optikszene()
    A, b = pipe.affine()
    out = {}
    for name, (faktor, kw) in {"faktor_eins": (1.0, {}), "faktor_tiefe": (_FAKTOR, {}),
                               "eng": (_FAKTOR, {"toleranz": 0.1}),
                               "weit": (1.0, {"toleranz": 5.0})}.items():
        _teile(name, o.anteil_auf_flaeche(pipe, punkte, A, b, faktor, **kw), out)
    # Lücken in der Karte: Fotopunkte über Zellen ohne Höhe zählen nicht mit,
    # weder als Treffer noch als Fehlgriff
    rest, n_ueber = _mit_loechern(punkte, welt)
    for name, (faktor, kw) in {"luecken.faktor_eins": (1.0, {}),
                               "luecken.faktor_tiefe": (_FAKTOR, {}),
                               "luecken.weit": (_FAKTOR, {"toleranz": 5.0})}.items():
        _teile(name, o.anteil_auf_flaeche(pipe, rest, A, b, faktor, **kw), out)
    out["luecken.fotopunkte_ueber_luecke"] = n_ueber
    out["luecken.punkte"] = len(rest)
    _teile("versetzt", o.anteil_auf_flaeche(pipe, punkte, A, b + [0.7, 0.4, 0.3], _FAKTOR), out)
    # Fotopunkte bis zu einer Zelle links und unterhalb des Rasterursprungs
    _teile("knapp_daneben", o.anteil_auf_flaeche(pipe, punkte, A, b + [-0.3, -0.3, 0], _FAKTOR),
           out)
    _teile("knapp_links", o.anteil_auf_flaeche(pipe, punkte, A, b + [-0.3, 0, 0], 1.0), out)
    _teile("knapp_unten", o.anteil_auf_flaeche(pipe, punkte, A, b + [0, -0.3, 0], _FAKTOR,
                                               toleranz=0.1), out)
    _teile("weit_daneben", o.anteil_auf_flaeche(pipe, punkte, A, b + [500.0, 0, 0], 1.0), out)
    ohne = _Pipe(cams={"names": np.array([])})
    _teile("ohne_fotopunkte", o.anteil_auf_flaeche(ohne, punkte, A, b, 1.0), out)
    return out


def _optik_rgb_cams():
    o = _core("optik")
    pipe, _, _ = _optikszene()
    out = _teile("eins", o.rgb_cams(pipe))
    _teile("faktor", o.rgb_cams(pipe, _FAKTOR), out)
    pipe.rgb_versatz = (12.5, -7.25)
    _teile("versatz", o.rgb_cams(pipe, 1.05), out)
    # zwei Brennweiten-Spalten
    n = len(pipe.cams["names"])
    pin = _Pipe(cams=dict(pipe.cams, model="PINHOLE",
                          params=np.tile([3000.0, 3010.0, 2000.0, 1500.0], (n, 1))))
    _teile("pinhole", o.rgb_cams(pin, 1.1), out)
    _teile("cams_dict", o.cams_dict(dict(pipe.cams, params=pipe.cams["params"].astype(
        np.float32), size=pipe.cams["size"].astype(np.int64))), out)
    # die Kameras der Pipeline bleiben, wie sie waren
    out["danach.params"] = pipe.cams["params"]
    return out


def _optik_thermal_cams():
    o = _core("optik")
    pipe, _, _ = _optikszene()
    n = len(pipe.cams["names"])
    out = {"ohne_alles": int(o.thermal_cams(pipe, None) is None)}
    exif = _Pipe(cams=pipe.cams, thermal=dict(
        pipe.cams, size=np.tile([1280.0, 1024.0], (n, 1)),
        params=np.tile([1422.0, 640.0, 512.0], (n, 1)), model=np.array("SIMPLE_PINHOLE")))
    _teile("exif", o.thermal_cams(exif, None), out)
    _teile("exif_faktor", o.thermal_cams(exif, None, faktor=0.97, rgb_faktor=1.08), out)
    out["exif_danach.params"] = exif._thermal["params"]
    _teile("kal", o.thermal_cams(pipe, _KAL), out)
    _teile("kal_faktoren", o.thermal_cams(pipe, _KAL, faktor=0.98, rgb_faktor=1.086), out)
    pipe.thermal_versatz = (3.0, -2.5)
    _teile("kal_versatz", o.thermal_cams(pipe, dict(_KAL, rgb_faktor=1.05),
                                         rgb_faktor=1.086), out)
    ohne_rgb = {k: v for k, v in _KAL.items() if k != "rgb_faktor"}
    _teile("kal_ohne_rgb_faktor", o.thermal_cams(pipe, ohne_rgb, rgb_faktor=1.086), out)
    return out


def _optik_datei():
    o = _core("optik")
    ordner = _ordner("optik_datei")
    ziel = os.path.join(ordner, "flug")
    pfad = os.path.join(ziel, o.DATEI)
    out = _teile("ohne_datei", o.laden(ziel))
    out["ohne_datei.vorhanden"] = int(o.vorhanden(ziel))
    voll = {"rgb_faktor": 1.086, "thermal_faktor": 0.99, "thermal": dict(_KAL, mi_nachher=0.412),
            "dz": -1.23, "hoehe": {"dz": -1.23, "n": 41, "n_spitze": 37, "iqr": [1.1, 1.4]},
            "verfahren": "tiefe", "korrektur": _korrektur(), "notiz": "Maßstab geprüft"}
    o.speichern(ziel, voll)
    out["voll.datei"] = _bytes(pfad)
    out["voll.vorhanden"] = int(o.vorhanden(ziel))
    _teile("voll.geladen", o.laden(ziel), out)
    # Faktoren als Text und ganze Zahl: laden macht float daraus
    o.speichern(ziel, {"rgb_faktor": "1.11", "thermal_faktor": 1, "thermal": None})
    out["text.datei"] = _bytes(pfad)
    _teile("text.geladen", o.laden(ziel), out)
    o.speichern(ziel, {"rgb_faktor": 1.2})
    _teile("teil.geladen", o.laden(ziel), out)
    o.speichern(ziel, {"rgb_faktor": "viel", "thermal_faktor": 0.9})
    _teile("unlesbarer_faktor.geladen", o.laden(ziel), out)
    with open(pfad, "w", encoding="utf-8") as fh:
        fh.write("{kaputt")
    _teile("kaputt.geladen", o.laden(ziel), out)
    out["dateien"] = "\n".join(sorted(os.listdir(ziel))).encode("utf-8")
    return out


def _optik_farbkonsistenz():
    o = _core("optik")
    cams, P = _farbszene()
    ordner, _ = _farbbilder("fk", cams, halb=(2,))
    fk = o.Farbkonsistenz(cams, ordner, breite=48)
    andere = dict(cams, params=cams["params"] * [1.07, 1.0, 1.0, 1.0])
    return {"bilder": np.stack(fk.bilder), "skala": np.asarray(fk.skala),
            "streuung": fk.streuung(P, cams, _FARB_A, _FARB_B),
            "andere_optik": fk.streuung(P, andere, _FARB_A, _FARB_B),
            "verschoben": fk.streuung(P, cams, _FARB_A, _FARB_B + [1000.0, 0.0, 0.0])}


# --------------------------------------------------------------- sichtbar

def _sichtbar_zellen():
    s = _core("sichtbar")
    rng = np.random.default_rng(31)
    u = rng.uniform(-5.0, 205.0, 3000)
    v = rng.uniform(-5.0, 155.0, 3000)
    out = {}
    for name, (W, H, skala) in {"halb": (200.0, 150.0, 0.5), "voll": (200.0, 150.0, 1.0),
                                "krumm": (199.0, 151.0, 0.37), "winzig": (3.0, 2.0, 0.1)}.items():
        lin, w, h = s._zellen(u, v, W, H, skala)
        out.update({f"{name}.lin": lin, f"{name}.w": w, f"{name}.h": h})
    lin, w, h = s._zellen(u.astype(np.float32), v.astype(np.float32), 200, 150, s.TIEFE_SKALA)
    out.update({"float32.lin": lin, "float32.w": w, "float32.h": h})
    return out


def _sichtbar_tiefenkarte():
    s = _core("sichtbar")
    rng = np.random.default_rng(32)
    u = rng.uniform(0.0, 200.0, 5000)
    v = rng.uniform(0.0, 150.0, 5000)
    z = rng.uniform(5.0, 60.0, 5000)
    out = {}
    for name, skala in (("halb", s.TIEFE_SKALA), ("voll", 1.0), ("grob", 0.1)):
        karte, lin = s._tiefenkarte(u, v, z, 200.0, 150.0, skala)
        out.update({f"{name}.karte": karte, f"{name}.lin": lin})
    karte, lin = s._tiefenkarte(u[:0], v[:0], z[:0], 200.0, 150.0, 0.5)
    out.update({"leer.karte": karte, "leer.lin": lin})
    return out


_SICHT_A = 1.5 * _rot(0.0, 0.0, 0.25)
_SICHT_B = np.array([4.0, -3.0, 1.0])


def _sichtszene(n_cams: int = 3):
    """Dach über einem Boden, freier Boden daneben, eine Wand; Kameras 30 m
    darüber. Punkte und Normalen im Kartenrahmen, Kameras davor."""
    rng = np.random.default_rng(33)
    g = np.mgrid[-4:4:0.1, -4:4:0.1].reshape(2, -1).T
    dach = np.c_[g, np.full(len(g), 5.0)]
    boden = np.c_[g * 0.5, np.zeros(len(g))]
    frei = np.c_[g * 0.5 + [9.0, 0.0], np.zeros(len(g))]
    w = np.mgrid[-3:3:0.1, 0:4:0.1].reshape(2, -1).T
    wand = np.c_[np.full(len(w), -7.0), w]
    P = np.vstack([dach, boden, frei, wand]) + rng.normal(0.0, 0.004, (3 * len(g) + len(w), 3))
    n = np.tile([0.0, 0.0, 1.0], (len(P), 1)).astype(np.float32)
    n[3 * len(g):] = (1.0, 0.0, 0.0)
    zentren = np.array([[-3.0, 0.5, 30.0], [0.0, -0.5, 30.5], [6.0, 0.0, 29.5]])[:n_cams]
    welt = _kameras(zentren, (200, 200), [200.0, 100.0, 100.0, -0.02], "SIMPLE_RADIAL",
                    rng, 0.03, "s")
    return _in_colmap(welt, _SICHT_A, _SICHT_B), P, n


#: Bildgrößen je Kamera (Kamera: 200 x 200) — halb, voll, anderes Seitenverhältnis
_SICHT_GROESSEN = [(100, 100), (200, 200), (120, 80)]


def _sichtbar_colorize():
    s = _core("sichtbar")
    cams, P, n = _sichtszene()
    ordner = _ordner("sichtbar")
    _bilder(ordner, cams["names"], _SICHT_GROESSEN, 404)
    out = {}

    fort = _Fortschritt()
    zeilen: list = []
    rgb, maske = s.colorize_sichtbar(P, cams, ordner, _SICHT_A, _SICHT_B, normalen_welt=n,
                                     progress=fort, log=zeilen.append)
    out.update({"farbe.rgb": rgb, "farbe.maske": maske,
                "farbe.log": "\n".join(zeilen).encode("utf-8")}, **fort.teile("farbe"))

    # mit Temperatur: eine Kamera ohne Temperaturbild
    quelle = _temperaturquelle(len(cams["names"]), (64, 80), 505, ohne=(2,))
    rgb, maske, temp = s.colorize_sichtbar(P, cams, ordner, _SICHT_A, _SICHT_B,
                                           normalen_welt=n, temperatur=quelle)
    out.update({"temperatur.rgb": rgb, "temperatur.maske": maske, "temperatur.temp": temp})

    # nur eine Stichprobe färben: ohne und mit der ganzen Karte für die Tiefe
    probe = np.arange(0, len(P), 40)
    rgb, maske = s.colorize_sichtbar(P[probe], cams, ordner, _SICHT_A, _SICHT_B,
                                     normalen_welt=n[probe])
    out.update({"probe_ohne_tiefe.rgb": rgb, "probe_ohne_tiefe.maske": maske})
    rgb, maske, temp = s.colorize_sichtbar(P[probe], cams, ordner, _SICHT_A, _SICHT_B,
                                           normalen_welt=n[probe], tiefe_punkte=P,
                                           temperatur=quelle)
    out.update({"probe_mit_tiefe.rgb": rgb, "probe_mit_tiefe.maske": maske,
                "probe_mit_tiefe.temp": temp})

    # eine Kamera, Bild halb so groß wie die Kamera
    cams1, P1, n1 = _sichtszene(n_cams=1)
    rgb, maske = s.colorize_sichtbar(P1, cams1, ordner, _SICHT_A, _SICHT_B, normalen_welt=n1)
    out.update({"einzeln_halb.rgb": rgb, "einzeln_halb.maske": maske})

    # Blickwinkel knapp um die Schwelle: Normalen so gestellt, dass der Kosinus
    # zum Blick der einen Kamera fein gestuft von 0,10 bis 0,14 läuft
    zentrum = cams1["C"][0] @ _SICHT_A.T + _SICHT_B
    blick = zentrum - P1
    blick /= np.linalg.norm(blick, axis=1, keepdims=True)
    quer = np.cross(blick, [0.0, 1.0, 0.0])
    quer /= np.linalg.norm(quer, axis=1, keepdims=True)
    cos = np.linspace(0.10, 0.14, len(P1))[np.random.default_rng(34).permutation(len(P1))]
    streifend = (cos[:, None] * blick + np.sqrt(1.0 - cos[:, None] ** 2) * quer).astype(
        np.float32)
    rgb, maske = s.colorize_sichtbar(P1, cams1, ordner, _SICHT_A, _SICHT_B,
                                     normalen_welt=streifend)
    out.update({"schwelle.rgb": rgb, "schwelle.maske": maske})

    # verschobene Lage: nichts wird getroffen
    rgb, maske = s.colorize_sichtbar(P, cams, ordner, _SICHT_A, _SICHT_B + [500.0, 0, 0],
                                     normalen_welt=n)
    out.update({"daneben.rgb": rgb, "daneben.maske": maske})
    return out


# ------------------------------------------------------------- temperatur

def _temperatur_abtasten():
    t = _core("temperatur")
    rng = np.random.default_rng(41)
    bild = rng.uniform(-10.0, 60.0, size=(51, 64)).astype(np.float32)
    bild[rng.integers(0, 51, 20), rng.integers(0, 64, 20)] = np.nan
    u = rng.uniform(-30.0, 1310.0, 4000)
    v = rng.uniform(-30.0, 1054.0, 4000)
    return {"doppelt": t.abtasten(bild, u, v, 1280, 1024),
            "gleich_gross": t.abtasten(bild, u / 20.0, v / 20.0, 64.0, 51.0),
            "krumm": t.abtasten(bild, u, v, 1279.0, 1023.0),
            "kleiner": t.abtasten(bild, u / 60.0, v / 60.0, 20, 16),
            "float32": t.abtasten(bild, u.astype(np.float32), v.astype(np.float32),
                                  np.float64(1280.0), np.float64(1024.0)),
            "float64_bild": t.abtasten(bild.astype(np.float64), u, v, 1280, 1024),
            "leer": t.abtasten(bild, u[:0], v[:0], 1280, 1024)}


def _rjpeg(roh: np.ndarray, lut: np.ndarray, hoehe: int, breite: int) -> bytes:
    """Kopf wie bei DJI: APP3 mit Rohwerten, APP5 mit Tabelle, SOF0 mit Bildgröße."""
    def seg(marker, nutz):
        return bytes([0xFF, marker]) + struct.pack(">H", len(nutz) + 2) + nutz

    teile = [b"\xff\xd8", seg(0xE0, b"JFIF\x00")]
    rohb = roh.astype("<u2").tobytes()
    for k in range(0, len(rohb), 65530):
        teile.append(seg(0xE3, rohb[k:k + 65530]))
    teile.append(seg(0xE5, lut.astype("<i2").tobytes() + b"\x00\x00"))
    teile.append(seg(0xC0, b"\x08" + struct.pack(">HH", hoehe, breite) + b"\x03"))
    teile.append(seg(0xDA, b"\x00"))
    return b"".join(teile)


def _temperatur_lies_rjpeg():
    t = _core("temperatur")
    ordner = _ordner("rjpeg")
    rng = np.random.default_rng(42)
    roh = rng.integers(3500, 5200, size=(128, 160)).astype(np.uint16)
    lut = np.clip((np.arange(t._TABELLE) - 3000) * 0.4 - 300, -300, 1600).astype(np.int16)

    def lies(name, daten):
        pfad = os.path.join(ordner, name)
        with open(pfad, "wb") as fh:
            fh.write(daten)
        return pfad, t.lies_rjpeg(pfad)

    pfad, bild = lies("gut_T.JPG", _rjpeg(roh, lut, 1024, 1280))
    out = {"gut": bild}
    kaputt = {"zu_gross": _rjpeg(np.full((128, 160), t._TABELLE, np.uint16), lut, 1024, 1280),
              "nicht_monoton": _rjpeg(roh, lut[::-1], 1024, 1280),
              "falsches_format": _rjpeg(roh[:, :150], lut, 1024, 1280),
              "kurze_tabelle": _rjpeg(roh, lut[:100], 1024, 1280),
              "ohne_rohwerte": b"\xff\xd8\xff\xe0\x00\x07JFIF\x00\xff\xda\x00\x03\x00"}
    for name, daten in kaputt.items():
        out[f"{name}.none"] = int(lies(f"{name}.JPG", daten)[1] is None)
    out["fehlt.none"] = int(t.lies_rjpeg(os.path.join(ordner, "fehlt.JPG")) is None)

    # quelle: Namen der RGB-Partner auf die Originale
    class _P:
        thermal_paare = {"a_W.JPG": pfad, "b_W.JPG": os.path.join(ordner, "ohne_rohwerte.JPG")}

    hole = t.quelle(_P())
    erst = hole(0, "/irgendwo/images/a_W.JPG")
    out.update({"quelle.treffer": erst,
                "quelle.behalten": int(hole(5, "a_W.JPG") is erst),
                "quelle.ohne_rohwerte": int(hole(1, "b_W.JPG") is None),
                "quelle.unbekannt": int(hole(2, "c_W.JPG") is None),
                "quelle.ohne_paare": int(t.quelle(object()) is None)})
    return out


# ------------------------------------------------------------------ Vertrag

PROBEN = {
    "meander.thin": _meander_thin,
    "meander.as_t3": _meander_as_t3,
    "meander.korrektur_anwenden": _meander_korrektur_anwenden,
    "meander.thermal_lage": _meander_thermal_lage,
    "meander.lage_affine": _meander_lage_affine,
    "meander.thermal_zuschlag": _meander_thermal_zuschlag,
    "meander.colorize_points": _meander_colorize_points,
    "meander.live_preview": _meander_live_preview,
    "meander.ebene": _meander_ebene,
    "meander.pruefe_ausrichtung": _meander_pruefe_ausrichtung,
    "meander.guete_mit_optik": _meander_guete_mit_optik,
    "optik.thermal_umrechnen": _optik_thermal_umrechnen,
    "optik.drehung": _optik_drehung,
    "optik.foto_hoehe": _optik_foto_hoehe,
    "optik.dsm": _optik_dsm,
    "optik.mi": _optik_mi,
    "optik.hoehe_aus_lrf": _optik_hoehe_aus_lrf,
    "optik.schaetze_rgb_faktor_tiefe": _optik_schaetze_rgb_faktor_tiefe,
    "optik.anteil_auf_flaeche": _optik_anteil_auf_flaeche,
    "optik.rgb_cams": _optik_rgb_cams,
    "optik.thermal_cams": _optik_thermal_cams,
    "optik.datei": _optik_datei,
    "optik.farbkonsistenz": _optik_farbkonsistenz,
    "sichtbar.zellen": _sichtbar_zellen,
    "sichtbar.tiefenkarte": _sichtbar_tiefenkarte,
    "sichtbar.colorize_sichtbar": _sichtbar_colorize,
    "temperatur.abtasten": _temperatur_abtasten,
    "temperatur.lies_rjpeg": _temperatur_lies_rjpeg,
}

#: Alle Proben kommen bitgleich wieder (zweimal im selben Prozess und in zwei
#: Prozessen gefahren); open3d-Normalen und die Suchen mit scipy sind bewusst
#: nicht dabei.
UNSTET: dict = {}


def _zweimal(argv=None) -> int:
    """Jede Probe zweimal fahren und die Hashes vergleichen."""
    import argparse

    parser = basis.argumente(argparse.ArgumentParser(description=_zweimal.__doc__))
    args = parser.parse_args(argv)
    basis.wurzel_setzen(args.wurzel)
    unstet, fehler, n_teile = [], [], 0
    for name in sorted(PROBEN):
        try:
            a, b = PROBEN[name](), PROBEN[name]()
        except Exception as exc:  # noqa: BLE001 — wird als Fehler genannt
            fehler.append(name)
            print(f"FEHLER  {name}: {exc.__class__.__name__}: {exc}")
            continue
        anders = sorted(t for t in set(a) | set(b)
                        if t not in a or t not in b
                        or basis.hash_wert(a[t]) != basis.hash_wert(b[t]))
        n_teile += len(a)
        if anders:
            unstet.append(name)
            print(f"UNSTET  {name}: {', '.join(anders)}")
        else:
            print(f"gleich  {name} ({len(a)} Teile)")
    print(f"{len(PROBEN)} Proben, {n_teile} Teile; unstet: "
          f"{', '.join(unstet) or 'keine'}; Fehler: {', '.join(fehler) or 'keine'}")
    return 1 if unstet or fehler else 0


if __name__ == "__main__":
    sys.exit(_zweimal())
