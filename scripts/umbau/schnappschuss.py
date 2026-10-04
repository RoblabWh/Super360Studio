#!/usr/bin/env python3
"""GUI-Schnappschuss: was das Hauptfenster zeigt, schaltet und speichert.

Baut das echte Fenster (``basis.fenster_bauen``) und hält in neun Teilen fest,
woran sich ein Umbau der Bedienoberfläche messen lässt:

a  Widget-Baum je Abschnitt der Seitenleiste, der Leiste über der 3D-Ansicht,
   der Statusleiste und der Tabs: Typ, Text, Attributname am Fenster, Tooltip,
   Wertebereich, Schritt, Suffix, Combo-Einträge, Reihenfolge; dazu die
   gemessene Breite ``_sidebar_breite``.
b  Einstellungen: Schlüssel, Typen und Werte von ``_collect_settings()`` nach
   Start, Vorgabe, Vorgabe plus Teilmenge, den Altwert-Übernahmen (Blaulicht,
   Mesh), dem Beispiel (jeder Schlüssel abweichend) und der Folge
   Beispiel -> Vorgabe. Das Beispiel liegt als ``settings_beispiel.json`` dabei.
c  Freigabe jeder Aktion, jedes ``_btn_*`` und jedes Reglers in den Zuständen
   aus :data:`ZUSTAENDE` (Stub-Objekte an ``_bag``, ``_rec``, ``_world`` …);
   je Voraussetzung gibt es einen Zustand ``ohne_<name>``, in dem alles außer
   ihr gilt, damit jede einzeln nachweisbar ist. Dazu RViz (Player gestoppt, offen, spielend), Blaulicht-Regler, Vorschau des
   zweiten Fluges, EDL und der Thermal-Haken nach der Flugwahl.
d  Handler-Namen aus ``ui/menubar.py`` gegen die Methoden des Fensters.
e  Menübaum: Pfad, Schlüssel, Rohtext mit ``&``, Kürzel, Tooltip, StatusTip,
   schaltbar, Kürzel-Kontext, nativeMenuBar.
f  Nebenfenster und Tab-Inhalte: Ausrichtfenster (mit und ohne Thermal),
   Export- und Projektdialog, 360°-Ansicht, GPS-Reiter, Kachel Explorationsgrad.
h  alle ``.connect(``-Aufrufe in ``ui/`` (Datei, Klasse, Funktion, Signal, Slot).
i  Methoden- und Attributmenge des Fensters nach ``__init__``.

Aufrufe::

    xvfb-run -a python3 scripts/umbau/schnappschuss.py --schreibe <dir> [--ersetzen]
    xvfb-run -a python3 scripts/umbau/schnappschuss.py --vergleiche [<dir>]
            [--teile a,b,…] [--erlaube abschnitte] [--erwartet <json>] [--h-ohne-datei]
    xvfb-run -a python3 scripts/umbau/schnappschuss.py --soll <json> [--abschnitt k]
            [--nur abschnittsliste|menue|befehle|freigabe|tooltips]
    python3 scripts/umbau/schnappschuss.py --pruefe-schema <json>

Überall ``--wurzel <dir>`` für einen anderen Stand (s. ``basis``). ``<dir>`` ist
ohne Angabe ``scripts/umbau/vorher/gui``. ``--schreibe`` verweigert ein
vorhandenes Ziel ohne ``--ersetzen``. ``--vergleiche`` gibt je Teil IDENTISCH
oder die Abweichungen als ``teil:pfad: vorher -> jetzt`` aus. Erfasst werden
immer alle Teile in derselben Folge, ``--teile`` wählt nur aus, was verglichen
wird. ``--soll`` prüft ohne ``--nur`` alles (auch ``abschnitte``, ``leiste``
und ``attribute``, die sich ebenfalls unter ``--nur`` nennen lassen), mit
``--abschnitt`` nur die Zeilen dieses Abschnitts und die Breite; der
Vorher-Stand für die Tooltips kommt aus ``--vorher <dir>``.

``--erlaube abschnitte``: der Zuschnitt der Seitenleiste darf sich ändern. In
Teil a zählt dann nur die Menge der Widgets (ohne Abschnitt, Tiefe und
Reihenfolge) und die Breite darf wachsen; in Teil b bleibt der Wert von
``sections`` außer Acht. ``--h-ohne-datei``: Teil h ohne Datei und Klasse, für
Methoden, die nur umgezogen sind. ``--erwartet`` nennt gewollte Abweichungen:
``{"teil:pfadmuster": "Grund"}``. Das Muster ist der Pfad, wie ``--vergleiche``
ihn ausgibt; einziger Platzhalter ist ``*`` (beliebig viele Zeichen), eckige
Klammern und Fragezeichen stehen für sich selbst.

Ende: 0 = gleich (bis auf erwartete) bzw. Soll erfüllt, 1 = Abweichung, 2 = Fehler.

Schema der ``soll_gui.json`` (``--pruefe-schema``, ``--soll``)::

    {
     "abschnitte": [{"schluessel": str, "titel": str, "offen": bool,
                     "zeilen": [{"attribut": str|null, "text": str|null,
                                 "unterblock": str|null}]}],
     "leiste": [{…}],
     "menue": [{"pfad": str, "schluessel": str|null, "text": str,
                "kuerzel": str, "schaltbar": bool}],
     "befehle": {schluessel: {"braucht": [str]|null, "frei_bei_busy": bool,
                              "tooltip": str}},
     "freigabe_sonder": {"rviz": {…}, "blaumaske": {…}, "vorschau": {…},
                         "edl": {…}, "thermal_haken": {…}},
     "entfallen": [str], "neu": [str | {"text": str}],
     "breite_min": int, "annahmen": {…},
     "tooltips": {"aenderungen": [{"vorher": str, "nachher": str|null,
                                   "grund": str}]}          (darf fehlen)
    }

* ``abschnitte``: Folge der Seitenleiste. ``titel`` wie angezeigt, ``offen``
  die Vorgabe. ``zeilen`` in Reihenfolge: ``attribut`` ist der Name am Fenster
  (``_btn_open``, ``_ext_spins[yaw]``) oder null für ein Widget ohne Namen,
  ``text`` die Beschriftung (bei Reglern die der Formularzeile, sonst der
  eigene Text; null = nicht geprüft), ``unterblock`` der Schlüssel des
  einklappbaren Unterblocks oder null. Als Unterblock gilt ein Widget mit
  ``key`` und ``is_expanded()``. Zwei Knöpfe einer Zeile sind zwei Einträge.
  Jede Soll-Zeile muss in dieser Folge vorkommen; ein benanntes Widget des
  Abschnitts, das im Soll fehlt, ist eine Abweichung. Abschnitts- und
  Unterblock-Schlüssel müssen in ``_collect_settings()["sections"]`` stehen.
* ``leiste``: die Einträge der Leiste über der 3D-Ansicht in Reihenfolge, wie
  in Teil a (``a_widgets.json``, ``leiste``); verglichen werden je Eintrag die
  genannten Felder.
* ``menue``: alle Einträge der Menüleiste in Reihenfolge, ohne Trennstriche und
  ohne die Auswahlpunkte der drei gefüllten Untermenüs. ``pfad`` sind die
  Menütitel mit ``&``, getrennt durch `` > ``; ``text`` mit ``&``; ``kuerzel``
  leer, wenn es keines gibt. ``tooltip`` und ``statustip`` dürfen dazu.
* ``befehle``: je Aktion. ``braucht`` null = wird nie geschaltet, sonst die
  Liste der Zustandsnamen aus :data:`ZUSTANDSNAMEN` (leer = nur bei laufendem
  Schritt gesperrt). ``tooltip`` ist der Fachtext (StatusTip der Aktion).
  Im Bereich ``befehle`` zählt außerdem jeder Handler-Name aus
  ``ui/menubar.py``, den das Fenster nicht als Methode hat, als Abweichung.
* ``freigabe_sonder``: je Block die erwarteten Werte in der Form des
  gleichnamigen Blocks von Teil c (``c_freigabe.json``); verglichen wird, was
  genannt ist.
* ``entfallen``: Attributnamen, die es am Fenster nicht mehr geben darf.
  ``neu``: Attributnamen, die es geben muss, oder ``{"text": …}`` für ein
  Widget der Seitenleiste mit diesem Text.
* ``breite_min``: ``_sidebar_breite`` darf nicht darunter liegen.
* ``tooltips``: jeder Tooltip des Vorher-Stands (Teile a, c, e, f) muss
  wörtlich wieder vorkommen oder unter ``aenderungen`` stehen; ``nachher``
  (falls nicht null) muss es im Ist geben.
* ``annahmen``: frei, wird nicht geprüft.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import inspect
import json
import os
import re
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import basis  # noqa: E402

TEILE = {"a": "Widget-Baum", "b": "Einstellungen", "c": "Freigabe",
         "d": "Handler", "e": "Menübaum", "f": "Nebenfenster",
         "h": "Verbindungen", "i": "Fenster"}
DATEIEN = {"a": "a_widgets.json", "b": "b_einstellungen.json",
           "c": "c_freigabe.json", "d": "d_handler.json", "e": "e_menue.json",
           "f": "f_nebenfenster.json", "h": "h_verbindungen.json",
           "i": "i_fenster.json"}
BEISPIEL_DATEI = "settings_beispiel.json"
STAND_DATEI = "stand.json"
VORGABE_ZIEL = os.path.join(basis.VORHER_ORDNER, "gui")

#: Zustandsnamen, wie sie ``braucht`` in der Soll-Beschreibung benutzt
ZUSTANDSNAMEN = ("bag", "rec", "world", "calib", "flug", "lage", "thermal",
                 "zweitflug", "fusion")
_KARTE = ("bag", "calib", "rec", "world")
#: Was ohne die genannte Voraussetzung ebenfalls nicht gelten kann: ob es
#: Thermalbilder gibt, weiß nur die Lage (``_hat_thermal`` fragt die Pipeline).
_HAENGT_AN = {"lage": ("thermal",)}


def _alles_ausser(name: str) -> tuple:
    """Alle Voraussetzungen bis auf ``name`` (und was an ihr hängt)."""
    weg = (name,) + _HAENGT_AN.get(name, ())
    return tuple(z for z in ZUSTANDSNAMEN if z not in weg)


#: (Name, was gilt). "busy" und "fest" (nicht abbrechbar) sind keine
#: Voraussetzungen, sondern der laufende Schritt. Die Zustände ``ohne_<name>``
#: trennen die Voraussetzungen: fehlt einer Freigabe eine Bedingung oder hängt
#: sie an der falschen, fällt das in genau einem von ihnen auf.
ZUSTAENDE = (
    ("leer", ("calib",)),
    ("bag", ("bag", "calib")),
    ("bag_ohne_kalibrierung", ("bag",)),
    ("karte", _KARTE),
    ("karte_ohne_bag", ("calib", "rec", "world")),
    ("karte_flug", _KARTE + ("flug",)),
    ("lage", _KARTE + ("flug", "lage")),
    ("lage_thermal", _KARTE + ("flug", "lage", "thermal")),
    ("zweitflug", _KARTE + ("zweitflug",)),
    ("fusion", _KARTE + ("fusion",)),
) + tuple(("ohne_" + z, _alles_ausser(z)) for z in ZUSTANDSNAMEN) + (
    ("beschaeftigt", ZUSTANDSNAMEN + ("busy",)),
    ("beschaeftigt_fest", ZUSTANDSNAMEN + ("busy", "fest")),
)
SONDER = ("rviz", "blaumaske", "vorschau", "edl", "thermal_haken")
SOLL_BEREICHE = ("abschnittsliste", "abschnitte", "leiste", "menue", "befehle",
                 "freigabe", "tooltips", "attribute")

_HANDLER = re.compile(r"^(_on_[A-Za-z0-9_]+|close)$")
_KONTEXT = {0: "WidgetShortcut", 1: "WindowShortcut", 2: "ApplicationShortcut",
            3: "WidgetWithChildrenShortcut"}
_FEHLT = "<fehlt>"


# ------------------------------------------------------------------ Helfer

def _adr(obj) -> int:
    from PyQt5 import sip
    return int(sip.unwrapinstance(obj))


def _json(wert):
    """Beliebigen Wert in etwas JSON-Fähiges wandeln (Tupel -> Liste)."""
    if wert is None or isinstance(wert, (bool, int, str)):
        return wert
    if isinstance(wert, float):
        return wert
    if isinstance(wert, dict):
        return {str(k): _json(v) for k, v in wert.items()}
    if isinstance(wert, (list, tuple)):
        return [_json(v) for v in wert]
    try:
        import numpy as np
        if isinstance(wert, np.generic):
            return _json(wert.item())
    except ImportError:
        pass
    return f"<{type(wert).__name__}>"


def _neutral(wert):
    """Jede Zeichenkette durch ``basis.pfad_neutral`` schicken."""
    if isinstance(wert, str):
        return basis.pfad_neutral(wert)
    if isinstance(wert, dict):
        return {basis.pfad_neutral(k): _neutral(v) for k, v in wert.items()}
    if isinstance(wert, list):
        return [_neutral(v) for v in wert]
    return wert


def _typ(wert) -> str:
    if isinstance(wert, list):
        return "list[" + ",".join(sorted({type(v).__name__ for v in wert})) + "]"
    if isinstance(wert, dict):
        return "dict[" + ",".join(sorted({type(v).__name__ for v in wert.values()})) + "]"
    return type(wert).__name__


def _eigenes(obj) -> bool:
    """Widget-Klasse aus ``ui/`` (und nicht aus Qt)?"""
    return type(obj).__module__.split(".")[0] == "ui"


def _ist_regler(w) -> bool:
    """Zusammengesetzter Regler (ui.feinregler): Wert, Maximum, eigenes Signal."""
    from PyQt5.QtWidgets import QAbstractSlider, QAbstractSpinBox
    return (_eigenes(w) and not isinstance(w, (QAbstractSpinBox, QAbstractSlider))
            and callable(getattr(w, "value", None))
            and callable(getattr(w, "maximum", None)))


def _unterblock(w):
    """Schlüssel, wenn ``w`` ein einklappbarer Block ist (sonst None)."""
    for name, offen in (("key", "is_expanded"), ("schluessel", "ist_offen")):
        key = getattr(w, name, None)
        if isinstance(key, str) and callable(getattr(w, offen, None)):
            return key
    return None


def namen_karte(fenster) -> dict:
    """Adresse -> Namen am Fenster, etwa ``_btn_open`` oder ``_ext_spins[yaw]``.

    Eigene Widgets des Fensters (3D-Ansicht, 360°-Ansicht, GPS-Reiter …)
    werden eine Ebene tief mit Punkt geführt: ``_cloud_view._farbe``.
    """
    from PyQt5.QtWidgets import QAction, QWidget
    karte: dict = {}

    def merke(obj, name):
        if isinstance(obj, (QWidget, QAction)):
            karte.setdefault(_adr(obj), []).append(name)
            return True
        return False

    def schau(besitzer, vor, tiefe):
        for name, wert in sorted(vars(besitzer).items()):
            voll = vor + name
            if merke(wert, voll):
                if tiefe == 0 and isinstance(wert, QWidget) and _eigenes(wert):
                    schau(wert, voll + ".", 1)
            elif isinstance(wert, dict):
                for k, v in wert.items():
                    key = ",".join(str(t) for t in k) if isinstance(k, tuple) else str(k)
                    merke(v, f"{voll}[{key}]")
            elif isinstance(wert, (list, tuple)):
                for i, v in enumerate(wert):
                    merke(v, f"{voll}[{i}]")

    schau(fenster, "", 0)
    for namen in karte.values():
        namen.sort(key=lambda n: ("[" in n, n))
    return karte


# -------------------------------------------------------------- Widget-Baum

def _eintrag(w, namen: dict) -> dict:
    """Was ein Widget für sich festhält."""
    from PyQt5 import QtWidgets as Q
    alle = namen.get(_adr(w), [])
    e: dict = {"typ": type(w).__name__, "attribut": alle[0] if alle else None,
               "text": None, "tooltip": w.toolTip()}
    if len(alle) > 1:
        e["auch"] = alle[1:]
    if w.isHidden():
        e["versteckt"] = True
    eltern = w.parentWidget()
    if eltern is not None and not w.isEnabledTo(eltern):
        e["gesperrt"] = True
    if w.styleSheet():
        e["stil"] = w.styleSheet()
    if isinstance(w, Q.QLabel):
        e["text"] = w.text()
        if w.wordWrap():
            e["umbruch"] = True
        pm = w.pixmap()
        if pm is not None and not pm.isNull():
            e["bild"] = [pm.width(), pm.height()]
    elif isinstance(w, Q.QAbstractButton):
        e["text"] = w.text()
        if w.isCheckable():
            e["schaltbar"] = True
            e["haken"] = bool(w.isChecked())
        if isinstance(eltern, Q.QDialogButtonBox):
            e["rolle"] = int(eltern.buttonRole(w))
    elif isinstance(w, Q.QGroupBox):
        e["text"] = w.title()
    elif isinstance(w, (Q.QSpinBox, Q.QDoubleSpinBox)):
        e.update(bereich=[w.minimum(), w.maximum()], schritt=w.singleStep(),
                 suffix=w.suffix(), wert=w.value())
        if w.prefix():
            e["praefix"] = w.prefix()
        if isinstance(w, Q.QDoubleSpinBox):
            e["dezimalen"] = w.decimals()
    elif isinstance(w, Q.QAbstractSlider):
        e.update(bereich=[w.minimum(), w.maximum()], schritt=w.singleStep(),
                 seitenschritt=w.pageStep(), wert=w.value())
    elif isinstance(w, Q.QComboBox):
        eintraege = []
        for i in range(w.count()):
            ein = {"text": w.itemText(i), "daten": _json(w.itemData(i))}
            modell = w.model()
            item = modell.item(i) if hasattr(modell, "item") else None
            if item is not None and not item.isEnabled():
                ein["gesperrt"] = True
            eintraege.append(ein)
        e.update(eintraege=eintraege, index=w.currentIndex())
    elif isinstance(w, Q.QProgressBar):
        e.update(bereich=[w.minimum(), w.maximum()], wert=w.value())
    elif isinstance(w, Q.QLineEdit):
        e["text"] = w.text()
        e["platzhalter"] = w.placeholderText()
    elif isinstance(w, Q.QTableWidget):
        kopf = []
        for c in range(w.columnCount()):
            item = w.horizontalHeaderItem(c)
            kopf.append(item.text() if item is not None else None)
        e.update(spalten=kopf, zeilen=w.rowCount())
        if 0 < w.rowCount() <= 20:
            zellen = []
            for r in range(w.rowCount()):
                reihe = []
                for c in range(w.columnCount()):
                    item = w.item(r, c)
                    reihe.append(None if item is None else
                                 {"text": item.text(), "tooltip": item.toolTip()})
                zellen.append(reihe)
            e["zellen"] = zellen
    elif isinstance(w, Q.QListWidget):
        e["zeilen"] = w.count()
    elif isinstance(w, Q.QPlainTextEdit):
        e.update(nur_lesen=bool(w.isReadOnly()), hoechstens=w.maximumBlockCount())
    elif isinstance(w, Q.QTabWidget):
        e["reiter"] = [w.tabText(i) for i in range(w.count())]
    if _ist_regler(w):
        e.update(wert=w.value(), maximum=w.maximum())
    key = _unterblock(w)
    if key is not None:
        e["unterblock_kopf"] = key
        offen = getattr(w, "is_expanded", None) or getattr(w, "ist_offen")
        e["offen"] = bool(offen())
    return e


def _aus_item(item, beschriftung):
    if item is None:
        return
    if item.widget() is not None:
        yield item.widget(), beschriftung
    elif item.layout() is not None:
        for w, b in _aus_layout(item.layout()):
            yield w, (b if b is not None else beschriftung)


def _aus_layout(lay):
    """Widgets eines Layouts in Reihenfolge, mit der Beschriftung ihrer Zeile."""
    from PyQt5 import QtWidgets as Q
    if isinstance(lay, Q.QFormLayout):
        for r in range(lay.rowCount()):
            kopf = lay.itemAt(r, Q.QFormLayout.LabelRole)
            feld = lay.itemAt(r, Q.QFormLayout.FieldRole)
            text = None
            if kopf is not None and isinstance(kopf.widget(), Q.QLabel):
                text = kopf.widget().text()
            ganz = lay.itemAt(r, Q.QFormLayout.SpanningRole)
            if ganz is not None:
                # eine Zeile über die volle Breite meldet Qt auch als Feld
                yield from _aus_item(ganz, None)
                continue
            if kopf is not None and feld is not None and kopf.widget() is not None:
                yield kopf.widget(), ("", "zeilenkopf")
            else:
                yield from _aus_item(kopf, None)
            yield from _aus_item(feld, text)
    else:
        for i in range(lay.count()):
            yield from _aus_item(lay.itemAt(i), None)


def _kinder(w):
    from PyQt5 import QtWidgets as Q
    blatt = (Q.QAbstractSpinBox, Q.QComboBox, Q.QAbstractSlider, Q.QAbstractButton,
             Q.QLabel, Q.QLineEdit, Q.QProgressBar, Q.QAbstractItemView,
             Q.QPlainTextEdit, Q.QTextEdit, Q.QGraphicsView)
    if isinstance(w, blatt):
        return
    if isinstance(w, Q.QScrollArea):
        if w.widget() is not None:
            yield w.widget(), None
    elif isinstance(w, (Q.QTabWidget, Q.QStackedWidget, Q.QSplitter)):
        for i in range(w.count()):
            yield w.widget(i), None
    elif w.layout() is not None:
        yield from _aus_layout(w.layout())


def baum(wurzel_widget, namen: dict, mit_wurzel: bool = False) -> list:
    """Flache Liste der Widgets unter ``wurzel_widget`` in Layout-Reihenfolge."""
    aus: list = []

    def geh(w, tiefe, beschriftung, block, regler):
        e = _eintrag(w, namen)
        e["tiefe"] = tiefe
        if isinstance(beschriftung, tuple):
            e["zeilenkopf"] = True
        elif beschriftung is not None:
            e["beschriftung"] = beschriftung
        if block is not None:
            e["unterblock"] = block
        if regler:
            e["teil_eines_reglers"] = True
        aus.append(e)
        block = e.get("unterblock_kopf", block)
        regler = regler or _ist_regler(w)
        for kind, b in _kinder(w):
            geh(kind, tiefe + 1, b, block, regler)

    if mit_wurzel:
        geh(wurzel_widget, 0, None, None, False)
    else:
        for kind, b in _kinder(wurzel_widget):
            geh(kind, 0, b, None, False)
    return aus


def abschnitte(fenster) -> list:
    """Die Abschnitte der Seitenleiste in Reihenfolge: ``[(schluessel, Section)]``."""
    stapel = fenster._sections
    roh = getattr(stapel, "_sections", None)
    if not isinstance(roh, dict):
        roh = stapel.sections()
    if isinstance(roh, dict):
        return list(roh.items())
    return [(s.key, s) for s in roh]


def _titel(sec) -> str:
    return str(getattr(sec, "_title", ""))


def teil_a(fenster, namen: dict) -> dict:
    from PyQt5 import QtWidgets as Q
    seite: dict = {}
    for key, sec in abschnitte(fenster):
        seite[key] = {"titel": _titel(sec), "offen": bool(sec.is_expanded()),
                      "widgets": baum(sec.content(), namen)}
    leiste = fenster._cloud_view.findChild(Q.QFrame, "wolkenleiste")
    status = [dict(_eintrag(w, namen), tiefe=0)
              for w in fenster.statusBar().children()
              if isinstance(w, Q.QWidget) and not isinstance(w, Q.QSizeGrip)]
    tabs = fenster._tabs
    return {
        "seitenleiste": {"reihenfolge": [k for k, _ in abschnitte(fenster)],
                         "abschnitte": seite,
                         "breite": int(fenster._sidebar_breite)},
        "leiste": baum(leiste, namen) if leiste is not None else [],
        "statusleiste": status,
        "tabs": [{"text": tabs.tabText(i), "typ": type(tabs.widget(i)).__name__,
                  "attribut": (namen.get(_adr(tabs.widget(i))) or [None])[0]}
                 for i in range(tabs.count())],
        "fenstertitel": fenster.windowTitle(),
    }


# -------------------------------------------------------------- Einstellungen

def _beispiel(start: dict) -> dict:
    """Je Schlüssel ein gültiger Wert, der vom Start abweicht."""
    bsp = {
        "pano_width": 1280, "config": "mid360.yaml", "rate": 0.5,
        "brightness_min": 35, "brightness_max": 210, "k_frames": 5,
        "sky_grow": 7, "lens_best": False, "edge_r": 520, "blue_filter": True,
        "blue_hue_lo": 180, "blue_hue_hi": 270, "blue_sat": 30, "blue_val": 45,
        "blue_neutral": 60, "mesh_voxel_cm": 8, "mesh_depth": 10,
        "mesh_trim": 12, "mesh_an": True, "mesh_stand": 3, "mesh_hybrid": False,
        "point_size": 3.5, "temperatur_anzeigen": False, "color_mode": "hoehe",
        "layer": "meander_rgb", "meander_dir": basis.ARBEIT_WURZEL,
        "meander_thermal": True, "meander_sichtbar": False, "meander_solo": True,
        "splat_raster": 0.08, "splat_anker_mio": 2.5, "splat_sh": 2,
        "splat_posen": False, "splat_pruefen": False, "splat_thermal": False,
        "splat_schritte": 6500, "splat_frames": 450,
        "splat_schritte_onboard": 22000, "rgb_versatz": [12.5, -7.25],
        "thermal_versatz": [-3.0, 4.5], "only_colored": True, "voxel": 0.1,
        "background": "hell", "edl": True, "show_path": True,
        "sections": {k: not v for k, v in start["sections"].items()},
        "sidebar": False,
    }
    ohne = sorted(set(start) - set(bsp))
    if ohne:
        raise RuntimeError(f"Einstellungsschlüssel ohne Beispielwert: {ohne} — "
                           "bitte in schnappschuss._beispiel nachtragen.")
    return {k: bsp[k] for k in start}


def teil_b(fenster) -> tuple:
    """Fälle der Einstellungen; zurück kommen (Teil, Beispiel-dict)."""
    from PyQt5.QtWidgets import QMessageBox

    def sammle() -> dict:
        s = fenster._collect_settings()
        return {k: {"typ": _typ(v), "wert": _json(v)} for k, v in s.items()}

    def wende_an(s: dict) -> dict:
        fenster._settings = dict(s)
        fenster._apply_settings_to_widgets()
        return sammle()

    faelle: dict = {}
    start = fenster._collect_settings()
    faelle["start"] = sammle()

    # Vorgabe über den Weg von "Einstellungen auf Vorgabe"; danach steht die
    # Vorbelegung in _settings.
    with basis.dialoge_abfangen() as dialoge:
        dialoge.antworte("QMessageBox.question", QMessageBox.Yes)
        fenster._on_settings_reset()
    vorgabe = dict(fenster._settings)
    faelle["vorgabe"] = sammle()

    def fall(name, **mehr):
        faelle[name] = wende_an(dict(vorgabe, **mehr))

    fall("vorgabe_plus_teilmenge", rate=0.5, k_frames=5, blue_filter=True,
         mesh_voxel_cm=8, color_mode="hoehe", splat_schritte=5000,
         meander_thermal=True, voxel=0.2)
    fall("blau_alt_200_240_40_60", blue_hue_lo=200, blue_hue_hi=240,
         blue_sat=40, blue_val=60)
    fall("blau_alt_170_250_25_40", blue_hue_lo=170, blue_hue_hi=250,
         blue_sat=25, blue_val=40)
    fall("mesh_5_11_ohne_stand", mesh_voxel_cm=5, mesh_depth=11)
    fall("mesh_5_11_mit_stand_3", mesh_voxel_cm=5, mesh_depth=11, mesh_stand=3)
    fall("mesh_trim_5_mit_stand_1", mesh_trim=5, mesh_stand=1)
    fall("mesh_trim_5_mit_stand_2", mesh_trim=5, mesh_stand=2)

    bsp = _beispiel(start)
    faelle["beispiel"] = wende_an(bsp)
    gesammelt = fenster._collect_settings()
    # Das Beispiel muss sich selbst wiedergeben, sonst taugt es nicht als Probe
    # für Laden und Speichern.
    ungleich = sorted(k for k in bsp if _json(gesammelt.get(k)) != _json(bsp[k])
                      and not (k == "edl" and not fenster._chk_edl.isEnabled()))
    if ungleich or set(gesammelt) != set(bsp):
        raise RuntimeError(f"Beispiel kommt nicht unverändert zurück: {ungleich}")
    gleich = sorted(k for k in bsp if _json(gesammelt[k]) == _json(start[k])
                    and k != "mesh_stand")
    if gleich:
        raise RuntimeError(f"Beispiel weicht in {gleich} nicht vom Start ab.")
    beispiel = _json(gesammelt)
    faelle["beispiel_dann_vorgabe"] = wende_an(vorgabe)

    # Ausgangslage wiederherstellen (Teil c baut darauf auf)
    fenster._meander_dir = None
    fenster._lbl_meander.setText("Kein Mäanderflug geladen.")
    wende_an(dict(vorgabe, **start))
    fenster._settings = vorgabe
    teil = {"vorgabe_dict": {k: {"typ": _typ(v), "wert": _json(v)}
                             for k, v in vorgabe.items()},
            "faelle": faelle}
    return teil, beispiel


# ------------------------------------------------------------------ Freigabe

class _StubBag:
    bag_path = "stub_bag"


class _StubRec:
    intensity = None
    n_scans = 0

    def path_positions(self):
        import numpy as np
        return np.zeros((0, 3))


class _StubPipe:
    def __init__(self, thermal: bool):
        import numpy as np
        self.yaw = float(np.radians(30.0))
        self.t = np.array([1.0, 2.0, 0.5])
        self._thermal = bool(thermal)

    def thermal_cams(self):
        return {"stub": True} if self._thermal else None


class _StubPlayer:
    def __init__(self):
        self.spielt = False
        self.offen = False

    def is_playing(self) -> bool:
        return self.spielt

    def rviz_running(self) -> bool:
        return self.offen

    def stop(self) -> None:
        pass


class _StubArbeiter:
    def __init__(self, cancellable: bool):
        import threading
        self.cancel = threading.Event()
        self.cancellable = bool(cancellable)

    def isFinished(self) -> bool:
        return True

    def wait(self, *_a) -> bool:
        return True


class _StubProjekt:
    dir = ""
    bag_name = "stub"

    def save_settings(self, _s) -> None:
        pass


def _schaltbare(fenster, namen: dict) -> dict:
    """Name -> Widget für alles, dessen Freigabe Teil c festhält.

    Knöpfe, Zahlenfelder, Schieber, Auswahlen und zusammengesetzte Regler, die
    das Fenster als Attribut oder in einem dict hält; dazu jedes Mitglied
    einer Sammlung (Blaulicht-Regler), auch wenn es nur ein Label ist.
    """
    from PyQt5 import QtWidgets as Q
    arten = (Q.QAbstractButton, Q.QAbstractSpinBox, Q.QAbstractSlider, Q.QComboBox)
    aus: dict = {}

    def nimm(w, immer=False):
        if isinstance(w, Q.QWidget) and (immer or isinstance(w, arten) or _ist_regler(w)):
            aus[namen[_adr(w)][0]] = w

    for wert in vars(fenster).values():
        if isinstance(wert, dict):
            for v in wert.values():
                nimm(v)
        elif isinstance(wert, (list, tuple)):
            for v in wert:
                nimm(v, immer=True)
        else:
            nimm(wert)
    return dict(sorted(aus.items()))


def _setze_zustand(fenster, gilt: tuple) -> None:
    import numpy as np
    da = set(gilt)
    punkte = np.zeros((8, 3), dtype=np.float32)
    fenster._bag = _StubBag() if "bag" in da else None
    fenster._rec = _StubRec() if "rec" in da else None
    fenster._world = punkte if "world" in da else None
    fenster._calib = "kalibrierung.json" if "calib" in da else ""
    fenster._meander_dir = basis.ARBEIT_WURZEL if "flug" in da else None
    fenster._meander_pipe = _StubPipe("thermal" in da) if "lage" in da else None
    zweit = "zweitflug" in da
    fenster._merge_rec = _StubRec() if zweit else None
    fenster._merge_bag = _StubBag() if zweit else None
    fenster._merge_cloud = punkte.copy() if zweit else None
    fenster._cloud_view.set_preview_cloud(punkte + 1.0 if zweit else None)
    farbe = (np.zeros((8, 3), np.uint8), np.ones(8, bool))
    fenster._layers = ({"onboard": farbe, "meander_rgb": farbe}
                       if "fusion" in da else {})
    fenster._busy = "busy" in da
    fenster._worker = _StubArbeiter(False) if "fest" in da else None


def teil_c(fenster, namen: dict) -> dict:
    from PyQt5 import QtWidgets as Q

    merken = ("_bag", "_rec", "_world", "_calib", "_meander_dir", "_meander_pipe",
              "_merge_rec", "_merge_bag", "_merge_cloud", "_layers", "_busy",
              "_worker", "_project", "_rviz_player", "_settings")
    alt = {n: getattr(fenster, n) for n in merken}
    lbl_alt = fenster._lbl_meander.text()
    widgets = _schaltbare(fenster, namen)
    aktionen = fenster._actions
    tip_start = {n: w.toolTip() for n, w in widgets.items()}
    tip_start.update({f"_actions[{k}]": a.toolTip() for k, a in aktionen.items()
                      if isinstance(a, Q.QAction)})
    blau = getattr(fenster, "_blue_widgets", ())
    blau_namen = sorted(namen.get(_adr(w), ["?"])[0] for w in blau)
    if "_btn_blue_preview" not in blau_namen and hasattr(fenster, "_btn_blue_preview"):
        blau_namen.append("_btn_blue_preview")
    blau_haken = bool(fenster._chk_blue.isChecked())

    def frei(w) -> bool:
        return bool(w.isEnabledTo(fenster))

    def rviz_stand() -> dict:
        aus = {n: frei(getattr(fenster, n)) for n in
               ("_btn_rviz_start", "_btn_rviz_stop", "_btn_rviz_replay")
               if hasattr(fenster, n)}
        aus.update({k: bool(aktionen[k].isEnabled()) for k in
                    ("rviz_start", "rviz_stop", "rviz_replay") if k in aktionen})
        aus["_lbl_rviz"] = fenster._lbl_rviz.text()
        return aus

    player = _StubPlayer()
    fenster._rviz_player = player
    zustaende: dict = {}
    sonder: dict = {k: {} for k in ("rviz", "blaumaske", "vorschau")}
    try:
        for name, gilt in ZUSTAENDE:
            _setze_zustand(fenster, gilt)
            fenster._update_enabled()
            sonder["rviz"][name] = {}
            for lage, (offen, spielt) in (("offen", (True, False)),
                                          ("spielend", (True, True)),
                                          ("gestoppt", (False, False))):
                player.offen, player.spielt = offen, spielt
                fenster._refresh_rviz_state()
                sonder["rviz"][name][lage] = rviz_stand()
            fenster._sync_preview_action()
            vorschau = aktionen.get("preview")
            sonder["vorschau"][name] = (
                {"frei": bool(vorschau.isEnabled()), "haken": bool(vorschau.isChecked())}
                if vorschau is not None else None)
            sonder["blaumaske"][name] = {}
            for an in (True, False):
                fenster._chk_blue.blockSignals(True)
                fenster._chk_blue.setChecked(an)
                fenster._chk_blue.blockSignals(False)
                fenster._update_blue_widgets()
                stand = {n: frei(widgets[n]) for n in blau_namen if n in widgets}
                if "blaumaske" in aktionen:
                    stand["_actions[blaumaske]"] = bool(aktionen["blaumaske"].isEnabled())
                sonder["blaumaske"][name]["filter_an" if an else "filter_aus"] = stand
            tips = {n: w.toolTip() for n, w in widgets.items()
                    if w.toolTip() != tip_start[n]}
            tips.update({f"_actions[{k}]": a.toolTip() for k, a in aktionen.items()
                         if isinstance(a, Q.QAction)
                         and a.toolTip() != tip_start[f"_actions[{k}]"]})
            zustaende[name] = {
                "gilt": sorted(gilt),
                "aktionen": {k: bool(a.isEnabled()) for k, a in aktionen.items()},
                "knoepfe": {n: frei(w) for n, w in widgets.items()
                            if n.startswith("_btn_")},
                "regler": {n: frei(w) for n, w in widgets.items()
                           if not n.startswith("_btn_")},
                "tooltips": tips,
                "texte": {n: getattr(fenster, n).text()
                          for n in ("_lbl_meander_lage",) if hasattr(fenster, n)},
            }

        # EDL: Menü folgt dem Haken der Seitenleiste (_fill_view_menus)
        _setze_zustand(fenster, ("calib",))
        fenster._update_enabled()
        chk = getattr(fenster, "_chk_edl", None)
        sonder["edl"] = {}
        if chk is not None and "edl" in aktionen:
            vor = (chk.isEnabled(), chk.isChecked())
            for lage, (an, haken) in (("frei", (True, False)),
                                      ("frei_mit_haken", (True, True)),
                                      ("gesperrt", (False, False))):
                chk.blockSignals(True)
                chk.setEnabled(an)
                chk.setChecked(haken)
                chk.blockSignals(False)
                fenster._fill_view_menus()
                sonder["edl"][lage] = {"frei": bool(aktionen["edl"].isEnabled()),
                                       "haken": bool(aktionen["edl"].isChecked())}
            chk.blockSignals(True)
            chk.setEnabled(vor[0])
            chk.setChecked(vor[1])
            chk.blockSignals(False)
            fenster._fill_view_menus()
        sonder["edl"]["verfuegbar"] = bool(fenster._cloud_view.edl_available)

        # Thermal-Haken: nach der Flugwahl nur mit Thermalbildern frei
        ordner = basis.arbeitsordner("schnappschuss")
        sonder["thermal_haken"] = {}
        for lage, dateien in (("flug_mit_thermal", ("a_V.JPG", "a_T.JPG", "b_V.JPG")),
                              ("flug_ohne_thermal", ("a_V.JPG",)),
                              ("flug_nur_jpg", ("a.jpg", "b.JPG"))):
            pfad = os.path.join(ordner, lage)
            os.makedirs(pfad)
            for datei in dateien:
                open(os.path.join(pfad, datei), "wb").close()
            _setze_zustand(fenster, _KARTE)
            fenster._project = _StubProjekt()
            fenster._chk_thermal.setEnabled(True)
            fenster._chk_thermal.blockSignals(True)
            fenster._chk_thermal.setChecked(True)
            fenster._chk_thermal.blockSignals(False)
            with basis.dialoge_abfangen() as dialoge:
                dialoge.antworte("QFileDialog.getExistingDirectory", pfad)
                fenster._on_meander_pick()
            sonder["thermal_haken"][lage] = {
                "frei": frei(fenster._chk_thermal),
                "haken": bool(fenster._chk_thermal.isChecked()),
                "_lbl_meander": fenster._lbl_meander.text(),
                "dialoge": [d["art"] for d in dialoge.protokoll]}
            fenster._project = None
    finally:
        fenster._chk_thermal.setEnabled(True)
        fenster._chk_thermal.blockSignals(True)
        fenster._chk_thermal.setChecked(False)
        fenster._chk_thermal.blockSignals(False)
        fenster._chk_blue.blockSignals(True)
        fenster._chk_blue.setChecked(blau_haken)
        fenster._chk_blue.blockSignals(False)
        fenster._cloud_view.set_preview_cloud(None)
        for n, wert in alt.items():
            setattr(fenster, n, wert)
        fenster._lbl_meander.setText(lbl_alt)
        fenster._update_blue_widgets()
        fenster._update_enabled()
        fenster._refresh_rviz_state()
        fenster._sync_preview_action()
    aus = {"zustaende": zustaende}
    aus.update(sonder)
    return aus


# ------------------------------------------------------------------- Handler

def teil_d(fenster) -> dict:
    pfad = os.path.join(basis.wurzel(), "ui", "menubar.py")
    with open(pfad, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    anzahl: dict = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and _HANDLER.match(node.value):
            anzahl[node.value] = anzahl.get(node.value, 0) + 1
    aus: dict = {}
    for name in sorted(anzahl):
        fn = getattr(fenster, name, None)
        try:
            signatur = str(inspect.signature(fn)) if callable(fn) else None
        except (TypeError, ValueError):
            signatur = "<Qt>"
        aus[name] = {"nennungen": anzahl[name], "vorhanden": callable(fn),
                     "signatur": signatur}
    return {"handler": aus}


# ------------------------------------------------------------------ Menübaum

def teil_e(fenster) -> dict:
    from PyQt5 import QtWidgets as Q
    from PyQt5.QtCore import Qt
    schluessel = {_adr(obj): key for key, obj in fenster._actions.items()}

    def eintraege(menu) -> list:
        liste = []
        for act in menu.actions():
            if act.isSeparator():
                liste.append({"trenner": True})
                continue
            unter = act.menu()
            key = schluessel.get(_adr(unter)) if unter is not None else None
            e = {"schluessel": key or schluessel.get(_adr(act)),
                 "text": act.text(),
                 "kuerzel": act.shortcut().toString(),
                 "tooltip": act.toolTip(), "statustip": act.statusTip(),
                 "schaltbar": bool(act.isCheckable()),
                 "kontext": _KONTEXT.get(int(act.shortcutContext()),
                                         str(int(act.shortcutContext()))),
                 "frei": bool(act.isEnabled())}
            if act.isCheckable():
                e["haken"] = bool(act.isChecked())
            if act.data() is not None:
                e["daten"] = _json(act.data())
            if unter is not None:
                e["untermenue"] = eintraege(unter)
                e["tooltips_sichtbar"] = bool(unter.toolTipsVisible())
            liste.append(e)
        return liste

    bar = fenster.menuBar()
    ecke = bar.cornerWidget(Qt.TopRightCorner)
    return {"native_menueleiste": bool(bar.isNativeMenuBar()),
            "ecke": type(ecke).__name__ if ecke is not None else None,
            "menues": eintraege(bar),
            "zahl": {"aktionen": sum(isinstance(a, Q.QAction)
                                     for a in fenster._actions.values()),
                     "untermenues": sum(isinstance(a, Q.QMenu)
                                        for a in fenster._actions.values())}}


def menue_flach(menues: list, pfad: str = "") -> list:
    """Menübaum als Liste wie ``menue`` der Soll-Beschreibung."""
    aus = []
    for e in menues:
        if e.get("trenner"):
            continue
        if pfad:
            aus.append({"pfad": pfad, "schluessel": e["schluessel"],
                        "text": e["text"], "kuerzel": e["kuerzel"],
                        "schaltbar": e["schaltbar"], "tooltip": e["tooltip"],
                        "statustip": e["statustip"]})
        if "untermenue" in e and not (e["schluessel"] or "").startswith("menu_"):
            aus += menue_flach(e["untermenue"],
                               f"{pfad} > {e['text']}" if pfad else e["text"])
    return aus


def _kuerzel(menues: list) -> list:
    aus = []
    for e in menues:
        if e.get("kuerzel"):
            aus.append(e["kuerzel"])
        aus += _kuerzel(e.get("untermenue", []))
    return aus


# -------------------------------------------------------------- Nebenfenster

def _ausrichtfenster(namen_von) -> dict:
    import numpy as np
    from ui.meander_align_window import MeanderAlignWindow

    class Pipe:
        yaw = np.radians(30.0)
        t = np.array([1.0, 2.0, 0.5])
        thermal_versatz = (0.0, 0.0)
        cams = {"xyz": np.random.default_rng(0).uniform(-5, 5, size=(500, 3)),
                "C": np.random.default_rng(1).uniform(-5, 5, size=(20, 3)) + [0, 0, 50],
                "names": np.array(["a"]), "Rcw": np.eye(3)[None],
                "tcw": np.zeros((1, 3)), "size": np.array([[100.0, 100.0]]),
                "params": np.array([[50.0, 50.0, 50.0, 0.0]]),
                "model": np.array("SIMPLE_RADIAL")}

        def affine(self):
            from scipy.spatial.transform import Rotation
            return (Rotation.from_euler("z", self.yaw).as_matrix(),
                    np.asarray(self.t, float))

        def rgb_cams(self):
            return self.cams

        def thermal_cams(self):
            return self.cams

    class Live:
        def colorize(self, P, A, b, cams=None):
            n = len(P)
            return np.zeros((n, 3), np.uint8), np.ones(n, bool)

    welt = np.random.default_rng(2).uniform(-10, 10, size=(5000, 3)).astype(np.float32)
    aus: dict = {}
    for name, kw in (
            ("mit_thermal_eingemessen",
             dict(live_th=Live(), thermal_zuschlag=(2.0, 1.5, 1.0),
                  optik={"rgb_faktor": 1.1, "thermal_faktor": 0.98,
                         "thermal": {"stub": True}})),
            ("mit_thermal", dict(live_th=Live(), optik={"rgb_faktor": 1.0,
                                                        "thermal_faktor": 1.0})),
            ("ohne_thermal", dict())):
        w = MeanderAlignWindow(welt, Pipe(), Live(), **kw)
        aus[name] = {"titel": w.windowTitle(), "widgets": baum(w, namen_von(w))}
        w.deleteLater()
    return aus


def _dialoge() -> dict:
    from ui.bundle_dialog import ExportDialog, ProjectOpenDialog
    info = {
        "recording": {"da": True, "bytes": 402_653_184},
        "colors": {"da": True, "bytes": 289_406_976,
                   "ebenen": ["onboard", "meander_rgb"]},
        "panos": {"da": True, "bytes": 1_610_612_736, "breiten": [1920]},
        "meander": {"da": False, "bytes": 0},
        "bags": {"da": True, "bytes": 25_769_803_776, "pfade": ["stub_bag"]},
    }
    ziel = os.path.join(basis.arbeitsordner("schnappschuss"), "export")
    projekte = [
        {"dir": "cache/a-1", "name": "flug_a", "bag": "bags/a", "n_scans": 100,
         "n_points": 1_000_000, "created": "2026-09-10T17:40", "quellen": [],
         "zusammengefuehrt": False},
        {"dir": "cache/ab-2", "name": "flug_a+flug_b", "bag": "bags/a",
         "n_scans": 300, "n_points": 24_300_000, "created": "2026-09-10T18:00",
         "quellen": ["bags/a", "bags/b"], "zusammengefuehrt": True},
    ]
    aus: dict = {}
    for name, dlg in (("export", ExportDialog("flug_a", info, ziel)),
                      ("export_ohne_ziel", ExportDialog("flug_a", info, "")),
                      ("projekt_oeffnen", ProjectOpenDialog(projekte)),
                      ("projekt_oeffnen_leer", ProjectOpenDialog([]))):
        lokal = namen_karte(dlg)
        aus[name] = {"titel": dlg.windowTitle(), "widgets": baum(dlg, lokal)}
        dlg.deleteLater()
    return aus


def _kachel() -> dict:
    from core.exploration import Explorationsgrad
    from ui.explorationsgrad import ExplorationsgradAnzeige

    def probe(**kw):
        werte = dict(
            prozent=None, prozent_flaeche=None, prozent_gesamt=96.8,
            prozent_flaeche_gesamt=100.0, stand_beginn=None, stand_ende=None,
            zuwachs=None, box_min=[9.5, -2.5, 0.5], box_max=[27.5, 19.0, 7.0],
            volumen_m3=2516.0, flaeche_m2=387.0, voxel_m=0.5, phasen=[],
            quelle="Probe", dauer_s=0.0, strecke_m=0.0, scans=1031,
            scans_phase=0, bag_dauer_s=103.8, bag="probe")
        werte.update(kw)
        return Explorationsgrad(**werte)

    aus: dict = {}
    for name in ("leer", "rechnet", "ohne_daten", "ganzer_flug", 96.8, 59.7, 31.2):
        a = ExplorationsgradAnzeige()
        if name == "leer":
            a.leeren()
        elif name == "rechnet":
            a.rechnet()
        elif name == "ohne_daten":
            a.ohne_daten("Das Bag enthält keine EPIC-Explorationsdaten.")
        elif name == "ganzer_flug":
            a.setze(probe())
        else:
            a.setze(probe(prozent=name, prozent_flaeche=99.0,
                          phasen=[(18.7, 101.7)], dauer_s=83.0))
        aus[str(name)] = {"widgets": baum(a, namen_karte(a), mit_wurzel=True),
                          "klickbar": bool(getattr(a, "_hat_wert", False)),
                          "zeiger": int(a.cursor().shape())}
        a.deleteLater()
    return aus


def teil_f(fenster, namen: dict) -> dict:
    tabs = fenster._tabs
    return {
        "ausrichtfenster": _ausrichtfenster(namen_karte),
        "dialoge": _dialoge(),
        "kachel": _kachel(),
        "tabs": {tabs.tabText(i): baum(tabs.widget(i), namen, mit_wurzel=True)
                 for i in range(tabs.count())},
    }


# -------------------------------------------------------------- Verbindungen

class _Verbindungen(ast.NodeVisitor):
    def __init__(self):
        self.klasse: list = []
        self.funktion: list = []
        self.aus: list = []

    def visit_ClassDef(self, node):
        self.klasse.append(node.name)
        self.generic_visit(node)
        self.klasse.pop()

    def visit_FunctionDef(self, node):
        self.funktion.append(node.name)
        self.generic_visit(node)
        self.funktion.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node):
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr == "connect":
            teile = [ast.unparse(a) for a in node.args]
            teile += [f"{k.arg}={ast.unparse(k.value)}" for k in node.keywords]
            self.aus.append({"klasse": ".".join(self.klasse) or None,
                             "funktion": ".".join(self.funktion) or "<modul>",
                             "signal": ast.unparse(f.value),
                             "slot": ", ".join(teile)})
        self.generic_visit(node)


def teil_h() -> dict:
    w = basis.wurzel()
    dateien: dict = {}
    for ordner, unter, namen in os.walk(os.path.join(w, "ui")):
        unter[:] = sorted(u for u in unter if u != "__pycache__")
        for name in sorted(namen):
            if not name.endswith(".py"):
                continue
            voll = os.path.join(ordner, name)
            with open(voll, encoding="utf-8") as fh:
                besucher = _Verbindungen()
                besucher.visit(ast.parse(fh.read()))
            if besucher.aus:
                dateien[os.path.relpath(voll, w).replace(os.sep, "/")] = besucher.aus
    return {"dateien": dateien}


def h_ohne_datei(teil: dict) -> dict:
    zeilen = sorted(f"{e['funktion']} | {e['signal']} | {e['slot']}"
                    for liste in teil["dateien"].values() for e in liste)
    return {"verbindungen": zeilen}


# ------------------------------------------------------------------- Fenster

def teil_i(fenster) -> dict:
    fremd = ("PyQt5", "sip", "builtins")
    ohne = {"__module__", "__doc__", "__dict__", "__weakref__", "__qualname__",
            "__annotations__", "__firstlineno__", "__static_attributes__"}
    klasse: dict = {}
    for k in reversed(type(fenster).__mro__):
        if k.__module__.split(".")[0] in fremd:
            continue
        for name, wert in vars(k).items():
            if name in ohne:
                continue
            if isinstance(wert, staticmethod):
                art, fn = "staticmethod", wert.__func__
            elif isinstance(wert, classmethod):
                art, fn = "classmethod", wert.__func__
            elif isinstance(wert, property):
                art, fn = "property", None
            elif inspect.isfunction(wert):
                art, fn = "methode", wert
            else:
                art, fn = f"klassenattribut:{type(wert).__name__}", None
            e = {"art": art}
            if fn is not None:
                e["signatur"] = str(inspect.signature(fn))
            klasse[name] = e
    return {"klasse": klasse,
            "attribute": {n: type(v).__name__ for n, v in vars(fenster).items()}}


# ------------------------------------------------------------------ Erfassen

def _timer_einmal(app, fenster) -> None:
    """Laufende periodische QTimer des Fensters anhalten und einmal auslösen.

    Sonst hängt, was ein solcher Timer setzt (etwa die Sperre der RViz-Knöpfe
    700 ms nach dem Bau), an der Rechnerlast: Ein Schnappschuss unter Last sah
    ihn schon, einer ohne Last noch nicht.
    """
    from PyQt5.QtCore import QTimer
    for timer in fenster.findChildren(QTimer):
        if timer.isActive() and not timer.isSingleShot():
            timer.stop()
            timer.timeout.emit()
    app.processEvents()


def erfasse() -> tuple:
    """Alle Teile am frisch gebauten Fenster: (teile, beispiel, zusammenfassung)."""
    fehler = basis.module_laden()
    app, fenster = basis.fenster_bauen()
    _timer_einmal(app, fenster)
    try:
        teile: dict = {}
        teile["i"] = teil_i(fenster)
        namen = namen_karte(fenster)
        teile["a"] = teil_a(fenster, namen)
        teile["e"] = teil_e(fenster)
        teile["d"] = teil_d(fenster)
        teile["f"] = teil_f(fenster, namen)
        teile["b"], beispiel = teil_b(fenster)
        teile["c"] = teil_c(fenster, namen)
        teile["h"] = teil_h()
    finally:
        # Das Beispiel schaltet EDL ein; beim Abbau meldet VTK dafür nicht
        # freigegebene Puffer. Ab hier wird nichts mehr erfasst.
        import vtk
        vtk.vtkObject.GlobalWarningDisplayOff()
        basis.fenster_schliessen(app, fenster)
    teile = {k: _neutral(_json(v)) for k, v in teile.items()}
    return teile, beispiel, zusammenfassung(teile, fehler)


def zusammenfassung(teile: dict, fehler: dict | None = None) -> dict:
    handler = teile["d"]["handler"]
    return {
        "abschnitte": len(teile["a"]["seitenleiste"]["reihenfolge"]),
        "sidebar_breite": teile["a"]["seitenleiste"]["breite"],
        "widgets_seitenleiste": sum(len(a["widgets"]) for a in
                                    teile["a"]["seitenleiste"]["abschnitte"].values()),
        "einstellungsschluessel": len(teile["b"]["faelle"]["start"]),
        "aktionen": teile["e"]["zahl"]["aktionen"],
        "untermenues": teile["e"]["zahl"]["untermenues"],
        "kuerzel": len(_kuerzel(teile["e"]["menues"])),
        "handler_on": sum(n.startswith("_on_") for n in handler),
        "handler_close": "close" in handler,
        "handler_fehlend": sorted(n for n, e in handler.items() if not e["vorhanden"]),
        "zustaende": len(teile["c"]["zustaende"]),
        "verbindungen": sum(len(v) for v in teile["h"]["dateien"].values()),
        "methoden": sum(e["art"] != "property" and not e["art"].startswith("klassen")
                        for e in teile["i"]["klasse"].values()),
        "attribute": len(teile["i"]["attribute"]),
        "nicht_ladbar": sorted(fehler or {}),
    }


def zeige_zusammenfassung(z: dict) -> None:
    print(f"Zusammenfassung: {z['abschnitte']} Abschnitte "
          f"({z['widgets_seitenleiste']} Widgets, Breite {z['sidebar_breite']} px), "
          f"{z['einstellungsschluessel']} Einstellungsschlüssel, "
          f"{z['aktionen']} Aktionen plus {z['untermenues']} Untermenüs, "
          f"{z['kuerzel']} Kürzel, {z['handler_on']} _on_-Handler-Strings"
          f"{' plus ' + repr('close') if z['handler_close'] else ''}"
          f"{' (fehlend: ' + ', '.join(z['handler_fehlend']) + ')' if z['handler_fehlend'] else ''}, "
          f"{z['zustaende']} Zustände, {z['verbindungen']} Verbindungen, "
          f"{z['methoden']} Methoden und {z['attribute']} Attribute am Fenster.")
    for name in z["nicht_ladbar"]:
        print(f"  nicht ladbar: {name}")


# ----------------------------------------------------------------- Schreiben

def _schreibe_json(pfad: str, wert) -> None:
    with open(pfad, "w", encoding="utf-8") as fh:
        json.dump(wert, fh, ensure_ascii=False, indent=1, sort_keys=True)
        fh.write("\n")


def _lies_json(pfad: str):
    with open(pfad, encoding="utf-8") as fh:
        return json.load(fh)


def schreibe(ziel: str, ersetzen: bool) -> int:
    eigene = list(DATEIEN.values()) + [BEISPIEL_DATEI, STAND_DATEI]
    if os.path.exists(ziel) and not ersetzen:
        print(f"Ziel vorhanden ({basis.pfad_neutral(ziel)}) — nur mit --ersetzen.")
        return 2
    teile, beispiel, z = erfasse()
    os.makedirs(ziel, exist_ok=True)
    for name in eigene:
        if os.path.exists(os.path.join(ziel, name)):
            os.remove(os.path.join(ziel, name))
    for k, datei in DATEIEN.items():
        _schreibe_json(os.path.join(ziel, datei), teile[k])
    _schreibe_json(os.path.join(ziel, BEISPIEL_DATEI), beispiel)
    stempel = basis.umgebungsstempel()
    _schreibe_json(os.path.join(ziel, STAND_DATEI),
                   {"zusammenfassung": z,
                    "umgebung": {k: stempel[k] for k in ("python", "PyQt5", "Qt", "VTK")}})
    print(f"Geschrieben: {len(DATEIEN)} Teile, {BEISPIEL_DATEI} und {STAND_DATEI} "
          f"nach {basis.pfad_neutral(ziel)}")
    zeige_zusammenfassung(z)
    return 0


# ---------------------------------------------------------------- Vergleichen

def _passt(voll: str, muster: str) -> bool:
    """Trifft ein Muster aus ``--erwartet`` den Pfad? Nur ``*`` ist Platzhalter."""
    if voll == muster:
        return True
    if "*" not in muster:
        return False
    return re.fullmatch(".*".join(re.escape(t) for t in muster.split("*")),
                        voll, re.DOTALL) is not None



def _marke(e, i: int) -> str:
    if isinstance(e, dict):
        for feld in ("attribut", "schluessel", "unterblock_kopf"):
            if isinstance(e.get(feld), str):
                return f"[{e[feld]}]"
        if "signal" in e:
            return f"[{e.get('funktion')}: {e['signal']}]"
        if isinstance(e.get("text"), str) and e["text"]:
            typ = f"{e['typ']}:" if "typ" in e else ""
            return f"[{typ}{e['text'][:40]}]"
    return f"[{i}]"


def _kurz(wert, laenge: int = 160) -> str:
    text = json.dumps(wert, ensure_ascii=False, sort_keys=True)
    return text if len(text) <= laenge else text[:laenge - 1] + "…"


def unterschiede(vorher, jetzt, pfad: str = "") -> list:
    """Abweichungen als Liste ``(pfad, vorher, jetzt)``.

    Listen werden als Folgen abgeglichen: ein eingefügtes oder entferntes
    Element verschiebt die übrigen nicht.
    """
    if isinstance(vorher, dict) and isinstance(jetzt, dict):
        aus = []
        for k in sorted(set(vorher) | set(jetzt)):
            p = f"{pfad}/{k}" if pfad else str(k)
            if k not in jetzt:
                aus.append((p, vorher[k], _FEHLT))
            elif k not in vorher:
                aus.append((p, _FEHLT, jetzt[k]))
            else:
                aus += unterschiede(vorher[k], jetzt[k], p)
        return aus
    if isinstance(vorher, list) and isinstance(jetzt, list):
        if vorher == jetzt:
            return []
        a = [json.dumps(v, ensure_ascii=False, sort_keys=True) for v in vorher]
        b = [json.dumps(v, ensure_ascii=False, sort_keys=True) for v in jetzt]
        aus = []
        folge = difflib.SequenceMatcher(None, a, b, autojunk=False)
        for art, i1, i2, j1, j2 in folge.get_opcodes():
            if art == "equal":
                continue
            if art == "replace" and i2 - i1 == j2 - j1:
                for d in range(i2 - i1):
                    aus += unterschiede(vorher[i1 + d], jetzt[j1 + d],
                                        pfad + _marke(vorher[i1 + d], i1 + d))
                continue
            for i in range(i1, i2):
                aus.append((pfad + _marke(vorher[i], i), vorher[i], _FEHLT))
            for j in range(j1, j2):
                aus.append((pfad + _marke(jetzt[j], j), _FEHLT, jetzt[j]))
        return aus
    if vorher != jetzt or type(vorher) is not type(jetzt):
        return [(pfad, vorher, jetzt)]
    return []


def _ohne_zuschnitt_a(teil: dict) -> dict:
    """Teil a ohne Zuschnitt: nur die Menge der Widgets der Seitenleiste."""
    aus = dict(teil)
    seite = teil["seitenleiste"]
    menge = []
    for a in seite["abschnitte"].values():
        for e in a["widgets"]:
            menge.append({k: v for k, v in e.items()
                          if k not in ("tiefe", "unterblock")})
    menge.sort(key=lambda e: json.dumps(e, ensure_ascii=False, sort_keys=True))
    aus["seitenleiste"] = {"widgets": menge}
    return aus


def _ohne_zuschnitt_b(teil: dict) -> dict:
    aus = json.loads(json.dumps(teil))
    for fall in aus["faelle"].values():
        if "sections" in fall:
            fall["sections"]["wert"] = "<Zuschnitt>"
    return aus


def vergleiche(ordner: str, nur: list, erlaube: set, erwartet_pfad: str | None,
               ohne_datei: bool) -> int:
    fehlend = [DATEIEN[k] for k in nur if not os.path.isfile(os.path.join(ordner, DATEIEN[k]))]
    if fehlend:
        print(f"Vorher-Stand unvollständig in {basis.pfad_neutral(ordner)}: "
              f"{', '.join(fehlend)} fehlt.")
        return 2
    erwartet: dict = {}
    if erwartet_pfad:
        erwartet = _lies_json(erwartet_pfad)
        if not isinstance(erwartet, dict):
            print("--erwartet: die Datei muss ein Objekt {\"teil:pfad\": grund} enthalten.")
            return 2
    teile, _beispiel, z = erfasse()
    benutzt: set = set()
    n_ab = n_erwartet = 0
    for k in nur:
        vorher = _lies_json(os.path.join(ordner, DATEIEN[k]))
        jetzt = teile[k]
        hinweis = ""
        if k == "h" and ohne_datei:
            vorher, jetzt = h_ohne_datei(vorher), h_ohne_datei(jetzt)
            hinweis = " (ohne Datei und Klasse)"
        if "abschnitte" in erlaube and k == "a":
            b_vor = vorher["seitenleiste"]["breite"]
            b_jetzt = jetzt["seitenleiste"]["breite"]
            alt = vorher["seitenleiste"]["reihenfolge"]
            neu = jetzt["seitenleiste"]["reihenfolge"]
            vorher, jetzt = _ohne_zuschnitt_a(vorher), _ohne_zuschnitt_a(jetzt)
            vorher["seitenleiste"]["breite_reicht"] = True
            jetzt["seitenleiste"]["breite_reicht"] = b_jetzt >= b_vor
            hinweis = (f" (Zuschnitt frei; Abschnitte vorher {', '.join(alt)} — "
                       f"jetzt {', '.join(neu)}; Breite {b_vor} -> {b_jetzt} px)")
        if "abschnitte" in erlaube and k == "b":
            vorher, jetzt = _ohne_zuschnitt_b(vorher), _ohne_zuschnitt_b(jetzt)
            hinweis = " (Wert von sections außer Acht)"
        ab = unterschiede(vorher, jetzt)
        offen = []
        for pfad, v, j in ab:
            voll = f"{k}:{pfad}"
            grund = next((m for m in erwartet if _passt(voll, m)), None)
            if grund is not None:
                benutzt.add(grund)
                n_erwartet += 1
            else:
                offen.append((voll, v, j))
        erw = len(ab) - len(offen)
        if not ab:
            print(f"Teil {k} ({TEILE[k]}): IDENTISCH{hinweis}")
        else:
            print(f"Teil {k} ({TEILE[k]}): {len(offen)} Abweichung(en)"
                  f"{f', {erw} weitere wie erwartet' if erw else ''}{hinweis}")
            for voll, v, j in offen:
                print(f"  {voll}: {_kurz(v)} -> {_kurz(j)}")
        n_ab += len(offen)
    for key in sorted(set(erwartet) - benutzt):
        print(f"Hinweis: erwartete Abweichung '{key}' ist nicht eingetreten.")
    zeige_zusammenfassung(z)
    if n_ab:
        print(f"ABWEICHUNG: {n_ab} in {len(nur)} Teil(en)"
              f"{f', {n_erwartet} weitere wie erwartet' if n_erwartet else ''}.")
        return 1
    if n_erwartet:
        print(f"WIE ERWARTET: {n_erwartet} benannte Abweichung(en), sonst gleich.")
        return 0
    print("IDENTISCH")
    return 0


# ------------------------------------------------------------------- Schema

def schema_fehler(soll) -> list:
    """Verstöße der Soll-Beschreibung gegen das Schema im Kopf der Datei."""
    f: list = []
    if not isinstance(soll, dict):
        return ["die Datei muss ein Objekt enthalten"]
    pflicht = {"abschnitte": list, "leiste": list, "menue": list, "befehle": dict,
               "freigabe_sonder": dict, "entfallen": list, "neu": list,
               "breite_min": int, "annahmen": dict}
    for key, typ in pflicht.items():
        if key not in soll:
            f.append(f"{key}: fehlt")
        elif not isinstance(soll[key], typ) or isinstance(soll[key], bool):
            f.append(f"{key}: erwartet {typ.__name__}")
    for key in sorted(set(soll) - set(pflicht) - {"tooltips"}):
        f.append(f"{key}: unbekannter Schlüssel")
    if f:
        return f

    def text_oder_null(w):
        return w is None or isinstance(w, str)

    gesehen: set = set()
    for i, a in enumerate(soll["abschnitte"]):
        wo = f"abschnitte[{i}]"
        if not isinstance(a, dict):
            f.append(f"{wo}: erwartet Objekt")
            continue
        wo = f"abschnitte[{a.get('schluessel', i)}]"
        if set(a) != {"schluessel", "titel", "offen", "zeilen"}:
            f.append(f"{wo}: Schlüssel müssen schluessel, titel, offen, zeilen sein")
            continue
        if not isinstance(a["schluessel"], str) or not isinstance(a["titel"], str) \
                or not isinstance(a["offen"], bool) or not isinstance(a["zeilen"], list):
            f.append(f"{wo}: falscher Typ in schluessel, titel, offen oder zeilen")
            continue
        if a["schluessel"] in gesehen:
            f.append(f"{wo}: Schlüssel doppelt")
        gesehen.add(a["schluessel"])
        for n, zeile in enumerate(a["zeilen"]):
            if not isinstance(zeile, dict) or set(zeile) != {"attribut", "text", "unterblock"} \
                    or not all(text_oder_null(zeile[k]) for k in zeile):
                f.append(f"{wo}/zeilen[{n}]: erwartet attribut, text, unterblock "
                         "(je Text oder null)")
            elif zeile["attribut"] is None and zeile["text"] is None:
                f.append(f"{wo}/zeilen[{n}]: weder attribut noch text")
    for i, e in enumerate(soll["leiste"]):
        if not isinstance(e, dict):
            f.append(f"leiste[{i}]: erwartet Objekt")
    in_menue: set = set()
    for i, e in enumerate(soll["menue"]):
        wo = f"menue[{i}]"
        if not isinstance(e, dict) or not {"pfad", "schluessel", "text", "kuerzel",
                                           "schaltbar"} <= set(e):
            f.append(f"{wo}: erwartet pfad, schluessel, text, kuerzel, schaltbar")
            continue
        if set(e) - {"pfad", "schluessel", "text", "kuerzel", "schaltbar",
                     "tooltip", "statustip"}:
            f.append(f"{wo}: unbekannte Schlüssel")
        if not isinstance(e["pfad"], str) or not e["pfad"] \
                or not text_oder_null(e["schluessel"]) or not isinstance(e["text"], str) \
                or not isinstance(e["kuerzel"], str) or not isinstance(e["schaltbar"], bool):
            f.append(f"{wo}: falscher Typ")
            continue
        if e["schluessel"] is not None:
            if e["schluessel"] in in_menue:
                f.append(f"{wo}: Schlüssel {e['schluessel']} doppelt")
            in_menue.add(e["schluessel"])
    for key, b in soll["befehle"].items():
        wo = f"befehle[{key}]"
        if not isinstance(b, dict) or set(b) != {"braucht", "frei_bei_busy", "tooltip"}:
            f.append(f"{wo}: erwartet braucht, frei_bei_busy, tooltip")
            continue
        if b["braucht"] is not None and (
                not isinstance(b["braucht"], list)
                or any(z not in ZUSTANDSNAMEN for z in b["braucht"])):
            f.append(f"{wo}: braucht ist null oder eine Liste aus "
                     f"{', '.join(ZUSTANDSNAMEN)}")
        if not isinstance(b["frei_bei_busy"], bool) or not isinstance(b["tooltip"], str):
            f.append(f"{wo}: falscher Typ in frei_bei_busy oder tooltip")
    befehle_im_menue = {k for k in in_menue if not k.startswith("menu_")}
    for key in sorted(befehle_im_menue - set(soll["befehle"])):
        f.append(f"menue: Schlüssel {key} steht nicht in befehle")
    for key in sorted(set(soll["befehle"]) - befehle_im_menue):
        f.append(f"befehle[{key}]: kommt im Menü nicht vor")
    for key in SONDER:
        if not isinstance(soll["freigabe_sonder"].get(key), dict):
            f.append(f"freigabe_sonder/{key}: fehlt oder kein Objekt")
    for key in sorted(set(soll["freigabe_sonder"]) - set(SONDER)):
        f.append(f"freigabe_sonder/{key}: unbekannter Block")
    for i, e in enumerate(soll["entfallen"]):
        if not isinstance(e, str) or not e.startswith("_"):
            f.append(f"entfallen[{i}]: erwartet einen Attributnamen")
    for i, e in enumerate(soll["neu"]):
        if isinstance(e, str) and e.startswith("_"):
            continue
        if not (isinstance(e, dict) and isinstance(e.get("text"), str)):
            f.append(f"neu[{i}]: erwartet einen Attributnamen oder {{\"text\": …}}")
    if soll["breite_min"] <= 0:
        f.append("breite_min: muss größer als 0 sein")
    if "tooltips" in soll:
        liste = soll["tooltips"].get("aenderungen") if isinstance(soll["tooltips"], dict) \
            else None
        if not isinstance(liste, list):
            f.append("tooltips: erwartet {\"aenderungen\": […]}")
        else:
            for i, e in enumerate(liste):
                if not isinstance(e, dict) or not isinstance(e.get("vorher"), str) \
                        or not text_oder_null(e.get("nachher")) \
                        or not isinstance(e.get("grund"), str) or not e.get("grund"):
                    f.append(f"tooltips/aenderungen[{i}]: erwartet vorher, nachher, grund")
    return f


def pruefe_schema(pfad: str) -> int:
    try:
        soll = _lies_json(pfad)
    except (OSError, ValueError) as exc:
        print(f"{pfad}: nicht lesbar ({exc})")
        return 2
    fehler = schema_fehler(soll)
    if fehler:
        print(f"Schema verletzt ({len(fehler)}):")
        for zeile in fehler:
            print(f"  {zeile}")
        return 1
    print("Schema gültig.")
    print("Abschnittsfolge: " + ", ".join(a["schluessel"] for a in soll["abschnitte"]))
    zeilen = sum(len(a["zeilen"]) for a in soll["abschnitte"])
    print(f"{len(soll['abschnitte'])} Abschnitte mit {zeilen} Zeilen, "
          f"{len(soll['leiste'])} Einträge der Leiste, {len(soll['menue'])} "
          f"Menüeinträge, {len(soll['befehle'])} Befehle, "
          f"{len(soll['entfallen'])} entfallen, {len(soll['neu'])} neu, "
          f"breite_min {soll['breite_min']}.")
    return 0


# ------------------------------------------------------------------ Soll-Ist

def _teilmenge(soll, ist, pfad: str) -> list:
    """Was in ``soll`` genannt ist, muss in ``ist`` so stehen."""
    if isinstance(soll, dict):
        if not isinstance(ist, dict):
            return [f"{pfad}: Soll {_kurz(soll)}, Ist {_kurz(ist)}"]
        aus = []
        for k, v in soll.items():
            if k not in ist:
                if v not in (None, "", False):
                    aus.append(f"{pfad}/{k}: Soll {_kurz(v)}, im Ist nicht vorhanden")
            else:
                aus += _teilmenge(v, ist[k], f"{pfad}/{k}")
        return aus
    if soll != ist:
        return [f"{pfad}: Soll {_kurz(soll)}, Ist {_kurz(ist)}"]
    return []


def _kandidaten(widgets: list) -> list:
    """Zeilen eines Abschnitts im Ist: (attribut, auch, text, unterblock)."""
    aus = []
    for e in widgets:
        if e.get("teil_eines_reglers") or e.get("zeilenkopf"):
            continue
        text = e.get("beschriftung") if e.get("beschriftung") is not None else e.get("text")
        if e["attribut"] is None and not text:
            continue
        aus.append({"attribut": e["attribut"], "auch": e.get("auch", []),
                    "text": text, "eigen": e.get("text"),
                    "unterblock": e.get("unterblock")})
    return aus


def _soll_abschnitt(a: dict, ist: dict | None, sections: dict) -> list:
    key = a["schluessel"]
    if ist is None:
        return [f"{key}: Abschnitt fehlt im Ist"]
    aus = []
    if key not in sections:
        aus.append(f"{key}: fehlt in _collect_settings()['sections']")
    for block in sorted({z["unterblock"] for z in a["zeilen"] if z["unterblock"]}):
        if block not in sections:
            aus.append(f"{key}: Unterblock {block} fehlt in "
                       "_collect_settings()['sections']")
    kand = _kandidaten(ist["widgets"])
    stelle = -1
    genannt: set = set()
    for n, z in enumerate(a["zeilen"]):
        wo = f"{key}/zeilen[{n}] {z['attribut'] or z['text']!r}"
        if z["attribut"] is not None:
            i = next((i for i, k in enumerate(kand)
                      if z["attribut"] == k["attribut"] or z["attribut"] in k["auch"]),
                     None)
            if i is None:
                aus.append(f"{wo}: Attribut fehlt im Abschnitt")
                continue
            genannt.add(i)
            if z["text"] is not None and z["text"] not in (kand[i]["text"], kand[i]["eigen"]):
                aus.append(f"{wo}: Text Soll {z['text']!r}, Ist {kand[i]['text']!r}")
        else:
            i = next((i for i, k in enumerate(kand)
                      if i > stelle and z["text"] in (k["text"], k["eigen"])), None)
            if i is None:
                aus.append(f"{wo}: kein Widget mit diesem Text an dieser Stelle")
                continue
            genannt.add(i)
        if kand[i]["unterblock"] != z["unterblock"]:
            aus.append(f"{wo}: Unterblock Soll {z['unterblock']!r}, "
                       f"Ist {kand[i]['unterblock']!r}")
        if i <= stelle:
            aus.append(f"{wo}: steht im Ist vor der vorigen Soll-Zeile")
        stelle = max(stelle, i)
    for i, k in enumerate(kand):
        if k["attribut"] is not None and i not in genannt:
            aus.append(f"{key}: {k['attribut']} ({k['text']!r}) steht im Abschnitt, "
                       "aber nicht im Soll")
    return aus


def _tooltips(teile: dict) -> set:
    """Alle Tooltip-Texte der Teile a, c, e und f."""
    aus: set = set()

    def geh(w, in_tooltips=False):
        if isinstance(w, dict):
            for k, v in w.items():
                if k in ("tooltip", "statustip") and isinstance(v, str):
                    aus.add(v)
                else:
                    geh(v, in_tooltips or k == "tooltips")
        elif isinstance(w, list):
            for v in w:
                geh(v, in_tooltips)
        elif in_tooltips and isinstance(w, str):
            aus.add(w)

    for k in ("a", "c", "e", "f"):
        geh(teile.get(k, {}))
    aus.discard("")
    return aus


def _sollfrei(befehl: dict, gilt: set) -> bool:
    if "busy" in gilt and not befehl["frei_bei_busy"]:
        return False
    return all(z in gilt for z in befehl["braucht"])


def soll_ist(pfad: str, abschnitt: str | None, nur: list, vorher_ordner: str) -> int:
    try:
        soll = _lies_json(pfad)
    except (OSError, ValueError) as exc:
        print(f"{pfad}: nicht lesbar ({exc})")
        return 2
    fehler = schema_fehler(soll)
    if fehler:
        print(f"Schema verletzt ({len(fehler)}):")
        for zeile in fehler:
            print(f"  {zeile}")
        return 2
    if abschnitt and abschnitt not in [a["schluessel"] for a in soll["abschnitte"]]:
        print(f"--abschnitt {abschnitt}: steht nicht in der Soll-Beschreibung.")
        return 2
    bereiche = list(nur) if nur else (["abschnitte"] if abschnitt else list(SOLL_BEREICHE))
    teile, _beispiel, z = erfasse()
    seite = teile["a"]["seitenleiste"]
    sections = teile["b"]["faelle"]["start"].get("sections", {}).get("wert", {})
    ergebnis: dict = {}

    def breite() -> list:
        if seite["breite"] < soll["breite_min"]:
            return [f"breite: {seite['breite']} px, Soll mindestens {soll['breite_min']}"]
        return []

    if "abschnittsliste" in bereiche:
        ab = []
        soll_folge = [a["schluessel"] for a in soll["abschnitte"]]
        if soll_folge != seite["reihenfolge"]:
            ab.append(f"Folge: Soll {', '.join(soll_folge)} — "
                      f"Ist {', '.join(seite['reihenfolge'])}")
        for a in soll["abschnitte"]:
            ist = seite["abschnitte"].get(a["schluessel"])
            if ist is None:
                continue
            for feld in ("titel", "offen"):
                if ist[feld] != a[feld]:
                    ab.append(f"{a['schluessel']}/{feld}: Soll {a[feld]!r}, "
                              f"Ist {ist[feld]!r}")
            if a["schluessel"] not in sections:
                ab.append(f"{a['schluessel']}: fehlt in _collect_settings()['sections']")
        ergebnis["abschnittsliste"] = ab + breite()
    if "abschnitte" in bereiche:
        ab = []
        for a in soll["abschnitte"]:
            if abschnitt and a["schluessel"] != abschnitt:
                continue
            ab += _soll_abschnitt(a, seite["abschnitte"].get(a["schluessel"]), sections)
        if abschnitt:
            ergebnis[f"abschnitt {abschnitt}"] = ab + breite()
        else:
            ergebnis["abschnitte"] = ab
    if "leiste" in bereiche:
        ab = []
        ist = teile["a"]["leiste"]
        if len(ist) != len(soll["leiste"]):
            ab.append(f"leiste: Soll {len(soll['leiste'])} Einträge, Ist {len(ist)}")
        for i, (s, e) in enumerate(zip(soll["leiste"], ist)):
            ab += _teilmenge(s, e, f"leiste{_marke(s, i)}")
        ergebnis["leiste"] = ab
    if "menue" in bereiche:
        ab = []
        if teile["e"]["native_menueleiste"]:
            ab.append("menue: nativeMenuBar ist nicht abgeschaltet")
        ist = menue_flach(teile["e"]["menues"])
        felder = ("pfad", "schluessel", "text", "kuerzel", "schaltbar")
        a = [json.dumps([e[k] for k in felder], ensure_ascii=False) for e in soll["menue"]]
        b = [json.dumps([e[k] for k in felder], ensure_ascii=False) for e in ist]
        for art, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b,
                                                           autojunk=False).get_opcodes():
            if art == "equal":
                for d in range(i2 - i1):
                    s, e = soll["menue"][i1 + d], ist[j1 + d]
                    for feld in ("tooltip", "statustip"):
                        if feld in s and s[feld] != e[feld]:
                            ab.append(f"menue[{s['schluessel']}]/{feld}: Soll "
                                      f"{s[feld]!r}, Ist {e[feld]!r}")
                continue
            for i in range(i1, i2):
                ab.append(f"menue: im Ist fehlt {a[i]}")
            for j in range(j1, j2):
                ab.append(f"menue: im Soll fehlt {b[j]}")
        kuerzel = sorted(_kuerzel(teile["e"]["menues"]))
        soll_kuerzel = sorted(e["kuerzel"] for e in soll["menue"] if e["kuerzel"])
        if kuerzel != soll_kuerzel:
            ab.append(f"menue: Kürzel Soll {soll_kuerzel}, Ist {kuerzel}")
        ergebnis["menue"] = ab
    flach = {e["schluessel"]: e for e in menue_flach(teile["e"]["menues"])
             if e["schluessel"]}
    if "befehle" in bereiche:
        ab = []
        ist_keys = {k for k, e in teile["c"]["zustaende"]["leer"]["aktionen"].items()
                    if not k.startswith("menu_")}
        for key in sorted(set(soll["befehle"]) - ist_keys):
            ab.append(f"befehle[{key}]: Aktion fehlt im Ist")
        for key in sorted(ist_keys - set(soll["befehle"])):
            ab.append(f"befehle[{key}]: Aktion steht nicht im Soll")
        for key in sorted(set(soll["befehle"]) & ist_keys):
            e = flach.get(key)
            if e is None:
                ab.append(f"befehle[{key}]: nicht in der Menüleiste")
            elif e["statustip"] != soll["befehle"][key]["tooltip"]:
                ab.append(f"befehle[{key}]/tooltip: Soll "
                          f"{soll['befehle'][key]['tooltip']!r}, Ist {e['statustip']!r}")
        # ein Befehl, dessen Handler es am Fenster nicht gibt, tut nichts
        for name in z["handler_fehlend"]:
            ab.append(f"befehle: Handler {name} aus ui/menubar.py fehlt am Fenster")
        ergebnis["befehle"] = ab
    if "freigabe" in bereiche:
        ab = []
        for name, stand in teile["c"]["zustaende"].items():
            gilt = set(stand["gilt"])
            for key, b in sorted(soll["befehle"].items()):
                if b["braucht"] is None or key not in stand["aktionen"]:
                    continue
                if stand["aktionen"][key] != _sollfrei(b, gilt):
                    ab.append(f"freigabe/{name}/{key}: Soll "
                              f"{'frei' if _sollfrei(b, gilt) else 'gesperrt'} (braucht "
                              f"{', '.join(b['braucht']) or 'nichts'}), Ist "
                              f"{'frei' if stand['aktionen'][key] else 'gesperrt'}")
        for block in SONDER:
            ab += _teilmenge(soll["freigabe_sonder"][block], teile["c"].get(block),
                             f"freigabe_sonder/{block}")
        ergebnis["freigabe"] = ab
    if "tooltips" in bereiche:
        ab = []
        dateien = {k: os.path.join(vorher_ordner, DATEIEN[k]) for k in ("a", "c", "e", "f")}
        if not all(os.path.isfile(p) for p in dateien.values()):
            ab.append(f"tooltips: Vorher-Stand fehlt in {basis.pfad_neutral(vorher_ordner)}")
        else:
            vorher = _tooltips({k: _lies_json(p) for k, p in dateien.items()})
            ist = _tooltips(teile)
            liste = (soll.get("tooltips") or {}).get("aenderungen", [])
            benannt = {e["vorher"]: e for e in liste}
            for text in sorted(vorher - ist):
                if text not in benannt:
                    ab.append(f"tooltips: fehlt ohne benannte Änderung: {text!r}")
            for e in liste:
                if e["nachher"] is not None and e["nachher"] not in ist:
                    ab.append(f"tooltips: geänderter Text kommt im Ist nicht vor: "
                              f"{e['nachher']!r} ({e['grund']})")
                if e["vorher"] not in vorher:
                    ab.append(f"tooltips: benannte Änderung ohne Vorher-Text: "
                              f"{e['vorher']!r}")
        ergebnis["tooltips"] = ab
    if "attribute" in bereiche:
        ab = []
        attribute = teile["i"]["attribute"]
        texte = {t for a in seite["abschnitte"].values() for k in _kandidaten(a["widgets"])
                 for t in (k["text"], k["eigen"]) if t}
        for name in soll["entfallen"]:
            if name in attribute:
                ab.append(f"entfallen: {name} gibt es noch")
        for e in soll["neu"]:
            if isinstance(e, str) and e not in attribute:
                ab.append(f"neu: {e} fehlt")
            elif isinstance(e, dict) and e["text"] not in texte:
                ab.append(f"neu: kein Widget der Seitenleiste mit dem Text {e['text']!r}")
        ergebnis["attribute"] = ab

    n = 0
    for name, ab in ergebnis.items():
        ab = list(dict.fromkeys(ab))
        n += len(ab)
        print(f"{name}: {'bestanden' if not ab else f'{len(ab)} Abweichung(en)'}")
        for zeile in ab:
            print(f"  {zeile}")
    zeige_zusammenfassung(z)
    print("SOLL-IST: bestanden" if n == 0 else f"SOLL-IST: {n} Abweichung(en)")
    return 0 if n == 0 else 1


# ---------------------------------------------------------------------- main

def _liste(text: str | None, erlaubt, was: str) -> list:
    if not text:
        return []
    werte = [t.strip() for t in text.split(",") if t.strip()]
    fremd = [w for w in werte if w not in erlaubt]
    if fremd:
        print(f"{was}: unbekannt {', '.join(fremd)} (möglich: {', '.join(erlaubt)})")
        raise SystemExit(2)
    return werte


def main(argv=None) -> int:
    parser = basis.argumente(argparse.ArgumentParser(
        description=__doc__.splitlines()[0]))
    parser.add_argument("--schreibe", metavar="DIR", default=None)
    parser.add_argument("--ersetzen", action="store_true")
    parser.add_argument("--vergleiche", metavar="DIR", nargs="?", const=VORGABE_ZIEL,
                        default=None)
    parser.add_argument("--teile", default=None, metavar="a,b,…")
    parser.add_argument("--erlaube", default=None, metavar="abschnitte")
    parser.add_argument("--erwartet", default=None, metavar="JSON")
    parser.add_argument("--h-ohne-datei", action="store_true")
    parser.add_argument("--soll", metavar="JSON", default=None)
    parser.add_argument("--abschnitt", default=None, metavar="K")
    parser.add_argument("--nur", default=None,
                        metavar="abschnittsliste|menue|befehle|freigabe|tooltips")
    parser.add_argument("--vorher", metavar="DIR", default=VORGABE_ZIEL,
                        help="Vorher-Stand für --soll (Tooltips)")
    parser.add_argument("--pruefe-schema", metavar="JSON", default=None)
    args = parser.parse_args(argv)
    modi = [m for m in (args.schreibe, args.vergleiche, args.soll, args.pruefe_schema)
            if m is not None]
    if len(modi) != 1:
        parser.print_help()
        return 2
    # relative Pfade gelten ab dem Aufrufort; wurzel_setzen wechselt den Ordner
    for name in ("schreibe", "vergleiche", "soll", "pruefe_schema", "erwartet",
                 "vorher", "wurzel"):
        if getattr(args, name):
            setattr(args, name, os.path.abspath(getattr(args, name)))
    if args.pruefe_schema:
        return pruefe_schema(args.pruefe_schema)
    teile = _liste(args.teile, tuple(TEILE), "--teile") or list(TEILE)
    erlaube = set(_liste(args.erlaube, ("abschnitte",), "--erlaube"))
    nur = _liste(args.nur, SOLL_BEREICHE, "--nur")
    basis.wurzel_setzen(args.wurzel)
    if args.schreibe:
        return schreibe(args.schreibe, args.ersetzen)
    if args.vergleiche:
        return vergleiche(args.vergleiche, teile, erlaube, args.erwartet,
                          args.h_ohne_datei)
    return soll_ist(args.soll, args.abschnitt, nur, args.vorher)


if __name__ == "__main__":
    sys.exit(main())
