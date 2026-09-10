"""Menueleiste von Super360 Studio.

Die Leiste haelt die Befehle, die Seitenleiste die Einstellungen. Wer nur
oeffnen, rechnen und exportieren will, kommt allein ueber das Menue durch den
Ablauf; wer an Schwellen dreht, klappt rechts den passenden Abschnitt auf.

:func:`build` haengt die Menues an ein Fenster und gibt alle Aktionen als dict
zurueck, damit das Fenster sie je nach Zustand scharf oder stumpf schalten kann
(s. ``MainWindow._update_enabled``). Fehlt ein Handler am Fenster, bleibt die
Aktion dauerhaft stumpf statt beim Klick zu scheitern — so laesst sich die
Leiste unabhaengig von den Bausteinen dahinter erweitern.
"""

from __future__ import annotations

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence

# (Schluessel, Beschriftung, Handler am Fenster, Kuerzel, Tooltip)
_DATEI = (
    ("open", "Rosbag öffnen …", "_on_open_clicked", "Ctrl+O",
     "Einen Flug öffnen; er wird zur Arbeitswolke."),
    ("open_project", "Berechnetes Projekt öffnen …", "_on_open_project", "Ctrl+Shift+P",
     "Ein Projekt aus dem Cache — der einzige Weg zu zusammengeführten Karten."),
    ("merge", "Zweiten Flug dazuladen …", "_on_merge_pick", "Ctrl+Shift+O",
     "Zweites Rosbag laden und mit dem offenen zusammenführen."),
    ("meander", "Mäanderflug laden …", "_on_meander_pick", "",
     "Bilder eines DJI-Kartierungsfluges für die Einfärbung laden."),
    (None, None, None, None, None),
    ("project_export", "Projekt exportieren …", "_on_export_project", "Ctrl+Shift+E",
     "Alle berechneten Daten in einen Ordner kopieren, den man weiterreichen kann."),
    ("project_import", "Projekt importieren …", "_on_import_project", "Ctrl+I",
     "Ein exportiertes Projekt aus seinem Ordner wieder öffnen."),
    (None, None, None, None, None),
    ("export_ply", "Exportieren als PLY/PCD …", "_on_export_plypcd", "Ctrl+E",
     "Die angezeigte Farbebene mitschreiben."),
    ("export_las", "Exportieren als LAS …", "_on_export_las", "",
     "Georeferenziert, sofern das GPS brauchbar war."),
    ("screenshot", "Ansicht als Bild speichern …", "_on_screenshot", "Ctrl+P", ""),
    (None, None, None, None, None),
    ("quit", "Beenden", "close", "Ctrl+Q", ""),
)

_AENDERN = (
    ("autocal", "Auto-Kalibrierung (grob)", "_on_autocal_clicked", "",
     "Sucht die Kamera-Extrinsik über die Foto-Konsistenz."),
    ("overlay", "Überlagerung prüfen", "_on_overlay_clicked", "",
     "Pano mit projizierten Lidar-Punkten zur Sichtprüfung."),
    ("extrinsic_reset", "Extrinsik zurücksetzen", "_on_extrinsic_reset", "", ""),
    (None, None, None, None, None),
    ("merge_reset", "Ausrichtung verwerfen", "_on_merge_discard", "",
     "Den dazugeladenen zweiten Flug wieder entfernen."),
    (None, None, None, None, None),
    ("settings_reset", "Einstellungen auf Vorgabe", "_on_settings_reset", "", ""),
)

_ANSICHT_ENDE = (
    ("reset_cam", "Kamera zurücksetzen", "_on_reset_camera", "R", ""),
    ("cut_reset", "Höhenschnitt aufheben", "_on_cut_reset", "Ctrl+H",
     "Die volle Höhe wieder zeigen."),
    (None, None, None, None, None),
    ("sidebar", "Seitenleiste", "_on_toggle_sidebar", "Ctrl+B", ""),
    ("expand_all", "Alle Abschnitte aufklappen", "_on_expand_all", "", ""),
    ("collapse_all", "Alle Abschnitte zuklappen", "_on_collapse_all", "", ""),
)

_WERKZEUGE = (
    ("fastlio", "Karte berechnen (FAST-LIO2)", "_on_fastlio_clicked", "F5", ""),
    ("colorize", "Einfärben (360°-Kamera)", "_on_colorize_clicked", "F6", ""),
    ("meander_run", "Mäander-Einfärbung …", "_on_meander_run", "F7",
     "Punktwolke aus den Bildern eines Kartierungsfluges einfärben."),
    ("merge_apply", "Flüge zusammenführen", "_on_merge_apply", "", ""),
    (None, None, None, None, None),
    ("measure", "Messen", "_on_toggle_measure", "M",
     "Zwei Klicks in die Wolke: Abstand in Metern."),
    (None, None, None, None, None),
    ("rviz_start", "RViz-Wiedergabe starten", "_on_rviz_start", "", ""),
    ("rviz_stop", "RViz-Wiedergabe beenden", "_on_rviz_stop", "", ""),
)


def _add(menu: QtWidgets.QMenu, win, eintraege, actions: dict,
         checkable: set | None = None) -> None:
    checkable = checkable or set()
    for key, label, handler, kuerzel, tip in eintraege:
        if key is None:
            menu.addSeparator()
            continue
        act = QtWidgets.QAction(label, win)
        if kuerzel:
            act.setShortcut(QKeySequence(kuerzel))
            act.setShortcutContext(Qt.ApplicationShortcut)
        if tip:
            act.setToolTip(tip)
            act.setStatusTip(tip)
        if key in checkable:
            act.setCheckable(True)
        fn = getattr(win, handler, None)
        if callable(fn):
            act.triggered.connect(lambda _checked=False, f=fn: f())
        else:
            act.setEnabled(False)  # Baustein noch nicht da
            act.setToolTip((tip + "  ") if tip else "" + "(noch nicht verfügbar)")
        menu.addAction(act)
        actions[key] = act


def build(win) -> dict:
    """Menueleiste an ``win`` haengen; gibt {Schluessel: QAction} zurueck."""
    bar = win.menuBar()
    bar.setNativeMenuBar(False)   # unter Wayland/GNOME sonst unsichtbar
    actions: dict = {}

    _add(bar.addMenu("&Datei"), win, _DATEI, actions)
    _add(bar.addMenu("&Ändern"), win, _AENDERN, actions)

    ansicht = bar.addMenu("&Ansicht")
    actions["menu_farbquelle"] = ansicht.addMenu("Farbquelle")
    actions["menu_hintergrund"] = ansicht.addMenu("Hintergrund")
    actions["menu_tab"] = ansicht.addMenu("Bereich")
    edl = QtWidgets.QAction("Eye-Dome-Beleuchtung", win)
    edl.setCheckable(True)
    fn = getattr(win, "_on_menu_edl", None)
    if callable(fn):
        edl.toggled.connect(fn)
    else:
        edl.setEnabled(False)
    ansicht.addAction(edl)
    actions["edl"] = edl
    ansicht.addSeparator()
    _add(ansicht, win, _ANSICHT_ENDE, actions,
         checkable={"sidebar", "measure"})

    _add(bar.addMenu("&Werkzeuge"), win, _WERKZEUGE, actions,
         checkable={"measure"})

    hilfe = bar.addMenu("&Hilfe")
    ueber = QtWidgets.QAction("Über Super360 Studio", win)
    fn = getattr(win, "_on_about", None)
    if callable(fn):
        ueber.triggered.connect(lambda _c=False, f=fn: f())
    else:
        ueber.setEnabled(False)
    hilfe.addAction(ueber)
    actions["about"] = ueber
    return actions


def fill_radio_menu(menu: QtWidgets.QMenu, win, eintraege, handler,
                    aktuell=None) -> dict:
    """Untermenue als Auswahlgruppe fuellen (Farbquelle, Hintergrund, Bereich)."""
    menu.clear()
    gruppe = QtWidgets.QActionGroup(win)
    gruppe.setExclusive(True)
    out: dict = {}
    for data, label in eintraege:
        act = QtWidgets.QAction(label, win)
        act.setCheckable(True)
        act.setChecked(data == aktuell)
        act.setData(data)
        gruppe.addAction(act)
        menu.addAction(act)
        act.triggered.connect(lambda _c=False, d=data: handler(d))
        out[data] = act
    return out
