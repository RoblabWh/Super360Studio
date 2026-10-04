"""Menueleiste von Super360 Studio.

Die Leiste haelt die Befehle, die Seitenleiste die Einstellungen. Wer nur
oeffnen, rechnen und exportieren will, kommt allein ueber das Menue durch den
Ablauf; wer an Schwellen dreht, klappt rechts den passenden Abschnitt auf.

:func:`build` haengt die Menues an ein Fenster und gibt alle Aktionen als dict
zurueck, damit das Fenster sie je nach Zustand scharf oder stumpf schalten kann
(s. ``MainWindow._update_enabled``). Fehlt ein Handler am Fenster, bleibt die
Aktion dauerhaft stumpf statt beim Klick zu scheitern — so laesst sich die
Leiste unabhaengig von den Bausteinen dahinter erweitern.

:data:`BEFEHLE` beschreibt jeden Befehl an einer Stelle: Text im Menue und am
Knopf der Seitenleiste, Handler, Kuerzel, Tooltip und die Voraussetzungen, die
ihn freigeben. :func:`baue` erzeugt daraus Menues und Aktionen und scheitert
laut, wenn dem Fenster ein Handler fehlt; :func:`schalte` gibt die Aktionen je
Zustand frei und nennt im Tooltip, was fehlt.
"""

from __future__ import annotations

from typing import NamedTuple

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
    ("project_import", "Projektordner öffnen …", "_on_import_project", "Ctrl+I",
     "Ein exportiertes Projekt direkt in seinem Ordner öffnen — ohne Kopie,\n"
     "ohne Rückfrage. Änderungen werden sofort dort gespeichert."),
    (None, None, None, None, None),
    ("export_ply", "Exportieren als PLY/PCD …", "_on_export_plypcd", "Ctrl+E",
     "Die angezeigte Farbebene mitschreiben."),
    ("export_las", "Exportieren als LAS …", "_on_export_las", "",
     "Georeferenziert, sofern das GPS brauchbar war."),
    ("mesh_cc", "Mesh in CloudCompare zeigen", "_on_mesh_cloudcompare", "",
     "Dreiecksnetz aus der Wolke erzeugen (Einstellungen im Abschnitt Export)\n"
     "und in CloudCompare öffnen."),
    ("cloud_cc", "Punktwolke in CloudCompare öffnen", "_on_cloud_cloudcompare", "", ""),
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
    ("preview", "Zweiten Flug (orange) anzeigen", "_on_toggle_preview", "",
     "Die Vorschau des dazugeladenen Fluges. Orange heißt: noch nicht\n"
     "übernommen — sie gehört noch nicht zur Karte."),
    (None, None, None, None, None),
    ("sidebar", "Seitenleiste", "_on_toggle_sidebar", "Ctrl+B", ""),
    ("expand_all", "Alle Abschnitte aufklappen", "_on_expand_all", "", ""),
    ("collapse_all", "Alle Abschnitte zuklappen", "_on_collapse_all", "", ""),
)

_WERKZEUGE = (
    ("fastlio", "Karte berechnen (FAST-LIO2)", "_on_fastlio_clicked", "F5", ""),
    ("exploration", "Explorationsgrad neu berechnen", "_on_exploration_neu", "",
     "Den Wert oben rechts noch einmal aus dem Bag rechnen, am Cache vorbei."),
    ("colorize", "Einfärben (360°-Kamera)", "_on_colorize_clicked", "F6", ""),
    ("meander_run", "Mäander-Einfärbung …", "_on_meander_run", "F7",
     "Punktwolke aus den Bildern eines Kartierungsfluges einfärben."),
    ("splat_meander", "Mäander per Gaussian Splat", "_on_splat_maeander", "",
     "Farben aus allen Bildern des Mäanderfluges zugleich lernen (GPU)."),
    ("splat_onboard", "Onboard per Gaussian Splat", "_on_splat_onboard", "",
     "Farben aus den Frames der 360°-Kamera zugleich lernen (GPU)."),
    ("splat_gemeinsam", "Onboard + Mäander per Gaussian Splat", "_on_splat_gemeinsam", "",
     "Ein Splat aus beiden Flügen, Farbe im Mäander, Farbmatrix je Onboard-Bild (GPU)."),
    ("fusion", "Onboard + Mäander fusionieren", "_on_fusion", "",
     "Onboard-Farben an den Mäander angleichen und nach Flächenlage mischen:\n"
     "Dächer und Boden aus dem Mäander, Fassaden aus der 360°-Kamera."),
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
         checkable={"sidebar", "preview"})

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


# ------------------------------------------------------------ Befehlstabelle

#: Voraussetzungen, die ``braucht`` nennen darf, und der Satz, den der
#: Tooltip anhaengt, solange sie fehlt
GRUENDE = {
    "bag": "Erst ein Rosbag öffnen.",
    "rec": "Erst die Karte berechnen oder ein Projekt öffnen.",
    "world": "Erst die Karte berechnen oder ein Projekt öffnen.",
    "calib": "Keine Kamerakalibrierung gefunden.",
    "flug": "Erst einen Mäanderflug wählen.",
    "lage": "Erst „Ausrichten“ laufen lassen.",
    "thermal": "Keine Thermalbilder in diesem Lauf.",
    "zweitflug": "Erst einen zweiten Flug laden.",
    "fusion": "Braucht eine Farbebene der 360°-Kamera und eine des Mäanderfluges.",
}
GRUND_BESCHAEFTIGT = "Gesperrt, solange ein Schritt läuft."


class Menue(NamedTuple):
    """Ein Menue der Leiste (Pfad mit einem Titel) oder ein Untermenue.

    Ein Untermenue mit ``schluessel`` fuellt das Fenster selbst (Farbe,
    Hintergrund, Bereich); es steht unter diesem Schluessel in ``actions``.
    """
    pfad: tuple
    schluessel: str | None = None
    trenner: bool = False   # Trennstrich davor


class Befehl(NamedTuple):
    """Ein Befehl: Menueeintrag und, wo es ihn gibt, Knopf der Seitenleiste.

    ``braucht``: None = wird hier nie geschaltet (bleibt frei oder hat eine
    eigene Freigabe), () = nur waehrend eines laufenden Schritts gesperrt,
    sonst die Namen aus :data:`GRUENDE`, die alle gelten muessen.
    ``signal``: "triggered" ruft den Handler ohne Argument, "toggled" mit
    dem neuen Hakenzustand. ``knopf``: kuerzerer Text am Knopf, sonst ``text``.
    """
    schluessel: str
    text: str
    handler: str
    menue: tuple
    kuerzel: str = ""
    tooltip: str = ""
    braucht: tuple | None = ()
    frei_bei_busy: bool = False
    schaltbar: bool = False
    signal: str = "triggered"
    knopf: str | None = None
    trenner: bool = False   # Trennstrich davor


_M_DATEI = ("&Datei",)
_M_ABLAUF = ("A&blauf",)
_M_KALIB = _M_ABLAUF + ("Kamera-Kalibrierung",)
_M_ZUSAMMEN = _M_ABLAUF + ("Zusammenführen",)
_M_EINFAERBUNG = _M_ABLAUF + ("Einfärbung (360°-Kamera)",)
_M_MAEANDER = _M_ABLAUF + ("Mäander-Einfärbung",)
_M_SPLAT = _M_ABLAUF + ("Gaussian Splat und Fusion",)
_M_ANSICHT = ("&Ansicht",)
_M_WERKZEUGE = ("&Werkzeuge",)
_M_RVIZ = _M_WERKZEUGE + ("RViz-Wiedergabe",)
_M_HILFE = ("&Hilfe",)

_KARTE = ("bag", "rec", "calib")   # Bag, Karte und Kamerakalibrierung

# In Menuefolge; ein Menue steht vor seinem ersten Eintrag.
_TABELLE = (
    Menue(_M_DATEI),
    Befehl("open", "Rosbag öffnen …", "_on_open_clicked", _M_DATEI, "Ctrl+O",
           "Einen Flug öffnen; er wird zur Arbeitswolke."),
    Befehl("open_project", "Projekt aus dem Cache öffnen …", "_on_open_project", _M_DATEI,
           "Ctrl+Shift+P",
           "Ein Projekt aus dem Cache — der einzige Weg zu zusammengeführten Karten.",
           knopf="Projekt öffnen …"),
    Befehl("project_import", "Exportiertes Projekt öffnen …", "_on_import_project", _M_DATEI,
           "Ctrl+I",
           "Ein exportiertes Projekt direkt in seinem Ordner öffnen — ohne Kopie,\n"
           "ohne Rückfrage. Änderungen werden sofort dort gespeichert."),
    Befehl("merge", "Zweiten Flug laden …", "_on_merge_pick", _M_DATEI, "Ctrl+Shift+O",
           "Zweites Rosbag dazuladen. Dessen Karte muss berechnet sein —\n"
           "sonst wird gefragt, ob sie jetzt berechnet werden soll.\n"
           "Zweites Rosbag laden und mit dem offenen zusammenführen.",
           braucht=("rec",), trenner=True),
    Befehl("meander", "Mäanderflug wählen …", "_on_meander_pick", _M_DATEI, "",
           "Ordner mit den Bildern eines DJI-Kartierungsfluges.\n"
           "Gesucht werden die _V.JPG, die _T.JPG sind die Thermalbilder.\n"
           "Bilder eines DJI-Kartierungsfluges für die Einfärbung laden.",
           braucht=("rec",)),
    Befehl("project_export", "Projekt exportieren …", "_on_export_project", _M_DATEI,
           "Ctrl+Shift+E",
           "Alle berechneten Daten in einen Ordner kopieren, den man weiterreichen kann.",
           braucht=("rec",), trenner=True),
    Befehl("export_ply", "Punktwolke als PLY/PCD speichern …", "_on_export_plypcd", _M_DATEI,
           "Ctrl+E", "Die angezeigte Farbebene mitschreiben.",
           braucht=("world",), knopf="PLY/PCD speichern …"),
    Befehl("export_las", "Punktwolke als LAS speichern …", "_on_export_las", _M_DATEI, "",
           "Georeferenziert, sofern das GPS brauchbar war.",
           braucht=("world",), knopf="LAS speichern …"),
    Befehl("cloud_cc", "Punktwolke in CloudCompare öffnen", "_on_cloud_cloudcompare",
           _M_DATEI, braucht=("world",)),
    Befehl("mesh_cc", "Mesh in CloudCompare öffnen", "_on_mesh_cloudcompare", _M_DATEI, "",
           "Dreiecksnetz mit den Farben der angezeigten Ebene; gespeichert im\n"
           "Projektordner unter mesh/.\n"
           "Dreiecksnetz aus der Wolke erzeugen (Einstellungen im Abschnitt Mesh)\n"
           "und in CloudCompare öffnen.",
           braucht=("world",)),
    Befehl("screenshot", "3D-Ansicht als Bild speichern …", "_on_screenshot", _M_DATEI,
           "Ctrl+P", braucht=None),
    Befehl("quit", "Beenden", "close", _M_DATEI, "Ctrl+Q", braucht=None, trenner=True),

    Menue(_M_ABLAUF),
    Befehl("fastlio", "Karte berechnen (FAST-LIO2)", "_on_fastlio_clicked", _M_ABLAUF, "F5",
           braucht=("bag",), knopf="Karte berechnen"),
    Menue(_M_KALIB, trenner=True),
    Befehl("autocal", "Automatisch kalibrieren (grob)", "_on_autocal_clicked", _M_KALIB, "",
           "Sucht die Kamera-Extrinsik über die Foto-Konsistenz.", braucht=_KARTE),
    Befehl("overlay", "Überlagerung prüfen", "_on_overlay_clicked", _M_KALIB, "",
           "Pano mit projizierten Lidar-Punkten zur Sichtprüfung.", braucht=_KARTE),
    Befehl("extrinsic_reset", "Extrinsik zurücksetzen", "_on_extrinsic_reset", _M_KALIB),
    Menue(_M_ZUSAMMEN),
    Befehl("merge_auto", "Automatisch ausrichten", "_on_merge_auto", _M_ZUSAMMEN, "",
           "Globale Suche (FGR über FPFH) plus ICP von grob nach fein.\n"
           "Dauert je nach Wolkengröße ein bis mehrere Minuten.",
           braucht=("rec", "zweitflug")),
    Befehl("merge_icp", "Nur fein ausrichten (ICP)", "_on_merge_icp", _M_ZUSAMMEN, "",
           "Verfeinert nur die aktuelle Lage — nach einer Handjustage genug.",
           braucht=("rec", "zweitflug")),
    Befehl("merge_apply", "Zusammenführen", "_on_merge_apply", _M_ZUSAMMEN, "",
           "Schreibt eine gemeinsame Aufzeichnung und öffnet sie als Arbeitswolke.\n"
           "Sie lässt sich danach als Ganzes einfärben und exportieren.",
           braucht=("rec", "zweitflug")),
    Befehl("merge_reset", "Zweiten Flug verwerfen", "_on_merge_discard", _M_ZUSAMMEN, "",
           "Den dazugeladenen zweiten Flug wieder entfernen.",
           braucht=("rec", "zweitflug")),
    Menue(_M_EINFAERBUNG),
    Befehl("colorize", "Einfärben", "_on_colorize_clicked", _M_EINFAERBUNG, "F6",
           braucht=_KARTE),
    Befehl("blaumaske", "Blaumaske zeigen", "_on_blue_preview_clicked", _M_EINFAERBUNG, "",
           "Markiert im aktuellen Kamerabild magenta, was als Blaulicht gilt.",
           braucht=("bag",), frei_bei_busy=True),
    Menue(_M_MAEANDER),
    Befehl("meander_auto", "Automatisch: ausrichten bis zur Farbe", "_on_meander_auto",
           _M_MAEANDER, "",
           "Alles hintereinander: Ausrichten, Optik einmessen, Feinausrichten,\n"
           "Einfärben mit Sichtprüfung. Rund sechs Minuten.",
           braucht=("rec", "flug")),
    Befehl("meander_align", "Ausrichten", "_on_meander_align", _M_MAEANDER, "",
           "Grob per Kreuzkorrelation über den Gierwinkel, fein über den\n"
           "Höhenunterschied zum Rastermodell der Wolke. Kein ICP — das würde\n"
           "an Gebäudekanten verkippen und die Lotrechte zerstören.",
           braucht=("rec", "flug")),
    Befehl("meander_optik", "Optik einmessen", "_on_meander_einmessen", _M_MAEANDER, "",
           "Höhe über den Laser-Entfernungsmesser der Drohne, RGB-Brennweite\n"
           "über die Farbkonsistenz der Bilder, Thermalkamera (Brennweite,\n"
           "Verzeichnung, Schielwinkel) gegen das RGB-Bild desselben Auslösers.\n"
           "Rund eine Minute. Läuft nach dem ersten Ausrichten von selbst.",
           braucht=("lage",)),
    Befehl("meander_fein", "Feinausrichten", "_on_meander_fein", _M_MAEANDER, "",
           "Fotomodell auf die Karte legen: Neigung und Höhe über die\n"
           "Oberfläche, Versatz in der Ebene über die Kanten. Gegengeprüft über\n"
           "die Farbkonsistenz — was nicht hilft, wird nicht übernommen.",
           braucht=("lage",)),
    Befehl("meander_fenster", "Im Fenster justieren …", "_on_meander_fenster", _M_MAEANDER,
           "",
           "Eigenes Fenster: Karte und Flug übereinander, live verschieben,\n"
           "mit Farbvorschau. Das Hauptfenster bleibt unberührt.",
           braucht=("lage",)),
    Befehl("meander_run", "Einfärben", "_on_meander_run", _M_MAEANDER, "F7",
           "Punktwolke aus den Bildern eines Kartierungsfluges einfärben.",
           braucht=("rec", "flug")),
    Menue(_M_SPLAT),
    Befehl("splat_pruefen", "GPU und Interpreter prüfen", "_on_splat_pruefen", _M_SPLAT),
    Befehl("splat_meander", "Mäanderflug per Splat einfärben", "_on_splat_maeander",
           _M_SPLAT, "",
           "Farben aus allen Bildern des Mäanderfluges zugleich lernen (GPU).",
           braucht=("rec", "flug")),
    Befehl("splat_onboard", "360°-Kamera per Splat einfärben", "_on_splat_onboard",
           _M_SPLAT, "",
           "Nimmt die Extrinsik aus dem Abschnitt Kamera-Kalibrierung, Helligkeitsfenster\n"
           "und Himmelssaum aus dem Abschnitt Einfärbung (360°-Kamera).\n"
           "Farben aus den Frames der 360°-Kamera zugleich lernen (GPU).",
           braucht=_KARTE),
    Befehl("fusion", "Fusionieren (ohne Splat)", "_on_fusion", _M_SPLAT, "",
           "Onboard-Farben an den Mäander angleichen und nach Flächenlage mischen:\n"
           "Dächer und Boden aus dem Mäander, Fassaden aus der 360°-Kamera.",
           braucht=("world", "fusion")),
    Befehl("splat_gemeinsam", "Beide in einem Splat einfärben", "_on_splat_gemeinsam",
           _M_SPLAT, "",
           "Ein Splat aus beiden Flügen. Die Farbe steht im Mäander, jedes Onboard-Bild\n"
           "bekommt seine eigene Farbmatrix. Schritte: beide Felder zusammen.\n"
           "Liegt eine Fusion (ohne Splat) vor, startet die Farbmatrix dort.\n"
           "Ein Splat aus beiden Flügen, Farbe im Mäander, Farbmatrix je Onboard-Bild (GPU).",
           braucht=_KARTE + ("flug",)),

    Menue(_M_ANSICHT),
    Menue(_M_ANSICHT + ("Farbe",), "menu_farbquelle"),
    Menue(_M_ANSICHT + ("Hintergrund",), "menu_hintergrund"),
    Menue(_M_ANSICHT + ("Bereich",), "menu_tab"),
    Befehl("edl", "Kantenbetonung (EDL)", "_on_menu_edl", _M_ANSICHT,
           braucht=None, schaltbar=True, signal="toggled"),
    Befehl("reset_cam", "Ansicht zurücksetzen", "_on_reset_camera", _M_ANSICHT, "R",
           braucht=None, trenner=True),
    Befehl("cut_reset", "Höhenschnitt aufheben", "_on_cut_reset", _M_ANSICHT, "Ctrl+H",
           "Die volle Höhe wieder zeigen.", braucht=None),
    Befehl("preview", "Zweiten Flug (orange) zeigen", "_on_toggle_preview", _M_ANSICHT, "",
           "Die Vorschau des dazugeladenen Fluges. Orange heißt: noch nicht\n"
           "übernommen — sie gehört noch nicht zur Karte.",
           braucht=None, schaltbar=True),
    Befehl("sidebar", "Seitenleiste", "_on_toggle_sidebar", _M_ANSICHT, "Ctrl+B",
           braucht=None, schaltbar=True, trenner=True),
    Befehl("expand_all", "Alle Abschnitte aufklappen", "_on_expand_all", _M_ANSICHT,
           braucht=None),
    Befehl("collapse_all", "Alle Abschnitte zuklappen", "_on_collapse_all", _M_ANSICHT,
           braucht=None),

    Menue(_M_WERKZEUGE),
    Befehl("measure", "Messen", "_on_toggle_measure", _M_WERKZEUGE, "M",
           "Zwei Klicks in die Wolke: Abstand in Metern.",
           braucht=("world",), schaltbar=True),
    Befehl("exploration", "Explorationsgrad neu berechnen", "_on_exploration_neu",
           _M_WERKZEUGE, "",
           "Den Wert oben rechts noch einmal aus dem Bag rechnen, am Cache vorbei.",
           braucht=("bag",)),
    Menue(_M_RVIZ, trenner=True),
    Befehl("rviz_start", "Starten", "_on_rviz_start", _M_RVIZ, braucht=None),
    Befehl("rviz_stop", "Beenden", "_on_rviz_stop", _M_RVIZ, braucht=None),
    Befehl("rviz_replay", "Wiederholen", "_on_rviz_replay", _M_RVIZ, "",
           "Leert die RViz-Anzeige und spielt den Bag von vorn ab.", braucht=None),
    Befehl("settings_reset", "Einstellungen auf Vorgabe", "_on_settings_reset",
           _M_WERKZEUGE, trenner=True),

    Menue(_M_HILFE),
    Befehl("about", "Über Super360 Studio", "_on_about", _M_HILFE, braucht=None),
)

#: {Schluessel: Befehl} in Menuefolge
BEFEHLE = {b.schluessel: b for b in _TABELLE if isinstance(b, Befehl)}
#: Schluessel der Untermenues, die das Fenster fuellt
UNTERMENUES = tuple(m.schluessel for m in _TABELLE if isinstance(m, Menue) and m.schluessel)


def baue(win) -> dict:
    """Menueleiste aus :data:`BEFEHLE` an ``win`` haengen.

    Gibt {Schluessel: QAction} zurueck, dazu die Untermenues aus
    :data:`UNTERMENUES` als QMenu. Fehlt dem Fenster ein Handler, wird nichts
    gebaut und ein AttributeError nennt alle fehlenden.
    """
    fehlend = sorted({b.handler for b in BEFEHLE.values()
                      if not callable(getattr(win, b.handler, None))})
    if fehlend:
        raise AttributeError("Dem Fenster fehlen Handler der Befehlstabelle: "
                             + ", ".join(fehlend))
    bar = win.menuBar()
    bar.setNativeMenuBar(False)   # unter Wayland/GNOME sonst unsichtbar
    menues: dict = {}
    actions: dict = {}
    for e in _TABELLE:
        if isinstance(e, Menue):
            if len(e.pfad) == 1:
                menu = bar.addMenu(e.pfad[0])
            else:
                oben = menues[e.pfad[:-1]]
                if e.trenner:
                    oben.addSeparator()
                menu = oben.addMenu(e.pfad[-1])
            menu.setToolTipsVisible(True)
            menues[e.pfad] = menu
            if e.schluessel:
                actions[e.schluessel] = menu
            continue
        menu = menues[e.menue]
        if e.trenner:
            menu.addSeparator()
        act = QtWidgets.QAction(e.text, win)
        if e.kuerzel:
            act.setShortcut(QKeySequence(e.kuerzel))
            act.setShortcutContext(Qt.ApplicationShortcut)
        if e.tooltip:
            act.setToolTip(e.tooltip)
            act.setStatusTip(e.tooltip)
        if e.schaltbar:
            act.setCheckable(True)
        fn = getattr(win, e.handler)
        if e.signal == "toggled":
            act.toggled.connect(fn)
        else:
            act.triggered.connect(lambda _checked=False, f=fn: f())
        menu.addAction(act)
        actions[e.schluessel] = act
    return actions


def _tooltip(befehl: Befehl, gruende: list) -> str:
    if not gruende:
        return befehl.tooltip
    grund = " ".join(gruende)
    return f"{befehl.tooltip}\n\n{grund}" if befehl.tooltip else grund


def schalte(actions: dict, zustand: dict, busy: bool) -> None:
    """Aktionen nach ``zustand`` ({Name aus GRUENDE: bool}) freigeben.

    Waehrend eines Schritts (``busy``) ist jeder Befehl mit ``braucht`` gesperrt,
    ausser er ist ``frei_bei_busy``. Ein gesperrter Befehl traegt den Grund
    unter seinem Tooltip; frei bekommt er den Tooltip der Tabelle zurueck. Der
    StatusTip bleibt immer der Tooltip der Tabelle. Befehle mit ``braucht``
    None und Schluessel, die in ``actions`` fehlen, bleiben unberuehrt.
    """
    for key, b in BEFEHLE.items():
        act = actions.get(key)
        if act is None or b.braucht is None:
            continue
        if busy and not b.frei_bei_busy:
            gruende = [GRUND_BESCHAEFTIGT]
        else:
            gruende = []
            for name in b.braucht:
                if not zustand.get(name) and GRUENDE[name] not in gruende:
                    gruende.append(GRUENDE[name])
        act.setEnabled(not gruende)
        act.setToolTip(_tooltip(b, gruende))


if __name__ == "__main__":
    import ast
    import copy
    import glob
    import inspect
    import os
    import re
    import sys

    # Was die Leiste schon immer hatte und was dazukommt; die Kuerzel sind
    # eingeuebt und duerfen nicht wandern.
    ALT = {"open", "open_project", "merge", "meander", "project_export", "project_import",
           "export_ply", "export_las", "mesh_cc", "cloud_cc", "screenshot", "quit",
           "autocal", "overlay", "extrinsic_reset", "merge_reset", "settings_reset",
           "edl", "reset_cam", "cut_reset", "preview", "sidebar", "expand_all",
           "collapse_all", "fastlio", "exploration", "colorize", "meander_run",
           "splat_meander", "splat_onboard", "splat_gemeinsam", "fusion", "merge_apply",
           "measure", "rviz_start", "rviz_stop", "about"}
    NEU = {"merge_auto", "merge_icp", "blaumaske", "meander_auto", "meander_align",
           "meander_optik", "meander_fein", "meander_fenster", "splat_pruefen",
           "rviz_replay"}
    KUERZEL = {"open": "Ctrl+O", "open_project": "Ctrl+Shift+P", "merge": "Ctrl+Shift+O",
               "project_export": "Ctrl+Shift+E", "project_import": "Ctrl+I",
               "export_ply": "Ctrl+E", "screenshot": "Ctrl+P", "quit": "Ctrl+Q",
               "reset_cam": "R", "cut_reset": "Ctrl+H", "sidebar": "Ctrl+B",
               "fastlio": "F5", "colorize": "F6", "meander_run": "F7", "measure": "M"}
    NIE = {"quit", "about", "reset_cam", "cut_reset", "sidebar", "expand_all",
           "collapse_all", "screenshot", "edl", "preview", "rviz_start", "rviz_stop",
           "rviz_replay"}
    NUR_BUSY = {"open", "open_project", "project_import", "splat_pruefen",
                "extrinsic_reset", "settings_reset"}

    assert len(ALT) == 37 and len(NEU) == 10 and len(KUERZEL) == 15
    assert set(BEFEHLE) == ALT | NEU, sorted(set(BEFEHLE) ^ (ALT | NEU))
    assert len(_TABELLE) - len(BEFEHLE) == len({m.pfad for m in _TABELLE
                                                if isinstance(m, Menue)})
    assert UNTERMENUES == ("menu_farbquelle", "menu_hintergrund", "menu_tab")
    assert {k: b.kuerzel for k, b in BEFEHLE.items() if b.kuerzel} == KUERZEL
    assert {k for k, b in BEFEHLE.items() if b.braucht is None} == NIE
    assert {k for k, b in BEFEHLE.items() if b.braucht == ()} == NUR_BUSY
    assert {k for k, b in BEFEHLE.items() if b.frei_bei_busy} == {"blaumaske"}
    assert {k for k, b in BEFEHLE.items() if b.schaltbar} == {"sidebar", "measure",
                                                             "preview", "edl"}
    assert {k for k, b in BEFEHLE.items() if b.signal != "triggered"} == {"edl"}
    assert BEFEHLE["edl"].signal == "toggled"
    assert set(BEFEHLE["meander_run"].braucht) == {"rec", "flug"}
    for k in ("merge_auto", "merge_icp", "merge_apply", "merge_reset"):
        assert set(BEFEHLE[k].braucht) == {"rec", "zweitflug"}, k
    for b in BEFEHLE.values():
        assert set(b.braucht or ()) <= set(GRUENDE), b.schluessel
        assert b.signal in ("triggered", "toggled"), b.schluessel
        assert b.signal == "triggered" or b.schaltbar, b.schluessel
        assert not re.search(r"Abschnitt \d", b.tooltip), b.schluessel
        assert "&" not in b.text, b.schluessel

    # Signaturen der Handler aus dem Quelltext des Fensters; ein Stub mit
    # genau diesen Signaturen bekommt jeden Aufruf mit.
    hier = os.path.dirname(os.path.abspath(__file__))
    quellen = [os.path.join(hier, "main_window.py")] + sorted(
        glob.glob(os.path.join(hier, "fenster", "*.py")))
    gesucht = {b.handler for b in BEFEHLE.values()}
    signaturen: dict = {}
    for pfad in quellen:
        with open(pfad, encoding="utf-8") as fh:
            baum = ast.parse(fh.read())
        for klasse in (n for n in baum.body if isinstance(n, ast.ClassDef)):
            for fn in klasse.body:
                if isinstance(fn, ast.FunctionDef) and fn.name in gesucht \
                        and fn.name not in signaturen:
                    args = copy.deepcopy(fn.args)
                    for a in args.posonlyargs + args.args + args.kwonlyargs:
                        a.annotation = None
                    for a in (args.vararg, args.kwarg):
                        if a is not None:
                            a.annotation = None
                    args.defaults = [ast.Constant(None) for _ in args.defaults]
                    args.kw_defaults = [None if d is None else ast.Constant(None)
                                        for d in args.kw_defaults]
                    ns: dict = {}
                    exec(f"def f({ast.unparse(args)}): pass", ns)
                    signaturen[fn.name] = inspect.signature(ns["f"])
    ohne_quelle = gesucht - set(signaturen)
    qt_eigen = {h for h in ohne_quelle if hasattr(QtWidgets.QMainWindow, h)}
    # noch nicht am Fenster: die beiden parameterlosen Huellen fuers Ausrichten
    noch_nicht = ohne_quelle - qt_eigen
    assert noch_nicht <= {BEFEHLE["merge_auto"].handler, BEFEHLE["merge_icp"].handler}, \
        sorted(noch_nicht)
    leer_sig = inspect.signature(lambda self: None)
    for h in ohne_quelle:
        signaturen[h] = leer_sig
    assert str(signaturen[BEFEHLE["edl"].handler]) == "(self, on)"

    def _stub_methode(name, sig):
        def methode(self, *args, **kwargs):
            try:
                sig.bind(self, *args, **kwargs)
            except TypeError as exc:
                self.fehler.append((name, str(exc)))
                return
            self.aufrufe.append((name, args))
        return methode

    class _Stub(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.aufrufe: list = []
            self.fehler: list = []

    for name, sig in signaturen.items():
        setattr(_Stub, name, _stub_methode(name, sig))

    app = QtWidgets.QApplication(sys.argv)

    # fehlt ein Handler, wird nichts gebaut
    class _Unvollstaendig(_Stub):
        pass
    fehlt = BEFEHLE["fusion"].handler
    setattr(_Unvollstaendig, fehlt, None)
    w0 = _Unvollstaendig()
    try:
        baue(w0)
    except AttributeError as exc:
        assert fehlt in str(exc), exc
    else:
        raise AssertionError("baue() ohne Handler ist nicht gescheitert")
    assert not w0.menuBar().actions()

    win = _Stub()
    actions = baue(win)
    bar = win.menuBar()
    assert not bar.isNativeMenuBar()
    assert set(actions) == set(BEFEHLE) | set(UNTERMENUES)
    for key in UNTERMENUES:
        assert isinstance(actions[key], QtWidgets.QMenu), key
    titel = [a.text() for a in bar.actions()]
    assert titel == ["&Datei", "A&blauf", "&Ansicht", "&Werkzeuge", "&Hilfe"], titel

    def _menues(menu, pfad):
        yield pfad, menu
        for a in menu.actions():
            if a.menu() is not None:
                yield from _menues(a.menu(), pfad + (a.text(),))

    alle_menues = {}
    for a in bar.actions():
        alle_menues.update(dict(_menues(a.menu(), (a.text(),))))
    assert set(alle_menues) == {m.pfad for m in _TABELLE if isinstance(m, Menue)}
    for pfad, menu in alle_menues.items():
        assert menu.toolTipsVisible(), pfad
    for key, b in BEFEHLE.items():
        act = actions[key]
        assert act in alle_menues[b.menue].actions(), key
        assert act.text() == b.text and act.isCheckable() == b.schaltbar, key
        assert act.shortcut().toString() == b.kuerzel, key
        if b.kuerzel:
            assert act.shortcutContext() == Qt.ApplicationShortcut, key
        assert act.statusTip() == b.tooltip, key
        assert act.isEnabled(), key
    assert {k: a.shortcut().toString() for k, a in actions.items()
            if isinstance(a, QtWidgets.QAction) and not a.shortcut().isEmpty()} == KUERZEL

    # jede Aktion einmal ausloesen
    for key, b in BEFEHLE.items():
        act = actions[key]
        win.aufrufe.clear()
        if b.signal == "toggled":
            act.setChecked(not act.isChecked())
            assert win.aufrufe == [(b.handler, (True,))], (key, win.aufrufe)
            assert type(win.aufrufe[0][1][0]) is bool
            win.aufrufe.clear()
            act.setChecked(False)
            assert win.aufrufe == [(b.handler, (False,))], (key, win.aufrufe)
        else:
            act.trigger()
            assert win.aufrufe == [(b.handler, ())], (key, win.aufrufe)
    assert not win.fehler, win.fehler

    tip_vorher = {k: actions[k].toolTip() for k in BEFEHLE}
    alles = {name: True for name in GRUENDE}
    geschaltet = [k for k, b in BEFEHLE.items() if b.braucht is not None]

    # laufender Schritt: nur frei_bei_busy und die nie geschalteten bleiben frei
    schalte(actions, alles, busy=True)
    for key, b in BEFEHLE.items():
        act = actions[key]
        if b.braucht is None:
            assert act.isEnabled() and act.toolTip() == tip_vorher[key], key
        elif b.frei_bei_busy:
            assert act.isEnabled() and act.toolTip() == tip_vorher[key], key
        else:
            assert not act.isEnabled(), key
            assert act.toolTip().endswith(GRUND_BESCHAEFTIGT), key
        assert act.statusTip() == b.tooltip, key
    schalte(actions, alles, busy=False)
    for key in geschaltet:
        assert actions[key].isEnabled(), key
        assert actions[key].toolTip() == tip_vorher[key], key

    # jede Voraussetzung einzeln wegnehmen: genau ihre Befehle werden grau,
    # mit ihrem Grund, und bekommen den Tooltip danach zurueck
    for name in GRUENDE:
        schalte(actions, dict(alles, **{name: False}), busy=False)
        for key in geschaltet:
            b, act = BEFEHLE[key], actions[key]
            if name in b.braucht:
                assert not act.isEnabled(), (name, key)
                assert act.toolTip().endswith(GRUENDE[name]), (name, key)
                if b.tooltip:
                    assert act.toolTip().startswith(b.tooltip + "\n\n"), (name, key)
            else:
                assert act.isEnabled() and act.toolTip() == tip_vorher[key], (name, key)
            assert act.statusTip() == b.tooltip, (name, key)
        schalte(actions, alles, busy=False)
    schalte(actions, {}, busy=False)
    tip = actions["meander_run"].toolTip()
    assert tip == (BEFEHLE["meander_run"].tooltip + "\n\n" + GRUENDE["rec"] + " "
                   + GRUENDE["flug"]), tip
    assert actions["fusion"].toolTip().count(GRUENDE["world"]) == 1
    for key in NUR_BUSY:
        assert actions[key].isEnabled(), key
    schalte(actions, alles, busy=False)
    assert {k: actions[k].toolTip() for k in BEFEHLE} == tip_vorher

    # die heutige Leiste baut weiter wie bisher
    alt = build(_Stub())
    assert set(alt) == ALT | set(UNTERMENUES), sorted(set(alt) ^ (ALT | set(UNTERMENUES)))
    assert [k for k in ALT if alt[k].isCheckable()] == [k for k in ALT if k in
                                                        ("edl", "preview", "sidebar",
                                                         "measure")]

    print(f"menubar SELFTEST OK: {len(BEFEHLE)} Befehle ({len(ALT)} bisherige, "
          f"{len(NEU)} neue), {len(KUERZEL)} Kürzel, {len(alle_menues)} Menüs"
          + (f"; noch nicht am Fenster: {', '.join(sorted(noch_nicht))}"
             if noch_nicht else ""))
