"""Anzeige, Farbebenen und Menü (Mixin des Hauptfensters).

Schreibt am Hauptfenster: _chk_edl, _chk_only_colored, _chk_path, _color_mode,
_colors, _combo_bg, _combo_voxel, _layer_key, _layers, _loading_ui, _point_size,
_temperatur, _temperatur_anzeigen, _temperaturen, _valid.
"""
from __future__ import annotations

import os

import numpy as np

from PyQt5.QtCore import QLocale
from PyQt5.QtWidgets import QFormLayout, QMessageBox, QWidget

from core.ebenen import THERMAL, lade_farbdateien, lade_temperatur, laden
from core.project import Project

from ui import menubar as menubar_mod
from ui.bausteine import _wrappable, auswahl, haken, speicherpfad, still_setzen

#: Farben der 3D-Ansicht, wie Leiste und Menue Ansicht ▸ Farbe sie anbieten:
#: (Schluessel, Ebene, Text). 'rgb:<ebene>' zeigt eine Farbebene, die uebrigen
#: sind Farbmodi der Ansicht. Die Ebenen stehen in der Reihenfolge von
#: Project.LAYERS; fehlt die gewaehlte, gilt die erste vorhandene.
_FARBEN = (
    ("uniform", None, "Einheitsfarbe"),
    ("intensitaet", None, "Intensität"),
    ("hoehe", None, "Höhe"),
    ("rgb:onboard", "onboard", "RGB Onboard"),
    ("rgb:onboard_splat", "onboard_splat", "RGB Onboard (Splat)"),
    ("rgb:meander_rgb", "meander_rgb", "RGB Mäander"),
    ("rgb:meander_splat", "meander_splat", "RGB Mäander (Splat)"),
    ("rgb:meander_thermal", "meander_thermal", "Thermal Mäander"),
    ("rgb:meander_thermal_splat", "meander_thermal_splat", "Temperatur Mäander (Splat)"),
    ("rgb:fusion", "fusion", "RGB Fusion"),
    ("rgb:fusion_splat", "fusion_splat", "RGB Fusion (Splat)"),
)
_EBENEN = tuple(ebene for _key, ebene, _text in _FARBEN if ebene)
if _EBENEN != tuple(Project.LAYERS) or any(
        key != f"rgb:{ebene}" for key, ebene, _text in _FARBEN if ebene):
    raise RuntimeError(f"Die Farbtabelle passt nicht zu Project.LAYERS: {_EBENEN} "
                       f"gegen {tuple(Project.LAYERS)}.")
#: Farbmodi der Ansicht; einen unbekannten aus den Einstellungen nimmt sie als RGB
_FARBMODI = ("rgb",) + tuple(key for key, ebene, _text in _FARBEN if ebene is None)
_VOXEL_ITEMS = (("Aus", 0.0), ("0,05 m", 0.05), ("0,10 m", 0.10), ("0,20 m", 0.20))
_BG_ITEMS = (("Dunkel", "dunkel"), ("Hell", "hell"))


def _punktgroesse(wert) -> float:
    """Punktgroesse auf 0,01 px gerundet (halbe Hundertstel nach oben, wie Qt
    rundet) und auf 0,5 bis 10 px begrenzt."""
    v = float(QLocale.c().toString(float(wert), "f", 2))
    if v < 0.5:
        return 0.5
    return v if v <= 10.0 else 10.0         # NaN wird zu 10


def _farbmodus(wert) -> str:
    """Farbmodus aus den Einstellungen; ein unbekannter gilt als RGB."""
    return wert if isinstance(wert, str) and wert in _FARBMODI else _FARBMODI[0]


def _ebene_oder_ersatz(key, ebenen) -> str:
    """Die gewaehlte Ebene, wenn sie geladen ist; sonst die erste geladene in
    Tabellenreihenfolge, ohne geladene Ebene 'onboard'."""
    if key in _EBENEN and key in ebenen:
        return key
    return next((e for e in _EBENEN if e in ebenen), "onboard")


class AnzeigeMixin:
    def _abschnitt_anzeige(self) -> QWidget:
        box = QWidget()
        form = _wrappable(QFormLayout(box))
        self._chk_only_colored = haken("Nur eingefärbte Punkte",
                                       slot=self._on_display_changed)
        form.addRow(self._chk_only_colored)
        self._combo_voxel = auswahl(_VOXEL_ITEMS, slot=self._on_display_changed)
        form.addRow("Anzeige-Voxel:", self._combo_voxel)
        self._combo_bg = auswahl(_BG_ITEMS, slot=self._on_display_changed)
        form.addRow("Hintergrund:", self._combo_bg)
        self._chk_edl = haken("Kantenbetonung (EDL)", slot=self._on_display_changed)
        if not self._cloud_view.edl_available:
            self._chk_edl.setEnabled(False)
            self._chk_edl.setToolTip(
                "EDL wird von dieser VTK-Installation nicht unterstützt.")
        form.addRow(self._chk_edl)
        self._chk_path = haken("Flugbahn zeigen", slot=self._on_display_changed)
        form.addRow(self._chk_path)
        return box

    def _sync_after_display(self) -> None:
        self._sync_menu_state()

    def _push_display_settings(self) -> None:
        cv = self._cloud_view
        cv.set_point_size(float(self._point_size))
        cv.set_color_mode(self._color_mode)
        cv.set_only_colored(self._chk_only_colored.isChecked())
        cv.set_voxel_display(float(self._combo_voxel.currentData()))
        cv.set_background(self._combo_bg.currentData())
        cv.set_eyedome(self._chk_edl.isChecked())
        if self._chk_path.isChecked() and self._rec is not None:
            cv.set_path(self._rec.path_positions())
        else:
            cv.set_path(None)
        self._sync_farbleiste()

    # ------------------------------------------------ Leiste ueber der Ansicht

    def _farbleiste_key(self) -> str:
        if self._color_mode == "rgb":
            return f"rgb:{self._layer_key}"
        return str(self._color_mode)

    def _farbeintraege(self) -> list:
        """Die Farbtabelle als [(Schluessel, Text, vorhanden)] fuer Leiste und Menue."""
        hat_int = self._rec is not None and getattr(self._rec, "intensity", None) is not None
        eintraege = []
        for key, ebene, text in _FARBEN:
            if ebene:
                da = ebene in self._layers
            elif key == "intensitaet":
                da = hat_int
            else:
                da = True
            eintraege.append((key, text, da))
        return eintraege

    def _sync_farbleiste(self) -> None:
        """Leiste und Menue Ansicht ▸ Farbe an Farbe und vorhandene Ebenen angleichen."""
        eintraege = self._farbeintraege()
        self._cloud_view.set_farbmodi(eintraege, self._farbleiste_key())
        self._farbmenue_abgleichen(eintraege)

    def _setze_farbe(self, schluessel: str, speichern: bool = True) -> None:
        """Farbe der Ansicht setzen, ein Schluessel aus der Farbtabelle.

        'rgb:<ebene>' stellt auf RGB und macht die Ebene zur Farbquelle, wenn sie
        geladen ist; die uebrigen sind Farbmodi. Ansicht, Leiste, Menue und
        (mit ``speichern``) die Einstellungen ziehen nach.
        """
        if schluessel.startswith("rgb:"):
            ebene = schluessel[4:]
            paar = self._layers.get(ebene)
            if (self._world is not None and paar is not None
                    and ebene == self._layer_key and self._colors is paar[0]):
                # Die Ebene steht schon in der Wolke: nur der Modus wechselt.
                self._color_mode = "rgb"
                self._push_display_settings()
                if speichern:
                    self._save_settings()
                return
            if paar is not None:
                self._layer_key = ebene
            self._color_mode = "rgb"
        elif schluessel in _FARBMODI:
            self._color_mode = schluessel
            if self._world is not None:
                # Ein reiner Farbmodus aendert an Wolke, Farben und Maske nichts:
                # den Modus durchreichen genuegt. Die Wolke neu aufzubauen
                # kostete bei 24 Mio. Punkten einige Sekunden.
                self._push_display_settings()
                if speichern:
                    self._save_settings()
                return
        self._farbe_anwenden()
        if speichern:
            self._save_settings()

    def _farbe_anwenden(self) -> None:
        """Gewaehlte Ebene und Farbmodus in Ansicht, Leiste und Menue bringen."""
        self._apply_layer()
        if self._world is None:
            # ohne Wolke endet _apply_layer vor der Ansicht
            self._push_display_settings()

    def _on_farbleiste(self, key: str) -> None:
        self._setze_farbe(key)

    def _on_leiste_punktgroesse(self, wert: float) -> None:
        self._point_size = _punktgroesse(wert)
        self._save_settings()

    def _on_leiste_temperatur(self, an: bool) -> None:
        self._temperatur_anzeigen = bool(an)
        self._save_settings()

    def _on_display_changed(self, *_a) -> None:
        if self._loading_ui:
            return
        self._push_display_settings()
        self._sync_after_display()
        self._save_settings()

    # ======================================================== Farbebenen

    def _reload_layers(self) -> None:
        """Alle vorhandenen Farbebenen des Projekts einlesen.

        Jede Ebene wird gegen die Punktzahl geprueft; was nicht passt, faellt
        weg statt die Anzeige zu verfaelschen. Leiste und Menue bieten danach
        nur an, was wirklich da ist.
        """
        self._layers = {}
        self._temperaturen = {}
        if self._project is None or self._rec is None:
            self._refresh_layer_combo()
            return
        n = int(self._rec.n_points)
        for key in Project.LAYERS:
            if not self._project.has_layer(key):
                continue
            if key == "onboard":
                try:
                    from core.colorizer import rec_fingerprint
                    fp = rec_fingerprint(self._rec)
                except Exception as exc:  # noqa: BLE001 — Kompat: ohne Prüfung laden
                    fp = None
                    self._log(f"Aufzeichnungs-Fingerprint nicht verfügbar: {exc}")
                colors, valid, err = lade_farbdateien(
                    self._project.layer_dir(key), n, expected_fingerprint=fp,
                    log_cb=self._log)
                if err:
                    self._log(err)
                    continue
                paar = (colors, valid)
            else:
                paar = laden(self._project.layer_dir(key), n)
                if paar is None:
                    self._log(f"Farbebene '{key}' passt nicht zur Wolke — ignoriert.")
                    continue
            self._layers[key] = paar
        for key in THERMAL:
            if key not in self._layers:
                continue
            t = lade_temperatur(self._project.layer_dir(key), n)
            if t is not None:
                self._temperaturen[key] = t
                gut = np.isfinite(t)
                if gut.any():
                    tt = t[gut]
                    herkunft = " (Splat)" if key.endswith("_splat") else ""
                    self._log(f"Temperaturen{herkunft} für {gut.mean() * 100:.0f} % der "
                              f"Punkte, {np.percentile(tt, 1):.1f} bis "
                              f"{np.percentile(tt, 99):.1f} °C — beim Überfahren mit der "
                              f"Maus zu sehen.")
            elif key == "meander_thermal":
                self._log("Die Thermal-Ebene hat noch keine Temperaturen — einmal neu "
                          "einfärben, dann zeigt die Maus sie an.")
        self._refresh_layer_combo()       # setzt ueber _apply_layer auch die Temperatur

    def _ebenen_melden(self, ebenen: dict, zusatz: str = "") -> None:
        """Je neu berechneter Ebene den eingefaerbten Anteil ins Protokoll.

        ``ebenen`` bildet den Ebenenschluessel auf den Anteil (0..1) ab;
        ``zusatz`` folgt dem Satz nach einem Leerzeichen.
        """
        zusatz = zusatz.strip()
        for key, anteil in ebenen.items():
            satz = f"Farbebene '{key}': {anteil * 100:.1f} % der Punkte eingefärbt."
            self._log(f"{satz} {zusatz}" if zusatz else satz)

    def _ebene_eintragen(self, key: str, rgb, valid) -> None:
        """Eine eben berechnete Farbebene eintragen, ohne sie von der Platte zu lesen.

        Leiste und Menue bieten sie danach an; gezeigt wird sie erst mit
        ``_setze_farbe`` oder ``_ebene_waehlen``.
        """
        if key not in _EBENEN:
            raise ValueError(f"Unbekannte Farbebene: {key}")
        self._layers[key] = (rgb, valid)
        self._sync_farbleiste()

    def _ebene_waehlen(self, key) -> bool:
        """Eine geladene Ebene als Farbquelle waehlen; der Farbmodus bleibt.

        Wendet an, speichert und protokolliert, wenn sich die Farbquelle
        aendert. False, wenn die Ebene nicht geladen ist.
        """
        if key not in self._layers:
            return False
        if key != self._layer_key:
            self._layer_key = key
            self._farbe_anwenden()
            self._save_settings()
            text = next((t for _k, e, t in _FARBEN if e == key), key)
            self._log(f"Farbquelle: {text}")
        return True

    def _waehle(self, combo, data) -> bool:
        """Den Eintrag mit diesen Daten waehlen (mit Signal); False, wenn er fehlt."""
        idx = combo.findData(data)
        if idx < 0:
            return False
        combo.setCurrentIndex(idx)
        return True

    def _refresh_layer_combo(self) -> None:
        """Farbquelle, Leiste und Menue auf die vorhandenen Ebenen setzen.

        Fehlt die gewaehlte Ebene, etwa die aus den Einstellungen, gilt die
        erste vorhandene in Tabellenreihenfolge, ohne Ebenen 'onboard'.
        """
        self._layer_key = _ebene_oder_ersatz(self._layer_key, self._layers)
        self._apply_layer()
        self._farbmenue_abgleichen()

    def _fill_layer_menu(self) -> None:
        """Ansicht ▸ Farbe aus der Farbtabelle fuellen, als Auswahl wie die Leiste."""
        menubar_mod.fill_radio_menu(
            self._actions["menu_farbquelle"], self,
            [(key, text) for key, _ebene, text in _FARBEN],
            self._on_menu_layer, self._farbleiste_key())
        self._farbmenue_abgleichen()

    def _farbmenue_abgleichen(self, eintraege=None) -> None:
        """Haken und Freigabe unter Ansicht ▸ Farbe wie in der Leiste; Fehlendes ist grau."""
        if eintraege is None:
            eintraege = self._farbeintraege()
        da = {key: vorhanden for key, _text, vorhanden in eintraege}
        aktuell = self._farbleiste_key()
        for act in self._actions["menu_farbquelle"].actions():
            act.setEnabled(bool(da.get(act.data(), False)))
            act.setChecked(act.data() == aktuell)

    def _on_menu_layer(self, data) -> None:
        """Ansicht ▸ Farbe; wie frueher protokolliert ein Wechsel der Ebene die Farbquelle."""
        vorher = self._layer_key
        self._setze_farbe(data)
        if self._layer_key != vorher:
            text = next((t for _k, e, t in _FARBEN if e == self._layer_key),
                        self._layer_key)
            self._log(f"Farbquelle: {text}")

    def _apply_layer(self) -> None:
        """Die gewaehlte Ebene in die Ansicht schieben.

        Fehlt fuer die gewaehlte Quelle eine Einfaerbung, waere "RGB" eine
        einfarbig graue Wolke — richtig gerechnet, aber nichtssagend. Dann
        lieber auf Hoehe umschalten und es sagen.
        """
        # Bis zum Laden der Ebenen steht hier, was die settings.json nennt —
        # auch etwas, das kein Ebenenschluessel sein kann.
        ebene = self._layer_key if isinstance(self._layer_key, str) else None
        paar = self._layers.get(ebene)
        self._colors, self._valid = paar if paar else (None, None)
        # Die Maus zeigt die Temperatur der angezeigten Thermalebene; in jeder
        # anderen Ebene die der direkten, sonst die des Splats.
        self._temperatur = self._temperaturen.get(ebene)
        if self._temperatur is None:
            self._temperatur = next((self._temperaturen[k] for k in THERMAL
                                     if k in self._temperaturen), None)
        self._cloud_view.set_temperatur(self._temperatur)
        if self._world is None:
            return
        if self._colors is None and self._color_mode == "rgb":
            self._color_mode = "hoehe"
            self._log("Für diese Farbquelle gibt es noch keine Einfärbung — "
                      "die Ansicht steht auf Höhe statt auf einfarbigem Grau.")
        # Eine fast leere Ebene plus "Nur eingefaerbte Punkte" ergibt eine leere
        # Ansicht — und die sieht aus, als sei das Modell weg. Das darf nie
        # passieren, also lieber den Haken loesen und es sagen.
        if (self._valid is not None and self._chk_only_colored.isChecked()
                and float(self._valid.mean()) < 0.01):
            self._loading_ui = True
            self._chk_only_colored.setChecked(False)
            self._loading_ui = False
            self._log(f"Diese Farbquelle hat nur {100.0 * self._valid.mean():.1f} % "
                      f"eingefärbte Punkte — 'Nur eingefärbte Punkte' wurde gelöst, "
                      f"sonst bliebe die Ansicht leer.")
        self._cloud_view.set_cloud(
            self._world, self._colors,
            self._rec.intensity if self._rec is not None else None, self._valid)
        self._push_display_settings()
        self._mesh_farben_zeigen()

    # =============================================================== Menü

    def _fill_view_menus(self) -> None:
        """Die drei Auswahl-Untermenues fuellen und mit der Sidebar gleichziehen."""
        self._fill_layer_menu()
        menubar_mod.fill_radio_menu(
            self._actions["menu_hintergrund"], self,
            [(self._combo_bg.itemData(i), self._combo_bg.itemText(i))
             for i in range(self._combo_bg.count())],
            self._on_menu_background, self._combo_bg.currentData())
        menubar_mod.fill_radio_menu(
            self._actions["menu_tab"], self,
            [(i, self._tabs.tabText(i)) for i in range(self._tabs.count())],
            self._tabs.setCurrentIndex, self._tabs.currentIndex())
        self._actions["edl"].setChecked(self._chk_edl.isChecked())
        self._actions["edl"].setEnabled(self._chk_edl.isEnabled())
        # isHidden statt isVisible: vor dem ersten show() ist nichts sichtbar
        self._actions["sidebar"].setChecked(not self._sidebar_scroll.isHidden())

    def _sync_menu_state(self) -> None:
        """Hintergrund und Kantenbetonung im Menue an die Seitenleiste angleichen
        (ohne Rueckkopplung)."""
        menu = self._actions.get("menu_hintergrund")
        if menu is not None:
            for act in menu.actions():
                act.setChecked(act.data() == self._combo_bg.currentData())
        edl = self._actions.get("edl")
        if edl is not None:
            still_setzen(edl, self._chk_edl.isChecked())

    def _sync_tab_menu(self, index: int) -> None:
        """Haken unter Ansicht ▸ Bereich auf den gezeigten Reiter setzen."""
        for act in self._actions["menu_tab"].actions():
            act.setChecked(act.data() == index)

    def _on_menu_background(self, data) -> None:
        self._waehle(self._combo_bg, data)

    def _on_menu_edl(self, on: bool) -> None:
        if self._chk_edl.isEnabled():
            self._chk_edl.setChecked(bool(on))

    def _set_sidebar_visible(self, on: bool) -> None:
        self._sidebar_scroll.setVisible(bool(on))
        act = self._actions.get("sidebar")
        if act is not None:
            still_setzen(act, bool(on))

    def _on_toggle_sidebar(self) -> None:
        self._set_sidebar_visible(self._sidebar_scroll.isHidden())
        self._save_settings()

    def _on_expand_all(self) -> None:
        self._sections.set_all(True)
        self._save_settings()

    def _on_collapse_all(self) -> None:
        self._sections.set_all(False)
        self._save_settings()

    def _on_toggle_measure(self) -> None:
        an = not self._cloud_view.measure_enabled()
        self._cloud_view.set_measure(an)
        act = self._actions.get("measure")
        if act is not None:
            still_setzen(act, an)
        self._tabs.setCurrentIndex(0)
        self._status_lbl.setText(
            "Messen: zwei Klicks in die Wolke setzen die Marken (Esc verwirft)."
            if an else "Bereit")
        if not an:
            self._mess_lbl.setText("")

    def _on_measured(self, a, b) -> None:
        """Auslese unter der 3D-Ansicht; Werte in Originalkoordinaten."""
        if a is None:
            self._mess_lbl.setText("")
            return
        if b is None:
            self._mess_lbl.setText(
                f"A = ({a[0]:.3f}  {a[1]:.3f}  {a[2]:.3f}) m — zweiten Punkt wählen")
            return
        d = np.asarray(b) - np.asarray(a)
        strecke = float(np.linalg.norm(d))
        waagerecht = float(np.linalg.norm(d[:2]))
        self._mess_lbl.setText(
            f"A = ({a[0]:.3f}  {a[1]:.3f}  {a[2]:.3f})    "
            f"B = ({b[0]:.3f}  {b[1]:.3f}  {b[2]:.3f})    "
            f"Abstand {strecke:.3f} m    waagerecht {waagerecht:.3f} m    "
            f"Höhe {d[2]:+.3f} m    ΔX {d[0]:+.3f}  ΔY {d[1]:+.3f}  ΔZ {d[2]:+.3f}")
        self._log(f"Messung: {strecke:.3f} m (waagerecht {waagerecht:.3f} m, "
                  f"Höhe {d[2]:+.3f} m)")

    def _on_toggle_preview(self) -> None:
        an = not self._cloud_view.preview_visible()
        self._cloud_view.set_preview_visible(an)
        self._sync_preview_action()

    def _sync_preview_action(self) -> None:
        act = self._actions.get("preview")
        if act is None:
            return
        still_setzen(act, self._cloud_view.preview_visible())
        act.setEnabled(self._cloud_view.has_preview())

    def _on_reset_camera(self) -> None:
        self._cloud_view.reset_camera()

    def _on_cut_reset(self) -> None:
        self._cloud_view.cut_bar.reset()

    def _on_screenshot(self) -> None:
        start = os.path.join(self._project.dir if self._project else "",
                             "ansicht.png")
        path = speicherpfad(self, "Ansicht speichern", start, "PNG-Datei (*.png)", ".png")
        if not path:
            return
        try:
            self._cloud_view.screenshot(path)
        except RuntimeError as exc:
            self._show_error("3D-Ansicht als Bild speichern", str(exc))
            return
        self._log(f"Ansicht gespeichert: {path}")

    def _on_about(self) -> None:
        QMessageBox.about(
            self, "Über Super360 Studio",
            "<b>Super360 Studio</b><br><br>"
            "Rosbag → FAST-LIO2-Punktwolke, 360°-Video, Einfärbung, "
            "Zusammenführen und Export.<br>"
            "Bis 2026-09 hieß das Programm „RosBag Suite 360\".<br><br>"
            "<a href='https://github.com/LenaKremer98/Super360Studio'>"
            "github.com/LenaKremer98/Super360Studio</a>")
