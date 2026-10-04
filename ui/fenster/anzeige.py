"""Anzeige, Farbebenen und Menü (Mixin des Hauptfensters).

Schreibt am Hauptfenster: _chk_edl, _chk_only_colored, _chk_path, _colors,
_combo_bg, _combo_colormode, _combo_layer, _combo_voxel, _layer_key, _layers,
_loading_ui, _spin_pointsize, _temperatur, _temperatur_anzeigen, _temperaturen,
_valid.
"""
from __future__ import annotations

import os

import numpy as np

from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QMessageBox, QWidget,
)

from core.ebenen import THERMAL, lade_farbdateien, lade_temperatur, laden
from core.project import Project

from ui import menubar as menubar_mod
from ui.bausteine import _compact_combo, _wrappable, speicherpfad, still_setzen

#: Farbebenen fuer die Auswahl — Reihenfolge wie in Project.LAYERS
_LAYER_LABELS = (
    ("onboard", "Onboard RGB (360°-Kamera)"),
    ("onboard_splat", "Onboard RGB, Gaussian Splat"),
    ("meander_rgb", "Mäander RGB (DJI)"),
    ("meander_splat", "Mäander RGB, Gaussian Splat"),
    ("meander_thermal", "Mäander Thermal (DJI)"),
    ("meander_thermal_splat", "Mäander Temperatur, Gaussian Splat"),
    ("fusion", "Fusion Onboard + Mäander"),
    ("fusion_splat", "Fusion, gemeinsames Gaussian Splat"),
)
_COLOR_MODE_ITEMS = (("RGB (eingefärbt)", "rgb"), ("Höhe", "hoehe"),
                     ("Intensität", "intensitaet"), ("Einfarbig", "uniform"))
#: Farbmodi der Leiste ueber der 3D-Ansicht: Farbmodus und Farbquelle in einem
_FARBLEISTE = (("uniform", "Einheitsfarbe"), ("intensitaet", "Intensität"),
               ("hoehe", "Höhe"), ("rgb:onboard", "RGB Onboard"),
               ("rgb:onboard_splat", "RGB Onboard (Splat)"),
               ("rgb:meander_rgb", "RGB Mäander"),
               ("rgb:meander_splat", "RGB Mäander (Splat)"),
               ("rgb:meander_thermal", "Thermal Mäander"),
               ("rgb:meander_thermal_splat", "Temperatur Mäander (Splat)"),
               ("rgb:fusion", "RGB Fusion"),
               ("rgb:fusion_splat", "RGB Fusion (Splat)"))
_VOXEL_ITEMS = (("Aus", 0.0), ("0,05 m", 0.05), ("0,10 m", 0.10), ("0,20 m", 0.20))
_BG_ITEMS = (("Dunkel", "dunkel"), ("Hell", "hell"))


class AnzeigeMixin:
    def _group_display(self) -> QWidget:
        box = QWidget()
        form = _wrappable(QFormLayout(box))
        self._spin_pointsize = QDoubleSpinBox()
        self._spin_pointsize.setRange(0.5, 10.0)
        self._spin_pointsize.setSingleStep(0.25)
        self._spin_pointsize.setDecimals(2)
        self._spin_pointsize.setSuffix(" px")
        self._spin_pointsize.setValue(2.0)
        self._spin_pointsize.valueChanged.connect(self._on_display_changed)
        form.addRow("Punktgröße:", self._spin_pointsize)
        self._combo_colormode = _compact_combo(QComboBox())
        for label, data in _COLOR_MODE_ITEMS:
            self._combo_colormode.addItem(label, data)
        self._combo_colormode.currentIndexChanged.connect(self._on_display_changed)
        form.addRow("Farbmodus:", self._combo_colormode)
        self._combo_layer = _compact_combo(QComboBox())
        self._combo_layer.setToolTip(
            "Welche Einfärbung gezeigt wird. Angeboten wird, was berechnet ist.")
        for data, label in _LAYER_LABELS:
            self._combo_layer.addItem(label, data)
        self._combo_layer.currentIndexChanged.connect(self._on_layer_changed)
        form.addRow("Farbquelle:", self._combo_layer)
        self._chk_only_colored = QCheckBox("Nur eingefärbte Punkte")
        self._chk_only_colored.toggled.connect(self._on_display_changed)
        form.addRow(self._chk_only_colored)
        self._combo_voxel = _compact_combo(QComboBox())
        for label, data in _VOXEL_ITEMS:
            self._combo_voxel.addItem(label, data)
        self._combo_voxel.currentIndexChanged.connect(self._on_display_changed)
        form.addRow("Anzeige-Voxel:", self._combo_voxel)
        self._combo_bg = _compact_combo(QComboBox())
        for label, data in _BG_ITEMS:
            self._combo_bg.addItem(label, data)
        self._combo_bg.currentIndexChanged.connect(self._on_display_changed)
        form.addRow("Hintergrund:", self._combo_bg)
        self._chk_edl = QCheckBox("EDL (Eye-Dome Lighting)")
        self._chk_edl.toggled.connect(self._on_display_changed)
        # Verfügbarkeit wird nach dem Bau der CloudView geprüft (s. _build_ui).
        form.addRow(self._chk_edl)
        self._chk_path = QCheckBox("Trajektorie zeigen")
        self._chk_path.toggled.connect(self._on_display_changed)
        form.addRow(self._chk_path)
        return box

    def _sync_after_display(self) -> None:
        if hasattr(self, "_actions"):
            self._sync_menu_state()

    def _push_display_settings(self) -> None:
        cv = self._cloud_view
        cv.set_point_size(float(self._spin_pointsize.value()))
        cv.set_color_mode(self._combo_colormode.currentData())
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
        modus = self._combo_colormode.currentData()
        return f"rgb:{self._layer_key}" if modus == "rgb" else str(modus)

    def _sync_farbleiste(self) -> None:
        """Auswahl der Leiste an Seitenleiste und vorhandene Ebenen angleichen."""
        hat_int = self._rec is not None and getattr(self._rec, "intensity", None) is not None
        eintraege = []
        for key, text in _FARBLEISTE:
            if key.startswith("rgb:"):
                da = key[4:] in self._layers
            elif key == "intensitaet":
                da = hat_int
            else:
                da = True
            eintraege.append((key, text, da))
        self._cloud_view.set_farbmodi(eintraege, self._farbleiste_key())

    def _on_farbleiste(self, key: str) -> None:
        """Farbmodus aus der Leiste: setzt Farbmodus und -quelle der Seitenleiste."""
        if key.startswith("rgb:"):
            ebene = key[4:]
            if ebene != self._layer_key:
                i = self._combo_layer.findData(ebene)
                if i >= 0:
                    self._loading_ui = True
                    self._combo_layer.setCurrentIndex(i)
                    self._loading_ui = False
                    self._layer_key = ebene
                    self._apply_layer()
                    if hasattr(self, "_actions"):
                        self._fill_layer_menu()
            modus = "rgb"
        else:
            modus = key
        i = self._combo_colormode.findData(modus)
        if i >= 0 and i != self._combo_colormode.currentIndex():
            self._combo_colormode.setCurrentIndex(i)      # loest die Anzeige aus
        else:
            self._push_display_settings()
            self._sync_after_display()
        self._save_settings()

    def _on_leiste_punktgroesse(self, wert: float) -> None:
        still_setzen(self._spin_pointsize, float(wert))
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
        weg statt die Anzeige zu verfaelschen. Die Auswahlliste zeigt danach
        nur, was wirklich da ist.
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
                except Exception:  # noqa: BLE001
                    fp = None
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

    def _ebene_waehlen(self, key) -> bool:
        """Eine geladene Ebene als Farbquelle waehlen, wie von Hand.

        Speichert und protokolliert ueber ``_on_layer_changed``; der Farbmodus
        bleibt. False, wenn die Ebene nicht geladen ist.
        """
        return key in self._layers and self._waehle(self._combo_layer, key)

    def _waehle(self, combo, data) -> bool:
        """Den Eintrag mit diesen Daten waehlen (mit Signal); False, wenn er fehlt."""
        idx = combo.findData(data)
        if idx < 0:
            return False
        combo.setCurrentIndex(idx)
        return True

    def _refresh_layer_combo(self) -> None:
        """Auswahlliste auf die vorhandenen Ebenen setzen."""
        alt = self._layer_key
        self._loading_ui = True
        try:
            self._combo_layer.clear()
            for data, label in _LAYER_LABELS:
                if data in self._layers:
                    self._combo_layer.addItem(label, data)
            if self._combo_layer.count() == 0:
                self._combo_layer.addItem("keine Einfärbung", "onboard")
                self._combo_layer.setEnabled(False)
            else:
                self._combo_layer.setEnabled(True)
            idx = self._combo_layer.findData(alt)
            if idx < 0:
                idx = 0
            self._combo_layer.setCurrentIndex(idx)
            self._layer_key = self._combo_layer.currentData() or "onboard"
        finally:
            self._loading_ui = False
        self._apply_layer()
        if hasattr(self, "_actions"):
            self._fill_layer_menu()

    def _fill_layer_menu(self) -> None:
        menubar_mod.fill_radio_menu(
            self._actions["menu_farbquelle"], self,
            [(self._combo_layer.itemData(i), self._combo_layer.itemText(i))
             for i in range(self._combo_layer.count())],
            self._on_menu_layer, self._layer_key)

    def _on_menu_layer(self, data) -> None:
        self._waehle(self._combo_layer, data)

    def _on_layer_changed(self, *_a) -> None:
        if self._loading_ui:
            return
        self._layer_key = self._combo_layer.currentData() or "onboard"
        self._apply_layer()
        self._save_settings()
        self._log(f"Farbquelle: {self._combo_layer.currentText()}")

    def _apply_layer(self) -> None:
        """Die gewaehlte Ebene in die Ansicht schieben.

        Fehlt fuer die gewaehlte Quelle eine Einfaerbung, waere "RGB" eine
        einfarbig graue Wolke — richtig gerechnet, aber nichtssagend. Dann
        lieber auf Hoehe umschalten und es sagen.
        """
        paar = self._layers.get(self._layer_key)
        self._colors, self._valid = paar if paar else (None, None)
        # Die Maus zeigt die Temperatur der angezeigten Thermalebene; in jeder
        # anderen Ebene die der direkten, sonst die des Splats.
        self._temperatur = self._temperaturen.get(self._layer_key)
        if self._temperatur is None:
            self._temperatur = next((self._temperaturen[k] for k in THERMAL
                                     if k in self._temperaturen), None)
        self._cloud_view.set_temperatur(self._temperatur)
        if self._world is None:
            return
        if self._colors is None and self._combo_colormode.currentData() == "rgb":
            self._loading_ui = True
            gewaehlt = self._waehle(self._combo_colormode, "hoehe")
            self._loading_ui = False
            if gewaehlt:
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
        self._actions["sidebar"].setChecked(self._sidebar_scroll.isVisible())

    def _sync_menu_state(self) -> None:
        """Haken im Menue an die Seitenleiste angleichen (ohne Rueckkopplung)."""
        for key, combo in (("menu_farbquelle", self._combo_colormode),
                           ("menu_hintergrund", self._combo_bg)):
            menu = self._actions.get(key)
            if menu is None:
                continue
            for act in menu.actions():
                act.setChecked(act.data() == combo.currentData())
        edl = self._actions.get("edl")
        if edl is not None:
            still_setzen(edl, self._chk_edl.isChecked())

    def _on_menu_background(self, data) -> None:
        self._waehle(self._combo_bg, data)

    def _on_menu_edl(self, on: bool) -> None:
        if self._chk_edl.isEnabled():
            self._chk_edl.setChecked(bool(on))

    def _set_sidebar_visible(self, on: bool) -> None:
        self._sidebar_scroll.setVisible(bool(on))
        act = self._actions.get("sidebar") if hasattr(self, "_actions") else None
        if act is not None:
            still_setzen(act, bool(on))

    def _on_toggle_sidebar(self) -> None:
        self._set_sidebar_visible(not self._sidebar_scroll.isVisible())
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
        act = self._actions.get("preview") if hasattr(self, "_actions") else None
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
            self._show_error("Screenshot", str(exc))
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
