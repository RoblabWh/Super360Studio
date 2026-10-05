"""Einstellungen (Mixin des Hauptfensters).

Schreibt am Hauptfenster: _color_mode, _layer_key, _loading_ui, _meander_dir,
_meander_solo, _point_size, _settings, _temperatur_anzeigen.
"""
from __future__ import annotations

import os
from typing import NamedTuple

from PyQt5.QtWidgets import QMessageBox

from ui.fenster.anzeige import _farbmodus, _punktgroesse


class _Einstellung(NamedTuple):
    """Eine Zeile der Einstellungstabelle.

    ``bindung`` ist das Widget- oder Zustandsattribut des Fensters, ``lesart``
    sagt, wie Wert und Bindung ineinander uebergehen (``_SCHREIBEN``/``_LESEN``).
    ``art``: ``vorbelegt`` steht in _DEFAULT_SETTINGS; ``rueckfall`` nicht, dann
    gilt die Vorgabe, wenn der Schluessel fehlt; ``nur_vorhanden`` laesst den
    Zustand stehen, wenn der Schluessel fehlt; ``fest`` wird immer mit der
    Vorgabe gespeichert und nie angewandt.
    """
    schluessel: str
    vorgabe: object
    bindung: str
    lesart: str
    art: str = "vorbelegt"


# In der Reihenfolge, in der die Einstellungen angewandt werden.
_EINSTELLUNGEN: tuple = (
    # ohne Wirkung (die Breite ist fest), wird nur gelesen und weiter gespeichert
    _Einstellung("pano_width", 1920, "_settings", "durchreichen"),
    _Einstellung("config", "whs_dense.yaml", "_combo_config", "daten"),
    _Einstellung("rate", 1.0, "_spin_rate", "komma"),
    _Einstellung("brightness_min", 20, "_sld_bmin", "zahl"),
    _Einstellung("brightness_max", 235, "_sld_bmax", "zahl"),
    _Einstellung("k_frames", 3, "_spin_kframes", "zahl"),
    _Einstellung("sky_grow", 4, "_spin_sky", "zahl"),
    _Einstellung("lens_best", True, "_chk_lens", "haken"),
    _Einstellung("edge_r", 600, "_spin_edge", "zahl"),
    _Einstellung("blue_filter", False, "_chk_blue", "haken"),
    _Einstellung("blue_hue_lo", 150, "_sld_blue_lo", "zahl"),
    _Einstellung("blue_hue_hi", 290, "_sld_blue_hi", "zahl"),
    _Einstellung("blue_sat", 5, "_sld_blue_sat", "zahl"),
    _Einstellung("blue_val", 10, "_sld_blue_val", "zahl"),
    _Einstellung("blue_neutral", 100, "_sld_blue_neutral", "zahl"),
    _Einstellung("mesh_voxel_cm", 4, "_spin_mesh_voxel", "zahl"),
    _Einstellung("mesh_depth", 12, "_spin_mesh_depth", "zahl"),
    _Einstellung("mesh_trim", 0, "_spin_mesh_trim", "zahl"),
    # ohne Wirkung, wird immer mit der Vorgabe gespeichert
    _Einstellung("mesh_stand", 3, "", "", "fest"),
    _Einstellung("mesh_hybrid", True, "_chk_mesh_hybrid", "haken"),
    _Einstellung("mesh_an", False, "_cloud_view", "mesh_schalter"),
    _Einstellung("point_size", 2, "_point_size", "punktgroesse"),
    _Einstellung("temperatur_anzeigen", True, "_temperatur_anzeigen", "temperatur",
                 "rueckfall"),
    _Einstellung("color_mode", "rgb", "_color_mode", "farbmodus"),
    _Einstellung("only_colored", True, "_chk_only_colored", "haken"),
    _Einstellung("voxel", 0.0, "_combo_voxel", "naeherung"),
    _Einstellung("background", "dunkel", "_combo_bg", "daten"),
    _Einstellung("edl", False, "_chk_edl", "haken_wenn_frei"),
    _Einstellung("show_path", False, "_chk_path", "haken"),
    _Einstellung("meander_thermal", False, "_chk_thermal", "haken", "rueckfall"),
    _Einstellung("meander_sichtbar", True, "_chk_sichtbar", "haken", "rueckfall"),
    # ohne Wirkung, wird nur gelesen und weiter gespeichert
    _Einstellung("meander_solo", True, "_meander_solo", "merker", "rueckfall"),
    _Einstellung("splat_raster", 0.05, "_combo_splat_raster", "naeherung", "rueckfall"),
    _Einstellung("splat_anker_mio", 4.0, "_spin_splat_anker", "komma", "rueckfall"),
    _Einstellung("splat_sh", 1, "_combo_splat_sh", "daten_ganz", "rueckfall"),
    _Einstellung("splat_posen", True, "_chk_splat_posen", "haken", "rueckfall"),
    _Einstellung("splat_pruefen", True, "_chk_splat_pruefen", "haken", "rueckfall"),
    _Einstellung("splat_thermal", True, "_chk_splat_thermal", "haken", "rueckfall"),
    _Einstellung("splat_schritte", 3000, "_spin_splat_schritte", "zahl", "rueckfall"),
    _Einstellung("splat_frames", 300, "_spin_splat_frames", "zahl", "rueckfall"),
    _Einstellung("splat_schritte_onboard", 15000, "_spin_splat_schritte_onboard", "zahl",
                 "rueckfall"),
    _Einstellung("rgb_versatz", [0.0, 0.0], "_spin_optik", "versatz", "rueckfall"),
    _Einstellung("thermal_versatz", [0.0, 0.0], "_spin_optik", "versatz", "rueckfall"),
    _Einstellung("meander_dir", "", "_meander_dir", "maeanderordner", "nur_vorhanden"),
    _Einstellung("layer", "onboard", "_layer_key", "ebene"),
    _Einstellung("sections", {}, "_sections", "abschnitte", "nur_vorhanden"),
    _Einstellung("sidebar", True, "_sidebar_scroll", "seitenleiste", "rueckfall"),
)

_DEFAULT_SETTINGS: dict = {e.schluessel: e.vorgabe for e in _EINSTELLUNGEN
                           if e.art == "vorbelegt"}


def _auswahl_index(combo, wert: float, vorgabe: float) -> int:
    """Eintrag mit diesem Wert, sonst der mit der Vorgabe, sonst der erste."""
    for ziel in (float(wert), vorgabe):
        idx = next((i for i in range(combo.count())
                    if abs(combo.itemData(i) - ziel) < 1e-9), None)
        if idx is not None:
            return idx
    return 0


def _optik(e: _Einstellung) -> str:
    return e.schluessel.split("_")[0]


def _setze_temperatur(f, wert) -> None:
    f._temperatur_anzeigen = bool(wert)
    f._cloud_view.set_temperatur_anzeigen(f._temperatur_anzeigen)


def _setze_versatz(f, e: _Einstellung, wert) -> None:
    v = wert or [0.0, 0.0]
    f._spin_optik[(_optik(e), "u")].setValue(float(v[0]))
    f._spin_optik[(_optik(e), "v")].setValue(float(v[1]))


def _setze_maeanderordner(f, wert) -> None:
    d = wert or ""
    if d and os.path.isdir(d):
        f._meander_dir = d
        f._lbl_meander.setText(f"{os.path.basename(d)} (aus den "
                               f"Einstellungen)")
        # wie nach „Mäanderflug wählen …“: ohne Thermalbilder kein Haken
        try:
            n_t = len([n for n in os.listdir(d) if n.upper().endswith("_T.JPG")])
        except OSError as exc:
            # nicht lesbar (Rechte, abgezogener Datenträger): Haken frei lassen,
            # das Öffnen des Projekts darf daran nicht scheitern
            f._log(f"Mäanderordner nicht lesbar: {exc}")
            return
        f._chk_thermal.setEnabled(n_t > 0)
        if n_t == 0:
            f._chk_thermal.setChecked(False)


def _setze_ebene(f, wert) -> None:
    f._layer_key = wert


# lesart -> (Fenster, Zeile, Wert) setzt die Bindung
_SCHREIBEN: dict = {
    "durchreichen": lambda f, e, w: None,
    "zahl": lambda f, e, w: getattr(f, e.bindung).setValue(int(w)),
    "komma": lambda f, e, w: getattr(f, e.bindung).setValue(float(w)),
    "haken": lambda f, e, w: getattr(f, e.bindung).setChecked(bool(w)),
    "haken_wenn_frei": lambda f, e, w: getattr(f, e.bindung).setChecked(
        bool(w) and getattr(f, e.bindung).isEnabled()),
    "daten": lambda f, e, w: getattr(f, e.bindung).setCurrentIndex(
        max(0, getattr(f, e.bindung).findData(w))),
    "daten_ganz": lambda f, e, w: getattr(f, e.bindung).setCurrentIndex(
        max(0, getattr(f, e.bindung).findData(int(w)))),
    "naeherung": lambda f, e, w: getattr(f, e.bindung).setCurrentIndex(
        _auswahl_index(getattr(f, e.bindung), w, e.vorgabe)),
    "mesh_schalter": lambda f, e, w: f._cloud_view.set_mesh_schalter(bool(w)),
    "punktgroesse": lambda f, e, w: setattr(f, e.bindung, _punktgroesse(w)),
    "farbmodus": lambda f, e, w: setattr(f, e.bindung, _farbmodus(w)),
    "merker": lambda f, e, w: setattr(f, e.bindung, bool(w)),
    "temperatur": lambda f, e, w: _setze_temperatur(f, w),
    "versatz": _setze_versatz,
    "maeanderordner": lambda f, e, w: _setze_maeanderordner(f, w),
    "ebene": lambda f, e, w: _setze_ebene(f, w),
    "abschnitte": lambda f, e, w: f._sections.set_states(w or {}),
    "seitenleiste": lambda f, e, w: f._set_sidebar_visible(bool(w)),
}

# lesart -> (Fenster, Zeile) liefert den zu speichernden Wert
_LESEN: dict = {
    "durchreichen": lambda f, e: int(f._settings.get(e.schluessel, e.vorgabe)),
    "zahl": lambda f, e: int(getattr(f, e.bindung).value()),
    "komma": lambda f, e: float(getattr(f, e.bindung).value()),
    "haken": lambda f, e: bool(getattr(f, e.bindung).isChecked()),
    "haken_wenn_frei": lambda f, e: bool(getattr(f, e.bindung).isChecked()),
    "daten": lambda f, e: getattr(f, e.bindung).currentData(),
    "daten_ganz": lambda f, e: int(getattr(f, e.bindung).currentData()),
    "naeherung": lambda f, e: float(getattr(f, e.bindung).currentData()),
    "mesh_schalter": lambda f, e: bool(f._cloud_view.mesh_an()),
    "punktgroesse": lambda f, e: float(getattr(f, e.bindung)),
    "farbmodus": lambda f, e: getattr(f, e.bindung),
    "merker": lambda f, e: bool(getattr(f, e.bindung)),
    "temperatur": lambda f, e: bool(f._temperatur_anzeigen),
    "versatz": lambda f, e: f._meander_versatz(_optik(e)),
    "maeanderordner": lambda f, e: f._meander_dir or "",
    "ebene": lambda f, e: f._layer_key,
    "abschnitte": lambda f, e: f._sections.states(),
    # isHidden statt isVisible: vor dem ersten show() ist nichts sichtbar
    "seitenleiste": lambda f, e: not f._sidebar_scroll.isHidden(),
}


class EinstellungenMixin:
    # ========================================================== Einstellungen

    def _apply_settings_to_widgets(self) -> None:
        werte = dict(self._settings)
        self._loading_ui = True
        try:
            for e in _EINSTELLUNGEN:
                if e.art == "fest" or (e.art == "nur_vorhanden"
                                       and e.schluessel not in werte):
                    continue
                _SCHREIBEN[e.lesart](self, e, werte.get(e.schluessel, e.vorgabe))
        finally:
            self._loading_ui = False
        self._push_display_settings()

    def _collect_settings(self) -> dict:
        return {e.schluessel: e.vorgabe if e.art == "fest" else _LESEN[e.lesart](self, e)
                for e in _EINSTELLUNGEN}

    def _save_settings(self) -> None:
        if self._loading_ui or self._project is None:
            return
        self._settings = self._collect_settings()
        try:
            self._project.save_settings(self._settings)
        except (OSError, RuntimeError) as exc:
            self._log(f"Einstellungen nicht gespeichert: {exc}")

    def _on_setting_changed(self, *_a) -> None:
        self._save_settings()

    def _on_settings_reset(self) -> None:
        if QMessageBox.question(
                self, "Einstellungen zurücksetzen",
                "Alle Einstellungen dieses Projekts auf die Vorgabe setzen?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        self._settings = dict(_DEFAULT_SETTINGS)
        self._apply_settings_to_widgets()
        self._save_settings()
        self._log("Einstellungen auf Vorgabe gesetzt.")
