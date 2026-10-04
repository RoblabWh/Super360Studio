"""Einstellungen (Mixin des Hauptfensters).

Schreibt am Hauptfenster: _layer_key, _loading_ui, _meander_dir, _settings,
_temperatur_anzeigen.
"""
from __future__ import annotations

import os

from PyQt5.QtWidgets import QMessageBox

_DEFAULT_SETTINGS: dict = {
    "pano_width": 1920,
    "config": "whs_dense.yaml",
    "rate": 1.0,
    "brightness_min": 20,
    "brightness_max": 235,
    "k_frames": 3,
    "sky_grow": 4,
    "lens_best": True,
    "edge_r": 600,
    "blue_filter": False,
    "blue_hue_lo": 150,
    "blue_hue_hi": 290,
    "blue_sat": 5,
    "blue_val": 10,
    "blue_neutral": 100,
    "mesh_voxel_cm": 4,
    "mesh_depth": 12,
    "mesh_trim": 0,
    "mesh_hybrid": True,
    "mesh_an": False,
    "point_size": 2,
    "color_mode": "rgb",
    "layer": "onboard",
    "only_colored": True,
    "voxel": 0.0,
    "background": "dunkel",
    "edl": False,
    "show_path": False,
}


class EinstellungenMixin:
    # ========================================================== Einstellungen

    def _apply_settings_to_widgets(self) -> None:
        s = self._settings
        self._loading_ui = True
        try:
            idx = self._combo_config.findData(s.get("config", "whs_dense.yaml"))
            self._combo_config.setCurrentIndex(max(0, idx))
            self._spin_rate.setValue(float(s.get("rate", 1.0)))
            self._sld_bmin.setValue(int(s.get("brightness_min", 20)))
            self._sld_bmax.setValue(int(s.get("brightness_max", 235)))
            self._spin_kframes.setValue(int(s.get("k_frames", 3)))
            self._spin_sky.setValue(int(s.get("sky_grow", 4)))
            self._chk_lens.setChecked(bool(s.get("lens_best", True)))
            self._spin_edge.setValue(int(s.get("edge_r", 600)))
            self._chk_blue.setChecked(bool(s.get("blue_filter", False)))
            d = _DEFAULT_SETTINGS
            blau = [s.get(k, d[k]) for k in ("blue_hue_lo", "blue_hue_hi", "blue_sat",
                                               "blue_val")]
            if blau in ([200, 240, 40, 60], [170, 250, 25, 40]):
                # unveraenderte fruehere Standardwerte: liessen cyanblaues und
                # blassblaues Blaulicht durch, daher auf die neuen
                blau = [d[k] for k in ("blue_hue_lo", "blue_hue_hi", "blue_sat", "blue_val")]
            self._sld_blue_lo.setValue(int(blau[0]))
            self._sld_blue_hi.setValue(int(blau[1]))
            self._sld_blue_sat.setValue(int(blau[2]))
            self._sld_blue_val.setValue(int(blau[3]))
            self._sld_blue_neutral.setValue(int(s.get("blue_neutral", d["blue_neutral"])))
            self._spin_mesh_voxel.setValue(int(s.get("mesh_voxel_cm", d["mesh_voxel_cm"])))
            self._spin_mesh_depth.setValue(int(s.get("mesh_depth", d["mesh_depth"])))
            stand = int(s.get("mesh_stand", 0) or 0)
            trim = int(s.get("mesh_trim", d["mesh_trim"]))
            if trim == 5 and stand < 2:
                trim = 0      # frueherer Standard, stanzte kleine Loecher
            self._spin_mesh_trim.setValue(trim)
            if stand < 3 and (self._spin_mesh_voxel.value(), self._spin_mesh_depth.value()) \
                    == (5, 11):
                # fruehere Standardwerte: zu grob (Poisson-Zelle ~12 cm)
                self._spin_mesh_voxel.setValue(int(d["mesh_voxel_cm"]))
                self._spin_mesh_depth.setValue(int(d["mesh_depth"]))
            self._chk_mesh_hybrid.setChecked(bool(s.get("mesh_hybrid", d["mesh_hybrid"])))
            self._cloud_view.set_mesh_schalter(bool(s.get("mesh_an", False)))
            self._spin_pointsize.setValue(float(s.get("point_size", 2.0)))
            self._temperatur_anzeigen = bool(s.get("temperatur_anzeigen", True))
            self._cloud_view.set_temperatur_anzeigen(self._temperatur_anzeigen)
            idx = self._combo_colormode.findData(s.get("color_mode", "rgb"))
            self._combo_colormode.setCurrentIndex(max(0, idx))
            self._chk_only_colored.setChecked(bool(s.get("only_colored", False)))
            voxel = float(s.get("voxel", 0.0))
            idx = next((i for i in range(self._combo_voxel.count())
                        if abs(self._combo_voxel.itemData(i) - voxel) < 1e-9), 0)
            self._combo_voxel.setCurrentIndex(idx)
            idx = self._combo_bg.findData(s.get("background", "dunkel"))
            self._combo_bg.setCurrentIndex(max(0, idx))
            self._chk_edl.setChecked(bool(s.get("edl", False)) and self._chk_edl.isEnabled())
            self._chk_path.setChecked(bool(s.get("show_path", False)))
            self._chk_thermal.setChecked(bool(s.get("meander_thermal", False)))
            self._chk_sichtbar.setChecked(bool(s.get("meander_sichtbar", True)))
            self._chk_solo.setChecked(bool(s.get("meander_solo", True)))
            raster = float(s.get("splat_raster", 0.05))
            idx = next((i for i in range(self._combo_splat_raster.count())
                        if abs(self._combo_splat_raster.itemData(i) - raster) < 1e-9), 2)
            self._combo_splat_raster.setCurrentIndex(idx)
            self._spin_splat_anker.setValue(float(s.get("splat_anker_mio", 4.0)))
            self._combo_splat_sh.setCurrentIndex(
                max(0, self._combo_splat_sh.findData(int(s.get("splat_sh", 1)))))
            self._chk_splat_posen.setChecked(bool(s.get("splat_posen", True)))
            self._chk_splat_pruefen.setChecked(bool(s.get("splat_pruefen", True)))
            self._chk_splat_thermal.setChecked(bool(s.get("splat_thermal", True)))
            self._spin_splat_schritte.setValue(int(s.get("splat_schritte", 3000)))
            self._spin_splat_frames.setValue(int(s.get("splat_frames", 300)))
            self._spin_splat_schritte_onboard.setValue(
                int(s.get("splat_schritte_onboard", 15000)))
            for optik, key in (("rgb", "rgb_versatz"),
                               ("thermal", "thermal_versatz")):
                v = s.get(key) or [0.0, 0.0]
                self._spin_optik[(optik, "u")].setValue(float(v[0]))
                self._spin_optik[(optik, "v")].setValue(float(v[1]))
            d = s.get("meander_dir") or ""
            if d and os.path.isdir(d):
                self._meander_dir = d
                self._lbl_meander.setText(f"{os.path.basename(d)} (aus den "
                                          f"Einstellungen)")
            self._layer_key = s.get("layer", "onboard")
            self._sections.set_states(s.get("sections") or {})
            self._set_sidebar_visible(bool(s.get("sidebar", True)))
        finally:
            self._loading_ui = False
        self._push_display_settings()

    def _collect_settings(self) -> dict:
        return {
            "pano_width": int(self._settings.get("pano_width", 1920)),
            "config": self._combo_config.currentData(),
            "rate": float(self._spin_rate.value()),
            "brightness_min": int(self._sld_bmin.value()),
            "brightness_max": int(self._sld_bmax.value()),
            "k_frames": int(self._spin_kframes.value()),
            "sky_grow": int(self._spin_sky.value()),
            "lens_best": bool(self._chk_lens.isChecked()),
            "edge_r": int(self._spin_edge.value()),
            "blue_filter": bool(self._chk_blue.isChecked()),
            "blue_hue_lo": int(self._sld_blue_lo.value()),
            "blue_hue_hi": int(self._sld_blue_hi.value()),
            "blue_sat": int(self._sld_blue_sat.value()),
            "blue_val": int(self._sld_blue_val.value()),
            "blue_neutral": int(self._sld_blue_neutral.value()),
            "mesh_voxel_cm": int(self._spin_mesh_voxel.value()),
            "mesh_depth": int(self._spin_mesh_depth.value()),
            "mesh_trim": int(self._spin_mesh_trim.value()),
            "mesh_an": bool(self._cloud_view.mesh_an()),
            # Marke, ab der mesh_trim bewusst gesetzt ist (nicht in den
            # Standardwerten, sonst griffe die Uebernahme oben nie)
            "mesh_stand": 3,
            "mesh_hybrid": bool(self._chk_mesh_hybrid.isChecked()),
            "point_size": float(self._spin_pointsize.value()),
            "temperatur_anzeigen": bool(self._temperatur_anzeigen),
            "color_mode": self._combo_colormode.currentData(),
            "layer": self._layer_key,
            "meander_dir": self._meander_dir or "",
            "meander_thermal": bool(self._chk_thermal.isChecked()),
            "meander_sichtbar": bool(self._chk_sichtbar.isChecked()),
            "meander_solo": bool(self._chk_solo.isChecked()),
            "splat_raster": float(self._combo_splat_raster.currentData()),
            "splat_anker_mio": float(self._spin_splat_anker.value()),
            "splat_sh": int(self._combo_splat_sh.currentData()),
            "splat_posen": bool(self._chk_splat_posen.isChecked()),
            "splat_pruefen": bool(self._chk_splat_pruefen.isChecked()),
            "splat_thermal": bool(self._chk_splat_thermal.isChecked()),
            "splat_schritte": int(self._spin_splat_schritte.value()),
            "splat_frames": int(self._spin_splat_frames.value()),
            "splat_schritte_onboard": int(self._spin_splat_schritte_onboard.value()),
            "rgb_versatz": self._meander_versatz("rgb"),
            "thermal_versatz": self._meander_versatz("thermal"),
            "only_colored": bool(self._chk_only_colored.isChecked()),
            "voxel": float(self._combo_voxel.currentData()),
            "background": self._combo_bg.currentData(),
            "edl": bool(self._chk_edl.isChecked()),
            "show_path": bool(self._chk_path.isChecked()),
            "sections": self._sections.states(),
            "sidebar": bool(self._sidebar_scroll.isVisible()),
        }

    def _save_settings(self) -> None:
        if self._loading_ui or self._project is None:
            return
        self._settings = self._collect_settings()
        try:
            self._project.save_settings(self._settings)
        except RuntimeError as exc:
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
