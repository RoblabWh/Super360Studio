"""360°-Einfärbung (Mixin des Hauptfensters).

Schreibt am Hauptfenster: _blue_widgets, _btn_blue_defaults, _btn_blue_preview,
_btn_colorize, _chk_blue, _chk_lens, _colors, _lbl_blue_band, _loading_ui,
_overlay_dialogs, _sld_blue_hi, _sld_blue_lo, _sld_blue_neutral, _sld_blue_sat,
_sld_blue_val, _sld_bmax, _sld_bmin, _spin_edge, _spin_kframes, _spin_sky,
_valid.
"""
from __future__ import annotations

import os

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox, QFormLayout, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QSlider, QSpinBox, QWidget,
)

from core.ebenen import lade_farbdateien as _load_color_files
from core.gemeinsam import fmt_int as _fmt_int

from ui.bausteine import _ImageDialog, _wrappable
from ui.fenster.einstellungen import _DEFAULT_SETTINGS


class EinfaerbungMixin:
    def _slider_row(self, minimum: int, maximum: int, value: int
                    ) -> tuple[QSlider, QLabel, QWidget]:
        holder = QWidget()
        lay = QHBoxLayout(holder)
        lay.setContentsMargins(0, 0, 0, 0)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        lbl = QLabel(str(value))
        lbl.setMinimumWidth(30)
        lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        slider.valueChanged.connect(lambda v, l=lbl: l.setText(str(v)))
        lay.addWidget(slider, 1)
        lay.addWidget(lbl)
        return slider, lbl, holder

    def _update_blue_widgets(self, *_args) -> None:
        """Farbtonband neu zeichnen, Blaulicht-Regler nur mit Filter aktiv."""
        an = self._chk_blue.isChecked()
        for w in self._blue_widgets:
            w.setEnabled(an)
        lo, hi = self._sld_blue_lo.value(), self._sld_blue_hi.value()
        w, h = 240, 14
        hue = np.linspace(0.0, 360.0, w, endpoint=False, dtype=np.float32)
        drin = (hue >= lo) & (hue <= hi) if lo <= hi else (hue >= lo) | (hue <= hi)
        hsv = np.stack([hue, np.ones(w, np.float32),
                        np.where(drin, 1.0, 0.3).astype(np.float32)], axis=1)
        import cv2
        rgb = cv2.cvtColor(hsv.reshape(1, w, 3), cv2.COLOR_HSV2RGB)
        rgb = np.ascontiguousarray(
            np.repeat((rgb * 255).astype(np.uint8), h, axis=0))
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self._lbl_blue_band.setPixmap(QPixmap.fromImage(img))

    def _blue_args(self) -> dict:
        return {"blue_hue_lo": float(self._sld_blue_lo.value()),
                "blue_hue_hi": float(self._sld_blue_hi.value()),
                "blue_sat": float(self._sld_blue_sat.value()) / 100.0,
                "blue_val": float(self._sld_blue_val.value())}

    def _on_blue_defaults(self) -> None:
        """Blaulicht-Regler auf die Standardwerte (gemessen am Nachtflug)."""
        d = _DEFAULT_SETTINGS
        self._sld_blue_lo.setValue(int(d["blue_hue_lo"]))
        self._sld_blue_hi.setValue(int(d["blue_hue_hi"]))
        self._sld_blue_sat.setValue(int(d["blue_sat"]))
        self._sld_blue_val.setValue(int(d["blue_val"]))
        self._sld_blue_neutral.setValue(int(d["blue_neutral"]))

    def _on_blue_preview_clicked(self) -> None:
        if self._bag is None:
            return
        colorizer = self._mit_colorizer("Blaumaske")
        if colorizer is None:
            return
        frame_idx = self._aktueller_frame()
        a = self._blue_args()
        try:
            img = self._bag.read_camera(frame_idx)
        except Exception as exc:  # noqa: BLE001 - Bag-Lesefehler anzeigen
            self._show_error("Blaumaske", f"Frame {frame_idx} nicht lesbar: {exc}")
            return
        bild, anteil = colorizer.blue_preview(
            img, a["blue_hue_lo"], a["blue_hue_hi"], a["blue_sat"], a["blue_val"])
        self._zeige_bild(f"Blaumaske — Frame {frame_idx}: {100.0 * anteil:.1f} % "
                         f"der Pixel gelten als Blaulicht", bild)

    def _abschnitt_einfaerbung(self) -> QWidget:
        box = QWidget()
        form = _wrappable(QFormLayout(box))
        self._sld_bmin, _, row_min = self._slider_row(0, 255, 20)
        self._sld_bmax, _, row_max = self._slider_row(0, 255, 235)
        self._sld_bmin.valueChanged.connect(self._on_setting_changed)
        self._sld_bmax.valueChanged.connect(self._on_setting_changed)
        form.addRow("Helligkeit min:", row_min)
        form.addRow("Helligkeit max:", row_max)
        self._spin_kframes = QSpinBox()
        self._spin_kframes.setRange(1, 10)
        self._spin_kframes.setValue(3)
        self._spin_kframes.valueChanged.connect(self._on_setting_changed)
        form.addRow("K Frames:", self._spin_kframes)
        self._spin_sky = QSpinBox()
        self._spin_sky.setRange(0, 20)
        self._spin_sky.setValue(4)
        self._spin_sky.setSuffix(" px")
        self._spin_sky.setToolTip(
            "Sperrt den Saum um ausgebrannte Himmelsflächen. Dort mischen Blur und\n"
            "Farbsaum Himmel und Objekt zu Grauweiß, das unter 'Helligkeit max'\n"
            "durchrutscht und Baumkronen weiß überzieht. 0 schaltet die Sperre ab.")
        self._spin_sky.valueChanged.connect(self._on_setting_changed)
        form.addRow("Himmelssaum:", self._spin_sky)

        # Linsenrand: dort ist das Bild vignettiert und kippt ins Gruenblaue,
        # an der Grenze der beiden Fisheyes entstehen daraus sichtbare Kanten.
        self._chk_lens = QCheckBox("Bessere Linse je Frame")
        self._chk_lens.setChecked(True)
        self._chk_lens.setToolTip(
            "Sieht ein Frame den Punkt mit beiden Fisheyes, zählt die Linse, die\n"
            "ihn näher am Bildzentrum hat. Sonst gilt cam0 bis zum Rand.")
        self._chk_lens.toggled.connect(self._on_setting_changed)
        form.addRow(self._chk_lens)
        self._spin_edge = QSpinBox()
        self._spin_edge.setRange(400, 700)
        self._spin_edge.setSingleStep(10)
        self._spin_edge.setValue(600)
        self._spin_edge.setSuffix(" px")
        self._spin_edge.setToolTip(
            "Farbproben weiter als so viele Pixel vom Fisheye-Zentrum zählen nur,\n"
            "wenn ein anderer Frame den Punkt nicht näher an der Mitte sieht.\n"
            "Keine Probe wird verworfen, die Dichte bleibt. 700 schaltet ab.")
        self._spin_edge.valueChanged.connect(self._on_setting_changed)
        form.addRow("Linsenrand ab:", self._spin_edge)

        # Blaulicht: blinkt, also gibt es fast immer einen Frame ohne
        form.addRow(QLabel("<b>Blaulicht</b>"))
        self._chk_blue = QCheckBox("Blaulicht filtern")
        self._chk_blue.setToolTip(
            "Proben im gewählten Blaubereich zählen nur, wenn es für den Punkt\n"
            "keine andere gibt; gibt es nur blaue, zählt die am wenigsten blaue.\n"
            "Dazu kommen Frames bei ±0,2 bis ±1,2 s, damit eine dunkle Blinkphase\n"
            "dabei ist. Was danach noch blau ist, zieht 'Restblau neutralisieren'\n"
            "Richtung Grau.")
        self._chk_blue.toggled.connect(self._on_setting_changed)
        self._chk_blue.toggled.connect(self._update_blue_widgets)
        form.addRow(self._chk_blue)
        d = _DEFAULT_SETTINGS
        self._sld_blue_lo, _, row_blo = self._slider_row(0, 360, d["blue_hue_lo"])
        self._sld_blue_hi, _, row_bhi = self._slider_row(0, 360, d["blue_hue_hi"])
        self._sld_blue_sat, _, row_bsat = self._slider_row(0, 100, d["blue_sat"])
        self._sld_blue_val, _, row_bval = self._slider_row(0, 255, d["blue_val"])
        self._sld_blue_neutral, _, row_bneu = self._slider_row(0, 100, d["blue_neutral"])
        self._sld_blue_neutral.setToolTip(
            "Wo Blaulicht eine Fläche in jedem Frame anstrahlt, gibt es keine\n"
            "unbeleuchtete Probe. Punkte, deren Farbe am Ende im Blaubereich liegt,\n"
            "werden bei gleicher Helligkeit so weit Richtung Grau gezogen.\n"
            "0 = aus, 100 = ganz grau. Echt blaue Flächen werden dabei auch grau.")
        self._lbl_blue_band = QLabel()
        self._lbl_blue_band.setToolTip(
            "Farbton 0–360°: der helle Bereich wird als Blaulicht behandelt.\n"
            "Ist 'von' größer als 'bis', läuft der Bereich über Rot hinweg.")
        for sld in (self._sld_blue_lo, self._sld_blue_hi, self._sld_blue_sat,
                    self._sld_blue_val, self._sld_blue_neutral):
            sld.valueChanged.connect(self._on_setting_changed)
        self._sld_blue_lo.valueChanged.connect(self._update_blue_widgets)
        self._sld_blue_hi.valueChanged.connect(self._update_blue_widgets)
        form.addRow("Farbton von (°):", row_blo)
        form.addRow("Farbton bis (°):", row_bhi)
        form.addRow(self._lbl_blue_band)
        form.addRow("Sättigung min (%):", row_bsat)
        form.addRow("Helligkeit min:", row_bval)
        form.addRow("Restblau neutralisieren (%):", row_bneu)
        self._btn_blue_defaults = QPushButton("Standardwerte")
        self._btn_blue_defaults.setToolTip(
            "Farbton 150–290°, Sättigung ab 5 %, Helligkeit ab 10, Restblau\n"
            "100 % — eingestellt an einem Nachtflug mit Einsatzfahrzeugen, bis\n"
            "kein Blaulicht mehr zu sehen war. Grün (Bäume) bleibt.")
        self._btn_blue_defaults.clicked.connect(self._on_blue_defaults)
        form.addRow(self._btn_blue_defaults)
        self._btn_blue_preview = QPushButton("Blaumaske im Frame zeigen")
        self._btn_blue_preview.setToolTip(
            "Markiert im aktuellen Kamerabild magenta, was als Blaulicht gilt.")
        self._btn_blue_preview.clicked.connect(self._on_blue_preview_clicked)
        form.addRow(self._btn_blue_preview)
        self._blue_widgets = (self._sld_blue_lo, self._sld_blue_hi, self._sld_blue_sat,
                              self._sld_blue_val, self._sld_blue_neutral,
                              self._btn_blue_defaults, self._lbl_blue_band,
                              self._btn_blue_preview)
        self._update_blue_widgets()

        self._btn_colorize = QPushButton("Einfärben")
        self._btn_colorize.clicked.connect(self._on_colorize_clicked)
        form.addRow(self._btn_colorize)
        return box

    # ============================================================== Einfärbung

    @staticmethod
    def _import_colorizer():
        try:
            from core import colorizer
            return colorizer
        except ImportError as exc:
            raise RuntimeError(f"Modul 'colorizer' ist nicht verfügbar: {exc}") from exc

    def _mit_colorizer(self, titel: str):
        """Das Einfärbe-Modul; fehlt es, ein Fehlerdialog unter ``titel`` und None."""
        try:
            return self._import_colorizer()
        except RuntimeError as exc:
            self._show_error(titel, str(exc))
            return None

    def _aktueller_frame(self) -> int:
        """Frame der Rundumansicht; ist keiner angezeigt, die Mitte des Bags."""
        frame_idx = self._pano_view.current_index
        if frame_idx < 0:
            frame_idx = max(0, self._n_frames // 2)
        return frame_idx

    def _zeige_bild(self, titel: str, bgr: np.ndarray) -> None:
        """BGR-Bild in einem eigenen Fenster; geschlossene fallen aus der Liste."""
        dlg = _ImageDialog(titel, bgr, self)
        self._overlay_dialogs = [d for d in self._overlay_dialogs if d.isVisible()]
        self._overlay_dialogs.append(dlg)
        dlg.show()

    def _onboard_kontext(self) -> dict:
        """Extrinsik, Abschnitte und Maskenwerte für einen Lauf mit den Onboard-Bildern.

        Ohne zusammengeführte Karte ist der ganze Flug ein Abschnitt.
        """
        rec = self._rec
        T = self._extrinsic_from_spins()
        teile = list(self._parts) if self._parts else [(self._bag, 0, int(rec.n_scans))]
        masken = {"bmin": int(self._sld_bmin.value()), "bmax": int(self._sld_bmax.value()),
                  "sky_grow": int(self._spin_sky.value())}
        return {"T": T, "teile": teile, "masken": masken}

    def _on_colorize_clicked(self) -> None:
        if self._rec is None or self._bag is None:
            return
        colorizer = self._mit_colorizer("Einfärben")
        if colorizer is None:
            return
        if self._merge_rec is not None:
            weiter = QMessageBox.question(
                self, "Zweiter Flug nicht übernommen",
                "Es ist ein zweiter Flug geladen (die orangen Punkte), aber "
                "noch nicht übernommen.\n\nEingefärbt wird nur der offene "
                "Flug; die orange Vorschau bleibt unverändert liegen und "
                "verdeckt das Ergebnis.\n\nTrotzdem einfärben?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if weiter != QMessageBox.Yes:
                return
        T = self._extrinsic_from_spins()
        try:
            self._project.save_extrinsic(T)
        except RuntimeError as exc:
            self._log(f"Extrinsik nicht gespeichert: {exc}")
        params = colorizer.ColorizeParams(
            brightness_min=int(self._sld_bmin.value()),
            brightness_max=int(self._sld_bmax.value()),
            k_frames=int(self._spin_kframes.value()),
            sky_grow=int(self._spin_sky.value()),
            lens_best=bool(self._chk_lens.isChecked()),
            edge_r=float(self._spin_edge.value()),
            blue_filter=bool(self._chk_blue.isChecked()),
            blue_neutral=float(self._sld_blue_neutral.value()) / 100.0,
            **self._blue_args(),
            T_imu_cam0=T)
        rec, bag, calib = self._rec, self._bag, self._calib
        parts = self._parts
        out_dir = self._project.layer_dir("onboard")

        def job(progress_cb, cancel, log_cb):
            # Kurzer Test vor dem langen Lauf: sitzt die Extrinsik auf einem
            # Gipfel oder auf einer Flanke? Der absolute Score ist zwischen
            # Fluegen nicht vergleichbar, eine verdrehte Extrinsik faellt daher
            # sonst nicht auf — sie kostet aber die halbe Farbqualitaet.
            progress_cb(0.01, "Prüfe Extrinsik …")
            try:
                chk = colorizer.check_extrinsic(rec, bag, calib, T, cancel=cancel)
            except RuntimeError as exc:
                log_cb(f"Extrinsik-Prüfung übersprungen: {exc}")
            else:
                log_cb(f"Extrinsik-Güte (Foto-Konsistenz): {chk['score']:.3f}; "
                       f"bestes erreichbares {chk['best_score']:.3f} "
                       f"{chk['dist_deg']:.1f}° daneben.")
                if chk["suspect"]:
                    log_cb(
                        "WARNUNG: die gespeicherte Extrinsik ist deutlich verdreht. "
                        "Das kostet spürbar Farbqualität — Lauf abbrechen, "
                        "'Auto-Kalibrierung (grob)' starten und neu einfärben.")
            return colorizer.colorize(rec, bag, calib, params, out_dir,
                                      progress_cb=progress_cb, cancel=cancel,
                                      parts=parts)

        self._start_worker("Färbe Punktwolke ein …", job, self._on_colorize_done)

    def _on_colorize_done(self, res: dict) -> None:
        n_valid = int(res.get("n_valid", 0))
        frac = float(res.get("frac_valid", 0.0))
        wo = f"GPU ({res['gpu']})" if res.get("gpu") else "CPU"
        self._log(f"Einfärbung fertig: {_fmt_int(n_valid)} Punkte gültig "
                  f"({100.0 * frac:.1f} %), gerechnet auf {wo}.")
        n_sky = int(res.get("n_sky_blocked", 0))
        if n_sky:
            self._log(f"Himmelssaum-Sperre: {_fmt_int(n_sky)} Farbproben verworfen "
                      f"(Saum {self._spin_sky.value()} px um ausgebrannte Flächen).")
        n_edge = int(res.get("n_edge_outvoted", 0))
        if n_edge:
            self._log(f"Linsenrand: {_fmt_int(n_edge)} Farbproben hinter Proben näher "
                      f"an der Bildmitte zurückgestellt.")
        n_blue = int(res.get("n_blue_outvoted", 0))
        if n_blue or res.get("n_blue_kept"):
            self._log(f"Blaulicht: {_fmt_int(n_blue)} blaue Farbproben zurückgestellt; "
                      f"{_fmt_int(int(res.get('n_blue_kept', 0)))} Punkte hatten nur "
                      f"blaue Proben.")
        n_neu = int(res.get("n_blue_neutral", 0))
        if n_neu:
            self._log(f"Restblau: {_fmt_int(n_neu)} Punkte Richtung Grau gezogen "
                      f"({self._sld_blue_neutral.value()} %).")
        # Bei einer zusammengefuehrten Karte je Abschnitt ausweisen: sonst
        # sieht man nur eine Gesamtquote und merkt nicht, dass ein ganzer Flug
        # leer geblieben ist.
        colors, valid, err = _load_color_files(self._project.layer_dir("onboard"),
                                               self._rec.n_points)
        if err:
            self._show_error("Einfärben", err)
            return
        from core.colorizer import abschnittsquote
        for teil, proxy, a, b, anteil in abschnittsquote(self._rec, valid, self._parts):
            self._log(f"   Abschnitt {teil} "
                      f"({os.path.basename(proxy.bag_path)}): "
                      f"{100.0 * anteil:.1f} % von {_fmt_int(b - a)} Punkten.")
            if anteil < 0.02:
                self._log(f"   WARNUNG: Abschnitt {teil} ist praktisch leer "
                          f"geblieben — vermutlich fehlt für dieses Bag die "
                          f"Kamera oder es liegt nicht mehr an seinem Ort.")
        self._colors, self._valid = colors, valid
        # Eine leere Anzeige ist der schlechteste Ausgang: bei "Nur eingefärbte
        # Punkte" verschwindet die ganze Wolke, und uebrig bleibt nur, was sonst
        # noch im Bild ist. Lieber den Haken loesen und es sagen.
        if n_valid == 0 and self._chk_only_colored.isChecked():
            self._loading_ui = True
            self._chk_only_colored.setChecked(False)
            self._loading_ui = False
            self._log("Kein Punkt wurde eingefärbt — 'Nur eingefärbte Punkte' "
                      "wurde gelöst, sonst bliebe die Ansicht leer.")
        self._cloud_view.set_cloud(self._world, self._colors,
                                   self._rec.intensity, self._valid)
        idx = self._combo_colormode.findData("rgb")
        self._combo_colormode.setCurrentIndex(idx)  # löst _on_display_changed aus
        self._push_display_settings()
