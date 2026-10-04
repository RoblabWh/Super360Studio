"""Kamera-Kalibrierung (Mixin des Hauptfensters).

Schreibt am Hauptfenster: _loading_ui.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from PyQt5.QtWidgets import QMessageBox

_AUTOCAL_WEAK_SCORE = 0.30


class KalibrierungMixin:
    # ============================================================== Extrinsik

    def _extrinsic_from_spins(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler(
            "ZYX",
            [self._ext_spins["yaw"].value(), self._ext_spins["pitch"].value(),
             self._ext_spins["roll"].value()], degrees=True).as_matrix()
        T[:3, 3] = [self._ext_spins["x"].value(), self._ext_spins["y"].value(),
                    self._ext_spins["z"].value()]
        return T

    def _spins_from_extrinsic(self, T: np.ndarray) -> None:
        ypr = Rotation.from_matrix(np.asarray(T)[:3, :3]).as_euler("ZYX", degrees=True)
        vals = {"yaw": ypr[0], "pitch": ypr[1], "roll": ypr[2],
                "x": T[0, 3], "y": T[1, 3], "z": T[2, 3]}
        self._loading_ui = True
        try:
            for key, val in vals.items():
                self._ext_spins[key].setValue(float(val))
        finally:
            self._loading_ui = False

    def _speichere_extrinsik(self, T: np.ndarray) -> None:
        """Extrinsik ins Projekt schreiben; ohne Projekt nichts, Fehler nur ins Log."""
        if self._project is None:
            return
        try:
            self._project.save_extrinsic(T)
        except RuntimeError as exc:
            self._log(f"Extrinsik nicht gespeichert: {exc}")

    def _on_extrinsic_changed(self, *_a) -> None:
        if self._loading_ui or self._project is None:
            return
        self._speichere_extrinsik(self._extrinsic_from_spins())

    def _on_extrinsic_reset(self) -> None:
        self._spins_from_extrinsic(np.eye(4))
        self._log("Extrinsik auf Identität zurückgesetzt.")

    def _on_overlay_clicked(self) -> None:
        if self._rec is None or self._bag is None:
            return
        colorizer = self._mit_colorizer("Overlay-Vorschau")
        if colorizer is None:
            return
        frame_idx = self._aktueller_frame()
        T = self._extrinsic_from_spins()
        rec, bag, calib = self._rec, self._bag, self._calib

        def job(progress_cb, cancel, log_cb):
            progress_cb(0.2, f"Erzeuge Overlay für Frame {frame_idx} …")
            return colorizer.overlay_preview(rec, bag, calib, T, frame_idx, stride=50)

        def on_done(img) -> None:
            self._zeige_bild(f"Overlay-Vorschau — Frame {frame_idx}", img)

        # Einzelner Bibliotheksaufruf ohne Cancel-Auswertung — nicht abbrechbar.
        self._start_worker("Erzeuge Overlay-Vorschau …", job, on_done,
                           cancellable=False)

    def _on_autocal_clicked(self) -> None:
        if self._rec is None or self._bag is None:
            return
        colorizer = self._mit_colorizer("Auto-Kalibrierung")
        if colorizer is None:
            return
        T_init = self._extrinsic_from_spins()
        rec, bag, calib = self._rec, self._bag, self._calib

        def job(progress_cb, cancel, log_cb):
            return colorizer.auto_calibrate(rec, bag, calib, T_init=T_init,
                                            progress_cb=progress_cb, cancel=cancel)

        def on_done(result) -> None:
            T, score = result
            self._spins_from_extrinsic(np.asarray(T))
            self._speichere_extrinsik(np.asarray(T))
            self._log(f"Auto-Kalibrierung fertig — Score {score:.3f}.")
            if score < _AUTOCAL_WEAK_SCORE and not self._autotest_active():
                QMessageBox.warning(
                    self, "Auto-Kalibrierung",
                    f"Schwacher Kalibrier-Score ({score:.3f}) — Ergebnis bitte mit "
                    "der Overlay-Vorschau prüfen und ggf. von Hand nachjustieren.")

        self._start_worker("Auto-Kalibrierung läuft …", job, on_done)
