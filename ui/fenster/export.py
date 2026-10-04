"""Export (Mixin des Hauptfensters).

Schreibt am Hauptfenster: _btn_cloud_cc, _btn_export_las, _btn_export_plypcd,
_georef, _lbl_georef.
"""
from __future__ import annotations

import os

import numpy as np

from PyQt5.QtWidgets import QLabel, QMessageBox, QVBoxLayout, QWidget

from core import georef
from core.gemeinsam import GRAU_ANZEIGE, fmt_int as _fmt_int

from ui.bausteine import speicherpfad


class ExportMixin:
    def _abschnitt_export(self) -> QWidget:
        box = QWidget()
        lay = QVBoxLayout(box)
        self._btn_export_plypcd = self._befehlsknopf("export_ply")
        self._btn_export_las = self._befehlsknopf("export_las")
        lay.addWidget(self._btn_export_plypcd)
        lay.addWidget(self._btn_export_las)
        self._lbl_georef = QLabel(self._georef_text(None))
        self._lbl_georef.setWordWrap(True)
        lay.addWidget(self._lbl_georef)
        self._btn_cloud_cc = self._befehlsknopf("cloud_cc")
        lay.addWidget(self._btn_cloud_cc)
        lay.addWidget(self._befehlsknopf("project_export"))
        return box

    # ================================================================== Export

    def _export_arrays(self) -> tuple[np.ndarray, np.ndarray | None]:
        pts, cols = self._world, self._colors
        if (self._chk_only_colored.isChecked() and self._valid is not None
                and cols is not None):
            mask = self._valid
            pts = pts[mask]
            cols = cols[mask]
        return pts, cols

    @staticmethod
    def _export_schreibjob(pts: np.ndarray, cols: np.ndarray | None, path: str):
        """Job, der die Punkte als PLY/PCD nach ``path`` schreibt."""
        def job(progress_cb, cancel, log_cb):
            progress_cb(0.2, f"Schreibe {os.path.basename(path)} …")
            georef.export_ply_pcd(pts, cols, path)
            return path
        return job

    def _on_export_plypcd(self) -> None:
        if self._world is None:
            return
        start = os.path.join(self._project.dir if self._project else "",
                             "punktwolke.ply")
        path = speicherpfad(self, "Punktwolke speichern", start,
                            "PLY-Datei (*.ply);;PCD-Datei (*.pcd)", (".ply", ".pcd"))
        if not path:
            return
        pts, cols = self._export_arrays()
        job = self._export_schreibjob(pts, cols, path)

        # Einzelner Bibliotheksaufruf ohne Cancel-Auswertung — nicht abbrechbar.
        self._start_worker("Exportiere PLY/PCD …", job,
                           lambda p: self._log(f"Export abgeschlossen: {p} "
                                               f"({_fmt_int(len(pts))} Punkte)."),
                           cancellable=False)

    def _on_export_las(self) -> None:
        if self._world is None:
            return
        start = os.path.join(self._project.dir if self._project else "",
                             "punktwolke.las")
        path = speicherpfad(self, "LAS speichern", start, "LAS-Datei (*.las)", ".las")
        if not path:
            return
        pts, cols = self._export_arrays()
        geo = self._georef

        def job(progress_cb, cancel, log_cb):
            progress_cb(0.2, f"Schreibe {os.path.basename(path)} …")
            georef.export_las(pts, cols, geo, path)
            return path

        suffix = (f" (georeferenziert, EPSG:{geo.utm_epsg})" if geo is not None
                  else " (lokales LIO-System)")
        # Einzelner Bibliotheksaufruf ohne Cancel-Auswertung — nicht abbrechbar.
        self._start_worker("Exportiere LAS …", job,
                           lambda p: self._log(f"Export abgeschlossen: {p}"
                                               f"{suffix}, {_fmt_int(len(pts))} Punkte."),
                           cancellable=False)

    def _cc_arrays(self) -> tuple[np.ndarray, np.ndarray | None]:
        """Punkte wie angezeigt; ohne 'Nur eingefärbte' die ungefärbten grau
        wie in der Ansicht (schwarz würde das Mesh fleckig machen)."""
        pts, cols = self._export_arrays()
        if (cols is not None and self._valid is not None
                and not self._chk_only_colored.isChecked()):
            cols = cols.copy()
            cols[~self._valid] = GRAU_ANZEIGE
        return pts, cols

    def _cc_open(self, paths: list[str]) -> None:
        from core import mesh as mesh_mod
        try:
            cmd = mesh_mod.open_in_cloudcompare(paths)
        except RuntimeError as exc:
            self._log(str(exc))
            if not self._autotest_active():
                QMessageBox.information(self, "CloudCompare", str(exc)
                                        + "\n\nDie Datei liegt unter:\n" + "\n".join(paths))
            return
        self._log(f"CloudCompare gestartet: {' '.join(cmd[:-len(paths)])} "
                  f"({', '.join(os.path.basename(p) for p in paths)}).")

    def _on_cloud_cloudcompare(self) -> None:
        if self._world is None or self._project is None:
            return
        pts, cols = self._cc_arrays()
        path = os.path.join(self._project.dir, "mesh", "punktwolke.ply")
        schreiben = self._export_schreibjob(pts, cols, path)

        def job(progress_cb, cancel, log_cb):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            return schreiben(progress_cb, cancel, log_cb)

        self._start_worker("Schreibe Punktwolke für CloudCompare …", job,
                           lambda p: self._cc_open([p]), cancellable=False)

    @staticmethod
    def _georef_text(result: georef.GeorefResult | None) -> str:
        """Statuszeile im Abschnitt Export: in welchem System LAS schreibt."""
        if result is None:
            return "LAS im lokalen System — Georeferenz im Reiter GPS"
        return f"LAS in UTM, EPSG:{result.utm_epsg}"

    def _georef_setzen(self, result: georef.GeorefResult | None) -> None:
        """Georeferenz der offenen Wolke setzen (None: keine)."""
        self._georef = result
        self._lbl_georef.setText(self._georef_text(result))

    def _on_georef_ready(self, result) -> None:
        self._georef_setzen(result)
        self._log(f"Georeferenzierung bereit: EPSG:{result.utm_epsg}, "
                  f"RMS {result.rms_m:.2f} m, {result.n_used} Fixe — "
                  "LAS-Export verwendet jetzt UTM-Koordinaten.")
