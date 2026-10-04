"""Export (Mixin des Hauptfensters).

Schreibt am Hauptfenster: _btn_cloud_cc, _btn_export_las, _btn_export_plypcd,
_btn_mesh_cc, _chk_mesh_hybrid, _georef, _spin_mesh_depth, _spin_mesh_trim,
_spin_mesh_voxel, _valid.
"""
from __future__ import annotations

import os

import numpy as np

from PyQt5.QtWidgets import (
    QCheckBox, QFormLayout, QLabel, QMessageBox, QPushButton, QSpinBox,
    QVBoxLayout, QWidget,
)

from core import georef
from core.gemeinsam import GRAU_ANZEIGE, fmt_int as _fmt_int

from ui.bausteine import _wrappable, speicherpfad
from ui.fenster.einstellungen import _DEFAULT_SETTINGS


class ExportMixin:
    def _group_export(self) -> QWidget:
        box = QWidget()
        lay = QVBoxLayout(box)
        self._btn_export_plypcd = QPushButton("PLY/PCD speichern…")
        self._btn_export_plypcd.clicked.connect(self._on_export_plypcd)
        self._btn_export_las = QPushButton("LAS speichern…")
        self._btn_export_las.clicked.connect(self._on_export_las)
        lay.addWidget(self._btn_export_plypcd)
        lay.addWidget(self._btn_export_las)

        # Mesh fuer CloudCompare (s. core/mesh.py)
        lay.addWidget(QLabel("<b>CloudCompare</b>"))
        form = _wrappable(QFormLayout())
        d = _DEFAULT_SETTINGS
        self._spin_mesh_voxel = QSpinBox()
        self._spin_mesh_voxel.setRange(1, 50)
        self._spin_mesh_voxel.setValue(d["mesh_voxel_cm"])
        self._spin_mesh_voxel.setSuffix(" cm")
        self._spin_mesh_voxel.setToolTip(
            "Je Rasterzelle ein Punkt, bevor das Netz entsteht. Kleiner = feiner,\n"
            "aber deutlich langsamer (5 cm: ganzer Flug ~1,5 min).")
        self._spin_mesh_depth = QSpinBox()
        self._spin_mesh_depth.setRange(8, 13)
        self._spin_mesh_depth.setValue(d["mesh_depth"])
        self._spin_mesh_depth.setToolTip(
            "Poisson-Tiefe: 2^Tiefe Zellen über die größte Ausdehnung der Karte.\n"
            "12 bei ~250 m sind ~6 cm; jede Stufe mehr halbiert die Zellen und\n"
            "kostet etwa das Vierfache an Zeit und Speicher.")
        self._chk_mesh_hybrid = QCheckBox("Laub als Punkte (Hybrid)")
        self._chk_mesh_hybrid.setChecked(bool(d["mesh_hybrid"]))
        self._chk_mesh_hybrid.setToolTip(
            "Nur Flächen vernetzen — Boden, Wände, Dächer, Fahrzeuge. Laub, Masten\n"
            "und Rohre bleiben Punkte: als Mesh würden sie zu Klumpen und Wülsten.\n"
            "Aus: alles wird vernetzt.")
        self._spin_mesh_trim = QSpinBox()
        self._spin_mesh_trim.setRange(0, 30)
        self._spin_mesh_trim.setValue(d["mesh_trim"])
        self._spin_mesh_trim.setSuffix(" %")
        self._spin_mesh_trim.setToolTip(
            "Zusätzlich die Ecken mit der geringsten Punktdichte entfernen. Flächen\n"
            "ohne Messung in der Nähe fallen ohnehin weg; mehr als 0 stanzt kleine\n"
            "Löcher in gleichmäßig abgetastete Wände.")
        for w in (self._spin_mesh_voxel, self._spin_mesh_depth, self._spin_mesh_trim):
            w.valueChanged.connect(self._on_setting_changed)
            w.valueChanged.connect(self._on_mesh_param_changed)
        self._chk_mesh_hybrid.toggled.connect(self._on_setting_changed)
        self._chk_mesh_hybrid.toggled.connect(self._on_mesh_param_changed)
        form.addRow("Mesh-Raster:", self._spin_mesh_voxel)
        form.addRow("Detail (Tiefe):", self._spin_mesh_depth)
        form.addRow("Ränder kürzen:", self._spin_mesh_trim)
        form.addRow(self._chk_mesh_hybrid)
        lay.addLayout(form)
        self._btn_mesh_cc = QPushButton("Mesh erzeugen und in CloudCompare zeigen")
        self._btn_mesh_cc.setToolTip(
            "Dreiecksnetz mit den Farben der angezeigten Ebene; gespeichert im\n"
            "Projektordner unter mesh/.")
        self._btn_mesh_cc.clicked.connect(self._on_mesh_cloudcompare)
        self._btn_cloud_cc = QPushButton("Punktwolke in CloudCompare öffnen")
        self._btn_cloud_cc.clicked.connect(self._on_cloud_cloudcompare)
        lay.addWidget(self._btn_mesh_cc)
        lay.addWidget(self._btn_cloud_cc)
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

    def _georef_setzen(self, result: georef.GeorefResult | None) -> None:
        """Georeferenz der offenen Wolke setzen (None: keine)."""
        self._georef = result

    def _on_georef_ready(self, result) -> None:
        self._georef_setzen(result)
        self._log(f"Georeferenzierung bereit: EPSG:{result.utm_epsg}, "
                  f"RMS {result.rms_m:.2f} m, {result.n_used} Fixe — "
                  "LAS-Export verwendet jetzt UTM-Koordinaten.")
