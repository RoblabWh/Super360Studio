"""Aufnahme, Karte und Projekt (Mixin des Hauptfensters).

Schreibt am Hauptfenster: _bag, _bag_info, _btn_fastlio, _btn_open,
_btn_open_project, _colors, _combo_config, _exploration, _fixes, _info_table,
_layers, _lbl_fastlio, _meander_pipe, _n_frames, _pano_failed, _pano_src,
_parts, _project, _quality, _rec, _settings, _spin_rate, _valid, _world.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QFileDialog, QFormLayout, QHeaderView, QLabel, QMessageBox,
    QPlainTextEdit, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget,
)

from core import exploration, georef
from core.bag_reader import ThreadLocalBag
from core.exploration import (
    holen as _exploration_holen, rechnen_und_ablegen as _exploration_rechnen,
)
from core.gemeinsam import fmt_int as _fmt_int
from core.kalibrierung import REPO_WURZEL
from core.project import Project
from core.recording import lade_mit_hinweis

from ui.bausteine import _wrappable, auswahl, knopfzeile, zahl
from ui.fenster.einstellungen import _DEFAULT_SETTINGS
from ui.pano_view import StitchingPanoSource

_CONFIG_ITEMS = (("Maximal dicht (whs_dense.yaml)", "whs_dense.yaml"),
                 ("Schnell (mid360.yaml)", "mid360.yaml"))
#: Breite der gestitchten Panoramen in px
_PANO_BREITE = 1920


def _karten_zeile(n_scans, erwartet, n_points, drops) -> str:
    """Statuszeile unter "Karte berechnen"."""
    return (f"Scans: {n_scans}/{erwartet} · "
            f"Punkte: {_fmt_int(n_points)} · Drops: {drops}")


class ProjektMixin:
    def _abschnitt_aufnahme(self) -> QWidget:
        box = QWidget()
        lay = QVBoxLayout(box)
        self._btn_open = self._befehlsknopf("open")
        self._btn_open_project = self._befehlsknopf("open_project")
        lay.addWidget(knopfzeile(self._btn_open, self._btn_open_project))
        self._info_table = QTableWidget(0, 2, box)
        self._info_table.setHorizontalHeaderLabels(["Eigenschaft", "Wert"])
        self._info_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._info_table.horizontalHeader().setStretchLastSection(True)
        self._info_table.verticalHeader().setVisible(False)
        self._info_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._info_table.setSelectionMode(QTableWidget.NoSelection)
        self._info_table.setMinimumHeight(230)
        self._info_table.setWordWrap(False)
        lay.addWidget(self._info_table)
        return box

    def _abschnitt_karte(self) -> QWidget:
        box = QWidget()
        form = _wrappable(QFormLayout(box))
        self._combo_config = auswahl(_CONFIG_ITEMS, slot=self._on_setting_changed)
        form.addRow("Konfiguration:", self._combo_config)
        self._spin_rate = zahl(0.25, 2.0, 1.0, 0.25, dezimalen=2,
                               slot=self._on_setting_changed)
        form.addRow("Abspielrate:", self._spin_rate)
        self._btn_fastlio = self._befehlsknopf("fastlio")
        form.addRow(self._btn_fastlio)
        self._lbl_fastlio = QLabel("Noch keine Karte berechnet.")
        self._lbl_fastlio.setWordWrap(True)
        form.addRow(self._lbl_fastlio)
        return box

    # ========================================================= Rosbag öffnen

    def _beschaeftigt_melden(self, lang: bool) -> bool:
        """Läuft noch ein Arbeitsschritt, das sagen und True liefern."""
        if not self._busy:
            return False
        if lang:
            QMessageBox.information(
                self, "Beschäftigt",
                "Es läuft noch ein Arbeitsschritt — bitte warten oder abbrechen.")
        else:
            QMessageBox.information(self, "Beschäftigt",
                                    "Es läuft noch ein Arbeitsschritt.")
        return True

    def _on_open_clicked(self) -> None:
        if self._beschaeftigt_melden(True):
            return
        path = QFileDialog.getExistingDirectory(
            self, "Rosbag-Ordner öffnen", os.path.join(REPO_WURZEL, "ui"))
        if path:
            self._open_bag(path)

    def _open_bag(self, path: str, projekt=None) -> None:
        """Bag oeffnen; ``projekt`` statt des Cache-Projekts zum Bag (Ordner
        eines geoeffneten Exports)."""
        if self._beschaeftigt_melden(True):
            return
        self._clear_bag_state()
        self._grad_anzeige.rechnet()
        path = os.path.abspath(path)

        def job(progress_cb, cancel, log_cb):
            progress_cb(0.05, "Öffne Bag …")
            proxy = ThreadLocalBag(path)
            info = proxy.info()
            progress_cb(0.25, "Indiziere Kamera-Frames …")
            n_frames = len(proxy.camera_stamps()) if info.camera_topic else 0
            if cancel.is_set():
                raise RuntimeError("Abgebrochen")
            progress_cb(0.6, "Lese GPS-Daten …")
            fixes = proxy.read_gps()
            quality = georef.assess(fixes)
            project = projekt if projekt is not None else Project(path)
            progress_cb(0.75, "Explorationsgrad …")
            grad, grad_hinweis = _exploration_holen(path, project, log_cb,
                                                    progress_cb, cancel,
                                                    von=0.75, bis=0.98)
            progress_cb(1.0, "Bag geöffnet")
            return {"path": path, "proxy": proxy, "info": info,
                    "n_frames": n_frames, "fixes": fixes,
                    "quality": quality, "project": project,
                    "exploration": grad, "exploration_hinweis": grad_hinweis}

        self._start_worker(f"Öffne Bag: {os.path.basename(path)} …", job, self._on_bag_opened)

    def _clear_bag_state(self) -> None:
        self._bag = None
        self._bag_info = None
        self._project = None
        self._rec = None
        self._world = None
        self._colors = None
        self._valid = None
        self._fixes = None
        self._quality = None
        self._georef_setzen(None)
        self._exploration = None
        self._grad_anzeige.leeren()
        self._pano_src = None
        self._pano_failed = False
        self._pano_view.set_source(None)
        self._cloud_view.set_cloud(None)
        self._cloud_view.set_path(None)
        self._mesh_vergessen()
        self._parts = None
        self._layers = {}
        self._meander_pipe = None
        self._live_clear()
        self._meander_setze_thermal((0.0, 0.0, 0.0))
        self._meander_setze_optik({"rgb_faktor": 1.0, "thermal_faktor": 1.0,
                                   "thermal": None})
        self._merge_reset_state()
        self._gps_panel.set_quality(None, None, None)
        self._info_table.setRowCount(0)
        self._lbl_fastlio.setText("Noch keine Karte berechnet.")
        self._frame_lbl.setText("")
        self.setWindowTitle("Super360 Studio")

    def _on_bag_opened(self, res: dict) -> None:
        self._bag = res["proxy"]
        self._bag_info = res["info"]
        self._n_frames = res["n_frames"]
        self._fixes = res["fixes"]
        self._quality = res["quality"]
        self._project = res["project"]
        self._exploration = res.get("exploration")
        self._zeige_exploration(res.get("exploration_hinweis") or "")
        self.setWindowTitle(f"Super360 Studio — {self._project.bag_name}")
        self._projekt_aktivieren(self._project)

        self._populate_info_table()
        self._gps_panel.set_quality(self._quality, self._fixes, None)
        info = self._bag_info
        self._log(f"Bag geöffnet: {info.path} — Dauer {info.duration:.1f} s, "
                  f"{self._n_frames} Kamera-Frames, {len(self._fixes)} GPS-Fixe "
                  f"(GPS {'nutzbar' if self._quality.usable else 'unbrauchbar'}).")
        # Kette entkoppelt: das Pano ist optional — eine zwischengespeicherte
        # Aufzeichnung muss auch ohne Kamera-Topic bzw. bei Pano-Fehlern laden.
        if not info.camera_topic:
            self._log("Kein Kamera-Topic im Bag — 360°-Video/Einfärbung nicht verfügbar.")
        self._update_enabled()
        if self._calib and info.camera_topic:
            self._start_pano_job()
        elif self._project.has_recording():
            self._start_recording_load()

    def _projekt_aktivieren(self, project) -> None:
        """Einstellungen und Extrinsik des Projekts in die Regler laden."""
        self._settings = dict(_DEFAULT_SETTINGS)
        try:
            self._settings.update(project.load_settings())
        except RuntimeError as exc:
            self._log(str(exc))
        self._apply_settings_to_widgets()

        T = None
        try:
            T = project.load_extrinsic()
        except RuntimeError as exc:
            self._log(str(exc))
        self._spins_from_extrinsic(T if T is not None else np.eye(4))

    # ======================================================= Explorationsgrad

    def _zeige_exploration(self, hinweis: str = "") -> None:
        """Kachel oben rechts auf den Stand von ``self._exploration`` bringen."""
        grad = self._exploration
        if grad is not None:
            self._grad_anzeige.setze(grad)
            self._log(f"Explorationsgrad: {grad.kurz()} — Phasen: {grad.quelle}.")
        elif hinweis:
            self._grad_anzeige.ohne_daten(hinweis)
            self._log(f"Kein Explorationsgrad: {hinweis}")
        else:
            self._grad_anzeige.leeren()

    def _on_exploration_zeigen(self) -> None:
        """Voller Bericht — auf Klick in die Kachel."""
        grad = self._exploration
        if grad is None:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Explorationsgrad")
        lay = QVBoxLayout(dlg)
        kopf = QLabel(f"<b style='font-size:16px'>{grad.kurz()}</b>", dlg)
        lay.addWidget(kopf)
        feld = QPlainTextEdit(dlg)
        feld.setReadOnly(True)
        feld.setPlainText(grad.text())
        feld.setLineWrapMode(QPlainTextEdit.NoWrap)
        schrift = feld.font()
        schrift.setFamily("monospace")   # die Kennzahlen stehen in Spalten
        feld.setFont(schrift)
        lay.addWidget(feld, 1)
        knopf = QPushButton("Schließen", dlg)
        knopf.clicked.connect(dlg.accept)
        lay.addWidget(knopf, 0, Qt.AlignRight)
        dlg.resize(660, 500)
        dlg.exec_()

    def _on_exploration_neu(self) -> None:
        """Explorationsgrad neu rechnen, am Cache vorbei."""
        if self._bag_info is None or self._project is None:
            return
        pfad, projekt = self._bag_info.path, self._project

        def job(progress_cb, cancel, log_cb):
            return _exploration_rechnen(pfad, projekt, log_cb, progress_cb, cancel)

        def fertig(res):
            grad, hinweis = res
            self._exploration = grad
            self._zeige_exploration(hinweis)

        self._grad_anzeige.rechnet()
        self._start_worker("Explorationsgrad berechnen …", job, fertig)

    def _populate_info_table(self) -> None:
        info = self._bag_info
        rows: list[tuple[str, str]] = [
            ("Pfad", info.path),
            ("Dauer", f"{info.duration:.1f} s"),
            ("Kamera-Frames", _fmt_int(self._n_frames)),
            ("Kamera-Topic", info.camera_topic or "—"),
            ("Lidar-Topic", info.lidar_topic or "—"),
            ("GPS-Fix-Topic", info.gps_fix_topic or "—"),
        ]
        for name, (typ, count) in sorted(info.topics.items()):
            rows.append((name, f"{typ.rsplit('/', 1)[-1]} × {_fmt_int(count)}"))
        self._info_table.setRowCount(len(rows))
        for r, (key, val) in enumerate(rows):
            for c, text in enumerate((key, val)):
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                self._info_table.setItem(r, c, item)

    # ============================================================ Pano-Quelle

    def _start_pano_job(self) -> None:
        bag, project = self._bag, self._project
        width = _PANO_BREITE
        pano_dir = project.pano_dir(width)
        calib = self._calib
        self._pano_failed = False

        def job(progress_cb, cancel, log_cb):
            progress_cb(0.05, "Baue 360°-Stitcher (LUT) …")
            src = StitchingPanoSource(bag, calib, width, pano_dir)
            if cancel.is_set():
                raise RuntimeError("Abgebrochen")
            if src.count > 0:
                progress_cb(0.7, "Stitche erstes Panorama …")
                src.get_pano(0)
            progress_cb(1.0, "360°-Quelle bereit")
            return src

        def on_pano_failed(msg: str) -> None:
            # Pano-Fehler dürfen die restliche Kette (3D-Karte, GPS) nicht blockieren.
            self._pano_failed = True
            self._log(f"FEHLER — 360°-Video: {msg}")
            self._status_lbl.setText("360°-Video nicht verfügbar.")
            if self._project is not None and self._project.has_recording():
                self._start_recording_load()

        self._start_worker("Bereite 360°-Video vor …", job, self._on_pano_ready,
                           on_failed=on_pano_failed)

    def _on_pano_ready(self, src: StitchingPanoSource) -> None:
        self._pano_src = src
        self._pano_view.set_source(src)
        self._log(f"360°-Video bereit: {src.count} Frames, {src.fps:.1f} fps, "
                  f"Breite {_PANO_BREITE} px.")
        if self._project is not None and self._project.has_recording():
            self._start_recording_load()

    def _on_pano_frame(self, idx: int, stamp: float) -> None:
        src = self._pano_src
        if src is None or src.count == 0:
            return
        t_rel = stamp - float(src.stamps[0])
        self._frame_lbl.setText(f"360°: Frame {idx + 1}/{src.count} — t={t_rel:.2f} s")

    # ===================================================== Recording/3D-Karte

    def _start_recording_load(self) -> None:
        project = self._project
        rec_dir = project.recording_dir()

        def job(progress_cb, cancel, log_cb):
            progress_cb(0.02, "Lade FAST-LIO-Aufzeichnung …")
            rec = lade_mit_hinweis(rec_dir, project.bag_path, log_cb)
            world = rec.world_points(
                progress_cb=lambda f, m: progress_cb(0.05 + 0.85 * f, m), cancel=cancel)
            return {"rec": rec, "world": world}

        self._start_worker("Lade Punktwolke …", job, self._on_recording_loaded)

    def _parts_from_meta(self, meta: dict) -> Optional[list]:
        """Abschnitte einer zusammengefuehrten Aufzeichnung aus ihrer meta.json.

        Die Aufzeichnung weiss selbst, aus welchen Bags sie besteht — die UI
        soll sich das nicht merken muessen. Vorher stand die Liste nur direkt
        nach dem Zusammenfuehren im Speicher; wurde sie verworfen, faerbte die
        gesamte Wolke aus der Kamera des ERSTEN Bags. Fuer die Scans des
        zweiten gibt es dort keine Frames im Zeitfenster, also blieben sie
        ungefaerbt.
        """
        quellen = meta.get("sources") or []
        if len(quellen) < 2:
            return None
        teile = []
        fehlend = []
        for q in quellen:
            bag = q.get("bag")
            r = q.get("scan_range")
            if not bag or not r or len(r) != 2:
                self._log("Zusammengeführte Aufzeichnung ohne brauchbare "
                          "Quellenangabe — Einfärbung nutzt nur das erste Bag.")
                return None
            if not os.path.exists(bag):
                fehlend.append(bag)
            teile.append((ThreadLocalBag(bag), int(r[0]), int(r[1])))
        namen = ", ".join(os.path.basename(q["bag"]) for q in quellen)
        self._log(f"Zusammengeführte Aufzeichnung aus {len(teile)} Flügen: {namen}. "
                  f"Die Einfärbung nutzt für jeden Abschnitt seine eigene Kamera.")
        for b in fehlend:
            self._log(f"WARNUNG: Quell-Bag nicht am Ort: {b} — dieser Abschnitt "
                      f"lässt sich nicht einfärben.")
        return teile

    def _on_recording_loaded(self, res: dict) -> None:
        self._rec = res["rec"]
        self._world = res["world"]
        # Immer aus der Aufzeichnung ableiten, nicht aus dem Sitzungsgedaechtnis
        self._parts = self._parts_from_meta(self._rec.meta)
        if self._parts:
            self._bag = self._parts[0][0]
        # liest die Farbebenen und zeigt die gewaehlte samt Wolke an
        self._reload_layers()
        # neue Punktwolke, neues Mesh: das alte gehoerte zu einer anderen
        self._mesh_vergessen()
        self._mesh_sicherstellen()
        if self._quality is not None:
            self._gps_panel.set_quality(self._quality, self._fixes, self._rec)
        meta = self._rec.meta
        expected = int(meta.get("expected_scans", self._rec.n_scans))
        drops = max(0, expected - self._rec.n_scans)
        self._lbl_fastlio.setText(
            _karten_zeile(self._rec.n_scans, expected, self._rec.n_points, drops))
        n_col = int(self._valid.sum()) if self._valid is not None else 0
        col_txt = (f", {_fmt_int(n_col)} eingefärbt" if self._colors is not None else "")
        self._log(f"Punktwolke geladen: {self._rec.n_scans} Scans, "
                  f"{_fmt_int(self._rec.n_points)} Punkte{col_txt}.")
        self._update_enabled()

    # ================================================================ FAST-LIO

    def _fastlio_job(self, bag_path: str, out_dir: str):
        """FAST-LIO-Job mit Konfiguration und Rate der Regler: (job, config, rate).

        Die Regler werden hier im GUI-Thread gelesen; der Job selbst fasst das
        Fenster nicht an.
        """
        config = self._combo_config.currentData()
        rate = float(self._spin_rate.value())

        def job(progress_cb, cancel, log_cb):
            from core.fastlio_runner import FastLioRunner
            runner = FastLioRunner()
            return runner.run(bag_path, out_dir, config=config, rate=rate,
                              progress_cb=progress_cb, cancel=cancel, log_cb=log_cb)

        return job, config, rate

    def _on_fastlio_clicked(self) -> None:
        if self._bag is None or self._project is None:
            return
        job, config, rate = self._fastlio_job(self._bag.bag_path,
                                              self._project.recording_dir())
        # Alte Aufzeichnung VOR dem Start vollständig loslassen: die neue
        # Aufzeichnung ersetzt recording/ — offene np.memmaps auf den alten
        # Dateien würden sonst als veraltete Anzeige weiterleben (bzw. bei
        # Truncation einen SIGBUS riskieren). Auch die Georeferenzierung passt
        # nicht mehr zur neuen Flugbahn.
        self._rec = None
        self._world = None
        self._colors = None
        self._valid = None
        self._georef_setzen(None)
        self._cloud_view.set_cloud(None)
        self._cloud_view.set_path(None)
        if self._quality is not None:
            self._gps_panel.set_quality(self._quality, self._fixes, None)
        self._lbl_fastlio.setText("Karte wird berechnet …")
        self._start_worker(f"FAST-LIO2 läuft ({config}, Rate {rate:g}×) …",
                           job, self._on_fastlio_done)

    def _on_fastlio_done(self, result) -> None:
        self._lbl_fastlio.setText(
            _karten_zeile(result.n_scans, result.expected_scans, result.n_points,
                          result.dropped_scans))
        self._log(f"FAST-LIO2 fertig in {result.duration_s:.1f} s: "
                  f"{result.n_scans} Scans, {_fmt_int(result.n_points)} Punkte, "
                  f"{result.dropped_scans} Drops.")
        # alte Farben passen nicht mehr zur neuen Aufzeichnung
        self._colors = None
        self._valid = None
        self._start_recording_load()

    # ========================================== Projekt aus- und einpacken

    def _bag_paths(self) -> list:
        """Alle Bags, aus denen das offene Projekt stammt (bei Fusion mehrere)."""
        if self._parts:
            return [p[0].bag_path for p in self._parts]
        return [self._bag.bag_path] if self._bag is not None else []

    def _on_export_project(self) -> None:
        if self._project is None or not self._project.has_recording():
            QMessageBox.information(
                self, "Projekt exportieren",
                "Es ist noch keine Karte berechnet — ohne sie gibt es kein "
                "Projekt zum Mitnehmen.")
            return
        from core import bundle
        from ui.bundle_dialog import ExportDialog
        info = bundle.describe(self._project, self._bag_paths())
        vorschlag = os.path.join(os.path.expanduser("~"),
                                 f"{self._project.bag_name}_projekt")
        dlg = ExportDialog(self._project.bag_name, info, vorschlag, self)
        if dlg.exec_() != QDialog.Accepted:
            return
        ziel, wahl = dlg.ziel(), dlg.auswahl()
        proj, bags, calib = self._project, self._bag_paths(), self._calib
        extra = {"kennzahlen": {
            "n_scans": int(self._rec.n_scans) if self._rec else None,
            "n_points": int(self._rec.n_points) if self._rec else None,
            "ebenen": sorted(self._layers),
        }}

        def job(progress_cb, cancel, log_cb):
            return bundle.export_project(
                proj, ziel, wahl, bag_paths=bags, calib_path=calib,
                meta_extra=extra, progress=progress_cb,
                cancel=lambda: cancel.is_set())

        def fertig(manifest: dict) -> None:
            drin = [k for k, v in manifest["inhalt"].items() if v]
            self._log(f"Projekt exportiert nach {ziel} "
                      f"({bundle.fmt_size(manifest['bytes'])}): "
                      + ", ".join(drin) + ".")
            if not manifest["inhalt"].get("bags"):
                self._log("Ohne Rosbag — auf einem anderen Rechner fehlen damit "
                          "360°-Video und erneutes Einfärben; Karte, Farben, "
                          "Messen und Export bleiben.")

        self._start_worker(f"Exportiere das Projekt nach {os.path.basename(ziel)} …",
                           job, fertig)

    def _on_open_project(self) -> None:
        """Projekt aus dem Cache oeffnen — auch zusammengefuehrte."""
        if self._beschaeftigt_melden(False):
            return
        from ui.bundle_dialog import ProjectOpenDialog
        projekte = Project.list_projects()
        if not projekte:
            QMessageBox.information(
                self, "Projekt öffnen",
                "Im Cache liegt noch kein berechnetes Projekt.")
            return
        dlg = ProjectOpenDialog(projekte, self)
        if dlg.exec_() != QDialog.Accepted:
            return
        d = dlg.gewaehlt()
        if not d:
            return
        eintrag = next((e for e in projekte if e["dir"] == d), None)
        try:
            project = Project.from_dir(d)
        except RuntimeError as exc:
            self._show_error("Projekt öffnen", str(exc))
            return
        self._oeffne_projekt(project, project.bag_path,
                             not eintrag or eintrag["zusammengefuehrt"])

    def _on_import_project(self) -> None:
        """Exportierten Projektordner direkt oeffnen (s. core.bundle.oeffnen).

        Keine Kopie in den Cache und darum auch keine Frage, ob ein vorhandenes
        Projekt ersetzt werden soll: gearbeitet wird im Ordner selbst, und jede
        Aenderung wird dort sofort gespeichert.
        """
        if self._beschaeftigt_melden(False):
            return
        src = QFileDialog.getExistingDirectory(
            self, "Ordner eines exportierten Projekts", os.path.expanduser("~"))
        if not src:
            return
        from core import bundle
        try:
            res = bundle.oeffnen(src)
        except RuntimeError as exc:
            self._show_error("Projekt öffnen", str(exc))
            return
        m = res["manifest"]
        self._log(f"Projektordner geöffnet: {src} (exportiert am "
                  f"{m.get('erstellt', '?')}). Änderungen werden direkt dort "
                  f"gespeichert.")
        for b in res["fehlende_bags"]:
            self._log(f"Rosbag nicht am Ort: {b} — 360°-Video und Einfärben aus "
                      f"der Bordkamera fallen für diesen Abschnitt aus.")
        vorhanden = [b for b in res["bags"] if b and os.path.exists(b)]
        self._oeffne_projekt(res["project"], vorhanden[0] if vorhanden else None,
                             res["zusammengefuehrt"], projekt=res["project"])

    def _oeffne_projekt(self, project, bag_path, zusammengefuehrt,
                        projekt=None) -> None:
        """Einzelne Fluege mit Bag am Ort gehen den normalen Weg — dann stehen
        auch das 360-Video und die GPS-Pruefung zur Verfuegung. Zusammengefuehrte
        haben keinen einzelnen Bagpfad und werden ohne Bag geoeffnet, ebenso ein
        Flug, dessen Bag fehlt. ``projekt`` geht an :meth:`_open_bag`."""
        if not zusammengefuehrt and bag_path and os.path.exists(bag_path):
            self._open_bag(bag_path, projekt=projekt)
        else:
            self._open_project_only(project)

    def _open_project_only(self, project) -> None:
        """Projekt ohne Bag oeffnen — nur, was aus dem Cache lebt."""
        self._clear_bag_state()
        self._project = project
        self._projekt_aktivieren(project)
        # Der Explorationsgrad gehoert zum Projekt. Ohne Bag laesst er sich nicht
        # neu rechnen, der gespeicherte Stand gilt aber weiter.
        gespeichert = project.load_exploration()
        try:
            self._exploration, hinweis = exploration.aus_cache(gespeichert)
        except (TypeError, ValueError) as exc:
            hinweis = ""
            self._log(f"Gespeicherter Explorationsgrad unbrauchbar: {exc}")
        self._zeige_exploration(hinweis)
        self.setWindowTitle(f"Super360 Studio — {project.bag_name} (ohne Bag)")
        self._update_enabled()
        if project.has_recording():
            self._start_recording_load()
