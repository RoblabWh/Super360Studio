"""Zusammenführen (Mixin des Hauptfensters).

Schreibt am Hauptfenster: _bag, _bag_info, _btn_merge_apply, _btn_merge_auto,
_btn_merge_drop, _btn_merge_icp, _btn_merge_pick, _colors, _georef, _lbl_merge,
_merge_bag, _merge_center, _merge_cloud, _merge_kipp_grad, _merge_rec,
_merge_T, _merge_T_basis, _merge_T_kipp, _pano_src, _parts, _project,
_spin_merge, _valid.
"""
from __future__ import annotations

import os

import numpy as np

from PyQt5.QtWidgets import (
    QFileDialog, QFormLayout, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QWidget,
)

from core.bag_reader import BagReader
from core.bag_reader import ThreadLocalBag
from core.gemeinsam import fmt_int as _fmt_int
from core.merge import lage_aus_reglern
from core.project import Project
from core.recording import lade_mit_hinweis

from ui.bausteine import _wrappable, still_setzen


class ZusammenfuehrenMixin:
    def _abschnitt_zusammen(self) -> QWidget:
        box = QWidget()
        form = _wrappable(QFormLayout(box))
        self._btn_merge_pick = QPushButton("Zweiten Flug wählen …")
        self._btn_merge_pick.setToolTip(
            "Zweites Rosbag dazuladen. Dessen Karte muss berechnet sein —\n"
            "sonst wird gefragt, ob sie jetzt berechnet werden soll.")
        self._btn_merge_pick.clicked.connect(self._on_merge_pick)
        form.addRow(self._btn_merge_pick)
        self._lbl_merge = QLabel("Kein zweiter Flug geladen.")
        self._lbl_merge.setWordWrap(True)
        form.addRow(self._lbl_merge)

        from ui.feinregler import regler_grad, regler_meter
        self._spin_merge = {}
        for key, label, sp in (("x", "X", regler_meter(500.0)),
                               ("y", "Y", regler_meter(500.0)),
                               ("z", "Z", regler_meter(500.0)),
                               ("yaw", "Gier", regler_grad())):
            sp.valueChanged.connect(lambda *_: self._merge_timer.start())
            form.addRow(label, sp)
            self._spin_merge[key] = sp

        row = QWidget()
        hl = QHBoxLayout(row)
        hl.setContentsMargins(0, 0, 0, 0)
        self._btn_merge_auto = QPushButton("Auto-Ausrichten")
        self._btn_merge_auto.setToolTip(
            "Globale Suche (FGR über FPFH) plus ICP von grob nach fein.\n"
            "Dauert je nach Wolkengröße ein bis mehrere Minuten.")
        self._btn_merge_auto.clicked.connect(self._on_merge_auto)
        self._btn_merge_icp = QPushButton("Nur ICP")
        self._btn_merge_icp.setToolTip(
            "Verfeinert nur die aktuelle Lage — nach einer Handjustage genug.")
        self._btn_merge_icp.clicked.connect(self._on_merge_icp)
        hl.addWidget(self._btn_merge_auto)
        hl.addWidget(self._btn_merge_icp)
        form.addRow(row)

        row2 = QWidget()
        hl2 = QHBoxLayout(row2)
        hl2.setContentsMargins(0, 0, 0, 0)
        self._btn_merge_apply = QPushButton("Übernehmen")
        self._btn_merge_apply.setToolTip(
            "Schreibt eine gemeinsame Aufzeichnung und öffnet sie als Arbeitswolke.\n"
            "Sie lässt sich danach als Ganzes einfärben und exportieren.")
        self._btn_merge_apply.clicked.connect(self._on_merge_apply)
        self._btn_merge_drop = QPushButton("Verwerfen")
        self._btn_merge_drop.clicked.connect(self._on_merge_discard)
        hl2.addWidget(self._btn_merge_apply)
        hl2.addWidget(self._btn_merge_drop)
        form.addRow(row2)
        return box

    # ========================================================= Zusammenführen

    def _merge_reset_state(self) -> None:
        self._merge_bag = None
        self._merge_rec = None
        self._merge_cloud = None
        self._merge_T = np.eye(4)
        self._merge_T_basis = np.eye(4)
        self._merge_center = np.zeros(3)
        self._merge_T_kipp = np.eye(4)
        self._merge_kipp_grad = None
        self._cloud_view.set_preview_cloud(None)
        self._lbl_merge.setText("Kein zweiter Flug geladen.")
        self._merge_spins_null()

    def _merge_T_from_spins(self) -> np.ndarray:
        """Handjustage: um den Schwerpunkt der zweiten Wolke gieren, dann schieben.

        Die Regler sind ein Versatz zur Lage der letzten Ausrichtung
        (``_merge_T_basis``), nicht zum Ladeort. Nach 'Nur ICP' stehen sie auf
        null; galten sie ab Ursprung, fiel die Wolke beim ersten Reglerschritt
        auf ihren Ladeort zurück.
        """
        return lage_aus_reglern(
            self._spin_merge["yaw"].value(),
            [self._spin_merge[k].value() for k in ("x", "y", "z")],
            self._merge_center, basis=self._merge_T_basis)

    def _merge_spins_null(self) -> None:
        """Regler der Handjustage auf null, ohne die Vorschau neu zu rechnen."""
        for sp in self._spin_merge.values():
            still_setzen(sp, 0.0)

    def _merge_refresh_preview(self) -> None:
        if self._merge_cloud is None:
            return
        pts = (self._merge_cloud @ self._merge_T[:3, :3].T) + self._merge_T[:3, 3]
        self._cloud_view.set_preview_cloud(pts.astype(np.float32))

    def _on_merge_manual(self) -> None:
        if self._merge_cloud is None:
            return
        self._merge_T = self._merge_T_from_spins()
        self._merge_refresh_preview()

    def _on_merge_pick(self) -> None:
        if self._busy or self._rec is None or self._project is None:
            QMessageBox.information(
                self, "Zusammenführen",
                "Erst einen Flug öffnen und seine Karte berechnen — der ist "
                "dann der Bezug, auf den der zweite gelegt wird.")
            return
        start = os.path.dirname(os.path.abspath(self._bag.bag_path))
        path = QFileDialog.getExistingDirectory(self, "Zweites Rosbag wählen", start)
        if not path:
            return
        path = os.path.abspath(path)
        if path == os.path.abspath(self._bag.bag_path):
            QMessageBox.warning(self, "Zusammenführen",
                                "Das ist derselbe Flug wie der offene.")
            return
        try:
            project_b = Project(path)
        except RuntimeError as exc:
            self._show_error("Zusammenführen", str(exc))
            return
        if not project_b.has_recording():
            try:
                with BagReader(path) as br:
                    dauer = br.info().duration
            except Exception as exc:  # noqa: BLE001
                self._show_error("Zusammenführen", f"Bag nicht lesbar: {exc}")
                return
            antwort = QMessageBox.question(
                self, "Karte fehlt",
                f"Für '{os.path.basename(path)}' ist noch keine Karte berechnet.\n\n"
                f"Der FAST-LIO-Lauf dauert ungefähr so lange wie der Flug, hier "
                f"etwa {dauer / 60.0:.1f} Minuten bei Rate 1×.\n\n"
                f"Jetzt berechnen?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if antwort != QMessageBox.Yes:
                self._log(f"Zusammenführen abgebrochen: '{os.path.basename(path)}' "
                          f"hat keine berechnete Karte.")
                return
            self._merge_run_fastlio(path, project_b)
            return
        self._merge_load_second(path, project_b)

    def _merge_run_fastlio(self, path: str, project_b) -> None:
        """FAST-LIO fuer den zweiten Flug, danach direkt weiter im Merge-Ablauf."""
        job, _config, _rate = self._fastlio_job(path, project_b.recording_dir())

        def fertig(result) -> None:
            self._log(f"Karte für den zweiten Flug fertig: {result.n_scans} Scans, "
                      f"{_fmt_int(result.n_points)} Punkte.")
            self._merge_load_second(path, project_b)

        self._start_worker(
            f"FAST-LIO2 für den zweiten Flug ({os.path.basename(path)}) …",
            job, fertig)

    def _merge_load_second(self, path: str, project_b) -> None:
        rec_dir = project_b.recording_dir()
        rec_a = self._rec
        teile_a = ([(p[0].bag_path, p[1], p[2]) for p in self._parts] if self._parts
                   else [(self._bag.bag_path, 0, int(rec_a.n_scans))])

        def job(progress_cb, cancel, log_cb):
            from core import merge as merge_mod
            progress_cb(0.1, "Lade zweite Aufzeichnung …")
            rec_b = lade_mit_hinweis(rec_dir, path, log_cb, praefix="Zweiter Flug — ")
            # Gleich lotrecht stellen wie den offenen Flug. Recording.load
            # schafft das nur mit Ruhefenster am Bag-Anfang; startete die
            # Aufnahme in der Luft, bliebe B um die Einbaulage gekippt und keine
            # Suche um die Hochachse faende die richtige Lage.
            progress_cb(0.3, "Lotrechte beider Flüge aus der IMU …")
            oben_b = merge_mod.lotrechte_aus_flug(rec_b, [(path, 0, rec_b.n_scans)])
            oben_a = merge_mod.lotrechte_aus_flug(rec_a, teile_a)
            if oben_a is None and rec_a.gravity_level is not None:
                oben_a = np.array([0.0, 0.0, 1.0])   # beim Laden lotrecht gestellt
            T_kipp, kipp = np.eye(4), None
            if oben_a is not None and oben_b is not None:
                T_kipp, kipp = merge_mod.kippausgleich(oben_a, oben_b)
                rec_b.poses = merge_mod.transform_poses(rec_b.poses, T_kipp)
                if kipp >= 1.0:
                    log_cb(f"Zweiter Flug um {kipp:.1f}° gekippt gegenüber dem "
                           f"offenen (Lotrechte aus der IMU über den ganzen Flug) "
                           f"— ausgeglichen.")
            else:
                log_cb("Zweiter Flug: Lotrechte nicht messbar — er bleibt, wie "
                       "FAST-LIO ihn liefert. Steht er schief, findet "
                       "Auto-Ausrichten ihn womöglich nicht.")
            progress_cb(0.6, "Dünne für die Vorschau aus …")
            wolke = merge_mod.cloud_for_registration(rec_b)
            return {"rec": rec_b, "cloud": wolke, "path": path,
                    "proxy": ThreadLocalBag(path), "T_kipp": T_kipp, "kipp": kipp}

        self._start_worker("Lade zweiten Flug …", job, self._on_merge_loaded)

    def _on_merge_loaded(self, res: dict) -> None:
        self._merge_rec = res["rec"]
        self._merge_bag = res["proxy"]
        self._merge_cloud = res["cloud"]
        self._merge_center = (self._merge_cloud.mean(axis=0)
                              if len(self._merge_cloud) else np.zeros(3))
        self._merge_T = np.eye(4)
        self._merge_T_basis = np.eye(4)
        self._merge_T_kipp = np.asarray(res["T_kipp"], dtype=np.float64)
        self._merge_kipp_grad = res["kipp"]
        self._merge_spins_null()
        self._merge_refresh_preview()
        name = os.path.basename(res["path"])
        self._lbl_merge.setText(
            f"{name}: {_fmt_int(self._merge_rec.n_points)} Punkte, "
            f"{self._merge_rec.n_scans} Scans — noch nicht ausgerichtet.")
        self._log(f"Zweiter Flug geladen: {name} "
                  f"({_fmt_int(self._merge_rec.n_points)} Punkte). Orange und "
                  f"halbdurchsichtig dargestellt — das ist eine VORSCHAU und "
                  f"gehört erst nach 'Übernehmen' zur Karte. Ausblenden über "
                  f"Ansicht → Zweiten Flug anzeigen.")
        self._sync_preview_action()

    def _on_merge_auto(self) -> None:
        self._on_merge_align("auto")

    def _on_merge_icp(self) -> None:
        self._on_merge_align("icp")

    def _on_merge_align(self, mode: str) -> None:
        if self._merge_rec is None or self._rec is None:
            QMessageBox.information(self, "Zusammenführen",
                                    "Erst einen zweiten Flug laden.")
            return
        rec_a = self._rec
        cloud_b = self._merge_cloud
        T_init = self._merge_T.copy()

        def job(progress_cb, cancel, log_cb):
            from core import merge as merge_mod
            progress_cb(0.02, "Dünne die erste Wolke aus …")
            cloud_a = merge_mod.cloud_for_registration(rec_a)
            return merge_mod.register(
                cloud_a, cloud_b, T_init=T_init, mode=mode,
                progress_cb=lambda f, m: progress_cb(0.05 + 0.95 * f, m),
                cancel=cancel)

        self._start_worker(
            "Richte aus (globale Suche + ICP) …" if mode == "auto"
            else "Verfeinere mit ICP …", job, self._on_merge_aligned)

    def _on_merge_aligned(self, res: dict) -> None:
        self._merge_T = np.asarray(res["T"], dtype=np.float64)
        # Neue Basis: die Regler gehen auf null und verschieben ab jetzt
        # relativ zu dieser Lage (s. _merge_T_from_spins).
        self._merge_T_basis = self._merge_T.copy()
        self._merge_refresh_preview()
        self._merge_spins_null()
        fit, rmse = res["fitness"], res["rmse"]
        self._lbl_merge.setText(
            f"Ausgerichtet über '{res['kandidat']}': Trefferquote {fit:.2f}, "
            f"Restfehler {rmse:.3f} m. Die Regler verschieben ab hier "
            f"relativ zu dieser Lage.")
        self._log(f"Ausrichtung: Kandidat '{res['kandidat']}', Trefferquote "
                  f"{fit:.2f}, Restfehler {rmse:.3f} m.")
        rang = res.get("rangliste") or []
        if len(rang) > 1:
            self._log("Bewertung der besten Startlagen (gleiche Zahl = gleiche "
                      "Lage gefunden): " + ", ".join(
                          f"{name} {wert:+.3f}" for wert, name, *_ in rang[:5]))
        if fit < 0.3:
            self._log("WARNUNG: Trefferquote unter 0,3 — die Wolken überlappen "
                      "vermutlich zu wenig. Von Hand grob zusammenschieben und "
                      "'Nur ICP' nachlaufen lassen.")
        elif rmse > 0.30:
            self._log(f"Hinweis: Restfehler {rmse:.2f} m ist für Innenräume viel. "
                      f"Die Lage stimmt grob, sitzt aber nicht sauber — vor dem "
                      f"Übernehmen im Viewer prüfen und ggf. von Hand nachziehen.")

    def _on_merge_discard(self) -> None:
        if self._merge_rec is None:
            return
        self._merge_reset_state()
        self._sync_preview_action()
        self._log("Zweiter Flug verworfen.")

    def _on_merge_apply(self) -> None:
        if self._merge_rec is None or self._rec is None or self._project is None:
            QMessageBox.information(self, "Zusammenführen",
                                    "Erst einen zweiten Flug laden und ausrichten.")
            return
        bag_a = os.path.abspath(self._bag.bag_path)
        bag_b = os.path.abspath(self._merge_bag.bag_path)
        ziel = os.path.join(os.path.dirname(bag_a),
                            f"{os.path.basename(bag_a)}+{os.path.basename(bag_b)}")
        project_m = Project(ziel)
        rec_a, rec_b = self._rec, self._merge_rec
        T = self._merge_T.copy()
        out_dir = project_m.recording_dir()
        # T_ab gilt fuer B nach dem Kippausgleich; von B wie geladen nach A
        # fuehrt T_ab @ T_kipp_b.
        info = {"fitness": None,
                "T_kipp_b": self._merge_T_kipp.tolist(),
                "kipp_b_grad": self._merge_kipp_grad}

        def job(progress_cb, cancel, log_cb):
            from core import merge as merge_mod
            meta = merge_mod.merge_recordings(
                rec_a, rec_b, T, out_dir, bag_a, bag_b,
                info=info, progress_cb=progress_cb, cancel=cancel)
            progress_cb(0.99, "Lade zusammengeführte Aufzeichnung …")
            return {"project": project_m, "meta": meta}

        self._start_worker("Führe die Flüge zusammen …", job, self._on_merge_applied)

    def _on_merge_applied(self, res: dict) -> None:
        project_m = res["project"]
        meta = res["meta"]
        quellen = meta["sources"]
        # Die zusammengefuehrte Aufzeichnung wird die Arbeitswolke. Der zeitlich
        # fruehere Flug fuehrt (Pano-Tab und GPS haengen an ihm), die Einfaerbung
        # bekommt ueber _parts fuer jeden Abschnitt die richtige Kamera.
        self._merge_reset_state()
        self._sync_preview_action()
        self._project = project_m
        self._parts = [(ThreadLocalBag(q["bag"]), q["scan_range"][0], q["scan_range"][1])
                       for q in quellen]
        self._bag = self._parts[0][0]
        try:
            self._bag_info = self._bag.info()
        except Exception as exc:  # noqa: BLE001
            self._log(f"Info des führenden Bags nicht lesbar: {exc}")
        self._colors = None
        self._valid = None
        self._georef = None
        self._pano_src = None
        self._pano_view.set_source(None)
        self.setWindowTitle(f"Super360 Studio — {project_m.bag_name}")
        self._log(f"Zusammengeführt: {meta['n_scans']} Scans, "
                  f"{_fmt_int(meta['n_points'])} Punkte aus "
                  + " + ".join(os.path.basename(q["bag"]) for q in quellen) + ".")
        self._log("Das 360°-Video und die GPS-Prüfung zeigen weiter den zeitlich "
                  "ersten Flug; die Einfärbung nutzt für jeden Abschnitt die "
                  "Kamera seines eigenen Bags.")
        self._start_recording_load()
