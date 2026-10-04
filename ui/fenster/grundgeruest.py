"""Grundgerüst: Protokoll, Freigabe, Arbeiter (Mixin des Hauptfensters).

Schreibt am Hauptfenster: _auto_kette, _autotest_failed, _busy, _retired,
_worker.
"""
from __future__ import annotations

import time
import traceback
from typing import Callable

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QMessageBox

from ui.jobs import Worker


class GrundgeruestMixin:
    # =========================================================== Grundgerüst

    def _log(self, line: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        for part in str(line).splitlines() or [""]:
            self._log_edit.appendPlainText(f"[{stamp}] {part}")

    def _show_error(self, title: str, msg: str) -> None:
        self._log(f"FEHLER — {title}: {msg}")
        if self._autotest_active():
            self._autotest_failed = True
            return
        QMessageBox.critical(self, title, msg)

    def _autotest_active(self) -> bool:
        return bool(getattr(self, "_autotest_path", None))

    def _update_enabled(self) -> None:
        busy = self._busy
        has_bag = self._bag is not None
        has_rec = self._rec is not None
        has_world = self._world is not None
        self._btn_open.setEnabled(not busy)
        self._btn_fastlio.setEnabled(not busy and has_bag)
        for b in (self._btn_autocal, self._btn_overlay, self._btn_colorize):
            b.setEnabled(not busy and has_bag and has_rec and bool(self._calib))
        for b in (self._btn_export_plypcd, self._btn_export_las, self._btn_mesh_cc,
                  self._btn_cloud_cc):
            b.setEnabled(not busy and has_world)
        # Schritte, die nur die Karte brauchen und ohne Bag weiterlaufen
        for b in (self._btn_merge_pick, self._btn_meander_pick,
                  self._btn_merge_auto, self._btn_merge_icp,
                  self._btn_merge_apply, self._btn_merge_drop):
            b.setEnabled(not busy and has_rec)
        # Maeander: erst mit gewaehltem Flug, und die Handregler erst, wenn
        # es eine Lage gibt, auf die sie sich beziehen koennen. Ein Regler,
        # der stillschweigend nichts tut, ist schlimmer als ein grauer.
        hat_flug = bool(self._meander_dir)
        for b in (self._btn_meander_align, self._btn_meander_run):
            b.setEnabled(not busy and has_rec and hat_flug)
            if not hat_flug:
                b.setToolTip("Erst einen Mäanderflug wählen.")
        hat_lage = (self._meander_pipe is not None
                    and getattr(self._meander_pipe, "yaw", None) is not None)
        self._btn_meander_fenster.setEnabled(not busy and hat_lage)
        for key, sp in self._spin_meander.items():
            sp.setEnabled(not busy and hat_lage)
            sp.setToolTip(
                "RGB: Zuschlag auf die gefundene Lage; wirkt sofort in der Wolke."
                if hat_lage else
                "Erst 'Ausrichten' laufen lassen — vorher gibt es keine Lage, "
                "auf die sich die Regler beziehen könnten.")
        self._btn_meander_optik.setEnabled(not busy and hat_lage)
        self._btn_meander_fein.setEnabled(not busy and hat_lage)
        self._btn_meander_auto.setEnabled(not busy and has_rec and hat_flug)
        self._btn_splat_pruefen.setEnabled(not busy)
        self._btn_splat_maeander.setEnabled(not busy and has_rec and hat_flug)
        self._btn_splat_onboard.setEnabled(not busy and has_bag and has_rec
                                           and bool(self._calib))
        self._btn_splat_gemeinsam.setEnabled(not busy and has_bag and has_rec
                                             and hat_flug and bool(self._calib))
        self._massstab["rgb"].setEnabled(not busy and hat_lage)
        hat_thermal = hat_lage and self._hat_thermal()
        self._massstab["thermal"].setEnabled(not busy and hat_thermal)
        for key, sp in self._spin_meander_th.items():
            sp.setEnabled(not busy and hat_thermal)
            if hat_thermal:
                sp.setToolTip(
                    "Thermal: Zuschlag auf die RGB-Lage; wirkt sofort und zeigt "
                    "dabei die Thermalvorschau.\nWird RGB verschoben, zieht "
                    "Thermal mit. Der Wert bleibt im Projekt gespeichert.")
            elif hat_lage:
                sp.setToolTip("Keine Thermalbilder in diesem Lauf — "
                              "„Thermalbilder mitrechnen“ anhaken und neu "
                              "ausrichten.")
            else:
                sp.setToolTip("Erst 'Ausrichten' laufen lassen.")
        self._lbl_meander_lage.setText(self._meander_zustand_text())
        for key, an in (("project_export", not busy and has_rec),
                        ("project_import", not busy),
                        ("measure", not busy and has_world),
                        ("fastlio", not busy and has_bag),
                        ("exploration", not busy and has_bag),
                        ("colorize", not busy and has_bag and has_rec
                         and bool(self._calib)),
                        ("meander_run", not busy and has_rec),
                        ("splat_meander", not busy and has_rec
                         and bool(self._meander_dir)),
                        ("splat_onboard", not busy and has_bag and has_rec
                         and bool(self._calib)),
                        ("splat_gemeinsam", not busy and has_bag and has_rec
                         and bool(self._meander_dir) and bool(self._calib)),
                        ("fusion", not busy and has_world
                         and self._fusion_quellen() is not None),
                        ("meander", not busy and has_rec),
                        ("merge", not busy and has_rec),
                        ("merge_apply", not busy and has_rec),
                        ("export_ply", not busy and has_world),
                        ("export_las", not busy and has_world)):
            act = self._actions.get(key)
            if act is not None:
                act.setEnabled(bool(an))
        can_cancel = busy and (self._worker is None or self._worker.cancellable)
        self._btn_cancel.setEnabled(can_cancel)
        self._btn_cancel.setToolTip(
            "Dieser Schritt kann nicht abgebrochen werden."
            if busy and not can_cancel else "")

    def _set_busy(self, busy: bool, text: str | None = None) -> None:
        self._busy = busy
        self._status_pbar.setVisible(busy)
        if busy:
            self._status_pbar.setRange(0, 0)  # unbestimmt bis erster Fortschritt
        if text is not None:
            self._status_lbl.setText(text)
        self._update_enabled()

    def _on_progress(self, frac: float, msg: str) -> None:
        self._status_pbar.setRange(0, 1000)
        self._status_pbar.setValue(int(max(0.0, min(1.0, frac)) * 1000))
        self._status_lbl.setText(msg)

    def _on_cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel.set()
            self._status_lbl.setText("Breche ab …")

    def _start_worker(self, text: str, job: Callable,
                      on_done: Callable[[object], None],
                      extra_progress: Callable[[float, str], None] | None = None,
                      on_failed: Callable[[str], None] | None = None,
                      cancellable: bool = True) -> None:
        if self._closing:
            return
        self._retired = [w for w in self._retired if not w.isFinished()]
        worker = Worker(job, self, cancellable=cancellable)
        self._worker = worker
        worker.progress.connect(self._on_progress)
        if extra_progress is not None:
            worker.progress.connect(extra_progress)
        worker.log.connect(self._log)
        worker.finished.connect(lambda res, w=worker, cb=on_done: self._worker_done(w, cb, res))
        worker.failed.connect(lambda msg, w=worker, t=text, cb=on_failed:
                              self._worker_failed(w, t, msg, cb))
        self._log(text)
        self._set_busy(True, text)
        worker.start()

    def _worker_done(self, worker: Worker, on_done: Callable, result: object) -> None:
        self._retire(worker)
        if self._closing:
            return  # Fenster schließt bereits — keine Folgeschritte mehr anstoßen
        self._set_busy(False, "Bereit")
        try:
            on_done(result)
        except Exception as exc:  # noqa: BLE001
            self._log(traceback.format_exc())
            self._show_error("Interner Fehler", str(exc))
        QTimer.singleShot(0, self._mesh_nachholen)

    def _worker_failed(self, worker: Worker, title: str, msg: str,
                       on_failed: Callable[[str], None] | None = None) -> None:
        self._retire(worker)
        if self._auto_kette:
            self._auto_kette = False
            self._log("Automatik angehalten — der Schritt davor ist fehlgeschlagen.")
        # Nach einem Fehlschlag darf keine Solo-Vorschau die Karte verdecken:
        # sonst sieht ein abgebrochener Lauf so aus, als sei das Modell weg.
        if self._cloud_view.has_color_preview():
            self._live_hide()
            self._log("Vorschau geräumt — die Karte ist wieder sichtbar.")
        if self._closing:
            return  # Fenster schließt bereits — keine Dialoge/Folgeschritte mehr
        self._set_busy(False, "Bereit")
        QTimer.singleShot(0, self._mesh_nachholen)
        if msg == "Abgebrochen":
            self._log(f"Abgebrochen — {title}")
            self._status_lbl.setText("Abgebrochen.")
            return
        if on_failed is not None:
            try:
                on_failed(msg)
            except Exception as exc:  # noqa: BLE001
                self._log(traceback.format_exc())
                self._show_error("Interner Fehler", str(exc))
            return
        self._show_error(title, msg)

    def _retire(self, worker: Worker) -> None:
        if self._worker is worker:
            self._worker = None
        self._retired.append(worker)
