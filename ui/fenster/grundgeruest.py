"""Grundgerüst: Protokoll, Freigabe, Arbeiter (Mixin des Hauptfensters).

Schreibt am Hauptfenster: _auto_kette, _autotest_failed, _busy, _retired,
_worker.
"""
from __future__ import annotations

import time
import traceback
from typing import Callable

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QMessageBox, QPushButton

from ui import menubar as menubar_mod
from ui.bausteine import aktionsknopf
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

    def _befehlsknopf(self, schluessel: str) -> QPushButton:
        """Knopf der Seitenleiste zum Befehl ``schluessel`` der Befehlstabelle.

        Er folgt seiner Aktion (Freigabe, Tooltip samt fehlender Voraussetzung)
        und trägt den Knopftext der Tabelle.
        """
        return aktionsknopf(self._actions[schluessel],
                            menubar_mod.BEFEHLE[schluessel].knopf)

    def _update_enabled(self) -> None:
        busy = self._busy
        lage = self._hat_lage
        # Was die Befehle der Tabelle voraussetzen (menubar.GRUENDE); ob es
        # Thermalbilder gibt, weiß erst die ausgerichtete Pipeline.
        zustand = {
            "bag": self._bag is not None,
            "rec": self._rec is not None,
            "world": self._world is not None,
            "calib": bool(self._calib),
            "flug": bool(self._meander_dir),
            "lage": lage,
            "thermal": lage and self._hat_thermal(),
            "zweitflug": self._merge_rec is not None,
            "fusion": self._fusion_quellen() is not None,
        }
        # Die Knöpfe der Seitenleiste folgen ihren Aktionen.
        menubar_mod.schalte(self._actions, zustand, busy)
        # Was kein Befehl ist: der Lagetext des Mäanders und der Abbrechen-Knopf
        self._freigabe_maeander(zustand, busy)
        can_cancel = busy and (self._worker is None or self._worker.cancellable)
        self._btn_cancel.setEnabled(can_cancel)
        self._btn_cancel.setToolTip(
            "Dieser Schritt kann nicht abgebrochen werden."
            if busy and not can_cancel else "")
        # Starten der RViz-Wiedergabe hängt am laufenden Schritt
        self._refresh_rviz_state()

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
                      on_failed: Callable[[str], None] | None = None,
                      cancellable: bool = True) -> None:
        if self._closing:
            return
        if self._busy:
            # Ein Schritt nach dem anderen: ein zweiter Arbeiter überschriebe
            # self._worker, den laufenden könnte dann niemand mehr abbrechen
            # oder beim Schließen abwarten.
            self._log(f"Nicht gestartet — es läuft noch ein Arbeitsschritt: {text}")
            self._auto_kette = False
            QMessageBox.information(
                self, "Beschäftigt",
                f"„{text.rstrip(' …')}“ wurde nicht gestartet — es läuft noch ein "
                "Arbeitsschritt. Bitte warten oder abbrechen.")
            return
        self._retired = [w for w in self._retired if not w.isFinished()]
        worker = Worker(job, self, cancellable=cancellable)
        self._worker = worker
        worker.progress.connect(self._on_progress)
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
