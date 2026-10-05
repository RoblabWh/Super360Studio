"""Wiedergabe (RViz) (Mixin des Hauptfensters).

Schreibt am Hauptfenster: _btn_rviz_replay, _btn_rviz_start, _btn_rviz_stop,
_lbl_rviz, _rviz_text, _rviz_timer, _rviz_worker.
"""
from __future__ import annotations

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget

from ui.bausteine import knopfzeile
from ui.jobs import Worker

#: Die drei Befehle der Wiedergabe in der Befehlstabelle
_RVIZ_BEFEHLE = ("rviz_start", "rviz_stop", "rviz_replay")


class WiedergabeMixin:
    def _abschnitt_wiedergabe(self) -> QWidget:
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.addWidget(QLabel("Spielt den geöffneten Bag in RViz ab."))
        self._btn_rviz_start = self._befehlsknopf("rviz_start")
        self._btn_rviz_stop = self._befehlsknopf("rviz_stop")
        self._btn_rviz_replay = self._befehlsknopf("rviz_replay")
        lay.addWidget(knopfzeile(self._btn_rviz_start, self._btn_rviz_stop,
                                 self._btn_rviz_replay))
        self._lbl_rviz = QLabel("Gestoppt")
        lay.addWidget(self._lbl_rviz)
        # Knopfzustand an der echten Prozesslage ausrichten (der Player kann
        # auch von selbst enden, wenn der Bag durchgelaufen ist). Gestartet
        # wird der Takt am Ende von _build_ui.
        self._rviz_timer = QTimer(self)
        self._rviz_timer.timeout.connect(self._refresh_rviz_state)
        return box

    def _refresh_rviz_state(self) -> None:
        if self._closing:
            return
        if self._rviz_worker is not None:
            # Solange Beenden oder Wiederholen läuft, hält der Player zeitweise
            # seine Sperre; ihn jetzt zu fragen, könnte die Oberfläche anhalten.
            for key in _RVIZ_BEFEHLE:
                self._actions[key].setEnabled(False)
            return
        playing = self._rviz_player.is_playing()
        open_ = self._rviz_player.rviz_running()
        self._lbl_rviz.setText(
            "Läuft" if playing else ("RViz offen — Bag durchgelaufen"
                                     if open_ else "Gestoppt"))
        has_bag = self._bag is not None
        self._actions["rviz_start"].setEnabled(has_bag and not playing and not self._busy)
        self._actions["rviz_stop"].setEnabled(open_ or playing)
        self._actions["rviz_replay"].setEnabled(open_ or playing)

    def _rviz_job(self, text: str, fn, eigener: bool = False) -> None:
        """RvizPlayer-Aufruf im Worker (start/replay brauchen ggf. reindex).

        Starten ist ein Schritt wie jeder andere. Beenden und Wiederholen
        (``eigener``) laufen in einem eigenen Arbeiter, auch während ein
        Schritt läuft: er setzt weder _busy noch _worker, und solange er
        läuft, sind die drei Befehle der Wiedergabe gesperrt. Die Statuszeile
        zeigt ihn, solange kein Schritt läuft (s. _rviz_status).
        """
        def job(progress_cb, cancel, log_cb):
            progress_cb(0.1, text)
            fn()
            progress_cb(1.0, text)
            return None

        if eigener and self._closing:
            return
        if self._rviz_worker is not None:
            self._log(f"Nicht gestartet — die RViz-Wiedergabe ist noch beschäftigt: {text}")
            return
        if not eigener:
            self._start_worker(text, job, lambda _r: self._refresh_rviz_state(),
                               on_failed=lambda _m: self._refresh_rviz_state(),
                               cancellable=False)
            return
        worker = Worker(job, self, cancellable=False)
        self._rviz_worker = worker
        self._rviz_text = text
        worker.log.connect(self._log)
        worker.progress.connect(self._rviz_fortschritt)
        worker.finished.connect(lambda _r, w=worker: self._rviz_fertig(w))
        worker.failed.connect(lambda _m, w=worker: self._rviz_fertig(w, gescheitert=True))
        self._log(text)
        self._rviz_status()
        self._refresh_rviz_state()
        worker.start()

    def _rviz_status(self) -> None:
        """Den eigenen Arbeiter der Wiedergabe in der Statuszeile zeigen.

        Die Statuszeile gehört einem laufenden Schritt; nur ohne ihn stehen
        dort Text und Fortschritt der Wiedergabe.
        """
        if self._busy or self._rviz_worker is None:
            return
        self._status_pbar.setRange(0, 0)
        self._status_pbar.setVisible(True)
        self._status_lbl.setText(self._rviz_text)

    def _rviz_fortschritt(self, frac: float, msg: str) -> None:
        if not self._busy:
            self._on_progress(frac, msg)

    def _rviz_fertig(self, worker: Worker, gescheitert: bool = False) -> None:
        """Eigener Arbeiter der Wiedergabe ist durch, fertig oder gescheitert."""
        if self._rviz_worker is worker:
            self._rviz_worker = None
        self._retire(worker)
        if not self._busy and not self._closing:
            # Die Meldung des Fehlers steht schon im Protokoll (Worker.log)
            self._status_pbar.setVisible(False)
            self._status_lbl.setText(
                f"Fehlgeschlagen — {self._rviz_text.rstrip(' …')} (s. Protokoll)"
                if gescheitert else "Bereit")
        self._refresh_rviz_state()

    def _on_rviz_start(self) -> None:
        if self._bag is None:
            return
        bag = self._bag.bag_path
        self._rviz_job("Starte RViz-Wiedergabe …",
                       lambda: self._rviz_player.start(bag))

    def _on_rviz_stop(self) -> None:
        self._rviz_job("Stoppe RViz-Wiedergabe …",
                       lambda: self._rviz_player.stop(), eigener=True)

    def _on_rviz_replay(self) -> None:
        self._rviz_job("Wiederhole (Anzeige wird geleert) …",
                       lambda: self._rviz_player.replay(), eigener=True)
