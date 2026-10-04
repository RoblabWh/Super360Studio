"""Wiedergabe (RViz) (Mixin des Hauptfensters).

Schreibt am Hauptfenster: _btn_rviz_replay, _btn_rviz_start, _btn_rviz_stop,
_lbl_rviz, _rviz_timer.
"""
from __future__ import annotations

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)


class WiedergabeMixin:
    def _abschnitt_wiedergabe(self) -> QWidget:
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.addWidget(QLabel("Spielt den geöffneten Bag in RViz ab."))
        row = QHBoxLayout()
        self._btn_rviz_start = QPushButton("Start")
        self._btn_rviz_stop = QPushButton("Stopp")
        self._btn_rviz_replay = QPushButton("Wiederholen")
        self._btn_rviz_start.clicked.connect(self._on_rviz_start)
        self._btn_rviz_stop.clicked.connect(self._on_rviz_stop)
        self._btn_rviz_replay.clicked.connect(self._on_rviz_replay)
        self._btn_rviz_replay.setToolTip(
            "Leert die RViz-Anzeige und spielt den Bag von vorn ab.")
        for b in (self._btn_rviz_start, self._btn_rviz_stop, self._btn_rviz_replay):
            row.addWidget(b)
        lay.addLayout(row)
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
        playing = self._rviz_player.is_playing()
        open_ = self._rviz_player.rviz_running()
        self._lbl_rviz.setText(
            "Läuft" if playing else ("RViz offen — Bag durchgelaufen"
                                     if open_ else "Gestoppt"))
        has_bag = self._bag is not None
        self._btn_rviz_start.setEnabled(has_bag and not playing and not self._busy)
        self._btn_rviz_stop.setEnabled(open_ or playing)
        self._btn_rviz_replay.setEnabled(open_ or playing)

    def _rviz_job(self, text: str, fn) -> None:
        """RvizPlayer-Aufruf im Worker (start/replay brauchen ggf. reindex)."""
        def job(progress_cb, cancel, log_cb):
            progress_cb(0.1, text)
            fn()
            progress_cb(1.0, text)
            return None
        self._start_worker(text, job, lambda _r: self._refresh_rviz_state(),
                           on_failed=lambda _m: self._refresh_rviz_state(),
                           cancellable=False)

    def _on_rviz_start(self) -> None:
        if self._bag is None:
            return
        bag = self._bag.bag_path
        self._rviz_job("Starte RViz-Wiedergabe …",
                       lambda: self._rviz_player.start(bag))

    def _on_rviz_stop(self) -> None:
        self._rviz_job("Stoppe RViz-Wiedergabe …",
                       lambda: self._rviz_player.stop())

    def _on_rviz_replay(self) -> None:
        self._rviz_job("Wiederhole (Anzeige wird geleert) …",
                       lambda: self._rviz_player.replay())
