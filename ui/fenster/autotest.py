"""Autotest (Mixin des Hauptfensters).

Schreibt am Hauptfenster: _autotest_deadline, _autotest_timer.
"""
from __future__ import annotations

import json
import os
import time

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication


class AutotestMixin:
    # ================================================================ Autotest

    def _autotest_begin(self) -> None:
        os.makedirs(self._autotest_out, exist_ok=True)
        self._log(f"[Autotest] Öffne {self._autotest_path} …")
        self._autotest_deadline = time.monotonic() + 420.0
        self._open_bag(self._autotest_path)
        self._autotest_timer = QTimer(self)
        self._autotest_timer.setInterval(500)
        self._autotest_timer.timeout.connect(self._autotest_poll)
        self._autotest_timer.start()

    def _autotest_poll(self) -> None:
        if self._autotest_failed:
            print("AUTOTEST FEHLGESCHLAGEN: Worker-Fehler (s. Protokoll).", flush=True)
            QApplication.instance().exit(2)
            return
        if time.monotonic() > self._autotest_deadline:
            print("AUTOTEST FEHLGESCHLAGEN: Zeitüberschreitung.", flush=True)
            QApplication.instance().exit(3)
            return
        if self._busy or self._worker is not None or self._bag is None:
            return
        if (self._calib and self._bag_info is not None
                and self._bag_info.camera_topic and self._pano_src is None
                and not self._pano_failed):
            return
        if self._project is not None and self._project.has_recording() and self._rec is None:
            return
        self._autotest_timer.stop()
        self._log("[Autotest] Artefakte geladen — erzeuge Screenshots.")
        self._autotest_shots()

    def _autotest_shots(self) -> None:
        names = ("tab1_3d_karte", "tab2_360_video", "tab3_gps", "tab4_protokoll")
        delay_show, delay_snap = 400, 1400
        t = 200
        for i, name in enumerate(names):
            QTimer.singleShot(t, lambda i=i: self._tabs.setCurrentIndex(i))
            t += delay_show
            QTimer.singleShot(t, lambda i=i, n=name: self._autotest_snap(i, n))
            t += delay_snap
        QTimer.singleShot(t, self._autotest_finish)

    def _autotest_snap(self, tab_idx: int, name: str) -> None:
        path = os.path.join(self._autotest_out, f"{name}.png")
        ok = self.grab().save(path)
        print(f"[Autotest] Screenshot Tab {tab_idx}: {path} ({'ok' if ok else 'FEHLER'})",
              flush=True)
        if tab_idx == 0:
            vtk_path = os.path.join(self._autotest_out, "cloud_vtk.png")
            try:
                self._cloud_view.screenshot(vtk_path)
                print(f"[Autotest] VTK-Screenshot: {vtk_path}", flush=True)
            except RuntimeError as exc:
                print(f"[Autotest] VTK-Screenshot fehlgeschlagen: {exc}", flush=True)

    def _autotest_finish(self) -> None:
        expected = [os.path.join(self._autotest_out, f"{n}.png")
                    for n in ("tab1_3d_karte", "tab2_360_video", "tab3_gps",
                              "tab4_protokoll")]
        missing = [p for p in expected if not os.path.isfile(p)]
        log_text = self._log_edit.toPlainText()
        with open(os.path.join(self._autotest_out, "protokoll.txt"), "w",
                  encoding="utf-8") as fh:
            fh.write(log_text)
        summary = {
            "bag": self._autotest_path,
            "n_frames": self._n_frames,
            "n_scans": self._rec.n_scans if self._rec else 0,
            "n_points": self._rec.n_points if self._rec else 0,
            "colors_loaded": self._colors is not None,
            "gps_usable": bool(self._quality.usable) if self._quality else None,
            "gps_reasons": list(self._quality.reasons) if self._quality else [],
            "pano_frames": self._pano_src.count if self._pano_src else 0,
            "log_lines": log_text.count("\n") + 1 if log_text else 0,
        }
        with open(os.path.join(self._autotest_out, "summary.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)
        if missing or self._autotest_failed:
            print(f"AUTOTEST FEHLGESCHLAGEN: fehlende Screenshots {missing}", flush=True)
            QApplication.instance().exit(2)
            return
        print(f"AUTOTEST OK: {json.dumps(summary, ensure_ascii=False)}", flush=True)
        QApplication.instance().exit(0)
