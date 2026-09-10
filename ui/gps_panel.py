"""GpsPanel: GPS-Qualitätsbericht (Ampel, Kennzahlen, Gründe) + Georeferenzierung."""
from __future__ import annotations

from typing import Optional, Sequence

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QListWidget, QMessageBox,
    QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

try:
    from core import georef
except ImportError:  # direkter Skript-Start (Selbsttest): Paketwurzel nachrüsten
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core import georef

_DOT_COLORS = {"rot": "#e53935", "gelb": "#fdd835", "gruen": "#43a047",
               "grau": "#757575"}
_FIX_TYPE_NAMES = {0: "kein GPS", 1: "kein Fix", 2: "2D", 3: "3D", 4: "DGPS",
                   5: "RTK Float", 6: "RTK Fixed", 7: "statisch", 8: "PPP"}


class _AlignWorker(QThread):
    """Kleiner lokaler Worker: führt georef.align im Hintergrund aus."""

    progressed = pyqtSignal(float, str)
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, rec, fixes, quality, parent: QWidget | None = None):
        super().__init__(parent)
        self._rec, self._fixes, self._quality = rec, fixes, quality

    def run(self) -> None:
        try:
            result = georef.align(
                self._rec, self._fixes, self._quality,
                progress_cb=lambda f, m: self.progressed.emit(f, m))
            self.succeeded.emit(result)
        except Exception as exc:  # noqa: BLE001 — UI zeigt die Meldung
            self.failed.emit(str(exc))


class GpsPanel(QWidget):
    """Zeigt GpsQuality + startet die Georeferenzierung (QThread-Worker)."""

    georefReady = pyqtSignal(object)  # GeorefResult

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._quality: Optional[georef.GpsQuality] = None
        self._fixes: Optional[Sequence] = None
        self._recording = None
        self._worker: Optional[_AlignWorker] = None
        # Generationszähler: jede set_quality-Übergabe entwertet Ergebnisse
        # noch laufender Align-Worker (verhindert Bag-übergreifend veraltete
        # Georeferenzierungen).
        self._gen = 0

        root = QVBoxLayout(self)
        root.setSpacing(10)

        # Ampel
        head = QHBoxLayout()
        self._dot = QLabel()
        self._dot.setFixedSize(22, 22)
        self._status_lbl = QLabel("Keine GPS-Daten geladen")
        self._status_lbl.setStyleSheet("font-weight: bold;")
        head.addWidget(self._dot)
        head.addWidget(self._status_lbl, 1)
        root.addLayout(head)
        self._set_dot("grau")

        # Kennzahlen
        metrics_box = QGroupBox("Kennzahlen")
        form = QFormLayout(metrics_box)
        self._lbl_total = QLabel("–")
        self._lbl_good = QLabel("–")
        self._lbl_hist = QLabel("–")
        self._lbl_hist.setWordWrap(True)
        self._lbl_sats = QLabel("–")
        self._lbl_eph = QLabel("–")
        self._lbl_baseline = QLabel("–")
        form.addRow("Fixe gesamt:", self._lbl_total)
        form.addRow("Gute Fixe:", self._lbl_good)
        form.addRow("fix_type-Histogramm:", self._lbl_hist)
        form.addRow("Satelliten (Median):", self._lbl_sats)
        form.addRow("eph (Median):", self._lbl_eph)
        form.addRow("Baseline:", self._lbl_baseline)
        root.addWidget(metrics_box)

        # Gründe
        reasons_box = QGroupBox("Gründe / Hinweise")
        rb_lay = QVBoxLayout(reasons_box)
        self._reasons = QListWidget()
        self._reasons.setSelectionMode(QListWidget.NoSelection)
        self._reasons.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        rb_lay.addWidget(self._reasons)
        root.addWidget(reasons_box, 1)

        # Aktion
        self._btn = QPushButton("Georeferenzierung ausführen")
        self._btn.setEnabled(False)
        self._btn.clicked.connect(self._start_align)
        root.addWidget(self._btn)
        self._progress_lbl = QLabel("")
        self._progress_lbl.setWordWrap(True)
        root.addWidget(self._progress_lbl)

        # Ergebnis
        self._result_box = QGroupBox("Ergebnis der Georeferenzierung")
        res_form = QFormLayout(self._result_box)
        self._lbl_epsg = QLabel("–")
        self._lbl_rms = QLabel("–")
        self._lbl_used = QLabel("–")
        self._lbl_origin = QLabel("–")
        self._lbl_utm = QLabel("–")
        res_form.addRow("UTM-EPSG:", self._lbl_epsg)
        res_form.addRow("RMS:", self._lbl_rms)
        res_form.addRow("Verwendete Fixe:", self._lbl_used)
        res_form.addRow("Ursprung (Lat/Lon/Alt):", self._lbl_origin)
        res_form.addRow("UTM-Offset (E/N):", self._lbl_utm)
        self._hint_lbl = QLabel("")
        self._hint_lbl.setWordWrap(True)
        self._hint_lbl.setStyleSheet("color: #4FC3F7;")
        res_form.addRow(self._hint_lbl)
        self._result_box.setVisible(False)
        root.addWidget(self._result_box)

    # ------------------------------------------------------------------ API
    def set_quality(self, quality: georef.GpsQuality, fixes: Sequence,
                    recording) -> None:
        """Neue Daten anzeigen. ``recording`` darf None sein (noch keine Karte)."""
        self._gen += 1  # laufende Align-Worker liefern ab jetzt veraltete Ergebnisse
        self._quality, self._fixes, self._recording = quality, fixes, recording
        self._result_box.setVisible(False)
        self._progress_lbl.setText("")

        if quality is None:
            self._set_dot("grau")
            self._status_lbl.setText("Keine GPS-Daten geladen")
            self._btn.setEnabled(False)
            return

        if not quality.usable:
            self._set_dot("rot")
            self._status_lbl.setText("GPS unbrauchbar — keine Georeferenzierung möglich")
        elif ((quality.median_eph_cm is not None and quality.median_eph_cm > 200)
              or quality.n_good < 0.5 * max(quality.n_total, 1)):
            self._set_dot("gelb")
            self._status_lbl.setText("GPS eingeschränkt nutzbar")
        else:
            self._set_dot("gruen")
            self._status_lbl.setText("GPS-Qualität gut")

        self._lbl_total.setText(str(quality.n_total))
        self._lbl_good.setText(str(quality.n_good))
        if quality.fix_type_hist:
            self._lbl_hist.setText(", ".join(
                f"{ft} ({_FIX_TYPE_NAMES.get(ft, '?')}): {n}"
                for ft, n in sorted(quality.fix_type_hist.items())))
        else:
            self._lbl_hist.setText("kein GPSRAW vorhanden")
        self._lbl_sats.setText(
            f"{quality.median_sats:.0f}" if quality.median_sats is not None
            else "unbekannt")
        self._lbl_eph.setText(
            f"{quality.median_eph_cm:.0f} cm" if quality.median_eph_cm is not None
            else "unbekannt")
        self._lbl_baseline.setText(f"{quality.baseline_m:.1f} m")

        self._reasons.clear()
        if quality.reasons:
            self._reasons.addItems(quality.reasons)
        elif quality.usable:
            self._reasons.addItem("Keine Beanstandungen — Georeferenzierung möglich.")

        can_run = bool(quality.usable) and self._worker is None
        self._btn.setEnabled(can_run and recording is not None)
        if quality.usable and recording is None:
            self._btn.setToolTip("Zuerst Punktwolke berechnen (FAST-LIO2).")
            self._progress_lbl.setText(
                "Hinweis: Zuerst die Karte berechnen, dann georeferenzieren.")
        else:
            self._btn.setToolTip("")

    # ------------------------------------------------------------- intern
    def _set_dot(self, color_key: str) -> None:
        self._dot.setStyleSheet(
            f"background-color: {_DOT_COLORS[color_key]};"
            "border-radius: 11px; border: 1px solid rgba(0,0,0,80);")

    def _start_align(self) -> None:
        if self._worker is not None or self._quality is None:
            return
        if self._recording is None:
            QMessageBox.warning(self, "Georeferenzierung",
                                "Keine Punktwolken-Aufzeichnung vorhanden — "
                                "zuerst die Karte berechnen.")
            return
        self._btn.setEnabled(False)
        self._progress_lbl.setText("Georeferenzierung läuft …")
        gen = self._gen  # Datensatz-Generation beim Start festhalten
        self._worker = _AlignWorker(self._recording, self._fixes, self._quality, self)
        self._worker.progressed.connect(
            lambda f, m: self._progress_lbl.setText(f"{int(f * 100)} % — {m}"))
        self._worker.succeeded.connect(lambda res, g=gen: self._on_success(res, g))
        self._worker.failed.connect(lambda msg, g=gen: self._on_failed(msg, g))
        self._worker.finished.connect(self._on_worker_done)
        self._worker.start()

    def _on_worker_done(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
        self._worker = None
        usable = self._quality is not None and self._quality.usable
        self._btn.setEnabled(usable and self._recording is not None)

    def _on_success(self, result: object, gen: int) -> None:
        if gen != self._gen:
            # Ergebnis gehört zu einem inzwischen ersetzten Datensatz
            # (anderes Bag / neue Aufzeichnung) — verwerfen, kein Signal.
            self._progress_lbl.setText(
                "Veraltetes Georeferenzierungs-Ergebnis verworfen "
                "(Daten wurden inzwischen gewechselt).")
            return
        res: georef.GeorefResult = result  # type: ignore[assignment]
        lat, lon, alt = res.origin_llh
        self._lbl_epsg.setText(f"EPSG:{res.utm_epsg}")
        self._lbl_rms.setText(f"{res.rms_m:.2f} m")
        self._lbl_used.setText(str(res.n_used))
        self._lbl_origin.setText(f"{lat:.7f}°, {lon:.7f}°, {alt:.1f} m")
        self._lbl_utm.setText(f"{res.utm_offset[0]:.2f} m, {res.utm_offset[1]:.2f} m")
        self._hint_lbl.setText(
            f"Der LAS-Export verwendet nun UTM-Koordinaten (EPSG:{res.utm_epsg}). "
            "PLY/PCD bleiben im lokalen LIO-Koordinatensystem.")
        self._result_box.setVisible(True)
        self._progress_lbl.setText("Georeferenzierung abgeschlossen.")
        self.georefReady.emit(res)

    def _on_failed(self, message: str, gen: int) -> None:
        if gen != self._gen:
            return  # Fehler eines inzwischen ersetzten Datensatzes — nicht anzeigen
        self._progress_lbl.setText("Georeferenzierung fehlgeschlagen.")
        QMessageBox.critical(self, "Georeferenzierung fehlgeschlagen", message)

    def shutdown(self) -> None:
        """Vor dem Beenden der Anwendung aufrufen: wartet auf den Align-Worker.

        Verhindert Qts »QThread: Destroyed while thread is still running«-Abort,
        wenn das Fenster während einer laufenden Georeferenzierung geschlossen
        wird. georef.align kennt kein Abbruch-Event, daher großzügig gedeckelt
        warten (30 s — align braucht auch auf langen Bags nur Sekunden).
        """
        worker = self._worker
        if worker is not None and worker.isRunning():
            worker.wait(30000)


# --------------------------------------------------------------------------
if __name__ == "__main__":
    import math
    import os
    import sys
    from types import SimpleNamespace

    import numpy as np

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtCore import QEventLoop, QTimer
    from PyQt5.QtWidgets import QApplication

    EVID = ("/tmp/super360_modtests/"
            "gps_panel")
    os.makedirs(EVID, exist_ok=True)
    app = QApplication(sys.argv)

    # --- synthetischer usable-Fall (Gerade, 100 m, Yaw 25°) ---
    class StubRec:
        def __init__(self, stamps, positions):
            self.stamps, self.positions = stamps, positions

        def interpolate_pose(self, t):
            if t < self.stamps[0] - 0.15 or t > self.stamps[-1] + 0.15:
                return None
            t = float(np.clip(t, self.stamps[0], self.stamps[-1]))
            T = np.eye(4)
            T[:3, 3] = [float(np.interp(t, self.stamps, self.positions[:, k]))
                        for k in range(3)]
            return T

        def path_positions(self):
            return self.positions

    rng = np.random.default_rng(7)
    stamps = 500.0 + np.linspace(0.0, 40.0, 200)
    traj = np.column_stack([np.linspace(0.0, 100.0, 200), np.zeros(200), np.zeros(200)])
    rec = StubRec(stamps, traj)
    yaw = math.radians(25.0)
    R = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                  [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1.0]])
    fix_t = 500.0 + np.linspace(0.5, 39.5, 60)
    p = np.column_stack([np.interp(fix_t, stamps, traj[:, k]) for k in range(3)])
    q = p @ R.T + np.array([5.0, 8.0, 2.0]) + rng.normal(0, 0.4, p.shape)
    llh = georef.enu_to_llh(q, (51.574, 7.027, 60.0))
    fixes_syn = [SimpleNamespace(
        stamp=float(fix_t[i]), lat=float(llh[i, 0]), lon=float(llh[i, 1]),
        alt=float(llh[i, 2]), status=0, service=1, cov_east_m=0.6, cov_north_m=0.6,
        cov_up_m=1.0, cov_type=2, fix_type=4, eph_cm=70, epv_cm=110, satellites=15)
        for i in range(len(fix_t))]
    qual_syn = georef.assess(fixes_syn)
    print(f"synthetisch: usable={qual_syn.usable} n_good={qual_syn.n_good} "
          f"baseline={qual_syn.baseline_m:.1f} m")
    assert qual_syn.usable

    panel = GpsPanel()
    panel.resize(560, 820)
    panel.show()
    panel.set_quality(qual_syn, fixes_syn, rec)
    app.processEvents()
    assert panel._btn.isEnabled(), "Knopf muss bei usable+Recording aktiv sein"
    p1 = os.path.join(EVID, "panel_synthetic_usable.png")
    panel.grab().save(p1)
    print("PNG:", p1)

    # Georeferenzierung über den Knopf (Worker-Pfad) ausführen
    results = []
    panel.georefReady.connect(results.append)
    loop = QEventLoop()
    panel.georefReady.connect(lambda _res: loop.quit())
    QTimer.singleShot(15000, loop.quit)
    panel._btn.click()
    loop.exec_()
    app.processEvents()
    assert results, "georefReady wurde nicht emittiert"
    res = results[0]
    print(f"GeorefResult: EPSG={res.utm_epsg} RMS={res.rms_m:.3f} m "
          f"n_used={res.n_used} origin=({res.origin_llh[0]:.6f}, "
          f"{res.origin_llh[1]:.6f}, {res.origin_llh[2]:.1f})")
    yaw_est = math.degrees(math.atan2(res.T_enu_world[1, 0], res.T_enu_world[0, 0]))
    print(f"Yaw geschätzt: {yaw_est:.2f}° (wahr 25°)")
    assert abs(yaw_est - 25.0) < 1.0
    p2 = os.path.join(EVID, "panel_synthetic_result.png")
    panel.grab().save(p2)
    print("PNG:", p2)

    # --- realer seg0-Fall (GPS tot) ---
    from pathlib import Path

    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_types_from_msg, get_typestore

    ts_store = get_typestore(Stores.ROS2_HUMBLE)
    ts_store.register(get_types_from_msg(
        Path("/opt/ros/humble/share/mavros_msgs/msg/GPSRAW.msg").read_text(),
        "mavros_msgs/msg/GPSRAW"))
    nav, raws = [], []
    bagp = Path("/home/lena/RosBagSuper_Gui/rosbag_2026-07-11_15-37-07_seg0")
    with AnyReader([bagp], default_typestore=ts_store) as reader:
        conns = [cn for cn in reader.connections
                 if cn.topic in ("/mavros/global_position/raw/fix",
                                 "/mavros/gpsstatus/gps1/raw")]
        for conn, _, raw in reader.messages(connections=conns):
            m = reader.deserialize(raw, conn.msgtype)
            st = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            (nav if conn.msgtype.endswith("NavSatFix") else raws).append((st, m))
    raw_stamps = np.array([r[0] for r in raws])
    fixes_real = []
    for st, m in nav:
        j = int(np.argmin(np.abs(raw_stamps - st))) if len(raws) else -1
        gr = raws[j][1] if j >= 0 and abs(raw_stamps[j] - st) <= 0.3 else None
        cov = np.asarray(m.position_covariance, dtype=np.float64)
        fixes_real.append(SimpleNamespace(
            stamp=st, lat=float(m.latitude), lon=float(m.longitude),
            alt=float(m.altitude), status=int(m.status.status),
            service=int(m.status.service),
            cov_east_m=float(np.sqrt(max(cov[0], 0.0))),
            cov_north_m=float(np.sqrt(max(cov[4], 0.0))),
            cov_up_m=float(np.sqrt(max(cov[8], 0.0))),
            cov_type=int(m.position_covariance_type),
            fix_type=int(gr.fix_type) if gr else None,
            eph_cm=int(gr.eph) if gr else None,
            epv_cm=int(gr.epv) if gr else None,
            satellites=int(gr.satellites_visible) if gr else None))
    qual_real = georef.assess(fixes_real)
    print(f"seg0: usable={qual_real.usable} n_total={qual_real.n_total} "
          f"n_good={qual_real.n_good} Gründe={len(qual_real.reasons)}")
    assert not qual_real.usable

    panel.set_quality(qual_real, fixes_real, None)
    app.processEvents()
    assert not panel._btn.isEnabled(), "Knopf muss bei unusable deaktiviert sein"
    p3 = os.path.join(EVID, "panel_seg0_unusable.png")
    panel.grab().save(p3)
    print("PNG:", p3)

    print("ALLE GPSPANEL-SELBSTTESTS BESTANDEN")
