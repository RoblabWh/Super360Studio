"""PanoView — 360°-Panorama-Player für Super360 Studio.

QGraphicsView-basierter Player mit Mausrad-Zoom (10 %–1600 %, unter dem
Mauszeiger verankert), Drag-Pan, stamps-basiertem Playback (Wanduhr-verankert:
bei Überlast werden Frames übersprungen, nie nachgezogen) und einem
Prefetch-QThread, der als EINZIGER ``PanoSource.get_pano`` aufruft.

Bilder sind core-konventionsgemäß BGR; erst hier wird zu RGB/QImage gewandelt.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QKeySequence, QPainter
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QShortcut,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ui.bausteine import bgr_zu_pixmap


# --------------------------------------------------------------------------- #
# Quellen
# --------------------------------------------------------------------------- #
class PanoSource:
    """Abstrakte, threadsichere Quelle für Panorama-Frames (BGR uint8)."""

    count: int = 0
    stamps: np.ndarray = np.zeros(0, dtype=np.float64)
    fps: float = 0.0

    def get_pano(self, idx: int) -> np.ndarray:
        raise NotImplementedError


class StitchingPanoSource(PanoSource):
    """BagReader + EquirectStitcher mit JPEG-Disk-Cache (Qualität 92) und RAM-LRU(8).

    ``cache_dir`` entspricht ``cache/<bag>/pano_<W>/`` (index.json + %06d.jpg).
    """

    _LRU_SIZE = 8
    _JPEG_QUALITY = 92

    def __init__(self, bag, calib_json: str, width: int, cache_dir: str):
        # Lazy-Import: core-Module werden nur hier gebraucht; der Selbsttest
        # dieses Moduls läuft mit synthetischer Quelle völlig ohne core/.
        try:
            from ..core.stitcher import EquirectStitcher  # type: ignore
        except ImportError:
            from core.stitcher import EquirectStitcher  # type: ignore

        self._bag = bag
        self._width = int(width)
        self._dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # Cache-Identität: md5 der Kalibrierungsdatei + Breite. Passt der
        # vorhandene index.json nicht (oder fehlt der Schlüssel — Altbestand),
        # wurden die JPEGs mit einer anderen Kalibrierung gestitcht → einmal
        # den kompletten pano_<W>-Inhalt wischen und index.json neu schreiben.
        try:
            with open(calib_json, "rb") as fh:
                self._calib_md5 = hashlib.md5(fh.read()).hexdigest()
        except OSError as exc:
            raise RuntimeError(
                f"Kalibrierung nicht lesbar: {calib_json} ({exc})"
            ) from exc
        index_path = os.path.join(cache_dir, "index.json")
        if os.path.exists(index_path) and not self._index_matches(index_path):
            self._wipe_cache_dir(cache_dir)

        self._stitcher = EquirectStitcher(
            calib_json, width=self._width, lut_cache_dir=cache_dir
        )
        st = np.asarray(bag.camera_stamps(), dtype=np.float64)
        self.stamps = st
        self.count = int(st.size)
        if self.count >= 2 and st[-1] > st[0]:
            self.fps = float((self.count - 1) / (st[-1] - st[0]))
        else:
            self.fps = 20.0

        self._lock = threading.RLock()
        self._lru: "OrderedDict[int, np.ndarray]" = OrderedDict()

        if not os.path.exists(index_path):
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "stamps": [float(s) for s in st],
                        "width": self._width,
                        "calib_md5": self._calib_md5,
                    },
                    f,
                )

    def _index_matches(self, index_path: str) -> bool:
        """True, wenn index.json zur aktuellen Kalibrierung + Breite passt."""
        try:
            with open(index_path, encoding="utf-8") as fh:
                idx = json.load(fh)
            return (
                isinstance(idx, dict)
                and idx.get("calib_md5") == self._calib_md5
                and int(idx.get("width", -1)) == self._width
            )
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            return False

    @staticmethod
    def _wipe_cache_dir(cache_dir: str) -> None:
        """Alle Dateien (JPEGs, index.json, LUT-npz) im Pano-Cache entfernen."""
        for name in os.listdir(cache_dir):
            path = os.path.join(cache_dir, name)
            if os.path.isfile(path):
                try:
                    os.remove(path)
                except OSError:
                    pass  # best effort; Breiten-Check in get_pano bleibt als Netz

    def _jpg_path(self, idx: int) -> str:
        return os.path.join(self._dir, "%06d.jpg" % idx)

    def get_pano(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= self.count:
            raise IndexError(f"Frame-Index {idx} außerhalb 0..{self.count - 1}")
        with self._lock:
            if idx in self._lru:
                self._lru.move_to_end(idx)
                return self._lru[idx]
            pano: Optional[np.ndarray] = None
            path = self._jpg_path(idx)
            if os.path.exists(path):
                pano = cv2.imread(path, cv2.IMREAD_COLOR)
                if pano is not None and pano.shape[1] != self._width:
                    pano = None  # veralteter Cache-Eintrag
            if pano is None:
                raw = self._bag.read_camera(idx)
                pano = self._stitcher.stitch(raw)
                cv2.imwrite(
                    path, pano, [int(cv2.IMWRITE_JPEG_QUALITY), self._JPEG_QUALITY]
                )
            self._lru[idx] = pano
            while len(self._lru) > self._LRU_SIZE:
                self._lru.popitem(last=False)
            return pano


# --------------------------------------------------------------------------- #
# Prefetch-Thread
# --------------------------------------------------------------------------- #
class _Prefetcher(QThread):
    """Einziger Aufrufer von ``PanoSource.get_pano``; füllt einen Dict-Cache."""

    panoReady = pyqtSignal(int)

    _MAX_CACHE = 12

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cond = threading.Condition()
        self._source: Optional[PanoSource] = None
        self._gen = 0  # Generationszähler: entwertet in-flight-Ergebnisse
        self._wanted: list[int] = []
        self._cache: dict[int, np.ndarray] = {}
        self._center = 0
        self._quit = False

    def set_source(self, src: Optional[PanoSource]) -> None:
        with self._cond:
            self._source = src
            self._gen += 1
            self._wanted = []
            self._cache.clear()
            self._center = 0
            self._cond.notify()

    def request(self, indices: list[int]) -> None:
        """Ersetzt die Wunschliste (neuester Bedarf gewinnt → Frame-Drops)."""
        with self._cond:
            if indices:
                self._center = indices[0]
            self._wanted = [i for i in indices if i not in self._cache]
            self._cond.notify()

    def get(self, idx: int) -> Optional[np.ndarray]:
        with self._cond:
            return self._cache.get(idx)

    def shutdown(self) -> None:
        with self._cond:
            self._quit = True
            self._cond.notify()
        self.wait(5000)

    def run(self) -> None:
        while True:
            with self._cond:
                while not self._quit and (self._source is None or not self._wanted):
                    self._cond.wait()
                if self._quit:
                    return
                idx = self._wanted.pop(0)
                src = self._source
                gen = self._gen
            try:
                pano = src.get_pano(idx)  # bewusst außerhalb des Locks
            except Exception:
                pano = None
            with self._cond:
                if gen != self._gen:
                    continue  # Quelle gewechselt → Ergebnis verwerfen
                if pano is not None:
                    self._cache[idx] = pano
                    if len(self._cache) > self._MAX_CACHE:
                        far = sorted(
                            self._cache,
                            key=lambda k: abs(k - self._center),
                            reverse=True,
                        )
                        for k in far[: len(self._cache) - self._MAX_CACHE]:
                            del self._cache[k]
            if pano is not None:
                self.panoReady.emit(idx)


# --------------------------------------------------------------------------- #
# Zoombare Ansicht
# --------------------------------------------------------------------------- #
class _ZoomGraphicsView(QGraphicsView):
    ZOOM_MIN = 0.1
    ZOOM_MAX = 16.0
    ZOOM_STEP = 1.25

    def __init__(self, scene: QGraphicsScene, parent=None):
        super().__init__(scene, parent)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setBackgroundBrush(QColor(18, 18, 18))
        self.user_zoomed = False
        self._fit_item: Optional[QGraphicsPixmapItem] = None

    def set_fit_item(self, item: QGraphicsPixmapItem) -> None:
        self._fit_item = item

    def fit(self) -> None:
        """Einpassen; setzt den Nutzer-Zoom-Status zurück."""
        it = self._fit_item
        if it is None or it.pixmap().isNull():
            return
        self.resetTransform()
        self.fitInView(it, Qt.KeepAspectRatio)
        self.user_zoomed = False

    def apply_zoom(self, factor: float) -> None:
        cur = float(self.transform().m11())
        if cur <= 0.0:
            return
        new = max(self.ZOOM_MIN, min(self.ZOOM_MAX, cur * factor))
        f = new / cur
        if abs(f - 1.0) < 1e-9:
            return
        self.scale(f, f)
        self.user_zoomed = True

    def wheelEvent(self, ev) -> None:
        self.apply_zoom(
            self.ZOOM_STEP if ev.angleDelta().y() > 0 else 1.0 / self.ZOOM_STEP
        )
        ev.accept()

    def mouseDoubleClickEvent(self, ev) -> None:
        self.fit()
        ev.accept()

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        if not self.user_zoomed:
            self.fit()


# --------------------------------------------------------------------------- #
# Hauptwidget
# --------------------------------------------------------------------------- #
class PanoView(QWidget):
    """360°-Player: Play/Pause (Leertaste), Frame ±1 (←/→), Speed, Seek, Zoom."""

    frameChanged = pyqtSignal(int, float)  # (idx, stamp)

    _SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0)
    _TICK_MS = 15

    def __init__(self, parent=None):
        super().__init__(parent)
        self._source: Optional[PanoSource] = None
        self._cur_idx = -1
        self._current_pano: Optional[np.ndarray] = None  # volle Auflösung, BGR
        self._pending_idx: Optional[int] = None
        self._playing = False
        self._speed = 1.0
        self._t0_wall = 0.0
        self._t0_stamp = 0.0
        self._shutdown_done = False
        self.dropped_frames = 0  # übersprungene Frames seit letztem set_source

        # --- Ansicht -------------------------------------------------------
        self._scene = QGraphicsScene(self)
        self._item = QGraphicsPixmapItem()
        self._item.setTransformationMode(Qt.SmoothTransformation)
        self._scene.addItem(self._item)
        self._view = _ZoomGraphicsView(self._scene, self)
        self._view.set_fit_item(self._item)

        # --- Steuerleiste ----------------------------------------------------
        self._play_btn = QToolButton(self)
        self._play_btn.setText("▶")
        self._play_btn.setToolTip("Abspielen/Pause (Leertaste)")
        self._play_btn.clicked.connect(self._toggle_play)

        self._prev_btn = QToolButton(self)
        self._prev_btn.setText("⏮")
        self._prev_btn.setToolTip("Ein Bild zurück (←)")
        self._prev_btn.clicked.connect(lambda: self._step(-1))

        self._next_btn = QToolButton(self)
        self._next_btn.setText("⏭")
        self._next_btn.setToolTip("Ein Bild vor (→)")
        self._next_btn.clicked.connect(lambda: self._step(+1))

        self._speed_combo = QComboBox(self)
        self._speed_combo.setToolTip("Abspielgeschwindigkeit")
        for s in self._SPEEDS:
            label = ("%g" % s).replace(".", ",") + "×"
            self._speed_combo.addItem(label, float(s))
        self._speed_combo.setCurrentIndex(self._SPEEDS.index(1.0))
        self._speed_combo.currentIndexChanged.connect(self._on_speed_changed)

        self._slider = QSlider(Qt.Horizontal, self)
        self._slider.setToolTip("Zeitposition (loslassen springt zum Frame)")
        self._slider.setRange(0, 0)
        self._slider.sliderReleased.connect(self._on_slider_released)

        self._label = QLabel("Frame –/– — t=–", self)

        self._fit_btn = QToolButton(self)
        self._fit_btn.setText("Einpassen")
        self._fit_btn.setToolTip("Ansicht einpassen (auch Doppelklick)")
        self._fit_btn.clicked.connect(self._view.fit)

        self._shot_btn = QToolButton(self)
        self._shot_btn.setText("Bild speichern …")
        self._shot_btn.setToolTip("Aktuelles Panorama in voller Auflösung speichern")
        self._shot_btn.clicked.connect(self._on_screenshot)

        self._rot_chk = QCheckBox("180°", self)
        self._rot_chk.setToolTip("Bild um 180° drehen (Kamera kopfüber montiert)")
        self._rot_chk.setChecked(True)
        self._rot_chk.toggled.connect(self._on_rotate_toggled)

        controls = QHBoxLayout()
        controls.setContentsMargins(4, 2, 4, 4)
        controls.setSpacing(6)
        for w in (self._play_btn, self._prev_btn, self._next_btn, self._speed_combo):
            controls.addWidget(w)
        controls.addWidget(self._slider, 1)
        controls.addWidget(self._label)
        controls.addWidget(self._rot_chk)
        controls.addWidget(self._fit_btn)
        controls.addWidget(self._shot_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._view, 1)
        layout.addLayout(controls)

        # --- Tastatur --------------------------------------------------------
        for key, fn in (
            (Qt.Key_Space, self._toggle_play),
            (Qt.Key_Left, lambda: self._step(-1)),
            (Qt.Key_Right, lambda: self._step(+1)),
        ):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(fn)

        # --- Playback / Prefetch ---------------------------------------------
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.PreciseTimer)
        self._timer.setInterval(self._TICK_MS)
        self._timer.timeout.connect(self._on_tick)

        self._prefetch = _Prefetcher(self)
        self._prefetch.panoReady.connect(self._on_pano_ready)
        self._prefetch.start()

        from PyQt5.QtWidgets import QApplication

        app = QApplication.instance()
        if app is not None:
            # Kind-Widgets bekommen kein closeEvent → Thread beim App-Ende stoppen.
            app.aboutToQuit.connect(self._shutdown)

    # ---------------------------------------------------------------- API ---
    def set_source(self, src: PanoSource) -> None:
        self._set_playing(False)
        self._source = src
        self._cur_idx = -1
        self._current_pano = None
        self._pending_idx = None
        self.dropped_frames = 0
        self._prefetch.set_source(src)
        n = int(src.count) if src is not None else 0
        self._slider.blockSignals(True)
        self._slider.setRange(0, max(0, n - 1))
        self._slider.setValue(0)
        self._slider.blockSignals(False)
        self._view.user_zoomed = False
        self._update_label()
        if n > 0:
            self._show_frame(0)

    @property
    def current_index(self) -> int:
        return self._cur_idx

    @property
    def is_playing(self) -> bool:
        return self._playing

    # ---------------------------------------------------------- Playback ---
    def _toggle_play(self) -> None:
        self._set_playing(not self._playing)

    def _set_playing(self, on: bool) -> None:
        src = self._source
        if on:
            if src is None or src.count == 0:
                return
            anchor_idx = max(0, self._cur_idx)
            if self._cur_idx >= src.count - 1:
                anchor_idx = 0  # vom Ende: neu starten
                self._show_frame(0)
            self._t0_wall = time.monotonic()
            self._t0_stamp = float(src.stamps[anchor_idx])
            self._playing = True
            self._timer.start()
            self._play_btn.setText("⏸")
        else:
            self._playing = False
            self._timer.stop()
            self._play_btn.setText("▶")

    def _on_speed_changed(self, i: int) -> None:
        data = self._speed_combo.itemData(i)
        if data is None:
            return
        self._speed = float(data)
        if self._playing and self._source is not None and self._cur_idx >= 0:
            # neu verankern, damit der Tempo-Wechsel ab jetzt gilt
            self._t0_wall = time.monotonic()
            self._t0_stamp = float(self._source.stamps[self._cur_idx])

    def _set_speed(self, speed: float) -> None:
        i = self._speed_combo.findData(float(speed))
        if i >= 0:
            self._speed_combo.setCurrentIndex(i)
            self._on_speed_changed(i)

    def _nearest_idx(self, target_stamp: float) -> int:
        src = self._source
        stamps = src.stamps
        j = int(np.searchsorted(stamps, target_stamp))
        if j <= 0:
            return 0
        if j >= src.count:
            return src.count - 1
        return j if (stamps[j] - target_stamp) <= (target_stamp - stamps[j - 1]) else j - 1

    def _on_tick(self) -> None:
        src = self._source
        if not self._playing or src is None or src.count == 0:
            return
        elapsed = time.monotonic() - self._t0_wall
        target = self._t0_stamp + elapsed * self._speed
        if target >= float(src.stamps[-1]):
            self._set_playing(False)
            if self._cur_idx != src.count - 1:
                self._show_frame(src.count - 1)
            return
        idx = self._nearest_idx(target)
        if idx != self._cur_idx or self._pending_idx is not None:
            self._show_frame(idx)

    def _step(self, delta: int) -> None:
        if self._source is None or self._source.count == 0:
            return
        self._set_playing(False)
        base = self._cur_idx if self._cur_idx >= 0 else 0
        self._seek_to(base + delta)

    def _seek_to(self, idx: int) -> None:
        src = self._source
        if src is None or src.count == 0:
            return
        idx = max(0, min(src.count - 1, int(idx)))
        self._show_frame(idx)
        if self._playing:
            self._t0_wall = time.monotonic()
            self._t0_stamp = float(src.stamps[idx])

    def _on_slider_released(self) -> None:
        self._seek_to(self._slider.value())

    # ----------------------------------------------------------- Anzeige ---
    def _show_frame(self, idx: int) -> None:
        """Frame anfordern; zeichnet sofort bei Cache-Treffer, sonst pending."""
        src = self._source
        if src is None or src.count == 0:
            return
        idx = max(0, min(src.count - 1, idx))
        self._prefetch.request(list(range(idx, min(src.count, idx + 5))))
        pano = self._prefetch.get(idx)
        if pano is None:
            self._pending_idx = idx  # letztes Bild bleibt stehen, Repaint via Signal
            return
        self._pending_idx = None
        self._paint(idx, pano)

    def _on_pano_ready(self, idx: int) -> None:
        if self._source is None:
            return
        if self._pending_idx is not None and idx == self._pending_idx:
            pano = self._prefetch.get(idx)
            if pano is not None:
                self._pending_idx = None
                self._paint(idx, pano)
            return
        if self._playing and idx > self._cur_idx:
            # Verspätet fertiger Frame: anzeigen, sofern nicht neuer als das
            # aktuelle Wanduhr-Ziel (sonst wäre es vorzeitiges Prefetch-Material).
            elapsed = time.monotonic() - self._t0_wall
            target = self._t0_stamp + elapsed * self._speed
            if idx <= self._nearest_idx(min(target, float(self._source.stamps[-1]))):
                pano = self._prefetch.get(idx)
                if pano is not None:
                    self._pending_idx = None
                    self._paint(idx, pano)

    def _on_rotate_toggled(self) -> None:
        if self._cur_idx >= 0 and self._current_pano is not None:
            self._paint(self._cur_idx, self._current_pano)

    def _paint(self, idx: int, pano: np.ndarray) -> None:
        if self._playing and self._cur_idx >= 0 and idx > self._cur_idx + 1:
            self.dropped_frames += idx - self._cur_idx - 1
        disp = cv2.flip(pano, -1) if self._rot_chk.isChecked() else pano
        h, w = disp.shape[:2]
        pm = bgr_zu_pixmap(disp)
        size_changed = self._item.pixmap().size() != pm.size()
        self._item.setPixmap(pm)
        if size_changed:
            self._scene.setSceneRect(0, 0, w, h)
            if not self._view.user_zoomed:
                self._view.fit()
        self._cur_idx = idx
        self._current_pano = pano
        if not self._slider.isSliderDown():
            self._slider.blockSignals(True)
            self._slider.setValue(idx)
            self._slider.blockSignals(False)
        self._update_label()
        self.frameChanged.emit(int(idx), float(self._source.stamps[idx]))

    def _update_label(self) -> None:
        src = self._source
        if src is None or src.count == 0 or self._cur_idx < 0:
            self._label.setText("Frame –/– — t=–")
            return
        t = float(src.stamps[self._cur_idx] - src.stamps[0])
        self._label.setText(f"Frame {self._cur_idx + 1}/{src.count} — t={t:.2f}s")

    # -------------------------------------------------------- Screenshot ---
    def _save_pano(self, path: str) -> bool:
        """Speichert das aktuell angezeigte Panorama in voller Quellauflösung."""
        if self._current_pano is None:
            return False
        pano = self._current_pano
        if self._rot_chk.isChecked():
            pano = cv2.flip(pano, -1)
        try:
            return bool(cv2.imwrite(path, pano))
        except cv2.error:
            return False

    def _on_screenshot(self) -> None:
        if self._current_pano is None:
            QMessageBox.warning(self, "Bild speichern", "Kein Panorama geladen.")
            return
        default = f"pano_{max(0, self._cur_idx):06d}.png"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Panorama speichern",
            default,
            "PNG-Bild (*.png);;JPEG-Bild (*.jpg *.jpeg)",
        )
        if not path:
            return
        if not self._save_pano(path):
            QMessageBox.warning(
                self, "Bild speichern", f"Datei konnte nicht geschrieben werden:\n{path}"
            )

    # ----------------------------------------------------------- Aufräumen -
    def _shutdown(self) -> None:
        if self._shutdown_done:
            return
        self._shutdown_done = True
        self._timer.stop()
        self._prefetch.shutdown()

    def closeEvent(self, ev) -> None:
        self._shutdown()
        super().closeEvent(ev)


# --------------------------------------------------------------------------- #
# Selbsttest (synthetische Quelle, offscreen)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PyQt5.QtCore import QEventLoop
    from PyQt5.QtWidgets import QApplication

    EVID = (
        "/tmp/super360_modtests/"
        "pano_view"
    )
    os.makedirs(EVID, exist_ok=True)

    class SyntheticSource(PanoSource):
        """Prozedurale Frames: Gitter, wandernde Marker, eingebrannte Frame-Nr."""

        def __init__(self, count=30, fps=20.0, w=1920, h=960, delay=0.0):
            self.count = count
            self.fps = fps
            self.stamps = np.arange(count, dtype=np.float64) / fps
            self._w, self._h, self._delay = w, h, delay
            self.calls = 0

        def get_pano(self, idx: int) -> np.ndarray:
            self.calls += 1
            if self._delay:
                time.sleep(self._delay)
            w, h = self._w, self._h
            img = np.zeros((h, w, 3), np.uint8)
            img[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
            img[:, :, 1] = np.linspace(0, 160, h, dtype=np.uint8)[:, None]
            for x in range(0, w, 240):
                cv2.line(img, (x, 0), (x, h), (90, 90, 90), 1)
            for y in range(0, h, 240):
                cv2.line(img, (0, y), (w, y), (90, 90, 90), 1)
            cx = int((idx + 0.5) / self.count * w)
            cv2.circle(img, (cx, h // 2), 50, (0, 215, 255), -1)
            cy = int((idx + 0.5) / self.count * h)
            cv2.circle(img, (w // 4, cy), 35, (255, 80, 80), -1)
            cv2.putText(
                img, f"{idx:02d}", (w // 2 - 170, h // 2 - 120),
                cv2.FONT_HERSHEY_SIMPLEX, 6.0, (255, 255, 255), 14, cv2.LINE_AA,
            )
            cv2.putText(
                img, f"t={self.stamps[idx]:.3f}s", (40, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 255, 200), 3, cv2.LINE_AA,
            )
            return img

    def spin(ms: int) -> None:
        loop = QEventLoop()
        QTimer.singleShot(ms, loop.quit)
        loop.exec_()

    def wait_until(pred, timeout_ms=3000) -> bool:
        deadline = time.monotonic() + timeout_ms / 1000.0
        while time.monotonic() < deadline:
            if pred():
                return True
            spin(10)
        return bool(pred())

    fails: list[str] = []

    def check(cond: bool, msg: str) -> None:
        status = "PASS" if cond else "FAIL"
        print(f"[{status}] {msg}")
        if not cond:
            fails.append(msg)

    app = QApplication(sys.argv)
    pv = PanoView()
    pv.resize(1280, 760)
    pv.show()

    # ---- schnelle Quelle -------------------------------------------------
    src = SyntheticSource()
    pv.set_source(src)
    check(wait_until(lambda: pv.current_index == 0), "erster Frame angezeigt")
    pv.grab().save(os.path.join(EVID, "01_fit_frame0.png"))

    events: list[tuple[float, int, float]] = []
    pv.frameChanged.connect(lambda i, t: events.append((time.monotonic(), i, t)))

    # 1 s bei 2× abspielen (Quelle: 30 Frames @20 fps → Ende nach 0,725 s bei 2×)
    pv._set_speed(2.0)
    pv._toggle_play()
    t0 = pv._t0_wall
    spin(1000)

    errs = [i - (tw - t0) * 2.0 * src.fps for (tw, i, _t) in events]
    max_err = max(abs(e) for e in errs) if errs else float("inf")
    intervals = np.diff([tw for (tw, _i, _t) in events]) if len(events) > 1 else []
    print(
        f"2x-Playback: {len(events)} Frames angezeigt, max. Stamps-Fehler "
        f"{max_err:.2f} Frames, mittl. Anzeigeintervall "
        f"{np.mean(intervals) * 1000:.1f} ms, Drops={pv.dropped_frames}"
    )
    check(max_err <= 2.0, f"Stamps-korrektes Playback (max. Fehler {max_err:.2f} <= 2 Frames)")
    check(pv.current_index == src.count - 1, "Ende erreicht (Frame 29)")
    check(not pv.is_playing, "Auto-Pause am Ende")
    check(len(events) >= 12, f"genug Frames angezeigt ({len(events)})")

    # ---- Seek --------------------------------------------------------------
    pv._slider.setValue(25)
    pv._on_slider_released()
    check(wait_until(lambda: pv.current_index == 25), "Seek auf Frame 25")
    check(pv._slider.value() == 25, "Slider synchron (25)")
    check(
        pv._label.text() == f"Frame 26/30 — t={src.stamps[25]:.2f}s",
        f"Label korrekt ('{pv._label.text()}')",
    )

    # ---- Zoom + Pan ----------------------------------------------------------
    pv._view.fit()
    fit_scale = float(pv._view.transform().m11())
    for _ in range(4):
        pv._view.apply_zoom(1.25)
    z4 = float(pv._view.transform().m11())
    check(
        abs(z4 - fit_scale * 1.25**4) < 1e-6,
        f"4 Zoom-Schritte: {fit_scale:.3f} → {z4:.3f} (erwartet {fit_scale * 1.25**4:.3f})",
    )
    hs, vs = pv._view.horizontalScrollBar(), pv._view.verticalScrollBar()
    hs.setValue(hs.value() + 400)
    vs.setValue(vs.value() + 200)
    spin(50)
    pv.grab().save(os.path.join(EVID, "02_zoom4_pan_frame25.png"))
    for _ in range(30):
        pv._view.apply_zoom(1.25)
    zmax = float(pv._view.transform().m11())
    check(zmax <= 16.0 + 1e-9, f"Zoom-Obergrenze 16 eingehalten ({zmax:.3f})")
    for _ in range(60):
        pv._view.apply_zoom(0.8)
    zmin = float(pv._view.transform().m11())
    check(zmin >= 0.1 - 1e-9, f"Zoom-Untergrenze 0.1 eingehalten ({zmin:.3f})")
    pv._view.fit()
    check(pv._view.user_zoomed is False, "Einpassen setzt Nutzer-Zoom zurück")

    # ---- Screenshots -----------------------------------------------------------
    pano_path = os.path.join(EVID, "03_pano_frame25_fullres.png")
    check(pv._save_pano(pano_path), "interner Panorama-Save")
    im = cv2.imread(pano_path)
    check(
        im is not None and im.shape == (960, 1920, 3),
        f"gespeichertes Pano volle Auflösung {None if im is None else im.shape}",
    )
    pv.grab().save(os.path.join(EVID, "04_widget_fit_frame25.png"))

    # ---- langsame Quelle (get_pano schläft 120 ms) → Frame-Drops ---------------
    slow = SyntheticSource(delay=0.12)
    t_req = time.monotonic()
    pv.set_source(slow)
    check(wait_until(lambda: pv.current_index == 0, 3000), "langsame Quelle: erster Frame")
    first_latency = time.monotonic() - t_req

    events.clear()
    pv._set_speed(1.0)
    heartbeats: list[float] = []
    hb = QTimer()
    hb.setTimerType(Qt.PreciseTimer)
    hb.setInterval(10)
    hb.timeout.connect(lambda: heartbeats.append(time.monotonic()))
    hb.start()
    pv._toggle_play()
    t0s = pv._t0_wall
    spin(1200)
    hb.stop()
    elapsed = time.monotonic() - t0s
    if pv.is_playing:
        pv._toggle_play()

    shown = [i for (_tw, i, _t) in events]
    target_idx = min(slow.count - 1, elapsed * slow.fps)
    hb_gaps = np.diff(heartbeats) if len(heartbeats) > 1 else np.array([0.0])
    print(
        f"Langsame Quelle: Latenz 1. Frame {first_latency * 1000:.0f} ms; "
        f"{elapsed:.2f} s @1x → Soll≈Frame {target_idx:.1f}; angezeigt {len(shown)} "
        f"Frames {shown}; letzter={pv.current_index}; Drops={pv.dropped_frames}; "
        f"max. UI-Loop-Lücke {hb_gaps.max() * 1000:.0f} ms"
    )
    check(len(shown) < target_idx - 2, f"Frames wurden verworfen ({len(shown)} < {target_idx:.0f})")
    check(
        pv.current_index >= target_idx - 6,
        f"kein Nachlauf: letzter Frame {pv.current_index} >= Soll-6 ({target_idx - 6:.1f})",
    )
    check(all(b > a for a, b in zip(shown, shown[1:])), "Anzeige monoton aufsteigend")
    check(
        float(hb_gaps.max()) < 0.08,
        f"UI-Thread nie blockiert (max. Lücke {hb_gaps.max() * 1000:.0f} ms < 80 ms)",
    )
    pv.grab().save(os.path.join(EVID, "05_slow_source_final.png"))

    pv.close()
    print(f"Beweisdateien: {EVID}")
    if fails:
        print(f"SELBSTTEST FEHLGESCHLAGEN ({len(fails)}): {fails}")
        sys.exit(1)
    print("SELBSTTEST OK")
    sys.exit(0)
