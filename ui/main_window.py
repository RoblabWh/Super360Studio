"""MainWindow von Super360 Studio.

Sidebar (Pipeline + Einstellungen) links, zentrale Tabs (3D-Karte / 360°-Video /
GPS / Protokoll), Statusleiste mit Fortschritt und Abbrechen-Knopf. Alle langen
Operationen laufen in einem generischen :class:`Worker` (QThread); Qt-Widgets
werden ausschließlich im GUI-Thread berührt (Signale).

Autotest-Haken: Ist die Umgebungsvariable ``SUPER360_AUTOTEST=<bagpfad>``
gesetzt, öffnet das Fenster beim Start automatisch dieses Bag, wartet auf die
Cache-Artefakte, schaltet durch alle Tabs, legt Screenshots unter
``SUPER360_AUTOTEST_OUT`` ab und beendet die Anwendung mit Exit-Code 0.
Ohne die Variable ist der Haken ein No-Op.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import tempfile
import threading
import time
import traceback
from typing import Callable, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QMainWindow,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea,
    QSlider, QSpinBox, QTableWidget, QTableWidgetItem, QTabWidget,
    QVBoxLayout, QWidget, QApplication,
)

try:
    from core.bag_reader import BagInfo, BagReader
    from core.project import Project
    from core.recording import Recording
    from core.rviz_player import RvizPlayer, RvizPlayerError
    from core import georef
except ImportError:  # direkter Skript-Start: Paketwurzel nachrüsten
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.bag_reader import BagInfo, BagReader
    from core.project import Project
    from core.recording import Recording
    from core.rviz_player import RvizPlayer, RvizPlayerError
    from core import georef

from ui.cloud_view import CloudView
from ui.gps_panel import GpsPanel
from ui.pano_view import PanoView, StitchingPanoSource

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 2026-07-25: neue Basalt-Kalibrierung (RoblabWh-Fork, pro Kamera headless)
# + photometrisch verfeinerte Extrinsik (rx 4.6°, ry 182.4°) + Vignette-Profil.
# Die Kopien in calib/ machen das Repo eigenstaendig; danach die Originale
# im Stitcher-Projekt als Fallback.
_CALIB_CANDIDATES = (
    os.path.join(_REPO_ROOT, "calib", "calib_result_new2.json"),
    os.path.join(_REPO_ROOT, "calib", "calib_new_vign.json"),
    os.path.join(_REPO_ROOT, "calib", "calib_new_refined.json"),
    "/home/lena/RosBagSuper_Gui/Super360_Stitcher_rosbag/work/calib_result_new2/calibration.json",
    "/home/lena/RosBagSuper_Gui/Super360_Stitcher_rosbag/work/calib_new_vign/calibration.json",
    "/home/lena/RosBagSuper_Gui/Super360_Stitcher_rosbag/work/calib_new_refined/calibration.json",
)

_DEFAULT_SETTINGS: dict = {
    "pano_width": 1920,
    "config": "whs_dense.yaml",
    "rate": 1.0,
    "brightness_min": 20,
    "brightness_max": 235,
    "k_frames": 3,
    "sky_grow": 4,
    "point_size": 2,
    "color_mode": "rgb",
    "only_colored": True,
    "voxel": 0.0,
    "background": "dunkel",
    "edl": False,
    "show_path": False,
}

_COLOR_MODE_ITEMS = (("RGB (eingefärbt)", "rgb"), ("Höhe", "hoehe"),
                     ("Intensität", "intensitaet"), ("Einfarbig", "uniform"))
_VOXEL_ITEMS = (("Aus", 0.0), ("0,05 m", 0.05), ("0,10 m", 0.10), ("0,20 m", 0.20))
_BG_ITEMS = (("Dunkel", "dunkel"), ("Hell", "hell"))
_CONFIG_ITEMS = (("Maximal dicht (whs_dense.yaml)", "whs_dense.yaml"),
                 ("Schnell (mid360.yaml)", "mid360.yaml"))
_AUTOCAL_WEAK_SCORE = 0.30


def _default_calib() -> str:
    for p in _CALIB_CANDIDATES:
        if os.path.isfile(p):
            return p
    raise RuntimeError("Keine Kamera-Kalibrierung gefunden "
                       f"(gesucht: {', '.join(_CALIB_CANDIDATES)})")


def _fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", ".")


class ThreadLocalBag:
    """BagReader-Fassade mit einem echten Reader je Thread.

    rosbags öffnet sqlite3-Verbindungen ohne ``check_same_thread=False`` —
    ein Reader darf daher nur in seinem Erzeuger-Thread lesen. Worker- und
    Prefetch-Threads bekommen hier je einen eigenen BagReader (Aufbau des
    Kamera-Index dauert < 0,2 s).
    """

    def __init__(self, bag_path: str):
        self.bag_path = str(bag_path)
        self._local = threading.local()

    def _bag(self) -> BagReader:
        bag = getattr(self._local, "bag", None)
        if bag is None:
            bag = BagReader(self.bag_path)
            self._local.bag = bag
        return bag

    def info(self) -> BagInfo:
        return self._bag().info()

    def camera_stamps(self) -> np.ndarray:
        return self._bag().camera_stamps()

    def read_camera(self, idx: int) -> np.ndarray:
        return self._bag().read_camera(idx)

    def read_camera_jpeg(self, idx: int) -> bytes:
        return self._bag().read_camera_jpeg(idx)

    def iter_camera(self, start: int = 0, stop: int | None = None):
        return self._bag().iter_camera(start, stop)

    def read_gps(self):
        return self._bag().read_gps()

    def close(self) -> None:
        bag = getattr(self._local, "bag", None)
        if bag is not None:
            bag.close()
            self._local.bag = None


class Worker(QThread):
    """Generischer Hintergrund-Arbeiter: ``fn(progress_cb, cancel, log_cb)``.

    ``cancellable=False`` für Jobs, die aus einem einzelnen Bibliotheksaufruf
    bestehen (Export, Overlay) und das Cancel-Event ohnehin nicht auswerten
    können — der Abbrechen-Knopf bleibt dann ehrlich deaktiviert.
    """

    progress = pyqtSignal(float, str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, fn: Callable, parent=None, cancellable: bool = True):
        super().__init__(parent)
        self._fn = fn
        self.cancel = threading.Event()
        self.cancellable = bool(cancellable)

    def run(self) -> None:
        # Emits bewusst außerhalb des except-Blocks (kein aktiver Exception-
        # Zustand während der Signal-Zustellung).
        result = None
        error: str | None = None
        error_line = ""
        try:
            result = self._fn(progress_cb=self._emit_progress,
                              cancel=self.cancel, log_cb=self.log.emit)
        except Exception as exc:  # noqa: BLE001 — UI zeigt die Meldung
            error = str(exc) or exc.__class__.__name__
            error_line = "".join(traceback.format_exception_only(exc)).strip()
        if error is not None:
            self.log.emit(error_line)
            self.failed.emit(error)
        else:
            self.finished.emit(result)

    def _emit_progress(self, frac: float, msg: str) -> None:
        self.progress.emit(float(frac), str(msg))


class _ImageDialog(QDialog):
    """Einfacher Bild-Dialog (BGR-Eingabe, skaliert auf max. 1400×800)."""

    def __init__(self, title: str, bgr: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        h, w = rgb.shape[:2]
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pm = QPixmap.fromImage(img)
        if w > 1400 or h > 800:
            pm = pm.scaled(1400, 800, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        lbl = QLabel(self)
        lbl.setPixmap(pm)
        lay = QVBoxLayout(self)
        lay.addWidget(lbl)
        btn = QPushButton("Schließen", self)
        btn.clicked.connect(self.accept)
        lay.addWidget(btn, alignment=Qt.AlignRight)


def _compact_combo(combo: QComboBox) -> QComboBox:
    """Verhindert, dass lange Eintragstexte die Sidebar-Mindestbreite sprengen."""
    combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(10)
    _cap_width(combo, "Maximal dicht (whs_de", 44)
    return combo


def _cap_width(widget: QWidget, sample_text: str, extra_px: int) -> None:
    """Deckelt die Breite fontabhängig — hält die Sidebar bei jedem DPI schmal.

    Ein via setMaximumWidth gesetztes Maximum begrenzt auch das effektive
    Layout-Minimum (smartMinSize), das sonst vom breiten minimumSizeHint kommt.
    """
    fm = widget.fontMetrics()
    widget.setMaximumWidth(fm.horizontalAdvance(sample_text) + extra_px)


def _wrappable(form: QFormLayout) -> QFormLayout:
    form.setRowWrapPolicy(QFormLayout.WrapLongRows)
    form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    return form


def _load_color_files(colors_dir: str, n_points: int,
                      expected_fingerprint: str | None = None,
                      log_cb: Callable[[str], None] | None = None
                      ) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    """colors.bin (RGB uint8 N×3) + valid.bin (uint8 N) laden und prüfen.

    ``expected_fingerprint``: Fingerprint der aktuell geladenen Aufzeichnung
    (``core.colorizer.rec_fingerprint``). Steht in colors/meta.json ein anderer
    ``rec_fingerprint``, stammt der Farb-Cache von einer früheren Aufzeichnung
    und wird ignoriert. Fehlt der Schlüssel (älterer Cache), wird er aus
    Kompatibilität akzeptiert. Die Farben werden unverändert zurückgegeben
    (ungefärbte Punkte bleiben 0,0,0 — Grau-Ersatz ist Sache der Anzeige).
    """
    try:
        colors = np.fromfile(os.path.join(colors_dir, "colors.bin"), dtype=np.uint8)
        valid = np.fromfile(os.path.join(colors_dir, "valid.bin"), dtype=np.uint8)
    except OSError as exc:
        return None, None, f"Farben nicht lesbar: {exc}"
    if colors.size != 3 * n_points or valid.size != n_points:
        return None, None, (f"Farb-Cache passt nicht zur Punktwolke "
                            f"({colors.size // 3} Farben, {n_points} Punkte) — ignoriert.")
    if expected_fingerprint is not None:
        stored = None
        try:
            with open(os.path.join(colors_dir, "meta.json"), encoding="utf-8") as fh:
                stored = json.load(fh).get("rec_fingerprint")
        except (OSError, ValueError):
            stored = None
        if stored is not None and stored != expected_fingerprint:
            return None, None, ("Farb-Cache stammt von einer anderen Aufzeichnung "
                                "— ignoriert.")
        if stored is None and log_cb is not None:
            log_cb("Farb-Cache ohne Aufzeichnungs-Fingerprint (älterer Stand) — "
                   "wird übernommen.")
    return colors.reshape(-1, 3), valid.astype(bool), None


class MainWindow(QMainWindow):
    """Hauptfenster: Sidebar-Pipeline + Tabs, ein Worker je Schritt."""

    # RvizPlayer protokolliert aus seinen Reader-Threads. Ein Signal ist der
    # einzige threadsichere Weg in den GUI-Thread — QTimer.singleShot() aus
    # einem Fremd-Thread feuert NIE (dieser Thread hat keine Event-Loop).
    rvizLog = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Super360 Studio")
        self.resize(1600, 950)

        # ------------------------------------------------------------ Zustand
        self._bag: Optional[ThreadLocalBag] = None
        self._bag_info: Optional[BagInfo] = None
        self._n_frames = 0
        self._project: Optional[Project] = None
        self._settings: dict = dict(_DEFAULT_SETTINGS)
        self._rec: Optional[Recording] = None
        self._world: Optional[np.ndarray] = None
        self._colors: Optional[np.ndarray] = None
        self._valid: Optional[np.ndarray] = None
        self._fixes: Optional[list] = None
        self._quality = None
        self._georef = None
        self._pano_src: Optional[StitchingPanoSource] = None
        self._worker: Optional[Worker] = None
        self._retired: list[Worker] = []
        self._busy = False
        self._loading_ui = False
        self._closing = False
        self._pano_failed = False
        self._overlay_dialogs: list[_ImageDialog] = []
        self.rvizLog.connect(self._log)
        self._rviz_player = RvizPlayer(log_cb=self.rvizLog.emit)
        try:
            self._calib = _default_calib()
        except RuntimeError:
            self._calib = ""

        self._build_ui()
        self._update_enabled()
        if not self._calib:
            self._log("WARNUNG: Keine Kamera-Kalibrierung gefunden — "
                      "Pano/Einfärbung nicht verfügbar.")

        # -------------------------------------------------------- Autotest
        self._autotest_path = os.environ.get("SUPER360_AUTOTEST") or None
        self._autotest_out = (os.environ.get("SUPER360_AUTOTEST_OUT")
                              or os.path.join(tempfile.gettempdir(), "super360studio_autotest"))
        self._autotest_failed = False
        self._autotest_deadline = 0.0
        self._autotest_timer: Optional[QTimer] = None
        if self._autotest_path:
            QTimer.singleShot(800, self._autotest_begin)

    # ================================================================== UI-Bau

    def _build_ui(self) -> None:
        central = QWidget(self)
        root = QHBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        sidebar = self._build_sidebar()
        scroll = QScrollArea(self)
        scroll.setWidget(sidebar)
        scroll.setWidgetResizable(True)
        # Soll ~360-400 px; bei großen Systemfonts (Hi-DPI) so weit aufweiten,
        # dass nichts abgeschnitten wird (Fontbreiten skalieren die Minima).
        need = (sidebar.minimumSizeHint().width()
                + scroll.verticalScrollBar().sizeHint().width() + 10)
        scroll.setFixedWidth(max(400, min(600, need)))
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        root.addWidget(scroll)

        self._tabs = QTabWidget(self)
        self._cloud_view = CloudView(self)
        # EDL-Verfügbarkeit lässt sich erst mit existierender CloudView bestimmen
        # (die Sidebar samt Checkbox wird oben vor der CloudView gebaut).
        if not self._cloud_view.edl_available:
            self._chk_edl.setEnabled(False)
            self._chk_edl.setToolTip(
                "EDL wird von dieser VTK-Installation nicht unterstützt.")
        self._pano_view = PanoView(self)
        self._gps_panel = GpsPanel(self)
        self._log_edit = QPlainTextEdit(self)
        self._log_edit.setReadOnly(True)
        self._log_edit.setMaximumBlockCount(20000)
        self._tabs.addTab(self._cloud_view, "3D-Karte")
        self._tabs.addTab(self._pano_view, "360°-Video")
        self._tabs.addTab(self._gps_panel, "GPS")
        self._tabs.addTab(self._log_edit, "Protokoll")
        root.addWidget(self._tabs, 1)
        self.setCentralWidget(central)

        # ------------------------------------------------------ Statusleiste
        sb = self.statusBar()
        self._status_lbl = QLabel("Bereit", self)
        sb.addWidget(self._status_lbl, 1)
        self._frame_lbl = QLabel("", self)
        sb.addPermanentWidget(self._frame_lbl)
        self._status_pbar = QProgressBar(self)
        self._status_pbar.setMaximumWidth(220)
        self._status_pbar.setVisible(False)
        sb.addPermanentWidget(self._status_pbar)
        self._btn_cancel = QPushButton("Abbrechen", self)
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._on_cancel)
        sb.addPermanentWidget(self._btn_cancel)

        self._pano_view.frameChanged.connect(self._on_pano_frame)
        self._gps_panel.georefReady.connect(self._on_georef_ready)

    def _build_sidebar(self) -> QWidget:
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(8)
        lay.addWidget(self._group_rosbag())
        lay.addWidget(self._group_fastlio())
        lay.addWidget(self._group_colorize())
        lay.addWidget(self._group_display())
        lay.addWidget(self._group_rviz())
        lay.addWidget(self._group_export())
        lay.addStretch(1)
        return panel

    def _group_rosbag(self) -> QGroupBox:
        box = QGroupBox("1. Rosbag")
        lay = QVBoxLayout(box)
        self._btn_open = QPushButton("Bag öffnen…")
        self._btn_open.clicked.connect(self._on_open_clicked)
        lay.addWidget(self._btn_open)
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

    def _group_fastlio(self) -> QGroupBox:
        box = QGroupBox("2. Punktwolke (FAST-LIO2)")
        form = _wrappable(QFormLayout(box))
        self._combo_config = _compact_combo(QComboBox())
        for label, data in _CONFIG_ITEMS:
            self._combo_config.addItem(label, data)
        self._combo_config.currentIndexChanged.connect(self._on_setting_changed)
        form.addRow("Konfiguration:", self._combo_config)
        self._spin_rate = QDoubleSpinBox()
        self._spin_rate.setRange(0.25, 2.0)
        self._spin_rate.setSingleStep(0.25)
        self._spin_rate.setValue(1.0)
        self._spin_rate.valueChanged.connect(self._on_setting_changed)
        form.addRow("Abspielrate:", self._spin_rate)
        self._btn_fastlio = QPushButton("Karte berechnen")
        self._btn_fastlio.clicked.connect(self._on_fastlio_clicked)
        form.addRow(self._btn_fastlio)
        self._pbar_fastlio = QProgressBar()
        self._pbar_fastlio.setRange(0, 1000)
        self._pbar_fastlio.setValue(0)
        form.addRow(self._pbar_fastlio)
        self._lbl_fastlio = QLabel("Noch keine Karte berechnet.")
        self._lbl_fastlio.setWordWrap(True)
        form.addRow(self._lbl_fastlio)
        return box

    def _slider_row(self, minimum: int, maximum: int, value: int
                    ) -> tuple[QSlider, QLabel, QWidget]:
        holder = QWidget()
        lay = QHBoxLayout(holder)
        lay.setContentsMargins(0, 0, 0, 0)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        lbl = QLabel(str(value))
        lbl.setMinimumWidth(30)
        lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        slider.valueChanged.connect(lambda v, l=lbl: l.setText(str(v)))
        lay.addWidget(slider, 1)
        lay.addWidget(lbl)
        return slider, lbl, holder

    def _group_colorize(self) -> QGroupBox:
        box = QGroupBox("3. Einfärbung")
        form = _wrappable(QFormLayout(box))
        self._sld_bmin, _, row_min = self._slider_row(0, 255, 20)
        self._sld_bmax, _, row_max = self._slider_row(0, 255, 235)
        self._sld_bmin.valueChanged.connect(self._on_setting_changed)
        self._sld_bmax.valueChanged.connect(self._on_setting_changed)
        form.addRow("Helligkeit min:", row_min)
        form.addRow("Helligkeit max:", row_max)
        self._spin_kframes = QSpinBox()
        self._spin_kframes.setRange(1, 10)
        self._spin_kframes.setValue(3)
        self._spin_kframes.valueChanged.connect(self._on_setting_changed)
        form.addRow("K Frames:", self._spin_kframes)
        self._spin_sky = QSpinBox()
        self._spin_sky.setRange(0, 20)
        self._spin_sky.setValue(4)
        self._spin_sky.setSuffix(" px")
        self._spin_sky.setToolTip(
            "Sperrt den Saum um ausgebrannte Himmelsflächen. Dort mischen Blur und\n"
            "Farbsaum Himmel und Objekt zu Grauweiß, das unter 'Helligkeit max'\n"
            "durchrutscht und Baumkronen weiß überzieht. 0 schaltet die Sperre ab.")
        self._spin_sky.valueChanged.connect(self._on_setting_changed)
        form.addRow("Himmelssaum:", self._spin_sky)

        grid_holder = QWidget()
        grid = QGridLayout(grid_holder)
        grid.setContentsMargins(0, 0, 0, 0)
        self._ext_spins: dict[str, QDoubleSpinBox] = {}
        specs = (("yaw", "Yaw °", -180.0, 180.0, 0.5, 2),
                 ("pitch", "Pitch °", -180.0, 180.0, 0.5, 2),
                 ("roll", "Roll °", -180.0, 180.0, 0.5, 2),
                 ("x", "x m", -2.0, 2.0, 0.005, 3),
                 ("y", "y m", -2.0, 2.0, 0.005, 3),
                 ("z", "z m", -2.0, 2.0, 0.005, 3))
        for i, (key, label, lo, hi, step, dec) in enumerate(specs):
            spin = QDoubleSpinBox()
            _cap_width(spin, "-180,000", 48)
            spin.setRange(lo, hi)
            spin.setSingleStep(step)
            spin.setDecimals(dec)
            spin.valueChanged.connect(self._on_extrinsic_changed)
            self._ext_spins[key] = spin
            grid.addWidget(QLabel(label), i // 2, (i % 2) * 2)
            grid.addWidget(spin, i // 2, (i % 2) * 2 + 1)
        form.addRow(QLabel("Extrinsik Kamera↔IMU:"))
        form.addRow(grid_holder)

        self._btn_autocal = QPushButton("Auto-Kalibrierung (grob)")
        self._btn_autocal.clicked.connect(self._on_autocal_clicked)
        self._btn_overlay = QPushButton("Overlay-Vorschau")
        self._btn_overlay.clicked.connect(self._on_overlay_clicked)
        self._btn_colorize = QPushButton("Einfärben")
        self._btn_colorize.clicked.connect(self._on_colorize_clicked)
        form.addRow(self._btn_autocal)
        form.addRow(self._btn_overlay)
        form.addRow(self._btn_colorize)
        return box

    def _group_display(self) -> QGroupBox:
        box = QGroupBox("4. Anzeige")
        form = _wrappable(QFormLayout(box))
        self._spin_pointsize = QSpinBox()
        self._spin_pointsize.setRange(1, 8)
        self._spin_pointsize.setValue(2)
        self._spin_pointsize.valueChanged.connect(self._on_display_changed)
        form.addRow("Punktgröße:", self._spin_pointsize)
        self._combo_colormode = _compact_combo(QComboBox())
        for label, data in _COLOR_MODE_ITEMS:
            self._combo_colormode.addItem(label, data)
        self._combo_colormode.currentIndexChanged.connect(self._on_display_changed)
        form.addRow("Farbmodus:", self._combo_colormode)
        self._chk_only_colored = QCheckBox("Nur eingefärbte Punkte")
        self._chk_only_colored.toggled.connect(self._on_display_changed)
        form.addRow(self._chk_only_colored)
        self._combo_voxel = _compact_combo(QComboBox())
        for label, data in _VOXEL_ITEMS:
            self._combo_voxel.addItem(label, data)
        self._combo_voxel.currentIndexChanged.connect(self._on_display_changed)
        form.addRow("Anzeige-Voxel:", self._combo_voxel)
        self._combo_bg = _compact_combo(QComboBox())
        for label, data in _BG_ITEMS:
            self._combo_bg.addItem(label, data)
        self._combo_bg.currentIndexChanged.connect(self._on_display_changed)
        form.addRow("Hintergrund:", self._combo_bg)
        self._chk_edl = QCheckBox("EDL (Eye-Dome Lighting)")
        self._chk_edl.toggled.connect(self._on_display_changed)
        # Verfügbarkeit wird nach dem Bau der CloudView geprüft (s. _build_ui).
        form.addRow(self._chk_edl)
        self._chk_path = QCheckBox("Trajektorie zeigen")
        self._chk_path.toggled.connect(self._on_display_changed)
        form.addRow(self._chk_path)
        return box

    def _group_rviz(self) -> QGroupBox:
        box = QGroupBox("5. RViz-Wiedergabe")
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
        # auch von selbst enden, wenn der Bag durchgelaufen ist).
        self._rviz_timer = QTimer(self)
        self._rviz_timer.timeout.connect(self._refresh_rviz_state)
        self._rviz_timer.start(700)
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

    def _group_export(self) -> QGroupBox:
        box = QGroupBox("5. Export")
        lay = QVBoxLayout(box)
        self._btn_export_plypcd = QPushButton("PLY/PCD speichern…")
        self._btn_export_plypcd.clicked.connect(self._on_export_plypcd)
        self._btn_export_las = QPushButton("LAS speichern…")
        self._btn_export_las.clicked.connect(self._on_export_las)
        lay.addWidget(self._btn_export_plypcd)
        lay.addWidget(self._btn_export_las)
        return box

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
        for b in (self._btn_export_plypcd, self._btn_export_las):
            b.setEnabled(not busy and has_world)
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

    def _worker_failed(self, worker: Worker, title: str, msg: str,
                       on_failed: Callable[[str], None] | None = None) -> None:
        self._retire(worker)
        if self._closing:
            return  # Fenster schließt bereits — keine Dialoge/Folgeschritte mehr
        self._set_busy(False, "Bereit")
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

    # ========================================================== Einstellungen

    def _apply_settings_to_widgets(self) -> None:
        s = self._settings
        self._loading_ui = True
        try:
            idx = self._combo_config.findData(s.get("config", "whs_dense.yaml"))
            self._combo_config.setCurrentIndex(max(0, idx))
            self._spin_rate.setValue(float(s.get("rate", 1.0)))
            self._sld_bmin.setValue(int(s.get("brightness_min", 20)))
            self._sld_bmax.setValue(int(s.get("brightness_max", 235)))
            self._spin_kframes.setValue(int(s.get("k_frames", 3)))
            self._spin_sky.setValue(int(s.get("sky_grow", 4)))
            self._spin_pointsize.setValue(int(s.get("point_size", 2)))
            idx = self._combo_colormode.findData(s.get("color_mode", "rgb"))
            self._combo_colormode.setCurrentIndex(max(0, idx))
            self._chk_only_colored.setChecked(bool(s.get("only_colored", False)))
            voxel = float(s.get("voxel", 0.0))
            idx = next((i for i in range(self._combo_voxel.count())
                        if abs(self._combo_voxel.itemData(i) - voxel) < 1e-9), 0)
            self._combo_voxel.setCurrentIndex(idx)
            idx = self._combo_bg.findData(s.get("background", "dunkel"))
            self._combo_bg.setCurrentIndex(max(0, idx))
            self._chk_edl.setChecked(bool(s.get("edl", False)) and self._chk_edl.isEnabled())
            self._chk_path.setChecked(bool(s.get("show_path", False)))
        finally:
            self._loading_ui = False
        self._push_display_settings()

    def _collect_settings(self) -> dict:
        return {
            "pano_width": int(self._settings.get("pano_width", 1920)),
            "config": self._combo_config.currentData(),
            "rate": float(self._spin_rate.value()),
            "brightness_min": int(self._sld_bmin.value()),
            "brightness_max": int(self._sld_bmax.value()),
            "k_frames": int(self._spin_kframes.value()),
            "sky_grow": int(self._spin_sky.value()),
            "point_size": int(self._spin_pointsize.value()),
            "color_mode": self._combo_colormode.currentData(),
            "only_colored": bool(self._chk_only_colored.isChecked()),
            "voxel": float(self._combo_voxel.currentData()),
            "background": self._combo_bg.currentData(),
            "edl": bool(self._chk_edl.isChecked()),
            "show_path": bool(self._chk_path.isChecked()),
        }

    def _save_settings(self) -> None:
        if self._loading_ui or self._project is None:
            return
        self._settings = self._collect_settings()
        try:
            self._project.save_settings(self._settings)
        except RuntimeError as exc:
            self._log(f"Einstellungen nicht gespeichert: {exc}")

    def _on_setting_changed(self, *_a) -> None:
        self._save_settings()

    def _push_display_settings(self) -> None:
        cv = self._cloud_view
        cv.set_point_size(int(self._spin_pointsize.value()))
        cv.set_color_mode(self._combo_colormode.currentData())
        cv.set_only_colored(self._chk_only_colored.isChecked())
        cv.set_voxel_display(float(self._combo_voxel.currentData()))
        cv.set_background(self._combo_bg.currentData())
        cv.set_eyedome(self._chk_edl.isChecked())
        if self._chk_path.isChecked() and self._rec is not None:
            cv.set_path(self._rec.path_positions())
        else:
            cv.set_path(None)

    def _on_display_changed(self, *_a) -> None:
        if self._loading_ui:
            return
        self._push_display_settings()
        self._save_settings()

    # ============================================================== Extrinsik

    def _extrinsic_from_spins(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler(
            "ZYX",
            [self._ext_spins["yaw"].value(), self._ext_spins["pitch"].value(),
             self._ext_spins["roll"].value()], degrees=True).as_matrix()
        T[:3, 3] = [self._ext_spins["x"].value(), self._ext_spins["y"].value(),
                    self._ext_spins["z"].value()]
        return T

    def _spins_from_extrinsic(self, T: np.ndarray) -> None:
        ypr = Rotation.from_matrix(np.asarray(T)[:3, :3]).as_euler("ZYX", degrees=True)
        vals = {"yaw": ypr[0], "pitch": ypr[1], "roll": ypr[2],
                "x": T[0, 3], "y": T[1, 3], "z": T[2, 3]}
        self._loading_ui = True
        try:
            for key, val in vals.items():
                self._ext_spins[key].setValue(float(val))
        finally:
            self._loading_ui = False

    def _on_extrinsic_changed(self, *_a) -> None:
        if self._loading_ui or self._project is None:
            return
        try:
            self._project.save_extrinsic(self._extrinsic_from_spins())
        except RuntimeError as exc:
            self._log(f"Extrinsik nicht gespeichert: {exc}")

    # ============================================================ Bag öffnen

    def _on_open_clicked(self) -> None:
        if self._busy:
            QMessageBox.information(
                self, "Beschäftigt",
                "Es läuft noch ein Arbeitsschritt — bitte warten oder abbrechen.")
            return
        path = QFileDialog.getExistingDirectory(
            self, "Rosbag-Ordner öffnen", os.path.dirname(os.path.abspath(__file__)))
        if path:
            self._open_bag(path)

    def _open_bag(self, path: str) -> None:
        if self._busy:
            QMessageBox.information(
                self, "Beschäftigt",
                "Es läuft noch ein Arbeitsschritt — bitte warten oder abbrechen.")
            return
        self._clear_bag_state()
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
            project = Project(path)
            try:
                with open(project.gps_json(), "w", encoding="utf-8") as fh:
                    json.dump([dataclasses.asdict(f) for f in fixes], fh)
            except OSError as exc:
                log_cb(f"gps.json nicht geschrieben: {exc}")
            progress_cb(1.0, "Bag geöffnet")
            return {"path": path, "proxy": proxy, "info": info,
                    "n_frames": n_frames, "fixes": fixes,
                    "quality": quality, "project": project}

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
        self._georef = None
        self._pano_src = None
        self._pano_failed = False
        self._pano_view.set_source(None)
        self._cloud_view.set_cloud(None)
        self._cloud_view.set_path(None)
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
        self.setWindowTitle(f"Super360 Studio — {self._project.bag_name}")

        self._settings = dict(_DEFAULT_SETTINGS)
        try:
            self._settings.update(self._project.load_settings())
        except RuntimeError as exc:
            self._log(str(exc))
        self._apply_settings_to_widgets()

        T = None
        try:
            T = self._project.load_extrinsic()
        except RuntimeError as exc:
            self._log(str(exc))
        self._spins_from_extrinsic(T if T is not None else np.eye(4))

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
        if self._calib and info.camera_topic:
            self._start_pano_job()
        elif self._project.has_recording():
            self._start_recording_load()

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
        width = int(self._settings.get("pano_width", 1920))
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
                  f"Breite {self._settings.get('pano_width', 1920)} px.")
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
            rec = Recording.load(rec_dir, bag_path=project.bag_path)
            note = rec.level_note()
            if note:
                log_cb(note)
            world = rec.world_points(
                progress_cb=lambda f, m: progress_cb(0.05 + 0.85 * f, m), cancel=cancel)
            colors = valid = None
            if project.has_colors():
                progress_cb(0.95, "Lade Farben …")
                try:
                    from core.colorizer import rec_fingerprint
                    fingerprint = rec_fingerprint(rec)
                except Exception as exc:  # noqa: BLE001 — Kompat: ohne Prüfung laden
                    fingerprint = None
                    log_cb(f"Aufzeichnungs-Fingerprint nicht verfügbar: {exc}")
                colors, valid, err = _load_color_files(
                    project.colors_dir(), rec.n_points,
                    expected_fingerprint=fingerprint, log_cb=log_cb)
                if err:
                    log_cb(err)
            return {"rec": rec, "world": world, "colors": colors, "valid": valid}

        self._start_worker("Lade Punktwolke …", job, self._on_recording_loaded)

    def _on_recording_loaded(self, res: dict) -> None:
        self._rec = res["rec"]
        self._world = res["world"]
        self._colors = res["colors"]
        self._valid = res["valid"]
        if self._colors is None and self._combo_colormode.currentData() == "rgb":
            # ohne Farben wäre "rgb" einfarbig — Höhe ist die aussagekräftige Ansicht
            idx = self._combo_colormode.findData("hoehe")
            self._loading_ui = True
            self._combo_colormode.setCurrentIndex(idx)
            self._loading_ui = False
        self._cloud_view.set_cloud(self._world, self._colors,
                                   self._rec.intensity, self._valid)
        self._push_display_settings()
        if self._quality is not None:
            self._gps_panel.set_quality(self._quality, self._fixes, self._rec)
        meta = self._rec.meta
        expected = int(meta.get("expected_scans", self._rec.n_scans))
        drops = max(0, expected - self._rec.n_scans)
        self._lbl_fastlio.setText(
            f"Scans: {self._rec.n_scans}/{expected} · "
            f"Punkte: {_fmt_int(self._rec.n_points)} · Drops: {drops}")
        n_col = int(self._valid.sum()) if self._valid is not None else 0
        col_txt = (f", {_fmt_int(n_col)} eingefärbt" if self._colors is not None else "")
        self._log(f"Punktwolke geladen: {self._rec.n_scans} Scans, "
                  f"{_fmt_int(self._rec.n_points)} Punkte{col_txt}.")
        self._update_enabled()

    # ================================================================ FAST-LIO

    def _on_fastlio_clicked(self) -> None:
        if self._bag is None or self._project is None:
            return
        bag_path = self._bag.bag_path
        out_dir = self._project.recording_dir()
        config = self._combo_config.currentData()
        rate = float(self._spin_rate.value())
        # Alte Aufzeichnung VOR dem Start vollständig loslassen: die neue
        # Aufzeichnung ersetzt recording/ — offene np.memmaps auf den alten
        # Dateien würden sonst als veraltete Anzeige weiterleben (bzw. bei
        # Truncation einen SIGBUS riskieren). Auch die Georeferenzierung passt
        # nicht mehr zur neuen Trajektorie.
        self._rec = None
        self._world = None
        self._colors = None
        self._valid = None
        self._georef = None
        self._cloud_view.set_cloud(None)
        self._cloud_view.set_path(None)
        if self._quality is not None:
            self._gps_panel.set_quality(self._quality, self._fixes, None)
        self._lbl_fastlio.setText("Karte wird berechnet …")
        self._pbar_fastlio.setRange(0, 1000)
        self._pbar_fastlio.setValue(0)

        def job(progress_cb, cancel, log_cb):
            from core.fastlio_runner import FastLioRunner
            runner = FastLioRunner()
            return runner.run(bag_path, out_dir, config=config, rate=rate,
                              progress_cb=progress_cb, cancel=cancel, log_cb=log_cb)

        def extra(frac: float, _msg: str) -> None:
            self._pbar_fastlio.setValue(int(max(0.0, min(1.0, frac)) * 1000))

        self._start_worker(f"FAST-LIO2 läuft ({config}, Rate {rate:g}×) …",
                           job, self._on_fastlio_done, extra_progress=extra)

    def _on_fastlio_done(self, result) -> None:
        self._pbar_fastlio.setValue(1000)
        self._lbl_fastlio.setText(
            f"Scans: {result.n_scans}/{result.expected_scans} · "
            f"Punkte: {_fmt_int(result.n_points)} · Drops: {result.dropped_scans}")
        self._log(f"FAST-LIO2 fertig in {result.duration_s:.1f} s: "
                  f"{result.n_scans} Scans, {_fmt_int(result.n_points)} Punkte, "
                  f"{result.dropped_scans} Drops.")
        # alte Farben passen nicht mehr zur neuen Aufzeichnung
        self._colors = None
        self._valid = None
        self._start_recording_load()

    # ============================================================== Einfärbung

    @staticmethod
    def _import_colorizer():
        try:
            from core import colorizer
            return colorizer
        except ImportError as exc:
            raise RuntimeError(f"Modul 'colorizer' ist nicht verfügbar: {exc}") from exc

    def _on_colorize_clicked(self) -> None:
        if self._rec is None or self._bag is None:
            return
        try:
            colorizer = self._import_colorizer()
        except RuntimeError as exc:
            self._show_error("Einfärben", str(exc))
            return
        T = self._extrinsic_from_spins()
        try:
            self._project.save_extrinsic(T)
        except RuntimeError as exc:
            self._log(f"Extrinsik nicht gespeichert: {exc}")
        params = colorizer.ColorizeParams(
            brightness_min=int(self._sld_bmin.value()),
            brightness_max=int(self._sld_bmax.value()),
            k_frames=int(self._spin_kframes.value()),
            sky_grow=int(self._spin_sky.value()),
            T_imu_cam0=T)
        rec, bag, calib = self._rec, self._bag, self._calib
        out_dir = self._project.colors_dir()

        def job(progress_cb, cancel, log_cb):
            # Kurzer Test vor dem langen Lauf: sitzt die Extrinsik auf einem
            # Gipfel oder auf einer Flanke? Der absolute Score ist zwischen
            # Fluegen nicht vergleichbar, eine verdrehte Extrinsik faellt daher
            # sonst nicht auf — sie kostet aber die halbe Farbqualitaet.
            progress_cb(0.01, "Prüfe Extrinsik …")
            try:
                chk = colorizer.check_extrinsic(rec, bag, calib, T, cancel=cancel)
            except RuntimeError as exc:
                log_cb(f"Extrinsik-Prüfung übersprungen: {exc}")
            else:
                log_cb(f"Extrinsik-Güte (Foto-Konsistenz): {chk['score']:.3f}; "
                       f"bestes erreichbares {chk['best_score']:.3f} "
                       f"{chk['dist_deg']:.1f}° daneben.")
                if chk["suspect"]:
                    log_cb(
                        "WARNUNG: die gespeicherte Extrinsik ist deutlich verdreht. "
                        "Das kostet spürbar Farbqualität — Lauf abbrechen, "
                        "'Auto-Kalibrierung (grob)' starten und neu einfärben.")
            return colorizer.colorize(rec, bag, calib, params, out_dir,
                                      progress_cb=progress_cb, cancel=cancel)

        self._start_worker("Färbe Punktwolke ein …", job, self._on_colorize_done)

    def _on_colorize_done(self, res: dict) -> None:
        n_valid = int(res.get("n_valid", 0))
        frac = float(res.get("frac_valid", 0.0))
        self._log(f"Einfärbung fertig: {_fmt_int(n_valid)} Punkte gültig "
                  f"({100.0 * frac:.1f} %).")
        n_sky = int(res.get("n_sky_blocked", 0))
        if n_sky:
            self._log(f"Himmelssaum-Sperre: {_fmt_int(n_sky)} Farbproben verworfen "
                      f"(Saum {self._spin_sky.value()} px um ausgebrannte Flächen).")
        colors, valid, err = _load_color_files(self._project.colors_dir(),
                                               self._rec.n_points)
        if err:
            self._show_error("Einfärben", err)
            return
        self._colors, self._valid = colors, valid
        self._cloud_view.set_cloud(self._world, self._colors,
                                   self._rec.intensity, self._valid)
        idx = self._combo_colormode.findData("rgb")
        self._combo_colormode.setCurrentIndex(idx)  # löst _on_display_changed aus
        self._push_display_settings()

    def _on_overlay_clicked(self) -> None:
        if self._rec is None or self._bag is None:
            return
        try:
            colorizer = self._import_colorizer()
        except RuntimeError as exc:
            self._show_error("Overlay-Vorschau", str(exc))
            return
        frame_idx = self._pano_view.current_index
        if frame_idx < 0:
            frame_idx = max(0, self._n_frames // 2)
        T = self._extrinsic_from_spins()
        rec, bag, calib = self._rec, self._bag, self._calib

        def job(progress_cb, cancel, log_cb):
            progress_cb(0.2, f"Erzeuge Overlay für Frame {frame_idx} …")
            return colorizer.overlay_preview(rec, bag, calib, T, frame_idx, stride=50)

        def on_done(img) -> None:
            dlg = _ImageDialog(f"Overlay-Vorschau — Frame {frame_idx}", img, self)
            self._overlay_dialogs = [d for d in self._overlay_dialogs if d.isVisible()]
            self._overlay_dialogs.append(dlg)
            dlg.show()

        # Einzelner Bibliotheksaufruf ohne Cancel-Auswertung — nicht abbrechbar.
        self._start_worker("Erzeuge Overlay-Vorschau …", job, on_done,
                           cancellable=False)

    def _on_autocal_clicked(self) -> None:
        if self._rec is None or self._bag is None:
            return
        try:
            colorizer = self._import_colorizer()
        except RuntimeError as exc:
            self._show_error("Auto-Kalibrierung", str(exc))
            return
        T_init = self._extrinsic_from_spins()
        rec, bag, calib = self._rec, self._bag, self._calib

        def job(progress_cb, cancel, log_cb):
            return colorizer.auto_calibrate(rec, bag, calib, T_init=T_init,
                                            progress_cb=progress_cb, cancel=cancel)

        def on_done(result) -> None:
            T, score = result
            self._spins_from_extrinsic(np.asarray(T))
            try:
                self._project.save_extrinsic(np.asarray(T))
            except RuntimeError as exc:
                self._log(f"Extrinsik nicht gespeichert: {exc}")
            self._log(f"Auto-Kalibrierung fertig — Score {score:.3f}.")
            if score < _AUTOCAL_WEAK_SCORE and not self._autotest_active():
                QMessageBox.warning(
                    self, "Auto-Kalibrierung",
                    f"Schwacher Kalibrier-Score ({score:.3f}) — Ergebnis bitte mit "
                    "der Overlay-Vorschau prüfen und ggf. von Hand nachjustieren.")

        self._start_worker("Auto-Kalibrierung läuft …", job, on_done)

    # ================================================================== Export

    def _export_arrays(self) -> tuple[np.ndarray, np.ndarray | None]:
        pts, cols = self._world, self._colors
        if (self._chk_only_colored.isChecked() and self._valid is not None
                and cols is not None):
            mask = self._valid
            pts = pts[mask]
            cols = cols[mask]
        return pts, cols

    def _on_export_plypcd(self) -> None:
        if self._world is None:
            return
        start = os.path.join(self._project.dir if self._project else "",
                             "punktwolke.ply")
        path, chosen = QFileDialog.getSaveFileName(
            self, "Punktwolke speichern", start,
            "PLY-Datei (*.ply);;PCD-Datei (*.pcd)")
        if not path:
            return
        if not path.lower().endswith((".ply", ".pcd")):
            path += ".pcd" if "pcd" in chosen.lower() else ".ply"
        pts, cols = self._export_arrays()

        def job(progress_cb, cancel, log_cb):
            progress_cb(0.2, f"Schreibe {os.path.basename(path)} …")
            georef.export_ply_pcd(pts, cols, path)
            return path

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
        path, _ = QFileDialog.getSaveFileName(
            self, "LAS speichern", start, "LAS-Datei (*.las)")
        if not path:
            return
        if not path.lower().endswith(".las"):
            path += ".las"
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

    def _on_georef_ready(self, result) -> None:
        self._georef = result
        self._log(f"Georeferenzierung bereit: EPSG:{result.utm_epsg}, "
                  f"RMS {result.rms_m:.2f} m, {result.n_used} Fixe — "
                  "LAS-Export verwendet jetzt UTM-Koordinaten.")

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

    # ================================================================ Lifecycle

    def closeEvent(self, event) -> None:
        self._closing = True
        if self._worker is not None:
            self._worker.cancel.set()
            self._status_lbl.setText("Breche ab — warte auf Aufräumen …")
            self.statusBar().repaint()
            # Unbegrenzt warten (Korrektheit vor Reaktionszeit): das Aufräumen
            # der Subprozesse (fast_lio, bag play, Recorder) muss vollständig
            # durchlaufen; ein zerstörter, noch laufender QThread würde Qt mit
            # "QThread: Destroyed while thread is still running" abbrechen
            # lassen und ROS-Prozesse verwaisen.
            self._worker.wait()
        for w in self._retired:
            w.wait(30000)
        try:
            self._rviz_player.stop()      # sonst bleiben rviz2/bag play verwaist
        except Exception as exc:
            print(f"RViz-Aufräumen: {exc}", file=sys.stderr)
        self._gps_panel.shutdown()
        self._save_settings()
        super().closeEvent(event)


if __name__ == "__main__":
    os.environ.setdefault("DISPLAY", ":0")
    app = QApplication(sys.argv)
    try:
        import qdarktheme
        qdarktheme.setup_theme("dark", custom_colors={"primary": "#4FC3F7"})
    except Exception as exc:  # Theme ist Kosmetik — ohne weiterlaufen
        print(f"qdarktheme nicht aktiv: {exc}", file=sys.stderr)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
