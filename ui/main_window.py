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
    QSplitter,
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
from ui.collapsible import SectionStack
from ui import menubar as menubar_mod
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

#: Farbebenen fuer die Auswahl — Reihenfolge wie in Project.LAYERS
_LAYER_LABELS = (
    ("onboard", "Onboard RGB (360°-Kamera)"),
    ("meander_rgb", "Mäander RGB (DJI)"),
    ("meander_thermal", "Mäander Thermal (DJI)"),
)

#: So viele Punkte gehen in die Live-Vorschau der Handjustage. Bei 50.000
#: dauert ein Durchlauf rund 80 ms — schnell genug, um dem Regler zu folgen.
_LIVE_PUNKTE = 50_000

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
    "layer": "onboard",
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
        # Zusammenfuehren: zweiter Flug, seine Lage und die Quellen-Abschnitte
        # einer bereits zusammengefuehrten Aufzeichnung (fuer die Einfaerbung).
        self._merge_bag: Optional[ThreadLocalBag] = None
        self._merge_rec = None
        self._merge_cloud: Optional[np.ndarray] = None
        self._merge_T = np.eye(4)
        self._merge_center = np.zeros(3)
        self._parts: Optional[list] = None
        # Farbebenen: Schluessel -> (rgb, maske). 'onboard' kommt aus der
        # 360-Kamera, die beiden anderen aus dem Maeanderflug.
        self._layers: dict = {}
        self._layer_key = "onboard"
        self._meander_pipe = None
        self._meander_dir: Optional[str] = None
        # Live-Vorschau der Handjustage: verkleinerte Bilder im Speicher
        # plus eine Stichprobe der Wolke. Ein Durchlauf kostet damit rund
        # 80 ms statt Minuten, die Wolke folgt dem Regler.
        self._live = None
        self._live_pts: Optional[np.ndarray] = None
        self._live_gemeckert = False   # Warnung bei 0 % nur einmal je Sitzung
        self._live_timer = QTimer(self)
        self._live_timer.setSingleShot(True)
        self._live_timer.setInterval(120)
        self._live_timer.timeout.connect(self._live_update)
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
        self._sidebar_scroll = QScrollArea(self)
        self._sidebar_scroll.setWidget(sidebar)
        self._sidebar_scroll.setWidgetResizable(True)
        # Soll ~360-400 px; bei großen Systemfonts (Hi-DPI) so weit aufweiten,
        # dass nichts abgeschnitten wird (Fontbreiten skalieren die Minima).
        # Breite am BREITESTEN Abschnitt messen, nicht am Stapel: dessen
        # sizeHint bleibt hinter seinen Kindern zurueck (die Kopfzeilen duerfen
        # sich dehnen), und die Leiste waere dann zu schmal. Zugeklappte
        # Abschnitte zaehlen mit, sonst haengt die Breite davon ab, was beim
        # Start zufaellig offen ist.
        need = 0
        for sec in self._sections._sections.values():
            need = max(need, sec.content().sizeHint().width())
        need += self._sidebar_scroll.verticalScrollBar().sizeHint().width() + 26
        self._sidebar_breite = max(400, min(720, need))
        self._sidebar_scroll.setMinimumWidth(300)
        self._sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._tabs = QTabWidget(self)
        self._cloud_view = CloudView(self)
        # 3D-Ansicht plus Auslese-Zeile fuers Messen darunter
        karte = QWidget(self)
        karte_lay = QVBoxLayout(karte)
        karte_lay.setContentsMargins(0, 0, 0, 0)
        karte_lay.setSpacing(2)
        karte_lay.addWidget(self._cloud_view, 1)
        self._mess_lbl = QLabel("", self)
        self._mess_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._mess_lbl.setStyleSheet("padding: 2px 6px;")
        karte_lay.addWidget(self._mess_lbl, 0)
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
        self._tabs.addTab(karte, "3D-Karte")
        self._tabs.addTab(self._pano_view, "360°-Video")
        self._tabs.addTab(self._gps_panel, "GPS")
        self._tabs.addTab(self._log_edit, "Protokoll")
        # Arbeitsfläche links, Bedienung rechts — der Splitter lässt die
        # Seitenleiste in der Breite ziehen, Ctrl+B blendet sie ganz aus.
        self._splitter = QSplitter(Qt.Horizontal, self)
        self._splitter.addWidget(self._tabs)
        self._splitter.addWidget(self._sidebar_scroll)
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 0)
        self._splitter.setCollapsible(0, False)
        self._splitter.setSizes([1200, self._sidebar_breite])
        root.addWidget(self._splitter, 1)
        self.setCentralWidget(central)

        # ------------------------------------------------------- Menüleiste
        self._actions = menubar_mod.build(self)
        self._fill_view_menus()
        self._refresh_layer_combo()   # ohne Projekt: 'keine Einfärbung'
        self._sync_preview_action()

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

        self._cloud_view.measured.connect(self._on_measured)
        self._pano_view.frameChanged.connect(self._on_pano_frame)
        self._gps_panel.georefReady.connect(self._on_georef_ready)

    # Reihenfolge der Seitenleiste: der Arbeitsablauf von oben nach unten,
    # nicht die Reihenfolge, in der die Teile entstanden sind.
    _SECTIONS = (
        ("aufnahme", "1 · Aufnahme", "_group_rosbag", True),
        ("karte", "2 · Karte (FAST-LIO2)", "_group_fastlio", True),
        ("einfaerbung", "3 · Einfärbung (360°-Kamera)", "_group_colorize", True),
        ("maeander", "4 · Mäander-Einfärbung", "_group_meander", False),
        ("zusammen", "5 · Zusammenführen", "_group_merge", False),
        ("anzeige", "6 · Anzeige", "_group_display", True),
        ("wiedergabe", "7 · Wiedergabe (RViz)", "_group_rviz", False),
        ("export", "8 · Export", "_group_export", True),
    )

    def _build_sidebar(self) -> QWidget:
        self._sections = SectionStack()
        for key, titel, bauen, offen in self._SECTIONS:
            self._sections.add(key, titel, getattr(self, bauen)(), offen)
        self._sections.finish()
        self._sections.toggled.connect(lambda *_: self._on_setting_changed())
        return self._sections

    def _group_rosbag(self) -> QWidget:
        box = QWidget()
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

    def _group_fastlio(self) -> QWidget:
        box = QWidget()
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

    def _group_colorize(self) -> QWidget:
        box = QWidget()
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

    def _group_meander(self) -> QWidget:
        box = QWidget()
        form = _wrappable(QFormLayout(box))
        self._btn_meander_pick = QPushButton("Mäanderflug wählen …")
        self._btn_meander_pick.setToolTip(
            "Ordner mit den Bildern eines DJI-Kartierungsfluges.\n"
            "Gesucht werden die _V.JPG, die _T.JPG sind die Thermalbilder.")
        self._btn_meander_pick.clicked.connect(self._on_meander_pick)
        form.addRow(self._btn_meander_pick)
        self._lbl_meander = QLabel("Kein Mäanderflug geladen.")
        self._lbl_meander.setWordWrap(True)
        form.addRow(self._lbl_meander)

        self._chk_thermal = QCheckBox("Thermalbilder mitrechnen")
        self._chk_thermal.setToolTip(
            "Färbt ein zweites Mal mit den _T.JPG und legt eine eigene Ebene an.\n"
            "Eine zweite Rekonstruktion braucht es nicht — beide Optiken sitzen\n"
            "auf derselben Gimbal und lösen zusammen aus.")
        self._chk_thermal.stateChanged.connect(self._on_setting_changed)
        form.addRow(self._chk_thermal)

        row = QWidget()
        hl = QHBoxLayout(row)
        hl.setContentsMargins(0, 0, 0, 0)
        self._btn_meander_align = QPushButton("Ausrichten")
        self._btn_meander_align.setToolTip(
            "Grob per Kreuzkorrelation über den Gierwinkel, fein über den\n"
            "Höhenunterschied zum Rastermodell der Wolke. Kein ICP — das würde\n"
            "an Gebäudekanten verkippen und die Lotrechte zerstören.")
        self._btn_meander_align.clicked.connect(self._on_meander_align)
        self._btn_meander_run = QPushButton("Einfärben")
        self._btn_meander_run.clicked.connect(self._on_meander_run)
        hl.addWidget(self._btn_meander_align)
        hl.addWidget(self._btn_meander_run)
        form.addRow(row)
        self._btn_meander_fenster = QPushButton("Überlagern und justieren …")
        self._btn_meander_fenster.setToolTip(
            "Eigenes Fenster: Karte und Flug übereinander, live verschieben,\n"
            "mit Farbvorschau. Das Hauptfenster bleibt unberührt.")
        self._btn_meander_fenster.clicked.connect(self._on_meander_fenster)
        form.addRow(self._btn_meander_fenster)

        # Handjustage: verschiebt die Fotopunkte starr gegen die Wolke
        grid_holder = QWidget()
        grid = QGridLayout(grid_holder)
        grid.setContentsMargins(0, 0, 0, 0)
        self._spin_meander = {}
        for col, (key, label, rng, step, suffix) in enumerate((
                ("yaw", "Gier", 180.0, 0.5, "°"),
                ("x", "X", 500.0, 0.5, " m"),
                ("y", "Y", 500.0, 0.5, " m"))):
            sp = QDoubleSpinBox()
            sp.setRange(-rng, rng)
            sp.setSingleStep(step)
            sp.setDecimals(2)
            sp.setSuffix(suffix)
            sp.valueChanged.connect(self._on_meander_manual)
            grid.addWidget(QLabel(label), 0, col)
            grid.addWidget(sp, 1, col)
            self._spin_meander[key] = sp
        form.addRow("Lage von Hand:", grid_holder)
        self._chk_solo = QCheckBox("Während der Justage nur die Vorschau zeigen")
        self._chk_solo.setChecked(False)
        self._chk_solo.setToolTip(
            "Blendet die volle Karte aus, solange die Vorschau läuft.\n"
            "50.000 Stichprobenpunkte gehen in 24 Millionen sonst unter.\n"
            "Abschalten zeigt beides übereinander.")
        self._chk_solo.stateChanged.connect(self._on_solo_changed)
        form.addRow(self._chk_solo)
        self._lbl_meander_lage = QLabel("")
        self._lbl_meander_lage.setWordWrap(True)
        form.addRow(self._lbl_meander_lage)

        # Hauptpunkt-Versatz je Optik: wirkt wie eine Verkippung der Kamera
        # gegen die Achse, die COLMAP angenommen hat, und waechst mit dem
        # Abstand — anders als die Regler darueber, die starr schieben.
        self._spin_optik = {}
        for optik, titel, tip in (
                ("rgb", "RGB-Optik", "Versatz des Bildhauptpunkts in Pixeln des RGB-Bildes."),
                ("thermal", "Thermal-Optik", "Dasselbe für die Thermaloptik — eigener Wert, "
                                             "es ist ein zweites Objektiv.")):
            holder = QWidget()
            g = QGridLayout(holder)
            g.setContentsMargins(0, 0, 0, 0)
            for col, (achse, label) in enumerate((("u", "rechts"), ("v", "unten"))):
                sp = QDoubleSpinBox()
                sp.setRange(-400.0, 400.0)
                sp.setSingleStep(1.0)
                sp.setDecimals(1)
                sp.setSuffix(" px")
                sp.setToolTip(tip)
                sp.valueChanged.connect(self._on_setting_changed)
                g.addWidget(QLabel(label), 0, col)
                g.addWidget(sp, 1, col)
                self._spin_optik[(optik, achse)] = sp
            form.addRow(f"{titel}:", holder)
        return box

    def _group_merge(self) -> QWidget:
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

        grid_holder = QWidget()
        grid = QGridLayout(grid_holder)
        grid.setContentsMargins(0, 0, 0, 0)
        self._spin_merge = {}
        for col, (key, label, rng, step, suffix) in enumerate((
                ("x", "X", 500.0, 0.1, " m"), ("y", "Y", 500.0, 0.1, " m"),
                ("z", "Z", 500.0, 0.1, " m"), ("yaw", "Gier", 180.0, 1.0, "°"))):
            sp = QDoubleSpinBox()
            sp.setRange(-rng, rng)
            sp.setSingleStep(step)
            sp.setDecimals(2)
            sp.setSuffix(suffix)
            sp.valueChanged.connect(self._on_merge_manual)
            grid.addWidget(QLabel(label), 0, col)
            grid.addWidget(sp, 1, col)
            self._spin_merge[key] = sp
        form.addRow(grid_holder)

        row = QWidget()
        hl = QHBoxLayout(row)
        hl.setContentsMargins(0, 0, 0, 0)
        self._btn_merge_auto = QPushButton("Auto-Ausrichten")
        self._btn_merge_auto.setToolTip(
            "Globale Suche (FGR über FPFH) plus ICP von grob nach fein.\n"
            "Dauert je nach Wolkengröße ein bis mehrere Minuten.")
        self._btn_merge_auto.clicked.connect(lambda: self._on_merge_align("auto"))
        self._btn_merge_icp = QPushButton("Nur ICP")
        self._btn_merge_icp.setToolTip(
            "Verfeinert nur die aktuelle Lage — nach einer Handjustage genug.")
        self._btn_merge_icp.clicked.connect(lambda: self._on_merge_align("icp"))
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

    def _group_display(self) -> QWidget:
        box = QWidget()
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
        self._combo_layer = _compact_combo(QComboBox())
        self._combo_layer.setToolTip(
            "Welche Einfärbung gezeigt wird. Angeboten wird, was berechnet ist.")
        for data, label in _LAYER_LABELS:
            self._combo_layer.addItem(label, data)
        self._combo_layer.currentIndexChanged.connect(self._on_layer_changed)
        form.addRow("Farbquelle:", self._combo_layer)
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

    def _group_rviz(self) -> QWidget:
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

    def _group_export(self) -> QWidget:
        box = QWidget()
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
                "Zuschlag auf die gefundene Lage; wirkt sofort in der Wolke."
                if hat_lage else
                "Erst 'Ausrichten' laufen lassen — vorher gibt es keine Lage, "
                "auf die sich die Regler beziehen könnten.")
        if hasattr(self, "_lbl_meander_lage"):
            self._lbl_meander_lage.setText(self._meander_zustand_text())
        if hasattr(self, "_actions"):
            for key, an in (("project_export", not busy and has_rec),
                            ("project_import", not busy),
                            ("measure", not busy and has_world),
                            ("fastlio", not busy and has_bag),
                            ("colorize", not busy and has_bag and has_rec
                             and bool(self._calib)),
                            ("meander_run", not busy and has_rec),
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

    def _worker_failed(self, worker: Worker, title: str, msg: str,
                       on_failed: Callable[[str], None] | None = None) -> None:
        self._retire(worker)
        # Nach einem Fehlschlag darf keine Solo-Vorschau die Karte verdecken:
        # sonst sieht ein abgebrochener Lauf so aus, als sei das Modell weg.
        if self._cloud_view.has_color_preview():
            self._live_hide()
            self._log("Vorschau geräumt — die Karte ist wieder sichtbar.")
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
            self._chk_thermal.setChecked(bool(s.get("meander_thermal", False)))
            self._chk_solo.setChecked(bool(s.get("meander_solo", True)))
            for optik, key in (("rgb", "rgb_versatz"),
                               ("thermal", "thermal_versatz")):
                v = s.get(key) or [0.0, 0.0]
                self._spin_optik[(optik, "u")].setValue(float(v[0]))
                self._spin_optik[(optik, "v")].setValue(float(v[1]))
            d = s.get("meander_dir") or ""
            if d and os.path.isdir(d):
                self._meander_dir = d
                self._lbl_meander.setText(f"{os.path.basename(d)} (aus den "
                                          f"Einstellungen)")
            self._layer_key = s.get("layer", "onboard")
            self._sections.set_states(s.get("sections") or {})
            self._set_sidebar_visible(bool(s.get("sidebar", True)))
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
            "layer": self._layer_key,
            "meander_dir": self._meander_dir or "",
            "meander_thermal": bool(self._chk_thermal.isChecked()),
            "meander_solo": bool(self._chk_solo.isChecked()),
            "rgb_versatz": self._meander_versatz("rgb"),
            "thermal_versatz": self._meander_versatz("thermal"),
            "only_colored": bool(self._chk_only_colored.isChecked()),
            "voxel": float(self._combo_voxel.currentData()),
            "background": self._combo_bg.currentData(),
            "edl": bool(self._chk_edl.isChecked()),
            "show_path": bool(self._chk_path.isChecked()),
            "sections": self._sections.states(),
            "sidebar": bool(self._sidebar_scroll.isVisible()),
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

    def _sync_after_display(self) -> None:
        if hasattr(self, "_actions"):
            self._sync_menu_state()

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
        self._sync_after_display()
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
        self._parts = None
        self._layers = {}
        self._meander_pipe = None
        self._live_clear()
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
        self._colors = res["colors"]
        self._valid = res["valid"]
        # Immer aus der Aufzeichnung ableiten, nicht aus dem Sitzungsgedaechtnis
        self._parts = self._parts_from_meta(self._rec.meta)
        if self._parts:
            self._bag = self._parts[0][0]
        self._reload_layers()
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
        if self._busy:
            QMessageBox.information(self, "Beschäftigt",
                                    "Es läuft noch ein Arbeitsschritt.")
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
        # Einzelne Fluege gehen den normalen Weg — dann stehen auch das
        # 360-Video und die GPS-Pruefung zur Verfuegung. Zusammengefuehrte
        # haben keinen einzelnen Bagpfad und werden aus dem Cache geoeffnet.
        if eintrag and not eintrag["zusammengefuehrt"] and \
                project.bag_path and os.path.exists(project.bag_path):
            self._open_bag(project.bag_path)
            return
        self._open_project_only(project)

    def _on_import_project(self) -> None:
        if self._busy:
            QMessageBox.information(self, "Beschäftigt",
                                    "Es läuft noch ein Arbeitsschritt.")
            return
        src = QFileDialog.getExistingDirectory(
            self, "Ordner eines exportierten Projekts", os.path.expanduser("~"))
        if not src:
            return
        from core import bundle
        try:
            manifest = bundle.read_manifest(src)
        except RuntimeError as exc:
            self._show_error("Projekt importieren", str(exc))
            return
        bags = bundle.bag_paths_after_import(src, manifest)
        ziel_bag = bags[0] if bags else manifest["projekt"]["bag"]
        try:
            project = Project(ziel_bag)
        except RuntimeError as exc:
            self._show_error("Projekt importieren", str(exc))
            return
        if project.has_recording():
            frage = QMessageBox.question(
                self, "Projekt bereits vorhanden",
                f"Für '{project.bag_name}' liegt hier schon ein Projekt im "
                f"Cache. Soll es durch das importierte ersetzt werden?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if frage != QMessageBox.Yes:
                return

        def job(progress_cb, cancel, log_cb):
            return bundle.import_project(
                src, project, progress=progress_cb,
                cancel=lambda: cancel.is_set())

        self._start_worker(f"Importiere Projekt aus {os.path.basename(src)} …",
                           job, self._on_project_imported)

    def _on_project_imported(self, res: dict) -> None:
        from core import bundle
        m = res["manifest"]
        self._log(f"Projekt importiert ({bundle.fmt_size(res['bytes'])}), "
                  f"exportiert am {m.get('erstellt', '?')}.")
        bags = [b for b in res["bags"] if b]
        fehlt = res["fehlende_bags"]
        if fehlt:
            for b in fehlt:
                self._log(f"Rosbag nicht am Ort: {b}")
            self._log("Ohne Bag: 360°-Video und erneutes Einfärben stehen nicht "
                      "zur Verfügung. Karte, Farben, Messen und Export schon.")
            self._open_project_only(res["project"])
        else:
            self._open_bag(bags[0])

    def _open_project_only(self, project) -> None:
        """Projekt ohne Bag oeffnen — nur, was aus dem Cache lebt."""
        self._clear_bag_state()
        self._project = project
        self._bag = None
        self._bag_info = None
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
        self.setWindowTitle(f"Super360 Studio — {project.bag_name} (ohne Bag)")
        self._update_enabled()
        if project.has_recording():
            self._start_recording_load()

    # ================================================= Mäander-Einfärbung

    def _meander_versatz(self, optik: str) -> list:
        return [float(self._spin_optik[(optik, "u")].value()),
                float(self._spin_optik[(optik, "v")].value())]

    def _on_meander_pick(self) -> None:
        if self._rec is None or self._project is None:
            QMessageBox.information(
                self, "Mäander-Einfärbung",
                "Erst einen Flug öffnen und seine Karte berechnen — sie ist die "
                "Wolke, die eingefärbt wird.")
            return
        start = self._meander_dir or os.path.expanduser("~")
        path = QFileDialog.getExistingDirectory(
            self, "Ordner mit den Bildern des Mäanderfluges", start)
        if not path:
            return
        n_v = len([f for f in os.listdir(path) if f.upper().endswith("_V.JPG")])
        n_t = len([f for f in os.listdir(path) if f.upper().endswith("_T.JPG")])
        if n_v == 0:
            n_v = len([f for f in os.listdir(path)
                       if f.upper().endswith((".JPG", ".JPEG"))
                       and not f.upper().endswith("_T.JPG")])
        if n_v == 0:
            self._show_error("Mäander-Einfärbung",
                             f"In '{os.path.basename(path)}' liegen keine JPEGs.")
            return
        self._meander_dir = path
        self._meander_pipe = None      # Pipeline wird beim nächsten Lauf neu gebaut
        self._live_clear()
        self._chk_thermal.setEnabled(n_t > 0)
        if n_t == 0:
            self._chk_thermal.setChecked(False)
        self._lbl_meander.setText(
            f"{os.path.basename(path)}: {n_v} RGB-Bilder"
            + (f", {n_t} Thermalbilder" if n_t else ", keine Thermalbilder"))
        self._log(f"Mäanderflug gewählt: {path} — {n_v} RGB, {n_t} Thermal.")
        self._save_settings()
        self._update_enabled()

    def _meander_args(self) -> dict:
        """Alles, was der Arbeitsthread braucht — im GUI-Thread eingesammelt.

        Widgets duerfen nur hier gelesen werden. Der Worker laeuft in einem
        eigenen Thread, und Qt-Widgets von dort anzufassen ist ein Fehler, der
        sich erst spaeter und schlecht reproduzierbar zeigt.
        """
        return {
            "points": self._world,
            "photo_dir": self._meander_dir,
            "work_dir": self._project.meander_work_dir(),
            "thermal": bool(self._chk_thermal.isChecked()),
            "rgb_versatz": self._meander_versatz("rgb"),
            "thermal_versatz": self._meander_versatz("thermal"),
        }

    @staticmethod
    def _meander_build(args: dict, log_cb):
        """Pipeline aufsetzen; die Arbeitswolke geht als cloud.npy hinein."""
        from core import meander as meander_mod
        return meander_mod.build_pipeline(
            args["points"], args["photo_dir"], args["work_dir"],
            thermal=args["thermal"], rgb_versatz=args["rgb_versatz"],
            thermal_versatz=args["thermal_versatz"], log=log_cb)

    def _meander_ask_colmap(self) -> bool:
        """Vor einer Rekonstruktion fragen — die dauert eine halbe Stunde."""
        work = self._project.meander_work_dir()
        if os.path.exists(os.path.join(work, "cameras.npz")):
            return True
        if os.path.exists(os.path.join(work, "sparse", "0", "cameras.bin")):
            return True
        from core import meander as meander_mod
        if meander_mod.find_colmap_python() is None:
            self._show_error(
                "Mäander-Einfärbung",
                "Für die Rekonstruktion wird ein Interpreter mit pycolmap "
                "gebraucht, es ist keiner gefunden worden. Entweder pycolmap "
                "installieren oder ein fertiges COLMAP-Modell als sparse/0 in "
                f"'{work}' ablegen.")
            return False
        n = len([f for f in os.listdir(self._meander_dir)
                 if f.upper().endswith((".JPG", ".JPEG"))
                 and not f.upper().endswith("_T.JPG")])
        return QMessageBox.question(
            self, "Rekonstruktion nötig",
            f"Für diesen Flug gibt es noch kein COLMAP-Modell.\n\n"
            f"Die Rekonstruktion von {n} Bildern dauert etwa eine halbe Stunde. "
            f"Danach liegt sie im Arbeitsordner und wird wiederverwendet.\n\n"
            f"Jetzt rechnen?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes

    def _on_meander_align(self) -> None:
        if not self._meander_dir or self._world is None:
            QMessageBox.information(self, "Mäander-Einfärbung",
                                    "Erst einen Mäanderflug wählen.")
            return
        if not self._meander_ask_colmap():
            return
        args = self._meander_args()
        bauen = self._meander_build

        def job(progress_cb, cancel, log_cb):
            from core import meander as meander_mod
            pipe = bauen(args, log_cb)
            pipe._cancel = lambda: cancel.is_set()
            vor = meander_mod.prepare(
                pipe, progress=lambda f, m: progress_cb(0.05 + 0.7 * f, m))
            k = meander_mod.align(
                pipe, progress=lambda f, m: progress_cb(0.75 + 0.25 * f, m))
            return {"pipe": pipe, "vor": vor, "kennwerte": k}

        self._start_worker("Richte den Mäanderflug aus …", job,
                           self._on_meander_aligned)

    def _on_meander_aligned(self, res: dict) -> None:
        self._meander_pipe = res["pipe"]
        v, k = res["vor"], res["kennwerte"]
        anteil = k.get("anteil_auf_flaeche")
        med = k.get("median_abweichung")
        self._lbl_meander.setText(
            f"Ausgerichtet: {k['yaw_deg']:.2f}°"
            + (f", {anteil * 100:.0f} % der Fotopunkte auf der Oberfläche "
               f"(Median {med:.2f} m)" if anteil is not None else ""))
        self._log(f"Mäander: {v['kameras']} Kameras, Maßstab {v['massstab']:.3f}, "
                  f"GPS-Residuum {v['gps_residuum']:.2f} m.")
        self._log(f"Ausrichtung: {k['yaw_deg']:.2f}°, Versatz "
                  f"{np.round(k['t'], 2).tolist()} m.")
        from core import meander as meander_mod
        schlecht = meander_mod.pruefe_ausrichtung(k)
        if schlecht:
            self._log("WARNUNG: " + schlecht)
            self._lbl_meander.setText(
                f"Ausrichtung fraglich: {k['yaw_deg']:.2f}°, nur "
                f"{anteil * 100:.1f} % auf der Oberfläche.")
        for key in ("yaw", "x", "y"):
            sp = self._spin_meander[key]
            sp.blockSignals(True)
            sp.setValue(0.0)
            sp.blockSignals(False)
        self._update_enabled()
        self._start_live_preview()

    def _start_live_preview(self) -> None:
        """Verkleinerte Bilder laden, damit die Handjustage live wirkt."""
        pipe = self._meander_pipe
        if pipe is None or pipe.cams is None or self._world is None:
            return
        welt = self._world
        n = len(welt)
        schritt = max(1, n // _LIVE_PUNKTE)
        stich = np.ascontiguousarray(welt[::schritt][:_LIVE_PUNKTE],
                                     dtype=np.float64)
        cams = pipe.rgb_cams()
        bilder = pipe._p("images")

        def job(progress_cb, cancel, log_cb):
            from core import meander as meander_mod
            live = meander_mod.LivePreview(
                cams, bilder, progress=progress_cb,
                cancel=lambda: cancel.is_set())
            return {"live": live, "pts": stich}

        def fertig(res: dict) -> None:
            self._live = res["live"]
            self._live_pts = res["pts"]
            self._log(f"Live-Vorschau bereit: {_fmt_int(len(res['pts']))} "
                      f"Punkte, {len(res['live'].bilder)} verkleinerte Bilder. "
                      f"Sie erscheint beim ersten Zug an Gier, X oder Y und "
                      f"blendet die Karte dabei aus (Haken darüber schaltet "
                      f"das ab). Die Karte bleibt bis dahin stehen.")
            self._update_enabled()
            # Bewusst KEIN _live_update hier: die Karte soll nach dem
            # Ausrichten stehen bleiben. Wer nichts justiert, will sie sehen.

        self._start_worker("Lade Vorschaubilder für die Handjustage …",
                           job, fertig)

    def _on_solo_changed(self) -> None:
        self._cloud_view.set_preview_solo(bool(self._chk_solo.isChecked()))
        if hasattr(self, "_lbl_meander_lage"):
            self._lbl_meander_lage.setText(self._meander_zustand_text())
        self._save_settings()

    def _meander_zustand_text(self) -> str:
        """Was die Handregler gerade koennen — und was fehlt, wenn nicht."""
        if not self._meander_dir:
            return "Noch kein Mäanderflug gewählt."
        pipe = self._meander_pipe
        if pipe is None or getattr(pipe, "yaw", None) is None:
            return ("Noch nicht ausgerichtet — die Regler brauchen eine Lage, "
                    "auf die sie sich beziehen. Erst „Ausrichten“.")
        if self._live is None:
            return (f"Ausgerichtet auf {np.degrees(pipe.yaw):.2f}°. "
                    f"Vorschaubilder werden noch geladen …")
        if self._cloud_view.has_color_preview():
            wo = ("die volle Karte ist solange ausgeblendet"
                  if not self._cloud_view.map_visible()
                  else "über der vollen Karte")
            return (f"Ausgerichtet auf {np.degrees(pipe.yaw):.2f}°. Gezeigt wird "
                    f"die Vorschau aus {_fmt_int(len(self._live_pts))} Punkten, "
                    f"{wo}. Magenta sind die Kamerastandorte, Grau ist von "
                    f"keinem Bild getroffen.")
        return (f"Ausgerichtet auf {np.degrees(pipe.yaw):.2f}°, Live-Vorschau "
                f"mit {_fmt_int(len(self._live_pts))} Punkten bereit.")

    def _meander_lage(self) -> tuple:
        """Ausgerichtete Lage plus Handjustage: (yaw_grad, t als 3er-Vektor).

        Die Hoehe bleibt stehen — geregelt werden nur Gier, X und Y. Sie muss
        aber mitgeführt werden: ``register.affine`` rechnet mit drei
        Komponenten, ein 2er-Vektor bricht dort ab.
        """
        from core import meander as meander_mod
        pipe = self._meander_pipe
        t = meander_mod.as_t3(pipe.t)
        t[0] += float(self._spin_meander["x"].value())
        t[1] += float(self._spin_meander["y"].value())
        return (float(np.degrees(pipe.yaw)) + float(self._spin_meander["yaw"].value()),
                t)

    def _on_meander_fenster(self) -> None:
        """Ausrichtfenster oeffnen: Karte und Flug uebereinander, live justierbar."""
        pipe = self._meander_pipe
        if pipe is None or getattr(pipe, "yaw", None) is None:
            QMessageBox.information(
                self, "Überlagern",
                "Erst „Ausrichten“ laufen lassen — das Fenster zeigt die "
                "gefundene Lage und lässt sie von Hand nachziehen.")
            return
        if self._world is None:
            return
        from ui.meander_align_window import MeanderAlignWindow
        stich = self._live_pts
        if stich is None:
            n = len(self._world)
            stich = np.ascontiguousarray(
                self._world[::max(1, n // _LIVE_PUNKTE)][:_LIVE_PUNKTE],
                dtype=np.float64)
        fenster = MeanderAlignWindow(self._world, pipe, self._live, stich, self)
        fenster.uebernommen.connect(self._on_meander_fenster_lage)
        fenster.setAttribute(Qt.WA_DeleteOnClose, True)
        self._meander_fenster = fenster       # Referenz halten, sonst weg
        fenster.show()
        if self._live is None:
            self._log("Das Ausrichtfenster zeigt die Überlagerung. Für die "
                      "Farbvorschau darin werden die Vorschaubilder gebraucht — "
                      "die lädt „Ausrichten“ im Anschluss.")

    def _on_meander_fenster_lage(self, yaw_deg: float, t) -> None:
        """Lage aus dem Ausrichtfenster als neue Basis uebernehmen."""
        from core import meander as meander_mod
        meander_mod.set_manual(self._meander_pipe, float(yaw_deg), t)
        for sp in self._spin_meander.values():
            sp.blockSignals(True)
            sp.setValue(0.0)
            sp.blockSignals(False)
        self._log(f"Lage aus dem Ausrichtfenster übernommen: Gier "
                  f"{yaw_deg:.2f}°, Versatz {np.round(np.asarray(t)[:2], 2).tolist()} m.")
        if hasattr(self, "_lbl_meander_lage"):
            self._lbl_meander_lage.setText(self._meander_zustand_text())

    def _on_meander_manual(self) -> None:
        """Handjustage anwenden und die Vorschau nachziehen.

        Die Basislage bleibt stehen, die Regler sind ein Zuschlag darauf —
        sonst wuerde jeder Reglerzug auf dem vorigen aufbauen und man kaeme nie
        zurueck. Neu gerechnet wird erst nach kurzer Ruhe (Timer), damit ein
        Ziehen nicht Dutzende Durchlaeufe ausloest.
        """
        if self._meander_pipe is None or self._meander_pipe.yaw is None:
            self._log("Handjustage ohne Wirkung: es gibt noch keine "
                      "Ausrichtung. Erst „Ausrichten“ laufen lassen.")
            return
        if self._live is None:
            self._log("Die Vorschaubilder sind noch nicht geladen — die "
                      "Regler wirken, sobald sie da sind.")
            return
        self._live_timer.start()

    def _live_update(self) -> None:
        """Stichprobe mit der aktuellen Lage einfaerben und anzeigen."""
        pipe = self._meander_pipe
        if pipe is None or pipe.yaw is None or self._live is None \
                or self._live_pts is None:
            return
        yaw, t = self._meander_lage()
        # Lage in der Pipeline setzen, damit affine() sie sieht, und danach
        # zuruecklegen — die Basis bleibt, die Regler sind nur ein Zuschlag.
        alt_yaw, alt_t = pipe.yaw, np.asarray(pipe.t, dtype=float).copy()
        try:
            pipe.yaw = float(np.radians(yaw))
            pipe.t = t
            A, b = pipe.affine()
            t0 = time.perf_counter()
            rgb, maske = self._live.colorize(self._live_pts, A, b)
        except Exception as exc:  # noqa: BLE001
            # Ohne diesen Fang verschluckt Qt den Fehler im Timer-Slot und der
            # Regler sieht aus, als bewirke er nichts.
            self._log(f"Live-Vorschau fehlgeschlagen: {exc}")
            self._log(traceback.format_exc())
            self._live_timer.stop()
            return
        finally:
            pipe.yaw, pipe.t = alt_yaw, alt_t
        rgb = rgb.copy()
        rgb[~maske] = 60          # nicht getroffen: dunkel, nicht Fallback-grau
        anteil = float(maske.mean())

        # Kamerastandorte mit einzeichnen. Sieht man nur graue Punkte, ist die
        # erste Frage, ob der Flug ueberhaupt ueber dieser Wolke lag — und das
        # beantwortet ein Blick auf die Standorte sofort.
        punkte, farben = self._live_pts, rgb
        try:
            C = np.asarray(self._meander_pipe.cams["C"], dtype=np.float64)
            C_welt = (np.asarray(A) @ C.T).T + np.asarray(b)
            punkte = np.vstack([self._live_pts, C_welt])
            farben = np.vstack([rgb, np.tile(np.array([255, 0, 200], np.uint8),
                                             (len(C_welt), 1))])
        except Exception:  # noqa: BLE001 — ohne Kameras eben nur die Punkte
            C_welt = None
        self._cloud_view.set_color_preview(
            punkte, farben, solo=bool(self._chk_solo.isChecked()))

        if anteil < 0.05 and not self._live_gemeckert:
            self._live_gemeckert = True
            hinweis = ("Die Vorschau trifft fast nichts: nur "
                       f"{100.0 * anteil:.1f} % der Punkte liegen in einem Bild. "
                       "Alles Graue ist ungetroffen.")
            if C_welt is not None and len(self._live_pts):
                mitte_w = self._live_pts.mean(axis=0)
                mitte_c = C_welt.mean(axis=0)
                abstand = float(np.linalg.norm(mitte_c[:2] - mitte_w[:2]))
                ausdehnung = float(np.linalg.norm(
                    self._live_pts[:, :2].max(0) - self._live_pts[:, :2].min(0)))
                hinweis += (f" Die Kameras (magenta) liegen im Mittel {abstand:.0f} m "
                            f"von der Wolkenmitte entfernt, die Wolke selbst misst "
                            f"{ausdehnung:.0f} m.")
                if abstand > ausdehnung:
                    hinweis += (" Das ist weiter weg als die Wolke breit ist — "
                                "entweder deckt der Mäanderflug dieses Gebiet gar "
                                "nicht ab, oder die Ausrichtung sitzt völlig falsch.")
                else:
                    hinweis += (" Die Kameras liegen über der Wolke; dann fehlt es "
                                "an der Ausrichtung — Gier grob durchdrehen und "
                                "auf die Trefferquote schauen.")
            self._log("WARNUNG: " + hinweis)
        if hasattr(self, "_lbl_meander_lage"):
            self._lbl_meander_lage.setText(self._meander_zustand_text())
        self._status_lbl.setText(
            f"Vorschau: Gier {yaw:.2f}°, Versatz {t[0]:+.1f}/{t[1]:+.1f} m — "
            f"{100.0 * maske.mean():.0f} % getroffen "
            f"({(time.perf_counter() - t0) * 1000:.0f} ms)")

    def _meander_apply_manual(self) -> None:
        """Handjustage endgueltig in die Pipeline schreiben (vor dem Einfaerben)."""
        pipe = self._meander_pipe
        if pipe is None or pipe.yaw is None:
            return
        from core import meander as meander_mod
        yaw, t = self._meander_lage()
        meander_mod.set_manual(pipe, yaw, t)
        for sp in self._spin_meander.values():   # Zuschlag ist verrechnet
            sp.blockSignals(True)
            sp.setValue(0.0)
            sp.blockSignals(False)

    def _live_hide(self) -> None:
        """Nur die Anzeige raeumen; die geladenen Bilder bleiben im Speicher,
        damit ein Nachjustieren danach weiter sofort wirkt."""
        self._live_timer.stop()
        self._cloud_view.set_color_preview(None)

    def _live_clear(self) -> None:
        self._live = None
        self._live_pts = None
        self._live_gemeckert = False
        self._live_hide()

    def _on_meander_run(self) -> None:
        if not self._meander_dir or self._world is None or self._project is None:
            QMessageBox.information(self, "Mäander-Einfärbung",
                                    "Erst einen Mäanderflug wählen.")
            return
        if not self._meander_ask_colmap():
            return
        self._meander_apply_manual()
        pipe = self._meander_pipe
        args = self._meander_args()
        bauen = self._meander_build
        welt = self._world
        thermal = bool(self._chk_thermal.isChecked())
        proj = self._project

        def job(progress_cb, cancel, log_cb):
            from core import meander as meander_mod
            p = pipe
            if p is None:
                p = bauen(args, log_cb)
                p._cancel = lambda: cancel.is_set()
                meander_mod.prepare(
                    p, progress=lambda f, m: progress_cb(0.02 + 0.38 * f, m))
                k = meander_mod.align(
                    p, progress=lambda f, m: progress_cb(0.40 + 0.10 * f, m))
                # Lieber hier abbrechen als Minuten in eine falsche Lage
                # stecken: die Trefferquote beim Einfaerben merkt den
                # Fehlgriff nicht, sie liegt auch dann nahe 100 %.
                schlecht = meander_mod.pruefe_ausrichtung(k)
                if schlecht:
                    raise RuntimeError(schlecht)
            p._cancel = lambda: cancel.is_set()
            A, b = p.affine()
            ergebnis = {"pipe": p, "ebenen": {}}
            progress_cb(0.52, "Färbe die volle Wolke aus den RGB-Bildern …")
            rgb, maske = meander_mod.colorize_points(
                welt, p.rgb_cams(), p._p("images"), A, b,
                progress=lambda f, m: progress_cb(0.52 + 0.28 * f, m),
                cancel=lambda: cancel.is_set())
            meander_mod.save_layer(
                proj.layer_dir("meander_rgb"), rgb, maske,
                {"quelle": "meander_rgb", "flug": p.photo_dir,
                 "yaw_deg": float(np.degrees(p.yaw)),
                 "anteil": float(maske.mean()),
                 "rgb_versatz": [float(x) for x in p.rgb_versatz]})
            ergebnis["ebenen"]["meander_rgb"] = float(maske.mean())
            if thermal and p.thermal_cams() is not None:
                progress_cb(0.82, "Färbe aus den Thermalbildern …")
                trgb, tmaske = meander_mod.colorize_points(
                    welt, p.thermal_cams(), p._p("thermal"), A, b,
                    progress=lambda f, m: progress_cb(0.82 + 0.16 * f, m),
                    cancel=lambda: cancel.is_set())
                meander_mod.save_layer(
                    proj.layer_dir("meander_thermal"), trgb, tmaske,
                    {"quelle": "meander_thermal", "flug": p.photo_dir,
                     "anteil": float(tmaske.mean()),
                     "thermal_versatz": [float(x) for x in p.thermal_versatz]})
                ergebnis["ebenen"]["meander_thermal"] = float(tmaske.mean())
            elif thermal:
                log_cb("Thermal übersprungen: die Optik fehlt (keine Brennweite "
                       "im EXIF der _T.JPG).")
            progress_cb(1.0, "Mäander-Einfärbung fertig")
            return ergebnis

        self._start_worker("Mäander-Einfärbung läuft …", job, self._on_meander_done)

    def _on_meander_done(self, res: dict) -> None:
        self._meander_pipe = res["pipe"]
        for key, anteil in res["ebenen"].items():
            self._log(f"Farbebene '{key}': {anteil * 100:.1f} % der Punkte "
                      f"eingefärbt.")
        if res["ebenen"].get("meander_thermal", 1.0) < 0.9:
            self._log("Der Rest liegt außerhalb der Thermalbilder — die sehen "
                      "einen schmaleren Ausschnitt als die RGB-Kamera.")
        self._live_hide()
        self._reload_layers()
        if self._live is None and self._meander_pipe is not None:
            self._start_live_preview()   # Nachjustieren soll sofort wirken
        if "meander_rgb" in self._layers:
            idx = self._combo_layer.findData("meander_rgb")
            if idx >= 0:
                self._combo_layer.setCurrentIndex(idx)
        self._update_enabled()

    # ======================================================== Farbebenen

    def _reload_layers(self) -> None:
        """Alle vorhandenen Farbebenen des Projekts einlesen.

        Jede Ebene wird gegen die Punktzahl geprueft; was nicht passt, faellt
        weg statt die Anzeige zu verfaelschen. Die Auswahlliste zeigt danach
        nur, was wirklich da ist.
        """
        self._layers = {}
        if self._project is None or self._rec is None:
            self._refresh_layer_combo()
            return
        from core import meander as meander_mod
        n = int(self._rec.n_points)
        for key in Project.LAYERS:
            if not self._project.has_layer(key):
                continue
            if key == "onboard":
                try:
                    from core.colorizer import rec_fingerprint
                    fp = rec_fingerprint(self._rec)
                except Exception:  # noqa: BLE001
                    fp = None
                colors, valid, err = _load_color_files(
                    self._project.layer_dir(key), n, expected_fingerprint=fp,
                    log_cb=self._log)
                if err:
                    self._log(err)
                    continue
                paar = (colors, valid)
            else:
                paar = meander_mod.load_layer(self._project.layer_dir(key), n)
                if paar is None:
                    self._log(f"Farbebene '{key}' passt nicht zur Wolke — ignoriert.")
                    continue
            self._layers[key] = paar
        self._refresh_layer_combo()

    def _refresh_layer_combo(self) -> None:
        """Auswahlliste auf die vorhandenen Ebenen setzen."""
        alt = self._layer_key
        self._loading_ui = True
        try:
            self._combo_layer.clear()
            for data, label in _LAYER_LABELS:
                if data in self._layers:
                    self._combo_layer.addItem(label, data)
            if self._combo_layer.count() == 0:
                self._combo_layer.addItem("keine Einfärbung", "onboard")
                self._combo_layer.setEnabled(False)
            else:
                self._combo_layer.setEnabled(True)
            idx = self._combo_layer.findData(alt)
            if idx < 0:
                idx = 0
            self._combo_layer.setCurrentIndex(idx)
            self._layer_key = self._combo_layer.currentData() or "onboard"
        finally:
            self._loading_ui = False
        self._apply_layer()
        if hasattr(self, "_actions"):
            self._fill_layer_menu()

    def _fill_layer_menu(self) -> None:
        menubar_mod.fill_radio_menu(
            self._actions["menu_farbquelle"], self,
            [(self._combo_layer.itemData(i), self._combo_layer.itemText(i))
             for i in range(self._combo_layer.count())],
            self._on_menu_layer, self._layer_key)

    def _on_menu_layer(self, data) -> None:
        idx = self._combo_layer.findData(data)
        if idx >= 0:
            self._combo_layer.setCurrentIndex(idx)

    def _on_layer_changed(self, *_a) -> None:
        if self._loading_ui:
            return
        self._layer_key = self._combo_layer.currentData() or "onboard"
        self._apply_layer()
        self._save_settings()
        self._log(f"Farbquelle: {self._combo_layer.currentText()}")

    def _apply_layer(self) -> None:
        """Die gewaehlte Ebene in die Ansicht schieben.

        Fehlt fuer die gewaehlte Quelle eine Einfaerbung, waere "RGB" eine
        einfarbig graue Wolke — richtig gerechnet, aber nichtssagend. Dann
        lieber auf Hoehe umschalten und es sagen.
        """
        paar = self._layers.get(self._layer_key)
        self._colors, self._valid = paar if paar else (None, None)
        if self._world is None:
            return
        if self._colors is None and self._combo_colormode.currentData() == "rgb":
            idx = self._combo_colormode.findData("hoehe")
            if idx >= 0:
                self._loading_ui = True
                self._combo_colormode.setCurrentIndex(idx)
                self._loading_ui = False
                self._log("Für diese Farbquelle gibt es noch keine Einfärbung — "
                          "die Ansicht steht auf Höhe statt auf einfarbigem Grau.")
        # Eine fast leere Ebene plus "Nur eingefaerbte Punkte" ergibt eine leere
        # Ansicht — und die sieht aus, als sei das Modell weg. Das darf nie
        # passieren, also lieber den Haken loesen und es sagen.
        if (self._valid is not None and self._chk_only_colored.isChecked()
                and float(self._valid.mean()) < 0.01):
            self._loading_ui = True
            self._chk_only_colored.setChecked(False)
            self._loading_ui = False
            self._log(f"Diese Farbquelle hat nur {100.0 * self._valid.mean():.1f} % "
                      f"eingefärbte Punkte — 'Nur eingefärbte Punkte' wurde gelöst, "
                      f"sonst bliebe die Ansicht leer.")
        self._cloud_view.set_cloud(
            self._world, self._colors,
            self._rec.intensity if self._rec is not None else None, self._valid)
        self._push_display_settings()

    # =============================================================== Menü

    def _fill_view_menus(self) -> None:
        """Die drei Auswahl-Untermenues fuellen und mit der Sidebar gleichziehen."""
        self._fill_layer_menu()
        menubar_mod.fill_radio_menu(
            self._actions["menu_hintergrund"], self,
            [(self._combo_bg.itemData(i), self._combo_bg.itemText(i))
             for i in range(self._combo_bg.count())],
            self._on_menu_background, self._combo_bg.currentData())
        menubar_mod.fill_radio_menu(
            self._actions["menu_tab"], self,
            [(i, self._tabs.tabText(i)) for i in range(self._tabs.count())],
            self._tabs.setCurrentIndex, self._tabs.currentIndex())
        self._actions["edl"].setChecked(self._chk_edl.isChecked())
        self._actions["edl"].setEnabled(self._chk_edl.isEnabled())
        self._actions["sidebar"].setChecked(self._sidebar_scroll.isVisible())

    def _sync_menu_state(self) -> None:
        """Haken im Menue an die Seitenleiste angleichen (ohne Rueckkopplung)."""
        for key, combo, handler in (("menu_farbquelle", self._combo_colormode, None),
                                    ("menu_hintergrund", self._combo_bg, None)):
            menu = self._actions.get(key)
            if menu is None:
                continue
            for act in menu.actions():
                act.setChecked(act.data() == combo.currentData())
        edl = self._actions.get("edl")
        if edl is not None:
            edl.blockSignals(True)
            edl.setChecked(self._chk_edl.isChecked())
            edl.blockSignals(False)

    def _on_menu_colormode(self, data) -> None:
        idx = self._combo_colormode.findData(data)
        if idx >= 0:
            self._combo_colormode.setCurrentIndex(idx)

    def _on_menu_background(self, data) -> None:
        idx = self._combo_bg.findData(data)
        if idx >= 0:
            self._combo_bg.setCurrentIndex(idx)

    def _on_menu_edl(self, on: bool) -> None:
        if self._chk_edl.isEnabled():
            self._chk_edl.setChecked(bool(on))

    def _set_sidebar_visible(self, on: bool) -> None:
        self._sidebar_scroll.setVisible(bool(on))
        act = self._actions.get("sidebar") if hasattr(self, "_actions") else None
        if act is not None:
            act.blockSignals(True)
            act.setChecked(bool(on))
            act.blockSignals(False)

    def _on_toggle_sidebar(self) -> None:
        self._set_sidebar_visible(not self._sidebar_scroll.isVisible())
        self._save_settings()

    def _on_expand_all(self) -> None:
        self._sections.set_all(True)
        self._save_settings()

    def _on_collapse_all(self) -> None:
        self._sections.set_all(False)
        self._save_settings()

    def _on_toggle_measure(self) -> None:
        an = not self._cloud_view.measure_enabled()
        self._cloud_view.set_measure(an)
        act = self._actions.get("measure")
        if act is not None:
            act.blockSignals(True)
            act.setChecked(an)
            act.blockSignals(False)
        self._tabs.setCurrentIndex(0)
        self._status_lbl.setText(
            "Messen: zwei Klicks in die Wolke setzen die Marken (Esc verwirft)."
            if an else "Bereit")
        if not an:
            self._mess_lbl.setText("")

    def _on_measured(self, a, b) -> None:
        """Auslese unter der 3D-Ansicht; Werte in Originalkoordinaten."""
        if a is None:
            self._mess_lbl.setText("")
            return
        if b is None:
            self._mess_lbl.setText(
                f"A = ({a[0]:.3f}  {a[1]:.3f}  {a[2]:.3f}) m — zweiten Punkt wählen")
            return
        d = np.asarray(b) - np.asarray(a)
        strecke = float(np.linalg.norm(d))
        waagerecht = float(np.linalg.norm(d[:2]))
        self._mess_lbl.setText(
            f"A = ({a[0]:.3f}  {a[1]:.3f}  {a[2]:.3f})    "
            f"B = ({b[0]:.3f}  {b[1]:.3f}  {b[2]:.3f})    "
            f"Abstand {strecke:.3f} m    waagerecht {waagerecht:.3f} m    "
            f"Höhe {d[2]:+.3f} m    ΔX {d[0]:+.3f}  ΔY {d[1]:+.3f}  ΔZ {d[2]:+.3f}")
        self._log(f"Messung: {strecke:.3f} m (waagerecht {waagerecht:.3f} m, "
                  f"Höhe {d[2]:+.3f} m)")

    def _on_toggle_preview(self) -> None:
        an = not self._cloud_view.preview_visible()
        self._cloud_view.set_preview_visible(an)
        self._sync_preview_action()

    def _sync_preview_action(self) -> None:
        act = self._actions.get("preview") if hasattr(self, "_actions") else None
        if act is None:
            return
        act.blockSignals(True)
        act.setChecked(self._cloud_view.preview_visible())
        act.setEnabled(self._cloud_view.has_preview())
        act.blockSignals(False)

    def _on_reset_camera(self) -> None:
        self._cloud_view.reset_camera()

    def _on_cut_reset(self) -> None:
        self._cloud_view.cut_bar.reset()

    def _on_screenshot(self) -> None:
        start = os.path.join(self._project.dir if self._project else "",
                             "ansicht.png")
        path, _ = QFileDialog.getSaveFileName(
            self, "Ansicht speichern", start, "PNG-Datei (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        try:
            self._cloud_view.screenshot(path)
        except RuntimeError as exc:
            self._show_error("Screenshot", str(exc))
            return
        self._log(f"Ansicht gespeichert: {path}")

    def _on_extrinsic_reset(self) -> None:
        self._spins_from_extrinsic(np.eye(4))
        self._log("Extrinsik auf Identität zurückgesetzt.")

    def _on_settings_reset(self) -> None:
        if QMessageBox.question(
                self, "Einstellungen zurücksetzen",
                "Alle Einstellungen dieses Projekts auf die Vorgabe setzen?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        self._settings = dict(_DEFAULT_SETTINGS)
        self._apply_settings_to_widgets()
        self._save_settings()
        self._log("Einstellungen auf Vorgabe gesetzt.")

    def _on_about(self) -> None:
        QMessageBox.about(
            self, "Über Super360 Studio",
            "<b>Super360 Studio</b><br><br>"
            "Rosbag → FAST-LIO2-Punktwolke, 360°-Video, Einfärbung, "
            "Zusammenführen und Export.<br>"
            "Bis 2026-09 hieß das Programm „RosBag Suite 360\".<br><br>"
            "<a href='https://github.com/LenaKremer98/Super360Studio'>"
            "github.com/LenaKremer98/Super360Studio</a>")

    # ========================================================= Zusammenführen

    def _merge_reset_state(self) -> None:
        self._merge_bag = None
        self._merge_rec = None
        self._merge_cloud = None
        self._merge_T = np.eye(4)
        self._merge_center = np.zeros(3)
        self._cloud_view.set_preview_cloud(None)
        if hasattr(self, "_lbl_merge"):
            self._lbl_merge.setText("Kein zweiter Flug geladen.")
            for sp in self._spin_merge.values():
                sp.blockSignals(True)
                sp.setValue(0.0)
                sp.blockSignals(False)

    def _merge_T_from_spins(self) -> np.ndarray:
        """Handjustage: um den Schwerpunkt der zweiten Wolke gieren, dann schieben."""
        yaw = np.radians(float(self._spin_merge["yaw"].value()))
        R = np.eye(4)
        R[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
        hin = np.eye(4)
        hin[:3, 3] = self._merge_center
        weg = np.eye(4)
        weg[:3, 3] = -self._merge_center
        T = hin @ R @ weg
        T[:3, 3] += [float(self._spin_merge[k].value()) for k in ("x", "y", "z")]
        return T

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
        config = self._combo_config.currentData()
        rate = float(self._spin_rate.value())
        out_dir = project_b.recording_dir()

        def job(progress_cb, cancel, log_cb):
            from core.fastlio_runner import FastLioRunner
            runner = FastLioRunner()
            return runner.run(path, out_dir, config=config, rate=rate,
                              progress_cb=progress_cb, cancel=cancel, log_cb=log_cb)

        def fertig(result) -> None:
            self._log(f"Karte für den zweiten Flug fertig: {result.n_scans} Scans, "
                      f"{_fmt_int(result.n_points)} Punkte.")
            self._merge_load_second(path, project_b)

        self._start_worker(
            f"FAST-LIO2 für den zweiten Flug ({os.path.basename(path)}) …",
            job, fertig)

    def _merge_load_second(self, path: str, project_b) -> None:
        rec_dir = project_b.recording_dir()

        def job(progress_cb, cancel, log_cb):
            from core import merge as merge_mod
            progress_cb(0.1, "Lade zweite Aufzeichnung …")
            rec_b = Recording.load(rec_dir, bag_path=path)
            note = rec_b.level_note()
            if note:
                log_cb(f"Zweiter Flug — {note}")
            progress_cb(0.6, "Dünne für die Vorschau aus …")
            wolke = merge_mod.cloud_for_registration(rec_b)
            return {"rec": rec_b, "cloud": wolke, "path": path,
                    "proxy": ThreadLocalBag(path)}

        self._start_worker("Lade zweiten Flug …", job, self._on_merge_loaded)

    def _on_merge_loaded(self, res: dict) -> None:
        self._merge_rec = res["rec"]
        self._merge_bag = res["proxy"]
        self._merge_cloud = res["cloud"]
        self._merge_center = (self._merge_cloud.mean(axis=0)
                              if len(self._merge_cloud) else np.zeros(3))
        self._merge_T = np.eye(4)
        for sp in self._spin_merge.values():
            sp.blockSignals(True)
            sp.setValue(0.0)
            sp.blockSignals(False)
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
        self._merge_refresh_preview()
        for sp in self._spin_merge.values():   # Handfelder gelten jetzt nicht mehr
            sp.blockSignals(True)
            sp.setValue(0.0)
            sp.blockSignals(False)
        fit, rmse = res["fitness"], res["rmse"]
        self._lbl_merge.setText(
            f"Ausgerichtet über '{res['kandidat']}': Trefferquote {fit:.2f}, "
            f"Restfehler {rmse:.3f} m.")
        self._log(f"Ausrichtung: Kandidat '{res['kandidat']}', Trefferquote "
                  f"{fit:.2f}, Restfehler {rmse:.3f} m.")
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

        def job(progress_cb, cancel, log_cb):
            from core import merge as merge_mod
            meta = merge_mod.merge_recordings(
                rec_a, rec_b, T, out_dir, bag_a, bag_b,
                info={"fitness": None}, progress_cb=progress_cb, cancel=cancel)
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
        if self._merge_rec is not None:
            weiter = QMessageBox.question(
                self, "Zweiter Flug nicht übernommen",
                "Es ist ein zweiter Flug geladen (die orangen Punkte), aber "
                "noch nicht übernommen.\n\nEingefärbt wird nur der offene "
                "Flug; die orange Vorschau bleibt unverändert liegen und "
                "verdeckt das Ergebnis.\n\nTrotzdem einfärben?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if weiter != QMessageBox.Yes:
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
        parts = self._parts
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
                                      progress_cb=progress_cb, cancel=cancel,
                                      parts=parts)

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
        # Bei einer zusammengefuehrten Karte je Abschnitt ausweisen: sonst
        # sieht man nur eine Gesamtquote und merkt nicht, dass ein ganzer Flug
        # leer geblieben ist.
        colors, valid, err = _load_color_files(self._project.colors_dir(),
                                               self._rec.n_points)
        if err:
            self._show_error("Einfärben", err)
            return
        if self._parts and valid is not None:
            for teil, (proxy, von, bis) in enumerate(self._parts, start=1):
                a = int(self._rec.offsets[von])
                b = int(self._rec.offsets[min(bis, self._rec.n_scans)])
                if b <= a:
                    continue
                anteil = float(valid[a:b].mean())
                self._log(f"   Abschnitt {teil} "
                          f"({os.path.basename(proxy.bag_path)}): "
                          f"{100.0 * anteil:.1f} % von {_fmt_int(b - a)} Punkten.")
                if anteil < 0.02:
                    self._log(f"   WARNUNG: Abschnitt {teil} ist praktisch leer "
                              f"geblieben — vermutlich fehlt für dieses Bag die "
                              f"Kamera oder es liegt nicht mehr an seinem Ort.")
        self._colors, self._valid = colors, valid
        # Eine leere Anzeige ist der schlechteste Ausgang: bei "Nur eingefärbte
        # Punkte" verschwindet die ganze Wolke, und uebrig bleibt nur, was sonst
        # noch im Bild ist. Lieber den Haken loesen und es sagen.
        if n_valid == 0 and self._chk_only_colored.isChecked():
            self._loading_ui = True
            self._chk_only_colored.setChecked(False)
            self._loading_ui = False
            self._log("Kein Punkt wurde eingefärbt — 'Nur eingefärbte Punkte' "
                      "wurde gelöst, sonst bliebe die Ansicht leer.")
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
