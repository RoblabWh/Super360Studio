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

import os
import sys
import tempfile
from typing import Optional

import numpy as np

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QFormLayout, QHBoxLayout, QLabel, QMainWindow, QPlainTextEdit, QProgressBar,
    QPushButton, QScrollArea, QSplitter, QTabWidget, QVBoxLayout, QWidget,
    QApplication,
)

try:
    from core.bag_reader import BagInfo, ThreadLocalBag
    from core.kalibrierung import default_calib as _default_calib
    from core.project import Project
    from core.recording import Recording
    from core.rviz_player import RvizPlayer
except ImportError:  # direkter Skript-Start: Paketwurzel nachrüsten
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.bag_reader import BagInfo, ThreadLocalBag
    from core.kalibrierung import default_calib as _default_calib
    from core.project import Project
    from core.recording import Recording
    from core.rviz_player import RvizPlayer

from ui.bausteine import _ImageDialog, einmal_timer
from ui.cloud_view import CloudView
from ui.collapsible import SectionStack
from ui.explorationsgrad import ExplorationsgradAnzeige
from ui import menubar as menubar_mod
from ui.gps_panel import GpsPanel
from ui.jobs import Worker
from ui.pano_view import PanoView, StitchingPanoSource
from ui.fenster.einstellungen import _DEFAULT_SETTINGS

from ui.fenster.grundgeruest import GrundgeruestMixin
from ui.fenster.einstellungen import EinstellungenMixin
from ui.fenster.projekt import ProjektMixin
from ui.fenster.zusammenfuehren import ZusammenfuehrenMixin
from ui.fenster.kalibrierung import KalibrierungMixin
from ui.fenster.einfaerbung import EinfaerbungMixin
from ui.fenster.maeander import MaeanderMixin
from ui.fenster.maeander_justage import MaeanderJustageMixin
from ui.fenster.splat import SplatMixin
from ui.fenster.mesh import MeshMixin
from ui.fenster.export import ExportMixin
from ui.fenster.anzeige import AnzeigeMixin
from ui.fenster.wiedergabe import WiedergabeMixin
from ui.fenster.autotest import AutotestMixin


class MainWindow(GrundgeruestMixin,
                 EinstellungenMixin,
                 ProjektMixin,
                 ZusammenfuehrenMixin,
                 KalibrierungMixin,
                 EinfaerbungMixin,
                 MaeanderMixin,
                 MaeanderJustageMixin,
                 SplatMixin,
                 MeshMixin,
                 ExportMixin,
                 AnzeigeMixin,
                 WiedergabeMixin,
                 AutotestMixin,
                 QMainWindow):
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
        # Lage nach der letzten Ausrichtung; die Handregler wirken obendrauf
        self._merge_T_basis = np.eye(4)
        self._merge_center = np.zeros(3)
        # Kippausgleich des zweiten Flugs (4x4) und sein Winkel, s. core.merge
        self._merge_T_kipp = np.eye(4)
        self._merge_kipp_grad: Optional[float] = None
        self._parts: Optional[list] = None
        # Farbebenen: Schluessel -> (rgb, maske). 'onboard' kommt aus der
        # 360-Kamera, die beiden anderen aus dem Maeanderflug.
        self._layers: dict = {}
        self._layer_key = "onboard"
        # Farbmodus und Punktgroesse der 3D-Ansicht
        self._color_mode = "rgb"
        self._point_size = 2.0
        self._meander_pipe = None
        self._meander_dir: Optional[str] = None
        # Ausrichtfenster des Maeanderflugs, solange es offen ist (s. _on_meander_fenster)
        self._meander_fenster = None
        # Verkleinerte Bilder fuer die Farbvorschau im Ausrichtfenster: ein
        # Durchlauf kostet damit Millisekunden statt Minuten.
        self._live = None
        self._live_th = None           # dasselbe fuer die Thermalbilder
        # Optik der Maeanderkameras: Massstab je Optik, Thermal-Einmessung
        self._optik: dict = {"rgb_faktor": 1.0, "thermal_faktor": 1.0, "thermal": None}
        # Thermal-Zuschlag auf die RGB-Lage: (Gier Grad, X m, Y m)
        self._th_zuschlag = (0.0, 0.0, 0.0)
        # Einstellung meander_solo: wird gelesen und gespeichert, wirkt nicht
        self._meander_solo = False
        # Temperatur je Punkt aus der Thermal-Mäanderebene (°C, NaN wo keine)
        self._temperatur: Optional[np.ndarray] = None
        self._temperaturen: dict = {}      # Thermalebene -> Temperatur je Punkt
        self._temperatur_anzeigen = True
        # "Automatisch bis zur Farbe": die Schritte stossen einander an
        self._auto_kette = False
        self._optik_neu_messen = False   # frisch ausgerichtet: Hoehe gilt nicht mehr
        # Schieber am zweiten Flug: erst nach kurzer Ruhe neu transformieren
        self._merge_timer = einmal_timer(self, 60, self._on_merge_manual)
        self._n_frames = 0
        self._project: Optional[Project] = None
        self._settings: dict = dict(_DEFAULT_SETTINGS)
        self._rec: Optional[Recording] = None
        self._world: Optional[np.ndarray] = None
        self._colors: Optional[np.ndarray] = None
        self._valid: Optional[np.ndarray] = None
        self._fixes: Optional[list] = None
        self._quality = None
        # Explorationsgrad des offenen Bags (core.exploration.Explorationsgrad)
        self._exploration = None
        self._georef = None
        self._pano_src: Optional[StitchingPanoSource] = None
        self._worker: Optional[Worker] = None
        self._retired: list[Worker] = []
        # Beenden und Wiederholen der RViz-Wiedergabe laufen in einem eigenen
        # Arbeiter neben dem Schritt (s. _rviz_job)
        self._rviz_worker: Optional[Worker] = None
        self._busy = False
        self._loading_ui = False
        self._closing = False
        self._pano_failed = False
        # Mesh der offenen Punktwolke (core/mesh.py): Geometrie einmal je
        # Aufzeichnung, Eckfarben je Ebene zwischengespeichert
        self._mesh_geom: Optional[dict] = None
        self._mesh_geom_dir: Optional[str] = None
        self._mesh_farbcache: dict = {}
        self._mesh_wartet = False
        self._mesh_laeuft = False
        self._mesh_timer = einmal_timer(self, 1500)
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
        # ------------------------------------------------- Aktionen und Menü
        # Zuerst: so steht jede Aktion schon, wenn die Seitenleiste entsteht.
        self._actions = menubar_mod.baue(self)
        # Explorationsgrad ganz oben rechts: in der Ecke der Menueleiste steht
        # er ueber allen Bereichen und bleibt beim Tabwechsel stehen.
        self._grad_anzeige = ExplorationsgradAnzeige(self)
        self._grad_anzeige.angeklickt.connect(self._on_exploration_zeigen)
        self.menuBar().setCornerWidget(self._grad_anzeige, Qt.TopRightCorner)

        # -------------------------------------------------------------- Tabs
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
        self._pano_view = PanoView(self)
        self._gps_panel = GpsPanel(self)
        self._log_edit = QPlainTextEdit(self)
        self._log_edit.setReadOnly(True)
        self._log_edit.setMaximumBlockCount(20000)
        self._tabs.addTab(karte, "3D-Karte")
        self._tabs.addTab(self._pano_view, "360°-Video")
        self._tabs.addTab(self._gps_panel, "GPS")
        self._tabs.addTab(self._log_edit, "Protokoll")

        # ------------------------------------------------------ Seitenleiste
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
        # Abschnitte und Unterbloecke zaehlen mit, sonst haengt die Breite davon
        # ab, was beim Start zufaellig offen ist. Dazu muessen die breiteste
        # Beschriftung und das breiteste Feld der Formulare nebeneinander
        # passen, gleich in welchem Abschnitt sie stehen: sonst haengt die
        # Breite davon ab, wie die Zeilen auf die Abschnitte verteilt sind.
        need = kopf = feld = rand = 0
        for sec in self._sections.sections(mit_unterbloecken=True):
            need = max(need, sec.content().sizeHint().width())
            for form in sec.content().findChildren(QFormLayout):
                m = form.contentsMargins()
                rand = max(rand, m.left() + max(0, form.horizontalSpacing()) + m.right())
                for zeile in range(form.rowCount()):
                    k = form.itemAt(zeile, QFormLayout.LabelRole)
                    f = form.itemAt(zeile, QFormLayout.FieldRole)
                    if k is not None and f is not None:
                        kopf = max(kopf, k.sizeHint().width())
                        feld = max(feld, f.sizeHint().width())
        need = max(need, kopf + feld + rand)
        need += self._sidebar_scroll.verticalScrollBar().sizeHint().width() + 26
        self._sidebar_breite = max(400, min(720, need))
        self._sidebar_scroll.setMinimumWidth(300)
        self._sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
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
        # Die Fokuskette folgt der Bauordnung, darin stehen die Reiter vorn;
        # den Tastaturfokus beim Start bekommt trotzdem die Seitenleiste.
        self._sidebar_scroll.setFocus()

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

        # ------------------------------------------------- Signalverdrahtung
        # Menue mit der Seitenleiste gleichziehen; der Haken unter
        # Ansicht ▸ Bereich folgt dem gezeigten Reiter.
        self._fill_view_menus()
        self._refresh_layer_combo()   # ohne Projekt: 'keine Einfärbung'
        self._sync_preview_action()
        self._tabs.currentChanged.connect(self._sync_tab_menu)
        self._cloud_view.measured.connect(self._on_measured)
        # Leiste ueber der 3D-Ansicht: gleiche Wirkung wie Seitenleiste und Menue
        self._cloud_view.farbmodus_gewaehlt.connect(self._on_farbleiste)
        self._cloud_view.punktgroesse_geaendert.connect(self._on_leiste_punktgroesse)
        self._cloud_view.messen_angefordert.connect(self._on_toggle_measure)
        self._cloud_view.temperatur_umgeschaltet.connect(self._on_leiste_temperatur)
        self._cloud_view.mesh_umgeschaltet.connect(self._on_leiste_mesh)
        self._mesh_timer.timeout.connect(self._mesh_timer_abgelaufen)
        self._pano_view.frameChanged.connect(self._on_pano_frame)
        self._gps_panel.georefReady.connect(self._on_georef_ready)

        # -------------------------------------------------------- Timerstart
        # Die RViz-Knoepfe folgen der Prozesslage erst, wenn alles steht.
        self._rviz_timer.start(700)

    # Reihenfolge der Seitenleiste: der Arbeitsablauf von oben nach unten,
    # nicht die Reihenfolge, in der die Teile entstanden sind. Nummern tragen
    # nur die Schritte des Ablaufs; Anzeige und Wiedergabe folgen ohne.
    _SECTIONS = (
        ("aufnahme", "1 · Aufnahme", "_abschnitt_aufnahme", True),
        ("karte", "2 · Karte (FAST-LIO2)", "_abschnitt_karte", True),
        ("extrinsik", "3 · Kamera-Kalibrierung", "_abschnitt_extrinsik", False),
        ("zusammen", "4 · Zusammenführen (optional)", "_abschnitt_zusammen", False),
        ("einfaerbung", "5 · Einfärbung (360°-Kamera)", "_abschnitt_einfaerbung", True),
        ("maeander", "6 · Mäander-Einfärbung", "_abschnitt_maeander", False),
        ("splat", "7 · Gaussian Splat und Fusion", "_abschnitt_splat", False),
        ("mesh", "8 · Mesh", "_abschnitt_mesh", False),
        ("export", "9 · Export", "_abschnitt_export", True),
        ("anzeige", "Anzeige", "_abschnitt_anzeige", True),
        ("wiedergabe", "Wiedergabe (RViz)", "_abschnitt_wiedergabe", False),
    )

    def _build_sidebar(self) -> QWidget:
        self._sections = SectionStack()
        for key, titel, bauen, offen in self._SECTIONS:
            self._sections.add(key, titel, getattr(self, bauen)(), offen)
        self._sections.finish()
        self._sections.toggled.connect(lambda *_: self._on_setting_changed())
        return self._sections

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
        if self._rviz_worker is not None:
            # Beenden oder Wiederholen der RViz-Wiedergabe läuft noch; erst
            # danach darf der Player unten geräumt werden.
            self._status_lbl.setText("Warte auf die RViz-Wiedergabe …")
            self.statusBar().repaint()
            self._rviz_worker.wait()
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
