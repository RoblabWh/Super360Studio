"""CloudView: VTK point cloud viewer widget for Super360 Studio.

Renders large point clouds (tested up to 9M points) via vtkPolyData with a
direct vertex cell array (no vtkVertexGlyphFilter). Supports color modes
rgb/hoehe/intensitaet/uniform, a valid mask filter ("nur eingefaerbte
Punkte"), display voxel downsampling, a trajectory polyline, EDL shading
(with graceful fallback) and screenshots. All calls GUI-thread only.

Am rechten Rand liegt der Hoehenschnitt (:class:`CutBar`): zwei Griffe
spannen die sichtbare Schicht auf, so laesst sich das Dach abnehmen und in
ein Gebaeude hineinschauen. Geschnitten wird ueber vtkPlane am Mapper, also
auf der Grafikkarte — die Geometrie wird dabei nicht neu aufgebaut, das
Ziehen bleibt auch bei Millionen Punkten fluessig. Die Trajektorie haengt an
einem eigenen Mapper und bleibt ungeschnitten sichtbar.

**Maus wie im VS-Code-Punktwolken-Viewer** (``PointCloudMerger/vscode-
pointcloud-viewer``): links ziehen dreht um den Zielpunkt (Gier und Nick, Z
oben, 0,006 rad je Pixel), rechts ziehen oder Umschalt/Strg und ziehen
verschiebt in der Bildebene, das Mausrad zoomt mit ``exp(0,0012 * Delta)``,
Blickwinkel 50°. Ein Klick beim Messen ist ein Klick, solange die Maus dabei
unter 5 Pixeln bleibt — gedreht werden kann also auch waehrend des Messens.

**Leiste oben** wie dort: Farbe, Punktgroesse (in Vierteln), Messen,
Temperatur anzeigen, Ansicht zuruecksetzen. Mit Temperatur zeigt die Maus
ueber einem Punkt dessen Temperatur — in jedem Farbmodus.
"""
from __future__ import annotations

import math

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

try:  # pin the Qt binding before the interactor module is imported
    import vtkmodules.qt as _vtk_qt

    if _vtk_qt.PyQtImpl is None:
        _vtk_qt.PyQtImpl = "PyQt5"
except ImportError:
    pass
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

_ACCENT = (0x4F / 255.0, 0xC3 / 255.0, 0xF7 / 255.0)  # #4FC3F7
_BG = {"dunkel": (0.102, 0.110, 0.125), "hell": (0.93, 0.94, 0.955)}
_UNIFORM_COLOR = {"dunkel": (0.80, 0.82, 0.85), "hell": (0.22, 0.25, 0.28)}
_COLOR_MODES = ("rgb", "hoehe", "intensitaet", "uniform")
_ID_DTYPE = np.int64 if vtk.vtkIdTypeArray().GetDataTypeSize() == 8 else np.int32

_BLICKWINKEL = 50.0   # Grad, wie im VS-Code-Viewer
_DREH = 0.006         # rad je Pixel
_ZOOM = 0.0012        # je Browser-Pixel Mausrad; eine Raste sind dort 100
_NICK_MAX = 1.553     # knapp unter 90°, sonst kippt die Hochachse
_HOVER_PUNKTE = 1_500_000   # so viele Punkte mit Temperatur prueft das Hovern

_CUT_W = 62        # px Gesamtbreite der Leiste
_CUT_TRACK_W = 10  # px Breite der Schiene
_CUT_PAD = 20      # px oben/unten fuer die Beschriftung
_CUT_GRIP = 7      # px halbe Hoehe eines Griffs


def _turbo_lut() -> np.ndarray:
    """256x3 uint8 turbo colormap (polynomial approximation, matplotlib-free)."""
    t = np.linspace(0.0, 1.0, 256)
    r = 34.61 + t * (1172.33 - t * (10793.56 - t * (33300.12 - t * (38394.49 - t * 14825.05))))
    g = 23.31 + t * (557.33 + t * (1225.33 - t * (3574.96 - t * (1073.77 + t * 707.56))))
    b = 27.2 + t * (3211.1 - t * (15327.97 - t * (27814.0 - t * (22569.18 - t * 6838.66))))
    return np.clip(np.stack([r, g, b], axis=1), 0.0, 255.0).astype(np.uint8)


_TURBO = _turbo_lut()


def _scalar_to_rgb(vals: np.ndarray, lut: np.ndarray | None) -> np.ndarray:
    """Percentile-scale (2..98) scalars to uint8 RGB; lut=None gives greys."""
    if vals.size == 0:
        return np.empty((0, 3), np.uint8)
    vals = vals.astype(np.float32, copy=False)
    lo, hi = np.percentile(vals, [2.0, 98.0])
    if hi - lo < 1e-9:
        hi = lo + 1e-9
    idx = (np.clip((vals - lo) / (hi - lo), 0.0, 1.0) * 255.0).astype(np.uint8)
    if lut is None:
        return np.repeat(idx[:, None], 3, axis=1)
    return lut[idx]


def _strip_numpy_ref(vtk_arr):
    """numpy-Referenz vom VTK-Wrapper entfernen (Lebensdauer regeln wir selbst).

    numpy_to_vtk(deep=False) haengt das numpy-Array als ``_numpy_reference`` an
    den Python-Wrapper. Stirbt der Wrapper, waehrend das C++-Objekt weiterlebt,
    "ghostet" VTK den Wrapper samt Dict — die numpy-Puffer bleiben dann auch
    nach set_cloud(None) unbegrenzt gepinnt (empirisch auf VTK 9.1 verifiziert).
    CloudView haelt die Puffer selbst in ``_vtk_refs``; die Wrapper-Referenz ist
    daher redundant und wird geloescht, damit set_cloud(None) wirklich freigibt.
    """
    try:
        del vtk_arr._numpy_reference
    except AttributeError:
        pass
    return vtk_arr


def _voxel_first_indices(pts: np.ndarray, voxel: float) -> np.ndarray:
    """Indices of the first point per occupied voxel (numpy grid hash)."""
    if len(pts) == 0:
        return np.empty(0, np.int64)
    g = np.floor(pts.astype(np.float64) / voxel).astype(np.int64)
    g -= g.min(axis=0)
    dims = g.max(axis=0) + 1
    if float(dims[0]) * float(dims[1]) * float(dims[2]) < 2**62:
        key = (g[:, 0] * dims[1] + g[:, 1]) * dims[2] + g[:, 2]
    else:  # degenerate extent: fall back to xor hash (collisions tolerable for display)
        key = (g[:, 0] * 73856093) ^ (g[:, 1] * 19349663) ^ (g[:, 2] * 83492791)
    _, first = np.unique(key, return_index=True)
    return np.sort(first)


class CutBar(QtWidgets.QWidget):
    """Senkrechte Leiste mit zwei Griffen: die sichtbare Hoehenschicht.

    Vorbild ist die Leiste des VS-Code-Punktwolken-Viewers aus PointCloudMerger.
    Der obere Griff setzt die obere Schnittebene, der untere die untere; die
    Werte stehen in Metern in den Koordinaten der Wolke. Ziehen bewegt einen
    Griff, ein Klick auf die Schiene holt den naeheren Griff dorthin, das
    Mausrad schiebt die ganze Schicht.
    """

    changed = QtCore.pyqtSignal(float, float)  # untere, obere Ebene in Metern

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setFixedWidth(_CUT_W)
        self.setCursor(QtCore.Qt.SizeVerCursor)
        self.setToolTip(
            "Höhenschnitt: Griffe ziehen, auf die Schiene klicken holt den\n"
            "näheren Griff, Mausrad verschiebt die ganze Schicht.")
        self._min = 0.0
        self._max = 1.0
        self._lo = 0.0   # Anteil 0..1, unten
        self._hi = 1.0   # Anteil 0..1, oben
        self._drag: str | None = None
        self._enabled = False

    # ------------------------------------------------------------- Zustand

    def set_range(self, zmin: float, zmax: float) -> None:
        """Hoehenbereich der Wolke setzen; der Schnitt geht dabei auf ganz auf."""
        if not np.isfinite(zmin) or not np.isfinite(zmax) or zmax - zmin < 1e-6:
            self._enabled = False
            self._min, self._max = 0.0, 1.0
        else:
            self._enabled = True
            self._min, self._max = float(zmin), float(zmax)
        self._lo, self._hi = 0.0, 1.0
        self.update()
        self.changed.emit(*self.planes())

    def planes(self) -> tuple[float, float]:
        """Aktuelle Schnittebenen in Metern (untere, obere)."""
        span = self._max - self._min
        return self._min + self._lo * span, self._min + self._hi * span

    def is_cut(self) -> bool:
        return self._enabled and (self._lo > 0.0 or self._hi < 1.0)

    def reset(self) -> None:
        if self._lo == 0.0 and self._hi == 1.0:
            return
        self._lo, self._hi = 0.0, 1.0
        self.update()
        self.changed.emit(*self.planes())

    # -------------------------------------------------------------- Geometrie

    def _track(self) -> QtCore.QRect:
        x = (self.width() - _CUT_TRACK_W) // 2
        return QtCore.QRect(x, _CUT_PAD, _CUT_TRACK_W,
                            max(self.height() - 2 * _CUT_PAD, 1))

    def _y(self, frac: float) -> int:
        """Anteil -> y (0 unten, 1 oben)."""
        tr = self._track()
        return int(round(tr.bottom() - frac * (tr.height() - 1)))

    def _frac(self, y: int) -> float:
        tr = self._track()
        return float(np.clip((tr.bottom() - y) / max(tr.height() - 1, 1), 0.0, 1.0))

    # ---------------------------------------------------------------- Malen

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt)
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        tr = self._track()
        an = QtGui.QColor(*[int(c * 255) for c in _ACCENT])
        if not self._enabled:
            an.setAlpha(70)
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor(255, 255, 255, 28))
        p.drawRoundedRect(tr, 5, 5)
        y_lo, y_hi = self._y(self._lo), self._y(self._hi)
        band = QtCore.QRect(tr.left(), y_hi, tr.width(), max(y_lo - y_hi, 1))
        p.setBrush(an)
        p.drawRoundedRect(band, 5, 5)
        gw = self.width() - 12
        for y in (y_hi, y_lo):
            g = QtCore.QRect(6, y - _CUT_GRIP // 2, gw, _CUT_GRIP)
            p.setBrush(QtGui.QColor(240, 244, 250) if self._enabled
                       else QtGui.QColor(150, 155, 165))
            p.drawRoundedRect(g, 3, 3)
        lo_m, hi_m = self.planes()
        f = p.font()
        f.setPointSizeF(max(f.pointSizeF() - 1.5, 6.5))
        p.setFont(f)
        p.setPen(QtGui.QColor(215, 220, 230) if self._enabled
                 else QtGui.QColor(130, 135, 145))
        txt = ("—", "—") if not self._enabled else (f"{hi_m:.1f} m", f"{lo_m:.1f} m")
        p.drawText(QtCore.QRect(0, 2, self.width(), _CUT_PAD - 4),
                   QtCore.Qt.AlignCenter, txt[0])
        p.drawText(QtCore.QRect(0, self.height() - _CUT_PAD + 2, self.width(),
                                _CUT_PAD - 4), QtCore.Qt.AlignCenter, txt[1])
        p.end()

    # ---------------------------------------------------------------- Maus

    def _nearest(self, y: int) -> str:
        return "hi" if abs(y - self._y(self._hi)) <= abs(y - self._y(self._lo)) else "lo"

    def _move_to(self, which: str, y: int) -> None:
        f = self._frac(y)
        if which == "hi":
            self._hi = max(f, self._lo)
        else:
            self._lo = min(f, self._hi)
        self.update()
        self.changed.emit(*self.planes())

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt)
        if not self._enabled or event.button() != QtCore.Qt.LeftButton:
            return
        self._drag = self._nearest(event.y())
        self._move_to(self._drag, event.y())

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 (Qt)
        if self._drag is not None:
            self._move_to(self._drag, event.y())

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 (Qt)
        self._drag = None

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 (Qt)
        self.reset()

    def wheelEvent(self, event) -> None:  # noqa: N802 (Qt)
        """Die ganze Schicht verschieben, Dicke bleibt."""
        if not self._enabled:
            return
        step = (event.angleDelta().y() / 120.0) * 0.02
        d = self._hi - self._lo
        lo = float(np.clip(self._lo + step, 0.0, 1.0 - d))
        self._lo, self._hi = lo, lo + d
        self.update()
        self.changed.emit(*self.planes())
        event.accept()


class CloudView(QtWidgets.QWidget):
    """VTK-Punktwolken-Viewer (Trackball-Kamera, EDL optional)."""

    #: (A, B) in Weltkoordinaten; B ist None, solange nur A gesetzt ist
    measured = QtCore.pyqtSignal(object, object)
    #: Leiste: Farbmodus gewaehlt (Schluessel, s. set_farbmodi)
    farbmodus_gewaehlt = QtCore.pyqtSignal(str)
    #: Leiste: Punktgroesse in Pixeln (Viertelschritte)
    punktgroesse_geaendert = QtCore.pyqtSignal(float)
    #: Leiste oder Taste: Messen soll umschalten
    messen_angefordert = QtCore.pyqtSignal()
    #: Leiste: Temperatur beim Hovern zeigen an/aus
    temperatur_umgeschaltet = QtCore.pyqtSignal(bool)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        aussen = QtWidgets.QVBoxLayout(self)
        aussen.setContentsMargins(0, 0, 0, 0)
        aussen.setSpacing(0)
        aussen.addWidget(self._leiste_bauen())
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        aussen.addLayout(layout, 1)
        self._vtkw = QVTKRenderWindowInteractor(self)
        self._vtkw.setMouseTracking(True)          # fuer die Temperatur beim Hovern
        self._vtkw.setFocusPolicy(QtCore.Qt.StrongFocus)
        self._vtkw.setCursor(QtCore.Qt.OpenHandCursor)
        layout.addWidget(self._vtkw, 1)

        side = QtWidgets.QVBoxLayout()
        side.setContentsMargins(2, 4, 4, 4)
        side.setSpacing(4)
        self.cut_bar = CutBar(self)
        side.addWidget(self.cut_bar, 1)
        self._cut_reset = QtWidgets.QPushButton("alles")
        self._cut_reset.setToolTip("Höhenschnitt aufheben")
        self._cut_reset.setFixedWidth(_CUT_W)
        self._cut_reset.clicked.connect(self.cut_bar.reset)
        side.addWidget(self._cut_reset, 0)
        layout.addLayout(side, 0)
        self.cut_bar.changed.connect(self._on_cut_changed)

        self._renderer = vtk.vtkRenderer()
        self._background = "dunkel"
        self._renderer.SetBackground(*_BG[self._background])
        rw = self._vtkw.GetRenderWindow()
        rw.SetMultiSamples(0)  # required for correct render-pass (EDL) output
        rw.AddRenderer(self._renderer)
        # Die Maus steuert diese Klasse selbst (wie der VS-Code-Viewer); der
        # VTK-Stil bekommt keine Maus- und Tastenereignisse mehr zu sehen.
        rw.GetInteractor().SetInteractorStyle(vtk.vtkInteractorStyleUser())
        self._renderer.GetActiveCamera().SetViewAngle(_BLICKWINKEL)
        self._kam = {"yaw": -math.pi / 4, "pitch": 0.5, "dist": 10.0,
                     "ziel": np.zeros(3), "radius": 1.0}
        self._ziehen = 0            # 0 nichts, 1 drehen, 2 verschieben
        self._maus_letzt = QtCore.QPoint()
        self._maus_start = QtCore.QPoint()
        # Temperatur beim Hovern: Werte je Punkt, Stichprobe zum Suchen
        self._temperatur: np.ndarray | None = None
        self._temp_an = True
        self._hover_pts: np.ndarray | None = None
        self._hover_temp: np.ndarray | None = None
        self._hover_pos = QtCore.QPoint()
        self._hover_timer = QtCore.QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.setInterval(45)
        self._hover_timer.timeout.connect(self._hover_zeigen)

        self._mapper = vtk.vtkPolyDataMapper()
        self._mapper.SetColorModeToDirectScalars()
        self._actor = vtk.vtkActor()
        self._actor.SetMapper(self._mapper)
        self._actor.GetProperty().SetPointSize(2)
        self._renderer.AddActor(self._actor)
        # Hoehenschnitt auf der Grafikkarte: nur am Wolken-Mapper, damit die
        # Trajektorie ungeschnitten sichtbar bleibt.
        self._plane_lo = vtk.vtkPlane()
        self._plane_lo.SetNormal(0.0, 0.0, 1.0)
        self._plane_hi = vtk.vtkPlane()
        self._plane_hi.SetNormal(0.0, 0.0, -1.0)
        self._cut_active = False
        self._path_actor: vtk.vtkActor | None = None
        # Zweite Wolke fuer die Merge-Vorschau: eigener Mapper, einfarbig,
        # damit sich beim Ausrichten sofort sehen laesst, was wohin wandert.
        self._prev_mapper = vtk.vtkPolyDataMapper()
        self._prev_actor = vtk.vtkActor()
        self._prev_actor.SetMapper(self._prev_mapper)
        # Klein und halbdurchsichtig: die Vorschau soll die Karte zeigen, auf
        # die sie gelegt wird, und sie nicht zudecken. Voll deckend war sie
        # ein oranger Teppich, unter dem von der Karte nichts mehr zu sehen war.
        self._prev_actor.GetProperty().SetPointSize(1)
        self._prev_actor.GetProperty().SetColor(1.0, 0.55, 0.20)
        self._prev_actor.GetProperty().SetOpacity(0.55)
        self._prev_actor.SetVisibility(False)
        self._prev_wanted = False   # vom Anwender gewuenscht (Ansicht-Menue)
        # Zweite, eingefaerbte Vorschau: die Stichprobe der Maeander-
        # Handjustage. Eigener Actor, damit sie sich mit der orangen
        # Merge-Vorschau nicht ins Gehege kommt. Groessere Punkte als die
        # Karte, damit sie darauf sichtbar bleibt.
        self._cprev_mapper = vtk.vtkPolyDataMapper()
        self._cprev_mapper.SetColorModeToDirectScalars()
        self._cprev_actor = vtk.vtkActor()
        self._cprev_actor.SetMapper(self._cprev_mapper)
        self._cprev_actor.GetProperty().SetPointSize(5)
        self._cprev_actor.SetVisibility(False)
        self._renderer.AddActor(self._cprev_actor)
        self._cprev_refs: list = []
        self._renderer.AddActor(self._prev_actor)
        self._prev_refs: list = []

        # EDL (eye-dome lighting) — availability exposed as attribute.
        self.edl_available: bool = False
        self._edl_pass = None
        self._edl_on = False
        try:
            steps = vtk.vtkRenderStepsPass()
            edl = vtk.vtkEDLShading()
            edl.SetDelegatePass(steps)
            if not hasattr(self._renderer, "SetPass"):
                raise AttributeError("Renderer ohne SetPass")
            self._edl_pass = edl
            self.edl_available = True
        except Exception:
            self.edl_available = False

        # original data (kept so toggles can re-filter)
        self._points: np.ndarray | None = None
        self._colors: np.ndarray | None = None
        self._intensity: np.ndarray | None = None
        self._valid: np.ndarray | None = None
        # display state
        self._color_mode = "rgb"
        self._only_colored = False
        self._voxel = 0.0
        self._poly: vtk.vtkPolyData | None = None
        self._sel: np.ndarray | None = None
        self._disp_points: np.ndarray | None = None
        self._rgb_np: np.ndarray | None = None
        self._vtk_refs: list = []  # keep numpy buffers alive for deep=False arrays
        self._initialized = False
        self._had_cloud = False
        # Messen: zwei Marken, Linie dazwischen, Abstand am Mittelpunkt
        self._measure_on = False
        self._meas: list[np.ndarray] = []
        self._meas_actors: list = []
        self._vtkw.installEventFilter(self)

    # ------------------------------------------------------------------ Leiste

    def _leiste_bauen(self) -> QtWidgets.QWidget:
        """Leiste oben wie im VS-Code-Viewer: Farbe, Punkte, Messen, Temperatur."""
        leiste = QtWidgets.QFrame(self)
        leiste.setObjectName("wolkenleiste")
        # Farben aus der Palette, nicht fest: helle Schrift waere ohne das
        # dunkle Theme auf hellem Grund unsichtbar
        leiste.setStyleSheet("#wolkenleiste { border-bottom: 1px solid rgba(128,128,128,90); }")
        lay = QtWidgets.QHBoxLayout(leiste)
        lay.setContentsMargins(10, 4, 10, 4)
        lay.setSpacing(14)

        def gruppe(*widgets):
            g = QtWidgets.QHBoxLayout()
            g.setSpacing(5)
            for w in widgets:
                g.addWidget(w)
            lay.addLayout(g)

        self._farbe = QtWidgets.QComboBox()
        self._farbe.setMinimumContentsLength(14)
        self._farbe.setToolTip("Farbmodus der Wolke")
        self._farbe.activated.connect(
            lambda i: self.farbmodus_gewaehlt.emit(str(self._farbe.itemData(i))))
        gruppe(QtWidgets.QLabel("Farbe"), self._farbe)

        self._groesse = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._groesse.setRange(2, 40)          # Viertelpixel: 0,5 .. 10 px
        self._groesse.setValue(8)
        self._groesse.setFixedWidth(110)
        self._groesse.setToolTip("Punktgröße in Pixeln, in Viertelschritten")
        self._groesse_lbl = QtWidgets.QLabel("2")
        self._groesse_lbl.setMinimumWidth(34)
        self._groesse.valueChanged.connect(self._groesse_gezogen)
        gruppe(QtWidgets.QLabel("Punkte"), self._groesse, self._groesse_lbl)

        self._btn_messen = QtWidgets.QPushButton("Messen")
        self._btn_messen.setCheckable(True)
        self._btn_messen.setToolTip("zwei Punkte anklicken (M) — Esc verwirft")
        self._btn_messen.clicked.connect(lambda: self.messen_angefordert.emit())
        gruppe(self._btn_messen)

        self._chk_temp = QtWidgets.QCheckBox("Temperatur anzeigen")
        self._chk_temp.setChecked(True)
        self._chk_temp.setEnabled(False)
        self._chk_temp.setToolTip(
            "Zeigt beim Überfahren eines Punktes seine Temperatur — in jedem\n"
            "Farbmodus. Braucht eine Thermal-Einfärbung aus dem Mäanderflug.")
        self._chk_temp.toggled.connect(self._temp_umgeschaltet)
        gruppe(self._chk_temp)

        knopf = QtWidgets.QPushButton("Ansicht zurücksetzen")
        knopf.setToolTip("R")
        knopf.clicked.connect(self.reset_camera)
        gruppe(knopf)
        lay.addStretch(1)
        hinweis = QtWidgets.QLabel("links: drehen · rechts/Umschalt: verschieben · Rad: zoomen")
        hinweis.setStyleSheet("font-style: italic;")
        hinweis.setEnabled(False)          # gedaempft, in jedem Theme lesbar
        lay.addWidget(hinweis)
        return leiste

    def set_farbmodi(self, eintraege: list, aktuell: str) -> None:
        """Auswahl der Leiste fuellen: [(schluessel, text, verfuegbar), ...]."""
        self._farbe.blockSignals(True)
        self._farbe.clear()
        for key, text, da in eintraege:
            self._farbe.addItem(text, key)
            item = self._farbe.model().item(self._farbe.count() - 1)
            item.setEnabled(bool(da))
            if not da:
                item.setToolTip("für dieses Projekt nicht vorhanden")
        self._farbe.blockSignals(False)
        self.setze_farbmodus(aktuell)

    def setze_farbmodus(self, key: str) -> None:
        i = self._farbe.findData(key)
        if i >= 0 and i != self._farbe.currentIndex():
            self._farbe.blockSignals(True)
            self._farbe.setCurrentIndex(i)
            self._farbe.blockSignals(False)

    def _groesse_gezogen(self, v: int) -> None:
        wert = v / 4.0
        self._groesse_lbl.setText(f"{wert:g}".replace(".", ","))
        self._actor.GetProperty().SetPointSize(wert)
        self._render()
        self.punktgroesse_geaendert.emit(wert)

    def _temp_umgeschaltet(self, an: bool) -> None:
        self._temp_an = bool(an)
        if not an:
            QtWidgets.QToolTip.hideText()
        self.temperatur_umgeschaltet.emit(bool(an))

    def set_temperatur_anzeigen(self, an: bool) -> None:
        self._chk_temp.blockSignals(True)
        self._chk_temp.setChecked(bool(an))
        self._chk_temp.blockSignals(False)
        self._temp_an = bool(an)

    def set_temperatur(self, temp: np.ndarray | None) -> None:
        """Temperatur je Punkt der Wolke (°C, NaN wo keine) oder None."""
        if temp is not None:
            temp = np.asarray(temp, dtype=np.float32).ravel()
            if self._points is not None and len(temp) != len(self._points):
                raise ValueError("Temperaturen passen nicht zur Punktanzahl.")
        self._temperatur = temp
        self._hover_pts = self._hover_temp = None
        da = temp is not None and bool(np.isfinite(temp[:: max(1, len(temp) // 10000)]).any())
        self._chk_temp.setEnabled(da)
        self._chk_temp.setToolTip(
            "Zeigt beim Überfahren eines Punktes seine Temperatur — in jedem "
            "Farbmodus." if da else
            "Keine Temperaturen: erst mit dem Mäanderflug und „Thermalbilder "
            "mitrechnen“ einfärben.")

    # ------------------------------------------------------------- Hoehenschnitt

    def _on_cut_changed(self, lo: float, hi: float) -> None:
        """Schnittebenen am Mapper nachziehen (kein Geometrie-Neuaufbau)."""
        aktiv = self.cut_bar.is_cut()
        self._plane_lo.SetOrigin(0.0, 0.0, lo)
        self._plane_hi.SetOrigin(0.0, 0.0, hi)
        if aktiv and not self._cut_active:
            for m in (self._mapper, self._prev_mapper, self._cprev_mapper):
                m.AddClippingPlane(self._plane_lo)
                m.AddClippingPlane(self._plane_hi)
        elif not aktiv and self._cut_active:
            for m in (self._mapper, self._prev_mapper, self._cprev_mapper):
                m.RemoveClippingPlane(self._plane_lo)
                m.RemoveClippingPlane(self._plane_hi)
        self._cut_active = aktiv
        self._render()

    def cut_planes(self) -> tuple[float, float] | None:
        """Aktive Schnittebenen (unten, oben) oder None, wenn nicht geschnitten."""
        return self.cut_bar.planes() if self.cut_bar.is_cut() else None

    def _refresh_cut_range(self) -> None:
        """Hoehenbereich aus der geladenen Wolke uebernehmen."""
        if self._points is None or len(self._points) == 0:
            self.cut_bar.set_range(0.0, 0.0)
            return
        z = self._points[:, 2]
        self.cut_bar.set_range(float(z.min()), float(z.max()))

    # --------------------------------------------------------------- Messen

    def set_measure(self, on: bool) -> None:
        """Messmodus schalten; beim Ausschalten wird die Messung verworfen."""
        on = bool(on)
        if on == self._measure_on:
            return
        self._measure_on = on
        self._vtkw.setCursor(QtCore.Qt.CrossCursor if on else QtCore.Qt.OpenHandCursor)
        self._btn_messen.blockSignals(True)
        self._btn_messen.setChecked(on)
        self._btn_messen.blockSignals(False)
        if not on:
            self.clear_measure()

    def measure_enabled(self) -> bool:
        return self._measure_on

    def clear_measure(self) -> None:
        self._meas = []
        for a in self._meas_actors:
            self._renderer.RemoveActor(a)
        self._meas_actors = []
        self.measured.emit(None, None)
        self._render()

    def pick_point(self, x: int, y: int, radius_px: float = 14.0) -> np.ndarray | None:
        """Sichtbaren Punkt nahe (x, y) in Widget-Koordinaten suchen.

        Genommen wird der Punkt, welcher dem Klick am naechsten liegt und dabei
        der Kamera am naechsten steht — sonst greift man durch eine Wand
        hindurch. Gesucht wird nur unter den ANGEZEIGTEN Punkten und, bei
        aktivem Hoehenschnitt, nur innerhalb der sichtbaren Schicht: was
        weggeschnitten ist, laesst sich auch nicht anklicken.
        """
        P = self._disp_points
        if P is None or len(P) == 0 or not self._initialized:
            return None
        i = self._naechster(P, x, y, radius_px)
        return None if i < 0 else np.array(P[i], dtype=np.float64)

    def _naechster(self, P: np.ndarray, x: int, y: int, radius_px: float) -> int:
        """Index des Punktes in ``P`` nahe (x, y), der der Kamera am naechsten
        steht; -1, wenn keiner. Beachtet den Hoehenschnitt."""
        w = max(self._vtkw.width(), 1)
        h = max(self._vtkw.height(), 1)
        cam = self._renderer.GetActiveCamera()
        m = cam.GetCompositeProjectionTransformMatrix(w / h, -1.0, 1.0)
        M = np.array([[m.GetElement(i, j) for j in range(4)] for i in range(4)],
                     dtype=np.float32)
        pts = np.asarray(P, dtype=np.float32)
        wq = pts @ M[3, :3] + M[3, 3]
        cx = pts @ M[0, :3] + M[0, 3]
        cy = pts @ M[1, :3] + M[1, 3]
        vorn = wq > 1e-9
        wq = np.where(vorn, wq, 1.0)
        sx = (cx / wq * 0.5 + 0.5) * w
        sy = (1.0 - (cy / wq * 0.5 + 0.5)) * h        # Qt zaehlt von oben
        nah = vorn & ((sx - x) ** 2 + (sy - y) ** 2 <= radius_px * radius_px)
        schnitt = self.cut_planes()
        if schnitt is not None:
            nah &= (pts[:, 2] >= schnitt[0]) & (pts[:, 2] <= schnitt[1])
        kand = np.flatnonzero(nah)
        if not len(kand):
            return -1
        return int(kand[int(np.argmin(wq[kand]))])      # kleinste Tiefe gewinnt

    def _add_measure_point(self, p: np.ndarray) -> None:
        if len(self._meas) >= 2:      # dritter Klick faengt neu an
            self.clear_measure()
        self._meas.append(p)
        self._rebuild_measure()
        a = self._meas[0]
        b = self._meas[1] if len(self._meas) > 1 else None
        self.measured.emit(a, b)

    def _rebuild_measure(self) -> None:
        for a in self._meas_actors:
            self._renderer.RemoveActor(a)
        self._meas_actors = []
        if not self._meas:
            self._render()
            return
        pts = vtk.vtkPoints()
        for p in self._meas:
            pts.InsertNextPoint(*p)
        poly = vtk.vtkPolyData()
        poly.SetPoints(pts)
        verts = vtk.vtkCellArray()
        for i in range(len(self._meas)):
            verts.InsertNextCell(1)
            verts.InsertCellPoint(i)
        poly.SetVerts(verts)
        mk = vtk.vtkPolyDataMapper()
        mk.SetInputData(poly)
        ak = vtk.vtkActor()
        ak.SetMapper(mk)
        pr = ak.GetProperty()
        pr.SetPointSize(13)
        pr.SetColor(1.0, 0.85, 0.2)
        if hasattr(pr, "RenderPointsAsSpheresOn"):
            pr.RenderPointsAsSpheresOn()
        self._renderer.AddActor(ak)
        self._meas_actors.append(ak)

        if len(self._meas) == 2:
            a, b = self._meas
            line = vtk.vtkLineSource()
            line.SetPoint1(*a)
            line.SetPoint2(*b)
            ml = vtk.vtkPolyDataMapper()
            ml.SetInputConnection(line.GetOutputPort())
            al = vtk.vtkActor()
            al.SetMapper(ml)
            al.GetProperty().SetColor(1.0, 0.85, 0.2)
            al.GetProperty().SetLineWidth(2)
            self._renderer.AddActor(al)
            self._meas_actors.append(al)
            txt = vtk.vtkBillboardTextActor3D()
            txt.SetPosition(*((a + b) / 2.0))
            txt.SetInput(f"{float(np.linalg.norm(b - a)):.3f} m")
            tp = txt.GetTextProperty()
            tp.SetFontSize(17)
            tp.SetColor(1.0, 0.9, 0.35)
            tp.SetJustificationToCentered()
            tp.SetBold(True)
            self._renderer.AddActor(txt)
            self._meas_actors.append(txt)
        self._render()

    def eventFilter(self, obj, event):  # noqa: N802 (Qt)
        """Maus und Tasten der 3D-Ansicht — Steuerung wie im VS-Code-Viewer.

        Alle Maus- und Tastenereignisse werden hier verbraucht; VTK sieht
        keines. Sonst dreht sein eigener Stil mit, und dessen Tasten (q, e
        beenden) waeren eine Falle.
        """
        if obj is not self._vtkw:
            return super().eventFilter(obj, event)
        et = event.type()
        E = QtCore.QEvent
        if et == E.MouseButtonPress:
            self._vtkw.setFocus()
            links = event.button() == QtCore.Qt.LeftButton
            mod = event.modifiers() & (QtCore.Qt.ShiftModifier | QtCore.Qt.ControlModifier)
            self._ziehen = 1 if (links and not mod) else 2
            self._maus_letzt = self._maus_start = event.pos()
            QtWidgets.QToolTip.hideText()
            if not self._measure_on:
                self._vtkw.setCursor(QtCore.Qt.ClosedHandCursor)
            return True
        if et == E.MouseMove:
            if self._ziehen:
                d = event.pos() - self._maus_letzt
                self._maus_letzt = event.pos()
                if self._ziehen == 1:
                    self.drehen(d.x(), d.y())
                else:
                    self.verschieben(d.x(), d.y())
            elif self._temp_an and self._temperatur is not None:
                self._hover_pos = event.pos()
                self._hover_timer.start()
            return True
        if et == E.MouseButtonRelease:
            bewegt = (event.pos() - self._maus_start).manhattanLength()
            war = self._ziehen
            self._ziehen = 0
            self._vtkw.setCursor(QtCore.Qt.CrossCursor if self._measure_on
                                 else QtCore.Qt.OpenHandCursor)
            # Ein Klick ist ein Klick, solange die Maus dabei stehen bleibt
            if self._measure_on and war == 1 and event.button() == QtCore.Qt.LeftButton \
                    and bewegt < 5:
                p = self.pick_point(event.x(), event.y(), radius_px=16.0)
                if p is not None:
                    self._add_measure_point(p)
            return True
        if et == E.MouseButtonDblClick:
            return True
        if et == E.Wheel:
            # Qt: 120 je Raste; der Browser liefert dafuer 100 Pixel
            schritte = event.angleDelta().y() / 120.0
            if schritte:
                self.zoomen(-schritte * 100.0)
            return True
        if et == E.KeyPress:
            if event.key() == QtCore.Qt.Key_Escape:
                self.clear_measure()
            return True
        if et == E.KeyRelease:
            return True
        if et == E.Leave:
            self._hover_timer.stop()
            QtWidgets.QToolTip.hideText()
        return super().eventFilter(obj, event)

    # -------------------------------------------------------------- Kamera

    def drehen(self, dx: float, dy: float) -> None:
        k = self._kam
        k["yaw"] -= dx * _DREH
        k["pitch"] = max(-_NICK_MAX, min(_NICK_MAX, k["pitch"] + dy * _DREH))
        self._kamera_setzen()

    def verschieben(self, dx: float, dy: float) -> None:
        """In der Bildebene schieben; Schrittweite haengt am Abstand."""
        k = self._kam
        schritt = k["dist"] * math.tan(math.radians(_BLICKWINKEL / 2.0)) * 2.0 \
            / max(1, self._vtkw.height())
        rechts, oben = self._bildachsen()
        k["ziel"] = k["ziel"] - rechts * dx * schritt + oben * dy * schritt
        self._kamera_setzen()

    def zoomen(self, delta_browser: float) -> None:
        k = self._kam
        k["dist"] *= math.exp(delta_browser * _ZOOM)
        k["dist"] = max(k["radius"] * 1e-4, min(k["radius"] * 200.0, k["dist"]))
        self._kamera_setzen()

    def _augenrichtung(self) -> np.ndarray:
        k = self._kam
        cp, sp = math.cos(k["pitch"]), math.sin(k["pitch"])
        cy, sy = math.cos(k["yaw"]), math.sin(k["yaw"])
        return np.array([cp * cy, cp * sy, sp])

    def _bildachsen(self) -> tuple:
        blick = -self._augenrichtung()               # vom Auge zum Ziel
        rechts = np.cross(blick, [0.0, 0.0, 1.0])
        n = np.linalg.norm(rechts)
        rechts = rechts / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
        oben = np.cross(rechts, blick)
        return rechts, oben / max(np.linalg.norm(oben), 1e-9)

    def _kamera_setzen(self) -> None:
        k = self._kam
        cam = self._renderer.GetActiveCamera()
        auge = k["ziel"] + k["dist"] * self._augenrichtung()
        cam.SetFocalPoint(*k["ziel"])
        cam.SetPosition(*auge)
        cam.SetViewUp(0.0, 0.0, 1.0)
        cam.SetViewAngle(_BLICKWINKEL)
        nah = max(k["dist"] * 0.001, 1e-4)
        cam.SetClippingRange(nah, k["dist"] * 4.0 + k["radius"] * 6.0)
        self._render()

    # --------------------------------------------------------- Temperatur

    def _hover_vorbereiten(self) -> bool:
        """Stichprobe der angezeigten Punkte, die eine Temperatur haben."""
        if self._hover_pts is not None:
            return len(self._hover_pts) > 0
        if self._temperatur is None or self._disp_points is None:
            return False
        idx = self._sel if self._sel is not None else None
        temp = self._temperatur if idx is None else self._temperatur[idx]
        gut = np.flatnonzero(np.isfinite(temp))
        if len(gut) > _HOVER_PUNKTE:
            gut = gut[:: len(gut) // _HOVER_PUNKTE + 1]
        self._hover_pts = np.ascontiguousarray(self._disp_points[gut], dtype=np.float32)
        self._hover_temp = np.ascontiguousarray(temp[gut], dtype=np.float32)
        return len(gut) > 0

    def temperatur_bei(self, x: int, y: int, radius_px: float = 9.0) -> float | None:
        """Temperatur des Punktes unter (x, y), oder None."""
        if self._temperatur is None or not self._hover_vorbereiten():
            return None
        i = self._naechster(self._hover_pts, x, y, radius_px)
        return None if i < 0 else float(self._hover_temp[i])

    def _hover_zeigen(self) -> None:
        if not self._temp_an or self._ziehen:
            return
        pos = self._hover_pos
        t = self.temperatur_bei(pos.x(), pos.y())
        if t is None:
            QtWidgets.QToolTip.hideText()
            return
        QtWidgets.QToolTip.showText(self._vtkw.mapToGlobal(pos + QtCore.QPoint(14, 10)),
                                    f"{t:.1f} °C".replace(".", ","), self._vtkw)

    # ------------------------------------------------------------- Vorschau

    def _point_poly(self, pts: np.ndarray) -> tuple:
        """vtkPolyData aus einer Punktliste; gibt (poly, refs) zurueck.

        Die numpy-Puffer muessen am Leben bleiben, solange VTK sie benutzt —
        deshalb wandern sie als refs mit zurueck und werden am Widget gehalten.
        """
        n = len(pts)
        vtk_pts = vtk.vtkPoints()
        vtk_pts.SetData(_strip_numpy_ref(
            numpy_to_vtk(pts, deep=False, array_type=vtk.VTK_FLOAT)))
        poly = vtk.vtkPolyData()
        poly.SetPoints(vtk_pts)
        verts = vtk.vtkCellArray()
        refs: list = [pts]
        try:
            offsets = np.arange(n + 1, dtype=_ID_DTYPE)
            conn = np.arange(n, dtype=_ID_DTYPE)
            verts.SetData(_strip_numpy_ref(numpy_to_vtkIdTypeArray(offsets, deep=False)),
                          _strip_numpy_ref(numpy_to_vtkIdTypeArray(conn, deep=False)))
            refs += [offsets, conn]
        except (AttributeError, TypeError):  # pre-9.0 fallback
            legacy = np.empty(2 * n, dtype=_ID_DTYPE)
            legacy[0::2] = 1
            legacy[1::2] = np.arange(n, dtype=_ID_DTYPE)
            verts.SetCells(n, _strip_numpy_ref(numpy_to_vtkIdTypeArray(legacy, deep=False)))
            refs.append(legacy)
        poly.SetVerts(verts)
        return poly, refs

    def set_color_preview(self, points: np.ndarray | None,
                          rgb: np.ndarray | None = None,
                          solo: bool = True) -> None:
        """Eingefaerbte Stichprobe zeigen (Maeander-Handjustage).

        Damit folgt die Wolke dem Regler: die Stichprobe wird bei jeder
        Aenderung neu eingefaerbt und hier ersetzt. ``None`` raeumt sie weg.

        ``solo`` blendet die Karte waehrenddessen aus, und das ist der
        Normalfall. Eine Stichprobe von 50.000 Punkten sind bei einer Karte
        aus 24 Millionen zwei Promille — als Staub darueber gestreut sieht
        man von einer Farbaenderung nichts. Allein gezeigt ist sie die
        ganze Ansicht, und jeder Reglerzug ist sofort zu sehen.
        """
        if points is None or len(points) == 0:
            self._cprev_actor.SetVisibility(False)
            self._cprev_mapper.SetInputData(vtk.vtkPolyData())
            self._cprev_refs = []
            self._actor.SetVisibility(True)
            self._render()
            return
        pts = np.ascontiguousarray(np.asarray(points).reshape(-1, 3), dtype=np.float32)
        poly, refs = self._point_poly(pts)
        if rgb is not None:
            farben = np.ascontiguousarray(np.asarray(rgb).reshape(-1, 3), dtype=np.uint8)
            if len(farben) != len(pts):
                raise ValueError("Farben passen nicht zur Punktanzahl.")
            arr = _strip_numpy_ref(numpy_to_vtk(farben, deep=False,
                                                array_type=vtk.VTK_UNSIGNED_CHAR))
            arr.SetName("vorschau")
            poly.GetPointData().SetScalars(arr)
            refs.append(farben)
            self._cprev_mapper.ScalarVisibilityOn()
        else:
            self._cprev_mapper.ScalarVisibilityOff()
        self._cprev_refs = refs
        self._cprev_mapper.SetInputData(poly)
        self._cprev_actor.SetVisibility(True)
        self._actor.SetVisibility(not solo)
        if self._cut_active:
            self._cprev_mapper.RemoveAllClippingPlanes()
            self._cprev_mapper.AddClippingPlane(self._plane_lo)
            self._cprev_mapper.AddClippingPlane(self._plane_hi)
        self._render()

    def has_color_preview(self) -> bool:
        return bool(self._cprev_actor.GetVisibility())

    def set_preview_solo(self, on: bool) -> None:
        """Karte waehrend der Vorschau aus- oder wieder einblenden."""
        if not self.has_color_preview():
            self._actor.SetVisibility(True)
        else:
            self._actor.SetVisibility(not bool(on))
        self._render()

    def map_visible(self) -> bool:
        return bool(self._actor.GetVisibility())

    def set_preview_cloud(self, points: np.ndarray | None,
                          color: tuple[float, float, float] = (1.0, 0.55, 0.20)) -> None:
        """Zweite Wolke einfarbig darueberlegen (Merge-Vorschau); None entfernt sie.

        Der Hoehenschnitt gilt auch hier, sonst haenge die Vorschau ueber einer
        aufgeschnittenen Karte.
        """
        if points is None or len(points) == 0:
            self._prev_wanted = False
            self._prev_actor.SetVisibility(False)
            self._prev_mapper.SetInputData(vtk.vtkPolyData())
            self._prev_refs = []
            self._render()
            return
        pts = np.ascontiguousarray(np.asarray(points).reshape(-1, 3), dtype=np.float32)
        n = len(pts)
        vtk_pts = vtk.vtkPoints()
        vtk_pts.SetData(_strip_numpy_ref(
            numpy_to_vtk(pts, deep=False, array_type=vtk.VTK_FLOAT)))
        poly = vtk.vtkPolyData()
        poly.SetPoints(vtk_pts)
        verts = vtk.vtkCellArray()
        refs: list = [pts]
        try:
            offsets = np.arange(n + 1, dtype=_ID_DTYPE)
            conn = np.arange(n, dtype=_ID_DTYPE)
            verts.SetData(_strip_numpy_ref(numpy_to_vtkIdTypeArray(offsets, deep=False)),
                          _strip_numpy_ref(numpy_to_vtkIdTypeArray(conn, deep=False)))
            refs += [offsets, conn]
        except (AttributeError, TypeError):  # pre-9.0 fallback
            legacy = np.empty(2 * n, dtype=_ID_DTYPE)
            legacy[0::2] = 1
            legacy[1::2] = np.arange(n, dtype=_ID_DTYPE)
            verts.SetCells(n, _strip_numpy_ref(numpy_to_vtkIdTypeArray(legacy, deep=False)))
            refs.append(legacy)
        poly.SetVerts(verts)
        self._prev_refs = refs
        self._prev_mapper.SetInputData(poly)
        self._prev_actor.GetProperty().SetColor(*color)
        self._prev_wanted = True
        self._prev_actor.SetVisibility(True)
        if self._cut_active:
            self._prev_mapper.RemoveAllClippingPlanes()
            self._prev_mapper.AddClippingPlane(self._plane_lo)
            self._prev_mapper.AddClippingPlane(self._plane_hi)
        self._render()

    def has_preview(self) -> bool:
        """Ist eine Vorschau geladen (auch wenn sie gerade ausgeblendet ist)?"""
        return bool(self._prev_wanted)

    def preview_visible(self) -> bool:
        return bool(self._prev_actor.GetVisibility())

    def set_preview_visible(self, on: bool) -> None:
        """Vorschau ein- oder ausblenden, ohne sie zu verwerfen."""
        self._prev_actor.SetVisibility(bool(on) and self._prev_wanted)
        self._render()

    # ------------------------------------------------------------------ data

    def set_cloud(self, points: np.ndarray, colors: np.ndarray | None = None,
                  intensity: np.ndarray | None = None,
                  valid: np.ndarray | None = None) -> None:
        if points is None:
            # Alle gehaltenen Puffer freigeben (auch Anzeige-/VTK-Referenzen),
            # sonst bleiben ~Hunderte MB der alten Wolke fuer die Session liegen.
            self._points = self._colors = self._intensity = self._valid = None
            self.set_temperatur(None)
            self._mapper.SetInputData(vtk.vtkPolyData())
            self._poly = None
            self._sel = None
            self._disp_points = None
            self._rgb_np = None
            self._vtk_refs = []
            self._had_cloud = False  # naechste Wolke passt die Kamera neu ein
            self._refresh_cut_range()
            self._render()
            return
        pts = np.ascontiguousarray(np.asarray(points).reshape(-1, 3), dtype=np.float32)
        n = len(pts)
        if colors is not None:
            colors = np.asarray(colors).reshape(-1, 3)
            if len(colors) != n:
                raise ValueError("Farben passen nicht zur Punktanzahl.")
            if colors.dtype != np.uint8:
                if np.issubdtype(colors.dtype, np.floating) and colors.size and colors.max() <= 1.0:
                    colors = colors * 255.0
                colors = np.clip(colors, 0, 255).astype(np.uint8)
            colors = np.ascontiguousarray(colors)
        if intensity is not None:
            intensity = np.asarray(intensity, dtype=np.float32).ravel()
            if len(intensity) != n:
                raise ValueError("Intensitäten passen nicht zur Punktanzahl.")
        if valid is not None:
            valid = np.asarray(valid).ravel().astype(bool)
            if len(valid) != n:
                raise ValueError("Gültigkeitsmaske passt nicht zur Punktanzahl.")
        self._points, self._colors = pts, colors
        self._intensity, self._valid = intensity, valid
        if self._temperatur is not None and len(self._temperatur) != n:
            self.set_temperatur(None)      # gehoerte zu einer anderen Wolke
        # Eine neue Wolke raeumt eine alte Farbvorschau weg und wird immer
        # gezeigt. Die Vorschau gehoert zu einer laufenden Justage; sobald
        # sich die Wolke darunter aendert, passt sie nicht mehr — und eine
        # Solo-Vorschau wuerde die neue Wolke sonst weiter verdecken.
        self.set_color_preview(None)
        self._actor.SetVisibility(True)
        self._refresh_cut_range()
        self._rebuild_geometry()
        if not self._had_cloud:
            self._had_cloud = True
            self.reset_camera()
        else:
            self._render()

    def set_path(self, positions: np.ndarray | None) -> None:
        if self._path_actor is not None:
            self._renderer.RemoveActor(self._path_actor)
            self._path_actor = None
        if positions is not None:
            pos = np.ascontiguousarray(np.asarray(positions, dtype=np.float32).reshape(-1, 3))
            if len(pos) >= 2:
                poly = vtk.vtkPolyData()
                vp = vtk.vtkPoints()
                vp.SetData(numpy_to_vtk(pos, deep=True, array_type=vtk.VTK_FLOAT))
                poly.SetPoints(vp)
                cell = np.empty(len(pos) + 1, dtype=_ID_DTYPE)
                cell[0] = len(pos)
                cell[1:] = np.arange(len(pos), dtype=_ID_DTYPE)
                lines = vtk.vtkCellArray()
                lines.SetCells(1, numpy_to_vtkIdTypeArray(cell, deep=True))
                poly.SetLines(lines)
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(poly)
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                prop = actor.GetProperty()
                prop.SetColor(*_ACCENT)
                prop.SetLineWidth(3.0)
                prop.LightingOff()
                self._renderer.AddActor(actor)
                self._path_actor = actor
        self._render()

    # --------------------------------------------------------------- options

    def set_point_size(self, size: float) -> None:
        """Punktgroesse in Pixeln, 0,5 bis 10 in Viertelschritten.

        Gebrochene Groessen wirken: die Grafikkarte setzt je Punkt die Pixel,
        deren Mitte im Quadrat der Groesse liegt — bei 1,5 sind das je nach
        Lage ein oder zwei, im Mittel also dazwischen.
        """
        wert = round(max(0.5, min(10.0, float(size))) * 4.0) / 4.0
        self._groesse.blockSignals(True)
        self._groesse.setValue(int(round(wert * 4)))
        self._groesse.blockSignals(False)
        self._groesse_lbl.setText(f"{wert:g}".replace(".", ","))
        self._actor.GetProperty().SetPointSize(wert)
        self._render()

    def set_color_mode(self, mode: str) -> None:
        if mode not in _COLOR_MODES:
            raise ValueError(f"Unbekannter Farbmodus: {mode!r}")
        if mode != self._color_mode:
            self._color_mode = mode
            self._update_colors()
            self._render()

    def set_only_colored(self, on: bool) -> None:
        on = bool(on)
        if on != self._only_colored:
            self._only_colored = on
            self._rebuild_geometry()
            self._render()

    def set_voxel_display(self, voxel: float) -> None:
        voxel = max(0.0, float(voxel))
        if voxel != self._voxel:
            self._voxel = voxel
            self._rebuild_geometry()
            self._render()

    def set_background(self, name: str) -> None:
        if name not in _BG:
            raise ValueError(f"Unbekannter Hintergrund: {name!r}")
        self._background = name
        self._renderer.SetBackground(*_BG[name])
        self._actor.GetProperty().SetColor(*_UNIFORM_COLOR[name])  # used in uniform mode
        self._render()

    def set_eyedome(self, on: bool) -> None:
        if not self.edl_available:
            self._edl_on = False
            return
        self._edl_on = bool(on)
        try:
            self._renderer.SetPass(self._edl_pass if self._edl_on else None)
        except Exception:
            self.edl_available = False
            self._edl_on = False
        self._render()

    # --------------------------------------------------------------- actions

    def reset_camera(self) -> None:
        """Startansicht wie im VS-Code-Viewer: auf den Schwerpunkt, schraeg
        von oben (Gier −45°, Nick 0,5 rad), Abstand 2,2 × der Radius, in dem
        95 % der Punkte liegen — ein einzelner Ausreisser schiebt die Kamera
        sonst so weit weg, dass die Wolke als Fleck in der Mitte steht."""
        k = self._kam
        P = self._points
        if P is not None and len(P) > 0:
            probe = np.asarray(P[:: max(1, len(P) // 200_000)], dtype=np.float64)
            mitte = probe.mean(axis=0)
            abst = np.linalg.norm(probe - mitte, axis=1)
            k["ziel"] = mitte
            k["radius"] = max(0.5 * float(np.linalg.norm(probe.max(0) - probe.min(0))), 1e-3)
            sicht = float(np.percentile(abst, 95.0)) or k["radius"]
            k["dist"] = max(sicht * 2.2, 1e-3)
        k["yaw"], k["pitch"] = -math.pi / 4, 0.5
        self._kamera_setzen()

    def screenshot(self, path: str) -> None:
        if not self._initialized:
            raise RuntimeError("Screenshot erst möglich, wenn das Widget angezeigt wurde.")
        rw = self._vtkw.GetRenderWindow()
        rw.Render()
        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(rw)
        w2i.ReadFrontBufferOff()
        w2i.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(path)
        writer.SetInputConnection(w2i.GetOutputPort())
        writer.Write()

    # ------------------------------------------------------------- internals

    def _rebuild_geometry(self) -> None:
        """Apply valid mask + voxel downsample, then build fresh polydata."""
        if self._points is None:
            return
        sel: np.ndarray | None = None
        if self._only_colored and self._valid is not None:
            sel = np.flatnonzero(self._valid)
        if self._voxel > 0.0:
            pts = self._points if sel is None else self._points[sel]
            keep = _voxel_first_indices(pts, self._voxel)
            sel = keep if sel is None else sel[keep]
        self._sel = sel
        self._hover_pts = self._hover_temp = None     # Stichprobe neu ziehen
        disp = self._points if sel is None else self._points[sel]
        self._disp_points = np.ascontiguousarray(disp, dtype=np.float32)
        n = len(self._disp_points)

        refs: list = [self._disp_points]
        vtk_pts = vtk.vtkPoints()
        vtk_pts.SetData(_strip_numpy_ref(
            numpy_to_vtk(self._disp_points, deep=False, array_type=vtk.VTK_FLOAT)))
        poly = vtk.vtkPolyData()
        poly.SetPoints(vtk_pts)
        verts = vtk.vtkCellArray()
        try:
            offsets = np.arange(n + 1, dtype=_ID_DTYPE)
            conn = np.arange(n, dtype=_ID_DTYPE)
            verts.SetData(_strip_numpy_ref(numpy_to_vtkIdTypeArray(offsets, deep=False)),
                          _strip_numpy_ref(numpy_to_vtkIdTypeArray(conn, deep=False)))
            refs += [offsets, conn]
        except (AttributeError, TypeError):  # pre-9.0 fallback: legacy cell layout
            legacy = np.empty(2 * n, dtype=_ID_DTYPE)
            legacy[0::2] = 1
            legacy[1::2] = np.arange(n, dtype=_ID_DTYPE)
            verts.SetCells(n, _strip_numpy_ref(numpy_to_vtkIdTypeArray(legacy, deep=False)))
            refs.append(legacy)
        poly.SetVerts(verts)
        self._vtk_refs = refs
        self._poly = poly
        self._mapper.SetInputData(poly)
        self._update_colors()

    def _update_colors(self) -> None:
        if self._poly is None:
            return
        sel = self._sel
        rgb: np.ndarray | None = None
        if self._disp_points is not None and len(self._disp_points) > 0:
            mode = self._color_mode
            if mode == "rgb" and self._colors is not None:
                rgb = self._colors if sel is None else self._colors[sel]
                if self._valid is not None and not self._only_colored:
                    # Nicht eingefaerbte Punkte neutral grau anzeigen — auf einer
                    # Kopie, das Farb-Array des Aufrufers wird nie veraendert.
                    vmask = self._valid if sel is None else self._valid[sel]
                    invalid = ~vmask
                    if invalid.any():
                        if rgb is self._colors:
                            rgb = rgb.copy()
                        rgb[invalid] = 90
            elif mode == "hoehe":
                rgb = _scalar_to_rgb(self._disp_points[:, 2], _TURBO)
            elif mode == "intensitaet" and self._intensity is not None:
                vals = self._intensity if sel is None else self._intensity[sel]
                rgb = _scalar_to_rgb(vals, None)
        if rgb is None:  # "uniform" or missing data for the requested mode
            self._mapper.ScalarVisibilityOff()
            self._actor.GetProperty().SetColor(*_UNIFORM_COLOR[self._background])
        else:
            self._rgb_np = np.ascontiguousarray(rgb, dtype=np.uint8)
            arr = _strip_numpy_ref(
                numpy_to_vtk(self._rgb_np, deep=False, array_type=vtk.VTK_UNSIGNED_CHAR))
            arr.SetName("farben")
            self._poly.GetPointData().SetScalars(arr)
            self._mapper.SetColorModeToDirectScalars()
            self._mapper.ScalarVisibilityOn()
        self._poly.Modified()

    def _render(self) -> None:
        if self._initialized:
            self._vtkw.GetRenderWindow().Render()

    # ------------------------------------------------------------- lifecycle

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._initialized:
            self._vtkw.Initialize()
            self._vtkw.Start()  # no-op under Qt event loop, standard pattern
            self._initialized = True
        self._render()

    def closeEvent(self, event) -> None:
        try:
            rw = self._vtkw.GetRenderWindow()
            if self._edl_pass is not None:
                self._renderer.SetPass(None)
                self._edl_pass.ReleaseGraphicsResources(rw)
            rw.Finalize()
        except Exception:
            pass
        try:
            self._vtkw.close()
        except Exception:
            pass
        super().closeEvent(event)


if __name__ == "__main__":
    import os
    import sys
    import time

    from PyQt5 import QtCore

    ev_dir = ("/tmp/super360_modtests/"
              "cloud_view")
    os.makedirs(ev_dir, exist_ok=True)
    os.environ.setdefault("DISPLAY", ":0")

    app = QtWidgets.QApplication(sys.argv)
    view = CloudView()
    view.resize(1280, 800)
    view.setWindowTitle("CloudView Selbsttest")
    view.show()
    app.processEvents()

    report: list[str] = []

    def snap(name: str) -> None:
        view.screenshot(os.path.join(ev_dir, name))
        app.processEvents()

    def timed(label: str, fn) -> float:
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        report.append(f"{label}: {dt:.3f} s")
        return dt

    rng = np.random.default_rng(42)

    # ---- 2M synthetic terrain-like points -----------------------------------
    N2 = 2_000_000
    xy = rng.uniform(-25.0, 25.0, (N2, 2)).astype(np.float32)
    z = (2.0 * np.sin(xy[:, 0] * 0.35) * np.cos(xy[:, 1] * 0.28)
         + 0.15 * rng.standard_normal(N2)).astype(np.float32)
    pts2 = np.column_stack([xy, z])
    colors2 = np.empty((N2, 3), np.uint8)
    colors2[:, 0] = ((xy[:, 0] + 25.0) / 50.0 * 255.0).astype(np.uint8)
    colors2[:, 1] = ((xy[:, 1] + 25.0) / 50.0 * 255.0).astype(np.uint8)
    colors2[:, 2] = np.clip((z + 3.0) / 6.0 * 255.0, 0, 255).astype(np.uint8)
    inten2 = (np.abs(np.sin(xy[:, 0] * 0.9)) * 80.0
              + 20.0 * rng.random(N2)).astype(np.float32)
    valid2 = xy[:, 0] > 0.0  # spatial half-false mask
    t_path = np.linspace(0.0, 4.0 * np.pi, 500)
    path = np.column_stack([10.0 * np.cos(t_path), 10.0 * np.sin(t_path),
                            3.0 + 0.2 * t_path]).astype(np.float32)

    report.append(f"EDL verfügbar: {view.edl_available}")
    timed("2M set_cloud + erster Render", lambda: view.set_cloud(pts2, colors2, inten2, valid2))
    view.set_path(path)
    view.reset_camera()
    view._renderer.GetActiveCamera().Elevation(-35.0)
    view.reset_camera()
    timed("2M Screenshot rgb (01)", lambda: snap("01_rgb.png"))

    cam = view._renderer.GetActiveCamera()
    cam.Azimuth(25.0)
    timed("2M Interaktions-Render (Rotation)", view._render)

    timed("Farbmodus hoehe (02)", lambda: (view.set_color_mode("hoehe"), snap("02_hoehe.png")))
    timed("Farbmodus intensitaet (03)",
          lambda: (view.set_color_mode("intensitaet"), snap("03_intensitaet.png")))
    timed("Farbmodus uniform (04)", lambda: (view.set_color_mode("uniform"), snap("04_uniform.png")))

    view.set_color_mode("rgb")
    timed("Voxel 0.10 m (05)", lambda: (view.set_voxel_display(0.1), snap("05_voxel_0.10.png")))
    report.append(f"  Punkte nach Voxel 0.10: {view._poly.GetNumberOfPoints():,}")
    view.set_voxel_display(0.0)
    timed("Nur eingefärbte Punkte (06)",
          lambda: (view.set_only_colored(True), snap("06_nur_eingefaerbt.png")))
    report.append(f"  Punkte nach valid-Maske: {view._poly.GetNumberOfPoints():,} "
                  f"(erwartet ~{int(valid2.sum()):,})")
    view.set_only_colored(False)
    timed("EDL an (07)", lambda: (view.set_eyedome(True), snap("07_edl_an.png")))
    timed("EDL aus (08)", lambda: (view.set_eyedome(False), snap("08_edl_aus.png")))

    # ---- 9M points, rgb mode -------------------------------------------------
    N9 = 9_000_000
    pts9 = rng.uniform(-40.0, 40.0, (N9, 3)).astype(np.float32)
    pts9[:, 2] *= 0.15
    colors9 = rng.integers(0, 255, (N9, 3), dtype=np.uint8)
    timed("9M set_cloud + Render", lambda: view.set_cloud(pts9, colors9))
    view.reset_camera()
    timed("9M Screenshot rgb (09)", lambda: snap("09_9mio_rgb.png"))
    cam = view._renderer.GetActiveCamera()
    cam.Azimuth(30.0)
    timed("9M Interaktions-Render (Rotation)", view._render)

    view.close()
    QtCore.QTimer.singleShot(150, app.quit)
    app.exec_()
    print("== CloudView Selbsttest ==")
    print("\n".join(report))
    print(f"Screenshots: {ev_dir}")
    sys.exit(0)
