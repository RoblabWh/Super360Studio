"""Eigenes Fenster zum Ausrichten des Maeanderfluges auf die Karte.

Nach dem Vorbild der Pipeline aus PointCloudMerger: beide Wolken **uebereinander**
sehen und die eine von Hand auf die andere schieben. Dass das ein eigenes Fenster
bekommt, hat einen Grund — im Hauptfenster musste die Karte fuer die Vorschau
ausgeblendet werden, weil eine Stichprobe von 50.000 Punkten in 24 Millionen
untergeht. Hier ist das Ueberlagern der Zweck, nicht die Stoerung: die Karte
liegt bewusst ausgeduennt darunter, damit die Fotopunkte darauf zu sehen sind.

Drei Ansichten, umschaltbar:

* **Überlagerung** — Karte in Grau (nach Hoehe schattiert), die Fotopunkte des
  Fluges in Magenta, mit eigener Thermallage zusaetzlich in Orange. Damit sieht
  man auf einen Blick, ob der Flug ueberhaupt ueber diesem Gebiet lag und wie
  weit die Lage daneben liegt.
* **Farbvorschau RGB / Thermal** — eine Stichprobe der Karte, eingefaerbt mit
  der aktuellen Lage der jeweiligen Optik. Damit sieht man, ob die Farben auf
  den richtigen Strukturen landen.

Gier, X und Y gibt es zweimal: fuer RGB als Zuschlag auf die gefundene Lage
(dazu Z), fuer Thermal als Zuschlag auf die RGB-Lage (beide Optiken haengen an
derselben Gimbal, s. ``core.meander.thermal_lage``). Dazu je ein Massstab — die
Brennweite, s. ``core.optik``. Jeder Wert hat einen groben und einen feinen
Schieber (``ui.feinregler``). Alle wirken sofort. Das Hauptfenster bleibt
unberuehrt, bis „Lage übernehmen“.

Gezeichnet wird als **Bild**, nicht mit einem zweiten 3D-Fenster: zwei
OpenGL-Kontexte in einer Anwendung sind je nach Grafiktreiber und Sitzung eine
Quelle schwarzer Fenster. Das Bild ist trotzdem eine richtige Ansicht — Mausrad
zoomt, Ziehen dreht oder verschiebt. Gerechnet wird das in numpy mit
Tiefenpuffer: ein Durchlauf ueber 400.000 Punkte kostet einige zehn
Millisekunden, das reicht, damit das Bild der Maus folgt.
"""

from __future__ import annotations

import time
import traceback

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from core import meander as meander_mod
from core import optik as optik_mod
from ui.bausteine import still_setzen
from ui.feinregler import regler_grad, regler_meter, regler_prozent

_KARTE_PUNKTE = 300_000      # so viel Karte liegt als Untergrund darunter
_FOTO_PUNKTE = 60_000        # ... und so viele Fotopunkte darueber. Mehr
                             # deckt die Karte zu, statt sie zu zeigen.
_FARB_PUNKTE = 150_000       # Stichprobe fuer die Farbvorschau
_MAGENTA = np.array([255, 0, 200], np.uint8)
_ORANGE = np.array([255, 150, 0], np.uint8)
_UNGETROFFEN = np.array([60, 60, 64], np.uint8)
_HINTERGRUND = (20, 22, 28)

#: Blickrichtungen als (Azimut, Elevation) in Grad
_BLICKE = (("von oben", (0.0, 90.0)),
           ("von vorn", (0.0, 0.0)),
           ("von der Seite", (90.0, 0.0)),
           ("schräg", (-30.0, 35.0)))


def _grau_nach_hoehe(P: np.ndarray) -> np.ndarray:
    """Karte grau, aber nach Hoehe schattiert.

    Einheitliches Grau zeigt von oben nur den Umriss der Karte. Mit der Hoehe
    als Helligkeit stehen Daecher, Baeume und Kanten darin — genau die
    Strukturen, auf die man die Fotopunkte schiebt.
    """
    if len(P) == 0:
        return np.empty((0, 3), np.uint8)
    z = P[:, 2]
    lo, hi = np.percentile(z, [3.0, 97.0])
    v = np.clip((z - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    hell = 70.0 + 150.0 * v
    return (hell[:, None] * np.array([0.93, 0.96, 1.0])).astype(np.uint8)


class Ansicht(QtWidgets.QWidget):
    """Punkte als Bild, mit der Maus zu drehen, zu verschieben und zu zoomen.

    Parallelprojektion mit Tiefenpuffer. Die Ebenen werden der Reihe nach
    gezeichnet, jede spaetere liegt oben — so bleiben die Fotopunkte ueber der
    Karte sichtbar, auch wenn sie ein Stueck darunter liegen.

    Mausrad: zoomen (zum Mauszeiger hin). Links ziehen: drehen. Rechts,
    Mitte oder Umschalt+links ziehen: verschieben. Doppelklick: einpassen.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 420)
        self.setFocusPolicy(QtCore.Qt.WheelFocus)
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent, True)
        self._ebenen: list = []          # [(P float32 (N,3), C uint8 (N,3))]
        self._einpassen_an: np.ndarray | None = None
        self._az, self._el = 0.0, 90.0
        self._mitte = np.zeros(3)
        self._massstab: float | None = None   # Pixel je Meter
        self.punktgroesse = 2
        self._bild: QtGui.QImage | None = None
        self._puffer = None               # QImage haelt keine Kopie
        self._schmutzig = True
        self._maus = None
        self._modus_maus = None

    # --------------------------------------------------------------- Inhalt

    def setze_ebenen(self, ebenen: list, einpassen_an=None) -> None:
        """Neue Punkte zeigen. Die Ansicht bleibt stehen, wo sie war — beim
        Justieren soll sich die Karte nicht unter der Maus wegbewegen."""
        self._ebenen = [(np.ascontiguousarray(P, dtype=np.float32),
                         np.ascontiguousarray(C, dtype=np.uint8))
                        for P, C in ebenen if len(P)]
        if einpassen_an is not None:
            self._einpassen_an = np.asarray(einpassen_an, dtype=np.float32)
        if self._massstab is None:
            self.einpassen()
        self._neu()

    def blick(self, az: float, el: float) -> None:
        self._az, self._el = float(az), float(el)
        self.einpassen()

    def einpassen(self) -> None:
        """Mitte und Massstab so, dass die Karte das Fenster fuellt."""
        P = self._einpassen_an
        if P is None or not len(P):
            P = self._ebenen[0][0] if self._ebenen else None
        if P is None or not len(P):
            return
        if len(P) > 40_000:
            P = P[:: len(P) // 40_000 + 1]
        r, u, _ = self._achsen()
        c0 = np.median(P, axis=0).astype(np.float64)
        Q = (P - c0) @ np.stack([r, u]).T
        lo = np.percentile(Q, 0.5, axis=0)
        hi = np.percentile(Q, 99.5, axis=0)
        self._mitte = c0 + r * (lo[0] + hi[0]) / 2.0 + u * (lo[1] + hi[1]) / 2.0
        spanne = np.maximum(hi - lo, 1e-3)
        W, H = max(self.width(), 320), max(self.height(), 240)
        self._massstab = float(min((W - 16) / spanne[0], (H - 16) / spanne[1]))
        self._neu()

    def zoom(self, faktor: float, um=None) -> None:
        """Um ``faktor`` vergroessern; ``um`` ist der Bildpunkt, der stehen bleibt."""
        if self._massstab is None:
            return
        alt = self._massstab
        neu = float(np.clip(alt * faktor, 1e-3, 1e4))
        if um is not None:
            r, u, _ = self._achsen()
            ox = (um[0] - self.width() / 2.0) / alt
            oy = -(um[1] - self.height() / 2.0) / alt
            self._mitte = self._mitte + (r * ox + u * oy) * (1.0 - alt / neu)
        self._massstab = neu
        self._neu()

    def verschieben(self, dx_px: float, dy_px: float) -> None:
        if self._massstab is None:
            return
        r, u, _ = self._achsen()
        self._mitte = self._mitte - (r * dx_px - u * dy_px) / self._massstab
        self._neu()

    def drehen(self, daz: float, del_: float) -> None:
        self._az = (self._az + daz) % 360.0
        self._el = float(np.clip(self._el + del_, -90.0, 90.0))
        self._neu()

    def _achsen(self) -> tuple:
        """Bildachsen in Weltkoordinaten: rechts, oben, zum Betrachter hin."""
        a, e = np.radians(self._az), np.radians(self._el)
        ca, sa, ce, se = np.cos(a), np.sin(a), np.cos(e), np.sin(e)
        rz = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
        rechts = rz @ np.array([1.0, 0.0, 0.0])
        oben = rz @ np.array([0.0, se, ce])
        hin = rz @ np.array([0.0, -ce, se])
        return rechts, oben, hin

    # ------------------------------------------------------------- Zeichnen

    def rendern(self, W: int, H: int) -> np.ndarray:
        """Bild (H, W, 3) uint8 der aktuellen Ansicht."""
        bild = np.empty((H * W, 3), np.uint8)
        bild[:] = _HINTERGRUND
        belegt = np.zeros(H * W, bool)
        if self._massstab is None or not self._ebenen:
            return bild.reshape(H, W, 3)
        r, u, z = self._achsen()
        M = np.stack([r, u, z]).astype(np.float32)
        mitte = self._mitte.astype(np.float32)
        s = np.float32(self._massstab)
        for P, C in self._ebenen:
            Q = (P - mitte) @ M.T
            x = Q[:, 0] * s + np.float32(W / 2.0)
            y = np.float32(H / 2.0) - Q[:, 1] * s
            drin = (x >= 0) & (x < W) & (y >= 0) & (y < H)
            if not drin.any():
                continue
            lin = y[drin].astype(np.int64) * W + x[drin].astype(np.int64)
            tiefe = -Q[drin, 2]                    # klein = nah
            puffer = np.full(H * W, np.inf, np.float32)
            np.minimum.at(puffer, lin, tiefe)
            vorn = tiefe <= puffer[lin]
            bild[lin[vorn]] = C[drin][vorn]
            belegt[lin[vorn]] = True
        bild = bild.reshape(H, W, 3)
        belegt = belegt.reshape(H, W)
        # Punktgroesse: Luecken aus den Nachbarn fuellen, nie Gezeichnetes
        # ueberschreiben — die Fotopunkte bleiben so, wie sie liegen.
        for _ in range(max(0, int(self.punktgroesse) - 1)):
            neu_b, neu_c = belegt.copy(), bild.copy()
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                ys = slice(max(dy, 0), H + min(dy, 0))
                yd = slice(max(-dy, 0), H + min(-dy, 0))
                xs = slice(max(dx, 0), W + min(dx, 0))
                xd = slice(max(-dx, 0), W + min(-dx, 0))
                ziel = ~neu_b[yd, xd] & belegt[ys, xs]
                neu_c[yd, xd][ziel] = bild[ys, xs][ziel]
                neu_b[yd, xd] |= ziel
            bild, belegt = neu_c, neu_b
        return bild

    def _neu(self) -> None:
        self._schmutzig = True
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt)
        W, H = max(self.width(), 1), max(self.height(), 1)
        if self._schmutzig or self._bild is None or \
                (self._bild.width(), self._bild.height()) != (W, H):
            arr = np.ascontiguousarray(self.rendern(W, H))
            self._puffer = arr
            self._bild = QtGui.QImage(arr.data, W, H, 3 * W,
                                      QtGui.QImage.Format_RGB888)
            self._schmutzig = False
        p = QtGui.QPainter(self)
        p.drawImage(0, 0, self._bild)
        self._zeichne_massstab(p, W, H)
        hinweis = ("Mausrad: zoomen · links ziehen: drehen · rechts/Mitte "
                   "ziehen: verschieben · Doppelklick: einpassen")
        breite = p.fontMetrics().horizontalAdvance(hinweis)
        p.fillRect(0, H - 24, breite + 16, 24, QtGui.QColor(14, 16, 20, 200))
        p.setPen(QtGui.QColor(170, 175, 185))
        p.drawText(8, H - 8, hinweis)
        p.end()

    def _zeichne_massstab(self, p: QtGui.QPainter, W: int, H: int) -> None:
        """Massstabsbalken oben links — X und Y sind Meter, man soll sie sehen."""
        if not self._massstab:
            return
        ziel_m = 120.0 / self._massstab
        stufe = 10.0 ** np.floor(np.log10(ziel_m))
        laenge_m = max(m for m in (stufe, 2 * stufe, 5 * stufe) if m <= ziel_m)
        px = int(round(laenge_m * self._massstab))
        p.fillRect(4, 8, px + 70, 26, QtGui.QColor(14, 16, 20, 200))
        p.setPen(QtGui.QPen(QtGui.QColor(230, 230, 235), 2))
        p.drawLine(12, 22, 12 + px, 22)
        p.drawLine(12, 17, 12, 27)
        p.drawLine(12 + px, 17, 12 + px, 27)
        text = f"{laenge_m:g} m" if laenge_m >= 1 else f"{laenge_m * 100:g} cm"
        p.drawText(18 + px, 27, text)

    # ----------------------------------------------------------------- Maus

    def wheelEvent(self, event) -> None:  # noqa: N802 (Qt)
        stufen = event.angleDelta().y() / 120.0
        if stufen:
            pos = event.pos()
            self.zoom(1.2 ** stufen, (pos.x(), pos.y()))
        event.accept()

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt)
        self._maus = event.pos()
        links = event.button() == QtCore.Qt.LeftButton
        umschalt = bool(event.modifiers() & QtCore.Qt.ShiftModifier)
        self._modus_maus = "drehen" if links and not umschalt else "schieben"
        self.setCursor(QtCore.Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 (Qt)
        if self._maus is None:
            return
        d = event.pos() - self._maus
        self._maus = event.pos()
        if self._modus_maus == "drehen":
            self.drehen(-0.4 * d.x(), -0.4 * d.y())
        else:
            self.verschieben(d.x(), d.y())

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 (Qt)
        self._maus = None
        self.unsetCursor()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 (Qt)
        self.einpassen()

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt)
        k = event.key()
        if k in (QtCore.Qt.Key_Plus, QtCore.Qt.Key_Equal):
            self.zoom(1.25)
        elif k == QtCore.Qt.Key_Minus:
            self.zoom(0.8)
        else:
            super().keyPressEvent(event)


class MeanderAlignWindow(QtWidgets.QDialog):
    """Karte und Mäanderflug übereinander, live justierbar — RGB und Thermal.

    Rechts die Regler, je Wert ein grober und ein feiner Schieber. RGB: Gier,
    X, Y, Z als Zuschlag auf die gefundene Lage, dazu der Massstab (Brennweite).
    Thermal: Gier, X, Y als Zuschlag auf die RGB-Lage, dazu sein Massstab.
    """

    #: dict mit yaw, t (RGB-Lage), thermal (Zuschlag), rgb_faktor, thermal_faktor
    uebernommen = QtCore.pyqtSignal(object)

    def __init__(self, world: np.ndarray, pipe, live, live_th=None,
                 thermal_zuschlag=(0.0, 0.0, 0.0), thermal_da: bool = False,
                 optik: dict | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Mäanderflug ausrichten")
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.Window
                            | QtCore.Qt.WindowMaximizeButtonHint)
        self.resize(1400, 900)
        self._pipe = pipe
        self._live = live
        self._live_th = live_th
        self._thermal_da = bool(thermal_da or live_th is not None)
        optik = dict(optik or {})
        self._thermal_kal = optik.get("thermal")
        n = len(world)
        # Untergrund: die Karte ausgeduennt, damit die Fotopunkte darauf liegen
        self._karte = np.ascontiguousarray(
            world[:: max(1, n // _KARTE_PUNKTE)][:_KARTE_PUNKTE], dtype=np.float32)
        self._karte_farbe = _grau_nach_hoehe(self._karte)
        self._stich = np.ascontiguousarray(
            world[:: max(1, n // _FARB_PUNKTE)][:_FARB_PUNKTE], dtype=np.float64)
        self._basis_yaw = float(np.degrees(pipe.yaw))
        # as_t3, nicht [:3]: aus einem 2er bleibt sonst ein 2er, affine()
        # bricht ab und das Fenster bleibt schwarz.
        self._basis_t = meander_mod.as_t3(pipe.t)
        self._start = {"th": tuple(float(v) for v in thermal_zuschlag),
                       "rgb_faktor": float(optik.get("rgb_faktor", 1.0)),
                       "thermal_faktor": float(optik.get("thermal_faktor", 1.0))}

        haupt = QtWidgets.QHBoxLayout(self)
        links = QtWidgets.QVBoxLayout()
        self.ansicht = Ansicht(self)
        links.addWidget(self.ansicht, 1)
        self._lbl = QtWidgets.QLabel("")
        self._lbl.setWordWrap(True)
        links.addWidget(self._lbl)
        haupt.addLayout(links, 1)

        leiste = QtWidgets.QWidget()
        form = QtWidgets.QVBoxLayout(leiste)
        form.setContentsMargins(4, 0, 4, 0)
        rolle = QtWidgets.QScrollArea()
        rolle.setWidgetResizable(True)
        rolle.setWidget(leiste)
        rolle.setMinimumWidth(410)
        rolle.setMaximumWidth(480)
        haupt.addWidget(rolle)

        anzeige = QtWidgets.QFormLayout()
        self._modus = QtWidgets.QComboBox()
        self._modus.addItem("Überlagerung (Karte grau, Flug magenta)", "ueber")
        self._modus.addItem("Farbvorschau RGB", "farbe_rgb")
        self._modus.addItem("Farbvorschau Thermal", "farbe_th")
        if not self._thermal_da:
            self._modus.model().item(2).setEnabled(False)
        self._modus.currentIndexChanged.connect(lambda *_: self._neu())
        anzeige.addRow("Ansicht:", self._modus)
        blick = QtWidgets.QHBoxLayout()
        self._richtung = QtWidgets.QComboBox()
        for name, b in _BLICKE:
            self._richtung.addItem(name, b)
        self._richtung.activated.connect(
            lambda *_: self.ansicht.blick(*self._richtung.currentData()))
        blick.addWidget(self._richtung, 1)
        knopf_fit = QtWidgets.QPushButton("einpassen")
        knopf_fit.setToolTip("Ansicht so zoomen, dass Karte und Flug zu sehen sind "
                             "(auch: Doppelklick ins Bild).")
        knopf_fit.clicked.connect(self.ansicht.einpassen)
        blick.addWidget(knopf_fit)
        anzeige.addRow("Blick:", blick)
        self._punkt = QtWidgets.QSpinBox()
        self._punkt.setRange(1, 5)
        self._punkt.setValue(self.ansicht.punktgroesse)
        self._punkt.valueChanged.connect(self._punktgroesse)
        anzeige.addRow("Punktgröße:", self._punkt)
        form.addLayout(anzeige)

        # RGB: Zuschlag auf die gefundene Lage, Massstab absolut
        self._spins: dict = {}
        box = QtWidgets.QGroupBox("RGB — Zuschlag auf die gefundene Lage")
        f = QtWidgets.QFormLayout(box)
        for key, label, regler in (("yaw", "Gier", regler_grad()),
                                   ("x", "X", regler_meter()),
                                   ("y", "Y", regler_meter()),
                                   ("z", "Z (Höhe)", regler_meter(50.0))):
            regler.valueChanged.connect(lambda *_: self._angefasst("rgb"))
            f.addRow(label, regler)
            self._spins[key] = regler
        self._massstab_rgb = regler_prozent()
        self._massstab_rgb.setValue((self._start["rgb_faktor"] - 1.0) * 100.0)
        self._massstab_rgb.setToolTip(
            "Brennweite der RGB-Kamera gegenüber COLMAP, in Prozent.\n"
            "Zu kurz, und jedes Bild landet zu klein auf der Karte — am Rand\n"
            "um Meter, in jedem Bild anders. In der Seitenansicht liegen die\n"
            "Fotopunkte dann über der Karte; stimmt er, liegen sie darauf.")
        self._massstab_rgb.valueChanged.connect(lambda *_: self._angefasst("rgb"))
        f.addRow("Maßstab", self._massstab_rgb)
        form.addWidget(box)

        # Thermal: Zuschlag auf die RGB-Lage, Massstab auf die Thermaloptik
        self._spins_th: dict = {}
        box = QtWidgets.QGroupBox("Thermal — Zuschlag auf die RGB-Lage")
        f = QtWidgets.QFormLayout(box)
        for (key, label, regler), wert in zip((("yaw", "Gier", regler_grad()),
                                               ("x", "X", regler_meter()),
                                               ("y", "Y", regler_meter())),
                                              self._start["th"]):
            regler.setValue(wert)
            regler.valueChanged.connect(lambda *_: self._angefasst("thermal"))
            f.addRow(label, regler)
            self._spins_th[key] = regler
        self._massstab_th = regler_prozent()
        self._massstab_th.setValue((self._start["thermal_faktor"] - 1.0) * 100.0)
        self._massstab_th.setToolTip("Brennweite der Thermalkamera gegenüber ihrer "
                                     "Einmessung, in Prozent.")
        self._massstab_th.valueChanged.connect(lambda *_: self._angefasst("thermal"))
        f.addRow("Maßstab", self._massstab_th)
        self._lbl_kal = QtWidgets.QLabel(
            "Thermaloptik eingemessen." if self._thermal_kal else
            "Thermaloptik nicht eingemessen — im Hauptfenster „Optik einmessen“.")
        self._lbl_kal.setWordWrap(True)
        self._lbl_kal.setStyleSheet("color: #9aa0aa;")
        f.addRow(self._lbl_kal)
        form.addWidget(box)
        if not self._thermal_da:
            box.setEnabled(False)
            box.setToolTip("Keine Thermalbilder in diesem Lauf — im Hauptfenster "
                           "„Thermalbilder mitrechnen“ anhaken und neu ausrichten.")

        knopf_null = QtWidgets.QPushButton("zurücksetzen")
        knopf_null.setToolTip("Alle Regler auf den Stand beim Öffnen des Fensters.")
        knopf_null.clicked.connect(self._zuruecksetzen)
        form.addWidget(knopf_null)
        form.addStretch(1)

        knoepfe = QtWidgets.QDialogButtonBox()
        self._btn_ok = knoepfe.addButton("Lage übernehmen",
                                         QtWidgets.QDialogButtonBox.AcceptRole)
        knoepfe.addButton("Schließen", QtWidgets.QDialogButtonBox.RejectRole)
        knoepfe.accepted.connect(self._uebernehmen)
        knoepfe.rejected.connect(self.reject)
        form.addWidget(knoepfe)

        # Neu gerechnet wird erst nach kurzer Ruhe, sonst loest ein Ziehen am
        # Regler Dutzende Durchlaeufe aus. Drehen und Zoomen rechnet nichts
        # neu, das zeichnet nur das vorhandene Bild anders.
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(90)
        self._timer.timeout.connect(self._neu)

    # ------------------------------------------------------------------ Lage

    def lage(self) -> tuple:
        """RGB-Lage: (yaw_grad, t3) aus Basislage plus Reglerzuschlag."""
        t = self._basis_t.copy()
        t[0] += float(self._spins["x"].value())
        t[1] += float(self._spins["y"].value())
        t[2] += float(self._spins["z"].value())
        return self._basis_yaw + float(self._spins["yaw"].value()), t

    def thermal_zuschlag(self) -> tuple:
        return tuple(float(self._spins_th[k].value()) for k in ("yaw", "x", "y"))

    def lage_thermal(self) -> tuple:
        """Thermal-Lage: RGB-Lage plus Thermal-Zuschlag."""
        return meander_mod.thermal_lage(*self.lage(), self.thermal_zuschlag())

    def rgb_faktor(self) -> float:
        return 1.0 + float(self._massstab_rgb.value()) / 100.0

    def thermal_faktor(self) -> float:
        return 1.0 + float(self._massstab_th.value()) / 100.0

    def cams(self, thermal: bool) -> dict | None:
        """Kameras mit dem Massstab der Regler (und der Thermal-Einmessung)."""
        if thermal:
            return optik_mod.thermal_cams(self._pipe, self._thermal_kal,
                                          self.thermal_faktor(), self.rgb_faktor())
        return optik_mod.rgb_cams(self._pipe, self.rgb_faktor())

    def _angefasst(self, optik: str) -> None:
        # Wer an Thermal dreht, will Thermal sehen
        if optik == "thermal" and self._modus.currentData() == "farbe_rgb" \
                and self._thermal_da:
            self._modus.blockSignals(True)
            self._modus.setCurrentIndex(self._modus.findData("farbe_th"))
            self._modus.blockSignals(False)
        self._timer.start()

    def _zuruecksetzen(self) -> None:
        werte = [(self._spins[k], 0.0) for k in ("yaw", "x", "y", "z")]
        werte += list(zip((self._spins_th[k] for k in ("yaw", "x", "y")),
                          self._start["th"]))
        werte += [(self._massstab_rgb, (self._start["rgb_faktor"] - 1.0) * 100.0),
                  (self._massstab_th, (self._start["thermal_faktor"] - 1.0) * 100.0)]
        for regler, wert in werte:
            still_setzen(regler, wert)
        self._neu()

    def _punktgroesse(self, wert: int) -> None:
        self.ansicht.punktgroesse = int(wert)
        self.ansicht._neu()

    # ------------------------------------------------------------- Zeichnen

    def _neu(self) -> None:
        """Ansicht neu aufbauen. Faengt alles ab — ein Fehler in einem
        Timer-Slot wird von Qt sonst verschluckt und das Fenster bleibt
        einfach schwarz, ohne dass irgendwo etwas steht."""
        try:
            modus = self._modus.currentData()
            if modus == "ueber":
                self._zeige_ueberlagerung()
            else:
                self._zeige_farben(modus == "farbe_th")
        except Exception as exc:  # noqa: BLE001
            self._lbl.setText(f"Ansicht nicht aufbaubar: {type(exc).__name__}: {exc}")
            traceback.print_exc()

    def _foto_punkte(self, A, b) -> np.ndarray:
        """Fotopunkte des Fluges im Rahmen der Karte.

        Bevorzugt die Punktwolke aus der Rekonstruktion; fehlt sie, tun es die
        Kamerastandorte — Hauptsache, man sieht, wo der Flug liegt. Die Hoehe
        folgt dem RGB-Massstab: die Rekonstruktion lief mit fester Brennweite,
        und wie weit die Punkte unter den Kameras liegen, haengt an ihr.
        """
        cams = self._pipe.cams
        A, b = np.asarray(A), np.asarray(b)
        try:
            C = (A @ np.asarray(cams["C"], dtype=np.float64).T).T + b
        except Exception:  # noqa: BLE001
            C = np.empty((0, 3))
        for schluessel in ("xyz", "C"):
            try:
                P = np.asarray(cams[schluessel], dtype=np.float64)
            except Exception:  # noqa: BLE001
                continue
            if P.ndim == 2 and len(P):
                if len(P) > _FOTO_PUNKTE:
                    P = P[:: len(P) // _FOTO_PUNKTE + 1]
                P = (A @ P.T).T + b
                if schluessel == "xyz":
                    P = optik_mod.foto_hoehe(P, C, self.rgb_faktor())
                return P
        return np.empty((0, 3))

    def _zeige_ueberlagerung(self) -> None:
        yaw, t = self.lage()
        foto = self._foto_punkte(*meander_mod.lage_affine(self._pipe, yaw, t))
        ebenen = [(self._karte, self._karte_farbe)]
        zusatz = ""
        if self._thermal_da and any(self.thermal_zuschlag()):
            foto_th = self._foto_punkte(
                *meander_mod.lage_affine(self._pipe, *self.lage_thermal()))
            ebenen.append((foto_th, np.tile(_ORANGE, (len(foto_th), 1))))
            dg, dx, dy = self.thermal_zuschlag()
            zusatz = (f" Thermal (orange) liegt {dg:+.3f}°, {dx:+.2f}/{dy:+.2f} m "
                      f"daneben.")
        k = getattr(self._pipe, "s360_korrektur", None)
        if k:
            zusatz += (f" Feinausrichtung aktiv: Neigung {k['neigung_grad'][0]:+.2f}°/"
                       f"{k['neigung_grad'][1]:+.2f}°, Versatz {k['versatz_m'][0]:+.2f}/"
                       f"{k['versatz_m'][1]:+.2f} m.")
        ebenen.append((foto, np.tile(_MAGENTA, (len(foto), 1))))
        # Einpassen auf Karte UND Flug: liegt der Flug daneben, muss man ihn
        # trotzdem sehen, sonst sucht man ihn vergeblich.
        self.ansicht.setze_ebenen(
            ebenen, einpassen_an=np.vstack([self._karte, foto.astype(np.float32)])
            if len(foto) else self._karte)
        if len(foto):
            karte = self._karte
            abstand = float(np.linalg.norm(foto[:, :2].mean(0) - karte[:, :2].mean(0)))
            gross = float(np.linalg.norm(karte[:, :2].max(0) - karte[:, :2].min(0)))
            self._lbl.setText(
                f"RGB: Gier {yaw:.3f}°, Versatz {t[0]:+.2f}/{t[1]:+.2f}/{t[2]:+.2f} m, "
                f"Maßstab {self.rgb_faktor():.4f} — {len(foto)} Fotopunkte (magenta) "
                f"über {len(karte)} Kartenpunkten (grau, hell = hoch). Schwerpunkte "
                f"{abstand:.0f} m auseinander, Karte {gross:.0f} m breit."
                + ("  Der Flug liegt neben der Karte — schieben oder anderen Flug "
                   "wählen." if abstand > gross else "") + zusatz
                + "  Tipp: in der Seitenansicht zeigt sich, ob die Fotopunkte auf "
                  "der Karte liegen oder darüber schweben.")
        else:
            self._lbl.setText("Keine Fotopunkte im Modell gefunden.")

    def _zeige_farben(self, thermal: bool) -> None:
        live = self._live_th if thermal else self._live
        name = "Thermal" if thermal else "RGB"
        if live is None:
            self.ansicht.setze_ebenen([], einpassen_an=self._karte)
            self._lbl.setText(f"Für die Farbvorschau {name} fehlen die "
                              f"Vorschaubilder — sie laden nach „Ausrichten“ im "
                              f"Hauptfenster.")
            return
        yaw, t = self.lage_thermal() if thermal else self.lage()
        A, b = meander_mod.lage_affine(self._pipe, yaw, t)
        cams = self.cams(thermal)
        t0 = time.perf_counter()
        rgb, maske = live.colorize(self._stich, A, b, cams=cams)
        dt = (time.perf_counter() - t0) * 1000.0
        rgb = rgb.copy()
        rgb[~maske] = _UNGETROFFEN
        self.ansicht.setze_ebenen([(self._stich, rgb)], einpassen_an=self._karte)
        faktor = self.thermal_faktor() if thermal else self.rgb_faktor()
        self._lbl.setText(
            f"{name}: Gier {yaw:.3f}°, Versatz {t[0]:+.2f}/{t[1]:+.2f}/{t[2]:+.2f} m, "
            f"Maßstab {faktor:.4f} — {100.0 * maske.mean():.1f} % der "
            f"{len(self._stich)} Punkte getroffen ({dt:.0f} ms). Dunkelgrau ist von "
            f"keinem Bild getroffen.")

    # ----------------------------------------------------------------- Ende

    def showEvent(self, event) -> None:  # noqa: N802 (Qt)
        super().showEvent(event)
        if not getattr(self, "_erstmalig", False):
            self._erstmalig = True
            QtCore.QTimer.singleShot(60, self._erstes_bild)

    def _erstes_bild(self) -> None:
        self._neu()
        self.ansicht.einpassen()

    def ergebnis(self) -> dict:
        yaw, t = self.lage()
        return {"yaw": yaw, "t": t, "thermal": self.thermal_zuschlag(),
                "rgb_faktor": self.rgb_faktor(),
                "thermal_faktor": self.thermal_faktor()}

    def _uebernehmen(self) -> None:
        self.uebernommen.emit(self.ergebnis())
        self.accept()


if __name__ == "__main__":
    import os
    import sys

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication(sys.argv)

    class FakePipe:
        yaw = np.radians(30.0)
        t = np.array([1.0, 2.0, 0.5])
        thermal_versatz = (0.0, 0.0)
        cams = {"xyz": np.random.default_rng(0).uniform(-5, 5, size=(500, 3)),
                "C": np.random.default_rng(1).uniform(-5, 5, size=(20, 3)) + [0, 0, 50],
                "names": np.array(["a"]), "Rcw": np.eye(3)[None], "tcw": np.zeros((1, 3)),
                "size": np.array([[100.0, 100.0]]),
                "params": np.array([[50.0, 50.0, 50.0, 0.0]]),
                "model": np.array("SIMPLE_RADIAL")}

        def affine(self):
            from scipy.spatial.transform import Rotation
            A = Rotation.from_euler("z", self.yaw).as_matrix()
            return A, np.asarray(self.t, float)

        def rgb_cams(self):
            return self.cams

        def thermal_cams(self):
            return self.cams

    class FakeLive:
        """Faerbt nach x: links rot, rechts blau; merkt sich die Brennweite."""
        f = None

        def colorize(self, P, A, b, cams=None):
            FakeLive.f = None if cams is None else float(cams["params"][0][0])
            Pl = (np.linalg.inv(A) @ (np.asarray(P) - b).T).T
            rgb = np.where(Pl[:, :1] < 0, [200, 30, 30], [30, 30, 200]).astype(np.uint8)
            return rgb, np.abs(Pl[:, 0]) < 8

    def hell(arr):
        return float((arr.astype(int).sum(axis=2) > 90).mean())

    welt = np.random.default_rng(2).uniform(-10, 10, size=(50000, 3)).astype(np.float32)
    w = MeanderAlignWindow(welt, FakePipe(), FakeLive(), live_th=FakeLive(),
                           thermal_zuschlag=(0.0, 0.0, 0.0),
                           optik={"rgb_faktor": 1.1, "thermal_faktor": 1.0})
    w.resize(1300, 800)
    w.show()
    app.processEvents()
    assert abs(w.lage()[0] - 30.0) < 1e-9
    assert np.allclose(w.lage()[1], [1.0, 2.0, 0.5])
    assert abs(w.rgb_faktor() - 1.1) < 1e-9, w.rgb_faktor()
    w._spins["x"].setValue(3.0)
    w._spins["yaw"].setValue(-5.0)
    w._spins["z"].setValue(-0.37)          # Zentimeter ueber den Feinschieber
    yaw, t = w.lage()
    assert abs(yaw - 25.0) < 1e-9 and np.allclose(t, [4.0, 2.0, 0.13]), (yaw, t)
    print(f"Lage mit Zuschlag: Gier {yaw:.2f}, t {np.round(t, 2).tolist()}")
    assert abs(np.degrees(FakePipe.yaw) - 30.0) < 1e-9, "Basislage wurde veraendert"

    # Thermal: Zuschlag kommt auf die RGB-Lage
    w._spins_th["x"].setValue(1.5)
    w._spins_th["yaw"].setValue(2.0)
    yaw_t, t_t = w.lage_thermal()
    assert abs(yaw_t - 27.0) < 1e-9 and np.allclose(t_t, [5.5, 2.0, 0.13]), (yaw_t, t_t)
    print(f"Thermal-Lage: Gier {yaw_t:.2f}, t {np.round(t_t, 2).tolist()}")

    w._neu()
    print("Überlagerung:", w._lbl.text()[:110])
    assert "magenta" in w._lbl.text() and "Thermal (orange)" in w._lbl.text()
    W, H = w.ansicht.width(), w.ansicht.height()
    bild = w.ansicht.rendern(W, H)
    orange = ((bild[..., 0] == 255) & (bild[..., 1] == 150)).sum()
    magenta = ((bild[..., 0] == 255) & (bild[..., 2] == 200)).sum()
    print(f"Bild {W}x{H}, {100 * hell(bild):.0f} % gezeichnet, "
          f"{magenta} px magenta, {orange} px orange")
    assert hell(bild) > 0.05, "Bild ist fast schwarz"
    assert magenta > 0 and orange > 0, "Fotopunkte fehlen im Bild"

    # Massstab senkt die Fotopunkte (sie liegen bei zu kurzer Brennweite zu hoch)
    A, b = meander_mod.lage_affine(w._pipe, *w.lage())
    z1 = w._foto_punkte(A, b)[:, 2].mean()
    w._massstab_rgb.setValue(0.0)
    z0 = w._foto_punkte(A, b)[:, 2].mean()
    assert z1 < z0 - 1.0, (z1, z0)
    w._massstab_rgb.setValue(10.0)
    print(f"Maßstab 1.10 senkt die Fotopunkte um {z0 - z1:.1f} m")

    # Navigation: zoomen, verschieben, drehen aendern das Bild, nicht die Lage
    s0, m0 = w.ansicht._massstab, w.ansicht._mitte.copy()
    w.ansicht.zoom(2.0, (W / 2, H / 2))
    assert abs(w.ansicht._massstab - 2 * s0) < 1e-6
    assert np.allclose(w.ansicht._mitte, m0), "Zoom zur Mitte hat verschoben"
    w.ansicht.zoom(1.5, (0, 0))
    assert not np.allclose(w.ansicht._mitte, m0), "Zoom zum Rand bleibt stehen"
    vorher = w.ansicht.rendern(W, H)
    w.ansicht.verschieben(80, 0)
    nachher = w.ansicht.rendern(W, H)
    assert not np.array_equal(vorher, nachher), "Verschieben ohne Wirkung"
    w.ansicht.drehen(0.0, -60.0)
    assert abs(w.ansicht._el - 30.0) < 1e-9
    schraeg = w.ansicht.rendern(W, H)
    assert hell(schraeg) > 0.01 and not np.array_equal(schraeg, nachher)
    w.ansicht.blick(90.0, 0.0)
    seite = w.ansicht.rendern(W, H)
    assert hell(seite) > 0.05, "Seitenansicht leer"
    assert abs(yaw - w.lage()[0]) < 1e-9, "Navigation hat die Lage geaendert"
    t0 = time.perf_counter()
    for _ in range(5):
        w.ansicht.rendern(W, H)
    print(f"Navigation: zoomen, schieben, drehen wirken; "
          f"{(time.perf_counter() - t0) * 200:.0f} ms je Bild")

    # Ansicht bleibt stehen, wenn die Lage sich aendert
    w.ansicht.blick(0.0, 90.0)
    w.ansicht.zoom(3.0, (W / 2, H / 2))
    s1, m1 = w.ansicht._massstab, w.ansicht._mitte.copy()
    w._spins["x"].setValue(5.0)
    w._neu()
    assert w.ansicht._massstab == s1 and np.allclose(w.ansicht._mitte, m1), \
        "Justieren hat die Ansicht versetzt"
    print("Justieren laesst die Ansicht stehen")

    # Farbvorschau: RGB und Thermal einzeln, mit dem Massstab der Regler
    w._modus.setCurrentIndex(w._modus.findData("farbe_rgb"))
    assert "RGB:" in w._lbl.text() and "getroffen" in w._lbl.text(), w._lbl.text()
    assert abs(FakeLive.f - 55.0) < 1e-9, FakeLive.f       # 50 px * 1.10
    w._spins_th["y"].setValue(1.0)          # Thermal angefasst -> Thermal zeigen
    w._massstab_th.setValue(-2.0)
    w._neu()
    assert w._modus.currentData() == "farbe_th", w._modus.currentData()
    assert "Thermal:" in w._lbl.text(), w._lbl.text()
    assert abs(FakeLive.f - 49.0) < 1e-9, FakeLive.f       # ohne Einmessung: EXIF * 0.98
    print("Farbvorschau:", w._lbl.text()[:90])

    empfangen = []
    w.uebernommen.connect(empfangen.append)
    w._uebernehmen()
    e = empfangen[0]
    assert e["thermal"] == (2.0, 1.5, 1.0) and abs(e["rgb_faktor"] - 1.1) < 1e-9 \
        and abs(e["thermal_faktor"] - 0.98) < 1e-9, e
    print("übernehmen liefert Lage, Thermal-Zuschlag und beide Maßstäbe")

    w._zuruecksetzen()
    assert abs(w.lage()[0] - 30.0) < 1e-9 and w.thermal_zuschlag() == (0.0, 0.0, 0.0)
    assert abs(w.rgb_faktor() - 1.1) < 1e-9 and abs(w.thermal_faktor() - 1.0) < 1e-9
    print("zurücksetzen stellt den Stand beim Öffnen wieder her")

    # Alte align.json: t nur mit x und y. Das Fenster muss trotzdem zeichnen.
    class AltePipe(FakePipe):
        t = np.array([1.0, 2.0])

        def affine(self):
            from colorize_pipeline import register as reg
            return reg.affine(self.yaw, self.t, 1.0, np.eye(3), np.zeros(3))

    meander_mod.find_pipeline()
    w2 = MeanderAlignWindow(welt, AltePipe(), None)
    w2.resize(1100, 700)
    w2.show()
    app.processEvents()
    assert np.allclose(w2.lage()[1], [1.0, 2.0, 0.0]), w2.lage()[1]
    w2._neu()
    assert "nicht aufbaubar" not in w2._lbl.text(), w2._lbl.text()
    assert hell(w2.ansicht.rendern(700, 500)) > 0.05, "2er-t: Bild schwarz"
    assert not w2._spins_th["x"].isEnabledTo(w2), "Thermal ohne Bilder bedienbar"
    print("2er-Verschiebung aus alter align.json wird aufgefüllt, Bild da; "
          "ohne Thermalbilder sind die Thermalregler aus")
    print("meander_align_window SELFTEST OK")
