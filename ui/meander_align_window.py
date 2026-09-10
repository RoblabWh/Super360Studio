"""Eigenes Fenster zum Ausrichten des Maeanderfluges auf die Karte.

Nach dem Vorbild der Pipeline aus PointCloudMerger: beide Wolken **uebereinander**
sehen und die eine von Hand auf die andere schieben. Dass das ein eigenes Fenster
bekommt, hat einen Grund — im Hauptfenster musste die Karte fuer die Vorschau
ausgeblendet werden, weil eine Stichprobe von 50.000 Punkten in 24 Millionen
untergeht. Hier ist das Ueberlagern der Zweck, nicht die Stoerung: die Karte
liegt bewusst ausgeduennt darunter, damit die Fotopunkte darauf zu sehen sind.

Zwei Ansichten, umschaltbar:

* **Überlagerung** — Karte in Grau, die Fotopunkte des Fluges in Magenta. Damit
  sieht man auf einen Blick, ob der Flug ueberhaupt ueber diesem Gebiet lag und
  wie weit die Lage daneben liegt.
* **Farbvorschau** — die Stichprobe der Karte, eingefaerbt mit der aktuellen
  Lage. Damit sieht man, ob die Farben auf den richtigen Strukturen landen.

Gier, X und Y wirken in beiden sofort. Das Hauptfenster bleibt unberuehrt.

Gezeichnet wird als **Bild**, nicht mit einem zweiten 3D-Fenster: zwei
OpenGL-Kontexte in einer Anwendung sind je nach Grafiktreiber und Sitzung eine
Quelle schwarzer Fenster, und fuers Ausrichten reicht der Blick von oben. Ein
Durchlauf ueber 360.000 Punkte kostet wenige Millisekunden, das Bild folgt dem
Regler also genauso wie eine 3D-Ansicht — nur zuverlaessig.
"""

from __future__ import annotations

import os
import sys
import time
import traceback

import numpy as np
from PyQt5 import QtCore, QtWidgets

from PyQt5 import QtGui

_KARTE_PUNKTE = 300_000      # so viel Karte liegt als Untergrund darunter
_FOTO_PUNKTE = 60_000        # ... und so viele Fotopunkte darueber. Mehr
                             # deckt die Karte zu, statt sie zu zeigen.
_GRAU = np.array([120, 124, 130], np.uint8)
_MAGENTA = np.array([255, 0, 200], np.uint8)
_UNGETROFFEN = np.array([60, 60, 64], np.uint8)


class MeanderAlignWindow(QtWidgets.QDialog):
    """Karte und Mäanderflug übereinander, live justierbar."""

    #: (yaw_grad, t3) wurde übernommen
    uebernommen = QtCore.pyqtSignal(float, object)

    def __init__(self, world: np.ndarray, pipe, live, stichprobe: np.ndarray,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Mäanderflug ausrichten")
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.Window)
        self.resize(1150, 780)
        self._pipe = pipe
        self._live = live
        self._stich = np.asarray(stichprobe, dtype=np.float64)
        # Untergrund: die Karte ausgeduennt, damit die Fotopunkte darauf liegen
        n = len(world)
        schritt = max(1, n // _KARTE_PUNKTE)
        self._karte = np.ascontiguousarray(world[::schritt][:_KARTE_PUNKTE],
                                           dtype=np.float32)
        self._basis_yaw = float(np.degrees(pipe.yaw))
        self._basis_t = np.asarray(pipe.t, dtype=float).ravel()[:3].copy()

        lay = QtWidgets.QVBoxLayout(self)
        self.bild = QtWidgets.QLabel("")
        self.bild.setAlignment(QtCore.Qt.AlignCenter)
        self.bild.setMinimumSize(640, 420)
        self.bild.setStyleSheet("background: #14161c;")
        lay.addWidget(self.bild, 1)

        leiste = QtWidgets.QHBoxLayout()
        self._modus = QtWidgets.QComboBox()
        self._modus.addItem("Überlagerung (Karte grau, Flug magenta)", "ueber")
        self._modus.addItem("Farbvorschau (Karte eingefärbt)", "farbe")
        self._modus.currentIndexChanged.connect(lambda *_: self._neu())
        leiste.addWidget(QtWidgets.QLabel("Ansicht:"))
        leiste.addWidget(self._modus, 1)
        self._richtung = QtWidgets.QComboBox()
        self._richtung.addItem("von oben", (0, 1))
        self._richtung.addItem("von vorn", (0, 2))
        self._richtung.addItem("von der Seite", (1, 2))
        self._richtung.currentIndexChanged.connect(lambda *_: self._neu())
        leiste.addWidget(self._richtung)

        self._spins: dict = {}
        for key, label, rng, step, suffix in (("yaw", "Gier", 180.0, 0.5, "°"),
                                              ("x", "X", 500.0, 0.5, " m"),
                                              ("y", "Y", 500.0, 0.5, " m")):
            sp = QtWidgets.QDoubleSpinBox()
            sp.setRange(-rng, rng)
            sp.setSingleStep(step)
            sp.setDecimals(2)
            sp.setSuffix(suffix)
            sp.setToolTip("Zuschlag auf die gefundene Lage; wirkt sofort.")
            sp.valueChanged.connect(self._angefasst)
            leiste.addWidget(QtWidgets.QLabel(label))
            leiste.addWidget(sp)
            self._spins[key] = sp
        knopf_null = QtWidgets.QPushButton("zurücksetzen")
        knopf_null.clicked.connect(self._zuruecksetzen)
        leiste.addWidget(knopf_null)
        lay.addLayout(leiste)

        self._lbl = QtWidgets.QLabel("")
        self._lbl.setWordWrap(True)
        lay.addWidget(self._lbl)

        knoepfe = QtWidgets.QDialogButtonBox(self)
        self._btn_ok = knoepfe.addButton("Lage übernehmen",
                                         QtWidgets.QDialogButtonBox.AcceptRole)
        knoepfe.addButton("Schließen", QtWidgets.QDialogButtonBox.RejectRole)
        knoepfe.accepted.connect(self._uebernehmen)
        knoepfe.rejected.connect(self.reject)
        lay.addWidget(knoepfe)

        # Neu gerechnet wird erst nach kurzer Ruhe, sonst loest ein Ziehen am
        # Regler Dutzende Durchlaeufe aus.
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(110)
        self._timer.timeout.connect(self._neu)

    # ------------------------------------------------------------------ Lage

    def lage(self) -> tuple:
        """(yaw_grad, t3) aus Basislage plus Reglerzuschlag."""
        t = self._basis_t.copy()
        t[0] += float(self._spins["x"].value())
        t[1] += float(self._spins["y"].value())
        return self._basis_yaw + float(self._spins["yaw"].value()), t

    def _angefasst(self) -> None:
        self._timer.start()

    def _zuruecksetzen(self) -> None:
        for sp in self._spins.values():
            sp.blockSignals(True)
            sp.setValue(0.0)
            sp.blockSignals(False)
        self._neu()

    def _affin(self):
        yaw, t = self.lage()
        alt_yaw, alt_t = self._pipe.yaw, np.asarray(self._pipe.t, float).copy()
        try:
            self._pipe.yaw = float(np.radians(yaw))
            self._pipe.t = t
            return self._pipe.affine()
        finally:
            self._pipe.yaw, self._pipe.t = alt_yaw, alt_t

    # ------------------------------------------------------------- Zeichnen

    def _neu(self) -> None:
        """Ansicht neu aufbauen. Faengt alles ab — ein Fehler in einem
        Timer-Slot wird von Qt sonst verschluckt und das Fenster bleibt
        einfach schwarz, ohne dass irgendwo etwas steht."""
        try:
            A, b = self._affin()
            yaw, t = self.lage()
            if self._modus.currentData() == "ueber":
                self._zeige_ueberlagerung(A, b, yaw, t)
            else:
                self._zeige_farben(A, b, yaw, t)
        except Exception as exc:  # noqa: BLE001
            self._lbl.setText(f"Ansicht nicht aufbaubar: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return


    def _foto_punkte(self, A, b) -> np.ndarray:
        """Fotopunkte des Fluges im Rahmen der Karte.

        Bevorzugt die Punktwolke aus der Rekonstruktion; fehlt sie, tun es die
        Kamerastandorte — Hauptsache, man sieht, wo der Flug liegt.
        """
        cams = self._pipe.cams
        for schluessel in ("xyz", "C"):
            try:
                P = np.asarray(cams[schluessel], dtype=np.float64)
            except Exception:  # noqa: BLE001
                continue
            if P.ndim == 2 and len(P):
                if len(P) > _FOTO_PUNKTE:
                    P = P[:: len(P) // _FOTO_PUNKTE + 1]
                return (np.asarray(A) @ P.T).T + np.asarray(b)
        return np.empty((0, 3))

    def _zeige_ueberlagerung(self, A, b, yaw, t) -> None:
        foto = self._foto_punkte(A, b)
        karte = self._karte
        pts = np.vstack([karte, foto.astype(np.float32)]) if len(foto) else karte
        farben = np.vstack([np.tile(_GRAU, (len(karte), 1)),
                            np.tile(_MAGENTA, (len(foto), 1))]) if len(foto) \
            else np.tile(_GRAU, (len(karte), 1))
        self._male(pts, farben)
        if len(foto):
            mitte_k = karte[:, :2].mean(0)
            mitte_f = foto[:, :2].mean(0)
            abstand = float(np.linalg.norm(mitte_f - mitte_k))
            gross = float(np.linalg.norm(karte[:, :2].max(0) - karte[:, :2].min(0)))
            self._lbl.setText(
                f"Gier {yaw:.2f}°, Versatz {t[0]:+.1f}/{t[1]:+.1f} m — "
                f"{len(foto)} Fotopunkte (magenta) über {len(karte)} Kartenpunkten "
                f"(grau). Schwerpunkte {abstand:.0f} m auseinander, Karte {gross:.0f} m "
                f"breit."
                + ("  Der Flug liegt neben der Karte — schieben oder anderen Flug "
                   "wählen." if abstand > gross else ""))
        else:
            self._lbl.setText("Keine Fotopunkte im Modell gefunden.")

    def _zeige_farben(self, A, b, yaw, t) -> None:
        if self._live is None:
            self._lbl.setText("Für die Farbvorschau fehlen die Vorschaubilder.")
            return
        t0 = time.perf_counter()
        rgb, maske = self._live.colorize(self._stich, A, b)
        dt = (time.perf_counter() - t0) * 1000.0
        rgb = rgb.copy()
        rgb[~maske] = _UNGETROFFEN
        self._male(self._stich.astype(np.float32), rgb)
        self._lbl.setText(
            f"Gier {yaw:.2f}°, Versatz {t[0]:+.1f}/{t[1]:+.1f} m — "
            f"{100.0 * maske.mean():.1f} % der {len(self._stich)} Punkte getroffen "
            f"({dt:.0f} ms). Dunkelgrau ist von keinem Bild getroffen.")

    def _male(self, punkte: np.ndarray, farben: np.ndarray) -> None:
        """Punkte als Bild zeichnen (Parallelprojektion auf zwei Achsen).

        Jeder Bildpunkt bekommt den Mittelwert der Farben, die auf ihn fallen.
        Bei 360.000 Punkten sind das wenige Millisekunden, das reicht fuer eine
        Anzeige, die dem Regler folgt.
        """
        au, av = self._richtung.currentData()
        W = max(self.bild.width(), 320)
        H = max(self.bild.height(), 240)
        P = np.asarray(punkte, dtype=np.float32)
        C = np.asarray(farben, dtype=np.float32)
        if len(P) == 0:
            self.bild.setPixmap(QtGui.QPixmap())
            return
        lo = np.percentile(P[:, [au, av]], 0.5, axis=0)
        hi = np.percentile(P[:, [au, av]], 99.5, axis=0)
        spanne = np.maximum(hi - lo, 1e-6)
        # Seitenverhaeltnis halten, sonst verzerrt die Ansicht
        s = min((W - 8) / spanne[0], (H - 8) / spanne[1])
        xi = np.clip(((P[:, au] - lo[0]) * s + 4).astype(np.int32), 0, W - 1)
        yi = np.clip((H - 5 - (P[:, av] - lo[1]) * s).astype(np.int32), 0, H - 1)
        summe = np.zeros((H, W, 3), np.float32)
        anzahl = np.zeros((H, W), np.float32)
        np.add.at(summe, (yi, xi), C)
        np.add.at(anzahl, (yi, xi), 1.0)
        nz = anzahl > 0
        bild = np.full((H, W, 3), 20, np.uint8)
        bild[nz] = (summe[nz] / anzahl[nz][:, None]).astype(np.uint8)
        bild = np.ascontiguousarray(bild)
        qi = QtGui.QImage(bild.data, W, H, 3 * W, QtGui.QImage.Format_RGB888)
        self._bildpuffer = bild          # QImage haelt keine Kopie
        self.bild.setPixmap(QtGui.QPixmap.fromImage(qi))

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt)
        super().resizeEvent(event)
        self._timer.start()

    # ----------------------------------------------------------------- Ende

    def showEvent(self, event) -> None:  # noqa: N802 (Qt)
        super().showEvent(event)
        if not getattr(self, "_erstmalig", False):
            self._erstmalig = True
            QtCore.QTimer.singleShot(60, self._erstes_bild)

    def _erstes_bild(self) -> None:
        self._neu()

    def _uebernehmen(self) -> None:
        yaw, t = self.lage()
        self.uebernommen.emit(yaw, t)
        self.accept()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)

    class FakePipe:
        yaw = np.radians(30.0)
        t = np.array([1.0, 2.0, 0.5])
        cams = {"xyz": np.random.default_rng(0).uniform(-5, 5, size=(500, 3)),
                "C": np.random.default_rng(1).uniform(-5, 5, size=(20, 3))}

        def affine(self):
            from scipy.spatial.transform import Rotation
            A = Rotation.from_euler("z", self.yaw).as_matrix()
            return A, np.asarray(self.t, float)

    welt = np.random.default_rng(2).uniform(-10, 10, size=(50000, 3)).astype(np.float32)
    w = MeanderAlignWindow(welt, FakePipe(), None, welt[::10])
    assert abs(w.lage()[0] - 30.0) < 1e-9
    assert np.allclose(w.lage()[1], [1.0, 2.0, 0.5])
    w._spins["x"].setValue(3.0)
    w._spins["yaw"].setValue(-5.0)
    yaw, t = w.lage()
    assert abs(yaw - 25.0) < 1e-9 and np.allclose(t, [4.0, 2.0, 0.5]), (yaw, t)
    print(f"Lage mit Zuschlag: Gier {yaw:.2f}, t {np.round(t, 2).tolist()}")
    A, b = w._affin()
    assert abs(np.degrees(FakePipe.yaw) - 30.0) < 1e-9, "Basislage wurde veraendert"
    foto = w._foto_punkte(A, b)
    assert len(foto) == 500, len(foto)
    w._zeige_ueberlagerung(A, b, yaw, t)
    print("Überlagerung:", w._lbl.text()[:96])
    assert "magenta" in w._lbl.text()
    pm = w.bild.pixmap()
    assert pm is not None and not pm.isNull(), "kein Bild gezeichnet"
    img = pm.toImage()
    px = [img.pixelColor(x, y) for y in range(0, img.height(), 7)
          for x in range(0, img.width(), 7)]
    hell = sum(1 for c in px if c.red() + c.green() + c.blue() > 90)
    print(f"Bild {img.width()}x{img.height()}, {100 * hell / len(px):.0f} % der "
          f"Stichprobenpixel gezeichnet")
    assert hell > 0, "Bild ist komplett schwarz"
    w._zuruecksetzen()
    assert abs(w.lage()[0] - 30.0) < 1e-9
    print("zurücksetzen stellt die Basislage wieder her")
    print("meander_align_window SELFTEST OK")
