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
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
from PyQt5 import QtCore, QtWidgets

try:  # Paket-Import (App) vs. Direktstart des Selbsttests
    from ui.cloud_view import CloudView
except ImportError:  # pragma: no cover
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ui.cloud_view import CloudView

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
        self.view = CloudView(self)
        lay.addWidget(self.view, 1)

        leiste = QtWidgets.QHBoxLayout()
        self._modus = QtWidgets.QComboBox()
        self._modus.addItem("Überlagerung (Karte grau, Flug magenta)", "ueber")
        self._modus.addItem("Farbvorschau (Karte eingefärbt)", "farbe")
        self._modus.currentIndexChanged.connect(lambda *_: self._neu())
        leiste.addWidget(QtWidgets.QLabel("Ansicht:"))
        leiste.addWidget(self._modus, 1)

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
        try:
            A, b = self._affin()
        except Exception as exc:  # noqa: BLE001
            self._lbl.setText(f"Lage nicht berechenbar: {exc}")
            return
        yaw, t = self.lage()
        if self._modus.currentData() == "ueber":
            self._zeige_ueberlagerung(A, b, yaw, t)
        else:
            self._zeige_farben(A, b, yaw, t)

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
        self.view.set_cloud(pts, farben, None, np.ones(len(pts), bool))
        self.view.set_color_mode("rgb")
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
        self.view.set_cloud(self._stich.astype(np.float32), rgb, None,
                            np.ones(len(self._stich), bool))
        self.view.set_color_mode("rgb")
        self._lbl.setText(
            f"Gier {yaw:.2f}°, Versatz {t[0]:+.1f}/{t[1]:+.1f} m — "
            f"{100.0 * maske.mean():.1f} % der {len(self._stich)} Punkte getroffen "
            f"({dt:.0f} ms). Dunkelgrau ist von keinem Bild getroffen.")

    # ----------------------------------------------------------------- Ende

    def showEvent(self, event) -> None:  # noqa: N802 (Qt)
        super().showEvent(event)
        if not getattr(self, "_erstmalig", False):
            self._erstmalig = True
            QtCore.QTimer.singleShot(60, self._erstes_bild)

    def _erstes_bild(self) -> None:
        self._neu()
        self.view.reset_camera()

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
    w._zuruecksetzen()
    assert abs(w.lage()[0] - 30.0) < 1e-9
    print("zurücksetzen stellt die Basislage wieder her")
    print("meander_align_window SELFTEST OK")
