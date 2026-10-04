"""Regler aus zwei Schiebern: grob fuer den Weg, fein fuer den letzten Zentimeter.

Ein einzelner Schieber kann nicht beides. Soll er 250 m abdecken, ist ein
Pixel Mausweg einen halben Meter — zum Uebereinanderlegen viel zu grob. Soll er
Zentimeter koennen, kommt man nie an. Also zwei, die sich addieren: der grobe
bringt die Lage in die Naehe, der feine legt sie auf den Zentimeter hin. Das
Zahlenfeld daneben zeigt die Summe und nimmt einen getippten Wert an.

Nach aussen verhaelt sich der Regler wie ein ``QDoubleSpinBox`` — ``value``,
``setValue``, ``valueChanged``, ``blockSignals``, ``setEnabled`` — und kann ihn
darum ersetzen, ohne dass die Stellen, die ihn lesen, etwas merken.

Die Rasterfunktionen (``auf_raster``, ``raster_grad`` …) rechnen ohne Qt genau
das, was der Regler nach ``setValue`` zeigt: Wer einen Wert ohne Regler haelt,
legt ihn damit auf dasselbe Raster.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets


def _stufen(grob: float, grob_schritt: float, fein: float,
            fein_schritt: float) -> tuple[float, int, float, int]:
    """Schrittweiten und Stufenzahl je Schieber: (Grobschritt, Grobstufen, Feinschritt, Feinstufen)."""
    return (float(grob_schritt), int(round(grob / grob_schritt)),
            float(fein_schritt), int(round(fein / fein_schritt)))


def _zerlegen(wert: float, gs: float, ng: int, fs: float, nf: int) -> tuple[int, int]:
    """Wert auf die beiden Schieber verteilen; am Anschlag ±(ng*gs + nf*fs)."""
    grenze = ng * gs + nf * fs
    wert = max(-grenze, min(grenze, wert))
    g = int(round(wert / gs))
    g = max(-ng, min(ng, g))
    f = int(round((wert - g * gs) / fs))
    f = max(-nf, min(nf, f))
    return g, f


def auf_raster(wert: float, grob: float, grob_schritt: float, fein: float,
               fein_schritt: float) -> float:
    """Der Wert, den ein FeinRegler mit diesen Zahlen nach ``setValue(wert)`` zeigt."""
    gs, ng, fs, nf = _stufen(grob, grob_schritt, fein, fein_schritt)
    g, f = _zerlegen(float(wert), gs, ng, fs, nf)
    return g * gs + f * fs


class FeinRegler(QtWidgets.QWidget):
    """Grob- plus Feinschieber mit Zahlenfeld; der Wert ist die Summe."""

    valueChanged = QtCore.pyqtSignal(float)

    def __init__(self, grob: float, grob_schritt: float, fein: float,
                 fein_schritt: float, einheit: str = "", dezimalen: int = 2,
                 fein_text: str | None = None, parent=None):
        super().__init__(parent)
        self._gs, self._ng, self._fs, self._nf = _stufen(grob, grob_schritt,
                                                         fein, fein_schritt)
        self._max = self._ng * self._gs + self._nf * self._fs
        self._still = False

        self._grob = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._grob.setRange(-self._ng, self._ng)
        self._grob.setPageStep(max(1, self._ng // 20))
        self._fein = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._fein.setRange(-self._nf, self._nf)
        self._fein.setPageStep(max(1, self._nf // 10))
        for s in (self._grob, self._fein):
            s.setMinimumWidth(70)
            s.valueChanged.connect(self._geschoben)
        self._grob.setToolTip(f"grob: ±{grob:g}{einheit}, Schritt {grob_schritt:g}{einheit}")
        self._fein.setToolTip(f"fein: ±{fein:g}{einheit}, Schritt {fein_schritt:g}{einheit}"
                              " — für das letzte Stück")

        self._zahl = QtWidgets.QDoubleSpinBox()
        self._zahl.setRange(-self._max, self._max)
        self._zahl.setDecimals(dezimalen)
        self._zahl.setSingleStep(self._fs)
        self._zahl.setSuffix(einheit)
        self._zahl.setKeyboardTracking(False)
        self._zahl.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self._zahl.setAlignment(QtCore.Qt.AlignRight)
        self._zahl.setMinimumWidth(78)
        self._zahl.valueChanged.connect(self._getippt)

        null = QtWidgets.QToolButton()
        null.setText("↺")
        null.setToolTip("auf null zurück")
        null.setAutoRaise(True)
        null.clicked.connect(lambda: self._setzen(0.0, melden=True))

        lay = QtWidgets.QGridLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setHorizontalSpacing(4)
        lay.setVerticalSpacing(0)
        klein = self.font()
        klein.setPointSizeF(max(6.0, klein.pointSizeF() * 0.8))
        for zeile, (text, schieber) in enumerate(((("grob"), self._grob),
                                                  (fein_text or "fein", self._fein))):
            lbl = QtWidgets.QLabel(text)
            lbl.setFont(klein)
            lbl.setStyleSheet("color: #9aa0aa;")
            lay.addWidget(lbl, zeile, 0)
            lay.addWidget(schieber, zeile, 1)
        lay.addWidget(self._zahl, 0, 2)
        lay.addWidget(null, 1, 2, QtCore.Qt.AlignRight)
        lay.setColumnStretch(1, 1)

    # ------------------------------------------------------------ Wert

    def value(self) -> float:
        return self._grob.value() * self._gs + self._fein.value() * self._fs

    def setValue(self, wert: float) -> None:  # noqa: N802 (Qt-Name)
        self._setzen(float(wert), melden=True)

    def maximum(self) -> float:
        return self._max

    def _setzen(self, wert: float, melden: bool) -> None:
        g, f = _zerlegen(wert, self._gs, self._ng, self._fs, self._nf)
        alt = self.value()
        self._still = True
        try:
            self._grob.setValue(g)
            self._fein.setValue(f)
            self._zahl.setValue(self.value())
        finally:
            self._still = False
        if melden and self.value() != alt:
            self.valueChanged.emit(self.value())

    def _geschoben(self, *_a) -> None:
        if self._still:
            return
        self._still = True
        try:
            self._zahl.setValue(self.value())
        finally:
            self._still = False
        self.valueChanged.emit(self.value())

    def _getippt(self, wert: float) -> None:
        if self._still:
            return
        self._setzen(float(wert), melden=True)


# Raster der Regler: (grob, Grobschritt, fein, Feinschritt); grob ist die Vorgabe
_METER = (250.0, 0.1, 1.0, 0.01)
_GRAD = (180.0, 0.1, 1.0, 0.005)
_PROZENT = (20.0, 0.1, 1.0, 0.01)


def regler_meter(grob: float = _METER[0]) -> FeinRegler:
    """Verschiebung in Metern: grob in Dezimetern, fein in Zentimetern (±1 m)."""
    return FeinRegler(grob, *_METER[1:], " m", 2, "fein cm")


def regler_grad(grob: float = _GRAD[0]) -> FeinRegler:
    """Winkel: grob in Zehntelgrad, fein in 0,005° (bei 50 m unter einem Zentimeter)."""
    return FeinRegler(grob, *_GRAD[1:], "°", 3)


def regler_prozent(grob: float = _PROZENT[0]) -> FeinRegler:
    """Massstab in Prozent: grob in 0,1 %, fein in 0,01 %."""
    return FeinRegler(grob, *_PROZENT[1:], " %", 2)


def regler_pixel(grob: float = 400.0) -> FeinRegler:
    """Hauptpunkt in Pixeln: grob ganze Pixel, fein Zwanzigstel."""
    return FeinRegler(grob, 1.0, 5.0, 0.05, " px", 2)


def regler_millimeter() -> FeinRegler:
    """Kurze Strecke in Metern (±2 m): grob in Zentimetern, fein in Millimetern."""
    return FeinRegler(2.0, 0.01, 0.05, 0.001, " m", 3, "fein mm")


def raster_meter(wert: float, grob: float = _METER[0]) -> float:
    """Wert auf dem Raster von :func:`regler_meter`."""
    return auf_raster(wert, grob, *_METER[1:])


def raster_grad(wert: float, grob: float = _GRAD[0]) -> float:
    """Wert auf dem Raster von :func:`regler_grad`."""
    return auf_raster(wert, grob, *_GRAD[1:])


def raster_prozent(wert: float, grob: float = _PROZENT[0]) -> float:
    """Wert auf dem Raster von :func:`regler_prozent`."""
    return auf_raster(wert, grob, *_PROZENT[1:])


if __name__ == "__main__":
    import os
    import sys

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication(sys.argv)
    r = regler_meter()
    gesehen = []
    r.valueChanged.connect(gesehen.append)
    r.setValue(12.34)
    assert abs(r.value() - 12.34) < 1e-9, r.value()
    assert r._grob.value() == 123 and r._fein.value() == 4, (r._grob.value(), r._fein.value())
    assert gesehen and abs(gesehen[-1] - 12.34) < 1e-9
    r._fein.setValue(-37)                       # am Feinschieber ziehen
    assert abs(r.value() - (12.3 - 0.37)) < 1e-9, r.value()
    assert abs(r._zahl.value() - r.value()) < 1e-9, "Zahlenfeld nicht nachgezogen"
    r._zahl.setValue(-242.46)                   # eintippen
    assert abs(r.value() + 242.46) < 1e-9, r.value()
    r.setValue(9999)                            # ausserhalb: am Anschlag
    assert abs(r.value() - r.maximum()) < 1e-9
    n = len(gesehen)
    r.blockSignals(True)
    r.setValue(1.0)
    r.blockSignals(False)
    assert len(gesehen) == n and abs(r.value() - 1.0) < 1e-9, "blockSignals wirkt nicht"
    g = regler_grad()
    g.setValue(0.123)
    assert abs(g.value() - 0.125) < 1e-9, g.value()   # auf 0,005° gerundet
    p = regler_prozent()
    p.setValue(11.07)
    assert abs(p.value() - 11.07) < 1e-9
    print(f"Meter {r.value():.2f}, Grad {g.value():.3f}, Prozent {p.value():.2f} — "
          f"grob+fein addieren sich, Zahlenfeld und blockSignals stimmen")

    mm = regler_millimeter()
    mm.setValue(0.1234)
    assert abs(mm.value() - 0.123) < 1e-12 and abs(mm.maximum() - 2.05) < 1e-12, mm.value()

    # Rasterfunktionen gegen den Regler: bitgleich, auch jenseits des Anschlags,
    # genau am Anschlag und genau zwischen zwei Rasterpunkten
    import random
    zufall = random.Random(20261003)
    for name, regler, raster in (
            ("raster_grad", regler_grad(), raster_grad),
            ("raster_meter", regler_meter(), raster_meter),
            ("raster_meter(50)", regler_meter(50.0), lambda w: raster_meter(w, 50.0)),
            ("raster_prozent", regler_prozent(), raster_prozent)):
        m = regler.maximum()
        werte = [zufall.uniform(-1.5 * m, 1.5 * m) for _ in range(1000)]
        werte += [0.0, -0.0, m, -m, m + 1e-9, -m - 1e-9, 2 * m, -2 * m,
                  0.5 * regler._gs, -0.5 * regler._gs, 0.5 * regler._fs,
                  regler._gs + 0.5 * regler._fs, -regler._gs - 0.5 * regler._fs]
        for w in werte:
            regler.setValue(w)
            soll = regler.value()
            ist = raster(w)
            assert ist.hex() == soll.hex(), (name, w, ist, soll)
    print("Rasterfunktionen bitgleich zum Regler (je 1000 Zufallswerte und Ränder)")
    print("feinregler SELFTEST OK")
