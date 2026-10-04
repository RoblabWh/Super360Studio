"""Wiederverwendbare Bausteine der Bedienoberfläche.

Knöpfe, Zahlenfelder, Auswahlen, Haken und Schieber entstehen hier mit ihren
Grundeinstellungen; ``aktionsknopf`` und ``aktionshaken`` spiegeln eine
QAction, sodass Menü und Seitenleiste denselben Zustand zeigen.
"""
from __future__ import annotations

import numpy as np
from PyQt5.QtCore import QEvent, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QAbstractButton, QAction, QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
    QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QSlider, QSpinBox, QVBoxLayout, QWidget,
)

from ui.collapsible import Section


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


# ------------------------------------------------------------------ Knöpfe

def knopf(text: str, slot=None, tip: str = "", fett: bool = False) -> QPushButton:
    """Druckknopf; ``slot`` wird ohne Argument aufgerufen."""
    btn = QPushButton(text)
    if tip:
        btn.setToolTip(tip)
    if fett:
        btn.setStyleSheet("font-weight: bold;")
    if slot is not None:
        btn.clicked.connect(lambda _c=False: slot())
    return btn


def _ohne_kuerzel(text: str) -> str:
    """Menütext ohne Tastenkürzel-&, ein doppeltes && bleibt ein sichtbares &."""
    return text.replace("&&", "\0").replace("&", "").replace("\0", "&&")


class _FolgtAktion:
    """Gemeinsamer Teil von :func:`aktionsknopf` und :func:`aktionshaken`.

    Das Widget hängt die Aktion per addAction an und liest sie bei jedem
    ActionChanged neu aus. Dieses Ereignis schickt Qt auch dann, wenn die
    Aktion unter blockSignals gesetzt wird – ihr Signal changed bliebe dann
    aus. Der Haken gehört allein der Aktion: ein Klick schaltet das Widget
    nicht selbst um, sondern löst die Aktion aus, und das Widget folgt ihr.
    """

    _aktion: QAction | None = None
    _kurz: str | None = None

    def _anhaengen(self, action: QAction, kurz: str | None) -> None:
        self._aktion = action
        self._kurz = kurz
        self.clicked.connect(self._geklickt)
        self.addAction(action)
        self._folgen()

    def actionEvent(self, event) -> None:  # noqa: N802 (Qt-Name)
        if event.action() is self._aktion:
            if event.type() == QEvent.ActionRemoved:
                self._aktion = None
            else:
                self._folgen()
        super().actionEvent(event)

    def nextCheckState(self) -> None:  # noqa: N802 (Qt-Name)
        pass

    def _geklickt(self, *_a) -> None:
        if self._aktion is not None:
            self._aktion.trigger()

    def _folgen(self) -> None:
        act = self._aktion
        if act is None:
            return
        tip = act.toolTip()
        alt = self.blockSignals(True)
        try:
            self.setText(self._kurz if self._kurz is not None
                         else _ohne_kuerzel(act.text()))
            # ohne eigenen Tooltip liefert QAction ihren Text – den nicht wiederholen
            self.setToolTip("" if tip == act.iconText() else tip)
            self.setEnabled(act.isEnabled())
            self.setCheckable(act.isCheckable())
            self.setChecked(act.isChecked())
        finally:
            self.blockSignals(alt)


class _AktionsKnopf(_FolgtAktion, QPushButton):
    pass


class _AktionsHaken(_FolgtAktion, QCheckBox):
    pass


def aktionsknopf(action: QAction, kurz: str | None = None) -> QPushButton:
    """Knopf zu einer Aktion: folgt Text (oder zeigt ``kurz``), Tooltip, Freigabe, Haken."""
    btn = _AktionsKnopf()
    btn._anhaengen(action, kurz)
    return btn


def aktionshaken(action: QAction) -> QCheckBox:
    """Haken zu einer schaltbaren Aktion: folgt Text, Tooltip, Freigabe und Haken."""
    chk = _AktionsHaken()
    chk._anhaengen(action, None)
    return chk


def knopfzeile(*knoepfe: QWidget) -> QWidget:
    """Knöpfe nebeneinander, ohne eigenen Rand."""
    zeile = QWidget()
    hl = QHBoxLayout(zeile)
    hl.setContentsMargins(0, 0, 0, 0)
    for k in knoepfe:
        hl.addWidget(k)
    return zeile


# --------------------------------------------------------- Werte und Auswahl

def zahl(minimum, maximum, wert, schritt=None, *, dezimalen: int | None = None,
         suffix: str = "", tip: str = "", slot=None) -> QSpinBox | QDoubleSpinBox:
    """Zahlenfeld: ohne ``dezimalen`` ganzzahlig (QSpinBox), sonst QDoubleSpinBox."""
    if dezimalen is None:
        spin = QSpinBox()
        spin.setRange(int(minimum), int(maximum))
        if schritt is not None:
            spin.setSingleStep(int(schritt))
    else:
        spin = QDoubleSpinBox()
        spin.setDecimals(int(dezimalen))
        spin.setRange(float(minimum), float(maximum))
        if schritt is not None:
            spin.setSingleStep(float(schritt))
    if suffix:
        spin.setSuffix(suffix)
    spin.setValue(int(wert) if dezimalen is None else float(wert))
    if tip:
        spin.setToolTip(tip)
    if slot is not None:
        spin.valueChanged.connect(slot)
    return spin


def auswahl(eintraege, aktuell=None, slot=None, tip: str = "",
            muster: str = "Maximal dicht (whs_de", rand: int = 44) -> QComboBox:
    """Auswahlliste aus Texten oder (Text, Daten); schmal gehalten wie die Seitenleiste.

    ``aktuell`` wählt den Eintrag mit diesen Daten vor, bevor ``slot`` an
    currentIndexChanged hängt. Die Breite deckelt ``muster`` plus ``rand`` Pixel.
    """
    combo = QComboBox()
    combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(10)
    _cap_width(combo, muster, rand)
    for e in eintraege:
        if isinstance(e, str):
            combo.addItem(e)
        else:
            combo.addItem(e[0], e[1])
    if aktuell is not None:
        idx = combo.findData(aktuell)
        if idx >= 0:
            combo.setCurrentIndex(idx)
    if tip:
        combo.setToolTip(tip)
    if slot is not None:
        combo.currentIndexChanged.connect(slot)
    return combo


def haken(text: str, an: bool = False, slot=None, tip: str = "") -> QCheckBox:
    """Haken; ``slot`` hängt an toggled und bekommt den neuen Zustand."""
    chk = QCheckBox(text)
    chk.setChecked(bool(an))
    if tip:
        chk.setToolTip(tip)
    if slot is not None:
        chk.toggled.connect(slot)
    return chk


def schieber(minimum: int, maximum: int, wert: int, slot=None
             ) -> tuple[QSlider, QWidget]:
    """Waagerechter Schieber mit Zahl rechts daneben; gibt (Schieber, Zeile) zurück."""
    zeile = QWidget()
    lay = QHBoxLayout(zeile)
    lay.setContentsMargins(0, 0, 0, 0)
    slider = QSlider(Qt.Horizontal)
    slider.setRange(minimum, maximum)
    slider.setValue(wert)
    lbl = QLabel(str(wert))
    lbl.setMinimumWidth(30)
    lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    slider.valueChanged.connect(lambda v, l=lbl: l.setText(str(v)))
    if slot is not None:
        slider.valueChanged.connect(slot)
    lay.addWidget(slider, 1)
    lay.addWidget(lbl)
    return slider, zeile


def reglergruppe(form: QFormLayout, felder, slot=None) -> dict:
    """Je Feld (Schlüssel, Beschriftung, Regler) eine Formularzeile; gibt {Schlüssel: Regler}.

    ``slot`` hängt an valueChanged jedes Reglers und bekommt den neuen Wert.
    """
    regler = {}
    for key, label, r in felder:
        if slot is not None:
            r.valueChanged.connect(slot)
        form.addRow(label, r)
        regler[key] = r
    return regler


def still_setzen(ziel, wert) -> None:
    """Wert setzen, ohne dass ``ziel`` ein Signal sendet.

    Bei einer QAction ist der Wert ihr Haken. Angehängte aktionsknopf und
    aktionshaken folgen trotzdem (über ActionChanged), ohne selbst ein Signal
    zu senden. Ein aktionsknopf oder aktionshaken setzt seine Aktion. Eine
    Auswahl wählt den Eintrag mit diesen Daten (fehlt er, bleibt sie, wie sie
    ist); Knöpfe und Haken nehmen den Wert als Haken, Textfelder als Text,
    alles andere über setValue.
    """
    if isinstance(ziel, _FolgtAktion) and ziel._aktion is not None:
        ziel = ziel._aktion
    alt = ziel.blockSignals(True)
    try:
        if isinstance(ziel, (QAction, QAbstractButton)):
            ziel.setChecked(bool(wert))
        elif isinstance(ziel, QComboBox):
            idx = ziel.findData(wert)
            if idx >= 0:
                ziel.setCurrentIndex(idx)
        elif isinstance(ziel, QLineEdit):
            ziel.setText(str(wert))
        else:
            ziel.setValue(wert)
    finally:
        ziel.blockSignals(alt)


# ------------------------------------------------------------ Unterblock

class Unterblock(Section):
    """Zwischenüberschrift mit eigenem Formular ``.form`` im Inhalt eines Abschnitts.

    Einklappbar ist er eine Section mit Pfeil; sein Zustand wird erst
    gespeichert, wenn ihn das Fenster mit ``SectionStack.melde_an`` unter
    ``schluessel`` (etwa ``maeander.hauptpunkt``) anmeldet. Nicht einklappbar
    ist die Kopfzeile nur Überschrift und der Inhalt immer offen.
    """

    def __init__(self, titel: str, einklappbar: bool = False, offen: bool = True,
                 schluessel: str = "", parent: QWidget | None = None):
        inhalt = QWidget()
        form = _wrappable(QFormLayout(inhalt))
        form.setContentsMargins(8, 0, 0, 0)
        super().__init__(schluessel, titel, inhalt, bool(offen) or not einklappbar, parent)
        self.form = form
        self._einklappbar = bool(einklappbar)
        if self._einklappbar:
            self._head.setStyleSheet(
                "QToolButton { border: none; text-align: left; padding: 3px 2px;"
                " font-weight: 600; }"
                "QToolButton:hover { background: rgba(255,255,255,22); border-radius: 4px; }")
        else:
            self._head.setCheckable(False)
            self._head.setFocusPolicy(Qt.NoFocus)
            self._head.setCursor(Qt.ArrowCursor)
            self._head.setStyleSheet(
                "QToolButton { border: none; text-align: left; padding: 3px 2px;"
                " font-weight: 600; }")
        self._sync()

    def einklappbar(self) -> bool:
        return getattr(self, "_einklappbar", True)

    def is_expanded(self) -> bool:
        return True if not self.einklappbar() else super().is_expanded()

    def set_expanded(self, on: bool) -> None:
        if self.einklappbar():
            super().set_expanded(on)

    def _on_clicked(self) -> None:
        if self.einklappbar():
            super()._on_clicked()

    def _sync(self) -> None:
        if self.einklappbar():
            super()._sync()
        else:
            self._head.setText(self._title)
            self._content.setVisible(True)


# ------------------------------------------------------------ Verschiedenes

def bgr_zu_pixmap(bgr: np.ndarray) -> QPixmap:
    """BGR-Bild (H×W×3, uint8) als QPixmap; die Pixel werden kopiert."""
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    h, w = rgb.shape[:2]
    img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(img)


def speicherpfad(parent, titel: str, start: str, filter: str, endung) -> str:
    """Speichern-Dialog; leer bei Abbruch, sonst der Pfad mit passender Endung.

    ``endung`` ist eine Endung (".png") oder mehrere (".ply", ".pcd"). Trägt
    der Pfad keine davon, kommt die des gewählten Filters dazu, sonst die erste.
    """
    endungen = (endung,) if isinstance(endung, str) else tuple(endung)
    path, gewaehlt = QFileDialog.getSaveFileName(parent, titel, start, filter)
    if not path:
        return ""
    if not path.lower().endswith(endungen):
        gewaehlt = (gewaehlt or "").lower()
        path += next((e for e in endungen if "*" + e in gewaehlt), endungen[0])
    return path


def einmal_timer(parent, ms: int, slot=None) -> QTimer:
    """Einmal-Timer mit festem Abstand; ``start()`` setzt ihn zurück."""
    timer = QTimer(parent)
    timer.setSingleShot(True)
    timer.setInterval(ms)
    if slot is not None:
        timer.timeout.connect(slot)
    return timer


if __name__ == "__main__":
    import os
    import sys

    from PyQt5.QtWidgets import QApplication

    from ui.collapsible import SectionStack
    from ui.feinregler import regler_meter

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication(sys.argv)
    besitzer = QWidget()

    def zaehler(signal) -> list:
        gesehen: list = []
        signal.connect(lambda *a: gesehen.append(a))
        return gesehen

    # aktionsknopf: folgt der Aktion auch unter blockSignals, Klick genau einmal
    act = QAction("&Rechnen", besitzer)
    act.setToolTip("rechnet")
    ausgeloest = zaehler(act.triggered)
    btn = aktionsknopf(act)
    kurz = aktionsknopf(act, kurz="Los")
    assert (btn.text(), btn.toolTip(), btn.isEnabled(), btn.isCheckable()) == \
        ("Rechnen", "rechnet", True, False), btn.text()
    act.blockSignals(True)
    act.setEnabled(False)
    act.setText("Neu rechnen")
    act.setToolTip("fehlt: Karte")
    act.blockSignals(False)
    assert (btn.text(), btn.toolTip(), btn.isEnabled()) == ("Neu rechnen", "fehlt: Karte", False)
    assert (kurz.text(), kurz.toolTip(), kurz.isEnabled()) == ("Los", "fehlt: Karte", False)
    btn.click()
    assert ausgeloest == [], "gesperrter Knopf hat ausgelöst"
    act.setEnabled(True)
    btn.click()
    assert len(ausgeloest) == 1, ausgeloest
    act.blockSignals(True)
    act.setCheckable(True)
    act.setChecked(True)
    act.blockSignals(False)
    assert btn.isCheckable() and btn.isChecked() and kurz.isChecked()
    umgeschaltet = zaehler(act.toggled)
    btn.click()
    assert len(ausgeloest) == 2 and umgeschaltet == [(False,)], (ausgeloest, umgeschaltet)
    assert not act.isChecked() and not btn.isChecked() and not kurz.isChecked()
    ohne_tip = QAction("Nur Text", besitzer)
    assert aktionsknopf(ohne_tip).toolTip() == "", "Text als Tooltip wiederholt"

    # aktionshaken: dasselbe als Haken
    vorschau = QAction("Vorschau zeigen", besitzer)
    vorschau.setCheckable(True)
    v_ausgeloest = zaehler(vorschau.triggered)
    v_umgeschaltet = zaehler(vorschau.toggled)
    chk = aktionshaken(vorschau)
    chk_signale = zaehler(chk.toggled)
    chk_klicks = zaehler(chk.clicked)
    assert isinstance(chk, QCheckBox) and chk.text() == "Vorschau zeigen" and not chk.isChecked()
    vorschau.blockSignals(True)
    vorschau.setChecked(True)
    vorschau.setEnabled(False)
    vorschau.setText("Zweiten Flug zeigen")
    vorschau.setToolTip("orange")
    vorschau.blockSignals(False)
    assert (chk.isChecked(), chk.isEnabled(), chk.text(), chk.toolTip()) == \
        (True, False, "Zweiten Flug zeigen", "orange")
    assert v_umgeschaltet == [] and chk_signale == []
    vorschau.setEnabled(True)
    chk.click()
    assert v_ausgeloest == [(False,)] and v_umgeschaltet == [(False,)], v_ausgeloest
    assert not chk.isChecked() and not vorschau.isChecked()
    assert len(chk_klicks) == 1 and chk_signale == [], chk_signale

    # still_setzen: kein Signal am Widget, angehängte Widgets folgen der Aktion
    n_klicks = len(chk_klicks)
    v_geaendert = zaehler(vorschau.changed)
    still_setzen(vorschau, True)
    assert vorschau.isChecked() and chk.isChecked()
    assert v_umgeschaltet == [(False,)] and v_geaendert == [] and len(v_ausgeloest) == 1
    assert chk_signale == [] and len(chk_klicks) == n_klicks
    still_setzen(chk, False)                       # über den Haken: setzt die Aktion
    assert not vorschau.isChecked() and not chk.isChecked() and chk_signale == []
    assert v_umgeschaltet == [(False,)]
    h = haken("Blaulicht filtern", slot=lambda an: None)
    sp = zahl(0, 20, 4, suffix=" px")
    dsp = zahl(0.25, 2.0, 1.0, 0.25, dezimalen=2)
    cb = auswahl([("Dicht", "dense"), ("Grob", "grob")], aktuell="grob")
    sl, _zeile = schieber(0, 255, 20)
    fr = regler_meter()
    le = QLineEdit()
    for w, sig, wert, lesen in (
            (h, h.toggled, True, h.isChecked), (sp, sp.valueChanged, 7, sp.value),
            (dsp, dsp.valueChanged, 1.75, dsp.value),
            (cb, cb.currentIndexChanged, "dense", cb.currentData),
            (sl, sl.valueChanged, 99, sl.value), (fr, fr.valueChanged, 12.34, fr.value),
            (le, le.textChanged, "abc", le.text)):
        gesehen = zaehler(sig)
        still_setzen(w, wert)
        assert gesehen == [], (type(w).__name__, gesehen)
        assert (abs(lesen() - wert) < 1e-9 if isinstance(wert, float) else lesen() == wert), \
            (type(w).__name__, lesen())
        assert not w.signalsBlocked()
    still_setzen(cb, "gibt es nicht")
    assert cb.currentData() == "dense"
    sp.blockSignals(True)
    still_setzen(sp, 3)
    assert sp.signalsBlocked(), "still_setzen hebt eine vorhandene Sperre auf"
    sp.blockSignals(False)

    # Unterblock: Zustand über SectionStack.states(), Umschalten meldet toggled
    stack = SectionStack()
    inhalt = QWidget()
    form = _wrappable(QFormLayout(inhalt))
    ub = Unterblock("Hauptpunkt", True, False, "maeander.hauptpunkt")
    ub.form.addRow("rechts", regler_meter())
    form.addRow(ub)
    titel = Unterblock("Blaulicht")
    titel.form.addRow("Farbton", QSpinBox())
    form.addRow(titel)
    stack.add("maeander", "Mäander-Einfärbung", inhalt, False)
    stack.add("anzeige", "Anzeige", QWidget(), True)
    stack.finish()
    stack.melde_an(ub)
    gemeldet = zaehler(stack.toggled)
    assert stack.states() == {"maeander": False, "anzeige": True,
                              "maeander.hauptpunkt": False}, stack.states()
    assert ub.form.rowCount() == 1 and ub.content().isHidden()
    ub._head.click()
    assert gemeldet == [("maeander.hauptpunkt", True)], gemeldet
    assert stack.states()["maeander.hauptpunkt"] is True and not ub.content().isHidden()
    stack.set_states({"maeander.hauptpunkt": False})
    assert stack.states()["maeander.hauptpunkt"] is False
    stack.set_all(True)
    assert all(stack.states().values())
    stack.set_all(False)
    assert not any(stack.states().values())
    assert ub in stack.sections(mit_unterbloecken=True), "zugeklappter Unterblock fehlt"
    assert ub not in stack.sections() and len(stack.sections()) == 2
    assert titel.is_expanded() and not titel.content().isHidden()
    assert titel._head.text() == "Blaulicht" and titel.form.rowCount() == 1
    titel._head.click()
    titel.set_expanded(False)
    assert titel.is_expanded() and not titel.content().isHidden()
    try:
        stack.melde_an(titel)
    except ValueError:
        pass
    else:
        raise AssertionError("nicht einklappbarer Unterblock angemeldet")

    # Knopf, Knopfzeile, Zahl, Auswahl, Haken, Schieber, Reglergruppe
    gerufen: list = []
    k = knopf("Standardwerte", lambda: gerufen.append("k"), tip="setzt zurück", fett=True)
    k.click()
    assert gerufen == ["k"] and k.toolTip() == "setzt zurück" and "bold" in k.styleSheet()
    zeile = knopfzeile(k, knopf("Blaumaske zeigen"))
    assert zeile.layout().count() == 2 and zeile.layout().contentsMargins().left() == 0
    assert type(sp) is QSpinBox and (sp.minimum(), sp.maximum(), sp.suffix()) == (0, 20, " px")
    assert type(dsp) is QDoubleSpinBox and dsp.decimals() == 2 and dsp.singleStep() == 0.25
    vergleich = _compact_combo(QComboBox())
    assert cb.maximumWidth() == vergleich.maximumWidth()
    assert cb.sizeAdjustPolicy() == vergleich.sizeAdjustPolicy()
    assert cb.minimumContentsLength() == vergleich.minimumContentsLength()
    assert auswahl(["a", "b"]).count() == 2
    h2 = haken("x", an=True, slot=lambda an: gerufen.append(an))
    h2.setChecked(False)
    assert gerufen[-1] is False
    sl2, zeile2 = schieber(0, 360, 210, slot=lambda v: gerufen.append(v))
    sl2.setValue(222)
    assert gerufen[-1] == 222 and zeile2.findChild(QLabel).text() == "222"
    rbox = QWidget()
    rform = QFormLayout(rbox)
    gruppe = reglergruppe(rform, (("x", "X", regler_meter(500.0)),
                                  ("z", "Z", regler_meter(500.0))),
                          lambda v: gerufen.append(("regler", v)))
    assert list(gruppe) == ["x", "z"] and rform.rowCount() == 2
    gruppe["z"].setValue(1.5)
    assert gerufen[-1] == ("regler", 1.5)

    # bgr_zu_pixmap, speicherpfad, einmal_timer
    bild = np.zeros((4, 6, 3), np.uint8)
    bild[..., 2] = 255                                   # BGR: rot
    pm = bgr_zu_pixmap(bild)
    assert (pm.width(), pm.height()) == (6, 4)
    assert pm.toImage().pixelColor(0, 0).getRgb()[:3] == (255, 0, 0)
    antworten: list = []
    echt = QFileDialog.getSaveFileName
    QFileDialog.getSaveFileName = staticmethod(lambda *a: antworten.pop(0))
    try:
        antworten[:] = [("", ""), ("/x/bild", "PNG-Datei (*.png)"),
                        ("/x/bild.PNG", ""), ("/x/wolke", "PCD-Datei (*.pcd)"),
                        ("/x/wolke", "PLY-Datei (*.ply)"), ("/x/wolke", ""),
                        ("/x/wolke.pcd", "PLY-Datei (*.ply)")]
        assert speicherpfad(None, "t", "s", "PNG-Datei (*.png)", ".png") == ""
        assert speicherpfad(None, "t", "s", "PNG-Datei (*.png)", ".png") == "/x/bild.png"
        assert speicherpfad(None, "t", "s", "PNG-Datei (*.png)", ".png") == "/x/bild.PNG"
        f = "PLY-Datei (*.ply);;PCD-Datei (*.pcd)"
        assert speicherpfad(None, "t", "s", f, (".ply", ".pcd")) == "/x/wolke.pcd"
        assert speicherpfad(None, "t", "s", f, (".ply", ".pcd")) == "/x/wolke.ply"
        assert speicherpfad(None, "t", "s", f, (".ply", ".pcd")) == "/x/wolke.ply"
        assert speicherpfad(None, "t", "s", f, (".ply", ".pcd")) == "/x/wolke.pcd"
    finally:
        QFileDialog.getSaveFileName = echt
    t = einmal_timer(besitzer, 400, lambda: gerufen.append("timer"))
    assert t.isSingleShot() and t.interval() == 400 and t.parent() is besitzer
    t.timeout.emit()
    assert gerufen[-1] == "timer"
    assert not einmal_timer(besitzer, 1500).isActive()
    print("bausteine SELFTEST OK")
