#!/usr/bin/env python3
"""Anzeige zu ``entschaerfen.py``: links das Original, rechts das Ergebnis.

Liest jede Sekunde ``stand.json`` aus dem Ausgabeordner. Solange "Neueste
Iteration folgen" angehakt ist, springt die rechte Seite auf jede neu fertig
gewordene Iteration; ein Klick in die Tabelle haelt eine bestimmte fest.

Aufruf:

    entschaerfen_gui.py <ausgabeordner>

Bedienung: Mausrad zoomt, Ziehen verschiebt — beide Seiten laufen gemeinsam.
Pfeil links/rechts wechselt das Bild, Pfeil hoch/runter die Iteration. Der
Regler "Aufhellen" wirkt auf beide Seiten gleich und aendert nichts an den
Dateien.
"""

from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

# `import cv2` biegt die Qt-Plugin-Pfade prozessweit auf sein eigenes Qt um;
# mit dem PyQt5 aus apt startet dann kein Fenster (s. core/rviz_player.py).
for _var in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR"):
    os.environ.pop(_var, None)

from PyQt5 import QtCore, QtGui, QtWidgets  # noqa: E402


def _de(x: float, n: int = 2, vorzeichen: bool = False) -> str:
    return f"{x:{'+' if vorzeichen else ''}.{n}f}".replace(".", ",")


class Ansicht(QtWidgets.QGraphicsView):
    """Bildflaeche mit Zoom am Mauszeiger; meldet jede Aenderung des Ausschnitts."""

    bewegt = QtCore.pyqtSignal()

    def __init__(self, titel: str):
        super().__init__()
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.bild = self.scene().addPixmap(QtGui.QPixmap())
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QtGui.QColor("#101216"))
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.titel = QtWidgets.QLabel(titel, self)
        self.titel.setStyleSheet("background: rgba(0,0,0,170); color: white; padding: 4px 8px;"
                                 "font-weight: bold;")
        self.titel.move(8, 8)
        self.kern = QtWidgets.QLabel(self)
        self.kern.setStyleSheet("background: black; border: 1px solid #666;")
        self.kern.hide()
        self._still = False
        self.horizontalScrollBar().valueChanged.connect(self._melde)
        self.verticalScrollBar().valueChanged.connect(self._melde)

    def _melde(self) -> None:
        if not self._still:
            self.bewegt.emit()

    def setze_titel(self, text: str) -> None:
        self.titel.setText(text)
        self.titel.adjustSize()

    def setze_bild(self, pix: QtGui.QPixmap) -> None:
        neu = self.bild.pixmap().size() != pix.size()
        self.bild.setPixmap(pix)
        if neu:
            self.scene().setSceneRect(QtCore.QRectF(pix.rect()))
            self.einpassen()

    def setze_kern(self, pix: QtGui.QPixmap | None) -> None:
        if pix is None or pix.isNull():
            self.kern.hide()
            return
        self.kern.setPixmap(pix)
        self.kern.adjustSize()
        self.kern.show()
        self._lege_kern()

    def _lege_kern(self) -> None:
        self.kern.move(self.width() - self.kern.width() - 8, 8)

    def einpassen(self) -> None:
        self.fitInView(self.bild, QtCore.Qt.KeepAspectRatio)
        self._melde()

    def uebernimm(self, andere: "Ansicht") -> None:
        self._still = True
        self.setTransform(andere.transform())
        self.horizontalScrollBar().setValue(andere.horizontalScrollBar().value())
        self.verticalScrollBar().setValue(andere.verticalScrollBar().value())
        self._still = False

    def wheelEvent(self, e: QtGui.QWheelEvent) -> None:
        f = 1.25 if e.angleDelta().y() > 0 else 0.8
        self.scale(f, f)
        self._melde()

    def resizeEvent(self, e: QtGui.QResizeEvent) -> None:
        super().resizeEvent(e)
        self._lege_kern()


class Fenster(QtWidgets.QWidget):
    SPALTEN = ["Nr", "Wertung", "PSNR-Gewinn dB", "SIFT-Inlier", "zum Original", "Verfahren"]

    def __init__(self, ordner: str):
        super().__init__()
        self.ordner = ordner
        self.stand: dict = {}
        self._zeit = 0.0
        self._nr = 0          # gewaehlte Iteration (1-basiert), 0 = keine
        self._name = ""       # gewaehltes Bild
        self.setWindowTitle("Entschaerfen — Original und Ergebnis")
        self.resize(1700, 1000)

        self.links = Ansicht("Original")
        self.rechts = Ansicht("Ergebnis")
        self.links.bewegt.connect(lambda: self.rechts.uebernimm(self.links))
        self.rechts.bewegt.connect(lambda: self.links.uebernimm(self.rechts))
        bilder = QtWidgets.QSplitter()
        bilder.addWidget(self.links)
        bilder.addWidget(self.rechts)

        self.liste = QtWidgets.QListWidget()
        self.liste.setFixedWidth(120)
        self.liste.currentTextChanged.connect(self._bild_gewaehlt)

        self.folgen = QtWidgets.QCheckBox("Neueste Iteration folgen")
        self.folgen.setChecked(True)
        self.folgen.toggled.connect(lambda an: an and self._lies(erzwingen=True))
        self.hell = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.hell.setRange(0, 100)
        self.hell.setValue(40)
        self.hell.setFixedWidth(180)
        self.hell.valueChanged.connect(self._zeige)
        passen = QtWidgets.QPushButton("Einpassen")
        passen.clicked.connect(self.links.einpassen)
        self.beste_knopf = QtWidgets.QPushButton("Zur besten")
        self.beste_knopf.clicked.connect(self._zur_besten)
        self.status = QtWidgets.QLabel("warte auf stand.json …")
        leiste = QtWidgets.QHBoxLayout()
        for w in (self.folgen, self.beste_knopf, passen, QtWidgets.QLabel("Aufhellen"), self.hell):
            leiste.addWidget(w)
        leiste.addWidget(self.status, 1)

        self.tabelle = QtWidgets.QTableWidget(0, len(self.SPALTEN))
        self.tabelle.setHorizontalHeaderLabels(self.SPALTEN)
        self.tabelle.verticalHeader().hide()
        self.tabelle.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tabelle.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.tabelle.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tabelle.horizontalHeader().setStretchLastSection(True)
        self.tabelle.setFixedHeight(230)
        self.tabelle.cellClicked.connect(self._zeile_geklickt)

        oben = QtWidgets.QHBoxLayout()
        oben.addWidget(self.liste)
        oben.addWidget(bilder, 1)
        alles = QtWidgets.QVBoxLayout(self)
        alles.addLayout(oben, 1)
        alles.addLayout(leiste)
        alles.addWidget(self.tabelle)

        self.uhr = QtCore.QTimer(self)
        self.uhr.timeout.connect(self._lies)
        self.uhr.start(1000)
        self._lies()

    # -- Stand lesen ------------------------------------------------------

    def _lies(self, erzwingen: bool = False) -> None:
        pfad = os.path.join(self.ordner, "stand.json")
        try:
            zeit = os.path.getmtime(pfad)
            if zeit == self._zeit and not erzwingen:
                return
            with open(pfad, encoding="utf-8") as f:
                self.stand = json.load(f)
            self._zeit = zeit
        except (OSError, ValueError):
            return
        its = self.stand.get("iterationen", [])
        self._fuelle_tabelle(its)
        ref = self.stand.get("referenz")
        text = self.stand.get("status", "")
        if ref:
            text += f"   |   Original: {_de(ref['inlier'], 0)} Inlier je Paar"
        self.status.setText(text)
        if its and (self.folgen.isChecked() or self._nr == 0):
            self._waehle(its[-1]["nr"])

    def _fuelle_tabelle(self, its: list[dict]) -> None:
        beste = self.stand.get("beste")
        self.tabelle.setRowCount(len(its))
        for z, it in enumerate(its):
            hat = "wertung" in it
            werte = [str(it["nr"]),
                     _de(it["wertung"], 2, True) if hat else "",
                     _de(it["psnr_gewinn"], 2, True) if hat else "",
                     _de(it["inlier"], 0) if hat else "",
                     _de(it["inlier_rel"], 2) + "×" if hat else "",
                     it["beschreibung"] + ("   ★ beste" if it["nr"] == beste else "")]
            for s, w in enumerate(werte):
                zelle = QtWidgets.QTableWidgetItem(w)
                if it["nr"] == beste:
                    zelle.setForeground(QtGui.QColor("#4ec97a"))
                    f = zelle.font()
                    f.setBold(True)
                    zelle.setFont(f)
                self.tabelle.setItem(z, s, zelle)
        self.tabelle.resizeColumnsToContents()

    # -- Auswahl ----------------------------------------------------------

    def _iteration(self) -> dict | None:
        its = self.stand.get("iterationen", [])
        return its[self._nr - 1] if 0 < self._nr <= len(its) else None

    def _waehle(self, nr: int) -> None:
        self._nr = nr
        it = self._iteration()
        if it is None:
            return
        self.tabelle.selectRow(nr - 1)
        self.tabelle.scrollToItem(self.tabelle.item(nr - 1, 0))
        namen = it["bilder"]
        if [self.liste.item(i).data(QtCore.Qt.UserRole) for i in range(self.liste.count())] != namen:
            self.liste.blockSignals(True)
            self.liste.clear()
            for n in namen:
                # DJI_20260919022417_0001_W.JPG -> 0001
                teile = os.path.splitext(n)[0].split("_")
                eintrag = QtWidgets.QListWidgetItem(teile[2] if len(teile) > 2 else n)
                eintrag.setData(QtCore.Qt.UserRole, n)
                self.liste.addItem(eintrag)
            self.liste.blockSignals(False)
        if self._name not in namen:
            self._name = namen[0]
        self.liste.blockSignals(True)
        self.liste.setCurrentRow(namen.index(self._name))
        self.liste.blockSignals(False)
        self._zeige()

    def _zeile_geklickt(self, zeile: int, _spalte: int) -> None:
        self.folgen.setChecked(False)
        self._waehle(zeile + 1)

    def _zur_besten(self) -> None:
        if self.stand.get("beste"):
            self.folgen.setChecked(False)
            self._waehle(self.stand["beste"])

    def _bild_gewaehlt(self, _text: str) -> None:
        eintrag = self.liste.currentItem()
        if eintrag is not None:
            self._name = eintrag.data(QtCore.Qt.UserRole)
            self._zeige()

    # -- Bilder -----------------------------------------------------------

    def _pix(self, pfad: str) -> QtGui.QPixmap | None:
        g = cv2.imread(pfad, cv2.IMREAD_GRAYSCALE)
        if g is None:
            return None
        # Aufhellen als Gammakurve: 0 = unveraendert, 100 = Exponent 0,3
        e = 1.0 - 0.007 * self.hell.value()
        kurve = (255.0 * (np.arange(256) / 255.0) ** e + 0.5).astype(np.uint8)
        g = np.ascontiguousarray(kurve[g])
        h, w = g.shape
        return QtGui.QPixmap.fromImage(
            QtGui.QImage(g.data, w, h, w, QtGui.QImage.Format_Grayscale8).copy())

    def _zeige(self) -> None:
        it = self._iteration()
        if it is None or not self._name:
            return
        stamm = os.path.splitext(self._name)[0]
        a = self._pix(os.path.join(self.stand["quelle"], self._name))
        b = self._pix(os.path.join(self.ordner, it["ordner"], stamm + ".jpg"))
        if a is not None:
            self.links.setze_bild(a)
        if b is not None:
            self.rechts.setze_bild(b)
        self.links.setze_titel(f"Original  {self._name}")
        self.rechts.setze_titel(f"Iteration {it['nr']}: {it['beschreibung']}")
        kern = QtGui.QPixmap(os.path.join(self.ordner, it["ordner"], stamm + ".kern.png"))
        self.rechts.setze_kern(kern)

    def keyPressEvent(self, e: QtGui.QKeyEvent) -> None:
        t = e.key()
        if t in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Right):
            z = self.liste.currentRow() + (1 if t == QtCore.Qt.Key_Right else -1)
            if 0 <= z < self.liste.count():
                self.liste.setCurrentRow(z)
        elif t in (QtCore.Qt.Key_Up, QtCore.Qt.Key_Down):
            nr = self._nr + (1 if t == QtCore.Qt.Key_Down else -1)
            if 0 < nr <= len(self.stand.get("iterationen", [])):
                self.folgen.setChecked(False)
                self._waehle(nr)
        else:
            super().keyPressEvent(e)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    app = QtWidgets.QApplication(sys.argv[:1])
    try:
        import qdarktheme
        qdarktheme.setup_theme("dark")
    except Exception:
        pass
    f = Fenster(os.path.abspath(sys.argv[1]))
    f.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
