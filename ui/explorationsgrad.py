"""Anzeige des Explorationsgrades — die Kachel oben rechts im Fenster.

Der Explorationsgrad ist die Kennzahl des Fluges: welchen Anteil des
vorgegebenen Zielgebiets die Drohne im Explorationsmodus tatsaechlich
beobachtet hat (gerechnet in :mod:`core.exploration`). Er gehoert nicht in
eine Tabelle, die man erst aufklappt, sondern dorthin, wo er beim Arbeiten
immer im Blick ist: als Kachel in der rechten Ecke der Menueleiste, sichtbar
in jedem Bereich — 3D-Karte, Video, GPS, Protokoll.

Die Kachel traegt drei Zeilen Information in zwei Zeilen Platz: die Ueberschrift,
darunter den Bezug (Explorationsmodus mit seiner Dauer — oder "ganzer Flug",
wenn niemand autonom flog, denn dieselbe Zahl bedeutet dann etwas anderes), und
rechts gross den Wert. Die Farbe ist eine grobe Ampel und ersetzt kein Urteil;
sie soll nur verhindern, dass ein magerer Lauf wie ein guter aussieht.

Ein Klick auf die Kachel meldet :attr:`angeklickt` — das Fenster zeigt darauf
den vollstaendigen Bericht.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from core.gemeinsam import de

#: Schwellen der Ampel in Prozent und die zugehoerigen Farben. Bewusst kraeftig
#: gewaehlt, damit sie auf dunklem wie hellem Grund tragen.
_GUT = 85.0
_MITTEL = 60.0
_FARBE_GUT = "#4ec97a"
_FARBE_MITTEL = "#e3b341"
_FARBE_SCHWACH = "#e8734b"
_FARBE_LEER = "#8b949e"

#: Feste Breiten fuer Text und Wert. Die Menueleiste fragt die Groesse der
#: Eckkachel genau einmal ab, wenn sie ihre Geometrie legt — waechst der Inhalt
#: danach von "—" auf "96,8 %", bleibt der reservierte Platz der alte und die
#: Zahl wird abgeschnitten. Mit festen Mindestbreiten ist der Platzbedarf in
#: jedem Zustand derselbe, und es gibt nichts abzuschneiden.
_BREITE_TEXT = 152
_BREITE_WERT = 92


def _farbe(prozent: float) -> str:
    if prozent >= _GUT:
        return _FARBE_GUT
    if prozent >= _MITTEL:
        return _FARBE_MITTEL
    return _FARBE_SCHWACH


class ExplorationsgradAnzeige(QtWidgets.QFrame):
    """Kachel mit dem Explorationsgrad; anklickbar fuer den vollen Bericht."""

    #: Kachel angeklickt (nur wenn ein Ergebnis anliegt)
    angeklickt = QtCore.pyqtSignal()

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("explorationsgrad")
        self._hat_wert = False

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(11, 3, 11, 3)
        lay.setSpacing(10)

        links = QtWidgets.QVBoxLayout()
        links.setContentsMargins(0, 0, 0, 0)
        links.setSpacing(0)
        self._titel = QtWidgets.QLabel("Explorationsgrad", self)
        self._titel.setStyleSheet("font-size: 10px; font-weight: bold;")
        self._titel.setMinimumWidth(_BREITE_TEXT)
        self._bezug = QtWidgets.QLabel("—", self)
        self._bezug.setStyleSheet(f"font-size: 10px; color: {_FARBE_LEER};")
        self._bezug.setMinimumWidth(_BREITE_TEXT)
        links.addWidget(self._titel)
        links.addWidget(self._bezug)
        lay.addLayout(links)

        self._wert = QtWidgets.QLabel("—", self)
        self._wert.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self._wert.setMinimumWidth(_BREITE_WERT)
        lay.addWidget(self._wert)

        self.setCursor(QtCore.Qt.ArrowCursor)
        self._male(_FARBE_LEER)
        self.leeren()

    # ------------------------------------------------------------- Zustaende

    def leeren(self, grund: str = "Noch kein Flug geöffnet.") -> None:
        """Kein Ergebnis: Strich statt Zahl."""
        self._setze("—", "—", _FARBE_LEER, grund, False)

    def rechnet(self) -> None:
        self._setze("…", "wird berechnet", _FARBE_LEER,
                    "Der Explorationsgrad wird gerade aus dem Bag gerechnet.", False)

    def ohne_daten(self, grund: str) -> None:
        """Bag ohne EPIC-Daten: der Grad ist nicht berechenbar, kein Fehler."""
        self._setze("—", "keine Daten", _FARBE_LEER,
                    f"Kein Explorationsgrad für diesen Flug.\n{grund}", False)

    def setze(self, grad) -> None:
        """Ergebnis anzeigen (``core.exploration.Explorationsgrad``)."""
        bezug = grad.bezug
        if grad.hat_phase:
            bezug += f" · {de(grad.dauer_s)} s"
        self._setze(f"{de(grad.wert)} %", bezug, _farbe(grad.wert),
                    grad.text() + "\n\n(Klicken für den vollen Bericht)", True)

    # ----------------------------------------------------------------- intern

    def _setze(self, wert: str, bezug: str, farbe: str, tip: str, klickbar: bool) -> None:
        # Der Bezug ist der einzige Text, der laenger werden kann (lange Phase);
        # lieber mit Ellipse kuerzen als die Kachel breiter werden lassen.
        gekuerzt = self._bezug.fontMetrics().elidedText(
            bezug, QtCore.Qt.ElideRight, _BREITE_TEXT)
        self._wert.setText(wert)
        self._wert.setStyleSheet(
            f"font-size: 19px; font-weight: bold; color: {farbe};")
        self._bezug.setText(gekuerzt)
        self._male(farbe)
        self.setToolTip(tip)
        self._hat_wert = klickbar
        self.setCursor(QtCore.Qt.PointingHandCursor if klickbar
                       else QtCore.Qt.ArrowCursor)

    def _male(self, farbe: str) -> None:
        # Rahmen und Fond aus derselben Farbe: der Fond bleibt durchscheinend,
        # damit die Kachel in beiden Themes zum Untergrund passt.
        self.setStyleSheet(
            "#explorationsgrad {"
            f" border: 1px solid {farbe};"
            " border-radius: 6px;"
            f" background: rgba({int(farbe[1:3], 16)}, {int(farbe[3:5], 16)},"
            f" {int(farbe[5:7], 16)}, 28); }}"
        )

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt)
        if self._hat_wert and event.button() == QtCore.Qt.LeftButton:
            self.angeklickt.emit()
        super().mousePressEvent(event)


if __name__ == "__main__":
    import sys

    from core.exploration import Explorationsgrad

    app = QtWidgets.QApplication(sys.argv)
    fenster = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(fenster)

    def probe(**kw):
        grundwerte = dict(
            prozent=None, prozent_flaeche=None, prozent_gesamt=96.8,
            prozent_flaeche_gesamt=100.0, stand_beginn=None, stand_ende=None,
            zuwachs=None, box_min=[9.5, -2.5, 0.5], box_max=[27.5, 19.0, 7.0],
            volumen_m3=2516.0, flaeche_m2=387.0, voxel_m=0.5, phasen=[],
            quelle="Probe", dauer_s=0.0, strecke_m=0.0, scans=1031,
            scans_phase=0, bag_dauer_s=103.8, bag="probe")
        grundwerte.update(kw)
        return Explorationsgrad(**grundwerte)

    for zustand in ("leer", "rechnet", "ohne", 96.8, 59.7, 31.2):
        a = ExplorationsgradAnzeige(fenster)
        if zustand == "leer":
            a.leeren()
        elif zustand == "rechnet":
            a.rechnet()
        elif zustand == "ohne":
            a.ohne_daten("Das Bag enthält keine EPIC-Explorationsdaten.")
        else:
            a.setze(probe(prozent=zustand, prozent_flaeche=99.0,
                          phasen=[(18.7, 101.7)], dauer_s=83.0))
        a.angeklickt.connect(lambda: print("angeklickt"))
        lay.addWidget(a)
    fenster.resize(360, 260)
    fenster.show()
    print("explorationsgrad: Fenster offen — sechs Zustände von oben nach unten.")
    sys.exit(app.exec_())
