"""Ausklappbare Abschnitte fuer die Seitenleiste.

Ein Abschnitt ist eine Kopfzeile mit Pfeil plus ein beliebiges Inhalts-Widget.
Klick auf die Kopfzeile klappt zu und wieder auf; der Zustand laesst sich lesen
und setzen, damit ihn das Fenster in den Einstellungen ablegen kann.

Bewusst kein QToolBox: der laesst immer nur einen Abschnitt offen. Beim Arbeiten
mit dieser Anwendung sind aber regelmaessig mehrere gleichzeitig im Blick, etwa
die Einfaerbung und die Anzeige.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

_PFEIL_AUF = "▾"   # nach unten zeigend = offen
_PFEIL_ZU = "▸"    # nach rechts zeigend = geschlossen


class Section(QtWidgets.QWidget):
    """Kopfzeile mit Pfeil, darunter der Inhalt; klappt auf Klick zu."""

    toggled = QtCore.pyqtSignal(str, bool)  # Schluessel, offen

    def __init__(self, key: str, title: str, content: QtWidgets.QWidget,
                 expanded: bool = True, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.key = str(key)
        self._content = content
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self._head = QtWidgets.QToolButton(self)
        self._head.setCheckable(True)
        self._head.setChecked(bool(expanded))
        self._head.setAutoRaise(True)
        self._head.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self._head.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                 QtWidgets.QSizePolicy.Fixed)
        self._head.setCursor(QtCore.Qt.PointingHandCursor)
        self._title = str(title)
        self._head.setStyleSheet(
            "QToolButton { border: none; text-align: left; padding: 6px 4px;"
            " font-weight: 600; }"
            "QToolButton:hover { background: rgba(255,255,255,22); border-radius: 4px; }")
        self._head.clicked.connect(self._on_clicked)
        lay.addWidget(self._head)
        lay.addWidget(content)
        self._sync()

    # ---------------------------------------------------------------- Zustand

    def is_expanded(self) -> bool:
        return bool(self._head.isChecked())

    def set_expanded(self, on: bool) -> None:
        if bool(on) == self.is_expanded():
            return
        self._head.setChecked(bool(on))
        self._sync()

    def set_title(self, title: str) -> None:
        self._title = str(title)
        self._sync()

    def content(self) -> QtWidgets.QWidget:
        return self._content

    # ------------------------------------------------------------------ intern

    def _on_clicked(self) -> None:
        self._sync()
        self.toggled.emit(self.key, self.is_expanded())

    def _sync(self) -> None:
        offen = self.is_expanded()
        self._head.setText(f"{_PFEIL_AUF if offen else _PFEIL_ZU}  {self._title}")
        self._content.setVisible(offen)


class SectionStack(QtWidgets.QWidget):
    """Mehrere :class:`Section` untereinander, mit Zustand als dict."""

    toggled = QtCore.pyqtSignal(str, bool)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self._lay = QtWidgets.QVBoxLayout(self)
        self._lay.setContentsMargins(4, 4, 4, 4)
        self._lay.setSpacing(2)
        self._sections: dict[str, Section] = {}

    def add(self, key: str, title: str, content: QtWidgets.QWidget,
            expanded: bool = True) -> Section:
        sec = Section(key, title, content, expanded, self)
        sec.toggled.connect(self.toggled)
        self._sections[sec.key] = sec
        self._lay.addWidget(sec)
        return sec

    def finish(self) -> None:
        """Nach dem letzten add aufrufen: schiebt alles nach oben zusammen."""
        self._lay.addStretch(1)

    def section(self, key: str) -> Section | None:
        return self._sections.get(str(key))

    def states(self) -> dict:
        return {k: s.is_expanded() for k, s in self._sections.items()}

    def set_states(self, states: dict) -> None:
        for k, on in (states or {}).items():
            sec = self._sections.get(str(k))
            if sec is not None:
                sec.set_expanded(bool(on))

    def set_all(self, on: bool) -> None:
        for sec in self._sections.values():
            sec.set_expanded(bool(on))


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    stack = SectionStack()
    for i, (k, t) in enumerate((("a", "Aufnahme"), ("b", "Karte"), ("c", "Anzeige"))):
        inner = QtWidgets.QWidget()
        il = QtWidgets.QVBoxLayout(inner)
        il.addWidget(QtWidgets.QPushButton(f"Knopf in {t}"))
        stack.add(k, t, inner, expanded=(i == 0))
    stack.finish()
    assert stack.states() == {"a": True, "b": False, "c": False}, stack.states()
    stack.section("b").set_expanded(True)
    assert stack.states()["b"] is True
    stack.set_states({"a": False, "c": True})
    assert stack.states() == {"a": False, "b": True, "c": True}, stack.states()
    stack.set_all(False)
    assert not any(stack.states().values())
    print("collapsible SELFTEST OK:", stack.states())
