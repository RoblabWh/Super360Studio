"""Ausklappbare Abschnitte fuer die Seitenleiste.

Ein Abschnitt ist eine Kopfzeile mit Pfeil plus ein beliebiges Inhalts-Widget.
Klick auf die Kopfzeile klappt zu und wieder auf; der Zustand laesst sich lesen
und setzen, damit ihn das Fenster in den Einstellungen ablegen kann.

Bewusst kein QToolBox: der laesst immer nur einen Abschnitt offen. Beim Arbeiten
mit dieser Anwendung sind aber regelmaessig mehrere gleichzeitig im Blick, etwa
die Einfaerbung und die Anzeige.

Ein einklappbarer Unterblock im Inhalt eines Abschnitts ist selbst eine
:class:`Section`. Mit :meth:`SectionStack.melde_an` angemeldet, steht sein
Zustand unter einem Schluessel wie ``maeander.hauptpunkt`` neben denen der
Abschnitte, und sein Auf- und Zuklappen meldet der Stapel wie das eines
Abschnitts.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

_PFEIL_AUF = "▾"   # nach unten zeigend = offen
_PFEIL_ZU = "▸"    # nach rechts zeigend = geschlossen


class Section(QtWidgets.QWidget):
    """Kopfzeile mit Pfeil, darunter der Inhalt; klappt auf Klick zu."""

    toggled = QtCore.pyqtSignal(str, bool)  # Schluessel, offen
    #: Klasse der Kopfzeile; ein Unterblock nimmt eine, deren Text umbricht
    _kopf_klasse = QtWidgets.QToolButton

    def __init__(self, key: str, title: str, content: QtWidgets.QWidget,
                 expanded: bool = True, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.key = str(key)
        self._content = content
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self._head = self._kopf_klasse(self)
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

    def content(self) -> QtWidgets.QWidget:
        return self._content

    def einklappbar(self) -> bool:
        return True

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
        self._unterbloecke: dict[str, Section] = {}

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

    def melde_an(self, section: Section) -> Section:
        """Einklappbaren Unterblock anmelden: Zustand und toggled wie ein Abschnitt."""
        key = section.key
        if not key or not section.einklappbar():
            raise ValueError(f"Unterblock ohne Schluessel oder nicht einklappbar: {key!r}")
        if key in self._sections or key in self._unterbloecke:
            raise ValueError(f"Schluessel doppelt: {key!r}")
        section.toggled.connect(self.toggled)
        self._unterbloecke[key] = section
        return section

    def section(self, key: str) -> Section | None:
        return self._sections.get(str(key))

    def sections(self, mit_unterbloecken: bool = False) -> list[Section]:
        """Alle Abschnitte in Bauordnung, auf Wunsch samt Unterbloecken – auch zugeklappte."""
        alle = list(self._sections.values())
        if mit_unterbloecken:
            alle += list(self._unterbloecke.values())
        return alle

    def kopfbreite(self) -> int:
        """Breite, in der jeder Abschnittskopf ganz zu lesen ist, samt Rand des Stapels."""
        m = self._lay.contentsMargins()
        kopf = max((s._head.sizeHint().width() for s in self.sections()), default=0)
        return kopf + m.left() + m.right()

    def states(self) -> dict:
        return {s.key: s.is_expanded() for s in self.sections(mit_unterbloecken=True)}

    def set_states(self, states: dict) -> None:
        for k, on in (states or {}).items():
            sec = self._sections.get(str(k))
            if sec is None:
                sec = self._unterbloecke.get(str(k))
            if sec is not None:
                sec.set_expanded(bool(on))

    def set_all(self, on: bool) -> None:
        for sec in self.sections(mit_unterbloecken=True):
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

    # Unterblock im Inhalt von "b": eigener Zustand, Umschalten meldet der Stapel
    gemeldet = []
    stack.toggled.connect(lambda k, on: gemeldet.append((k, on)))
    unter = Section("b.unter", "Unterblock", QtWidgets.QLabel("Inhalt"), expanded=False)
    stack.section("b").content().layout().addWidget(unter)
    assert stack.melde_an(unter) is unter
    assert stack.states() == {"a": False, "b": False, "c": False, "b.unter": False}, stack.states()
    unter._head.click()
    assert gemeldet == [("b.unter", True)], gemeldet
    assert stack.states()["b.unter"] is True
    stack.set_states({"b.unter": False, "a": True})
    assert stack.states() == {"a": True, "b": False, "c": False, "b.unter": False}, stack.states()
    stack.set_all(True)
    assert all(stack.states().values())
    stack.set_all(False)
    assert not any(stack.states().values())
    assert stack.sections() == [stack.section(k) for k in "abc"]
    assert stack.kopfbreite() == 8 + max(s._head.sizeHint().width() for s in stack.sections())
    assert stack.sections(mit_unterbloecken=True)[-1] is unter     # auch zugeklappt
    for falsch in (unter, Section("a", "gleich", QtWidgets.QWidget()),
                   Section("", "ohne", QtWidgets.QWidget())):
        try:
            stack.melde_an(falsch)
        except ValueError:
            continue
        raise AssertionError(f"melde_an nimmt {falsch.key!r} an")
    print("collapsible SELFTEST OK:", stack.states())
