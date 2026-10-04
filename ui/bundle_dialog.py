"""Dialog fuer den Projekt-Export: was mitkommt und wohin.

Die Auswahl steht in Groessen, nicht in Haken allein — der Unterschied zwischen
einem Export mit und ohne Rosbag sind schnell 24 GB, und das soll man sehen,
bevor man auf Exportieren drueckt.
"""

from __future__ import annotations

import os
import sys

from PyQt5 import QtCore, QtWidgets

try:  # Paket-Import (App) vs. Direktstart des Selbsttests
    from core import bundle
except ImportError:  # pragma: no cover
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core import bundle


class ExportDialog(QtWidgets.QDialog):
    """Teile waehlen und Zielordner bestimmen."""

    def __init__(self, projekt_name: str, info: dict,
                 vorschlag: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Projekt exportieren")
        self.setMinimumWidth(560)
        lay = QtWidgets.QVBoxLayout(self)

        kopf = QtWidgets.QLabel(
            f"<b>{projekt_name}</b><br>Alles Angehakte wird in den Zielordner "
            f"kopiert. Die Punktwolke ist immer dabei, ohne sie gibt es kein "
            f"Projekt.")
        kopf.setWordWrap(True)
        lay.addWidget(kopf)

        kasten = QtWidgets.QGroupBox("Inhalt")
        form = QtWidgets.QVBoxLayout(kasten)
        self._boxen: dict = {}
        for key, (label, pflicht) in bundle.TEILE.items():
            eintrag = info.get(key, {"da": False, "bytes": 0})
            cb = QtWidgets.QCheckBox()
            cb.setText(f"{label} — {bundle.fmt_size(eintrag['bytes'])}"
                       if eintrag["da"] else f"{label} — nicht vorhanden")
            cb.setEnabled(bool(eintrag["da"]) and not pflicht)
            cb.setChecked(bool(eintrag["da"]) and (pflicht or key != "bags"))
            if key == "bags" and eintrag["da"]:
                cb.setToolTip(
                    "Die Rohdaten. Ohne sie bleiben Karte, Farben, Messen und\n"
                    "Export erhalten; das 360°-Video und ein erneutes Einfärben\n"
                    "brauchen das Bag.")
            if key == "colors" and eintrag.get("ebenen"):
                cb.setToolTip("Ebenen: " + ", ".join(eintrag["ebenen"]))
            cb.stateChanged.connect(self._summe)
            form.addWidget(cb)
            self._boxen[key] = (cb, eintrag)
        lay.addWidget(kasten)

        self._lbl_summe = QtWidgets.QLabel()
        lay.addWidget(self._lbl_summe)

        zeile = QtWidgets.QHBoxLayout()
        self._pfad = QtWidgets.QLineEdit(vorschlag)
        self._pfad.setPlaceholderText("Zielordner — leer oder neu")
        self._pfad.textChanged.connect(self._pruefen)
        knopf = QtWidgets.QPushButton("Durchsuchen …")
        knopf.clicked.connect(self._waehlen)
        zeile.addWidget(QtWidgets.QLabel("Ziel:"))
        zeile.addWidget(self._pfad, 1)
        zeile.addWidget(knopf)
        lay.addLayout(zeile)

        self._hinweis = QtWidgets.QLabel()
        self._hinweis.setWordWrap(True)
        lay.addWidget(self._hinweis)

        self._buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, self)
        self._buttons.button(QtWidgets.QDialogButtonBox.Ok).setText("Exportieren")
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        lay.addWidget(self._buttons)

        self._summe()
        self._pruefen()

    # ------------------------------------------------------------------ intern

    def _waehlen(self) -> None:
        start = self._pfad.text() or os.path.expanduser("~")
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Zielordner wählen oder anlegen", start,
            QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.DontUseNativeDialog)
        if d:
            self._pfad.setText(d)

    def _summe(self) -> None:
        n = sum(e["bytes"] for cb, e in self._boxen.values()
                if cb.isChecked() and e["da"])
        self._lbl_summe.setText(f"Zusammen etwa <b>{bundle.fmt_size(n)}</b>")

    #: Hinweis je Zustand des Zielordners aus ``bundle.pruefe_ziel``
    _HINWEISE = {
        "neu": "Wird angelegt.",
        "projekt": ("Der Ordner enthält bereits ein Projekt — es wird "
                    "überschrieben."),
        "fremd": ("Der Ordner ist nicht leer und enthält kein Projekt. "
                  "Bitte einen leeren oder neuen wählen."),
    }

    def _pruefen(self) -> None:
        p = self._pfad.text().strip()
        ok, zustand = bundle.pruefe_ziel(p) if p else (False, "")
        if zustand == "datei":      # eine Datei als Ziel weist erst der Export ab
            ok = True
        self._hinweis.setText(self._HINWEISE.get(zustand, ""))
        self._buttons.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(ok)

    # ----------------------------------------------------------------- Ergebnis

    def auswahl(self) -> dict:
        return {k: bool(cb.isChecked()) for k, (cb, _e) in self._boxen.items()}

    def ziel(self) -> str:
        return self._pfad.text().strip()


class ProjectOpenDialog(QtWidgets.QDialog):
    """Vorhandene Projekte des Caches zur Auswahl.

    Der normale Weg oeffnet ein Rosbag; zusammengefuehrte Projekte haben aber
    keinen Bagpfad, unter dem man sie wiederfaende — ihr Schluessel ist ein
    erfundener Pfad "A+B". Ohne diese Liste waeren sie nach dem Schliessen weg.
    """

    def __init__(self, projekte: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Projekt öffnen")
        self.resize(760, 420)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(QtWidgets.QLabel(
            "Berechnete Projekte im Cache. Zusammengeführte sind markiert — "
            "sie lassen sich nur hier öffnen."))
        self._tab = QtWidgets.QTableWidget(len(projekte), 5, self)
        self._tab.setHorizontalHeaderLabels(
            ["Projekt", "Scans", "Punkte", "Art", "berechnet"])
        self._tab.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._tab.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self._tab.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self._tab.verticalHeader().setVisible(False)
        for r, e in enumerate(projekte):
            werte = (e["name"],
                     f"{e['n_scans']}",
                     f"{e['n_points'] / 1e6:.1f} Mio.",
                     "zusammengeführt" if e["zusammengefuehrt"] else "einzeln",
                     (e["created"] or "").replace("T", " ")[:16])
            for c, v in enumerate(werte):
                it = QtWidgets.QTableWidgetItem(v)
                if c == 0:
                    it.setData(QtCore.Qt.UserRole, e["dir"])
                    tip = e["bag"]
                    if e["quellen"]:
                        tip = "\n".join(e["quellen"])
                    it.setToolTip(tip)
                self._tab.setItem(r, c, it)
        self._tab.resizeColumnsToContents()
        self._tab.horizontalHeader().setStretchLastSection(True)
        self._tab.doubleClicked.connect(lambda *_: self.accept())
        if projekte:
            self._tab.selectRow(0)
        lay.addWidget(self._tab, 1)

        knoepfe = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Open | QtWidgets.QDialogButtonBox.Cancel, self)
        knoepfe.accepted.connect(self.accept)
        knoepfe.rejected.connect(self.reject)
        knoepfe.button(QtWidgets.QDialogButtonBox.Open).setEnabled(bool(projekte))
        lay.addWidget(knoepfe)

    def gewaehlt(self) -> str | None:
        r = self._tab.currentRow()
        if r < 0:
            return None
        it = self._tab.item(r, 0)
        return it.data(QtCore.Qt.UserRole) if it else None


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    info = {
        "recording": {"da": True, "bytes": 402_653_184},
        "colors": {"da": True, "bytes": 289_406_976, "ebenen": ["onboard", "meander_rgb"]},
        "panos": {"da": True, "bytes": 1_610_612_736, "breiten": [1920]},
        "meander": {"da": True, "bytes": 268_435_456},
        "bags": {"da": True, "bytes": 25_769_803_776, "pfade": ["/x/bag"]},
    }
    d = ExportDialog("rosbag_2026-09-08_12-56-28", info, "/tmp/export")
    assert d.auswahl()["recording"] and not d.auswahl()["bags"], d.auswahl()
    assert "GB" in d._lbl_summe.text(), d._lbl_summe.text()
    print("Vorgabe ohne Bag:", d._lbl_summe.text())
    d._boxen["bags"][0].setChecked(True)
    assert d.auswahl()["bags"]
    print("mit Bag:         ", d._lbl_summe.text())
    d._pfad.setText("/tmp")           # nicht leer, kein Projekt -> abgelehnt
    ok = d._buttons.button(QtWidgets.QDialogButtonBox.Ok).isEnabled()
    print("/tmp erlaubt?", ok, "|", d._hinweis.text()[:50])
    assert not ok
    d._pfad.setText("/tmp/gibtsnochnicht")
    assert d._buttons.button(QtWidgets.QDialogButtonBox.Ok).isEnabled()
    print("neuer Ordner:", d._hinweis.text())
    projekte = [
        {"dir": "/c/a-1", "name": "flug_a", "bag": "/b/a", "n_scans": 100,
         "n_points": 1_000_000, "created": "2026-09-10T17:40", "quellen": [],
         "zusammengefuehrt": False},
        {"dir": "/c/ab-2", "name": "flug_a+flug_b", "bag": "/b/a", "n_scans": 300,
         "n_points": 24_300_000, "created": "2026-09-10T18:00",
         "quellen": ["/b/a", "/b/b"], "zusammengefuehrt": True},
    ]
    o = ProjectOpenDialog(projekte)
    assert o.gewaehlt() == "/c/a-1", o.gewaehlt()
    o._tab.selectRow(1)
    assert o.gewaehlt() == "/c/ab-2"
    assert o._tab.item(1, 3).text() == "zusammengeführt"
    print("Projektliste:", [o._tab.item(r, 0).text() for r in range(o._tab.rowCount())])
    leer = ProjectOpenDialog([])
    assert leer.gewaehlt() is None
    print("bundle_dialog SELFTEST OK")
