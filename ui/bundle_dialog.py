"""Dialog fuer den Projekt-Export: was mitkommt und wohin.

Die Auswahl steht in Groessen, nicht in Haken allein — der Unterschied zwischen
einem Export mit und ohne Rosbag sind schnell 24 GB, und das soll man sehen,
bevor man auf Exportieren drueckt.
"""

from __future__ import annotations

import os
import sys

from PyQt5 import QtWidgets

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

    def _pruefen(self) -> None:
        p = self._pfad.text().strip()
        ok = bool(p)
        text = ""
        if p and os.path.isdir(p) and os.listdir(p):
            if os.path.isfile(os.path.join(p, bundle.MANIFEST)):
                text = ("Der Ordner enthält bereits ein Projekt — es wird "
                        "überschrieben.")
            else:
                text = ("Der Ordner ist nicht leer und enthält kein Projekt. "
                        "Bitte einen leeren oder neuen wählen.")
                ok = False
        elif p and not os.path.exists(p):
            text = "Wird angelegt."
        self._hinweis.setText(text)
        self._buttons.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(ok)

    # ----------------------------------------------------------------- Ergebnis

    def auswahl(self) -> dict:
        return {k: bool(cb.isChecked()) for k, (cb, _e) in self._boxen.items()}

    def ziel(self) -> str:
        return self._pfad.text().strip()


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
    print("bundle_dialog SELFTEST OK")
