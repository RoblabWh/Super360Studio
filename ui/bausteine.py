"""Wiederverwendbare Bausteine der Bedienoberfläche."""
from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QComboBox, QDialog, QFormLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)


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
