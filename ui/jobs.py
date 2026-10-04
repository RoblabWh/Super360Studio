"""Der Hintergrund-Arbeiter des Hauptfensters.

Alle langen Schritte laufen in einem :class:`Worker` (QThread). Der Job bekommt
``progress_cb``, ``cancel`` und ``log_cb`` und fasst keine Qt-Widgets an; was er
meldet, kommt über Signale in den GUI-Thread.
"""
from __future__ import annotations

import threading
import traceback
from typing import Callable

from PyQt5.QtCore import QThread, pyqtSignal


class Worker(QThread):
    """Generischer Hintergrund-Arbeiter: ``fn(progress_cb, cancel, log_cb)``.

    ``cancellable=False`` für Jobs, die aus einem einzelnen Bibliotheksaufruf
    bestehen (Export, Overlay) und das Cancel-Event ohnehin nicht auswerten
    können — der Abbrechen-Knopf bleibt dann ehrlich deaktiviert.
    """

    progress = pyqtSignal(float, str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, fn: Callable, parent=None, cancellable: bool = True):
        super().__init__(parent)
        self._fn = fn
        self.cancel = threading.Event()
        self.cancellable = bool(cancellable)

    def run(self) -> None:
        # Emits bewusst außerhalb des except-Blocks (kein aktiver Exception-
        # Zustand während der Signal-Zustellung).
        result = None
        error: str | None = None
        error_line = ""
        try:
            result = self._fn(progress_cb=self._emit_progress,
                              cancel=self.cancel, log_cb=self.log.emit)
        except Exception as exc:  # noqa: BLE001 — UI zeigt die Meldung
            error = str(exc) or exc.__class__.__name__
            error_line = "".join(traceback.format_exception_only(exc)).strip()
        if error is not None:
            self.log.emit(error_line)
            self.failed.emit(error)
        else:
            self.finished.emit(result)

    def _emit_progress(self, frac: float, msg: str) -> None:
        self.progress.emit(float(frac), str(msg))
