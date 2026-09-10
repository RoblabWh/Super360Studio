#!/usr/bin/env python3
"""Einstiegspunkt von Super360 Studio: QApplication + dunkles Theme + MainWindow."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# PyQt5 muss vor qdarktheme importiert werden (Binding-Erkennung).
from PyQt5.QtWidgets import QApplication  # noqa: E402


def main() -> int:
    os.environ.setdefault("DISPLAY", ":0")
    app = QApplication(sys.argv)
    app.setApplicationName("Super360 Studio")
    # Verknüpft Fenster mit dem Starter (Dock-Icon); auf Wayland setzt das die app_id.
    app.setDesktopFileName("super360studio")
    try:
        import qdarktheme
        qdarktheme.setup_theme("dark", custom_colors={"primary": "#4FC3F7"})
    except Exception as exc:  # Theme ist Kosmetik — ohne weiterlaufen
        print(f"qdarktheme nicht aktiv: {exc}", file=sys.stderr)

    from ui.main_window import MainWindow

    win = MainWindow()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
