"""Application entry point."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from app.config import ensure_dirs


def main() -> int:
    ensure_dirs()
    app = QApplication(sys.argv)
    # Defer import so config dirs exist before any subwidget queries them
    from app.ui.main_window import MainWindow
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
