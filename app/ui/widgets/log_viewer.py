"""Dark log viewer for tailing subprocess output."""

from __future__ import annotations

from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import QPlainTextEdit, QWidget


class LogViewer(QPlainTextEdit):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setObjectName("log_viewer")
        self.setFont(QFont("Consolas", 9))
        self.setMaximumBlockCount(10_000)

    def append_line(self, line: str) -> None:
        self.appendPlainText(line)
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.setTextCursor(cursor)
