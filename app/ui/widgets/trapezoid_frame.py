"""Plain rectangular module container with a top-aligned title."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QFrame, QLabel, QSizePolicy, QVBoxLayout, QWidget


class TrapezoidFrame(QFrame):
    """Container that paints a simple rounded rectangle with a title at the top.

    (Name kept for import stability; it no longer draws a trapezoid.)
    """

    def __init__(self, title: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("trapezoid_frame")
        # Content-sized height so blocks stay dense and cluster at the top.
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        self._title = title
        self._title_label = QLabel(title)
        self._title_label.setObjectName("trapezoid_title")
        self._title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        layout = QVBoxLayout(self)
        # Title/buttons hug the top-left; tight margins keep the block compact.
        layout.setContentsMargins(14, 8, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(self._title_label, 0, Qt.AlignTop | Qt.AlignLeft)
        self._body_layout = QVBoxLayout()
        layout.addLayout(self._body_layout)

    def body_layout(self) -> QVBoxLayout:
        return self._body_layout

    def setTitle(self, t: str) -> None:  # noqa: N802
        self._title = t
        self._title_label.setText(t)

    def paintEvent(self, _e) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        bg = self.palette().color(self.backgroundRole())
        if not bg.isValid():
            bg = QColor("#2b2b2b")

        rect = self.rect().adjusted(0, 0, -1, -1)
        painter.setBrush(bg.lighter(112))
        pen = QPen(bg.lighter(160))
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawRoundedRect(rect, 6, 6)
