"""Decorative frame with trapezoidal top blending into a rectangle."""

from __future__ import annotations

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPolygon
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget


class TrapezoidFrame(QFrame):
    """Container that paints a trapezoid 'header' merged into the body rectangle."""

    def __init__(self, title: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("trapezoid_frame")
        self.setMinimumHeight(80)

        self._title = title
        self._header_h = 22
        self._title_label = QLabel(title)
        self._title_label.setObjectName("trapezoid_title")
        # Left-aligned title, cleared from the slanted top-left corner.
        self._title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._title_label.setContentsMargins(18, 0, 0, 0)

        layout = QVBoxLayout(self)
        # Tight margins so the title/buttons hug the top and blocks stay dense.
        layout.setContentsMargins(18, self._header_h + 2, 10, 10)
        layout.setSpacing(6)
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

        w = self.width()
        h = self.height()
        inset = 14
        hh = self._header_h

        path = QPainterPath()
        path.moveTo(0, hh)
        path.lineTo(inset, 0)
        path.lineTo(w - inset, 0)
        path.lineTo(w, hh)
        path.lineTo(w, h)
        path.lineTo(0, h)
        path.closeSubpath()

        bg = self.palette().color(self.backgroundRole())
        if not bg.isValid():
            bg = QColor("#2b2b2b")
        painter.fillPath(path, bg.lighter(110))

        pen = QPen(bg.lighter(160))
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawPolygon(QPolygon([
            QPoint(0, hh), QPoint(inset, 0), QPoint(w - inset, 0),
            QPoint(w, hh), QPoint(w, h - 1), QPoint(0, h - 1),
        ]))
