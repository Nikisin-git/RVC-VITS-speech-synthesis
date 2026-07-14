"""Push button with 45°-cut (chamfered) corners, custom-painted.

QSS `border-radius` only rounds corners; a true 45° cut needs an octagonal
shape drawn by hand. This button paints that polygon, wraps long labels, and
reacts to hover/press. Colours are theme-neutral so it looks consistent
across the app's four themes.
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPolygon
from PySide6.QtWidgets import QPushButton, QWidget


class ChamferButton(QPushButton):
    def __init__(self, text: str = "", chamfer: int = 12,
                 parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self._chamfer = chamfer
        self._hover = False
        self.setCursor(Qt.PointingHandCursor)
        # Narrow, square-ish footprint.
        self.setFixedWidth(210)
        self.setMinimumHeight(64)

    def enterEvent(self, e) -> None:  # noqa: N802
        self._hover = True
        self.update()
        super().enterEvent(e)

    def leaveEvent(self, e) -> None:  # noqa: N802
        self._hover = False
        self.update()
        super().leaveEvent(e)

    def paintEvent(self, _e) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        c = min(self._chamfer, w // 2, h // 2)
        poly = QPolygon([
            QPoint(c, 0), QPoint(w - c, 0),
            QPoint(w, c), QPoint(w, h - c),
            QPoint(w - c, h), QPoint(c, h),
            QPoint(0, h - c), QPoint(0, c),
        ])

        if not self.isEnabled():
            bg, border, fg = QColor("#3a3a40"), QColor("#4a4a52"), QColor("#888")
        elif self.isDown():
            bg, border, fg = QColor("#2c3a55"), QColor("#5a7ab0"), QColor("#eaf0ff")
        elif self._hover:
            bg, border, fg = QColor("#4a5f88"), QColor("#7a9cd8"), QColor("#ffffff")
        else:
            bg, border, fg = QColor("#3a4a6a"), QColor("#5a7ab0"), QColor("#f0f4ff")

        pen = QPen(border)
        pen.setWidth(2)
        p.setPen(pen)
        p.setBrush(bg)
        p.drawPolygon(poly)

        p.setPen(fg)
        # Word-wrap so long labels fit the narrow button.
        p.drawText(self.rect().adjusted(6, 4, -6, -4),
                   Qt.AlignCenter | Qt.TextWordWrap, self.text())
