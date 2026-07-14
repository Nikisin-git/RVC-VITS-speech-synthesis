"""Push button with 45°-cut (chamfered) corners, custom-painted.

QSS `border-radius` only rounds corners; a true 45° cut needs an octagonal
shape drawn by hand. This button paints that polygon, wraps long labels, and
reacts to hover/press. Colours are set per theme via set_theme_colors().
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
        self.setFixedWidth(230)
        self.setMinimumHeight(66)
        # Larger font so labels nearly reach the button edges.
        f = self.font()
        f.setPointSize(12)
        self.setFont(f)

        # Default (dark) colour scheme; overridden by set_theme_colors().
        self._base = QColor("#3a4a6a")
        self._hover_c = QColor("#4a5f88")
        self._pressed = QColor("#2c3a55")
        self._border = QColor("#5a7ab0")
        self._text = QColor("#f0f4ff")

    def set_theme_colors(self, base: str, hover: str, pressed: str,
                         border: str, text: str) -> None:
        self._base = QColor(base)
        self._hover_c = QColor(hover)
        self._pressed = QColor(pressed)
        self._border = QColor(border)
        self._text = QColor(text)
        self.update()

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
            bg = self._base.darker(140)
            border = self._border.darker(140)
            fg = QColor("#888")
        elif self.isDown():
            bg, border, fg = self._pressed, self._border.lighter(120), self._text
        elif self._hover:
            bg, border, fg = self._hover_c, self._border.lighter(130), self._text
        else:
            bg, border, fg = self._base, self._border, self._text

        pen = QPen(border)
        pen.setWidth(2)
        p.setPen(pen)
        p.setBrush(bg)
        p.drawPolygon(poly)

        p.setPen(fg)
        p.drawText(self.rect().adjusted(4, 3, -4, -3),
                   Qt.AlignCenter | Qt.TextWordWrap, self.text())
