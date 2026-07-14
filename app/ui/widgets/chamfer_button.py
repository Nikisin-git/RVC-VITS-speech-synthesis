"""Push button with 45°-cut (chamfered) corners, custom-painted.

QSS `border-radius` only rounds corners; a true 45° cut needs an octagonal
shape drawn by hand. This button paints that polygon, wraps long labels, and
reacts to hover/press. Colours are set per theme via set_theme_colors().
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygon
from PySide6.QtWidgets import QPushButton, QWidget


class ChamferButton(QPushButton):
    def __init__(self, text: str = "", chamfer: int = 12,
                 parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self._chamfer = chamfer
        self._hover = False
        # Alumni Sans is condensed/tall, so it needs a larger point size than
        # a normal sans to read at the same visual weight.
        self._font_family = "Alumni Sans"
        self._font_pt = 19
        self.setCursor(Qt.PointingHandCursor)
        # Wide enough that the longest label fits on a single line.
        self.setFixedWidth(300)
        self.setMinimumHeight(50)

        # Default (dark-gray) colour scheme; overridden by set_theme_colors().
        self._base = QColor("#3a3a3e")
        self._hover_c = QColor("#4a4a50")
        self._pressed = QColor("#2a2a2e")
        self._border = QColor("#5a5a62")
        self._text = QColor("#f0f0f0")

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
        # Build the font EXPLICITLY (family + size + weight). The themes'
        # global `* { font-size: 12px }` QSS rule overrides setFont() on the
        # widget, so the painter font is the only reliable control. Alumni Sans
        # is a variable font whose default instance is Thin, so we pin Normal
        # weight; if the family isn't loaded Qt falls back automatically.
        font = QFont(self._font_family, self._font_pt)
        font.setWeight(QFont.Weight.Normal)
        p.setFont(font)
        # Single line, centered, small interior margin.
        p.drawText(self.rect().adjusted(8, 4, -8, -4),
                   Qt.AlignCenter, self.text())
