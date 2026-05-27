"""Slider with a synchronized numeric input."""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox, QHBoxLayout, QLabel, QSlider, QSpinBox, QVBoxLayout, QWidget,
)


class SliderWithInput(QWidget):
    value_changed = Signal(float)

    def __init__(
        self,
        label: str,
        minimum: float,
        maximum: float,
        step: float = 1.0,
        default: float | None = None,
        decimals: int = 0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._min = float(minimum)
        self._max = float(maximum)
        self._step = float(step)
        self._decimals = int(decimals)
        self._scale = 10 ** self._decimals

        if default is None:
            default = minimum

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.addWidget(QLabel(label))

        row = QHBoxLayout()
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(int(self._min * self._scale), int(self._max * self._scale))
        self._slider.setSingleStep(max(1, int(self._step * self._scale)))
        row.addWidget(self._slider, stretch=1)

        if decimals == 0:
            self._input: QSpinBox | QDoubleSpinBox = QSpinBox()
            self._input.setRange(int(self._min), int(self._max))
            self._input.setSingleStep(int(self._step))
        else:
            self._input = QDoubleSpinBox()
            self._input.setDecimals(self._decimals)
            self._input.setRange(self._min, self._max)
            self._input.setSingleStep(self._step)
        self._input.setFixedWidth(80)
        row.addWidget(self._input)

        layout.addLayout(row)

        self._slider.valueChanged.connect(self._on_slider)
        self._input.valueChanged.connect(self._on_input)
        self.set_value(default)

    def _on_slider(self, raw: int) -> None:
        val = raw / self._scale
        if abs(self._input.value() - val) > 1e-6:
            self._input.blockSignals(True)
            self._input.setValue(val if self._decimals else int(val))
            self._input.blockSignals(False)
        self.value_changed.emit(float(val))

    def _on_input(self, val: float | int) -> None:
        raw = int(round(float(val) * self._scale))
        if self._slider.value() != raw:
            self._slider.blockSignals(True)
            self._slider.setValue(raw)
            self._slider.blockSignals(False)
        self.value_changed.emit(float(val))

    def value(self) -> float:
        return float(self._input.value())

    def set_value(self, val: float) -> None:
        clamped = max(self._min, min(self._max, float(val)))
        self._input.setValue(clamped if self._decimals else int(clamped))


def build_recommendation_label(builder: Callable[[], str]) -> QLabel:
    lbl = QLabel(builder())
    lbl.setObjectName("recommendation")
    return lbl
