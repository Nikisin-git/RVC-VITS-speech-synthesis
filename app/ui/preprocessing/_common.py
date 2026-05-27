"""Shared helpers for preprocessing windows."""

from __future__ import annotations

from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QLabel, QRadioButton, QWidget


def make_format_radio(parent: QWidget) -> tuple[QHBoxLayout, QButtonGroup, QRadioButton, QRadioButton]:
    """Build a `Формат: () wav  () mp3` row."""
    layout = QHBoxLayout()
    layout.addWidget(QLabel("Формат сохранения:"))
    rb_wav = QRadioButton("wav")
    rb_mp3 = QRadioButton("mp3")
    rb_wav.setChecked(True)
    group = QButtonGroup(parent)
    group.addButton(rb_wav)
    group.addButton(rb_mp3)
    layout.addWidget(rb_wav)
    layout.addWidget(rb_mp3)
    layout.addStretch(1)
    return layout, group, rb_wav, rb_mp3
