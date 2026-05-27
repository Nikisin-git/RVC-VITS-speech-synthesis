"""Compact audio player based on QMediaPlayer + QAudioOutput."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSlider, QWidget


class AudioPlayer(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._player = QMediaPlayer(self)
        self._output = QAudioOutput(self)
        self._player.setAudioOutput(self._output)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)

        self._btn = QPushButton("▶")
        self._btn.setFixedWidth(36)
        self._btn.clicked.connect(self._toggle)
        layout.addWidget(self._btn)

        self._pos = QSlider(Qt.Horizontal)
        self._pos.setEnabled(False)
        self._pos.sliderMoved.connect(self._player.setPosition)
        layout.addWidget(self._pos, stretch=1)

        self._label = QLabel("—")
        self._label.setMinimumWidth(80)
        layout.addWidget(self._label)

        self._player.positionChanged.connect(self._on_position)
        self._player.durationChanged.connect(self._on_duration)
        self._player.playbackStateChanged.connect(self._on_state)

    def load(self, path: Path) -> None:
        self._player.stop()
        self._player.setSource(QUrl.fromLocalFile(str(path)))
        self._pos.setEnabled(True)
        self._label.setText(Path(path).name)

    def _toggle(self) -> None:
        if self._player.playbackState() == QMediaPlayer.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def _on_state(self, state) -> None:
        self._btn.setText("⏸" if state == QMediaPlayer.PlayingState else "▶")

    def _on_position(self, ms: int) -> None:
        self._pos.blockSignals(True)
        self._pos.setValue(ms)
        self._pos.blockSignals(False)

    def _on_duration(self, ms: int) -> None:
        self._pos.setRange(0, max(0, ms))
