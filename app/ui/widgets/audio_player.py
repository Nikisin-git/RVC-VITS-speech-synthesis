"""Compact audio player that decodes to PCM and streams via QAudioSink.

Why not QMediaPlayer: its file playback delegates to a system backend
(Windows Media Foundation by default), which prematurely emits EndOfMedia on
longer PCM-WAV files — the saved file is complete, but the tail is dropped on
playback. The bug scales with file length, so slow-tempo (= longer) TTS output
gets cut off while fast-tempo output plays fine.

To sidestep the whole class of backend/codec bugs we decode the file to int16
PCM ourselves (via librosa, which also handles mp3/flac) and feed the raw
samples to QAudioSink. Playback then always runs to the true end of the buffer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QBuffer, QByteArray, QIODevice, Qt, QTimer
from PySide6.QtMultimedia import QAudio, QAudioFormat, QAudioSink, QMediaDevices
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSlider, QWidget


def _fmt_time(ms: int) -> str:
    s = max(0, ms) // 1000
    return f"{s // 60}:{s % 60:02d}"


class AudioPlayer(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sink: QAudioSink | None = None
        self._qba = QByteArray()
        self._buffer = QBuffer(self)
        self._sr = 22050
        self._channels = 1
        self._bytes_per_frame = 2
        self._total_frames = 0
        self._duration_ms = 0
        self._base_frame = 0     # frame offset from which the current start() runs
        self._finished = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)

        self._btn = QPushButton("▶")
        self._btn.setFixedWidth(36)
        self._btn.setEnabled(False)
        self._btn.clicked.connect(self._toggle)
        layout.addWidget(self._btn)

        self._pos = QSlider(Qt.Horizontal)
        self._pos.setEnabled(False)
        self._pos.sliderMoved.connect(self._seek_ms)
        layout.addWidget(self._pos, stretch=1)

        self._label = QLabel("—")
        self._label.setMinimumWidth(96)
        layout.addWidget(self._label)

        # Polls processedUSecs() to advance the position slider while playing.
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._tick)

    # --- loading ---
    def load(self, path: Path) -> None:
        self._teardown()
        import librosa
        # sr=None keeps the native rate; mono=False preserves stereo if present.
        y, sr = librosa.load(str(path), sr=None, mono=False)
        if y.ndim == 1:
            y = y[np.newaxis, :]           # (1, N)
        y = np.clip(y, -1.0, 1.0)
        pcm = (y.T * 32767.0).astype("<i2")  # (N, C) interleaved little-endian
        pcm = np.ascontiguousarray(pcm)

        self._sr = int(sr)
        self._channels = int(y.shape[0])
        self._bytes_per_frame = 2 * self._channels
        self._total_frames = int(y.shape[1])
        self._duration_ms = int(self._total_frames * 1000 / max(1, self._sr))

        self._qba = QByteArray(pcm.tobytes())
        self._buffer = QBuffer(self)
        self._buffer.setData(self._qba)
        self._buffer.open(QIODevice.ReadOnly)

        fmt = QAudioFormat()
        fmt.setSampleRate(self._sr)
        fmt.setChannelCount(self._channels)
        fmt.setSampleFormat(QAudioFormat.Int16)
        self._sink = QAudioSink(QMediaDevices.defaultAudioOutput(), fmt, self)
        self._sink.stateChanged.connect(self._on_state)

        self._base_frame = 0
        self._finished = False
        self._pos.setRange(0, self._duration_ms)
        self._pos.setValue(0)
        self._pos.setEnabled(True)
        self._btn.setEnabled(True)
        self._btn.setText("▶")
        self._label.setText(f"0:00 / {_fmt_time(self._duration_ms)}")

    # --- controls ---
    def _toggle(self) -> None:
        if self._sink is None:
            return
        st = self._sink.state()
        if st == QAudio.ActiveState:
            self._sink.suspend()
            self._timer.stop()
            self._btn.setText("▶")
        elif st == QAudio.SuspendedState:
            self._sink.resume()
            self._timer.start()
            self._btn.setText("⏸")
        else:
            # Stopped / Idle / fresh: (re)start. Replay from the top if we
            # reached the end last time, otherwise from the current marker.
            self._start_from(0 if self._finished else self._base_frame)

    def _start_from(self, frame: int) -> None:
        if self._sink is None:
            return
        frame = max(0, min(frame, self._total_frames))
        if self._sink.state() != QAudio.StoppedState:
            self._sink.stop()
        self._buffer.seek(frame * self._bytes_per_frame)
        self._base_frame = frame
        self._finished = False
        self._sink.start(self._buffer)
        self._timer.start()
        self._btn.setText("⏸")

    def _seek_ms(self, ms: int) -> None:
        if self._sink is None:
            return
        frame = int(ms * self._sr / 1000)
        if self._sink.state() in (QAudio.ActiveState, QAudio.SuspendedState):
            self._start_from(frame)
        else:
            # Paused/stopped: move the marker; playback will resume from here.
            self._base_frame = frame
            self._finished = False
            self._pos.setValue(ms)
            self._label.setText(f"{_fmt_time(ms)} / {_fmt_time(self._duration_ms)}")

    # --- updates ---
    def _current_frame(self) -> int:
        if self._sink is None:
            return self._base_frame
        # processedUSecs() counts audio played since the last start(); it holds
        # steady while suspended and resets on stop(). _base_frame carries the
        # seek origin so the absolute position stays correct across seeks.
        played = self._sink.processedUSecs() * self._sr // 1_000_000
        return self._base_frame + int(played)

    def _tick(self) -> None:
        ms = int(self._current_frame() * 1000 / max(1, self._sr))
        ms = min(ms, self._duration_ms)
        self._pos.blockSignals(True)
        self._pos.setValue(ms)
        self._pos.blockSignals(False)
        self._label.setText(f"{_fmt_time(ms)} / {_fmt_time(self._duration_ms)}")

    def _on_state(self, state) -> None:
        # IdleState with an in-memory buffer means the data ran out = finished.
        if state == QAudio.IdleState:
            self._finished = True
            self._timer.stop()
            self._btn.setText("▶")
            self._pos.setValue(self._duration_ms)
            self._label.setText(
                f"{_fmt_time(self._duration_ms)} / {_fmt_time(self._duration_ms)}")

    def _teardown(self) -> None:
        self._timer.stop()
        if self._sink is not None:
            self._sink.stop()
            self._sink = None
        if self._buffer.isOpen():
            self._buffer.close()
