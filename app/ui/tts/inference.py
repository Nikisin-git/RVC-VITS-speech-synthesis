"""Window: VITS / Coqui-TTS inference."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox, QHBoxLayout, QLabel, QMessageBox, QPushButton, QTextEdit, QVBoxLayout, QWidget,
)

from app.config import SCRIPTS_DIR, VITS_DIR
from app.ui.preprocessing._common import make_format_radio
from app.ui.widgets.audio_player import AudioPlayer
from app.ui.widgets.drag_drop_area import DragDropArea
from app.ui.widgets.progress_dialog import ProgressDialog
from app.ui.widgets.slider_with_input import SliderWithInput
from app.workers.base_worker import ProcessWorker


class TtsInferenceWindow(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("Преобразование текста в речь (VITS)")
        self.resize(840, 720)

        layout = QVBoxLayout(self)

        model_box = QGroupBox("Модель")
        ml = QHBoxLayout(model_box)
        self._dd_g = DragDropArea("G.pth", allowed_exts=(".pth",), single_file=True)
        self._dd_d = DragDropArea("D.pth", allowed_exts=(".pth",), single_file=True)
        self._dd_cfg = DragDropArea("config.json", allowed_exts=(".json",), single_file=True)
        ml.addWidget(self._dd_g)
        ml.addWidget(self._dd_d)
        ml.addWidget(self._dd_cfg)
        layout.addWidget(model_box)

        reset = QPushButton("Сбросить параметры")
        reset.clicked.connect(self._reset)
        layout.addWidget(reset)

        layout.addWidget(QLabel("Текст:"))
        self._text = QTextEdit()
        self._text.setPlaceholderText("Введите текст для синтеза...")
        layout.addWidget(self._text, stretch=1)

        params = QGroupBox("Параметры генерации")
        pl = QVBoxLayout(params)
        self._length = SliderWithInput("Темп речи", 0.1, 2.0, 0.05, 1.0, decimals=2)
        self._pitch = SliderWithInput("Высота голоса, полутонов", -12, 12, 1, 0, decimals=0)
        pl.addWidget(self._length)
        pl.addWidget(self._pitch)
        layout.addWidget(params)

        bottom = QHBoxLayout()
        self._btn_run = QPushButton("Выполнить генерацию")
        self._btn_run.clicked.connect(self._run)
        bottom.addWidget(self._btn_run)
        fmt_row, _, self._rb_wav, _ = make_format_radio(self)
        bottom.addLayout(fmt_row)
        layout.addLayout(bottom)

        self._player = AudioPlayer()
        layout.addWidget(self._player)

        self._metrics = QLabel("WER появится после генерации.")
        self._metrics.setObjectName("metrics")
        layout.addWidget(self._metrics)

    def _reset(self) -> None:
        self._length.set_value(1.0)
        self._pitch.set_value(0)

    def _run(self) -> None:
        g = self._dd_g.files()
        cfg = self._dd_cfg.files()
        text = self._text.toPlainText().strip()
        if not (g and cfg and text):
            QMessageBox.warning(self, "Вход", "Нужны G.pth, config.json и текст.")
            return
        fmt = "wav" if self._rb_wav.isChecked() else "mp3"
        model_name = Path(g[0]).stem

        VITS_DIR.mkdir(parents=True, exist_ok=True)
        payload = VITS_DIR / "_last_text.txt"
        payload.write_text(text, encoding="utf-8")

        args = [
            "--generator", g[0],
            "--config", cfg[0],
            "--text-file", str(payload),
            "--length-scale", str(self._length.value()),
            "--pitch-shift", str(int(self._pitch.value())),
            "--format", fmt,
            "--model-name", model_name,
            "--compute-metrics",
        ]

        dlg = ProgressDialog("Генерация речи...", show_log=True, parent=self)
        worker = ProcessWorker("tts_infer", SCRIPTS_DIR / "run_vits_infer.py", args)
        worker.line_received.connect(dlg.append_log)
        holder: dict = {}

        def _line(line: str) -> None:
            if line.startswith("RESULT_JSON="):
                try:
                    holder["data"] = json.loads(line[len("RESULT_JSON="):])
                except json.JSONDecodeError:
                    pass

        worker.line_received.connect(_line)

        def _done(code: int) -> None:
            if code != 0:
                dlg.finish_error(f"Код {code}")
                return
            data = holder.get("data", {})
            out = data.get("output")
            if out:
                self._player.load(Path(out))
            wer = data.get("wer")
            self._metrics.setText(f"WER: {wer:.3f}" if wer is not None else "WER недоступен.")
            dlg.finish_success()

        worker.finished.connect(_done)
        dlg.cancel_requested.connect(worker.kill)
        worker.start()
        dlg.exec()
