"""Window: filter speech fragments — drop filler-only / empty / repetitive clips."""

from __future__ import annotations

import tempfile
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QVBoxLayout, QWidget,
)

from app.config import FILTERED_DIR, SCRIPTS_DIR
from app.ui.widgets.progress_dialog import ProgressDialog
from app.ui.widgets.slider_with_input import SliderWithInput
from app.workers.base_worker import ProcessWorker

_AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg")


class SegmentFilterWindow(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("Удаление слов-паразитов (фильтр фрагментов)")
        self.resize(720, 520)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Каждый фрагмент распознаётся Whisper; «мусорные» отбрасываются, "
            "хорошие копируются в папку Filtered. Оригиналы не изменяются."))

        # input folder
        folder_box = QGroupBox("Папка с фрагментами")
        fl = QHBoxLayout(folder_box)
        self._dir = QLineEdit()
        self._dir.setPlaceholderText("…\\Cuts\\Sliced")
        btn = QPushButton("Обзор…")
        btn.setFixedWidth(90)
        btn.clicked.connect(self._browse)
        fl.addWidget(self._dir, stretch=1)
        fl.addWidget(btn)
        layout.addWidget(folder_box)

        # which rules
        rules_box = QGroupBox("Что удалять")
        rl = QVBoxLayout(rules_box)
        self._cb_empty = QCheckBox("Пустые / без речи")
        self._cb_filler = QCheckBox("Только слова-паразиты (ну, эээ, типа…)")
        self._cb_repeat = QCheckBox("Повторы / заикания")
        self._cb_lowconf = QCheckBox("Неразборчивые (низкая уверенность распознавания)")
        for cb in (self._cb_empty, self._cb_filler, self._cb_repeat, self._cb_lowconf):
            cb.setChecked(True)
            rl.addWidget(cb)
        layout.addWidget(rules_box)

        # strictness
        self._strict = SliderWithInput("Строгость (больше — удаляет смелее)", 1, 5, 1, 3, decimals=0)
        layout.addWidget(self._strict)

        self._btn_run = QPushButton("Выполнить")
        self._btn_run.clicked.connect(self._run)
        layout.addWidget(self._btn_run)

        self._workers: list[ProcessWorker] = []

    def _browse(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Папка с фрагментами", self._dir.text().strip())
        if d:
            self._dir.setText(d)

    def _thresholds(self) -> tuple[float, float, float]:
        s = int(self._strict.value())
        no_speech = round(0.90 - 0.09 * s, 3)   # ниже -> строже
        logprob = round(-1.45 + 0.15 * s, 3)    # выше -> строже
        comp = round(2.90 - 0.14 * s, 3)        # ниже -> строже
        return no_speech, logprob, comp

    def _run(self) -> None:
        folder = self._dir.text().strip()
        if not folder or not Path(folder).is_dir():
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Папка", "Укажите существующую папку с фрагментами.")
            return
        files = sorted(
            str(p) for p in Path(folder).iterdir()
            if p.is_file() and p.suffix.lower() in _AUDIO_EXTS
        )
        if not files:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Папка", "В папке нет аудиофайлов (.wav/.mp3).")
            return

        # Pass the list via a manifest file (avoids the command-line length limit).
        lst = tempfile.NamedTemporaryFile(
            mode="w", suffix="_filter_inputs.txt", delete=False, encoding="utf-8")
        lst.write("\n".join(files))
        lst.close()

        no_speech, logprob, comp = self._thresholds()
        args = [
            "--input-list", lst.name,
            "--out-dir", str(FILTERED_DIR),
            "--no-speech-thr", str(no_speech),
            "--logprob-thr", str(logprob),
            "--comp-thr", str(comp),
        ]
        if not self._cb_empty.isChecked():
            args.append("--keep-empty")
        if not self._cb_filler.isChecked():
            args.append("--keep-filler-only")
        if not self._cb_repeat.isChecked():
            args.append("--keep-repetition")
        if not self._cb_lowconf.isChecked():
            args.append("--keep-low-conf")

        dlg = ProgressDialog(f"Фильтрация {len(files)} фрагментов...", show_log=True, parent=self)
        worker = ProcessWorker("segment_filter", SCRIPTS_DIR / "run_segment_filter.py", args)
        worker.line_received.connect(dlg.append_log)

        def _done(code: int) -> None:
            if code == 0:
                dlg.set_status(f"Готово. Отфильтрованные — в папке Filtered.")
                dlg.finish_success()
            else:
                dlg.finish_error(f"Ошибка, код {code}")

        worker.finished.connect(_done)
        worker.failed.connect(lambda msg: dlg.finish_error(f"Не удалось запустить: {msg}"))
        dlg.cancel_requested.connect(worker.kill)
        self._workers.append(worker)
        worker.start()
        dlg.exec()
