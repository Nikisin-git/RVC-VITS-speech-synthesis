"""Window: remove silent regions > N sec."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton, QVBoxLayout, QWidget

from app.config import SCRIPTS_DIR
from app.ui.preprocessing._common import make_format_radio
from app.ui.widgets.drag_drop_area import DragDropArea
from app.ui.widgets.progress_dialog import ProgressDialog
from app.ui.widgets.slider_with_input import SliderWithInput
from app.workers.base_worker import ProcessWorker


class SilenceRemoverWindow(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("Удаление тихих мест")
        self.resize(640, 480)

        layout = QVBoxLayout(self)
        self._dd = DragDropArea("Перетащите .mp3/.wav сюда")
        layout.addWidget(self._dd)

        fmt_row, _, self._rb_wav, _ = make_format_radio(self)
        layout.addLayout(fmt_row)

        self._dur = SliderWithInput("Допустимая длительность паузы, сек",
                                     0.5, 5.0, step=0.5, default=1.0, decimals=1)
        layout.addWidget(self._dur)

        self._thr = SliderWithInput("Порог тишины, дБ",
                                     -100, 0, step=1, default=-40, decimals=0)
        layout.addWidget(self._thr)

        self._btn_run = QPushButton("Выполнить")
        self._btn_run.clicked.connect(self._run)
        layout.addWidget(self._btn_run)

        self._workers: list[ProcessWorker] = []

    def _run(self) -> None:
        files = self._dd.files()
        if not files:
            return
        fmt = "wav" if self._rb_wav.isChecked() else "mp3"
        max_sec = self._dur.value()
        thr_db = self._thr.value()
        dlg = ProgressDialog("Удаление тишины...", show_log=True, parent=self)
        remaining = list(files)
        produced: list[str] = []

        def _next() -> None:
            if not remaining:
                dlg.set_status(f"Готово. Файлов: {len(produced)}")
                dlg.finish_success()
                return
            path = remaining.pop(0)
            dlg.set_status(f"Обработка: {Path(path).name}")
            worker = ProcessWorker(
                "silence_remover", SCRIPTS_DIR / "run_silence_remover.py",
                ["--input", path, "--format", fmt,
                 "--max-silence", str(max_sec),
                 "--threshold-db", str(thr_db)],
            )
            worker.line_received.connect(dlg.append_log)

            def _done(code: int) -> None:
                if code == 0:
                    produced.append(path)
                    _next()
                else:
                    dlg.finish_error(f"Ошибка, код {code}")

            worker.finished.connect(_done)
            dlg.cancel_requested.connect(worker.kill)
            self._workers.append(worker)
            worker.start()

        _next()
        dlg.exec()
