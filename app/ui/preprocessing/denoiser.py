"""Window: DeepFilterNet 3 speech denoiser."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton, QVBoxLayout, QWidget

from app.config import SCRIPTS_DIR
from app.ui.preprocessing._common import make_format_radio
from app.ui.widgets.drag_drop_area import DragDropArea
from app.ui.widgets.progress_dialog import ProgressDialog
from app.workers.base_worker import ProcessWorker


class DenoiserWindow(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("Шумоподавление (DeepFilterNet 3)")
        self.resize(640, 420)

        layout = QVBoxLayout(self)
        self._dd = DragDropArea("Перетащите .mp3/.wav сюда")
        layout.addWidget(self._dd)

        fmt_row, _, self._rb_wav, _ = make_format_radio(self)
        layout.addLayout(fmt_row)

        self._btn_run = QPushButton("Выполнить")
        self._btn_run.clicked.connect(self._run)
        layout.addWidget(self._btn_run)

        self._workers: list[ProcessWorker] = []

    def _run(self) -> None:
        files = self._dd.files()
        if not files:
            return
        fmt = "wav" if self._rb_wav.isChecked() else "mp3"
        dlg = ProgressDialog("Идёт шумоподавление...", show_log=True, parent=self)
        remaining = list(files)
        produced: list[str] = []

        def _next() -> None:
            if not remaining:
                dlg.set_status(f"Готово. Файлов: {len(produced)}")
                dlg.finish_success()
                return
            path = remaining.pop(0)
            dlg.set_status(f"Обработка: {Path(path).name}")
            worker = ProcessWorker("denoiser", SCRIPTS_DIR / "run_denoiser.py",
                                   ["--input", path, "--format", fmt])
            worker.line_received.connect(dlg.append_log)

            def _done(code: int) -> None:
                if code == 0:
                    produced.append(path)
                    _next()
                else:
                    dlg.finish_error(f"Денойзер завершился с кодом {code}")

            worker.finished.connect(_done)
            dlg.cancel_requested.connect(worker.kill)
            self._workers.append(worker)
            worker.start()

        _next()
        dlg.exec()
