"""Window: timer-based or VAD-based audio slicing with tail policy."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup, QGroupBox, QHBoxLayout, QPushButton, QRadioButton, QVBoxLayout, QWidget,
)

from app.config import SCRIPTS_DIR
from app.ui.preprocessing._common import make_format_radio
from app.ui.widgets.drag_drop_area import DragDropArea
from app.ui.widgets.progress_dialog import ProgressDialog
from app.ui.widgets.slider_with_input import SliderWithInput
from app.workers.base_worker import ProcessWorker


class SlicerWindow(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("Нарезка на фрагменты")
        self.resize(680, 580)

        layout = QVBoxLayout(self)
        self._dd = DragDropArea("Перетащите .mp3/.wav сюда")
        layout.addWidget(self._dd)

        fmt_row, _, self._rb_wav, _ = make_format_radio(self)
        layout.addLayout(fmt_row)

        # save shape
        shape_box = QGroupBox("Сохранить аудиозаписи в виде")
        shape_l = QHBoxLayout(shape_box)
        self._rb_single = QRadioButton("единая дорожка")
        self._rb_multi = QRadioButton("несколько фрагментов")
        self._rb_multi.setChecked(True)
        shape_g = QButtonGroup(self)
        shape_g.addButton(self._rb_single)
        shape_g.addButton(self._rb_multi)
        shape_l.addWidget(self._rb_single)
        shape_l.addWidget(self._rb_multi)
        layout.addWidget(shape_box)

        self._length = SliderWithInput("Длительность фрагмента, сек", 5, 20, step=1, default=10, decimals=0)
        layout.addWidget(self._length)

        # slicing mode
        mode_box = QGroupBox("Способ нарезки")
        mode_l = QHBoxLayout(mode_box)
        self._rb_timer = QRadioButton("По таймеру")
        self._rb_vad = QRadioButton("По тишине (VAD)")
        self._rb_timer.setChecked(True)
        mode_g = QButtonGroup(self)
        mode_g.addButton(self._rb_timer)
        mode_g.addButton(self._rb_vad)
        mode_l.addWidget(self._rb_timer)
        mode_l.addWidget(self._rb_vad)
        layout.addWidget(mode_box)

        # tail policy
        tail_box = QGroupBox("Что делать с фрагментами короче 4 секунд")
        tail_l = QHBoxLayout(tail_box)
        self._rb_keep = QRadioButton("Сохранить как есть")
        self._rb_merge = QRadioButton("Дописать к предыдущему")
        self._rb_drop = QRadioButton("Отбросить")
        self._rb_keep.setChecked(True)
        tail_g = QButtonGroup(self)
        tail_g.addButton(self._rb_keep)
        tail_g.addButton(self._rb_merge)
        tail_g.addButton(self._rb_drop)
        tail_l.addWidget(self._rb_keep)
        tail_l.addWidget(self._rb_merge)
        tail_l.addWidget(self._rb_drop)
        layout.addWidget(tail_box)

        self._btn_run = QPushButton("Выполнить")
        self._btn_run.clicked.connect(self._run)
        layout.addWidget(self._btn_run)

        self._workers: list[ProcessWorker] = []

    def _tail_mode(self) -> str:
        if self._rb_merge.isChecked():
            return "merge"
        if self._rb_drop.isChecked():
            return "drop"
        return "keep"

    def _run(self) -> None:
        files = self._dd.files()
        if not files:
            return
        fmt = "wav" if self._rb_wav.isChecked() else "mp3"
        single = self._rb_single.isChecked()
        mode = "vad" if self._rb_vad.isChecked() else "timer"
        length = int(self._length.value())
        tail = self._tail_mode()

        dlg = ProgressDialog("Идёт нарезка..." if not single else "Идёт склейка...",
                             show_log=True, parent=self)

        # In single-track mode all uploaded files are concatenated into one
        # output, so we make a single CLI call with multiple --input flags.
        # In multi-fragment mode each file is processed independently.
        if single:
            batches: list[list[str]] = [list(files)]
        else:
            batches = [[f] for f in files]

        remaining = list(batches)
        produced: list[str] = []

        def _next() -> None:
            if not remaining:
                dlg.set_status(f"Готово. Файлов обработано: {len(produced)}")
                dlg.finish_success()
                return
            batch = remaining.pop(0)
            label = Path(batch[0]).name if len(batch) == 1 else f"{len(batch)} файлов"
            dlg.set_status(f"Обработка: {label}")
            args: list[str] = []
            for path in batch:
                args += ["--input", path]
            args += [
                "--format", fmt,
                "--mode", mode, "--length", str(length),
                "--tail-mode", tail,
            ]
            if single:
                args.append("--single-track")
            worker = ProcessWorker("slicer", SCRIPTS_DIR / "run_slicer.py", args)
            worker.line_received.connect(dlg.append_log)

            def _done(code: int) -> None:
                if code == 0:
                    produced.extend(batch)
                    _next()
                else:
                    dlg.finish_error(f"Ошибка, код {code}")

            worker.finished.connect(_done)
            dlg.cancel_requested.connect(worker.kill)
            self._workers.append(worker)
            worker.start()

        _next()
        dlg.exec()
