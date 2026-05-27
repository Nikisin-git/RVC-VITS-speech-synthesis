"""Window: RVC training."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QGroupBox, QLabel, QLineEdit, QMessageBox,
    QPushButton, QVBoxLayout, QWidget,
)

from app.config import SAMPLE_RATES, SCRIPTS_DIR, F0_METHODS
from app.ui.widgets.drag_drop_area import DragDropArea
from app.ui.widgets.progress_dialog import ProgressDialog
from app.ui.widgets.slider_with_input import SliderWithInput
from app.utils.env_check import EnvReport, recommended_batch_size
from app.utils.path_utils import validate_model_name
from app.workers.base_worker import ProcessWorker, make_cancel_flag
from app.workers.log_parser import parse_rvc_train_line
from app.workers.progress_tracker import ProgressState


class RvcTrainWindow(QWidget):
    def __init__(self, parent: QWidget | None = None, env: EnvReport | None = None) -> None:
        super().__init__(parent, flags=Qt.Window)
        self.setWindowTitle("Обучение голосовой модели (RVC)")
        self.resize(720, 720)
        self._env = env
        self._worker: ProcessWorker | None = None

        layout = QVBoxLayout(self)

        self._dd = DragDropArea("Перетащите файлы датасета (.wav/.mp3)")
        layout.addWidget(self._dd)

        # model name
        layout.addWidget(QLabel("Имя модели:"))
        self._name = QLineEdit()
        self._name.setPlaceholderText("Только буквы/цифры/_/-, без пробелов и спецсимволов")
        self._name.textChanged.connect(self._validate_name)
        layout.addWidget(self._name)
        self._name_error = QLabel("")
        self._name_error.setObjectName("error_label")
        layout.addWidget(self._name_error)

        # sample rate
        sr_box = QGroupBox("Параметры обучения")
        sr_l = QVBoxLayout(sr_box)
        sr_l.addWidget(QLabel("Частота дискретизации:"))
        self._sr = QComboBox()
        self._sr.addItems(SAMPLE_RATES)
        self._sr.setCurrentText("40k")
        sr_l.addWidget(self._sr)

        sr_l.addWidget(QLabel("Метод извлечения F0 (питча):"))
        self._f0 = QComboBox()
        for k, v in F0_METHODS.items():
            self._f0.addItem(k, v)
        self._f0.setCurrentText("rmvpe_gpu")
        self._f0.currentIndexChanged.connect(self._update_f0_hint)
        sr_l.addWidget(self._f0)
        self._f0_hint = QLabel(F0_METHODS["rmvpe_gpu"])
        self._f0_hint.setWordWrap(True)
        self._f0_hint.setObjectName("hint")
        sr_l.addWidget(self._f0_hint)

        self._save_every = SliderWithInput("Частота сохранения чекпоинта, эпох",
                                            5, 50, step=1, default=10, decimals=0)
        self._epochs = SliderWithInput("Количество эпох", 10, 10000, step=10, default=200, decimals=0)

        recommended = recommended_batch_size(env.gpu_vram_gb if env else None)
        self._batch = SliderWithInput("batch_size", 4, 32, step=1, default=recommended, decimals=0)
        if env and env.gpu_vram_gb:
            sr_l.addWidget(QLabel(f"Рекомендация для {env.gpu_vram_gb:.1f} ГБ VRAM: {recommended}"))

        sr_l.addWidget(self._save_every)
        sr_l.addWidget(self._epochs)
        sr_l.addWidget(self._batch)

        self._zip = QCheckBox("Создать .zip архив после обучения")
        self._zip.setChecked(True)
        sr_l.addWidget(self._zip)

        layout.addWidget(sr_box)

        self._btn_start = QPushButton("Начать обучение")
        self._btn_start.clicked.connect(self._start)
        layout.addWidget(self._btn_start)

        if env is not None and not env.cuda_available:
            self._btn_start.setEnabled(False)
            warn = QLabel("CUDA не обнаружена — обучение недоступно.")
            warn.setObjectName("error_label")
            layout.addWidget(warn)

    def _update_f0_hint(self) -> None:
        key = self._f0.currentText()
        self._f0_hint.setText(F0_METHODS.get(key, ""))

    def _validate_name(self, text: str) -> None:
        ok, err = validate_model_name(text)
        self._name_error.setText("" if ok else (err or ""))

    def _start(self) -> None:
        if self._env is not None and not self._env.cuda_available:
            QMessageBox.warning(self, "GPU", "CUDA не обнаружена. Обучение невозможно.")
            return
        files = self._dd.files()
        if not files:
            QMessageBox.warning(self, "Датасет", "Добавьте хотя бы один файл.")
            return
        ok, err = validate_model_name(self._name.text())
        if not ok:
            QMessageBox.warning(self, "Имя модели", err or "Некорректное имя.")
            return

        # write dataset list to a temp manifest the CLI reads
        from app.config import TRAINING_DIR
        TRAINING_DIR.mkdir(parents=True, exist_ok=True)
        manifest = TRAINING_DIR / f"_{self._name.text()}_dataset.txt"
        manifest.write_text("\n".join(files), encoding="utf-8")

        cancel_flag = make_cancel_flag(f"rvc_train_{self._name.text()}")

        args = [
            "--dataset-list", str(manifest),
            "--model-name", self._name.text(),
            "--sample-rate", self._sr.currentText(),
            "--f0-method", self._f0.currentText(),
            "--epochs", str(int(self._epochs.value())),
            "--save-every", str(int(self._save_every.value())),
            "--batch-size", str(int(self._batch.value())),
            "--cancel-flag", str(cancel_flag),
        ]
        if self._zip.isChecked():
            args.append("--create-zip")

        dlg = ProgressDialog("Обучение голосовой модели...", show_log=True, determinate=True, parent=self)
        state = ProgressState()

        worker = ProcessWorker("rvc_train", SCRIPTS_DIR / "run_rvc_train.py", args)
        worker.set_cancel_flag(cancel_flag)
        self._worker = worker

        def _line(line: str) -> None:
            dlg.append_log(line)
            ev = parse_rvc_train_line(line)
            if ev is None:
                return
            state.apply(ev)
            if ev.kind == "epoch":
                dlg.set_status(f"Эпоха {state.current_epoch}/{state.total_epochs or '?'} "
                               f"(loss={state.last_loss}, чекпоинтов: {state.checkpoints_saved})")
                dlg.set_percent(state.percent)
            elif ev.kind == "stage":
                dlg.set_status(f"Этап: {state.stage}")

        worker.line_received.connect(_line)

        def _done(code: int) -> None:
            try:
                manifest.unlink(missing_ok=True)
            except OSError:
                pass
            if code == 0:
                dlg.set_status("Обучение завершено.")
                dlg.finish_success()
            else:
                dlg.finish_error(f"Процесс завершился с кодом {code}")

        worker.finished.connect(_done)

        cancel_count = {"n": 0}

        def _on_cancel() -> None:
            cancel_count["n"] += 1
            if cancel_count["n"] == 1:
                worker.request_soft_cancel()
                dlg.append_log(">>> Отмена запрошена. Жду завершения текущей эпохи. "
                               "Нажмите 'Отмена' ещё раз для жёсткой остановки.")
            else:
                dlg.append_log(">>> Жёсткое прерывание.")
                worker.kill()

        dlg.cancel_requested.connect(_on_cancel)
        worker.start()
        dlg.exec()
