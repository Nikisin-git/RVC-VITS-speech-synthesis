"""Window: VITS / Coqui-TTS training."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QRadioButton, QVBoxLayout, QWidget,
)

from app.config import SCRIPTS_DIR
from app.core.tts.manifest import validate_manifest
from app.ui.widgets.drag_drop_area import DragDropArea
from app.ui.widgets.progress_dialog import ProgressDialog
from app.ui.widgets.slider_with_input import SliderWithInput
from app.utils.env_check import EnvReport, recommended_batch_size
from app.utils.path_utils import validate_model_name
from app.workers.base_worker import ProcessWorker, make_cancel_flag
from app.workers.log_parser import parse_tts_train_line
from app.workers.progress_tracker import ProgressState

_FIRST_TIME_HINT = (
    "Для обучения собственной TTS-модели потребуется датасет от 10 до 20 часов "
    "аудио высокого качества. Для дообучения существующей модели достаточно 10–30 минут."
)


class TtsTrainWindow(QWidget):
    _hint_shown = False

    def __init__(self, parent: QWidget | None = None, env: EnvReport | None = None) -> None:
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("Обучение TTS-модели")
        self.resize(820, 720)
        self._env = env

        layout = QVBoxLayout(self)

        if not TtsTrainWindow._hint_shown:
            TtsTrainWindow._hint_shown = True
            QMessageBox.information(self, "Подсказка", _FIRST_TIME_HINT)

        # mode
        mode_box = QGroupBox("Режим обучения")
        mode_l = QVBoxLayout(mode_box)
        radios = QHBoxLayout()
        self._rb_scratch = QRadioButton("С нуля (10–20 часов)")
        self._rb_finetune = QRadioButton("Дообучение (10–30 минут)")
        self._rb_finetune.setChecked(True)
        g = QButtonGroup(self)
        g.addButton(self._rb_scratch)
        g.addButton(self._rb_finetune)
        radios.addWidget(self._rb_scratch)
        radios.addWidget(self._rb_finetune)
        mode_l.addLayout(radios)

        # Pretrained checkpoint — required for fine-tuning. On a small dataset
        # fine-tuning a pretrained VITS is the only way to avoid the GAN
        # divergence that turns from-scratch runs into buzzing.
        self._pretrained_label = QLabel(
            "Базовый чекпойнт (.pth) для дообучения — обязателен в режиме «Дообучение»:")
        mode_l.addWidget(self._pretrained_label)
        self._dd_pretrained = DragDropArea(
            "Перетащите базовую модель .pth", allowed_exts=(".pth",), single_file=True)
        mode_l.addWidget(self._dd_pretrained)
        layout.addWidget(mode_box)

        self._rb_scratch.toggled.connect(self._on_mode_changed)
        self._on_mode_changed()

        # two columns
        cols = QHBoxLayout()
        audio_box = QGroupBox("Аудио")
        al = QVBoxLayout(audio_box)
        self._dd_audio = DragDropArea("Перетащите .wav файлы", allowed_exts=(".wav",))
        al.addWidget(self._dd_audio)
        cols.addWidget(audio_box, stretch=1)

        man_box = QGroupBox("Манифест")
        ml = QVBoxLayout(man_box)
        ml.addWidget(QLabel("LJSpeech-формат: имя_файла.wav|расшифровка"))
        self._dd_manifest = DragDropArea("Перетащите .csv", allowed_exts=(".csv",), single_file=True)
        ml.addWidget(self._dd_manifest)
        cols.addWidget(man_box, stretch=1)
        layout.addLayout(cols)

        # save settings
        save_box = QGroupBox("Настройки сохранения")
        sl = QVBoxLayout(save_box)
        sl.addWidget(QLabel("Имя модели:"))
        self._name = QLineEdit()
        self._name.textChanged.connect(self._validate_name)
        sl.addWidget(self._name)
        self._name_error = QLabel("")
        self._name_error.setObjectName("error_label")
        sl.addWidget(self._name_error)

        self._save_every = SliderWithInput("Частота сохранения, эпох", 5, 50, 1, 10, decimals=0)
        self._epochs = SliderWithInput("Количество эпох", 10, 10000, 10, 500, decimals=0)
        recommended = recommended_batch_size(env.gpu_vram_gb if env else None)
        self._batch = SliderWithInput("batch_size", 4, 32, 1, recommended, decimals=0)
        sl.addWidget(self._save_every)
        sl.addWidget(self._epochs)
        sl.addWidget(self._batch)
        layout.addWidget(save_box)

        self._btn_start = QPushButton("Начать обучение")
        self._btn_start.clicked.connect(self._start)
        layout.addWidget(self._btn_start)

        if env is not None and not env.cuda_available:
            self._btn_start.setEnabled(False)
            warn = QLabel("CUDA не обнаружена — обучение недоступно.")
            warn.setObjectName("error_label")
            layout.addWidget(warn)

    def _on_mode_changed(self) -> None:
        # Pretrained checkpoint only matters for fine-tuning.
        finetune = self._rb_finetune.isChecked()
        self._pretrained_label.setEnabled(finetune)
        self._dd_pretrained.setEnabled(finetune)

    def _validate_name(self, text: str) -> None:
        ok, err = validate_model_name(text)
        self._name_error.setText("" if ok else (err or ""))

    def _start(self) -> None:
        audio_files = self._dd_audio.files()
        manifest = self._dd_manifest.files()
        ok, err = validate_model_name(self._name.text())
        if not (audio_files and manifest):
            QMessageBox.warning(self, "Вход", "Добавьте аудио и манифест.")
            return
        if not ok:
            QMessageBox.warning(self, "Имя модели", err or "")
            return
        if self._env is not None and not self._env.cuda_available:
            QMessageBox.warning(self, "GPU", "CUDA не обнаружена.")
            return

        finetune = self._rb_finetune.isChecked()
        pretrained = self._dd_pretrained.files()
        if finetune and not pretrained:
            reply = QMessageBox.warning(
                self, "Дообучение без базовой модели",
                "Выбран режим «Дообучение», но не указан базовый чекпойнт .pth.\n\n"
                "Без него дообучение превращается в обучение с нуля, которое на "
                "небольшом датасете обычно расходится и даёт на выходе жужжание.\n\n"
                "Продолжить обучение с нуля?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        audio_dir = Path(audio_files[0]).parent
        report = validate_manifest(Path(manifest[0]), audio_dir)
        if not report.ok:
            msg = "Ошибки в манифесте:\n\n" + "\n".join(report.errors[:20])
            if len(report.errors) > 20:
                msg += f"\n... и ещё {len(report.errors) - 20}"
            QMessageBox.critical(self, "Манифест", msg)
            return

        # Fine-tune only if a base checkpoint was actually provided; otherwise
        # fall back to scratch so the trainer picks the from-scratch LR.
        effective_finetune = finetune and bool(pretrained)
        cancel_flag = make_cancel_flag(f"tts_train_{self._name.text()}")
        args = [
            "--audio-dir", str(audio_dir),
            "--manifest", manifest[0],
            "--model-name", self._name.text(),
            "--mode", "finetune" if effective_finetune else "scratch",
            "--epochs", str(int(self._epochs.value())),
            "--save-every", str(int(self._save_every.value())),
            "--batch-size", str(int(self._batch.value())),
            "--cancel-flag", str(cancel_flag),
        ]
        if effective_finetune:
            args += ["--pretrained", pretrained[0]]

        # The trainer writes events.out.tfevents.* into model_dir(name); the
        # live chart polls anything under that root.
        from app.core.tts.trainer import model_dir as _tts_model_dir
        chart_dir = _tts_model_dir(self._name.text())
        dlg = ProgressDialog(
            "Обучение TTS-модели...",
            show_log=True, determinate=True, parent=self,
            chart_event_dir=chart_dir, chart_framework="tts",
        )
        state = ProgressState()
        worker = ProcessWorker("tts_train", SCRIPTS_DIR / "run_vits_train.py", args)
        worker.set_cancel_flag(cancel_flag)

        def _line(line: str) -> None:
            dlg.append_log(line)
            ev = parse_tts_train_line(line)
            if ev is None:
                return
            state.apply(ev)
            if ev.kind == "epoch":
                dlg.set_status(f"Эпоха {state.current_epoch}/{state.total_epochs or '?'}")
                dlg.set_percent(state.percent)

        worker.line_received.connect(_line)

        def _done(code: int) -> None:
            if code == 0:
                dlg.finish_success()
            else:
                dlg.finish_error(f"Код {code}")

        worker.finished.connect(_done)

        cancel_count = {"n": 0}

        def _on_cancel() -> None:
            cancel_count["n"] += 1
            if cancel_count["n"] == 1:
                worker.request_soft_cancel()
                dlg.append_log(">>> Мягкая отмена запрошена.")
            else:
                worker.kill()
                dlg.append_log(">>> Жёсткое прерывание.")

        dlg.cancel_requested.connect(_on_cancel)
        worker.start()
        dlg.exec()
