"""Window: TTS training — Coqui VITS or HuggingFace VITS (mms-tts-rus)."""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QRadioButton, QVBoxLayout, QWidget,
)

from app.config import PROJECT_ROOT, SCRIPTS_DIR, VITS_DIR
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
    "Для русского голоса рекомендуется движок «HuggingFace VITS»: он дообучает "
    "готовую модель mms-tts-rus на вашем дикторе (10–30 минут аудио достаточно). "
    "Coqui VITS оставлен для англоязычных/своих моделей."
)

_HF_SETUP_HINT = (
    "Для движка «HuggingFace VITS» нужна разовая подготовка (см. "
    "docs/HF_VITS_FINETUNE.md):\n\n"
    "1. Склонировать репозиторий finetune-hf-vits.\n"
    "2. Собрать monotonic_align.\n"
    "3. Подготовить базовую модель с дискриминатором "
    "(mms-tts-rus-with-disc).\n\n"
    "Укажите пути к папке репозитория и к папке базовой модели ниже."
)


class _FolderPicker(QWidget):
    """QLineEdit + 'Обзор…' button that selects a directory."""

    def __init__(self, placeholder: str = "", default: str = "",
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._edit = QLineEdit()
        self._edit.setPlaceholderText(placeholder)
        if default:
            self._edit.setText(default)
        btn = QPushButton("Обзор…")
        btn.setFixedWidth(90)
        btn.clicked.connect(self._browse)
        lay.addWidget(self._edit, stretch=1)
        lay.addWidget(btn)

    def _browse(self) -> None:
        start = self._edit.text().strip() or ""
        d = QFileDialog.getExistingDirectory(self, "Выбор папки", start)
        if d:
            self._edit.setText(d)

    def path(self) -> str:
        return self._edit.text().strip()

    def setEnabled(self, enabled: bool) -> None:  # noqa: N802
        super().setEnabled(enabled)


def _default_repo() -> str:
    """Best guess for the finetune-hf-vits repo location."""
    env = os.environ.get("VOICEGEN_FINETUNE_HF_VITS")
    if env and Path(env).is_dir():
        return env
    guess = PROJECT_ROOT / "finetune-hf-vits"
    return str(guess) if guess.is_dir() else ""


def _default_base_model(repo: str) -> str:
    if repo:
        cand = Path(repo) / "mms-tts-rus-with-disc"
        if cand.is_dir():
            return str(cand)
    return ""


class TtsTrainWindow(QWidget):
    _hint_shown = False

    def __init__(self, parent: QWidget | None = None, env: EnvReport | None = None) -> None:
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("Обучение TTS-модели")
        self.resize(860, 780)
        self._env = env

        layout = QVBoxLayout(self)

        if not TtsTrainWindow._hint_shown:
            TtsTrainWindow._hint_shown = True
            QMessageBox.information(self, "Подсказка", _FIRST_TIME_HINT)

        # --- engine selector ---
        engine_box = QGroupBox("Движок обучения")
        engine_l = QHBoxLayout(engine_box)
        self._rb_hf = QRadioButton("HuggingFace VITS (русский, mms-tts-rus)")
        self._rb_coqui = QRadioButton("Coqui VITS")
        self._rb_hf.setChecked(True)
        eg = QButtonGroup(self)
        eg.addButton(self._rb_hf)
        eg.addButton(self._rb_coqui)
        engine_l.addWidget(self._rb_hf)
        engine_l.addWidget(self._rb_coqui)
        layout.addWidget(engine_box)

        # --- HuggingFace VITS box ---
        repo0 = _default_repo()
        self._hf_box = QGroupBox("HuggingFace VITS — пути (разовая настройка)")
        hf_l = QVBoxLayout(self._hf_box)
        hf_l.addWidget(QLabel("Репозиторий finetune-hf-vits:"))
        self._hf_repo = _FolderPicker("…\\finetune-hf-vits", repo0)
        hf_l.addWidget(self._hf_repo)
        hf_l.addWidget(QLabel("Базовая модель с дискриминатором (mms-tts-rus-with-disc):"))
        self._hf_base = _FolderPicker("…\\mms-tts-rus-with-disc", _default_base_model(repo0))
        hf_l.addWidget(self._hf_base)
        _hint = QLabel(_HF_SETUP_HINT)
        _hint.setObjectName("hint")
        _hint.setWordWrap(True)
        hf_l.addWidget(_hint)
        layout.addWidget(self._hf_box)

        # --- Coqui mode box ---
        self._coqui_box = QGroupBox("Coqui VITS — режим обучения")
        mode_l = QVBoxLayout(self._coqui_box)
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
        self._pretrained_label = QLabel(
            "Базовый чекпойнт (.pth) для дообучения — обязателен в режиме «Дообучение»:")
        mode_l.addWidget(self._pretrained_label)
        self._dd_pretrained = DragDropArea(
            "Перетащите базовую модель .pth", allowed_exts=(".pth",), single_file=True)
        mode_l.addWidget(self._dd_pretrained)
        layout.addWidget(self._coqui_box)

        self._rb_scratch.toggled.connect(self._on_mode_changed)
        self._rb_hf.toggled.connect(self._on_engine_changed)

        # --- two columns: audio + manifest ---
        cols = QHBoxLayout()
        audio_box = QGroupBox("Аудио")
        al = QVBoxLayout(audio_box)
        # Folder picker is the primary way (works for thousands of clips);
        # the drag-drop below stays as an option for small sets.
        al.addWidget(QLabel("Папка с .wav (рекомендуется для больших датасетов):"))
        self._audio_dir = _FolderPicker("…\\wavs")
        al.addWidget(self._audio_dir)
        al.addWidget(QLabel("или перетащите отдельные файлы:"))
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

        # --- save settings ---
        save_box = QGroupBox("Настройки обучения")
        sl = QVBoxLayout(save_box)
        sl.addWidget(QLabel("Имя модели:"))
        self._name = QLineEdit()
        self._name.textChanged.connect(self._validate_name)
        sl.addWidget(self._name)
        self._name_error = QLabel("")
        self._name_error.setObjectName("error_label")
        sl.addWidget(self._name_error)

        self._save_every = SliderWithInput("Частота сохранения чекпойнта, эпох", 5, 50, 1, 10, decimals=0)
        self._epochs = SliderWithInput("Количество эпох", 10, 10000, 10, 100, decimals=0)
        recommended = recommended_batch_size(env.gpu_vram_gb if env else None)
        self._batch = SliderWithInput("batch_size", 4, 32, 1, min(8, recommended), decimals=0)
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

        self._on_engine_changed()
        self._on_mode_changed()

    # --- visibility toggles ---
    def _on_engine_changed(self) -> None:
        hf = self._rb_hf.isChecked()
        self._hf_box.setVisible(hf)
        self._coqui_box.setVisible(not hf)
        # save-every applies to both engines now (checkpoint frequency in epochs)

    def _on_mode_changed(self) -> None:
        finetune = self._rb_finetune.isChecked()
        self._pretrained_label.setEnabled(finetune)
        self._dd_pretrained.setEnabled(finetune)

    def _validate_name(self, text: str) -> None:
        ok, err = validate_model_name(text)
        self._name_error.setText("" if ok else (err or ""))

    # --- launch ---
    def _start(self) -> None:
        audio_files = self._dd_audio.files()
        manifest = self._dd_manifest.files()
        ok, err = validate_model_name(self._name.text())

        # Audio directory: prefer the explicitly picked folder (works for
        # thousands of clips); otherwise fall back to the dropped files' parent.
        picked_dir = self._audio_dir.path()
        if picked_dir:
            audio_dir = Path(picked_dir)
            if not audio_dir.is_dir():
                QMessageBox.warning(self, "Аудио", "Указанная папка с аудио не найдена.")
                return
        elif audio_files:
            audio_dir = Path(audio_files[0]).parent
        else:
            QMessageBox.warning(self, "Вход", "Выберите папку с аудио или перетащите .wav файлы.")
            return

        if not manifest:
            QMessageBox.warning(self, "Вход", "Добавьте манифест (.csv).")
            return
        if not ok:
            QMessageBox.warning(self, "Имя модели", err or "")
            return
        if self._env is not None and not self._env.cuda_available:
            QMessageBox.warning(self, "GPU", "CUDA не обнаружена.")
            return
        report = validate_manifest(Path(manifest[0]), audio_dir)
        if not report.ok:
            msg = "Ошибки в манифесте:\n\n" + "\n".join(report.errors[:20])
            if len(report.errors) > 20:
                msg += f"\n... и ещё {len(report.errors) - 20}"
            QMessageBox.critical(self, "Манифест", msg)
            return

        if self._rb_hf.isChecked():
            self._start_hf(audio_dir, manifest[0])
        else:
            self._start_coqui(audio_dir, manifest[0])

    # --- HuggingFace VITS ---
    def _start_hf(self, audio_dir: Path, manifest: str) -> None:
        repo = self._hf_repo.path()
        base = self._hf_base.path()
        if not repo or not (Path(repo) / "run_vits_finetuning.py").exists():
            QMessageBox.critical(
                self, "finetune-hf-vits не найден",
                "Не найден репозиторий finetune-hf-vits (нет run_vits_finetuning.py).\n\n"
                + _HF_SETUP_HINT)
            return
        if not base or not (Path(base) / "config.json").exists():
            QMessageBox.critical(
                self, "Базовая модель не найдена",
                "В указанной папке нет config.json базовой модели.\n\n" + _HF_SETUP_HINT)
            return

        name = self._name.text()
        out_dir = VITS_DIR / "hf_vits_finetune" / name
        args = [
            "--manifest", manifest,
            "--audio-dir", str(audio_dir),
            "--model-name", name,
            "--base-model", base,
            "--output-dir", str(out_dir),
            "--epochs", str(int(self._epochs.value())),
            "--batch-size", str(int(self._batch.value())),
            "--save-every", str(int(self._save_every.value())),
            "--repo", repo,
        ]
        # HF Trainer logs its own tensorboard scalars (not our Coqui/RVC names),
        # so the live curve chart is skipped here — the log shows step progress.
        dlg = ProgressDialog("Обучение HuggingFace VITS...", show_log=True, parent=self)
        worker = ProcessWorker("hf_vits_train", SCRIPTS_DIR / "run_hf_vits_train.py", args)
        worker.line_received.connect(dlg.append_log)

        def _done(code: int) -> None:
            if code == 0:
                dlg.set_status(f"Готово. Модель: {out_dir / 'run'}")
                dlg.finish_success()
            else:
                dlg.finish_error(f"Код {code}")

        worker.finished.connect(_done)
        dlg.cancel_requested.connect(worker.kill)
        worker.start()
        dlg.exec()

    # --- Coqui VITS ---
    def _start_coqui(self, audio_dir: Path, manifest: str) -> None:
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

        effective_finetune = finetune and bool(pretrained)
        cancel_flag = make_cancel_flag(f"tts_train_{self._name.text()}")
        args = [
            "--audio-dir", str(audio_dir),
            "--manifest", manifest,
            "--model-name", self._name.text(),
            "--mode", "finetune" if effective_finetune else "scratch",
            "--epochs", str(int(self._epochs.value())),
            "--save-every", str(int(self._save_every.value())),
            "--batch-size", str(int(self._batch.value())),
            "--cancel-flag", str(cancel_flag),
        ]
        if effective_finetune:
            args += ["--pretrained", pretrained[0]]

        from app.core.tts.trainer import model_dir as _tts_model_dir
        chart_dir = _tts_model_dir(self._name.text())
        dlg = ProgressDialog(
            "Обучение Coqui VITS...",
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
