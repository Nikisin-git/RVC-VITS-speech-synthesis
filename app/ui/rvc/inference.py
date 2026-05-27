"""Window: RVC inference with full postprocess + metrics."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox, QHBoxLayout, QLabel, QMessageBox, QPushButton, QTabWidget,
    QVBoxLayout, QWidget,
)

from app.config import INFERENCE_RESULTS_DIR, SCRIPTS_DIR
from app.ui.preprocessing._common import make_format_radio
from app.ui.widgets.audio_player import AudioPlayer
from app.ui.widgets.drag_drop_area import DragDropArea
from app.ui.widgets.progress_dialog import ProgressDialog
from app.ui.widgets.slider_with_input import SliderWithInput
from app.workers.base_worker import ProcessWorker


class _PostprocessTabs(QTabWidget):
    """Reverb / Compressor / Filters / NoiseGate tabs."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # reverb
        rev = QWidget()
        rev_l = QVBoxLayout(rev)
        self.room = SliderWithInput("Размер комнаты", 0.0, 1.0, 0.05, 0.15, decimals=2)
        self.wet = SliderWithInput("Уровень влажности", 0.0, 1.0, 0.05, 0.2, decimals=2)
        self.dry = SliderWithInput("Уровень сухости", 0.0, 1.0, 0.05, 0.8, decimals=2)
        self.damp = SliderWithInput("Демпфирование", 0.0, 1.0, 0.05, 0.7, decimals=2)
        for w in (self.room, self.wet, self.dry, self.damp):
            rev_l.addWidget(w)
        self.addTab(rev, "Ревербация")

        # compressor
        comp = QWidget()
        comp_l = QVBoxLayout(comp)
        self.ratio = SliderWithInput("Соотношение", 1, 20, 1, 4, decimals=0)
        self.thr = SliderWithInput("Порог, дБ", -20, -1, 1, -10, decimals=0)
        comp_l.addWidget(self.ratio)
        comp_l.addWidget(self.thr)
        self.addTab(comp, "Компрессор")

        # filters
        filt = QWidget()
        filt_l = QVBoxLayout(filt)
        self.hp = SliderWithInput("Фильтр нижних частот, Гц", 20, 2000, 10, 80, decimals=0)
        self.lp = SliderWithInput("Фильтр высоких частот, Гц", 2000, 20000, 100, 12000, decimals=0)
        self.lim = SliderWithInput("Лимитер-порог, дБ", -12, 0, 1, -1, decimals=0)
        filt_l.addWidget(self.hp)
        filt_l.addWidget(self.lp)
        filt_l.addWidget(self.lim)
        self.addTab(filt, "Фильтры")

        # gate
        gate = QWidget()
        gate_l = QVBoxLayout(gate)
        self.gate_thr = SliderWithInput("Порог, дБ", -80, 0, 1, -40, decimals=0)
        self.gate_ratio = SliderWithInput("Соотношение", 1, 20, 1, 10, decimals=0)
        self.gate_attack = SliderWithInput("Атака, мс", 1, 100, 1, 1, decimals=0)
        self.gate_release = SliderWithInput("Спад, мс", 1, 1000, 10, 100, decimals=0)
        gate_l.addWidget(self.gate_thr)
        gate_l.addWidget(self.gate_ratio)
        gate_l.addWidget(self.gate_attack)
        gate_l.addWidget(self.gate_release)
        self.addTab(gate, "Подавление шума")

    def to_dict(self) -> dict:
        return {
            "reverb": {"room_size": self.room.value(), "wet_level": self.wet.value(),
                       "dry_level": self.dry.value(), "damping": self.damp.value()},
            "compressor": {"ratio": self.ratio.value(), "threshold_db": self.thr.value()},
            "filters": {"low_cut_hz": int(self.hp.value()), "high_cut_hz": int(self.lp.value()),
                        "limiter_threshold_db": self.lim.value()},
            "gate": {"threshold_db": self.gate_thr.value(), "ratio": self.gate_ratio.value(),
                      "attack_ms": self.gate_attack.value(), "release_ms": self.gate_release.value()},
        }


class RvcInferenceWindow(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("Преобразование голоса (RVC)")
        self.resize(900, 780)

        layout = QVBoxLayout(self)

        # two-column inputs
        cols = QHBoxLayout()

        model_box = QGroupBox("Модель")
        ml = QVBoxLayout(model_box)
        ml.addWidget(QLabel(".pth"))
        self._dd_pth = DragDropArea("Перетащите .pth", allowed_exts=(".pth",), single_file=True)
        ml.addWidget(self._dd_pth)
        ml.addWidget(QLabel(".index"))
        self._dd_index = DragDropArea("Перетащите .index", allowed_exts=(".index",), single_file=True)
        ml.addWidget(self._dd_index)
        cols.addWidget(model_box, stretch=1)

        audio_box = QGroupBox("Входное аудио")
        al = QVBoxLayout(audio_box)
        self._dd_audio = DragDropArea("Перетащите аудиофайл", single_file=True)
        al.addWidget(self._dd_audio)
        cols.addWidget(audio_box, stretch=1)

        layout.addLayout(cols)

        reset = QPushButton("Сброс всех параметров")
        reset.clicked.connect(self._reset)
        layout.addWidget(reset)

        # conversion params
        conv = QGroupBox("Настройка преобразования голоса")
        cl = QVBoxLayout(conv)
        self._pitch = SliderWithInput("Тон, полутонов", -12, 12, 1, 0, decimals=0)
        self._index_rate = SliderWithInput("Скорость индексации", 0.0, 1.0, 0.05, 0.5, decimals=2)
        self._filter_radius = SliderWithInput("Радиус фильтра", 1, 5, 1, 3, decimals=0)
        self._rms = SliderWithInput("Скорость смешивания RMS", 0.0, 1.0, 0.05, 0.25, decimals=2)
        self._protect = SliderWithInput("Скорость защиты", 0.0, 0.5, 0.01, 0.33, decimals=2)
        for w in (self._pitch, self._index_rate, self._filter_radius, self._rms, self._protect):
            cl.addWidget(w)
        layout.addWidget(conv)

        # postprocess
        mix_box = QGroupBox("Настройка сведения аудио")
        mix_l = QVBoxLayout(mix_box)
        self._post = _PostprocessTabs()
        mix_l.addWidget(self._post)
        layout.addWidget(mix_box)

        # bottom: run + format + player + metrics
        bottom = QHBoxLayout()
        self._btn_run = QPushButton("Выполнить генерацию")
        self._btn_run.clicked.connect(self._run)
        bottom.addWidget(self._btn_run)
        fmt_row, _, self._rb_wav, _ = make_format_radio(self)
        bottom.addLayout(fmt_row)
        layout.addLayout(bottom)

        self._player = AudioPlayer()
        layout.addWidget(self._player)

        self._metrics = QLabel("Метрики появятся после генерации.")
        self._metrics.setObjectName("metrics")
        layout.addWidget(self._metrics)

    def _reset(self) -> None:
        # Reset numeric params but keep files
        self._pitch.set_value(0)
        self._index_rate.set_value(0.5)
        self._filter_radius.set_value(3)
        self._rms.set_value(0.25)
        self._protect.set_value(0.33)

    def _run(self) -> None:
        pth = self._dd_pth.files()
        index = self._dd_index.files()
        audio = self._dd_audio.files()
        if not (pth and index and audio):
            QMessageBox.warning(self, "Вход", "Нужны .pth, .index и аудиофайл.")
            return
        fmt = "wav" if self._rb_wav.isChecked() else "mp3"
        post_cfg = self._post.to_dict()
        post_json = INFERENCE_RESULTS_DIR / "_last_post.json"
        INFERENCE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        post_json.write_text(json.dumps(post_cfg, ensure_ascii=False), encoding="utf-8")

        model_name = Path(pth[0]).stem
        args = [
            "--pth", pth[0],
            "--index", index[0],
            "--input", audio[0],
            "--pitch", str(int(self._pitch.value())),
            "--index-rate", str(self._index_rate.value()),
            "--filter-radius", str(int(self._filter_radius.value())),
            "--rms-mix-rate", str(self._rms.value()),
            "--protect", str(self._protect.value()),
            "--format", fmt,
            "--model-name", model_name,
            "--postprocess-json", str(post_json),
            "--compute-metrics",
        ]

        dlg = ProgressDialog("Идёт преобразование голоса...", show_log=True, parent=self)
        worker = ProcessWorker("rvc_infer", SCRIPTS_DIR / "run_rvc_infer.py", args)
        worker.line_received.connect(dlg.append_log)

        result_holder: dict = {}

        def _line(line: str) -> None:
            if line.startswith("RESULT_JSON="):
                try:
                    result_holder["data"] = json.loads(line[len("RESULT_JSON="):])
                except json.JSONDecodeError:
                    pass

        worker.line_received.connect(_line)

        def _done(code: int) -> None:
            if code != 0:
                dlg.finish_error(f"Код {code}")
                return
            data = result_holder.get("data", {})
            out = data.get("output")
            if out:
                self._player.load(Path(out))
            wer = data.get("wer")
            secs = data.get("secs")
            self._metrics.setText(
                f"WER: {wer:.3f} | SECS: {secs:.3f}" if wer is not None and secs is not None
                else "Метрики недоступны."
            )
            dlg.set_status("Готово.")
            dlg.finish_success()

        worker.finished.connect(_done)
        dlg.cancel_requested.connect(worker.kill)
        worker.start()
        dlg.exec()
