"""Main application window — three blocks + theme switcher + info."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget,
)

from app.config import APP_NAME, APP_VERSION, AVAILABLE_THEMES, DEFAULT_THEME
from app.ui.info_dialog import InfoDialog
from app.ui.preprocessing.denoiser import DenoiserWindow
from app.ui.preprocessing.silence_remover import SilenceRemoverWindow
from app.ui.preprocessing.slicer import SlicerWindow
from app.ui.preprocessing.vocal_remover import VocalRemoverWindow
from app.ui.rvc.inference import RvcInferenceWindow
from app.ui.rvc.train import RvcTrainWindow
from app.ui.tts.inference import TtsInferenceWindow
from app.ui.tts.train import TtsTrainWindow
from app.ui.widgets.chamfer_button import ChamferButton
from app.ui.widgets.trapezoid_frame import TrapezoidFrame
from app.utils.env_check import run_checks

_STYLES_DIR = Path(__file__).resolve().parent.parent / "styles"


def load_theme(theme: str) -> str:
    qss = _STYLES_DIR / f"{theme}.qss"
    if not qss.exists():
        return ""
    return qss.read_text(encoding="utf-8")


class _BlockButton(ChamferButton):
    """Narrow, square-ish button with 45°-cut corners for the main menu."""


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        self.resize(960, 640)
        self._env = run_checks()
        self._children: list[QWidget] = []

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 12, 16, 12)

        # --- top bar ---
        top = QHBoxLayout()
        top.addWidget(QLabel(f"<h2>{APP_NAME}</h2>"))
        top.addStretch(1)
        top.addWidget(QLabel("Тема:"))
        self._theme_box = QComboBox()
        self._theme_box.addItems(AVAILABLE_THEMES)
        self._theme_box.setCurrentText(DEFAULT_THEME)
        self._theme_box.currentTextChanged.connect(self._apply_theme)
        top.addWidget(self._theme_box)

        info_btn = QPushButton("i")
        info_btn.setFixedWidth(36)
        info_btn.setToolTip("О приложении")
        info_btn.clicked.connect(self._show_info)
        top.addWidget(info_btn)
        root.addLayout(top)

        # --- three blocks, stacked top-to-bottom ---
        blocks = QVBoxLayout()
        blocks.setSpacing(16)
        root.addLayout(blocks, stretch=1)

        def _centered_row(*buttons: QWidget) -> QHBoxLayout:
            row = QHBoxLayout()
            row.addStretch(1)
            for b in buttons:
                row.addWidget(b)
            row.addStretch(1)
            return row

        # 1. Предобработка
        b1 = TrapezoidFrame("Предобработка аудиозаписей")
        btn_edit = _BlockButton("Редактирование аудио")
        btn_edit.clicked.connect(self._show_preprocess_menu)
        b1.body_layout().addLayout(_centered_row(btn_edit))
        blocks.addWidget(b1, stretch=1)

        # 2. RVC
        b2 = TrapezoidFrame("Преобразование голоса")
        btn_rvc_train = _BlockButton("Обучить модель (RVC)")
        btn_rvc_train.clicked.connect(self._open_rvc_train)
        btn_rvc_infer = _BlockButton("Преобразовать голос")
        btn_rvc_infer.clicked.connect(self._open_rvc_infer)
        b2.body_layout().addLayout(_centered_row(btn_rvc_train, btn_rvc_infer))
        blocks.addWidget(b2, stretch=1)

        # 3. TTS
        b3 = TrapezoidFrame("Преобразование текста в речь")
        btn_tts_train = _BlockButton("Обучить модель (TTS)")
        btn_tts_train.clicked.connect(self._open_tts_train)
        btn_tts_infer = _BlockButton("Преобразовать текст в речь")
        btn_tts_infer.clicked.connect(self._open_tts_infer)
        b3.body_layout().addLayout(_centered_row(btn_tts_train, btn_tts_infer))
        blocks.addWidget(b3, stretch=1)

        # status bar
        status_lines = []
        status_lines.append("CUDA OK" if self._env.cuda_available else "CUDA отсутствует — обучение недоступно")
        status_lines.append("FFmpeg OK" if self._env.ffmpeg_available else "FFmpeg не найден")
        self.statusBar().showMessage(" • ".join(status_lines))

        self.setCentralWidget(central)
        self._apply_theme(DEFAULT_THEME)

    # --- theme ---
    def _apply_theme(self, theme: str) -> None:
        qss = load_theme(theme)
        app = self.window().window()  # type: ignore
        from PySide6.QtWidgets import QApplication
        QApplication.instance().setStyleSheet(qss)

    # --- info ---
    def _show_info(self) -> None:
        dlg = InfoDialog(self, env=self._env)
        dlg.exec()

    # --- preprocess menu ---
    def _show_preprocess_menu(self) -> None:
        from PySide6.QtWidgets import QDialog, QVBoxLayout
        dlg = QDialog(self)
        dlg.setWindowTitle("Редактирование аудио")
        lay = QVBoxLayout(dlg)
        for label, slot in [
            ("Отделение вокала от инструментала", self._open_vocal_remover),
            ("Шумоподавление", self._open_denoiser),
            ("Удаление тихих мест", self._open_silence_remover),
            ("Нарезка на фрагменты", self._open_slicer),
        ]:
            btn = QPushButton(label)
            btn.setMinimumHeight(36)
            btn.clicked.connect(lambda *_, s=slot: (dlg.accept(), s()))
            lay.addWidget(btn)
        dlg.exec()

    # --- per-feature openers ---
    def _open_vocal_remover(self) -> None:
        self._open(VocalRemoverWindow)

    def _open_denoiser(self) -> None:
        self._open(DenoiserWindow)

    def _open_silence_remover(self) -> None:
        self._open(SilenceRemoverWindow)

    def _open_slicer(self) -> None:
        self._open(SlicerWindow)

    def _open_rvc_train(self) -> None:
        self._open(RvcTrainWindow, env=self._env)

    def _open_rvc_infer(self) -> None:
        self._open(RvcInferenceWindow)

    def _open_tts_train(self) -> None:
        self._open(TtsTrainWindow, env=self._env)

    def _open_tts_infer(self) -> None:
        self._open(TtsInferenceWindow)

    def _open(self, cls, **kwargs) -> None:
        w = cls(parent=self, **kwargs)
        w.setAttribute(Qt.WA_DeleteOnClose)
        self._children.append(w)
        w.show()
