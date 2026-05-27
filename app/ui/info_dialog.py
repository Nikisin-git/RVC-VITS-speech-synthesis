"""About dialog: app name, version, requirements, env check, docs links."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget

from app.config import APP_NAME, APP_VERSION
from app.utils.env_check import EnvReport, run_checks

_REQUIREMENTS = """\
Минимальные требования:
  GPU: NVIDIA, 6 ГБ VRAM, CUDA CC ≥ 7.0
  CUDA Toolkit: 11.8 или 12.1
  RAM: 16 ГБ (рекомендуется 32 ГБ)
  Диск: 30 ГБ свободно (рекомендуется 100 ГБ)
  CPU: 4 ядра с AVX2 (рекомендуется 8+)
  Python: 3.10.x (строго)
  ОС: Windows 10 21H2+ / Ubuntu 22.04+

Рекомендуется (обучение):
  GPU: RTX 3060/4060+ с 12 ГБ VRAM
"""


class InfoDialog(QDialog):
    def __init__(self, parent: QWidget | None = None, env: EnvReport | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("О приложении")
        self.setModal(True)
        self.setMinimumWidth(520)

        env = env or run_checks()
        layout = QVBoxLayout(self)

        header = QLabel(f"<h2>{APP_NAME}</h2><p>Версия {APP_VERSION}</p>")
        header.setTextFormat(Qt.RichText)
        layout.addWidget(header)

        layout.addWidget(QLabel("<b>Системные требования</b>"))
        req = QTextEdit()
        req.setReadOnly(True)
        req.setPlainText(_REQUIREMENTS)
        req.setMaximumHeight(180)
        layout.addWidget(req)

        layout.addWidget(QLabel("<b>Проверка окружения</b>"))
        check = QTextEdit()
        check.setReadOnly(True)
        check.setPlainText("\n".join(env.to_lines()))
        check.setMaximumHeight(150)
        layout.addWidget(check)

        layout.addWidget(QLabel(
            '<a href="https://github.com/Nikisin-git/RVC-VITS-speech-synthesis">Документация / Исходный код</a>'
        ))
        layout.itemAt(layout.count() - 1).widget().setOpenExternalLinks(True)

        btn = QPushButton("Закрыть")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)
