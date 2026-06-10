"""Modal progress dialog with cancel button and a dark log viewer below."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QMessageBox, QProgressBar, QPushButton,
    QTabWidget, QVBoxLayout, QWidget,
)

from app.ui.widgets.log_viewer import LogViewer


class ProgressDialog(QDialog):
    cancel_requested = Signal()

    def __init__(
        self,
        title: str = "Идёт обработка...",
        show_log: bool = True,
        determinate: bool = False,
        parent: QWidget | None = None,
        chart_event_dir: Path | None = None,
        chart_framework: str | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(720 if chart_event_dir else 560)
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)

        layout = QVBoxLayout(self)

        self._status = QLabel(title)
        layout.addWidget(self._status)

        self._bar = QProgressBar()
        if determinate:
            self._bar.setRange(0, 100)
            self._bar.setValue(0)
        else:
            self._bar.setRange(0, 0)
        layout.addWidget(self._bar)

        self._log: LogViewer | None = None
        self._chart = None
        if show_log and chart_event_dir and chart_framework:
            # Both log and chart wanted — show them in a tab widget.
            tabs = QTabWidget()
            self._log = LogViewer()
            self._log.setMinimumHeight(220)
            tabs.addTab(self._log, "Лог")
            try:
                from app.ui.widgets.training_chart import TrainingChartWidget
                self._chart = TrainingChartWidget(chart_event_dir, chart_framework)
                tabs.addTab(self._chart, "График обучения")
            except Exception as e:
                err = QLabel(f"График недоступен: {e}")
                err.setStyleSheet("color: #c84040; padding: 8px;")
                tabs.addTab(err, "График обучения")
            layout.addWidget(tabs, stretch=1)
        elif show_log:
            self._log = LogViewer()
            self._log.setMinimumHeight(180)
            layout.addWidget(self._log, stretch=1)

        row = QHBoxLayout()
        row.addStretch(1)
        self._btn_cancel = QPushButton("Отмена")
        self._btn_cancel.clicked.connect(self._confirm_cancel)
        row.addWidget(self._btn_cancel)
        layout.addLayout(row)

    def set_status(self, text: str) -> None:
        self._status.setText(text)

    def set_percent(self, percent: int) -> None:
        if self._bar.maximum() == 0:
            self._bar.setRange(0, 100)
        self._bar.setValue(max(0, min(100, int(percent))))

    def append_log(self, line: str) -> None:
        if self._log is not None:
            self._log.append_line(line)

    def finish_success(self) -> None:
        self._bar.setRange(0, 100)
        self._bar.setValue(100)
        self._status.setText("Готово.")
        title = self.windowTitle().rstrip(".").rstrip()
        self.setWindowTitle(f"{title} — готово")
        self._btn_cancel.setText("Закрыть")
        self._btn_cancel.clicked.disconnect()
        self._btn_cancel.clicked.connect(self.accept)
        if self._chart is not None:
            # Final refresh, then stop polling.
            self._chart.refresh()
            self._chart.stop()

    def finish_error(self, msg: str) -> None:
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._status.setText(f"Ошибка: {msg}")
        self._btn_cancel.setText("Закрыть")
        self._btn_cancel.clicked.disconnect()
        self._btn_cancel.clicked.connect(self.reject)
        if self._chart is not None:
            self._chart.stop()

    def _confirm_cancel(self) -> None:
        reply = QMessageBox.question(
            self,
            "Подтверждение",
            "Вы уверены, что хотите остановить процесс?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.cancel_requested.emit()
