"""Live training-loss chart for ProgressDialog.

Polls TF-event files in the run directory every few seconds and redraws
the standard training_curves figure with matplotlib. Used by RVC and TTS
training windows alike — the framework parameter switches the scalar set.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class TrainingChartWidget(QWidget):
    def __init__(
        self,
        event_dir: Path,
        framework: str,
        refresh_seconds: int = 30,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._event_dir = Path(event_dir)
        self._framework = framework
        self._canvas = None
        self._figure = None
        self._ax = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        try:
            import matplotlib
            matplotlib.use("QtAgg")
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        except Exception as e:
            layout.addWidget(QLabel(f"matplotlib недоступен: {e}"))
            return

        self._figure = Figure(figsize=(8, 4), dpi=110)
        self._ax = self._figure.add_subplot(111)
        self._figure.subplots_adjust(left=0.10, right=0.78, top=0.92, bottom=0.14)
        self._canvas = FigureCanvas(self._figure)
        layout.addWidget(self._canvas)

        self._placeholder = QLabel("Ожидание первых записей TensorBoard…")
        self._placeholder.setStyleSheet("color: #888; padding: 6px;")
        layout.addWidget(self._placeholder)

        self._timer = QTimer(self)
        self._timer.setInterval(refresh_seconds * 1000)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        # First draw attempt after a short delay so the trainer has time to
        # create the event file.
        QTimer.singleShot(2500, self.refresh)

    def stop(self) -> None:
        if hasattr(self, "_timer"):
            self._timer.stop()

    def refresh(self) -> None:
        if self._ax is None:
            return
        try:
            from app.core.metrics.training_curves import (
                _RVC_SCALARS, _TTS_SCALARS, _find_event_dirs, _match_curve,
                ema_smooth, read_scalars,
            )
        except Exception as e:
            self._placeholder.setText(f"Ошибка импорта: {e}")
            return

        # Diagnostics first: tell the user what we see at the path so a wrong
        # event_dir or a missing tensorboard package becomes obvious.
        event_dirs = _find_event_dirs(self._event_dir)
        if not event_dirs:
            self._placeholder.setText(
                f"Tfevents ещё не созданы в {self._event_dir}. "
                "Это нормально в первые секунды обучения; если сообщение не "
                "уходит — проверьте, что путь к каталогу логов верный."
            )
            return

        try:
            scalars = read_scalars(self._event_dir)
        except Exception as e:
            self._placeholder.setText(f"Не удалось прочитать события: {e}")
            return

        keys = _RVC_SCALARS if self._framework == "rvc" else _TTS_SCALARS
        palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]

        self._ax.clear()
        drew_anything = False
        total_points = 0
        for (short, _), colour in zip(keys.items(), palette):
            c = _match_curve(scalars, short)
            if c is None or not c.values:
                continue
            total_points += len(c.values)
            self._ax.plot(c.steps, c.values, color=colour, alpha=0.18, linewidth=0.8)
            self._ax.plot(c.steps, ema_smooth(c.values, 0.7),
                          color=colour, linewidth=1.6, label=short)
            drew_anything = True

        if not drew_anything:
            tags = sorted(scalars.keys())[:6]
            self._placeholder.setText(
                f"События найдены ({len(scalars)} тегов), но ни один из "
                f"ожидаемых лоссов ещё не записан. Первые теги: {', '.join(tags)}"
            )
            return

        self._placeholder.hide()
        self._ax.set_xlabel("Шаг обучения")
        self._ax.set_ylabel("Значение функции потерь")
        self._ax.set_title(f"Шагов записано: {total_points}", fontsize=9, loc="right", color="#888")
        self._ax.grid(True, alpha=0.25)
        self._ax.set_yscale("symlog", linthresh=0.5)
        self._ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                        frameon=False, fontsize=8)
        self._canvas.draw_idle()
