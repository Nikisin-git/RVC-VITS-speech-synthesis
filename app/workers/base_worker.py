"""QProcess wrapper used by all UI dialogs that launch heavy CLI scripts."""

from __future__ import annotations

import datetime as _dt
import os
import sys
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

from app.config import LOGS_DIR


class ProcessWorker(QObject):
    """Run a Python CLI script in a child process, tee output to a log file."""

    line_received = Signal(str)            # raw stdout/stderr line
    finished = Signal(int)                 # exit code
    failed = Signal(str)                   # error string
    started = Signal()

    def __init__(self, label: str, script: Path, args: list[str],
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._label = label
        self._script = Path(script)
        self._args = [str(a) for a in args]
        self._cancel_flag: Path | None = None
        self._proc: QProcess | None = None
        self._log_path = self._make_log_path()
        self._log_fh = None  # type: ignore[assignment]

    def _make_log_path(self) -> Path:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        ts = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        safe = "".join(c for c in self._label if c.isalnum() or c in "-_") or "process"
        return LOGS_DIR / f"{ts}_{safe}.log"

    @property
    def log_path(self) -> Path:
        return self._log_path

    def set_cancel_flag(self, path: Path) -> None:
        self._cancel_flag = path

    def request_soft_cancel(self) -> None:
        """Write the cancel flag file so the child can exit cleanly."""
        if self._cancel_flag is not None:
            try:
                self._cancel_flag.parent.mkdir(parents=True, exist_ok=True)
                self._cancel_flag.touch()
            except OSError as e:
                self.failed.emit(f"Не удалось создать flag-файл отмены: {e}")

    def kill(self) -> None:
        if self._proc is not None and self._proc.state() != QProcess.NotRunning:
            self._proc.kill()

    def start(self) -> None:
        proc = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        proc.setProcessEnvironment(env)
        proc.setProcessChannelMode(QProcess.MergedChannels)

        proc.readyReadStandardOutput.connect(self._on_stdout)
        proc.finished.connect(self._on_finished)
        proc.errorOccurred.connect(self._on_error)

        self._log_fh = open(self._log_path, "w", encoding="utf-8")  # noqa: SIM115
        self._proc = proc

        program = sys.executable
        args = [str(self._script), *self._args]
        proc.start(program, args)
        self.started.emit()

    def _on_stdout(self) -> None:
        assert self._proc is not None
        data = bytes(self._proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in data.splitlines():
            self.line_received.emit(line)
            if self._log_fh is not None:
                self._log_fh.write(line + "\n")
                self._log_fh.flush()

    def _on_finished(self, code: int, _status) -> None:
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None
        if self._cancel_flag is not None:
            try:
                if self._cancel_flag.exists():
                    self._cancel_flag.unlink()
            except OSError:
                pass
        self.finished.emit(int(code))

    def _on_error(self, _err) -> None:
        if self._proc is not None:
            self.failed.emit(self._proc.errorString())


def make_cancel_flag(name: str) -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    return LOGS_DIR / f".cancel_{name}_{os.getpid()}.flag"


def install_line_handler(worker: ProcessWorker, handler: Callable[[str], None]) -> None:
    worker.line_received.connect(handler)
