"""Drag&drop file area + 'Attach files' button + list with per-row delete."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QVBoxLayout, QWidget,
)

from app.config import SUPPORTED_INPUT_FORMATS


class _FileRow(QWidget):
    remove_clicked = Signal(str)

    def __init__(self, path: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._path = path
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 2, 6, 2)
        lay.addWidget(QLabel(Path(path).name), stretch=1)
        btn = QPushButton("×")
        btn.setFixedWidth(28)
        btn.clicked.connect(lambda: self.remove_clicked.emit(self._path))
        lay.addWidget(btn)


class DragDropArea(QWidget):
    files_changed = Signal(list)  # list[str]

    def __init__(self, label: str = "Перетащите файлы сюда",
                 allowed_exts: tuple[str, ...] = SUPPORTED_INPUT_FORMATS,
                 single_file: bool = False,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._allowed = tuple(e.lower() for e in allowed_exts)
        self._single = single_file
        self._files: list[str] = []

        self.setAcceptDrops(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self._hint = QLabel(label)
        self._hint.setAlignment(Qt.AlignCenter)
        self._hint.setObjectName("dragdrop_hint")
        self._hint.setMinimumHeight(60)
        layout.addWidget(self._hint)

        self._list = QListWidget()
        self._list.setMinimumHeight(80)
        layout.addWidget(self._list, stretch=1)

        row = QHBoxLayout()
        row.addStretch(1)
        self._btn_attach = QPushButton("Прикрепить файлы")
        self._btn_attach.clicked.connect(self._open_dialog)
        row.addWidget(self._btn_attach)
        layout.addLayout(row)

    def files(self) -> list[str]:
        return list(self._files)

    def clear(self) -> None:
        self._files.clear()
        self._list.clear()
        self.files_changed.emit([])

    def add_files(self, paths: list[str]) -> None:
        for p in paths:
            if not p.lower().endswith(self._allowed):
                continue
            if p in self._files:
                continue
            if self._single:
                self._files = [p]
                self._list.clear()
            else:
                self._files.append(p)
            item = QListWidgetItem()
            row = _FileRow(p)
            row.remove_clicked.connect(self._remove)
            item.setSizeHint(row.sizeHint())
            self._list.addItem(item)
            self._list.setItemWidget(item, row)
        self.files_changed.emit(self._files)

    def _open_dialog(self) -> None:
        filt = "Audio (" + " ".join(f"*{e}" for e in self._allowed) + ")"
        if self._single:
            path, _ = QFileDialog.getOpenFileName(self, "Выбор файла", "", filt)
            if path:
                self.add_files([path])
        else:
            paths, _ = QFileDialog.getOpenFileNames(self, "Выбор файлов", "", filt)
            if paths:
                self.add_files(paths)

    def _remove(self, path: str) -> None:
        try:
            idx = self._files.index(path)
        except ValueError:
            return
        self._files.pop(idx)
        self._list.takeItem(idx)
        self.files_changed.emit(self._files)

    # --- drag/drop ---
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if paths:
            self.add_files(paths)
        event.acceptProposedAction()
