"""Path helpers with Windows MAX_PATH (260 chars) protection."""

from __future__ import annotations

import os
import re
from pathlib import Path

WINDOWS_MAX_PATH = 260
_INVALID_NAME_CHARS = re.compile(r'[\\/:*?"<>|]')


def sanitize_filename(name: str) -> str:
    """Strip characters forbidden in filenames on Windows."""
    return _INVALID_NAME_CHARS.sub("", name).strip()


def validate_model_name(name: str) -> tuple[bool, str | None]:
    """Validate model name. Returns (ok, error_message)."""
    if not name:
        return False, "Имя модели не может быть пустым."
    if " " in name:
        return False, "Пробелы запрещены."
    bad = _INVALID_NAME_CHARS.search(name)
    if bad:
        return False, f"Запрещённый символ: {bad.group()}"
    return True, None


def shorten_for_windows(target: Path, max_path: int = WINDOWS_MAX_PATH, min_stem: int = 4) -> Path:
    """If full path exceeds max_path, truncate stem until it fits."""
    target = Path(target)
    if os.name != "nt":
        return target
    full = str(target)
    if len(full) <= max_path:
        return target
    parent = str(target.parent)
    suffix = target.suffix
    overhead = len(parent) + 1 + len(suffix)
    available = max_path - overhead
    if available < min_stem:
        return target
    new_stem = target.stem[:available]
    return target.parent / f"{new_stem}{suffix}"


def unique_path(target: Path) -> Path:
    """Return a path that does not yet exist by appending _N suffix."""
    target = Path(target)
    if not target.exists():
        return target
    parent, stem, suffix = target.parent, target.stem, target.suffix
    i = 1
    while True:
        cand = parent / f"{stem}_{i}{suffix}"
        if not cand.exists():
            return cand
        i += 1
