"""Workspace helpers.

Upstream RVC reads/writes everything relative to `./logs/<exp_name>/`. We let
the caller specify any working directory, then chdir into it before invoking
upstream code.
"""

from __future__ import annotations

import contextlib
import os
import shutil
from pathlib import Path


@contextlib.contextmanager
def chdir(path: Path):
    """`os.chdir` into `path` and restore previous cwd on exit."""
    prev = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(prev)


def ensure_experiment_layout(workspace: Path, exp_name: str) -> Path:
    """Create `<workspace>/logs/<exp_name>/` and return that path."""
    workspace = Path(workspace).resolve()
    exp_dir = workspace / "logs" / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def materialize_dataset(input_dir: Path, target: Path) -> Path:
    """Copy/symlink audio files from `input_dir` to `target` for upstream preprocess."""
    target.mkdir(parents=True, exist_ok=True)
    for src in Path(input_dir).iterdir():
        if src.suffix.lower() in (".wav", ".mp3", ".flac", ".m4a", ".ogg"):
            dst = target / src.name
            if dst.exists():
                continue
            try:
                os.symlink(src, dst)
            except (OSError, NotImplementedError):
                shutil.copy2(src, dst)
    return target
