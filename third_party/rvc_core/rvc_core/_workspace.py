"""Workspace helpers.

Upstream RVC expects assets/, configs/, i18n/ and logs/ to be relative to the
current working directory. We chdir into the vendored tree so those relative
paths resolve correctly, run the upstream pipeline inside `_vendored/logs/<exp>/`,
then copy or move the produced artifacts to whatever `--logs-dir` the caller
asked for.
"""

from __future__ import annotations

import contextlib
import os
import shutil
from pathlib import Path

import rvc_core  # noqa: F401  side-effect: sys.path setup


def vendored_workspace() -> Path:
    return rvc_core.VENDORED_PATH


def _ensure_logs_redirected() -> None:
    """If VOICEGEN_RVC_LOGS is set, transparently send <vendored>/logs there
    via a Windows directory junction. Lets the user keep RVC's heavy disk
    traffic (tfevents flushes, .pth snapshots) outside OneDrive/Documents
    without touching the upstream code that hardcodes a relative 'logs/'
    path inside its working directory.
    """
    target = os.environ.get("VOICEGEN_RVC_LOGS")
    if not target:
        return
    target_path = Path(target).resolve()
    target_path.mkdir(parents=True, exist_ok=True)

    logs_link = vendored_workspace() / "logs"
    if logs_link.exists() or logs_link.is_symlink():
        # Already pointing where the user asked? Nothing to do.
        try:
            if logs_link.resolve() == target_path:
                return
        except OSError:
            pass
        # Existing 'logs' is something else — don't clobber it.
        return

    try:
        # Try a junction first (no admin required on Windows).
        if os.name == "nt":
            import subprocess
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(logs_link), str(target_path)],
                check=True, capture_output=True,
            )
        else:
            os.symlink(target_path, logs_link, target_is_directory=True)
    except Exception as e:
        print(f"WARN: cannot redirect RVC logs to {target_path}: {e}", flush=True)


def vendored_exp_dir(exp_name: str) -> Path:
    _ensure_logs_redirected()
    d = vendored_workspace() / "logs" / exp_name
    d.mkdir(parents=True, exist_ok=True)
    return d


@contextlib.contextmanager
def chdir(path: Path):
    """`os.chdir` into `path` and restore previous cwd on exit."""
    prev = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(prev)


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


def copy_artifacts(src_exp_dir: Path, dst_logs_dir: Path | None) -> None:
    """If `dst_logs_dir` is set and differs from `src_exp_dir`, copy everything over."""
    if dst_logs_dir is None:
        return
    dst = Path(dst_logs_dir).resolve()
    src = Path(src_exp_dir).resolve()
    if dst == src:
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
