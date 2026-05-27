"""Environment checks: CUDA, FFmpeg, disk space."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from app.config import MIN_FREE_DISK_GB, USER_DATA_DIR


@dataclass
class EnvReport:
    cuda_available: bool
    cuda_version: str | None
    gpu_name: str | None
    gpu_vram_gb: float | None
    driver_version: str | None
    ffmpeg_available: bool
    ffmpeg_version: str | None
    espeak_ng_available: bool
    free_disk_gb: float
    disk_ok: bool

    @property
    def can_train(self) -> bool:
        return self.cuda_available and self.ffmpeg_available and self.disk_ok

    def to_lines(self) -> list[str]:
        lines = []
        lines.append(f"CUDA доступна: {'да' if self.cuda_available else 'нет'}")
        if self.cuda_available:
            lines.append(f"  Версия CUDA: {self.cuda_version}")
            lines.append(f"  GPU: {self.gpu_name}")
            lines.append(f"  VRAM: {self.gpu_vram_gb:.1f} ГБ")
            if self.driver_version:
                lines.append(f"  Драйвер NVIDIA: {self.driver_version}")
        lines.append(f"FFmpeg: {'да' if self.ffmpeg_available else 'НЕТ'}")
        if self.ffmpeg_version:
            lines.append(f"  {self.ffmpeg_version}")
        lines.append(f"espeak-ng: {'да' if self.espeak_ng_available else 'нет (нужен для русского TTS)'}")
        lines.append(f"Свободно на диске: {self.free_disk_gb:.1f} ГБ "
                     f"({'OK' if self.disk_ok else f'< {MIN_FREE_DISK_GB} ГБ'})")
        return lines


def _check_cuda() -> tuple[bool, str | None, str | None, float | None]:
    try:
        import torch
    except ImportError:
        return False, None, None, None
    if not torch.cuda.is_available():
        return False, None, None, None
    cuda_version = torch.version.cuda
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024 ** 3)
    return True, cuda_version, props.name, vram_gb


def _check_nvidia_driver() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip().splitlines()[0]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _check_ffmpeg() -> tuple[bool, str | None]:
    path = shutil.which("ffmpeg")
    if not path:
        return False, None
    try:
        out = subprocess.run([path, "-version"], capture_output=True, text=True, timeout=5)
        first_line = out.stdout.splitlines()[0] if out.stdout else ""
        return True, first_line
    except (subprocess.TimeoutExpired, OSError):
        return True, None


def _check_espeak() -> bool:
    return shutil.which("espeak-ng") is not None or shutil.which("espeak") is not None


def _check_disk(target: Path) -> tuple[float, bool]:
    target.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(target)
    free_gb = usage.free / (1024 ** 3)
    return free_gb, free_gb >= MIN_FREE_DISK_GB


def run_checks() -> EnvReport:
    cuda_ok, cuda_ver, gpu_name, vram = _check_cuda()
    driver = _check_nvidia_driver()
    ff_ok, ff_ver = _check_ffmpeg()
    espeak = _check_espeak()
    free_gb, disk_ok = _check_disk(USER_DATA_DIR)
    return EnvReport(
        cuda_available=cuda_ok,
        cuda_version=cuda_ver,
        gpu_name=gpu_name,
        gpu_vram_gb=vram,
        driver_version=driver,
        ffmpeg_available=ff_ok,
        ffmpeg_version=ff_ver,
        espeak_ng_available=espeak,
        free_disk_gb=free_gb,
        disk_ok=disk_ok,
    )


def recommended_batch_size(vram_gb: float | None) -> int:
    if vram_gb is None:
        return 4
    if vram_gb <= 6:
        return 4
    if vram_gb <= 8:
        return 8
    if vram_gb <= 12:
        return 12
    return 16
