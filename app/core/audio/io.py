"""Centralized audio loading/saving via librosa + soundfile + pydub."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np


def load_audio(path: str | Path, sr: int | None = None, mono: bool = True) -> tuple[np.ndarray, int]:
    """Load audio as float32 in [-1, 1]. Returns (samples, sample_rate)."""
    import librosa
    audio, real_sr = librosa.load(str(path), sr=sr, mono=mono)
    return audio.astype(np.float32, copy=False), int(real_sr)


def save_audio(path: str | Path, audio: np.ndarray, sr: int, fmt: str = "wav") -> Path:
    """Save audio to wav or mp3 (mp3 via ffmpeg)."""
    import soundfile as sf
    path = Path(path)
    fmt = fmt.lower().lstrip(".")
    audio = np.clip(audio, -1.0, 1.0)
    if fmt == "wav":
        path = path.with_suffix(".wav")
        sf.write(str(path), audio, sr, subtype="PCM_16")
        return path
    if fmt == "mp3":
        tmp_wav = path.with_suffix(".tmp.wav")
        sf.write(str(tmp_wav), audio, sr, subtype="PCM_16")
        path = path.with_suffix(".mp3")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-i", str(tmp_wav), "-codec:a", "libmp3lame", "-b:a", "192k", str(path)],
                check=True,
            )
        finally:
            tmp_wav.unlink(missing_ok=True)
        return path
    raise ValueError(f"Unsupported output format: {fmt}")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
