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


def _replace_ext(path: Path, new_ext: str) -> Path:
    """Replace or append an extension without misinterpreting dots inside
    the stem (Path.with_suffix treats anything after the last dot as a
    suffix and would mangle names like 'song.net_remix')."""
    new_ext = new_ext if new_ext.startswith(".") else "." + new_ext
    name = path.name
    common = (".wav", ".mp3", ".tmp.wav", ".tmp.mp3", ".flac", ".ogg")
    for ext in sorted(common, key=len, reverse=True):
        if name.lower().endswith(ext):
            name = name[: -len(ext)]
            break
    return path.with_name(name + new_ext)


def _encode_mp3(tmp_wav: Path, out_path: Path) -> Path:
    """Try MP3 encoders in order: libmp3lame, then ffmpeg's bundled 'mp3'.
    If both fail (no libmp3lame, no licensed mp3 encoder in this build),
    fall back to keeping the .wav next to the requested .mp3 path."""
    for codec in ("libmp3lame", "mp3"):
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-i", str(tmp_wav), "-codec:a", codec, "-b:a", "192k", str(out_path)],
                check=True,
            )
            return out_path
        except subprocess.CalledProcessError:
            continue
    # Both encoders missing — keep the WAV so the user still gets the result.
    fallback = _replace_ext(out_path, ".wav")
    tmp_wav.replace(fallback)
    print(f"WARN: ffmpeg has no MP3 encoder; kept {fallback}", flush=True)
    return fallback


def save_audio(path: str | Path, audio: np.ndarray, sr: int, fmt: str = "wav") -> Path:
    """Save audio to wav or mp3 (mp3 via ffmpeg)."""
    import soundfile as sf
    path = Path(path)
    fmt = fmt.lower().lstrip(".")
    audio = np.clip(audio, -1.0, 1.0)
    if fmt == "wav":
        path = _replace_ext(path, ".wav")
        sf.write(str(path), audio, sr, subtype="PCM_16")
        return path
    if fmt == "mp3":
        tmp_wav = _replace_ext(path, ".tmp.wav")
        sf.write(str(tmp_wav), audio, sr, subtype="PCM_16")
        out_path = _replace_ext(path, ".mp3")
        try:
            return _encode_mp3(tmp_wav, out_path)
        finally:
            if tmp_wav.exists():
                tmp_wav.unlink(missing_ok=True)
    raise ValueError(f"Unsupported output format: {fmt}")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
