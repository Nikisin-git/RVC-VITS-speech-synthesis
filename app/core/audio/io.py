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


def _mp3_is_valid(path: Path) -> bool:
    """Verify ffmpeg can actually read the MP3 back. Some LGPL ffmpeg builds
    have no libmp3lame and their native 'mp3' encoder writes malformed files
    (unseekable, bad frame size) that then fail on playback and metric loads.
    A quick decode-to-null probe catches that."""
    if not path.exists() or path.stat().st_size < 512:
        return False
    try:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
            check=True, capture_output=True,
        )
        return True
    except Exception:
        return False


def _encode_mp3_lameenc(tmp_wav: Path, out_path: Path, bitrate: int = 192) -> Path | None:
    """Encode MP3 with the `lameenc` package (bundles LAME, no ffmpeg codec
    needed — works regardless of the system ffmpeg build). Returns the output
    path on success, or None if lameenc isn't installed / fails so the caller
    can fall back to ffmpeg."""
    try:
        import lameenc
        import numpy as np
        import soundfile as sf
        data, real_sr = sf.read(str(tmp_wav), dtype="int16")
        channels = 1 if data.ndim == 1 else data.shape[1]
        enc = lameenc.Encoder()
        enc.set_bit_rate(bitrate)
        enc.set_in_sample_rate(int(real_sr))
        enc.set_channels(channels)
        enc.set_quality(2)  # 2 = high quality, reasonable speed
        mp3 = enc.encode(np.ascontiguousarray(data).tobytes())
        mp3 += enc.flush()
        Path(out_path).write_bytes(mp3)
        return out_path
    except Exception as e:
        print(f"INFO: lameenc unavailable ({e}); trying ffmpeg for MP3.", flush=True)
        return None


def _encode_mp3(tmp_wav: Path, out_path: Path) -> Path:
    """Encode to MP3. Prefer the self-contained `lameenc` (works on any
    machine); fall back to ffmpeg's libmp3lame, then verify the result and
    fall back to a valid WAV if the only available encoder produced a corrupt
    file (LGPL ffmpeg builds' native 'mp3' encoder does this)."""
    via_lame = _encode_mp3_lameenc(tmp_wav, out_path)
    if via_lame is not None and _mp3_is_valid(via_lame):
        return via_lame

    for codec in ("libmp3lame", "mp3"):
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-i", str(tmp_wav), "-codec:a", codec, "-b:a", "192k", str(out_path)],
                check=True,
            )
        except subprocess.CalledProcessError:
            continue
        if _mp3_is_valid(out_path):
            return out_path
        # Encoder ran but produced a broken file — discard and try the next.
        out_path.unlink(missing_ok=True)
    # No working MP3 encoder — keep the WAV so the user gets a usable file.
    fallback = _replace_ext(out_path, ".wav")
    tmp_wav.replace(fallback)
    print(
        f"WARN: в этой сборке ffmpeg нет рабочего MP3-кодировщика (libmp3lame). "
        f"Сохранил WAV: {fallback}. Для MP3 установите ffmpeg с libmp3lame или "
        f"выбирайте формат WAV.",
        flush=True,
    )
    return fallback


def _to_soundfile_layout(audio: np.ndarray) -> np.ndarray:
    """Convert librosa's (channels, frames) layout to soundfile's (frames, channels).
    librosa.load(mono=False) returns shape (C, N); sf.write expects (N, C). If we
    pass (C, N) directly, libsndfile sees C frames with N channels and dies with
    'Format not recognised' (or similar) when writing the WAV header.
    """
    if audio.ndim == 2 and audio.shape[0] < audio.shape[1]:
        return np.ascontiguousarray(audio.T)
    return audio


def save_audio(path: str | Path, audio: np.ndarray, sr: int, fmt: str = "wav") -> Path:
    """Save audio to wav or mp3 (mp3 via ffmpeg)."""
    import soundfile as sf
    path = Path(path)
    fmt = fmt.lower().lstrip(".")
    audio = np.clip(audio, -1.0, 1.0)
    audio = _to_soundfile_layout(audio)
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
