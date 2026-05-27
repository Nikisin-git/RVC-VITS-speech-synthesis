"""Remove silent regions exceeding a duration threshold."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.config import UNMUTED_DIR
from app.core.audio.io import ensure_dir, load_audio, save_audio


def _detect_nonsilent_intervals(audio: np.ndarray, sr: int, top_db: float,
                                 frame_length: int = 2048, hop_length: int = 512) -> list[tuple[int, int]]:
    """Returns sample-index intervals (start, end) of non-silent regions."""
    import librosa
    intervals = librosa.effects.split(audio, top_db=top_db,
                                       frame_length=frame_length, hop_length=hop_length)
    return [(int(s), int(e)) for s, e in intervals]


def remove_silence(
    input_path: Path,
    max_silence_sec: float,
    silence_threshold_db: float,
    output_format: str = "wav",
) -> Path:
    """Drop silent gaps longer than `max_silence_sec`. Threshold in dBFS (e.g., -40)."""
    input_path = Path(input_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    audio, sr = load_audio(input_path, sr=None, mono=True)

    # librosa.effects.split uses top_db: how many dB below ref is considered silent.
    # silence_threshold_db is signed (e.g., -40). Convert to a positive top_db relative to peak.
    peak = np.max(np.abs(audio)) + 1e-9
    peak_db = 20.0 * np.log10(peak)
    top_db = max(1.0, peak_db - silence_threshold_db)

    intervals = _detect_nonsilent_intervals(audio, sr, top_db=top_db)
    if not intervals:
        out = ensure_dir(UNMUTED_DIR) / f"{input_path.stem} [unmuted]"
        return save_audio(out, audio, sr, output_format)

    max_gap_samples = int(max_silence_sec * sr)
    pieces: list[np.ndarray] = []
    prev_end = intervals[0][0]
    for start, end in intervals:
        gap = start - prev_end
        if pieces:
            keep = min(gap, max_gap_samples)
            if keep > 0:
                pieces.append(audio[prev_end:prev_end + keep])
        pieces.append(audio[start:end])
        prev_end = end

    result = np.concatenate(pieces) if pieces else audio
    out = ensure_dir(UNMUTED_DIR) / f"{input_path.stem} [unmuted]"
    return save_audio(out, result, sr, output_format)
