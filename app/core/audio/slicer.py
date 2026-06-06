"""Slice audio into fragments by timer or by VAD (silero-vad)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from app.config import CUTS_DIR
from app.core.audio.io import ensure_dir, load_audio, save_audio


class SliceMode(str, Enum):
    TIMER = "timer"
    VAD = "vad"


class TailMode(str, Enum):
    KEEP = "keep"
    MERGE = "merge"
    DROP = "drop"


@dataclass
class SlicerConfig:
    mode: SliceMode = SliceMode.TIMER
    target_seconds: int = 10
    tail_mode: TailMode = TailMode.KEEP
    min_tail_seconds: float = 4.0
    output_format: str = "wav"
    single_track: bool = False


def _vad_speech_intervals(audio: np.ndarray, sr: int) -> list[tuple[int, int]]:
    """Use silero-vad to find speech intervals."""
    import torch
    from silero_vad import load_silero_vad, get_speech_timestamps  # type: ignore

    model = load_silero_vad()
    tensor = torch.from_numpy(audio).float()
    if sr != 16000:
        import torchaudio
        tensor = torchaudio.functional.resample(tensor, sr, 16000)
        speech = get_speech_timestamps(tensor, model, sampling_rate=16000)
        ratio = sr / 16000
        return [(int(s["start"] * ratio), int(s["end"] * ratio)) for s in speech]
    speech = get_speech_timestamps(tensor, model, sampling_rate=sr)
    return [(int(s["start"]), int(s["end"])) for s in speech]


def _slice_by_timer(audio: np.ndarray, sr: int, target_seconds: int) -> list[np.ndarray]:
    step = int(target_seconds * sr)
    return [audio[i:i + step] for i in range(0, len(audio), step)]


def _slice_by_vad(audio: np.ndarray, sr: int, target_seconds: int) -> list[np.ndarray]:
    """Greedy: accumulate VAD intervals until total length >= target."""
    target = target_seconds * sr
    intervals = _vad_speech_intervals(audio, sr)
    if not intervals:
        return _slice_by_timer(audio, sr, target_seconds)
    chunks: list[np.ndarray] = []
    cur_start = intervals[0][0]
    cur_end = intervals[0][1]
    for start, end in intervals[1:]:
        if end - cur_start >= target:
            chunks.append(audio[cur_start:cur_end])
            cur_start = start
        cur_end = end
    chunks.append(audio[cur_start:cur_end])
    return chunks


def _apply_tail_policy(chunks: list[np.ndarray], sr: int, cfg: SlicerConfig) -> list[np.ndarray]:
    min_samples = int(cfg.min_tail_seconds * sr)
    if not chunks:
        return chunks
    if len(chunks[-1]) >= min_samples:
        return chunks
    if cfg.tail_mode == TailMode.KEEP:
        return chunks
    if cfg.tail_mode == TailMode.DROP:
        return chunks[:-1] if len(chunks) > 1 else chunks
    if cfg.tail_mode == TailMode.MERGE and len(chunks) >= 2:
        merged = np.concatenate([chunks[-2], chunks[-1]])
        return chunks[:-2] + [merged]
    return chunks


def slice_file(input_paths: Path | list[Path], cfg: SlicerConfig) -> list[Path]:
    """Slice file(s). With single_track=True, concatenates all inputs into one
    output. Otherwise processes each file independently and returns every chunk.
    """
    if isinstance(input_paths, (str, Path)):
        input_paths = [Path(input_paths)]
    input_paths = [Path(p).resolve() for p in input_paths]
    if not input_paths:
        return []

    if cfg.single_track:
        audios: list[np.ndarray] = []
        sr_ref: int | None = None
        for p in input_paths:
            audio, sr = load_audio(p, sr=sr_ref, mono=True)
            if sr_ref is None:
                sr_ref = sr
            audios.append(audio)
        merged = np.concatenate(audios) if len(audios) > 1 else audios[0]
        out_dir = ensure_dir(CUTS_DIR / "Single track")
        # Name after the first file; if multiple, append count for clarity.
        stem = input_paths[0].stem
        if len(input_paths) > 1:
            stem = f"{stem}_and_{len(input_paths) - 1}_more"
        return [save_audio(out_dir / f"{stem} [single]", merged, sr_ref, cfg.output_format)]

    outputs: list[Path] = []
    out_dir = ensure_dir(CUTS_DIR / "Sliced")
    for input_path in input_paths:
        audio, sr = load_audio(input_path, sr=None, mono=True)
        stem = input_path.stem
        if cfg.mode == SliceMode.VAD:
            chunks = _slice_by_vad(audio, sr, cfg.target_seconds)
        else:
            chunks = _slice_by_timer(audio, sr, cfg.target_seconds)
        chunks = _apply_tail_policy(chunks, sr, cfg)
        for idx, chunk in enumerate(chunks, start=1):
            if chunk.size == 0:
                continue
            out_path = save_audio(out_dir / f"{stem}_{idx:03d} [sliced]", chunk, sr, cfg.output_format)
            outputs.append(out_path)
    return outputs
