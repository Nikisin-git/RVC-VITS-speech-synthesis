"""VITS inference + optional pitch shift / time stretch postprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import VITS_DIR
from app.core.audio.io import ensure_dir, save_audio
from app.utils.path_utils import shorten_for_windows
from app.utils.transliterate import text_to_filename


@dataclass
class TtsInferConfig:
    generator_pth: Path
    config_json: Path
    text: str
    length_scale: float = 1.0       # 0.1..2.0
    pitch_shift_semitones: int = 0  # -12..+12
    output_format: str = "wav"
    model_name: str = "model"
    language: str | None = "ru"


def _synthesize(cfg: TtsInferConfig) -> tuple[np.ndarray, int]:
    """Synthesize raw audio using Coqui-TTS API."""
    from TTS.utils.synthesizer import Synthesizer  # type: ignore

    syn = Synthesizer(
        tts_checkpoint=str(cfg.generator_pth),
        tts_config_path=str(cfg.config_json),
        use_cuda=False,  # let the env decide; safe default
    )
    # length_scale supported by VITS via `length_scale` kwarg in some versions;
    # use synthesize_speech, otherwise fall back to time-stretch.
    wav = syn.tts(cfg.text, language_name=cfg.language) if hasattr(syn, "tts") else syn.tts_inference(cfg.text)
    audio = np.asarray(wav, dtype=np.float32)
    sr = int(getattr(syn, "output_sample_rate", 22050))
    return audio, sr


def _apply_length_scale(audio: np.ndarray, sr: int, length_scale: float) -> np.ndarray:
    if abs(length_scale - 1.0) < 1e-3:
        return audio
    import librosa
    rate = 1.0 / max(length_scale, 1e-3)
    return librosa.effects.time_stretch(audio, rate=rate).astype(np.float32)


def _apply_pitch_shift(audio: np.ndarray, sr: int, semitones: int) -> np.ndarray:
    if semitones == 0:
        return audio
    import librosa
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=float(semitones)).astype(np.float32)


def infer(cfg: TtsInferConfig) -> Path:
    audio, sr = _synthesize(cfg)
    audio = _apply_length_scale(audio, sr, cfg.length_scale)
    audio = _apply_pitch_shift(audio, sr, cfg.pitch_shift_semitones)

    out_dir = ensure_dir(VITS_DIR / "Inference Results")
    safe_stem = text_to_filename(cfg.text, max_len=20)
    target = out_dir / f"{safe_stem}_{cfg.model_name}_ver"
    target_with_ext = target.with_suffix(f".{cfg.output_format.lower()}")
    final_target = shorten_for_windows(target_with_ext)
    return save_audio(final_target.with_suffix(""), audio, sr, cfg.output_format)
