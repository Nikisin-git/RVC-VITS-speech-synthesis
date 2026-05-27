"""Speaker Embedding Cosine Similarity via resemblyzer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

_ENCODER = None


def _get_encoder():
    global _ENCODER
    if _ENCODER is None:
        from resemblyzer import VoiceEncoder  # type: ignore
        _ENCODER = VoiceEncoder()
    return _ENCODER


def embed(audio_path: Path) -> np.ndarray:
    from resemblyzer import preprocess_wav  # type: ignore
    enc = _get_encoder()
    wav = preprocess_wav(Path(audio_path))
    return enc.embed_utterance(wav)


def compute_secs(reference_audio: Path, target_audio: Path) -> float:
    """Cosine similarity between speaker embeddings. Range [-1, 1], higher is better."""
    a = embed(reference_audio)
    b = embed(target_audio)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return cos
