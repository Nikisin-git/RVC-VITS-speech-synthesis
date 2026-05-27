"""Word Error Rate via Whisper transcription + jiwer."""

from __future__ import annotations

import re
from pathlib import Path

_WHISPER_MODEL = None
_MODEL_NAME = "small"


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)
    return text


def _get_whisper(model_name: str | None = None):
    global _WHISPER_MODEL, _MODEL_NAME
    name = model_name or _MODEL_NAME
    if _WHISPER_MODEL is None or _MODEL_NAME != name:
        import whisper  # openai-whisper
        _WHISPER_MODEL = whisper.load_model(name)
        _MODEL_NAME = name
    return _WHISPER_MODEL


def transcribe(audio_path: Path, language: str | None = None, model_name: str | None = None) -> str:
    """Transcribe audio with Whisper. Returns plain text."""
    model = _get_whisper(model_name)
    result = model.transcribe(str(audio_path), language=language, fp16=False)
    return str(result.get("text", "")).strip()


def compute_wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate (0..1+). Uses jiwer if available, falls back to manual."""
    ref = _normalize(reference)
    hyp = _normalize(hypothesis)
    if not ref:
        return 1.0 if hyp else 0.0
    try:
        import jiwer  # type: ignore
        return float(jiwer.wer(ref, hyp))
    except ImportError:
        return _levenshtein_wer(ref.split(), hyp.split())


def _levenshtein_wer(ref: list[str], hyp: list[str]) -> float:
    m, n = len(ref), len(hyp)
    if m == 0:
        return 1.0 if n else 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n] / m


def wer_from_audio(reference_audio: Path, hypothesis_audio: Path,
                   language: str | None = "ru", model_name: str | None = None) -> dict:
    """Transcribe two audios and compute WER between transcriptions."""
    ref_text = transcribe(reference_audio, language, model_name)
    hyp_text = transcribe(hypothesis_audio, language, model_name)
    return {
        "reference_text": ref_text,
        "hypothesis_text": hyp_text,
        "wer": compute_wer(ref_text, hyp_text),
    }


def wer_from_text(reference_text: str, hypothesis_audio: Path,
                  language: str | None = "ru", model_name: str | None = None) -> dict:
    """Compare known reference text against Whisper transcription of generated audio."""
    hyp_text = transcribe(hypothesis_audio, language, model_name)
    return {
        "reference_text": reference_text,
        "hypothesis_text": hyp_text,
        "wer": compute_wer(reference_text, hyp_text),
    }
