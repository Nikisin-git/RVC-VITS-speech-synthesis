"""Shared TTS metric computation for both Coqui and HF-VITS inference.

Computes WER (typed text vs Whisper transcript of the output), SECS and MCD
(output vs a reference_speaker.wav of the target speaker saved at train time).
Each metric is independent — one failing never blocks the others. Returns a
dict merged into RESULT_JSON.
"""

from __future__ import annotations

from pathlib import Path


def compute_tts_metrics(text: str, out_path: Path, reference_dir: Path,
                        language: str | None = "ru") -> dict:
    """reference_dir is the folder holding the model; we look for
    reference_speaker.wav in it for SECS/MCD."""
    result: dict = {}
    ref_path = Path(reference_dir) / "reference_speaker.wav"

    # WER — compares the text you asked to synthesize against Whisper's
    # transcription of the produced audio.
    try:
        from app.core.metrics.wer import wer_from_text
        wer = wer_from_text(text, out_path, language=language).get("wer")
        result["wer"] = wer
        print(f"WER: {wer:.3f}", flush=True)
    except Exception as me:
        print(f"WARN: WER failed: {me}", flush=True)
        result["wer_error"] = str(me)

    # SECS — speaker similarity vs the target speaker reference.
    if ref_path.exists():
        try:
            from app.core.metrics.secs import compute_secs
            result["secs"] = compute_secs(ref_path, out_path)
            result["secs_reference"] = str(ref_path)
            print(f"SECS: {result['secs']:.3f} (reference: {ref_path.name})", flush=True)
        except Exception as se:
            print(f"WARN: SECS failed: {type(se).__name__}: {se}", flush=True)
            result["secs_error"] = f"{type(se).__name__}: {se}"
    else:
        print("SECS: пропущен (reference_speaker.wav не найден рядом с моделью).", flush=True)
        result["secs_error"] = "reference_speaker.wav not found"

    # MCD — mel cepstral distortion vs the same reference.
    if ref_path.exists():
        try:
            from app.core.metrics.mcd import compute_mcd
            result["mcd"] = compute_mcd(ref_path, out_path)
            result["mcd_reference"] = str(ref_path)
            print(f"MCD: {result['mcd']:.2f} dB (reference: {ref_path.name})", flush=True)
        except Exception as me:
            print(f"WARN: MCD failed: {type(me).__name__}: {me}", flush=True)
            result["mcd_error"] = f"{type(me).__name__}: {me}"
    else:
        result["mcd_error"] = "reference_speaker.wav not found"

    return result
