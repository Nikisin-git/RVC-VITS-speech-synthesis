"""UTMOS — reference-free naturalness (MOS) prediction for synthesized speech.

UTMOS predicts the Mean Opinion Score (roughly 1..5) that a human listening
panel would give for the *naturalness* of a speech clip, WITHOUT needing a
reference recording. Higher is better:
    ~4.0 и выше   естественно, близко к живой записи
    3.0 - 4.0     приемлемо, слышна синтетичность
    < 3.0         заметные артефакты / роботизированность
Живые студийные записи обычно набирают ~4.2-4.5.

Because it needs no reference, the same metric applies to both TTS output and
RVC voice-conversion output — it scores the clip on its own.

Implementation: the lightweight SpeechMOS UTMOS22 predictor
(tarepan/SpeechMOS) loaded via torch.hub — no fairseq dependency. The model is
downloaded once and cached by torch.hub. It expects 16 kHz mono audio.
"""

from __future__ import annotations

from pathlib import Path

# torch.hub spec — pinned so results are reproducible across machines.
_HUB_REPO = "tarepan/SpeechMOS:v1.2.0"
_HUB_MODEL = "utmos22_strong"
_TARGET_SR = 16000

_model = None  # cached predictor (loading it is the expensive part)


def _get_model():
    global _model
    if _model is None:
        import os
        import torch
        # Offline route: if VOICEGEN_SPEECHMOS_REPO points at a local clone of
        # tarepan/SpeechMOS, load the code from there instead of letting
        # torch.hub download the repo zip from github (which some networks
        # block). The model weights are still fetched once and cached under
        # ~/.cache/torch/hub, so a machine that cannot reach github at all
        # needs that cache copied over from one that can.
        local_repo = os.environ.get("VOICEGEN_SPEECHMOS_REPO")
        try:
            if local_repo and Path(local_repo).is_dir():
                _model = torch.hub.load(local_repo, _HUB_MODEL,
                                        source="local", trust_repo=True)
            else:
                _model = torch.hub.load(_HUB_REPO, _HUB_MODEL, trust_repo=True)
        except Exception as e:
            raise RuntimeError(
                "Не удалось загрузить модель UTMOS через torch.hub. Обычно это "
                "разовая загрузка с github.com — проверьте доступ в интернет и "
                "повторите (после первой успешной загрузки модель кэшируется в "
                "~/.cache/torch/hub и качаться больше не будет). Для полностью "
                "офлайн-машины склонируйте github.com/tarepan/SpeechMOS и укажите "
                f"путь в переменной VOICEGEN_SPEECHMOS_REPO. Исходная ошибка: {e}"
            ) from e
        _model.eval()
    return _model


def compute_utmos(audio_path: Path) -> float:
    """Predicted naturalness MOS (≈1..5, higher = better). Reference-free.

    Raises on empty/unreadable audio or if torch/the model can't be loaded, so
    the caller can record the error without blocking the other metrics.
    """
    import librosa
    import numpy as np
    import torch

    wave, _ = librosa.load(str(audio_path), sr=_TARGET_SR, mono=True)
    if wave is None or len(wave) == 0:
        raise ValueError("пустой аудиофайл — нечего оценивать.")

    model = _get_model()
    x = torch.from_numpy(np.ascontiguousarray(wave, dtype="float32")).unsqueeze(0)
    with torch.no_grad():
        score = model(x, _TARGET_SR)
    return float(score.squeeze().item())
