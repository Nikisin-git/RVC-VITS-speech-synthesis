"""Speech denoising via DeepFilterNet 3."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.config import DENOISING_DIR
from app.core.audio.io import ensure_dir, load_audio, save_audio

_DF_MODEL = None
_DF_STATE = None
_DF_SR = 48000


def _get_model():
    global _DF_MODEL, _DF_STATE
    if _DF_MODEL is None:
        from df.enhance import init_df  # type: ignore
        _DF_MODEL, _DF_STATE, _ = init_df()
    return _DF_MODEL, _DF_STATE


def denoise(input_path: Path, output_format: str = "wav") -> Path:
    """Apply DeepFilterNet 3 to a single file. Returns output path."""
    from df.enhance import enhance  # type: ignore
    import torch

    input_path = Path(input_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    out_dir = ensure_dir(DENOISING_DIR)
    model, state = _get_model()

    audio, sr = load_audio(input_path, sr=_DF_SR, mono=True)
    tensor = torch.from_numpy(audio).unsqueeze(0)
    print(f"[denoiser] enhancing {input_path.name} ({len(audio)/sr:.2f}s @ {sr} Hz)", flush=True)
    enhanced = enhance(model, state, tensor)
    enhanced_np = enhanced.squeeze(0).cpu().numpy().astype(np.float32)

    out_path = save_audio(out_dir / f"{input_path.stem} [denoised]", enhanced_np, _DF_SR, output_format)
    return out_path
