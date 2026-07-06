"""HuggingFace Transformers VITS inference (MMS / mms-tts-rus family).

Separate from the Coqui path in `inferencer.py` because the two use
incompatible checkpoint formats. This one loads a `transformers` VitsModel
(config.json with `"architectures": ["VitsModel"]`), which is what
facebook/mms-tts-rus and models fine-tuned from it use.

Detection: a HF-Transformers VITS config.json contains
`"architectures": ["VitsModel"]` and/or `"model_type": "vits"` with a
`transformers_version` field — as opposed to Coqui's top-level `"model"`
string plus `"audio"`/`"model_args"` blocks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import VITS_DIR
from app.core.audio.io import ensure_dir, save_audio
from app.utils.path_utils import shorten_for_windows
from app.utils.transliterate import text_to_filename


@dataclass
class HfVitsInferConfig:
    model_dir: Path                 # folder or HF id with config.json + weights
    text: str
    length_scale: float = 1.0       # >1 slower speech (maps to speaking_rate)
    noise_scale: float | None = None
    output_format: str = "wav"
    model_name: str = "model"


def is_hf_vits_config(config_path: Path) -> bool:
    """True if config.json is a HuggingFace Transformers VITS config."""
    try:
        data = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    archs = data.get("architectures") or []
    if isinstance(archs, list) and any("VitsModel" in str(a) for a in archs):
        return True
    # Fallback: HF configs carry transformers_version; Coqui ones don't.
    return data.get("model_type") == "vits" and "transformers_version" in data


def _synthesize(cfg: HfVitsInferConfig) -> tuple[np.ndarray, int]:
    import torch
    from transformers import VitsModel, AutoTokenizer  # type: ignore

    src = str(cfg.model_dir)
    model = VitsModel.from_pretrained(src)
    tokenizer = AutoTokenizer.from_pretrained(src)

    # length_scale in our UI is "slower when >1"; HF exposes speaking_rate
    # where LOWER is slower. Invert so the control feels consistent.
    if abs(cfg.length_scale - 1.0) > 1e-3:
        model.speaking_rate = 1.0 / max(cfg.length_scale, 1e-3)
    if cfg.noise_scale is not None:
        model.noise_scale = float(cfg.noise_scale)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    inputs = tokenizer(cfg.text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**inputs).waveform  # (1, num_samples)
    audio = output[0].detach().cpu().numpy().astype(np.float32)
    sr = int(model.config.sampling_rate)
    return audio, sr


def infer(cfg: HfVitsInferConfig) -> Path:
    audio, sr = _synthesize(cfg)
    out_dir = ensure_dir(VITS_DIR / "Inference Results")
    safe_stem = text_to_filename(cfg.text, max_len=20)
    target = out_dir / f"{safe_stem}_{cfg.model_name}_ver"
    target_with_ext = target.with_suffix(f".{cfg.output_format.lower()}")
    final_target = shorten_for_windows(target_with_ext)
    return save_audio(final_target.with_suffix(""), audio, sr, cfg.output_format)
