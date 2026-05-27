"""RVC inference: voice conversion + optional pedalboard post-chain + metrics."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from app.config import INFERENCE_RESULTS_DIR
from app.core.audio.io import ensure_dir, load_audio, save_audio
from app.core.postprocess.pedalboard_chain import PostProcessChain, apply as apply_postprocess


@dataclass
class RvcInferConfig:
    pth: Path
    index: Path
    input_audio: Path
    pitch: int = 0
    index_rate: float = 0.5
    filter_radius: int = 3
    rms_mix_rate: float = 0.25
    protect: float = 0.33
    output_format: str = "wav"
    model_name: str = "model"
    postprocess: PostProcessChain = field(default_factory=PostProcessChain)
    enable_postprocess: bool = True


def _infer_subprocess(cfg: RvcInferConfig, tmp_out: Path) -> None:
    cmd = [
        sys.executable, "-m", "rvc_core.infer_cli",
        "--pth", str(cfg.pth),
        "--index", str(cfg.index),
        "--input", str(cfg.input_audio),
        "--output", str(tmp_out),
        "--pitch", str(cfg.pitch),
        "--index-rate", str(cfg.index_rate),
        "--filter-radius", str(cfg.filter_radius),
        "--rms-mix-rate", str(cfg.rms_mix_rate),
        "--protect", str(cfg.protect),
    ]
    print(f"[rvc-infer] running: {' '.join(cmd)}", flush=True)
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        raise RuntimeError(f"RVC inference failed with exit {ret.returncode}")


def infer(cfg: RvcInferConfig) -> Path:
    out_dir = ensure_dir(INFERENCE_RESULTS_DIR)
    src_stem = cfg.input_audio.stem
    out_stem = f"{src_stem}_{cfg.model_name}_ver"
    tmp_out = out_dir / f"{out_stem}.tmp.wav"

    _infer_subprocess(cfg, tmp_out)
    audio, sr = load_audio(tmp_out, sr=None, mono=True)
    tmp_out.unlink(missing_ok=True)

    if cfg.enable_postprocess:
        audio = apply_postprocess(audio, sr, cfg.postprocess)

    final = save_audio(out_dir / out_stem, audio, sr, cfg.output_format)
    return final
