"""RVC training orchestration.

Calls into the bundled fork at `third_party/rvc_core/`. The fork exposes
CLI-friendly entrypoints for preprocess, feature extraction, F0 extraction,
training and index building. We chain them here and print progress lines
in a stable format consumed by `app.workers.log_parser`.

NOTE: the actual RVC fork is not vendored in this commit. The CLI script
`scripts/run_rvc_train.py` calls `train_full()` here; if the fork is not
installed, training fails with a clear message.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

from app.config import RVC_MODELS_DIR, TRAINING_DIR


@dataclass
class RvcTrainConfig:
    dataset_dir: Path
    model_name: str
    sample_rate: str = "40k"            # "32k" | "40k" | "48k"
    f0_method: str = "rmvpe_gpu"        # pm | harvest | rmvpe | rmvpe_gpu
    epochs: int = 200
    save_every: int = 10
    batch_size: int = 12
    create_zip: bool = True


def _rvc_module(name: str) -> list[str]:
    return [sys.executable, "-m", f"rvc_core.{name}"]


def _atomic_replace(src: Path, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    tmp.replace(dst)


def _checkpoint_paths(model_name: str) -> tuple[Path, Path]:
    weights_dir = TRAINING_DIR / "Assets" / "Weights" / model_name
    logs_dir = TRAINING_DIR / "logs" / model_name
    weights_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return weights_dir, logs_dir


def _zip_final(model_name: str, pth: Path, index: Path) -> Path:
    out_dir = RVC_MODELS_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{model_name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pth, pth.name)
        z.write(index, index.name)
    return zip_path


def train_full(cfg: RvcTrainConfig, cancel_flag: Path | None = None) -> dict:
    """Run the full RVC pipeline. Soft-cancellable via `cancel_flag` file.

    Pipeline (all subprocess to the RVC fork):
        1. preprocess (resample, slice into 3s)
        2. extract_f0 (chosen method)
        3. extract_feature (hubert)
        4. train (epochs)
        5. train_index (faiss IVF)
    """
    weights_dir, logs_dir = _checkpoint_paths(cfg.model_name)

    common = [
        "--exp-name", cfg.model_name,
        "--sr", cfg.sample_rate,
        "--dataset-dir", str(cfg.dataset_dir),
        "--logs-dir", str(logs_dir),
    ]

    stages: list[tuple[str, list[str]]] = [
        ("preprocess", _rvc_module("preprocess") + common),
        ("extract_f0", _rvc_module("extract_f0") + common + ["--f0-method", cfg.f0_method]),
        ("extract_feature", _rvc_module("extract_feature") + common),
        ("train", _rvc_module("train") + common + [
            "--epochs", str(cfg.epochs),
            "--save-every", str(cfg.save_every),
            "--batch-size", str(cfg.batch_size),
            "--weights-dir", str(weights_dir),
        ] + (["--cancel-flag", str(cancel_flag)] if cancel_flag else [])),
        ("train_index", _rvc_module("train_index") + common),
    ]

    for stage_name, cmd in stages:
        print(f"[rvc] === stage: {stage_name} ===", flush=True)
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            raise RuntimeError(f"RVC stage '{stage_name}' failed with exit {ret.returncode}")
        if cancel_flag and cancel_flag.exists():
            print("[rvc] cancellation requested between stages — stopping", flush=True)
            break

    # Locate produced artifacts
    final_pth = sorted(weights_dir.glob(f"{cfg.model_name}_*.pth"))[-1] if any(weights_dir.glob("*.pth")) else None
    final_index = next(iter(logs_dir.glob(f"added_*_{cfg.model_name}_v2.index")), None)

    result: dict = {
        "weights_dir": str(weights_dir),
        "logs_dir": str(logs_dir),
        "final_pth": str(final_pth) if final_pth else None,
        "final_index": str(final_index) if final_index else None,
    }

    # Save one representative audio fragment from the training set as
    # reference_speaker.wav next to the weights. SECS at inference time
    # uses this to measure output↔target-speaker similarity, which is the
    # canonical metric — comparing to the input audio (a different speaker)
    # would only measure how poorly the conversion erased the source voice.
    try:
        gt_wavs_dir = logs_dir / "0_gt_wavs"
        if gt_wavs_dir.is_dir():
            samples = sorted(gt_wavs_dir.glob("*.wav"))
            if samples:
                ref_dst = weights_dir / "reference_speaker.wav"
                shutil.copy2(samples[len(samples) // 2], ref_dst)
                result["reference_speaker"] = str(ref_dst)
    except Exception as e:
        print(f"WARN: failed to save reference_speaker.wav: {e}", flush=True)

    if cfg.create_zip and final_pth and final_index:
        zip_path = _zip_final(cfg.model_name, final_pth, final_index)
        result["zip"] = str(zip_path)

    (logs_dir / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    # Render training curves into PNGs alongside the weights for the report.
    try:
        from app.core.metrics.training_curves import generate_curves
        ckpt_steps = sorted({
            int(p.stem.rsplit("_", 1)[-1].lstrip("s"))
            for p in weights_dir.glob(f"{cfg.model_name}_e*_s*.pth")
            if p.stem.rsplit("_", 1)[-1].lstrip("s").isdigit()
        })
        plots = generate_curves(
            event_root=logs_dir, output_dir=logs_dir,
            framework="rvc", checkpoint_steps=ckpt_steps or None,
        )
        if plots.get("training_curves"):
            result["training_curves"] = plots["training_curves"]
        if plots.get("gan_balance"):
            result["gan_balance"] = plots["gan_balance"]
    except Exception as e:
        print(f"WARN: failed to render training curves: {e}", flush=True)

    print(f"[rvc] training done: {result}", flush=True)
    return result
