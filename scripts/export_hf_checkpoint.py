#!/usr/bin/env python
"""Convert an HF-VITS training checkpoint into an inference-ready model.

finetune-hf-vits saves intermediate checkpoints with Accelerate
(run/checkpoint-<step>/model.safetensors = generator, model_1.safetensors =
discriminator, plus optimizer/scheduler state). Those weights are still in
the *weight-normalized* form used during training (weight_g / weight_v), and
the config/tokenizer live only in the parent run/ folder. Loading such a
checkpoint straight into VitsModel leaves the decoder (HiFi-GAN vocoder)
weights unmatched → the vocoder emits buzzing instead of speech.

The trainer's FINAL save fixes this by calling `decoder.remove_weight_norm()`
before save_pretrained. This script replicates that on any checkpoint:

  1. build a VitsModelForPreTraining from run/config.json,
  2. load the checkpoint's generator weights (weight-norm form),
  3. remove_weight_norm() to collapse them to the inference form,
  4. save_pretrained() + copy the tokenizer → a self-contained folder.

Usage:
    python scripts/export_hf_checkpoint.py --checkpoint <run/checkpoint-STEP>
    # optional: --run <run dir>  --output <dir>  (defaults: parent / <ckpt>_inference)
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

_TOKENIZER_FILES = (
    "tokenizer_config.json", "vocab.json", "special_tokens_map.json",
    "added_tokens.json", "tokenizer.json", "preprocessor_config.json",
)


def _import_pretraining_class():
    """VitsModelForPreTraining lives in transformers (>=4.4x); fall back to the
    finetune-hf-vits repo if a given transformers build lacks it."""
    try:
        from transformers import VitsModelForPreTraining  # type: ignore
        return VitsModelForPreTraining
    except Exception:
        repo = os.environ.get("VOICEGEN_FINETUNE_HF_VITS")
        if repo and Path(repo).is_dir():
            sys.path.insert(0, str(repo))
        # Some repo layouts expose it under utils/
        from utils.modeling_vits_training import VitsModelForPreTraining  # type: ignore
        return VitsModelForPreTraining


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="run/checkpoint-<step> dir")
    p.add_argument("--run", default=None, help="run/ dir (default: checkpoint's parent)")
    p.add_argument("--output", default=None,
                   help="output dir (default: <checkpoint>_inference)")
    args = p.parse_args()

    ckpt = Path(args.checkpoint)
    run = Path(args.run) if args.run else ckpt.parent
    out = Path(args.output) if args.output else ckpt.parent / f"{ckpt.name}_inference"

    weights = ckpt / "model.safetensors"
    if not weights.exists():
        print(f"ERROR: no model.safetensors in {ckpt}", file=sys.stderr)
        return 1
    if not (run / "config.json").exists():
        print(f"ERROR: no config.json in {run} — pass --run <run dir>.", file=sys.stderr)
        return 1

    import torch  # noqa: F401
    from safetensors.torch import load_file
    from transformers import VitsConfig

    VitsModelForPreTraining = _import_pretraining_class()

    print("Строю модель из config.json …", flush=True)
    config = VitsConfig.from_pretrained(str(run))
    model = VitsModelForPreTraining(config)

    print("Загружаю веса чекпойнта …", flush=True)
    state = load_file(str(weights))
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Report only counts — a handful of discriminator keys are expected to be
    # missing here (its weights are in model_1.safetensors, not needed for TTS).
    print(f"  загружено; отсутствуют {len(missing)}, лишних {len(unexpected)}", flush=True)

    print("Свёртываю weight-norm декодера (иначе будет жужжание) …", flush=True)
    model.decoder.remove_weight_norm()
    try:
        for d in model.discriminator.discriminators:
            d.remove_weight_norm()
    except Exception:
        pass  # discriminator not needed for inference

    out.mkdir(parents=True, exist_ok=True)
    print(f"Сохраняю инференс-модель в {out} …", flush=True)
    model.save_pretrained(str(out))

    # Copy tokenizer / feature extractor so the folder is self-contained.
    for name in _TOKENIZER_FILES:
        src = run / name
        if src.exists():
            shutil.copy2(src, out / name)
    # reference_speaker.wav for SECS/MCD, if present next to the run.
    ref = run / "reference_speaker.wav"
    if ref.exists():
        shutil.copy2(ref, out / "reference_speaker.wav")

    print(f"OK. Укажите в форме config.json: {out / 'config.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
