#!/usr/bin/env python
"""Make an HF-VITS training checkpoint usable for inference.

finetune-hf-vits saves intermediate checkpoints via Accelerate, so a
run/checkpoint-<step>/ folder contains model.safetensors (generator),
model_1.safetensors (discriminator), optimizer/scheduler state — but NO
config.json or tokenizer. Those live in the parent run/ folder.

This copies the missing aux files (config.json + tokenizer + feature
extractor) from the run/ folder into the checkpoint folder, so you can then
point the TTS inference form at <checkpoint>/config.json. The checkpoint's
own model.safetensors (that epoch's generator weights) is used as-is.

Usage:
    python scripts/export_hf_checkpoint.py --checkpoint <path-to-checkpoint-dir>
    # optional: --run <path-to-run-dir>   (default: checkpoint's parent)
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

_AUX_FILES = (
    "config.json",
    "tokenizer_config.json",
    "vocab.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "tokenizer.json",
    "preprocessor_config.json",
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="path to a checkpoint-<step> dir")
    p.add_argument("--run", default=None,
                   help="run/ dir with config.json + tokenizer (default: checkpoint's parent)")
    args = p.parse_args()

    ckpt = Path(args.checkpoint)
    run = Path(args.run) if args.run else ckpt.parent

    if not (ckpt / "model.safetensors").exists():
        print(f"ERROR: no model.safetensors in {ckpt} — is this a checkpoint dir?",
              file=sys.stderr)
        return 1
    if not (run / "config.json").exists():
        print(f"ERROR: no config.json in {run} — point --run at the run/ folder.",
              file=sys.stderr)
        return 1

    copied = []
    for name in _AUX_FILES:
        src = run / name
        if src.exists():
            shutil.copy2(src, ckpt / name)
            copied.append(name)

    print(f"OK: {ckpt} готов к инференсу.")
    print(f"Скопировано из {run}: {', '.join(copied)}")
    print(f"Укажите в форме config.json: {ckpt / 'config.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
