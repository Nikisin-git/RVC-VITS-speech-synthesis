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


def _import_repo_classes(repo: str | None):
    """VitsModelForPreTraining / VitsConfig live in the finetune-hf-vits repo's
    `utils` package (not in stock transformers). Put the repo on sys.path and
    import them from there, exactly as run_vits_finetuning.py does."""
    cand = repo or os.environ.get("VOICEGEN_FINETUNE_HF_VITS")
    if not cand or not Path(cand).is_dir():
        raise SystemExit(
            "ERROR: не найден репозиторий finetune-hf-vits. Передайте --repo "
            "<path> или задайте переменную VOICEGEN_FINETUNE_HF_VITS. Класс "
            "VitsModelForPreTraining определён в нём (пакет utils), а не в "
            "transformers."
        )
    repo_path = str(Path(cand).resolve())
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    # `utils` may pull matplotlib plotting helpers that call the removed
    # tostring_rgb; restore it so the import doesn't explode.
    try:
        import numpy as _np
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        if not hasattr(FigureCanvasAgg, "tostring_rgb"):
            FigureCanvasAgg.tostring_rgb = lambda self: _np.asarray(
                self.buffer_rgba())[:, :, :3].tobytes()
    except Exception:
        pass
    from utils import VitsModelForPreTraining, VitsConfig  # type: ignore
    return VitsModelForPreTraining, VitsConfig


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="run/checkpoint-<step> dir")
    p.add_argument("--run", default=None, help="run/ dir (default: checkpoint's parent)")
    p.add_argument("--output", default=None,
                   help="output dir (default: <checkpoint>_inference)")
    p.add_argument("--repo", default=None,
                   help="finetune-hf-vits repo (default: VOICEGEN_FINETUNE_HF_VITS)")
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

    # VitsModelForPreTraining and the matching VitsConfig come from the repo.
    VitsModelForPreTraining, VitsConfig = _import_repo_classes(args.repo)

    print("Строю модель из config.json …", flush=True)
    config = VitsConfig.from_pretrained(str(run))
    model = VitsModelForPreTraining(config)
    # VitsModelForPreTraining does NOT apply weight-norm in __init__ — the
    # trainer calls apply_weight_norm() before training, so the checkpoint's
    # decoder/flow/posterior weights are stored in weight-norm form
    # (weight_g/weight_v). Apply it here too so the checkpoint keys match and
    # remove_weight_norm() below has something to collapse.
    model.apply_weight_norm()

    print("Загружаю веса чекпойнта …", flush=True)
    state = load_file(str(weights))
    missing, unexpected = model.load_state_dict(state, strict=False)
    # A few discriminator keys may be missing (its weights are in
    # model_1.safetensors, not needed for TTS). If the counts are still large
    # after apply_weight_norm, the load didn't align — warn loudly.
    print(f"  загружено; отсутствуют {len(missing)}, лишних {len(unexpected)}", flush=True)
    if len(missing) > 20 and len(unexpected) > 20:
        print("  WARN: много несовпавших ключей — результат может быть некорректным.", flush=True)

    # Replicate the trainer's FINAL save EXACTLY: it collapses weight-norm only
    # on the decoder (and discriminators), leaving flow/posterior in weight-norm
    # form. Matching this byte-for-byte guarantees the same format as the
    # working run/ model — removing flow/posterior too would break loading.
    print("Свёртываю weight-norm декодера (как при финальном сохранении) …", flush=True)
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
