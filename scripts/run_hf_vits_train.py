#!/usr/bin/env python
"""CLI: fine-tune a HuggingFace-Transformers VITS (facebook/mms-tts-rus family)
on a local dataset, via the ylacombe/finetune-hf-vits training loop.

The upstream `run_vits_finetuning.py` only loads data through
`load_dataset(dataset_name, ...)` with no local-path support, and it lives in
a separate repo with a Cython `monotonic_align` build step and a discriminator
checkpoint requirement. This wrapper:

  1. locates the cloned finetune-hf-vits repo (via --repo or
     VOICEGEN_FINETUNE_HF_VITS),
  2. builds the audiofolder dataset from the manifest + wavs,
  3. writes a training config tuned for small-dataset fine-tuning,
  4. monkeypatches datasets.load_dataset so dataset_name="audiofolder"
     transparently loads our local data_dir,
  5. runs the upstream training entrypoint in-process.

Prerequisites the user must set up once (documented in DEPLOYMENT.md):
  - git clone https://github.com/ylacombe/finetune-hf-vits
  - pip install -r finetune-hf-vits/requirements.txt
  - build monotonic_align (cd monotonic_align && python setup.py build_ext --inplace)
  - a base checkpoint WITH a discriminator (convert_original_discriminator_checkpoint.py)
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401


def _gpu_lacks_tensor_cores() -> bool:
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        name = torch.cuda.get_device_name(0).lower()
        return any(s in name for s in ("gtx 16", "gtx 15", "gtx 10", "gtx titan", "mx "))
    except Exception:
        return False


def _find_repo(explicit: str | None) -> Path:
    cand = explicit or os.environ.get("VOICEGEN_FINETUNE_HF_VITS")
    if not cand:
        raise SystemExit(
            "ERROR: не указан путь к finetune-hf-vits. Клонируйте репозиторий "
            "(git clone https://github.com/ylacombe/finetune-hf-vits) и передайте "
            "--repo <path> либо задайте переменную VOICEGEN_FINETUNE_HF_VITS."
        )
    repo = Path(cand)
    script = repo / "run_vits_finetuning.py"
    if not script.exists():
        raise SystemExit(f"ERROR: не найден {script}. Проверьте путь к репозиторию.")
    return repo


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--audio-dir", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--base-model", default="facebook/mms-tts-rus",
                   help="HF id or local path of the base VITS *with discriminator*.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--repo", help="path to a cloned finetune-hf-vits repo")
    args = p.parse_args()

    repo = _find_repo(args.repo)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = output_dir / "_dataset"

    # 1. Build the audiofolder dataset from the manifest.
    prep = Path(__file__).parent / "prepare_hf_vits_dataset.py"
    ret = subprocess.run(
        [sys.executable, str(prep), "--manifest", args.manifest,
         "--audio-dir", args.audio_dir, "--output", str(dataset_dir)],
    )
    if ret.returncode != 0:
        print("ERROR: подготовка датасета не удалась.", file=sys.stderr)
        return 1

    # 2. Write the training config. fp16 off on cards without Tensor Cores
    #    (fp16 GAN training diverges on GTX 16xx — same root cause as the
    #    Coqui buzzing). Loss weights follow the upstream English example.
    use_fp16 = not _gpu_lacks_tensor_cores()
    config = {
        "project_name": args.model_name,
        "push_to_hub": False,
        "overwrite_output_dir": True,
        "output_dir": str(output_dir / "run"),
        "dataset_name": "audiofolder",
        "audio_column_name": "audio",
        "text_column_name": "text",
        "train_split_name": "train",
        "eval_split_name": "train",
        "override_speaker_embeddings": True,
        "max_duration_in_seconds": 20,
        "min_duration_in_seconds": 1.0,
        "max_tokens_length": 500,
        "model_name_or_path": args.base_model,
        "preprocessing_num_workers": 2,
        "do_train": True,
        "num_train_epochs": args.epochs,
        "gradient_accumulation_steps": 1,
        "per_device_train_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "adam_beta1": 0.8,
        "adam_beta2": 0.99,
        "warmup_ratio": 0.01,
        "do_eval": True,
        "eval_steps": 50,
        "per_device_eval_batch_size": max(2, args.batch_size // 2),
        "max_eval_samples": 25,
        "do_step_schedule_per_epoch": True,
        "weight_disc": 3,
        "weight_fmaps": 1,
        "weight_gen": 1,
        "weight_kl": 1.5,
        "weight_duration": 1,
        "weight_mel": 35,
        "fp16": use_fp16,
        "seed": 456,
    }
    config_path = output_dir / "train_config.json"
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    if not use_fp16:
        print("[hf-vits] GPU без Tensor Cores → fp16=False (стабильнее)", flush=True)

    # 3. Monkeypatch load_dataset so "audiofolder" pulls our local data_dir.
    import datasets  # type: ignore
    _orig_load = datasets.load_dataset

    def _patched_load(path, *a, **kw):
        if path == "audiofolder" and "data_dir" not in kw:
            kw["data_dir"] = str(dataset_dir)
        return _orig_load(path, *a, **kw)

    datasets.load_dataset = _patched_load
    # The upstream script imports load_dataset by name at module load, so also
    # patch the attribute it will look up after runpy imports it. runpy runs
    # the module fresh, so patching the datasets module object is enough.

    # 4. Run the upstream trainer in-process from the repo directory.
    os.environ.setdefault("MPLBACKEND", "Agg")
    prev_cwd = os.getcwd()
    os.chdir(str(repo))
    sys.path.insert(0, str(repo))
    sys.argv = [str(repo / "run_vits_finetuning.py"), str(config_path)]
    try:
        print(f"[hf-vits] launching finetune-hf-vits with {config_path}", flush=True)
        runpy.run_path(str(repo / "run_vits_finetuning.py"), run_name="__main__")
    finally:
        os.chdir(prev_cwd)

    result = {
        "output_dir": str(output_dir / "run"),
        "config": str(config_path),
        "dataset": str(dataset_dir),
    }
    print(f"RESULT: {json.dumps(result, ensure_ascii=False)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
