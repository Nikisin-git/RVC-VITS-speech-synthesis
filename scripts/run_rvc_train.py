#!/usr/bin/env python
"""CLI: RVC full training pipeline."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import _bootstrap  # noqa: F401

from app.core.rvc.trainer import RvcTrainConfig, train_full


def _materialize_dataset(list_path: Path) -> Path:
    """Copy files referenced in the list into a temp directory expected by RVC."""
    tmp = Path(tempfile.mkdtemp(prefix="rvc_ds_"))
    for line in list_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        src = Path(line)
        if src.exists():
            shutil.copy2(src, tmp / src.name)
    return tmp


def main() -> int:
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset-dir")
    src.add_argument("--dataset-list", help="text file with one path per line")
    p.add_argument("--model-name", required=True)
    p.add_argument("--sample-rate", default="40k", choices=["32k", "40k", "48k"])
    p.add_argument("--f0-method", default="rmvpe_gpu",
                   choices=["pm", "harvest", "rmvpe", "rmvpe_gpu"])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--create-zip", action="store_true")
    p.add_argument("--cancel-flag")
    args = p.parse_args()

    if args.dataset_dir:
        dataset_dir = Path(args.dataset_dir)
    else:
        dataset_dir = _materialize_dataset(Path(args.dataset_list))

    cfg = RvcTrainConfig(
        dataset_dir=dataset_dir,
        model_name=args.model_name,
        sample_rate=args.sample_rate,
        f0_method=args.f0_method,
        epochs=args.epochs,
        save_every=args.save_every,
        batch_size=args.batch_size,
        create_zip=args.create_zip,
    )
    cancel = Path(args.cancel_flag) if args.cancel_flag else None

    try:
        result = train_full(cfg, cancel_flag=cancel)
        print(f"RESULT: {result}", flush=True)
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
