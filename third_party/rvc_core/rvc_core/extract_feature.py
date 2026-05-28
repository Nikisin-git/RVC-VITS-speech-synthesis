"""CLI wrapper around upstream `infer/modules/train/extract_feature_print.py`."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

import torch

import rvc_core  # noqa: F401
from rvc_core._workspace import chdir, copy_artifacts, vendored_exp_dir, vendored_workspace


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--exp-name", required=True)
    p.add_argument("--sr", default="40k", choices=["32k", "40k", "48k"])
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--logs-dir", required=True)
    p.add_argument("--version", default="v2", choices=["v1", "v2"])
    p.add_argument("--gpu-index", default="0")
    p.add_argument("--half", action="store_true")
    args = p.parse_args()

    exp_dir = vendored_exp_dir(args.exp_name)
    exp_rel = f"logs/{args.exp_name}"

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    script = vendored_workspace() / "infer" / "modules" / "train" / "extract_feature_print.py"

    with chdir(vendored_workspace()):
        sys.argv = [
            str(script),
            device,
            "1", "0",
            str(args.gpu_index),
            exp_rel,
            args.version,
            "true" if args.half else "false",
        ]
        print(f"[rvc_core.extract_feature] argv={sys.argv[1:]}", flush=True)
        runpy.run_path(str(script), run_name="__main__")

    copy_artifacts(exp_dir, Path(args.logs_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
