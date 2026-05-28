"""CLI wrapper around upstream F0 extraction scripts.

Routes pm/harvest/rmvpe through `extract_f0_print.py`,
and rmvpe_gpu through `extract_f0_rmvpe.py`.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import runpy
import sys
from pathlib import Path

import rvc_core  # noqa: F401
from rvc_core._workspace import chdir


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--exp-name", required=True)
    p.add_argument("--sr", default="40k", choices=["32k", "40k", "48k"])
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--logs-dir", required=True)
    p.add_argument("--f0-method", default="rmvpe_gpu",
                   choices=["pm", "harvest", "rmvpe", "rmvpe_gpu"])
    p.add_argument("--n-procs", type=int, default=max(1, multiprocessing.cpu_count() // 2))
    p.add_argument("--gpu-index", default="0")
    p.add_argument("--half", action="store_true",
                   help="Use FP16 for rmvpe_gpu (faster on RTX cards).")
    args = p.parse_args()

    logs_dir = Path(args.logs_dir).resolve()
    workspace = logs_dir.parent.parent
    exp_rel = f"logs/{args.exp_name}"
    upstream_dir = rvc_core.VENDORED_PATH / "infer" / "modules" / "train" / "extract"

    with chdir(workspace):
        if args.f0_method != "rmvpe_gpu":
            script = upstream_dir / "extract_f0_print.py"
            # exp_dir n_p f0method
            sys.argv = [str(script), exp_rel, str(args.n_procs), args.f0_method]
        else:
            script = upstream_dir / "extract_f0_rmvpe.py"
            # n_part i_part i_gpu exp_dir is_half
            sys.argv = [
                str(script),
                "1", "0",
                str(args.gpu_index),
                exp_rel,
                "True" if args.half else "False",
            ]
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
        print(f"[rvc_core.extract_f0] script={script.name} argv={sys.argv[1:]}", flush=True)
        runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
