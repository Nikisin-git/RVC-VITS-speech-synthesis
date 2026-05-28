"""CLI wrapper around upstream F0 extraction scripts."""

from __future__ import annotations

import argparse
import multiprocessing
import os
import runpy
import sys
from pathlib import Path

import rvc_core  # noqa: F401
from rvc_core._workspace import chdir, copy_artifacts, vendored_exp_dir, vendored_workspace


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
    p.add_argument("--half", action="store_true")
    args = p.parse_args()

    exp_dir = vendored_exp_dir(args.exp_name)
    exp_rel = f"logs/{args.exp_name}"
    upstream_dir = vendored_workspace() / "infer" / "modules" / "train" / "extract"

    with chdir(vendored_workspace()):
        if args.f0_method != "rmvpe_gpu":
            script = upstream_dir / "extract_f0_print.py"
            sys.argv = [str(script), exp_rel, str(args.n_procs), args.f0_method]
        else:
            script = upstream_dir / "extract_f0_rmvpe.py"
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

    copy_artifacts(exp_dir, Path(args.logs_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
