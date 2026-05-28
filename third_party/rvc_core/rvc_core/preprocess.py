"""CLI wrapper around upstream `infer/modules/train/preprocess.py`."""

from __future__ import annotations

import argparse
import multiprocessing
import runpy
import sys
from pathlib import Path

import rvc_core  # noqa: F401
from rvc_core._workspace import chdir, copy_artifacts, vendored_exp_dir, vendored_workspace

_SR_MAP = {"32k": 32000, "40k": 40000, "48k": 48000}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--exp-name", required=True)
    p.add_argument("--sr", default="40k", choices=["32k", "40k", "48k"])
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--logs-dir", required=True)
    p.add_argument("--n-procs", type=int, default=max(1, multiprocessing.cpu_count() // 2))
    p.add_argument("--per", type=float, default=3.7)
    p.add_argument("--no-parallel", action="store_true")
    args = p.parse_args()

    exp_dir = vendored_exp_dir(args.exp_name)
    sr_hz = _SR_MAP[args.sr]
    upstream_script = vendored_workspace() / "infer" / "modules" / "train" / "preprocess.py"

    with chdir(vendored_workspace()):
        sys.argv = [
            str(upstream_script),
            str(Path(args.dataset_dir).resolve()),
            str(sr_hz),
            str(args.n_procs),
            f"logs/{args.exp_name}",
            "True" if args.no_parallel else "False",
            f"{args.per:.1f}",
        ]
        print(f"[rvc_core.preprocess] argv={sys.argv[1:]}", flush=True)
        runpy.run_path(str(upstream_script), run_name="__main__")

    copy_artifacts(exp_dir, Path(args.logs_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
