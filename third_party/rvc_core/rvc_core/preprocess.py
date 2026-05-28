"""CLI wrapper around upstream `infer/modules/train/preprocess.py`."""

from __future__ import annotations

import argparse
import multiprocessing
import runpy
import shutil
import sys
from pathlib import Path

import rvc_core  # noqa: F401
from rvc_core._workspace import chdir, copy_artifacts, vendored_exp_dir, vendored_workspace

_SR_MAP = {"32k": 32000, "40k": 40000, "48k": 48000}


def _reset_if_sr_changed(exp_dir: Path, sr: str) -> None:
    """Wipe the experiment directory if the previous run used a different sr.

    Without this, switching from --sr 32k to --sr 40k inside the same
    experiment leaves 32k wavs/features/config behind and the 40k pretrained
    weights fail to load with 'size mismatch' errors.
    """
    stamp = exp_dir / "_sr.stamp"
    prev = stamp.read_text(encoding="utf-8").strip() if stamp.exists() else None
    if prev and prev != sr:
        print(f"[rvc_core.preprocess] sr changed ({prev} -> {sr}), clearing {exp_dir}", flush=True)
        for child in exp_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
    stamp.write_text(sr, encoding="utf-8")


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
    _reset_if_sr_changed(exp_dir, args.sr)
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
