"""CLI wrapper around upstream `infer/modules/train/train.py`.

Responsibilities:
- Build `<logs>/filelist.txt` from `0_gt_wavs`, `3_feature768`, `2a_f0`, `2b-f0nsf`.
- Copy the right config (`configs/v2/<sr>.json`) into `<logs>/config.json`.
- Build the upstream argv (`-e/-sr/-f0/-bs/-te/-se/-pg/-pd/-l/-c/-sw/-v`).
- chdir into the workspace and runpy the upstream training entrypoint.
- Honour a `--cancel-flag` file for soft cancellation by checking it between
  epochs (best-effort: a hook patches `torch.save` to short-circuit when the
  flag appears).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import runpy
import shutil
import sys
from pathlib import Path

import rvc_core  # noqa: F401
from rvc_core import _paths
from rvc_core._workspace import chdir


def _write_filelist(exp_dir: Path, sr: str, version: str, has_f0: bool, spk_id: int = 0) -> None:
    gt = exp_dir / "0_gt_wavs"
    feat = exp_dir / ("3_feature256" if version == "v1" else "3_feature768")
    fea_dim = 256 if version == "v1" else 768

    if has_f0:
        f0_dir = exp_dir / "2a_f0"
        f0nsf_dir = exp_dir / "2b-f0nsf"
        names = (
            {p.stem for p in gt.glob("*.wav")}
            & {p.stem for p in feat.glob("*.npy")}
            & {p.stem for p in f0_dir.iterdir()}
            & {p.stem for p in f0nsf_dir.iterdir()}
        )
    else:
        names = {p.stem for p in gt.glob("*.wav")} & {p.stem for p in feat.glob("*.npy")}

    rows = []
    for name in names:
        if has_f0:
            rows.append(
                f"{gt}/{name}.wav|{feat}/{name}.npy|{exp_dir}/2a_f0/{name}.wav.npy|"
                f"{exp_dir}/2b-f0nsf/{name}.wav.npy|{spk_id}"
            )
        else:
            rows.append(f"{gt}/{name}.wav|{feat}/{name}.npy|{spk_id}")

    # Upstream appends two "mute" placeholder lines from logs/mute/...; the
    # vendored tree ships them under _vendored/logs/mute. Use absolute paths.
    mute_root = rvc_core.VENDORED_PATH / "logs" / "mute"
    if mute_root.exists():
        for _ in range(2):
            if has_f0:
                rows.append(
                    f"{mute_root}/0_gt_wavs/mute{sr}.wav|"
                    f"{mute_root}/3_feature{fea_dim}/mute.npy|"
                    f"{mute_root}/2a_f0/mute.wav.npy|"
                    f"{mute_root}/2b-f0nsf/mute.wav.npy|{spk_id}"
                )
            else:
                rows.append(
                    f"{mute_root}/0_gt_wavs/mute{sr}.wav|"
                    f"{mute_root}/3_feature{fea_dim}/mute.npy|{spk_id}"
                )

    random.shuffle(rows)
    (exp_dir / "filelist.txt").write_text("\n".join(rows), encoding="utf-8")
    print(f"[rvc_core.train] filelist: {len(rows)} rows", flush=True)


def _write_config(exp_dir: Path, sr: str, version: str) -> None:
    src = _paths.vendored_config(sr, version)
    dst = exp_dir / "config.json"
    if dst.exists():
        return
    shutil.copy2(src, dst)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--exp-name", required=True)
    p.add_argument("--sr", default="40k", choices=["32k", "40k", "48k"])
    p.add_argument("--dataset-dir", required=False)
    p.add_argument("--logs-dir", required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--weights-dir", required=False)
    p.add_argument("--version", default="v2", choices=["v1", "v2"])
    p.add_argument("--gpu-index", default="0")
    p.add_argument("--has-f0", type=int, default=1, choices=[0, 1])
    p.add_argument("--save-latest", type=int, default=1, choices=[0, 1])
    p.add_argument("--cache-gpu", type=int, default=0, choices=[0, 1])
    p.add_argument("--save-every-weights", type=int, default=1, choices=[0, 1])
    p.add_argument("--cancel-flag")
    args = p.parse_args()

    logs_dir = Path(args.logs_dir).resolve()
    workspace = logs_dir.parent.parent
    exp_dir = logs_dir
    _write_filelist(exp_dir, args.sr, args.version, bool(args.has_f0))
    _write_config(exp_dir, args.sr, args.version)

    pg = _paths.pretrained_v2(args.sr, "G")
    pd = _paths.pretrained_v2(args.sr, "D")
    if not pg.exists() or not pd.exists():
        print(
            f"WARN: pretrained weights missing: {pg} / {pd}. "
            "Training will run from scratch (very slow). "
            "See DEPLOYMENT.md, section 'Pretrained RVC weights'.",
            flush=True,
        )
        pg_arg = ""
        pd_arg = ""
    else:
        pg_arg = str(pg)
        pd_arg = str(pd)

    script = rvc_core.VENDORED_PATH / "infer" / "modules" / "train" / "train.py"
    argv = [str(script),
            "-e", args.exp_name,
            "-sr", args.sr,
            "-f0", str(args.has_f0),
            "-bs", str(args.batch_size),
            "-g", str(args.gpu_index),
            "-te", str(args.epochs),
            "-se", str(args.save_every),
            "-l", str(args.save_latest),
            "-c", str(args.cache_gpu),
            "-sw", str(args.save_every_weights),
            "-v", args.version]
    if pg_arg:
        argv += ["-pg", pg_arg]
    if pd_arg:
        argv += ["-pd", pd_arg]

    if args.cancel_flag:
        os.environ["RVC_CANCEL_FLAG"] = str(Path(args.cancel_flag).resolve())

    with chdir(workspace):
        sys.argv = argv
        print(f"[rvc_core.train] argv={argv[1:]}", flush=True)
        runpy.run_path(str(script), run_name="__main__")

    # If weights-dir specified, copy any produced *.pth from workspace's
    # assets/weights into the caller's target.
    if args.weights_dir:
        produced = workspace / "assets" / "weights"
        target = Path(args.weights_dir)
        target.mkdir(parents=True, exist_ok=True)
        for f in produced.glob(f"{args.exp_name}*.pth"):
            shutil.copy2(f, target / f.name)
        print(f"[rvc_core.train] copied weights to {target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
