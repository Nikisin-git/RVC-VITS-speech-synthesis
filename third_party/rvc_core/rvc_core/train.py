"""CLI wrapper around upstream `infer/modules/train/train.py`."""

from __future__ import annotations

import argparse
import os
import random
import runpy
import shutil
import sys
from pathlib import Path

import rvc_core  # noqa: F401
from rvc_core import _paths
from rvc_core._workspace import chdir, copy_artifacts, vendored_exp_dir, vendored_workspace


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
                f"{gt}/{name}.wav|{feat}/{name}.npy|{f0_dir}/{name}.wav.npy|"
                f"{f0nsf_dir}/{name}.wav.npy|{spk_id}"
            )
        else:
            rows.append(f"{gt}/{name}.wav|{feat}/{name}.npy|{spk_id}")

    mute_root = vendored_workspace() / "logs" / "mute"
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

    exp_dir = vendored_exp_dir(args.exp_name)
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

    script = vendored_workspace() / "infer" / "modules" / "train" / "train.py"
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

    with chdir(vendored_workspace()):
        sys.argv = argv
        print(f"[rvc_core.train] argv={argv[1:]}", flush=True)
        runpy.run_path(str(script), run_name="__main__")

    if args.weights_dir:
        produced = vendored_workspace() / "assets" / "weights"
        target = Path(args.weights_dir)
        target.mkdir(parents=True, exist_ok=True)
        if produced.exists():
            for f in produced.glob(f"{args.exp_name}*.pth"):
                shutil.copy2(f, target / f.name)
            print(f"[rvc_core.train] copied weights to {target}", flush=True)

    copy_artifacts(exp_dir, Path(args.logs_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
