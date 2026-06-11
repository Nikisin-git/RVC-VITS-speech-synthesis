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

    def _stems(d: Path, suffix: str) -> dict[str, str]:
        """Map base stem -> actual filename, stripping common upstream suffixes."""
        out = {}
        if not d.exists():
            return out
        for p in d.iterdir():
            if not p.name.lower().endswith(suffix):
                continue
            stem = p.name
            for s in (".wav.npy", ".wav", ".npy"):
                if stem.endswith(s):
                    stem = stem[: -len(s)]
                    break
            out[stem] = p.name
        return out

    gt_map = _stems(gt, ".wav")
    feat_map = _stems(feat, ".npy")
    print(f"[rvc_core.train] dirs: gt={len(gt_map)} feat={len(feat_map)}", flush=True)

    if has_f0:
        f0_dir = exp_dir / "2a_f0"
        f0nsf_dir = exp_dir / "2b-f0nsf"
        f0_map = _stems(f0_dir, ".npy")
        f0nsf_map = _stems(f0nsf_dir, ".npy")
        print(f"[rvc_core.train] dirs: f0={len(f0_map)} f0nsf={len(f0nsf_map)}", flush=True)
        names = set(gt_map) & set(feat_map) & set(f0_map) & set(f0nsf_map)
    else:
        names = set(gt_map) & set(feat_map)

    rows = []
    for name in names:
        if has_f0:
            rows.append(
                f"{gt}/{gt_map[name]}|{feat}/{feat_map[name]}|"
                f"{f0_dir}/{f0_map[name]}|{f0nsf_dir}/{f0nsf_map[name]}|{spk_id}"
            )
        else:
            rows.append(f"{gt}/{gt_map[name]}|{feat}/{feat_map[name]}|{spk_id}")

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
    """Always overwrite config.json so a switched sr/version doesn't leak from
    a previous run inside the same experiment directory. Also adjust a few
    knobs that the vendor defaults set too aggressively for small / mid
    datasets running on a single consumer GPU:

    * log_interval -> 25 (vendor default 200; smaller so progress shows up
      every few seconds on a 600-700 sample dataset).
    * fp16_run is forced OFF when the active CUDA device is from the
      GTX 16xx / GTX 15xx / GTX 10xx families. These cards report compute
      capability 6.x or 7.5 but have no Tensor Cores, so fp16 is software-
      emulated and runs noticeably slower than fp32 on real workloads.
    """
    import json as _json
    src = _paths.vendored_config(sr, version)
    dst = exp_dir / "config.json"
    shutil.copy2(src, dst)
    try:
        data = _json.loads(dst.read_text(encoding="utf-8"))
        train_cfg = data.setdefault("train", {})
        train_cfg["log_interval"] = 25
        if _gpu_lacks_tensor_cores():
            train_cfg["fp16_run"] = False
            print("[rvc_core] GPU без Tensor Cores → forced fp16_run=False", flush=True)
        dst.write_text(_json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


def _gpu_lacks_tensor_cores() -> bool:
    """Return True for Nvidia consumer cards that report a CUDA cap >= 7.0
    but have no Tensor Cores (Turing GTX 16xx) or are below 7.0 (Pascal).
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        name = torch.cuda.get_device_name(0).lower()
        bad_substrings = ("gtx 16", "gtx 15", "gtx 10", "gtx titan", "mx ")
        return any(s in name for s in bad_substrings)
    except Exception:
        return False


def _enable_lightweight_logging() -> None:
    """Skip matplotlib mel-spectrogram rendering in the vendored Trainer.
    On Windows with OneDrive watching the output folder, those three
    figures per log step (rendered via plot_spectrogram_to_numpy) are by
    far the most expensive part of summarize() — 0.5-1.0 s each. Replace
    the plot function with a tiny np.zeros so summarize() returns instantly
    and the optimizer can keep stepping.
    """
    if os.environ.get("VOICEGEN_RVC_HEAVY_LOG") == "1":
        return
    try:
        import numpy as _np
        from infer.lib.train import utils as _utils
        _utils.plot_spectrogram_to_numpy = lambda *a, **kw: _np.zeros(
            (8, 8, 4), dtype=_np.uint8
        )
        print("[rvc_core] mel-spectrogram TB plotting → no-op (faster training)", flush=True)
    except Exception:
        # If the patch fails, fall back to the original (slow) behaviour.
        pass


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
    # Caching the dataset in GPU memory removes the disk-read bottleneck
    # entirely; safe to default ON for the small datasets typical of RVC
    # fine-tuning (<2000 fragments fit in 6 GB VRAM with batch_size <= 8).
    p.add_argument("--cache-gpu", type=int, default=1, choices=[0, 1])
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

    # Windows PyTorch wheels ship without libuv support; force the legacy TCP
    # rendezvous transport so torch.distributed.init_process_group works.
    # See pytorch/pytorch#129070.
    os.environ.setdefault("USE_LIBUV", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    # Upstream's process_ckpt.savee writes inference-ready weights via the
    # relative path "assets/weights/<exp>.pth"; that resolves against cwd
    # (= _vendored after chdir). The directory has to exist before training
    # starts or every save crashes with 'Parent directory assets/weights
    # does not exist'.
    (vendored_workspace() / "assets" / "weights").mkdir(parents=True, exist_ok=True)

    # Tell matplotlib to use a headless backend before anything inside the
    # vendored training imports it. Saves the ~2s backend probe on Windows.
    os.environ.setdefault("MPLBACKEND", "Agg")

    with chdir(vendored_workspace()):
        sys.argv = argv
        _enable_lightweight_logging()
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
