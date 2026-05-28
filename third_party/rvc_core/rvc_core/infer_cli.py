"""CLI wrapper for single-file RVC inference via upstream `VC` class.

Output is written as a WAV at `--output` (16-bit PCM). The caller can convert
to MP3 separately if needed.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import rvc_core  # noqa: F401  side-effect: sys.path setup
from rvc_core import _paths


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pth", required=True, help="Generator .pth checkpoint")
    p.add_argument("--index", required=True, help="FAISS .index file (or '' to disable)")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--pitch", type=int, default=0)
    p.add_argument("--index-rate", type=float, default=0.5)
    p.add_argument("--filter-radius", type=int, default=3)
    p.add_argument("--rms-mix-rate", type=float, default=0.25)
    p.add_argument("--protect", type=float, default=0.33)
    p.add_argument("--f0-method", default="rmvpe", choices=["pm", "harvest", "rmvpe", "crepe"])
    p.add_argument("--resample-sr", type=int, default=0)
    p.add_argument("--speaker-id", type=int, default=0)
    args = p.parse_args()

    # Upstream's Config and VC class look up assets via paths relative to cwd.
    # chdir into the vendored tree where assets/, configs/, i18n/ all live.
    workspace = _paths.VENDORED
    os.chdir(str(workspace))

    os.environ.setdefault("rmvpe_root", str(workspace / "assets" / "rmvpe"))
    os.environ.setdefault("weight_root", str(Path(args.pth).parent))
    os.environ.setdefault("index_root", str(Path(args.index).parent) if args.index else "")
    os.environ.setdefault("hubert_path", str(_paths.hubert_path()))

    from configs.config import Config  # type: ignore
    from infer.modules.vc.modules import VC  # type: ignore
    import soundfile as sf

    config = Config()
    vc = VC(config)

    weight_name = Path(args.pth).name
    print(f"[rvc_core.infer_cli] loading {weight_name}", flush=True)
    vc.get_vc(weight_name)

    msg, (sr, audio) = vc.vc_single(
        args.speaker_id,
        args.input,
        args.pitch,
        None,                 # f0_file
        args.f0_method,
        args.index,
        "",                   # file_index2
        args.index_rate,
        args.filter_radius,
        args.resample_sr,
        args.rms_mix_rate,
        args.protect,
    )
    if audio is None:
        print(f"ERROR: {msg}", file=sys.stderr, flush=True)
        return 1

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), audio, sr, subtype="PCM_16")
    print(f"[rvc_core.infer_cli] wrote {out} ({sr} Hz)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
