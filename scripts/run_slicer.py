#!/usr/bin/env python
"""CLI: audio slicing (timer or VAD)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from app.core.audio.slicer import SlicerConfig, SliceMode, TailMode, slice_file


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", action="append", required=True,
                   help="Input file. Repeat to pass multiple (concatenated in --single-track mode).")
    p.add_argument("--format", default="wav", choices=["wav", "mp3"])
    p.add_argument("--mode", choices=["timer", "vad"], default="timer")
    p.add_argument("--length", type=int, default=10)
    p.add_argument("--tail-mode", choices=["keep", "merge", "drop"], default="keep")
    p.add_argument("--single-track", action="store_true")
    args = p.parse_args()
    cfg = SlicerConfig(
        mode=SliceMode(args.mode),
        target_seconds=args.length,
        tail_mode=TailMode(args.tail_mode),
        output_format=args.format,
        single_track=args.single_track,
    )
    try:
        outs = slice_file([Path(p) for p in args.input], cfg)
        for o in outs:
            print(f"output: {o}", flush=True)
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
