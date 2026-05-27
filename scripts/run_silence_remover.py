#!/usr/bin/env python
"""CLI: silence removal."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from app.core.audio.silence_remover import remove_silence


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--format", default="wav", choices=["wav", "mp3"])
    p.add_argument("--max-silence", type=float, required=True)
    p.add_argument("--threshold-db", type=float, default=-40.0)
    args = p.parse_args()
    try:
        out = remove_silence(Path(args.input), args.max_silence, args.threshold_db, args.format)
        print(f"output: {out}", flush=True)
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
