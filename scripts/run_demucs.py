#!/usr/bin/env python
"""CLI: Demucs htdemucs_ft separation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from app.core.audio.vocal_separator import separate


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--format", default="wav", choices=["wav", "mp3"])
    args = p.parse_args()
    try:
        vocals, music = separate(Path(args.input), args.format)
        print(f"vocals: {vocals}", flush=True)
        print(f"music:  {music}", flush=True)
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
