#!/usr/bin/env python
"""CLI: filter speech fragments (drop filler-only / empty / repetitive / noisy).

Pass fragments with repeated --input, or a manifest via --input-list (preferred
for large folders — avoids the Windows command-line length limit). Kept files
are copied to --out-dir; a filter_report.csv is written there too.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from app.config import FILTERED_DIR
from app.core.audio.segment_filter import FilterOptions, filter_fragments


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", action="append", default=[], help="Fragment file (repeatable).")
    p.add_argument("--input-list", help="Text file: one fragment path per line (UTF-8).")
    p.add_argument("--out-dir", default=str(FILTERED_DIR))
    p.add_argument("--language", default="ru")
    p.add_argument("--model", default=None, help="Whisper model name (default: shared).")
    # which rules are active (default: all on)
    p.add_argument("--keep-empty", action="store_true", help="не удалять пустые/без речи")
    p.add_argument("--keep-filler-only", action="store_true", help="не удалять только-паразиты")
    p.add_argument("--keep-repetition", action="store_true", help="не удалять повторы/заикания")
    p.add_argument("--keep-low-conf", action="store_true", help="не удалять с низкой уверенностью")
    # thresholds
    p.add_argument("--no-speech-thr", type=float, default=0.6)
    p.add_argument("--logprob-thr", type=float, default=-1.0)
    p.add_argument("--comp-thr", type=float, default=2.4)
    args = p.parse_args()

    inputs: list[str] = list(args.input or [])
    if args.input_list:
        for line in Path(args.input_list).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                inputs.append(line)
    if not inputs:
        print("ERROR: нет входных файлов (--input или --input-list).", file=sys.stderr, flush=True)
        return 1

    opts = FilterOptions(
        drop_empty=not args.keep_empty,
        drop_filler_only=not args.keep_filler_only,
        drop_repetition=not args.keep_repetition,
        drop_low_conf=not args.keep_low_conf,
        no_speech_thr=args.no_speech_thr,
        logprob_thr=args.logprob_thr,
        comp_ratio_thr=args.comp_thr,
    )
    try:
        filter_fragments([Path(x) for x in inputs], Path(args.out_dir),
                         language=args.language, model_name=args.model, opts=opts)
        return 0
    except Exception as e:
        import traceback
        print(f"ERROR: {e}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
