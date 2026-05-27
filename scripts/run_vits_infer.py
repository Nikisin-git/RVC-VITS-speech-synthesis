#!/usr/bin/env python
"""CLI: VITS inference + optional WER metric."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from app.core.tts.inferencer import TtsInferConfig, infer


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--generator", required=True)
    p.add_argument("--config", required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text")
    src.add_argument("--text-file")
    p.add_argument("--length-scale", type=float, default=1.0)
    p.add_argument("--pitch-shift", type=int, default=0)
    p.add_argument("--format", default="wav", choices=["wav", "mp3"])
    p.add_argument("--model-name", default="model")
    p.add_argument("--language", default="ru")
    p.add_argument("--compute-metrics", action="store_true")
    args = p.parse_args()

    text = args.text if args.text else Path(args.text_file).read_text(encoding="utf-8")

    cfg = TtsInferConfig(
        generator_pth=Path(args.generator),
        config_json=Path(args.config),
        text=text,
        length_scale=args.length_scale,
        pitch_shift_semitones=args.pitch_shift,
        output_format=args.format,
        model_name=args.model_name,
        language=args.language,
    )
    try:
        out_path = infer(cfg)
        result = {"output": str(out_path)}
        if args.compute_metrics:
            try:
                from app.core.metrics.wer import wer_from_text
                wer = wer_from_text(text, out_path, language=args.language).get("wer")
                result["wer"] = wer
            except Exception as me:
                print(f"WARN: WER failed: {me}", flush=True)
        print(f"RESULT_JSON={json.dumps(result, ensure_ascii=False)}", flush=True)
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
