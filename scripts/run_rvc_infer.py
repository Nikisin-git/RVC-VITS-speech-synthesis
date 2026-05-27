#!/usr/bin/env python
"""CLI: RVC inference + postprocess + metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from app.core.postprocess.pedalboard_chain import (
    CompressorConfig, FiltersConfig, NoiseGateConfig, PostProcessChain, ReverbConfig,
)
from app.core.rvc.inferencer import RvcInferConfig, infer


def _load_postprocess(path: Path | None) -> PostProcessChain:
    if not path or not path.exists():
        return PostProcessChain()
    data = json.loads(path.read_text(encoding="utf-8"))
    return PostProcessChain(
        reverb=ReverbConfig(**data.get("reverb", {})),
        compressor=CompressorConfig(**data.get("compressor", {})),
        filters=FiltersConfig(**data.get("filters", {})),
        gate=NoiseGateConfig(**data.get("gate", {})),
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pth", required=True)
    p.add_argument("--index", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--pitch", type=int, default=0)
    p.add_argument("--index-rate", type=float, default=0.5)
    p.add_argument("--filter-radius", type=int, default=3)
    p.add_argument("--rms-mix-rate", type=float, default=0.25)
    p.add_argument("--protect", type=float, default=0.33)
    p.add_argument("--format", default="wav", choices=["wav", "mp3"])
    p.add_argument("--model-name", default="model")
    p.add_argument("--postprocess-json")
    p.add_argument("--compute-metrics", action="store_true")
    p.add_argument("--output")
    args = p.parse_args()

    post = _load_postprocess(Path(args.postprocess_json)) if args.postprocess_json else PostProcessChain()
    cfg = RvcInferConfig(
        pth=Path(args.pth), index=Path(args.index), input_audio=Path(args.input),
        pitch=args.pitch, index_rate=args.index_rate, filter_radius=args.filter_radius,
        rms_mix_rate=args.rms_mix_rate, protect=args.protect,
        output_format=args.format, model_name=args.model_name,
        postprocess=post, enable_postprocess=True,
    )

    try:
        out_path = infer(cfg)
        result = {"output": str(out_path)}

        if args.compute_metrics:
            try:
                from app.core.metrics.wer import wer_from_audio
                from app.core.metrics.secs import compute_secs
                wer = wer_from_audio(Path(args.input), out_path, language="ru").get("wer")
                secs = compute_secs(Path(args.input), out_path)
                result.update({"wer": wer, "secs": secs})
            except Exception as me:
                print(f"WARN: metrics failed: {me}", flush=True)

        print(f"RESULT_JSON={json.dumps(result, ensure_ascii=False)}", flush=True)
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
