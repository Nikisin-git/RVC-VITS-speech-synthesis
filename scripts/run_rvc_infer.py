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
        result: dict = {"output": str(out_path)}

        if args.compute_metrics:
            # Compute WER and SECS independently — a failure in one (e.g. Whisper
            # model download timing out) shouldn't blank out the other.
            print("Считаются метрики качества...", flush=True)
            try:
                from app.core.metrics.wer import wer_from_audio
                # language=None → Whisper auto-detects (so English inputs don't
                # get force-transcribed as Russian and score WER ~1.0).
                wer_info = wer_from_audio(Path(args.input), out_path, language=None)
                result["wer"] = wer_info.get("wer")
                ref_text = wer_info.get("reference_text", "")
                hyp_text = wer_info.get("hypothesis_text", "")
                print(f"WER: {result['wer']:.3f}", flush=True)
                # Whisper hallucinates on sung audio, so dump transcripts to help
                # the user judge whether a high WER reflects bad RVC quality or
                # just unreliable transcription on music.
                print(f"  reference (вход): {ref_text[:200]}", flush=True)
                print(f"  hypothesis (выход): {hyp_text[:200]}", flush=True)
            except Exception as we:
                import traceback
                print(f"WARN: WER failed: {type(we).__name__}: {we}", flush=True)
                traceback.print_exc()
                result["wer_error"] = f"{type(we).__name__}: {we}"

            try:
                from app.core.metrics.secs import compute_secs
                result["secs"] = compute_secs(Path(args.input), out_path)
                print(f"SECS: {result['secs']:.3f}", flush=True)
            except Exception as se:
                import traceback
                print(f"WARN: SECS failed: {type(se).__name__}: {se}", flush=True)
                traceback.print_exc()
                result["secs_error"] = f"{type(se).__name__}: {se}"

        print(f"RESULT_JSON={json.dumps(result, ensure_ascii=False)}", flush=True)
        return 0
    except Exception as e:
        import traceback
        print(f"ERROR: {e}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
