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
                print(f"WER: {wer:.3f}", flush=True)
            except Exception as me:
                print(f"WARN: WER failed: {me}", flush=True)
                result["wer_error"] = str(me)

            # SECS for TTS only makes sense against a sample of the target
            # speaker; without one we'd be comparing to the user's prompt
            # text (no audio embedding) or to the model's own output (~1.0
            # by definition). The TTS trainer saves reference_speaker.wav
            # next to best_model.pth; use it if present.
            try:
                from app.core.metrics.secs import compute_secs
                ref_path = Path(args.generator).parent / "reference_speaker.wav"
                if ref_path.exists():
                    result["secs"] = compute_secs(ref_path, out_path)
                    result["secs_reference"] = str(ref_path)
                    print(f"SECS: {result['secs']:.3f} (reference: {ref_path.name})", flush=True)
                else:
                    print(
                        "SECS: пропущен (reference_speaker.wav не найден рядом с моделью; "
                        "переобучите модель, чтобы получить эталонный сэмпл говорящего).",
                        flush=True,
                    )
                    result["secs_error"] = "reference_speaker.wav not found"
            except Exception as se:
                print(f"WARN: SECS failed: {type(se).__name__}: {se}", flush=True)
                result["secs_error"] = f"{type(se).__name__}: {se}"

            # MCD: spectral distance vs the target-speaker reference. Same
            # reference file as SECS; skip with a clear message if absent.
            try:
                from app.core.metrics.mcd import compute_mcd
                ref_path = Path(args.generator).parent / "reference_speaker.wav"
                if ref_path.exists():
                    result["mcd"] = compute_mcd(ref_path, out_path)
                    result["mcd_reference"] = str(ref_path)
                    print(f"MCD: {result['mcd']:.2f} dB (reference: {ref_path.name})", flush=True)
                else:
                    print(
                        "MCD: пропущен (reference_speaker.wav не найден рядом с моделью).",
                        flush=True,
                    )
                    result["mcd_error"] = "reference_speaker.wav not found"
            except Exception as me:
                print(f"WARN: MCD failed: {type(me).__name__}: {me}", flush=True)
                result["mcd_error"] = f"{type(me).__name__}: {me}"
        print(f"RESULT_JSON={json.dumps(result, ensure_ascii=False)}", flush=True)
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
