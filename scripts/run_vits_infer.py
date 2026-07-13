#!/usr/bin/env python
"""CLI: VITS inference (Coqui or HuggingFace-Transformers) + optional metrics.

The model format is auto-detected from config.json: a HF-Transformers VITS
(`architectures: [VitsModel]`) routes to the transformers path, anything else
to the Coqui path. This lets the same TTS form serve both a Coqui-trained
model and a fine-tuned facebook/mms-tts-rus derivative.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--generator", required=True,
                   help="Coqui: path to best_model.pth. HF: ignored (model dir "
                        "is inferred from --config's parent).")
    p.add_argument("--config", required=True, help="config.json of the model.")
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
    config_path = Path(args.config)

    try:
        from app.core.tts.hf_vits import is_hf_vits_config
        if is_hf_vits_config(config_path):
            # HuggingFace Transformers VITS — the whole model lives in the
            # folder containing config.json.
            from app.core.tts.hf_vits import HfVitsInferConfig, infer as hf_infer
            model_dir = config_path.parent
            out_path = hf_infer(HfVitsInferConfig(
                model_dir=model_dir, text=text,
                length_scale=args.length_scale,
                output_format=args.format, model_name=args.model_name,
            ))
            # reference_speaker.wav lives in the run/ folder; when inferring
            # from a checkpoint subfolder, look one level up too.
            reference_dir = model_dir
            if not (model_dir / "reference_speaker.wav").exists() \
                    and (model_dir.parent / "reference_speaker.wav").exists():
                reference_dir = model_dir.parent
        else:
            from app.core.tts.inferencer import TtsInferConfig, infer as coqui_infer
            out_path = coqui_infer(TtsInferConfig(
                generator_pth=Path(args.generator), config_json=config_path,
                text=text, length_scale=args.length_scale,
                pitch_shift_semitones=args.pitch_shift,
                output_format=args.format, model_name=args.model_name,
                language=args.language,
            ))
            reference_dir = Path(args.generator).parent

        result = {"output": str(out_path)}
        if args.compute_metrics:
            from app.core.metrics.tts_eval import compute_tts_metrics
            result.update(compute_tts_metrics(text, out_path, reference_dir, args.language))

        print(f"RESULT_JSON={json.dumps(result, ensure_ascii=False)}", flush=True)
        return 0
    except Exception as e:
        import traceback
        print(f"ERROR: {e}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
