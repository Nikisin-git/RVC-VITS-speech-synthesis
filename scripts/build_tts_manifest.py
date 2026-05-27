#!/usr/bin/env python
"""Build a draft LJSpeech-style manifest from a folder of .wav files using Whisper.

Output rows: `<filename>|<auto-transcription>`. Human review is expected afterwards.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--audio-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--language", default="ru")
    p.add_argument("--model", default="small")
    args = p.parse_args()

    audio_dir = Path(args.audio_dir)
    out_path = Path(args.output)

    from app.core.metrics.wer import transcribe

    wavs = sorted(audio_dir.glob("*.wav"))
    if not wavs:
        print(f"No .wav files in {audio_dir}", file=sys.stderr)
        return 1

    lines: list[str] = []
    for i, wav in enumerate(wavs, start=1):
        print(f"[{i}/{len(wavs)}] {wav.name}", flush=True)
        try:
            text = transcribe(wav, language=args.language, model_name=args.model)
        except Exception as e:
            print(f"  WARN: {e}", flush=True)
            text = ""
        lines.append(f"{wav.name}|{text}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} ({len(lines)} rows)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
