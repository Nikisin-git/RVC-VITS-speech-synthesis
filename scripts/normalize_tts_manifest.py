#!/usr/bin/env python
"""Convert a hand-made or so-vits-svc style manifest into the LJSpeech format
that Coqui-TTS expects.

Input rows (any of these work):
    <absolute_or_relative_path>.wav|<text>|<speaker_id>
    <path>.wav|<text>
    <stem>|<text>

Output rows (always 3 columns, what Coqui ljspeech formatter reads):
    <stem>|<text>|<normalized_text>

Where `stem` is the file name without `.wav`, and `normalized_text` is by
default a copy of `text` (optionally lowercased).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to existing manifest CSV")
    p.add_argument("--output", required=True, help="Where to write the cleaned metadata.csv")
    p.add_argument("--audio-dir", required=True,
                   help="Folder where the .wav files actually live (will be checked).")
    p.add_argument("--lowercase", action="store_true",
                   help="Lowercase the text (recommended — Cyrillic uppercase used as "
                        "stress markers confuses espeak).")
    p.add_argument("--strict", action="store_true",
                   help="Fail if any row references a missing file. Default: skip and warn.")
    args = p.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    audio_dir = Path(args.audio_dir)

    if not src.exists():
        print(f"ERROR: input not found: {src}", file=sys.stderr)
        return 1
    if not audio_dir.is_dir():
        print(f"ERROR: audio dir not found: {audio_dir}", file=sys.stderr)
        return 1

    lines = [ln.strip() for ln in src.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        print(f"ERROR: input is empty: {src}", file=sys.stderr)
        return 1

    out_rows: list[str] = []
    missing: list[str] = []
    skipped_bad: list[str] = []

    for ln in lines:
        cols = ln.split("|")
        if len(cols) < 2:
            skipped_bad.append(ln[:80])
            continue
        raw_path, text = cols[0], cols[1]
        # Strip any directory + extension to leave just the stem
        stem = Path(raw_path).stem
        if not stem:
            skipped_bad.append(ln[:80])
            continue
        wav = audio_dir / f"{stem}.wav"
        if not wav.exists():
            missing.append(stem)
            if args.strict:
                continue
            # In non-strict mode we still keep the row — user might rename later.

        text = text.strip()
        if args.lowercase:
            text = text.lower()
        if not text:
            skipped_bad.append(ln[:80])
            continue

        out_rows.append(f"{stem}|{text}|{text}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(out_rows) + "\n", encoding="utf-8")

    print(f"OK: wrote {len(out_rows)} rows to {dst}", flush=True)
    if missing:
        print(f"WARN: {len(missing)} files not found in {audio_dir} "
              f"(first 5: {missing[:5]})", flush=True)
    if skipped_bad:
        print(f"WARN: skipped {len(skipped_bad)} malformed rows "
              f"(first 3: {skipped_bad[:3]})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
