#!/usr/bin/env python
"""Convert an LJSpeech-style manifest + wav folder into a HuggingFace
`audiofolder` dataset directory for finetune-hf-vits.

Output layout (what `datasets.load_dataset("audiofolder", data_dir=OUT)` reads):

    OUT/
      metadata.csv        # columns: file_name,text
      <stem>.wav
      ...

The `file_name` column must be relative to OUT, so we copy (or hardlink) the
referenced wavs next to metadata.csv.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import _bootstrap  # noqa: F401


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="stem|text|text CSV")
    p.add_argument("--audio-dir", required=True, help="folder with the .wav files")
    p.add_argument("--output", required=True, help="dataset dir to create")
    p.add_argument("--link", action="store_true",
                   help="hardlink wavs instead of copying (saves disk).")
    args = p.parse_args()

    manifest = Path(args.manifest)
    audio_dir = Path(args.audio_dir)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if not manifest.exists():
        print(f"ERROR: manifest not found: {manifest}", file=sys.stderr)
        return 1

    rows = [ln.strip() for ln in manifest.read_text(encoding="utf-8").splitlines() if ln.strip()]
    written = 0
    missing = 0
    with (out / "metadata.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "text"])
        for ln in rows:
            parts = ln.split("|")
            if len(parts) < 2:
                continue
            stem, text = parts[0], parts[1]
            src = audio_dir / f"{stem}.wav"
            if not src.exists():
                src = audio_dir / stem  # already had an extension?
            if not src.exists():
                missing += 1
                continue
            dst = out / f"{stem}.wav"
            if not dst.exists():
                if args.link:
                    try:
                        import os
                        os.link(src, dst)
                    except OSError:
                        shutil.copy2(src, dst)
                else:
                    shutil.copy2(src, dst)
            writer.writerow([f"{stem}.wav", text])
            written += 1

    print(f"OK: wrote {written} rows to {out/'metadata.csv'}", flush=True)
    if missing:
        print(f"WARN: {missing} referenced wavs not found in {audio_dir}", flush=True)
    if written == 0:
        print("ERROR: no rows written — check manifest/audio paths.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
