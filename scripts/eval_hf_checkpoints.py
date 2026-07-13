#!/usr/bin/env python
"""Audition every HF-VITS checkpoint in a run and tabulate quality metrics.

For each run/checkpoint-<step>/ folder this:
  1. converts it to an inference-ready model (export_hf_checkpoint.py —
     rebuilds config, collapses weight-norm),
  2. synthesizes one test phrase and computes WER (+ SECS/MCD if a speaker
     reference is available) via run_vits_infer.py,
  3. records the numbers.

Finally it writes a CSV and prints a table sorted by WER, so you can pick the
epoch that actually sounds best instead of assuming "more epochs = better".

Usage:
    python scripts/eval_hf_checkpoints.py --run <run dir> --text "Съешь ещё этих мягких булок" \
        [--reference <clip.wav>] [--steps-per-epoch 625] [--keep]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from app.config import SCRIPTS_DIR


def _checkpoints(run: Path) -> list[Path]:
    cks = [d for d in run.iterdir() if d.is_dir() and re.fullmatch(r"checkpoint-\d+", d.name)]
    return sorted(cks, key=lambda d: int(d.name.split("-")[1]))


def _step(ckpt: Path) -> int:
    return int(ckpt.name.split("-")[1])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="run/ dir containing checkpoint-* folders")
    p.add_argument("--text", required=True, help="test phrase to synthesize")
    p.add_argument("--reference", default=None,
                   help="speaker reference .wav for SECS/MCD (optional)")
    p.add_argument("--steps-per-epoch", type=int, default=None,
                   help="to show an epoch column (= dataset_rows / batch_size)")
    p.add_argument("--keep", action="store_true",
                   help="keep the converted <checkpoint>_inference folders (default: delete)")
    p.add_argument("--output", default=None, help="report dir (default: <run>/_eval)")
    args = p.parse_args()

    run = Path(args.run)
    if not (run / "config.json").exists():
        print(f"ERROR: {run} не похож на run/ (нет config.json).", file=sys.stderr)
        return 1
    checkpoints = _checkpoints(run)
    if not checkpoints:
        print(f"ERROR: в {run} нет папок checkpoint-*.", file=sys.stderr)
        return 1

    out_dir = Path(args.output) if args.output else run / "_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    # If a reference clip is given, drop it into run/ so the metric step finds it.
    if args.reference:
        ref = Path(args.reference)
        if ref.exists():
            shutil.copy2(ref, run / "reference_speaker.wav")
            print(f"Референс для SECS/MCD: {ref.name}", flush=True)

    phrase_file = out_dir / "_phrase.txt"
    phrase_file.write_text(args.text, encoding="utf-8")

    export = SCRIPTS_DIR / "export_hf_checkpoint.py"
    infer = SCRIPTS_DIR / "run_vits_infer.py"
    rows = []

    for ck in checkpoints:
        step = _step(ck)
        epoch = round(step / args.steps_per_epoch, 1) if args.steps_per_epoch else None
        tag = f"эпоха {epoch}" if epoch is not None else f"шаг {step}"
        print(f"\n=== {ck.name} ({tag}) ===", flush=True)

        exp = ck.parent / f"{ck.name}_inference"
        r = subprocess.run([sys.executable, str(export), "--checkpoint", str(ck),
                            "--run", str(run), "--output", str(exp)])
        if r.returncode != 0:
            print(f"  пропуск: конвертация не удалась", flush=True)
            continue

        cfg = exp / "config.json"
        proc = subprocess.run(
            [sys.executable, str(infer), "--config", str(cfg), "--generator", str(cfg),
             "--text-file", str(phrase_file), "--format", "wav",
             "--model-name", ck.name, "--compute-metrics"],
            capture_output=True, text=True,
        )
        data = {}
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT_JSON="):
                try:
                    data = json.loads(line[len("RESULT_JSON="):])
                except json.JSONDecodeError:
                    pass
        wer = data.get("wer")
        secs = data.get("secs")
        mcd = data.get("mcd")
        outwav = data.get("output", "")
        print(f"  WER={wer} SECS={secs} MCD={mcd}", flush=True)
        print(f"  аудио: {outwav}", flush=True)
        rows.append({
            "checkpoint": ck.name, "step": step, "epoch": epoch,
            "wer": wer, "secs": secs, "mcd": mcd, "audio": outwav,
        })

        if not args.keep:
            shutil.rmtree(exp, ignore_errors=True)

    # CSV report
    report = out_dir / "checkpoints_report.csv"
    with report.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["checkpoint", "step", "epoch", "wer", "secs", "mcd", "audio"])
        w.writeheader()
        w.writerows(rows)

    # Sorted summary (lower WER = better; None sinks to the bottom).
    print("\n\n============ ИТОГ (сортировка по WER, меньше = лучше) ============", flush=True)
    ranked = sorted(rows, key=lambda r: (r["wer"] is None, r["wer"] if r["wer"] is not None else 9e9))
    hdr = f"{'checkpoint':<22}{'эпоха':>7}{'WER':>8}{'SECS':>8}{'MCD':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in ranked:
        ep = f"{r['epoch']}" if r["epoch"] is not None else "-"
        wer = f"{r['wer']:.3f}" if r["wer"] is not None else "-"
        secs = f"{r['secs']:.3f}" if r["secs"] is not None else "-"
        mcd = f"{r['mcd']:.1f}" if r["mcd"] is not None else "-"
        print(f"{r['checkpoint']:<22}{ep:>7}{wer:>8}{secs:>8}{mcd:>8}")
    if ranked and ranked[0]["wer"] is not None:
        print(f"\nЛучший по WER: {ranked[0]['checkpoint']}")
    print(f"\nCSV: {report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
