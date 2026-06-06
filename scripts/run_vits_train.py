#!/usr/bin/env python
"""CLI: VITS / Coqui-TTS training."""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path

import _bootstrap  # noqa: F401


def _patch_filehandler_for_windows() -> None:
    """Force logging.FileHandler to open with delay=True and mode='a'.

    Coqui-Trainer opens trainer_*_log.txt with mode='w', which on Windows
    grabs an exclusive handle. If Windows Defender / Search / OneDrive
    briefly touches the file at the same moment, Trainer dies with
    WinError 32. delay=True postpones the actual open until the first
    log emit, and mode='a' uses non-exclusive append — together they
    sidestep both races.
    """
    if sys.platform != "win32":
        return
    _orig_init = logging.FileHandler.__init__

    def _patched(self, filename, mode="a", encoding=None, delay=True, errors=None):
        if mode == "w":
            mode = "a"
        _orig_init(self, filename, mode=mode, encoding=encoding, delay=True, errors=errors)

    logging.FileHandler.__init__ = _patched  # type: ignore[assignment]


_patch_filehandler_for_windows()

from app.core.tts.trainer import TtsTrainConfig, TtsTrainMode, train  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--audio-dir", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--mode", choices=["scratch", "finetune"], default="finetune")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--language", default="ru")
    p.add_argument("--sample-rate", type=int, default=22050)
    p.add_argument("--pretrained")
    p.add_argument("--cancel-flag")
    args = p.parse_args()

    cfg = TtsTrainConfig(
        audio_dir=Path(args.audio_dir),
        manifest_path=Path(args.manifest),
        model_name=args.model_name,
        mode=TtsTrainMode(args.mode),
        epochs=args.epochs,
        save_every=args.save_every,
        batch_size=args.batch_size,
        language=args.language,
        sample_rate=args.sample_rate,
        pretrained_checkpoint=Path(args.pretrained) if args.pretrained else None,
    )
    cancel = Path(args.cancel_flag) if args.cancel_flag else None
    try:
        result = train(cfg, cancel_flag=cancel)
        print(f"RESULT: {result}", flush=True)
        return 0
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
