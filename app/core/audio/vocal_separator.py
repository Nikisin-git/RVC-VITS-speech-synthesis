"""Vocal/instrumental separation via Demucs htdemucs_ft."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from app.config import VOCAL_REMOVER_DIR
from app.core.audio.io import ensure_dir, load_audio, save_audio

MODEL = "htdemucs_ft"


def _safe_stem(name: str) -> str:
    """A filename stem Windows can use as a folder name. Demucs creates an
    output folder named after the input stem, and Windows forbids reserved
    characters and silently strips trailing dots/spaces — so a file like
    'track....mp3' breaks the save. Sanitize those away."""
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    name = name.rstrip(" .").strip()
    return name or "input"


def separate(input_path: Path, output_format: str = "wav") -> tuple[Path, Path]:
    """Run Demucs htdemucs_ft on a single file.

    Returns (vocals_path, music_path) in the project's VOCAL_REMOVER_DIR tree.
    Demucs is invoked as a subprocess to keep its torch model isolated.
    """
    input_path = Path(input_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    vocals_dir = ensure_dir(VOCAL_REMOVER_DIR / "vocals")
    music_dir = ensure_dir(VOCAL_REMOVER_DIR / "music")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_out = Path(tmpdir) / "demucs_out"
        # Demucs names its output folder after the input file's stem and writes
        # <stem>/vocals.wav. A stem with trailing dots/spaces or reserved chars
        # (e.g. "track....mp3") can't be created as a folder on Windows, so the
        # save crashes. Hand Demucs a sanitized copy and keep the clean name.
        stem = _safe_stem(input_path.stem)
        safe_input = Path(tmpdir) / f"{stem}{input_path.suffix}"
        shutil.copy2(input_path, safe_input)

        cmd = [
            sys.executable, "-m", "demucs.separate",
            "-n", MODEL,
            "--two-stems=vocals",
            "-o", str(tmp_out),
            str(safe_input),
        ]
        print(f"[demucs] running: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)

        # Demucs writes to <out>/<model>/<track_stem>/{vocals,no_vocals}.wav
        produced = tmp_out / MODEL / stem
        vocals_src = produced / "vocals.wav"
        music_src = produced / "no_vocals.wav"
        if not vocals_src.exists() or not music_src.exists():
            raise RuntimeError(f"Demucs did not produce expected outputs in {produced}")

        vocals_audio, sr = load_audio(vocals_src, sr=None, mono=False)
        music_audio, _ = load_audio(music_src, sr=None, mono=False)

        vocals_out = save_audio(vocals_dir / f"{stem} [vocal]", vocals_audio, sr, output_format)
        music_out = save_audio(music_dir / f"{stem} [music]", music_audio, sr, output_format)

    return vocals_out, music_out


def cleanup_partial(stem: str) -> None:
    """Remove half-written outputs (used on cancellation)."""
    stem = _safe_stem(stem)
    for sub in ("vocals", "music"):
        d = VOCAL_REMOVER_DIR / sub
        if not d.exists():
            continue
        for ext in (".wav", ".mp3"):
            f = d / f"{stem} [{'vocal' if sub == 'vocals' else 'music'}]{ext}"
            if f.exists():
                f.unlink()
    shutil.rmtree(VOCAL_REMOVER_DIR / "_tmp", ignore_errors=True)
