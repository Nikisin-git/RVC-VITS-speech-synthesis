"""Speech denoising via DeepFilterNet 3.

Processes audio in overlapping chunks and streams the result straight to the
output file. This keeps both GPU memory (one chunk at a time) and CPU memory
(no full-length buffer) bounded, so hours-long files denoise on a small GPU
instead of hitting CUDA out-of-memory. A short margin is trimmed from each
internal chunk edge (where the model is still "warming up") so the seams
between chunks are inaudible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.config import DENOISING_DIR
from app.core.audio.io import _encode_mp3, _replace_ext, ensure_dir, load_audio

_DF_MODEL = None
_DF_STATE = None
_DF_SR = 48000


def _get_model():
    global _DF_MODEL, _DF_STATE
    if _DF_MODEL is None:
        from df.enhance import init_df  # type: ignore
        _DF_MODEL, _DF_STATE, _ = init_df()
    return _DF_MODEL, _DF_STATE


def denoise(input_path: Path, output_format: str = "wav",
            chunk_seconds: float = 30.0) -> Path:
    """Apply DeepFilterNet 3 to a single file in chunks. Returns output path."""
    from df.enhance import enhance  # type: ignore
    import soundfile as sf
    import torch

    input_path = Path(input_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    out_dir = ensure_dir(DENOISING_DIR)
    model, state = _get_model()

    audio, sr = load_audio(input_path, sr=_DF_SR, mono=True)
    total = len(audio)

    # Overlapping windows: a `margin` of warm-up audio is processed but trimmed
    # from each internal edge, so the kept regions tile the timeline seamlessly.
    window = max(1, int(chunk_seconds * _DF_SR))
    margin = int(min(0.5, chunk_seconds / 4.0) * _DF_SR)
    hop = max(1, window - 2 * margin)
    # Stop as soon as a window reaches the end, otherwise a trailing short
    # window would overlap and duplicate the tail.
    starts: list[int] = []
    s = 0
    while True:
        starts.append(s)
        if s + window >= total:
            break
        s += hop
    n = len(starts)
    print(f"[denoiser] enhancing {input_path.name} ({total/sr:.2f}s @ {sr} Hz) "
          f"кусками по {chunk_seconds:.0f}s → {n} шт.", flush=True)

    target = out_dir / f"{input_path.stem} [denoised]"
    tmp_wav = _replace_ext(target, ".tmp.wav")
    writer = sf.SoundFile(str(tmp_wav), mode="w", samplerate=_DF_SR,
                          channels=1, subtype="PCM_16")
    try:
        for idx, start in enumerate(starts, 1):
            seg = audio[start:start + window]
            tensor = torch.from_numpy(np.ascontiguousarray(seg)).unsqueeze(0)
            enhanced = enhance(model, state, tensor)
            enh = enhanced.squeeze(0).cpu().numpy().astype(np.float32)

            left = 0 if start == 0 else margin
            right = 0 if start + len(seg) >= total else margin
            writer.write(enh[left:len(seg) - right] if right else enh[left:])

            del tensor, enhanced, enh
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[denoiser] кусок {idx}/{n}", flush=True)
    finally:
        writer.close()

    if output_format.lower() == "mp3":
        out_path = _encode_mp3(tmp_wav, _replace_ext(target, ".mp3"))
        if out_path != tmp_wav:
            tmp_wav.unlink(missing_ok=True)
        return out_path
    final = _replace_ext(target, ".wav")
    tmp_wav.replace(final)
    return final
