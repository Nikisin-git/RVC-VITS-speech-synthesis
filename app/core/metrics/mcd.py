"""Mel Cepstral Distortion (MCD) computation.

MCD measures the average spectral distance between two audio signals in the
mel-cepstral domain. It is the standard objective quality metric for voice
conversion and TTS evaluation.

Formula:
    MCD[dB] = (10/ln(10)) * mean_t( sqrt(2 * sum_k((c_ref[t,k] - c_syn[t,k])^2)) )

where c are the mel-cepstral coefficients excluding c0 (energy) and t is the
frame index after DTW alignment. Lower values are always better and a perfectly
identical signal scores 0 dB.

Two backends, chosen automatically by `compute_mcd`:

1. CLASSICAL (preferred) — `pyworld` (CheapTrick spectral envelope) + `pysptk`
   (order-24 mel-cepstrum). This is the textbook pipeline, so the usual
   thresholds apply:
       < 6 dB   excellent
       6 - 8    good
       8 - 12   acceptable
       > 12     poor
   Empirically: two parallel utterances of one speaker score ~2, a clearly
   different timbre ~5.

2. LIBROSA FALLBACK — used only when pyworld/pysptk are not installed. It works
   from librosa MFCCs (no extra dependency) but the coefficients are on a
   DIFFERENT, much larger scale (parallel same-speaker ~20, natural-vs-TTS ~50),
   so the thresholds above DO NOT apply. Treat it as a RELATIVE metric only:
   between two models the lower value is spectrally closer to the target.

NOTE: Strict MCD assumes parallel utterances of the same text. When the
reference and target say different things, DTW still yields a meaningful average
distance, but the absolute number is inflated (in both backends).

MEMORY: DTW builds full N x M cost/accumulation matrices. With the classical
backend's 5 ms frames a 3-minute song is ~36k frames, i.e. ~10 GB per matrix —
enough to push Windows into swap and freeze the whole machine. So each signal is
trimmed of edge silence and cropped to its first `max_seconds` (default 20 s)
before feature extraction, and a hard cap on the DTW matrix size is enforced.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Analysis window per file. 20 s at 5 ms frames = 4000 frames -> ~130 MB per
# DTW matrix, which is safe on any machine.
DEFAULT_MAX_SECONDS = 20.0
# Hard guard on N*M DTW cells (float64): 25M cells ~ 200 MB per matrix.
_MAX_DTW_CELLS = 25_000_000


def _load_cropped(path: Path, sr: int, max_seconds: float | None) -> np.ndarray:
    """Load mono audio, trim edge silence, keep at most `max_seconds`."""
    import librosa

    y, _ = librosa.load(str(path), sr=sr, mono=True)
    if len(y) == 0:
        raise ValueError("Один из входных файлов пуст — нечего сравнивать.")
    y, _ = librosa.effects.trim(y, top_db=40)
    if max_seconds is not None and max_seconds > 0:
        y = y[: int(max_seconds * sr)]
    if len(y) == 0:
        raise ValueError("Один из входных файлов содержит только тишину.")
    return y


def _check_dtw_size(n: int, m: int) -> None:
    if n * m > _MAX_DTW_CELLS:
        raise MemoryError(
            f"DTW {n}x{m} кадров потребовал бы ~{n * m * 8 / 1e9:.1f} ГБ на матрицу — "
            f"уменьшите max_seconds."
        )


def compute_mcd(
    reference_audio: Path,
    target_audio: Path,
    sr: int = 22050,
    n_mfcc: int = 13,
    n_fft: int = 1024,
    hop_length: int = 256,
    max_seconds: float | None = DEFAULT_MAX_SECONDS,
) -> float:
    """Compute MCD in dB between two audio files. Lower is better.

    Only the first `max_seconds` of each file (after trimming edge silence) are
    analysed, so DTW memory stays bounded on long inputs such as full songs.

    Uses the classical pyworld+pysptk pipeline (textbook dB scale) when those
    packages are available, and only falls back to the librosa-MFCC
    approximation (relative-only, larger scale) if they are not installed.
    A failure *inside* the classical backend is raised, not silently downgraded,
    so a run never mixes the two scales.
    """
    try:
        return _compute_mcd_classical(
            reference_audio, target_audio, sr, max_seconds=max_seconds)
    except ImportError:
        return _compute_mcd_librosa(
            reference_audio, target_audio, sr, n_mfcc, n_fft, hop_length,
            max_seconds=max_seconds)


def _compute_mcd_classical(
    reference_audio: Path,
    target_audio: Path,
    sr: int = 22050,
    order: int = 24,
    frame_period: float = 5.0,
    max_seconds: float | None = DEFAULT_MAX_SECONDS,
) -> float:
    """Classical MCD via pyworld spectral envelope + pysptk mel-cepstrum.

    Produces values on the standard MCD dB scale (see module docstring). Raises
    ImportError if pyworld/pysptk are missing so the caller can fall back.
    """
    import librosa
    import pysptk
    import pyworld

    # The all-pass constant alpha is sample-rate dependent; pysptk derives the
    # value that best warps a linear axis to the mel axis for this sr.
    alpha = pysptk.util.mcepalpha(sr)

    def _mel_cepstrum(path: Path) -> np.ndarray:
        y = _load_cropped(path, sr, max_seconds)
        # pyworld needs C-contiguous float64.
        y = np.ascontiguousarray(y, dtype=np.float64)
        f0, t = pyworld.harvest(y, sr, frame_period=frame_period)
        sp = pyworld.cheaptrick(y, f0, t, sr)          # (frames, fft/2+1)
        return pysptk.sp2mc(sp, order=order, alpha=alpha)  # (frames, order+1)

    mc_ref = _mel_cepstrum(reference_audio)[:, 1:]  # drop c0 (energy/level)
    mc_syn = _mel_cepstrum(target_audio)[:, 1:]
    if mc_ref.shape[0] == 0 or mc_syn.shape[0] == 0:
        raise ValueError("Не удалось извлечь мел-кепстр — слишком короткое аудио.")

    _check_dtw_size(mc_ref.shape[0], mc_syn.shape[0])
    # DTW over the mel-cepstral sequences (librosa expects features x time).
    _, wp = librosa.sequence.dtw(X=mc_ref.T, Y=mc_syn.T, metric="euclidean")
    diff = mc_ref[wp[:, 0]] - mc_syn[wp[:, 1]]
    per_frame = np.sqrt(2.0 * np.sum(diff ** 2, axis=1))
    return float((10.0 / np.log(10.0)) * np.mean(per_frame))


def _compute_mcd_librosa(
    reference_audio: Path,
    target_audio: Path,
    sr: int = 22050,
    n_mfcc: int = 13,
    n_fft: int = 1024,
    hop_length: int = 256,
    max_seconds: float | None = DEFAULT_MAX_SECONDS,
) -> float:
    """Dependency-free fallback MCD from librosa MFCCs.

    RELATIVE metric only — the absolute value is on a much larger scale than the
    classical MCD (see module docstring), so do not read it against the < 6 dB
    thresholds. Reliable for ranking models against the same reference.
    """
    import librosa

    y_ref = _load_cropped(reference_audio, sr, max_seconds)
    y_syn = _load_cropped(target_audio, sr, max_seconds)

    # n_mfcc+1 so we can drop c0 (energy) — energy is usually excluded from
    # MCD because it depends on recording level, not on spectral shape.
    mfcc_ref = librosa.feature.mfcc(
        y=y_ref, sr=sr, n_mfcc=n_mfcc + 1,
        n_fft=n_fft, hop_length=hop_length,
    )[1:]
    mfcc_syn = librosa.feature.mfcc(
        y=y_syn, sr=sr, n_mfcc=n_mfcc + 1,
        n_fft=n_fft, hop_length=hop_length,
    )[1:]

    # Cepstral mean normalization: subtract the per-coefficient time mean from
    # each sequence. Removes the constant channel/recording bias so the
    # distance reflects spectral *shape* differences, not level offsets.
    mfcc_ref = mfcc_ref - mfcc_ref.mean(axis=1, keepdims=True)
    mfcc_syn = mfcc_syn - mfcc_syn.mean(axis=1, keepdims=True)

    _check_dtw_size(mfcc_ref.shape[1], mfcc_syn.shape[1])
    # DTW alignment: returns a warping path of (i, j) frame pairs.
    _, wp = librosa.sequence.dtw(X=mfcc_ref, Y=mfcc_syn, metric="euclidean")
    if len(wp) == 0:
        raise ValueError("DTW не смог выровнять последовательности MFCC.")

    diffs = mfcc_ref[:, wp[:, 0]] - mfcc_syn[:, wp[:, 1]]
    # librosa MFCCs come from a dB-scaled (10*log10) mel spectrum, so the
    # classical (10/ln10) constant is already baked in and must not be applied
    # again — MCD here is simply sqrt(2*sum(diff^2)) averaged over frames.
    per_frame = np.sqrt(2.0 * np.sum(diffs ** 2, axis=0))
    return float(np.mean(per_frame))
