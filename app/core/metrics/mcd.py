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
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def compute_mcd(
    reference_audio: Path,
    target_audio: Path,
    sr: int = 22050,
    n_mfcc: int = 13,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> float:
    """Compute MCD in dB between two audio files. Lower is better.

    Uses the classical pyworld+pysptk pipeline (textbook dB scale) when those
    packages are available, and only falls back to the librosa-MFCC
    approximation (relative-only, larger scale) if they are not installed.
    A failure *inside* the classical backend is raised, not silently downgraded,
    so a run never mixes the two scales.
    """
    try:
        return _compute_mcd_classical(reference_audio, target_audio, sr)
    except ImportError:
        return _compute_mcd_librosa(
            reference_audio, target_audio, sr, n_mfcc, n_fft, hop_length)


def _compute_mcd_classical(
    reference_audio: Path,
    target_audio: Path,
    sr: int = 22050,
    order: int = 24,
    frame_period: float = 5.0,
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
        y, _ = librosa.load(str(path), sr=sr, mono=True)
        if len(y) == 0:
            raise ValueError("Один из входных файлов пуст — нечего сравнивать.")
        # pyworld needs C-contiguous float64.
        y = np.ascontiguousarray(y, dtype=np.float64)
        f0, t = pyworld.harvest(y, sr, frame_period=frame_period)
        sp = pyworld.cheaptrick(y, f0, t, sr)          # (frames, fft/2+1)
        return pysptk.sp2mc(sp, order=order, alpha=alpha)  # (frames, order+1)

    mc_ref = _mel_cepstrum(reference_audio)[:, 1:]  # drop c0 (energy/level)
    mc_syn = _mel_cepstrum(target_audio)[:, 1:]
    if mc_ref.shape[0] == 0 or mc_syn.shape[0] == 0:
        raise ValueError("Не удалось извлечь мел-кепстр — слишком короткое аудио.")

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
) -> float:
    """Dependency-free fallback MCD from librosa MFCCs.

    RELATIVE metric only — the absolute value is on a much larger scale than the
    classical MCD (see module docstring), so do not read it against the < 6 dB
    thresholds. Reliable for ranking models against the same reference.
    """
    import librosa

    y_ref, _ = librosa.load(str(reference_audio), sr=sr, mono=True)
    y_syn, _ = librosa.load(str(target_audio), sr=sr, mono=True)

    if len(y_ref) == 0 or len(y_syn) == 0:
        raise ValueError("Один из входных файлов пуст — нечего сравнивать.")

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
