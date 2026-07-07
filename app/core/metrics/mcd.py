"""Mel Cepstral Distortion (MCD) computation.

MCD measures the average spectral distance between two audio signals in the
mel-cepstral domain. It is the standard objective quality metric for voice
conversion and TTS evaluation.

Formula:
    MCD[dB] = (10/ln(10)) * mean_t( sqrt(2 * sum_k((c_ref[t,k] - c_syn[t,k])^2)) )

where c are the MFCC coefficients excluding c0 (energy) and t is the frame
index after DTW alignment. Lower values are better; a perfectly identical
signal scores 0 dB. Typical thresholds:
    < 6 dB   excellent
    6 - 8    good
    8 - 12   acceptable
    > 12     poor

This implementation uses librosa's MFCCs and DTW so we do not pull a new
dependency. The classical pyworld/pysptk pipeline produces slightly different
absolute numbers but the same relative ordering.

NOTE: Strict MCD assumes parallel utterances of the same text. When the
reference and target audio say different things (as for RVC/TTS where we
have a single representative speaker sample), DTW alignment lets us still
compute a meaningful average spectral distance, but the absolute number
should be treated as approximate.
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
    """Compute MCD in dB between two audio files. Lower is better."""
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
    # librosa MFCCs already come from a dB-scaled (10*log10) mel spectrum,
    # i.e. they are (10/ln10)*natural-log-cepstrum. The classical MCD constant
    # (10/ln10) is therefore ALREADY baked into the coefficients, so we must
    # NOT multiply by it again — MCD is simply sqrt(2*sum(diff^2)) averaged
    # over frames. (Multiplying again inflated the value ~4.34x.)
    per_frame = np.sqrt(2.0 * np.sum(diffs ** 2, axis=0))
    return float(np.mean(per_frame))
