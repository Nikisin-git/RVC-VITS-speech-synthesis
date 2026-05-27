"""Pedalboard-based audio post-processing chain.

Applies reverb, compression, filters and a noise gate, all configurable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ReverbConfig:
    room_size: float = 0.15
    wet_level: float = 0.2
    dry_level: float = 0.8
    damping: float = 0.7


@dataclass
class CompressorConfig:
    ratio: float = 4.0
    threshold_db: float = -10.0


@dataclass
class FiltersConfig:
    low_cut_hz: int = 80
    high_cut_hz: int = 12000
    limiter_threshold_db: float = -1.0


@dataclass
class NoiseGateConfig:
    threshold_db: float = -40
    ratio: float = 10
    attack_ms: float = 1
    release_ms: float = 100


@dataclass
class PostProcessChain:
    reverb: ReverbConfig = field(default_factory=ReverbConfig)
    compressor: CompressorConfig = field(default_factory=CompressorConfig)
    filters: FiltersConfig = field(default_factory=FiltersConfig)
    gate: NoiseGateConfig = field(default_factory=NoiseGateConfig)
    enable_reverb: bool = True
    enable_compressor: bool = True
    enable_filters: bool = True
    enable_gate: bool = True


def apply(audio: np.ndarray, sr: int, cfg: PostProcessChain) -> np.ndarray:
    """Apply the configured chain to a mono float32 signal."""
    from pedalboard import (  # type: ignore
        Pedalboard, Reverb, Compressor, HighpassFilter, LowpassFilter, Limiter, NoiseGate,
    )

    board: list = []
    if cfg.enable_gate:
        g = cfg.gate
        board.append(NoiseGate(threshold_db=g.threshold_db, ratio=g.ratio,
                               attack_ms=g.attack_ms, release_ms=g.release_ms))
    if cfg.enable_filters:
        f = cfg.filters
        if f.low_cut_hz > 0:
            board.append(HighpassFilter(cutoff_frequency_hz=float(f.low_cut_hz)))
        if f.high_cut_hz > 0:
            board.append(LowpassFilter(cutoff_frequency_hz=float(f.high_cut_hz)))
    if cfg.enable_compressor:
        c = cfg.compressor
        board.append(Compressor(threshold_db=c.threshold_db, ratio=c.ratio))
    if cfg.enable_reverb:
        r = cfg.reverb
        board.append(Reverb(room_size=r.room_size, wet_level=r.wet_level,
                            dry_level=r.dry_level, damping=r.damping))
    if cfg.enable_filters:
        board.append(Limiter(threshold_db=cfg.filters.limiter_threshold_db))

    if not board:
        return audio.astype(np.float32, copy=False)
    pb = Pedalboard(board)
    out = pb(audio.astype(np.float32, copy=False), sr)
    return out
