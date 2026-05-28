"""Resolve paths to pretrained weights and vendored configs."""

from __future__ import annotations

import os
from pathlib import Path

import rvc_core  # ensures _vendored is on sys.path

VENDORED: Path = rvc_core.VENDORED_PATH


def assets_root() -> Path:
    """Override-able via VOICEGEN_RVC_ASSETS env var."""
    env = os.environ.get("VOICEGEN_RVC_ASSETS")
    if env:
        return Path(env)
    return VENDORED / "assets"


def pretrained_v2(sr: str, kind: str) -> Path:
    """Return path to `assets/pretrained_v2/{f0|}{G|D}{sr}.pth`."""
    fname = f"f0{kind}{sr}.pth"
    return assets_root() / "pretrained_v2" / fname


def hubert_path() -> Path:
    return assets_root() / "hubert" / "hubert_base.pt"


def rmvpe_path() -> Path:
    return assets_root() / "rmvpe" / "rmvpe.pt"


def vendored_config(sr: str, version: str = "v2") -> Path:
    """Path to the upstream JSON config for the given (sr, version).

    Mirrors upstream's quirk: 40k has no v2 config, always falls back to v1.
    """
    if version == "v1" or sr == "40k":
        return VENDORED / "configs" / "v1" / f"{sr}.json"
    return VENDORED / "configs" / "v2" / f"{sr}.json"
