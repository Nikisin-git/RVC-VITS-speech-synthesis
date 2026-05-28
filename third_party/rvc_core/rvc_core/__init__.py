"""Vendored RVC core (commit 7ef1986).

The upstream repository uses top-level absolute imports like
`from infer.lib.audio import load_audio` and `from configs.config import Config`.
We expose `_vendored/` as a `sys.path` entry so those imports resolve when
any rvc_core CLI module is run via `python -m rvc_core.<name>`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_VENDORED = Path(__file__).resolve().parent / "_vendored"
if str(_VENDORED) not in sys.path:
    sys.path.insert(0, str(_VENDORED))

# Many upstream modules read PROJECT_ROOT from env; honor that for callers
# that want vendored assets resolved from a custom location.
os.environ.setdefault("RVC_CORE_VENDORED", str(_VENDORED))

__all__ = ["VENDORED_PATH"]
VENDORED_PATH = _VENDORED
