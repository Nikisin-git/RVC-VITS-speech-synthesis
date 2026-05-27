"""Regex-based parsers for subprocess log output.

Each parser extracts progress-relevant fields (epoch, elapsed, checkpoint events).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ProgressEvent:
    kind: str                 # "epoch" | "checkpoint" | "stage" | "metric" | "info"
    epoch: int | None = None
    total_epochs: int | None = None
    elapsed: str | None = None
    message: str = ""
    loss: float | None = None


_RVC_EPOCH_RE = re.compile(
    r"epoch\s+(?P<epoch>\d+)(?:\s*/\s*(?P<total>\d+))?.*?loss(?:_[a-z])?\s*[:=]\s*(?P<loss>[\d.]+)",
    re.IGNORECASE,
)
_RVC_SAVE_RE = re.compile(
    r"(?:saving|saved)\s+(?:ckpt|checkpoint).*?epoch[_\s]?(?P<epoch>\d+)",
    re.IGNORECASE,
)
_TTS_EPOCH_RE = re.compile(
    r">\s*EPOCH:\s*(?P<epoch>\d+)\s*/\s*(?P<total>\d+)",
    re.IGNORECASE,
)
_TTS_CHECKPOINT_RE = re.compile(r"checkpoint\s+(?:saved|written).*?step[_\s]?(?P<step>\d+)", re.IGNORECASE)
_STAGE_RE = re.compile(r"===\s*stage:\s*(?P<stage>\w+)\s*===")
_ELAPSED_RE = re.compile(r"(?P<elapsed>\d+:\d{2}(?::\d{2})?)")


def parse_rvc_train_line(line: str) -> ProgressEvent | None:
    s = _STAGE_RE.search(line)
    if s:
        return ProgressEvent(kind="stage", message=s.group("stage"))
    m = _RVC_EPOCH_RE.search(line)
    if m:
        elapsed = _ELAPSED_RE.search(line)
        return ProgressEvent(
            kind="epoch",
            epoch=int(m.group("epoch")),
            total_epochs=int(m.group("total")) if m.group("total") else None,
            loss=float(m.group("loss")),
            elapsed=elapsed.group("elapsed") if elapsed else None,
            message=line.strip(),
        )
    m = _RVC_SAVE_RE.search(line)
    if m:
        return ProgressEvent(kind="checkpoint", epoch=int(m.group("epoch")), message=line.strip())
    return None


def parse_tts_train_line(line: str) -> ProgressEvent | None:
    m = _TTS_EPOCH_RE.search(line)
    if m:
        return ProgressEvent(
            kind="epoch",
            epoch=int(m.group("epoch")),
            total_epochs=int(m.group("total")),
            message=line.strip(),
        )
    m = _TTS_CHECKPOINT_RE.search(line)
    if m:
        return ProgressEvent(kind="checkpoint", message=line.strip())
    return None


def parse_inference_line(line: str) -> ProgressEvent | None:
    """Inference is single-shot; surface 'done' and any error lines."""
    lower = line.lower()
    if "error" in lower or "traceback" in lower:
        return ProgressEvent(kind="info", message=line.strip())
    if "done" in lower or "finished" in lower or "written" in lower:
        return ProgressEvent(kind="info", message=line.strip())
    return None
