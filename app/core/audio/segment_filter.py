"""Filter sliced speech fragments for dataset cleaning.

Transcribes each short fragment with the same Whisper model used for WER and
drops the ones that are not useful training material:
  * пустые / без речи,
  * состоящие только из слов-паразитов («ну», «эээ», «типа», …),
  * с повторами / заиканием,
  * с низкой уверенностью распознавания (обычно шум/неразборчиво).
Good fragments are copied to an output folder; originals are never touched.

The decision logic (`classify`) is pure and unit-testable; the Whisper call
lives in `filter_fragments`.
"""

from __future__ import annotations

import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

# Contentless Russian fillers / discourse particles. A fragment is dropped as
# "filler-only" when EVERY token is in this set, so including ordinary short
# words here is safe (they only matter when the clip is nothing but them).
FILLERS = {
    "ну", "нуу", "нууу", "э", "ээ", "эээ", "эм", "ээм", "эх",
    "м", "мм", "ммм", "мда", "а", "аа", "ааа", "о", "оо",
    "типа", "вот", "значит", "короче", "блин", "это", "самое",
    "как", "бы", "в", "общем", "вообще", "ваще", "так", "угу", "ага",
}


@dataclass
class FilterOptions:
    drop_empty: bool = True
    drop_filler_only: bool = True
    drop_repetition: bool = True
    drop_low_conf: bool = True
    no_speech_thr: float = 0.6      # no_speech_prob выше -> это не речь
    logprob_thr: float = -1.0       # avg_logprob ниже -> ненадёжно
    comp_ratio_thr: float = 2.4     # compression_ratio выше -> повторы/зацикливание


def _tokens(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return text.split()


def _collapse(tok: str) -> str:
    """Collapse runs of a repeated char so elongations match fillers:
    'нууу' -> 'ну', 'типаа' -> 'типа'."""
    return re.sub(r"(.)\1+", r"\1", tok)


def _is_filler(tok: str) -> bool:
    return tok in FILLERS or _collapse(tok) in FILLERS


def _has_stutter(toks: list[str]) -> bool:
    """Consecutive duplicate tokens signal a stutter (я-я-я, на на на)."""
    run = 1
    for i in range(1, len(toks)):
        if toks[i] == toks[i - 1]:
            run += 1
            if run >= 3 or (run >= 2 and len(toks[i]) <= 3):
                return True
        else:
            run = 1
    return False


def classify(text: str, no_speech_prob: float, avg_logprob: float,
             comp_ratio: float, opts: FilterOptions) -> tuple[str, str]:
    """Return ('keep'|'drop', reason). Order: empty -> filler-only ->
    repetition -> low confidence."""
    toks = _tokens(text)
    if opts.drop_empty and (not toks or no_speech_prob >= opts.no_speech_thr):
        return "drop", "пусто/нет речи"
    if opts.drop_filler_only and toks and all(_is_filler(t) for t in toks):
        return "drop", "только слова-паразиты"
    if opts.drop_repetition and (comp_ratio >= opts.comp_ratio_thr or _has_stutter(toks)):
        return "drop", "повторы/заикание"
    if opts.drop_low_conf and avg_logprob <= opts.logprob_thr:
        return "drop", "низкая уверенность ASR"
    return "keep", "ок"


def _metrics(result: dict, text: str) -> tuple[float, float, float]:
    segs = result.get("segments") or []
    if not segs:
        # No segments at all -> treat as no speech (empty rule catches it).
        return (1.0, 0.0, 0.0)
    no_speech = max(s.get("no_speech_prob", 0.0) for s in segs)
    avg_lp = sum(s.get("avg_logprob", 0.0) for s in segs) / len(segs)
    comp = max(s.get("compression_ratio", 0.0) for s in segs)
    return (no_speech, avg_lp, comp)


def filter_fragments(paths: list[Path], out_dir: Path, language: str | None = "ru",
                     model_name: str | None = None, opts: FilterOptions | None = None,
                     copy: bool = True) -> list[dict]:
    """Classify each fragment, copy the kept ones to out_dir, write a CSV report.
    Returns the per-file result rows."""
    from app.core.metrics.wer import _get_whisper

    opts = opts or FilterOptions()
    model = _get_whisper(model_name)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    n = len(paths)
    kept = 0
    for i, p in enumerate(paths, 1):
        p = Path(p)
        try:
            result = model.transcribe(str(p), language=language, fp16=False)
            text = str(result.get("text", "")).strip()
            no_speech, avg_lp, comp = _metrics(result, text)
            decision, reason = classify(text, no_speech, avg_lp, comp, opts)
        except Exception as e:  # keep going; a bad file shouldn't abort the run
            decision, reason, text = "error", f"{type(e).__name__}: {e}", ""
            no_speech = avg_lp = comp = 0.0

        if decision == "keep":
            kept += 1
            if copy:
                shutil.copy2(p, out_dir / p.name)

        rows.append({
            "file": p.name, "decision": decision, "reason": reason,
            "no_speech_prob": round(no_speech, 3), "avg_logprob": round(avg_lp, 3),
            "compression_ratio": round(comp, 3), "text": text,
        })
        print(f"[filter] {i}/{n} {decision}: {p.name} — {reason}", flush=True)

    report = out_dir / "filter_report.csv"
    with report.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "file", "decision", "reason", "no_speech_prob", "avg_logprob",
            "compression_ratio", "text"])
        w.writeheader()
        w.writerows(rows)

    dropped = n - kept - sum(1 for r in rows if r["decision"] == "error")
    errors = sum(1 for r in rows if r["decision"] == "error")
    print(f"[filter] Итог: оставлено {kept}, удалено {dropped}, ошибок {errors} из {n}.",
          flush=True)
    print(f"[filter] Отчёт: {report}", flush=True)
    return rows
