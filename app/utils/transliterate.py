"""Russian-to-Latin transliteration for TTS output filenames."""

from __future__ import annotations

import re

try:
    from transliterate import translit as _translit
    _HAVE_LIB = True
except ImportError:
    _HAVE_LIB = False

_FALLBACK = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")


def _fallback_translit(text: str) -> str:
    out = []
    for ch in text:
        lower = ch.lower()
        if lower in _FALLBACK:
            translated = _FALLBACK[lower]
            out.append(translated.upper() if ch.isupper() else translated)
        else:
            out.append(ch)
    return "".join(out)


def text_to_filename(text: str, max_len: int = 20) -> str:
    """Transliterate, drop punctuation, replace spaces with `_`, truncate."""
    if not text:
        return "untitled"
    if _HAVE_LIB:
        try:
            translit = _translit(text, "ru", reversed=True)
        except Exception:
            translit = _fallback_translit(text)
    else:
        translit = _fallback_translit(text)
    cleaned = _PUNCT_RE.sub("", translit)
    cleaned = _WS_RE.sub("_", cleaned).strip("_")
    if not cleaned:
        return "untitled"
    return cleaned[:max_len]
