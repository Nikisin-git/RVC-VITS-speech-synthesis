"""LJSpeech manifest validation for TTS training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ManifestReport:
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rows: list[tuple[str, str]] = field(default_factory=list)


def validate_manifest(manifest_path: Path, audio_dir: Path) -> ManifestReport:
    """Validate a LJSpeech-style CSV: `<filename>|<text>`, UTF-8, no header."""
    report = ManifestReport()
    manifest_path = Path(manifest_path)
    audio_dir = Path(audio_dir)

    if not manifest_path.exists():
        report.ok = False
        report.errors.append(f"Манифест не найден: {manifest_path}")
        return report

    try:
        raw = manifest_path.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as e:
            report.ok = False
            report.errors.append(f"Файл не в UTF-8: {e}")
            return report
    except OSError as e:
        report.ok = False
        report.errors.append(f"Не удалось прочитать манифест: {e}")
        return report

    referenced: set[str] = set()
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        if "|" not in line:
            report.errors.append(f"Строка {lineno}: отсутствует разделитель '|'")
            continue
        parts = line.split("|", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            report.errors.append(f"Строка {lineno}: пустое имя файла или текст")
            continue
        filename, transcript = parts[0].strip(), parts[1].strip()
        referenced.add(filename)
        # Coqui's ljspeech formatter does `cols[0] + ".wav"`, so the manifest
        # is allowed to drop the extension. Accept both forms.
        candidate = audio_dir / filename
        if not candidate.exists() and not filename.lower().endswith(".wav"):
            candidate = audio_dir / f"{filename}.wav"
        if not candidate.exists():
            report.errors.append(f"Строка {lineno}: файл не найден — {filename}")
            continue
        referenced.add(candidate.name)
        report.rows.append((filename, transcript))

    on_disk = {p.name for p in audio_dir.glob("*.wav")}
    orphans = on_disk - referenced
    if orphans:
        sample = ", ".join(sorted(orphans)[:5])
        report.warnings.append(f"В каталоге есть аудио, не упомянутые в манифесте ({len(orphans)}): {sample}...")

    if report.errors:
        report.ok = False
    return report
