from pathlib import Path

from app.core.tts.manifest import validate_manifest


def test_manifest_ok(tmp_path: Path):
    audio_dir = tmp_path / "wavs"
    audio_dir.mkdir()
    (audio_dir / "a.wav").write_bytes(b"")
    (audio_dir / "b.wav").write_bytes(b"")
    mf = tmp_path / "m.csv"
    mf.write_text("a.wav|hello\nb.wav|world\n", encoding="utf-8")
    rep = validate_manifest(mf, audio_dir)
    assert rep.ok
    assert len(rep.rows) == 2


def test_manifest_missing_file(tmp_path: Path):
    audio_dir = tmp_path / "wavs"
    audio_dir.mkdir()
    (audio_dir / "a.wav").write_bytes(b"")
    mf = tmp_path / "m.csv"
    mf.write_text("a.wav|hello\nb.wav|missing\n", encoding="utf-8")
    rep = validate_manifest(mf, audio_dir)
    assert not rep.ok
    assert any("b.wav" in e for e in rep.errors)


def test_manifest_no_separator(tmp_path: Path):
    audio_dir = tmp_path / "wavs"
    audio_dir.mkdir()
    mf = tmp_path / "m.csv"
    mf.write_text("broken_line_without_pipe\n", encoding="utf-8")
    rep = validate_manifest(mf, audio_dir)
    assert not rep.ok


def test_manifest_orphan_warning(tmp_path: Path):
    audio_dir = tmp_path / "wavs"
    audio_dir.mkdir()
    (audio_dir / "a.wav").write_bytes(b"")
    (audio_dir / "orphan.wav").write_bytes(b"")
    mf = tmp_path / "m.csv"
    mf.write_text("a.wav|hello\n", encoding="utf-8")
    rep = validate_manifest(mf, audio_dir)
    assert rep.ok  # warnings only
    assert any("не упомянут" in w for w in rep.warnings)
