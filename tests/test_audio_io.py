from pathlib import Path

from app.core.audio.io import _replace_ext


def test_replace_ext_basic():
    assert _replace_ext(Path("song.wav"), ".mp3") == Path("song.mp3")


def test_replace_ext_dot_in_stem():
    # The bug that hit RVC inference: dots inside the stem must survive.
    p = Path("track_muzter.net_Andrey_ver")
    assert _replace_ext(p, ".mp3") == Path("track_muzter.net_Andrey_ver.mp3")
    assert _replace_ext(p, ".tmp.wav") == Path("track_muzter.net_Andrey_ver.tmp.wav")


def test_replace_ext_strips_known_double_ext():
    assert _replace_ext(Path("track.tmp.wav"), ".mp3") == Path("track.mp3")


def test_replace_ext_no_existing_ext():
    assert _replace_ext(Path("plain"), ".wav") == Path("plain.wav")


def test_replace_ext_path_preserved():
    p = Path("/abs/dir/with.dots/file.net_ver")
    out = _replace_ext(p, ".mp3")
    assert out == Path("/abs/dir/with.dots/file.net_ver.mp3")
