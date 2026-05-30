from pathlib import Path

import numpy as np

from app.core.audio.io import _replace_ext, _to_soundfile_layout


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


def test_soundfile_layout_mono_unchanged():
    audio = np.zeros(44100, dtype=np.float32)
    assert _to_soundfile_layout(audio).shape == (44100,)


def test_soundfile_layout_stereo_transposed():
    # librosa (channels, frames) → soundfile (frames, channels)
    audio = np.zeros((2, 44100), dtype=np.float32)
    out = _to_soundfile_layout(audio)
    assert out.shape == (44100, 2)


def test_soundfile_layout_already_correct():
    # Short clips where channels >= frames would be ambiguous, but
    # real audio always has frames >> channels. Make sure we don't
    # double-transpose when input is already in soundfile layout.
    audio = np.zeros((44100, 2), dtype=np.float32)
    out = _to_soundfile_layout(audio)
    assert out.shape == (44100, 2)
