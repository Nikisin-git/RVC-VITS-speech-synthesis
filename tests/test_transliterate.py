from app.utils.transliterate import text_to_filename


def test_transliterate_basic():
    out = text_to_filename("Привет, мир!", max_len=20)
    assert "_" in out or out
    assert "," not in out and "!" not in out
    assert len(out) <= 20


def test_transliterate_truncates():
    long = "Это очень длинная фраза для синтеза речи"
    out = text_to_filename(long, max_len=20)
    assert len(out) <= 20


def test_transliterate_empty():
    assert text_to_filename("") == "untitled"
