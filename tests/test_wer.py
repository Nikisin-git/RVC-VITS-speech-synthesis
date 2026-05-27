from app.core.metrics.wer import compute_wer


def test_wer_identical():
    assert compute_wer("привет мир", "привет мир") == 0.0


def test_wer_one_substitution():
    # 1 substitution out of 2 words = 0.5
    assert abs(compute_wer("привет мир", "привет код") - 0.5) < 0.01


def test_wer_empty_reference():
    assert compute_wer("", "") == 0.0
    assert compute_wer("", "hello") >= 0.99
