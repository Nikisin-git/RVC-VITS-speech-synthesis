from app.utils.path_utils import sanitize_filename, validate_model_name


def test_sanitize_filename_strips_invalid():
    assert sanitize_filename('hello/world:test*?"<>|') == "helloworldtest"


def test_validate_model_name_ok():
    ok, err = validate_model_name("MyModel_v1")
    assert ok and err is None


def test_validate_model_name_empty():
    ok, err = validate_model_name("")
    assert not ok and "пуст" in err.lower()


def test_validate_model_name_space():
    ok, err = validate_model_name("My Model")
    assert not ok and "пробел" in err.lower()


def test_validate_model_name_special():
    ok, err = validate_model_name("Bad*Name")
    assert not ok and "*" in err
