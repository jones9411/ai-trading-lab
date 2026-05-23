import pytest

from app.ml.model_storage import (
    DEFAULT_MODEL_NAME,
    get_model_path,
    get_safe_model_symbol,
    load_model,
    model_exists,
    save_model,
)


def test_get_safe_model_symbol_normalizes_symbol_for_filename():
    safe_symbol = get_safe_model_symbol("vod.l")

    assert safe_symbol == "VOD_L"


def test_get_model_path_builds_expected_filename(tmp_path):
    model_path = get_model_path(
        symbol="aapl",
        model_name=DEFAULT_MODEL_NAME,
        output_dir=tmp_path,
    )

    assert model_path.name == "AAPL_baseline_model.joblib"
    assert model_path.parent == tmp_path


def test_save_model_creates_model_file(tmp_path):
    model = {
        "model_type": "test_model",
        "version": 1,
    }

    model_path = save_model(
        model=model,
        symbol="aapl",
        output_dir=tmp_path,
    )

    assert model_path.exists()
    assert model_path.name == "AAPL_baseline_model.joblib"


def test_load_model_returns_saved_model(tmp_path):
    model = {
        "model_type": "test_model",
        "version": 1,
    }

    save_model(
        model=model,
        symbol="aapl",
        output_dir=tmp_path,
    )

    loaded_model = load_model(
        symbol="aapl",
        output_dir=tmp_path,
    )

    assert loaded_model == model


def test_model_exists_returns_true_when_model_file_exists(tmp_path):
    model = {
        "model_type": "test_model",
    }

    save_model(
        model=model,
        symbol="aapl",
        output_dir=tmp_path,
    )

    assert model_exists(
        symbol="aapl",
        output_dir=tmp_path,
    )


def test_load_model_raises_error_when_model_file_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        load_model(
            symbol="aapl",
            output_dir=tmp_path,
        )