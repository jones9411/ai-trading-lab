from pathlib import Path
from typing import Any

import joblib

from app.services.price_service import PROJECT_ROOT, normalize_symbol


MODEL_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "models"

DEFAULT_MODEL_NAME = "baseline_model"


def get_safe_model_symbol(symbol: str) -> str:
    """
    Convert a stock symbol into a safe filename-friendly version.

    Examples:
    AAPL -> AAPL
    VOD.L -> VOD_L
    BRK-B -> BRK_B
    """

    normalized_symbol = normalize_symbol(symbol)

    return normalized_symbol.replace(".", "_").replace("-", "_")


def get_model_path(
    symbol: str,
    model_name: str = DEFAULT_MODEL_NAME,
    output_dir: Path = MODEL_DATA_DIR,
) -> Path:
    """
    Build the path where a model artifact should be saved.
    """

    safe_symbol = get_safe_model_symbol(symbol)

    return output_dir / f"{safe_symbol}_{model_name}.joblib"


def ensure_model_data_dir_exists(
    output_dir: Path = MODEL_DATA_DIR,
) -> None:
    """
    Make sure the model artifact directory exists.
    """

    output_dir.mkdir(parents=True, exist_ok=True)


def save_model(
    model: Any,
    symbol: str,
    model_name: str = DEFAULT_MODEL_NAME,
    output_dir: Path = MODEL_DATA_DIR,
) -> Path:
    """
    Save a trained model to disk using joblib.
    """

    ensure_model_data_dir_exists(output_dir)

    model_path = get_model_path(
        symbol=symbol,
        model_name=model_name,
        output_dir=output_dir,
    )

    joblib.dump(model, model_path)

    return model_path


def load_model(
    symbol: str,
    model_name: str = DEFAULT_MODEL_NAME,
    output_dir: Path = MODEL_DATA_DIR,
) -> Any:
    """
    Load a trained model from disk.
    """

    model_path = get_model_path(
        symbol=symbol,
        model_name=model_name,
        output_dir=output_dir,
    )

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    return joblib.load(model_path)


def model_exists(
    symbol: str,
    model_name: str = DEFAULT_MODEL_NAME,
    output_dir: Path = MODEL_DATA_DIR,
) -> bool:
    """
    Check whether a saved model exists.
    """

    model_path = get_model_path(
        symbol=symbol,
        model_name=model_name,
        output_dir=output_dir,
    )

    return model_path.exists()
