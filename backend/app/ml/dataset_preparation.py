from datetime import date
from pathlib import Path

import pandas as pd

from app.services.feature_service import create_price_features
from app.services.label_service import create_price_labels
from app.services.price_service import PROJECT_ROOT, get_price_data, normalize_symbol


FEATURE_COLUMNS = [
    "daily_return",
    "sma_5",
    "sma_20",
    "rolling_volatility_20",
]

TARGET_COLUMN = "target_up_next_day"

PROCESSED_ML_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "ml"


def validate_feature_columns(
    feature_columns: list[str],
    target_column: str = TARGET_COLUMN,
) -> None:
    """
    Check that the model feature columns do not contain target/answer columns.

    This helps protect us from target leakage.
    """

    if target_column in feature_columns:
        raise ValueError(
            f"Target leakage detected. "
            f"The target column '{target_column}' cannot be used as a feature."
        )

    leaking_columns = [
        column for column in feature_columns if column.startswith("target_")
    ]

    if leaking_columns:
        raise ValueError(
            f"Target leakage detected. "
            f"These columns should not be model features: {leaking_columns}"
        )


def prepare_ml_dataset(
    data: pd.DataFrame,
    feature_columns: list[str] | None = None,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """
    Prepare a clean machine-learning dataset.

    This function:
    1. Checks for target leakage
    2. Checks that required columns exist
    3. Drops incomplete rows
    4. Converts the target column to integers
    """

    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS

    validate_feature_columns(
        feature_columns=feature_columns,
        target_column=target_column,
    )

    required_columns = feature_columns + [target_column]

    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    dataset = data.dropna(subset=required_columns).copy()

    dataset[target_column] = dataset[target_column].astype(int)

    return dataset


def build_ml_dataset(
    symbol: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Build a full ML dataset for one symbol.

    This function combines the pipeline:

    prices -> features -> labels -> clean ML dataset
    """

    normalized_symbol = normalize_symbol(symbol)

    price_data = get_price_data(
        symbol=normalized_symbol,
        start_date=start_date,
        end_date=end_date,
    )

    if price_data.empty:
        raise ValueError(f"No price data found for {normalized_symbol}.")

    featured_data = create_price_features(price_data)
    labeled_data = create_price_labels(featured_data)

    dataset = prepare_ml_dataset(labeled_data)

    return dataset


def split_features_and_target(
    dataset: pd.DataFrame,
    feature_columns: list[str] | None = None,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split a prepared dataset into X and y.

    X contains the model input features.
    y contains the answer the model learns to predict.
    """

    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS

    validate_feature_columns(
        feature_columns=feature_columns,
        target_column=target_column,
    )

    x = dataset[feature_columns].copy()
    y = dataset[target_column].copy()

    return x, y


def split_dataset_by_time(
    dataset: pd.DataFrame,
    train_ratio: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into training and test sets while preserving time order.

    The older rows are used for training.
    The newer rows are used for testing.
    """

    if dataset.empty:
        raise ValueError("Dataset is empty. Cannot split empty data.")

    if train_ratio <= 0 or train_ratio >= 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    split_index = int(len(dataset) * train_ratio)

    train_data = dataset.iloc[:split_index].copy()
    test_data = dataset.iloc[split_index:].copy()

    if train_data.empty or test_data.empty:
        raise ValueError("Train/test split failed. Not enough rows available.")

    return train_data, test_data


def create_train_test_data(
    dataset: pd.DataFrame,
    train_ratio: float = 0.8,
) -> dict:
    """
    Create train/test data for model training.

    Returns:
    - train_data
    - test_data
    - x_train
    - y_train
    - x_test
    - y_test
    """

    train_data, test_data = split_dataset_by_time(
        dataset=dataset,
        train_ratio=train_ratio,
    )

    x_train, y_train = split_features_and_target(train_data)
    x_test, y_test = split_features_and_target(test_data)

    return {
        "train_data": train_data,
        "test_data": test_data,
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
    }


def get_ml_dataset_path(
    symbol: str,
    output_dir: Path = PROCESSED_ML_DATA_DIR,
) -> Path:
    """
    Build the file path for a processed ML dataset.
    """

    normalized_symbol = normalize_symbol(symbol)

    safe_symbol = normalized_symbol.replace(".", "_").replace("-", "_")

    return output_dir / f"{safe_symbol}_ml_dataset.csv"


def save_ml_dataset(
    dataset: pd.DataFrame,
    symbol: str,
    output_dir: Path = PROCESSED_ML_DATA_DIR,
) -> Path:
    """
    Save a processed ML dataset to data/processed/ml/.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = get_ml_dataset_path(
        symbol=symbol,
        output_dir=output_dir,
    )

    dataset.to_csv(dataset_path)

    return dataset_path