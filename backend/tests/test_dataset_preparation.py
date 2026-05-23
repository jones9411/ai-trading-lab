import pandas as pd
import pytest

from app.ml.dataset_preparation import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    prepare_ml_dataset,
    save_ml_dataset,
    split_dataset_by_time,
    split_features_and_target,
    validate_feature_columns,
)


def make_sample_dataset():
    return pd.DataFrame(
        {
            "daily_return": [0.01, None, 0.03, 0.04],
            "sma_5": [100.0, 101.0, 102.0, 103.0],
            "sma_20": [99.0, 100.0, 101.0, 102.0],
            "rolling_volatility_20": [0.20, 0.21, 0.22, 0.23],
            "target_up_next_day": [1, 0, None, 0],
        }
    )


def test_prepare_ml_dataset_drops_incomplete_rows():
    raw_dataset = make_sample_dataset()

    prepared_dataset = prepare_ml_dataset(raw_dataset)

    assert len(prepared_dataset) == 2
    assert prepared_dataset[TARGET_COLUMN].tolist() == [1, 0]


def test_validate_feature_columns_rejects_target_column():
    leaking_features = FEATURE_COLUMNS + [TARGET_COLUMN]

    with pytest.raises(ValueError, match="Target leakage detected"):
        validate_feature_columns(leaking_features)


def test_split_features_and_target_returns_x_and_y():
    raw_dataset = make_sample_dataset()
    prepared_dataset = prepare_ml_dataset(raw_dataset)

    x, y = split_features_and_target(prepared_dataset)

    assert list(x.columns) == FEATURE_COLUMNS
    assert y.name == TARGET_COLUMN
    assert len(x) == len(y)


def test_split_dataset_by_time_keeps_oldest_rows_for_training():
    dataset = pd.DataFrame(
        {
            "marker": list(range(10)),
            "daily_return": [0.01] * 10,
            "sma_5": [100.0] * 10,
            "sma_20": [99.0] * 10,
            "rolling_volatility_20": [0.20] * 10,
            "target_up_next_day": [1, 0] * 5,
        }
    )

    train_data, test_data = split_dataset_by_time(
        dataset=dataset,
        train_ratio=0.8,
    )

    assert train_data["marker"].tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
    assert test_data["marker"].tolist() == [8, 9]


def test_save_ml_dataset_creates_csv_file(tmp_path):
    raw_dataset = make_sample_dataset()
    prepared_dataset = prepare_ml_dataset(raw_dataset)

    dataset_path = save_ml_dataset(
        dataset=prepared_dataset,
        symbol="aapl",
        output_dir=tmp_path,
    )

    assert dataset_path.exists()
    assert dataset_path.name == "AAPL_ml_dataset.csv"

    loaded_dataset = pd.read_csv(dataset_path)

    assert len(loaded_dataset) == 2