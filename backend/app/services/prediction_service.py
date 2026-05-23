from datetime import date

import pandas as pd

from app.ml.dataset_preparation import FEATURE_COLUMNS, build_ml_dataset
from app.ml.train_baseline_model import train_baseline_model
from app.services.feature_service import create_price_features
from app.services.price_service import get_price_data, normalize_symbol


TRAINING_START_DATE = date(2020, 1, 1)
PREDICTION_THRESHOLD = 0.5


def convert_probability_to_signal(
    probability_up: float,
    threshold: float = PREDICTION_THRESHOLD,
) -> str:
    """
    Convert a probability into a simple readable signal.

    If probability_up is 0.50 or higher, we return "up".
    Otherwise, we return "not_up".
    """

    if probability_up >= threshold:
        return "up"

    return "not_up"


def get_date_from_feature_row(feature_row: pd.Series) -> date:
    """
    Extract the date from a pandas feature row.

    Our price data usually stores the date in the DataFrame index.
    This helper converts that index value into a normal Python date.
    """

    row_date = feature_row.name

    if hasattr(row_date, "date"):
        return row_date.date()

    return pd.to_datetime(row_date).date()


def get_latest_feature_row(
    symbol: str,
    start_date: date,
    end_date: date,
) -> pd.Series:
    """
    Get the latest row that has all required feature values.

    We do not need labels for prediction.
    We only need the latest complete feature row.
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

    clean_features = featured_data.dropna(subset=FEATURE_COLUMNS).copy()

    if clean_features.empty:
        raise ValueError(
            f"Not enough feature data available for {normalized_symbol}."
        )

    return clean_features.iloc[-1]


def get_prediction_for_symbol(symbol: str) -> dict:
    """
    Train the baseline model and return a prediction for the latest feature row.

    This is intentionally simple for now.
    Later we will save and load trained model files instead of retraining here.
    """

    normalized_symbol = normalize_symbol(symbol)

    start_date = TRAINING_START_DATE
    end_date = date.today()

    dataset = build_ml_dataset(
        symbol=normalized_symbol,
        start_date=start_date,
        end_date=end_date,
    )

    if dataset.empty:
        raise ValueError(f"Not enough training data available for {normalized_symbol}.")

    training_results = train_baseline_model(dataset)

    model = training_results["model"]

    latest_feature_row = get_latest_feature_row(
        symbol=normalized_symbol,
        start_date=start_date,
        end_date=end_date,
    )

    latest_features = latest_feature_row[FEATURE_COLUMNS].to_frame().T

    probability_up = float(model.predict_proba(latest_features)[0][1])
    signal = convert_probability_to_signal(probability_up)

    prediction_date = get_date_from_feature_row(latest_feature_row)

    return {
        "symbol": normalized_symbol,
        "date": prediction_date,
        "probability_up": probability_up,
        "signal": signal,
    }