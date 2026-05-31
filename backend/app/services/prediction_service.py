from datetime import date, timedelta

import pandas as pd

from app.ml.dataset_preparation import FEATURE_COLUMNS
from app.ml.model_storage import load_model, model_exists
from app.services.feature_service import create_price_features
from app.services.price_service import get_price_data, normalize_symbol


def get_prediction_date_from_row_index(row_index) -> date:
    """
    Convert the DataFrame row index into a normal Python date.

    The row index may be a pandas Timestamp, datetime, or another date-like value.
    The API response should return a clean date object.
    """
    return pd.to_datetime(row_index).date()


def get_latest_complete_feature_row(symbol: str) -> pd.Series:
    """
    Get the most recent row that has all the feature columns needed by the model.

    We fetch about 1 year of price data because rolling features, like SMA 20 and
    rolling volatility, need enough previous rows before they become usable.
    """
    end_date = date.today()
    start_date = end_date - timedelta(days=365)

    price_data = get_price_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
    )

    if price_data.empty:
        raise ValueError(f"No price data found for {symbol}")

    feature_data = create_price_features(price_data)

    complete_feature_data = feature_data.dropna(subset=FEATURE_COLUMNS)

    if complete_feature_data.empty:
        raise ValueError(
            f"Not enough complete feature data found for {symbol}. "
            "Try fetching more historical price data first."
        )

    return complete_feature_data.iloc[-1]


def build_prediction_input(latest_feature_row: pd.Series) -> pd.DataFrame:
    """
    Convert the latest feature row into the 2D table shape expected by scikit-learn.

    scikit-learn models expect prediction input to look like a small DataFrame:

        daily_return | sma_5 | sma_20 | rolling_volatility_20

    Even though we only predict one row, it still needs to be a table.
    """
    return pd.DataFrame(
        [latest_feature_row[FEATURE_COLUMNS].to_dict()],
        columns=FEATURE_COLUMNS,
    )


def get_prediction_for_symbol(symbol: str) -> dict:
    """
    Load a saved model and use it to make the latest prediction for a symbol.

    This function should not train a model.
    Training happens separately in:

        python -m app.ml.train_baseline_model

    This function only performs inference.
    """
    normalized_symbol = normalize_symbol(symbol)

    if not model_exists(normalized_symbol):
        raise FileNotFoundError(
            f"No saved model found for {normalized_symbol}. "
            "Run this command first: python -m app.ml.train_baseline_model"
        )

    model = load_model(normalized_symbol)

    latest_feature_row = get_latest_complete_feature_row(normalized_symbol)
    prediction_input = build_prediction_input(latest_feature_row)

    prediction = int(model.predict(prediction_input)[0])
    probability_up = float(model.predict_proba(prediction_input)[0][1])

    signal = "up" if prediction == 1 else "down"

    prediction_date = get_prediction_date_from_row_index(latest_feature_row.name)

    return {
        "symbol": normalized_symbol,
        "date": prediction_date,
        "probability_up": probability_up,
        "signal": signal,
    }