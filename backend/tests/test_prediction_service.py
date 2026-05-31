import pandas as pd
import pytest

from app.services import prediction_service


class FakeSavedModel:
    def predict(self, prediction_input):
        return [1]

    def predict_proba(self, prediction_input):
        return [[0.25, 0.75]]


def create_test_price_data():
    dates = pd.date_range(start="2024-01-01", periods=40, freq="D")

    close_prices = list(range(100, 140))

    return pd.DataFrame(
        {
            "Open": close_prices,
            "High": [price + 1 for price in close_prices],
            "Low": [price - 1 for price in close_prices],
            "Close": close_prices,
            "Volume": [1000000 for _ in close_prices],
        },
        index=dates,
    )


def test_get_prediction_for_symbol_loads_saved_model(monkeypatch):
    price_data = create_test_price_data()

    def fake_model_exists(symbol):
        return True

    def fake_load_model(symbol):
        return FakeSavedModel()

    def fake_get_price_data(symbol, start_date, end_date):
        return price_data

    monkeypatch.setattr(prediction_service, "model_exists", fake_model_exists)
    monkeypatch.setattr(prediction_service, "load_model", fake_load_model)
    monkeypatch.setattr(prediction_service, "get_price_data", fake_get_price_data)

    result = prediction_service.get_prediction_for_symbol("AAPL")

    assert result["symbol"] == "AAPL"
    assert result["date"] == pd.Timestamp("2024-02-09").date()
    assert result["probability_up"] == pytest.approx(0.75)
    assert result["signal"] == "up"


def test_get_prediction_for_symbol_raises_error_when_model_is_missing(monkeypatch):
    def fake_model_exists(symbol):
        return False

    monkeypatch.setattr(prediction_service, "model_exists", fake_model_exists)

    with pytest.raises(FileNotFoundError, match="No saved model found for AAPL"):
        prediction_service.get_prediction_for_symbol("AAPL")