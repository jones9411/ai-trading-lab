from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_get_features_for_valid_symbol_returns_200():
    response = client.get(
        "/api/features/AAPL?start_date=2024-01-01&end_date=2024-03-31"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["symbol"] == "AAPL"
    assert "features" in data
    assert isinstance(data["features"], list)
    assert len(data["features"]) > 0

    first_feature = data["features"][0]

    assert "date" in first_feature
    assert "close" in first_feature
    assert "daily_return" in first_feature
    assert "sma_5" in first_feature
    assert "sma_20" in first_feature
    assert "rolling_volatility_20" in first_feature


def test_get_features_with_invalid_date_range_returns_400():
    response = client.get(
        "/api/features/AAPL?start_date=2024-01-10&end_date=2024-01-01"
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "start_date cannot be later than end_date"


def test_get_features_with_invalid_symbol_format_returns_422():
    response = client.get(
        "/api/features/THIS_SYMBOL_IS_TOO_LONG?start_date=2024-01-01&end_date=2024-01-10"
    )

    assert response.status_code == 422