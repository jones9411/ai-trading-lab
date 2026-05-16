from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_check_returns_healthy_status():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
    }


def test_get_prices_for_valid_symbol_returns_200():
    response = client.get(
        "/api/prices/AAPL?start_date=2024-01-01&end_date=2024-01-10"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["symbol"] == "AAPL"
    assert "prices" in data
    assert isinstance(data["prices"], list)
    assert len(data["prices"]) > 0


def test_get_prices_with_invalid_date_range_returns_400():
    response = client.get(
        "/api/prices/AAPL?start_date=2024-01-10&end_date=2024-01-01"
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "start_date cannot be later than end_date"


def test_get_prices_with_invalid_symbol_format_returns_422():
    response = client.get(
        "/api/prices/THIS_SYMBOL_IS_TOO_LONG?start_date=2024-01-01&end_date=2024-01-10"
    )

    assert response.status_code == 422
    