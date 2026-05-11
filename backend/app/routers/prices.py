from datetime import date

from fastapi import APIRouter, HTTPException

from app.schemas.prices import PriceResponse

router = APIRouter(
    prefix="/api/prices",
    tags=["prices"],
)


@router.get("/{symbol}", response_model=PriceResponse)
def get_prices(
    symbol: str,
    start_date: date | None = None,
    end_date: date | None = None,
):
    if start_date is not None and end_date is not None and start_date > end_date:
        raise HTTPException(
            status_code=400,
            detail="start_date cannot be later than end_date",
        )

    fake_prices = [
        {
            "date": "2026-04-01",
            "open": 188.20,
            "high": 191.10,
            "low": 187.50,
            "close": 190.25,
            "volume": 53200000,
        },
        {
            "date": "2026-04-02",
            "open": 190.25,
            "high": 193.40,
            "low": 189.80,
            "close": 192.75,
            "volume": 48700000,
        },
        {
            "date": "2026-04-03",
            "open": 192.75,
            "high": 194.20,
            "low": 190.60,
            "close": 191.10,
            "volume": 50100000,
        },
    ]

    filtered_prices = []

    for price in fake_prices:
        price_date = date.fromisoformat(price["date"])

        if start_date is not None and price_date < start_date:
            continue

        if end_date is not None and price_date > end_date:
            continue

        filtered_prices.append(price)

    return {
        "symbol": symbol.upper(),
        "prices": filtered_prices,
    }