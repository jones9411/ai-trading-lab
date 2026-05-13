from datetime import date

from fastapi import APIRouter, HTTPException, Path

from app.schemas.prices import PriceResponse
from app.services.price_service import get_prices_for_symbol, normalize_symbol

router = APIRouter(
    prefix="/api/prices",
    tags=["prices"],
)


@router.get("/{symbol}", response_model=PriceResponse)
def get_prices(
    symbol: str = Path(
        ...,
        min_length=1,
        max_length=10,
        pattern=r"^[A-Za-z0-9.\-]+$",
        description="Stock symbol, for example AAPL, MSFT, TSLA, or VOD.L.",
    ),
    start_date: date | None = None,
    end_date: date | None = None,
):
    normalized_symbol = normalize_symbol(symbol)

    if start_date is not None and end_date is not None and start_date > end_date:
        raise HTTPException(
            status_code=400,
            detail="start_date cannot be later than end_date",
        )

    response = get_prices_for_symbol(
        symbol=normalized_symbol,
        start_date=start_date,
        end_date=end_date,
    )

    if len(response["prices"]) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No prices found for {normalized_symbol} in the requested date range",
        )

    return response