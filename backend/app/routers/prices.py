from datetime import date

from fastapi import APIRouter, HTTPException

from app.schemas.prices import PriceResponse
from app.services.price_service import get_prices_for_symbol

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

    return get_prices_for_symbol(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
    )