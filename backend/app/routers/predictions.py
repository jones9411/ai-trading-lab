from fastapi import APIRouter, HTTPException, Path

from app.schemas.predictions import PredictionResponse
from app.services.prediction_service import get_prediction_for_symbol
from app.services.price_service import normalize_symbol

router = APIRouter(
    prefix="/api/predictions",
    tags=["predictions"],
)


@router.get("/{symbol}", response_model=PredictionResponse)
def get_prediction(
    symbol: str = Path(
        ...,
        min_length=1,
        max_length=10,
        pattern=r"^[A-Za-z0-9.\-]+$",
        description="Stock symbol, for example AAPL, MSFT, TSLA, or VOD.L.",
    ),
):
    normalized_symbol = normalize_symbol(symbol)

    try:
        return get_prediction_for_symbol(normalized_symbol)

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error