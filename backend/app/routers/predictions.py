import logging

from fastapi import APIRouter, HTTPException, Path

from app.schemas.predictions import PredictionResponse
from app.services.prediction_service import get_prediction_for_symbol


logger = logging.getLogger(__name__)

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
    try:
        return get_prediction_for_symbol(symbol)

    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    except Exception as error:
        logger.exception("Prediction failed for symbol %s", symbol)

        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check backend logs for details.",
        ) from error