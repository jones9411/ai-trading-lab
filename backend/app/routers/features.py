from datetime import date

import pandas as pd
from fastapi import APIRouter, HTTPException, Path

from app.schemas.features import FeatureResponse
from app.services.feature_service import create_price_features
from app.services.price_service import (
    get_default_end_date,
    get_default_start_date,
    get_price_data,
    normalize_symbol,
)


router = APIRouter(
    prefix="/api/features",
    tags=["features"],
)


def value_or_none(value):
    if pd.isna(value):
        return None

    return value


def convert_feature_data_to_feature_bars(data: pd.DataFrame) -> list[dict]:
    feature_bars = []

    for row_date, row in data.iterrows():
        feature_bar = {
            "date": row_date.date().isoformat(),
            "close": round(float(row["Close"]), 2),
            "daily_return": value_or_none(row["daily_return"]),
            "sma_5": value_or_none(row["sma_5"]),
            "sma_20": value_or_none(row["sma_20"]),
            "rolling_volatility_20": value_or_none(
                row["rolling_volatility_20"]
            ),
        }

        feature_bars.append(feature_bar)

    return feature_bars


@router.get("/{symbol}", response_model=FeatureResponse)
def get_features(
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
        
    if start_date is None:
        start_date = get_default_start_date()

    if end_date is None:
        end_date = get_default_end_date()

    price_data = get_price_data(
        symbol=normalized_symbol,
        start_date=start_date,
        end_date=end_date,
    )

    if price_data.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No price data found for {normalized_symbol} in the requested date range",
        )

    feature_data = create_price_features(price_data)
    feature_bars = convert_feature_data_to_feature_bars(feature_data)

    return {
        "symbol": normalized_symbol,
        "features": feature_bars,
    }
    