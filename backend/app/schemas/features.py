from datetime import date

from pydantic import BaseModel


class FeatureBar(BaseModel):
    date: date
    close: float
    daily_return: float | None = None
    sma_5: float | None = None
    sma_20: float | None = None
    rolling_volatility_20: float | None = None


class FeatureResponse(BaseModel):
    symbol: str
    features: list[FeatureBar]