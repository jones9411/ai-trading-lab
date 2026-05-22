from datetime import date

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    symbol: str
    date: date
    probability_up: float = Field(..., ge=0.0, le=1.0)
    signal: str