from pydantic import BaseModel


class PriceBar(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class PriceResponse(BaseModel):
    symbol: str
    prices: list[PriceBar]