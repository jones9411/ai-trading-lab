from datetime import date, timedelta

import yfinance as yf


def normalize_symbol(symbol: str) -> str:
    return symbol.upper().strip()


def get_default_start_date() -> date:
    return date.today() - timedelta(days=30)


def get_default_end_date() -> date:
    return date.today()


def fetch_price_data_from_yfinance(
    symbol: str,
    start_date: date | None = None,
    end_date: date | None = None,
):
    normalized_symbol = normalize_symbol(symbol)

    if start_date is None:
        start_date = get_default_start_date()

    if end_date is None:
        end_date = get_default_end_date()

    # yfinance treats the end date like an exclusive boundary.
    # Our API is easier for beginners if end_date is inclusive,
    # so we add one day before sending it to yfinance.
    yf_end_date = end_date + timedelta(days=1)

    ticker = yf.Ticker(normalized_symbol)

    data = ticker.history(
        start=start_date.isoformat(),
        end=yf_end_date.isoformat(),
        interval="1d",
        auto_adjust=False,
    )

    return data


def convert_yfinance_data_to_price_bars(data):
    price_bars = []

    for row_date, row in data.iterrows():
        price_bar = {
            "date": row_date.date().isoformat(),
            "open": round(float(row["Open"]), 2),
            "high": round(float(row["High"]), 2),
            "low": round(float(row["Low"]), 2),
            "close": round(float(row["Close"]), 2),
            "volume": int(row["Volume"]),
        }

        price_bars.append(price_bar)

    return price_bars


def get_prices_for_symbol(
    symbol: str,
    start_date: date | None = None,
    end_date: date | None = None,
):
    normalized_symbol = normalize_symbol(symbol)

    data = fetch_price_data_from_yfinance(
        symbol=normalized_symbol,
        start_date=start_date,
        end_date=end_date,
    )

    if data.empty:
        return {
            "symbol": normalized_symbol,
            "prices": [],
        }

    price_bars = convert_yfinance_data_to_price_bars(data)

    return {
        "symbol": normalized_symbol,
        "prices": price_bars,
    }