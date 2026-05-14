from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PRICE_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "prices"


def normalize_symbol(symbol: str) -> str:
    return symbol.upper().strip()


def get_default_start_date() -> date:
    return date.today() - timedelta(days=30)


def get_default_end_date() -> date:
    return date.today()


def get_price_csv_path(symbol: str) -> Path:
    normalized_symbol = normalize_symbol(symbol)
    safe_symbol = normalized_symbol.replace(".", "_").replace("-", "_")
    return PRICE_DATA_DIR / f"{safe_symbol}.csv"


def ensure_price_data_dir_exists() -> None:
    PRICE_DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_price_data_from_yfinance(
    symbol: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    normalized_symbol = normalize_symbol(symbol)

    yf_end_date = end_date + timedelta(days=1)

    ticker = yf.Ticker(normalized_symbol)

    data = ticker.history(
        start=start_date.isoformat(),
        end=yf_end_date.isoformat(),
        interval="1d",
        auto_adjust=False,
    )

    return data


def save_price_data_to_csv(symbol: str, data: pd.DataFrame) -> None:
    ensure_price_data_dir_exists()

    csv_path = get_price_csv_path(symbol)

    data.to_csv(csv_path)


def load_price_data_from_csv(symbol: str) -> pd.DataFrame:
    csv_path = get_price_csv_path(symbol)

    data = pd.read_csv(
        csv_path,
        index_col=0,
        parse_dates=True,
    )

    return data


def local_price_data_exists(symbol: str) -> bool:
    csv_path = get_price_csv_path(symbol)
    return csv_path.exists()


def filter_data_by_date_range(
    data: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    filtered_data = data[
        (data.index.date >= start_date)
        & (data.index.date <= end_date)
    ]

    return filtered_data


def get_price_data(
    symbol: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    if local_price_data_exists(symbol):
        print(f"Loading {symbol} from local CSV")
        data = load_price_data_from_csv(symbol)
    else:
        print(f"Fetching {symbol} from yfinance")
        data = fetch_price_data_from_yfinance(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )

        if not data.empty:
            save_price_data_to_csv(symbol=symbol, data=data)

    filtered_data = filter_data_by_date_range(
        data=data,
        start_date=start_date,
        end_date=end_date,
    )

    return filtered_data


def convert_yfinance_data_to_price_bars(data: pd.DataFrame) -> list[dict]:
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

    if start_date is None:
        start_date = get_default_start_date()

    if end_date is None:
        end_date = get_default_end_date()

    data = get_price_data(
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