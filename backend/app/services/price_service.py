import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PRICE_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "prices"

MARKET_CLOSED_TOLERANCE_DAYS = 5

logger = logging.getLogger(__name__)

def normalize_symbol(symbol: str) -> str:
    return symbol.upper().strip()


def get_default_start_date() -> date:
    return date.today() - timedelta(days=30)


def get_default_end_date() -> date:
    return date.today()


def get_price_csv_path(symbol: str) -> Path:
    normalized_symbol = normalize_symbol(symbol)
    safe_symbol = normalized_symbol.replace(".", "_")
    return PRICE_DATA_DIR / f"{safe_symbol}.csv"


def ensure_price_data_dir_exists() -> None:
    PRICE_DATA_DIR.mkdir(parents=True, exist_ok=True)


def clean_price_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data

    cleaned_data = data.copy()

    datetime_index = pd.to_datetime(cleaned_data.index)

    if datetime_index.tz is not None:
        datetime_index = datetime_index.tz_localize(None)

    cleaned_data.index = datetime_index.normalize()

    cleaned_data = cleaned_data.sort_index()
    cleaned_data = cleaned_data[~cleaned_data.index.duplicated(keep="last")]

    return cleaned_data


def fetch_price_data_from_yfinance(
    symbol: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    normalized_symbol = normalize_symbol(symbol)

    logger.info(
        "Fetching price data from yfinance for %s from %s to %s",
        normalized_symbol,
        start_date,
        end_date,
    )

    yf_end_date = end_date + timedelta(days=1)

    ticker = yf.Ticker(normalized_symbol)

    data = ticker.history(
        start=start_date.isoformat(),
        end=yf_end_date.isoformat(),
        interval="1d",
        auto_adjust=False,
    )

    cleaned_data = clean_price_dataframe(data)

    logger.info(
        "Fetched %s rows from yfinance for %s",
        len(cleaned_data),
        normalized_symbol,
    )

    return cleaned_data


def save_price_data_to_csv(symbol: str, data: pd.DataFrame) -> None:
    ensure_price_data_dir_exists()

    csv_path = get_price_csv_path(symbol)
    cleaned_data = clean_price_dataframe(data)

    logger.info(
        "Saving %s rows of price data for %s to %s",
        len(cleaned_data),
        normalize_symbol(symbol),
        csv_path,
    )

    cleaned_data.to_csv(csv_path, index_label="Date")


def load_price_data_from_csv(symbol: str) -> pd.DataFrame:
    csv_path = get_price_csv_path(symbol)

    logger.info(
        "Loading local price data for %s from %s",
        normalize_symbol(symbol),
        csv_path,
    )

    data = pd.read_csv(
        csv_path,
        index_col=0,
        parse_dates=True,
    )

    cleaned_data = clean_price_dataframe(data)

    logger.info(
        "Loaded %s rows of local price data for %s",
        len(cleaned_data),
        normalize_symbol(symbol),
    )

    return cleaned_data

def local_price_data_exists(symbol: str) -> bool:
    csv_path = get_price_csv_path(symbol)
    return csv_path.exists()


def filter_data_by_date_range(
    data: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    if data.empty:
        return data

    start_timestamp = pd.Timestamp(start_date)
    end_timestamp = pd.Timestamp(end_date)

    return data[
        (data.index >= start_timestamp)
        & (data.index <= end_timestamp)
    ]


def cached_data_covers_date_range(
    data: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> bool:
    if data.empty:
        return False

    cached_start_date = data.index.min().date()
    cached_end_date = data.index.max().date()

    start_is_covered = cached_start_date <= (
        start_date + timedelta(days=MARKET_CLOSED_TOLERANCE_DAYS)
    )

    end_is_covered = cached_end_date >= (
        end_date - timedelta(days=MARKET_CLOSED_TOLERANCE_DAYS)
    )

    return start_is_covered and end_is_covered


def combine_price_data(
    old_data: pd.DataFrame,
    new_data: pd.DataFrame,
) -> pd.DataFrame:
    combined_data = pd.concat([old_data, new_data])
    combined_data = clean_price_dataframe(combined_data)

    return combined_data


def get_price_data(
    symbol: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    normalized_symbol = normalize_symbol(symbol)

    logger.info(
        "Getting price data for %s from %s to %s",
        normalized_symbol,
        start_date,
        end_date,
    )

    if local_price_data_exists(normalized_symbol):
        logger.info("Local cache file exists for %s", normalized_symbol)

        cached_data = load_price_data_from_csv(normalized_symbol)

        if cached_data_covers_date_range(
            data=cached_data,
            start_date=start_date,
            end_date=end_date,
        ):
            logger.info(
                "Using local cached price data for %s",
                normalized_symbol,
            )

            return filter_data_by_date_range(
                data=cached_data,
                start_date=start_date,
                end_date=end_date,
            )

        logger.info(
            "Local cache for %s does not fully cover requested date range. Fetching fresh data.",
            normalized_symbol,
        )

        fetched_data = fetch_price_data_from_yfinance(
            symbol=normalized_symbol,
            start_date=start_date,
            end_date=end_date,
        )

        if fetched_data.empty:
            logger.warning(
                "yfinance returned no new data for %s from %s to %s. Returning filtered cached data.",
                normalized_symbol,
                start_date,
                end_date,
            )

            return filter_data_by_date_range(
                data=cached_data,
                start_date=start_date,
                end_date=end_date,
            )

        combined_data = combine_price_data(
            old_data=cached_data,
            new_data=fetched_data,
        )

        save_price_data_to_csv(
            symbol=normalized_symbol,
            data=combined_data,
        )

        return filter_data_by_date_range(
            data=combined_data,
            start_date=start_date,
            end_date=end_date,
        )

    logger.info(
        "No local cache file found for %s. Fetching from yfinance.",
        normalized_symbol,
    )

    fetched_data = fetch_price_data_from_yfinance(
        symbol=normalized_symbol,
        start_date=start_date,
        end_date=end_date,
    )

    if fetched_data.empty:
        logger.warning(
            "No price data found for %s from %s to %s",
            normalized_symbol,
            start_date,
            end_date,
        )
    else:
        save_price_data_to_csv(
            symbol=normalized_symbol,
            data=fetched_data,
        )

    return filter_data_by_date_range(
        data=fetched_data,
        start_date=start_date,
        end_date=end_date,
    )

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