from datetime import date


FAKE_PRICE_DATA = {
    "AAPL": [
        {
            "date": "2026-04-01",
            "open": 188.20,
            "high": 191.10,
            "low": 187.50,
            "close": 190.25,
            "volume": 53200000,
        },
        {
            "date": "2026-04-02",
            "open": 190.25,
            "high": 193.40,
            "low": 189.80,
            "close": 192.75,
            "volume": 48700000,
        },
        {
            "date": "2026-04-03",
            "open": 192.75,
            "high": 194.20,
            "low": 190.60,
            "close": 191.10,
            "volume": 50100000,
        },
    ],
    "MSFT": [
        {
            "date": "2026-04-01",
            "open": 415.10,
            "high": 420.30,
            "low": 412.80,
            "close": 418.40,
            "volume": 28100000,
        },
        {
            "date": "2026-04-02",
            "open": 418.40,
            "high": 422.75,
            "low": 416.90,
            "close": 421.20,
            "volume": 26400000,
        },
        {
            "date": "2026-04-03",
            "open": 421.20,
            "high": 423.50,
            "low": 417.60,
            "close": 419.85,
            "volume": 29700000,
        },
    ],
    "TSLA": [
        {
            "date": "2026-04-01",
            "open": 172.50,
            "high": 178.20,
            "low": 169.90,
            "close": 176.35,
            "volume": 91200000,
        },
        {
            "date": "2026-04-02",
            "open": 176.35,
            "high": 181.00,
            "low": 174.10,
            "close": 179.60,
            "volume": 87400000,
        },
        {
            "date": "2026-04-03",
            "open": 179.60,
            "high": 180.40,
            "low": 171.80,
            "close": 173.25,
            "volume": 95800000,
        },
    ],
}


def normalize_symbol(symbol: str) -> str:
    return symbol.upper().strip()


def has_price_data_for_symbol(symbol: str) -> bool:
    normalized_symbol = normalize_symbol(symbol)
    return normalized_symbol in FAKE_PRICE_DATA


def get_fake_prices(symbol: str):
    normalized_symbol = normalize_symbol(symbol)
    return FAKE_PRICE_DATA.get(normalized_symbol, [])


def filter_prices_by_date(
    prices: list[dict],
    start_date: date | None = None,
    end_date: date | None = None,
):
    filtered_prices = []

    for price in prices:
        price_date = date.fromisoformat(price["date"])

        if start_date is not None and price_date < start_date:
            continue

        if end_date is not None and price_date > end_date:
            continue

        filtered_prices.append(price)

    return filtered_prices


def get_prices_for_symbol(
    symbol: str,
    start_date: date | None = None,
    end_date: date | None = None,
):
    normalized_symbol = normalize_symbol(symbol)

    prices = get_fake_prices(normalized_symbol)

    filtered_prices = filter_prices_by_date(
        prices=prices,
        start_date=start_date,
        end_date=end_date,
    )

    return {
        "symbol": normalized_symbol,
        "prices": filtered_prices,
    }