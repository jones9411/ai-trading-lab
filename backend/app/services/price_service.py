from datetime import date


def get_fake_prices():
    return [
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
    ]


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
    prices = get_fake_prices()

    filtered_prices = filter_prices_by_date(
        prices=prices,
        start_date=start_date,
        end_date=end_date,
    )

    return {
        "symbol": symbol.upper(),
        "prices": filtered_prices,
    }