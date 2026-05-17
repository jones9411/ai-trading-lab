import pandas as pd


def add_daily_return(data: pd.DataFrame) -> pd.DataFrame:
    featured_data = data.copy()

    featured_data["daily_return"] = featured_data["Close"].pct_change()

    return featured_data


def add_simple_moving_average(
    data: pd.DataFrame,
    window: int,
    column_name: str,
) -> pd.DataFrame:
    featured_data = data.copy()

    featured_data[column_name] = featured_data["Close"].rolling(
        window=window,
    ).mean()

    return featured_data


def add_rolling_volatility(
    data: pd.DataFrame,
    window: int,
    column_name: str,
) -> pd.DataFrame:
    featured_data = data.copy()

    featured_data[column_name] = featured_data["daily_return"].rolling(
        window=window,
    ).std()

    return featured_data


def create_price_features(data: pd.DataFrame) -> pd.DataFrame:
    featured_data = data.copy()

    featured_data = add_daily_return(featured_data)

    featured_data = add_simple_moving_average(
        data=featured_data,
        window=5,
        column_name="sma_5",
    )

    featured_data = add_simple_moving_average(
        data=featured_data,
        window=20,
        column_name="sma_20",
    )

    featured_data = add_rolling_volatility(
        data=featured_data,
        window=20,
        column_name="rolling_volatility_20",
    )

    return featured_data