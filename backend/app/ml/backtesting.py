from collections.abc import Sequence

import pandas as pd


RETURN_COLUMN = "target_next_day_return"


def create_long_flat_positions(predictions: Sequence[int]) -> list[int]:
    """
    Convert model predictions into long/flat positions.

    Prediction:
    1 = model predicts up, so we go long
    0 = model predicts not up, so we stay flat

    Position:
    1 = invested
    0 = not invested
    """

    return [1 if int(prediction) == 1 else 0 for prediction in predictions]


def calculate_total_return(returns: pd.Series) -> float:
    """
    Calculate compounded total return from a series of daily returns.

    Example:
    Day 1 return: 10%
    Day 2 return: 5%

    Total return:
    (1 + 0.10) * (1 + 0.05) - 1
    """

    if returns.empty:
        return 0.0

    return float((1 + returns).prod() - 1)


def build_long_flat_backtest_frame(
    test_data: pd.DataFrame,
    predictions: Sequence[int],
    return_column: str = RETURN_COLUMN,
) -> pd.DataFrame:
    """
    Build a DataFrame showing strategy returns and buy-and-hold returns.

    The model strategy is:
    - prediction 1 -> long
    - prediction 0 -> flat
    """

    if return_column not in test_data.columns:
        raise ValueError(f"Missing required return column: {return_column}")

    if len(test_data) != len(predictions):
        raise ValueError(
            "Prediction count must match the number of test rows. "
            f"Got {len(predictions)} predictions for {len(test_data)} rows."
        )

    backtest_data = test_data.copy()

    backtest_data["prediction"] = [int(prediction) for prediction in predictions]
    backtest_data["position"] = create_long_flat_positions(predictions)

    backtest_data["strategy_return"] = (
        backtest_data["position"] * backtest_data[return_column]
    )

    backtest_data["buy_hold_return"] = backtest_data[return_column]

    backtest_data["strategy_equity_curve"] = (
        1 + backtest_data["strategy_return"]
    ).cumprod()

    backtest_data["buy_hold_equity_curve"] = (
        1 + backtest_data["buy_hold_return"]
    ).cumprod()

    return backtest_data


def summarize_backtest(backtest_data: pd.DataFrame) -> dict:
    """
    Create a simple summary of backtest performance.
    """

    if backtest_data.empty:
        raise ValueError("Backtest data is empty.")

    strategy_total_return = calculate_total_return(
        backtest_data["strategy_return"]
    )

    buy_hold_total_return = calculate_total_return(
        backtest_data["buy_hold_return"]
    )

    total_days = len(backtest_data)
    days_in_market = int(backtest_data["position"].sum())

    return {
        "backtest_rows": total_days,
        "days_in_market": days_in_market,
        "percent_days_in_market": days_in_market / total_days,
        "strategy_total_return": strategy_total_return,
        "buy_hold_total_return": buy_hold_total_return,
        "strategy_average_daily_return": float(
            backtest_data["strategy_return"].mean()
        ),
        "buy_hold_average_daily_return": float(
            backtest_data["buy_hold_return"].mean()
        ),
    }


def run_long_flat_backtest(
    test_data: pd.DataFrame,
    predictions: Sequence[int],
) -> dict:
    """
    Run a simple long/flat backtest.

    Returns:
    - backtest_data: row-by-row backtest results
    - summary: headline performance numbers
    """

    backtest_data = build_long_flat_backtest_frame(
        test_data=test_data,
        predictions=predictions,
    )

    summary = summarize_backtest(backtest_data)

    return {
        "backtest_data": backtest_data,
        "summary": summary,
    }
    