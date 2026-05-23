import pandas as pd
import pytest

from app.ml.backtesting import (
    build_long_flat_backtest_frame,
    calculate_total_return,
    create_long_flat_positions,
    run_long_flat_backtest,
)


def test_create_long_flat_positions_converts_predictions_to_positions():
    predictions = [1, 0, 1, 0]

    positions = create_long_flat_positions(predictions)

    assert positions == [1, 0, 1, 0]


def test_calculate_total_return_compounds_returns():
    returns = pd.Series([0.10, 0.05])

    total_return = calculate_total_return(returns)

    assert total_return == pytest.approx(0.155)


def test_build_long_flat_backtest_frame_calculates_strategy_returns():
    test_data = pd.DataFrame(
        {
            "target_next_day_return": [0.10, -0.05, 0.02],
        }
    )
    predictions = [1, 0, 1]

    backtest_data = build_long_flat_backtest_frame(
        test_data=test_data,
        predictions=predictions,
    )

    assert backtest_data["position"].tolist() == [1, 0, 1]
    assert backtest_data["strategy_return"].tolist() == pytest.approx(
        [0.10, 0.00, 0.02]
    )
    assert backtest_data["buy_hold_return"].tolist() == pytest.approx(
        [0.10, -0.05, 0.02]
    )


def test_run_long_flat_backtest_returns_summary():
    test_data = pd.DataFrame(
        {
            "target_next_day_return": [0.10, -0.05, 0.02],
        }
    )
    predictions = [1, 0, 1]

    results = run_long_flat_backtest(
        test_data=test_data,
        predictions=predictions,
    )

    summary = results["summary"]

    assert summary["backtest_rows"] == 3
    assert summary["days_in_market"] == 2
    assert summary["percent_days_in_market"] == pytest.approx(2 / 3)
    assert summary["strategy_total_return"] == pytest.approx(0.122)
    assert summary["buy_hold_total_return"] == pytest.approx(0.0659)


def test_backtest_raises_error_when_prediction_count_does_not_match_rows():
    test_data = pd.DataFrame(
        {
            "target_next_day_return": [0.10, -0.05, 0.02],
        }
    )
    predictions = [1, 0]

    with pytest.raises(ValueError, match="Prediction count must match"):
        build_long_flat_backtest_frame(
            test_data=test_data,
            predictions=predictions,
        )


def test_backtest_raises_error_when_return_column_is_missing():
    test_data = pd.DataFrame(
        {
            "some_other_column": [0.10, -0.05, 0.02],
        }
    )
    predictions = [1, 0, 1]

    with pytest.raises(ValueError, match="Missing required return column"):
        build_long_flat_backtest_frame(
            test_data=test_data,
            predictions=predictions,
        )