from datetime import date

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.ml.model_storage import save_model
from app.ml.backtesting import run_long_flat_backtest
from app.ml.risk_management import DEFAULT_MAX_ALLOCATION, DEFAULT_STOP_LOSS_PCT
from app.ml.dataset_preparation import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_ml_dataset,
    create_train_test_data,
    save_ml_dataset,
)
from app.ml.model_evaluation import calculate_classification_metrics


def train_baseline_model(dataset: pd.DataFrame) -> dict:
    """
    Train a simple baseline machine-learning model.

    We use LogisticRegression because the target is binary:

    1 = up tomorrow
    0 = not up tomorrow
    """

    train_test_data = create_train_test_data(dataset)

    train_data = train_test_data["train_data"]
    test_data = train_test_data["test_data"]

    x_train = train_test_data["x_train"]
    y_train = train_test_data["y_train"]

    x_test = train_test_data["x_test"]
    y_test = train_test_data["y_test"]

    if y_train.nunique() < 2:
        raise ValueError(
            "Training labels contain only one class. "
            "The model needs both 0 and 1 examples."
        )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )

    model.fit(x_train, y_train)

    model_predictions = model.predict(x_test)
    model_probability_up = model.predict_proba(x_test)[:, 1]

    model_metrics = calculate_classification_metrics(
        y_true=y_test,
        y_pred=model_predictions,
        y_probability_up=model_probability_up,
    )

    most_common_training_label = y_train.mode()[0]
    naive_predictions = [most_common_training_label] * len(y_test)

    naive_metrics = calculate_classification_metrics(
        y_true=y_test,
        y_pred=naive_predictions,
    )

    backtest_results = run_long_flat_backtest(
        test_data=test_data,
        predictions=model_predictions,
        max_allocation=DEFAULT_MAX_ALLOCATION,
        stop_loss_pct=DEFAULT_STOP_LOSS_PCT,
    )

    return {
        "model": model,
        "training_rows": len(train_data),
        "test_rows": len(test_data),
        "most_common_training_label": int(most_common_training_label),
        "model_metrics": model_metrics,
        "naive_metrics": naive_metrics,
        "backtest": backtest_results,
    }


def print_metric_summary(title: str, metrics: dict) -> None:
    """
    Print one readable block of classification metrics.
    """

    print(title)
    print("-" * len(title))
    print(f"Accuracy:  {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")

    average_confidence = metrics["average_prediction_confidence"]

    if average_confidence is not None:
        print(f"Average prediction confidence: {average_confidence:.3f}")

    print()
    print("Confusion matrix values:")
    print(f"True negatives:  {metrics['true_negative']}")
    print(f"False positives: {metrics['false_positive']}")
    print(f"False negatives: {metrics['false_negative']}")
    print(f"True positives:  {metrics['true_positive']}")
    print()


def print_backtest_summary(backtest: dict) -> None:
    """
    Print a readable summary of the simple long/flat backtest.
    """

    summary = backtest["summary"]

    print("Simple long/flat backtest with risk controls")
    print("--------------------------------------------")
    print(f"Backtest rows: {summary['backtest_rows']}")
    print(
        "Days in market: "
        f"{summary['days_in_market']} "
        f"({summary['percent_days_in_market']:.1%})"
    )
    print(
        "Configured max allocation: "
        f"{summary['configured_max_allocation']:.1%}"
    )

    stop_loss_pct = summary["configured_stop_loss_pct"]

    if stop_loss_pct is None:
        print("Configured stop-loss: None")
    else:
        print(f"Configured stop-loss: {stop_loss_pct:.1%}")

    print(
        "Average position allocation: "
        f"{summary['average_position_allocation']:.1%}"
    )
    print()
    print(f"Strategy total return:     {summary['strategy_total_return']:.2%}")
    print(f"Buy-and-hold total return: {summary['buy_hold_total_return']:.2%}")
    print()
    print(
        "Strategy average daily return:     "
        f"{summary['strategy_average_daily_return']:.4%}"
    )
    print(
        "Buy-and-hold average daily return: "
        f"{summary['buy_hold_average_daily_return']:.4%}"
    )
    print()
    print(
        "Worst strategy daily return:     "
        f"{summary['worst_strategy_daily_return']:.2%}"
    )
    print(
        "Worst buy-and-hold daily return: "
        f"{summary['worst_buy_hold_daily_return']:.2%}"
    )
    print()

    if summary["strategy_total_return"] > summary["buy_hold_total_return"]:
        print("Backtest result: The strategy beat buy-and-hold.")
    elif summary["strategy_total_return"] < summary["buy_hold_total_return"]:
        print("Backtest result: The strategy did NOT beat buy-and-hold.")
    else:
        print("Backtest result: The strategy matched buy-and-hold.")

    print()


def print_training_summary(
    symbol: str,
    dataset: pd.DataFrame,
    results: dict,
    dataset_path=None,
    model_path=None,
) -> None:
    """
    Print a readable summary of the training run.
    """

    model_metrics = results["model_metrics"]
    naive_metrics = results["naive_metrics"]

    print()
    print("Baseline model training complete")
    print("--------------------------------")
    print(f"Symbol: {symbol}")
    print(f"Rows after cleaning: {len(dataset)}")
    print(f"Feature columns: {FEATURE_COLUMNS}")
    print(f"Target column: {TARGET_COLUMN}")

    if dataset_path is not None:
        print(f"Saved ML dataset: {dataset_path}")
        
    if model_path is not None:
        print(f"Saved model artifact: {model_path}")

    print()
    print(f"Training rows: {results['training_rows']}")
    print(f"Test rows: {results['test_rows']}")
    print()
    print(f"Most common training label: {results['most_common_training_label']}")
    print()

    print_metric_summary(
        title="Naive baseline metrics",
        metrics=naive_metrics,
    )

    print_metric_summary(
        title="Model metrics",
        metrics=model_metrics,
    )

    if model_metrics["accuracy"] > naive_metrics["accuracy"]:
        print("Result: The model beat the naive baseline on accuracy.")
    elif model_metrics["accuracy"] < naive_metrics["accuracy"]:
        print("Result: The model did NOT beat the naive baseline on accuracy.")
    else:
        print("Result: The model matched the naive baseline on accuracy.")

    print()

    print_backtest_summary(results["backtest"])

    print("Important reminder:")
    print("This is a simplified educational backtest.")
    print("It does not include fees, slippage, spread, taxes, or risk controls.")
    print("A real trading system needs much more validation before paper trading or live use.")


def main() -> None:
    symbol = "AAPL"
    start_date = date(2020, 1, 1)
    end_date = date.today()

    dataset = build_ml_dataset(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
    )

    dataset_path = save_ml_dataset(
        dataset=dataset,
        symbol=symbol,
    )

    results = train_baseline_model(dataset)

    model_path = save_model(
        model=results["model"],
        symbol=symbol,
    )

    print_training_summary(
        symbol=symbol,
        dataset=dataset,
        results=results,
        dataset_path=dataset_path,
        model_path=model_path,
    )


if __name__ == "__main__":
    main()