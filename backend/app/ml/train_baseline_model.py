from datetime import date

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.ml.model_evaluation import calculate_classification_metrics
from app.services.feature_service import create_price_features
from app.services.label_service import create_price_labels
from app.services.price_service import get_price_data, normalize_symbol


FEATURE_COLUMNS = [
    "daily_return",
    "sma_5",
    "sma_20",
    "rolling_volatility_20",
]

TARGET_COLUMN = "target_up_next_day"


def build_training_dataset(
    symbol: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Build a clean machine-learning dataset for one stock symbol.

    This function:
    1. Loads price data
    2. Creates feature columns
    3. Creates label columns
    4. Removes rows with missing feature or target values
    """

    normalized_symbol = normalize_symbol(symbol)

    price_data = get_price_data(
        symbol=normalized_symbol,
        start_date=start_date,
        end_date=end_date,
    )

    featured_data = create_price_features(price_data)
    labeled_data = create_price_labels(featured_data)

    dataset = labeled_data.dropna(
        subset=FEATURE_COLUMNS + [TARGET_COLUMN]
    ).copy()

    dataset[TARGET_COLUMN] = dataset[TARGET_COLUMN].astype(int)

    return dataset


def split_train_test_by_time(
    dataset: pd.DataFrame,
    train_ratio: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into training and test sets while keeping time order.

    The first 80% of rows are used for training.
    The final 20% of rows are used for testing.
    """

    if dataset.empty:
        raise ValueError("Dataset is empty. Cannot split empty data.")

    split_index = int(len(dataset) * train_ratio)

    train_data = dataset.iloc[:split_index]
    test_data = dataset.iloc[split_index:]

    if train_data.empty or test_data.empty:
        raise ValueError("Train/test split failed. Not enough rows available.")

    return train_data, test_data


def train_baseline_model(dataset: pd.DataFrame) -> dict:
    """
    Train a simple baseline machine-learning model.

    We use LogisticRegression because the target is binary:

    1 = up tomorrow
    0 = not up tomorrow
    """

    train_data, test_data = split_train_test_by_time(dataset)

    x_train = train_data[FEATURE_COLUMNS]
    y_train = train_data[TARGET_COLUMN]

    x_test = test_data[FEATURE_COLUMNS]
    y_test = test_data[TARGET_COLUMN]

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

    return {
        "model": model,
        "training_rows": len(train_data),
        "test_rows": len(test_data),
        "most_common_training_label": int(most_common_training_label),
        "model_metrics": model_metrics,
        "naive_metrics": naive_metrics,
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


def print_training_summary(
    symbol: str,
    dataset: pd.DataFrame,
    results: dict,
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
    print("Important reminder:")
    print("Accuracy, precision, recall, and confidence do not mean profit.")
    print("A trading system also needs backtesting, fees, slippage, and risk controls.")


def main() -> None:
    symbol = "AAPL"
    start_date = date(2020, 1, 1)
    end_date = date.today()

    dataset = build_training_dataset(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
    )

    results = train_baseline_model(dataset)

    print_training_summary(
        symbol=symbol,
        dataset=dataset,
        results=results,
    )


if __name__ == "__main__":
    main()