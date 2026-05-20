import pytest

from app.ml.model_evaluation import calculate_classification_metrics


def test_calculate_classification_metrics_returns_expected_values():
    y_true = [1, 0, 1, 0]
    y_pred = [1, 0, 0, 0]
    y_probability_up = [0.8, 0.2, 0.4, 0.3]

    metrics = calculate_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_probability_up=y_probability_up,
    )

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(0.5)

    assert metrics["true_negative"] == 2
    assert metrics["false_positive"] == 0
    assert metrics["false_negative"] == 1
    assert metrics["true_positive"] == 1

    assert metrics["average_prediction_confidence"] == pytest.approx(0.725)


def test_calculate_classification_metrics_handles_no_positive_predictions():
    y_true = [1, 0, 1, 0]
    y_pred = [0, 0, 0, 0]

    metrics = calculate_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
    )

    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.0)
    assert metrics["recall"] == pytest.approx(0.0)

    assert metrics["true_negative"] == 2
    assert metrics["false_positive"] == 0
    assert metrics["false_negative"] == 2
    assert metrics["true_positive"] == 0

    assert metrics["average_prediction_confidence"] is None