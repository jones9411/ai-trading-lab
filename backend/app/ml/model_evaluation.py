from collections.abc import Sequence

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score


def calculate_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_probability_up: Sequence[float] | None = None,
) -> dict:
    """
    Calculate useful classification metrics for our up/down prediction model.

    y_true:
        The real answers.

    y_pred:
        The model's predicted answers.

    y_probability_up:
        Optional probability that the model assigned to class 1, meaning UP.
    """

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    true_negative = int(matrix[0][0])
    false_positive = int(matrix[0][1])
    false_negative = int(matrix[1][0])
    true_positive = int(matrix[1][1])

    average_prediction_confidence = None

    if y_probability_up is not None:
        prediction_confidences = [
            max(float(probability_up), 1.0 - float(probability_up))
            for probability_up in y_probability_up
        ]

        if len(prediction_confidences) > 0:
            average_prediction_confidence = sum(prediction_confidences) / len(
                prediction_confidences
            )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
        "average_prediction_confidence": average_prediction_confidence,
    }