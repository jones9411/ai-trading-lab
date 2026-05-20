import pandas as pd
import pytest

from app.services.label_service import create_price_labels


def test_create_price_labels_adds_expected_columns():
    data = pd.DataFrame(
        {
            "Close": [
                100,
                105,
                103,
            ]
        }
    )

    labelled_data = create_price_labels(data)

    assert "target_next_day_return" in labelled_data.columns
    assert "target_up_next_day" in labelled_data.columns


def test_next_day_return_is_calculated_correctly():
    data = pd.DataFrame(
        {
            "Close": [
                100,
                110,
            ]
        }
    )

    labelled_data = create_price_labels(data)

    assert labelled_data.loc[0, "target_next_day_return"] == pytest.approx(0.1)


def test_target_up_next_day_is_1_when_next_day_close_is_higher():
    data = pd.DataFrame(
        {
            "Close": [
                100,
                110,
            ]
        }
    )

    labelled_data = create_price_labels(data)

    assert labelled_data.loc[0, "target_up_next_day"] == 1


def test_target_up_next_day_is_0_when_next_day_close_is_lower():
    data = pd.DataFrame(
        {
            "Close": [
                100,
                90,
            ]
        }
    )

    labelled_data = create_price_labels(data)

    assert labelled_data.loc[0, "target_up_next_day"] == 0


def test_last_row_target_is_missing_because_next_day_is_unknown():
    data = pd.DataFrame(
        {
            "Close": [
                100,
                110,
            ]
        }
    )

    labelled_data = create_price_labels(data)

    assert pd.isna(labelled_data.loc[1, "target_next_day_return"])
    assert pd.isna(labelled_data.loc[1, "target_up_next_day"])