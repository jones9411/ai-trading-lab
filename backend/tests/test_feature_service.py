import pytest
import pandas as pd

from app.services.feature_service import create_price_features


def test_create_price_features_adds_expected_columns():
    data = pd.DataFrame(
        {
            "Close": [
                100,
                101,
                102,
                103,
                104,
                105,
                106,
                107,
                108,
                109,
                110,
                111,
                112,
                113,
                114,
                115,
                116,
                117,
                118,
                119,
                120,
            ]
        }
    )

    featured_data = create_price_features(data)

    assert "daily_return" in featured_data.columns
    assert "sma_5" in featured_data.columns
    assert "sma_20" in featured_data.columns
    assert "rolling_volatility_20" in featured_data.columns


def test_daily_return_is_calculated_correctly():
    data = pd.DataFrame(
        {
            "Close": [
                100,
                110,
            ]
        }
    )

    featured_data = create_price_features(data)

    assert featured_data.loc[1, "daily_return"] == pytest.approx(0.1)


def test_sma_5_is_calculated_correctly():
    data = pd.DataFrame(
        {
            "Close": [
                100,
                101,
                102,
                103,
                104,
            ]
        }
    )

    featured_data = create_price_features(data)

    assert featured_data.loc[4, "sma_5"] == 102