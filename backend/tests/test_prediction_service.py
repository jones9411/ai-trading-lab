import pytest

from app.services.prediction_service import convert_probability_to_signal


def test_convert_probability_to_signal_returns_up_at_threshold():
    signal = convert_probability_to_signal(0.5)

    assert signal == "up"


def test_convert_probability_to_signal_returns_up_above_threshold():
    signal = convert_probability_to_signal(0.75)

    assert signal == "up"


def test_convert_probability_to_signal_returns_not_up_below_threshold():
    signal = convert_probability_to_signal(0.49)

    assert signal == "not_up"


def test_convert_probability_to_signal_supports_custom_threshold():
    signal = convert_probability_to_signal(
        probability_up=0.55,
        threshold=0.6,
    )

    assert signal == "not_up"