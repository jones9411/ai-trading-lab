import pytest

from app.ml.risk_management import (
    apply_max_allocation_to_positions,
    apply_stop_loss_to_return,
    calculate_position_size,
    calculate_strategy_return,
    validate_max_allocation,
    validate_research_mode,
    validate_stop_loss_pct,
)


def test_validate_research_mode_does_not_raise():
    validate_research_mode()


def test_calculate_position_size_uses_max_allocation():
    position_size = calculate_position_size(
        account_value=10_000,
        max_allocation=0.50,
    )

    assert position_size == pytest.approx(5_000)


def test_apply_max_allocation_to_positions_converts_long_flat_positions():
    allocations = apply_max_allocation_to_positions(
        positions=[1, 0, 1],
        max_allocation=0.25,
    )

    assert allocations == pytest.approx([0.25, 0.0, 0.25])


def test_apply_stop_loss_to_return_caps_large_loss():
    adjusted_return = apply_stop_loss_to_return(
        asset_return=-0.10,
        stop_loss_pct=0.02,
    )

    assert adjusted_return == pytest.approx(-0.02)


def test_apply_stop_loss_to_return_keeps_smaller_loss():
    adjusted_return = apply_stop_loss_to_return(
        asset_return=-0.01,
        stop_loss_pct=0.02,
    )

    assert adjusted_return == pytest.approx(-0.01)


def test_calculate_strategy_return_uses_allocation_and_stop_loss():
    strategy_return = calculate_strategy_return(
        asset_return=-0.10,
        position_allocation=0.50,
        stop_loss_pct=0.02,
    )

    assert strategy_return == pytest.approx(-0.01)


def test_validate_max_allocation_rejects_invalid_value():
    with pytest.raises(ValueError, match="max_allocation"):
        validate_max_allocation(1.50)


def test_validate_stop_loss_pct_rejects_invalid_value():
    with pytest.raises(ValueError, match="stop_loss_pct"):
        validate_stop_loss_pct(1.25)