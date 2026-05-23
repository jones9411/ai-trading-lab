RESEARCH_MODE_ONLY = True

DEFAULT_INITIAL_CAPITAL = 10_000.0
DEFAULT_MAX_ALLOCATION = 0.50
DEFAULT_STOP_LOSS_PCT = 0.02


def validate_research_mode() -> None:
    """
    Guardrail for this project.

    This project is currently for research, learning, backtesting,
    and paper-trading preparation only.
    """

    if not RESEARCH_MODE_ONLY:
        raise RuntimeError(
            "Live trading is not enabled in this project. "
            "This system is research/paper-trading only."
        )


def validate_max_allocation(max_allocation: float) -> None:
    """
    Validate that max_allocation is between 0 and 1.

    Example:
    0.50 means 50% of capital.
    """

    if max_allocation < 0 or max_allocation > 1:
        raise ValueError("max_allocation must be between 0 and 1.")


def validate_stop_loss_pct(stop_loss_pct: float | None) -> None:
    """
    Validate a stop-loss percentage.

    Example:
    0.02 means 2%.
    """

    if stop_loss_pct is None:
        return

    if stop_loss_pct <= 0 or stop_loss_pct >= 1:
        raise ValueError("stop_loss_pct must be greater than 0 and less than 1.")


def calculate_position_size(
    account_value: float,
    max_allocation: float = DEFAULT_MAX_ALLOCATION,
) -> float:
    """
    Calculate maximum position size based on account value and allocation.

    Example:
    account_value = 10000
    max_allocation = 0.50

    position_size = 5000
    """

    if account_value <= 0:
        raise ValueError("account_value must be greater than 0.")

    validate_max_allocation(max_allocation)

    return float(account_value * max_allocation)


def apply_max_allocation_to_positions(
    positions: list[int],
    max_allocation: float = DEFAULT_MAX_ALLOCATION,
) -> list[float]:
    """
    Convert long/flat positions into position allocations.

    position 1 means long, so use max_allocation.
    position 0 means flat, so use 0 allocation.
    """

    validate_max_allocation(max_allocation)

    return [
        max_allocation if int(position) == 1 else 0.0
        for position in positions
    ]


def apply_stop_loss_to_return(
    asset_return: float,
    stop_loss_pct: float | None = DEFAULT_STOP_LOSS_PCT,
) -> float:
    """
    Apply a simplified stop-loss to a single asset return.

    Example:
    asset_return = -0.10
    stop_loss_pct = 0.02

    adjusted return = -0.02

    This is simplified because daily data cannot perfectly simulate
    real intraday stop-loss execution.
    """

    validate_stop_loss_pct(stop_loss_pct)

    if stop_loss_pct is None:
        return float(asset_return)

    return float(max(asset_return, -stop_loss_pct))


def calculate_strategy_return(
    asset_return: float,
    position_allocation: float,
    stop_loss_pct: float | None = DEFAULT_STOP_LOSS_PCT,
) -> float:
    """
    Calculate the portfolio return for one strategy row.

    Example:
    asset_return = 0.04
    position_allocation = 0.50

    strategy_return = 0.02
    """

    validate_max_allocation(position_allocation)

    adjusted_asset_return = apply_stop_loss_to_return(
        asset_return=asset_return,
        stop_loss_pct=stop_loss_pct,
    )

    return float(position_allocation * adjusted_asset_return)
