import numpy as np
import pandas as pd

from alpha_factor_engine.research.metrics import (
    max_drawdown,
    moving_block_bootstrap_sharpe,
    performance_summary,
)


def test_max_drawdown_includes_initial_wealth():
    returns = pd.Series([-0.10, 0.05])

    assert np.isclose(max_drawdown(returns), -0.10)


def test_performance_summary_uses_arithmetic_sharpe_and_downside_deviation():
    monthly = np.array([0.10, -0.05, 0.02, -0.01], dtype=float)
    frame = pd.DataFrame(
        {
            "net_return": monthly,
            "turnover": 0.0,
            "trading_cost": 0.0,
            "borrow_cost": 0.0,
        }
    )

    summary = performance_summary(frame)
    annualized_arithmetic_return = monthly.mean() * 12
    annualized_volatility = monthly.std(ddof=1) * np.sqrt(12)
    downside_deviation = np.sqrt(np.mean(np.minimum(monthly, 0.0) ** 2)) * np.sqrt(12)

    assert np.isclose(summary["annualized_arithmetic_return"], annualized_arithmetic_return)
    assert np.isclose(summary["sharpe"], annualized_arithmetic_return / annualized_volatility)
    assert np.isclose(summary["sortino"], annualized_arithmetic_return / downside_deviation)


def test_bootstrap_returns_nan_bounds_for_zero_variance_series_without_warning():
    summary = moving_block_bootstrap_sharpe(pd.Series(np.zeros(24)))

    assert all(np.isnan(value) for value in summary.values())
