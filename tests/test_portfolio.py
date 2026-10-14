from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from alpha_factor_engine.research.portfolio import build_long_short_portfolio


def _portfolio_predictions(months: int = 2, names: int = 50) -> pd.DataFrame:
    rows = []
    signal_dates = pd.date_range("2023-01-31", periods=months, freq="ME", tz="UTC")
    entry_dates = pd.date_range("2023-02-01", periods=months, freq="MS", tz="UTC")
    exit_dates = pd.date_range("2023-03-01", periods=months, freq="MS", tz="UTC")
    for month_index, (signal_date, entry_date, exit_date) in enumerate(
        zip(signal_dates, entry_dates, exit_dates)
    ):
        for index in range(names):
            target_return = 0.0
            if month_index == 0:
                target_return = (index - names / 2) / 500
            rows.append(
                {
                    "signal_date": signal_date,
                    "ticker": f"T{index:02d}",
                    "ensemble_score": index,
                    "target_return": target_return,
                    "beta_12m": 0.8 + 0.1 * (index % 2),
                    "log_market_cap": 20.0 + 0.1 * (index % 5),
                    "entry_at": entry_date + timedelta(hours=14, minutes=30),
                    "exit_at": exit_date + timedelta(hours=14, minutes=30),
                }
            )
    return pd.DataFrame(rows)


def test_portfolio_is_dollar_neutral_and_charges_entry_and_terminal_costs():
    cost_bps = 15.0
    predictions = _portfolio_predictions(months=1)
    predictions["target_return"] = 0.0
    returns, holdings = build_long_short_portfolio(
        predictions,
        trading_cost_bps_one_way=cost_bps,
        annual_short_borrow_bps=0.0,
    )
    exposure = holdings.groupby("signal_date")["weight"].sum()

    assert np.allclose(exposure, 0, atol=1e-10)
    assert holdings["weight"].abs().max() <= 0.10 + 1e-10
    assert np.isclose(returns.loc[0, "beta_exposure"], 0.0, atol=1e-8)
    assert np.isclose(returns.loc[0, "size_exposure"], 0.0, atol=1e-8)
    assert np.isclose(returns.loc[0, "turnover"], 2.0)
    entry_cost = cost_bps / 10_000
    expected_liquidation = entry_cost * (1 - entry_cost)
    assert np.isclose(returns.loc[0, "terminal_liquidation_turnover"], 1.0)
    assert np.isclose(returns.loc[0, "terminal_liquidation_cost"], expected_liquidation)
    assert np.isclose(returns.loc[0, "trading_cost"], entry_cost + expected_liquidation)


def test_portfolio_turnover_uses_drifted_pretrade_weights():
    cost_bps = 100.0
    predictions = _portfolio_predictions(months=2)
    returns, holdings = build_long_short_portfolio(
        predictions,
        trading_cost_bps_one_way=cost_bps,
        annual_short_borrow_bps=0.0,
    )

    dates = sorted(holdings["signal_date"].unique())
    first = holdings.loc[holdings["signal_date"].eq(dates[0])].set_index("ticker")
    second = holdings.loc[holdings["signal_date"].eq(dates[1])].set_index("ticker")
    first_gross_return = float((first["weight"] * first["target_return"]).sum())
    drifted = first["weight"] * (1 + first["target_return"]) / (1 + first_gross_return)
    expected_transition = float((second["weight"] - drifted).abs().sum())
    liquidation_turnover = returns.loc[1, "terminal_liquidation_turnover"]
    reported_transition = returns.loc[1, "turnover"] - liquidation_turnover

    assert expected_transition > 0
    assert np.isclose(reported_transition, expected_transition)


def test_infeasible_max_name_weight_fails_closed():
    with pytest.raises(ValueError, match="infeasible|max_name_weight"):
        build_long_short_portfolio(
            _portfolio_predictions(months=1, names=20),
            max_name_weight=0.10,
            annual_short_borrow_bps=0.0,
        )


def test_portfolio_handles_constant_exposure_columns():
    predictions = _portfolio_predictions(months=1)
    predictions["beta_12m"] = 1.0
    predictions["log_market_cap"] = 20.0

    returns, holdings = build_long_short_portfolio(
        predictions,
        annual_short_borrow_bps=0.0,
    )

    assert np.isclose(holdings["weight"].sum(), 0.0)
    assert np.isclose(returns.loc[0, "beta_exposure"], 0.0)
    assert np.isclose(returns.loc[0, "size_exposure"], 0.0)


def test_portfolio_rejects_partial_future_label_coverage():
    rows = []
    for index in range(20):
        rows.append(
            {
                "signal_date": pd.Timestamp("2024-01-31", tz="UTC"),
                "ticker": f"T{index:02d}",
                "ensemble_score": index,
                "target_return": np.nan if index == 0 else index / 100,
                "beta_12m": 1.0,
                "log_market_cap": 20.0,
                "entry_at": pd.Timestamp("2024-02-01T14:30:00Z"),
                "exit_at": pd.Timestamp("2024-03-01T14:30:00Z"),
            }
        )
    with pytest.raises(RuntimeError, match="partial target coverage"):
        build_long_short_portfolio(pd.DataFrame(rows))
