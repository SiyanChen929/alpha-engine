import numpy as np
import pandas as pd
import pytest

from alpha_factor_engine.research.momentum import (
    build_momentum_champion,
    compute_12_1_momentum,
    run_momentum_champion,
)


def _signal_inputs(names: int = 8):
    tickers = [f"T{index:02d}" for index in range(names)]
    dates = {
        "twelve_month_date": pd.Timestamp("2023-01-03"),
        "one_month_date": pd.Timestamp("2023-12-01"),
        "formation_date": pd.Timestamp("2023-12-29"),
    }
    signal_at = pd.Timestamp("2023-12-29T23:59:00Z")
    rows = []
    end_values = [108, 107, 106, 105, 104, 103, 103, 103]
    for index, ticker in enumerate(tickers):
        for label, date in dates.items():
            value = 100.0
            if label == "one_month_date":
                value = float(end_values[index])
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "total_return_index": value,
                    "available_at": signal_at - pd.Timedelta(hours=1),
                }
            )
    prices = pd.DataFrame(rows)
    decisions = pd.DataFrame(
        [
            {
                "signal_at": signal_at,
                **dates,
            }
        ]
    )
    executions = pd.DataFrame(
        [
            {
                "signal_at": signal_at,
                "ticker": ticker,
                "entry_at": pd.Timestamp("2024-01-02T14:30:00Z"),
                "exit_at": pd.Timestamp("2024-02-01T14:30:00Z"),
                "entry_price": 100.0,
                "exit_price": 101.0 + index,
            }
            for index, ticker in enumerate(tickers)
        ]
    )
    return tickers, prices, decisions, executions


def test_exact_endpoints_top_six_and_deterministic_tie_break():
    tickers, prices, decisions, executions = _signal_inputs()

    returns, holdings, signals = run_momentum_champion(
        prices,
        decisions,
        executions,
        universe=tickers,
        trading_cost_bps_one_way=0.0,
    )

    assert signals["twelve_month_date"].eq(pd.Timestamp("2023-01-03")).all()
    assert signals["one_month_date"].eq(pd.Timestamp("2023-12-01")).all()
    assert np.isclose(signals.loc[signals["ticker"].eq("T00"), "momentum_12_1"].iloc[0], 0.08)
    # T05, T06, and T07 tie for the final slot; lexical permanent-ID order wins.
    assert holdings.sort_values("momentum_rank")["ticker"].tolist() == tickers[:6]
    assert np.allclose(holdings["weight"], 1 / 6)
    assert returns.loc[0, "long_count"] == 6
    assert returns.loc[0, "entry_at"] == pd.Timestamp("2024-01-02T14:30:00Z")
    assert returns.loc[0, "exit_at"] == pd.Timestamp("2024-02-01T14:30:00Z")


def _two_month_candidates() -> pd.DataFrame:
    rows = []
    schedules = [
        (
            pd.Timestamp("2024-01-31T23:59:00Z"),
            pd.Timestamp("2024-02-01T14:30:00Z"),
            pd.Timestamp("2024-03-01T14:30:00Z"),
        ),
        (
            pd.Timestamp("2024-02-29T23:59:00Z"),
            pd.Timestamp("2024-03-01T14:30:00Z"),
            pd.Timestamp("2024-04-01T13:30:00Z"),
        ),
    ]
    for month, (signal_at, entry_at, exit_at) in enumerate(schedules):
        for index in range(7):
            rows.append(
                {
                    "signal_at": signal_at,
                    "ticker": f"T{index:02d}",
                    "momentum_12_1": 10 - index,
                    "entry_at": entry_at,
                    "exit_at": exit_at,
                    "entry_price": 100.0,
                    "exit_price": 100.0 + (index + 1 if month == 0 else 0.0),
                }
            )
    return pd.DataFrame(rows)


def test_turnover_uses_drifted_weights_and_charges_terminal_liquidation():
    cost_bps = 100.0
    returns, holdings = build_momentum_champion(
        _two_month_candidates(),
        trading_cost_bps_one_way=cost_bps,
    )

    first_signal = holdings["signal_at"].min()
    first = holdings.loc[holdings["signal_at"].eq(first_signal)].set_index("ticker")
    second = holdings.loc[holdings["signal_at"].ne(first_signal)].set_index("ticker")
    gross = float((first["weight"] * first["target_return"]).sum())
    drifted = first["weight"] * (1 + first["target_return"]) / (1 + gross)
    expected_transition = float((second["weight"] - drifted).abs().sum())
    liquidation_turnover = returns.loc[1, "terminal_liquidation_turnover"]

    assert np.isclose(returns.loc[0, "turnover"], 1.0)
    assert expected_transition > 0
    assert np.isclose(returns.loc[1, "turnover"] - liquidation_turnover, expected_transition)
    assert np.isclose(liquidation_turnover, 1.0)
    transition_cost = expected_transition * cost_bps / 10_000
    expected_liquidation_cost = (
        liquidation_turnover * cost_bps / 10_000 * (1 - transition_cost)
    )
    assert np.isclose(returns.loc[1, "terminal_liquidation_cost"], expected_liquidation_cost)
    assert np.isclose(returns.loc[1, "trading_cost"], transition_cost + expected_liquidation_cost)


def test_signal_rejects_endpoint_that_was_not_available():
    tickers, prices, decisions, _ = _signal_inputs()
    mask = prices["date"].eq(pd.Timestamp("2023-12-01")) & prices["ticker"].eq("T00")
    prices.loc[mask, "available_at"] = pd.Timestamp("2023-12-30T00:00:00Z")

    with pytest.raises(ValueError, match="unavailable"):
        compute_12_1_momentum(prices, decisions, universe=tickers)


def test_portfolio_rejects_noncausal_or_noncontiguous_execution_times():
    candidates = _two_month_candidates()
    candidates.loc[candidates["signal_at"].eq(candidates["signal_at"].min()), "entry_at"] = candidates[
        "signal_at"
    ].min()
    with pytest.raises(ValueError, match="signal_at < entry_at < exit_at"):
        build_momentum_champion(candidates)

    candidates = _two_month_candidates()
    second_signal = candidates["signal_at"].max()
    candidates.loc[candidates["signal_at"].eq(second_signal), "entry_at"] += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="contiguous"):
        build_momentum_champion(candidates)

    candidates = _two_month_candidates()
    candidates["entry_at"] += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="09:30"):
        build_momentum_champion(candidates)
