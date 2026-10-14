import numpy as np
import pandas as pd
import pytest

from alpha_factor_engine.research.diversified_momentum import (
    build_diversified_momentum_portfolio,
)


def _candidates(months: int = 1, names: int = 8) -> pd.DataFrame:
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
    ][:months]
    rows = []
    for month, (signal_at, entry_at, exit_at) in enumerate(schedules):
        for index in range(names):
            rows.append(
                {
                    "signal_at": signal_at,
                    "ticker": f"T{index:02d}",
                    "momentum_12_1": 10.0 - index,
                    "realized_vol_3m": 0.10 + 0.02 * index,
                    "beta_12m": 0.70 + 0.08 * index,
                    "price_available_at": signal_at - pd.Timedelta(hours=2),
                    "pit_sector": ["A", "A", "B", "B", "C", "C", "D", "D"][index],
                    "sector_available_at": signal_at - pd.Timedelta(days=30),
                    "entry_at": entry_at,
                    "exit_at": exit_at,
                    "entry_price": 100.0,
                    "exit_price": 100.0 + (index + 1 if month == 0 else 0.0),
                }
            )
    return pd.DataFrame(rows)


def test_deterministic_top_n_and_unconstrained_inverse_volatility_weights():
    candidates = _candidates(names=8)
    # Three names tie around the fourth slot; ticker order must resolve the tie.
    candidates.loc[candidates["ticker"].isin(["T03", "T04", "T05"]), "momentum_12_1"] = 7.0
    returns, holdings = build_diversified_momentum_portfolio(
        candidates.sample(frac=1, random_state=5),
        top_n=4,
        max_name_weight=0.50,
        beta_exposure_limit=2.0,
        trading_cost_bps_one_way=0.0,
        charge_terminal_liquidation=False,
    )

    assert holdings.sort_values("selection_rank")["ticker"].tolist() == ["T00", "T01", "T02", "T03"]
    expected = 1 / holdings["realized_vol_3m"].to_numpy()
    expected = expected / expected.sum()
    assert np.allclose(holdings["weight"], expected)
    assert np.allclose(holdings["weight"], holdings["inverse_volatility_base_weight"])
    assert np.isclose(holdings["weight"].sum(), 1.0)
    assert returns.loc[0, "long_count"] == 4


def test_name_sector_and_beta_limits_are_hard_constraints():
    candidates = _candidates(names=8)
    returns, holdings = build_diversified_momentum_portfolio(
        candidates,
        top_n=6,
        max_name_weight=0.24,
        beta_exposure_limit=0.95,
        sector_column="pit_sector",
        sector_available_at_column="sector_available_at",
        max_sector_weight=0.40,
        trading_cost_bps_one_way=0.0,
        charge_terminal_liquidation=False,
    )

    assert holdings["weight"].max() <= 0.24 + 1e-8
    assert holdings.groupby("pit_sector")["weight"].sum().max() <= 0.40 + 1e-8
    assert abs(float((holdings["weight"] * holdings["beta_12m"]).sum())) <= 0.95 + 1e-8
    assert np.isclose(holdings["weight"].sum(), 1.0)
    assert returns.loc[0, "largest_sector_weight"] <= 0.40 + 1e-8
    assert abs(returns.loc[0, "beta_exposure"]) <= 0.95 + 1e-8


def test_rank_buffer_retains_until_cutoff_and_fills_exactly_n_deterministically():
    candidates = _candidates(months=2, names=8)
    second_signal = candidates["signal_at"].max()
    second_scores = {
        "T00": 10.0,
        "T04": 9.0,
        "T05": 8.0,
        "T01": 7.0,
        "T02": 6.0,
        "T06": 5.0,
        "T03": 4.0,
        "T07": 3.0,
    }
    second = candidates["signal_at"].eq(second_signal)
    candidates.loc[second, "momentum_12_1"] = candidates.loc[second, "ticker"].map(second_scores)

    kwargs = {
        "top_n": 4,
        "retention_rank": 6,
        "max_name_weight": 0.50,
        "beta_exposure_limit": 2.0,
        "trading_cost_bps_one_way": 0.0,
        "charge_terminal_liquidation": False,
    }
    _, holdings = build_diversified_momentum_portfolio(
        candidates.sample(frac=1, random_state=11),
        **kwargs,
    )
    drift_variant = candidates.copy()
    drift_variant.loc[drift_variant["signal_at"].lt(second_signal), "exit_price"] *= np.linspace(
        0.5,
        2.0,
        8,
    )
    _, drift_holdings = build_diversified_momentum_portfolio(
        drift_variant.sample(frac=1, random_state=19),
        **kwargs,
    )

    selected = holdings.loc[holdings["signal_at"].eq(second_signal)].sort_values("selection_rank")
    drift_selected = drift_holdings.loc[
        drift_holdings["signal_at"].eq(second_signal)
    ].sort_values("selection_rank")
    assert selected["ticker"].tolist() == ["T00", "T04", "T01", "T02"]
    assert selected["universe_rank"].tolist() == [1, 2, 4, 5]
    assert selected["selection_reason"].tolist() == [
        "retained_within_buffer",
        "entered_top_n",
        "retained_within_buffer",
        "retained_within_buffer",
    ]
    assert selected["retention_rank_threshold"].eq(6).all()
    assert selected["ticker"].tolist() == drift_selected["ticker"].tolist()
    assert len(selected) == 4


def test_turnover_uses_drifted_weights_and_terminal_liquidation():
    cost_bps = 100.0
    returns, holdings = build_diversified_momentum_portfolio(
        _candidates(months=2),
        top_n=6,
        max_name_weight=0.30,
        beta_exposure_limit=2.0,
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
    expected_liquidation_cost = liquidation_turnover * cost_bps / 10_000 * (1 - transition_cost)
    assert np.isclose(returns.loc[1, "terminal_liquidation_cost"], expected_liquidation_cost)
    assert np.isclose(returns.loc[1, "trading_cost"], transition_cost + expected_liquidation_cost)


def test_future_or_missing_features_fail_closed():
    future = _candidates()
    future.loc[0, "price_available_at"] = future.loc[0, "signal_at"] + pd.Timedelta(seconds=1)
    with pytest.raises(ValueError, match="unavailable"):
        build_diversified_momentum_portfolio(future, top_n=6, max_name_weight=0.30)

    missing = _candidates()
    missing.loc[0, "realized_vol_3m"] = np.nan
    with pytest.raises(ValueError, match="missing or non-finite"):
        build_diversified_momentum_portfolio(missing, top_n=6, max_name_weight=0.30)

    sector_future = _candidates()
    sector_future.loc[0, "sector_available_at"] = sector_future.loc[0, "signal_at"] + pd.Timedelta(seconds=1)
    with pytest.raises(ValueError, match="sector label was unavailable"):
        build_diversified_momentum_portfolio(
            sector_future,
            top_n=6,
            max_name_weight=0.30,
            sector_column="pit_sector",
            sector_available_at_column="sector_available_at",
            max_sector_weight=0.50,
        )

    with pytest.raises(ValueError, match="non-PIT"):
        build_diversified_momentum_portfolio(
            _candidates().rename(columns={"pit_sector": "sic_division_current"}),
            top_n=6,
            max_name_weight=0.30,
            sector_column="sic_division_current",
            sector_available_at_column="sector_available_at",
            max_sector_weight=0.50,
        )


def test_infeasible_constraints_and_noncausal_execution_fail_closed():
    with pytest.raises(ValueError, match="cannot support"):
        build_diversified_momentum_portfolio(
            _candidates(),
            top_n=4,
            max_name_weight=0.20,
        )

    infeasible_beta = _candidates()
    infeasible_beta["beta_12m"] = 1.0
    with pytest.raises(ValueError, match="infeasible"):
        build_diversified_momentum_portfolio(
            infeasible_beta,
            top_n=6,
            max_name_weight=0.30,
            beta_exposure_limit=0.50,
        )

    noncausal = _candidates()
    noncausal["entry_at"] = noncausal["signal_at"]
    with pytest.raises(ValueError, match="signal_at < entry_at < exit_at"):
        build_diversified_momentum_portfolio(
            noncausal,
            top_n=6,
            max_name_weight=0.30,
        )
