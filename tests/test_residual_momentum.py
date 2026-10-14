from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_factor_engine.research.residual_momentum import (
    compute_market_residual_momentum,
    neutralize_cross_sectional_score,
)
from scripts.run_liquid_residual_momentum import (
    _build_label_independent_target_holdings,
    _complete_liquid_universe,
)


def _daily_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2018-01-02", periods=900)
    x = np.arange(len(dates), dtype=float)
    market_return = 0.00025 + 0.004 * np.sin(x / 13.0)
    a_return = 1.15 * market_return + 0.00045 + 0.0015 * np.sin(x / 7.0)
    b_return = 0.80 * market_return - 0.00015 + 0.0015 * np.cos(x / 9.0)
    rows = []
    for ticker, returns in {
        "SPY": market_return,
        "A": a_return,
        "B": b_return,
    }.items():
        close = 100.0 * np.cumprod(1.0 + returns)
        rows.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "date": dates,
                    "adj_close": close,
                    "available_at": pd.to_datetime(dates).tz_localize("UTC")
                    + pd.Timedelta(hours=21),
                }
            )
        )
    daily = pd.concat(rows, ignore_index=True)
    decisions = pd.DataFrame(
        {
            "formation_date": [dates[800], dates[850]],
            "signal_at": pd.to_datetime([dates[800], dates[850]]).tz_localize("UTC")
            + pd.Timedelta(days=1),
        }
    )
    return daily, decisions


def test_residual_momentum_orders_positive_firm_specific_trend_first():
    daily, decisions = _daily_fixture()
    scores, audit = compute_market_residual_momentum(
        daily,
        decisions.iloc[[0]],
        beta_window_sessions=500,
        minimum_beta_observations=400,
    )
    ranked = scores.set_index("ticker")["residual_momentum_score"]
    assert ranked["A"] > ranked["B"]
    assert scores["formation_observations"].min() == 232
    assert audit.iloc[0]["status"] == "scored"
    assert int(audit.iloc[0]["scored_names"]) == 2


def test_future_price_changes_do_not_change_earlier_signal():
    daily, decisions = _daily_fixture()
    original, _ = compute_market_residual_momentum(
        daily,
        decisions,
        beta_window_sessions=500,
        minimum_beta_observations=400,
    )
    changed = daily.copy()
    after_first = changed["date"].gt(decisions.iloc[0]["formation_date"])
    changed.loc[after_first & changed["ticker"].eq("A"), "adj_close"] *= 3.0
    revised, _ = compute_market_residual_momentum(
        changed,
        decisions,
        beta_window_sessions=500,
        minimum_beta_observations=400,
    )
    first = decisions.iloc[0]["formation_date"]
    left = original.loc[original["formation_date"].eq(first)].sort_values("ticker")
    right = revised.loc[revised["formation_date"].eq(first)].sort_values("ticker")
    np.testing.assert_allclose(
        left["residual_momentum_score"], right["residual_momentum_score"], atol=0, rtol=0
    )


def test_late_available_price_fails_closed():
    daily, decisions = _daily_fixture()
    used_row = daily["date"].eq(daily["date"].sort_values().unique()[700])
    daily.loc[used_row & daily["ticker"].eq("A"), "available_at"] = (
        decisions.iloc[0]["signal_at"] + pd.Timedelta(days=1)
    )
    with pytest.raises(ValueError, match="availability exceeds decision clock"):
        compute_market_residual_momentum(
            daily,
            decisions.iloc[[0]],
            beta_window_sessions=500,
            minimum_beta_observations=400,
        )


def test_cross_sectional_neutralization_removes_rank_exposure_projection():
    frame = pd.DataFrame(
        {
            "signal_date": [pd.Timestamp("2024-01-31", tz="UTC")] * 30,
            "score": np.linspace(-2, 2, 30) + np.sin(np.arange(30)),
            "beta": np.linspace(0.5, 1.5, 30),
            "size": np.linspace(10, 20, 30) ** 1.2,
        }
    )
    result = neutralize_cross_sectional_score(
        frame,
        score_column="score",
        exposure_columns=["beta", "size"],
        output_column="neutral",
    )
    residual = result["neutral"].to_numpy(float)
    for exposure in ["beta", "size"]:
        ranked = result[exposure].rank(pct=True).to_numpy(float)
        assert abs(float(np.dot(residual, ranked))) < 1e-10


def test_liquid_universe_membership_is_independent_of_future_labels():
    signal_date = pd.Timestamp("2024-02-01 01:00", tz="UTC")
    tickers = [f"S{number:02d}" for number in range(30)]
    panel = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-31")] * 30,
            "signal_date": [signal_date] * 30,
            "entry_at": [pd.Timestamp("2024-02-01 14:30", tz="UTC")] * 30,
            "exit_at": [pd.Timestamp("2024-03-01 14:30", tz="UTC")] * 30,
            "price_available_at": [signal_date - pd.Timedelta(hours=2)] * 30,
            "ticker": tickers,
            "eligible": [True] * 30,
            "target_return": np.linspace(-0.10, 0.10, 30),
            "pit_close": [20.0] * 30,
            "adv_usd_3m": np.arange(30, 0, -1, dtype=float) * 1_000_000,
            "beta_12m": np.linspace(0.7, 1.3, 30),
            "log_market_cap": np.linspace(20, 23, 30),
            "log_dollar_volume_3m": np.linspace(16, 18, 30),
            "idio_vol_12m": np.linspace(0.1, 0.4, 30),
        }
    )
    scores = pd.DataFrame(
        {
            "formation_date": [pd.Timestamp("2024-01-31")] * 30,
            "ticker": tickers,
            "signal_at": [signal_date] * 30,
            "market_beta_signal": np.linspace(0.8, 1.2, 30),
            "beta_observations": [252] * 30,
            "formation_observations": [232] * 30,
            "residual_return_sum_12_1": np.linspace(-0.2, 0.2, 30),
            "residual_volatility_daily_12_1": [0.02] * 30,
            "residual_momentum_score": np.linspace(-2, 2, 30),
            "exact_total_momentum_12_1": np.linspace(-0.1, 0.4, 30),
            "latest_input_available_at": [signal_date - pd.Timedelta(hours=2)] * 30,
        }
    )
    config = {
        "research": {"first_signal_date": "2024-01-01"},
        "universe": {
            "monthly_size": 25,
            "minimum_complete_cross_section": 20,
            "minimum_price": 5.0,
            "minimum_adv_usd": 1_000_000,
        },
        "development_family": {
            "neutralization_exposures": [
                "beta_12m",
                "log_market_cap",
                "log_dollar_volume_3m",
                "idio_vol_12m",
            ]
        },
    }

    signal_full, evaluation_full, audit_full = _complete_liquid_universe(
        panel, scores, config
    )
    missing = panel.copy()
    missing.loc[missing["ticker"].eq("S00"), "target_return"] = np.nan
    signal_missing, evaluation_missing, audit_missing = _complete_liquid_universe(
        missing, scores, config
    )

    assert signal_full["ticker"].tolist() == signal_missing["ticker"].tolist()
    targets_full = _build_label_independent_target_holdings(
        signal_full,
        score_column="exact_total_momentum_12_1",
        holdings_count=10,
        retention_rank=15,
    )
    targets_missing = _build_label_independent_target_holdings(
        signal_missing,
        score_column="exact_total_momentum_12_1",
        holdings_count=10,
        retention_rank=15,
    )
    assert targets_full["ticker"].tolist() == targets_missing["ticker"].tolist()
    assert len(evaluation_full) == 25
    assert evaluation_missing.empty
    assert audit_full.iloc[0]["evaluation_status"] == "evaluable"
    assert audit_missing.iloc[0]["evaluation_status"] == "partial_labels_excluded"
