import numpy as np
import pandas as pd
import pytest
from datetime import UTC, datetime

from alpha_factor_engine.contracts import PointInTimeViolation
from alpha_factor_engine.champion_pipeline import (
    _evaluation_cross_sections,
    _replace_with_exact_momentum,
    _write_governance_records,
    load_champion_config,
)
from alpha_factor_engine.research.champion_challenger import (
    add_beta_size_residual_target,
    add_decision_time,
    assert_panel_is_point_in_time,
    build_ranked_long_only,
    capacity_diagnostics,
    relative_return_frame,
    select_public_proxy_universe,
)


def _panel(months: int = 2, names: int = 8) -> pd.DataFrame:
    rows = []
    signal_dates = pd.date_range("2024-01-31", periods=months, freq="ME", tz="UTC")
    entry_dates = pd.date_range("2024-02-01", periods=months, freq="MS")
    exit_dates = pd.date_range("2024-03-01", periods=months, freq="MS")
    for month, (signal, entry, exit_at) in enumerate(zip(signal_dates, entry_dates, exit_dates)):
        for index in range(names):
            rows.append(
                {
                    "date": signal.tz_localize(None),
                    "signal_date": signal,
                    "ticker": f"T{index:02d}",
                    "eligible": True,
                    "log_market_cap": float(names - index),
                    "momentum_12_1": float(index),
                    "beta_12m": 0.8 + index / 100,
                    "adv_usd_3m": 1_000_000.0,
                    "target_return": (index - names / 2) / 1000,
                    "price_available_at": signal - pd.Timedelta(hours=1),
                    "fundamental_available_at": signal - pd.Timedelta(days=10),
                    "macro_available_at": signal - pd.Timedelta(days=1),
                    "entry_at": (entry + pd.Timedelta(hours=9, minutes=30))
                    .tz_localize("America/New_York")
                    .tz_convert("UTC"),
                    "exit_at": (exit_at + pd.Timedelta(hours=9, minutes=30))
                    .tz_localize("America/New_York")
                    .tz_convert("UTC"),
                    "entry_price": 100.0,
                    "exit_price": 100.0 + index + month,
                }
            )
    return pd.DataFrame(rows)


def test_decision_clock_and_availability_gate_fail_on_future_feature():
    panel = add_decision_time(_panel(), decision_time_et="20:00")
    quality = assert_panel_is_point_in_time(panel, feature_columns=["momentum_12_1"])
    assert quality["future_information_violations"] == 0
    assert panel["decision_time"].dt.tz_convert("America/New_York").dt.hour.eq(20).all()

    panel.loc[0, "price_available_at"] = panel.loc[0, "decision_time"] + pd.Timedelta(seconds=1)
    with pytest.raises(PointInTimeViolation, match="future=1"):
        assert_panel_is_point_in_time(panel, feature_columns=["momentum_12_1"])


def test_residual_target_has_zero_beta_size_projection_and_rejects_partial_labels():
    panel = add_beta_size_residual_target(_panel(months=1))
    residual = panel["residual_target_return"].to_numpy()
    design = np.column_stack(
        [
            np.ones(len(panel)),
            panel["beta_12m"].to_numpy(),
            panel["log_market_cap"].to_numpy(),
        ]
    )
    assert np.allclose(design.T @ residual, 0.0, atol=1e-10)

    partial = _panel(months=1)
    partial.loc[0, "target_return"] = np.nan
    with pytest.raises(PointInTimeViolation, match="partial target coverage"):
        add_beta_size_residual_target(partial)


def test_public_proxy_universe_is_deterministic_and_explicitly_diagnostic():
    panel = add_decision_time(_panel(months=1, names=8))
    selected = select_public_proxy_universe(
        panel.sample(frac=1, random_state=7), top_n=5, minimum_cross_section=5
    )

    assert selected["ticker"].tolist() == ["T00", "T01", "T02", "T03", "T04"]
    assert selected["universe_rank"].tolist() == [1, 2, 3, 4, 5]
    assert selected["universe_evidence_status"].str.contains("opened_diagnostic").all()


def test_ranked_long_only_uses_common_cost_engine_and_relative_returns():
    panel = add_decision_time(_panel(months=2, names=8))
    returns, holdings = build_ranked_long_only(
        panel,
        score_column="momentum_12_1",
        holdings_count=6,
        trading_cost_bps_one_way=15,
    )
    benchmark_panel = panel.assign(equal_score=0.0)
    benchmark, _ = build_ranked_long_only(
        benchmark_panel,
        score_column="equal_score",
        holdings_count=8,
        trading_cost_bps_one_way=15,
    )
    relative = relative_return_frame(returns, benchmark)

    assert len(holdings) == 12
    assert np.isclose(holdings.groupby("signal_at")["weight"].sum(), 1.0).all()
    assert len(relative) == 2
    assert np.isfinite(relative["net_return"]).all()
    assert np.allclose(
        relative["active_return"],
        relative["strategy_return"] - relative["benchmark_return"],
    )


def test_relative_returns_require_identical_execution_timestamps():
    panel = add_decision_time(_panel(months=1, names=8))
    strategy, _ = build_ranked_long_only(
        panel,
        score_column="momentum_12_1",
        holdings_count=6,
        trading_cost_bps_one_way=0,
    )
    benchmark, _ = build_ranked_long_only(
        panel.assign(equal_score=0.0),
        score_column="equal_score",
        holdings_count=8,
        trading_cost_bps_one_way=0,
    )
    benchmark["entry_at"] += pd.Timedelta(minutes=1)
    with pytest.raises(PointInTimeViolation, match="identical"):
        relative_return_frame(strategy, benchmark)


def test_evaluation_boundaries_keep_or_reject_whole_cross_sections():
    universe = select_public_proxy_universe(
        add_decision_time(_panel(months=2, names=8)),
        top_n=5,
        minimum_cross_section=5,
    )
    selected = _evaluation_cross_sections(
        universe,
        entry_at_or_after=pd.Timestamp("2024-01-01T00:00:00Z"),
        exit_at_or_before=pd.Timestamp("2025-01-01T00:00:00Z"),
    )
    assert selected.groupby("signal_date").size().eq(5).all()

    broken = universe.copy()
    broken.loc[broken.index[0], "exit_at"] = pd.NaT
    with pytest.raises(RuntimeError, match="partial exit_at"):
        _evaluation_cross_sections(
            broken,
            entry_at_or_after=pd.Timestamp("2024-01-01T00:00:00Z"),
            exit_at_or_before=pd.Timestamp("2025-01-01T00:00:00Z"),
        )


def test_ranked_long_only_fails_if_a_frozen_universe_score_is_missing():
    panel = add_decision_time(_panel(months=1, names=8))
    panel.loc[0, "momentum_12_1"] = np.nan
    with pytest.raises(PointInTimeViolation, match="missing"):
        build_ranked_long_only(
            panel,
            score_column="momentum_12_1",
            holdings_count=6,
            trading_cost_bps_one_way=0,
        )


def test_capacity_gate_measures_rebalances_and_terminal_liquidation():
    panel = add_decision_time(_panel(months=2, names=8))
    _, holdings = build_ranked_long_only(
        panel,
        score_column="momentum_12_1",
        holdings_count=6,
        trading_cost_bps_one_way=0,
    )
    summary, trades = capacity_diagnostics(
        holdings,
        portfolio_aum_usd=10_000,
        maximum_adv_participation=0.01,
    )

    assert summary["capacity_passed"] is True
    assert trades["trade_type"].eq("terminal_liquidation").any()

    failed, _ = capacity_diagnostics(
        holdings,
        portfolio_aum_usd=100_000_000,
        maximum_adv_participation=0.01,
    )
    assert failed["capacity_passed"] is False


def test_exact_momentum_uses_canonical_calendar_endpoints():
    dates = pd.bdate_range("2023-01-02", periods=253)
    formation = dates[-1]
    signal_at = (
        formation.tz_localize("America/New_York")
        + pd.Timedelta(hours=20)
    ).tz_convert("UTC")
    panel = pd.DataFrame(
        {
            "ticker": ["A"],
            "date": [formation],
            "signal_at": [signal_at],
            "eligible": [True],
            "momentum_12_1": [999.0],
        }
    )
    daily_rows = []
    for ticker in ["SPY", "A"]:
        for index, date in enumerate(dates):
            daily_rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "adj_close": 100.0 + index if ticker == "A" else 100.0,
                    "available_at": signal_at - pd.Timedelta(minutes=30),
                }
            )
    updated, signals, decisions = _replace_with_exact_momentum(
        panel,
        pd.DataFrame(daily_rows),
        benchmark_ticker="SPY",
        formation_sessions=252,
        skip_sessions=21,
    )
    expected = (100.0 + 231) / 100.0 - 1.0

    assert updated.loc[0, "momentum_12_1"] == pytest.approx(expected)
    assert signals.loc[0, "twelve_month_date"] == dates[0]
    assert signals.loc[0, "one_month_date"] == dates[-22]
    assert decisions.loc[0, "formation_date"] == formation


def test_exact_momentum_excludes_new_listing_until_full_history_exists():
    dates = pd.bdate_range("2023-01-02", periods=253)
    formation = dates[-1]
    signal_at = (
        formation.tz_localize("America/New_York") + pd.Timedelta(hours=20)
    ).tz_convert("UTC")
    panel = pd.DataFrame(
        {
            "ticker": ["A", "IPO"],
            "date": [formation, formation],
            "signal_at": [signal_at, signal_at],
            "eligible": [True, True],
            "momentum_12_1": [0.0, 0.0],
        }
    )
    rows = []
    for ticker, ticker_dates in [("SPY", dates), ("A", dates), ("IPO", dates[-50:])]:
        for index, date in enumerate(ticker_dates):
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "adj_close": 100.0 + index,
                    "available_at": signal_at - pd.Timedelta(minutes=30),
                }
            )

    updated, signals, decisions = _replace_with_exact_momentum(
        panel,
        pd.DataFrame(rows),
        benchmark_ticker="SPY",
        formation_sessions=252,
        skip_sessions=21,
    )

    assert signals["ticker"].tolist() == ["A"]
    assert not bool(updated.set_index("ticker").loc["IPO", "eligible"])
    assert decisions.loc[0, "insufficient_history_names"] == 1


def test_pipeline_governance_is_idempotent_and_enforces_freeze_deadline(tmp_path):
    config = load_champion_config("configs/champion_challenger.yml")
    config["prospective"] = {
        **config["prospective"],
        "strategy_version": "test_v1",
        "governance_dir": "governance/test_v1",
    }
    first, _, _ = _write_governance_records(
        config,
        project_root=tmp_path,
        config_sha256="a" * 64,
        source_tree_sha256="b" * 64,
        registered_at=datetime(2026, 9, 1, 12, tzinfo=UTC),
    )
    forecast_path = tmp_path / "governance/test_v1/sealed_forecasts.jsonl"
    original_forecast_bytes = forecast_path.read_bytes()
    repeated, _, _ = _write_governance_records(
        config,
        project_root=tmp_path,
        config_sha256="a" * 64,
        source_tree_sha256="b" * 64,
        registered_at=datetime(2026, 9, 2, 12, tzinfo=UTC),
    )
    assert repeated.registered_at == first.registered_at
    assert forecast_path.read_bytes() == original_forecast_bytes

    late_config = {
        **config,
        "prospective": {
            **config["prospective"],
            "strategy_version": "test_v2",
            "governance_dir": "governance/test_v2",
        },
    }
    with pytest.raises(RuntimeError, match="freeze deadline has passed"):
        _write_governance_records(
            late_config,
            project_root=tmp_path,
            config_sha256="c" * 64,
            source_tree_sha256="d" * 64,
            registered_at=datetime(2026, 9, 30, 1, tzinfo=UTC),
        )
