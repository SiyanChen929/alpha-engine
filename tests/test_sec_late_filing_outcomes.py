from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_factor_engine.research.sec_late_filing_outcomes import (
    LateFilingOutcomeError,
    attach_late_filing_outcomes,
    build_late_filing_schedule,
    circular_block_bootstrap_mean,
    gate_late_filing_confirmation,
    summarize_late_filing_events,
)


def _calendar(n: int = 90) -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=n)
    local_open = dates.tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30)
    local_close = dates.tz_localize("America/New_York") + pd.Timedelta(hours=16)
    return pd.DataFrame(
        {
            "session_date": dates,
            "market_open_utc": local_open.tz_convert("UTC"),
            "market_close_utc": local_close.tz_convert("UTC"),
        }
    )


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "issuer_cik": ["0000000001", "0000000001", "0000000002"],
            "accession": ["a", "b", "c"],
            "ticker": ["AAA", "AAA", "BBB"],
            "form": ["NT 10-K", "NT 10-Q", "NT 10-K"],
            "acceptance_at": pd.to_datetime(
                ["2023-02-13T12:00:00Z", "2023-02-14T12:00:00Z", "2023-02-13T22:00:00Z"],
                utc=True,
            ),
            "phase": ["confirmation", "confirmation", "sealed_2025"],
            "outcome_join_permitted": [False, False, False],
        }
    )


def _prices(calendar: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker, growth in [("AAA", 0.001), ("BBB", 0.0005), ("SPY", 0.0002)]:
        for index, session in calendar.iterrows():
            rows.append(
                {
                    "ticker": ticker,
                    "date": session["session_date"],
                    "timestamp": session["market_open_utc"],
                    "adj_open": 100.0 * (1.0 + growth) ** index,
                    "dollar_volume": 100_000_000.0,
                    "available_at": session["market_close_utc"] + pd.Timedelta(minutes=30),
                }
            )
    return pd.DataFrame(rows)


def test_schedule_uses_first_open_after_lag_and_suppresses_overlap() -> None:
    schedule, exclusions = build_late_filing_schedule(_candidates(), _calendar())
    assert set(schedule["accession"]) == {"a", "c"}
    assert exclusions.loc[exclusions["accession"].eq("b"), "reason"].item() == "overlap_until_prior_exit"
    a = schedule.loc[schedule["accession"].eq("a")].iloc[0]
    c = schedule.loc[schedule["accession"].eq("c")].iloc[0]
    assert a["entry_date"] == pd.Timestamp("2023-02-13")
    assert c["entry_date"] == pd.Timestamp("2023-02-14")
    assert a["exit_index"] - a["entry_index"] == 20
    assert a["entry_index"] - a["pre_index"] == 20


def test_outcome_attachment_rejects_sealed_events() -> None:
    schedule, _ = build_late_filing_schedule(_candidates(), _calendar())
    with pytest.raises(LateFilingOutcomeError, match="sealed 2025"):
        attach_late_filing_outcomes(schedule, _prices(_calendar()), _calendar())


def test_pair_return_and_two_leg_cost_are_reconstructed_exactly() -> None:
    calendar = _calendar()
    schedule, _ = build_late_filing_schedule(_candidates().iloc[[0]], calendar)
    events, exclusions = attach_late_filing_outcomes(
        schedule,
        _prices(calendar),
        calendar,
        minimum_adv_usd=5_000_000,
        portfolio_aum_usd=10_000_000,
    )
    assert exclusions.empty
    event = events.iloc[0]
    stock_return = (1.001**20) - 1.0
    spy_return = (1.0002**20) - 1.0
    gross = 0.5 * (spy_return - stock_return)
    traded = 1.0 + 0.5 * (1.0 + stock_return) + 0.5 * (1.0 + spy_return)
    assert event["pair_gross_return_20"] == pytest.approx(gross)
    assert event["pair_traded_notional_per_gross"] == pytest.approx(traded)
    assert event["pair_net_30bps_return_20"] == pytest.approx(gross - traded * 0.003)
    assert event["maximum_stock_trade_to_adv_at_10m"] == pytest.approx(
        max(0.05, 0.05 * (1.0 + stock_return))
    )


def test_summary_and_gate_use_entry_date_portfolios() -> None:
    rows = []
    for index in range(30):
        date = pd.Timestamp("2023-01-02") + pd.Timedelta(days=index)
        rows.append(
            {
                "entry_date": date,
                "entry_index": index * 2,
                "ticker": f"T{index:02d}",
                "form": "NT 10-Q",
                "pair_gross_return_20": 0.02,
                "pair_net_30bps_return_20": 0.01,
                "pair_net_50bps_return_20": 0.006,
                "pair_pre_event_gross_return_20": -0.002,
                "post_minus_pre_return_20": 0.022,
                "maximum_stock_trade_to_adv_at_10m": 0.005,
            }
        )
    events = pd.DataFrame(rows)
    # Add tiny deterministic variation so the HAC variance is nondegenerate.
    events["pair_net_30bps_return_20"] += np.linspace(-0.001, 0.001, len(events))
    summary = summarize_late_filing_events(events, trial_floor=131, hac_lags=3)
    gates = {
        "minimum_events": 20,
        "minimum_tickers": 20,
        "minimum_entry_dates": 20,
        "minimum_price_path_coverage": 0.8,
        "primary_net_mean_minimum": 0.0,
        "primary_newey_west_t_minimum": 3.0,
        "global_adjusted_p_maximum": 0.05,
        "stress_net_mean_minimum": 0.0,
        "post_minus_pre_mean_minimum": 0.0,
        "maximum_ticker_gross_contribution_share": 0.15,
        "minimum_leave_one_ticker_out_mean": 0.0,
        "maximum_trade_to_trailing_adv_at_10m": 0.01,
    }
    table = gate_late_filing_confirmation(summary, gates=gates, scheduled_events=30)
    assert table["passed"].all()


def test_liquidity_history_must_be_known_before_decision() -> None:
    calendar = _calendar()
    schedule, _ = build_late_filing_schedule(_candidates().iloc[[0]], calendar)
    prices = _prices(calendar)
    mask = prices["ticker"].eq("AAA") & prices["date"].lt(schedule.iloc[0]["entry_date"])
    prices.loc[mask, "available_at"] = pd.Timestamp("2026-01-01T00:00:00Z")
    events, exclusions = attach_late_filing_outcomes(schedule, prices, calendar)
    assert events.empty
    assert exclusions.iloc[0]["reason"] == "trailing_adv_incomplete_or_not_known"


def test_circular_bootstrap_is_centered_and_deterministic() -> None:
    values = np.linspace(0.001, 0.02, 30)
    first = circular_block_bootstrap_mean(values, block_length=4, draws=500, seed=7)
    second = circular_block_bootstrap_mean(values, block_length=4, draws=500, seed=7)
    assert first == second
    assert abs(first["centered_null_mean"]) < 0.001
    assert first["one_sided_p"] < 0.05
