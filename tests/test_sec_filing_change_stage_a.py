from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_factor_engine.research.sec_filing_change_stage_a import (
    FilingChangeStageAError,
    aggregate_entry_dates,
    attach_event_outcomes,
    build_calendar_preoutcome_schedule,
    build_preoutcome_schedule,
    calendar_portfolio_returns,
    development_gate_table,
    hac_mean_test,
    known_trial_floor_screen,
    monthly_portfolio_gate_table,
    paper_style_monthly_returns,
    reconcile_benchmark_to_exchange_calendar,
    summarize_event_sample,
    summarize_monthly_portfolio,
)
from alpha_factor_engine.research.sec_filing_change_stage_a import _portfolio_scope_returns


def _prices() -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-02", "2025-12-31")
    opens = (dates + pd.Timedelta(hours=9, minutes=30)).tz_localize(
        "America/New_York"
    ).tz_convert("UTC")
    market_returns = 0.0001 + 0.002 * np.sin(np.arange(len(dates)) / 11)
    rows = []
    for ticker, beta, start in (("SPY", 1.0, 100.0), ("AAA", 1.2, 50.0), ("BBB", 0.8, 70.0)):
        extra = 0 if ticker == "SPY" else 0.0002 * np.cos(np.arange(len(dates)) / 7)
        returns = beta * market_returns + extra
        values = start * np.cumprod(1 + returns)
        for index, date in enumerate(dates):
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "timestamp": opens[index],
                    "adj_open": values[index],
                    "close": values[index] * 1.001,
                    "volume": 1_000_000.0,
                }
            )
    return pd.DataFrame(rows)


def _signals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "current_accession": ["a", "b", "middle"],
            "cik": [1, 2, 3],
            "ticker": ["AAA", "BBB", "CCC"],
            "form": ["10-K", "10-Q", "10-K"],
            "current_accepted_at": [
                "2023-01-31T22:00:00Z",
                "2023-01-15T18:00:00Z",
                "2023-01-15T18:00:00Z",
            ],
            "similarity_quintile": [5, 1, 3],
            "sort_eligible": [True, True, True],
        }
    )


def test_schedule_uses_next_month_and_quarantines_one_full_open():
    prices = _prices()
    schedule, exclusions = build_preoutcome_schedule(_signals(), prices)
    assert len(schedule) == 2
    assert exclusions["current_accession"].tolist() == ["middle"]
    spy = prices.loc[prices["ticker"].eq("SPY")].sort_values("timestamp").reset_index(drop=True)
    a = schedule.set_index("current_accession").loc["a"]
    first_strict = spy.index[spy["timestamp"].gt(pd.Timestamp("2023-01-31T22:00:00Z"))][0]
    assert a["entry_index"] == first_strict + 1
    assert a["entry_date"].to_period("M") == pd.Period("2023-02", freq="M")
    assert a["exit_index"] == a["entry_index"] + 63
    assert a["entry_at"] > pd.Timestamp("2023-01-31T22:00:00Z")
    february = spy.loc[spy["date"].dt.to_period("M").eq(pd.Period("2023-02"))]
    assert a["entry_date"] == february.iloc[1]["date"]


def test_schedule_resolves_conflicts_and_overlap_before_outcomes():
    signals = pd.DataFrame(
        {
            "current_accession": ["a", "b", "c", "d"],
            "cik": [1, 1, 2, 2],
            "ticker": ["AAA", "AAA", "BBB", "BBB"],
            "form": ["10-K"] * 4,
            "current_accepted_at": [
                "2023-01-10T18:00:00Z",
                "2023-01-11T18:00:00Z",
                "2023-01-10T18:00:00Z",
                "2023-02-10T18:00:00Z",
            ],
            "similarity_quintile": [1, 5, 5, 5],
            "sort_eligible": [True] * 4,
        }
    )
    schedule, exclusions = build_preoutcome_schedule(signals, _prices())
    assert "AAA" not in set(schedule["ticker"])
    assert schedule.loc[schedule["ticker"].eq("BBB"), "current_accession"].tolist() == ["c"]
    assert set(exclusions["reason"]) == {
        "same_ticker_entry_direction_conflict",
        "overlap_until_prior_exit",
    }


def test_outcomes_use_prior_only_liquidity_beta_and_exact_costs():
    prices = _prices()
    schedule, _ = build_preoutcome_schedule(_signals(), prices)
    events, exclusions = attach_event_outcomes(schedule, prices)
    assert exclusions.empty
    assert len(events) == 2
    row = events.iloc[0]
    assert row["liquidity_last_date"] < row["entry_date"]
    assert row["beta_sample_end_date"] <= row["placebo_start_date"]
    assert row["beta_observations"] == 252
    gross = row["stock_weight"] * row["stock_return_63"] + row["benchmark_weight"] * row["spy_return_63"]
    assert row["hedged_gross_return_63"] == pytest.approx(gross)
    expected_cost = 0.003 * (
        abs(row["stock_weight"])
        + abs(row["benchmark_weight"])
        + abs(row["stock_weight"] * (1 + row["stock_return_63"]))
        + abs(row["benchmark_weight"] * (1 + row["spy_return_63"]))
    )
    assert row["traded_notional_cost_30bps"] == pytest.approx(expected_cost)
    assert row["hedged_net_return_30bps"] == pytest.approx(gross - expected_cost)


def test_missing_exact_exit_is_excluded_without_forward_fill():
    prices = _prices()
    schedule, _ = build_preoutcome_schedule(_signals().iloc[:1], prices)
    event = schedule.iloc[0]
    truncated = prices.loc[
        ~(prices["ticker"].eq("AAA") & prices["date"].eq(event["exit_date"]))
    ]
    events, exclusions = attach_event_outcomes(schedule, truncated)
    assert events.empty
    assert exclusions.iloc[0]["reason"] == "exact_stock_price_path_missing"


def test_date_aggregation_hac_and_known_trial_floor_screen():
    prices = _prices()
    schedule, _ = build_preoutcome_schedule(_signals(), prices)
    events, _ = attach_event_outcomes(schedule, prices)
    dates = aggregate_entry_dates(events)
    assert dates["event_count"].sum() == 2
    values = np.linspace(-0.01, 0.03, 80)
    test = hac_mean_test(values, session_indices=np.arange(80) * 2, lags=63)
    assert test["status"] == "OK"
    assert test["mean"] > 0
    screen = known_trial_floor_screen(test, trial_floor=126)
    assert screen["known_floor_adjusted_screening_p"] == pytest.approx(
        min(1, test["one_sided_p"] * 126)
    )
    assert screen["family_wise_error_control_established"] is False


def test_paper_style_portfolio_uses_latest_signal_and_three_month_hold():
    signals = _signals().iloc[:2].copy()
    signals["current_accepted_at"] = pd.to_datetime(signals["current_accepted_at"], utc=True)
    returns, monthly = paper_style_monthly_returns(signals, _prices(), holding_months=3)
    assert set(returns["quintile"]) == {1, 5}
    assert returns["month"].nunique() == 3
    assert monthly["q5_minus_q1_gross_return"].notna().all()


def test_calendar_schedule_keeps_consecutive_quarterly_cohorts():
    signals = pd.DataFrame(
        {
            "current_accession": ["a", "b"],
            "cik": [1, 1],
            "ticker": ["AAA", "AAA"],
            "form": ["10-Q", "10-Q"],
            "current_accepted_at": ["2023-01-10T18:00:00Z", "2023-03-10T18:00:00Z"],
            "similarity_quintile": [5, 5],
            "sort_eligible": [True, True],
        }
    )
    calendar, exclusions = build_calendar_preoutcome_schedule(signals, _prices())
    event, event_exclusions = build_preoutcome_schedule(signals, _prices())
    assert calendar["current_accession"].tolist() == ["a", "b"]
    assert not exclusions.get("reason", pd.Series(dtype=str)).eq("overlap_until_prior_exit").any()
    assert event["current_accession"].tolist() == ["a"]
    assert event_exclusions["reason"].eq("overlap_until_prior_exit").any()


def test_formal_calendar_portfolio_charges_turnover_and_uses_complete_month_grid():
    schedule, _ = build_calendar_preoutcome_schedule(_signals().iloc[:2], _prices())
    holdings, monthly, exclusions = calendar_portfolio_returns(schedule, _prices())
    assert exclusions.empty
    overall = monthly.loc[monthly["portfolio_scope"].eq("ALL")]
    assert len(overall) == 3
    assert overall.iloc[0]["start_turnover"] == pytest.approx(2.0)
    assert overall.iloc[-1]["terminal_turnover"] > 0
    assert (
        overall["q5_minus_q1_net30_return"]
        <= overall["q5_minus_q1_gross_return"]
    ).all()
    summary = summarize_monthly_portfolio(
        schedule,
        holdings,
        monthly,
        expected_month_start="2023-02",
        expected_month_end="2023-05",
        trial_floor=130,
    )
    assert summary["portfolio_months"] == 3
    assert summary["evaluation_months"] == 4
    assert summary["expected_month_coverage"] == pytest.approx(0.75)
    assert summary["missing_expected_months"] == ["2023-05"]
    assert summary["tests"]["net30"]["n"] == 3
    gates = monthly_portfolio_gate_table(
        summary,
        signed_events_minimum=1,
        unique_cik_minimum=1,
        portfolio_months_minimum=1,
        each_form_direction_minimum=1,
    )
    assert "minimum_portfolio_months" in set(gates["gate"])
    assert not gates.set_index("gate").loc["complete_expected_calendar_months", "passed"]


def test_rebalance_turnover_uses_gross_nav_normalized_drift():
    rows = []
    for ordinal, month, long_return, short_return in (
        (1, "2023-01", 0.20, 0.10),
        (2, "2023-02", 0.00, 0.00),
    ):
        for ticker, cik, direction, stock_return in (
            ("AAA", 1, 1, long_return),
            ("BBB", 2, -1, short_return),
        ):
            rows.append(
                {
                    "month": month,
                    "month_ordinal": ordinal,
                    "cohort": "2023-01",
                    "ticker": ticker,
                    "cik": cik,
                    "form": "10-K",
                    "quintile": 5 if direction == 1 else 1,
                    "direction": direction,
                    "current_accession": f"{ticker}-{month}",
                    "accepted_at": pd.Timestamp("2022-12-15", tz="UTC"),
                    "entry_date": pd.Timestamp(f"{month}-02"),
                    "entry_index": ordinal,
                    "exit_date": pd.Timestamp(f"{month}-28"),
                    "exit_index": ordinal + 1,
                    "gross_return": stock_return,
                }
            )
    _, monthly = _portfolio_scope_returns(
        pd.DataFrame(rows), scope="ALL", one_way_costs=(0.003, 0.005)
    )
    # End-Jan NAV is 1 + 20% - 10% = 1.10.  The long weight drifts to
    # 1.20/1.10 and the short to -1.10/1.10, so only 0.090909 is traded.
    assert monthly.iloc[1]["start_turnover"] == pytest.approx(1.0 / 11.0)


def test_unequal_cohorts_receive_equal_capital_by_frozen_assumption():
    rows = []
    specifications = (
        ("A", "A-L", 1, 1, 0.10),
        ("A", "A-S", 2, -1, 0.00),
        ("B", "B-L1", 3, 1, 0.00),
        ("B", "B-L2", 4, 1, 0.00),
        ("B", "B-L3", 5, 1, 0.00),
        ("B", "B-S", 6, -1, 0.00),
    )
    for cohort, ticker, cik, direction, stock_return in specifications:
        rows.append(
            {
                "month": "2023-01",
                "month_ordinal": 1,
                "cohort": cohort,
                "ticker": ticker,
                "cik": cik,
                "form": "10-K",
                "quintile": 5 if direction == 1 else 1,
                "direction": direction,
                "current_accession": ticker,
                "accepted_at": pd.Timestamp("2022-12-15", tz="UTC"),
                "entry_date": pd.Timestamp("2023-01-03"),
                "entry_index": 1,
                "exit_date": pd.Timestamp("2023-02-01"),
                "exit_index": 2,
                "gross_return": stock_return,
            }
        )
    holdings, monthly = _portfolio_scope_returns(
        pd.DataFrame(rows), scope="ALL", one_way_costs=(0.003, 0.005)
    )
    weight = holdings.loc[holdings["ticker"].eq("A-L"), "target_weight_component"].iloc[0]
    assert weight == pytest.approx(0.5)
    assert monthly.iloc[0]["q5_minus_q1_gross_return"] == pytest.approx(0.05)


def test_final_month_nonpositive_gross_nav_fails_closed():
    raw = pd.DataFrame(
        [
            {
                "month": "2023-01",
                "month_ordinal": 1,
                "cohort": "2023-01",
                "ticker": ticker,
                "cik": cik,
                "form": "10-K",
                "quintile": 5 if direction == 1 else 1,
                "direction": direction,
                "current_accession": ticker,
                "accepted_at": pd.Timestamp("2022-12-15", tz="UTC"),
                "entry_date": pd.Timestamp("2023-01-03"),
                "entry_index": 1,
                "exit_date": pd.Timestamp("2023-02-01"),
                "exit_index": 2,
                "gross_return": stock_return,
            }
            for ticker, cik, direction, stock_return in (
                ("AAA", 1, 1, -0.20),
                ("BBB", 2, -1, 1.10),
            )
        ]
    )
    with pytest.raises(FilingChangeStageAError, match="nonpositive"):
        _portfolio_scope_returns(raw, scope="ALL", one_way_costs=(0.003, 0.005))
    net_insolvent = raw.copy()
    net_insolvent.loc[net_insolvent["direction"].eq(1), "gross_return"] = 0.0
    net_insolvent.loc[net_insolvent["direction"].eq(-1), "gross_return"] = 0.99
    with pytest.raises(FilingChangeStageAError, match="net-cost NAV"):
        _portfolio_scope_returns(
            net_insolvent, scope="ALL", one_way_costs=(0.003, 0.005)
        )


def test_benchmark_clock_exactly_reconciles_to_independent_calendar():
    prices = _prices()
    spy = prices.loc[prices["ticker"].eq("SPY"), ["ticker", "date", "timestamp"]].copy()
    calendar = pd.DataFrame(
        {
            "session_date": spy["date"],
            "market_open_utc": spy["timestamp"],
            "market_close_utc": spy["timestamp"] + pd.Timedelta(hours=6, minutes=30),
        }
    )
    receipt = reconcile_benchmark_to_exchange_calendar(
        spy,
        calendar,
        required_start="2020-01-02",
        required_end="2025-12-31",
    )
    assert receipt["coverage"] == 1.0
    missing = spy.loc[spy["date"].ne(spy.iloc[100]["date"])]
    with pytest.raises(FilingChangeStageAError, match="missing=1"):
        reconcile_benchmark_to_exchange_calendar(
            missing,
            calendar,
            required_start="2020-01-02",
            required_end="2025-12-31",
        )


@pytest.mark.parametrize("bad_open", [0.0, np.nan])
def test_calendar_portfolio_rejects_invalid_exact_open(bad_open):
    prices = _prices()
    schedule, _ = build_calendar_preoutcome_schedule(_signals().iloc[:2], prices)
    first = schedule.iloc[0]
    mask = prices["ticker"].eq(first["ticker"]) & prices["date"].eq(first["entry_date"])
    prices.loc[mask, "adj_open"] = bad_open
    holdings, monthly, exclusions = calendar_portfolio_returns(schedule, prices)
    assert holdings.empty
    assert monthly.empty
    assert "invalid_exact_monthly_stock_price" in set(exclusions["reason"])


def test_benchmark_non_open_timestamp_fails_closed():
    prices = _prices()
    mask = prices["ticker"].eq("SPY")
    prices.loc[mask, "timestamp"] = prices.loc[mask, "timestamp"] + pd.Timedelta(minutes=1)
    with pytest.raises(FilingChangeStageAError, match="09:30"):
        build_preoutcome_schedule(_signals(), prices)


def test_summary_and_development_gates_do_not_promote_tiny_sample():
    prices = _prices()
    schedule, _ = build_preoutcome_schedule(_signals(), prices)
    events, _ = attach_event_outcomes(schedule, prices)
    summary = summarize_event_sample(
        events, trial_floor=130, inferential_test_names=("net30",)
    )
    assert summary["events"] == 2
    assert summary["tests"]["net30"]["disclosed_trial_floor"] == 130
    gates = development_gate_table(summary)
    assert not gates["passed"].all()
    assert not gates.set_index("gate").loc["minimum_signed_events", "passed"]


def test_undeclared_statistics_do_not_emit_p_values_and_incomplete_path_suppresses():
    prices = _prices()
    schedule, _ = build_preoutcome_schedule(_signals(), prices)
    events, _ = attach_event_outcomes(schedule, prices)
    descriptive = summarize_event_sample(events, trial_floor=130)
    assert descriptive["tests"]["gross"]["one_sided_p"] is None
    assert descriptive["tests"]["pre_placebo"]["one_sided_p"] is None
    inferred = summarize_event_sample(
        events,
        trial_floor=130,
        inferential_test_names=("pre_placebo",),
        inference_allowed=False,
    )
    assert inferred["tests"]["pre_placebo"]["one_sided_p"] is None
    assert inferred["tests"]["pre_placebo"]["status"].startswith("SUPPRESSED")
