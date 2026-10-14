"""Backward-only tradability-first repair for SEC reporting-lag sorting."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from alpha_factor_engine.contracts import validate_unique
from alpha_factor_engine.research.sec_reporting_lag_stage_a import (
    ReportingLagStageAError,
    _calendar_frame,
    _phase,
    prepare_annual_headers,
)


def _prepare_prices(prices: pd.DataFrame) -> pd.DataFrame:
    required = {"ticker", "date", "close", "volume"}
    missing = required.difference(prices.columns)
    if missing:
        raise ReportingLagStageAError(f"prices missing columns: {sorted(missing)}")
    frame = prices.loc[:, sorted(required)].copy()
    frame["ticker"] = frame["ticker"].astype("string").str.strip().str.upper()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
    if frame[["ticker", "date"]].isna().any().any():
        raise ReportingLagStageAError("prices contain invalid keys")
    validate_unique(frame, ["ticker", "date"], "tradability-first prices")
    return frame.sort_values(["ticker", "date"], kind="mergesort")


def build_tradability_first_plan(
    headers: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    cohorts: Mapping[str, Mapping[str, Any]],
    minimum_lag_days: int = 20,
    maximum_lag_days: int = 120,
    maximum_signal_age_days: int = 400,
    minimum_prior_close: float = 5.0,
    minimum_prior_stock_sessions: int = 252,
    adv_window_sessions: int = 63,
    minimum_prior_adv: float = 20_000_000.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Screen with data known at decision time, then assign lag quintiles."""

    annual, _ = prepare_annual_headers(headers)
    cal = _calendar_frame(calendar)
    cal["month"] = cal["session_date"].dt.to_period("M")
    panel = _prepare_prices(prices)
    by_ticker = {ticker: group.set_index("date") for ticker, group in panel.groupby("ticker", sort=False)}
    clocks: dict[pd.Period, dict[str, Any]] = {}
    for month, group in cal.groupby("month", sort=True):
        prior = cal.loc[cal["month"].lt(month)]
        if len(group) >= 2 and not prior.empty:
            clocks[month] = {
                "decision_at": prior.iloc[-1]["market_close_utc"] + pd.Timedelta(minutes=30),
                "decision_session": prior.iloc[-1]["session_date"],
                "entry_date": group.iloc[1]["session_date"],
                "entry_at": group.iloc[1]["market_open_utc"],
            }
    schedule_rows: list[dict[str, Any]] = []
    eligibility_rows: list[dict[str, Any]] = []
    for month in sorted(clocks):
        phase = _phase(month, cohorts)
        if phase is None or month + 1 not in clocks:
            continue
        clock = clocks[month]
        latest = annual.loc[annual["acceptance_at"].le(clock["decision_at"])].copy()
        latest = latest.sort_values(["ticker", "acceptance_at", "accession"], kind="mergesort").drop_duplicates("ticker", keep="last")
        latest["signal_age_days"] = (pd.Timestamp(clock["decision_session"]) - latest["filing_date"]).dt.days
        sessions = cal.loc[
            cal["session_date"].le(clock["decision_session"]), "session_date"
        ].tail(minimum_prior_stock_sessions)
        candidates: list[dict[str, Any]] = []
        for row in latest.itertuples(index=False):
            reason: str | None = None
            stock = by_ticker.get(str(row.ticker))
            prior_close = np.nan
            prior_adv = np.nan
            if not minimum_lag_days <= int(row.reporting_lag_days) <= maximum_lag_days:
                reason = "reporting_lag_out_of_bounds"
            elif not 0 <= int(row.signal_age_days) <= maximum_signal_age_days:
                reason = "signal_too_old"
            elif len(sessions) != minimum_prior_stock_sessions:
                reason = "benchmark_history_incomplete"
            elif stock is None:
                reason = "ticker_price_history_missing"
            elif not sessions.isin(stock.index).all():
                reason = "stock_history_incomplete"
            else:
                history = stock.loc[sessions]
                prior_close = float(history.iloc[-1]["close"])
                prior_adv = float((history["close"] * history["volume"]).tail(adv_window_sessions).median())
                if not np.isfinite(prior_close) or prior_close < minimum_prior_close:
                    reason = "prior_close_below_floor"
                elif not np.isfinite(prior_adv) or prior_adv < minimum_prior_adv:
                    reason = "prior_adv_below_floor"
            base = {
                "formation_month": str(month), "phase": phase, "cik": int(row.cik),
                "ticker": str(row.ticker), "source_accession": str(row.accession),
                "reporting_lag_days": int(row.reporting_lag_days), "signal_age_days": int(row.signal_age_days),
                "prior_close": prior_close, "prior_adv_usd": prior_adv,
                "eligibility_reason": reason or "eligible_before_sort", "eligible_before_sort": reason is None,
            }
            eligibility_rows.append(base)
            if reason is None:
                candidates.append({**base, "accepted_at": row.acceptance_at, "filing_date": row.filing_date, "report_date": row.report_date})
        eligible = pd.DataFrame(candidates)
        if len(eligible) < 5:
            continue
        eligible = eligible.sort_values(["reporting_lag_days", "source_accession", "ticker"], kind="mergesort").reset_index(drop=True)
        eligible["lag_quintile"] = (np.floor(np.arange(len(eligible)) * 5 / len(eligible)) + 1).astype(int)
        for row in eligible.loc[eligible["lag_quintile"].isin([1, 5])].itertuples(index=False):
            direction = 1 if int(row.lag_quintile) == 1 else -1
            next_clock = clocks[month + 1]
            schedule_rows.append({
                "current_accession": f"{row.source_accession}|{month}", "source_accession": row.source_accession,
                "cik": int(row.cik), "ticker": row.ticker, "form": "10-K", "accepted_at": row.accepted_at,
                "filing_date": row.filing_date, "report_date": row.report_date,
                "reporting_lag_days": int(row.reporting_lag_days), "signal_age_days": int(row.signal_age_days),
                "prior_close": float(row.prior_close), "prior_adv_usd": float(row.prior_adv_usd),
                "formation_month": str(month), "phase": phase, "decision_at": clock["decision_at"],
                "lag_quintile": int(row.lag_quintile), "similarity_quintile": 5 if direction == 1 else 1,
                "direction": direction, "entry_date": clock["entry_date"], "entry_at": clock["entry_at"],
                "calendar_exit_date": next_clock["entry_date"], "calendar_exit_at": next_clock["entry_at"],
                "outcome_join_permitted": phase != "sealed_2025", "outcome_join_performed": False,
            })
    schedule = pd.DataFrame(schedule_rows)
    if schedule.empty:
        raise ReportingLagStageAError("tradability-first schedule is empty")
    validate_unique(schedule, ["current_accession"], "tradability-first schedule")
    if not schedule["accepted_at"].le(schedule["decision_at"]).all():
        raise ReportingLagStageAError("future filing crossed decision clock")
    return schedule.sort_values(["formation_month", "direction", "ticker"], kind="mergesort"), pd.DataFrame(eligibility_rows)
