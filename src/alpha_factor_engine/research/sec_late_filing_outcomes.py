"""Frozen event schedule, outcome accounting, and inference for SEC Form NT.

The schedule is constructed before price values are supplied.  Callers must
exclude the sealed 2025 cohort before invoking :func:`attach_late_filing_outcomes`.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from alpha_factor_engine.contracts import validate_unique
from alpha_factor_engine.research.sec_filing_change_stage_a import (
    hac_mean_test,
    known_trial_floor_screen,
)


class LateFilingOutcomeError(ValueError):
    """Raised when a schedule, price path, or inference input is invalid."""


def prepare_xnys_calendar(calendar: pd.DataFrame) -> pd.DataFrame:
    required = {"session_date", "market_open_utc", "market_close_utc"}
    missing = required.difference(calendar.columns)
    if missing:
        raise LateFilingOutcomeError(f"XNYS calendar missing: {sorted(missing)}")
    frame = calendar.loc[:, sorted(required)].copy()
    frame["session_date"] = pd.to_datetime(
        frame["session_date"], errors="coerce"
    ).dt.normalize()
    frame["market_open_utc"] = pd.to_datetime(
        frame["market_open_utc"], utc=True, errors="coerce"
    )
    frame["market_close_utc"] = pd.to_datetime(
        frame["market_close_utc"], utc=True, errors="coerce"
    )
    if frame.isna().any().any():
        raise LateFilingOutcomeError("XNYS calendar contains invalid values")
    validate_unique(frame, ["session_date"], "late-filing XNYS calendar")
    frame = frame.sort_values("session_date", kind="mergesort").reset_index(drop=True)
    if not frame["market_open_utc"].is_monotonic_increasing:
        raise LateFilingOutcomeError("XNYS opens are not monotonic")
    local = frame["market_open_utc"].dt.tz_convert("America/New_York")
    if not (local.dt.hour.eq(9) & local.dt.minute.eq(30)).all():
        raise LateFilingOutcomeError("XNYS regular opens are not 09:30 New York")
    frame["session_index"] = np.arange(len(frame), dtype=int)
    return frame


def build_late_filing_schedule(
    candidates: pd.DataFrame,
    exchange_calendar: pd.DataFrame,
    *,
    decision_lag_minutes: int = 3,
    holding_sessions: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create event clocks and overlap decisions without any price input."""

    required = {
        "issuer_cik",
        "accession",
        "ticker",
        "form",
        "acceptance_at",
        "phase",
        "outcome_join_permitted",
    }
    missing = required.difference(candidates.columns)
    if missing:
        raise LateFilingOutcomeError(f"candidate events missing: {sorted(missing)}")
    if decision_lag_minutes < 0 or holding_sessions < 1:
        raise LateFilingOutcomeError("schedule lags must be nonnegative and nonzero")
    frame = candidates.copy()
    frame["ticker"] = frame["ticker"].astype("string").str.strip().str.upper()
    frame["acceptance_at"] = pd.to_datetime(
        frame["acceptance_at"], utc=True, errors="coerce"
    )
    if frame[["ticker", "acceptance_at"]].isna().any().any():
        raise LateFilingOutcomeError("candidate event keys or acceptance clocks are invalid")
    validate_unique(frame, ["accession"], "late-filing candidate accessions")
    calendar = prepare_xnys_calendar(exchange_calendar)
    opens_ns = calendar["market_open_utc"].astype("int64").to_numpy()

    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for event in frame.sort_values(
        ["acceptance_at", "ticker", "accession"], kind="mergesort"
    ).itertuples(index=False):
        decision_at = event.acceptance_at + pd.Timedelta(minutes=decision_lag_minutes)
        entry_index = int(np.searchsorted(opens_ns, decision_at.value, side="right"))
        exit_index = entry_index + holding_sessions
        pre_index = entry_index - holding_sessions
        if pre_index < 0 or exit_index >= len(calendar):
            exclusions.append(
                {
                    "accession": event.accession,
                    "ticker": event.ticker,
                    "reason": "calendar_path_incomplete",
                }
            )
            continue
        record = event._asdict()
        record.update(
            {
                "decision_at": decision_at,
                "pre_index": pre_index,
                "entry_index": entry_index,
                "exit_index": exit_index,
                "pre_date": calendar.iloc[pre_index]["session_date"],
                "entry_date": calendar.iloc[entry_index]["session_date"],
                "exit_date": calendar.iloc[exit_index]["session_date"],
                "entry_at": calendar.iloc[entry_index]["market_open_utc"],
                "exit_at": calendar.iloc[exit_index]["market_open_utc"],
                "holding_sessions": holding_sessions,
            }
        )
        rows.append(record)
    scheduled = pd.DataFrame(rows)
    if scheduled.empty:
        return scheduled, pd.DataFrame(exclusions)

    kept: list[dict[str, Any]] = []
    for ticker, group in scheduled.groupby("ticker", sort=True):
        last_exit = -1
        for event in group.sort_values(
            ["entry_index", "acceptance_at", "accession"], kind="mergesort"
        ).itertuples(index=False):
            if int(event.entry_index) < last_exit:
                exclusions.append(
                    {
                        "accession": event.accession,
                        "ticker": ticker,
                        "reason": "overlap_until_prior_exit",
                    }
                )
                continue
            kept.append(event._asdict())
            last_exit = int(event.exit_index)
    result = pd.DataFrame(kept).sort_values(
        ["entry_index", "ticker", "accession"], kind="mergesort"
    ).reset_index(drop=True)
    result["schedule_id"] = result["accession"].astype(str)
    validate_unique(result, ["schedule_id"], "late-filing schedule")
    if result.loc[result["phase"].eq("sealed_2025"), "outcome_join_permitted"].any():
        raise LateFilingOutcomeError("sealed 2025 schedule permits outcome access")
    return result, pd.DataFrame(exclusions)


def _prepare_prices(prices: pd.DataFrame) -> pd.DataFrame:
    required = {"ticker", "date", "timestamp", "adj_open", "dollar_volume", "available_at"}
    missing = required.difference(prices.columns)
    if missing:
        raise LateFilingOutcomeError(f"daily prices missing: {sorted(missing)}")
    frame = prices.loc[:, sorted(required)].copy()
    frame["ticker"] = frame["ticker"].astype("string").str.strip().str.upper()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["available_at"] = pd.to_datetime(
        frame["available_at"], utc=True, errors="coerce"
    )
    for column in ("adj_open", "dollar_volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["ticker", "date", "timestamp", "available_at"]].isna().any().any():
        raise LateFilingOutcomeError("daily prices contain invalid keys or clocks")
    validate_unique(frame, ["ticker", "date"], "late-filing daily prices")
    return frame.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)


def _row_on_date(frame: pd.DataFrame, date: pd.Timestamp) -> pd.Series | None:
    match = frame.loc[frame["date"].eq(pd.Timestamp(date).normalize())]
    if len(match) != 1:
        return None
    return match.iloc[0]


def _trailing_adv(
    stock: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    before_index: int,
    window: int,
    known_at: pd.Timestamp,
) -> tuple[float, pd.Timestamp] | None:
    if before_index < window:
        return None
    dates = set(calendar.iloc[before_index - window : before_index]["session_date"])
    history = stock.loc[stock["date"].isin(dates)].copy()
    if len(history) != window or history["date"].nunique() != window:
        return None
    values = history["dollar_volume"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        return None
    if not history["available_at"].le(known_at).all():
        return None
    return float(np.median(values)), pd.Timestamp(history["date"].max())


def attach_late_filing_outcomes(
    schedule: pd.DataFrame,
    prices: pd.DataFrame,
    exchange_calendar: pd.DataFrame,
    *,
    cost_bps_one_way: Iterable[int] = (30, 50),
    adv_window_sessions: int = 20,
    minimum_adv_usd: float = 5_000_000.0,
    portfolio_aum_usd: float = 10_000_000.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach exact-open pair returns only to nonsealed scheduled events."""

    if schedule.empty:
        return schedule.copy(), pd.DataFrame()
    if schedule["phase"].eq("sealed_2025").any():
        raise LateFilingOutcomeError("sealed 2025 events cannot enter outcome attachment")
    if schedule["outcome_join_permitted"].any():
        raise LateFilingOutcomeError("pre-outcome plan must remain fail-closed")
    costs = sorted({int(value) for value in cost_bps_one_way})
    if costs != [30, 50]:
        raise LateFilingOutcomeError("cost scenarios must be exactly 30 and 50 bps")
    if adv_window_sessions < 1 or minimum_adv_usd <= 0 or portfolio_aum_usd <= 0:
        raise LateFilingOutcomeError("liquidity and AUM parameters must be positive")
    calendar = prepare_xnys_calendar(exchange_calendar)
    px = _prepare_prices(prices)
    by_ticker = {ticker: group.reset_index(drop=True) for ticker, group in px.groupby("ticker")}
    spy = by_ticker.get("SPY")
    if spy is None:
        raise LateFilingOutcomeError("SPY benchmark is absent")

    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for event in schedule.sort_values(
        ["entry_index", "ticker", "accession"], kind="mergesort"
    ).itertuples(index=False):
        stock = by_ticker.get(str(event.ticker))
        if stock is None:
            exclusions.append({"schedule_id": event.schedule_id, "ticker": event.ticker, "reason": "ticker_price_history_absent"})
            continue
        stock_pre = _row_on_date(stock, event.pre_date)
        stock_entry = _row_on_date(stock, event.entry_date)
        stock_exit = _row_on_date(stock, event.exit_date)
        spy_pre = _row_on_date(spy, event.pre_date)
        spy_entry = _row_on_date(spy, event.entry_date)
        spy_exit = _row_on_date(spy, event.exit_date)
        exact_rows = (stock_pre, stock_entry, stock_exit, spy_pre, spy_entry, spy_exit)
        if any(row is None for row in exact_rows):
            exclusions.append({"schedule_id": event.schedule_id, "ticker": event.ticker, "reason": "exact_open_path_incomplete"})
            continue
        opens = np.array([row["adj_open"] for row in exact_rows], dtype=float)
        if not np.isfinite(opens).all() or (opens <= 0).any():
            exclusions.append({"schedule_id": event.schedule_id, "ticker": event.ticker, "reason": "invalid_adjusted_open"})
            continue
        entry_adv = _trailing_adv(
            stock,
            calendar,
            before_index=int(event.entry_index),
            window=adv_window_sessions,
            known_at=pd.Timestamp(event.decision_at),
        )
        exit_adv = _trailing_adv(
            stock,
            calendar,
            before_index=int(event.exit_index),
            window=adv_window_sessions,
            known_at=pd.Timestamp(event.exit_at),
        )
        if entry_adv is None or exit_adv is None:
            exclusions.append({"schedule_id": event.schedule_id, "ticker": event.ticker, "reason": "trailing_adv_incomplete_or_not_known"})
            continue
        if entry_adv[0] < minimum_adv_usd:
            exclusions.append({"schedule_id": event.schedule_id, "ticker": event.ticker, "reason": "entry_adv_below_frozen_minimum"})
            continue

        stock_return = float(stock_exit["adj_open"] / stock_entry["adj_open"] - 1.0)
        spy_return = float(spy_exit["adj_open"] / spy_entry["adj_open"] - 1.0)
        stock_pre_return = float(stock_entry["adj_open"] / stock_pre["adj_open"] - 1.0)
        spy_pre_return = float(spy_entry["adj_open"] / spy_pre["adj_open"] - 1.0)
        if min(1.0 + stock_return, 1.0 + spy_return) <= 0:
            raise LateFilingOutcomeError("pair leg has nonpositive terminal gross value")
        pair_gross = 0.5 * (spy_return - stock_return)
        pair_pre = 0.5 * (spy_pre_return - stock_pre_return)
        liquidation_notional = 0.5 * (1.0 + stock_return) + 0.5 * (1.0 + spy_return)
        traded_notional = 1.0 + liquidation_notional
        record = event._asdict()
        record.update(
            {
                "stock_entry_adj_open": float(stock_entry["adj_open"]),
                "stock_exit_adj_open": float(stock_exit["adj_open"]),
                "spy_entry_adj_open": float(spy_entry["adj_open"]),
                "spy_exit_adj_open": float(spy_exit["adj_open"]),
                "stock_return_20": stock_return,
                "spy_return_20": spy_return,
                "pair_gross_return_20": pair_gross,
                "pair_pre_event_gross_return_20": pair_pre,
                "post_minus_pre_return_20": pair_gross - pair_pre,
                "entry_trailing_adv_usd": entry_adv[0],
                "entry_adv_last_date": entry_adv[1],
                "exit_trailing_adv_usd": exit_adv[0],
                "exit_adv_last_date": exit_adv[1],
                "pair_traded_notional_per_gross": traded_notional,
                "maximum_stock_trade_to_adv_at_10m": max(
                    0.5 * portfolio_aum_usd / entry_adv[0],
                    0.5 * portfolio_aum_usd * (1.0 + stock_return) / exit_adv[0],
                ),
            }
        )
        for bps in costs:
            record[f"pair_net_{bps}bps_return_20"] = pair_gross - traded_notional * bps / 10_000.0
        rows.append(record)
    result = pd.DataFrame(rows)
    if not result.empty:
        validate_unique(result, ["schedule_id"], "late-filing event outcomes")
        numeric = [
            "stock_return_20",
            "spy_return_20",
            "pair_gross_return_20",
            "pair_pre_event_gross_return_20",
            "post_minus_pre_return_20",
            "pair_net_30bps_return_20",
            "pair_net_50bps_return_20",
            "maximum_stock_trade_to_adv_at_10m",
        ]
        if not np.isfinite(result[numeric].to_numpy(dtype=float)).all():
            raise LateFilingOutcomeError("event outcome contains nonfinite values")
        if not result["entry_adv_last_date"].lt(result["entry_date"]).all():
            raise LateFilingOutcomeError("entry liquidity window reaches the entry date")
        if not result["exit_adv_last_date"].lt(result["exit_date"]).all():
            raise LateFilingOutcomeError("exit liquidity window reaches the exit date")
    return result, pd.DataFrame(exclusions)


def aggregate_entry_date_returns(events: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "pair_gross_return_20",
        "pair_net_30bps_return_20",
        "pair_net_50bps_return_20",
        "pair_pre_event_gross_return_20",
        "post_minus_pre_return_20",
    ]
    required = {"entry_date", "entry_index", *metrics}
    missing = required.difference(events.columns)
    if missing:
        raise LateFilingOutcomeError(f"event outcomes missing: {sorted(missing)}")
    grouped = events.groupby("entry_date", sort=True)
    if grouped["entry_index"].nunique().gt(1).any():
        raise LateFilingOutcomeError("one entry date maps to multiple session indices")
    result = grouped[metrics].mean().reset_index()
    result["event_count"] = grouped.size().to_numpy()
    result["entry_index"] = grouped["entry_index"].first().to_numpy()
    return result


def _ticker_concentration(events: pd.DataFrame, metric: str) -> tuple[float, float]:
    if events.empty:
        return math.nan, math.nan
    counts = events.groupby("entry_date").size()
    contribution = events[metric] / events["entry_date"].map(counts)
    by_ticker = contribution.abs().groupby(events["ticker"]).sum()
    total = float(by_ticker.sum())
    maximum_share = float(by_ticker.max() / total) if total > 0 else math.nan
    loo_means: list[float] = []
    for ticker in sorted(events["ticker"].unique()):
        subset = events.loc[events["ticker"].ne(ticker)]
        if not subset.empty:
            loo_means.append(float(aggregate_entry_date_returns(subset)[metric].mean()))
    return maximum_share, min(loo_means) if loo_means else math.nan


def circular_block_bootstrap_mean(
    values: Iterable[float],
    *,
    block_length: int = 5,
    draws: int = 10_000,
    seed: int = 929,
) -> dict[str, Any]:
    """Centered circular-block null for the mean, returned as a sensitivity."""

    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3 or block_length < 1 or draws < 1:
        raise LateFilingOutcomeError("bootstrap inputs are insufficient")
    n = len(x)
    centered = x - x.mean()
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=float)
    blocks = int(math.ceil(n / block_length))
    offsets = np.arange(block_length, dtype=int)
    for draw in range(draws):
        starts = rng.integers(0, n, size=blocks)
        indices = ((starts[:, None] + offsets[None, :]) % n).reshape(-1)[:n]
        means[draw] = float(centered[indices].mean())
    observed = float(x.mean())
    p_value = float((1 + np.count_nonzero(means >= observed)) / (draws + 1))
    return {
        "status": "SENSITIVITY_ONLY",
        "n": n,
        "observed_mean": observed,
        "block_length": block_length,
        "draws": draws,
        "seed": seed,
        "centered_null_mean": float(means.mean()),
        "one_sided_p": p_value,
        "null_p05": float(np.quantile(means, 0.05)),
        "null_p95": float(np.quantile(means, 0.95)),
    }


def summarize_late_filing_events(
    events: pd.DataFrame,
    *,
    trial_floor: int,
    hac_lags: int = 3,
) -> dict[str, Any]:
    """Summarize one already-selected cohort with one formal primary test."""

    if events.empty:
        return {"events": 0, "entry_dates": 0, "tickers": 0}
    dates = aggregate_entry_date_returns(events)
    indices = dates["entry_index"].to_numpy(dtype=int)
    primary = known_trial_floor_screen(
        hac_mean_test(
            dates["pair_net_30bps_return_20"],
            session_indices=indices,
            lags=hac_lags,
            lag_unit="XNYS_session_distance",
        ),
        trial_floor=trial_floor,
    )
    bootstrap = circular_block_bootstrap_mean(
        dates["pair_net_30bps_return_20"],
        block_length=5,
        draws=10_000,
        seed=929,
    )
    gross = hac_mean_test(
        dates["pair_gross_return_20"],
        session_indices=indices,
        lags=hac_lags,
        lag_unit="XNYS_session_distance",
    )
    stress = hac_mean_test(
        dates["pair_net_50bps_return_20"],
        session_indices=indices,
        lags=hac_lags,
        lag_unit="XNYS_session_distance",
    )
    pre = hac_mean_test(
        dates["pair_pre_event_gross_return_20"],
        session_indices=indices,
        lags=hac_lags,
        lag_unit="XNYS_session_distance",
    )
    post_minus_pre = hac_mean_test(
        dates["post_minus_pre_return_20"],
        session_indices=indices,
        lags=hac_lags,
        lag_unit="XNYS_session_distance",
    )
    maximum_share, minimum_loo = _ticker_concentration(
        events, "pair_net_30bps_return_20"
    )
    return {
        "events": int(len(events)),
        "entry_dates": int(events["entry_date"].nunique()),
        "tickers": int(events["ticker"].nunique()),
        "forms": {key: int(value) for key, value in events["form"].value_counts().sort_index().items()},
        "gross": gross,
        "primary_net_30bps": primary,
        "primary_centered_circular_block_bootstrap": bootstrap,
        "stress_net_50bps": stress,
        "pre_event_placebo": pre,
        "post_minus_pre": post_minus_pre,
        "maximum_ticker_gross_contribution_share": maximum_share,
        "minimum_leave_one_ticker_out_mean": minimum_loo,
        "maximum_trade_to_adv_at_10m": float(events["maximum_stock_trade_to_adv_at_10m"].max()),
    }


def gate_late_filing_confirmation(
    summary: Mapping[str, Any],
    *,
    gates: Mapping[str, float | int],
    scheduled_events: int,
) -> pd.DataFrame:
    def value_or(mapping: Mapping[str, Any], key: str, fallback: float) -> float:
        value = mapping.get(key)
        return fallback if value is None else float(value)

    primary = summary.get("primary_net_30bps", {})
    stress = summary.get("stress_net_50bps", {})
    post_minus_pre = summary.get("post_minus_pre", {})
    coverage = float(summary.get("events", 0)) / scheduled_events if scheduled_events else 0.0
    rows = [
        ("minimum_events", int(summary.get("events", 0)) >= int(gates["minimum_events"]), summary.get("events"), gates["minimum_events"]),
        ("minimum_tickers", int(summary.get("tickers", 0)) >= int(gates["minimum_tickers"]), summary.get("tickers"), gates["minimum_tickers"]),
        ("minimum_entry_dates", int(summary.get("entry_dates", 0)) >= int(gates["minimum_entry_dates"]), summary.get("entry_dates"), gates["minimum_entry_dates"]),
        ("price_path_coverage", coverage >= float(gates["minimum_price_path_coverage"]), coverage, gates["minimum_price_path_coverage"]),
        ("primary_net_mean", value_or(primary, "mean", -math.inf) > float(gates["primary_net_mean_minimum"]), primary.get("mean"), gates["primary_net_mean_minimum"]),
        ("primary_hac_t", value_or(primary, "t", -math.inf) >= float(gates["primary_newey_west_t_minimum"]), primary.get("t"), gates["primary_newey_west_t_minimum"]),
        ("global_adjusted_p", value_or(primary, "known_floor_adjusted_screening_p", math.inf) <= float(gates["global_adjusted_p_maximum"]), primary.get("known_floor_adjusted_screening_p"), gates["global_adjusted_p_maximum"]),
        ("stress_net_mean", value_or(stress, "mean", -math.inf) > float(gates["stress_net_mean_minimum"]), stress.get("mean"), gates["stress_net_mean_minimum"]),
        ("post_minus_pre_mean", value_or(post_minus_pre, "mean", -math.inf) > float(gates["post_minus_pre_mean_minimum"]), post_minus_pre.get("mean"), gates["post_minus_pre_mean_minimum"]),
        ("ticker_contribution", float(summary.get("maximum_ticker_gross_contribution_share", math.inf)) <= float(gates["maximum_ticker_gross_contribution_share"]), summary.get("maximum_ticker_gross_contribution_share"), gates["maximum_ticker_gross_contribution_share"]),
        ("leave_one_ticker_out", float(summary.get("minimum_leave_one_ticker_out_mean", -math.inf)) > float(gates["minimum_leave_one_ticker_out_mean"]), summary.get("minimum_leave_one_ticker_out_mean"), gates["minimum_leave_one_ticker_out_mean"]),
        ("capacity_10m", float(summary.get("maximum_trade_to_adv_at_10m", math.inf)) <= float(gates["maximum_trade_to_trailing_adv_at_10m"]), summary.get("maximum_trade_to_adv_at_10m"), gates["maximum_trade_to_trailing_adv_at_10m"]),
    ]
    return pd.DataFrame(rows, columns=["gate", "passed", "observed", "threshold"])


__all__ = [
    "LateFilingOutcomeError",
    "aggregate_entry_date_returns",
    "attach_late_filing_outcomes",
    "build_late_filing_schedule",
    "circular_block_bootstrap_mean",
    "gate_late_filing_confirmation",
    "prepare_xnys_calendar",
    "summarize_late_filing_events",
]
