"""Execution-aware Stage A diagnostic for SEC filing-text change signals.

This module consumes already frozen document scores.  It does not download
filings and it does not choose a text model.  Signal schedules are built from
SEC clocks and the benchmark calendar before issuer forward returns are read.
Historical results from the bundled price panel remain ``INVALID`` because the
universe is a retrospective current-survivor ticker proxy.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import norm

from alpha_factor_engine.contracts import validate_unique


RESEARCH_VERDICT = "INVALID"
MODEL_VERDICT = "NOT VALIDATED"
DEPLOYMENT_VERDICT = "NOT AUTHORIZED"
BENCHMARK = "SPY"
PRIMARY_HORIZON_SESSIONS = 63
PRIMARY_ONE_WAY_COST = 0.003
STRESS_ONE_WAY_COST = 0.005


class FilingChangeStageAError(ValueError):
    """Raised when a Stage A calculation would violate the frozen protocol."""


def _prepare_benchmark_clock(prices: pd.DataFrame) -> pd.DataFrame:
    required = {"ticker", "date", "timestamp"}
    missing = required.difference(prices.columns)
    if missing:
        raise FilingChangeStageAError(
            f"benchmark clock missing columns: {sorted(missing)}"
        )
    frame = prices.loc[:, sorted(required)].copy()
    frame["ticker"] = frame["ticker"].astype("string").str.strip().str.upper()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if frame[["ticker", "date", "timestamp"]].isna().any().any():
        raise FilingChangeStageAError("benchmark clock contains null keys or timestamps")
    validate_unique(frame, ["ticker", "date"], "filing-change benchmark clock")
    return frame.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)


def _prepare_prices(prices: pd.DataFrame) -> pd.DataFrame:
    required = {"ticker", "date", "timestamp", "close", "volume", "adj_open"}
    missing = required.difference(prices.columns)
    if missing:
        raise FilingChangeStageAError(f"daily prices missing columns: {sorted(missing)}")
    frame = prices.loc[:, sorted(required)].copy()
    frame["ticker"] = frame["ticker"].astype("string").str.strip().str.upper()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    for column in ("close", "volume", "adj_open"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["ticker", "date", "timestamp"]].isna().any().any():
        raise FilingChangeStageAError("daily prices contain null keys or timestamps")
    validate_unique(frame, ["ticker", "date"], "filing-change daily prices")
    frame = frame.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)
    return frame


def _benchmark_sessions(prices: pd.DataFrame, benchmark: str = BENCHMARK) -> pd.DataFrame:
    spy = prices.loc[prices["ticker"].eq(benchmark)].copy()
    if spy.empty:
        raise FilingChangeStageAError(f"benchmark {benchmark} is absent")
    spy = spy.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    if not spy["timestamp"].is_monotonic_increasing:
        raise FilingChangeStageAError("benchmark timestamps are not monotonic")
    local = spy["timestamp"].dt.tz_convert("America/New_York")
    if not (local.dt.hour.eq(9) & local.dt.minute.eq(30)).all():
        raise FilingChangeStageAError("benchmark timestamps are not 09:30 New York opens")
    local_dates = local.dt.tz_localize(None).dt.normalize()
    if not local_dates.equals(spy["date"].reset_index(drop=True)):
        raise FilingChangeStageAError(
            "benchmark session dates differ from New York-local timestamp dates"
        )
    spy["session_index"] = np.arange(len(spy), dtype=int)
    spy["month"] = spy["date"].dt.to_period("M")
    return spy


def reconcile_benchmark_to_exchange_calendar(
    benchmark_clock: pd.DataFrame,
    exchange_calendar: pd.DataFrame,
    *,
    required_start: str | pd.Timestamp,
    required_end: str | pd.Timestamp,
    benchmark: str = BENCHMARK,
) -> dict[str, Any]:
    """Require exact benchmark-session agreement with a frozen XNYS snapshot."""

    clock = _prepare_benchmark_clock(benchmark_clock)
    spy = _benchmark_sessions(clock, benchmark)
    required = {"session_date", "market_open_utc", "market_close_utc"}
    missing = required.difference(exchange_calendar.columns)
    if missing:
        raise FilingChangeStageAError(
            f"exchange calendar missing columns: {sorted(missing)}"
        )
    calendar = exchange_calendar.loc[:, sorted(required)].copy()
    calendar["session_date"] = pd.to_datetime(
        calendar["session_date"], errors="coerce"
    ).dt.normalize()
    for column in ("market_open_utc", "market_close_utc"):
        calendar[column] = pd.to_datetime(calendar[column], utc=True, errors="coerce")
    if calendar.isna().any().any():
        raise FilingChangeStageAError("exchange calendar contains invalid values")
    validate_unique(calendar, ["session_date"], "XNYS exchange calendar")
    calendar = calendar.sort_values("session_date", kind="mergesort").reset_index(drop=True)
    start = pd.Timestamp(required_start).normalize().tz_localize(None)
    end = pd.Timestamp(required_end).normalize().tz_localize(None)
    if start > end:
        raise FilingChangeStageAError("exchange calendar reconciliation range is reversed")
    if calendar["session_date"].iloc[0] > start or calendar["session_date"].iloc[-1] < end:
        raise FilingChangeStageAError("exchange calendar does not span the required range")
    expected = calendar.loc[
        calendar["session_date"].between(start, end, inclusive="both")
    ].copy()
    actual = spy.loc[spy["date"].between(start, end, inclusive="both")].copy()
    expected_dates = set(expected["session_date"])
    actual_dates = set(actual["date"])
    if actual_dates != expected_dates:
        missing_dates = sorted(expected_dates - actual_dates)
        extra_dates = sorted(actual_dates - expected_dates)
        raise FilingChangeStageAError(
            "benchmark sessions differ from frozen XNYS calendar: "
            f"missing={len(missing_dates)} extra={len(extra_dates)}"
        )
    aligned = expected.merge(
        actual[["date", "timestamp"]],
        left_on="session_date",
        right_on="date",
        validate="one_to_one",
    )
    if not aligned["market_open_utc"].equals(aligned["timestamp"]):
        raise FilingChangeStageAError(
            "benchmark open timestamps differ from frozen XNYS calendar"
        )
    return {
        "calendar": "XNYS",
        "required_start": str(start.date()),
        "required_end": str(end.date()),
        "expected_sessions": int(len(expected)),
        "benchmark_sessions": int(len(actual)),
        "coverage": 1.0,
        "exact_session_dates": True,
        "exact_open_timestamps": True,
    }


def build_preoutcome_schedule(
    signals: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
    *,
    horizon_sessions: int = PRIMARY_HORIZON_SESSIONS,
    benchmark: str = BENCHMARK,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build tail-event entry/exit dates without opening issuer forward returns."""

    required = {
        "current_accession",
        "cik",
        "ticker",
        "form",
        "current_accepted_at",
        "similarity_quintile",
        "sort_eligible",
    }
    missing = required.difference(signals.columns)
    if missing:
        raise FilingChangeStageAError(f"signals missing columns: {sorted(missing)}")
    if horizon_sessions < 1:
        raise FilingChangeStageAError("horizon_sessions must be positive")
    frame = signals.copy()
    frame["current_accession"] = frame["current_accession"].astype("string").str.strip()
    frame["ticker"] = frame["ticker"].astype("string").str.strip().str.upper()
    frame["current_accepted_at"] = pd.to_datetime(
        frame["current_accepted_at"], utc=True, errors="coerce"
    )
    frame["similarity_quintile"] = pd.to_numeric(
        frame["similarity_quintile"], errors="coerce"
    ).astype("Int64")
    validate_unique(frame, ["current_accession"], "filing-change scores")
    eligible = (
        frame["sort_eligible"].eq(True)
        & frame["similarity_quintile"].isin([1, 5])
        & frame["current_accepted_at"].notna()
        & frame["ticker"].ne("")
    )
    exclusions: list[dict[str, Any]] = []
    for row in frame.loc[~eligible].itertuples(index=False):
        exclusions.append(
            {
                "current_accession": row.current_accession,
                "ticker": row.ticker,
                "reason": "not_sort_eligible_tail_signal",
            }
        )
    selected = frame.loc[eligible].copy()
    prices = _prepare_benchmark_clock(benchmark_prices)
    spy = _benchmark_sessions(prices, benchmark)
    timestamps = spy["timestamp"].astype("int64").to_numpy()
    months = spy["month"]
    # Use one common, conservative rebalance clock for every signal released in
    # a calendar month.  The second regular open of the following month is
    # necessarily at least the second benchmark open strictly after any
    # previous-month acceptance.  A common clock also makes the formal
    # calendar-time portfolio well defined.
    month_second: dict[pd.Period, int] = {}
    for month, group in spy.groupby("month", sort=True):
        if len(group) >= 2:
            month_second[month] = int(group.iloc[1]["session_index"])
    rows: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        accepted = row.current_accepted_at
        local_naive = accepted.tz_convert("America/New_York").tz_localize(None)
        release_month = local_naive.to_period("M")
        formation_month = release_month + 1
        second_month_index = month_second.get(formation_month)
        first_strict = int(np.searchsorted(timestamps, accepted.value, side="right"))
        safe_index = first_strict + 1  # quarantine one complete regular session
        if second_month_index is None or safe_index >= len(spy):
            exclusions.append(
                {
                    "current_accession": row.current_accession,
                    "ticker": row.ticker,
                    "reason": "formation_or_quarantine_open_unavailable",
                }
            )
            continue
        entry_index = int(second_month_index)
        exit_index = entry_index + int(horizon_sessions)
        placebo_end_index = first_strict - 1
        placebo_start_index = placebo_end_index - int(horizon_sessions)
        if (
            exit_index >= len(spy)
            or placebo_start_index < 1
            or spy.iloc[entry_index]["month"] != formation_month
            or entry_index < safe_index
        ):
            exclusions.append(
                {
                    "current_accession": row.current_accession,
                    "ticker": row.ticker,
                    "reason": "complete_clock_path_unavailable",
                }
            )
            continue
        rows.append(
            {
                "current_accession": str(row.current_accession),
                "cik": int(row.cik),
                "ticker": str(row.ticker),
                "form": str(row.form),
                "accepted_at": accepted,
                "release_month": str(release_month),
                "formation_month": str(formation_month),
                "similarity_quintile": int(row.similarity_quintile),
                "direction": 1 if int(row.similarity_quintile) == 5 else -1,
                "quarantine_date": spy.iloc[first_strict]["date"],
                "entry_date": spy.iloc[entry_index]["date"],
                "entry_at": spy.iloc[entry_index]["timestamp"],
                "entry_index": entry_index,
                "exit_date": spy.iloc[exit_index]["date"],
                "exit_at": spy.iloc[exit_index]["timestamp"],
                "exit_index": exit_index,
                "placebo_start_date": spy.iloc[placebo_start_index]["date"],
                "placebo_start_index": placebo_start_index,
                "placebo_end_date": spy.iloc[placebo_end_index]["date"],
                "placebo_end_index": placebo_end_index,
            }
        )
    schedule = pd.DataFrame(rows)
    if schedule.empty:
        return schedule, pd.DataFrame(exclusions)

    # Signals at one ticker/open that disagree are eliminated without consulting
    # issuer returns.  Same-direction filings are collapsed deterministically.
    conflict_keys = (
        schedule.groupby(["ticker", "entry_date"], sort=False)["direction"]
        .nunique()
        .loc[lambda values: values > 1]
        .index
    )
    if len(conflict_keys):
        conflict_set = set(conflict_keys)
        conflict_mask = schedule.apply(
            lambda row: (row["ticker"], row["entry_date"]) in conflict_set, axis=1
        )
        for row in schedule.loc[conflict_mask].itertuples(index=False):
            exclusions.append(
                {
                    "current_accession": row.current_accession,
                    "ticker": row.ticker,
                    "reason": "same_ticker_entry_direction_conflict",
                }
            )
        schedule = schedule.loc[~conflict_mask].copy()

    collapsed: list[dict[str, Any]] = []
    group_keys = [
        "ticker",
        "cik",
        "entry_date",
        "direction",
        "entry_index",
        "exit_index",
    ]
    for _, group in schedule.groupby(group_keys, sort=True):
        ordered = group.sort_values("current_accession", kind="mergesort")
        record = ordered.iloc[0].to_dict()
        record["accession_ids"] = "|".join(ordered["current_accession"].astype(str))
        record["collapsed_accession_count"] = len(ordered)
        collapsed.append(record)
    schedule = pd.DataFrame(collapsed).sort_values(
        ["ticker", "entry_index", "accession_ids"], kind="mergesort"
    )

    retained: list[pd.Series] = []
    for ticker, group in schedule.groupby("ticker", sort=True):
        prior_exit = -1
        for _, row in group.iterrows():
            if int(row.entry_index) <= prior_exit:
                exclusions.append(
                    {
                        "current_accession": row.accession_ids,
                        "ticker": ticker,
                        "reason": "overlap_until_prior_exit",
                    }
                )
                continue
            retained.append(row)
            prior_exit = int(row.exit_index)
    result = pd.DataFrame(retained).reset_index(drop=True)
    if not result.empty:
        result["schedule_id"] = (
            result["ticker"].astype(str)
            + "|"
            + result["entry_date"].dt.strftime("%Y-%m-%d")
            + "|"
            + result["direction"].astype(int).astype(str)
        )
        validate_unique(result, ["schedule_id"], "filing-change preoutcome schedule")
    return result, pd.DataFrame(exclusions)


def build_calendar_preoutcome_schedule(
    signals: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
    *,
    holding_months: int = 3,
    benchmark: str = BENCHMARK,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Freeze the formal monthly signal ledger without event-study filters.

    Quarterly signals are allowed to form consecutive three-month cohorts.
    Same-ticker signals within one formation month are resolved using only
    information known by the common second-open rebalance.
    """

    required = {
        "current_accession",
        "cik",
        "ticker",
        "form",
        "current_accepted_at",
        "similarity_quintile",
        "sort_eligible",
    }
    missing = required.difference(signals.columns)
    if missing:
        raise FilingChangeStageAError(f"signals missing columns: {sorted(missing)}")
    if holding_months < 1:
        raise FilingChangeStageAError("holding_months must be positive")
    frame = signals.copy()
    frame["current_accession"] = frame["current_accession"].astype("string").str.strip()
    frame["ticker"] = frame["ticker"].astype("string").str.strip().str.upper()
    frame["current_accepted_at"] = pd.to_datetime(
        frame["current_accepted_at"], utc=True, errors="coerce"
    )
    frame["similarity_quintile"] = pd.to_numeric(
        frame["similarity_quintile"], errors="coerce"
    ).astype("Int64")
    validate_unique(frame, ["current_accession"], "filing-change scores")
    eligible = (
        frame["sort_eligible"].eq(True)
        & frame["similarity_quintile"].isin([1, 5])
        & frame["current_accepted_at"].notna()
        & frame["ticker"].ne("")
    )
    exclusions: list[dict[str, Any]] = [
        {
            "current_accession": row.current_accession,
            "ticker": row.ticker,
            "reason": "not_sort_eligible_tail_signal",
        }
        for row in frame.loc[~eligible].itertuples(index=False)
    ]
    panel = _prepare_benchmark_clock(benchmark_prices)
    spy = _benchmark_sessions(panel, benchmark)
    timestamps = spy["timestamp"].astype("int64").to_numpy()
    month_open = _month_second_open(spy)
    rows: list[dict[str, Any]] = []
    for row in frame.loc[eligible].itertuples(index=False):
        accepted = row.current_accepted_at
        local_naive = accepted.tz_convert("America/New_York").tz_localize(None)
        release_month = local_naive.to_period("M")
        formation_month = release_month + 1
        entry_clock = month_open.get(formation_month)
        exit_clock = month_open.get(formation_month + holding_months)
        first_strict = int(np.searchsorted(timestamps, accepted.value, side="right"))
        safe_index = first_strict + 1
        if (
            entry_clock is None
            or exit_clock is None
            or safe_index >= len(spy)
            or entry_clock[0] < safe_index
        ):
            exclusions.append(
                {
                    "current_accession": row.current_accession,
                    "ticker": row.ticker,
                    "reason": "complete_calendar_clock_path_unavailable",
                }
            )
            continue
        rows.append(
            {
                "current_accession": str(row.current_accession),
                "cik": int(row.cik),
                "ticker": str(row.ticker),
                "form": str(row.form),
                "accepted_at": accepted,
                "release_month": str(release_month),
                "formation_month": str(formation_month),
                "similarity_quintile": int(row.similarity_quintile),
                "direction": 1 if int(row.similarity_quintile) == 5 else -1,
                "quarantine_date": spy.iloc[first_strict]["date"],
                "entry_date": entry_clock[1],
                "entry_at": spy.iloc[entry_clock[0]]["timestamp"],
                "entry_index": entry_clock[0],
                "calendar_exit_date": exit_clock[1],
                "calendar_exit_index": exit_clock[0],
            }
        )
    schedule = pd.DataFrame(rows)
    if schedule.empty:
        return schedule, pd.DataFrame(exclusions)

    conflict_keys = (
        schedule.groupby(["ticker", "entry_date"], sort=False)["direction"]
        .nunique()
        .loc[lambda values: values > 1]
        .index
    )
    if len(conflict_keys):
        conflicts = set(conflict_keys)
        mask = schedule.apply(
            lambda row: (row["ticker"], row["entry_date"]) in conflicts, axis=1
        )
        for row in schedule.loc[mask].itertuples(index=False):
            exclusions.append(
                {
                    "current_accession": row.current_accession,
                    "ticker": row.ticker,
                    "reason": "same_ticker_entry_direction_conflict",
                }
            )
        schedule = schedule.loc[~mask].copy()

    # One tradable issuer position per rebalance.  If same-direction forms land
    # in the same month, use the latest accepted filing; accession breaks exact
    # timestamp ties deterministically.
    schedule = schedule.sort_values(
        ["ticker", "entry_date", "accepted_at", "current_accession"],
        kind="mergesort",
    )
    duplicate = schedule.duplicated(["ticker", "entry_date"], keep="last")
    for row in schedule.loc[duplicate].itertuples(index=False):
        exclusions.append(
            {
                "current_accession": row.current_accession,
                "ticker": row.ticker,
                "reason": "same_ticker_month_superseded_by_later_filing",
            }
        )
    schedule = schedule.loc[~duplicate].copy().reset_index(drop=True)
    schedule["schedule_id"] = (
        schedule["ticker"].astype(str)
        + "|"
        + schedule["entry_date"].dt.strftime("%Y-%m-%d")
    )
    validate_unique(schedule, ["schedule_id"], "filing-change calendar schedule")
    return schedule, pd.DataFrame(exclusions)


def _lookup_row(frame: pd.DataFrame, date: pd.Timestamp) -> pd.Series | None:
    matched = frame.loc[frame["date"].eq(date)]
    return None if matched.empty else matched.iloc[0]


def attach_event_outcomes(
    schedule: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    minimum_prior_close: float = 5.0,
    adv_window: int = 63,
    minimum_prior_stock_sessions: int = 252,
    minimum_prior_adv: float = 20_000_000.0,
    beta_window: int = 252,
    minimum_beta_observations: int = 200,
    primary_one_way_cost: float = PRIMARY_ONE_WAY_COST,
    stress_one_way_cost: float = STRESS_ONE_WAY_COST,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach exact-date issuer outcomes after the schedule has been frozen."""

    required_schedule = {
        "schedule_id",
        "accession_ids",
        "cik",
        "ticker",
        "form",
        "accepted_at",
        "direction",
        "entry_date",
        "entry_at",
        "entry_index",
        "exit_date",
        "exit_index",
        "placebo_start_date",
        "placebo_start_index",
        "placebo_end_date",
        "placebo_end_index",
    }
    missing = required_schedule.difference(schedule.columns)
    if missing:
        raise FilingChangeStageAError(f"schedule missing columns: {sorted(missing)}")
    if min(adv_window, beta_window, minimum_beta_observations, minimum_prior_stock_sessions) < 1:
        raise FilingChangeStageAError("history windows must be positive")
    if minimum_prior_stock_sessions < adv_window:
        raise FilingChangeStageAError("minimum stock history is shorter than ADV window")
    panel = _prepare_prices(prices)
    spy = _benchmark_sessions(panel)
    spy_returns = spy.set_index("date")["adj_open"].pct_change(fill_method=None)
    by_ticker = {
        ticker: group.sort_values("date", kind="mergesort").reset_index(drop=True)
        for ticker, group in panel.groupby("ticker", sort=False)
    }
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for event in schedule.sort_values(["entry_index", "ticker"], kind="mergesort").itertuples(
        index=False
    ):
        stock = by_ticker.get(str(event.ticker))
        reason: str | None = None
        if stock is None:
            reason = "ticker_price_history_missing"
        else:
            entry = _lookup_row(stock, event.entry_date)
            exit_row = _lookup_row(stock, event.exit_date)
            placebo_start = _lookup_row(stock, event.placebo_start_date)
            placebo_end = _lookup_row(stock, event.placebo_end_date)
            history = stock.loc[stock["date"].lt(event.entry_date)]
            benchmark_history = spy.loc[
                spy["date"].lt(event.entry_date), "date"
            ].tail(minimum_prior_stock_sessions)
            if any(row is None for row in (entry, exit_row, placebo_start, placebo_end)):
                reason = "exact_stock_price_path_missing"
            elif any(
                not np.isfinite(float(row["adj_open"]))
                or float(row["adj_open"]) <= 0
                for row in (entry, exit_row, placebo_start, placebo_end)
                if row is not None
            ):
                reason = "invalid_exact_stock_price"
            elif len(benchmark_history) != minimum_prior_stock_sessions:
                reason = "insufficient_prior_stock_history"
            elif not benchmark_history.isin(set(history["date"])).all():
                reason = "incomplete_prior_stock_session_history"
            else:
                matched_history = history.set_index("date").loc[benchmark_history]
                prior = matched_history.iloc[-1]
                adv_values = (
                    matched_history["close"] * matched_history["volume"]
                ).tail(adv_window)
                if len(adv_values) != adv_window or not np.isfinite(adv_values).all():
                    reason = "prior_adv_window_incomplete"
                elif not np.isfinite(float(prior["close"])) or float(prior["close"]) < minimum_prior_close:
                    reason = "prior_close_below_floor"
                elif float(adv_values.median()) < minimum_prior_adv:
                    reason = "prior_adv_below_floor"
        if reason is not None:
            exclusions.append(
                {
                    "schedule_id": event.schedule_id,
                    "accession_ids": event.accession_ids,
                    "ticker": event.ticker,
                    "reason": reason,
                }
            )
            continue

        assert stock is not None
        assert entry is not None and exit_row is not None
        assert placebo_start is not None and placebo_end is not None
        stock_returns = (
            stock.set_index("date")["adj_open"]
            .reindex(spy["date"])
            .pct_change(fill_method=None)
        )
        beta_data = pd.concat(
            [stock_returns.rename("stock"), spy_returns.rename("spy")], axis=1, join="inner"
        ).dropna()
        beta_data = beta_data.loc[beta_data.index <= event.placebo_start_date].tail(beta_window)
        if len(beta_data) < minimum_beta_observations:
            exclusions.append(
                {
                    "schedule_id": event.schedule_id,
                    "accession_ids": event.accession_ids,
                    "ticker": event.ticker,
                    "reason": "insufficient_matched_beta_history",
                }
            )
            continue
        market_variance = float(beta_data["spy"].var(ddof=1))
        if not np.isfinite(market_variance) or market_variance <= 1e-20:
            exclusions.append(
                {
                    "schedule_id": event.schedule_id,
                    "accession_ids": event.accession_ids,
                    "ticker": event.ticker,
                    "reason": "degenerate_beta_market_variance",
                }
            )
            continue
        beta = float(beta_data[["stock", "spy"]].cov().iloc[0, 1] / market_variance)
        stock_return = float(exit_row["adj_open"] / entry["adj_open"] - 1.0)
        spy_return = float(
            spy.iloc[int(event.exit_index)]["adj_open"]
            / spy.iloc[int(event.entry_index)]["adj_open"]
            - 1.0
        )
        placebo_stock_return = float(
            placebo_end["adj_open"] / placebo_start["adj_open"] - 1.0
        )
        placebo_spy_return = float(
            spy.iloc[int(event.placebo_end_index)]["adj_open"]
            / spy.iloc[int(event.placebo_start_index)]["adj_open"]
            - 1.0
        )
        direction = int(event.direction)
        stock_weight = direction / (1.0 + abs(beta))
        benchmark_weight = -direction * beta / (1.0 + abs(beta))
        gross = stock_weight * stock_return + benchmark_weight * spy_return
        placebo_gross = (
            stock_weight * placebo_stock_return + benchmark_weight * placebo_spy_return
        )

        def cost(rate: float) -> float:
            return float(
                rate
                * (
                    abs(stock_weight)
                    + abs(benchmark_weight)
                    + abs(stock_weight * (1.0 + stock_return))
                    + abs(benchmark_weight * (1.0 + spy_return))
                )
            )

        record = event._asdict()
        record.update(
            {
                "prior_close": float(matched_history.iloc[-1]["close"]),
                "prior_adv_usd": float(
                    (matched_history["close"] * matched_history["volume"])
                    .tail(adv_window)
                    .median()
                ),
                "liquidity_last_date": matched_history.index[-1],
                "beta": beta,
                "beta_observations": int(len(beta_data)),
                "beta_sample_end_date": beta_data.index.max(),
                "stock_weight": stock_weight,
                "benchmark_weight": benchmark_weight,
                "stock_return_63": stock_return,
                "spy_return_63": spy_return,
                "hedged_gross_return_63": gross,
                "traded_notional_cost_30bps": cost(primary_one_way_cost),
                "traded_notional_cost_50bps": cost(stress_one_way_cost),
                "hedged_net_return_30bps": gross - cost(primary_one_way_cost),
                "hedged_net_return_50bps": gross - cost(stress_one_way_cost),
                "placebo_stock_return_63": placebo_stock_return,
                "placebo_spy_return_63": placebo_spy_return,
                "placebo_hedged_gross_return_63": placebo_gross,
            }
        )
        rows.append(record)
    events = pd.DataFrame(rows)
    if not events.empty:
        numeric = [
            "beta",
            "stock_weight",
            "benchmark_weight",
            "stock_return_63",
            "spy_return_63",
            "hedged_gross_return_63",
            "hedged_net_return_30bps",
            "hedged_net_return_50bps",
            "placebo_hedged_gross_return_63",
        ]
        if not np.isfinite(events[numeric].to_numpy(dtype=float)).all():
            raise FilingChangeStageAError("retained outcomes contain non-finite values")
        if not events["liquidity_last_date"].lt(events["entry_date"]).all():
            raise FilingChangeStageAError("liquidity used entry or future observations")
        if not events["beta_sample_end_date"].le(events["placebo_start_date"]).all():
            raise FilingChangeStageAError("beta window overlaps placebo or event return")
    return events, pd.DataFrame(exclusions)


def aggregate_entry_dates(events: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "hedged_gross_return_63",
        "hedged_net_return_30bps",
        "hedged_net_return_50bps",
        "placebo_hedged_gross_return_63",
    ]
    required = {"entry_date", "entry_index", *metrics}
    missing = required.difference(events.columns)
    if missing:
        raise FilingChangeStageAError(f"event outcomes missing columns: {sorted(missing)}")
    grouped = events.groupby("entry_date", sort=True)
    if grouped["entry_index"].nunique().gt(1).any():
        raise FilingChangeStageAError("entry date maps to multiple benchmark indices")
    result = grouped[metrics].mean().reset_index()
    result["event_count"] = grouped.size().to_numpy()
    result["entry_index"] = grouped["entry_index"].first().to_numpy()
    return result


def hac_mean_test(
    values: Iterable[float],
    *,
    session_indices: Iterable[int] | None = None,
    lags: int = PRIMARY_HORIZON_SESSIONS,
    lag_unit: str = "SPY_session_distance",
) -> dict[str, Any]:
    """One-sided mean test using a Bartlett HAC by benchmark-session distance."""

    x = np.asarray(list(values), dtype=float)
    indices = (
        np.arange(len(x), dtype=int)
        if session_indices is None
        else np.asarray(list(session_indices), dtype=int)
    )
    if len(indices) != len(x):
        raise FilingChangeStageAError("HAC values and indices differ in length")
    finite = np.isfinite(x)
    x = x[finite]
    indices = indices[finite]
    if len(indices) and np.any(np.diff(indices) <= 0):
        raise FilingChangeStageAError("HAC session indices must be strictly increasing")
    n = len(x)
    if n < 3:
        return {"status": "INSUFFICIENT_ENTRY_DATES", "n": n, "mean": float(x.mean()) if n else None, "t": None, "one_sided_p": None}
    mean = float(x.mean())
    centered = x - mean
    variance_sum = float(centered @ centered)
    for right in range(1, n):
        gaps = indices[right] - indices[:right]
        included = gaps <= lags
        if included.any():
            weights = 1.0 - gaps[included] / (lags + 1.0)
            variance_sum += float(2.0 * np.dot(weights, centered[right] * centered[:right][included]))
    variance_of_mean = variance_sum / (n * n)
    if not np.isfinite(variance_of_mean) or variance_of_mean <= 1e-24:
        return {"status": "DEGENERATE_VARIANCE", "n": n, "mean": mean, "t": None, "one_sided_p": None}
    standard_error = math.sqrt(variance_of_mean)
    t_value = mean / standard_error
    return {
        "status": "OK",
        "n": n,
        "mean": mean,
        "lags": lags,
        "lag_unit": lag_unit,
        "standard_error": standard_error,
        "t": float(t_value),
        "one_sided_p": float(norm.sf(t_value)),
    }


def known_trial_floor_screen(test: Mapping[str, Any], *, trial_floor: int) -> dict[str, Any]:
    if trial_floor < 1:
        raise FilingChangeStageAError("trial_floor must be positive")
    result = dict(test)
    raw_p = result.get("one_sided_p")
    result.update(
        {
            "disclosed_trial_floor": trial_floor,
            "one_sided_normal_critical_value": float(norm.ppf(1.0 - 0.05 / trial_floor)),
            "known_floor_adjusted_screening_p": (
                None if raw_p is None else min(1.0, float(raw_p) * trial_floor)
            ),
            "family_wise_error_control_established": False,
        }
    )
    return result


def _mean_test_record(
    values: Iterable[float],
    *,
    session_indices: Iterable[int],
    lags: int,
    lag_unit: str,
    trial_floor: int,
    inferential: bool,
    formal_mean_gate: bool,
    inference_allowed: bool,
) -> dict[str, Any]:
    value_series = pd.to_numeric(pd.Series(list(values)), errors="coerce")
    index_series = pd.to_numeric(pd.Series(list(session_indices)), errors="coerce")
    valid = value_series.notna() & np.isfinite(value_series) & index_series.notna()
    clean_values = value_series.loc[valid]
    clean_indices = index_series.loc[valid]
    if inferential and inference_allowed:
        result = known_trial_floor_screen(
            hac_mean_test(
                clean_values,
                session_indices=clean_indices,
                lags=lags,
                lag_unit=lag_unit,
            ),
            trial_floor=trial_floor,
        )
        result["predeclared_inferential_test"] = True
        return result
    return {
        "status": (
            "SUPPRESSED_INCOMPLETE_FORMAL_PRICE_PATH"
            if (inferential or formal_mean_gate) and not inference_allowed
            else (
                "PREDECLARED_MEAN_GATE_NO_P_VALUE"
                if formal_mean_gate
                else "DESCRIPTIVE_ONLY_NO_P_VALUE"
            )
        ),
        "n": int(len(clean_values)),
        "mean": float(clean_values.mean()) if len(clean_values) else None,
        "lags": lags,
        "lag_unit": lag_unit,
        "standard_error": None,
        "t": None,
        "one_sided_p": None,
        "disclosed_trial_floor": trial_floor if inferential else None,
        "one_sided_normal_critical_value": (
            float(norm.ppf(1.0 - 0.05 / trial_floor)) if inferential else None
        ),
        "known_floor_adjusted_screening_p": None,
        "family_wise_error_control_established": False,
        "predeclared_inferential_test": inferential,
    }


def summarize_event_sample(
    events: pd.DataFrame,
    *,
    trial_floor: int = 130,
    inferential_test_names: Iterable[str] = (),
    formal_mean_gate_names: Iterable[str] = (),
    inference_allowed: bool = True,
) -> dict[str, Any]:
    """Summarize one frozen candidate without changing its observations."""

    if events.empty:
        return {
            "events": 0,
            "unique_cik": 0,
            "entry_dates": 0,
            "tests": {},
            "by_form_direction": [],
            "by_form": [],
            "by_year": [],
            "maximum_single_cik_absolute_contribution_share": None,
            "maximum_entry_date_absolute_contribution_share": None,
            "minimum_leave_one_cik_out_net30_mean": None,
        }
    required = {
        "cik",
        "form",
        "direction",
        "entry_date",
        "entry_index",
        "hedged_gross_return_63",
        "hedged_net_return_30bps",
        "hedged_net_return_50bps",
        "placebo_hedged_gross_return_63",
    }
    missing = required.difference(events.columns)
    if missing:
        raise FilingChangeStageAError(f"events missing columns: {sorted(missing)}")
    dates = aggregate_entry_dates(events)
    indices = dates["entry_index"]
    inferential_names = set(inferential_test_names)
    formal_mean_names = set(formal_mean_gate_names)
    supported_names = {"gross", "net30", "net50", "pre_placebo", "post_minus_pre"}
    if (
        not inferential_names.issubset(supported_names)
        or not formal_mean_names.issubset(supported_names)
        or inferential_names.intersection(formal_mean_names)
    ):
        raise FilingChangeStageAError("unknown event inferential test name")
    tests: dict[str, Any] = {}
    for name, column in (
        ("gross", "hedged_gross_return_63"),
        ("net30", "hedged_net_return_30bps"),
        ("net50", "hedged_net_return_50bps"),
        ("pre_placebo", "placebo_hedged_gross_return_63"),
    ):
        tests[name] = _mean_test_record(
            dates[column],
            session_indices=indices,
            lags=63,
            lag_unit="session_index_distance",
            trial_floor=trial_floor,
            inferential=name in inferential_names,
            formal_mean_gate=name in formal_mean_names,
            inference_allowed=inference_allowed,
        )
    post_minus_pre = (
        dates["hedged_gross_return_63"]
        - dates["placebo_hedged_gross_return_63"]
    )
    tests["post_minus_pre"] = _mean_test_record(
        post_minus_pre,
        session_indices=indices,
        lags=63,
        lag_unit="session_index_distance",
        trial_floor=trial_floor,
        inferential="post_minus_pre" in inferential_names,
        formal_mean_gate="post_minus_pre" in formal_mean_names,
        inference_allowed=inference_allowed,
    )

    grouped_counts = events.groupby("entry_date", sort=False).size()
    contribution = events["hedged_net_return_30bps"] / events["entry_date"].map(
        grouped_counts
    ) / len(dates)
    gross_absolute_total = float(contribution.abs().sum())
    by_cik = contribution.abs().groupby(events["cik"]).sum()
    max_cik = (
        float(by_cik.max() / gross_absolute_total)
        if gross_absolute_total > 0 and not by_cik.empty
        else None
    )
    date_absolute = dates["hedged_net_return_30bps"].abs()
    max_date = (
        float(date_absolute.max() / date_absolute.sum())
        if float(date_absolute.sum()) > 0
        else None
    )
    leave_one_out: list[float] = []
    for cik in sorted(events["cik"].unique()):
        subset = events.loc[events["cik"].ne(cik)]
        if subset.empty:
            continue
        leave_one_out.append(
            float(aggregate_entry_dates(subset)["hedged_net_return_30bps"].mean())
        )
    by_form_direction: list[dict[str, Any]] = []
    for (form, direction), group in events.groupby(["form", "direction"], sort=True):
        date_means = group.groupby("entry_date", sort=True)["hedged_net_return_30bps"].mean()
        by_form_direction.append(
            {
                "form": form,
                "direction": int(direction),
                "events": int(len(group)),
                "unique_cik": int(group["cik"].nunique()),
                "entry_dates": int(len(date_means)),
                "net30_mean": float(date_means.mean()),
            }
        )
    by_form: list[dict[str, Any]] = []
    for form, group in events.groupby("form", sort=True):
        date_means = group.groupby("entry_date", sort=True)["hedged_net_return_30bps"].mean()
        by_form.append(
            {
                "form": form,
                "events": int(len(group)),
                "unique_cik": int(group["cik"].nunique()),
                "entry_dates": int(len(date_means)),
                "net30_mean": float(date_means.mean()),
            }
        )
    year_frame = dates.copy()
    year_frame["entry_year"] = pd.to_datetime(year_frame["entry_date"]).dt.year
    by_year = (
        year_frame.groupby("entry_year", sort=True)["hedged_net_return_30bps"]
        .mean()
        .rename("net30_mean")
        .reset_index()
        .to_dict("records")
    )
    return {
        "events": int(len(events)),
        "unique_cik": int(events["cik"].nunique()),
        "entry_dates": int(events["entry_date"].nunique()),
        "tests": tests,
        "by_form_direction": by_form_direction,
        "by_form": by_form,
        "by_year": by_year,
        "maximum_single_cik_absolute_contribution_share": max_cik,
        "maximum_entry_date_absolute_contribution_share": max_date,
        "minimum_leave_one_cik_out_net30_mean": (
            min(leave_one_out) if leave_one_out else None
        ),
    }


def development_gate_table(
    summary: Mapping[str, Any],
    *,
    signed_events_minimum: int = 1_500,
    unique_cik_minimum: int = 600,
    entry_dates_minimum: int = 36,
    each_form_direction_minimum: int = 250,
) -> pd.DataFrame:
    """Evaluate the frozen development gates for one similarity candidate."""

    tests = summary.get("tests", {})
    if not tests:
        return pd.DataFrame(
            [
                {
                    "gate": "nonempty_event_sample",
                    "passed": False,
                    "required": True,
                    "detail": 0,
                }
            ]
        )
    net30 = tests["net30"]
    gross = tests["gross"]
    net50 = tests["net50"]
    placebo = tests["pre_placebo"]
    post_minus_pre = tests["post_minus_pre"]
    buckets = summary["by_form_direction"]
    minimum_bucket = min((int(row["events"]) for row in buckets), default=0)
    bucket_keys = {(row["form"], int(row["direction"])) for row in buckets}
    form_rows = summary.get("by_form", [])
    both_forms_nonnegative = {
        str(row["form"]) for row in form_rows if float(row["net30_mean"]) >= 0
    } == {"10-K", "10-Q"}
    years_positive = bool(summary["by_year"]) and all(
        float(row["net30_mean"]) > 0 for row in summary["by_year"]
    )
    placebo_pass = not (
        placebo.get("mean") is not None
        and float(placebo["mean"]) > 0
        and placebo.get("one_sided_p") is not None
        and float(placebo["one_sided_p"]) <= 0.05
    )
    concentration_cik = summary["maximum_single_cik_absolute_contribution_share"]
    concentration_date = summary["maximum_entry_date_absolute_contribution_share"]
    leave_one_out = summary["minimum_leave_one_cik_out_net30_mean"]
    critical = net30["one_sided_normal_critical_value"]
    rows = [
        ("minimum_signed_events", summary["events"] >= signed_events_minimum, summary["events"]),
        ("minimum_unique_cik", summary["unique_cik"] >= unique_cik_minimum, summary["unique_cik"]),
        ("minimum_entry_dates", summary["entry_dates"] >= entry_dates_minimum, summary["entry_dates"]),
        (
            "minimum_each_form_direction",
            bucket_keys == {("10-K", -1), ("10-K", 1), ("10-Q", -1), ("10-Q", 1)}
            and minimum_bucket >= each_form_direction_minimum,
            minimum_bucket,
        ),
        ("gross_mean_positive", gross.get("mean") is not None and gross["mean"] > 0, gross.get("mean")),
        ("net30_mean_positive", net30.get("mean") is not None and net30["mean"] > 0, net30.get("mean")),
        ("net50_mean_positive", net50.get("mean") is not None and net50["mean"] > 0, net50.get("mean")),
        ("net30_known_floor_t", net30.get("t") is not None and net30["t"] >= critical, {"t": net30.get("t"), "critical": critical}),
        ("net30_known_floor_p", net30.get("known_floor_adjusted_screening_p") is not None and net30["known_floor_adjusted_screening_p"] <= 0.05, net30.get("known_floor_adjusted_screening_p")),
        ("each_calendar_year_positive", years_positive, summary["by_year"]),
        ("both_forms_nonnegative", both_forms_nonnegative, form_rows),
        ("pre_event_placebo_falsification", placebo_pass, {"mean": placebo.get("mean"), "p": placebo.get("one_sided_p")}),
        ("post_minus_pre_positive", post_minus_pre.get("mean") is not None and post_minus_pre["mean"] > 0, post_minus_pre.get("mean")),
        ("maximum_single_cik_contribution", concentration_cik is not None and concentration_cik <= 0.02, concentration_cik),
        ("maximum_entry_date_contribution", concentration_date is not None and concentration_date <= 0.05, concentration_date),
        ("leave_one_cik_out_positive", leave_one_out is not None and leave_one_out > 0, leave_one_out),
    ]
    return pd.DataFrame(
        [
            {"gate": name, "passed": bool(passed), "required": True, "detail": detail}
            for name, passed, detail in rows
        ]
    )


def _month_second_open(spy: pd.DataFrame) -> dict[pd.Period, tuple[int, pd.Timestamp]]:
    result: dict[pd.Period, tuple[int, pd.Timestamp]] = {}
    for month, group in spy.groupby("month", sort=True):
        if len(group) >= 2:
            row = group.iloc[1]
            result[month] = (int(row["session_index"]), row["date"])
    return result


def _eligible_calendar_holding_returns(
    schedule: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    holding_months: int,
    minimum_prior_close: float,
    adv_window: int,
    minimum_prior_stock_sessions: int,
    minimum_prior_adv: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "current_accession",
        "cik",
        "ticker",
        "form",
        "accepted_at",
        "formation_month",
        "similarity_quintile",
        "direction",
        "entry_date",
    }
    missing = required.difference(schedule.columns)
    if missing:
        raise FilingChangeStageAError(f"schedule missing columns: {sorted(missing)}")
    spy = _benchmark_sessions(panel)
    month_open = _month_second_open(spy)
    by_ticker = {
        ticker: group.sort_values("date", kind="mergesort").set_index("date")
        for ticker, group in panel.groupby("ticker", sort=False)
    }
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for event in schedule.sort_values(["entry_date", "ticker"], kind="mergesort").itertuples(
        index=False
    ):
        ticker = str(event.ticker)
        stock = by_ticker.get(ticker)
        formation_month = pd.Period(str(event.formation_month), freq="M")
        first_clock = month_open.get(formation_month)
        reason: str | None = None
        if stock is None:
            reason = "ticker_price_history_missing"
        elif first_clock is None or pd.Timestamp(event.entry_date) != first_clock[1]:
            raise FilingChangeStageAError(
                "calendar portfolio entry differs from frozen common second-open clock"
            )
        else:
            benchmark_history = spy.loc[
                spy["date"].lt(first_clock[1]), "date"
            ].tail(minimum_prior_stock_sessions)
            if len(benchmark_history) != minimum_prior_stock_sessions:
                reason = "insufficient_prior_stock_history"
            elif not benchmark_history.isin(stock.index).all():
                reason = "incomplete_prior_stock_session_history"
            else:
                history = stock.loc[benchmark_history]
                prior = history.iloc[-1]
                adv = (history["close"] * history["volume"]).tail(adv_window)
                if len(adv) != adv_window or not np.isfinite(adv).all():
                    reason = "prior_adv_window_incomplete"
                elif not np.isfinite(float(prior["close"])) or float(prior["close"]) < minimum_prior_close:
                    reason = "prior_close_below_floor"
                elif float(adv.median()) < minimum_prior_adv:
                    reason = "prior_adv_below_floor"
        paths: list[tuple[pd.Period, int, pd.Timestamp, int, pd.Timestamp]] = []
        if reason is None:
            assert stock is not None
            for offset in range(holding_months):
                month = formation_month + offset
                entry_clock = month_open.get(month)
                exit_clock = month_open.get(month + 1)
                if entry_clock is None or exit_clock is None:
                    reason = "complete_calendar_clock_path_unavailable"
                    break
                entry_index, entry_date = entry_clock
                exit_index, exit_date = exit_clock
                if entry_date not in stock.index or exit_date not in stock.index:
                    reason = "exact_monthly_stock_price_path_missing"
                    break
                entry_open = float(stock.loc[entry_date, "adj_open"])
                exit_open = float(stock.loc[exit_date, "adj_open"])
                if (
                    not np.isfinite(entry_open)
                    or not np.isfinite(exit_open)
                    or entry_open <= 0
                    or exit_open <= 0
                ):
                    reason = "invalid_exact_monthly_stock_price"
                    break
                paths.append((month, entry_index, entry_date, exit_index, exit_date))
        if reason is not None:
            exclusions.append(
                {
                    "current_accession": event.current_accession,
                    "ticker": ticker,
                    "reason": reason,
                }
            )
            continue
        assert stock is not None
        for month, entry_index, entry_date, exit_index, exit_date in paths:
            entry_open = float(stock.loc[entry_date, "adj_open"])
            exit_open = float(stock.loc[exit_date, "adj_open"])
            rows.append(
                {
                    "month": str(month),
                    "month_ordinal": int(month.ordinal),
                    "cohort": str(formation_month),
                    "ticker": ticker,
                    "cik": int(event.cik),
                    "form": str(event.form),
                    "quintile": int(event.similarity_quintile),
                    "direction": int(event.direction),
                    "current_accession": str(event.current_accession),
                    "accepted_at": event.accepted_at,
                    "entry_date": entry_date,
                    "entry_index": entry_index,
                    "exit_date": exit_date,
                    "exit_index": exit_index,
                    "gross_return": exit_open / entry_open - 1.0,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(exclusions)


def _portfolio_scope_returns(
    raw_holdings: pd.DataFrame,
    *,
    scope: str,
    one_way_costs: tuple[float, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = raw_holdings if scope == "ALL" else raw_holdings.loc[raw_holdings["form"].eq(scope)]
    target_rows: list[dict[str, Any]] = []
    for month, month_frame in frame.groupby("month", sort=True):
        valid_cohorts: list[pd.DataFrame] = []
        for _, cohort in month_frame.groupby("cohort", sort=True):
            if set(cohort["direction"].astype(int)) == {-1, 1}:
                valid_cohorts.append(cohort)
        if not valid_cohorts:
            continue
        cohort_scale = 1.0 / len(valid_cohorts)
        for cohort in valid_cohorts:
            for direction, side in cohort.groupby("direction", sort=True):
                side_scale = cohort_scale / len(side)
                for row in side.itertuples(index=False):
                    record = row._asdict()
                    record["portfolio_scope"] = scope
                    record["target_weight_component"] = int(direction) * side_scale
                    target_rows.append(record)
    components = pd.DataFrame(target_rows)
    if components.empty:
        return components, pd.DataFrame()
    weighted_rows: list[dict[str, Any]] = []
    for (month, ticker), group in components.groupby(["month", "ticker"], sort=True):
        if group["gross_return"].nunique(dropna=False) != 1:
            raise FilingChangeStageAError("ticker-month has inconsistent exact returns")
        row = group.iloc[-1].to_dict()
        row["target_weight"] = float(group["target_weight_component"].sum())
        row["component_count"] = int(len(group))
        weighted_rows.append(row)
    weighted = pd.DataFrame(weighted_rows).sort_values(
        ["month_ordinal", "ticker"], kind="mergesort"
    )

    monthly_rows: list[dict[str, Any]] = []
    previous_weights: dict[str, float] = {}
    previous_returns: dict[str, float] = {}
    previous_nav_growth: float | None = None
    previous_ordinal: int | None = None
    for month, group in weighted.groupby("month", sort=True):
        ordinal = int(group["month_ordinal"].iloc[0])
        if previous_ordinal is None or ordinal != previous_ordinal + 1:
            start_turnover = float(group["target_weight"].abs().sum())
            if monthly_rows and previous_weights:
                monthly_rows[-1]["terminal_turnover"] = float(
                    sum(
                        abs(weight * (1.0 + previous_returns[ticker]))
                        for ticker, weight in previous_weights.items()
                    )
                )
            previous_weights = {}
            previous_returns = {}
            previous_nav_growth = None
        else:
            if previous_nav_growth is None or previous_nav_growth <= 0:
                raise FilingChangeStageAError(
                    "calendar portfolio gross NAV is nonpositive before rebalance"
                )
            tickers = set(previous_weights) | set(group["ticker"].astype(str))
            target = dict(zip(group["ticker"].astype(str), group["target_weight"].astype(float)))
            start_turnover = float(
                sum(
                    abs(
                        target.get(ticker, 0.0)
                        - previous_weights.get(ticker, 0.0)
                        * (1.0 + previous_returns.get(ticker, 0.0))
                        / previous_nav_growth
                    )
                    for ticker in tickers
                )
            )
        long = group.loc[group["target_weight"].gt(0)]
        short = group.loc[group["target_weight"].lt(0)]
        long_return = float((long["target_weight"] * long["gross_return"]).sum())
        short_return = float((-short["target_weight"] * short["gross_return"]).sum())
        gross = long_return - short_return
        nav_growth = 1.0 + gross
        if not np.isfinite(nav_growth) or nav_growth <= 0:
            raise FilingChangeStageAError(
                "calendar portfolio gross NAV is nonpositive after monthly return"
            )
        monthly_rows.append(
            {
                "portfolio_scope": scope,
                "month": month,
                "month_ordinal": ordinal,
                "entry_date": group["entry_date"].iloc[0],
                "entry_index": int(group["entry_index"].iloc[0]),
                "exit_date": group["exit_date"].iloc[0],
                "q5_gross_return": long_return,
                "q1_gross_return": short_return,
                "q5_minus_q1_gross_return": gross,
                "start_turnover": start_turnover,
                "terminal_turnover": 0.0,
                "active_cohorts": int(group["cohort"].nunique()),
                "stocks": int(group["ticker"].nunique()),
                "gross_target_exposure": float(group["target_weight"].abs().sum()),
                "maximum_absolute_target_weight": float(group["target_weight"].abs().max()),
            }
        )
        previous_weights = dict(
            zip(group["ticker"].astype(str), group["target_weight"].astype(float))
        )
        previous_returns = dict(
            zip(group["ticker"].astype(str), group["gross_return"].astype(float))
        )
        previous_nav_growth = nav_growth
        previous_ordinal = ordinal
    if monthly_rows and previous_weights:
        monthly_rows[-1]["terminal_turnover"] = float(
            sum(
                abs(weight * (1.0 + previous_returns[ticker]))
                for ticker, weight in previous_weights.items()
            )
        )
    monthly = pd.DataFrame(monthly_rows)
    monthly["total_turnover"] = monthly["start_turnover"] + monthly["terminal_turnover"]
    monthly["q5_minus_q1_net30_return"] = (
        monthly["q5_minus_q1_gross_return"] - one_way_costs[0] * monthly["total_turnover"]
    )
    monthly["q5_minus_q1_net50_return"] = (
        monthly["q5_minus_q1_gross_return"] - one_way_costs[1] * monthly["total_turnover"]
    )
    if (
        monthly[["q5_minus_q1_net30_return", "q5_minus_q1_net50_return"]]
        .le(-1.0)
        .any()
        .any()
    ):
        raise FilingChangeStageAError(
            "calendar portfolio net-cost NAV is nonpositive after monthly return"
        )
    return components, monthly


def calendar_portfolio_returns(
    schedule: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    holding_months: int = 3,
    minimum_prior_close: float = 5.0,
    adv_window: int = 63,
    minimum_prior_stock_sessions: int = 252,
    minimum_prior_adv: float = 20_000_000.0,
    primary_one_way_cost: float = PRIMARY_ONE_WAY_COST,
    stress_one_way_cost: float = STRESS_ONE_WAY_COST,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Paper-inspired calendar portfolios on one frozen, common safe clock.

    Each monthly cohort is equal weighted within Q1 and Q5, three overlapping
    cohorts receive equal capital, and transaction costs are charged on actual
    target-weight turnover.  The output remains a retrospective proxy because
    the underlying security universe is not point in time.
    """

    if holding_months < 1:
        raise FilingChangeStageAError("holding_months must be positive")
    if (
        adv_window < 1
        or minimum_prior_stock_sessions < adv_window
        or min(primary_one_way_cost, stress_one_way_cost) < 0
    ):
        raise FilingChangeStageAError("calendar portfolio parameters are invalid")
    panel = _prepare_prices(prices)
    raw, exclusions = _eligible_calendar_holding_returns(
        schedule,
        panel,
        holding_months=holding_months,
        minimum_prior_close=minimum_prior_close,
        adv_window=adv_window,
        minimum_prior_stock_sessions=minimum_prior_stock_sessions,
        minimum_prior_adv=minimum_prior_adv,
    )
    if raw.empty:
        return raw, pd.DataFrame(), exclusions
    side_counts = raw.groupby(["month", "cohort"], sort=False)["direction"].nunique()
    invalid_cohorts = set(side_counts.loc[lambda values: values != 2].index)
    if invalid_cohorts:
        invalid_mask = raw.apply(
            lambda row: (row["month"], row["cohort"]) in invalid_cohorts, axis=1
        )
        invalid_rows = raw.loc[invalid_mask, ["current_accession", "ticker"]].drop_duplicates()
        invalid_rows["reason"] = "single_sided_calendar_cohort"
        exclusions = pd.concat([exclusions, invalid_rows], ignore_index=True)
        raw = raw.loc[~invalid_mask].copy()
    if raw.empty:
        return raw, pd.DataFrame(), exclusions
    holding_parts: list[pd.DataFrame] = []
    monthly_parts: list[pd.DataFrame] = []
    for scope in ("ALL", "10-K", "10-Q"):
        holdings, monthly = _portfolio_scope_returns(
            raw,
            scope=scope,
            one_way_costs=(primary_one_way_cost, stress_one_way_cost),
        )
        if not holdings.empty:
            holding_parts.append(holdings)
        if not monthly.empty:
            monthly_parts.append(monthly)
    return (
        pd.concat(holding_parts, ignore_index=True) if holding_parts else pd.DataFrame(),
        pd.concat(monthly_parts, ignore_index=True) if monthly_parts else pd.DataFrame(),
        exclusions,
    )


def summarize_monthly_portfolio(
    schedule: pd.DataFrame,
    holdings: pd.DataFrame,
    monthly: pd.DataFrame,
    *,
    expected_month_start: str,
    expected_month_end: str,
    trial_floor: int = 130,
    hac_lags: int = 3,
    primary_one_way_cost: float = PRIMARY_ONE_WAY_COST,
    stress_one_way_cost: float = STRESS_ONE_WAY_COST,
    inferential_test_names: Iterable[str] = ("net30",),
    formal_mean_gate_names: Iterable[str] = (),
    inference_allowed: bool = True,
) -> dict[str, Any]:
    all_months = monthly.loc[monthly.get("portfolio_scope", pd.Series(dtype=str)).eq("ALL")].copy()
    if all_months.empty:
        return {
            "signal_events": 0,
            "unique_cik": 0,
            "portfolio_months": 0,
            "evaluation_months": 0,
            "expected_month_coverage": 0.0,
            "missing_expected_months": [],
            "tests": {},
            "by_form_direction": [],
            "by_form": [],
            "by_year": [],
            "maximum_absolute_target_weight": None,
            "minimum_leave_one_cik_out_net30_mean": None,
            "leave_one_cik_out_complete": False,
        }
    expected = pd.period_range(expected_month_start, expected_month_end, freq="M")
    if expected.empty:
        raise FilingChangeStageAError("expected calendar evaluation range is empty")
    observed_periods = pd.PeriodIndex(all_months["month"], freq="M")
    if observed_periods.duplicated().any():
        raise FilingChangeStageAError("formal calendar portfolio has duplicate months")
    if not observed_periods.isin(expected).all():
        raise FilingChangeStageAError("formal calendar portfolio falls outside frozen cohort months")
    all_months = (
        all_months.assign(month_period=observed_periods)
        .set_index("month_period")
        .sort_index()
    )
    missing_expected = expected[~expected.isin(all_months.index)]
    expected_month_coverage = len(all_months) / len(expected)
    inferential_names = set(inferential_test_names)
    formal_mean_names = set(formal_mean_gate_names)
    supported_monthly = {"gross", "net30", "net50"}
    if (
        not inferential_names.issubset(supported_monthly)
        or not formal_mean_names.issubset(supported_monthly)
        or inferential_names.intersection(formal_mean_names)
    ):
        raise FilingChangeStageAError("unknown monthly inferential test name")
    tests: dict[str, Any] = {}
    for name, column in (
        ("gross", "q5_minus_q1_gross_return"),
        ("net30", "q5_minus_q1_net30_return"),
        ("net50", "q5_minus_q1_net50_return"),
    ):
        tests[name] = _mean_test_record(
            all_months[column],
            session_indices=all_months["month_ordinal"],
            lags=hac_lags,
            lag_unit="calendar_month_distance",
            trial_floor=trial_floor,
            inferential=name in inferential_names,
            formal_mean_gate=name in formal_mean_names,
            inference_allowed=inference_allowed,
        )
    required_schedule = {"current_accession", "cik", "form", "direction"}
    missing = required_schedule.difference(schedule.columns)
    if missing:
        raise FilingChangeStageAError(
            f"calendar schedule missing columns: {sorted(missing)}"
        )
    if "portfolio_scope" not in holdings.columns:
        raise FilingChangeStageAError("calendar holdings lack portfolio_scope")
    all_components = holdings.loc[holdings["portfolio_scope"].eq("ALL")].copy()
    if all_components.empty:
        raise FilingChangeStageAError("monthly portfolio exists without ALL holdings")
    signals = all_components.drop_duplicates("current_accession").copy()
    buckets = (
        signals.groupby(["form", "direction"], sort=True)
        .agg(events=("current_accession", "size"), unique_cik=("cik", "nunique"))
        .reset_index()
        .to_dict("records")
    )
    form_rows: list[dict[str, Any]] = []
    for form in ("10-K", "10-Q"):
        subset = monthly.loc[monthly["portfolio_scope"].eq(form)].copy()
        subset_periods = pd.PeriodIndex(subset["month"], freq="M") if not subset.empty else pd.PeriodIndex([], freq="M")
        if subset_periods.duplicated().any() or not subset_periods.isin(expected).all():
            raise FilingChangeStageAError("form portfolio months violate frozen evaluation range")
        subset = subset.assign(month_period=subset_periods).set_index("month_period")
        active = int(len(subset))
        form_rows.append(
            {
                "form": form,
                "portfolio_months": active,
                "evaluation_months": int(len(expected)),
                "net30_mean": (
                    float(subset["q5_minus_q1_net30_return"].mean())
                    if active
                    else None
                ),
            }
        )
    year_frame = all_months.copy()
    year_frame["entry_year"] = [period.year for period in year_frame.index]
    by_year = (
        year_frame.groupby("entry_year", sort=True)["q5_minus_q1_net30_return"]
        .mean()
        .rename("net30_mean")
        .reset_index()
        .to_dict("records")
    )
    cik_shares: list[float] = []
    for _, group in all_components.groupby("month", sort=True):
        security_weights = (
            group.groupby(["ticker", "cik"], sort=False)["target_weight_component"]
            .sum()
            .rename("weight")
            .reset_index()
        )
        gross = float(security_weights["weight"].abs().sum())
        if gross > 0:
            security_weights["absolute_weight"] = security_weights["weight"].abs()
            cik_shares.append(
                float(
                    security_weights.groupby("cik")["absolute_weight"].sum().max()
                    / gross
                )
            )
    absolute_months = all_months["q5_minus_q1_net30_return"].abs()
    maximum_month_contribution = (
        float(absolute_months.max() / absolute_months.sum())
        if float(absolute_months.sum()) > 0
        else None
    )

    # Exact issuer concentration falsification: remove each CIK, rebuild all
    # within-side/cohort weights and turnover, then rerun the monthly net-30
    # portfolio.  A deletion that destroys any frozen evaluation month is a
    # concentration failure, not a zero-return month.
    loo_means: list[float] = []
    loo_incomplete_ciks: list[int] = []
    expected_set = set(expected)
    raw_columns_to_drop = [
        column
        for column in ("portfolio_scope", "target_weight_component")
        if column in all_components.columns
    ]
    raw_components = all_components.drop(columns=raw_columns_to_drop)
    for cik in sorted(raw_components["cik"].dropna().astype(int).unique()):
        _, loo_monthly = _portfolio_scope_returns(
            raw_components.loc[raw_components["cik"].astype(int).ne(cik)],
            scope="ALL",
            one_way_costs=(primary_one_way_cost, stress_one_way_cost),
        )
        if loo_monthly.empty:
            loo_incomplete_ciks.append(cik)
            continue
        loo_periods = set(pd.PeriodIndex(loo_monthly["month"], freq="M"))
        if loo_periods != expected_set:
            loo_incomplete_ciks.append(cik)
            continue
        loo_means.append(float(loo_monthly["q5_minus_q1_net30_return"].mean()))
    loo_complete = (
        not len(missing_expected)
        and not loo_incomplete_ciks
        and len(loo_means) == int(raw_components["cik"].nunique())
        and bool(loo_means)
    )
    return {
        "signal_events": int(signals["current_accession"].nunique()),
        "unique_cik": int(signals["cik"].nunique()),
        "portfolio_months": int(len(all_months)),
        "evaluation_months": int(len(expected)),
        "expected_months_present": int(len(all_months)),
        "expected_month_coverage": float(expected_month_coverage),
        "missing_expected_months": [str(period) for period in missing_expected],
        "expected_month_start": str(expected[0]),
        "expected_month_end": str(expected[-1]),
        "tests": tests,
        "by_form_direction": buckets,
        "by_form": form_rows,
        "by_year": by_year,
        "maximum_absolute_target_weight": float(
            all_months["maximum_absolute_target_weight"].max()
        ),
        "maximum_single_cik_gross_weight_share": max(cik_shares) if cik_shares else None,
        "maximum_month_absolute_return_contribution_share": maximum_month_contribution,
        "minimum_leave_one_cik_out_net30_mean": (
            min(loo_means) if loo_complete else None
        ),
        "leave_one_cik_out_complete": loo_complete,
        "leave_one_cik_out_issuers_tested": int(len(loo_means)),
        "leave_one_cik_out_incomplete_issuers": loo_incomplete_ciks,
    }


def monthly_portfolio_gate_table(
    summary: Mapping[str, Any],
    *,
    signed_events_minimum: int,
    unique_cik_minimum: int,
    portfolio_months_minimum: int,
    each_form_direction_minimum: int,
    each_form_portfolio_months_minimum: int = 1,
    maximum_absolute_target_weight: float = 0.02,
    maximum_single_cik_gross_weight_share: float = 0.02,
    maximum_month_absolute_return_contribution_share: float = 0.05,
    minimum_leave_one_cik_out_net30_mean: float = 0.0,
) -> pd.DataFrame:
    tests = summary.get("tests", {})
    if not tests:
        return pd.DataFrame(
            [{"gate": "nonempty_monthly_portfolio", "passed": False, "required": True, "detail": 0}]
        )
    buckets = summary["by_form_direction"]
    bucket_keys = {(str(row["form"]), int(row["direction"])) for row in buckets}
    minimum_bucket = min((int(row["events"]) for row in buckets), default=0)
    form_rows = summary["by_form"]
    both_forms_nonnegative = {
        str(row["form"])
        for row in form_rows
        if row.get("net30_mean") is not None
        and int(row.get("portfolio_months", 0)) >= each_form_portfolio_months_minimum
        and float(row["net30_mean"]) >= 0
    } == {"10-K", "10-Q"}
    years_positive = bool(summary["by_year"]) and all(
        float(row["net30_mean"]) > 0 for row in summary["by_year"]
    )
    gross = tests["gross"]
    net30 = tests["net30"]
    net50 = tests["net50"]
    critical = net30["one_sided_normal_critical_value"]
    rows = [
        ("minimum_signed_events", summary["signal_events"] >= signed_events_minimum, summary["signal_events"]),
        ("minimum_unique_cik", summary["unique_cik"] >= unique_cik_minimum, summary["unique_cik"]),
        ("minimum_portfolio_months", summary["portfolio_months"] >= portfolio_months_minimum, summary["portfolio_months"]),
        (
            "complete_expected_calendar_months",
            summary.get("expected_month_coverage") == 1.0,
            {
                "coverage": summary.get("expected_month_coverage"),
                "missing": summary.get("missing_expected_months", []),
            },
        ),
        (
            "minimum_each_form_direction",
            bucket_keys == {("10-K", -1), ("10-K", 1), ("10-Q", -1), ("10-Q", 1)}
            and minimum_bucket >= each_form_direction_minimum,
            minimum_bucket,
        ),
        ("gross_mean_positive", gross.get("mean") is not None and gross["mean"] > 0, gross.get("mean")),
        ("net30_mean_positive", net30.get("mean") is not None and net30["mean"] > 0, net30.get("mean")),
        ("net50_mean_positive", net50.get("mean") is not None and net50["mean"] > 0, net50.get("mean")),
        ("net30_known_floor_t", net30.get("t") is not None and net30["t"] >= critical, {"t": net30.get("t"), "critical": critical}),
        ("net30_known_floor_p", net30.get("known_floor_adjusted_screening_p") is not None and net30["known_floor_adjusted_screening_p"] <= 0.05, net30.get("known_floor_adjusted_screening_p")),
        ("each_calendar_year_positive", years_positive, summary["by_year"]),
        ("both_form_portfolios_nonnegative", both_forms_nonnegative, form_rows),
        ("maximum_absolute_target_weight", summary["maximum_absolute_target_weight"] <= maximum_absolute_target_weight, summary["maximum_absolute_target_weight"]),
        (
            "maximum_single_cik_calendar_weight_share",
            summary.get("maximum_single_cik_gross_weight_share") is not None
            and summary["maximum_single_cik_gross_weight_share"]
            <= maximum_single_cik_gross_weight_share,
            summary.get("maximum_single_cik_gross_weight_share"),
        ),
        (
            "maximum_calendar_month_contribution",
            summary.get("maximum_month_absolute_return_contribution_share") is not None
            and summary["maximum_month_absolute_return_contribution_share"]
            <= maximum_month_absolute_return_contribution_share,
            summary.get("maximum_month_absolute_return_contribution_share"),
        ),
        (
            "leave_one_cik_out_calendar_positive",
            bool(summary.get("leave_one_cik_out_complete"))
            and summary.get("minimum_leave_one_cik_out_net30_mean") is not None
            and summary["minimum_leave_one_cik_out_net30_mean"]
            > minimum_leave_one_cik_out_net30_mean,
            {
                "complete": summary.get("leave_one_cik_out_complete"),
                "minimum_mean": summary.get("minimum_leave_one_cik_out_net30_mean"),
                "incomplete_issuers": summary.get(
                    "leave_one_cik_out_incomplete_issuers", []
                ),
            },
        ),
    ]
    return pd.DataFrame(
        [
            {"gate": name, "passed": bool(passed), "required": True, "detail": detail}
            for name, passed, detail in rows
        ]
    )


def paper_style_monthly_returns(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    holding_months: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compatibility wrapper around the formal frozen-clock portfolio."""

    panel = _prepare_prices(prices)
    schedule, _ = build_calendar_preoutcome_schedule(
        signals,
        panel.loc[panel["ticker"].eq(BENCHMARK)],
        holding_months=holding_months,
    )
    holdings, monthly, _ = calendar_portfolio_returns(
        schedule,
        panel,
        holding_months=holding_months,
    )
    return holdings, monthly
