"""Reaction signal, execution, and inference for frozen SEC Item 2.02 clocks."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

from alpha_factor_engine.contracts import validate_unique
from alpha_factor_engine.research.sec_filing_change_stage_a import hac_mean_test, known_trial_floor_screen


class Sec8KOutcomeError(ValueError):
    """Raised when event outcomes violate the frozen protocol."""


def assign_rolling_reaction_signals(
    events: pd.DataFrame,
    *,
    window_days: int = 730,
    minimum_history_events: int = 500,
    tail_probability: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign online tails from reactions on strictly earlier event dates."""

    required = {"accession", "ticker", "reaction_date", "reaction_abnormal_return"}
    missing = required.difference(events.columns)
    if missing:
        raise Sec8KOutcomeError(f"reaction events missing: {sorted(missing)}")
    if window_days < 1 or minimum_history_events < 1 or not 0 < tail_probability < 0.5:
        raise Sec8KOutcomeError("rolling threshold parameters are invalid")
    frame = events.copy()
    frame["reaction_date"] = pd.to_datetime(frame["reaction_date"], errors="raise").dt.normalize()
    frame = frame.sort_values(["reaction_date", "ticker", "accession"], kind="mergesort")
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for date, group in frame.groupby("reaction_date", sort=True):
        lower_date = pd.Timestamp(date) - pd.Timedelta(days=window_days)
        history = frame.loc[frame["reaction_date"].between(lower_date, pd.Timestamp(date), inclusive="left"), "reaction_abnormal_return"].to_numpy(dtype=float)
        if len(history) < minimum_history_events:
            for row in group.itertuples(index=False):
                exclusions.append({"accession": row.accession, "ticker": row.ticker, "reason": "rolling_reaction_history_below_minimum", "history_events": int(len(history))})
            continue
        lower = float(np.quantile(history, tail_probability, method="linear"))
        upper = float(np.quantile(history, 1 - tail_probability, method="linear"))
        if not lower < upper:
            raise Sec8KOutcomeError("rolling reaction thresholds are not ordered")
        for row in group.itertuples(index=False):
            value = float(row.reaction_abnormal_return)
            direction = 1 if value >= upper else (-1 if value <= lower else 0)
            if direction == 0:
                exclusions.append({"accession": row.accession, "ticker": row.ticker, "reason": "reaction_not_in_online_tail", "history_events": int(len(history))})
                continue
            record = row._asdict()
            record.update({"reaction_history_start": lower_date, "reaction_history_end_exclusive": pd.Timestamp(date), "reaction_history_events": int(len(history)), "lower_reaction_threshold": lower, "upper_reaction_threshold": upper, "direction": direction})
            rows.append(record)
    selected = pd.DataFrame(
        rows,
        columns=[
            *frame.columns,
            "reaction_history_start",
            "reaction_history_end_exclusive",
            "reaction_history_events",
            "lower_reaction_threshold",
            "upper_reaction_threshold",
            "direction",
        ],
    )
    if not selected.empty:
        validate_unique(selected, ["accession"], "Item 2.02 rolling reaction signals")
    return selected, pd.DataFrame(exclusions)


def _prepare_prices(prices: pd.DataFrame) -> pd.DataFrame:
    required = {"ticker", "date", "timestamp", "close", "adj_close", "adj_open", "dollar_volume", "available_at"}
    missing = required.difference(prices.columns)
    if missing:
        raise Sec8KOutcomeError(f"prices missing: {sorted(missing)}")
    frame = prices.loc[:, sorted(required)].copy()
    frame["ticker"] = frame["ticker"].astype("string").str.strip().str.upper()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True, errors="raise")
    for column in ("close", "adj_close", "adj_open", "dollar_volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    validate_unique(frame, ["ticker", "date"], "Item 2.02 prices")
    return frame.sort_values(["ticker", "date"], kind="mergesort")


def attach_reaction_and_forward_returns(
    schedule: pd.DataFrame,
    prices: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    minimum_prior_close: float = 5.0,
    minimum_prior_stock_sessions: int = 252,
    adv_window_sessions: int = 63,
    minimum_prior_adv: float = 20_000_000.0,
    threshold_window_days: int = 730,
    threshold_minimum_events: int = 500,
    tail_probability: float = 0.20,
    portfolio_aum_usd: float = 10_000_000.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Attach allowed prices, select online reaction tails, and suppress overlap."""

    if schedule.empty or schedule["phase"].eq("sealed_2025").any():
        raise Sec8KOutcomeError("outcome schedule is empty or contains sealed events")
    cal = calendar.copy()
    cal["session_date"] = pd.to_datetime(cal["session_date"], errors="raise").dt.normalize()
    cal = cal.sort_values("session_date", kind="mergesort").reset_index(drop=True)
    validate_unique(cal, ["session_date"], "Item 2.02 outcome calendar")
    session_index = dict(zip(cal["session_date"], np.arange(len(cal), dtype=int)))
    panel = _prepare_prices(prices)
    by_ticker = {ticker: group.set_index("date") for ticker, group in panel.groupby("ticker", sort=False)}
    spy = by_ticker.get("SPY")
    if spy is None:
        raise Sec8KOutcomeError("SPY is absent")
    preliminary: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for event in schedule.sort_values(["reaction_date", "ticker", "accession"], kind="mergesort").itertuples(index=False):
        stock = by_ticker.get(str(event.ticker))
        required_dates = [pd.Timestamp(event.prior_reaction_date), pd.Timestamp(event.reaction_date), pd.Timestamp(event.entry_date), pd.Timestamp(event.exit_date)]
        if stock is None or any(date not in stock.index or date not in spy.index for date in required_dates):
            exclusions.append({"accession": event.accession, "ticker": event.ticker, "reason": "exact_price_path_incomplete"})
            continue
        entry_index = session_index[pd.Timestamp(event.entry_date)]
        prior_sessions = cal.iloc[entry_index - minimum_prior_stock_sessions:entry_index]["session_date"] if entry_index >= minimum_prior_stock_sessions else pd.Series(dtype="datetime64[ns]")
        if len(prior_sessions) != minimum_prior_stock_sessions or not prior_sessions.isin(stock.index).all():
            exclusions.append({"accession": event.accession, "ticker": event.ticker, "reason": "prior_stock_history_incomplete"})
            continue
        history = stock.loc[prior_sessions]
        reaction_row = stock.loc[pd.Timestamp(event.reaction_date)]
        prior_row = stock.loc[pd.Timestamp(event.prior_reaction_date)]
        entry_row = stock.loc[pd.Timestamp(event.entry_date)]
        exit_row = stock.loc[pd.Timestamp(event.exit_date)]
        spy_reaction = spy.loc[pd.Timestamp(event.reaction_date)]
        spy_prior = spy.loc[pd.Timestamp(event.prior_reaction_date)]
        spy_entry = spy.loc[pd.Timestamp(event.entry_date)]
        spy_exit = spy.loc[pd.Timestamp(event.exit_date)]
        values = np.array([prior_row.adj_close, reaction_row.adj_close, entry_row.adj_open, exit_row.adj_open, spy_prior.adj_close, spy_reaction.adj_close, spy_entry.adj_open, spy_exit.adj_open], dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            exclusions.append({"accession": event.accession, "ticker": event.ticker, "reason": "invalid_price_value"})
            continue
        adv = float(history["dollar_volume"].tail(adv_window_sessions).median())
        exit_index = session_index[pd.Timestamp(event.exit_date)]
        exit_sessions = cal.iloc[exit_index - adv_window_sessions:exit_index]["session_date"]
        exit_adv = (
            float(stock.loc[exit_sessions, "dollar_volume"].median())
            if len(exit_sessions) == adv_window_sessions and exit_sessions.isin(stock.index).all()
            else math.nan
        )
        if float(reaction_row.close) < minimum_prior_close:
            exclusions.append({"accession": event.accession, "ticker": event.ticker, "reason": "reaction_close_below_floor"})
            continue
        if not np.isfinite(adv) or not np.isfinite(exit_adv) or adv < minimum_prior_adv:
            exclusions.append({"accession": event.accession, "ticker": event.ticker, "reason": "prior_adv_below_floor"})
            continue
        if max(pd.Timestamp(reaction_row.available_at), pd.Timestamp(spy_reaction.available_at)) >= pd.Timestamp(event.entry_at):
            raise Sec8KOutcomeError("reaction close was not available before entry")
        stock_reaction = float(reaction_row.adj_close / prior_row.adj_close - 1)
        spy_reaction_return = float(spy_reaction.adj_close / spy_prior.adj_close - 1)
        stock_forward = float(exit_row.adj_open / entry_row.adj_open - 1)
        spy_forward = float(spy_exit.adj_open / spy_entry.adj_open - 1)
        record = event._asdict()
        record.update({"entry_index": int(entry_index), "exit_index": int(exit_index), "reaction_abnormal_return": stock_reaction - spy_reaction_return, "stock_reaction_return": stock_reaction, "spy_reaction_return": spy_reaction_return, "stock_forward_return": stock_forward, "spy_forward_return": spy_forward, "entry_trailing_adv_usd": adv, "exit_trailing_adv_usd": exit_adv})
        preliminary.append(record)
    preliminary_frame = pd.DataFrame(preliminary)
    selected, threshold_exclusions = assign_rolling_reaction_signals(preliminary_frame, window_days=threshold_window_days, minimum_history_events=threshold_minimum_events, tail_probability=tail_probability)
    exclusions.extend(threshold_exclusions.to_dict("records"))
    kept: list[dict[str, Any]] = []
    for ticker, group in selected.groupby("ticker", sort=True):
        last_exit = -1
        for event in group.sort_values(["entry_index", "accession"], kind="mergesort").itertuples(index=False):
            if int(event.entry_index) < last_exit:
                exclusions.append({"accession": event.accession, "ticker": ticker, "reason": "same_ticker_overlap_until_exit"})
                continue
            record = event._asdict()
            direction = int(event.direction)
            pair_gross = 0.5 * direction * (float(event.stock_forward_return) - float(event.spy_forward_return))
            liquidation = 0.5 * (1 + float(event.stock_forward_return)) + 0.5 * (1 + float(event.spy_forward_return))
            traded = 1 + liquidation
            record.update({"pair_gross_return_20": pair_gross, "pair_traded_notional_per_gross": traded, "pair_net_30bps_return_20": pair_gross - traded * .003, "pair_net_50bps_return_20": pair_gross - traded * .005, "maximum_stock_trade_to_adv_at_10m": max(0.5 * portfolio_aum_usd / float(event.entry_trailing_adv_usd), 0.5 * portfolio_aum_usd * (1 + float(event.stock_forward_return)) / float(event.exit_trailing_adv_usd))})
            kept.append(record)
            last_exit = int(event.exit_index)
    result = pd.DataFrame(kept)
    if not result.empty:
        validate_unique(result, ["accession"], "Item 2.02 reaction outcomes")
    counts = {
        "scheduled_events": int(len(schedule)),
        "price_eligible_events": int(len(preliminary_frame)),
        "tail_selected_before_overlap": int(len(selected)),
        "retained_events": int(len(result)),
        "by_phase": {
            phase: {
                "scheduled_events": int(schedule["phase"].eq(phase).sum()),
                "price_eligible_events": int(preliminary_frame["phase"].eq(phase).sum()),
                "tail_selected_before_overlap": int(selected["phase"].eq(phase).sum()),
                "retained_events": int(result["phase"].eq(phase).sum()),
            }
            for phase in sorted(schedule["phase"].unique())
        },
    }
    return result, pd.DataFrame(exclusions), counts


def aggregate_entry_dates(events: pd.DataFrame) -> pd.DataFrame:
    metrics = ["pair_gross_return_20", "pair_net_30bps_return_20", "pair_net_50bps_return_20"]
    grouped = events.groupby("entry_date", sort=True)
    out = grouped[metrics].mean().reset_index()
    out["events"] = grouped.size().to_numpy()
    out["entry_index"] = grouped["entry_index"].first().to_numpy()
    return out


def summarize(events: pd.DataFrame, *, trial_floor: int, scheduled_events: int, price_eligible_events: int) -> dict[str, Any]:
    dates = aggregate_entry_dates(events)
    indices = dates["entry_index"].to_numpy(dtype=int)
    gross = hac_mean_test(dates["pair_gross_return_20"], session_indices=indices, lags=3, lag_unit="XNYS_session_distance")
    primary = known_trial_floor_screen(hac_mean_test(dates["pair_net_30bps_return_20"], session_indices=indices, lags=3, lag_unit="XNYS_session_distance"), trial_floor=trial_floor)
    stress = hac_mean_test(dates["pair_net_50bps_return_20"], session_indices=indices, lags=3, lag_unit="XNYS_session_distance")
    counts = events.groupby("entry_date").size()
    contributions = (events["pair_net_30bps_return_20"] / events["entry_date"].map(counts)).abs().groupby(events["ticker"]).sum()
    maximum_share = float(contributions.max() / contributions.sum())
    loo = []
    for ticker in events["ticker"].unique():
        subset = events.loc[events["ticker"].ne(ticker)]
        loo.append(float(aggregate_entry_dates(subset)["pair_net_30bps_return_20"].mean()))
    return {"events": int(len(events)), "tickers": int(events["ticker"].nunique()), "entry_dates": int(events["entry_date"].nunique()), "scheduled_events": int(scheduled_events), "price_eligible_events": int(price_eligible_events), "price_path_coverage": float(price_eligible_events / scheduled_events), "directions": {str(k): int(v) for k, v in events["direction"].value_counts().sort_index().items()}, "gross": gross, "primary_net_30bps": primary, "stress_net_50bps": stress, "maximum_ticker_gross_contribution_share": maximum_share, "minimum_leave_one_ticker_out_mean": min(loo), "maximum_trade_to_adv_at_10m": float(events["maximum_stock_trade_to_adv_at_10m"].max())}


def gate(summary: Mapping[str, Any], gates: Mapping[str, Any]) -> pd.DataFrame:
    primary, stress = summary["primary_net_30bps"], summary["stress_net_50bps"]
    rows = [
        ("minimum_events", summary["events"] >= gates["minimum_events"], summary["events"], gates["minimum_events"]),
        ("minimum_tickers", summary["tickers"] >= gates["minimum_tickers"], summary["tickers"], gates["minimum_tickers"]),
        ("minimum_entry_dates", summary["entry_dates"] >= gates["minimum_entry_dates"], summary["entry_dates"], gates["minimum_entry_dates"]),
        ("price_path_coverage", summary["price_path_coverage"] >= gates["minimum_price_path_coverage"], summary["price_path_coverage"], gates["minimum_price_path_coverage"]),
        ("primary_net_mean", primary["mean"] > gates["primary_net_mean_minimum"], primary["mean"], gates["primary_net_mean_minimum"]),
        ("primary_hac_t", primary["t"] >= gates["primary_newey_west_t_minimum"], primary["t"], gates["primary_newey_west_t_minimum"]),
        ("global_adjusted_p", primary["known_floor_adjusted_screening_p"] <= gates["global_adjusted_p_maximum"], primary["known_floor_adjusted_screening_p"], gates["global_adjusted_p_maximum"]),
        ("stress_net_mean", stress["mean"] > gates["stress_net_mean_minimum"], stress["mean"], gates["stress_net_mean_minimum"]),
        ("ticker_contribution", summary["maximum_ticker_gross_contribution_share"] <= gates["maximum_ticker_gross_contribution_share"], summary["maximum_ticker_gross_contribution_share"], gates["maximum_ticker_gross_contribution_share"]),
        ("leave_one_ticker_out", summary["minimum_leave_one_ticker_out_mean"] > gates["minimum_leave_one_ticker_out_mean"], summary["minimum_leave_one_ticker_out_mean"], gates["minimum_leave_one_ticker_out_mean"]),
        ("capacity_10m", summary["maximum_trade_to_adv_at_10m"] <= gates["maximum_trade_to_trailing_adv_at_10m"], summary["maximum_trade_to_adv_at_10m"], gates["maximum_trade_to_trailing_adv_at_10m"]),
    ]
    return pd.DataFrame(rows, columns=["gate", "passed", "observed", "threshold"])


__all__ = ["Sec8KOutcomeError", "assign_rolling_reaction_signals", "attach_reaction_and_forward_returns", "aggregate_entry_dates", "summarize", "gate"]
