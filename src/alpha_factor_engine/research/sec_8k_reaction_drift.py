"""Pre-price event clocks for SEC Item 2.02 reaction-drift research."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from alpha_factor_engine.contracts import validate_unique


class Sec8KReactionError(ValueError):
    """Raised when the frozen event-clock protocol is violated."""


def build_event_clock_plan(
    inventory: pd.DataFrame,
    universe: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    cohorts: Mapping[str, Mapping[str, Any]],
    reaction_delay_minutes: int = 30,
    holding_sessions: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required_events = {"issuer_cik", "accession", "filing_date", "acceptance_at", "items"}
    required_universe = {"cik", "ticker"}
    required_calendar = {"session_date", "market_open_utc", "market_close_utc"}
    if required_events.difference(inventory.columns):
        raise Sec8KReactionError("event inventory lacks required columns")
    if required_universe.difference(universe.columns):
        raise Sec8KReactionError("universe lacks required columns")
    if required_calendar.difference(calendar.columns):
        raise Sec8KReactionError("calendar lacks required columns")
    events = inventory.copy()
    events["issuer_cik_int"] = pd.to_numeric(events["issuer_cik"], errors="raise").astype(int)
    events["filing_date"] = pd.to_datetime(events["filing_date"], errors="raise").dt.normalize()
    events["acceptance_at"] = pd.to_datetime(events["acceptance_at"], utc=True, errors="raise")
    validate_unique(events, ["accession"], "Item 2.02 event inventory")
    mapping = universe[["cik", "ticker"]].copy()
    mapping["cik"] = pd.to_numeric(mapping["cik"], errors="raise").astype(int)
    mapping["ticker"] = mapping["ticker"].astype("string").str.strip().str.upper()
    validate_unique(mapping, ["cik"], "Item 2.02 CIK mapping")
    validate_unique(mapping, ["ticker"], "Item 2.02 ticker mapping")
    events = events.merge(mapping, left_on="issuer_cik_int", right_on="cik", how="left", validate="many_to_one")
    misses = events.loc[events["ticker"].isna(), ["issuer_cik", "accession"]].copy()
    events = events.loc[events["ticker"].notna()].copy()
    filing_public = events["filing_date"].dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    events["public_at"] = pd.concat([events["acceptance_at"], filing_public.rename("filing_public")], axis=1).max(axis=1)
    events["public_year_et"] = events["public_at"].dt.tz_convert("America/New_York").dt.year
    year_phase: dict[int, str] = {}
    for phase, spec in cohorts.items():
        for year in spec["public_years_et"]:
            if int(year) in year_phase:
                raise Sec8KReactionError("cohort years overlap")
            year_phase[int(year)] = str(phase)
    events["phase"] = events["public_year_et"].map(year_phase)
    events = events.loc[events["phase"].notna()].copy()
    events = events.sort_values(["ticker", "filing_date", "public_at", "accession"], kind="mergesort")
    duplicate = events.duplicated(["ticker", "filing_date"], keep="first")
    exclusions = events.loc[duplicate, ["accession", "ticker"]].copy()
    exclusions["reason"] = "same_ticker_filing_date_duplicate"
    events = events.loc[~duplicate].copy()
    cal = calendar.copy()
    cal["session_date"] = pd.to_datetime(cal["session_date"], errors="raise").dt.normalize()
    cal["market_open_utc"] = pd.to_datetime(cal["market_open_utc"], utc=True, errors="raise")
    cal["market_close_utc"] = pd.to_datetime(cal["market_close_utc"], utc=True, errors="raise")
    validate_unique(cal, ["session_date"], "Item 2.02 exchange calendar")
    cal = cal.sort_values("market_open_utc", kind="mergesort").reset_index(drop=True)
    closes = cal["market_close_utc"].astype("int64").to_numpy()
    rows: list[dict[str, Any]] = []
    for row in events.itertuples(index=False):
        eligible_close = pd.Timestamp(row.public_at) + pd.Timedelta(minutes=reaction_delay_minutes)
        reaction_index = int(np.searchsorted(closes, eligible_close.value, side="left"))
        prior_index = reaction_index - 1
        entry_index = reaction_index + 1
        exit_index = entry_index + holding_sessions
        if prior_index < 0 or exit_index >= len(cal):
            exclusions = pd.concat([exclusions, pd.DataFrame([{"accession": row.accession, "ticker": row.ticker, "reason": "complete_clock_path_unavailable"}])], ignore_index=True)
            continue
        rows.append({
            "accession": str(row.accession), "issuer_cik": int(row.issuer_cik_int), "ticker": str(row.ticker),
            "filing_date": row.filing_date, "acceptance_at": row.acceptance_at, "public_at": row.public_at,
            "public_year_et": int(row.public_year_et), "phase": str(row.phase), "items": str(row.items),
            "prior_reaction_date": cal.iloc[prior_index]["session_date"], "reaction_date": cal.iloc[reaction_index]["session_date"],
            "reaction_close_at": cal.iloc[reaction_index]["market_close_utc"], "entry_date": cal.iloc[entry_index]["session_date"],
            "entry_at": cal.iloc[entry_index]["market_open_utc"], "exit_date": cal.iloc[exit_index]["session_date"],
            "exit_at": cal.iloc[exit_index]["market_open_utc"], "outcome_join_permitted": str(row.phase) != "sealed_2025",
            "price_values_read": False,
        })
    schedule = pd.DataFrame(rows).sort_values(["entry_at", "ticker", "accession"], kind="mergesort").reset_index(drop=True)
    validate_unique(schedule, ["accession"], "Item 2.02 event clock schedule")
    if not (schedule["public_at"].lt(schedule["reaction_close_at"]) & schedule["reaction_close_at"].lt(schedule["entry_at"]) & schedule["entry_at"].lt(schedule["exit_at"])).all():
        raise Sec8KReactionError("event clock ordering failed")
    return schedule, exclusions.reset_index(drop=True), misses.reset_index(drop=True)


__all__ = ["Sec8KReactionError", "build_event_clock_plan"]
