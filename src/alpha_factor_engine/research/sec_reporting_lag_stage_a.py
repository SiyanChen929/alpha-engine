"""Header-only SEC reporting-lag signal construction.

The planner reads public filing clocks and a frozen exchange calendar only.  It
must be run and frozen before any issuer forward return is opened.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from alpha_factor_engine.contracts import validate_unique


class ReportingLagStageAError(ValueError):
    """Raised when the reporting-lag protocol would be violated."""


def _calendar_frame(calendar: pd.DataFrame) -> pd.DataFrame:
    required = {"session_date", "market_open_utc", "market_close_utc"}
    missing = required.difference(calendar.columns)
    if missing:
        raise ReportingLagStageAError(f"calendar missing columns: {sorted(missing)}")
    out = calendar.loc[:, sorted(required)].copy()
    out["session_date"] = pd.to_datetime(out["session_date"], errors="coerce").dt.normalize()
    for column in ("market_open_utc", "market_close_utc"):
        out[column] = pd.to_datetime(out[column], utc=True, errors="coerce")
    if out.isna().any().any():
        raise ReportingLagStageAError("calendar contains invalid values")
    validate_unique(out, ["session_date"], "reporting-lag XNYS calendar")
    return out.sort_values("session_date", kind="mergesort").reset_index(drop=True)


def prepare_annual_headers(headers: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize non-amended 10-K clocks and report signal-only exclusions."""

    required = {
        "issuer_cik", "ticker", "accession", "form", "filing_date",
        "report_date", "acceptance_at",
    }
    missing = required.difference(headers.columns)
    if missing:
        raise ReportingLagStageAError(f"headers missing columns: {sorted(missing)}")
    frame = headers.copy()
    frame["ticker"] = frame["ticker"].astype("string").str.strip().str.upper()
    frame["accession"] = frame["accession"].astype("string").str.strip()
    frame["report_date"] = pd.to_datetime(frame["report_date"], errors="coerce").dt.normalize()
    frame["filing_date"] = pd.to_datetime(frame["filing_date"], errors="coerce").dt.normalize()
    frame["acceptance_at"] = pd.to_datetime(frame["acceptance_at"], utc=True, errors="coerce")
    frame["cik"] = pd.to_numeric(frame["issuer_cik"], errors="coerce").astype("Int64")
    frame["reporting_lag_days"] = (frame["filing_date"] - frame["report_date"]).dt.days
    eligible = (
        frame["form"].eq("10-K")
        & frame["ticker"].ne("")
        & frame["accession"].ne("")
        & frame["cik"].notna()
        & frame["report_date"].notna()
        & frame["filing_date"].notna()
        & frame["acceptance_at"].notna()
    )
    exclusions = frame.loc[~eligible, ["accession", "ticker"]].copy()
    exclusions["reason"] = "not_complete_nonamended_10k_header"
    annual = frame.loc[eligible, [
        "cik", "ticker", "accession", "form", "filing_date", "report_date",
        "acceptance_at", "reporting_lag_days"
    ]].copy()
    validate_unique(annual, ["accession"], "reporting-lag annual headers")
    return annual.sort_values(["acceptance_at", "accession"], kind="mergesort"), exclusions


def _phase(month: pd.Period, cohorts: Mapping[str, Mapping[str, Any]]) -> str | None:
    for name, spec in cohorts.items():
        start = pd.Period(str(spec["formation_month_start"]), freq="M")
        end = pd.Period(str(spec["formation_month_end"]), freq="M")
        if start <= month <= end:
            return str(name)
    return None


def build_reporting_lag_plan(
    headers: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    cohorts: Mapping[str, Mapping[str, Any]],
    minimum_lag_days: int = 20,
    maximum_lag_days: int = 120,
    maximum_signal_age_days: int = 400,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create monthly tail signals using only information public by decision time."""

    annual, header_exclusions = prepare_annual_headers(headers)
    cal = _calendar_frame(calendar)
    cal["month"] = cal["session_date"].dt.to_period("M")
    clocks: dict[pd.Period, dict[str, Any]] = {}
    for month, group in cal.groupby("month", sort=True):
        if len(group) < 2:
            continue
        prior = cal.loc[cal["month"].lt(month)]
        if prior.empty:
            continue
        clocks[month] = {
            "decision_at": prior.iloc[-1]["market_close_utc"] + pd.Timedelta(minutes=30),
            "entry_date": group.iloc[1]["session_date"],
            "entry_at": group.iloc[1]["market_open_utc"],
        }
    rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    for month in sorted(clocks):
        phase = _phase(month, cohorts)
        next_clock = clocks.get(month + 1)
        if phase is None or next_clock is None:
            continue
        clock = clocks[month]
        available = annual.loc[annual["acceptance_at"].le(clock["decision_at"])].copy()
        if available.empty:
            continue
        available = available.sort_values(
            ["ticker", "acceptance_at", "accession"], kind="mergesort"
        ).drop_duplicates("ticker", keep="last")
        available["signal_age_days"] = (
            pd.Timestamp(clock["decision_at"]).normalize().tz_localize(None)
            - available["acceptance_at"].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
        ).dt.days
        available["formation_month"] = str(month)
        available["phase"] = phase
        available["decision_at"] = clock["decision_at"]
        available["lag_eligible"] = (
            available["reporting_lag_days"].between(minimum_lag_days, maximum_lag_days)
            & available["signal_age_days"].between(0, maximum_signal_age_days)
        )
        inventory_rows.extend(available.to_dict("records"))
        eligible = available.loc[available["lag_eligible"]].copy()
        if len(eligible) < 5:
            continue
        eligible = eligible.sort_values(
            ["reporting_lag_days", "accession", "ticker"], kind="mergesort"
        ).reset_index(drop=True)
        eligible["lag_quintile"] = (np.floor(np.arange(len(eligible)) * 5 / len(eligible)) + 1).astype(int)
        tail = eligible.loc[eligible["lag_quintile"].isin([1, 5])].copy()
        for row in tail.itertuples(index=False):
            direction = 1 if int(row.lag_quintile) == 1 else -1
            rows.append({
                "current_accession": f"{row.accession}|{month}",
                "source_accession": str(row.accession),
                "cik": int(row.cik),
                "ticker": str(row.ticker),
                "form": "10-K",
                "accepted_at": row.acceptance_at,
                "report_date": row.report_date,
                "filing_date": row.filing_date,
                "reporting_lag_days": int(row.reporting_lag_days),
                "signal_age_days": int(row.signal_age_days),
                "formation_month": str(month),
                "phase": phase,
                "decision_at": clock["decision_at"],
                "similarity_quintile": 5 if direction == 1 else 1,
                "lag_quintile": int(row.lag_quintile),
                "direction": direction,
                "entry_date": clock["entry_date"],
                "entry_at": clock["entry_at"],
                "calendar_exit_date": next_clock["entry_date"],
                "calendar_exit_at": next_clock["entry_at"],
                "outcome_join_permitted": phase != "sealed_2025",
                "outcome_join_performed": False,
            })
    schedule = pd.DataFrame(rows)
    if schedule.empty:
        raise ReportingLagStageAError("reporting-lag schedule is empty")
    validate_unique(schedule, ["current_accession"], "reporting-lag monthly schedule")
    for month, group in schedule.groupby("formation_month"):
        if set(group["direction"].astype(int)) != {-1, 1}:
            raise ReportingLagStageAError(f"formation month is single-sided: {month}")
        if not group["accepted_at"].le(group["decision_at"]).all():
            raise ReportingLagStageAError(f"future filing crossed decision clock: {month}")
    inventory = pd.DataFrame(inventory_rows)
    return schedule.sort_values(["formation_month", "direction", "ticker"], kind="mergesort"), inventory, header_exclusions
