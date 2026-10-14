"""Strict point-in-time quarterly fundamental releases, including derived Q4.

Version 2 fixes two material defects in the opened v1 diagnostic:

* a 10-K immediately supersedes the preceding 10-Q signal, even when a safe
  fourth-quarter value cannot be reconstructed; and
* every year-over-year comparison uses the same XBRL source tag.

Fourth-quarter flows prefer direct quarterly facts carried by the 10-K.  A
derived value is allowed only when annual and nine-month current/prior legs all
coexist in that same 10-K accession.  Cross-accession subtraction is excluded
because retrospective scope changes can make it economically invalid.  Because
Company Facts omits raw ``contextRef`` metadata, this is a compatible-context
diagnostic until raw filing XBRL is independently reconciled.  Ambiguous or
malformed contexts fail closed to missing values.  Missing values are intentional
supersession markers: an old Q3 signal is never carried through a newer 10-K.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any

import numpy as np
import pandas as pd


DURATION_TAGS: dict[str, tuple[str, ...]] = {
    "net_income": ("NetIncomeLoss", "ProfitLoss"),
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ),
    "gross_profit": ("GrossProfit",),
}
INSTANT_TAGS: dict[str, tuple[str, ...]] = {"assets": ("Assets",)}
RELEASE_FEATURES = (
    "earnings_change_on_assets",
    "sales_growth_yoy",
    "gross_margin_change_yoy",
)

QUARTER_MIN_DAYS = 70
QUARTER_MAX_DAYS = 120
Q3_YTD_MIN_DAYS = 230
Q3_YTD_MAX_DAYS = 310
ANNUAL_MIN_DAYS = 350
ANNUAL_MAX_DAYS = 380
YOY_MIN_DAYS = 350
YOY_MAX_DAYS = 380
MAX_10Q_LAG_DAYS = 120
MAX_10K_LAG_DAYS = 180
PERIOD_END_TOLERANCE_DAYS = 0
ANNUAL_START_TOLERANCE_DAYS = 1
Q3_TO_YEAR_END_MIN_DAYS = 70
Q3_TO_YEAR_END_MAX_DAYS = 120
MAX_COMPARATIVE_DURATION_DIFFERENCE_DAYS = 8
MAX_Q4_DURATION_DIFFERENCE_DAYS = 5
SEC_PROPAGATION_DELAY_MINUTES = 3


def filing_day_end_utc(value: object) -> pd.Timestamp:
    """Return a conservative public clock after the full SEC filing date."""

    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("filing date is missing")
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("America/New_York").tz_localize(None)
    day = stamp.normalize().tz_localize("America/New_York")
    return (day + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)).tz_convert(
        "UTC"
    )


def _sec_date(value: object) -> pd.Timestamp | None:
    if not isinstance(value, str):
        return None
    try:
        return pd.Timestamp(date.fromisoformat(value[:10]))
    except (TypeError, ValueError):
        return None


def _finite_float(value: object) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _unit_entries(
    us_gaap: Mapping[str, Any], tag: str, unit: str = "USD"
) -> list[dict[str, Any]]:
    concept = us_gaap.get(tag) or {}
    entries = (concept.get("units") or {}).get(unit) or []
    return [dict(item) for item in entries]


def _duration_candidates(
    us_gaap: Mapping[str, Any], ticker: str, cik: int
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for canonical, tags in DURATION_TAGS.items():
        for priority, tag in enumerate(tags):
            for item in _unit_entries(us_gaap, tag):
                if item.get("form") not in {"10-Q", "10-Q/A", "10-K", "10-K/A"}:
                    continue
                start = _sec_date(item.get("start"))
                end = _sec_date(item.get("end"))
                filed = _sec_date(item.get("filed"))
                value = _finite_float(item.get("val"))
                accession = str(item.get("accn") or "")
                if (
                    start is None
                    or end is None
                    or filed is None
                    or value is None
                    or not accession
                    or end > filed
                    or start >= end
                ):
                    continue
                rows.append(
                    {
                        "ticker": ticker,
                        "cik": int(cik),
                        "accession": accession,
                        "form": str(item.get("form")),
                        "fiscal_year": item.get("fy"),
                        "fiscal_period": str(item.get("fp") or ""),
                        "filed": filed,
                        "start": start,
                        "end": end,
                        "duration_days": int((end - start).days),
                        "concept": canonical,
                        "source_tag": tag,
                        "tag_priority": int(priority),
                        "unit": "USD",
                        "value": float(value),
                    }
                )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


def _instant_candidates(
    us_gaap: Mapping[str, Any], ticker: str, cik: int
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for canonical, tags in INSTANT_TAGS.items():
        for priority, tag in enumerate(tags):
            for item in _unit_entries(us_gaap, tag):
                if item.get("form") not in {"10-Q", "10-Q/A", "10-K", "10-K/A"}:
                    continue
                end = _sec_date(item.get("end"))
                filed = _sec_date(item.get("filed"))
                value = _finite_float(item.get("val"))
                accession = str(item.get("accn") or "")
                if (
                    end is None
                    or filed is None
                    or value is None
                    or not accession
                    or end > filed
                ):
                    continue
                rows.append(
                    {
                        "ticker": ticker,
                        "cik": int(cik),
                        "accession": accession,
                        "form": str(item.get("form")),
                        "fiscal_year": item.get("fy"),
                        "fiscal_period": str(item.get("fp") or ""),
                        "filed": filed,
                        "end": end,
                        "concept": canonical,
                        "source_tag": tag,
                        "tag_priority": int(priority),
                        "unit": "USD",
                        "value": float(value),
                    }
                )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


def _unambiguous_row(candidates: pd.DataFrame, order: list[str]) -> pd.Series | None:
    """Select one context only when its best key has one economic value."""

    if candidates.empty:
        return None
    ordered = candidates.sort_values(order, kind="mergesort")
    best = ordered.iloc[0]
    key_columns = [column for column in order if column in ordered.columns]
    same_key = pd.Series(True, index=ordered.index)
    for column in key_columns:
        same_key &= ordered[column].eq(best[column])
    tied = ordered.loc[same_key]
    if tied["value"].nunique(dropna=False) != 1:
        return None
    return tied.iloc[0]


def _availability(
    accession: str,
    filed: pd.Timestamp,
    acceptance: Mapping[str, object],
) -> tuple[pd.Timestamp, str]:
    exact = acceptance.get(accession)
    correction_at: object | None = None
    if isinstance(exact, Mapping):
        exact_value = exact.get("acceptance_at")
        correction_at = exact.get("correction_at")
        exact = (
            exact_value
            if exact_value is not None
            else exact.get("acceptanceDateTime")
        )
    if exact is None or pd.isna(exact):
        return filing_day_end_utc(filed), "conservative_filing_day_end_et"
    parsed = pd.to_datetime(exact, utc=True, errors="coerce")
    if pd.isna(parsed):
        return filing_day_end_utc(filed), "conservative_filing_day_end_et"
    available_at = pd.Timestamp(parsed)
    available_at += pd.Timedelta(minutes=SEC_PROPAGATION_DELAY_MINUTES)
    if correction_at is not None and pd.notna(correction_at):
        correction = pd.Timestamp(correction_at)
        correction = (
            correction.tz_localize("UTC")
            if correction.tzinfo is None
            else correction.tz_convert("UTC")
        )
        available_at = max(available_at, correction)
    acceptance_et_date = available_at.tz_convert("America/New_York").date()
    if acceptance_et_date < filed.date():
        available_at = max(available_at, filing_day_end_utc(filed))
    return available_at, "submissions_acceptance_plus_3m"


def _inventory_barriers(
    ticker: str,
    cik: int,
    records: list[dict[str, object]],
    inventory: Mapping[str, object],
) -> list[dict[str, object]]:
    """Overlay complete submissions headers and emit missing-fact barriers."""

    by_accession = {str(row["accession"]): row for row in records}
    for accession, metadata in inventory.items():
        if not isinstance(metadata, Mapping):
            continue
        form = str(metadata.get("form") or "")
        if form not in {"10-Q", "10-Q/A", "10-K", "10-K/A"}:
            continue
        filed = _sec_date(metadata.get("filing_date") or metadata.get("filingDate"))
        fiscal_end = _sec_date(
            metadata.get("report_date") or metadata.get("reportDate")
        )
        if filed is None:
            continue
        if fiscal_end is not None and fiscal_end > filed:
            fiscal_end = None
        maximum_lag = MAX_10K_LAG_DAYS if form.startswith("10-K") else MAX_10Q_LAG_DAYS
        available_at, clock_source = _availability(accession, filed, inventory)
        existing = by_accession.get(accession)
        valid_period = bool(
            fiscal_end is not None
            and 0 <= int((filed - fiscal_end).days) <= maximum_lag
        )
        existing_matches = bool(
            existing is not None
            and fiscal_end is not None
            and pd.Timestamp(existing["fiscal_end"]) == fiscal_end
        )
        if not valid_period or not existing_matches:
            existing = _empty_release(
                ticker=ticker,
                cik=cik,
                accession=accession,
                form=form,
                fiscal_year=metadata.get("fiscal_year"),
                fiscal_period="Q4" if form.startswith("10-K") else "UNKNOWN",
                fiscal_end=fiscal_end if fiscal_end is not None else pd.NaT,
                filed=filed,
                available_at=available_at,
                clock_source=clock_source,
                release_kind=(
                    "unknown_period_null_barrier"
                    if fiscal_end is None
                    else (
                        "late_periodic_null_barrier"
                        if not valid_period
                        else (
                            "inventory_10k_barrier"
                            if form.startswith("10-K")
                            else "inventory_10q_barrier"
                        )
                    )
                ),
            )
            existing["supersession_only"] = True
            by_accession[accession] = existing
        else:
            existing["form"] = form
            existing["filed"] = filed
            existing["release_lag_days"] = int((filed - fiscal_end).days)
            existing["available_at"] = available_at
            existing["clock_source"] = clock_source
        existing["barrier_inventory_source"] = "submissions_header"
        existing["header_inventory_verified"] = True

    for row in by_accession.values():
        row.setdefault("barrier_inventory_source", "companyfacts_target_or_assets")
        row.setdefault("header_inventory_verified", False)
    return list(by_accession.values())


def _empty_release(
    *,
    ticker: str,
    cik: int,
    accession: str,
    form: str,
    fiscal_year: object,
    fiscal_period: str,
    fiscal_end: pd.Timestamp,
    filed: pd.Timestamp,
    available_at: pd.Timestamp,
    clock_source: str,
    release_kind: str,
) -> dict[str, object]:
    record: dict[str, object] = {
        "ticker": ticker,
        "cik": int(cik),
        "accession": accession,
        "form": form,
        "fiscal_year": fiscal_year,
        "fiscal_period": fiscal_period,
        "fiscal_end": fiscal_end,
        "filed": filed,
        "release_lag_days": (
            int((filed - fiscal_end).days) if pd.notna(fiscal_end) else np.nan
        ),
        "available_at": available_at,
        "clock_source": clock_source,
        "release_kind": release_kind,
        "supersession_marker": True,
        "context_validation_level": "companyfacts_compatible_no_contextref",
    }
    for concept in DURATION_TAGS:
        record[f"{concept}_current"] = np.nan
        record[f"{concept}_prior_year"] = np.nan
        record[f"{concept}_source_tag"] = None
        record[f"{concept}_prior_source_tag"] = None
        record[f"{concept}_same_tag_comparator"] = False
        record[f"{concept}_annual"] = np.nan
        record[f"{concept}_annual_prior_year"] = np.nan
        record[f"{concept}_q3_ytd"] = np.nan
        record[f"{concept}_q3_ytd_prior_year"] = np.nan
        record[f"{concept}_q3_ytd_accession"] = None
        record[f"{concept}_q4_method"] = None
        record[f"{concept}_q4_days_current"] = np.nan
        record[f"{concept}_q4_days_prior"] = np.nan
        for leg in (
            "current_fact",
            "prior_fact",
            "annual_current",
            "annual_prior",
            "q3_ytd_current",
            "q3_ytd_prior",
        ):
            record[f"{concept}_{leg}_accession"] = None
            record[f"{concept}_{leg}_start"] = pd.NaT
            record[f"{concept}_{leg}_end"] = pd.NaT
            record[f"{concept}_{leg}_unit"] = None
        record[f"{concept}_q3_ytd_available_at"] = pd.NaT
    record["annual_available_at"] = available_at
    record["assets_current"] = np.nan
    record["earnings_change_on_assets"] = np.nan
    record["sales_growth_yoy"] = np.nan
    record["gross_margin_change_yoy"] = np.nan
    return record


def _select_asset(
    instant: pd.DataFrame,
    accession: str,
    fiscal_end: pd.Timestamp,
    form_prefix: str,
) -> pd.Series | None:
    if instant.empty:
        return None
    scope = instant.loc[
        instant["accession"].eq(accession)
        & instant["form"].str.startswith(form_prefix)
    ].copy()
    if scope.empty:
        return None
    scope["end_error"] = (scope["end"] - fiscal_end).abs().dt.days
    scope = scope.loc[scope["end_error"].eq(0)]
    return _unambiguous_row(
        scope, ["end_error", "tag_priority", "source_tag", "end"]
    )


def _add_features(record: dict[str, object]) -> None:
    def number(name: str) -> float:
        value = record.get(name)
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return float("nan")

    assets = number("assets_current")
    ni_now = number("net_income_current")
    ni_prior = number("net_income_prior_year")
    rev_now = number("revenue_current")
    rev_prior = number("revenue_prior_year")
    gp_now = number("gross_profit_current")
    gp_prior = number("gross_profit_prior_year")
    record["earnings_change_on_assets"] = (
        (ni_now - ni_prior) / assets
        if np.isfinite(ni_now)
        and np.isfinite(ni_prior)
        and np.isfinite(assets)
        and assets > 0
        else np.nan
    )
    record["sales_growth_yoy"] = (
        rev_now / rev_prior - 1
        if np.isfinite(rev_now) and np.isfinite(rev_prior) and rev_prior > 0
        else np.nan
    )
    record["gross_margin_change_yoy"] = (
        gp_now / rev_now - gp_prior / rev_prior
        if np.isfinite(gp_now)
        and np.isfinite(gp_prior)
        and np.isfinite(rev_now)
        and np.isfinite(rev_prior)
        and rev_now > 0
        and rev_prior > 0
        else np.nan
    )


def _extract_10q(
    ticker: str,
    cik: int,
    duration: pd.DataFrame,
    instant: pd.DataFrame,
    acceptance: Mapping[str, object],
) -> list[dict[str, object]]:
    if duration.empty:
        return []
    discrete_all = duration.loc[
        duration["form"].isin({"10-Q", "10-Q/A"})
        & duration["fiscal_period"].isin({"Q1", "Q2", "Q3"})
        & duration["duration_days"].between(QUARTER_MIN_DAYS, QUARTER_MAX_DAYS)
    ].copy()
    discrete_current = discrete_all.copy()
    discrete_current["filing_lag"] = (
        discrete_current["filed"] - discrete_current["end"]
    ).dt.days
    discrete_current = discrete_current.loc[
        discrete_current["filing_lag"].between(0, MAX_10Q_LAG_DAYS)
    ]
    output: list[dict[str, object]] = []
    for accession, current_rows in discrete_current.groupby("accession", sort=True):
        filing_rows = discrete_all.loc[discrete_all["accession"].eq(accession)].copy()
        fiscal_end = pd.Timestamp(current_rows["end"].max())
        current_scope = current_rows.loc[
            current_rows["end"].between(
                fiscal_end - pd.Timedelta(days=PERIOD_END_TOLERANCE_DAYS),
                fiscal_end + pd.Timedelta(days=PERIOD_END_TOLERANCE_DAYS),
            )
        ].copy()
        if current_scope.empty:
            continue
        anchor = current_scope.sort_values(
            ["filed", "tag_priority", "source_tag"], kind="mergesort"
        ).iloc[-1]
        filed = pd.Timestamp(anchor["filed"])
        available_at, clock_source = _availability(accession, filed, acceptance)
        if available_at <= fiscal_end.tz_localize("UTC"):
            raise ValueError(
                f"availability does not follow fiscal end: {ticker} {accession}"
            )
        record = _empty_release(
            ticker=ticker,
            cik=cik,
            accession=accession,
            form=str(anchor["form"]),
            fiscal_year=anchor["fiscal_year"],
            fiscal_period=str(anchor["fiscal_period"]),
            fiscal_end=fiscal_end,
            filed=filed,
            available_at=available_at,
            clock_source=clock_source,
            release_kind="reported_10q",
        )
        for concept in DURATION_TAGS:
            current_candidates = current_scope.loc[
                current_scope["concept"].eq(concept)
            ].sort_values(["tag_priority", "source_tag", "start"], kind="mergesort")
            chosen_current: pd.Series | None = None
            chosen_prior: pd.Series | None = None
            for source_tag, same_tag_current in current_candidates.groupby(
                "source_tag", sort=False
            ):
                current = _unambiguous_row(
                    same_tag_current,
                    ["tag_priority", "source_tag", "start", "end"],
                )
                if current is None:
                    continue
                prior = filing_rows.loc[
                    filing_rows["concept"].eq(concept)
                    & filing_rows["source_tag"].eq(source_tag)
                ].copy()
                prior["year_gap"] = fiscal_end - prior["end"]
                prior["year_gap"] = prior["year_gap"].dt.days
                prior = prior.loc[prior["year_gap"].between(YOY_MIN_DAYS, YOY_MAX_DAYS)]
                prior["duration_difference"] = (
                    prior["duration_days"] - int(current["duration_days"])
                ).abs()
                prior = prior.loc[
                    prior["duration_difference"].le(
                        MAX_COMPARATIVE_DURATION_DIFFERENCE_DAYS
                    )
                ]
                if prior.empty:
                    continue
                prior["gap_error"] = (prior["year_gap"] - 365).abs()
                candidate_prior = _unambiguous_row(
                    prior,
                    ["gap_error", "tag_priority", "source_tag", "end", "start"],
                )
                if candidate_prior is not None:
                    chosen_current = current
                    chosen_prior = candidate_prior
                    break
            if chosen_current is None or chosen_prior is None:
                continue
            tag = str(chosen_current["source_tag"])
            record[f"{concept}_current"] = float(chosen_current["value"])
            record[f"{concept}_prior_year"] = float(chosen_prior["value"])
            record[f"{concept}_source_tag"] = tag
            record[f"{concept}_prior_source_tag"] = str(chosen_prior["source_tag"])
            record[f"{concept}_same_tag_comparator"] = bool(
                tag == str(chosen_prior["source_tag"])
            )
            for leg, fact in (
                ("current_fact", chosen_current),
                ("prior_fact", chosen_prior),
            ):
                record[f"{concept}_{leg}_accession"] = str(fact["accession"])
                record[f"{concept}_{leg}_start"] = pd.Timestamp(fact["start"])
                record[f"{concept}_{leg}_end"] = pd.Timestamp(fact["end"])
                record[f"{concept}_{leg}_unit"] = str(fact["unit"])
        asset = _select_asset(instant, accession, fiscal_end, "10-Q")
        if asset is not None:
            record["assets_current"] = float(asset["value"])
        _add_features(record)
        record["supersession_only"] = not bool(
            np.isfinite(float(record["earnings_change_on_assets"]))
            and np.isfinite(float(record["sales_growth_yoy"]))
        )
        output.append(record)
    return output


def _direct_q4_pair(
    k_filing: pd.DataFrame,
    concept: str,
    fiscal_end: pd.Timestamp,
) -> tuple[pd.Series, pd.Series] | None:
    """Resolve current/prior direct Q4 facts from one 10-K accession."""

    current_scope = k_filing.loc[
        k_filing["concept"].eq(concept)
        & k_filing["duration_days"].between(QUARTER_MIN_DAYS, QUARTER_MAX_DAYS)
        & k_filing["end"].eq(fiscal_end)
    ].sort_values(["tag_priority", "source_tag", "start"], kind="mergesort")
    for source_tag, tagged in current_scope.groupby("source_tag", sort=False):
        current = _unambiguous_row(
            tagged, ["tag_priority", "source_tag", "start", "end"]
        )
        if current is None:
            continue
        prior = k_filing.loc[
            k_filing["concept"].eq(concept)
            & k_filing["source_tag"].eq(source_tag)
            & k_filing["duration_days"].between(QUARTER_MIN_DAYS, QUARTER_MAX_DAYS)
        ].copy()
        prior["end_gap"] = fiscal_end - prior["end"]
        prior["end_gap"] = prior["end_gap"].dt.days
        prior["duration_difference"] = (
            prior["duration_days"] - int(current["duration_days"])
        ).abs()
        prior = prior.loc[
            prior["end_gap"].between(YOY_MIN_DAYS, YOY_MAX_DAYS)
            & prior["duration_difference"].le(MAX_Q4_DURATION_DIFFERENCE_DAYS)
        ].copy()
        if prior.empty:
            continue
        prior["gap_error"] = (prior["end_gap"] - 365).abs()
        chosen_prior = _unambiguous_row(
            prior, ["gap_error", "tag_priority", "source_tag", "end", "start"]
        )
        if chosen_prior is not None:
            return current, chosen_prior
    return None


def _same_10k_derived_quad(
    k_filing: pd.DataFrame,
    concept: str,
    fiscal_end: pd.Timestamp,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series] | None:
    """Resolve annual and 9M current/prior legs carried by one 10-K."""

    annual_scope = k_filing.loc[
        k_filing["concept"].eq(concept)
        & k_filing["duration_days"].between(ANNUAL_MIN_DAYS, ANNUAL_MAX_DAYS)
        & k_filing["end"].eq(fiscal_end)
    ].sort_values(["tag_priority", "source_tag", "start"], kind="mergesort")
    for source_tag, tagged in annual_scope.groupby("source_tag", sort=False):
        annual_now = _unambiguous_row(
            tagged, ["tag_priority", "source_tag", "start", "end"]
        )
        if annual_now is None:
            continue
        annual_prior = k_filing.loc[
            k_filing["concept"].eq(concept)
            & k_filing["source_tag"].eq(source_tag)
            & k_filing["duration_days"].between(ANNUAL_MIN_DAYS, ANNUAL_MAX_DAYS)
        ].copy()
        annual_prior["end_gap"] = fiscal_end - annual_prior["end"]
        annual_prior["end_gap"] = annual_prior["end_gap"].dt.days
        annual_prior["start_gap"] = pd.Timestamp(annual_now["start"]) - annual_prior["start"]
        annual_prior["start_gap"] = annual_prior["start_gap"].dt.days
        annual_prior["duration_difference"] = (
            annual_prior["duration_days"] - int(annual_now["duration_days"])
        ).abs()
        annual_prior = annual_prior.loc[
            annual_prior["end_gap"].between(YOY_MIN_DAYS, YOY_MAX_DAYS)
            & annual_prior["start_gap"].between(YOY_MIN_DAYS, YOY_MAX_DAYS)
            & annual_prior["duration_difference"].le(
                MAX_COMPARATIVE_DURATION_DIFFERENCE_DAYS
            )
        ].copy()
        if annual_prior.empty:
            continue
        annual_prior["gap_error"] = (annual_prior["end_gap"] - 365).abs()
        prior_annual = _unambiguous_row(
            annual_prior,
            ["gap_error", "tag_priority", "source_tag", "end", "start"],
        )
        if prior_annual is None:
            continue

        ytd_now_scope = k_filing.loc[
            k_filing["concept"].eq(concept)
            & k_filing["source_tag"].eq(source_tag)
            & k_filing["duration_days"].between(Q3_YTD_MIN_DAYS, Q3_YTD_MAX_DAYS)
        ].copy()
        ytd_now_scope["start_error"] = (
            ytd_now_scope["start"] - pd.Timestamp(annual_now["start"])
        ).abs().dt.days
        ytd_now_scope["annual_end_gap"] = (
            fiscal_end - ytd_now_scope["end"]
        ).dt.days
        ytd_now_scope = ytd_now_scope.loc[
            ytd_now_scope["start_error"].le(ANNUAL_START_TOLERANCE_DAYS)
            & ytd_now_scope["annual_end_gap"].between(
                Q3_TO_YEAR_END_MIN_DAYS, Q3_TO_YEAR_END_MAX_DAYS
            )
        ].copy()
        ytd_now_scope["gap_error"] = (ytd_now_scope["annual_end_gap"] - 91).abs()
        ytd_now = _unambiguous_row(
            ytd_now_scope,
            ["start_error", "gap_error", "tag_priority", "source_tag", "end", "start"],
        )
        if ytd_now is None:
            continue

        ytd_prior = k_filing.loc[
            k_filing["concept"].eq(concept)
            & k_filing["source_tag"].eq(source_tag)
            & k_filing["duration_days"].between(Q3_YTD_MIN_DAYS, Q3_YTD_MAX_DAYS)
        ].copy()
        ytd_prior["end_gap"] = pd.Timestamp(ytd_now["end"]) - ytd_prior["end"]
        ytd_prior["end_gap"] = ytd_prior["end_gap"].dt.days
        ytd_prior["start_gap"] = pd.Timestamp(ytd_now["start"]) - ytd_prior["start"]
        ytd_prior["start_gap"] = ytd_prior["start_gap"].dt.days
        ytd_prior["annual_start_error"] = (
            ytd_prior["start"] - pd.Timestamp(prior_annual["start"])
        ).abs().dt.days
        ytd_prior["annual_end_gap"] = (
            pd.Timestamp(prior_annual["end"]) - ytd_prior["end"]
        ).dt.days
        ytd_prior["duration_difference"] = (
            ytd_prior["duration_days"] - int(ytd_now["duration_days"])
        ).abs()
        ytd_prior = ytd_prior.loc[
            ytd_prior["end_gap"].between(YOY_MIN_DAYS, YOY_MAX_DAYS)
            & ytd_prior["start_gap"].between(YOY_MIN_DAYS, YOY_MAX_DAYS)
            & ytd_prior["annual_start_error"].le(ANNUAL_START_TOLERANCE_DAYS)
            & ytd_prior["annual_end_gap"].between(
                Q3_TO_YEAR_END_MIN_DAYS, Q3_TO_YEAR_END_MAX_DAYS
            )
            & ytd_prior["duration_difference"].le(
                MAX_COMPARATIVE_DURATION_DIFFERENCE_DAYS
            )
        ].copy()
        if ytd_prior.empty:
            continue
        ytd_prior["gap_error"] = (ytd_prior["end_gap"] - 365).abs()
        prior_ytd = _unambiguous_row(
            ytd_prior,
            ["gap_error", "tag_priority", "source_tag", "end", "start"],
        )
        if prior_ytd is None:
            continue
        q4_days_now = int((fiscal_end - pd.Timestamp(ytd_now["end"])).days)
        q4_days_prior = int(
            (pd.Timestamp(prior_annual["end"]) - pd.Timestamp(prior_ytd["end"])).days
        )
        if abs(q4_days_now - q4_days_prior) > MAX_Q4_DURATION_DIFFERENCE_DAYS:
            continue
        return annual_now, prior_annual, ytd_now, prior_ytd
    return None


def _extract_10k(
    ticker: str,
    cik: int,
    duration: pd.DataFrame,
    instant: pd.DataFrame,
    acceptance: Mapping[str, object],
) -> list[dict[str, object]]:
    if duration.empty and instant.empty:
        return []
    annual_all = duration.loc[
        duration["form"].isin({"10-K", "10-K/A"})
        & duration["duration_days"].between(ANNUAL_MIN_DAYS, ANNUAL_MAX_DAYS)
    ].copy() if not duration.empty else pd.DataFrame()
    annual_current = annual_all.copy()
    if not annual_current.empty:
        annual_current["filing_lag"] = (
            annual_current["filed"] - annual_current["end"]
        ).dt.days
        annual_current = annual_current.loc[
            annual_current["filing_lag"].between(0, MAX_10K_LAG_DAYS)
        ]

    instant_k = instant.loc[instant["form"].isin({"10-K", "10-K/A"})].copy() if not instant.empty else pd.DataFrame()
    if not instant_k.empty:
        instant_k["filing_lag"] = (instant_k["filed"] - instant_k["end"]).dt.days
        instant_k = instant_k.loc[instant_k["filing_lag"].between(0, MAX_10K_LAG_DAYS)]

    accessions = sorted(
        set(annual_current.get("accession", pd.Series(dtype=str)).astype(str))
        | set(instant_k.get("accession", pd.Series(dtype=str)).astype(str))
    )
    if not accessions:
        return []

    output: list[dict[str, object]] = []
    for accession in accessions:
        k_filing = duration.loc[
            duration["accession"].eq(accession)
            & duration["form"].isin({"10-K", "10-K/A"})
        ].copy()
        annual_filing = annual_all.loc[annual_all["accession"].eq(accession)].copy()
        annual_current_filing = annual_current.loc[
            annual_current["accession"].eq(accession)
        ].copy()
        instant_filing = instant_k.loc[instant_k["accession"].eq(accession)].copy()
        end_candidates = pd.concat(
            [
                annual_current_filing[["end", "filed", "form", "fiscal_year"]],
                instant_filing[["end", "filed", "form", "fiscal_year"]],
            ],
            ignore_index=True,
        )
        if end_candidates.empty:
            continue
        fiscal_end = pd.Timestamp(end_candidates["end"].max())
        current_anchor = end_candidates.loc[
            end_candidates["end"].between(
                fiscal_end - pd.Timedelta(days=PERIOD_END_TOLERANCE_DAYS),
                fiscal_end + pd.Timedelta(days=PERIOD_END_TOLERANCE_DAYS),
            )
        ].sort_values(["filed", "form"], kind="mergesort")
        if current_anchor.empty:
            continue
        anchor = current_anchor.iloc[-1]
        filed = pd.Timestamp(anchor["filed"])
        available_at, clock_source = _availability(accession, filed, acceptance)
        if available_at <= fiscal_end.tz_localize("UTC"):
            raise ValueError(
                f"availability does not follow fiscal end: {ticker} {accession}"
            )
        record = _empty_release(
            ticker=ticker,
            cik=cik,
            accession=accession,
            form=str(anchor["form"]),
            fiscal_year=anchor["fiscal_year"],
            fiscal_period="Q4",
            fiscal_end=fiscal_end,
            filed=filed,
            available_at=available_at,
            clock_source=clock_source,
            release_kind="q4_10k_barrier",
        )
        for concept in DURATION_TAGS:
            direct = _direct_q4_pair(k_filing, concept, fiscal_end)
            if direct is not None:
                current, prior = direct
                tag = str(current["source_tag"])
                record[f"{concept}_current"] = float(current["value"])
                record[f"{concept}_prior_year"] = float(prior["value"])
                record[f"{concept}_source_tag"] = tag
                record[f"{concept}_prior_source_tag"] = tag
                record[f"{concept}_same_tag_comparator"] = True
                record[f"{concept}_q4_method"] = "direct_10k"
                record[f"{concept}_q4_days_current"] = int(current["duration_days"])
                record[f"{concept}_q4_days_prior"] = int(prior["duration_days"])
                for leg, fact in (("current_fact", current), ("prior_fact", prior)):
                    record[f"{concept}_{leg}_accession"] = str(fact["accession"])
                    record[f"{concept}_{leg}_start"] = pd.Timestamp(fact["start"])
                    record[f"{concept}_{leg}_end"] = pd.Timestamp(fact["end"])
                    record[f"{concept}_{leg}_unit"] = str(fact["unit"])
                continue

            quad = _same_10k_derived_quad(k_filing, concept, fiscal_end)
            if quad is not None:
                annual_now, annual_prior, ytd_now, ytd_prior = quad
                tag = str(annual_now["source_tag"])
                record[f"{concept}_annual"] = float(annual_now["value"])
                record[f"{concept}_annual_prior_year"] = float(annual_prior["value"])
                record[f"{concept}_q3_ytd"] = float(ytd_now["value"])
                record[f"{concept}_q3_ytd_prior_year"] = float(ytd_prior["value"])
                record[f"{concept}_current"] = float(annual_now["value"] - ytd_now["value"])
                record[f"{concept}_prior_year"] = float(annual_prior["value"] - ytd_prior["value"])
                record[f"{concept}_source_tag"] = tag
                record[f"{concept}_prior_source_tag"] = tag
                record[f"{concept}_same_tag_comparator"] = True
                record[f"{concept}_q4_method"] = "same_10k_derived"
                record[f"{concept}_q3_ytd_accession"] = accession
                record[f"{concept}_q3_ytd_available_at"] = available_at
                record[f"{concept}_q4_days_current"] = int(
                    (fiscal_end - pd.Timestamp(ytd_now["end"])).days
                )
                record[f"{concept}_q4_days_prior"] = int(
                    (pd.Timestamp(annual_prior["end"]) - pd.Timestamp(ytd_prior["end"])).days
                )
                for leg, fact in (
                    ("annual_current", annual_now),
                    ("annual_prior", annual_prior),
                    ("q3_ytd_current", ytd_now),
                    ("q3_ytd_prior", ytd_prior),
                ):
                    record[f"{concept}_{leg}_accession"] = str(fact["accession"])
                    record[f"{concept}_{leg}_start"] = pd.Timestamp(fact["start"])
                    record[f"{concept}_{leg}_end"] = pd.Timestamp(fact["end"])
                    record[f"{concept}_{leg}_unit"] = str(fact["unit"])
        asset = _select_asset(instant_k, accession, fiscal_end, "10-K")
        if asset is not None:
            record["assets_current"] = float(asset["value"])
        output.append(record)

    ordered = sorted(
        output,
        key=lambda row: (pd.Timestamp(row["available_at"]), str(row["accession"])),
    )
    for current in ordered:
        _add_features(current)
        current["supersession_only"] = not bool(
            np.isfinite(float(current["earnings_change_on_assets"]))
            and np.isfinite(float(current["sales_growth_yoy"]))
        )
        if not current["supersession_only"]:
            current["release_kind"] = "q4_10k_strict"
    return ordered


def _monotonic_fiscal_releases(records: list[dict[str, object]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True)
    frame["fiscal_end"] = pd.to_datetime(frame["fiscal_end"], errors="coerce")
    frame = frame.sort_values(
        ["ticker", "available_at", "fiscal_end", "accession"], kind="mergesort"
    ).drop_duplicates(["ticker", "accession"], keep="last")
    keep: list[int] = []
    for _, company in frame.groupby("ticker", sort=True):
        running_end: pd.Timestamp | None = None
        for index, row in company.iterrows():
            fiscal_end = pd.Timestamp(row["fiscal_end"])
            if pd.isna(fiscal_end):
                keep.append(index)
                continue
            if running_end is not None and fiscal_end < running_end:
                continue
            keep.append(index)
            running_end = fiscal_end if running_end is None else max(running_end, fiscal_end)
    return frame.loc[keep].sort_values(
        ["ticker", "available_at", "accession"], kind="mergesort"
    ).reset_index(drop=True)


def extract_quarterly_release_features(
    ticker: str,
    cik: int,
    payload: Mapping[str, Any],
    acceptance: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Extract strict same-tag 10-Q signals and conservative 10-K/Q4 markers."""

    acceptance = acceptance or {}
    us_gaap = ((payload.get("facts") or {}).get("us-gaap") or {})
    duration = _duration_candidates(us_gaap, ticker, cik)
    instant = _instant_candidates(us_gaap, ticker, cik)
    records = [
        *_extract_10q(ticker, cik, duration, instant, acceptance),
        *_extract_10k(ticker, cik, duration, instant, acceptance),
    ]
    records = _inventory_barriers(ticker, cik, records, acceptance)
    return _monotonic_fiscal_releases(records)


def align_quarterly_features(
    panel: pd.DataFrame,
    releases: pd.DataFrame,
    *,
    max_age_days: int = 180,
    operational_latency_minutes: int = 30,
) -> pd.DataFrame:
    """As-of align the latest filing marker and construct the primary score."""

    if missing := sorted(
        {"ticker", "signal_date", "entry_at"}.difference(panel.columns)
    ):
        raise ValueError(f"monthly panel missing columns: {missing}")
    required = {"ticker", "available_at", "accession", *RELEASE_FEATURES}
    if missing := sorted(required.difference(releases.columns)):
        raise ValueError(f"quarterly releases missing columns: {missing}")
    left = panel.copy()
    left["signal_date"] = pd.to_datetime(left["signal_date"], utc=True, errors="raise")
    left["entry_at"] = pd.to_datetime(left["entry_at"], utc=True, errors="raise")
    left["decision_at"] = left["entry_at"] - pd.to_timedelta(
        int(operational_latency_minutes), unit="minute"
    )
    if left["decision_at"].ge(left["entry_at"]).any():
        raise ValueError("decision_at must strictly precede entry_at")
    left["_row_id"] = np.arange(len(left), dtype=int)
    right = releases.copy()
    right["available_at"] = pd.to_datetime(right["available_at"], utc=True, errors="raise")
    if right.duplicated(["ticker", "available_at", "accession"]).any():
        raise ValueError("quarterly releases contain duplicate keys")
    right = right.rename(
        columns={"available_at": "quarterly_available_at", "cik": "issuer_cik"}
    )
    release_columns = [column for column in right if column != "ticker"]
    parts: list[pd.DataFrame] = []
    for ticker, decisions in left.groupby("ticker", sort=True):
        company = right.loc[right["ticker"].eq(ticker), release_columns].sort_values(
            ["quarterly_available_at", "accession"], kind="mergesort"
        )
        decisions = decisions.sort_values("decision_at", kind="mergesort")
        if company.empty:
            merged = decisions.copy()
        else:
            merged = pd.merge_asof(
                decisions,
                company,
                left_on="decision_at",
                right_on="quarterly_available_at",
                direction="backward",
                allow_exact_matches=False,
            )
        parts.append(merged)
    result = pd.concat(parts, ignore_index=True).sort_values("_row_id")
    if "quarterly_available_at" not in result:
        result["quarterly_available_at"] = pd.NaT
    for column in RELEASE_FEATURES:
        if column not in result:
            result[column] = np.nan
    result["quarterly_available_at"] = pd.to_datetime(
        result["quarterly_available_at"], utc=True, errors="coerce"
    )
    result["signal_date"] = pd.to_datetime(result["signal_date"], utc=True, errors="raise")
    known = result["quarterly_available_at"].notna()
    if (
        result.loc[known, "quarterly_available_at"]
        >= result.loc[known, "decision_at"]
    ).any():
        raise ValueError("future or simultaneous filing leaked into a decision")
    feature_age = result["decision_at"] - result["quarterly_available_at"]
    result["quarterly_feature_age_days"] = feature_age.dt.total_seconds() / 86_400
    fresh = known & feature_age.ge(pd.Timedelta(0)) & feature_age.le(
        pd.Timedelta(days=max_age_days)
    )
    for column in RELEASE_FEATURES:
        result.loc[~fresh, column] = np.nan
        result[f"rank_{column}"] = np.nan
    # One issuer contributes at most one security to each cross section.  The
    # representative is chosen before seeing the label: highest lagged ADV,
    # then ticker as a deterministic tie-break.
    result["issuer_representative"] = True
    has_issuer = result.get("issuer_cik", pd.Series(index=result.index)).notna()
    if has_issuer.any():
        adv = pd.to_numeric(
            result.get("adv_usd_3m", pd.Series(np.nan, index=result.index)),
            errors="coerce",
        ).fillna(-np.inf)
        candidates = result.loc[has_issuer, ["signal_date", "ticker", "issuer_cik"]].assign(
            _adv=adv.loc[has_issuer]
        )
        selected = (
            candidates.sort_values(
                ["signal_date", "issuer_cik", "_adv", "ticker"],
                ascending=[True, True, False, True],
                kind="mergesort",
            )
            .drop_duplicates(["signal_date", "issuer_cik"], keep="first")
            .index
        )
        duplicate_issuer = has_issuer & ~result.index.isin(selected)
        result.loc[duplicate_issuer, "issuer_representative"] = False
        result.loc[duplicate_issuer, list(RELEASE_FEATURES)] = np.nan
    primary_components = ["earnings_change_on_assets", "sales_growth_yoy"]
    complete = result[primary_components].notna().all(axis=1)
    for column in primary_components:
        result.loc[complete, f"rank_{column}"] = result.loc[complete].groupby(
            "signal_date"
        )[column].rank(method="average", pct=True)
    gross_known = result["gross_margin_change_yoy"].notna()
    result.loc[gross_known, "rank_gross_margin_change_yoy"] = result.loc[
        gross_known
    ].groupby("signal_date")["gross_margin_change_yoy"].rank(
        method="average", pct=True
    )
    result["quarterly_fundamental_momentum_score"] = np.nan
    result.loc[complete, "quarterly_fundamental_momentum_score"] = result.loc[
        complete,
        ["rank_earnings_change_on_assets", "rank_sales_growth_yoy"],
    ].mean(axis=1)
    result["quarterly_feature_eligible"] = complete
    return result.drop(columns="_row_id").reset_index(drop=True)


__all__ = [
    "DURATION_TAGS",
    "INSTANT_TAGS",
    "RELEASE_FEATURES",
    "align_quarterly_features",
    "extract_quarterly_release_features",
    "filing_day_end_utc",
]
