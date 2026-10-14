"""Point-in-time quarterly fundamental-momentum research primitives.

The extractor uses only three-month facts reported inside a 10-Q accession.  A
year-over-year comparator must appear in that same filing, so a later restated
Company Facts value cannot silently rewrite an earlier signal.  When the exact
SEC acceptance timestamp is unavailable, the observation becomes usable only
after the end of the SEC filing date in New York.
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


def _unit_entries(
    us_gaap: Mapping[str, Any], tag: str, unit: str = "USD"
) -> list[dict[str, Any]]:
    concept = us_gaap.get(tag) or {}
    entries = (concept.get("units") or {}).get(unit) or []
    return [dict(item) for item in entries]


def _sec_date(value: object) -> pd.Timestamp | None:
    """Parse SEC's ISO calendar dates without pandas' costly scalar inference."""

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


def _duration_candidates(
    us_gaap: Mapping[str, Any], ticker: str, cik: int
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for canonical, tags in DURATION_TAGS.items():
        for priority, tag in enumerate(tags):
            for item in _unit_entries(us_gaap, tag):
                if item.get("form") not in {"10-Q", "10-Q/A"}:
                    continue
                if item.get("fp") not in {"Q1", "Q2", "Q3"}:
                    continue
                start = _sec_date(item.get("start"))
                end = _sec_date(item.get("end"))
                filed = _sec_date(item.get("filed"))
                value = _finite_float(item.get("val"))
                accession = str(item.get("accn") or "")
                if start is None or end is None or filed is None or value is None or not accession:
                    continue
                # A completed accounting period cannot end after its filing date.
                # Company Facts occasionally carries malformed duplicate contexts.
                if end > filed:
                    continue
                duration_days = int((end - start).days)
                if not 70 <= duration_days <= 120:
                    continue
                rows.append(
                    {
                        "ticker": ticker,
                        "cik": int(cik),
                        "accession": accession,
                        "form": str(item.get("form")),
                        "fiscal_year": item.get("fy"),
                        "fiscal_period": str(item.get("fp")),
                        "filed": filed,
                        "start": start,
                        "end": end,
                        "duration_days": duration_days,
                        "concept": canonical,
                        "source_tag": tag,
                        "tag_priority": int(priority),
                        "value": float(value),
                    }
                )
    return pd.DataFrame(rows)


def _instant_candidates(
    us_gaap: Mapping[str, Any], ticker: str, cik: int
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for canonical, tags in INSTANT_TAGS.items():
        for priority, tag in enumerate(tags):
            for item in _unit_entries(us_gaap, tag):
                if item.get("form") not in {"10-Q", "10-Q/A"}:
                    continue
                end = _sec_date(item.get("end"))
                filed = _sec_date(item.get("filed"))
                value = _finite_float(item.get("val"))
                accession = str(item.get("accn") or "")
                if end is None or filed is None or value is None or not accession:
                    continue
                if end > filed:
                    continue
                rows.append(
                    {
                        "ticker": ticker,
                        "cik": int(cik),
                        "accession": accession,
                        "end": end,
                        "concept": canonical,
                        "source_tag": tag,
                        "tag_priority": int(priority),
                        "value": float(value),
                    }
                )
    return pd.DataFrame(rows)


def _one_fact(candidates: pd.DataFrame) -> pd.Series | None:
    if candidates.empty:
        return None
    ordered = candidates.sort_values(
        ["tag_priority", "source_tag", "start"], kind="mergesort"
    )
    return ordered.iloc[0]


def extract_quarterly_release_features(
    ticker: str,
    cik: int,
    payload: Mapping[str, Any],
    acceptance: Mapping[str, pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Extract same-filing quarterly changes with an auditable public clock."""

    acceptance = acceptance or {}
    us_gaap = ((payload.get("facts") or {}).get("us-gaap") or {})
    duration = _duration_candidates(us_gaap, ticker, cik)
    instant = _instant_candidates(us_gaap, ticker, cik)
    if duration.empty:
        return pd.DataFrame()

    output: list[dict[str, object]] = []
    for accession, filing_rows in duration.groupby("accession", sort=True):
        release_end = pd.Timestamp(filing_rows["end"].max())
        current_scope = filing_rows.loc[
            filing_rows["end"].between(
                release_end - pd.Timedelta(days=7),
                release_end + pd.Timedelta(days=7),
            )
        ]
        if current_scope.empty:
            continue
        anchor = current_scope.sort_values(
            ["end", "tag_priority"], ascending=[False, True], kind="mergesort"
        ).iloc[0]
        record: dict[str, object] = {
            "ticker": ticker,
            "cik": int(cik),
            "accession": accession,
            "form": anchor["form"],
            "fiscal_year": anchor["fiscal_year"],
            "fiscal_period": anchor["fiscal_period"],
            "fiscal_end": release_end,
            "filed": pd.Timestamp(anchor["filed"]),
        }
        for concept in DURATION_TAGS:
            concept_rows = filing_rows.loc[filing_rows["concept"].eq(concept)]
            current = _one_fact(
                concept_rows.loc[
                    concept_rows["end"].between(
                        release_end - pd.Timedelta(days=7),
                        release_end + pd.Timedelta(days=7),
                    )
                ]
            )
            if current is None:
                record[f"{concept}_current"] = np.nan
                record[f"{concept}_prior_year"] = np.nan
                continue
            prior_scope = concept_rows.copy()
            prior_scope["year_gap"] = (
                release_end - prior_scope["end"]
            ).dt.days
            prior_scope = prior_scope.loc[prior_scope["year_gap"].between(250, 470)]
            if not prior_scope.empty:
                prior_scope = prior_scope.assign(
                    gap_error=(prior_scope["year_gap"] - 365).abs()
                ).sort_values(
                    ["gap_error", "tag_priority", "end"],
                    ascending=[True, True, False],
                    kind="mergesort",
                )
                prior = prior_scope.iloc[0]
            else:
                prior = None
            record[f"{concept}_current"] = float(current["value"])
            record[f"{concept}_prior_year"] = (
                float(prior["value"]) if prior is not None else np.nan
            )
        asset_scope = instant.loc[instant["accession"].eq(accession)].copy()
        if not asset_scope.empty:
            asset_scope["end_gap"] = (asset_scope["end"] - release_end).abs().dt.days
            asset_scope = asset_scope.loc[asset_scope["end_gap"].le(31)].sort_values(
                ["end_gap", "tag_priority", "end"], kind="mergesort"
            )
        asset = asset_scope.iloc[0] if not asset_scope.empty else None
        record["assets_current"] = float(asset["value"]) if asset is not None else np.nan

        exact_clock = acceptance.get(accession)
        if exact_clock is not None and pd.notna(exact_clock):
            available_at = pd.Timestamp(exact_clock)
            if available_at.tzinfo is None:
                available_at = available_at.tz_localize("UTC")
            else:
                available_at = available_at.tz_convert("UTC")
            clock_source = "submissions_acceptance_datetime"
        else:
            available_at = filing_day_end_utc(record["filed"])
            clock_source = "conservative_filing_day_end_et"
        if available_at <= pd.Timestamp(release_end).tz_localize("UTC"):
            raise ValueError(
                "quarterly fact availability does not follow fiscal end: "
                f"ticker={ticker} accession={accession} fiscal_end={release_end} "
                f"available_at={available_at} clock_source={clock_source}"
            )
        record["available_at"] = available_at
        record["clock_source"] = clock_source

        assets = float(record["assets_current"])
        ni_now = float(record["net_income_current"])
        ni_prior = float(record["net_income_prior_year"])
        rev_now = float(record["revenue_current"])
        rev_prior = float(record["revenue_prior_year"])
        gp_now = float(record["gross_profit_current"])
        gp_prior = float(record["gross_profit_prior_year"])
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
        output.append(record)

    if not output:
        return pd.DataFrame()
    frame = pd.DataFrame(output)
    frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True)
    frame = frame.sort_values(
        ["ticker", "available_at", "accession"], kind="mergesort"
    ).drop_duplicates(["ticker", "accession"], keep="last")
    return frame.reset_index(drop=True)


def align_quarterly_features(
    panel: pd.DataFrame,
    releases: pd.DataFrame,
    *,
    max_age_days: int = 180,
) -> pd.DataFrame:
    """As-of align releases and construct a label-independent composite score."""

    required_panel = {"ticker", "signal_date"}
    if missing := sorted(required_panel.difference(panel.columns)):
        raise ValueError(f"monthly panel missing columns: {missing}")
    required_releases = {"ticker", "available_at", "accession", *RELEASE_FEATURES}
    if missing := sorted(required_releases.difference(releases.columns)):
        raise ValueError(f"quarterly releases missing columns: {missing}")
    left = panel.copy()
    left["signal_date"] = pd.to_datetime(left["signal_date"], utc=True, errors="raise")
    left["_row_id"] = np.arange(len(left), dtype=int)
    right = releases.copy()
    right["available_at"] = pd.to_datetime(
        right["available_at"], utc=True, errors="raise"
    )
    if right.duplicated(["ticker", "available_at", "accession"]).any():
        raise ValueError("quarterly releases contain duplicate keys")

    right = right.rename(columns={"available_at": "quarterly_available_at"})
    parts: list[pd.DataFrame] = []
    release_columns = [
        column for column in right.columns if column not in {"ticker", "cik"}
    ]
    for ticker, decisions in left.groupby("ticker", sort=True):
        company = right.loc[right["ticker"].eq(ticker), release_columns].sort_values(
            ["quarterly_available_at", "accession"], kind="mergesort"
        )
        decisions = decisions.sort_values("signal_date", kind="mergesort")
        if company.empty:
            merged = decisions.copy()
        else:
            merged = pd.merge_asof(
                decisions,
                company,
                left_on="signal_date",
                right_on="quarterly_available_at",
                direction="backward",
                allow_exact_matches=True,
            )
        parts.append(merged)
    result = pd.concat(parts, ignore_index=True).sort_values("_row_id")
    if "quarterly_available_at" not in result:
        result["quarterly_available_at"] = pd.NaT
    for column in RELEASE_FEATURES:
        if column not in result:
            result[column] = np.nan
    # Concatenating companies with and without releases can otherwise coerce
    # the all-NaT branch to a timezone-naive dtype.
    result["quarterly_available_at"] = pd.to_datetime(
        result["quarterly_available_at"], utc=True, errors="coerce"
    )
    result["signal_date"] = pd.to_datetime(
        result["signal_date"], utc=True, errors="raise"
    )
    known = result["quarterly_available_at"].notna()
    if (
        result.loc[known, "quarterly_available_at"]
        > result.loc[known, "signal_date"]
    ).any():
        raise ValueError("future quarterly filing leaked into a monthly decision")
    result["quarterly_feature_age_days"] = (
        result["signal_date"] - result["quarterly_available_at"]
    ).dt.total_seconds() / 86_400
    fresh = known & result["quarterly_feature_age_days"].between(0, max_age_days)
    for column in RELEASE_FEATURES:
        result.loc[~fresh, column] = np.nan
        result[f"rank_{column}"] = result.groupby("signal_date")[column].rank(
            method="average", pct=True
        )
    primary_available = result[
        ["rank_earnings_change_on_assets", "rank_sales_growth_yoy"]
    ].notna().all(axis=1)
    result["quarterly_fundamental_momentum_score"] = np.nan
    result.loc[primary_available, "quarterly_fundamental_momentum_score"] = result.loc[
        primary_available,
        ["rank_earnings_change_on_assets", "rank_sales_growth_yoy"],
    ].mean(axis=1)
    result["quarterly_feature_eligible"] = primary_available
    return result.drop(columns="_row_id").reset_index(drop=True)


__all__ = [
    "DURATION_TAGS",
    "INSTANT_TAGS",
    "RELEASE_FEATURES",
    "align_quarterly_features",
    "extract_quarterly_release_features",
    "filing_day_end_utc",
]
