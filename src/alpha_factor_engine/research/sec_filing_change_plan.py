"""Pre-outcome acquisition planning for the SEC filing-change experiment."""

from __future__ import annotations

from typing import Any

import pandas as pd

from alpha_factor_engine.contracts import validate_unique


class FilingChangePlanError(ValueError):
    """Raised when a header-only plan is ambiguous or incomplete."""


def attach_current_proxy_mapping(
    inventory: pd.DataFrame, universe: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach the frozen current-ticker proxy by CIK and expose all misses."""

    required_inventory = {
        "issuer_cik",
        "accession",
        "form",
        "filing_date",
        "report_date",
        "acceptance_at",
        "acceptance_at_raw",
        "primary_document",
        "primary_document_url",
    }
    required_universe = {
        "cik",
        "ticker",
        "proxy_universe_rank",
        "selection_as_of",
        "evidence_status",
    }
    missing_inventory = required_inventory.difference(inventory.columns)
    missing_universe = required_universe.difference(universe.columns)
    if missing_inventory:
        raise FilingChangePlanError(
            f"inventory missing columns: {sorted(missing_inventory)}"
        )
    if missing_universe:
        raise FilingChangePlanError(
            f"universe missing columns: {sorted(missing_universe)}"
        )
    mapping = universe.loc[:, sorted(required_universe)].copy()
    mapping["issuer_cik"] = (
        pd.to_numeric(mapping["cik"], errors="raise").astype("int64").map(lambda x: f"{x:010d}")
    )
    mapping["ticker"] = mapping["ticker"].astype("string").str.strip().str.upper()
    validate_unique(mapping, ["issuer_cik"], "current proxy CIK mapping")
    validate_unique(mapping, ["ticker"], "current proxy ticker mapping")
    if not mapping["evidence_status"].eq("current_survivor_proxy").all():
        raise FilingChangePlanError("universe is not uniformly labelled current_survivor_proxy")
    if mapping["selection_as_of"].nunique() != 1:
        raise FilingChangePlanError("universe has multiple selection_as_of dates")
    merged = inventory.merge(
        mapping.drop(columns=["cik"]), on="issuer_cik", how="left", validate="many_to_one"
    )
    misses = merged.loc[merged["ticker"].isna()].copy()
    attached = merged.loc[merged["ticker"].notna()].copy()
    return attached.reset_index(drop=True), misses.reset_index(drop=True)


def build_rough_prior_year_pairs(
    mapped_inventory: pd.DataFrame,
    *,
    signal_start: str | pd.Timestamp = "2019-01-01",
    signal_end: str | pd.Timestamp = "2025-12-31",
    minimum_gap_days: int = 300,
    maximum_gap_days: int = 430,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a header-only download superset of possible strict fiscal pairs.

    The result is an acquisition plan, not a tradable signal.  Final pairing
    must use DEI fiscal identities extracted from the downloaded documents.
    """

    required = {
        "issuer_cik",
        "ticker",
        "proxy_universe_rank",
        "accession",
        "form",
        "filing_date",
        "report_date",
        "acceptance_at",
        "acceptance_at_raw",
        "primary_document",
        "primary_document_url",
    }
    missing = required.difference(mapped_inventory.columns)
    if missing:
        raise FilingChangePlanError(f"mapped inventory missing: {sorted(missing)}")
    if minimum_gap_days >= maximum_gap_days:
        raise FilingChangePlanError("invalid report-date gap range")
    frame = mapped_inventory.copy().reset_index(drop=True)
    for column in ("filing_date", "report_date"):
        frame[column] = pd.to_datetime(frame[column], errors="raise").dt.normalize()
    frame["acceptance_at"] = pd.to_datetime(frame["acceptance_at"], utc=True, errors="raise")
    validate_unique(frame, ["accession"], "mapped periodic headers")
    start = pd.Timestamp(signal_start)
    end = pd.Timestamp(signal_end)
    if start > end:
        raise FilingChangePlanError("signal_start follows signal_end")
    possible: list[dict[str, Any]] = []
    no_pair: list[dict[str, Any]] = []
    for (issuer_cik, form), group in frame.groupby(["issuer_cik", "form"], sort=True):
        ordered = group.sort_values(
            ["report_date", "acceptance_at", "accession"], kind="mergesort"
        )
        for current in ordered.itertuples(index=False):
            if current.filing_date < start or current.filing_date > end:
                continue
            candidates = ordered.loc[
                ordered["acceptance_at"].lt(current.acceptance_at)
                & ordered["report_date"].lt(current.report_date)
            ].copy()
            candidates["report_gap_days"] = (
                current.report_date - candidates["report_date"]
            ).dt.days
            candidates = candidates.loc[
                candidates["report_gap_days"].between(
                    minimum_gap_days, maximum_gap_days, inclusive="both"
                )
            ]
            if candidates.empty:
                no_pair.append(
                    {
                        "issuer_cik": issuer_cik,
                        "ticker": current.ticker,
                        "form": form,
                        "current_accession": current.accession,
                        "current_filing_date": current.filing_date,
                        "reason": "no_header_level_prior_year_candidate",
                    }
                )
                continue
            candidates["distance_from_365"] = (
                candidates["report_gap_days"] - 365
            ).abs()
            candidates = candidates.sort_values(
                ["distance_from_365", "report_date", "acceptance_at", "accession"],
                ascending=[True, False, False, True],
                kind="mergesort",
            )
            for candidate_rank, (_, prior) in enumerate(candidates.iterrows(), start=1):
                possible.append(
                    {
                        "issuer_cik": issuer_cik,
                        "ticker": current.ticker,
                        "proxy_universe_rank": int(current.proxy_universe_rank),
                        "form": form,
                        "current_accession": current.accession,
                        "prior_accession": prior["accession"],
                        "current_filing_date": current.filing_date,
                        "prior_filing_date": prior["filing_date"],
                        "current_report_date": current.report_date,
                        "prior_report_date": prior["report_date"],
                        "current_acceptance_at": current.acceptance_at,
                        "prior_acceptance_at": prior["acceptance_at"],
                        "report_gap_days": int(prior["report_gap_days"]),
                        "rough_prior_candidates": int(len(candidates)),
                        "rough_candidate_rank": int(candidate_rank),
                        "pairing_status": "HEADER_ONLY_ACQUISITION_SUPERSET_NOT_SIGNAL",
                    }
                )
    pairs = pd.DataFrame(possible)
    if not pairs.empty:
        validate_unique(
            pairs,
            ["current_accession", "prior_accession"],
            "rough current-prior candidate pairs",
        )
        pairs = pairs.sort_values(
            ["current_acceptance_at", "issuer_cik", "current_accession"],
            kind="mergesort",
        ).reset_index(drop=True)
    return pairs, pd.DataFrame(no_pair)


def acquisition_documents(
    mapped_inventory: pd.DataFrame, pairs: pd.DataFrame
) -> pd.DataFrame:
    """Return every unique current/prior primary document needed by rough pairs."""

    if pairs.empty:
        return mapped_inventory.iloc[0:0].copy()
    accessions = set(pairs["current_accession"]) | set(pairs["prior_accession"])
    documents = mapped_inventory.loc[mapped_inventory["accession"].isin(accessions)].copy()
    missing = accessions.difference(documents["accession"])
    if missing:
        raise FilingChangePlanError(f"pair documents absent from inventory: {sorted(missing)[:5]}")
    validate_unique(documents, ["accession"], "acquisition documents")
    documents["document_role"] = "prior_only"
    documents.loc[documents["accession"].isin(set(pairs["current_accession"])), "document_role"] = (
        "current_or_reused_prior"
    )
    return documents.sort_values(
        ["proxy_universe_rank", "acceptance_at", "accession"], kind="mergesort"
    ).reset_index(drop=True)


def assign_pair_download_phases(
    pairs: pd.DataFrame,
    *,
    development_release_month_end: str,
    confirmation_release_month_end: str,
    smoke_rank_maximum: int = 25,
) -> pd.DataFrame:
    """Assign acquisition phases on the same New York release-month clock."""

    required = {
        "current_accession",
        "current_acceptance_at",
        "proxy_universe_rank",
    }
    missing = required.difference(pairs.columns)
    if missing:
        raise FilingChangePlanError(f"rough pairs missing phase columns: {sorted(missing)}")
    frame = pairs.copy()
    frame["current_acceptance_at"] = pd.to_datetime(
        frame["current_acceptance_at"], utc=True, errors="raise"
    )
    consistency = frame.groupby("current_accession", sort=False).agg(
        acceptance_values=("current_acceptance_at", "nunique"),
        rank_values=("proxy_universe_rank", "nunique"),
    )
    if (consistency[["acceptance_values", "rank_values"]] != 1).any().any():
        raise FilingChangePlanError(
            "rough candidates disagree on current filing phase metadata"
        )
    local = frame["current_acceptance_at"].dt.tz_convert("America/New_York")
    frame["current_release_month"] = local.dt.tz_localize(None).dt.to_period("M")
    development_end = pd.Period(development_release_month_end, freq="M")
    confirmation_end = pd.Period(confirmation_release_month_end, freq="M")
    if development_end >= confirmation_end:
        raise FilingChangePlanError("download phase release-month boundaries are invalid")
    rank = pd.to_numeric(frame["proxy_universe_rank"], errors="raise").astype(int)
    frame["download_phase"] = "20_sealed_2025_signal_only"
    frame.loc[
        frame["current_release_month"].le(confirmation_end), "download_phase"
    ] = "10_full_development_and_confirmation"
    frame.loc[
        rank.le(smoke_rank_maximum)
        & frame["current_release_month"].le(development_end),
        "download_phase",
    ] = "00_top25_engineering_smoke"
    phase_counts = frame.groupby("current_accession")["download_phase"].nunique()
    if not phase_counts.eq(1).all():
        raise FilingChangePlanError("one current filing maps to multiple download phases")
    return frame


__all__ = [
    "FilingChangePlanError",
    "acquisition_documents",
    "attach_current_proxy_mapping",
    "assign_pair_download_phases",
    "build_rough_prior_year_pairs",
]
