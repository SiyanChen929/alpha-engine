"""Pre-outcome planning helpers for exact SEC late-filing notifications.

This module reads only SEC submissions headers and frozen universe metadata.  It
does not read prices, construct returns, or inspect any outcome.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from alpha_factor_engine.contracts import validate_unique
from alpha_factor_engine.data.sec_periodic_filings import (
    ACCESSION_PATTERN,
    INVENTORY_COLUMNS,
    REQUIRED_HEADER_COLUMNS,
    PeriodicFilingDataError,
    _canonical_cik,
    _declared_history_sources,
    _filter_date,
    _header_table,
    _issuer_main_paths,
    _parse_iso_date,
    _read_gzip_json,
    build_primary_document_url,
    validate_acceptance_at,
    validate_primary_document,
)
from alpha_factor_engine.research.sec_filing_change_plan import (
    attach_current_proxy_mapping,
)


LATE_FORMS = frozenset({"NT 10-K", "NT 10-Q"})


class LateFilingPlanError(ValueError):
    """Raised when a late-filing pre-outcome plan is incomplete or ambiguous."""


def _late_source_rows(
    source: Any,
    *,
    issuer_cik: str,
    start_date: pd.Timestamp | None,
    end_date: pd.Timestamp | None,
) -> list[dict[str, Any]]:
    table = _header_table(source.payload, source=source.path)
    missing = [column for column in REQUIRED_HEADER_COLUMNS if column not in table]
    if missing:
        raise PeriodicFilingDataError(
            f"SEC submissions columns missing in {source.path}: {missing}"
        )
    if any(not isinstance(table[column], list) for column in REQUIRED_HEADER_COLUMNS):
        raise PeriodicFilingDataError(
            f"SEC submissions required columns are not lists in {source.path}"
        )
    lengths = {column: len(table[column]) for column in REQUIRED_HEADER_COLUMNS}
    if len(set(lengths.values())) != 1:
        raise PeriodicFilingDataError(
            f"SEC submissions column lengths differ in {source.path}: {lengths}"
        )

    rows: list[dict[str, Any]] = []
    for index in range(next(iter(lengths.values()), 0)):
        form = table["form"][index]
        if form not in LATE_FORMS:
            continue
        accession = table["accessionNumber"][index]
        if not isinstance(accession, str) or not ACCESSION_PATTERN.fullmatch(accession):
            raise PeriodicFilingDataError(
                f"invalid accession in {source.path}: {accession!r}"
            )
        filing_date = _parse_iso_date(
            table["filingDate"][index], field="filingDate", source=source.path
        )
        if start_date is not None and filing_date < start_date:
            continue
        if end_date is not None and filing_date > end_date:
            continue
        report_date = _parse_iso_date(
            table["reportDate"][index], field="reportDate", source=source.path
        )
        if report_date > filing_date:
            raise PeriodicFilingDataError(
                f"reportDate follows filingDate for {accession} in {source.path}"
            )
        acceptance_raw = table["acceptanceDateTime"][index]
        acceptance_at = validate_acceptance_at(acceptance_raw, source=source.path)
        primary_document = validate_primary_document(table["primaryDocument"][index])
        rows.append(
            {
                "issuer_cik": issuer_cik,
                "accession": accession,
                "form": form,
                "filing_date": filing_date,
                "report_date": report_date,
                "acceptance_at_raw": acceptance_raw,
                "acceptance_at": acceptance_at,
                "primary_document": primary_document,
                "primary_document_url": build_primary_document_url(
                    issuer_cik=issuer_cik,
                    accession=accession,
                    primary_document=primary_document,
                ),
                "source_kind": source.source_kind,
                "source_path": source.path.as_posix(),
                "source_file_sha256": source.file_sha256,
                "source_content_sha256": source.content_sha256,
            }
        )
    return rows


def load_late_filing_inventory(
    cache_dir: str | Path,
    *,
    issuer_ciks: Iterable[str | int],
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    allowed_source_paths: Iterable[str | Path],
) -> pd.DataFrame:
    """Load exact NT 10-K/NT 10-Q headers from a frozen submissions snapshot."""

    root = Path(cache_dir)
    allowed = {Path(path).resolve() for path in allowed_source_paths}
    if not allowed:
        raise LateFilingPlanError("frozen SEC source manifest is empty")
    start = _filter_date(start_date, name="start_date")
    end = _filter_date(end_date, name="end_date")
    if start is None or end is None or start > end:
        raise LateFilingPlanError("late-filing date range is invalid")

    rows: list[dict[str, Any]] = []
    normalized_ciks = [_canonical_cik(value) for value in issuer_ciks]
    if len(normalized_ciks) != len(set(normalized_ciks)):
        raise LateFilingPlanError("issuer CIK list contains duplicates")
    for issuer_cik, main_path in _issuer_main_paths(root, normalized_ciks):
        if main_path.resolve() not in allowed:
            raise LateFilingPlanError(
                f"SEC main submissions file is outside the frozen manifest: {main_path}"
            )
        main = _read_gzip_json(main_path, source_kind="recent")
        if _canonical_cik(main.payload.get("cik", "")) != issuer_cik:
            raise LateFilingPlanError(
                f"main submissions CIK conflicts with filename: {main_path}"
            )
        sources = [
            main,
            *_declared_history_sources(
                main,
                cache_root=root,
                issuer_cik=issuer_cik,
                start_date=start,
                end_date=end,
                allowed_source_paths=allowed,
            ),
        ]
        for source in sources:
            rows.extend(
                _late_source_rows(
                    source,
                    issuer_cik=issuer_cik,
                    start_date=start,
                    end_date=end,
                )
            )

    frame = pd.DataFrame(rows, columns=INVENTORY_COLUMNS)
    if frame.empty:
        return frame
    frame["filing_date"] = pd.to_datetime(frame["filing_date"], errors="raise")
    frame["report_date"] = pd.to_datetime(frame["report_date"], errors="raise")
    frame["acceptance_at"] = pd.to_datetime(
        frame["acceptance_at"], utc=True, errors="raise"
    )
    compare = [
        "issuer_cik",
        "form",
        "filing_date",
        "report_date",
        "acceptance_at_raw",
        "acceptance_at",
        "primary_document",
        "primary_document_url",
    ]
    conflicts: list[str] = []
    for accession, group in frame.groupby("accession", sort=False):
        if len(group) > 1 and any(group[column].nunique(dropna=False) != 1 for column in compare):
            conflicts.append(str(accession))
    if conflicts:
        raise LateFilingPlanError(
            f"conflicting late-filing headers for accessions: {conflicts[:5]}"
        )
    frame = (
        frame.sort_values(["accession", "source_path"], kind="mergesort")
        .drop_duplicates("accession", keep="first")
        .sort_values(["acceptance_at", "issuer_cik", "accession"], kind="mergesort")
        .reset_index(drop=True)
    )
    validate_unique(frame, ["accession"], "late-filing headers")
    return frame


def build_late_filing_candidate_plan(
    inventory: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    lag_bounds: Mapping[str, list[int] | tuple[int, int]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Map and freeze eligible events without accessing market outcomes."""

    mapped, mapping_misses = attach_current_proxy_mapping(inventory, universe)
    if not mapping_misses.empty:
        raise LateFilingPlanError("frozen CIK universe failed to map every late filing")
    frame = mapped.copy()
    frame["filing_lag_days"] = (
        frame["filing_date"] - frame["report_date"]
    ).dt.days.astype(int)
    local = frame["acceptance_at"].dt.tz_convert("America/New_York")
    frame["release_year_et"] = local.dt.year.astype(int)
    frame["release_month_et"] = local.dt.tz_localize(None).dt.to_period("M")
    frame["phase"] = ""
    frame.loc[frame["release_year_et"].between(2018, 2021), "phase"] = "development"
    frame.loc[frame["release_year_et"].between(2022, 2024), "phase"] = "confirmation"
    frame.loc[frame["release_year_et"].eq(2025), "phase"] = "sealed_2025"
    if frame["phase"].eq("").any():
        raise LateFilingPlanError("late-filing release year falls outside frozen cohorts")

    reasons: list[str] = []
    eligible: list[bool] = []
    for row in frame.itertuples(index=False):
        bounds = lag_bounds.get(str(row.form))
        if bounds is None or len(bounds) != 2:
            raise LateFilingPlanError(f"missing lag bounds for {row.form}")
        lower, upper = int(bounds[0]), int(bounds[1])
        if lower >= upper:
            raise LateFilingPlanError(f"invalid lag bounds for {row.form}")
        keep = lower <= int(row.filing_lag_days) <= upper
        eligible.append(keep)
        reasons.append("eligible" if keep else "report_lag_outside_frozen_form_bounds")
    frame["eligible"] = eligible
    frame["eligibility_reason"] = reasons
    frame["signal_direction"] = "short_issuer_long_spy"
    frame["outcome_join_performed"] = False
    frame["outcome_join_permitted"] = False

    candidates = frame.loc[frame["eligible"]].copy()
    exclusions = frame.loc[~frame["eligible"]].copy()
    validate_unique(candidates, ["accession"], "eligible late-filing candidates")
    return (
        candidates.reset_index(drop=True),
        exclusions.reset_index(drop=True),
        mapping_misses.reset_index(drop=True),
    )


__all__ = [
    "LATE_FORMS",
    "LateFilingPlanError",
    "build_late_filing_candidate_plan",
    "load_late_filing_inventory",
]
