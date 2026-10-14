"""Strict cached SEC Form 8-K event inventory for one exact item token."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from alpha_factor_engine.data.sec_periodic_filings import (
    ACCESSION_PATTERN,
    PeriodicFilingDataError,
    _canonical_cik,
    _declared_history_sources,
    _filter_date,
    _header_table,
    _issuer_main_paths,
    _parse_iso_date,
    _read_gzip_json,
    validate_acceptance_at,
)


REQUIRED_COLUMNS = (
    "accessionNumber",
    "filingDate",
    "reportDate",
    "acceptanceDateTime",
    "form",
    "items",
    "primaryDocument",
)


def _item_tokens(value: Any) -> set[str]:
    if not isinstance(value, str):
        return set()
    return {part.strip() for part in value.split(",") if part.strip()}


def _source_rows(
    source: Any,
    *,
    issuer_cik: str,
    required_item: str,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> list[dict[str, Any]]:
    table = _header_table(source.payload, source=source.path)
    missing = [column for column in REQUIRED_COLUMNS if column not in table]
    if missing:
        raise PeriodicFilingDataError(f"SEC 8-K columns missing in {source.path}: {missing}")
    if any(not isinstance(table[column], list) for column in REQUIRED_COLUMNS):
        raise PeriodicFilingDataError(f"SEC 8-K columns are not lists in {source.path}")
    lengths = {column: len(table[column]) for column in REQUIRED_COLUMNS}
    if len(set(lengths.values())) != 1:
        raise PeriodicFilingDataError(f"SEC 8-K column lengths differ in {source.path}: {lengths}")
    rows: list[dict[str, Any]] = []
    for index in range(next(iter(lengths.values()), 0)):
        if table["form"][index] != "8-K" or required_item not in _item_tokens(table["items"][index]):
            continue
        accession = table["accessionNumber"][index]
        if not isinstance(accession, str) or not ACCESSION_PATTERN.fullmatch(accession):
            raise PeriodicFilingDataError(f"invalid 8-K accession in {source.path}: {accession!r}")
        filing_date = _parse_iso_date(table["filingDate"][index], field="filingDate", source=source.path)
        if start is not None and filing_date < start:
            continue
        if end is not None and filing_date > end:
            continue
        report_raw = table["reportDate"][index]
        report_date = None if report_raw == "" else _parse_iso_date(report_raw, field="reportDate", source=source.path)
        acceptance_raw = table["acceptanceDateTime"][index]
        acceptance_at = validate_acceptance_at(acceptance_raw, source=source.path)
        acceptance_et_date = acceptance_at.tz_convert("America/New_York").tz_localize(None).normalize()
        rows.append(
            {
                "issuer_cik": issuer_cik,
                "accession": accession,
                "form": "8-K",
                "required_item": required_item,
                "items": str(table["items"][index]),
                "filing_date": filing_date,
                "report_date": report_date,
                "acceptance_at_raw": acceptance_raw,
                "acceptance_at": acceptance_at,
                "acceptance_et_date": acceptance_et_date,
                "filing_acceptance_et_date_delta_days": int((filing_date - acceptance_et_date).days),
                "primary_document": str(table["primaryDocument"][index]),
                "source_kind": source.source_kind,
                "source_path": source.path.as_posix(),
                "source_file_sha256": source.file_sha256,
                "source_content_sha256": source.content_sha256,
            }
        )
    return rows


def load_item_inventory(
    cache_dir: str | Path,
    *,
    required_item: str,
    issuer_ciks: Iterable[str | int] | None = None,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    allowed_source_paths: Iterable[str | Path] | None = None,
) -> pd.DataFrame:
    """Load non-amended 8-K headers containing one exact item token."""

    if not required_item or "," in required_item or required_item.strip() != required_item:
        raise PeriodicFilingDataError("required_item must be one non-empty exact item token")
    root = Path(cache_dir)
    allowed = None if allowed_source_paths is None else {Path(path).resolve() for path in allowed_source_paths}
    if allowed_source_paths is not None and not allowed:
        raise PeriodicFilingDataError("frozen SEC source manifest is empty")
    start = _filter_date(start_date, name="start_date")
    end = _filter_date(end_date, name="end_date")
    rows: list[dict[str, Any]] = []
    for issuer_cik, main_path in _issuer_main_paths(root, issuer_ciks):
        if allowed is not None and main_path.resolve() not in allowed:
            raise PeriodicFilingDataError(f"SEC main source is outside frozen manifest: {main_path}")
        main = _read_gzip_json(main_path, source_kind="recent")
        if _canonical_cik(main.payload.get("cik", "")) != issuer_cik:
            raise PeriodicFilingDataError(f"main submissions CIK conflict: {main_path}")
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
                _source_rows(
                    source,
                    issuer_cik=issuer_cik,
                    required_item=required_item,
                    start=start,
                    end=end,
                )
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    compare = ["issuer_cik", "required_item", "items", "filing_date", "acceptance_at_raw", "primary_document"]
    conflicts = [
        str(accession)
        for accession, group in frame.groupby("accession", sort=False)
        if len(group) > 1 and any(group[column].nunique(dropna=False) != 1 for column in compare)
    ]
    if conflicts:
        raise PeriodicFilingDataError(f"conflicting 8-K item accessions: {conflicts[:5]}")
    frame = frame.sort_values(["accession", "source_path"], kind="mergesort").drop_duplicates("accession", keep="first")
    frame["filing_date"] = pd.to_datetime(frame["filing_date"])
    frame["report_date"] = pd.to_datetime(frame["report_date"])
    frame["acceptance_at"] = pd.to_datetime(frame["acceptance_at"], utc=True)
    frame["acceptance_et_date"] = pd.to_datetime(frame["acceptance_et_date"])
    return frame.sort_values(["acceptance_at", "accession"], kind="mergesort").reset_index(drop=True)


__all__ = ["load_item_inventory"]
