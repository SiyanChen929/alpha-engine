"""Fail-closed SEC Form 3/4/5 bulk-data ingestion helpers.

The quarterly DERA data are useful for discovering and flattening ownership
filings, but they do not contain the intraday EDGAR acceptance timestamp.  A
historical event is therefore not research-eligible until its accession has
been joined to ``acceptanceDateTime`` from the SEC submissions history.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import re
import time
import zipfile
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping
from urllib.parse import urljoin

import pandas as pd
import requests

from alpha_factor_engine.contracts import (
    DatasetRecord,
    PointInTimeViolation,
    sha256_bytes,
    utc_now,
    validate_unique,
)


INSIDER_DATASET_PAGE = (
    "https://www.sec.gov/data-research/sec-markets-data/"
    "insider-transactions-data-sets"
)
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_ROOT = "https://www.sec.gov/Archives/edgar/data"
DAILY_MASTER_URL = (
    "https://www.sec.gov/Archives/edgar/daily-index/"
    "{year}/QTR{quarter}/master.{yyyymmdd}.idx"
)

REQUIRED_TABLE_FILES = {
    "SUBMISSION.tsv",
    "REPORTINGOWNER.tsv",
    "NONDERIV_TRANS.tsv",
    "FOOTNOTES.tsv",
}
METADATA_FILE_ALIASES = (
    "FORM_345_metadata.json",
    "insider_transactions_metadata.json",
)
# Kept for callers that used the original public constant.  Archive validation
# below accepts exactly one versioned metadata filename, while every table in
# this set remains mandatory.
REQUIRED_FILES = REQUIRED_TABLE_FILES | {METADATA_FILE_ALIASES[0]}

TABLE_PRIMARY_KEYS: dict[str, tuple[str, ...]] = {
    "SUBMISSION.tsv": ("ACCESSION_NUMBER",),
    "REPORTINGOWNER.tsv": ("ACCESSION_NUMBER", "RPTOWNERCIK"),
    "NONDERIV_TRANS.tsv": ("ACCESSION_NUMBER", "NONDERIV_TRANS_SK"),
    "FOOTNOTES.tsv": ("ACCESSION_NUMBER", "FOOTNOTE_ID"),
}

REQUIRED_VALUE_COLUMNS: dict[str, tuple[str, ...]] = {
    "SUBMISSION.tsv": (
        "ACCESSION_NUMBER",
        "FILING_DATE",
        "DOCUMENT_TYPE",
        "ISSUERCIK",
    ),
    "REPORTINGOWNER.tsv": ("ACCESSION_NUMBER", "RPTOWNERCIK"),
    "NONDERIV_TRANS.tsv": ("ACCESSION_NUMBER", "NONDERIV_TRANS_SK"),
    "FOOTNOTES.tsv": ("ACCESSION_NUMBER", "FOOTNOTE_ID"),
}

REQUIRED_COLUMNS: dict[str, set[str]] = {
    "SUBMISSION.tsv": {
        "ACCESSION_NUMBER",
        "FILING_DATE",
        "PERIOD_OF_REPORT",
        "DATE_OF_ORIG_SUB",
        "DOCUMENT_TYPE",
        "ISSUERCIK",
        "ISSUERNAME",
        "ISSUERTRADINGSYMBOL",
    },
    "REPORTINGOWNER.tsv": {
        "ACCESSION_NUMBER",
        "RPTOWNERCIK",
        "RPTOWNERNAME",
        "RPTOWNER_RELATIONSHIP",
        "RPTOWNER_TITLE",
    },
    "NONDERIV_TRANS.tsv": {
        "ACCESSION_NUMBER",
        "NONDERIV_TRANS_SK",
        "SECURITY_TITLE",
        "TRANS_DATE",
        "TRANS_FORM_TYPE",
        "TRANS_CODE",
        "EQUITY_SWAP_INVOLVED",
        "TRANS_TIMELINESS",
        "TRANS_SHARES",
        "TRANS_PRICEPERSHARE",
        "TRANS_ACQUIRED_DISP_CD",
        "SHRS_OWND_FOLWNG_TRANS",
        "DIRECT_INDIRECT_OWNERSHIP",
    },
    "FOOTNOTES.tsv": {"ACCESSION_NUMBER", "FOOTNOTE_ID", "FOOTNOTE_TXT"},
}

DATE_COLUMNS: dict[str, tuple[str, ...]] = {
    "SUBMISSION.tsv": ("FILING_DATE", "PERIOD_OF_REPORT", "DATE_OF_ORIG_SUB"),
    "NONDERIV_TRANS.tsv": ("TRANS_DATE", "DEEMED_EXECUTION_DATE"),
}

NUMERIC_COLUMNS = (
    "TRANS_SHARES",
    "TRANS_PRICEPERSHARE",
    "SHRS_OWND_FOLWNG_TRANS",
    "VALU_OWND_FOLWNG_TRANS",
)

ACCESSION_PATTERN = re.compile(r"^\d{10}-\d{2}-\d{6}$")
QUARTER_PATTERN = re.compile(r"(?P<year>20\d{2})q(?P<quarter>[1-4])_form345\.zip", re.I)
HISTORY_FILE_PATTERN = re.compile(r"^CIK\d{10}-submissions-\d{3}\.json$")
ACCEPTANCE_HEADER_PATTERN = re.compile(r"<ACCEPTANCE-DATETIME>(\d{14})")


class InsiderDataSchemaError(ValueError):
    """Raised when a bulk archive or submissions payload violates its contract."""


class AcceptanceClockError(PointInTimeViolation):
    """Raised when an event does not have a defensible SEC availability time."""


@dataclass(frozen=True)
class InsiderQuarterTables:
    quarter: str
    submissions: pd.DataFrame
    reporting_owners: pd.DataFrame
    nonderivative_transactions: pd.DataFrame
    footnotes: pd.DataFrame
    metadata: dict[str, Any]
    metadata_filename: str
    data_quality_issues: tuple[dict[str, Any], ...]
    archive_sha256: str


@dataclass(frozen=True)
class DailyMasterSnapshot:
    index_date: pd.Timestamp
    filings: pd.DataFrame
    source_url: str
    last_modified_at: pd.Timestamp
    etag: str | None
    retrieved_at: str
    content_sha256: str


class _QuarterLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href and QUARTER_PATTERN.search(href):
            self.hrefs.append(href)


def _normalize_quarter(value: str) -> str:
    normalized = str(value).strip().lower().replace(" ", "")
    if not re.fullmatch(r"20\d{2}q[1-4]", normalized):
        raise ValueError(f"invalid SEC insider quarter: {value!r}")
    return normalized


def discover_quarter_links(html: str, *, base_url: str = INSIDER_DATASET_PAGE) -> dict[str, str]:
    """Extract official quarterly ZIP URLs without assuming a stable SEC directory."""

    parser = _QuarterLinkParser()
    parser.feed(html)
    result: dict[str, str] = {}
    for href in parser.hrefs:
        match = QUARTER_PATTERN.search(href)
        assert match is not None
        quarter = f"{match.group('year')}q{match.group('quarter')}"
        url = urljoin(base_url, href)
        prior = result.get(quarter)
        if prior is not None and prior != url:
            raise InsiderDataSchemaError(
                f"multiple official URLs found for {quarter}: {prior!r}, {url!r}"
            )
        result[quarter] = url
    if not result:
        raise InsiderDataSchemaError("SEC insider download page contains no quarterly ZIP links")
    return dict(sorted(result.items()))


def daily_master_url(index_date: str | pd.Timestamp) -> str:
    """Return the official archived daily master-index URL for one date."""

    stamp = pd.Timestamp(index_date)
    if pd.isna(stamp):
        raise ValueError("index_date must be valid")
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("America/New_York").tz_localize(None)
    stamp = stamp.normalize()
    quarter = (stamp.month - 1) // 3 + 1
    return DAILY_MASTER_URL.format(
        year=stamp.year,
        quarter=quarter,
        yyyymmdd=stamp.strftime("%Y%m%d"),
    )


def parse_daily_master_index(
    content: bytes,
    *,
    index_date: str | pd.Timestamp,
    source_url: str,
    last_modified_at: str | pd.Timestamp,
    etag: str | None = None,
    retrieved_at: str | None = None,
) -> DailyMasterSnapshot:
    """Parse one immutable SEC daily master snapshot with public-time metadata."""

    if not content:
        raise InsiderDataSchemaError("SEC daily master index is empty")
    raw = content
    if content[:2] == b"\x1f\x8b":
        try:
            raw = gzip.decompress(content)
        except OSError as exc:
            raise InsiderDataSchemaError("SEC daily master gzip is corrupt") from exc
    try:
        text = raw.decode("latin-1")
    except UnicodeDecodeError as exc:  # pragma: no cover - latin-1 is exhaustive
        raise InsiderDataSchemaError("SEC daily master index is not decodable") from exc

    stamp = pd.Timestamp(index_date)
    if pd.isna(stamp):
        raise ValueError("index_date must be valid")
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("America/New_York").tz_localize(None)
    stamp = stamp.normalize()
    modified = pd.Timestamp(last_modified_at)
    if pd.isna(modified):
        raise AcceptanceClockError("daily master is missing Last-Modified")
    if modified.tzinfo is None:
        raise AcceptanceClockError("daily master Last-Modified must include a timezone")
    modified = modified.tz_convert("UTC")

    rows: list[dict[str, Any]] = []
    data_started = False
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("CIK|"):
            data_started = True
            continue
        if not data_started:
            continue
        if set(stripped) == {"-"}:
            continue
        parts = stripped.split("|", maxsplit=4)
        if len(parts) != 5:
            raise InsiderDataSchemaError(
                f"daily master malformed row {line_number}: {stripped[:120]!r}"
            )
        cik, company_name, form, date_filed, filename = (value.strip() for value in parts)
        if not re.fullmatch(r"\d{1,10}", cik):
            raise InsiderDataSchemaError(
                f"daily master invalid CIK at row {line_number}: {cik!r}"
            )
        pure = PurePosixPath(filename)
        if pure.is_absolute() or ".." in pure.parts or "\\" in filename:
            raise InsiderDataSchemaError(
                f"daily master unsafe filename at row {line_number}: {filename!r}"
            )
        accession_match = re.fullmatch(
            r"(?P<accession>\d{10}-\d{2}-\d{6})\.txt", pure.name
        )
        if accession_match is None:
            raise InsiderDataSchemaError(
                f"daily master filename lacks accession at row {line_number}: {filename!r}"
            )
        path_parts = pure.parts
        if len(path_parts) < 4 or path_parts[0:2] != ("edgar", "data"):
            raise InsiderDataSchemaError(
                f"daily master has unexpected archive path at row {line_number}: {filename!r}"
            )
        archive_cik = path_parts[2]
        if not re.fullmatch(r"\d{1,10}", archive_cik):
            raise InsiderDataSchemaError(
                f"daily master has invalid archive CIK at row {line_number}: {archive_cik!r}"
            )
        filed = pd.to_datetime(date_filed, format="%Y-%m-%d", errors="coerce")
        if pd.isna(filed):
            raise InsiderDataSchemaError(
                f"daily master has invalid filing date at row {line_number}: {date_filed!r}"
            )
        rows.append(
            {
                "accession": accession_match.group("accession"),
                "indexed_cik": f"{int(cik):010d}",
                "archive_cik": f"{int(archive_cik):010d}",
                "company_name": company_name,
                "form": form,
                "filing_date": filed,
                "filename": filename,
                "index_date": stamp,
                "daily_master_last_modified_at": modified,
                "daily_master_source_url": source_url,
                "daily_master_sha256": sha256_bytes(content),
            }
        )
    if not data_started:
        raise InsiderDataSchemaError("SEC daily master header was not found")
    filings = pd.DataFrame(rows)
    if filings.empty:
        filings = pd.DataFrame(
            columns=[
                "accession",
                "indexed_cik",
                "archive_cik",
                "company_name",
                "form",
                "filing_date",
                "filename",
                "index_date",
                "daily_master_last_modified_at",
                "daily_master_source_url",
                "daily_master_sha256",
            ]
        )
    return DailyMasterSnapshot(
        index_date=stamp,
        filings=filings,
        source_url=source_url,
        last_modified_at=modified,
        etag=etag,
        retrieved_at=retrieved_at or utc_now(),
        content_sha256=sha256_bytes(content),
    )


def build_first_daily_public_evidence(
    snapshots: Iterable[DailyMasterSnapshot],
    *,
    forms: tuple[str, ...] = ("4", "4/A"),
) -> pd.DataFrame:
    """Collapse duplicated issuer/owner index rows to first archived visibility."""

    snapshot_list = list(snapshots)
    if not snapshot_list:
        raise ValueError("snapshots must be non-empty")
    date_hashes: dict[pd.Timestamp, str] = {}
    frames: list[pd.DataFrame] = []
    for snapshot in snapshot_list:
        prior = date_hashes.get(snapshot.index_date)
        if prior is not None and prior != snapshot.content_sha256:
            raise AcceptanceClockError(
                f"multiple daily master snapshots for {snapshot.index_date.date()}"
            )
        date_hashes[snapshot.index_date] = snapshot.content_sha256
        frame = snapshot.filings.loc[snapshot.filings["form"].isin(forms)].copy()
        if not frame.empty:
            frames.append(frame)
    columns = [
        "accession",
        "first_index_date",
        "first_daily_master_last_modified_at",
        "first_daily_master_source_url",
        "first_daily_master_sha256",
        "first_day_index_row_count",
        "indexed_ciks",
        "archive_ciks",
        "filenames",
        "forms",
        "filing_dates",
    ]
    if not frames:
        return pd.DataFrame(columns=columns)
    combined = pd.concat(frames, ignore_index=True)
    first_date = combined.groupby("accession")["index_date"].transform("min")
    first = combined.loc[combined["index_date"].eq(first_date)].copy()

    def pipe(values: Iterable[object]) -> str:
        return "|".join(sorted({str(value) for value in values if pd.notna(value)}))

    consistency = first.groupby("accession").agg(
        modified_values=("daily_master_last_modified_at", "nunique"),
        source_values=("daily_master_source_url", "nunique"),
        hash_values=("daily_master_sha256", "nunique"),
    )
    bad = consistency[
        consistency["modified_values"].ne(1)
        | consistency["source_values"].ne(1)
        | consistency["hash_values"].ne(1)
    ]
    if not bad.empty:
        raise AcceptanceClockError(
            f"conflicting first daily public snapshots: {bad.index[:5].tolist()}"
        )
    result = (
        first.groupby("accession", sort=True)
        .agg(
            first_index_date=("index_date", "first"),
            first_daily_master_last_modified_at=(
                "daily_master_last_modified_at",
                "first",
            ),
            first_daily_master_source_url=("daily_master_source_url", "first"),
            first_daily_master_sha256=("daily_master_sha256", "first"),
            first_day_index_row_count=("accession", "size"),
            indexed_ciks=("indexed_cik", pipe),
            archive_ciks=("archive_cik", pipe),
            filenames=("filename", pipe),
            forms=("form", pipe),
            filing_dates=("filing_date", lambda values: pipe(value.date() for value in values)),
        )
        .reset_index()
    )
    validate_unique(result, ["accession"], "first SEC daily public evidence")
    return result[columns]


def _safe_member_names(archive: zipfile.ZipFile) -> dict[str, str]:
    result: dict[str, str] = {}
    for info in archive.infolist():
        name = info.filename
        pure = PurePosixPath(name)
        if pure.is_absolute() or ".." in pure.parts or "\\" in name:
            raise InsiderDataSchemaError(f"unsafe ZIP member path: {name!r}")
        basename = pure.name
        if not basename:
            continue
        key = basename.upper()
        if key in result:
            raise InsiderDataSchemaError(f"duplicate ZIP member basename: {basename!r}")
        result[key] = name
    return result


def _read_tsv(archive: zipfile.ZipFile, member: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(
            archive.open(member),
            sep="\t",
            dtype="string",
            keep_default_na=True,
            low_memory=False,
        )
    except Exception as exc:  # pragma: no cover - pandas error text is version-specific
        raise InsiderDataSchemaError(f"could not parse {member}: {exc}") from exc
    frame.columns = [str(column).strip().upper() for column in frame.columns]
    return frame


def _strict_numeric(frame: pd.DataFrame, column: str, *, table: str) -> None:
    if column not in frame:
        return
    raw = frame[column]
    parsed = pd.to_numeric(raw, errors="coerce")
    invalid = raw.notna() & raw.astype("string").str.strip().ne("") & parsed.isna()
    if invalid.any():
        examples = raw.loc[invalid].head(5).tolist()
        raise InsiderDataSchemaError(f"{table}.{column} has non-numeric values: {examples}")
    frame[column] = parsed


def _strict_dates(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    table: str,
    quarantine_invalid: bool = False,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for column in columns:
        if column not in frame:
            continue
        raw = frame[column]
        text = raw.astype("string").str.strip().str.upper()
        # The SEC bulk files use 01-JAN-0001 as a null sentinel in optional
        # date fields.  No other malformed non-empty date is repaired.
        normalized = raw.mask(text.eq("01-JAN-0001"))
        parsed = pd.to_datetime(normalized, format="%d-%b-%Y", errors="coerce")
        invalid = (
            normalized.notna()
            & normalized.astype("string").str.strip().ne("")
            & parsed.isna()
        )
        if invalid.any():
            examples = raw.loc[invalid].head(5).tolist()
            if not quarantine_invalid:
                raise InsiderDataSchemaError(
                    f"{table}.{column} has invalid dates: {examples}"
                )
            identifier_columns = [
                value
                for value in (
                    "ACCESSION_NUMBER",
                    "NONDERIV_TRANS_SK",
                )
                if value in frame.columns
            ]
            example_rows: list[dict[str, Any]] = []
            for index in raw.index[invalid][:5]:
                row = {
                    key: (
                        None
                        if pd.isna(frame.at[index, key])
                        else str(frame.at[index, key])
                    )
                    for key in identifier_columns
                }
                row["raw_value"] = str(raw.at[index])
                example_rows.append(row)
            issues.append(
                {
                    "table": table,
                    "column": column,
                    "invalid_count": int(invalid.sum()),
                    "action": "coerced_to_NaT_and_excluded_when_date_is_required",
                    "examples": example_rows,
                }
            )
        frame[column] = parsed
    return issues


def _normalize_cik_column(
    frame: pd.DataFrame,
    column: str,
    *,
    table: str,
) -> dict[str, Any] | None:
    """Canonicalize valid SEC CIKs without inferring or repairing digits."""

    raw = frame[column]
    text = raw.astype("string").str.strip()
    present = raw.notna() & text.ne("")
    invalid = present & ~text.fillna("").str.fullmatch(r"\d{1,10}")
    if invalid.any():
        raise InsiderDataSchemaError(
            f"{table}.{column} has invalid CIK values: "
            f"{raw.loc[invalid].head(5).tolist()}"
        )
    normalized = text.where(~present, text.str.zfill(10))
    changed = present & text.ne(normalized)
    frame[column] = normalized
    if not changed.any():
        return None
    examples: list[dict[str, Any]] = []
    for index in raw.index[changed][:5]:
        examples.append(
            {
                "ACCESSION_NUMBER": (
                    None
                    if "ACCESSION_NUMBER" not in frame.columns
                    or pd.isna(frame.at[index, "ACCESSION_NUMBER"])
                    else str(frame.at[index, "ACCESSION_NUMBER"])
                ),
                "raw_value": str(raw.at[index]),
                "normalized_value": str(normalized.at[index]),
            }
        )
    return {
        "table": table,
        "column": column,
        "normalized_count": int(changed.sum()),
        "action": "left_zero_padded_valid_decimal_cik_to_10_digits",
        "examples": examples,
    }


def _validate_metadata(
    metadata: Any,
    *,
    require_aff10b5one: bool,
) -> dict[str, set[str]]:
    """Validate the SEC CSVW metadata and return its declared table columns."""

    if not isinstance(metadata, Mapping):
        raise InsiderDataSchemaError("FORM_345_metadata.json must be a JSON object")
    dialect = metadata.get("dialect")
    if not isinstance(dialect, Mapping):
        raise InsiderDataSchemaError("FORM_345_metadata.json is missing dialect metadata")
    if dialect.get("delimiter") != "\t" or dialect.get("header") is not True:
        raise InsiderDataSchemaError(
            "FORM_345_metadata.json does not declare a tab-delimited header"
        )
    tables = metadata.get("tables")
    if not isinstance(tables, list) or not tables:
        raise InsiderDataSchemaError("FORM_345_metadata.json has no table definitions")

    declared: dict[str, set[str]] = {}
    for position, table in enumerate(tables):
        if not isinstance(table, Mapping):
            raise InsiderDataSchemaError(
                f"FORM_345_metadata.json table {position} is not an object"
            )
        raw_url = table.get("url")
        if not isinstance(raw_url, str) or not raw_url.strip():
            raise InsiderDataSchemaError(
                f"FORM_345_metadata.json table {position} has no URL"
            )
        name = PurePosixPath(raw_url).name.upper()
        if name in declared:
            raise InsiderDataSchemaError(
                f"FORM_345_metadata.json declares table twice: {name}"
            )
        schema = table.get("tableSchema")
        if not isinstance(schema, Mapping):
            raise InsiderDataSchemaError(f"metadata table {name} has no tableSchema")
        columns = schema.get("columns")
        if not isinstance(columns, list) or not columns:
            raise InsiderDataSchemaError(f"metadata table {name} has no columns")
        names: list[str] = []
        for column in columns:
            if not isinstance(column, Mapping) or not isinstance(column.get("name"), str):
                raise InsiderDataSchemaError(
                    f"metadata table {name} contains an invalid column definition"
                )
            column_name = str(column["name"]).strip().upper()
            if not column_name:
                raise InsiderDataSchemaError(
                    f"metadata table {name} contains an empty column name"
                )
            names.append(column_name)
        if len(names) != len(set(names)):
            raise InsiderDataSchemaError(f"metadata table {name} has duplicate columns")
        declared[name] = set(names)

        canonical_name = next(
            (value for value in TABLE_PRIMARY_KEYS if value.upper() == name),
            None,
        )
        if canonical_name is not None:
            raw_primary_key = schema.get("primaryKey")
            if isinstance(raw_primary_key, str):
                primary_key = (raw_primary_key.strip().upper(),)
            elif isinstance(raw_primary_key, list):
                primary_key = tuple(str(value).strip().upper() for value in raw_primary_key)
            else:
                primary_key = ()
            if primary_key != TABLE_PRIMARY_KEYS[canonical_name]:
                raise InsiderDataSchemaError(
                    f"metadata table {canonical_name} has unexpected primary key: "
                    f"{primary_key}"
                )

    required_tables = {name.upper() for name in TABLE_PRIMARY_KEYS}
    missing_tables = required_tables.difference(declared)
    if missing_tables:
        raise InsiderDataSchemaError(
            "FORM_345_metadata.json missing table definitions: "
            f"{sorted(missing_tables)}"
        )
    for canonical_name, required_columns in REQUIRED_COLUMNS.items():
        expected = set(required_columns)
        if canonical_name == "SUBMISSION.tsv" and require_aff10b5one:
            expected.add("AFF10B5ONE")
        missing_columns = expected.difference(declared[canonical_name.upper()])
        if missing_columns:
            raise InsiderDataSchemaError(
                f"metadata table {canonical_name} missing columns: {sorted(missing_columns)}"
            )
    return declared


def validate_insider_tables(
    submissions: pd.DataFrame,
    reporting_owners: pd.DataFrame,
    nonderivative_transactions: pd.DataFrame,
    footnotes: pd.DataFrame,
    *,
    require_aff10b5one: bool = False,
) -> None:
    """Validate keys, identifiers, and foreign keys without repairing input."""

    tables = {
        "SUBMISSION.tsv": submissions,
        "REPORTINGOWNER.tsv": reporting_owners,
        "NONDERIV_TRANS.tsv": nonderivative_transactions,
        "FOOTNOTES.tsv": footnotes,
    }
    for name, frame in tables.items():
        expected = set(REQUIRED_COLUMNS[name])
        if name == "SUBMISSION.tsv" and require_aff10b5one:
            expected.add("AFF10B5ONE")
        missing = expected.difference(frame.columns)
        if missing:
            raise InsiderDataSchemaError(f"{name} missing columns: {sorted(missing)}")
        for column in REQUIRED_VALUE_COLUMNS[name]:
            empty = frame[column].isna() | frame[column].astype("string").str.strip().eq("")
            if empty.any():
                raise InsiderDataSchemaError(
                    f"{name}.{column} contains missing key values"
                )

    validate_unique(submissions, ["ACCESSION_NUMBER"], "SEC insider submissions")
    validate_unique(
        reporting_owners,
        ["ACCESSION_NUMBER", "RPTOWNERCIK"],
        "SEC insider reporting owners",
    )
    validate_unique(
        nonderivative_transactions,
        ["ACCESSION_NUMBER", "NONDERIV_TRANS_SK"],
        "SEC insider non-derivative transactions",
    )
    validate_unique(footnotes, ["ACCESSION_NUMBER", "FOOTNOTE_ID"], "SEC insider footnotes")

    accessions = submissions["ACCESSION_NUMBER"].astype("string")
    invalid_accession = ~accessions.fillna("").str.fullmatch(ACCESSION_PATTERN)
    if invalid_accession.any():
        raise InsiderDataSchemaError(
            "SUBMISSION.tsv has invalid ACCESSION_NUMBER values: "
            f"{accessions.loc[invalid_accession].head(5).tolist()}"
        )
    invalid_cik = ~submissions["ISSUERCIK"].fillna("").astype("string").str.fullmatch(r"\d{10}")
    if invalid_cik.any():
        raise InsiderDataSchemaError(
            f"SUBMISSION.tsv has invalid ISSUERCIK values: "
            f"{submissions.loc[invalid_cik, 'ISSUERCIK'].head(5).tolist()}"
        )
    invalid_owner_cik = ~reporting_owners["RPTOWNERCIK"].fillna("").astype("string").str.fullmatch(
        r"\d{10}"
    )
    if invalid_owner_cik.any():
        raise InsiderDataSchemaError(
            "REPORTINGOWNER.tsv has invalid RPTOWNERCIK values: "
            f"{reporting_owners.loc[invalid_owner_cik, 'RPTOWNERCIK'].head(5).tolist()}"
        )

    accession_set = set(accessions)
    for name, frame in tables.items():
        if name == "SUBMISSION.tsv":
            continue
        orphan = ~frame["ACCESSION_NUMBER"].isin(accession_set)
        if orphan.any():
            examples = frame.loc[orphan, "ACCESSION_NUMBER"].head(5).tolist()
            raise InsiderDataSchemaError(f"{name} contains orphan accessions: {examples}")


def parse_insider_zip(content: bytes, *, quarter: str) -> InsiderQuarterTables:
    """Parse one official quarterly archive and enforce the documented schema."""

    normalized_quarter = _normalize_quarter(quarter)
    require_aff10b5one = int(normalized_quarter[:4]) >= 2023
    if not content:
        raise InsiderDataSchemaError("SEC insider ZIP is empty")
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as exc:
        raise InsiderDataSchemaError("SEC insider payload is not a valid ZIP") from exc
    with archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise InsiderDataSchemaError(f"SEC insider ZIP CRC failure: {corrupt}")
        members = _safe_member_names(archive)
        missing_files = {
            name for name in REQUIRED_TABLE_FILES if name.upper() not in members
        }
        if missing_files:
            raise InsiderDataSchemaError(
                f"SEC insider ZIP missing required files: {sorted(missing_files)}"
            )
        metadata_keys = [
            alias.upper()
            for alias in METADATA_FILE_ALIASES
            if alias.upper() in members
        ]
        if not metadata_keys:
            raise InsiderDataSchemaError(
                "SEC insider ZIP missing a supported metadata file: "
                f"{list(METADATA_FILE_ALIASES)}"
            )
        if len(metadata_keys) != 1:
            raise InsiderDataSchemaError(
                "SEC insider ZIP contains ambiguous metadata files: "
                f"{[members[key] for key in metadata_keys]}"
            )
        metadata_member = members[metadata_keys[0]]
        metadata_filename = PurePosixPath(metadata_member).name
        submissions = _read_tsv(archive, members["SUBMISSION.TSV"])
        owners = _read_tsv(archive, members["REPORTINGOWNER.TSV"])
        transactions = _read_tsv(archive, members["NONDERIV_TRANS.TSV"])
        footnotes = _read_tsv(archive, members["FOOTNOTES.TSV"])
        try:
            metadata = json.loads(archive.read(metadata_member))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise InsiderDataSchemaError(
                f"{metadata_filename} is invalid"
            ) from exc

    _validate_metadata(
        metadata,
        require_aff10b5one=require_aff10b5one,
    )
    # SEC archives may omit optional metadata columns and have historically
    # shipped unused footnote columns under a different name.  Signal-bearing
    # fields remain fail-closed through both metadata and TSV required subsets.

    data_quality_issues: list[dict[str, Any]] = []
    for frame, column, table in (
        (submissions, "ISSUERCIK", "SUBMISSION.tsv"),
        (owners, "RPTOWNERCIK", "REPORTINGOWNER.tsv"),
    ):
        issue = _normalize_cik_column(frame, column, table=table)
        if issue is not None:
            data_quality_issues.append(issue)
    # Filing date is the mandatory daily availability key and remains strict.
    _strict_dates(
        submissions,
        ("FILING_DATE",),
        table="SUBMISSION.tsv",
    )
    # SEC's official archives contain a small number of impossible year-00xx
    # values in optional report/original-submission dates and transaction dates.
    # They are never guessed or repaired.  The raw ZIP remains immutable, the
    # parsed value becomes NaT, and the accession-level finding is returned so
    # downstream event filters exclude the affected transaction fail-closed.
    data_quality_issues.extend(
        _strict_dates(
            submissions,
            ("PERIOD_OF_REPORT", "DATE_OF_ORIG_SUB"),
            table="SUBMISSION.tsv",
            quarantine_invalid=True,
        )
    )
    data_quality_issues.extend(
        _strict_dates(
            transactions,
            DATE_COLUMNS["NONDERIV_TRANS.tsv"],
            table="NONDERIV_TRANS.tsv",
            quarantine_invalid=True,
        )
    )
    for column in NUMERIC_COLUMNS:
        _strict_numeric(transactions, column, table="NONDERIV_TRANS.tsv")

    validate_insider_tables(
        submissions,
        owners,
        transactions,
        footnotes,
        require_aff10b5one=require_aff10b5one,
    )
    return InsiderQuarterTables(
        quarter=normalized_quarter,
        submissions=submissions,
        reporting_owners=owners,
        nonderivative_transactions=transactions,
        footnotes=footnotes,
        metadata=metadata,
        metadata_filename=metadata_filename,
        data_quality_issues=tuple(data_quality_issues),
        archive_sha256=sha256_bytes(content),
    )


def _columnar_submission_rows(
    payload: Mapping[str, Any], *, source: str
) -> list[dict[str, Any]]:
    recent: Mapping[str, Any]
    if "filings" in payload:
        filings = payload.get("filings")
        if not isinstance(filings, Mapping) or not isinstance(
            filings.get("recent"), Mapping
        ):
            raise InsiderDataSchemaError(
                f"SEC submissions payload has no filings.recent object in {source}"
            )
        recent = filings["recent"]
    else:
        recent = payload
    required = (
        "accessionNumber",
        "filingDate",
        "acceptanceDateTime",
        "form",
        "primaryDocument",
    )
    missing = [key for key in required if key not in recent]
    if missing:
        raise InsiderDataSchemaError(
            f"SEC submissions payload missing columns in {source}: {missing}"
        )
    invalid_types = [key for key in required if not isinstance(recent[key], list)]
    if invalid_types:
        raise InsiderDataSchemaError(
            f"SEC submissions payload has non-list columns in {source}: {invalid_types}"
        )
    lengths = {key: len(recent[key]) for key in required}
    if len(set(lengths.values())) != 1:
        raise InsiderDataSchemaError(
            f"SEC submissions column lengths differ in {source}: {lengths}"
        )
    count = next(iter(lengths.values()), 0)
    primary = recent["primaryDocument"]
    return [
        {
            "accession": str(recent["accessionNumber"][index]),
            "filing_date": recent["filingDate"][index],
            "acceptance_at_raw": recent["acceptanceDateTime"][index],
            "acceptance_at": recent["acceptanceDateTime"][index],
            "form": str(recent["form"][index]),
            "primary_document": primary[index],
            "acceptance_source": source,
        }
        for index in range(count)
    ]


def build_acceptance_map(
    payloads: Iterable[Mapping[str, Any]], *, sources: Iterable[str] | None = None
) -> pd.DataFrame:
    """Flatten current and historical submissions JSON into one accession clock."""

    payload_list = list(payloads)
    source_list = list(sources) if sources is not None else [f"payload:{i}" for i in range(len(payload_list))]
    if len(source_list) != len(payload_list):
        raise ValueError("sources and payloads must have equal length")
    rows: list[dict[str, Any]] = []
    for payload, source in zip(payload_list, source_list):
        rows.extend(_columnar_submission_rows(payload, source=source))
    columns = [
        "accession",
        "filing_date",
        "acceptance_at_raw",
        "acceptance_at",
        "form",
        "primary_document",
        "acceptance_source",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows, columns=columns)
    invalid_accession = ~frame["accession"].str.fullmatch(ACCESSION_PATTERN)
    if invalid_accession.any():
        raise InsiderDataSchemaError(
            f"submissions history has invalid accessions: "
            f"{frame.loc[invalid_accession, 'accession'].head(5).tolist()}"
        )
    frame["filing_date"] = pd.to_datetime(frame["filing_date"], errors="coerce")
    frame["acceptance_at"] = pd.to_datetime(frame["acceptance_at"], utc=True, errors="coerce")
    invalid_clock = frame["filing_date"].isna() | frame["acceptance_at"].isna()
    if invalid_clock.any():
        raise AcceptanceClockError(
            "SEC submissions history contains missing or invalid filing/acceptance times: "
            f"{frame.loc[invalid_clock, ['accession', 'filing_date', 'acceptance_at']].head(5).to_dict('records')}"
        )
    comparison = frame.copy()
    comparison["acceptance_ns"] = comparison["acceptance_at"].astype("int64")
    conflict = comparison.groupby("accession").agg(
        acceptance_values=("acceptance_ns", "nunique"),
        form_values=("form", "nunique"),
        filing_values=("filing_date", "nunique"),
        primary_document_values=("primary_document", "nunique"),
        acceptance_raw_values=("acceptance_at_raw", "nunique"),
    )
    conflict = conflict[
        conflict["acceptance_values"].gt(1)
        | conflict["form_values"].gt(1)
        | conflict["filing_values"].gt(1)
        | conflict["primary_document_values"].gt(1)
        | conflict["acceptance_raw_values"].gt(1)
    ]
    if not conflict.empty:
        raise AcceptanceClockError(
            f"conflicting submissions-history clocks for accessions: {conflict.index[:5].tolist()}"
        )
    return (
        frame.sort_values(
            ["accession", "primary_document", "acceptance_source"],
            kind="mergesort",
            na_position="last",
        )
        .drop_duplicates("accession", keep="first")
        .reset_index(drop=True)
    )


def attach_acceptance_clock(
    events: pd.DataFrame,
    acceptance_map: pd.DataFrame,
    *,
    accession_column: str = "accession",
    filing_date_column: str = "filing_date",
    form_column: str = "document_type",
    safety_delay: str | pd.Timedelta = "5min",
    allow_acceptance_only_diagnostic: bool = False,
) -> pd.DataFrame:
    """Attach an acceptance-only diagnostic clock.

    Acceptance is not proof of public dissemination. Strict historical event
    research must use :func:`attach_first_public_clock`; callers must opt in
    explicitly to this weaker diagnostic clock.
    """

    if not allow_acceptance_only_diagnostic:
        raise AcceptanceClockError(
            "acceptanceDateTime alone is not public-time evidence; "
            "use attach_first_public_clock or explicitly permit a diagnostic clock"
        )

    required_events = {accession_column, filing_date_column, form_column}
    missing = required_events.difference(events.columns)
    if missing:
        raise AcceptanceClockError(f"events missing clock columns: {sorted(missing)}")
    required_map = {"accession", "filing_date", "acceptance_at", "form"}
    missing_map = required_map.difference(acceptance_map.columns)
    if missing_map:
        raise AcceptanceClockError(f"acceptance map missing columns: {sorted(missing_map)}")
    if events.empty:
        result = events.copy()
        result["acceptance_at"] = pd.Series(dtype="datetime64[ns, UTC]")
        result["information_at"] = pd.Series(dtype="datetime64[ns, UTC]")
        return result

    validate_unique(acceptance_map, ["accession"], "SEC acceptance map")
    right = acceptance_map.rename(
        columns={
            "filing_date": "acceptance_filing_date",
            "form": "acceptance_form",
        }
    )
    result = events.merge(
        right,
        left_on=accession_column,
        right_on="accession",
        how="left",
        validate="many_to_one",
        suffixes=("", "_clock"),
    )
    missing_clock = result["acceptance_at"].isna()
    if missing_clock.any():
        raise AcceptanceClockError(
            f"events missing acceptanceDateTime: "
            f"{result.loc[missing_clock, accession_column].head(5).tolist()}"
        )
    event_date = pd.to_datetime(result[filing_date_column], errors="coerce").dt.date
    mapped_date = pd.to_datetime(result["acceptance_filing_date"], errors="coerce").dt.date
    acceptance_local_date = pd.to_datetime(result["acceptance_at"], utc=True).dt.tz_convert(
        "America/New_York"
    ).dt.date
    form_match = result[form_column].astype("string").eq(result["acceptance_form"].astype("string"))
    invalid = event_date.isna() | mapped_date.isna() | event_date.ne(mapped_date) | event_date.ne(
        acceptance_local_date
    ) | ~form_match
    if invalid.any():
        columns = [
            accession_column,
            filing_date_column,
            "acceptance_filing_date",
            "acceptance_at",
            form_column,
            "acceptance_form",
        ]
        raise AcceptanceClockError(
            "SEC event filing date/form conflicts with acceptance history: "
            f"{result.loc[invalid, columns].head(5).to_dict('records')}"
        )
    delay = pd.Timedelta(safety_delay)
    if delay < pd.Timedelta(0):
        raise ValueError("safety_delay must be non-negative")
    result["acceptance_at"] = pd.to_datetime(result["acceptance_at"], utc=True)
    result["information_at"] = result["acceptance_at"] + delay
    result["clock_claim_status"] = "acceptance_only_diagnostic_not_public_evidence"
    return result.drop(columns=["accession_clock"], errors="ignore")


def parse_acceptance_header(raw_submission: str | bytes) -> pd.Timestamp:
    """Parse the EDGAR SGML acceptance header, whose timestamp is New York time."""

    text = raw_submission.decode("utf-8", errors="strict") if isinstance(raw_submission, bytes) else raw_submission
    matches = ACCEPTANCE_HEADER_PATTERN.findall(text)
    if len(matches) != 1:
        raise AcceptanceClockError(
            f"expected exactly one ACCEPTANCE-DATETIME header, found {len(matches)}"
        )
    local = pd.to_datetime(matches[0], format="%Y%m%d%H%M%S", errors="raise")
    return local.tz_localize("America/New_York", ambiguous="raise", nonexistent="raise").tz_convert(
        "UTC"
    )


def reconcile_acceptance_headers(
    acceptance_map: pd.DataFrame,
    raw_submissions: Mapping[str, str | bytes],
    *,
    require_exact_api_match: bool = True,
) -> pd.DataFrame:
    """Reconcile submissions-API clocks to raw EDGAR SGML headers.

    SEC submissions JSON values are retained verbatim because their apparent
    timezone encoding is not uniformly reliable. The raw SGML header is parsed
    as a New York wall clock. A conflict is fatal by default; diagnostic runs
    may explicitly retain the raw-header clock while exposing the conflict.
    """

    required = {"accession", "acceptance_at", "acceptance_at_raw"}
    missing = required.difference(acceptance_map.columns)
    if missing:
        raise AcceptanceClockError(
            f"acceptance map missing reconciliation columns: {sorted(missing)}"
        )
    validate_unique(acceptance_map, ["accession"], "SEC acceptance map")
    expected = set(acceptance_map["accession"].astype("string"))
    supplied = {str(value) for value in raw_submissions}
    missing_raw = expected.difference(supplied)
    unexpected_raw = supplied.difference(expected)
    if missing_raw or unexpected_raw:
        raise AcceptanceClockError(
            "raw submission accession coverage mismatch: "
            f"missing={sorted(missing_raw)[:5]}, unexpected={sorted(unexpected_raw)[:5]}"
        )

    result = acceptance_map.copy()
    result["api_acceptance_at"] = pd.to_datetime(
        result["acceptance_at"], utc=True, errors="coerce"
    )
    if result["api_acceptance_at"].isna().any():
        raise AcceptanceClockError("acceptance map contains invalid API timestamps")
    raw_clocks = {
        accession: parse_acceptance_header(raw_submissions[accession])
        for accession in sorted(expected)
    }
    result["raw_header_acceptance_at"] = result["accession"].map(raw_clocks)
    result["api_raw_delta_seconds"] = (
        result["api_acceptance_at"] - result["raw_header_acceptance_at"]
    ).dt.total_seconds()
    result["api_clock_matches_raw"] = result["api_raw_delta_seconds"].eq(0)
    conflicts = ~result["api_clock_matches_raw"]
    if require_exact_api_match and conflicts.any():
        examples = result.loc[
            conflicts,
            [
                "accession",
                "acceptance_at_raw",
                "api_acceptance_at",
                "raw_header_acceptance_at",
                "api_raw_delta_seconds",
            ],
        ].head(5)
        raise AcceptanceClockError(
            "SEC submissions API clock conflicts with raw SGML header: "
            f"{examples.to_dict('records')}"
        )
    result["acceptance_at"] = result["raw_header_acceptance_at"]
    result["acceptance_clock_source"] = "raw_edgar_sgml_header"
    result["json_clock_conflict"] = conflicts
    return result


def attach_first_public_clock(
    events: pd.DataFrame,
    reconciled_clocks: pd.DataFrame,
    daily_evidence: pd.DataFrame,
    *,
    accession_column: str = "accession",
    form_column: str = "document_type",
) -> pd.DataFrame:
    """Attach a strict historical clock from raw acceptance and daily evidence.

    The archived daily-index HTTP ``Last-Modified`` timestamp is a conservative
    known-public-by bound. Callers must supply snapshots far enough back to find
    the accession's true first occurrence, including occurrences before a later
    PAC or filing-date change.
    """

    _event_required = {accession_column, form_column}
    missing_events = _event_required.difference(events.columns)
    if missing_events:
        raise AcceptanceClockError(
            f"events missing public-clock columns: {sorted(missing_events)}"
        )
    required_clocks = {
        "accession",
        "raw_header_acceptance_at",
        "acceptance_clock_source",
    }
    missing_clocks = required_clocks.difference(reconciled_clocks.columns)
    if missing_clocks:
        raise AcceptanceClockError(
            f"reconciled clocks missing columns: {sorted(missing_clocks)}"
        )
    required_daily = {
        "accession",
        "first_index_date",
        "first_daily_master_last_modified_at",
        "first_daily_master_source_url",
        "first_daily_master_sha256",
        "forms",
    }
    missing_daily = required_daily.difference(daily_evidence.columns)
    if missing_daily:
        raise AcceptanceClockError(
            f"daily public evidence missing columns: {sorted(missing_daily)}"
        )
    validate_unique(reconciled_clocks, ["accession"], "reconciled SEC clocks")
    validate_unique(daily_evidence, ["accession"], "SEC daily public evidence")

    clock_columns = list(required_clocks)
    daily_columns = list(required_daily)
    result = events.merge(
        reconciled_clocks[clock_columns],
        left_on=accession_column,
        right_on="accession",
        how="left",
        validate="many_to_one",
        suffixes=("", "_clock"),
    ).merge(
        daily_evidence[daily_columns],
        left_on=accession_column,
        right_on="accession",
        how="left",
        validate="many_to_one",
        suffixes=("", "_daily"),
    )
    raw_at = pd.to_datetime(
        result["raw_header_acceptance_at"], utc=True, errors="coerce"
    )
    public_at = pd.to_datetime(
        result["first_daily_master_last_modified_at"], utc=True, errors="coerce"
    )
    missing = raw_at.isna() | public_at.isna()
    if missing.any():
        raise AcceptanceClockError(
            "events lack raw acceptance or first daily public evidence: "
            f"{result.loc[missing, accession_column].head(5).tolist()}"
        )
    form_matches = [
        str(form) in {token for token in str(index_forms).split("|") if token}
        for form, index_forms in zip(result[form_column], result["forms"])
    ]
    if not all(form_matches):
        bad = result.loc[[not value for value in form_matches], accession_column]
        raise AcceptanceClockError(
            f"event forms conflict with first daily index: {bad.head(5).tolist()}"
        )
    result["accepted_at_raw"] = raw_at
    result["first_public_evidence_at"] = public_at
    result["information_at"] = pd.concat(
        [raw_at.rename("raw"), public_at.rename("public")], axis=1
    ).max(axis=1)
    result["public_evidence_lag_seconds"] = (public_at - raw_at).dt.total_seconds()
    result["public_clock_basis"] = (
        "max_raw_sgml_acceptance_and_first_archived_daily_master_last_modified"
    )
    invalid = ~(
        result["accepted_at_raw"].le(result["information_at"])
        & result["first_public_evidence_at"].le(result["information_at"])
    )
    if invalid.any():  # pragma: no cover - max() makes this defensive
        raise AcceptanceClockError("public information clock is internally inconsistent")
    return result.drop(columns=["accession_clock", "accession_daily"], errors="ignore")


def filing_archive_urls(
    *, issuer_cik: str | int, accession: str, primary_document: str | None = None
) -> dict[str, str]:
    """Construct deterministic official archive URLs for one accession."""

    accession = str(accession)
    if not ACCESSION_PATTERN.fullmatch(accession):
        raise ValueError(f"invalid accession: {accession!r}")
    cik_text = str(issuer_cik).strip()
    if not re.fullmatch(r"\d{1,10}", cik_text):
        raise ValueError(f"invalid issuer CIK: {issuer_cik!r}")
    cik = int(cik_text)
    compact = accession.replace("-", "")
    base = f"{ARCHIVE_ROOT}/{cik}/{compact}"
    result = {
        "directory_index": f"{base}/index.json",
        "complete_submission": f"{base}/{accession}.txt",
        "header_sgml": f"{base}/{accession}.hdr.sgml",
    }
    if primary_document:
        basename = PurePosixPath(str(primary_document)).name
        if not basename or basename in {".", ".."}:
            raise ValueError(f"invalid primary document: {primary_document!r}")
        result["primary_document"] = f"{base}/{basename}"
    return result


class SecInsiderBulkClient:
    """Small, rate-limited SEC downloader with immutable content-addressed cache files."""

    def __init__(
        self,
        *,
        cache_dir: str | Path = "data/cache/sec/insider",
        user_agent: str | None = None,
        timeout: int = 60,
        min_interval_seconds: float = 0.20,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.user_agent = user_agent or os.getenv("SEC_USER_AGENT")
        if (
            not self.user_agent
            or ".invalid" in self.user_agent.lower()
            or "example.com" in self.user_agent.lower()
            or not re.search(r"[^\s@]+@[^\s@]+\.[^\s@]+", self.user_agent)
        ):
            raise RuntimeError(
                "SEC_USER_AGENT must identify the application and contain a real contact email"
            )
        if min_interval_seconds < 0.1:
            raise ValueError("SEC request interval must be at least 0.1 seconds")
        self.timeout = int(timeout)
        self.min_interval_seconds = float(min_interval_seconds)
        self._last_request = 0.0
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"}
        )

    def _request(self, url: str) -> requests.Response:
        delay = self.min_interval_seconds - (time.monotonic() - self._last_request)
        if delay > 0:
            time.sleep(delay)
        response: requests.Response | None = None
        for attempt in range(5):
            response = self.session.get(url, timeout=self.timeout)
            self._last_request = time.monotonic()
            if response.status_code not in {403, 429, 500, 502, 503, 504}:
                break
            retry_after = response.headers.get("Retry-After")
            wait = float(retry_after) if retry_after and retry_after.isdigit() else 0.5 * 2**attempt
            time.sleep(wait)
        assert response is not None
        response.raise_for_status()
        return response

    def _get(self, url: str) -> bytes:
        return self._request(url).content

    def discover_quarters(self) -> dict[str, str]:
        raw = self._get(INSIDER_DATASET_PAGE)
        return discover_quarter_links(raw.decode("utf-8", errors="strict"))

    def download_quarter(
        self, quarter: str, *, url: str | None = None
    ) -> tuple[InsiderQuarterTables, DatasetRecord, Path]:
        normalized = _normalize_quarter(quarter)
        source_url = url or self.discover_quarters().get(normalized)
        if not source_url:
            raise FileNotFoundError(f"SEC insider quarter is not published: {normalized}")
        raw = self._get(source_url)
        parsed = parse_insider_zip(raw, quarter=normalized)
        path = self.cache_dir / "bulk" / normalized / f"{parsed.archive_sha256}.zip"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != raw:
            raise InsiderDataSchemaError(f"content-addressed cache conflict: {path}")
        if not path.exists():
            path.write_bytes(raw)
        record = DatasetRecord(
            provider="SEC DERA Insider Transactions",
            dataset=f"form345:{normalized}",
            retrieved_at=utc_now(),
            source_url=source_url,
            sha256=parsed.archive_sha256,
            rows=len(parsed.nonderivative_transactions),
            finality="as_filed_quarterly_flattening",
            revision_policy="immutable content-addressed ZIP; amendments retained separately",
            notes={
                "acceptance_clock": "absent; submissions API join required before research",
                "tables": sorted(REQUIRED_TABLE_FILES),
                "metadata_file": parsed.metadata_filename,
                "quarantined_optional_date_values": sum(
                    int(issue.get("invalid_count", 0))
                    for issue in parsed.data_quality_issues
                ),
                "normalized_cik_values": sum(
                    int(issue.get("normalized_count", 0))
                    for issue in parsed.data_quality_issues
                ),
            },
        )
        return parsed, record, path

    def download_daily_master(
        self, index_date: str | pd.Timestamp
    ) -> tuple[DailyMasterSnapshot, DatasetRecord, Path]:
        """Download one archived daily master and preserve public-time headers."""

        source_url = daily_master_url(index_date)
        response = self._request(source_url)
        last_modified_text = response.headers.get("Last-Modified")
        if not last_modified_text:
            raise AcceptanceClockError(
                f"daily master response has no Last-Modified header: {source_url}"
            )
        try:
            last_modified = pd.Timestamp(parsedate_to_datetime(last_modified_text)).tz_convert(
                "UTC"
            )
        except (TypeError, ValueError) as exc:
            raise AcceptanceClockError(
                f"daily master has invalid Last-Modified header: {last_modified_text!r}"
            ) from exc
        raw = response.content
        snapshot = parse_daily_master_index(
            raw,
            index_date=index_date,
            source_url=source_url,
            last_modified_at=last_modified,
            etag=response.headers.get("ETag"),
        )
        date_key = snapshot.index_date.strftime("%Y%m%d")
        suffix = ".idx.gz" if raw[:2] == b"\x1f\x8b" else ".idx"
        path = (
            self.cache_dir
            / "daily_master"
            / date_key
            / f"{snapshot.content_sha256}{suffix}"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != raw:
            raise InsiderDataSchemaError(f"content-addressed cache conflict: {path}")
        if not path.exists():
            path.write_bytes(raw)
        record = DatasetRecord(
            provider="SEC EDGAR Daily Index",
            dataset=f"daily_master:{date_key}",
            retrieved_at=snapshot.retrieved_at,
            source_url=source_url,
            sha256=snapshot.content_sha256,
            rows=len(snapshot.filings),
            max_available_at=snapshot.last_modified_at.isoformat(),
            finality="archived_daily_snapshot",
            revision_policy=(
                "preserve daily-first occurrence; reconcile later PAC changes separately"
            ),
            notes={
                "index_date": snapshot.index_date.date().isoformat(),
                "last_modified": snapshot.last_modified_at.isoformat(),
                "etag": snapshot.etag,
            },
        )
        return snapshot, record, path

    def download_raw_submission(
        self,
        *,
        issuer_cik: str | int,
        accession: str,
    ) -> tuple[bytes, pd.Timestamp, DatasetRecord, Path]:
        """Download and parse a complete filing without treating acceptance as public time."""

        source_url = filing_archive_urls(
            issuer_cik=issuer_cik, accession=accession
        )["complete_submission"]
        raw = self._get(source_url)
        accepted_at = parse_acceptance_header(raw)
        digest = sha256_bytes(raw)
        cik = int(str(issuer_cik).strip())
        path = (
            self.cache_dir
            / "raw_submissions"
            / f"CIK{cik:010d}"
            / accession
            / f"{digest}.txt"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != raw:
            raise InsiderDataSchemaError(f"content-addressed cache conflict: {path}")
        if not path.exists():
            path.write_bytes(raw)
        record = DatasetRecord(
            provider="SEC EDGAR Archives",
            dataset=f"complete_submission:{accession}",
            retrieved_at=utc_now(),
            source_url=source_url,
            sha256=digest,
            rows=1,
            finality="archived_as_disseminated_submission",
            revision_policy="immutable content-addressed response",
            notes={
                "accepted_at_raw": accepted_at.isoformat(),
                "public_time_warning": (
                    "acceptance is not sufficient; first daily public evidence required"
                ),
            },
        )
        return raw, accepted_at, record, path

    def fetch_submission_history(
        self, issuer_cik: str | int
    ) -> tuple[list[dict[str, Any]], list[DatasetRecord]]:
        cik_text = str(issuer_cik).strip()
        if not re.fullmatch(r"\d{1,10}", cik_text):
            raise ValueError(f"invalid issuer CIK: {issuer_cik!r}")
        cik = int(cik_text)
        main_url = SUBMISSIONS_URL.format(cik=cik)
        queue = [main_url]
        payloads: list[dict[str, Any]] = []
        records: list[DatasetRecord] = []
        for index, source_url in enumerate(queue):
            raw = self._get(source_url)
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise InsiderDataSchemaError(f"invalid submissions JSON: {source_url}") from exc
            digest = sha256_bytes(raw)
            path = self.cache_dir / "submissions" / f"CIK{cik:010d}" / f"{digest}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.read_bytes() != raw:
                raise InsiderDataSchemaError(f"content-addressed cache conflict: {path}")
            if not path.exists():
                path.write_bytes(raw)
            payloads.append(payload)
            records.append(
                DatasetRecord(
                    provider="SEC EDGAR Submissions",
                    dataset=f"submissions:{cik:010d}:{index}",
                    retrieved_at=utc_now(),
                    source_url=source_url,
                    sha256=digest,
                    rows=len(
                        (((payload.get("filings") or {}).get("recent") or {}).get("form") or [])
                        if "filings" in payload
                        else (payload.get("form") or [])
                    ),
                    finality="filed",
                    revision_policy="immutable content-addressed response",
                )
            )
            if index == 0:
                filings = payload.get("filings")
                if not isinstance(filings, Mapping):
                    raise InsiderDataSchemaError(
                        "main submissions JSON has no filings object"
                    )
                history_files = filings.get("files")
                if not isinstance(history_files, list):
                    raise InsiderDataSchemaError(
                        "main submissions JSON has no filings.files list"
                    )
                names: set[str] = set()
                for item in history_files:
                    if not isinstance(item, Mapping):
                        raise InsiderDataSchemaError(
                            "submissions-history file entry is not an object"
                        )
                    name = str(item.get("name", ""))
                    if not HISTORY_FILE_PATTERN.fullmatch(name):
                        raise InsiderDataSchemaError(
                            f"unsafe or unexpected submissions-history filename: {name!r}"
                        )
                    if name in names:
                        raise InsiderDataSchemaError(
                            f"duplicate submissions-history filename: {name!r}"
                        )
                    names.add(name)
                    queue.append(urljoin("https://data.sec.gov/submissions/", name))
        return payloads, records


__all__ = [
    "AcceptanceClockError",
    "DAILY_MASTER_URL",
    "DailyMasterSnapshot",
    "INSIDER_DATASET_PAGE",
    "InsiderDataSchemaError",
    "InsiderQuarterTables",
    "SecInsiderBulkClient",
    "attach_acceptance_clock",
    "attach_first_public_clock",
    "build_first_daily_public_evidence",
    "build_acceptance_map",
    "daily_master_url",
    "discover_quarter_links",
    "filing_archive_urls",
    "parse_acceptance_header",
    "parse_daily_master_index",
    "parse_insider_zip",
    "reconcile_acceptance_headers",
    "validate_insider_tables",
]
