"""Strict SEC 10-K/10-Q header inventory and primary-document cache.

This module deliberately stops at acquisition and lineage.  It does not parse
filing text, construct signals, or attach market outcomes.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import requests

from alpha_factor_engine.contracts import DatasetRecord, sha256_bytes, utc_now


ARCHIVE_ROOT = "https://www.sec.gov/Archives/edgar/data"
PERIODIC_FORMS = frozenset({"10-K", "10-Q"})

ACCESSION_PATTERN = re.compile(r"^\d{10}-\d{2}-\d{6}$")
ACCEPTANCE_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?Z$"
)
CIK_PATTERN = re.compile(r"^\d{1,10}$")
MAIN_FILE_PATTERN = re.compile(r"^CIK(?P<cik>\d{10})\.json\.gz$")
HISTORY_FILE_PATTERN = re.compile(
    r"^CIK(?P<cik>\d{10})-submissions-\d{3}\.json$"
)
PRIMARY_DOCUMENT_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,254}\.(?:htm|html)$", re.IGNORECASE
)
CONTENT_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
EMAIL_PATTERN = re.compile(r"[^\s<>@]+@[^\s<>@]+\.[A-Za-z]{2,}")

REQUIRED_HEADER_COLUMNS = (
    "accessionNumber",
    "filingDate",
    "reportDate",
    "acceptanceDateTime",
    "form",
    "primaryDocument",
)

INVENTORY_COLUMNS = (
    "issuer_cik",
    "accession",
    "form",
    "filing_date",
    "report_date",
    "acceptance_at_raw",
    "acceptance_at",
    "primary_document",
    "primary_document_url",
    "source_kind",
    "source_path",
    "source_file_sha256",
    "source_content_sha256",
)

RETRYABLE_HTTP_STATUS = frozenset({403, 408, 425, 429, 500, 502, 503, 504})
ALLOWED_HTML_MEDIA_TYPES = frozenset({"text/html", "application/xhtml+xml"})
ALLOWED_SGML_MEDIA_TYPES = frozenset(
    {"text/plain", "application/sgml", "application/octet-stream", "text/html"}
)
ACCEPTANCE_HEADER_PATTERN = re.compile(rb"<ACCEPTANCE-DATETIME>(\d{14})")
SEC_BLOCK_PAGE_MARKERS = (
    b"undeclared automated tool",
    b"request rate threshold exceeded",
    b"your request originates from an undeclared automated tool",
)


class PeriodicFilingDataError(ValueError):
    """Raised when SEC periodic-filing evidence violates its contract."""


@dataclass(frozen=True)
class _SourcePayload:
    payload: Mapping[str, Any]
    path: Path
    file_sha256: str
    content_sha256: str
    source_kind: str


def _canonical_cik(value: str | int) -> str:
    text = str(value).strip()
    if not CIK_PATTERN.fullmatch(text) or int(text) <= 0:
        raise PeriodicFilingDataError(f"invalid issuer CIK: {value!r}")
    return f"{int(text):010d}"


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PeriodicFilingDataError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _read_stable(path: Path) -> bytes:
    if path.is_symlink():
        raise PeriodicFilingDataError(f"symbolic links are not accepted: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    before = path.stat()
    content = path.read_bytes()
    after = path.stat()
    before_state = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_state = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_state != after_state or len(content) != after.st_size:
        raise PeriodicFilingDataError(f"file changed while being read: {path}")
    return content


def _read_gzip_json(path: Path, *, source_kind: str) -> _SourcePayload:
    compressed = _read_stable(path)
    try:
        content = gzip.decompress(compressed)
    except (EOFError, OSError) as exc:
        raise PeriodicFilingDataError(f"invalid gzip source: {path}") from exc
    try:
        decoded = content.decode("utf-8", errors="strict")
        payload = json.loads(decoded, object_pairs_hook=_strict_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PeriodicFilingDataError(f"invalid SEC submissions JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise PeriodicFilingDataError(
            f"SEC submissions JSON root is not an object: {path}"
        )
    return _SourcePayload(
        payload=payload,
        path=path.resolve(),
        file_sha256=sha256_bytes(compressed),
        content_sha256=sha256_bytes(content),
        source_kind=source_kind,
    )


def _parse_iso_date(value: Any, *, field: str, source: Path) -> pd.Timestamp:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise PeriodicFilingDataError(
            f"invalid {field} in {source}: {value!r}"
        )
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise PeriodicFilingDataError(
            f"invalid {field} in {source}: {value!r}"
        ) from exc
    if parsed.strftime("%Y-%m-%d") != value:
        raise PeriodicFilingDataError(
            f"non-canonical {field} in {source}: {value!r}"
        )
    return parsed


def validate_acceptance_at(value: Any, *, source: str | Path = "header") -> pd.Timestamp:
    """Validate the exact UTC timestamp syntax used by SEC submissions JSON."""

    if not isinstance(value, str) or not ACCEPTANCE_PATTERN.fullmatch(value):
        raise PeriodicFilingDataError(
            f"invalid acceptanceDateTime in {source}: {value!r}"
        )
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise PeriodicFilingDataError(
            f"invalid acceptanceDateTime in {source}: {value!r}"
        ) from exc
    if parsed.tzinfo is None:
        raise PeriodicFilingDataError(
            f"acceptanceDateTime is not timezone-aware in {source}: {value!r}"
        )
    return parsed.tz_convert("UTC")


def validate_primary_document(value: Any) -> str:
    """Require a single safe HTML basename, never a path or URL."""

    if not isinstance(value, str) or not PRIMARY_DOCUMENT_PATTERN.fullmatch(value):
        raise PeriodicFilingDataError(f"invalid primaryDocument: {value!r}")
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise PeriodicFilingDataError(f"invalid primaryDocument: {value!r}")
    return value


def build_primary_document_url(
    *, issuer_cik: str | int, accession: str, primary_document: str
) -> str:
    """Build the canonical SEC Archives URL for a periodic filing's HTML."""

    cik = _canonical_cik(issuer_cik)
    accession = str(accession)
    if not ACCESSION_PATTERN.fullmatch(accession):
        raise PeriodicFilingDataError(f"invalid accession: {accession!r}")
    document = validate_primary_document(primary_document)
    compact = accession.replace("-", "")
    return f"{ARCHIVE_ROOT}/{int(cik)}/{compact}/{document}"


def build_acceptance_header_url(*, issuer_cik: str | int, accession: str) -> str:
    """Build the canonical raw SGML header URL for an accession."""

    cik = _canonical_cik(issuer_cik)
    accession = str(accession)
    if not ACCESSION_PATTERN.fullmatch(accession):
        raise PeriodicFilingDataError(f"invalid accession: {accession!r}")
    compact = accession.replace("-", "")
    return f"{ARCHIVE_ROOT}/{int(cik)}/{compact}/{accession}.hdr.sgml"


def parse_raw_acceptance_header(content: bytes) -> pd.Timestamp:
    """Parse one EDGAR SGML acceptance wall clock as America/New_York."""

    matches = ACCEPTANCE_HEADER_PATTERN.findall(content)
    if len(matches) != 1:
        raise PeriodicFilingDataError(
            f"expected one ACCEPTANCE-DATETIME in raw SGML header, found {len(matches)}"
        )
    value = matches[0].decode("ascii")
    local = pd.to_datetime(value, format="%Y%m%d%H%M%S", errors="raise")
    try:
        return local.tz_localize(
            "America/New_York", ambiguous="raise", nonexistent="raise"
        ).tz_convert("UTC")
    except (TypeError, ValueError) as exc:
        raise PeriodicFilingDataError(
            f"raw SGML acceptance clock is ambiguous or nonexistent: {value}"
        ) from exc


def _header_table(payload: Mapping[str, Any], *, source: Path) -> Mapping[str, Any]:
    if "filings" not in payload:
        return payload
    filings = payload.get("filings")
    if not isinstance(filings, Mapping):
        raise PeriodicFilingDataError(f"filings is not an object in {source}")
    recent = filings.get("recent")
    if not isinstance(recent, Mapping):
        raise PeriodicFilingDataError(f"filings.recent is not an object in {source}")
    return recent


def _source_rows(
    source: _SourcePayload,
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
    invalid_types = [
        column for column in REQUIRED_HEADER_COLUMNS if not isinstance(table[column], list)
    ]
    if invalid_types:
        raise PeriodicFilingDataError(
            f"SEC submissions columns are not lists in {source.path}: {invalid_types}"
        )
    lengths = {column: len(table[column]) for column in REQUIRED_HEADER_COLUMNS}
    if len(set(lengths.values())) != 1:
        raise PeriodicFilingDataError(
            f"SEC submissions column lengths differ in {source.path}: {lengths}"
        )

    rows: list[dict[str, Any]] = []
    for index in range(next(iter(lengths.values()), 0)):
        form = table["form"][index]
        if form not in PERIODIC_FORMS:
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


def _filter_date(value: str | pd.Timestamp | None, *, name: str) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise PeriodicFilingDataError(f"invalid {name}: {value!r}") from exc
    if parsed.tzinfo is not None:
        parsed = parsed.tz_convert("UTC").tz_localize(None)
    if parsed != parsed.normalize():
        raise PeriodicFilingDataError(f"{name} must be a date, not a timestamp")
    return parsed


def _declared_history_sources(
    main: _SourcePayload,
    *,
    cache_root: Path,
    issuer_cik: str,
    start_date: pd.Timestamp | None,
    end_date: pd.Timestamp | None,
    allowed_source_paths: set[Path] | None = None,
) -> list[_SourcePayload]:
    filings = main.payload.get("filings")
    if not isinstance(filings, Mapping):
        raise PeriodicFilingDataError(f"filings is not an object in {main.path}")
    files = filings.get("files")
    if not isinstance(files, list):
        raise PeriodicFilingDataError(f"filings.files is not a list in {main.path}")

    seen: set[str] = set()
    selected: list[_SourcePayload] = []
    for item in files:
        if not isinstance(item, Mapping):
            raise PeriodicFilingDataError(
                f"filings.files entry is not an object in {main.path}"
            )
        name = item.get("name")
        if not isinstance(name, str):
            raise PeriodicFilingDataError(
                f"history filename is missing or invalid in {main.path}"
            )
        match = HISTORY_FILE_PATTERN.fullmatch(name)
        if not match or match.group("cik") != issuer_cik:
            raise PeriodicFilingDataError(
                f"unsafe or mismatched history filename in {main.path}: {name!r}"
            )
        if name in seen:
            raise PeriodicFilingDataError(
                f"duplicate history filename in {main.path}: {name!r}"
            )
        seen.add(name)
        filing_from = _parse_iso_date(
            item.get("filingFrom"), field="filingFrom", source=main.path
        )
        filing_to = _parse_iso_date(
            item.get("filingTo"), field="filingTo", source=main.path
        )
        if filing_from > filing_to:
            raise PeriodicFilingDataError(
                f"reversed history date bounds for {name!r}"
            )
        filing_count = item.get("filingCount")
        if not isinstance(filing_count, int) or isinstance(filing_count, bool) or filing_count < 0:
            raise PeriodicFilingDataError(
                f"invalid filingCount for {name!r}: {filing_count!r}"
            )
        if start_date is not None and filing_to < start_date:
            continue
        if end_date is not None and filing_from > end_date:
            continue
        path = (
            cache_root
            / "submissions_archive"
            / f"CIK{issuer_cik}"
            / f"{name}.gz"
        )
        if allowed_source_paths is not None and path.resolve() not in allowed_source_paths:
            raise PeriodicFilingDataError(
                f"declared SEC history is outside the frozen source manifest: {path}"
            )
        if not path.is_file():
            raise FileNotFoundError(
                f"declared SEC submissions history is absent: {path}"
            )
        selected.append(_read_gzip_json(path, source_kind="history"))
    return selected


def _issuer_main_paths(
    cache_root: Path, issuer_ciks: Iterable[str | int] | None
) -> list[tuple[str, Path]]:
    submissions_dir = cache_root / "submissions"
    if issuer_ciks is not None:
        normalized = [_canonical_cik(value) for value in issuer_ciks]
        if len(normalized) != len(set(normalized)):
            raise PeriodicFilingDataError("issuer_ciks contains duplicates")
        return [
            (cik, submissions_dir / f"CIK{cik}.json.gz")
            for cik in sorted(normalized)
        ]
    paths = sorted(submissions_dir.glob("CIK*.json.gz"))
    if not paths:
        raise FileNotFoundError(f"no SEC submissions caches found in {submissions_dir}")
    result: list[tuple[str, Path]] = []
    for path in paths:
        match = MAIN_FILE_PATTERN.fullmatch(path.name)
        if not match:
            raise PeriodicFilingDataError(f"unexpected submissions filename: {path}")
        result.append((match.group("cik"), path))
    return result


def _deduplicate_inventory(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    comparison_columns = (
        "issuer_cik",
        "form",
        "filing_date",
        "report_date",
        "acceptance_at_raw",
        "acceptance_at",
        "primary_document",
        "primary_document_url",
    )
    conflicts: list[str] = []
    for accession, group in frame.groupby("accession", sort=False):
        if len(group) <= 1:
            continue
        if any(group[column].nunique(dropna=False) != 1 for column in comparison_columns):
            conflicts.append(str(accession))
    if conflicts:
        raise PeriodicFilingDataError(
            f"conflicting periodic filing headers for accessions: {conflicts[:5]}"
        )
    return (
        frame.sort_values(
            ["accession", "source_path"], kind="mergesort"
        )
        .drop_duplicates("accession", keep="first")
        .sort_values(
            ["issuer_cik", "filing_date", "acceptance_at", "accession"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def load_periodic_filing_inventory(
    cache_dir: str | Path,
    *,
    issuer_ciks: Iterable[str | int] | None = None,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    allowed_source_paths: Iterable[str | Path] | None = None,
) -> pd.DataFrame:
    """Load non-amended 10-K/10-Q headers from cached SEC submissions files.

    Every historical file declared by the main submissions snapshot and
    overlapping the requested date range is mandatory.  Missing history,
    malformed arrays, or conflicting duplicate accessions fail closed.
    """

    root = Path(cache_dir)
    allowed = (
        None
        if allowed_source_paths is None
        else {Path(path).resolve() for path in allowed_source_paths}
    )
    if allowed_source_paths is not None and not allowed:
        raise PeriodicFilingDataError("frozen SEC source manifest is empty")
    start = _filter_date(start_date, name="start_date")
    end = _filter_date(end_date, name="end_date")
    if start is not None and end is not None and start > end:
        raise PeriodicFilingDataError("start_date follows end_date")

    rows: list[dict[str, Any]] = []
    for issuer_cik, main_path in _issuer_main_paths(root, issuer_ciks):
        if allowed is not None and main_path.resolve() not in allowed:
            raise PeriodicFilingDataError(
                f"SEC main submissions file is outside the frozen source manifest: {main_path}"
            )
        main = _read_gzip_json(main_path, source_kind="recent")
        payload_cik = _canonical_cik(main.payload.get("cik", ""))
        if payload_cik != issuer_cik:
            raise PeriodicFilingDataError(
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
                _source_rows(
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
    return _deduplicate_inventory(frame)


def _validate_user_agent(value: str | None) -> str:
    user_agent = value or os.getenv("SEC_USER_AGENT")
    if not user_agent or "\r" in user_agent or "\n" in user_agent:
        raise PeriodicFilingDataError(
            "SEC_USER_AGENT must identify the application and contain a real contact email"
        )
    match = EMAIL_PATTERN.search(user_agent)
    if match is None:
        raise PeriodicFilingDataError(
            "SEC_USER_AGENT must identify the application and contain a real contact email"
        )
    email = match.group(0).lower()
    domain = email.rsplit("@", 1)[1]
    placeholder = (
        domain in {"example.com", "example.org", "example.net", "localhost"}
        or domain.endswith(".invalid")
        or domain.endswith(".test")
    )
    identity = (user_agent[: match.start()] + user_agent[match.end() :]).strip(
        " \t()[]{}<>/;,:-_"
    )
    if placeholder or len(re.sub(r"[^A-Za-z0-9]", "", identity)) < 2:
        raise PeriodicFilingDataError(
            "SEC_USER_AGENT must identify the application and contain a real contact email"
        )
    return user_agent


def _validate_primary_html(content: bytes, *, source_url: str) -> None:
    if len(content) < 32:
        raise PeriodicFilingDataError(
            f"SEC primary document is empty or truncated: {source_url}"
        )
    sample = content[:65536].lower()
    if any(marker in sample for marker in SEC_BLOCK_PAGE_MARKERS):
        raise PeriodicFilingDataError(
            f"SEC returned an access-block page instead of a filing: {source_url}"
        )
    if b"<html" not in sample and b"<!doctype html" not in sample:
        raise PeriodicFilingDataError(
            f"SEC primary document is not recognizable HTML: {source_url}"
        )


def _cached_primary_content(path: Path, *, source_url: str) -> bytes:
    match = re.fullmatch(r"(?P<sha>[0-9a-f]{64})\.html\.gz", path.name)
    if match is None:
        raise PeriodicFilingDataError(
            f"unexpected primary-document cache filename: {path}"
        )
    compressed = _read_stable(path)
    try:
        content = gzip.decompress(compressed)
    except (EOFError, OSError) as exc:
        raise PeriodicFilingDataError(
            f"invalid cached primary-document gzip: {path}"
        ) from exc
    digest = sha256_bytes(content)
    if digest != match.group("sha"):
        raise PeriodicFilingDataError(
            f"cached primary-document hash conflicts with filename: {path}"
        )
    _validate_primary_html(content, source_url=source_url)
    return content


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_sidecar_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    raw = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb", prefix=".partial-", suffix=".tmp", dir=path.parent, delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
            _fsync_directory(path.parent)
        except FileExistsError:
            if _read_stable(path) != raw:
                raise PeriodicFilingDataError(f"cache provenance sidecar conflict: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _verify_sidecar(path: Path, expected: Mapping[str, Any]) -> None:
    if not path.is_file() or path.is_symlink():
        raise PeriodicFilingDataError(f"cache provenance sidecar is absent: {path}")
    try:
        payload = json.loads(
            _read_stable(path).decode("utf-8", errors="strict"),
            object_pairs_hook=_strict_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PeriodicFilingDataError(f"invalid cache provenance sidecar: {path}") from exc
    if payload != dict(expected):
        raise PeriodicFilingDataError(f"cache provenance sidecar mismatch: {path}")


def _write_primary_atomic(path: Path, content: bytes, *, source_url: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compressed = gzip.compress(content, compresslevel=6, mtime=0)
    handle = tempfile.NamedTemporaryFile(
        mode="wb", prefix=".partial-", suffix=".tmp", dir=path.parent, delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(compressed)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
            _fsync_directory(path.parent)
        except FileExistsError:
            existing = _cached_primary_content(path, source_url=source_url)
            if existing != content:
                raise PeriodicFilingDataError(
                    f"content-addressed primary-document cache conflict: {path}"
                )
    finally:
        if temporary.exists():
            temporary.unlink()
    verified = _cached_primary_content(path, source_url=source_url)
    if verified != content:  # pragma: no cover - defensive post-write check
        raise PeriodicFilingDataError(f"post-write verification failed: {path}")


class SecPeriodicFilingsClient:
    """Rate-limited, resumable SEC primary-HTML downloader."""

    def __init__(
        self,
        *,
        cache_dir: str | Path = "data/cache/sec/periodic_filings",
        user_agent: str | None = None,
        timeout: int = 60,
        min_interval_seconds: float = 0.20,
        max_attempts: int = 5,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.user_agent = _validate_user_agent(user_agent)
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if min_interval_seconds < 0.1:
            raise ValueError("SEC request interval must be at least 0.1 seconds")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.timeout = int(timeout)
        self.min_interval_seconds = float(min_interval_seconds)
        self.max_attempts = int(max_attempts)
        self._last_request = 0.0
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept-Encoding": "gzip, deflate",
                "Accept": "text/html,application/xhtml+xml,text/plain,application/sgml",
            }
        )

    def _retry_delay(self, response: requests.Response | None, attempt: int) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    return min(max(float(retry_after), 0.0), 60.0)
                except ValueError:
                    pass
        return min(0.5 * (2**attempt), 8.0)

    def _request(
        self,
        source_url: str,
        *,
        allowed_media_types: frozenset[str],
        artifact_name: str,
    ) -> requests.Response:
        response: requests.Response | None = None
        last_error: requests.RequestException | None = None
        for attempt in range(self.max_attempts):
            delay = self.min_interval_seconds - (
                time.monotonic() - self._last_request
            )
            if delay > 0:
                time.sleep(delay)
            try:
                response = self.session.get(
                    source_url,
                    timeout=self.timeout,
                    allow_redirects=False,
                )
                last_error = None
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_error = exc
                response = None
            finally:
                self._last_request = time.monotonic()

            retryable = response is None or response.status_code in RETRYABLE_HTTP_STATUS
            if retryable and attempt + 1 < self.max_attempts:
                time.sleep(self._retry_delay(response, attempt))
                continue
            break
        if response is None:
            assert last_error is not None
            raise last_error
        if 300 <= response.status_code < 400:
            raise PeriodicFilingDataError(
                f"SEC {artifact_name} request redirected: {source_url}"
            )
        response.raise_for_status()
        final_url = str(getattr(response, "url", source_url) or source_url)
        if final_url != source_url:
            raise PeriodicFilingDataError(
                f"SEC response URL conflicts with requested filing: {final_url!r}"
            )
        media_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if media_type not in allowed_media_types:
            raise PeriodicFilingDataError(
                f"unexpected SEC {artifact_name} media type: {media_type!r}"
            )
        return response

    def _artifact_directory(self, *, issuer_cik: str, accession: str) -> Path:
        return (
            self.cache_dir
            / "primary_documents"
            / f"CIK{issuer_cik}"
            / accession
        )

    def _existing_artifact(
        self,
        directory: Path,
        *,
        issuer_cik: str,
        accession: str,
        primary_document: str,
        source_url: str,
        expected_sha256: str | None,
    ) -> tuple[bytes, Path] | None:
        if not directory.exists():
            return None
        if directory.is_symlink() or not directory.is_dir():
            raise PeriodicFilingDataError(
                f"invalid primary-document cache directory: {directory}"
            )
        candidates = sorted(directory.glob("*.html.gz"))
        if len(candidates) > 1:
            raise PeriodicFilingDataError(
                f"conflicting cached primary documents for one accession: {directory}"
            )
        if not candidates:
            return None
        path = candidates[0]
        content = _cached_primary_content(path, source_url=source_url)
        digest = sha256_bytes(content)
        if expected_sha256 is not None and digest != expected_sha256:
            raise PeriodicFilingDataError(
                f"cached primary-document hash differs from expected hash: {path}"
            )
        _verify_sidecar(
            path.with_name(f"{digest}.meta.json"),
            {
                "schema_version": "sec_periodic_primary_cache_provenance_v1",
                "issuer_cik": issuer_cik,
                "accession": accession,
                "primary_document": primary_document,
                "source_url": source_url,
                "content_sha256": digest,
                "raw_bytes": len(content),
            },
        )
        return content, path

    @staticmethod
    def _record(
        *,
        issuer_cik: str,
        accession: str,
        primary_document: str,
        acceptance_at: pd.Timestamp,
        source_url: str,
        content: bytes,
        path: Path,
        cache_status: str,
        media_type: str | None,
    ) -> DatasetRecord:
        compressed = _read_stable(path)
        return DatasetRecord(
            provider="SEC EDGAR Archives",
            dataset=f"periodic_primary_document:{accession}",
            retrieved_at=utc_now(),
            source_url=source_url,
            sha256=sha256_bytes(content),
            rows=1,
            finality="archived_as_filed_primary_document",
            revision_policy="immutable content-addressed gzip; conflicts fail closed",
            notes={
                "issuer_cik": issuer_cik,
                "accession": accession,
                "primary_document": primary_document,
                "acceptance_at": acceptance_at.isoformat(),
                "cache_status": cache_status,
                "cache_path": path.resolve().as_posix(),
                "raw_bytes": len(content),
                "compressed_bytes": len(compressed),
                "compressed_sha256": sha256_bytes(compressed),
                "media_type": media_type,
                "claim": "acquisition only; acceptance is not first-public-time evidence",
            },
        )

    def download_primary_html(
        self,
        *,
        issuer_cik: str | int,
        accession: str,
        primary_document: str,
        acceptance_at: str,
        expected_sha256: str | None = None,
        refresh: bool = False,
    ) -> tuple[bytes, DatasetRecord, Path]:
        """Download or resume one immutable SEC primary HTML artifact."""

        cik = _canonical_cik(issuer_cik)
        accession = str(accession)
        if not ACCESSION_PATTERN.fullmatch(accession):
            raise PeriodicFilingDataError(f"invalid accession: {accession!r}")
        document = validate_primary_document(primary_document)
        accepted = validate_acceptance_at(acceptance_at, source=accession)
        if expected_sha256 is not None:
            expected_sha256 = str(expected_sha256).lower()
            if not CONTENT_HASH_PATTERN.fullmatch(expected_sha256):
                raise PeriodicFilingDataError(
                    f"invalid expected primary-document sha256: {expected_sha256!r}"
                )
        source_url = build_primary_document_url(
            issuer_cik=cik,
            accession=accession,
            primary_document=document,
        )
        directory = self._artifact_directory(issuer_cik=cik, accession=accession)
        existing = self._existing_artifact(
            directory,
            issuer_cik=cik,
            accession=accession,
            primary_document=document,
            source_url=source_url,
            expected_sha256=expected_sha256,
        )
        if existing is not None and not refresh:
            content, path = existing
            record = self._record(
                issuer_cik=cik,
                accession=accession,
                primary_document=document,
                acceptance_at=accepted,
                source_url=source_url,
                content=content,
                path=path,
                cache_status="cached_verified",
                media_type=None,
            )
            return content, record, path

        response = self._request(
            source_url,
            allowed_media_types=ALLOWED_HTML_MEDIA_TYPES,
            artifact_name="primary-document",
        )
        content = response.content
        _validate_primary_html(content, source_url=source_url)
        digest = sha256_bytes(content)
        if expected_sha256 is not None and digest != expected_sha256:
            raise PeriodicFilingDataError(
                "downloaded primary-document hash differs from expected hash"
            )
        if existing is not None:
            existing_content, existing_path = existing
            if existing_content != content:
                raise PeriodicFilingDataError(
                    "SEC primary-document content changed for an existing accession"
                )
            path = existing_path
            cache_status = "refreshed_identical"
        else:
            path = directory / f"{digest}.html.gz"
            _write_primary_atomic(path, content, source_url=source_url)
            _write_sidecar_atomic(
                path.with_name(f"{digest}.meta.json"),
                {
                    "schema_version": "sec_periodic_primary_cache_provenance_v1",
                    "issuer_cik": cik,
                    "accession": accession,
                    "primary_document": document,
                    "source_url": source_url,
                    "content_sha256": digest,
                    "raw_bytes": len(content),
                },
            )
            cache_status = "downloaded"
        media_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        record = self._record(
            issuer_cik=cik,
            accession=accession,
            primary_document=document,
            acceptance_at=accepted,
            source_url=source_url,
            content=content,
            path=path,
            cache_status=cache_status,
            media_type=media_type,
        )
        return content, record, path

    def download_acceptance_header(
        self,
        *,
        issuer_cik: str | int,
        accession: str,
        api_acceptance_at: str,
        maximum_delta_seconds: float = 1.0,
    ) -> tuple[bytes, pd.Timestamp, DatasetRecord, Path]:
        """Download, cache, and reconcile one raw ``.hdr.sgml`` clock."""

        cik = _canonical_cik(issuer_cik)
        accession = str(accession)
        if not ACCESSION_PATTERN.fullmatch(accession):
            raise PeriodicFilingDataError(f"invalid accession: {accession!r}")
        api_clock = validate_acceptance_at(api_acceptance_at, source=accession)
        if maximum_delta_seconds < 0:
            raise ValueError("maximum_delta_seconds must be nonnegative")
        source_url = build_acceptance_header_url(
            issuer_cik=cik, accession=accession
        )
        directory = (
            self.cache_dir / "acceptance_headers" / f"CIK{cik}" / accession
        )
        if directory.exists() and (directory.is_symlink() or not directory.is_dir()):
            raise PeriodicFilingDataError(
                f"invalid acceptance-header cache directory: {directory}"
            )
        candidates = sorted(directory.glob("*.hdr.sgml.gz")) if directory.exists() else []
        if len(candidates) > 1:
            raise PeriodicFilingDataError(
                f"conflicting cached acceptance headers for one accession: {directory}"
            )
        media_type: str | None = None
        cache_status = "cached_verified"
        if candidates:
            path = candidates[0]
            match = re.fullmatch(r"(?P<sha>[0-9a-f]{64})\.hdr\.sgml\.gz", path.name)
            if match is None:
                raise PeriodicFilingDataError(
                    f"unexpected acceptance-header cache filename: {path}"
                )
            try:
                content = gzip.decompress(_read_stable(path))
            except (EOFError, OSError) as exc:
                raise PeriodicFilingDataError(
                    f"invalid cached acceptance-header gzip: {path}"
                ) from exc
            if sha256_bytes(content) != match.group("sha"):
                raise PeriodicFilingDataError(
                    f"cached acceptance-header hash conflicts with filename: {path}"
                )
            _verify_sidecar(
                path.with_name(f"{match.group('sha')}.meta.json"),
                {
                    "schema_version": "sec_periodic_acceptance_cache_provenance_v1",
                    "issuer_cik": cik,
                    "accession": accession,
                    "source_url": source_url,
                    "content_sha256": match.group("sha"),
                    "raw_bytes": len(content),
                },
            )
        else:
            response = self._request(
                source_url,
                allowed_media_types=ALLOWED_SGML_MEDIA_TYPES,
                artifact_name="acceptance-header",
            )
            content = response.content
            media_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            digest = sha256_bytes(content)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{digest}.hdr.sgml.gz"
            compressed = gzip.compress(content, compresslevel=6, mtime=0)
            if path.exists() and _read_stable(path) != compressed:
                raise PeriodicFilingDataError(
                    f"content-addressed acceptance-header conflict: {path}"
                )
            if not path.exists():
                handle = tempfile.NamedTemporaryFile(
                    mode="wb", prefix=".partial-", suffix=".tmp", dir=directory, delete=False
                )
                temporary = Path(handle.name)
                try:
                    with handle:
                        handle.write(compressed)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.link(temporary, path)
                    _fsync_directory(directory)
                except FileExistsError:
                    if _read_stable(path) != compressed:
                        raise PeriodicFilingDataError(
                            f"content-addressed acceptance-header conflict: {path}"
                        )
                finally:
                    temporary.unlink(missing_ok=True)
            _write_sidecar_atomic(
                path.with_name(f"{digest}.meta.json"),
                {
                    "schema_version": "sec_periodic_acceptance_cache_provenance_v1",
                    "issuer_cik": cik,
                    "accession": accession,
                    "source_url": source_url,
                    "content_sha256": digest,
                    "raw_bytes": len(content),
                },
            )
            cache_status = "downloaded"
        raw_clock = parse_raw_acceptance_header(content)
        delta = abs((raw_clock - api_clock).total_seconds())
        if delta > maximum_delta_seconds:
            raise PeriodicFilingDataError(
                f"API/raw acceptance clock conflict for {accession}: {delta} seconds"
            )
        compressed = _read_stable(path)
        record = DatasetRecord(
            provider="SEC EDGAR Archives",
            dataset=f"acceptance_header:{accession}",
            retrieved_at=utc_now(),
            source_url=source_url,
            sha256=sha256_bytes(content),
            rows=1,
            finality="archived_raw_sgml_header",
            revision_policy="immutable content-addressed gzip; conflicts fail closed",
            notes={
                "issuer_cik": cik,
                "accession": accession,
                "api_acceptance_at": api_clock.isoformat(),
                "raw_header_acceptance_at": raw_clock.isoformat(),
                "api_raw_delta_seconds": delta,
                "cache_status": cache_status,
                "cache_path": path.resolve().as_posix(),
                "compressed_sha256": sha256_bytes(compressed),
                "media_type": media_type,
                "claim": "verified acceptance clock; not an exact first-public timestamp",
            },
        )
        return content, raw_clock, record, path


__all__ = [
    "ARCHIVE_ROOT",
    "INVENTORY_COLUMNS",
    "PERIODIC_FORMS",
    "PeriodicFilingDataError",
    "SecPeriodicFilingsClient",
    "build_acceptance_header_url",
    "build_primary_document_url",
    "load_periodic_filing_inventory",
    "parse_raw_acceptance_header",
    "validate_acceptance_at",
    "validate_primary_document",
]
