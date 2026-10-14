"""Offline, fail-closed ingestion for manually acquired SEC Form 4 assets.

The module never performs a network request.  It verifies an immutable input
manifest, validates the supplied bytes against the expected SEC data shape and
URL contract, and writes content-addressed cache copies plus an import receipt.
Manual transport is operator supplied and therefore cannot, by itself, prove
that bytes came from the declared URL.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

import pandas as pd

from alpha_factor_engine.data.sec_insider import (
    ACCESSION_PATTERN,
    INSIDER_DATASET_PAGE,
    AcceptanceClockError,
    InsiderDataSchemaError,
    build_acceptance_map,
    daily_master_url,
    discover_quarter_links,
    filing_archive_urls,
    parse_acceptance_header,
    parse_daily_master_index,
    parse_insider_zip,
)


OFFLINE_IMPORT_SCHEMA = "sec_form4_offline_import_manifest_v1"
OFFLINE_RECEIPT_SCHEMA = "sec_form4_offline_import_receipt_v1"
ACQUISITION_PLAN_SCHEMA = "sec_form4_bulk_acquisition_plan_v1"
RESEARCH_STATUS = "NOT VALIDATED"
TRANSPORT_ATTESTATION = "operator_supplied_manual_download"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
CIK_PATTERN = re.compile(r"\d{1,10}")
QUARTER_PATTERN = re.compile(r"20\d{2}q[1-4]")
DATE_ROLE_PATTERN = re.compile(r"daily_master:(\d{4}-\d{2}-\d{2})")
BULK_ROLE_PATTERN = re.compile(r"bulk:(20\d{2}q[1-4])")
RAW_ROLE_PATTERN = re.compile(r"raw_submission:(\d{10}-\d{2}-\d{6})")
SUBMISSIONS_ROLE_PATTERN = re.compile(r"submissions:(\d{10}):(main|history-\d{3})")


class OfflineImportError(ValueError):
    """Raised when an offline import bundle is ambiguous or invalid."""


@dataclass(frozen=True)
class VerifiedOfflineRecord:
    role: str
    path: Path
    content: bytes
    byte_count: int
    sha256: str
    media_type: str
    source_url: str
    metadata: Mapping[str, Any]
    validation: Mapping[str, Any]


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _read_stable(path: Path) -> bytes:
    if path.is_symlink():
        raise OfflineImportError(f"offline input cannot be a symlink: {path}")
    try:
        before = path.stat()
        content = path.read_bytes()
        after = path.stat()
    except OSError as exc:
        raise OfflineImportError(f"cannot read offline input: {path}") from exc
    before_signature = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_signature = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_signature != after_signature:
        raise OfflineImportError(f"offline input changed while reading: {path}")
    return content


def _strict_json(content: bytes, *, source: str) -> Mapping[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise OfflineImportError(
                    f"{source} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise OfflineImportError(f"{source} contains non-finite JSON {value}")

    try:
        payload = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except OfflineImportError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise OfflineImportError(f"{source} is not strict JSON") from exc
    if not isinstance(payload, Mapping):
        raise OfflineImportError(f"{source} must be a JSON object")
    return payload


def _exact_keys(payload: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    actual = set(payload)
    if actual != expected:
        raise OfflineImportError(
            f"{name} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _official_url(value: object, *, allowed_hosts: set[str]) -> str:
    url = str(value).strip()
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in allowed_hosts
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.query
        or parsed.fragment
    ):
        raise OfflineImportError(f"source URL is not an allowed SEC HTTPS URL: {url!r}")
    return url


def _normalize_cik(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not CIK_PATTERN.fullmatch(text):
        raise OfflineImportError(f"{name} is not a valid CIK: {value!r}")
    return f"{int(text):010d}"


def _validate_bulk(record: Mapping[str, Any], content: bytes) -> Mapping[str, Any]:
    match = BULK_ROLE_PATTERN.fullmatch(str(record["role"]))
    assert match is not None
    quarter = match.group(1)
    metadata = record["metadata"]
    _exact_keys(metadata, {"quarter"}, name=f"{record['role']} metadata")
    if str(metadata["quarter"]).lower() != quarter:
        raise OfflineImportError(f"{record['role']} quarter metadata conflicts with role")
    url = _official_url(record["source_url"], allowed_hosts={"www.sec.gov"})
    path_match = re.fullmatch(
        r"/files/(?:structureddata|datastandardsinnovation)/data/"
        r"insider-transactions-data-sets/(20\d{2}q[1-4])_form345\.zip",
        urlparse(url).path,
    )
    if path_match is None or path_match.group(1) != quarter:
        raise OfflineImportError(
            f"{record['role']} source URL does not identify the same official quarter"
        )
    parsed = parse_insider_zip(content, quarter=quarter)
    return {
        "asset_type": "bulk_quarter_zip",
        "quarter": quarter,
        "submissions": int(len(parsed.submissions)),
        "form4_submissions": int(
            parsed.submissions["DOCUMENT_TYPE"].isin(["4", "4/A"]).sum()
        ),
        "original_form4_submissions": int(
            parsed.submissions["DOCUMENT_TYPE"].eq("4").sum()
        ),
        "reporting_owners": int(len(parsed.reporting_owners)),
        "nonderivative_transactions": int(len(parsed.nonderivative_transactions)),
        "footnotes": int(len(parsed.footnotes)),
        "archive_sha256": parsed.archive_sha256,
        "metadata_filename": parsed.metadata_filename,
        "quarantined_optional_date_values": sum(
            int(issue.get("invalid_count", 0))
            for issue in parsed.data_quality_issues
        ),
        "normalized_cik_values": sum(
            int(issue.get("normalized_count", 0))
            for issue in parsed.data_quality_issues
        ),
        "data_quality_issues": list(parsed.data_quality_issues),
    }


def _validate_daily_master(
    record: Mapping[str, Any], content: bytes
) -> Mapping[str, Any]:
    match = DATE_ROLE_PATTERN.fullmatch(str(record["role"]))
    assert match is not None
    role_date = match.group(1)
    metadata = record["metadata"]
    _exact_keys(
        metadata,
        {"index_date", "last_modified_at", "etag"},
        name=f"{record['role']} metadata",
    )
    index_date = pd.Timestamp(metadata["index_date"])
    if (
        pd.isna(index_date)
        or index_date.tzinfo is not None
        or index_date.normalize() != index_date
        or index_date.date().isoformat() != role_date
    ):
        raise OfflineImportError(f"{record['role']} has conflicting index_date")
    modified = pd.Timestamp(metadata["last_modified_at"])
    if pd.isna(modified) or modified.tzinfo is None:
        raise OfflineImportError(
            f"{record['role']} last_modified_at must be timezone-aware"
        )
    url = _official_url(record["source_url"], allowed_hosts={"www.sec.gov"})
    if url != daily_master_url(index_date):
        raise OfflineImportError(f"{record['role']} source URL is not canonical")
    snapshot = parse_daily_master_index(
        content,
        index_date=index_date,
        source_url=url,
        last_modified_at=modified,
        etag=None if metadata["etag"] is None else str(metadata["etag"]),
    )
    return {
        "asset_type": "daily_master_index",
        "index_date": role_date,
        "last_modified_at": snapshot.last_modified_at.isoformat(),
        "filings": int(len(snapshot.filings)),
        "form4_rows": int(snapshot.filings["form"].isin(["4", "4/A"]).sum()),
        "content_sha256": snapshot.content_sha256,
    }


def _validate_raw_submission(
    record: Mapping[str, Any], content: bytes
) -> Mapping[str, Any]:
    match = RAW_ROLE_PATTERN.fullmatch(str(record["role"]))
    assert match is not None
    accession = match.group(1)
    metadata = record["metadata"]
    _exact_keys(
        metadata,
        {"accession", "archive_cik", "issuer_cik"},
        name=f"{record['role']} metadata",
    )
    if str(metadata["accession"]) != accession:
        raise OfflineImportError(f"{record['role']} accession metadata conflicts with role")
    archive_cik = _normalize_cik(metadata["archive_cik"], name="archive_cik")
    issuer_cik = _normalize_cik(metadata["issuer_cik"], name="issuer_cik")
    url = _official_url(record["source_url"], allowed_hosts={"www.sec.gov"})
    expected_url = filing_archive_urls(
        issuer_cik=archive_cik, accession=accession
    )["complete_submission"]
    if url != expected_url:
        raise OfflineImportError(f"{record['role']} source URL is not canonical")
    accepted_at = parse_acceptance_header(content)
    text_head = content[:4096].decode("latin-1", errors="strict")
    if "<SEC-DOCUMENT>" not in text_head or accession not in content.decode(
        "latin-1", errors="strict"
    ):
        raise OfflineImportError(
            f"{record['role']} does not look like its declared complete submission"
        )
    return {
        "asset_type": "raw_complete_submission",
        "accession": accession,
        "archive_cik": archive_cik,
        "issuer_cik": issuer_cik,
        "accepted_at_raw": accepted_at.isoformat(),
    }


def _validate_submissions_json(
    record: Mapping[str, Any], content: bytes
) -> Mapping[str, Any]:
    match = SUBMISSIONS_ROLE_PATTERN.fullmatch(str(record["role"]))
    assert match is not None
    role_cik, role_part = match.groups()
    metadata = record["metadata"]
    _exact_keys(
        metadata,
        {"cik", "part"},
        name=f"{record['role']} metadata",
    )
    cik = _normalize_cik(metadata["cik"], name="cik")
    part = str(metadata["part"])
    if cik != role_cik or part != role_part:
        raise OfflineImportError(f"{record['role']} metadata conflicts with role")
    url = _official_url(record["source_url"], allowed_hosts={"data.sec.gov"})
    if part == "main":
        expected = f"https://data.sec.gov/submissions/CIK{cik}.json"
    else:
        sequence = part.removeprefix("history-")
        expected = (
            f"https://data.sec.gov/submissions/"
            f"CIK{cik}-submissions-{sequence}.json"
        )
    if url != expected:
        raise OfflineImportError(f"{record['role']} source URL is not canonical")
    payload = _strict_json(content, source=record["role"])
    acceptance = build_acceptance_map([payload], sources=[url])
    return {
        "asset_type": "submissions_json",
        "cik": cik,
        "part": part,
        "filing_rows": int(len(acceptance)),
        "form4_rows": int(acceptance["form"].isin(["4", "4/A"]).sum()),
    }


def _validate_content(record: Mapping[str, Any], content: bytes) -> Mapping[str, Any]:
    role = str(record["role"])
    try:
        if BULK_ROLE_PATTERN.fullmatch(role):
            return _validate_bulk(record, content)
        if DATE_ROLE_PATTERN.fullmatch(role):
            return _validate_daily_master(record, content)
        if RAW_ROLE_PATTERN.fullmatch(role):
            return _validate_raw_submission(record, content)
        if SUBMISSIONS_ROLE_PATTERN.fullmatch(role):
            return _validate_submissions_json(record, content)
    except (InsiderDataSchemaError, AcceptanceClockError, ValueError) as exc:
        if isinstance(exc, OfflineImportError):
            raise
        raise OfflineImportError(f"{role} content validation failed: {exc}") from exc
    raise OfflineImportError(f"unsupported offline SEC role: {role!r}")


def load_verified_offline_bundle(
    manifest_path: str | Path,
) -> tuple[Mapping[str, Any], str, list[VerifiedOfflineRecord]]:
    """Verify one offline bundle and return the exact validated byte snapshots."""

    manifest = Path(manifest_path).resolve()
    manifest_content = _read_stable(manifest)
    payload = _strict_json(manifest_content, source=str(manifest))
    _exact_keys(
        payload,
        {
            "schema_version",
            "snapshot_id",
            "created_at",
            "transport_attestation",
            "records",
        },
        name="offline import manifest",
    )
    if payload["schema_version"] != OFFLINE_IMPORT_SCHEMA:
        raise OfflineImportError(
            f"unsupported offline import schema: {payload['schema_version']!r}"
        )
    if payload["transport_attestation"] != TRANSPORT_ATTESTATION:
        raise OfflineImportError(
            "offline import must state operator-supplied manual transport"
        )
    if not str(payload["snapshot_id"]).strip():
        raise OfflineImportError("snapshot_id cannot be empty")
    created_at = pd.Timestamp(payload["created_at"])
    if pd.isna(created_at) or created_at.tzinfo is None:
        raise OfflineImportError("created_at must be timezone-aware")
    raw_records = payload["records"]
    if not isinstance(raw_records, list) or not raw_records:
        raise OfflineImportError("offline import records must be a non-empty list")

    base = manifest.parent.resolve()
    seen_roles: set[str] = set()
    seen_paths: set[Path] = set()
    verified: list[VerifiedOfflineRecord] = []
    record_keys = {
        "role",
        "path",
        "bytes",
        "sha256",
        "media_type",
        "source_url",
        "metadata",
    }
    for position, raw in enumerate(raw_records):
        if not isinstance(raw, Mapping):
            raise OfflineImportError(f"record {position} is not an object")
        _exact_keys(raw, record_keys, name=f"offline import record {position}")
        role = str(raw["role"]).strip()
        if role in seen_roles:
            raise OfflineImportError(f"duplicate offline import role: {role}")
        seen_roles.add(role)
        relative = PurePosixPath(str(raw["path"]))
        if relative.is_absolute() or ".." in relative.parts or "\\" in str(raw["path"]):
            raise OfflineImportError(f"unsafe offline import path: {raw['path']!r}")
        path = (base / Path(*relative.parts)).resolve()
        try:
            path.relative_to(base)
        except ValueError as exc:
            raise OfflineImportError(f"offline input escapes bundle: {path}") from exc
        if path in seen_paths:
            raise OfflineImportError(f"offline input path is duplicated: {path}")
        seen_paths.add(path)
        if not isinstance(raw["bytes"], int) or raw["bytes"] <= 0:
            raise OfflineImportError(f"{role} has an invalid byte count")
        declared_sha = str(raw["sha256"]).lower()
        if not SHA256_PATTERN.fullmatch(declared_sha):
            raise OfflineImportError(f"{role} has an invalid SHA256")
        if not isinstance(raw["metadata"], Mapping):
            raise OfflineImportError(f"{role} metadata must be an object")
        content = _read_stable(path)
        actual_sha = _sha256(content)
        if len(content) != raw["bytes"] or actual_sha != declared_sha:
            raise OfflineImportError(
                f"manifest integrity mismatch for {role}: "
                f"declared={raw['bytes']}/{declared_sha}, "
                f"actual={len(content)}/{actual_sha}"
            )
        validation = _validate_content(raw, content)
        verified.append(
            VerifiedOfflineRecord(
                role=role,
                path=path,
                content=content,
                byte_count=len(content),
                sha256=actual_sha,
                media_type=str(raw["media_type"]),
                source_url=str(raw["source_url"]),
                metadata=dict(raw["metadata"]),
                validation=dict(validation),
            )
        )
    return payload, _sha256(manifest_content), verified


def _cache_path(cache_dir: Path, record: VerifiedOfflineRecord) -> Path:
    if BULK_ROLE_PATTERN.fullmatch(record.role):
        quarter = str(record.validation["quarter"])
        return cache_dir / "bulk" / quarter / f"{record.sha256}.zip"
    if DATE_ROLE_PATTERN.fullmatch(record.role):
        date_key = str(record.validation["index_date"]).replace("-", "")
        suffix = ".idx.gz" if record.content[:2] == b"\x1f\x8b" else ".idx"
        return cache_dir / "daily_master" / date_key / f"{record.sha256}{suffix}"
    if RAW_ROLE_PATTERN.fullmatch(record.role):
        return (
            cache_dir
            / "raw_submissions"
            / f"CIK{record.validation['archive_cik']}"
            / str(record.validation["accession"])
            / f"{record.sha256}.txt"
        )
    if SUBMISSIONS_ROLE_PATTERN.fullmatch(record.role):
        return (
            cache_dir
            / "submissions"
            / f"CIK{record.validation['cik']}"
            / f"{record.sha256}.json"
        )
    raise AssertionError(record.role)


def _write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = _read_stable(path)
        if existing != content:
            raise OfflineImportError(f"content-addressed cache conflict: {path}")
        return
    handle = tempfile.NamedTemporaryFile(dir=path.parent, delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    if _sha256(_read_stable(path)) != _sha256(content):  # pragma: no cover
        raise OfflineImportError(f"post-write verification failed: {path}")


def import_verified_offline_bundle(
    manifest_path: str | Path,
    *,
    cache_dir: str | Path,
    receipt_path: str | Path,
) -> Path:
    """Import verified bytes and write a receipt that permits no alpha claim."""

    manifest = Path(manifest_path).resolve()
    payload, manifest_sha, records = load_verified_offline_bundle(manifest)
    cache_root = Path(cache_dir).resolve()
    rows: list[dict[str, Any]] = []
    for record in records:
        target = _cache_path(cache_root, record)
        _write_atomic(target, record.content)
        rows.append(
            {
                "role": record.role,
                "source_url": record.source_url,
                "bundle_path": record.path.as_posix(),
                "cache_path": target.as_posix(),
                "bytes": record.byte_count,
                "sha256": record.sha256,
                "validation": record.validation,
            }
        )
    output = Path(receipt_path).resolve()
    if output.exists():
        if not verify_offline_import_receipt(output):  # pragma: no cover
            raise OfflineImportError("existing offline receipt did not verify")
        existing = _strict_json(_read_stable(output), source=str(output))
        existing_roles = {
            str(row.get("role")): (row.get("sha256"), row.get("cache_path"))
            for row in existing.get("records", [])
            if isinstance(row, Mapping)
        }
        new_roles = {
            str(row["role"]): (row["sha256"], row["cache_path"])
            for row in rows
        }
        if (
            existing.get("input_manifest_sha256") != manifest_sha
            or existing.get("snapshot_id") != payload["snapshot_id"]
            or existing_roles != new_roles
        ):
            raise OfflineImportError(
                "existing offline receipt belongs to a different import snapshot"
            )
        return output
    receipt = {
        "schema_version": OFFLINE_RECEIPT_SCHEMA,
        "generated_at": datetime.now(UTC).isoformat(),
        "research_status": RESEARCH_STATUS,
        "alpha_claim_permitted": False,
        "input_manifest_path": manifest.as_posix(),
        "input_manifest_sha256": manifest_sha,
        "snapshot_id": payload["snapshot_id"],
        "transport_attestation": payload["transport_attestation"],
        "transport_authenticity_independently_verified": False,
        "records": rows,
        "limitations": [
            "manual transport is operator supplied and not independently authenticated",
            "bulk quarterly data omit the strict historical public-availability clock",
            "no return or alpha claim is authorized by this import receipt",
        ],
    }
    content = (
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    _write_atomic(output, content)
    return output


def verify_offline_import_receipt(receipt_path: str | Path) -> bool:
    """Re-verify the source manifest and every imported cache byte."""

    receipt_path = Path(receipt_path).resolve()
    receipt = _strict_json(_read_stable(receipt_path), source=str(receipt_path))
    if receipt.get("schema_version") != OFFLINE_RECEIPT_SCHEMA:
        raise OfflineImportError("unsupported offline import receipt schema")
    if receipt.get("research_status") != RESEARCH_STATUS:
        raise OfflineImportError("offline import receipt research status drifted")
    if receipt.get("alpha_claim_permitted") is not False:
        raise OfflineImportError("offline import receipt cannot permit alpha claims")
    manifest_path = Path(str(receipt.get("input_manifest_path", ""))).resolve()
    manifest_content = _read_stable(manifest_path)
    if _sha256(manifest_content) != receipt.get("input_manifest_sha256"):
        raise OfflineImportError("offline input manifest hash drifted after import")
    _, _, verified = load_verified_offline_bundle(manifest_path)
    by_role = {record.role: record for record in verified}
    rows = receipt.get("records")
    if not isinstance(rows, list) or set(by_role) != {
        str(row.get("role")) for row in rows if isinstance(row, Mapping)
    }:
        raise OfflineImportError("offline receipt role coverage differs from manifest")
    for row in rows:
        if not isinstance(row, Mapping):
            raise OfflineImportError("offline receipt record is not an object")
        role = str(row["role"])
        source = by_role[role]
        cached = _read_stable(Path(str(row["cache_path"])).resolve())
        if (
            len(cached) != row.get("bytes")
            or _sha256(cached) != row.get("sha256")
            or cached != source.content
        ):
            raise OfflineImportError(f"offline cached bytes drifted for {role}")
    return True


def prepare_bulk_manual_import(
    source_zip: str | Path,
    *,
    quarter: str,
    source_url: str,
    bundle_dir: str | Path,
    snapshot_id: str,
) -> Path:
    """Freeze one manually downloaded official-compatible bulk ZIP as a bundle."""

    normalized = str(quarter).strip().lower()
    if not QUARTER_PATTERN.fullmatch(normalized):
        raise OfflineImportError(f"invalid quarter: {quarter!r}")
    source = Path(source_zip).resolve()
    content = _read_stable(source)
    digest = _sha256(content)
    relative = PurePosixPath("sources") / f"{normalized}_{digest}.zip"
    record = {
        "role": f"bulk:{normalized}",
        "path": relative.as_posix(),
        "bytes": len(content),
        "sha256": digest,
        "media_type": "application/zip",
        "source_url": source_url,
        "metadata": {"quarter": normalized},
    }
    # Validate URL, archive schema, and quarter before mutating the bundle.  An
    # interrupted earlier prepare may have written only the exact source bytes;
    # that state is safe to resume without deleting or overwriting anything.
    _validate_content(record, content)
    root = Path(bundle_dir).resolve()
    target = root / Path(*relative.parts)
    path = root / "offline_import_manifest.json"
    if path.exists():
        payload, _, records = load_verified_offline_bundle(path)
        if (
            payload.get("snapshot_id") != str(snapshot_id)
            or len(records) != 1
            or records[0].role != record["role"]
            or records[0].sha256 != digest
            or records[0].source_url != source_url
        ):
            raise OfflineImportError(
                "existing offline manifest belongs to a different source snapshot"
            )
        return path
    if root.exists():
        unexpected = [
            candidate
            for candidate in root.rglob("*")
            if candidate.is_file() and candidate.resolve() != target.resolve()
        ]
        if unexpected:
            raise OfflineImportError(
                f"bundle directory contains unexpected files: {unexpected[:5]}"
            )
    root.mkdir(parents=True, exist_ok=True)
    _write_atomic(target, content)
    manifest = {
        "schema_version": OFFLINE_IMPORT_SCHEMA,
        "snapshot_id": str(snapshot_id),
        "created_at": datetime.now(UTC).isoformat(),
        "transport_attestation": TRANSPORT_ATTESTATION,
        "records": [record],
    }
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    load_verified_offline_bundle(path)
    return path


def _quarter_sequence(start: str, end: str) -> list[str]:
    if not QUARTER_PATTERN.fullmatch(start) or not QUARTER_PATTERN.fullmatch(end):
        raise OfflineImportError("plan bounds must be YYYYqN")
    start_value = int(start[:4]) * 4 + int(start[-1]) - 1
    end_value = int(end[:4]) * 4 + int(end[-1]) - 1
    if start_value > end_value:
        raise OfflineImportError("plan start quarter follows end quarter")
    return [f"{value // 4:04d}q{value % 4 + 1}" for value in range(start_value, end_value + 1)]


def _validate_bulk_url(url_value: object, quarter: str) -> str:
    url = _official_url(url_value, allowed_hosts={"www.sec.gov"})
    path_match = re.fullmatch(
        r"/files/(?:structureddata|datastandardsinnovation)/data/"
        r"insider-transactions-data-sets/(20\d{2}q[1-4])_form345\.zip",
        urlparse(url).path,
    )
    if path_match is None or path_match.group(1) != quarter:
        raise OfflineImportError(
            f"official source URL does not identify quarter {quarter}: {url!r}"
        )
    return url


def _cached_quarter_status(
    cache_root: Path, quarter: str
) -> tuple[str, Path | None, str | None, int | None, list[str]]:
    candidates = sorted((cache_root / "bulk" / quarter).glob("*.zip"))
    valid: list[tuple[Path, str, int]] = []
    invalid: list[str] = []
    for candidate in candidates:
        try:
            content = _read_stable(candidate)
            parsed = parse_insider_zip(content, quarter=quarter)
            if candidate.stem != parsed.archive_sha256:
                raise OfflineImportError("cache filename is not its SHA256")
            valid.append((candidate, parsed.archive_sha256, len(content)))
        except (OfflineImportError, InsiderDataSchemaError) as exc:
            invalid.append(f"{candidate}: {exc}")
    if invalid:
        status = "invalid_cached_asset"
    elif len(valid) > 1:
        status = "ambiguous_multiple_cached_versions"
    elif len(valid) == 1:
        status = "verified_cached"
    else:
        status = "manual_download_required"
    unique_valid = valid[0] if status == "verified_cached" else None
    return (
        status,
        unique_valid[0] if unique_valid is not None else None,
        unique_valid[1] if unique_valid is not None else None,
        unique_valid[2] if unique_valid is not None else None,
        invalid,
    )


def _plan_payload(
    rows: list[dict[str, Any]], *, start_quarter: str, end_quarter: str
) -> Mapping[str, Any]:
    return {
        "schema_version": ACQUISITION_PLAN_SCHEMA,
        "generated_at": datetime.now(UTC).isoformat(),
        "research_status": RESEARCH_STATUS,
        "alpha_claim_permitted": False,
        "landing_page": INSIDER_DATASET_PAGE,
        "start_quarter": start_quarter,
        "end_quarter": end_quarter,
        "quarter_count": len(rows),
        "verified_cached_count": sum(row["status"] == "verified_cached" for row in rows),
        "manual_download_required_count": sum(
            row["status"] == "manual_download_required" for row in rows
        ),
        "blocking_cache_finding_count": sum(
            row["status"]
            in {"invalid_cached_asset", "ambiguous_multiple_cached_versions"}
            for row in rows
        ),
        "records": rows,
        "next_step": (
            "download only with a declared real SEC_USER_AGENT or manually in a "
            "browser; then prepare and import each file with exact URL and hash"
        ),
    }


def build_bulk_acquisition_plan(
    landing_page_html: str,
    *,
    start_quarter: str,
    end_quarter: str,
    cache_dir: str | Path,
) -> Mapping[str, Any]:
    """Build a complete quarter plan from an operator-saved official SEC page."""

    links = discover_quarter_links(landing_page_html, base_url=INSIDER_DATASET_PAGE)
    required = _quarter_sequence(start_quarter.lower(), end_quarter.lower())
    missing = [quarter for quarter in required if quarter not in links]
    if missing:
        raise OfflineImportError(
            f"official landing-page snapshot is missing requested quarters: {missing}"
        )
    cache_root = Path(cache_dir).resolve()
    rows: list[dict[str, Any]] = []
    for quarter in required:
        url = _validate_bulk_url(links[quarter], quarter)
        status, cache_path, sha256, byte_count, invalid = _cached_quarter_status(
            cache_root, quarter
        )
        rows.append(
            {
                "quarter": quarter,
                "source_url": url,
                "status": status,
                "cache_path": cache_path.as_posix() if cache_path is not None else None,
                "bytes": byte_count,
                "sha256": sha256,
                "displayed_size_mb": None,
                "invalid_cache_findings": invalid,
            }
        )
    return _plan_payload(rows, start_quarter=required[0], end_quarter=required[-1])


def build_bulk_acquisition_plan_from_catalog(
    catalog: pd.DataFrame,
    *,
    start_quarter: str,
    end_quarter: str,
    cache_dir: str | Path,
) -> Mapping[str, Any]:
    """Build a plan from a frozen catalog projected from the official SEC page."""

    required_columns = {"quarter", "source_url", "displayed_size_mb"}
    missing_columns = required_columns.difference(catalog.columns)
    if missing_columns:
        raise OfflineImportError(
            f"quarter catalog is missing columns: {sorted(missing_columns)}"
        )
    frame = catalog.copy()
    frame["quarter"] = frame["quarter"].astype("string").str.strip().str.lower()
    invalid_quarter = ~frame["quarter"].fillna("").str.fullmatch(QUARTER_PATTERN)
    if invalid_quarter.any():
        raise OfflineImportError("quarter catalog contains invalid quarter labels")
    if frame["quarter"].duplicated().any():
        raise OfflineImportError("quarter catalog contains duplicate quarters")
    frame["displayed_size_mb"] = pd.to_numeric(
        frame["displayed_size_mb"], errors="coerce"
    )
    if frame["displayed_size_mb"].isna().any() or frame["displayed_size_mb"].le(0).any():
        raise OfflineImportError("quarter catalog contains invalid displayed sizes")
    required = _quarter_sequence(start_quarter.lower(), end_quarter.lower())
    indexed = frame.set_index("quarter", verify_integrity=True)
    missing = [quarter for quarter in required if quarter not in indexed.index]
    if missing:
        raise OfflineImportError(f"quarter catalog is missing requested quarters: {missing}")
    cache_root = Path(cache_dir).resolve()
    rows: list[dict[str, Any]] = []
    for quarter in required:
        catalog_row = indexed.loc[quarter]
        url = _validate_bulk_url(catalog_row["source_url"], quarter)
        status, cache_path, sha256, byte_count, invalid = _cached_quarter_status(
            cache_root, quarter
        )
        rows.append(
            {
                "quarter": quarter,
                "source_url": url,
                "status": status,
                "cache_path": cache_path.as_posix() if cache_path is not None else None,
                "bytes": byte_count,
                "sha256": sha256,
                "displayed_size_mb": float(catalog_row["displayed_size_mb"]),
                "invalid_cache_findings": invalid,
            }
        )
    payload = dict(
        _plan_payload(rows, start_quarter=required[0], end_quarter=required[-1])
    )
    payload["catalog_transport_note"] = (
        "frozen semantic projection of the official landing page; verify the live "
        "page again before each download"
    )
    payload["displayed_total_mb"] = float(
        sum(row["displayed_size_mb"] for row in rows)
    )
    return payload


__all__ = [
    "ACQUISITION_PLAN_SCHEMA",
    "OFFLINE_IMPORT_SCHEMA",
    "OFFLINE_RECEIPT_SCHEMA",
    "OfflineImportError",
    "TRANSPORT_ATTESTATION",
    "VerifiedOfflineRecord",
    "build_bulk_acquisition_plan",
    "build_bulk_acquisition_plan_from_catalog",
    "import_verified_offline_bundle",
    "load_verified_offline_bundle",
    "prepare_bulk_manual_import",
    "verify_offline_import_receipt",
]
