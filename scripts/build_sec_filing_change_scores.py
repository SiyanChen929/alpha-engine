#!/usr/bin/env python3
"""Parse frozen SEC documents and build deterministic filing-change scores."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alpha_factor_engine.features.filing_change import (
    FilingChangeError,
    assign_release_month_quintiles,
    binary_jaccard_counts,
    build_strict_fiscal_pairs,
    clean_primary_filing,
    extract_fiscal_identity,
    raw_tf_cosine_counts,
)


DESIGN_RECEIPT_TOP_LEVEL = frozenset(
    {
        "schema_version",
        "created_at",
        "experiment",
        "status",
        "records",
        "primary_signal",
        "secondary_signal",
        "current_disclosed_trial_floor",
        "outcome_join_performed",
        "sealed_2025_signal_return_joined",
        "historical_alpha_verdict",
        "deployment_verdict",
        "non_overwrite_policy",
    }
)
PLAN_MANIFEST_INPUT_ROLES = frozenset(
    {
        "human_protocol",
        "frozen_proxy_universe",
        "sec_header_raw_manifest",
        "planner",
        "sec_periodic_data_code",
        "pair_plan_code",
    }
)
PLAN_MANIFEST_OUTPUT_ROLES = frozenset(
    {
        "mapped_header_inventory",
        "rough_pairing_ledger",
        "acquisition_documents",
        "header_pair_exclusions",
        "mapping_misses",
        "plan_summary",
    }
)
ACQUISITION_MANIFEST_INPUT_ROLES = frozenset(
    {
        "design_receipt",
        "frozen_acquisition_documents",
        "acquisition_runner",
        "sec_periodic_data_code",
    }
)
ACQUISITION_MANIFEST_OUTPUT_ROLES = frozenset(
    {
        "document_acquisition_audit",
        "document_acquisition_failures",
        "dataset_records",
        "acquisition_summary",
    }
)
PHASE_PRIORITY = {
    "00_top25_engineering_smoke": 0,
    "10_full_development_and_confirmation": 1,
    "20_sealed_2025_signal_only": 2,
}
ACQUISITION_AUDIT_COLUMNS = frozenset(
    {
        "issuer_cik",
        "accession",
        "form",
        "ticker",
        "download_phase",
        "api_acceptance_at_raw",
        "raw_header_acceptance_at",
        "api_raw_delta_seconds",
        "primary_document",
        "primary_url",
        "primary_sha256",
        "primary_raw_bytes",
        "primary_cache_path",
        "primary_cache_status",
        "header_sha256",
        "header_cache_path",
        "header_cache_status",
        "outcome_join_performed",
    }
)
ACQUISITION_SUMMARY_FIELDS = frozenset(
    {
        "schema_version",
        "created_at",
        "through_phase",
        "status",
        "eligible_documents_through_phase",
        "selection_limited",
        "requested_documents",
        "acquired_documents",
        "failed_documents",
        "acquisition_coverage",
        "unique_accessions",
        "unique_issuers",
        "raw_clock_exact_match_rate",
        "primary_raw_bytes",
        "outcome_join_performed",
        "historical_alpha_verdict",
        "deployment_verdict",
        "design_receipt_sha256",
    }
)
DATASET_RECORD_FIELDS = frozenset(
    {
        "provider",
        "dataset",
        "retrieved_at",
        "source_url",
        "sha256",
        "rows",
        "min_effective_at",
        "max_effective_at",
        "min_available_at",
        "max_available_at",
        "finality",
        "revision_policy",
        "notes",
    }
)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path, role: str) -> dict[str, object]:
    return {
        "role": role,
        "path": path.resolve().relative_to(ROOT.resolve()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _verify_receipt(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("design receipt must be a JSON object")
    if payload.get("schema_version") != "sec_filing_change_design_receipt_v2":
        raise ValueError("unsupported design receipt")
    if set(payload) != DESIGN_RECEIPT_TOP_LEVEL:
        raise ValueError(
            "design receipt top-level fields differ from the frozen contract"
        )
    required_roles = {
        "human_protocol",
        "preoutcome_plan_manifest",
        "frozen_acquisition_documents",
        "header_only_rough_pairs",
        "frozen_current_survivor_universe",
        "sec_submissions_raw_manifest",
        "frozen_retrospective_daily_prices",
        "xnys_regular_sessions",
        "xnys_calendar_receipt",
        "calendar_snapshot_generator",
        "sec_periodic_data_code",
        "filing_change_feature_code",
        "preoutcome_plan_code",
        "outcome_core_code",
        "plan_builder",
        "document_acquirer",
        "score_builder",
        "outcome_runner",
        "design_receipt_builder",
    }
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != len(required_roles):
        raise ValueError("design receipt has incomplete records")
    roles = [record.get("role") for record in records if isinstance(record, dict)]
    if (
        len(roles) != len(records)
        or set(roles) != required_roles
        or len(roles) != len(set(roles))
    ):
        raise ValueError("design receipt roles differ from the frozen contract")
    if (
        payload.get("primary_signal") != "jaccard_similarity"
        or payload.get("secondary_signal") != "cosine_similarity"
        or payload.get("current_disclosed_trial_floor") != 130
        or payload.get("experiment") != "sec_filing_change_stage_a_v1"
        or payload.get("status") != "FROZEN_PRE_OUTCOME"
        or payload.get("outcome_join_performed") is not False
        or payload.get("sealed_2025_signal_return_joined") is not False
        or payload.get("non_overwrite_policy") is not True
        or payload.get("historical_alpha_verdict") != "INVALID"
        or payload.get("deployment_verdict") != "NOT AUTHORIZED"
    ):
        raise ValueError("design receipt top-level constraints are invalid")
    created_at = payload.get("created_at")
    try:
        created_timestamp = datetime.fromisoformat(
            str(created_at).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ValueError("design receipt creation timestamp is invalid") from exc
    if not isinstance(created_at, str) or created_timestamp.tzinfo is None:
        raise ValueError("design receipt creation timestamp is invalid")
    receipt_sources: dict[str, Path] = {}
    for record in records:
        if set(record) != {"role", "path", "bytes", "sha256"}:
            raise ValueError(
                f"design receipt record is malformed: {record.get('role')}"
            )
        raw_path = record.get("path")
        expected_bytes = record.get("bytes")
        expected_hash = record.get("sha256")
        if (
            not isinstance(raw_path, str)
            or not raw_path
            or Path(raw_path).is_absolute()
            or not isinstance(expected_bytes, int)
            or isinstance(expected_bytes, bool)
            or expected_bytes < 0
            or not isinstance(expected_hash, str)
            or len(expected_hash) != 64
        ):
            raise ValueError(f"design receipt record is malformed: {record['role']}")
        candidate = ROOT / raw_path
        if candidate.is_symlink():
            raise ValueError(
                f"design receipt source may not be a symlink: {record['role']}"
            )
        source = candidate.resolve()
        source.relative_to(ROOT.resolve())
        if not source.is_file():
            raise FileNotFoundError(source)
        if (
            source.stat().st_size != expected_bytes
            or sha256_file(source) != expected_hash
        ):
            raise ValueError(f"design receipt hash mismatch: {record['role']}")
        receipt_sources[str(record["role"])] = source
    if len(receipt_sources) != len(set(receipt_sources.values())):
        raise ValueError("design receipt assigns multiple roles to one path")
    expected_code_paths = {
        "sec_periodic_data_code": SRC
        / "alpha_factor_engine/data/sec_periodic_filings.py",
        "filing_change_feature_code": SRC
        / "alpha_factor_engine/features/filing_change.py",
        "preoutcome_plan_code": SRC
        / "alpha_factor_engine/research/sec_filing_change_plan.py",
        "outcome_core_code": SRC
        / "alpha_factor_engine/research/sec_filing_change_stage_a.py",
        "plan_builder": ROOT / "scripts/build_sec_filing_change_acquisition_plan.py",
        "document_acquirer": ROOT / "scripts/acquire_sec_filing_change_documents.py",
        "score_builder": ROOT / "scripts/build_sec_filing_change_scores.py",
        "outcome_runner": ROOT / "scripts/run_sec_filing_change_stage_a.py",
        "design_receipt_builder": ROOT
        / "scripts/build_sec_filing_change_design_receipt.py",
        "calendar_snapshot_generator": ROOT / "scripts/build_xnys_calendar_snapshot.py",
    }
    mismatched_roles = sorted(
        role
        for role, expected in expected_code_paths.items()
        if receipt_sources[role] != expected.resolve()
    )
    if mismatched_roles:
        raise ValueError(
            f"design receipt code roles do not bind current files: {mismatched_roles}"
        )
    return payload


def _verify_manifest(
    path: Path,
    schema: str,
    *,
    expected_inputs: frozenset[str] | None = None,
    expected_outputs: frozenset[str] | None = None,
    expected_top_level: frozenset[str] | None = None,
) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"manifest must be a JSON object: {path}")
    if payload.get("schema_version") != schema:
        raise ValueError(f"unsupported manifest: {path}")
    if expected_top_level is not None and set(payload) != expected_top_level:
        raise ValueError(f"manifest top-level fields differ from the contract: {path}")
    created_at = payload.get("created_at")
    try:
        created_timestamp = datetime.fromisoformat(
            str(created_at).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ValueError(f"manifest creation timestamp is invalid: {path}") from exc
    if not isinstance(created_at, str) or created_timestamp.tzinfo is None:
        raise ValueError(f"manifest creation timestamp is invalid: {path}")
    inputs = payload.get("inputs")
    outputs = payload.get("outputs")
    if (
        not isinstance(inputs, list)
        or not inputs
        or not isinstance(outputs, list)
        or not outputs
    ):
        raise ValueError(f"manifest has incomplete records: {path}")
    combined = [*inputs, *outputs]
    roles = [record.get("role") for record in combined if isinstance(record, dict)]
    if len(roles) != len(combined) or any(
        not isinstance(role, str) or not role for role in roles
    ):
        raise ValueError(f"manifest has duplicate or invalid roles: {path}")
    if len(roles) != len(set(roles)):
        raise ValueError(f"manifest has duplicate or invalid roles: {path}")
    input_roles = {record["role"] for record in inputs}
    output_roles = {record["role"] for record in outputs}
    if expected_inputs is not None and input_roles != expected_inputs:
        raise ValueError(f"manifest input roles differ from the contract: {path}")
    if expected_outputs is not None and output_roles != expected_outputs:
        raise ValueError(f"manifest output roles differ from the contract: {path}")
    manifest_sources: list[Path] = []
    for record in combined:
        if set(record) != {"role", "path", "bytes", "sha256"}:
            raise ValueError(f"manifest record is malformed: {record.get('role')}")
        raw_path = record.get("path")
        expected_bytes = record.get("bytes")
        expected_hash = record.get("sha256")
        if (
            not isinstance(raw_path, str)
            or not raw_path
            or Path(raw_path).is_absolute()
            or not isinstance(expected_bytes, int)
            or isinstance(expected_bytes, bool)
            or expected_bytes < 0
            or not isinstance(expected_hash, str)
            or len(expected_hash) != 64
        ):
            raise ValueError(f"manifest record is malformed: {record['role']}")
        candidate = ROOT / raw_path
        if candidate.is_symlink():
            raise ValueError(f"manifest source may not be a symlink: {record['role']}")
        source = candidate.resolve()
        source.relative_to(ROOT.resolve())
        if not source.is_file() or source.stat().st_size != expected_bytes:
            raise ValueError(
                f"manifest file missing or size mismatch: {record['role']}"
            )
        if sha256_file(source) != expected_hash:
            raise ValueError(f"manifest hash mismatch: {record['role']}")
        manifest_sources.append(source)
    if len(manifest_sources) != len(set(manifest_sources)):
        raise ValueError(f"manifest assigns multiple roles to one path: {path}")
    return payload


def _manifest_path_for_role(payload: dict[str, object], role: str) -> Path:
    matches = [
        record
        for record in [*payload.get("inputs", []), *payload.get("outputs", [])]
        if record.get("role") == role
    ]
    if len(matches) != 1:
        raise ValueError(f"manifest must contain exactly one {role!r} record")
    return (ROOT / str(matches[0]["path"])).resolve()


def _receipt_path_for_role(payload: dict[str, object], role: str) -> Path:
    matches = [record for record in payload["records"] if record.get("role") == role]
    if len(matches) != 1:
        raise ValueError(f"design receipt must contain exactly one {role!r} record")
    return (ROOT / str(matches[0]["path"])).resolve()


def _read_csv_allow_empty(path: Path) -> pd.DataFrame:
    if not path.read_text(encoding="utf-8").strip():
        return pd.DataFrame()
    return pd.read_csv(path)


def _verified_cache_path(value: object, *, label: str) -> Path:
    raw_path = Path(str(value))
    if raw_path.is_symlink():
        raise ValueError(f"{label} cache path may not be a symlink: {raw_path}")
    path = raw_path.resolve()
    path.relative_to(ROOT.resolve())
    if not path.is_file():
        raise ValueError(f"{label} cache file is missing: {path}")
    return path


def _reconcile_acquisition_data(
    *,
    summary: dict[str, object],
    manifest_created_at: object,
    audit: pd.DataFrame,
    failures: pd.DataFrame,
    dataset_records: object,
    frozen_documents: pd.DataFrame,
    design_receipt_sha256: str,
) -> None:
    if set(summary) != ACQUISITION_SUMMARY_FIELDS:
        raise ValueError("acquisition summary fields differ from the contract")
    through_phase = summary.get("through_phase")
    if through_phase not in PHASE_PRIORITY:
        raise ValueError("acquisition summary has an unknown phase")
    if (
        summary.get("schema_version")
        != "sec_filing_change_document_acquisition_summary_v1"
        or summary.get("created_at") != manifest_created_at
        or summary.get("status") != "COMPLETE"
        or summary.get("outcome_join_performed") is not False
        or summary.get("selection_limited") is not False
        or summary.get("historical_alpha_verdict") != "INVALID"
        or summary.get("deployment_verdict") != "NOT AUTHORIZED"
        or summary.get("design_receipt_sha256") != design_receipt_sha256
    ):
        raise ValueError("document acquisition is incomplete or outcome-contaminated")
    if set(audit.columns) != ACQUISITION_AUDIT_COLUMNS:
        raise ValueError("document acquisition audit columns differ from the contract")
    if audit.empty:
        raise ValueError("document acquisition audit is empty")
    if audit.isna().any().any():
        raise ValueError("document acquisition audit has null values")
    if not audit["accession"].is_unique:
        raise ValueError("document acquisition audit has duplicate accessions")
    if (
        not pd.api.types.is_bool_dtype(audit["outcome_join_performed"].dtype)
        or audit["outcome_join_performed"].ne(False).any()
    ):
        raise ValueError("document acquisition audit contains an outcome join")
    if not failures.empty:
        raise ValueError("document acquisition failure ledger is not empty")
    required_frozen = {
        "issuer_cik",
        "accession",
        "form",
        "ticker",
        "acceptance_at_raw",
        "primary_document",
        "primary_document_url",
        "download_phase",
    }
    missing_frozen = required_frozen.difference(frozen_documents.columns)
    if missing_frozen:
        raise ValueError(
            f"frozen acquisition documents are missing columns: {sorted(missing_frozen)}"
        )
    if not frozen_documents["accession"].is_unique:
        raise ValueError("frozen acquisition documents have duplicate accessions")
    if not frozen_documents["download_phase"].isin(PHASE_PRIORITY).all():
        raise ValueError("frozen acquisition documents contain an unknown phase")
    cutoff = PHASE_PRIORITY[str(through_phase)]
    expected = frozen_documents.loc[
        frozen_documents["download_phase"].map(PHASE_PRIORITY).le(cutoff)
    ].copy()
    expected_accessions = set(expected["accession"].astype(str))
    actual_accessions = set(audit["accession"].astype(str))
    if expected_accessions != actual_accessions:
        raise ValueError("document acquisition audit differs from the frozen phase set")
    expected_by_accession = expected.set_index("accession")
    for row in audit.itertuples(index=False):
        frozen = expected_by_accession.loc[row.accession]
        comparisons = {
            "issuer_cik": (str(row.issuer_cik), str(frozen.issuer_cik)),
            "form": (str(row.form), str(frozen.form)),
            "ticker": (str(row.ticker), str(frozen.ticker)),
            "download_phase": (str(row.download_phase), str(frozen.download_phase)),
            "api_acceptance_at_raw": (
                str(row.api_acceptance_at_raw),
                str(frozen.acceptance_at_raw),
            ),
            "primary_document": (
                str(row.primary_document),
                str(frozen.primary_document),
            ),
            "primary_url": (str(row.primary_url), str(frozen.primary_document_url)),
        }
        mismatches = [
            field
            for field, (actual, planned) in comparisons.items()
            if actual != planned
        ]
        if mismatches:
            raise ValueError(
                f"document acquisition metadata differs from the frozen plan for "
                f"{row.accession}: {mismatches}"
            )
    api_clock = pd.to_datetime(
        audit["api_acceptance_at_raw"], errors="coerce", utc=True
    )
    raw_clock = pd.to_datetime(
        audit["raw_header_acceptance_at"], errors="coerce", utc=True
    )
    recorded_delta = pd.to_numeric(audit["api_raw_delta_seconds"], errors="coerce")
    calculated_delta = (raw_clock - api_clock).abs().dt.total_seconds()
    if (
        api_clock.isna().any()
        or raw_clock.isna().any()
        or recorded_delta.isna().any()
        or recorded_delta.lt(0).any()
        or recorded_delta.gt(1.0).any()
        or not np.allclose(recorded_delta, calculated_delta, rtol=0.0, atol=1e-9)
    ):
        raise ValueError("document acquisition acceptance-clock evidence is invalid")
    for hash_column in ("primary_sha256", "header_sha256"):
        if not audit[hash_column].astype("string").str.fullmatch(r"[0-9a-f]{64}").all():
            raise ValueError(f"document acquisition has invalid {hash_column}")
    primary_bytes = pd.to_numeric(audit["primary_raw_bytes"], errors="coerce")
    if primary_bytes.isna().any() or primary_bytes.le(0).any():
        raise ValueError("document acquisition has invalid primary byte counts")
    expected_count = len(expected)
    expected_summary = {
        "eligible_documents_through_phase": expected_count,
        "requested_documents": expected_count,
        "acquired_documents": len(audit),
        "failed_documents": len(failures),
        "unique_accessions": audit["accession"].nunique(),
        "unique_issuers": audit["issuer_cik"].nunique(),
        "primary_raw_bytes": int(primary_bytes.sum()),
    }
    if any(summary.get(key) != value for key, value in expected_summary.items()):
        raise ValueError("acquisition summary counts do not reconcile")
    coverage = summary.get("acquisition_coverage")
    exact_rate = summary.get("raw_clock_exact_match_rate")
    if (
        not isinstance(coverage, (int, float))
        or isinstance(coverage, bool)
        or not np.isclose(float(coverage), 1.0)
        or not isinstance(exact_rate, (int, float))
        or isinstance(exact_rate, bool)
        or not np.isclose(float(exact_rate), float(recorded_delta.eq(0).mean()))
    ):
        raise ValueError("acquisition summary rates do not reconcile")
    if not isinstance(dataset_records, list) or len(dataset_records) != 2 * len(audit):
        raise ValueError("dataset records do not cover every acquired document")
    if any(not isinstance(record, dict) for record in dataset_records):
        raise ValueError("dataset records contain a malformed entry")
    labels = [record.get("dataset") for record in dataset_records]
    if len(labels) != len(set(labels)):
        raise ValueError("dataset records contain duplicate dataset labels")
    records_by_label = {str(record["dataset"]): record for record in dataset_records}
    for row in audit.itertuples(index=False):
        primary_label = f"periodic_primary_document:{row.accession}"
        header_label = f"acceptance_header:{row.accession}"
        if (
            primary_label not in records_by_label
            or header_label not in records_by_label
        ):
            raise ValueError(f"dataset records are incomplete for {row.accession}")
        primary_record = records_by_label[primary_label]
        header_record = records_by_label[header_label]
        for record in (primary_record, header_record):
            if set(record) != DATASET_RECORD_FIELDS or record.get("rows") != 1:
                raise ValueError(f"dataset record is malformed for {row.accession}")
            if record.get("provider") != "SEC EDGAR Archives" or not isinstance(
                record.get("notes"), dict
            ):
                raise ValueError(
                    f"dataset record provenance is invalid for {row.accession}"
                )
        primary_notes = primary_record["notes"]
        header_notes = header_record["notes"]
        primary_path = _verified_cache_path(
            row.primary_cache_path, label="primary document"
        )
        header_path = _verified_cache_path(
            row.header_cache_path, label="acceptance header"
        )
        expected_primary = {
            "source_url": str(row.primary_url),
            "sha256": str(row.primary_sha256),
        }
        if any(
            primary_record.get(key) != value for key, value in expected_primary.items()
        ):
            raise ValueError(f"primary dataset record differs for {row.accession}")
        if (
            str(primary_notes.get("issuer_cik")) != str(row.issuer_cik)
            or primary_notes.get("accession") != row.accession
            or primary_notes.get("primary_document") != row.primary_document
            or Path(str(primary_notes.get("cache_path"))).resolve() != primary_path
            or primary_notes.get("raw_bytes") != int(row.primary_raw_bytes)
            or pd.Timestamp(primary_notes.get("acceptance_at"))
            != pd.Timestamp(row.api_acceptance_at_raw)
        ):
            raise ValueError(f"primary dataset notes differ for {row.accession}")
        compact = str(row.accession).replace("-", "")
        expected_header_url = (
            f"https://www.sec.gov/Archives/edgar/data/{int(row.issuer_cik)}/"
            f"{compact}/{row.accession}.hdr.sgml"
        )
        if (
            header_record.get("source_url") != expected_header_url
            or header_record.get("sha256") != row.header_sha256
            or str(header_notes.get("issuer_cik")) != str(row.issuer_cik)
            or header_notes.get("accession") != row.accession
            or Path(str(header_notes.get("cache_path"))).resolve() != header_path
            or pd.Timestamp(header_notes.get("api_acceptance_at"))
            != pd.Timestamp(row.api_acceptance_at_raw)
            or not np.isclose(
                float(header_notes.get("api_raw_delta_seconds")),
                float(row.api_raw_delta_seconds),
            )
            or pd.Timestamp(header_notes.get("raw_header_acceptance_at"))
            != pd.Timestamp(row.raw_header_acceptance_at)
        ):
            raise ValueError(f"header dataset record differs for {row.accession}")
        try:
            header_raw = gzip.decompress(header_path.read_bytes())
        except (EOFError, OSError) as exc:
            raise ValueError(
                f"invalid acceptance header gzip for {row.accession}"
            ) from exc
        if sha256_bytes(header_raw) != row.header_sha256:
            raise ValueError(f"acceptance header cache hash mismatch: {row.accession}")


def _coverage_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def _build_upstream_coverage(
    *,
    thresholds: dict[str, object],
    acquisition_summary: dict[str, object],
    acquisition_audit: pd.DataFrame,
    document_audit: pd.DataFrame,
    rough_pairs: pd.DataFrame,
    strict_pairs: pd.DataFrame,
    pair_exclusions: pd.DataFrame,
    mapped_headers: pd.DataFrame,
    mapping_misses: pd.DataFrame,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    required_thresholds = {
        "sec_raw_clock",
        "primary_document_retrieval",
        "prior_year_pairing",
        "parse_success",
        "ticker_mapping",
        "benchmark_clock",
    }
    if not required_thresholds.issubset(thresholds):
        raise ValueError("protocol coverage thresholds are incomplete")
    through_phase = str(acquisition_summary["through_phase"])
    if through_phase not in PHASE_PRIORITY:
        raise ValueError("acquisition summary has an unknown phase")
    required_rough_columns = {
        "form",
        "current_accession",
        "prior_accession",
        "current_acceptance_at",
        "current_release_month",
        "download_phase",
    }
    missing_rough = required_rough_columns.difference(rough_pairs.columns)
    if missing_rough:
        raise ValueError(f"rough-pair ledger missing columns: {sorted(missing_rough)}")
    if rough_pairs.duplicated(["current_accession", "prior_accession"]).any():
        raise ValueError("rough-pair ledger has duplicate candidate pairs")
    if not rough_pairs["download_phase"].isin(PHASE_PRIORITY).all():
        raise ValueError("rough-pair ledger has an unknown phase")
    accepted_clock = pd.to_datetime(
        rough_pairs["current_acceptance_at"], errors="raise", utc=True
    )
    expected_release_month = (
        accepted_clock.dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
        .dt.to_period("M")
    )
    supplied_release_month = pd.Series(
        pd.PeriodIndex(rough_pairs["current_release_month"].astype(str), freq="M"),
        index=rough_pairs.index,
    )
    if not supplied_release_month.equals(expected_release_month):
        raise ValueError("rough-pair release month differs from New York acceptance clock")
    current_metadata = [
        "form", "current_acceptance_at", "current_release_month", "download_phase"
    ]
    if rough_pairs[current_metadata].isna().any().any():
        raise ValueError("rough-pair ledger has null current-filing metadata")
    metadata_variants = rough_pairs.groupby("current_accession", dropna=False)[
        current_metadata
    ].nunique(dropna=False)
    if metadata_variants.gt(1).any().any():
        raise ValueError("rough-pair candidates disagree on current-filing metadata")
    cutoff = PHASE_PRIORITY[through_phase]
    phase_rough_pairs = rough_pairs.loc[
        rough_pairs["download_phase"].map(PHASE_PRIORITY).le(cutoff)
    ].copy()
    required_strict_columns = {
        "form",
        "current_accession",
        "prior_accession",
        "current_accepted_at",
    }
    missing_strict = required_strict_columns.difference(strict_pairs.columns)
    if missing_strict:
        raise ValueError(
            f"strict-pair ledger missing columns: {sorted(missing_strict)}"
        )
    if strict_pairs[list(required_strict_columns)].isna().any().any():
        raise ValueError("strict-pair ledger has null pair metadata")
    if not strict_pairs["current_accession"].is_unique:
        raise ValueError("strict-pair ledger has duplicate current accessions")
    rough_pair_keys = set(
        phase_rough_pairs[["current_accession", "prior_accession"]].itertuples(
            index=False, name=None
        )
    )
    strict_pair_keys = set(
        strict_pairs[["current_accession", "prior_accession"]].itertuples(
            index=False, name=None
        )
    )
    if not strict_pair_keys.issubset(rough_pair_keys):
        raise ValueError(
            "strict-pair ledger contains a pair outside the frozen rough-pair "
            "candidate scope"
        )
    mapping_denominator = len(mapped_headers) + len(mapping_misses)
    metrics = {
        "sec_raw_clock": float(acquisition_audit["api_raw_delta_seconds"].eq(0).mean()),
        "primary_document_retrieval": _coverage_rate(
            int(acquisition_summary["acquired_documents"]),
            int(acquisition_summary["eligible_documents_through_phase"]),
        ),
        "prior_year_pairing": _coverage_rate(
            strict_pairs["current_accession"].nunique(),
            phase_rough_pairs["current_accession"].nunique(),
        ),
        "parse_success": _coverage_rate(
            int(document_audit["parse_status"].eq("OK").sum()),
            len(document_audit),
        ),
        "ticker_mapping": _coverage_rate(len(mapped_headers), mapping_denominator),
        "benchmark_clock": None,
    }
    gates: dict[str, dict[str, object]] = {}
    for name in sorted(required_thresholds):
        threshold = thresholds[name]
        if (
            not isinstance(threshold, (int, float))
            or isinstance(threshold, bool)
            or not 0 <= float(threshold) <= 1
        ):
            raise ValueError(f"invalid coverage threshold: {name}")
        value = metrics[name]
        gates[name] = {
            "threshold": float(threshold),
            "value": value,
            "passed": None if value is None else bool(value >= float(threshold)),
            "evaluation_stage": (
                "outcome_runner" if name == "benchmark_clock" else "score_builder"
            ),
        }
    signal_build_names = required_thresholds - {"benchmark_clock"}
    signal_build_passed = all(
        bool(gates[name]["passed"]) for name in signal_build_names
    )
    coverage = {
        "schema_version": "sec_filing_change_upstream_coverage_v1",
        "status": (
            "PASS_UPSTREAM_SIGNAL_BUILD_COVERAGE"
            if signal_build_passed
            else "BLOCKED_DATA_COVERAGE"
        ),
        "all_signal_build_gates_passed": signal_build_passed,
        "benchmark_clock_deferred_to_outcome_runner": True,
        "gates": gates,
    }

    documents = document_audit.copy()
    documents["filing_year"] = (
        pd.to_datetime(documents["raw_header_acceptance_at"], errors="coerce", utc=True)
        .dt.tz_convert("America/New_York")
        .dt.year
    )
    if documents["filing_year"].isna().any():
        raise ValueError("document audit has an invalid raw acceptance year")
    phase_rough_pairs["filing_year"] = (
        pd.to_datetime(phase_rough_pairs["current_acceptance_at"], errors="coerce", utc=True)
        .dt.tz_convert("America/New_York")
        .dt.year
    )
    strict = strict_pairs.copy()
    strict["filing_year"] = (
        pd.to_datetime(strict["current_accepted_at"], errors="coerce", utc=True)
        .dt.tz_convert("America/New_York")
        .dt.year
    )
    exclusions = pair_exclusions.copy()
    if not exclusions.empty:
        if not {"accepted_at", "form", "reason"}.issubset(exclusions.columns):
            raise ValueError("pair-exclusion ledger is missing audit fields")
        exclusions["filing_year"] = (
            pd.to_datetime(exclusions["accepted_at"], errors="coerce", utc=True)
            .dt.tz_convert("America/New_York")
            .dt.year
        )
    else:
        exclusions = pd.DataFrame(columns=["filing_year", "form", "reason"])
    mapped = mapped_headers.copy()
    mapped["filing_year"] = (
        pd.to_datetime(mapped["acceptance_at"], errors="coerce", utc=True)
        .dt.tz_convert("America/New_York")
        .dt.year
    )
    misses = mapping_misses.copy()
    if not misses.empty:
        if not {"acceptance_at", "form"}.issubset(misses.columns):
            raise ValueError("mapping-miss ledger is missing year/form fields")
        misses["filing_year"] = (
            pd.to_datetime(misses["acceptance_at"], errors="coerce", utc=True)
            .dt.tz_convert("America/New_York")
            .dt.year
        )
    else:
        misses = pd.DataFrame(columns=["filing_year", "form"])
    frames = (documents, phase_rough_pairs, strict, exclusions, mapped, misses)
    if any(not {"filing_year", "form"}.issubset(frame.columns) for frame in frames):
        raise ValueError("coverage source is missing filing year or form")
    if any(frame[["filing_year", "form"]].isna().any().any() for frame in frames):
        raise ValueError("coverage source has invalid filing year or form")
    group_keys = sorted(
        {
            (int(row.filing_year), str(row.form))
            for frame in frames
            for row in frame[["filing_year", "form"]].dropna().itertuples(index=False)
        }
    )
    breakdown: list[dict[str, object]] = []
    for year, form in group_keys:
        document_group = documents.loc[
            documents["filing_year"].eq(year) & documents["form"].eq(form)
        ]
        rough_group = phase_rough_pairs.loc[
            phase_rough_pairs["filing_year"].eq(year)
            & phase_rough_pairs["form"].eq(form)
        ]
        strict_group = strict.loc[
            strict["filing_year"].eq(year) & strict["form"].eq(form)
        ]
        exclusion_group = exclusions.loc[
            exclusions["filing_year"].eq(year) & exclusions["form"].eq(form)
        ]
        mapped_group = mapped.loc[
            mapped["filing_year"].eq(year) & mapped["form"].eq(form)
        ]
        miss_group = misses.loc[
            misses["filing_year"].eq(year) & misses["form"].eq(form)
        ]
        raw_exact = int(
            document_group["accession"]
            .map(acquisition_audit.set_index("accession")["api_raw_delta_seconds"])
            .eq(0)
            .sum()
        )
        mapped_total = len(mapped_group) + len(miss_group)
        breakdown.append(
            {
                "filing_year": year,
                "form": form,
                "documents_planned": int(len(document_group)),
                "documents_retrieved": int(len(document_group)),
                "primary_document_retrieval": _coverage_rate(
                    len(document_group), len(document_group)
                ),
                "raw_clocks_exact": raw_exact,
                "sec_raw_clock": _coverage_rate(raw_exact, len(document_group)),
                "documents_parsed": int(document_group["parse_status"].eq("OK").sum()),
                "parse_success": _coverage_rate(
                    int(document_group["parse_status"].eq("OK").sum()),
                    len(document_group),
                ),
                "parse_exclusion_reasons": {
                    str(reason): int(count)
                    for reason, count in document_group.loc[
                        document_group["parse_status"].ne("OK"), "parse_error"
                    ]
                    .fillna("unspecified_parse_error")
                    .value_counts()
                    .sort_index()
                    .items()
                },
                "rough_pairs": int(len(rough_group)),
                "rough_current_filings": int(
                    rough_group["current_accession"].nunique()
                ),
                "strict_pairs": int(strict_group["current_accession"].nunique()),
                "prior_year_pairing": _coverage_rate(
                    strict_group["current_accession"].nunique(),
                    rough_group["current_accession"].nunique(),
                ),
                "pair_exclusion_reasons": {
                    str(reason): int(count)
                    for reason, count in exclusion_group["reason"]
                    .fillna("unspecified_pair_exclusion")
                    .value_counts()
                    .sort_index()
                    .items()
                },
                "mapped_headers": int(len(mapped_group)),
                "unmapped_headers": int(len(miss_group)),
                "ticker_mapping": _coverage_rate(len(mapped_group), mapped_total),
            }
        )
    return coverage, breakdown


def _atomic_gzip(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compressed = gzip.compress(raw, compresslevel=6, mtime=0)
    handle = tempfile.NamedTemporaryFile(
        mode="wb", prefix=".partial-", suffix=".tmp", dir=path.parent, delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(compressed)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError:
        if path.read_bytes() != compressed:
            raise ValueError(f"content-addressed clean artifact conflict: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _load_raw_primary(row: object) -> bytes:
    path = Path(str(row.primary_cache_path)).resolve()
    path.relative_to(ROOT.resolve())
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"invalid primary cache path: {path}")
    try:
        content = gzip.decompress(path.read_bytes())
    except (EOFError, OSError) as exc:
        raise ValueError(f"invalid primary cache gzip: {path}") from exc
    if sha256_bytes(content) != row.primary_sha256:
        raise ValueError(f"primary cache hash mismatch: {row.accession}")
    return content


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _load_counts(path: str, expected_hash: str) -> dict[str, int]:
    target = Path(path)
    raw = gzip.decompress(target.read_bytes())
    if sha256_bytes(raw) != expected_hash:
        raise ValueError(f"clean token-count hash mismatch: {target}")
    payload = json.loads(raw)
    return {str(term): int(count) for term, count in payload["term_counts"]}


def run(args: argparse.Namespace) -> Path:
    design = _verify_receipt(args.design_receipt)
    acquisition_manifest = _verify_manifest(
        args.acquisition_manifest,
        "sec_filing_change_document_acquisition_manifest_v1",
        expected_inputs=ACQUISITION_MANIFEST_INPUT_ROLES,
        expected_outputs=ACQUISITION_MANIFEST_OUTPUT_ROLES,
        expected_top_level=frozenset(
            {
                "schema_version",
                "created_at",
                "inputs",
                "outputs",
                "outcome_join_performed",
                "design_receipt_verified",
            }
        ),
    )
    if (
        acquisition_manifest.get("outcome_join_performed") is not False
        or acquisition_manifest.get("design_receipt_verified") is not True
    ):
        raise ValueError("acquisition manifest is outcome-contaminated or unverified")
    if (
        _manifest_path_for_role(acquisition_manifest, "design_receipt")
        != args.design_receipt.resolve()
    ):
        raise ValueError("acquisition manifest belongs to a different design receipt")
    acquisition_chain_roles = {
        "frozen_acquisition_documents": "frozen_acquisition_documents",
        "acquisition_runner": "document_acquirer",
        "sec_periodic_data_code": "sec_periodic_data_code",
    }
    for manifest_role, receipt_role in acquisition_chain_roles.items():
        if _manifest_path_for_role(
            acquisition_manifest, manifest_role
        ) != _receipt_path_for_role(design, receipt_role):
            raise ValueError(
                f"acquisition manifest {manifest_role} differs from the design receipt"
            )
    if (
        _manifest_path_for_role(acquisition_manifest, "document_acquisition_audit")
        != args.acquisition_audit.resolve()
    ):
        raise ValueError("acquisition audit differs from its frozen manifest")
    plan_manifest_record = [
        record
        for record in design["records"]
        if record["role"] == "preoutcome_plan_manifest"
    ][0]
    plan_manifest_path = (ROOT / str(plan_manifest_record["path"])).resolve()
    plan_manifest = _verify_manifest(
        plan_manifest_path,
        "sec_filing_change_acquisition_plan_manifest_v1",
        expected_inputs=PLAN_MANIFEST_INPUT_ROLES,
        expected_outputs=PLAN_MANIFEST_OUTPUT_ROLES,
        expected_top_level=frozenset(
            {
                "schema_version",
                "created_at",
                "inputs",
                "outputs",
                "outcome_join_performed",
            }
        ),
    )
    if (
        _manifest_path_for_role(plan_manifest, "mapped_header_inventory")
        != args.mapped_headers.resolve()
    ):
        raise ValueError("mapped headers differ from the frozen preoutcome plan")
    acquisition_summary_path = _manifest_path_for_role(
        acquisition_manifest, "acquisition_summary"
    )
    acquisition_summary = json.loads(
        acquisition_summary_path.read_text(encoding="utf-8")
    )
    if not isinstance(acquisition_summary, dict):
        raise ValueError("acquisition summary must be a JSON object")
    dataset_path = _manifest_path_for_role(acquisition_manifest, "dataset_records")
    dataset_payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(dataset_payload, dict) or set(dataset_payload) != {"records"}:
        raise ValueError("dataset-record payload differs from the contract")
    failure_path = _manifest_path_for_role(
        acquisition_manifest, "document_acquisition_failures"
    )
    audit = pd.read_parquet(args.acquisition_audit)
    failures = _read_csv_allow_empty(failure_path)
    frozen_documents = pd.read_parquet(
        _receipt_path_for_role(design, "frozen_acquisition_documents")
    )
    _reconcile_acquisition_data(
        summary=acquisition_summary,
        manifest_created_at=acquisition_manifest.get("created_at"),
        audit=audit,
        failures=failures,
        dataset_records=dataset_payload["records"],
        frozen_documents=frozen_documents,
        design_receipt_sha256=sha256_file(args.design_receipt),
    )
    rough_pair_path = _manifest_path_for_role(plan_manifest, "rough_pairing_ledger")
    mapping_miss_path = _manifest_path_for_role(plan_manifest, "mapping_misses")
    rough_pairs = pd.read_parquet(rough_pair_path)
    mapping_misses = _read_csv_allow_empty(mapping_miss_path)
    headers = pd.read_parquet(args.mapped_headers)
    protocol_path = _receipt_path_for_role(design, "human_protocol")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    try:
        coverage_thresholds = protocol["gates"]["coverage"]
    except (KeyError, TypeError) as exc:
        raise ValueError("protocol coverage thresholds are unavailable") from exc
    if not isinstance(coverage_thresholds, dict):
        raise ValueError("protocol coverage thresholds are invalid")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"non-overwriting score output exists: {output}")
    output.mkdir(parents=True)
    parser_hash = sha256_file(SRC / "alpha_factor_engine/features/filing_change.py")
    document_rows: list[dict[str, object]] = []
    for number, row in enumerate(
        audit.sort_values("accession").itertuples(index=False), start=1
    ):
        content = _load_raw_primary(row)
        try:
            fiscal = extract_fiscal_identity(content)
            cleaned = clean_primary_filing(content)
            counts = Counter(cleaned.tokens)
            token_payload = {
                "schema_version": "sec_filing_change_term_counts_v1",
                "accession": row.accession,
                "primary_sha256": row.primary_sha256,
                "parser_sha256": parser_hash,
                "token_count": cleaned.token_count,
                "unique_token_count": cleaned.unique_token_count,
                "term_counts": sorted(counts.items()),
            }
            raw = json.dumps(
                token_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode("utf-8")
            clean_hash = sha256_bytes(raw)
            clean_path = (
                output
                / "clean_token_counts"
                / f"CIK{row.issuer_cik}"
                / row.accession
                / f"{clean_hash}.json.gz"
            )
            _atomic_gzip(clean_path, raw)
            parse_status = "OK"
            parse_error = None
            metrics = {
                "visible_characters": cleaned.visible_characters,
                "token_count": cleaned.token_count,
                "unique_token_count": cleaned.unique_token_count,
                "tables_seen": cleaned.tables_seen,
                "numeric_tables_removed": cleaned.numeric_tables_removed,
                "nonnumeric_tables_retained": cleaned.nonnumeric_tables_retained,
                "clean_counts_path": clean_path.resolve().as_posix(),
                "clean_counts_sha256": clean_hash,
            }
        except (FilingChangeError, UnicodeError, ValueError) as exc:
            fiscal = {
                "fiscal_year": None,
                "fiscal_period": None,
                "period_end": None,
                "fiscal_identity_complete": False,
            }
            parse_status = "EXCLUDED"
            parse_error = f"{type(exc).__name__}: {exc}"
            metrics = {
                "visible_characters": None,
                "token_count": None,
                "unique_token_count": None,
                "tables_seen": None,
                "numeric_tables_removed": None,
                "nonnumeric_tables_retained": None,
                "clean_counts_path": None,
                "clean_counts_sha256": None,
            }
        document_rows.append(
            {
                "issuer_cik": row.issuer_cik,
                "accession": row.accession,
                "form": row.form,
                "ticker": row.ticker,
                "raw_header_acceptance_at": row.raw_header_acceptance_at,
                "primary_sha256": row.primary_sha256,
                "parser_sha256": parser_hash,
                **fiscal,
                **metrics,
                "parse_status": parse_status,
                "parse_error": parse_error,
            }
        )
        if number == 1 or number % 100 == 0 or number == len(audit):
            print(f"SEC filing parse {number}/{len(audit)}", flush=True)
    document_audit = pd.DataFrame(document_rows)
    usable_docs = document_audit.loc[document_audit["parse_status"].eq("OK")].copy()
    paired_headers = headers.merge(
        usable_docs[
            [
                "accession",
                "raw_header_acceptance_at",
                "fiscal_year",
                "fiscal_period",
                "period_end",
                "clean_counts_path",
                "clean_counts_sha256",
                "token_count",
            ]
        ],
        on="accession",
        how="inner",
        validate="one_to_one",
    )
    strict_input = paired_headers.rename(
        columns={
            "issuer_cik": "cik",
            "raw_header_acceptance_at": "accepted_at",
            "report_date": "report_date",
        }
    )
    strict_pairs, pair_exclusions = build_strict_fiscal_pairs(strict_input)
    doc_lookup = usable_docs.set_index("accession")
    score_rows: list[dict[str, object]] = []
    for pair in strict_pairs.itertuples(index=False):
        current_doc = doc_lookup.loc[pair.current_accession]
        prior_doc = doc_lookup.loc[pair.prior_accession]
        current_counts = _load_counts(
            current_doc.clean_counts_path, current_doc.clean_counts_sha256
        )
        prior_counts = _load_counts(
            prior_doc.clean_counts_path, prior_doc.clean_counts_sha256
        )
        text_eligible = (
            int(current_doc.token_count) >= 1000 and int(prior_doc.token_count) >= 1000
        )
        base = pair._asdict()
        base.update(
            {
                "current_token_count": int(current_doc.token_count),
                "prior_token_count": int(prior_doc.token_count),
                "text_eligible": text_eligible,
                "jaccard_similarity": (
                    binary_jaccard_counts(current_counts, prior_counts)
                    if text_eligible
                    else np.nan
                ),
                "cosine_similarity": (
                    raw_tf_cosine_counts(current_counts, prior_counts)
                    if text_eligible
                    else np.nan
                ),
            }
        )
        score_rows.append(base)
    raw_scores = pd.DataFrame(score_rows)
    text_scores = raw_scores.loc[raw_scores["text_eligible"].eq(True)].copy()
    jaccard, jaccard_cutoffs = assign_release_month_quintiles(
        text_scores,
        score_column="jaccard_similarity",
        minimum_monthly_observations=100,
    )
    jaccard = jaccard.rename(
        columns={
            "similarity_quintile": "jaccard_quintile",
            "sort_eligible": "jaccard_sort_eligible",
        }
    )
    cosine, cosine_cutoffs = assign_release_month_quintiles(
        text_scores,
        score_column="cosine_similarity",
        minimum_monthly_observations=100,
    )
    cosine = cosine[
        ["current_accession", "similarity_quintile", "sort_eligible"]
    ].rename(
        columns={
            "similarity_quintile": "cosine_quintile",
            "sort_eligible": "cosine_sort_eligible",
        }
    )
    scored = jaccard.merge(cosine, on="current_accession", validate="one_to_one")
    scored["similarity_quintile"] = scored["jaccard_quintile"]
    scored["sort_eligible"] = scored["jaccard_sort_eligible"]
    cutoffs = pd.concat([jaccard_cutoffs, cosine_cutoffs], ignore_index=True)
    upstream_coverage, coverage_by_year_form = _build_upstream_coverage(
        thresholds=coverage_thresholds,
        acquisition_summary=acquisition_summary,
        acquisition_audit=audit,
        document_audit=document_audit,
        rough_pairs=rough_pairs,
        strict_pairs=strict_pairs,
        pair_exclusions=pair_exclusions,
        mapped_headers=headers,
        mapping_misses=mapping_misses,
    )

    document_audit_path = output / "document_parse_audit.parquet"
    pair_path = output / "strict_fiscal_pairs.parquet"
    pair_exclusion_path = output / "strict_pair_exclusions.csv"
    raw_score_path = output / "raw_similarity_scores.parquet"
    score_path = output / "sorted_similarity_scores.parquet"
    cutoff_path = output / "monthly_cutoff_ledger.csv"
    document_audit.to_parquet(document_audit_path, index=False)
    strict_pairs.to_parquet(pair_path, index=False)
    pair_exclusions.to_csv(pair_exclusion_path, index=False)
    raw_scores.to_parquet(raw_score_path, index=False)
    scored.to_parquet(score_path, index=False)
    cutoffs.to_csv(cutoff_path, index=False)
    summary = {
        "schema_version": "sec_filing_change_score_summary_v2",
        "created_at": datetime.now(UTC).isoformat(),
        "status": (
            "DETERMINISTIC_SIGNAL_BUILD_ONLY"
            if upstream_coverage["all_signal_build_gates_passed"]
            else "BLOCKED_DATA_COVERAGE"
        ),
        "documents_attempted": int(len(document_audit)),
        "documents_parsed": int(document_audit["parse_status"].eq("OK").sum()),
        "fiscal_identity_complete": int(
            document_audit["fiscal_identity_complete"].sum()
        ),
        "strict_fiscal_pairs": int(len(strict_pairs)),
        "text_eligible_pairs": int(len(text_scores)),
        "primary_sort_eligible_pairs": int(scored["sort_eligible"].sum()),
        "primary_tail_signals": int(scored["similarity_quintile"].isin([1, 5]).sum()),
        "outcome_join_performed": False,
        "historical_alpha_verdict": "INVALID",
        "deployment_verdict": "NOT AUTHORIZED",
        "design_receipt_verified": design.get("status") == "FROZEN_PRE_OUTCOME",
        "upstream_coverage": upstream_coverage,
        "coverage_by_year_form": coverage_by_year_form,
    }
    summary_path = output / "score_summary.json"
    _write_json(summary_path, summary)
    manifest = {
        "schema_version": "sec_filing_change_score_manifest_v2",
        "created_at": summary["created_at"],
        "inputs": [
            _record(args.design_receipt, "design_receipt"),
            _record(protocol_path, "human_protocol"),
            _record(args.acquisition_manifest, "acquisition_manifest"),
            _record(acquisition_summary_path, "acquisition_summary"),
            _record(args.acquisition_audit, "acquisition_audit"),
            _record(failure_path, "acquisition_failures"),
            _record(dataset_path, "dataset_records"),
            _record(
                _receipt_path_for_role(design, "frozen_acquisition_documents"),
                "frozen_acquisition_documents",
            ),
            _record(args.mapped_headers, "mapped_header_inventory"),
            _record(rough_pair_path, "rough_pairing_ledger"),
            _record(mapping_miss_path, "mapping_misses"),
            _record(Path(__file__), "score_builder"),
            _record(
                SRC / "alpha_factor_engine/features/filing_change.py", "feature_code"
            ),
        ],
        "outputs": [
            _record(document_audit_path, "document_parse_audit"),
            _record(pair_path, "strict_fiscal_pairs"),
            _record(pair_exclusion_path, "strict_pair_exclusions"),
            _record(raw_score_path, "raw_similarity_scores"),
            _record(score_path, "sorted_similarity_scores"),
            _record(cutoff_path, "monthly_cutoff_ledger"),
            _record(summary_path, "score_summary"),
        ],
        "outcome_join_performed": False,
        "upstream_coverage_status": upstream_coverage["status"],
        "all_signal_build_coverage_gates_passed": upstream_coverage[
            "all_signal_build_gates_passed"
        ],
        "benchmark_clock_deferred_to_outcome_runner": True,
    }
    manifest_path = output / "score_manifest.json"
    _write_json(manifest_path, manifest)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--design-receipt",
        type=Path,
        default=ROOT / "artifacts/sec_filing_change_stage_a_v1/design_receipt_v2.json",
    )
    parser.add_argument(
        "--acquisition-manifest",
        type=Path,
        default=ROOT
        / "artifacts/sec_filing_change_stage_a_v1/acquisition_smoke_v2/acquisition_manifest.json",
    )
    parser.add_argument(
        "--acquisition-audit",
        type=Path,
        default=ROOT
        / "artifacts/sec_filing_change_stage_a_v1/acquisition_smoke_v2/document_acquisition_audit.parquet",
    )
    parser.add_argument(
        "--mapped-headers",
        type=Path,
        default=ROOT
        / "artifacts/sec_filing_change_stage_a_v1/preoutcome_plan_v3/mapped_header_inventory.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/sec_filing_change_stage_a_v1/scores_smoke_v2",
    )
    args = parser.parse_args()
    print(run(args))


if __name__ == "__main__":
    main()
