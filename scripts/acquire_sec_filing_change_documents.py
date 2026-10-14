#!/usr/bin/env python3
"""Acquire frozen SEC primary documents and raw acceptance headers."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlparse

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alpha_factor_engine.data.sec_periodic_filings import SecPeriodicFilingsClient


PHASE_PRIORITY = {
    "00_top25_engineering_smoke": 0,
    "10_full_development_and_confirmation": 1,
    "20_sealed_2025_signal_only": 2,
}
REQUIRED_DOCUMENT_COLUMNS = frozenset(
    {
        "issuer_cik",
        "accession",
        "form",
        "ticker",
        "acceptance_at",
        "acceptance_at_raw",
        "primary_document",
        "primary_document_url",
        "proxy_universe_rank",
        "download_phase",
        "outcome_join_permitted_during_acquisition",
    }
)
ALLOWED_DOCUMENT_COLUMNS = REQUIRED_DOCUMENT_COLUMNS | frozenset(
    {
        "filing_date",
        "report_date",
        "source_kind",
        "source_path",
        "source_file_sha256",
        "source_content_sha256",
        "evidence_status",
        "selection_as_of",
        "document_role",
    }
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


def _read_design_receipt(path: Path) -> dict[str, object]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise ValueError("design receipt must be a JSON object")
    if receipt.get("schema_version") != "sec_filing_change_design_receipt_v2":
        raise ValueError("unsupported design receipt")
    if set(receipt) != DESIGN_RECEIPT_TOP_LEVEL:
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
    records = receipt.get("records")
    if not isinstance(records, list) or len(records) != len(required_roles):
        raise ValueError("design receipt has incomplete records")
    roles = [item.get("role") for item in records if isinstance(item, dict)]
    if (
        len(roles) != len(records)
        or set(roles) != required_roles
        or len(roles) != len(set(roles))
    ):
        raise ValueError("design receipt roles differ from the frozen contract")
    if (
        receipt.get("primary_signal") != "jaccard_similarity"
        or receipt.get("secondary_signal") != "cosine_similarity"
        or receipt.get("current_disclosed_trial_floor") != 130
        or receipt.get("experiment") != "sec_filing_change_stage_a_v1"
        or receipt.get("status") != "FROZEN_PRE_OUTCOME"
        or receipt.get("outcome_join_performed") is not False
        or receipt.get("sealed_2025_signal_return_joined") is not False
        or receipt.get("non_overwrite_policy") is not True
        or receipt.get("historical_alpha_verdict") != "INVALID"
        or receipt.get("deployment_verdict") != "NOT AUTHORIZED"
    ):
        raise ValueError("design receipt top-level constraints are invalid")
    created_at = receipt.get("created_at")
    try:
        created_timestamp = datetime.fromisoformat(
            str(created_at).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ValueError("design receipt creation timestamp is invalid") from exc
    if not isinstance(created_at, str) or created_timestamp.tzinfo is None:
        raise ValueError("design receipt creation timestamp is invalid")
    receipt_sources: dict[str, Path] = {}
    for item in records:
        if set(item) != {"role", "path", "bytes", "sha256"}:
            raise ValueError(f"design receipt record is malformed: {item.get('role')}")
        raw_path = item.get("path")
        expected_bytes = item.get("bytes")
        expected_hash = item.get("sha256")
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
            raise ValueError(f"design receipt record is malformed: {item['role']}")
        candidate = ROOT / raw_path
        if candidate.is_symlink():
            raise ValueError(
                f"design receipt source may not be a symlink: {item['role']}"
            )
        source = candidate.resolve()
        source.relative_to(ROOT.resolve())
        if not source.is_file():
            raise FileNotFoundError(source)
        if (
            source.stat().st_size != expected_bytes
            or sha256_file(source) != expected_hash
        ):
            raise ValueError(f"design receipt hash mismatch: {item['role']}")
        receipt_sources[str(item["role"])] = source
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
    return receipt


def _validate_acquisition_documents(documents: pd.DataFrame) -> None:
    missing = REQUIRED_DOCUMENT_COLUMNS.difference(documents.columns)
    if missing:
        raise ValueError(f"acquisition plan missing columns: {sorted(missing)}")
    unexpected = set(documents.columns).difference(ALLOWED_DOCUMENT_COLUMNS)
    if unexpected:
        raise ValueError(
            f"acquisition plan has unexpected columns: {sorted(unexpected)}"
        )
    null_columns = sorted(
        column for column in REQUIRED_DOCUMENT_COLUMNS if documents[column].isna().any()
    )
    if null_columns:
        raise ValueError(f"acquisition plan has null required values: {null_columns}")
    if not documents["accession"].is_unique:
        raise ValueError("acquisition plan has duplicate accessions")
    ciks = documents["issuer_cik"].astype("string")
    accessions = documents["accession"].astype("string")
    if (
        not ciks.str.fullmatch(r"\d{10}").all()
        or pd.to_numeric(ciks, errors="coerce").le(0).any()
        or not accessions.str.fullmatch(r"\d{10}-\d{2}-\d{6}").all()
    ):
        raise ValueError("acquisition plan has invalid CIK or accession identifiers")
    if not documents["form"].isin({"10-K", "10-Q"}).all():
        raise ValueError("acquisition plan has unsupported filing forms")
    primary_documents = documents["primary_document"].astype("string")
    if not primary_documents.str.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*\.(?:htm|html)", case=False
    ).all():
        raise ValueError("acquisition plan has invalid primary document names")
    for cik, accession, document, raw_url in zip(
        ciks,
        accessions,
        primary_documents,
        documents["primary_document_url"].astype("string"),
        strict=True,
    ):
        parsed = urlparse(str(raw_url))
        expected_path = (
            f"/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/{document}"
        )
        if (
            parsed.scheme != "https"
            or parsed.netloc != "www.sec.gov"
            or parsed.path != expected_path
            or parsed.params
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                f"acquisition plan has invalid SEC primary URL: {accession}"
            )
    ranks = documents["proxy_universe_rank"]
    if not pd.api.types.is_integer_dtype(ranks.dtype) or ranks.le(0).any():
        raise ValueError("acquisition plan has invalid proxy universe ranks")
    acceptance_at = documents["acceptance_at"]
    if (
        not isinstance(acceptance_at.dtype, pd.DatetimeTZDtype)
        or str(acceptance_at.dtype.tz) != "UTC"
    ):
        raise ValueError("acquisition plan acceptance_at must be timezone-aware UTC")
    raw_acceptance = documents["acceptance_at_raw"].astype("string")
    if not raw_acceptance.str.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z"
    ).all():
        raise ValueError("acquisition plan has invalid raw acceptance timestamps")
    parsed_raw_acceptance = pd.to_datetime(
        raw_acceptance,
        format="%Y-%m-%dT%H:%M:%S.%fZ",
        errors="coerce",
        utc=True,
    )
    if (
        parsed_raw_acceptance.isna().any()
        or not parsed_raw_acceptance.eq(acceptance_at).all()
    ):
        raise ValueError("acquisition plan acceptance timestamps disagree")
    unknown_phase = ~documents["download_phase"].isin(PHASE_PRIORITY)
    if unknown_phase.any():
        values = sorted(
            {str(value) for value in documents.loc[unknown_phase, "download_phase"]}
        )
        raise ValueError(f"acquisition plan has unknown download phases: {values}")
    outcome_permission = documents["outcome_join_permitted_during_acquisition"]
    if (
        not pd.api.types.is_bool_dtype(outcome_permission.dtype)
        or outcome_permission.ne(False).any()
    ):
        raise ValueError("acquisition plan permits an outcome join")


def _acquisition_status(*, failure_count: int, selection_limited: bool) -> str:
    if failure_count:
        return "INCOMPLETE_RETRY_REQUIRED"
    if selection_limited:
        return "PARTIAL_LIMITED_ENGINEERING_SAMPLE"
    return "COMPLETE"


def _json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> Path:
    if args.through_phase not in PHASE_PRIORITY:
        raise ValueError(f"unknown phase: {args.through_phase}")
    design = _read_design_receipt(args.design_receipt)
    frozen_documents = [
        item
        for item in design["records"]
        if item["role"] == "frozen_acquisition_documents"
    ][0]
    documents_path = args.documents.resolve()
    expected_documents_path = (ROOT / str(frozen_documents["path"])).resolve()
    if documents_path != expected_documents_path:
        raise ValueError("documents path differs from the frozen design receipt")
    if (
        documents_path.stat().st_size != int(frozen_documents["bytes"])
        or sha256_file(documents_path) != frozen_documents["sha256"]
    ):
        raise ValueError("documents file differs from the frozen design receipt")
    documents = pd.read_parquet(documents_path)
    _validate_acquisition_documents(documents)
    cutoff = PHASE_PRIORITY[args.through_phase]
    selected = documents.loc[
        documents["download_phase"].map(PHASE_PRIORITY).le(cutoff)
    ].copy()
    selected = selected.sort_values(
        ["download_phase", "proxy_universe_rank", "acceptance_at", "accession"],
        kind="mergesort",
    )
    eligible_documents = len(selected)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        selected = selected.head(args.limit)
    selection_limited = len(selected) < eligible_documents
    if selected.empty:
        raise ValueError("selected acquisition phase contains no documents")
    client = SecPeriodicFilingsClient(
        cache_dir=args.cache_dir,
        user_agent=args.user_agent or os.getenv("SEC_USER_AGENT"),
        timeout=args.timeout,
        min_interval_seconds=args.minimum_interval_seconds,
        max_attempts=args.max_attempts,
    )
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"non-overwriting acquisition output exists: {output}")
    output.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    for number, row in enumerate(selected.itertuples(index=False), start=1):
        try:
            _, raw_clock, clock_record, clock_path = client.download_acceptance_header(
                issuer_cik=row.issuer_cik,
                accession=row.accession,
                api_acceptance_at=row.acceptance_at_raw,
                maximum_delta_seconds=1.0,
            )
            content, primary_record, primary_path = client.download_primary_html(
                issuer_cik=row.issuer_cik,
                accession=row.accession,
                primary_document=row.primary_document,
                acceptance_at=row.acceptance_at_raw,
            )
            if primary_record.source_url != row.primary_document_url:
                raise ValueError(
                    f"primary URL differs from frozen plan: {row.accession}"
                )
            rows.append(
                {
                    "issuer_cik": row.issuer_cik,
                    "accession": row.accession,
                    "form": row.form,
                    "ticker": row.ticker,
                    "download_phase": row.download_phase,
                    "api_acceptance_at_raw": row.acceptance_at_raw,
                    "raw_header_acceptance_at": raw_clock,
                    "api_raw_delta_seconds": clock_record.notes[
                        "api_raw_delta_seconds"
                    ],
                    "primary_document": row.primary_document,
                    "primary_url": primary_record.source_url,
                    "primary_sha256": primary_record.sha256,
                    "primary_raw_bytes": len(content),
                    "primary_cache_path": primary_path.resolve().as_posix(),
                    "primary_cache_status": primary_record.notes["cache_status"],
                    "header_sha256": clock_record.sha256,
                    "header_cache_path": clock_path.resolve().as_posix(),
                    "header_cache_status": clock_record.notes["cache_status"],
                    "outcome_join_performed": False,
                }
            )
            records.extend((asdict(clock_record), asdict(primary_record)))
        except Exception as exc:
            failure_rows.append(
                {
                    "issuer_cik": row.issuer_cik,
                    "accession": row.accession,
                    "form": row.form,
                    "ticker": row.ticker,
                    "download_phase": row.download_phase,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "retry_required": True,
                    "outcome_join_performed": False,
                }
            )
        if number == 1 or number % 25 == 0 or number == len(selected):
            print(f"SEC filing documents {number}/{len(selected)}", flush=True)
    audit = pd.DataFrame(rows)
    failures = pd.DataFrame(failure_rows)
    audit_path = output / "document_acquisition_audit.parquet"
    failures_path = output / "document_acquisition_failures.csv"
    audit.to_parquet(audit_path, index=False)
    failures.to_csv(failures_path, index=False)
    dataset_path = output / "dataset_records.json"
    _json(dataset_path, {"records": records})
    summary = {
        "schema_version": "sec_filing_change_document_acquisition_summary_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "through_phase": args.through_phase,
        "status": _acquisition_status(
            failure_count=len(failures), selection_limited=selection_limited
        ),
        "eligible_documents_through_phase": int(eligible_documents),
        "selection_limited": selection_limited,
        "requested_documents": int(len(selected)),
        "acquired_documents": int(len(audit)),
        "failed_documents": int(len(failures)),
        "acquisition_coverage": float(len(audit) / len(selected)),
        "unique_accessions": (
            int(audit["accession"].nunique()) if not audit.empty else 0
        ),
        "unique_issuers": int(audit["issuer_cik"].nunique()) if not audit.empty else 0,
        "raw_clock_exact_match_rate": (
            float(audit["api_raw_delta_seconds"].eq(0).mean())
            if not audit.empty
            else None
        ),
        "primary_raw_bytes": (
            int(audit["primary_raw_bytes"].sum()) if not audit.empty else 0
        ),
        "outcome_join_performed": False,
        "historical_alpha_verdict": "INVALID",
        "deployment_verdict": "NOT AUTHORIZED",
        "design_receipt_sha256": sha256_file(args.design_receipt),
    }
    summary_path = output / "acquisition_summary.json"
    _json(summary_path, summary)
    manifest = {
        "schema_version": "sec_filing_change_document_acquisition_manifest_v1",
        "created_at": summary["created_at"],
        "inputs": [
            _record(args.design_receipt, "design_receipt"),
            _record(args.documents, "frozen_acquisition_documents"),
            _record(Path(__file__), "acquisition_runner"),
            _record(
                SRC / "alpha_factor_engine/data/sec_periodic_filings.py",
                "sec_periodic_data_code",
            ),
        ],
        "outputs": [
            _record(audit_path, "document_acquisition_audit"),
            _record(failures_path, "document_acquisition_failures"),
            _record(dataset_path, "dataset_records"),
            _record(summary_path, "acquisition_summary"),
        ],
        "outcome_join_performed": False,
        "design_receipt_verified": True,
    }
    manifest_path = output / "acquisition_manifest.json"
    _json(manifest_path, manifest)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--design-receipt",
        type=Path,
        default=ROOT / "artifacts/sec_filing_change_stage_a_v1/design_receipt_v2.json",
    )
    parser.add_argument(
        "--documents",
        type=Path,
        default=ROOT
        / "artifacts/sec_filing_change_stage_a_v1/preoutcome_plan_v3/acquisition_documents.parquet",
    )
    parser.add_argument(
        "--through-phase",
        choices=tuple(PHASE_PRIORITY),
        default="00_top25_engineering_smoke",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "data/cache/sec/periodic_filings",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/sec_filing_change_stage_a_v1/acquisition_smoke_v2",
    )
    parser.add_argument("--user-agent")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--minimum-interval-seconds", type=float, default=0.20)
    parser.add_argument("--max-attempts", type=int, default=5)
    args = parser.parse_args()
    print(run(args))


if __name__ == "__main__":
    main()
