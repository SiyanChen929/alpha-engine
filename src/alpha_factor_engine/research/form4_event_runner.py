"""Manifest-verified, fail-closed runner for SEC Form 4 event diagnostics.

This runner is deliberately narrower than a full alpha-validation pipeline.  It
can turn a frozen input bundle into auditable event, matched-control, placebo,
and two-leg ledger diagnostics.  It never declares alpha ``VALIDATED``.  A
missing lineage or execution input blocks return/ledger output entirely.

Input files are read once, checked against an immutable SHA256/byte manifest,
and parsed from the verified bytes.  The real-data adapter contract is emitted
with every run so a synthetic fixture cannot be confused with historical
evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import platform
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from alpha_factor_engine.contracts import PointInTimeViolation, validate_unique
from alpha_factor_engine.data.sec_insider import (
    AcceptanceClockError,
    InsiderDataSchemaError,
    attach_first_public_clock,
    build_first_daily_public_evidence,
    parse_acceptance_header,
    parse_daily_master_index,
)
from alpha_factor_engine.research.form4_event_study import (
    RESEARCH_CANDIDATE,
    Form4EventStudyError,
    build_prior_close_eligibility_snapshot,
    canonical_frame_sha256,
    compute_open_to_open_returns,
    compute_pre_event_placebo,
    exchange_calendar_sha256,
    match_liquidity_risk_controls,
    suppress_causal_overlaps,
    validate_event_entry_clocks,
)


INPUT_SCHEMA = "form4_event_study_input_manifest_v1"
RUNNER_SCHEMA = "form4_event_study_run_manifest_v1"
ARTIFACT_SCHEMA = "form4_event_study_artifact_manifest_v1"
RESEARCH_STATUS = "NOT VALIDATED"
SYNTHETIC_SOURCE = "synthetic_fixture"
HISTORICAL_SOURCE = "historical_research"
ALLOWED_SOURCE_CLASSES = frozenset({SYNTHETIC_SOURCE, HISTORICAL_SOURCE})
MANDATORY_HORIZONS = frozenset({5, 20, 60})
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
ACCESSION_PATTERN = re.compile(r"\d{10}-\d{2}-\d{6}")


class Form4EventRunnerError(ValueError):
    """Base class for runner input or execution failures."""


class ManifestIntegrityError(Form4EventRunnerError):
    """Raised when bytes differ from their manifest declaration."""


class EvidenceGateError(Form4EventRunnerError):
    """Raised when verified bytes do not prove a required research gate."""


@dataclass(frozen=True)
class VerifiedRecord:
    role: str
    path: Path
    content: bytes
    sha256: str
    byte_count: int
    media_type: str
    metadata: Mapping[str, Any]


@dataclass
class GateBook:
    rows: list[dict[str, Any]]

    @classmethod
    def create(cls) -> "GateBook":
        return cls(rows=[])

    def add(
        self,
        gate: str,
        passed: bool,
        evidence: str,
        *,
        required_for_diagnostic: bool = True,
        required_for_research_candidate: bool = True,
    ) -> None:
        if any(row["gate"] == gate for row in self.rows):
            raise Form4EventRunnerError(f"duplicate gate: {gate}")
        self.rows.append(
            {
                "gate": gate,
                "passed": bool(passed),
                "required_for_diagnostic": bool(required_for_diagnostic),
                "required_for_research_candidate": bool(
                    required_for_research_candidate
                ),
                "evidence": str(evidence),
            }
        )

    def diagnostic_passed(self) -> bool:
        return all(
            (not row["required_for_diagnostic"]) or row["passed"]
            for row in self.rows
        )

    def research_candidate_passed(self) -> bool:
        return all(
            (not row["required_for_research_candidate"]) or row["passed"]
            for row in self.rows
        )


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_bytes(content: bytes, *, source: str) -> Mapping[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ManifestIntegrityError(
                    f"{source} contains duplicate JSON key {key!r}"
                )
            output[key] = value
        return output

    def reject_constant(value: str) -> None:
        raise ManifestIntegrityError(f"{source} contains non-finite JSON {value}")

    try:
        decoded = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except ManifestIntegrityError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ManifestIntegrityError(f"{source} is not valid strict JSON") from exc
    if not isinstance(decoded, Mapping):
        raise ManifestIntegrityError(f"{source} must contain a JSON object")
    return decoded


def _exact_keys(payload: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    actual = set(payload)
    if actual != expected:
        raise ManifestIntegrityError(
            f"{name} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _read_stable(path: Path) -> bytes:
    if path.is_symlink():
        raise ManifestIntegrityError(f"manifest input cannot be a symlink: {path}")
    try:
        before = path.stat()
        content = path.read_bytes()
        after = path.stat()
    except OSError as exc:
        raise ManifestIntegrityError(f"cannot read manifest input: {path}") from exc
    signature_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    signature_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if signature_before != signature_after:
        raise ManifestIntegrityError(f"manifest input changed while reading: {path}")
    return content


def load_verified_input_bundle(
    manifest_path: str | Path,
) -> tuple[Mapping[str, Any], str, list[VerifiedRecord]]:
    """Load a bundle using one stable byte snapshot per declared file."""

    manifest = Path(manifest_path).resolve()
    manifest_content = _read_stable(manifest)
    payload = _strict_json_bytes(manifest_content, source=str(manifest))
    _exact_keys(
        payload,
        {"schema_version", "snapshot_id", "source_class", "created_at", "records"},
        name="input manifest",
    )
    if payload["schema_version"] != INPUT_SCHEMA:
        raise ManifestIntegrityError(
            f"unsupported input manifest schema: {payload['schema_version']!r}"
        )
    if payload["source_class"] not in ALLOWED_SOURCE_CLASSES:
        raise ManifestIntegrityError(
            f"source_class must be one of {sorted(ALLOWED_SOURCE_CLASSES)}"
        )
    if not str(payload["snapshot_id"]).strip():
        raise ManifestIntegrityError("snapshot_id cannot be empty")
    created_at = pd.Timestamp(payload["created_at"])
    if pd.isna(created_at) or created_at.tzinfo is None:
        raise ManifestIntegrityError("created_at must be timezone-aware")
    raw_records = payload["records"]
    if not isinstance(raw_records, list) or not raw_records:
        raise ManifestIntegrityError("input manifest records must be a non-empty list")

    base = manifest.parent.resolve()
    records: list[VerifiedRecord] = []
    seen_roles: set[str] = set()
    seen_paths: set[Path] = set()
    for position, raw in enumerate(raw_records):
        if not isinstance(raw, Mapping):
            raise ManifestIntegrityError(f"input record {position} is not an object")
        _exact_keys(
            raw,
            {"role", "path", "bytes", "sha256", "media_type", "metadata"},
            name=f"input record {position}",
        )
        role = str(raw["role"]).strip()
        if not re.fullmatch(r"[a-z0-9_:-]+", role):
            raise ManifestIntegrityError(f"invalid input role: {role!r}")
        if role in seen_roles:
            raise ManifestIntegrityError(f"duplicate input role: {role}")
        seen_roles.add(role)
        relative = PurePosixPath(str(raw["path"]))
        if relative.is_absolute() or ".." in relative.parts or "\\" in str(raw["path"]):
            raise ManifestIntegrityError(
                f"input paths must be safe paths relative to manifest: {raw['path']!r}"
            )
        path = (base / Path(*relative.parts)).resolve()
        try:
            path.relative_to(base)
        except ValueError as exc:
            raise ManifestIntegrityError(f"input escapes bundle directory: {path}") from exc
        if path in seen_paths:
            raise ManifestIntegrityError(f"input path declared more than once: {path}")
        seen_paths.add(path)
        if not isinstance(raw["bytes"], int) or raw["bytes"] < 0:
            raise ManifestIntegrityError(f"invalid byte count for role {role}")
        declared_sha = str(raw["sha256"]).lower()
        if not SHA256_PATTERN.fullmatch(declared_sha):
            raise ManifestIntegrityError(f"invalid SHA256 for role {role}")
        if not isinstance(raw["metadata"], Mapping):
            raise ManifestIntegrityError(f"metadata for role {role} must be an object")
        content = _read_stable(path)
        actual_sha = _sha256_bytes(content)
        if len(content) != raw["bytes"] or actual_sha != declared_sha:
            raise ManifestIntegrityError(
                f"manifest integrity mismatch for role {role}: "
                f"declared bytes/hash={raw['bytes']}/{declared_sha}, "
                f"actual={len(content)}/{actual_sha}"
            )
        records.append(
            VerifiedRecord(
                role=role,
                path=path,
                content=content,
                sha256=actual_sha,
                byte_count=len(content),
                media_type=str(raw["media_type"]),
                metadata=dict(raw["metadata"]),
            )
        )
    return payload, _sha256_bytes(manifest_content), records


def _records_by_role(records: Sequence[VerifiedRecord]) -> dict[str, VerifiedRecord]:
    return {record.role: record for record in records}


def _required_single_roles() -> set[str]:
    return {
        "events",
        "exchange_calendar",
        "security_master",
        "daily_prices",
        "corporate_actions",
        "delistings",
        "delisting_payoffs",
        "cost_model",
        "run_spec",
    }


def _validate_source_class_provenance(
    source_class: str, records: Sequence[VerifiedRecord]
) -> None:
    """Prevent a synthetic bundle from being promoted by relabelling one field."""

    for record in records:
        serialized = json.dumps(
            record.metadata, sort_keys=True, default=str
        ).lower()
        if source_class == SYNTHETIC_SOURCE and "synthetic" not in serialized:
            raise EvidenceGateError(
                f"synthetic bundle record {record.role} lacks synthetic provenance"
            )
        if source_class == HISTORICAL_SOURCE and "synthetic" in serialized:
            raise EvidenceGateError(
                f"historical bundle record {record.role} contains synthetic provenance"
            )
    if source_class == HISTORICAL_SOURCE:
        for record in records:
            if record.role.startswith("raw_submission:"):
                url = str(record.metadata.get("source_url", ""))
                if not url.startswith("https://www.sec.gov/Archives/edgar/data/"):
                    raise EvidenceGateError(
                        f"historical raw submission lacks an official SEC archive URL: {record.role}"
                    )
            elif record.role.startswith("daily_master:"):
                url = str(record.metadata.get("source_url", ""))
                if not url.startswith(
                    "https://www.sec.gov/Archives/edgar/daily-index/"
                ):
                    raise EvidenceGateError(
                        f"historical daily master lacks an official SEC URL: {record.role}"
                    )
        role_map = _records_by_role(records)
        calendar = role_map.get("exchange_calendar")
        if calendar is None or str(calendar.metadata.get("exchange", "")) != "XNYS":
            raise EvidenceGateError(
                "historical exchange calendar must identify the XNYS calendar"
            )


def _csv(record: VerifiedRecord, *, dtype: Any = None) -> pd.DataFrame:
    try:
        return pd.read_csv(
            io.BytesIO(record.content),
            dtype=dtype,
            keep_default_na=True,
        )
    except Exception as exc:
        raise EvidenceGateError(f"cannot parse {record.role} as CSV: {exc}") from exc


def _json(record: VerifiedRecord) -> Mapping[str, Any]:
    try:
        return _strict_json_bytes(record.content, source=record.role)
    except ManifestIntegrityError as exc:
        raise EvidenceGateError(str(exc)) from exc


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, name: str) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise EvidenceGateError(f"{name} missing columns: {sorted(missing)}")


def _nonempty_text(values: pd.Series, *, name: str) -> pd.Series:
    result = values.astype("string").str.strip()
    invalid = values.isna() | result.eq("")
    if invalid.any():
        raise EvidenceGateError(
            f"{name} contains missing text: {values.loc[invalid].head(5).tolist()}"
        )
    return result


def _utc(values: pd.Series, *, name: str, allow_missing: bool = False) -> pd.Series:
    result = pd.to_datetime(values, utc=True, errors="coerce")
    present = values.notna() & values.astype("string").str.strip().ne("")
    invalid = present & result.isna()
    if not allow_missing:
        invalid |= ~present
    if invalid.any():
        raise EvidenceGateError(
            f"{name} contains invalid timestamps: {values.loc[invalid].head(5).tolist()}"
        )
    return result


def _dates(values: pd.Series, *, name: str) -> pd.Series:
    result = pd.to_datetime(values, errors="coerce")
    if result.isna().any():
        raise EvidenceGateError(
            f"{name} contains invalid dates: {values.loc[result.isna()].head(5).tolist()}"
        )
    return result.dt.normalize()


def _numbers(
    values: pd.Series, *, name: str, allow_missing: bool = False
) -> pd.Series:
    result = pd.to_numeric(values, errors="coerce")
    present = values.notna() & values.astype("string").str.strip().ne("")
    invalid = present & (result.isna() | ~np.isfinite(result))
    if not allow_missing:
        invalid |= ~present
    if invalid.any():
        raise EvidenceGateError(
            f"{name} contains invalid numbers: {values.loc[invalid].head(5).tolist()}"
        )
    return result.astype(float)


def _booleans(values: pd.Series, *, name: str) -> pd.Series:
    normalized = values.astype("string").str.strip().str.lower()
    result = pd.Series(pd.NA, index=values.index, dtype="boolean")
    result.loc[normalized.isin({"1", "true", "t", "yes", "y"})] = True
    result.loc[normalized.isin({"0", "false", "f", "no", "n"})] = False
    if result.isna().any():
        raise EvidenceGateError(
            f"{name} contains invalid booleans: "
            f"{values.loc[result.isna()].head(5).tolist()}"
        )
    return result


def _metadata_true(record: VerifiedRecord, key: str) -> bool:
    return record.metadata.get(key) is True


def _parse_run_spec(record: VerifiedRecord) -> Mapping[str, Any]:
    spec = _json(record)
    required = {
        "benchmark_ticker",
        "benchmark_security_id",
        "horizons",
        "primary_horizon_sessions",
        "controls_per_event",
        "adv_window_sessions",
        "volatility_window_sessions",
        "minimum_history_sessions",
        "minimum_price_usd",
        "minimum_adv_dollars",
        "maximum_staleness_calendar_days",
        "minimum_purchase_usd",
        "clock_search_start",
        "clock_search_end",
        "require_cross_quarter_daily_search",
        "price_basis",
    }
    missing = required.difference(spec)
    if missing:
        raise EvidenceGateError(f"run_spec missing keys: {sorted(missing)}")
    horizons = [int(value) for value in spec["horizons"]]
    if set(horizons) != MANDATORY_HORIZONS or len(horizons) != 3:
        raise EvidenceGateError("run_spec horizons must be exactly [5, 20, 60]")
    if int(spec["primary_horizon_sessions"]) != 20:
        raise EvidenceGateError("primary_horizon_sessions must be 20")
    for key in [
        "controls_per_event",
        "adv_window_sessions",
        "volatility_window_sessions",
        "minimum_history_sessions",
        "maximum_staleness_calendar_days",
    ]:
        if not isinstance(spec[key], int) or int(spec[key]) <= 0:
            raise EvidenceGateError(f"run_spec.{key} must be a positive integer")
    for key in ["minimum_price_usd", "minimum_adv_dollars", "minimum_purchase_usd"]:
        value = float(spec[key])
        if not math.isfinite(value) or value < 0:
            raise EvidenceGateError(f"run_spec.{key} must be finite and non-negative")
    if spec["price_basis"] not in {
        "split_dividend_adjusted_ohlc",
        "raw_ohlc_with_corporate_action_ledger",
    }:
        raise EvidenceGateError("run_spec.price_basis is unsupported")
    start = pd.Timestamp(spec["clock_search_start"])
    end = pd.Timestamp(spec["clock_search_end"])
    if pd.isna(start) or pd.isna(end) or start.tzinfo is not None or end.tzinfo is not None:
        raise EvidenceGateError("clock search bounds must be timezone-naive ISO dates")
    if start.normalize() != start or end.normalize() != end or start > end:
        raise EvidenceGateError("clock search bounds must be ordered whole dates")
    if not isinstance(spec["require_cross_quarter_daily_search"], bool):
        raise EvidenceGateError("require_cross_quarter_daily_search must be boolean")
    return dict(spec)


def _parse_cost_model(record: VerifiedRecord) -> Mapping[str, Any]:
    model = _json(record)
    required = {
        "model_version",
        "two_leg_complete",
        "event_one_way_bps",
        "control_one_way_bps",
        "control_borrow_annual_bps",
        "event_financing_annual_bps",
        "control_financing_annual_bps",
        "calendar_days_per_year",
        "included_components",
    }
    missing = required.difference(model)
    if missing:
        raise EvidenceGateError(f"cost_model missing keys: {sorted(missing)}")
    if model["two_leg_complete"] is not True:
        raise EvidenceGateError("cost_model.two_leg_complete must be true")
    for key in [
        "event_one_way_bps",
        "control_one_way_bps",
        "control_borrow_annual_bps",
        "event_financing_annual_bps",
        "control_financing_annual_bps",
    ]:
        value = float(model[key])
        if not math.isfinite(value) or value < 0:
            raise EvidenceGateError(f"cost_model.{key} must be finite and non-negative")
    if int(model["calendar_days_per_year"]) <= 0:
        raise EvidenceGateError("calendar_days_per_year must be positive")
    components = set(str(value) for value in model["included_components"])
    required_components = {
        "commission",
        "bid_ask_spread",
        "slippage",
        "market_impact",
        "borrow",
        "financing",
    }
    if not required_components.issubset(components):
        raise EvidenceGateError(
            "cost_model does not include all event/control execution components"
        )
    return dict(model)


def _parse_events(record: VerifiedRecord, spec: Mapping[str, Any]) -> pd.DataFrame:
    required_metadata = [
        "authoritative",
        "signal_filter_complete",
        "economic_event_dedup_complete",
        "amendments_retained_and_primary_excluded",
    ]
    if not all(_metadata_true(record, key) for key in required_metadata):
        raise EvidenceGateError(
            "events metadata does not prove source, signal-filter, amendment, and dedup completeness"
        )
    events = _csv(
        record,
        dtype={
            "event_id": "string",
            "accession": "string",
            "document_type": "string",
            "issuer_cik": "string",
            "ticker": "string",
            "security_id": "string",
            "sic2": "string",
        },
    )
    required = [
        "event_id",
        "accession",
        "filing_date",
        "document_type",
        "issuer_cik",
        "ticker",
        "security_id",
        "sic2",
        "transaction_code",
        "acquired_disposed_code",
        "equity_swap_involved",
        "rule_10b5_1",
        "qualifying_owner_role",
        "purchase_usd",
    ]
    _require_columns(events, required, name="events")
    if events.empty:
        raise EvidenceGateError("events input is empty")
    events["event_id"] = _nonempty_text(events["event_id"], name="events.event_id")
    events["accession"] = _nonempty_text(
        events["accession"], name="events.accession"
    )
    events["issuer_cik"] = _nonempty_text(
        events["issuer_cik"], name="events.issuer_cik"
    )
    events["ticker"] = _nonempty_text(events["ticker"], name="events.ticker").str.upper()
    events["security_id"] = _nonempty_text(
        events["security_id"], name="events.security_id"
    ).str.upper()
    events["sic2"] = _nonempty_text(events["sic2"], name="events.sic2").str.zfill(2)
    validate_unique(events, ["event_id"], "runner events")
    validate_unique(events, ["accession"], "runner event accessions")
    if (~events["accession"].str.fullmatch(ACCESSION_PATTERN)).any():
        raise EvidenceGateError("events contain invalid accession numbers")
    if (~events["issuer_cik"].str.fullmatch(r"\d{10}")).any():
        raise EvidenceGateError("events issuer_cik must be zero-padded 10-digit values")
    if (~events["sic2"].str.fullmatch(r"\d{2}")).any():
        raise EvidenceGateError("events sic2 must contain two digits")
    events["filing_date"] = _dates(events["filing_date"], name="events.filing_date")
    document = events["document_type"].astype("string").str.strip()
    transaction = events["transaction_code"].astype("string").str.strip().str.upper()
    acquired = events["acquired_disposed_code"].astype("string").str.strip().str.upper()
    swap = _booleans(events["equity_swap_involved"], name="events.equity_swap_involved")
    plan = _booleans(events["rule_10b5_1"], name="events.rule_10b5_1")
    roles = events["qualifying_owner_role"].astype("string").str.lower()
    purchase = _numbers(events["purchase_usd"], name="events.purchase_usd")
    invalid_signal = (
        ~document.eq("4")
        | ~transaction.eq("P")
        | ~acquired.eq("A")
        | swap.ne(False)
        | plan.ne(False)
        | ~roles.str.contains(r"officer|director", regex=True, na=False)
        | purchase.lt(float(spec["minimum_purchase_usd"]))
    )
    if invalid_signal.any():
        raise EvidenceGateError(
            "events include rows outside the frozen Form 4 code-P purchase signal"
        )
    events["purchase_usd"] = purchase
    return events


def _parse_calendar(record: VerifiedRecord) -> tuple[pd.DataFrame, pd.DatetimeIndex, str]:
    if not _metadata_true(record, "authoritative"):
        raise EvidenceGateError("exchange calendar metadata is not authoritative")
    calendar = _csv(record)
    _require_columns(calendar, ["open_at"], name="exchange_calendar")
    opens = pd.DatetimeIndex(
        pd.to_datetime(calendar["open_at"], utc=True, errors="coerce")
    )
    if opens.empty or opens.isna().any() or opens.has_duplicates:
        raise EvidenceGateError("exchange calendar contains invalid or duplicate opens")
    opens = opens.sort_values()
    digest = exchange_calendar_sha256(opens)
    # This also verifies 09:30 America/New_York and exact first-open semantics.
    probe = pd.DataFrame(
        {
            "event_id": ["calendar_probe"],
            "information_at": [opens[0] - pd.Timedelta(seconds=1)],
            "entry_at": [opens[0]],
            "research_scope": [RESEARCH_CANDIDATE],
        }
    )
    validate_event_entry_clocks(
        probe,
        opens,
        calendar_authoritative=True,
        calendar_provenance_sha256=digest,
    )
    return calendar, opens, digest


def _parse_security_master(record: VerifiedRecord) -> pd.DataFrame:
    required_metadata = [
        "authoritative",
        "point_in_time_complete",
        "issuer_cik_complete",
        "sic_history_complete",
    ]
    if not all(_metadata_true(record, key) for key in required_metadata):
        raise EvidenceGateError(
            "security master metadata does not prove permanent ID/CIK/SIC PIT completeness"
        )
    master = _csv(
        record,
        dtype={
            "security_id": "string",
            "issuer_cik": "string",
            "ticker": "string",
            "sic2": "string",
        },
    )
    required = [
        "security_id",
        "issuer_cik",
        "ticker",
        "sic2",
        "known_at",
        "effective_from",
        "effective_to",
        "is_primary_listing",
        "is_benchmark",
    ]
    _require_columns(master, required, name="security_master")
    if master.empty:
        raise EvidenceGateError("security_master is empty")
    for column in ["security_id", "issuer_cik", "ticker", "sic2"]:
        master[column] = _nonempty_text(
            master[column], name=f"security_master.{column}"
        )
    master["security_id"] = master["security_id"].str.upper()
    master["ticker"] = master["ticker"].str.upper()
    master["sic2"] = master["sic2"].str.zfill(2)
    if (~master["issuer_cik"].str.fullmatch(r"\d{10}")).any():
        raise EvidenceGateError("security_master issuer_cik must be 10 digits")
    if (~master["sic2"].str.fullmatch(r"\d{2}")).any():
        raise EvidenceGateError("security_master sic2 must contain two digits")
    master["known_at"] = _utc(master["known_at"], name="security_master.known_at")
    master["effective_from"] = _utc(
        master["effective_from"], name="security_master.effective_from"
    )
    master["effective_to"] = _utc(
        master["effective_to"], name="security_master.effective_to", allow_missing=True
    )
    master["is_primary_listing"] = _booleans(
        master["is_primary_listing"], name="security_master.is_primary_listing"
    )
    master["is_benchmark"] = _booleans(
        master["is_benchmark"], name="security_master.is_benchmark"
    )
    if (master["effective_to"].notna() & master["effective_from"].ge(master["effective_to"])).any():
        raise EvidenceGateError("security_master has non-positive effective intervals")
    ordered = master.sort_values(["security_id", "effective_from"], kind="mergesort")
    for security_id, group in ordered.groupby("security_id", sort=False):
        prior_end: pd.Timestamp | None = None
        first = True
        for row in group.itertuples(index=False):
            if first:
                prior_end = row.effective_to
                first = False
                continue
            if pd.isna(prior_end) or row.effective_from < prior_end:
                raise EvidenceGateError(
                    f"security_master has overlapping intervals for {security_id}"
                )
            prior_end = row.effective_to
    return master.reset_index(drop=True)


def _coverage_bounds(record: VerifiedRecord, *, role: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    if not _metadata_true(record, "authoritative") or not _metadata_true(
        record, "complete"
    ):
        raise EvidenceGateError(f"{role} metadata is not authoritative and complete")
    try:
        start = pd.Timestamp(record.metadata["coverage_start"]).normalize()
        end = pd.Timestamp(record.metadata["coverage_end"]).normalize()
    except (KeyError, TypeError, ValueError) as exc:
        raise EvidenceGateError(f"{role} lacks valid coverage bounds") from exc
    if start.tzinfo is not None or end.tzinfo is not None or start > end:
        raise EvidenceGateError(f"{role} coverage bounds are invalid")
    return start, end


def _parse_prices(
    record: VerifiedRecord,
    *,
    master: pd.DataFrame,
    opens: pd.DatetimeIndex,
    spec: Mapping[str, Any],
) -> tuple[pd.DataFrame, str]:
    if not _metadata_true(record, "authoritative") or not _metadata_true(
        record, "complete"
    ):
        raise EvidenceGateError("daily price metadata is not authoritative and complete")
    if record.metadata.get("price_basis") != spec["price_basis"]:
        raise EvidenceGateError("daily price basis differs from run_spec")
    prices = _csv(
        record,
        dtype={"security_id": "string", "ticker": "string"},
    )
    required = [
        "security_id",
        "ticker",
        "date",
        "open",
        "close",
        "volume",
        "available_at",
    ]
    _require_columns(prices, required, name="daily_prices")
    if prices.empty:
        raise EvidenceGateError("daily_prices is empty")
    prices["security_id"] = _nonempty_text(
        prices["security_id"], name="daily_prices.security_id"
    ).str.upper()
    prices["ticker"] = _nonempty_text(
        prices["ticker"], name="daily_prices.ticker"
    ).str.upper()
    prices["date"] = _dates(prices["date"], name="daily_prices.date")
    prices["available_at"] = _utc(
        prices["available_at"], name="daily_prices.available_at"
    )
    for column in ["open", "close", "volume"]:
        prices[column] = _numbers(prices[column], name=f"daily_prices.{column}")
    if prices[["open", "close"]].le(0).any().any() or prices["volume"].lt(0).any():
        raise EvidenceGateError("daily_prices contain non-positive prices or negative volume")
    validate_unique(prices, ["security_id", "date"], "runner daily prices")
    validate_unique(prices, ["ticker", "date"], "runner daily ticker prices")
    master_pairs = set(
        master[["security_id", "ticker"]].drop_duplicates().itertuples(index=False, name=None)
    )
    price_pairs = set(
        prices[["security_id", "ticker"]].drop_duplicates().itertuples(index=False, name=None)
    )
    if not price_pairs.issubset(master_pairs):
        raise EvidenceGateError("daily_prices include security/ticker pairs absent from master")
    calendar_dates = set(
        pd.to_datetime(opens.tz_convert("America/New_York").date)
    )
    if not set(prices["date"]).issubset(calendar_dates):
        raise EvidenceGateError("daily_prices contain dates absent from exchange calendar")
    benchmark = str(spec["benchmark_ticker"]).strip().upper()
    benchmark_security_id = str(spec["benchmark_security_id"]).strip().upper()
    benchmark_rows = prices.loc[
        prices["ticker"].eq(benchmark)
        & prices["security_id"].eq(benchmark_security_id)
    ]
    if benchmark_rows.empty:
        raise EvidenceGateError("benchmark ticker/security_id is absent from daily_prices")
    return prices, canonical_frame_sha256(prices)


def _validate_event_master(
    events: pd.DataFrame,
    master: pd.DataFrame,
) -> pd.DataFrame:
    output = events.copy()
    additions: list[dict[str, Any]] = []
    for event in output.itertuples(index=False):
        candidates = master.loc[master["security_id"].eq(event.security_id)].copy()
        candidates = candidates.loc[
            candidates["known_at"].le(event.information_at)
            & candidates["effective_from"].le(event.entry_at)
            & (
                candidates["effective_to"].isna()
                | candidates["effective_to"].gt(event.entry_at)
            )
            & candidates["is_primary_listing"].eq(True)
        ]
        if len(candidates) != 1:
            raise EvidenceGateError(
                f"event {event.event_id} does not have exactly one PIT security-master row"
            )
        row = candidates.iloc[0]
        for column in ["ticker", "issuer_cik", "sic2"]:
            if str(getattr(event, column)) != str(row[column]):
                raise EvidenceGateError(
                    f"event {event.event_id} {column} conflicts with PIT security master"
                )
        additions.append(
            {
                "security_master_known_at": row["known_at"],
                "security_master_effective_at": row["effective_from"],
                "sic_known_at": row["known_at"],
                "sic_effective_at": row["effective_from"],
            }
        )
    addition = pd.DataFrame(additions, index=output.index)
    for column in addition:
        output[column] = addition[column]
    return output


def _validate_auxiliary_ledgers(
    corporate_record: VerifiedRecord,
    delisting_record: VerifiedRecord,
    payoff_record: VerifiedRecord,
    *,
    price_dates: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    minimum = price_dates.min()
    maximum = price_dates.max()
    for record, role in [
        (corporate_record, "corporate_actions"),
        (delisting_record, "delistings"),
    ]:
        start, end = _coverage_bounds(record, role=role)
        if start > minimum or end < maximum:
            raise EvidenceGateError(f"{role} coverage does not span the price panel")

    corporate = _csv(corporate_record, dtype={"security_id": "string"})
    _require_columns(
        corporate,
        [
            "security_id",
            "effective_at",
            "known_at",
            "action_type",
            "split_factor",
            "cash_dividend",
        ],
        name="corporate_actions",
    )
    if not corporate.empty:
        corporate["security_id"] = _nonempty_text(
            corporate["security_id"], name="corporate_actions.security_id"
        ).str.upper()
        corporate["effective_at"] = _utc(
            corporate["effective_at"], name="corporate_actions.effective_at"
        )
        corporate["known_at"] = _utc(
            corporate["known_at"], name="corporate_actions.known_at"
        )
        if corporate["known_at"].gt(corporate["effective_at"]).any():
            raise EvidenceGateError(
                "corporate_actions include actions known after their effective timestamp"
            )

    delistings = _csv(delisting_record, dtype={"security_id": "string"})
    _require_columns(
        delistings,
        ["security_id", "delisting_at", "known_at", "outcome_status"],
        name="delistings",
    )
    if not delistings.empty:
        delistings["security_id"] = _nonempty_text(
            delistings["security_id"], name="delistings.security_id"
        ).str.upper()
        delistings["delisting_at"] = _utc(
            delistings["delisting_at"], name="delistings.delisting_at"
        )
        delistings["known_at"] = _utc(
            delistings["known_at"], name="delistings.known_at"
        )
        validate_unique(delistings, ["security_id", "delisting_at"], "delistings")

    payoffs = _csv(payoff_record, dtype={"security_id": "string"})
    if not _metadata_true(payoff_record, "authoritative") or not _metadata_true(
        payoff_record, "complete"
    ):
        raise EvidenceGateError(
            "delisting_payoffs metadata is not authoritative and complete"
        )
    _require_columns(
        payoffs,
        [
            "security_id",
            "entry_at",
            "horizon_sessions",
            "event_total_return_including_delisting",
        ],
        name="delisting_payoffs",
    )
    if not payoffs.empty:
        payoffs["security_id"] = _nonempty_text(
            payoffs["security_id"], name="delisting_payoffs.security_id"
        ).str.upper()
        payoffs["entry_at"] = _utc(
            payoffs["entry_at"], name="delisting_payoffs.entry_at"
        )
        payoffs["horizon_sessions"] = _numbers(
            payoffs["horizon_sessions"], name="delisting_payoffs.horizon_sessions"
        ).astype(int)
        payoffs["event_total_return_including_delisting"] = _numbers(
            payoffs["event_total_return_including_delisting"],
            name="delisting_payoffs.event_total_return_including_delisting",
        )
        validate_unique(
            payoffs,
            ["security_id", "entry_at", "horizon_sessions"],
            "delisting payoffs",
        )
    return corporate, delistings, payoffs


def _build_public_clocks(
    events: pd.DataFrame,
    records: Sequence[VerifiedRecord],
    *,
    opens: pd.DatetimeIndex,
    calendar_digest: str,
    spec: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_records = [record for record in records if record.role.startswith("raw_submission:")]
    daily_records = [record for record in records if record.role.startswith("daily_master:")]
    event_accessions = set(events["accession"])
    raw_by_accession: dict[str, VerifiedRecord] = {}
    raw_rows: list[dict[str, Any]] = []
    for record in raw_records:
        accession = str(record.metadata.get("accession", ""))
        if not ACCESSION_PATTERN.fullmatch(accession):
            raise EvidenceGateError(f"{record.role} lacks a valid accession metadata value")
        if accession in raw_by_accession:
            raise EvidenceGateError(f"multiple raw submissions for {accession}")
        if record.role != f"raw_submission:{accession}":
            raise EvidenceGateError(
                f"raw submission role does not match accession metadata: {record.role}"
            )
        if not _metadata_true(record, "authoritative"):
            raise EvidenceGateError(f"raw submission {accession} is not authoritative")
        event_row = events.loc[events["accession"].eq(accession)]
        if len(event_row) != 1 or str(record.metadata.get("issuer_cik", "")) != str(
            event_row.iloc[0]["issuer_cik"]
        ):
            raise EvidenceGateError(
                f"raw submission {accession} issuer metadata conflicts with event"
            )
        raw_by_accession[accession] = record
        raw_rows.append(
            {
                "accession": accession,
                "raw_header_acceptance_at": parse_acceptance_header(record.content),
                "acceptance_clock_source": "raw_edgar_sgml_header",
                "raw_submission_sha256": record.sha256,
            }
        )
    if set(raw_by_accession) != event_accessions:
        raise EvidenceGateError(
            "raw SEC submission coverage differs from event accessions; "
            f"missing={sorted(event_accessions - set(raw_by_accession))[:5]}, "
            f"extra={sorted(set(raw_by_accession) - event_accessions)[:5]}"
        )
    raw_clocks = pd.DataFrame(raw_rows)
    validate_unique(raw_clocks, ["accession"], "raw SEC clocks")

    snapshots = []
    snapshot_dates: list[pd.Timestamp] = []
    for record in daily_records:
        metadata = record.metadata
        if not _metadata_true(record, "authoritative"):
            raise EvidenceGateError(f"{record.role} is not authoritative")
        try:
            index_date = pd.Timestamp(metadata["index_date"]).normalize()
            modified = pd.Timestamp(metadata["last_modified_at"])
            source_url = str(metadata["source_url"])
        except (KeyError, TypeError, ValueError) as exc:
            raise EvidenceGateError(f"{record.role} has invalid metadata") from exc
        if index_date.tzinfo is not None or modified.tzinfo is None or not source_url:
            raise EvidenceGateError(f"{record.role} has invalid date/source metadata")
        if record.role != f"daily_master:{index_date.date().isoformat()}":
            raise EvidenceGateError(
                f"daily master role does not match index_date metadata: {record.role}"
            )
        snapshots.append(
            parse_daily_master_index(
                record.content,
                index_date=index_date,
                source_url=source_url,
                last_modified_at=modified,
                etag=str(metadata.get("etag", "")) or None,
                retrieved_at=str(metadata.get("retrieved_at", "")) or None,
            )
        )
        snapshot_dates.append(index_date)
    if not snapshots:
        raise EvidenceGateError("no daily master snapshots were supplied")
    if len(snapshot_dates) != len(set(snapshot_dates)):
        raise EvidenceGateError("daily master snapshot dates are duplicated")

    search_start = pd.Timestamp(spec["clock_search_start"]).normalize()
    search_end = pd.Timestamp(spec["clock_search_end"]).normalize()
    calendar_dates = pd.DatetimeIndex(
        pd.to_datetime(opens.tz_convert("America/New_York").date)
    )
    required_dates = set(
        calendar_dates[(calendar_dates >= search_start) & (calendar_dates <= search_end)]
    )
    actual_dates = set(snapshot_dates)
    if not required_dates.issubset(actual_dates):
        raise EvidenceGateError(
            "daily master coverage has exchange-session gaps: "
            f"{sorted(value.date().isoformat() for value in required_dates - actual_dates)[:5]}"
        )
    raw_local_dates = pd.DatetimeIndex(
        raw_clocks["raw_header_acceptance_at"].dt.tz_convert("America/New_York").dt.date
    )
    if search_start > raw_local_dates.min() or search_end < raw_local_dates.max():
        raise EvidenceGateError(
            "daily search bounds do not cover every raw acceptance local date"
        )
    quarter_keys = {
        (stamp.year, (stamp.month - 1) // 3 + 1) for stamp in required_dates
    }
    if spec["require_cross_quarter_daily_search"] and len(quarter_keys) < 2:
        raise EvidenceGateError("daily search does not span two calendar quarters")

    daily_evidence = build_first_daily_public_evidence(snapshots)
    if set(daily_evidence["accession"]) != event_accessions:
        raise EvidenceGateError(
            "first daily public evidence coverage differs from event accessions"
        )
    clocked = attach_first_public_clock(events, raw_clocks, daily_evidence)
    for row in clocked.itertuples(index=False):
        daily_row = daily_evidence.loc[
            daily_evidence["accession"].eq(row.accession)
        ].iloc[0]
        issuer = str(row.issuer_cik)
        indexed = set(str(daily_row["indexed_ciks"]).split("|"))
        archived = set(str(daily_row["archive_ciks"]).split("|"))
        if issuer not in indexed and issuer not in archived:
            raise EvidenceGateError(
                f"daily first evidence for {row.accession} lacks event issuer CIK"
            )
        raw_date = pd.Timestamp(row.accepted_at_raw).tz_convert(
            "America/New_York"
        ).date()
        if row.filing_date.date() != raw_date:
            raise EvidenceGateError(
                f"event filing date conflicts with raw acceptance for {row.accession}"
            )

    positions = opens.searchsorted(clocked["information_at"], side="right")
    if (positions >= len(opens)).any():
        raise EvidenceGateError("calendar has no executable open after an event")
    clocked["entry_at"] = pd.Series(opens.take(positions), index=clocked.index)
    clocked["research_scope"] = RESEARCH_CANDIDATE
    clocked = validate_event_entry_clocks(
        clocked,
        opens,
        calendar_authoritative=True,
        calendar_provenance_sha256=calendar_digest,
    )
    if not clocked["clock_verified"].all():
        raise EvidenceGateError("event entry clock failed authoritative verification")
    return clocked, daily_evidence


def _active_master(
    master: pd.DataFrame, *, information_at: pd.Timestamp, entry_at: pd.Timestamp
) -> pd.DataFrame:
    active = master.loc[
        master["known_at"].le(information_at)
        & master["effective_from"].le(entry_at)
        & (master["effective_to"].isna() | master["effective_to"].gt(entry_at))
        & master["is_primary_listing"].eq(True)
    ].copy()
    validate_unique(active, ["security_id"], "active PIT security master")
    validate_unique(active, ["ticker"], "active PIT ticker master")
    return active


def _eligibility_kwargs(
    *,
    spec: Mapping[str, Any],
    opens: pd.DatetimeIndex,
    calendar_digest: str,
    prices: pd.DataFrame,
    price_digest: str,
    corporate_record: VerifiedRecord,
    delisting_record: VerifiedRecord,
) -> dict[str, Any]:
    return {
        "exchange_sessions": opens,
        "calendar_authoritative": True,
        "calendar_provenance_sha256": calendar_digest,
        "price_available_column": "available_at",
        "security_master_clock_complete": True,
        "adv_window": int(spec["adv_window_sessions"]),
        "volatility_window": int(spec["volatility_window_sessions"]),
        "minimum_history_sessions": int(spec["minimum_history_sessions"]),
        "minimum_price": float(spec["minimum_price_usd"]),
        "minimum_adv_dollars": float(spec["minimum_adv_dollars"]),
        "maximum_staleness_calendar_days": int(
            spec["maximum_staleness_calendar_days"]
        ),
        "price_basis": str(spec["price_basis"]),
        "price_provenance_sha256": price_digest,
        "corporate_action_complete": True,
        "corporate_action_provenance_sha256": corporate_record.sha256,
        "corporate_action_evidence": corporate_record.content,
        "delisting_complete": True,
        "delisting_provenance_sha256": delisting_record.sha256,
        "delisting_evidence": delisting_record.content,
        "delisting_payoff_included": True,
    }


def _build_overlap_artifact(
    events: pd.DataFrame,
    *,
    opens: pd.DatetimeIndex,
    calendar_digest: str,
    horizon: int,
) -> pd.DataFrame:
    positions = opens.get_indexer(pd.DatetimeIndex(events["entry_at"]))
    if (positions < 0).any() or (positions + horizon >= len(opens)).any():
        raise EvidenceGateError("calendar cannot prove every primary exit open")
    return pd.DataFrame(
        {
            "event_id": events["event_id"].to_numpy(),
            "horizon_sessions": horizon,
            "entry_at": events["entry_at"].to_numpy(),
            "exit_open_at": opens.take(positions + horizon),
            "return_basis": "exact_independent_exchange_session_open_to_open",
            "calendar_authoritative": True,
            "calendar_provenance_sha256": calendar_digest,
        }
    )


def _match_controls(
    events: pd.DataFrame,
    *,
    master: pd.DataFrame,
    prices: pd.DataFrame,
    spec: Mapping[str, Any],
    opens: pd.DatetimeIndex,
    calendar_digest: str,
    price_digest: str,
    corporate_record: VerifiedRecord,
    delisting_record: VerifiedRecord,
) -> pd.DataFrame:
    matches: list[pd.DataFrame] = []
    known_events = (
        events.groupby("security_id", sort=True)["information_at"].min().to_dict()
    )
    eligibility_kwargs = _eligibility_kwargs(
        spec=spec,
        opens=opens,
        calendar_digest=calendar_digest,
        prices=prices,
        price_digest=price_digest,
        corporate_record=corporate_record,
        delisting_record=delisting_record,
    )
    for _, event in events.iterrows():
        active = _active_master(
            master,
            information_at=event["information_at"],
            entry_at=event["entry_at"],
        )
        active = active.loc[active["is_benchmark"].eq(False)].copy()
        decisions = pd.DataFrame(
            {
                "event_id": [
                    f"{event['event_id']}::candidate::{security_id}"
                    for security_id in active["security_id"]
                ],
                "ticker": active["ticker"].to_numpy(),
                "security_id": active["security_id"].to_numpy(),
                "issuer_cik": active["issuer_cik"].to_numpy(),
                "sic2": active["sic2"].to_numpy(),
                "information_at": event["information_at"],
                "entry_at": event["entry_at"],
                "security_master_known_at": active["known_at"].to_numpy(),
                "security_master_effective_at": active["effective_from"].to_numpy(),
                "sic_known_at": active["known_at"].to_numpy(),
                "sic_effective_at": active["effective_from"].to_numpy(),
                "research_scope": RESEARCH_CANDIDATE,
            }
        )
        candidates = build_prior_close_eligibility_snapshot(
            decisions,
            prices,
            **eligibility_kwargs,
        )
        candidates["known_event_at"] = candidates["security_id"].map(known_events)
        candidates["research_scope"] = RESEARCH_CANDIDATE
        event_frame = pd.DataFrame([event.copy()])
        event_frame["research_scope"] = RESEARCH_CANDIDATE
        match = match_liquidity_risk_controls(
            event_frame,
            candidates,
            exchange_sessions=opens,
            calendar_authoritative=True,
            calendar_provenance_sha256=calendar_digest,
            controls_per_event=int(spec["controls_per_event"]),
            feature_columns=("log_adv_dollars", "realized_volatility"),
            security_master_clock_complete=True,
            industry_clock_complete=True,
            known_event_clock_complete=True,
            known_event_provenance_sha256=canonical_frame_sha256(candidates),
        )
        matches.append(match)
    output = pd.concat(matches, ignore_index=True) if matches else pd.DataFrame()
    if output.empty:
        raise EvidenceGateError("control matching produced no rows")
    expected = int(spec["controls_per_event"])
    counts = output.loc[output["match_status"].eq("matched")].groupby("event_id").size()
    event_ids = set(events["event_id"])
    if set(counts.index) != event_ids or not counts.eq(expected).all():
        failed = output.loc[
            ~output["match_status"].eq("matched"),
            ["event_id", "match_status", "eligible_candidate_count"],
        ]
        raise EvidenceGateError(
            "every selected event must have the frozen number of controls; "
            f"unmatched={failed.head(5).to_dict('records')}"
        )
    return output


def _event_specific_payoffs(
    legs: pd.DataFrame, payoffs: pd.DataFrame
) -> pd.DataFrame:
    columns = [
        "event_id",
        "horizon_sessions",
        "event_total_return_including_delisting",
    ]
    if payoffs.empty:
        return pd.DataFrame(columns=columns)
    left = legs[["event_id", "security_id", "entry_at"]].drop_duplicates()
    joined = left.merge(
        payoffs,
        on=["security_id", "entry_at"],
        how="inner",
        validate="one_to_many",
    )
    return joined[
        ["event_id", "horizon_sessions", "event_total_return_including_delisting"]
    ]


def _ensure_delisting_payoffs(
    legs: pd.DataFrame,
    *,
    payoffs: pd.DataFrame,
    delistings: pd.DataFrame,
    opens: pd.DatetimeIndex,
    horizons: Sequence[int],
) -> None:
    if delistings.empty:
        return
    position_by_entry = {value: position for position, value in enumerate(opens)}
    payoff_keys = set(
        payoffs[["security_id", "entry_at", "horizon_sessions"]].itertuples(
            index=False, name=None
        )
    )
    for leg in legs.itertuples(index=False):
        position = position_by_entry.get(leg.entry_at)
        if position is None:
            raise EvidenceGateError("leg entry is absent from exchange calendar")
        security_delistings = delistings.loc[
            delistings["security_id"].eq(leg.security_id)
        ]
        for horizon in horizons:
            if position + horizon >= len(opens):
                raise EvidenceGateError("calendar cannot prove a leg exit")
            exit_at = opens[position + horizon]
            applicable = security_delistings.loc[
                security_delistings["delisting_at"].gt(leg.entry_at)
                & security_delistings["delisting_at"].le(exit_at)
            ]
            key = (leg.security_id, leg.entry_at, int(horizon))
            if not applicable.empty and key not in payoff_keys:
                raise EvidenceGateError(
                    f"missing delisting payoff for {leg.security_id}, horizon {horizon}"
                )


def _compute_leg_returns(
    legs: pd.DataFrame,
    *,
    prices: pd.DataFrame,
    opens: pd.DatetimeIndex,
    calendar_digest: str,
    horizons: Sequence[int],
    benchmark_ticker: str,
    one_way_cost_bps: float,
    price_basis: str,
    price_digest: str,
    corporate_record: VerifiedRecord,
    delisting_record: VerifiedRecord,
    event_payoffs: pd.DataFrame,
) -> pd.DataFrame:
    payoff_digest = canonical_frame_sha256(event_payoffs)
    result = compute_open_to_open_returns(
        legs,
        prices,
        exchange_sessions=opens,
        calendar_authoritative=True,
        calendar_provenance_sha256=calendar_digest,
        horizons=horizons,
        benchmark_ticker=benchmark_ticker,
        one_way_cost_bps=one_way_cost_bps,
        price_basis=price_basis,
        price_provenance_sha256=price_digest,
        corporate_action_complete=True,
        corporate_action_provenance_sha256=corporate_record.sha256,
        corporate_action_evidence=corporate_record.content,
        delisting_complete=True,
        delisting_provenance_sha256=delisting_record.sha256,
        delisting_evidence=delisting_record.content,
        delisting_payoff_included=True,
        delisting_payoffs=event_payoffs,
        delisting_payoff_provenance_sha256=payoff_digest,
    )
    identity = legs[["event_id", "security_id", "issuer_cik"]].drop_duplicates(
        "event_id"
    )
    return result.merge(
        identity,
        on="event_id",
        how="left",
        validate="many_to_one",
    )


def _control_legs(
    events: pd.DataFrame,
    matches: pd.DataFrame,
    master: pd.DataFrame,
) -> pd.DataFrame:
    event_lookup = events.set_index("event_id")
    rows: list[dict[str, Any]] = []
    for match in matches.itertuples(index=False):
        event = event_lookup.loc[match.event_id]
        master_rows = master.loc[
            master["security_id"].eq(str(match.control_permanent_id))
            & master["ticker"].eq(str(match.control_ticker))
        ]
        if master_rows.empty:
            raise EvidenceGateError("matched control is absent from security master")
        master_row = master_rows.iloc[0]
        rows.append(
            {
                "event_id": f"{match.event_id}::control::{int(match.control_rank)}",
                "parent_event_id": match.event_id,
                "control_rank": int(match.control_rank),
                "ticker": match.control_ticker,
                "security_id": match.control_permanent_id,
                "issuer_cik": match.control_issuer_cik,
                "information_at": event["information_at"],
                "entry_at": event["entry_at"],
                "clock_verified": True,
                "eligible": True,
                "selected_after_overlap": True,
                "security_master_known_at": master_row["known_at"],
                "security_master_effective_at": master_row["effective_from"],
                "research_scope": RESEARCH_CANDIDATE,
            }
        )
    return pd.DataFrame(rows)


def _build_two_leg_ledger(
    event_returns: pd.DataFrame,
    control_returns: pd.DataFrame,
    controls: pd.DataFrame,
    *,
    cost_model: Mapping[str, Any],
    evidence_class: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    control_map = controls.set_index("event_id")[["parent_event_id", "control_rank"]]
    control = control_returns.copy().join(control_map, on="event_id", validate="many_to_one")
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    days_per_year = float(cost_model["calendar_days_per_year"])
    for event_row in event_returns.itertuples(index=False):
        event_id = event_row.event_id
        horizon = int(event_row.horizon_sessions)
        selected_controls = control.loc[
            control["parent_event_id"].eq(event_id)
            & control["horizon_sessions"].eq(horizon)
        ].sort_values("control_rank")
        if selected_controls.empty:
            raise EvidenceGateError(f"event {event_id} has no control returns")
        control_count = len(selected_controls)
        event_weight = 0.5
        control_weight = -0.5 / control_count
        entry_local_date = pd.Timestamp(event_row.entry_at).tz_convert(
            "America/New_York"
        ).date()
        exit_local_date = pd.Timestamp(event_row.exit_open_at).tz_convert(
            "America/New_York"
        ).date()
        hold_days = float((exit_local_date - entry_local_date).days)
        if hold_days <= 0:
            raise EvidenceGateError("ledger contains a non-positive holding interval")

        def append_leg(
            *,
            leg_id: str,
            security_id: str,
            ticker: str,
            direction: str,
            weight: float,
            gross_return: float,
            entry_open: float,
            exit_open: float,
            execution_bps: float,
            borrow_bps: float,
            financing_bps: float,
            control_rank: int | None,
        ) -> None:
            execution_cost = abs(weight) * 2.0 * execution_bps / 10_000.0
            borrow_cost = abs(weight) * borrow_bps / 10_000.0 * hold_days / days_per_year
            financing_cost = (
                abs(weight) * financing_bps / 10_000.0 * hold_days / days_per_year
            )
            gross_contribution = weight * gross_return
            rows.append(
                {
                    "event_id": event_id,
                    "horizon_sessions": horizon,
                    "leg_id": leg_id,
                    "leg_direction": direction,
                    "control_rank": control_rank,
                    "security_id": security_id,
                    "ticker": ticker,
                    "entry_at": event_row.entry_at,
                    "exit_open_at": event_row.exit_open_at,
                    "holding_calendar_days": hold_days,
                    "gross_capital_weight": weight,
                    "entry_open": entry_open,
                    "exit_open": exit_open,
                    "security_gross_return": gross_return,
                    "gross_return_contribution": gross_contribution,
                    "execution_cost_contribution": execution_cost,
                    "borrow_cost_contribution": borrow_cost,
                    "financing_cost_contribution": financing_cost,
                    "net_return_contribution": gross_contribution
                    - execution_cost
                    - borrow_cost
                    - financing_cost,
                    "evidence_class": evidence_class,
                    "research_status": RESEARCH_STATUS,
                    "performance_claim_permitted": False,
                }
            )

        append_leg(
            leg_id=f"{event_id}::event::{horizon}",
            security_id=str(event_row.security_id),
            ticker=str(event_row.ticker),
            direction="LONG_EVENT",
            weight=event_weight,
            gross_return=float(event_row.event_gross_return),
            entry_open=float(event_row.entry_open),
            exit_open=float(event_row.exit_open)
            if pd.notna(event_row.exit_open)
            else np.nan,
            execution_bps=float(cost_model["event_one_way_bps"]),
            borrow_bps=0.0,
            financing_bps=float(cost_model["event_financing_annual_bps"]),
            control_rank=None,
        )
        for control_row in selected_controls.itertuples(index=False):
            append_leg(
                leg_id=f"{event_id}::control::{int(control_row.control_rank)}::{horizon}",
                security_id=str(control_row.security_id),
                ticker=str(control_row.ticker),
                direction="SHORT_MATCHED_CONTROL",
                weight=control_weight,
                gross_return=float(control_row.event_gross_return),
                entry_open=float(control_row.entry_open),
                exit_open=float(control_row.exit_open)
                if pd.notna(control_row.exit_open)
                else np.nan,
                execution_bps=float(cost_model["control_one_way_bps"]),
                borrow_bps=float(cost_model["control_borrow_annual_bps"]),
                financing_bps=float(cost_model["control_financing_annual_bps"]),
                control_rank=int(control_row.control_rank),
            )

        current = pd.DataFrame(
            [row for row in rows if row["event_id"] == event_id and row["horizon_sessions"] == horizon]
        )
        gross = float(current["gross_return_contribution"].sum())
        execution = float(current["execution_cost_contribution"].sum())
        borrow = float(current["borrow_cost_contribution"].sum())
        financing = float(current["financing_cost_contribution"].sum())
        net = float(current["net_return_contribution"].sum())
        control_average = float(selected_controls["event_gross_return"].mean())
        expected_gross = 0.5 * (
            float(event_row.event_gross_return) - control_average
        )
        reconciled = bool(
            np.isclose(gross, expected_gross, atol=1e-12, rtol=0)
            and np.isclose(net, gross - execution - borrow - financing, atol=1e-12, rtol=0)
            and np.isclose(current["gross_capital_weight"].abs().sum(), 1.0, atol=1e-12)
        )
        summaries.append(
            {
                "event_id": event_id,
                "ticker": event_row.ticker,
                "security_id": event_row.security_id,
                "information_at": event_row.information_at,
                "entry_at": event_row.entry_at,
                "exit_open_at": event_row.exit_open_at,
                "horizon_sessions": horizon,
                "event_gross_return": event_row.event_gross_return,
                "matched_control_average_gross_return": control_average,
                "event_minus_control_spread_gross": float(event_row.event_gross_return)
                - control_average,
                "matched_gross_return_on_gross_capital": gross,
                "two_leg_execution_cost": execution,
                "control_borrow_cost": borrow,
                "financing_cost": financing,
                "matched_net_return_on_gross_capital": net,
                "benchmark_return_diagnostic": event_row.benchmark_return,
                "event_market_adjusted_return_gross_diagnostic": event_row.market_adjusted_return_gross,
                "ledger_reconciled": reconciled,
                "return_status": event_row.return_status,
                "evidence_class": evidence_class,
                "research_status": RESEARCH_STATUS,
                "performance_claim_permitted": False,
            }
        )
    ledger = pd.DataFrame(rows)
    summary = pd.DataFrame(summaries)
    if ledger.empty or summary.empty or not summary["ledger_reconciled"].all():
        raise EvidenceGateError("two-leg ledger failed reconciliation")
    return ledger, summary


def _interface_contract() -> Mapping[str, Any]:
    return {
        "schema_version": "form4_event_study_real_data_interface_v1",
        "research_status": RESEARCH_STATUS,
        "performance_claim_permitted": False,
        "bundle_rules": [
            "all paths are relative to input_manifest.json",
            "every file is read once and verified by exact bytes and SHA256",
            "raw_submission:<accession> contains retained EDGAR SGML bytes",
            "daily_master:<date> covers every exchange session in the frozen search range",
            "missing or changed evidence blocks returns and the ledger",
        ],
        "required_roles": sorted(_required_single_roles()),
        "repeated_roles": {
            "raw_submission:<accession>": "one exact SGML payload per event accession",
            "daily_master:<YYYY-MM-DD>": (
                "one retained SEC daily master index per exchange session in search range"
            ),
        },
        "real_source_requirements": {
            "events": "official SEC Form 3/4/5 quarterly tables plus frozen code-P filter",
            "raw_submission": "SEC complete-submission SGML, content-addressed",
            "daily_master": (
                "archived SEC daily master bytes with HTTP Last-Modified and URL"
            ),
            "exchange_calendar": "authoritative XNYS regular-open snapshot",
            "security_master": (
                "effective-dated permanent security/share-class ID, ticker, issuer CIK, SIC"
            ),
            "daily_prices": (
                "permanent-ID keyed open/close/volume with point-in-time availability"
            ),
            "corporate_actions": "complete split/dividend/reorganization history",
            "delistings": "complete terminal outcome and delisting history",
            "cost_model": (
                "event and control entry/exit costs, impact, borrow, financing"
            ),
        },
        "next_validation_layer": [
            "multi-year event sample",
            "issuer and date clustered inference",
            "conditional random-event nulls",
            "multiple-testing correction",
            "capacity sensitivity",
            "chronological and held-out-issuer validation",
        ],
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _prepare_output_dir(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise Form4EventRunnerError(f"output path is not a directory: {path}")
        if any(path.iterdir()):
            raise Form4EventRunnerError(
                f"output directory must be new or empty to prevent stale artifacts: {path}"
            )
    else:
        path.mkdir(parents=True)


def _source_audit(records: Sequence[VerifiedRecord]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "role": record.role,
                "path": record.path.as_posix(),
                "bytes": record.byte_count,
                "sha256": record.sha256,
                "integrity_verified": True,
                "media_type": record.media_type,
            }
            for record in records
        ]
    )


def _write_artifact_manifest(
    output_dir: Path,
    *,
    input_manifest_path: Path,
    records: Sequence[VerifiedRecord],
    research_scope: str,
) -> Path:
    output_files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    project_root = Path(__file__).resolve().parents[3]
    code_paths = [
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("form4_event_study.py"),
        Path(__file__).resolve().parents[1] / "data/sec_insider.py",
        Path(__file__).resolve().parents[1] / "contracts.py",
        project_root / "scripts/run_form4_event_study.py",
        project_root / "scripts/build_form4_synthetic_fixture.py",
        project_root / "tests/test_form4_event_runner.py",
    ]
    paths: list[tuple[str, Path]] = [("input_manifest", input_manifest_path)]
    paths.extend(("input", record.path) for record in records)
    paths.extend(("runner_code", path) for path in code_paths if path.is_file())
    paths.extend(("output", path.resolve()) for path in output_files)
    seen: set[Path] = set()
    manifest_records: list[dict[str, Any]] = []
    for category, path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        manifest_records.append(
            {
                "category": category,
                "path": resolved.as_posix(),
                "bytes": resolved.stat().st_size,
                "sha256": _sha256_file(resolved),
            }
        )
    payload = {
        "schema_version": ARTIFACT_SCHEMA,
        "generated_at": datetime.now(UTC).isoformat(),
        "research_status": RESEARCH_STATUS,
        "research_scope": research_scope,
        "performance_claim_permitted": False,
        "records": manifest_records,
    }
    path = output_dir / "artifact_manifest.json"
    _write_json(path, payload)
    return path


def verify_artifact_manifest(manifest_path: str | Path) -> bool:
    """Independently recompute every non-self artifact-manifest record."""

    path = Path(manifest_path).resolve()
    payload = _strict_json_bytes(_read_stable(path), source=str(path))
    if payload.get("schema_version") != ARTIFACT_SCHEMA:
        raise ManifestIntegrityError("artifact manifest has an unsupported schema")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ManifestIntegrityError("artifact manifest has no records")
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ManifestIntegrityError("artifact manifest record is not an object")
        _exact_keys(
            record,
            {"category", "path", "bytes", "sha256"},
            name="artifact manifest record",
        )
        record_path = str(record["path"])
        if record_path in seen:
            raise ManifestIntegrityError(f"duplicate artifact record: {record_path}")
        seen.add(record_path)
        target = Path(record_path)
        content = _read_stable(target)
        if len(content) != record["bytes"] or _sha256_bytes(content) != record["sha256"]:
            raise ManifestIntegrityError(
                f"artifact manifest integrity mismatch: {target}"
            )
    return True


def _finalize_run(
    *,
    output_dir: Path,
    input_manifest_path: Path,
    input_manifest_sha256: str,
    manifest_payload: Mapping[str, Any],
    records: Sequence[VerifiedRecord],
    gates: GateBook,
    research_scope: str,
    diagnostic_returns_emitted: bool,
    blocked_reason: str | None,
    counts: Mapping[str, int] | None = None,
) -> dict[str, Path]:
    gate_payload = {
        "research_status": RESEARCH_STATUS,
        "research_scope": research_scope,
        "alpha_validated": False,
        "performance_claim_permitted": False,
        "diagnostic_returns_emitted": bool(diagnostic_returns_emitted),
        "all_required_diagnostic_gates_passed": gates.diagnostic_passed(),
        "all_required_research_candidate_gates_passed": (
            gates.research_candidate_passed()
        ),
        "gates": gates.rows,
    }
    _write_json(output_dir / "validation_gates.json", gate_payload)
    _source_audit(records).to_csv(output_dir / "source_audit.csv", index=False)
    _write_json(output_dir / "real_data_interface.json", _interface_contract())
    run_manifest = {
        "schema_version": RUNNER_SCHEMA,
        "generated_at": datetime.now(UTC).isoformat(),
        "run_id": f"form4-{input_manifest_sha256[:16]}",
        "snapshot_id": manifest_payload["snapshot_id"],
        "source_class": manifest_payload["source_class"],
        "research_status": RESEARCH_STATUS,
        "research_scope": research_scope,
        "alpha_validated": False,
        "performance_claim_permitted": False,
        "diagnostic_returns_emitted": bool(diagnostic_returns_emitted),
        "blocked_reason": blocked_reason,
        "input_manifest_path": input_manifest_path.as_posix(),
        "input_manifest_sha256": input_manifest_sha256,
        "input_record_count": len(records),
        "counts": dict(counts or {}),
        "limitations": [
            "runner mechanics do not constitute statistical alpha validation",
            "benchmark subtraction is diagnostic and is not an executed hedge",
            "prospective and multiple-testing gates remain outside this runner",
        ],
    }
    _write_json(output_dir / "run_manifest.json", run_manifest)
    try:
        import pyarrow
    except ImportError:  # pragma: no cover - project dependency is required
        pyarrow_version = "unavailable"
    else:
        pyarrow_version = pyarrow.__version__
    _write_json(
        output_dir / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow_version,
        },
    )

    if blocked_reason:
        report = f"""# Form 4 event-study runner\n\n## Verdict\n\n**NOT VALIDATED.** The run stopped before return and ledger publication.\n\n- Scope: `{research_scope}`\n- Blocking evidence: `{blocked_reason}`\n- Performance claim permitted: `false`\n- Diagnostic returns emitted: `false`\n\nNo missing input was replaced, inferred, or silently dropped. See\n`validation_gates.json`, `source_audit.csv`, and `real_data_interface.json`.\n"""
    else:
        source_warning = (
            "All numeric returns are synthetic test values."
            if manifest_payload["source_class"] == SYNTHETIC_SOURCE
            else "Historical diagnostics remain exploratory pending the next validation layer."
        )
        report = f"""# Form 4 event-study runner\n\n## Verdict\n\n**NOT VALIDATED.** {source_warning}\n\n- Scope: `{research_scope}`\n- Performance claim permitted: `false`\n- Diagnostic returns emitted: `true`\n- Alpha validated: `false`\n\nThe runner verified raw SGML acceptance, the earliest retained daily-index\nappearance across the frozen cross-quarter search, first-next regular open,\npermanent security/CIK/SIC lineage, corporate actions, delistings, exact 5/20/60\nopen-to-open horizons, matched controls, pre-event placebo, and both ledger legs.\nThose checks validate mechanics for this exact bundle only.\n"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    artifact_manifest = _write_artifact_manifest(
        output_dir,
        input_manifest_path=input_manifest_path,
        records=records,
        research_scope=research_scope,
    )
    return {
        "output_dir": output_dir,
        "run_manifest": output_dir / "run_manifest.json",
        "validation_gates": output_dir / "validation_gates.json",
        "artifact_manifest": artifact_manifest,
        "report": output_dir / "report.md",
    }


def run_form4_event_study(
    input_manifest_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Execute one immutable Form 4 diagnostic bundle.

    Missing semantic evidence yields a ``NOT VALIDATED`` evidence pack without
    return or ledger files.  Byte/hash drift raises ``ManifestIntegrityError``
    before an output directory is created.
    """

    manifest_path = Path(input_manifest_path).resolve()
    manifest_payload, manifest_sha256, records = load_verified_input_bundle(
        manifest_path
    )
    target = Path(output_dir).resolve()
    gates = GateBook.create()
    role_map = _records_by_role(records)
    missing_roles = sorted(_required_single_roles().difference(role_map))
    unexpected_roles = sorted(
        role
        for role in role_map
        if role not in _required_single_roles()
        and not role.startswith("raw_submission:")
        and not role.startswith("daily_master:")
    )
    raw_count = sum(record.role.startswith("raw_submission:") for record in records)
    daily_count = sum(record.role.startswith("daily_master:") for record in records)
    roles_pass = (
        not missing_roles
        and not unexpected_roles
        and raw_count > 0
        and daily_count > 0
    )
    gates.add(
        "required_manifest_roles_present",
        roles_pass,
        f"missing={missing_roles}; unexpected={unexpected_roles}; "
        f"raw_submissions={raw_count}; daily_masters={daily_count}",
    )
    gates.add(
        "all_input_bytes_match_manifest",
        True,
        f"{len(records)}/{len(records)} records verified from one stable read",
    )
    _prepare_output_dir(target)
    if not roles_pass:
        return _finalize_run(
            output_dir=target,
            input_manifest_path=manifest_path,
            input_manifest_sha256=manifest_sha256,
            manifest_payload=manifest_payload,
            records=records,
            gates=gates,
            research_scope="DIAGNOSTIC_ONLY_BLOCKED",
            diagnostic_returns_emitted=False,
            blocked_reason="required_manifest_roles_present",
        )

    try:
        _validate_source_class_provenance(
            str(manifest_payload["source_class"]), records
        )
        gates.add(
            "source_class_provenance_consistent",
            True,
            f"{manifest_payload['source_class']} agrees with every input record",
        )
        spec = _parse_run_spec(role_map["run_spec"])
        gates.add("frozen_run_spec_valid", True, "5/20/60 horizons and 20-session primary")
        cost_model = _parse_cost_model(role_map["cost_model"])
        gates.add(
            "two_leg_cost_model_complete",
            True,
            "event/control entry+exit, spread, slippage, impact, borrow, financing",
        )
        events = _parse_events(role_map["events"], spec)
        gates.add(
            "code_p_event_contract_valid",
            True,
            f"{len(events)} unique original Form 4 code-P events",
        )
        _, opens, calendar_digest = _parse_calendar(role_map["exchange_calendar"])
        gates.add(
            "authoritative_exchange_calendar_verified",
            True,
            f"{len(opens)} exact regular opens; digest={calendar_digest}",
        )
        master = _parse_security_master(role_map["security_master"])
        gates.add(
            "permanent_security_cik_sic_pit_verified",
            True,
            f"{len(master)} effective-dated security-master rows",
        )
        prices, price_digest = _parse_prices(
            role_map["daily_prices"], master=master, opens=opens, spec=spec
        )
        gates.add(
            "point_in_time_price_panel_verified",
            True,
            f"{len(prices)} permanent-ID keyed rows; digest={price_digest}",
        )
        _, delistings, payoffs = _validate_auxiliary_ledgers(
            role_map["corporate_actions"],
            role_map["delistings"],
            role_map["delisting_payoffs"],
            price_dates=prices["date"],
        )
        gates.add(
            "corporate_action_and_delisting_ledgers_verified",
            True,
            "complete coverage metadata, exact bytes, explicit payoff interface",
        )
        clocked, daily_evidence = _build_public_clocks(
            events,
            records,
            opens=opens,
            calendar_digest=calendar_digest,
            spec=spec,
        )
        gates.add(
            "raw_sec_acceptance_clock_verified",
            True,
            f"{len(clocked)}/{len(clocked)} raw SGML headers parsed",
        )
        gates.add(
            "cross_quarter_daily_first_clock_verified",
            True,
            f"{len(daily_evidence)}/{len(clocked)} first appearances derived from retained indexes",
        )
        clocked = _validate_event_master(clocked, master)
        eligibility = build_prior_close_eligibility_snapshot(
            clocked,
            prices,
            **_eligibility_kwargs(
                spec=spec,
                opens=opens,
                calendar_digest=calendar_digest,
                prices=prices,
                price_digest=price_digest,
                corporate_record=role_map["corporate_actions"],
                delisting_record=role_map["delistings"],
            ),
        )
        gates.add(
            "prior_information_eligibility_computed",
            True,
            f"{int(eligibility['eligible'].sum())}/{len(eligibility)} eligible from prior known closes",
        )
        overlap_artifact = _build_overlap_artifact(
            eligibility,
            opens=opens,
            calendar_digest=calendar_digest,
            horizon=int(spec["primary_horizon_sessions"]),
        )
        overlapped = suppress_causal_overlaps(
            eligibility,
            exchange_sessions=opens,
            calendar_authoritative=True,
            calendar_provenance_sha256=calendar_digest,
            horizon_artifact=overlap_artifact,
            horizon_artifact_sha256=canonical_frame_sha256(overlap_artifact),
            overlap_horizon_sessions=int(spec["primary_horizon_sessions"]),
        )
        selected = overlapped.loc[
            overlapped["eligible"].eq(True)
            & overlapped["selected_after_overlap"].eq(True)
        ].copy()
        if selected.empty:
            raise EvidenceGateError("no eligible non-overlapping events remain")
        gates.add(
            "causal_overlap_selection_verified",
            True,
            f"{len(selected)} selected; {len(overlapped) - len(selected)} ineligible/suppressed",
        )
        matches = _match_controls(
            selected,
            master=master,
            prices=prices,
            spec=spec,
            opens=opens,
            calendar_digest=calendar_digest,
            price_digest=price_digest,
            corporate_record=role_map["corporate_actions"],
            delisting_record=role_map["delistings"],
        )
        gates.add(
            "matched_same_sic_controls_complete",
            True,
            f"{len(matches)} controls for {len(selected)} events",
        )
        control_legs = _control_legs(selected, matches, master)
        event_legs = selected.copy()
        event_legs["selected_after_overlap"] = True
        event_legs["research_scope"] = RESEARCH_CANDIDATE
        all_legs = pd.concat(
            [
                event_legs[["event_id", "security_id", "entry_at"]],
                control_legs[["event_id", "security_id", "entry_at"]],
            ],
            ignore_index=True,
        )
        horizons = [int(value) for value in spec["horizons"]]
        _ensure_delisting_payoffs(
            all_legs,
            payoffs=payoffs,
            delistings=delistings,
            opens=opens,
            horizons=horizons,
        )
        event_payoffs = _event_specific_payoffs(event_legs, payoffs)
        control_payoffs = _event_specific_payoffs(control_legs, payoffs)
        event_return_rows = _compute_leg_returns(
            event_legs,
            prices=prices,
            opens=opens,
            calendar_digest=calendar_digest,
            horizons=horizons,
            benchmark_ticker=str(spec["benchmark_ticker"]),
            one_way_cost_bps=float(cost_model["event_one_way_bps"]),
            price_basis=str(spec["price_basis"]),
            price_digest=price_digest,
            corporate_record=role_map["corporate_actions"],
            delisting_record=role_map["delistings"],
            event_payoffs=event_payoffs,
        )
        control_return_rows = _compute_leg_returns(
            control_legs,
            prices=prices,
            opens=opens,
            calendar_digest=calendar_digest,
            horizons=horizons,
            benchmark_ticker=str(spec["benchmark_ticker"]),
            one_way_cost_bps=float(cost_model["control_one_way_bps"]),
            price_basis=str(spec["price_basis"]),
            price_digest=price_digest,
            corporate_record=role_map["corporate_actions"],
            delisting_record=role_map["delistings"],
            event_payoffs=control_payoffs,
        )
        if not event_return_rows["return_status"].isin({"ok", "ok_delisting_payoff"}).all():
            raise EvidenceGateError("one or more event horizons lacks an exact executable return")
        if not control_return_rows["return_status"].isin({"ok", "ok_delisting_payoff"}).all():
            raise EvidenceGateError("one or more control horizons lacks an exact executable return")
        gates.add(
            "exact_entry_exit_opens_verified",
            True,
            f"{len(event_return_rows)} event and {len(control_return_rows)} control horizon rows",
        )
        placebo = compute_pre_event_placebo(
            event_legs,
            prices,
            exchange_sessions=opens,
            calendar_authoritative=True,
            horizons=horizons,
            gap_sessions=1,
            benchmark_ticker=str(spec["benchmark_ticker"]),
            price_basis=str(spec["price_basis"]),
            corporate_action_complete=True,
        )
        if not placebo["placebo_status"].eq("ok").all():
            raise EvidenceGateError("pre-event placebo lacks exact complete opens")
        gates.add(
            "pre_event_placebo_complete",
            True,
            f"{len(placebo)} exact non-overlapping pre-event horizon rows",
        )
        evidence_class = (
            "SYNTHETIC_FIXTURE_ONLY"
            if manifest_payload["source_class"] == SYNTHETIC_SOURCE
            else "HISTORICAL_RESEARCH_DIAGNOSTIC"
        )
        ledger, return_summary = _build_two_leg_ledger(
            event_return_rows,
            control_return_rows,
            control_legs,
            cost_model=cost_model,
            evidence_class=evidence_class,
        )
        gates.add(
            "two_leg_ledger_reconciled",
            True,
            f"{len(ledger)} legs; {len(return_summary)}/{len(return_summary)} summaries reconciled",
        )
        gates.add(
            "pipeline_evidence_complete",
            True,
            "event, controls, 5/20/60 returns, placebo, and two-leg ledger completed",
        )
        is_real = manifest_payload["source_class"] == HISTORICAL_SOURCE
        gates.add(
            "historical_real_data_source",
            is_real,
            str(manifest_payload["source_class"]),
            required_for_diagnostic=False,
            required_for_research_candidate=True,
        )
        gates.add(
            "statistical_alpha_validation",
            False,
            "multi-year inference, nulls, multiple testing, capacity, and holdout remain pending",
            required_for_diagnostic=False,
            required_for_research_candidate=False,
        )

        selected["runner_evidence_class"] = evidence_class
        selected["runner_research_status"] = RESEARCH_STATUS
        selected["performance_claim_permitted"] = False
        matches["runner_evidence_class"] = evidence_class
        matches["runner_research_status"] = RESEARCH_STATUS
        matches["performance_claim_permitted"] = False
        placebo["evidence_class"] = evidence_class
        placebo["research_status"] = RESEARCH_STATUS
        placebo["performance_claim_permitted"] = False
        overlapped.to_parquet(target / "event_clusters_diagnostic.parquet", index=False)
        selected.to_parquet(target / "purchase_events_diagnostic.parquet", index=False)
        matches.to_parquet(target / "matched_controls_diagnostic.parquet", index=False)
        return_summary.to_csv(target / "event_returns_diagnostic.csv", index=False)
        ledger.to_csv(target / "trade_ledger_diagnostic.csv", index=False)
        placebo.to_csv(target / "pre_event_placebo_diagnostic.csv", index=False)
        event_return_rows.to_csv(target / "event_leg_returns_audit.csv", index=False)
        control_return_rows.to_csv(target / "control_leg_returns_audit.csv", index=False)
        exclusions = overlapped.loc[
            ~(
                overlapped["eligible"].eq(True)
                & overlapped["selected_after_overlap"].eq(True)
            )
        ].copy()
        exclusions.to_csv(target / "event_exclusions.csv", index=False)
        research_scope = "RESEARCH_CANDIDATE" if is_real else "SYNTHETIC_DIAGNOSTIC_ONLY"
        return _finalize_run(
            output_dir=target,
            input_manifest_path=manifest_path,
            input_manifest_sha256=manifest_sha256,
            manifest_payload=manifest_payload,
            records=records,
            gates=gates,
            research_scope=research_scope,
            diagnostic_returns_emitted=True,
            blocked_reason=None,
            counts={
                "input_events": len(events),
                "selected_events": len(selected),
                "matched_controls": len(matches),
                "event_horizon_rows": len(return_summary),
                "placebo_rows": len(placebo),
                "ledger_rows": len(ledger),
            },
        )
    except ManifestIntegrityError:
        raise
    except (
        EvidenceGateError,
        AcceptanceClockError,
        InsiderDataSchemaError,
        Form4EventStudyError,
        PointInTimeViolation,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        if not any(row["gate"] == "pipeline_evidence_complete" for row in gates.rows):
            gates.add("pipeline_evidence_complete", False, str(exc))
        return _finalize_run(
            output_dir=target,
            input_manifest_path=manifest_path,
            input_manifest_sha256=manifest_sha256,
            manifest_payload=manifest_payload,
            records=records,
            gates=gates,
            research_scope="DIAGNOSTIC_ONLY_BLOCKED",
            diagnostic_returns_emitted=False,
            blocked_reason=str(exc),
        )


__all__ = [
    "ARTIFACT_SCHEMA",
    "HISTORICAL_SOURCE",
    "INPUT_SCHEMA",
    "ManifestIntegrityError",
    "RESEARCH_STATUS",
    "SYNTHETIC_SOURCE",
    "EvidenceGateError",
    "Form4EventRunnerError",
    "load_verified_input_bundle",
    "run_form4_event_study",
    "verify_artifact_manifest",
]
