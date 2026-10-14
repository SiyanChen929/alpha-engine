#!/usr/bin/env python3
"""Freeze the SEC filing-change design before document or outcome access."""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

PLAN_MANIFEST_SCHEMA = "sec_filing_change_acquisition_plan_manifest_v1"
DESIGN_RECEIPT_SCHEMA = "sec_filing_change_design_receipt_v2"
CALENDAR_RECEIPT_SCHEMA = "xnys_calendar_snapshot_receipt_v1"
CALENDAR_RECEIPT_ROLES = frozenset(
    {"calendar_snapshot_generator", "xnys_regular_sessions"}
)
PLAN_INPUT_ROLES = frozenset(
    {
        "human_protocol",
        "frozen_proxy_universe",
        "sec_header_raw_manifest",
        "planner",
        "sec_periodic_data_code",
        "pair_plan_code",
    }
)
PLAN_OUTPUT_ROLES = frozenset(
    {
        "mapped_header_inventory",
        "rough_pairing_ledger",
        "acquisition_documents",
        "header_pair_exclusions",
        "mapping_misses",
        "plan_summary",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path, role: str) -> dict[str, object]:
    if path.is_symlink():
        raise ValueError(f"receipt source may not be a symlink: {path}")
    resolved = path.resolve()
    resolved.relative_to(ROOT.resolve())
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "role": role,
        "path": resolved.relative_to(ROOT.resolve()).as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _validated_record_map(
    records: object,
    *,
    expected_roles: frozenset[str],
    section: str,
) -> dict[str, Mapping[str, object]]:
    if not isinstance(records, list) or not records:
        raise ValueError(f"preoutcome plan manifest has empty {section}")
    if any(not isinstance(record, dict) for record in records):
        raise ValueError(f"preoutcome plan manifest has invalid {section} records")
    roles = [record.get("role") for record in records]
    if any(not isinstance(role, str) or not role for role in roles):
        raise ValueError(f"preoutcome plan manifest has invalid {section} roles")
    if len(roles) != len(set(roles)):
        raise ValueError(f"preoutcome plan manifest has duplicate {section} roles")
    actual_roles = set(roles)
    if actual_roles != expected_roles:
        missing = sorted(expected_roles - actual_roles)
        unexpected = sorted(actual_roles - expected_roles)
        raise ValueError(
            f"preoutcome plan manifest {section} roles differ from the contract; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return {str(record["role"]): record for record in records}


def _verify_plan_manifest(
    path: Path,
) -> tuple[dict[str, Mapping[str, object]], dict[str, Mapping[str, object]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("preoutcome plan manifest must be a JSON object")
    if payload.get("schema_version") != PLAN_MANIFEST_SCHEMA:
        raise ValueError("unsupported preoutcome plan manifest")
    expected_top_level = {
        "schema_version",
        "created_at",
        "inputs",
        "outputs",
        "outcome_join_performed",
    }
    if set(payload) != expected_top_level:
        raise ValueError(
            "preoutcome plan manifest top-level fields differ from the contract"
        )
    created_at = payload.get("created_at")
    if not isinstance(created_at, str) or not created_at.strip():
        raise ValueError("preoutcome plan manifest has no creation timestamp")
    try:
        created_timestamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            "preoutcome plan manifest creation timestamp is invalid"
        ) from exc
    if created_timestamp.tzinfo is None:
        raise ValueError("preoutcome plan manifest creation timestamp is invalid")
    if payload.get("outcome_join_performed") is not False:
        raise ValueError("preoutcome plan manifest records an outcome join")
    inputs = _validated_record_map(
        payload.get("inputs"), expected_roles=PLAN_INPUT_ROLES, section="input"
    )
    outputs = _validated_record_map(
        payload.get("outputs"), expected_roles=PLAN_OUTPUT_ROLES, section="output"
    )
    combined = {**inputs, **outputs}
    manifest_paths: list[Path] = []
    for role, record in combined.items():
        raw_path = record.get("path")
        if (
            not isinstance(raw_path, str)
            or not raw_path
            or Path(raw_path).is_absolute()
        ):
            raise ValueError(f"plan manifest has invalid path: {role}")
        candidate = ROOT / raw_path
        if candidate.is_symlink():
            raise ValueError(f"plan manifest source may not be a symlink: {role}")
        source = candidate.resolve()
        source.relative_to(ROOT.resolve())
        if not source.is_file():
            raise FileNotFoundError(source)
        expected_bytes = record.get("bytes")
        expected_hash = record.get("sha256")
        if (
            not isinstance(expected_bytes, int)
            or isinstance(expected_bytes, bool)
            or expected_bytes < 0
            or not isinstance(expected_hash, str)
            or len(expected_hash) != 64
        ):
            raise ValueError(f"plan manifest has invalid file metadata: {role}")
        if (
            source.stat().st_size != expected_bytes
            or sha256_file(source) != expected_hash
        ):
            raise ValueError(f"plan manifest hash mismatch: {role}")
        manifest_paths.append(source)
    if len(manifest_paths) != len(set(manifest_paths)):
        raise ValueError("preoutcome plan manifest assigns multiple roles to one path")
    return inputs, outputs


def _require_plan_binding(
    path: Path,
    record: Mapping[str, object],
    *,
    argument: str,
) -> None:
    actual = path.resolve()
    expected = (ROOT / str(record["path"])).resolve()
    if actual != expected:
        raise ValueError(f"{argument} differs from its preoutcome plan manifest role")
    if (
        actual.stat().st_size != record["bytes"]
        or sha256_file(actual) != record["sha256"]
    ):
        raise ValueError(f"{argument} differs from its preoutcome plan manifest hash")


def _verify_plan_bindings(
    args: argparse.Namespace,
    plan_inputs: Mapping[str, Mapping[str, object]],
    plan_outputs: Mapping[str, Mapping[str, object]],
) -> None:
    bindings = (
        (args.protocol, plan_inputs["human_protocol"], "protocol"),
        (args.universe, plan_inputs["frozen_proxy_universe"], "universe"),
        (
            args.header_manifest,
            plan_inputs["sec_header_raw_manifest"],
            "header manifest",
        ),
        (
            ROOT / "scripts/build_sec_filing_change_acquisition_plan.py",
            plan_inputs["planner"],
            "plan builder",
        ),
        (
            SRC / "alpha_factor_engine/data/sec_periodic_filings.py",
            plan_inputs["sec_periodic_data_code"],
            "SEC periodic data code",
        ),
        (
            SRC / "alpha_factor_engine/research/sec_filing_change_plan.py",
            plan_inputs["pair_plan_code"],
            "preoutcome plan code",
        ),
        (
            args.acquisition_documents,
            plan_outputs["acquisition_documents"],
            "acquisition documents",
        ),
        (args.rough_pairs, plan_outputs["rough_pairing_ledger"], "rough pairs"),
    )
    for bound_path, manifest_record, argument in bindings:
        _require_plan_binding(bound_path, manifest_record, argument=argument)


def _verify_protocol_paths(
    args: argparse.Namespace, protocol: Mapping[str, object]
) -> None:
    sources = protocol.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("protocol has no source contract")
    universe = sources.get("universe")
    prices = sources.get("daily_prices")
    exchange_calendar = sources.get("exchange_calendar")
    if (
        not isinstance(universe, dict)
        or not isinstance(prices, dict)
        or not isinstance(exchange_calendar, dict)
        or exchange_calendar.get("calendar") != "XNYS"
        or exchange_calendar.get("source_status")
        != "independent_versioned_community_ruleset_not_exchange_licensed_data"
    ):
        raise ValueError("protocol source contract is incomplete")
    configured = (
        (args.universe, universe.get("path"), "universe"),
        (args.header_manifest, sources.get("sec_header_manifest"), "header manifest"),
        (args.prices, prices.get("path"), "prices"),
        (
            args.xnys_regular_sessions,
            exchange_calendar.get("path"),
            "XNYS regular sessions",
        ),
        (
            args.xnys_calendar_receipt,
            exchange_calendar.get("receipt"),
            "XNYS calendar receipt",
        ),
        (
            args.calendar_snapshot_generator,
            exchange_calendar.get("generator"),
            "XNYS calendar generator",
        ),
    )
    for supplied, raw_expected, label in configured:
        if (
            not isinstance(raw_expected, str)
            or not raw_expected
            or Path(raw_expected).is_absolute()
        ):
            raise ValueError(f"protocol has an invalid canonical {label} path")
        expected = (ROOT / raw_expected).resolve()
        expected.relative_to(ROOT.resolve())
        if supplied.resolve() != expected:
            raise ValueError(f"{label} path differs from the protocol source contract")


def _verify_calendar_receipt(args: argparse.Namespace) -> None:
    payload = json.loads(args.xnys_calendar_receipt.read_text(encoding="utf-8"))
    expected_top_level = {
        "schema_version",
        "created_at",
        "calendar",
        "calendar_semantics",
        "first_session",
        "last_session",
        "sessions",
        "exchange_calendars_version",
        "tzdata_version",
        "source_status",
        "crosscheck_source",
        "package_source",
        "records",
    }
    if not isinstance(payload, dict) or set(payload) != expected_top_level:
        raise ValueError("XNYS calendar receipt top-level fields differ from contract")
    if (
        payload.get("schema_version") != CALENDAR_RECEIPT_SCHEMA
        or payload.get("calendar") != "XNYS"
        or payload.get("calendar_semantics")
        != "regular exchange sessions and scheduled open/close timestamps"
        or payload.get("source_status")
        != "INDEPENDENT_VERSIONED_COMMUNITY_RULESET_NOT_EXCHANGE_LICENSED_DATA"
    ):
        raise ValueError("XNYS calendar receipt semantics differ from contract")
    created_at = payload.get("created_at")
    try:
        created_timestamp = datetime.fromisoformat(
            str(created_at).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ValueError("XNYS calendar receipt timestamp is invalid") from exc
    if not isinstance(created_at, str) or created_timestamp.tzinfo is None:
        raise ValueError("XNYS calendar receipt timestamp is invalid")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != len(CALENDAR_RECEIPT_ROLES):
        raise ValueError("XNYS calendar receipt has incomplete records")
    roles = [record.get("role") for record in records if isinstance(record, dict)]
    if (
        len(roles) != len(records)
        or set(roles) != CALENDAR_RECEIPT_ROLES
        or len(roles) != len(set(roles))
    ):
        raise ValueError("XNYS calendar receipt roles differ from contract")
    expected_paths = {
        "calendar_snapshot_generator": args.calendar_snapshot_generator.resolve(),
        "xnys_regular_sessions": args.xnys_regular_sessions.resolve(),
    }
    for record in records:
        if set(record) != {"role", "path", "bytes", "sha256"}:
            raise ValueError("XNYS calendar receipt record is malformed")
        raw_path = record.get("path")
        expected_bytes = record.get("bytes")
        expected_hash = record.get("sha256")
        if (
            not isinstance(raw_path, str)
            or not raw_path
            or Path(raw_path).is_absolute()
            or not isinstance(expected_bytes, int)
            or isinstance(expected_bytes, bool)
            or expected_bytes < 1
            or not isinstance(expected_hash, str)
            or len(expected_hash) != 64
        ):
            raise ValueError("XNYS calendar receipt record is malformed")
        candidate = ROOT / raw_path
        if candidate.is_symlink():
            raise ValueError("XNYS calendar receipt source may not be a symlink")
        source = candidate.resolve()
        source.relative_to(ROOT.resolve())
        if source != expected_paths[str(record["role"])]:
            raise ValueError(
                f"XNYS calendar receipt role has a substituted path: {record['role']}"
            )
        if (
            not source.is_file()
            or source.stat().st_size != expected_bytes
            or sha256_file(source) != expected_hash
        ):
            raise ValueError(
                f"XNYS calendar receipt hash mismatch: {record['role']}"
            )

    with args.xnys_regular_sessions.open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != [
            "session_date",
            "market_open_utc",
            "market_close_utc",
        ]:
            raise ValueError("XNYS calendar snapshot columns differ from contract")
        rows = list(reader)
    sessions = payload.get("sessions")
    if (
        not isinstance(sessions, int)
        or isinstance(sessions, bool)
        or sessions < 1
        or len(rows) != sessions
    ):
        raise ValueError("XNYS calendar snapshot row count differs from receipt")
    dates: list[str] = []
    for row in rows:
        session_date = row.get("session_date")
        market_open = row.get("market_open_utc")
        market_close = row.get("market_close_utc")
        try:
            parsed_date = datetime.strptime(str(session_date), "%Y-%m-%d").date()
            parsed_open = datetime.fromisoformat(
                str(market_open).replace("Z", "+00:00")
            )
            parsed_close = datetime.fromisoformat(
                str(market_close).replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise ValueError("XNYS calendar snapshot has invalid timestamps") from exc
        if (
            not str(market_open).endswith("Z")
            or not str(market_close).endswith("Z")
            or parsed_open.utcoffset() != UTC.utcoffset(parsed_open)
            or parsed_close.utcoffset() != UTC.utcoffset(parsed_close)
            or parsed_open.date() != parsed_date
            or parsed_close.date() != parsed_date
            or parsed_open >= parsed_close
        ):
            raise ValueError("XNYS calendar snapshot has invalid session clocks")
        dates.append(str(session_date))
    if (
        dates != sorted(set(dates))
        or payload.get("first_session") != dates[0]
        or payload.get("last_session") != dates[-1]
    ):
        raise ValueError("XNYS calendar snapshot session range differs from receipt")


def _verify_protocol_contract(protocol: object) -> dict[str, object]:
    if (
        not isinstance(protocol, dict)
        or protocol.get("schema_version") != "sec_filing_change_stage_a_protocol_v1"
    ):
        raise ValueError("unsupported filing-change protocol")
    experiment = protocol.get("experiment")
    if (
        not isinstance(experiment, dict)
        or experiment.get("name") != "sec_filing_change_stage_a_v1"
    ):
        raise ValueError("protocol experiment differs from the frozen experiment")
    if experiment.get("selection_or_tuning_after_outcome") is not False:
        raise ValueError("protocol does not forbid post-outcome tuning")
    if experiment.get("historical_alpha_verdict") != "INVALID":
        raise ValueError("protocol does not lock the historical INVALID verdict")
    if experiment.get("deployment_verdict") != "NOT AUTHORIZED":
        raise ValueError("protocol does not prohibit deployment")
    signals = protocol.get("signals")
    inference = protocol.get("inference")
    document = protocol.get("document")
    gates = protocol.get("gates")
    expected_coverage = {
        "sec_raw_clock": 1.0,
        "primary_document_retrieval": 0.98,
        "prior_year_pairing": 0.80,
        "parse_success": 0.98,
        "ticker_mapping": 0.95,
        "benchmark_clock": 1.0,
    }
    if (
        not isinstance(signals, dict)
        or not isinstance(signals.get("primary"), dict)
        or not isinstance(signals.get("secondary"), dict)
        or not isinstance(signals.get("sort"), dict)
        or signals["primary"].get("name") != "jaccard_similarity"
        or signals["secondary"].get("name") != "cosine_similarity"
        or signals["sort"].get("minimum_monthly_observations") != 100
        or not isinstance(document, dict)
        or document.get("minimum_clean_tokens_each_document") != 1000
        or not isinstance(inference, dict)
        or inference.get("current_disclosed_trial_floor") != 130
        or not isinstance(gates, dict)
        or not isinstance(gates.get("coverage"), dict)
        or any(
            gates["coverage"].get(name) != value
            for name, value in expected_coverage.items()
        )
    ):
        raise ValueError(
            "protocol signal or inference contract differs from receipt v2"
        )
    return protocol


def run(args: argparse.Namespace) -> Path:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"non-overwriting design receipt exists: {output}")
    protocol = _verify_protocol_contract(
        json.loads(args.protocol.read_text(encoding="utf-8"))
    )
    _verify_protocol_paths(args, protocol)
    _verify_calendar_receipt(args)
    plan_inputs, plan_outputs = _verify_plan_manifest(args.plan_manifest)
    _verify_plan_bindings(args, plan_inputs, plan_outputs)
    paths = [
        (args.protocol, "human_protocol"),
        (args.plan_manifest, "preoutcome_plan_manifest"),
        (args.acquisition_documents, "frozen_acquisition_documents"),
        (args.rough_pairs, "header_only_rough_pairs"),
        (args.universe, "frozen_current_survivor_universe"),
        (args.header_manifest, "sec_submissions_raw_manifest"),
        (args.prices, "frozen_retrospective_daily_prices"),
        (args.xnys_regular_sessions, "xnys_regular_sessions"),
        (args.xnys_calendar_receipt, "xnys_calendar_receipt"),
        (args.calendar_snapshot_generator, "calendar_snapshot_generator"),
        (
            SRC / "alpha_factor_engine/data/sec_periodic_filings.py",
            "sec_periodic_data_code",
        ),
        (
            SRC / "alpha_factor_engine/features/filing_change.py",
            "filing_change_feature_code",
        ),
        (
            SRC / "alpha_factor_engine/research/sec_filing_change_plan.py",
            "preoutcome_plan_code",
        ),
        (
            SRC / "alpha_factor_engine/research/sec_filing_change_stage_a.py",
            "outcome_core_code",
        ),
        (ROOT / "scripts/build_sec_filing_change_acquisition_plan.py", "plan_builder"),
        (ROOT / "scripts/acquire_sec_filing_change_documents.py", "document_acquirer"),
        (ROOT / "scripts/build_sec_filing_change_scores.py", "score_builder"),
        (ROOT / "scripts/run_sec_filing_change_stage_a.py", "outcome_runner"),
        (Path(__file__), "design_receipt_builder"),
    ]
    records = [_record(path, role) for path, role in paths]
    if len({record["role"] for record in records}) != len(records):
        raise ValueError("duplicate receipt roles")
    receipt = {
        "schema_version": DESIGN_RECEIPT_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "experiment": protocol["experiment"]["name"],
        "status": "FROZEN_PRE_OUTCOME",
        "records": records,
        "primary_signal": protocol["signals"]["primary"]["name"],
        "secondary_signal": protocol["signals"]["secondary"]["name"],
        "current_disclosed_trial_floor": protocol["inference"][
            "current_disclosed_trial_floor"
        ],
        "outcome_join_performed": False,
        "sealed_2025_signal_return_joined": False,
        "historical_alpha_verdict": "INVALID",
        "deployment_verdict": "NOT AUTHORIZED",
        "non_overwrite_policy": True,
    }
    serialized = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    plan_root = ROOT / "artifacts/sec_filing_change_stage_a_v1/preoutcome_plan_v3"
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/sec_filing_change_stage_a_v1.json",
    )
    parser.add_argument(
        "--plan-manifest", type=Path, default=plan_root / "plan_manifest.json"
    )
    parser.add_argument(
        "--acquisition-documents",
        type=Path,
        default=plan_root / "acquisition_documents.parquet",
    )
    parser.add_argument(
        "--rough-pairs", type=Path, default=plan_root / "rough_pairing_ledger.parquet"
    )
    parser.add_argument(
        "--universe",
        type=Path,
        default=ROOT / "artifacts/public_broad_proxy_v1/data/frozen_proxy_universe.csv",
    )
    parser.add_argument(
        "--header-manifest",
        type=Path,
        default=ROOT / "artifacts/sec_periodic_filing_inventory_v1/raw_manifest.json",
    )
    parser.add_argument(
        "--prices",
        type=Path,
        default=ROOT / "artifacts/public_broad_proxy_v1/data/daily_prices.parquet",
    )
    parser.add_argument(
        "--xnys-regular-sessions",
        type=Path,
        default=ROOT
        / "artifacts/reference_calendars/xnys_v1/xnys_regular_sessions_2014_2028.csv",
    )
    parser.add_argument(
        "--xnys-calendar-receipt",
        type=Path,
        default=ROOT / "artifacts/reference_calendars/xnys_v1/calendar_receipt.json",
    )
    parser.add_argument(
        "--calendar-snapshot-generator",
        type=Path,
        default=ROOT / "scripts/build_xnys_calendar_snapshot.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/sec_filing_change_stage_a_v1/design_receipt_v2.json",
    )
    args = parser.parse_args()
    print(run(args))


if __name__ == "__main__":
    main()
