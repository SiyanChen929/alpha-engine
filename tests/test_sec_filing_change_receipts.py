import argparse
import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import acquire_sec_filing_change_documents as acquirer
from scripts import build_sec_filing_change_design_receipt as receipt_builder
from scripts import build_sec_filing_change_scores as scorer
from scripts import run_sec_filing_change_stage_a as runner


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _empty_design_receipt():
    return {
        "schema_version": "sec_filing_change_design_receipt_v2",
        "created_at": "2026-09-03T12:00:00+00:00",
        "experiment": "sec_filing_change_stage_a_v1",
        "status": "FROZEN_PRE_OUTCOME",
        "records": [],
        "primary_signal": "jaccard_similarity",
        "secondary_signal": "cosine_similarity",
        "current_disclosed_trial_floor": 130,
        "outcome_join_performed": False,
        "sealed_2025_signal_return_joined": False,
        "historical_alpha_verdict": "INVALID",
        "deployment_verdict": "NOT AUTHORIZED",
        "non_overwrite_policy": True,
    }


DESIGN_RECEIPT_ROLES = (
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
)


def test_acquirer_rejects_zero_record_or_unsealed_design_receipt(tmp_path):
    empty = _write(tmp_path / "empty.json", _empty_design_receipt())
    with pytest.raises(ValueError, match="incomplete"):
        acquirer._read_design_receipt(empty)


def test_scorer_rejects_zero_record_design_receipt(tmp_path):
    empty = _write(tmp_path / "empty.json", _empty_design_receipt())
    with pytest.raises(ValueError, match="incomplete"):
        scorer._verify_receipt(empty)


@pytest.mark.parametrize(
    "verifier", [acquirer._read_design_receipt, scorer._verify_receipt]
)
def test_receipt_v2_rejects_legacy_unopened_outcome_claim(tmp_path, verifier):
    payload = _empty_design_receipt()
    payload["sealed_2025_outcomes_opened"] = payload.pop(
        "sealed_2025_signal_return_joined"
    )
    path = _write(tmp_path / f"legacy-{verifier.__name__}.json", payload)
    with pytest.raises(ValueError, match="top-level fields"):
        verifier(path)


def test_runner_rejects_zero_record_manifest(tmp_path):
    empty = _write(
        tmp_path / "empty.json",
        {
            "schema_version": "sec_filing_change_score_manifest_v1",
            "inputs": [],
            "outputs": [],
        },
    )
    with pytest.raises(ValueError, match="no file records"):
        runner._verify_json(empty, "sec_filing_change_score_manifest_v1")


def _file_record(root: Path, relative_path: str, role: str) -> dict[str, object]:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"fixture for {role}\n", encoding="utf-8")
    content = path.read_bytes()
    return {
        "role": role,
        "path": relative_path,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _valid_design_receipt(root: Path) -> dict[str, object]:
    payload = _empty_design_receipt()
    code_paths = {
        "sec_periodic_data_code": "src/alpha_factor_engine/data/sec_periodic_filings.py",
        "filing_change_feature_code": "src/alpha_factor_engine/features/filing_change.py",
        "preoutcome_plan_code": "src/alpha_factor_engine/research/sec_filing_change_plan.py",
        "outcome_core_code": "src/alpha_factor_engine/research/sec_filing_change_stage_a.py",
        "plan_builder": "scripts/build_sec_filing_change_acquisition_plan.py",
        "document_acquirer": "scripts/acquire_sec_filing_change_documents.py",
        "score_builder": "scripts/build_sec_filing_change_scores.py",
        "outcome_runner": "scripts/run_sec_filing_change_stage_a.py",
        "design_receipt_builder": "scripts/build_sec_filing_change_design_receipt.py",
        "calendar_snapshot_generator": "scripts/build_xnys_calendar_snapshot.py",
    }
    payload["records"] = [
        _file_record(
            root,
            code_paths.get(role, f"receipt-inputs/{number:02d}.txt"),
            role,
        )
        for number, role in enumerate(DESIGN_RECEIPT_ROLES)
    ]
    return payload


@pytest.mark.parametrize(
    ("module", "verifier"),
    [
        (acquirer, acquirer._read_design_receipt),
        (scorer, scorer._verify_receipt),
    ],
)
def test_receipt_v2_requires_strict_record_metadata_and_utc_clock(
    tmp_path, monkeypatch, module, verifier
):
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "SRC", tmp_path / "src")
    payload = _valid_design_receipt(tmp_path)
    valid_path = _write(tmp_path / "valid-receipt.json", payload)
    verifier(valid_path)

    payload = _valid_design_receipt(tmp_path)
    score_builder = next(
        record for record in payload["records"] if record["role"] == "score_builder"
    )
    replacement = _file_record(
        tmp_path, "receipt-inputs/substituted-score-builder.py", "score_builder"
    )
    score_builder.update(replacement)
    substituted_path = _write(tmp_path / "substituted-code-receipt.json", payload)
    with pytest.raises(ValueError, match="code roles do not bind current files"):
        verifier(substituted_path)

    payload = _valid_design_receipt(tmp_path)
    payload["records"][0]["bytes"] = str(payload["records"][0]["bytes"])
    malformed_path = _write(tmp_path / "malformed-receipt.json", payload)
    with pytest.raises(ValueError, match="record is malformed"):
        verifier(malformed_path)

    payload = _valid_design_receipt(tmp_path)
    payload["created_at"] = "not-a-time"
    invalid_clock_path = _write(tmp_path / "invalid-clock-receipt.json", payload)
    with pytest.raises(ValueError, match="creation timestamp"):
        verifier(invalid_clock_path)


def _valid_plan_payload(root: Path) -> dict[str, object]:
    input_paths = {
        "human_protocol": "configs/protocol.json",
        "frozen_proxy_universe": "artifacts/universe.csv",
        "sec_header_raw_manifest": "artifacts/raw_manifest.json",
        "planner": "scripts/build_sec_filing_change_acquisition_plan.py",
        "sec_periodic_data_code": "src/alpha_factor_engine/data/sec_periodic_filings.py",
        "pair_plan_code": "src/alpha_factor_engine/research/sec_filing_change_plan.py",
    }
    output_paths = {
        "mapped_header_inventory": "artifacts/plan/mapped_headers.parquet",
        "rough_pairing_ledger": "artifacts/plan/rough_pairs.parquet",
        "acquisition_documents": "artifacts/plan/documents.parquet",
        "header_pair_exclusions": "artifacts/plan/exclusions.csv",
        "mapping_misses": "artifacts/plan/mapping_misses.csv",
        "plan_summary": "artifacts/plan/summary.json",
    }
    return {
        "schema_version": receipt_builder.PLAN_MANIFEST_SCHEMA,
        "created_at": "2026-09-03T12:00:00+00:00",
        "inputs": [
            _file_record(root, path, role) for role, path in input_paths.items()
        ],
        "outputs": [
            _file_record(root, path, role) for role, path in output_paths.items()
        ],
        "outcome_join_performed": False,
    }


def test_design_builder_rejects_empty_duplicate_and_unexpected_plan_roles(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(receipt_builder, "ROOT", tmp_path)
    monkeypatch.setattr(receipt_builder, "SRC", tmp_path / "src")
    valid = _valid_plan_payload(tmp_path)

    empty = dict(valid, inputs=[])
    empty_path = _write(tmp_path / "empty-plan.json", empty)
    with pytest.raises(ValueError, match="empty input"):
        receipt_builder._verify_plan_manifest(empty_path)

    duplicate = dict(valid, inputs=[*valid["inputs"], dict(valid["inputs"][0])])
    duplicate_path = _write(tmp_path / "duplicate-plan.json", duplicate)
    with pytest.raises(ValueError, match="duplicate input roles"):
        receipt_builder._verify_plan_manifest(duplicate_path)

    unexpected_inputs = [dict(record) for record in valid["inputs"]]
    unexpected_inputs[-1]["role"] = "outcome_prices"
    unexpected = dict(valid, inputs=unexpected_inputs)
    unexpected_path = _write(tmp_path / "unexpected-plan.json", unexpected)
    with pytest.raises(ValueError, match="unexpected=.*outcome_prices"):
        receipt_builder._verify_plan_manifest(unexpected_path)


def test_design_builder_rejects_invalid_plan_top_level(tmp_path, monkeypatch):
    monkeypatch.setattr(receipt_builder, "ROOT", tmp_path)
    valid = _valid_plan_payload(tmp_path)

    contaminated = dict(valid, outcome_join_performed=True)
    contaminated_path = _write(tmp_path / "contaminated-plan.json", contaminated)
    with pytest.raises(ValueError, match="outcome join"):
        receipt_builder._verify_plan_manifest(contaminated_path)

    unexpected = dict(valid, unreviewed_override=True)
    unexpected_path = _write(tmp_path / "extra-top-level.json", unexpected)
    with pytest.raises(ValueError, match="top-level fields"):
        receipt_builder._verify_plan_manifest(unexpected_path)


@pytest.mark.parametrize(
    "argument",
    ["protocol", "universe", "header_manifest", "acquisition_documents", "rough_pairs"],
)
def test_design_builder_binds_runtime_inputs_to_exact_plan_roles(
    tmp_path, monkeypatch, argument
):
    monkeypatch.setattr(receipt_builder, "ROOT", tmp_path)
    monkeypatch.setattr(receipt_builder, "SRC", tmp_path / "src")
    payload = _valid_plan_payload(tmp_path)
    manifest_path = _write(tmp_path / "plan-manifest.json", payload)
    plan_inputs, plan_outputs = receipt_builder._verify_plan_manifest(manifest_path)
    args = argparse.Namespace(
        protocol=tmp_path / str(plan_inputs["human_protocol"]["path"]),
        universe=tmp_path / str(plan_inputs["frozen_proxy_universe"]["path"]),
        header_manifest=tmp_path / str(plan_inputs["sec_header_raw_manifest"]["path"]),
        acquisition_documents=tmp_path
        / str(plan_outputs["acquisition_documents"]["path"]),
        rough_pairs=tmp_path / str(plan_outputs["rough_pairing_ledger"]["path"]),
    )
    receipt_builder._verify_plan_bindings(args, plan_inputs, plan_outputs)

    replacement = tmp_path / f"replacement-{argument}"
    replacement.write_text("fixture replacement\n", encoding="utf-8")
    setattr(args, argument, replacement)
    with pytest.raises(
        ValueError, match="differs from its preoutcome plan manifest role"
    ):
        receipt_builder._verify_plan_bindings(args, plan_inputs, plan_outputs)


@pytest.mark.parametrize("argument", ["universe", "header_manifest", "prices"])
def test_design_builder_binds_sources_to_protocol_canonical_paths(
    tmp_path, monkeypatch, argument
):
    monkeypatch.setattr(receipt_builder, "ROOT", tmp_path)
    protocol = {
        "sources": {
            "universe": {"path": "artifacts/universe.csv"},
            "sec_header_manifest": "artifacts/raw_manifest.json",
            "daily_prices": {"path": "artifacts/prices.parquet"},
            "exchange_calendar": {
                "path": "artifacts/xnys.csv",
                "receipt": "artifacts/xnys-receipt.json",
                "generator": "scripts/build-xnys.py",
                "calendar": "XNYS",
                "source_status": "independent_versioned_community_ruleset_not_exchange_licensed_data",
            },
        }
    }
    args = argparse.Namespace(
        universe=tmp_path / "artifacts/universe.csv",
        header_manifest=tmp_path / "artifacts/raw_manifest.json",
        prices=tmp_path / "artifacts/prices.parquet",
        xnys_regular_sessions=tmp_path / "artifacts/xnys.csv",
        xnys_calendar_receipt=tmp_path / "artifacts/xnys-receipt.json",
        calendar_snapshot_generator=tmp_path / "scripts/build-xnys.py",
    )
    receipt_builder._verify_protocol_paths(args, protocol)

    setattr(args, argument, tmp_path / f"replacement-{argument}")
    with pytest.raises(ValueError, match="differs from the protocol source contract"):
        receipt_builder._verify_protocol_paths(args, protocol)


@pytest.mark.parametrize(
    ("keys", "replacement"),
    [
        (("signals", "primary", "name"), "substituted_primary"),
        (("signals", "secondary", "name"), "substituted_secondary"),
        (("signals", "sort", "minimum_monthly_observations"), 99),
        (("document", "minimum_clean_tokens_each_document"), 999),
        (("inference", "current_disclosed_trial_floor"), 129),
        (("gates", "coverage", "parse_success"), 0.50),
    ],
)
def test_design_builder_locks_protocol_values_used_downstream(keys, replacement):
    protocol_path = receipt_builder.ROOT / "configs/sec_filing_change_stage_a_v1.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    receipt_builder._verify_protocol_contract(protocol)

    target = protocol
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = replacement
    with pytest.raises(ValueError, match="signal or inference contract"):
        receipt_builder._verify_protocol_contract(protocol)


def _valid_acquisition_documents() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "issuer_cik": ["0000000001"],
            "accession": ["0000000001-24-000001"],
            "form": ["10-K"],
            "ticker": ["AAA"],
            "acceptance_at": [pd.Timestamp("2024-02-20T20:00:00Z")],
            "acceptance_at_raw": ["2024-02-20T20:00:00.000Z"],
            "primary_document": ["annual.htm"],
            "primary_document_url": [
                "https://www.sec.gov/Archives/edgar/data/1/"
                "000000000124000001/annual.htm"
            ],
            "proxy_universe_rank": [1],
            "download_phase": ["00_top25_engineering_smoke"],
            "outcome_join_permitted_during_acquisition": [False],
        }
    )


@pytest.mark.parametrize(
    "column", ["form", "ticker", "acceptance_at", "proxy_universe_rank"]
)
def test_acquirer_schema_requires_columns_used_after_preflight(column):
    documents = _valid_acquisition_documents().drop(columns=column)
    with pytest.raises(ValueError, match=column):
        acquirer._validate_acquisition_documents(documents)


def test_acquirer_rejects_unknown_document_phase():
    documents = _valid_acquisition_documents()
    documents.loc[0, "download_phase"] = "99_unreviewed_outcome_phase"
    with pytest.raises(ValueError, match="unknown download phases"):
        acquirer._validate_acquisition_documents(documents)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("outcome_column", "unexpected columns"),
        ("cik", "CIK or accession"),
        ("accession", "CIK or accession"),
        ("form", "filing forms"),
        ("url", "SEC primary URL"),
        ("rank", "proxy universe ranks"),
        ("time", "timezone-aware UTC"),
        ("raw_time", "acceptance timestamps disagree"),
    ],
)
def test_acquirer_rejects_outcome_columns_and_malformed_plan_values(case, message):
    documents = _valid_acquisition_documents()
    if case == "outcome_column":
        documents["forward_return_63"] = 0.5
    elif case == "cik":
        documents.loc[0, "issuer_cik"] = "1"
    elif case == "accession":
        documents.loc[0, "accession"] = "../../escape"
    elif case == "form":
        documents.loc[0, "form"] = "8-K"
    elif case == "url":
        documents.loc[0, "primary_document_url"] = "https://example.com/annual.htm"
    elif case == "rank":
        documents.loc[0, "proxy_universe_rank"] = 0
    elif case == "time":
        documents["acceptance_at"] = [pd.Timestamp("2024-02-20T20:00:00")]
    elif case == "raw_time":
        documents.loc[0, "acceptance_at_raw"] = "2024-02-20T20:00:01.000Z"
    with pytest.raises(ValueError, match=message):
        acquirer._validate_acquisition_documents(documents)


def test_acquirer_rejects_numeric_outcome_permission_and_duplicate_accession():
    numeric_permission = _valid_acquisition_documents()
    numeric_permission["outcome_join_permitted_during_acquisition"] = 0
    with pytest.raises(ValueError, match="permits an outcome join"):
        acquirer._validate_acquisition_documents(numeric_permission)

    duplicates = pd.concat(
        [_valid_acquisition_documents(), _valid_acquisition_documents()],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="duplicate accessions"):
        acquirer._validate_acquisition_documents(duplicates)


def test_limited_acquisition_is_never_complete():
    assert (
        acquirer._acquisition_status(failure_count=0, selection_limited=True)
        == "PARTIAL_LIMITED_ENGINEERING_SAMPLE"
    )
    assert (
        acquirer._acquisition_status(failure_count=1, selection_limited=True)
        == "INCOMPLETE_RETRY_REQUIRED"
    )


def _dataset_record(*, dataset, source_url, sha256, notes, finality):
    return {
        "provider": "SEC EDGAR Archives",
        "dataset": dataset,
        "retrieved_at": "2026-09-03T12:00:00+00:00",
        "source_url": source_url,
        "sha256": sha256,
        "rows": 1,
        "min_effective_at": None,
        "max_effective_at": None,
        "min_available_at": None,
        "max_available_at": None,
        "finality": finality,
        "revision_policy": "immutable content-addressed gzip; conflicts fail closed",
        "notes": notes,
    }


def _valid_acquisition_bundle(root: Path):
    cik = "0000000001"
    accession = "0000000001-24-000001"
    accepted = "2024-02-20T20:00:00.000Z"
    primary_url = (
        "https://www.sec.gov/Archives/edgar/data/1/" "000000000124000001/annual.htm"
    )
    header_url = (
        "https://www.sec.gov/Archives/edgar/data/1/"
        "000000000124000001/0000000001-24-000001.hdr.sgml"
    )
    primary_raw = b"<html><body>filing</body></html>"
    header_raw = b"<ACCEPTANCE-DATETIME>20240220150000"
    primary_hash = hashlib.sha256(primary_raw).hexdigest()
    header_hash = hashlib.sha256(header_raw).hexdigest()
    primary_path = root / "cache/primary.json.gz"
    header_path = root / "cache/header.json.gz"
    primary_path.parent.mkdir(parents=True, exist_ok=True)
    primary_path.write_bytes(gzip.compress(primary_raw, mtime=0))
    header_path.write_bytes(gzip.compress(header_raw, mtime=0))
    audit = pd.DataFrame(
        {
            "issuer_cik": [cik],
            "accession": [accession],
            "form": ["10-K"],
            "ticker": ["AAA"],
            "download_phase": ["00_top25_engineering_smoke"],
            "api_acceptance_at_raw": [accepted],
            "raw_header_acceptance_at": [pd.Timestamp(accepted)],
            "api_raw_delta_seconds": [0.0],
            "primary_document": ["annual.htm"],
            "primary_url": [primary_url],
            "primary_sha256": [primary_hash],
            "primary_raw_bytes": [len(primary_raw)],
            "primary_cache_path": [primary_path.resolve().as_posix()],
            "primary_cache_status": ["downloaded"],
            "header_sha256": [header_hash],
            "header_cache_path": [header_path.resolve().as_posix()],
            "header_cache_status": ["downloaded"],
            "outcome_join_performed": [False],
        }
    )
    frozen = _valid_acquisition_documents()
    dataset_records = [
        _dataset_record(
            dataset=f"acceptance_header:{accession}",
            source_url=header_url,
            sha256=header_hash,
            finality="archived_raw_sgml_header",
            notes={
                "issuer_cik": cik,
                "accession": accession,
                "api_acceptance_at": pd.Timestamp(accepted).isoformat(),
                "raw_header_acceptance_at": pd.Timestamp(accepted).isoformat(),
                "api_raw_delta_seconds": 0.0,
                "cache_path": header_path.resolve().as_posix(),
            },
        ),
        _dataset_record(
            dataset=f"periodic_primary_document:{accession}",
            source_url=primary_url,
            sha256=primary_hash,
            finality="archived_as_filed_primary_document",
            notes={
                "issuer_cik": cik,
                "accession": accession,
                "primary_document": "annual.htm",
                "acceptance_at": pd.Timestamp(accepted).isoformat(),
                "cache_path": primary_path.resolve().as_posix(),
                "raw_bytes": len(primary_raw),
            },
        ),
    ]
    summary = {
        "schema_version": "sec_filing_change_document_acquisition_summary_v1",
        "created_at": "2026-09-03T12:00:00+00:00",
        "through_phase": "00_top25_engineering_smoke",
        "status": "COMPLETE",
        "eligible_documents_through_phase": 1,
        "selection_limited": False,
        "requested_documents": 1,
        "acquired_documents": 1,
        "failed_documents": 0,
        "acquisition_coverage": 1.0,
        "unique_accessions": 1,
        "unique_issuers": 1,
        "raw_clock_exact_match_rate": 1.0,
        "primary_raw_bytes": len(primary_raw),
        "outcome_join_performed": False,
        "historical_alpha_verdict": "INVALID",
        "deployment_verdict": "NOT AUTHORIZED",
        "design_receipt_sha256": "a" * 64,
    }
    return {
        "summary": summary,
        "manifest_created_at": summary["created_at"],
        "audit": audit,
        "failures": pd.DataFrame(),
        "dataset_records": dataset_records,
        "frozen_documents": frozen,
        "design_receipt_sha256": "a" * 64,
    }


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("limited", "incomplete or outcome-contaminated"),
        ("summary_count", "summary counts"),
        ("failure", "failure ledger"),
        ("accession", "frozen phase set"),
        ("metadata", "metadata differs"),
        ("clock", "acceptance-clock"),
        ("dataset", "dataset records do not cover"),
    ],
)
def test_scorer_reconciles_entire_acquisition_bundle(
    tmp_path, monkeypatch, case, message
):
    monkeypatch.setattr(scorer, "ROOT", tmp_path)
    bundle = _valid_acquisition_bundle(tmp_path)
    if case == "limited":
        bundle["summary"]["status"] = "PARTIAL_LIMITED_ENGINEERING_SAMPLE"
        bundle["summary"]["selection_limited"] = True
    elif case == "summary_count":
        bundle["summary"]["acquired_documents"] = 2
    elif case == "failure":
        bundle["failures"] = pd.DataFrame({"accession": ["failed"]})
    elif case == "accession":
        bundle["audit"].loc[0, "accession"] = "0000000001-24-999999"
    elif case == "metadata":
        bundle["audit"].loc[0, "ticker"] = "SWAP"
    elif case == "clock":
        bundle["audit"].loc[0, "api_raw_delta_seconds"] = 2.0
    elif case == "dataset":
        bundle["dataset_records"].pop()
    with pytest.raises(ValueError, match=message):
        scorer._reconcile_acquisition_data(**bundle)


def test_scorer_accepts_a_fully_reconciled_acquisition_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(scorer, "ROOT", tmp_path)
    scorer._reconcile_acquisition_data(**_valid_acquisition_bundle(tmp_path))


def _valid_acquisition_manifest(root: Path) -> dict[str, object]:
    return {
        "schema_version": "sec_filing_change_document_acquisition_manifest_v1",
        "created_at": "2026-09-03T12:00:00+00:00",
        "inputs": [
            _file_record(root, f"manifest-inputs/{role}.dat", role)
            for role in sorted(scorer.ACQUISITION_MANIFEST_INPUT_ROLES)
        ],
        "outputs": [
            _file_record(root, f"manifest-outputs/{role}.dat", role)
            for role in sorted(scorer.ACQUISITION_MANIFEST_OUTPUT_ROLES)
        ],
        "outcome_join_performed": False,
        "design_receipt_verified": True,
    }


def test_scorer_requires_exact_acquisition_manifest_roles_and_top_level(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(scorer, "ROOT", tmp_path)
    expected_top_level = frozenset(
        {
            "schema_version",
            "created_at",
            "inputs",
            "outputs",
            "outcome_join_performed",
            "design_receipt_verified",
        }
    )
    payload = _valid_acquisition_manifest(tmp_path)
    valid_path = _write(tmp_path / "valid-acquisition-manifest.json", payload)
    scorer._verify_manifest(
        valid_path,
        "sec_filing_change_document_acquisition_manifest_v1",
        expected_inputs=scorer.ACQUISITION_MANIFEST_INPUT_ROLES,
        expected_outputs=scorer.ACQUISITION_MANIFEST_OUTPUT_ROLES,
        expected_top_level=expected_top_level,
    )

    payload = _valid_acquisition_manifest(tmp_path)
    payload["inputs"][0]["role"] = "substituted_input"
    wrong_role_path = _write(tmp_path / "wrong-role-manifest.json", payload)
    with pytest.raises(ValueError, match="input roles differ"):
        scorer._verify_manifest(
            wrong_role_path,
            "sec_filing_change_document_acquisition_manifest_v1",
            expected_inputs=scorer.ACQUISITION_MANIFEST_INPUT_ROLES,
            expected_outputs=scorer.ACQUISITION_MANIFEST_OUTPUT_ROLES,
            expected_top_level=expected_top_level,
        )

    payload = _valid_acquisition_manifest(tmp_path)
    payload["outcome_override"] = True
    extra_field_path = _write(tmp_path / "extra-field-manifest.json", payload)
    with pytest.raises(ValueError, match="top-level fields"):
        scorer._verify_manifest(
            extra_field_path,
            "sec_filing_change_document_acquisition_manifest_v1",
            expected_inputs=scorer.ACQUISITION_MANIFEST_INPUT_ROLES,
            expected_outputs=scorer.ACQUISITION_MANIFEST_OUTPUT_ROLES,
            expected_top_level=expected_top_level,
        )


def _valid_coverage_inputs():
    accession = "0000000001-24-000001"
    accepted = pd.Timestamp("2024-02-20T20:00:00Z")
    return {
        "thresholds": {
            "sec_raw_clock": 1.0,
            "primary_document_retrieval": 0.98,
            "prior_year_pairing": 0.80,
            "parse_success": 0.98,
            "ticker_mapping": 0.95,
            "benchmark_clock": 1.0,
        },
        "acquisition_summary": {
            "through_phase": "00_top25_engineering_smoke",
            "acquired_documents": 1,
            "eligible_documents_through_phase": 1,
        },
        "acquisition_audit": pd.DataFrame(
            {"accession": [accession], "api_raw_delta_seconds": [0.0]}
        ),
        "document_audit": pd.DataFrame(
            {
                "accession": [accession],
                "form": ["10-K"],
                "raw_header_acceptance_at": [accepted],
                "parse_status": ["OK"],
                "parse_error": [None],
            }
        ),
        "rough_pairs": pd.DataFrame(
            {
                "form": ["10-K"],
                "current_accession": [accession],
                "prior_accession": ["0000000001-23-000001"],
                "current_acceptance_at": [accepted],
                "current_release_month": ["2024-02"],
                "download_phase": ["00_top25_engineering_smoke"],
            }
        ),
        "strict_pairs": pd.DataFrame(
            {
                "form": ["10-K"],
                "current_accession": [accession],
                "prior_accession": ["0000000001-23-000001"],
                "current_accepted_at": [accepted],
            }
        ),
        "pair_exclusions": pd.DataFrame(columns=["accepted_at", "form", "reason"]),
        "mapped_headers": pd.DataFrame({"form": ["10-K"], "acceptance_at": [accepted]}),
        "mapping_misses": pd.DataFrame(columns=["form", "acceptance_at"]),
    }


def test_score_coverage_is_recomputable_and_defers_benchmark_clock():
    coverage, breakdown = scorer._build_upstream_coverage(**_valid_coverage_inputs())
    assert coverage["status"] == "PASS_UPSTREAM_SIGNAL_BUILD_COVERAGE"
    assert coverage["all_signal_build_gates_passed"] is True
    assert coverage["gates"]["benchmark_clock"]["passed"] is None
    assert breakdown == [
        {
            "filing_year": 2024,
            "form": "10-K",
            "documents_planned": 1,
            "documents_retrieved": 1,
            "primary_document_retrieval": 1.0,
            "raw_clocks_exact": 1,
            "sec_raw_clock": 1.0,
            "documents_parsed": 1,
            "parse_success": 1.0,
            "parse_exclusion_reasons": {},
            "rough_pairs": 1,
            "rough_current_filings": 1,
            "strict_pairs": 1,
            "prior_year_pairing": 1.0,
            "pair_exclusion_reasons": {},
            "mapped_headers": 1,
            "unmapped_headers": 0,
            "ticker_mapping": 1.0,
        }
    ]


def test_score_coverage_counts_multiple_rough_candidates_once():
    inputs = _valid_coverage_inputs()
    second = inputs["rough_pairs"].copy()
    second.loc[0, "prior_accession"] = "0000000001-23-000002"
    inputs["rough_pairs"] = pd.concat(
        [inputs["rough_pairs"], second], ignore_index=True
    )
    coverage, breakdown = scorer._build_upstream_coverage(**inputs)
    assert coverage["gates"]["prior_year_pairing"]["value"] == 1.0
    assert breakdown[0]["rough_pairs"] == 2
    assert breakdown[0]["rough_current_filings"] == 1

    inputs["rough_pairs"].loc[1, "form"] = "10-Q"
    with pytest.raises(ValueError, match="disagree on current-filing metadata"):
        scorer._build_upstream_coverage(**inputs)


def test_score_coverage_rejects_strict_pair_with_substituted_prior():
    inputs = _valid_coverage_inputs()
    inputs["strict_pairs"].loc[0, "prior_accession"] = "0000000001-23-999999"
    with pytest.raises(ValueError, match="outside the frozen rough-pair candidate"):
        scorer._build_upstream_coverage(**inputs)


def test_score_coverage_blocks_failed_parse_or_pairing_gate():
    inputs = _valid_coverage_inputs()
    inputs["document_audit"].loc[0, "parse_status"] = "EXCLUDED"
    inputs["strict_pairs"] = inputs["strict_pairs"].iloc[0:0]
    coverage, _ = scorer._build_upstream_coverage(**inputs)
    assert coverage["status"] == "BLOCKED_DATA_COVERAGE"
    assert coverage["gates"]["parse_success"]["passed"] is False
    assert coverage["gates"]["prior_year_pairing"]["passed"] is False
