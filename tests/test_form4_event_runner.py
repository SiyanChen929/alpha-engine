from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alpha_factor_engine.research.form4_event_runner import (
    ManifestIntegrityError,
    run_form4_event_study,
    verify_artifact_manifest,
)


BUILDER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/build_form4_synthetic_fixture.py"
)
SPEC = importlib.util.spec_from_file_location(
    "form4_synthetic_fixture_builder_under_test", BUILDER_PATH
)
assert SPEC is not None and SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)
build_synthetic_bundle = BUILDER.build_synthetic_bundle


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rewrite_manifest(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def test_synthetic_fixture_runs_complete_diagnostic_and_reconciles_ledger(tmp_path):
    manifest = build_synthetic_bundle(tmp_path / "bundle")
    output = tmp_path / "output"

    paths = run_form4_event_study(manifest, output)

    run = _read_json(paths["run_manifest"])
    gates = _read_json(paths["validation_gates"])
    assert run["research_status"] == "NOT VALIDATED"
    assert run["research_scope"] == "SYNTHETIC_DIAGNOSTIC_ONLY"
    assert run["alpha_validated"] is False
    assert run["performance_claim_permitted"] is False
    assert run["diagnostic_returns_emitted"] is True
    assert gates["all_required_diagnostic_gates_passed"] is True
    assert gates["all_required_research_candidate_gates_passed"] is False
    assert {
        row["gate"]: row["passed"] for row in gates["gates"]
    }["historical_real_data_source"] is False

    returns = pd.read_csv(output / "event_returns_diagnostic.csv")
    ledger = pd.read_csv(output / "trade_ledger_diagnostic.csv")
    placebo = pd.read_csv(output / "pre_event_placebo_diagnostic.csv")
    controls = pd.read_parquet(output / "matched_controls_diagnostic.parquet")
    events = pd.read_parquet(output / "purchase_events_diagnostic.parquet")

    assert returns["horizon_sessions"].tolist() == [5, 20, 60]
    assert placebo["horizon_sessions"].tolist() == [5, 20, 60]
    assert placebo["placebo_status"].eq("ok").all()
    assert len(controls) == 2
    assert controls["match_status"].eq("matched").all()
    assert len(events) == 1
    assert pd.Timestamp(events.loc[0, "information_at"]) == pd.Timestamp(
        "2025-01-02T13:00:00Z"
    )
    assert pd.Timestamp(events.loc[0, "entry_at"]) == pd.Timestamp(
        "2025-01-02T14:30:00Z"
    )
    assert len(ledger) == 9
    assert set(ledger["leg_direction"]) == {
        "LONG_EVENT",
        "SHORT_MATCHED_CONTROL",
    }
    assert ledger["performance_claim_permitted"].eq(False).all()
    assert returns["ledger_reconciled"].eq(True).all()

    twenty = ledger.loc[ledger["horizon_sessions"].eq(20)]
    assert np.isclose(twenty["gross_capital_weight"].abs().sum(), 1.0)
    assert np.isclose(twenty["execution_cost_contribution"].sum(), 0.0035)
    independently_rebuilt_net = (
        twenty["gross_return_contribution"].sum()
        - twenty["execution_cost_contribution"].sum()
        - twenty["borrow_cost_contribution"].sum()
        - twenty["financing_cost_contribution"].sum()
    )
    reported_net = returns.loc[
        returns["horizon_sessions"].eq(20),
        "matched_net_return_on_gross_capital",
    ].iloc[0]
    assert np.isclose(independently_rebuilt_net, reported_net, atol=1e-12)

    # Independent raw-input reconstruction: do not call the runner's return or
    # ledger helpers. Rebuild the 20-session event/control return and every cost.
    raw_prices = pd.read_csv(manifest.parent / "tables/daily_prices.csv")
    cost_model = _read_json(manifest.parent / "spec/cost_model.json")
    summary_20 = returns.loc[returns["horizon_sessions"].eq(20)].iloc[0]
    entry_date = pd.Timestamp(summary_20["entry_at"]).tz_convert(
        "America/New_York"
    ).date().isoformat()
    exit_date = pd.Timestamp(summary_20["exit_open_at"]).tz_convert(
        "America/New_York"
    ).date().isoformat()

    def raw_return(ticker: str) -> float:
        security = raw_prices.loc[raw_prices["ticker"].eq(ticker)].set_index("date")
        return float(security.loc[exit_date, "open"] / security.loc[entry_date, "open"] - 1.0)

    event_raw_return = raw_return("EVT")
    control_raw_returns = [raw_return(ticker) for ticker in controls["control_ticker"]]
    hold_days = float(
        (
            pd.Timestamp(exit_date).date() - pd.Timestamp(entry_date).date()
        ).days
    )
    gross_from_raw = 0.5 * event_raw_return - 0.25 * sum(control_raw_returns)
    execution_from_spec = (
        0.5 * 2 * cost_model["event_one_way_bps"] / 10_000
        + 0.5 * 2 * cost_model["control_one_way_bps"] / 10_000
    )
    borrow_from_spec = (
        0.5
        * cost_model["control_borrow_annual_bps"]
        / 10_000
        * hold_days
        / cost_model["calendar_days_per_year"]
    )
    financing_from_spec = (
        0.5
        * (
            cost_model["event_financing_annual_bps"]
            + cost_model["control_financing_annual_bps"]
        )
        / 10_000
        * hold_days
        / cost_model["calendar_days_per_year"]
    )
    manual_net_from_raw = (
        gross_from_raw
        - execution_from_spec
        - borrow_from_spec
        - financing_from_spec
    )
    assert np.isclose(manual_net_from_raw, reported_net, atol=1e-12)
    assert set(ledger.loc[ledger["horizon_sessions"].eq(60), "holding_calendar_days"]) == {
        88.0
    }

    assert not (output / "event_returns.csv").exists()
    assert not (output / "trade_ledger.csv").exists()
    assert verify_artifact_manifest(paths["artifact_manifest"]) is True


def test_missing_corporate_action_role_blocks_all_return_and_ledger_outputs(tmp_path):
    manifest = build_synthetic_bundle(tmp_path / "bundle")
    payload = _read_json(manifest)
    payload["records"] = [
        record for record in payload["records"] if record["role"] != "corporate_actions"
    ]
    _rewrite_manifest(manifest, payload)

    output = tmp_path / "blocked"
    paths = run_form4_event_study(manifest, output)

    run = _read_json(paths["run_manifest"])
    gates = _read_json(paths["validation_gates"])
    required_gate = {
        row["gate"]: row for row in gates["gates"]
    }["required_manifest_roles_present"]
    assert required_gate["passed"] is False
    assert "corporate_actions" in required_gate["evidence"]
    assert run["research_status"] == "NOT VALIDATED"
    assert run["research_scope"] == "DIAGNOSTIC_ONLY_BLOCKED"
    assert run["diagnostic_returns_emitted"] is False
    assert not list(output.glob("*return*"))
    assert not list(output.glob("*ledger*"))


def test_raw_submission_byte_tamper_fails_before_output_creation(tmp_path):
    manifest = build_synthetic_bundle(tmp_path / "bundle")
    payload = _read_json(manifest)
    raw_record = next(
        record
        for record in payload["records"]
        if record["role"].startswith("raw_submission:")
    )
    raw_path = manifest.parent / raw_record["path"]
    raw_path.write_bytes(raw_path.read_bytes() + b"tampered")
    output = tmp_path / "should-not-exist"

    with pytest.raises(ManifestIntegrityError, match="integrity mismatch"):
        run_form4_event_study(manifest, output)

    assert not output.exists()


def test_gap_in_cross_quarter_daily_search_blocks_returns(tmp_path):
    manifest = build_synthetic_bundle(tmp_path / "bundle")
    payload = _read_json(manifest)
    payload["records"] = [
        record
        for record in payload["records"]
        if record["role"] != "daily_master:2024-12-31"
    ]
    _rewrite_manifest(manifest, payload)

    output = tmp_path / "blocked"
    paths = run_form4_event_study(manifest, output)

    run = _read_json(paths["run_manifest"])
    gates = _read_json(paths["validation_gates"])
    failure = {row["gate"]: row for row in gates["gates"]}[
        "pipeline_evidence_complete"
    ]
    assert failure["passed"] is False
    assert "exchange-session gaps" in failure["evidence"]
    assert run["diagnostic_returns_emitted"] is False
    assert not (output / "event_returns_diagnostic.csv").exists()
    assert not (output / "trade_ledger_diagnostic.csv").exists()


def test_incomplete_two_leg_cost_model_blocks_without_performance(tmp_path):
    manifest = build_synthetic_bundle(tmp_path / "bundle")
    payload = _read_json(manifest)
    cost_record = next(
        record for record in payload["records"] if record["role"] == "cost_model"
    )
    cost_path = manifest.parent / cost_record["path"]
    cost_model = _read_json(cost_path)
    cost_model["two_leg_complete"] = False
    content = (
        json.dumps(cost_model, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    cost_path.write_bytes(content)
    cost_record["bytes"] = len(content)
    cost_record["sha256"] = hashlib.sha256(content).hexdigest()
    _rewrite_manifest(manifest, payload)

    output = tmp_path / "blocked"
    paths = run_form4_event_study(manifest, output)

    run = _read_json(paths["run_manifest"])
    assert run["research_status"] == "NOT VALIDATED"
    assert run["diagnostic_returns_emitted"] is False
    assert "two_leg_complete" in run["blocked_reason"]
    assert not (output / "event_returns_diagnostic.csv").exists()
    assert not (output / "trade_ledger_diagnostic.csv").exists()


def test_synthetic_bundle_cannot_be_promoted_by_relabelling_source_class(tmp_path):
    manifest = build_synthetic_bundle(tmp_path / "bundle")
    payload = _read_json(manifest)
    payload["source_class"] = "historical_research"
    _rewrite_manifest(manifest, payload)

    output = tmp_path / "blocked"
    paths = run_form4_event_study(manifest, output)

    run = _read_json(paths["run_manifest"])
    assert run["research_status"] == "NOT VALIDATED"
    assert run["research_scope"] == "DIAGNOSTIC_ONLY_BLOCKED"
    assert "synthetic provenance" in run["blocked_reason"]
    assert not (output / "event_returns_diagnostic.csv").exists()
    assert not (output / "trade_ledger_diagnostic.csv").exists()


def test_artifact_manifest_detects_post_run_output_change(tmp_path):
    manifest = build_synthetic_bundle(tmp_path / "bundle")
    output = tmp_path / "output"
    paths = run_form4_event_study(manifest, output)
    assert verify_artifact_manifest(paths["artifact_manifest"]) is True

    report = output / "report.md"
    report.write_text(report.read_text(encoding="utf-8") + "\nchanged\n", encoding="utf-8")

    with pytest.raises(ManifestIntegrityError, match="integrity mismatch"):
        verify_artifact_manifest(paths["artifact_manifest"])
