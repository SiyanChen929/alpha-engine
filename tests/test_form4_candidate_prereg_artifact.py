import json
from pathlib import Path

import pandas as pd
import pytest

from alpha_factor_engine.research.form4_candidate_prereg import verify_closed_manifest


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "artifacts/form4_event_alpha_v1/candidate_preregistration_2020q1_2025q4_v2"
)

pytestmark = pytest.mark.skipif(
    not ARTIFACT.exists(),
    reason="requires the generated Form 4 candidate-census artifact (not redistributed)",
)


def _json(name: str) -> dict:
    return json.loads((ARTIFACT / name).read_text(encoding="utf-8"))


def test_candidate_census_counts_reconcile_and_no_return_column_exists():
    summary = _json("run_summary.json")
    candidates = pd.read_parquet(ARTIFACT / "candidate_accessions.parquet")
    years = pd.read_csv(ARTIFACT / "candidate_counts_by_year.csv")

    assert len(candidates) == summary["structural_candidate_accessions"] == 25_965
    assert years["structural_candidate_accessions"].sum() == len(candidates)
    assert candidates["preclock_evaluation_candidate"].sum() == summary[
        "preclock_evaluation_candidates"
    ]
    assert candidates["preliminary_primary_candidate"].sum() == summary[
        "preliminary_primary_opportunistic_candidates"
    ]
    assert not candidates["research_eligible"].any()
    assert not any(
        token in column.lower()
        for column in candidates.columns
        for token in ["return", "pnl", "alpha", "future_price"]
    )


def test_governance_failure_is_explicit_and_chronologically_correct():
    summary = _json("run_summary.json")
    record = _json("governance_failure_record.json")
    gates = pd.read_csv(ARTIFACT / "validation_gates.csv")

    assert pd.Timestamp(record["candidate_census_generated_at"]) < pd.Timestamp(
        record["protocol_frozen_at"]
    )
    assert record["passed"] is False
    assert record["return_or_price_outcomes_opened"] is False
    assert summary["candidate_census_validly_preregistered"] is False
    gate = gates.set_index("gate").loc["preregistration_precedes_candidate_generation"]
    assert str(gate["passed"]).lower() == "false"


def test_acquisition_counts_reconcile_and_manifest_is_closed():
    summary = _json("run_summary.json")
    raw = pd.read_csv(ARTIFACT / "raw_sgml_acquisition_plan.csv")
    daily = pd.read_csv(ARTIFACT / "daily_index_acquisition_plan.csv")

    assert raw["accession"].is_unique
    assert len(raw) == summary["raw_sgml_assets_required"] == 19_883
    assert len(daily) == summary["daily_index_weekday_assets_planned"] == 1_576
    assert summary["total_planned_sec_gets_before_extensions"] == len(raw) + len(daily)
    assert verify_closed_manifest(ARTIFACT / "artifact_manifest.json")
