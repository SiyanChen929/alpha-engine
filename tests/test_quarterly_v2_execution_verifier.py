from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = ROOT / "scripts/verify_quarterly_v2_execution_diagnostic_v1.py"
SPEC = importlib.util.spec_from_file_location("quarterly_v2_execution_verifier", VERIFIER_PATH)
assert SPEC is not None and SPEC.loader is not None
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)

DEFAULT_CONFIG = VERIFIER.DEFAULT_CONFIG
DEFAULT_PREREG = VERIFIER.DEFAULT_PREREG
DEFAULT_REPORTED = VERIFIER.DEFAULT_REPORTED
DEFAULT_SOURCE = VERIFIER.DEFAULT_SOURCE
absolute_trade_sum = VERIFIER.absolute_trade_sum
compare_frames = VERIFIER.compare_frames
marked_weights = VERIFIER.marked_weights
run_audit = VERIFIER.run_audit
sha256_file = VERIFIER.sha256_file
stable_equal_count_deciles = VERIFIER.stable_equal_count_deciles
verify_manifest_records = VERIFIER.verify_manifest_records


def test_verifier_has_no_forbidden_project_imports() -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "scripts/verify_quarterly_v2_execution_diagnostic_v1.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not any(name.startswith("alpha_factor_engine") for name in imported)
    assert "run_quarterly_v2_execution_diagnostic_v1" not in imported


def test_stable_deciles_ignore_returns_and_use_frozen_tie_breaks() -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["Z", "A", "B", "C", "D", "E", "F", "G", "H", "I"],
            "issuer_cik": [2, 2, 1, 3, 4, 5, 6, 7, 8, 9],
            "score": [0, 0, 0, 1, 2, 3, 4, 5, 6, 7],
            "target_return": list(range(10)),
        }
    )
    ranked = stable_equal_count_deciles(frame, score_column="score")
    perturbed = frame.assign(target_return=list(reversed(range(10))))
    ranked_perturbed = stable_equal_count_deciles(perturbed, score_column="score")
    assert ranked["ticker"].tolist()[:3] == ["B", "A", "Z"]
    assert ranked[["ticker", "decile"]].reset_index(drop=True).equals(
        ranked_perturbed[["ticker", "decile"]].reset_index(drop=True)
    )


def test_mark_to_market_turnover_hand_calculation() -> None:
    marked = marked_weights(["A", "B"], {"A": 0.2, "B": -0.2})
    assert marked == pytest.approx({"A": 0.6, "B": 0.4})
    assert absolute_trade_sum(["B", "C"], marked) == pytest.approx(1.2)
    assert absolute_trade_sum(["A", "B"], None) == pytest.approx(1.0)


def test_fault_injection_turnover_tampering_is_detected() -> None:
    expected = pd.DataFrame(
        {
            "signal_date": ["2024-01-01"],
            "long_names": [10],
            "total_one_way_traded_notional": [2.0],
        }
    )
    observed = expected.copy()
    observed.loc[0, "total_one_way_traded_notional"] = 1.0
    result = compare_frames(
        expected,
        observed,
        keys=["signal_date"],
        exact_columns=["long_names"],
        numeric_columns=["total_one_way_traded_notional"],
    )
    assert result["passed"] is False
    assert (
        result["numeric_errors"]["total_one_way_traded_notional"][
            "mismatch_count_over_tolerance"
        ]
        == 1
    )


def test_fault_injection_cost_tampering_is_detected() -> None:
    expected = pd.DataFrame(
        {
            "signal_date": ["2024-01-01"],
            "cost_25bps": [0.005],
            "net_return_25bps": [0.015],
        }
    )
    observed = expected.copy()
    observed.loc[0, "cost_25bps"] = 0.0005
    result = compare_frames(
        expected,
        observed,
        keys=["signal_date"],
        exact_columns=[],
        numeric_columns=["cost_25bps", "net_return_25bps"],
    )
    assert result["passed"] is False
    assert result["numeric_errors"]["cost_25bps"]["mismatch_count_over_tolerance"] == 1


def test_fault_injection_manifest_tampering_is_detected(tmp_path: Path) -> None:
    payload = tmp_path / "payload.txt"
    payload.write_text("abc", encoding="utf-8")
    manifest_path = tmp_path / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "role": "output",
                        "path": payload.resolve().as_posix(),
                        "bytes": payload.stat().st_size,
                        "sha256": sha256_file(payload),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    clean = verify_manifest_records(manifest_path, tmp_path)
    assert clean["record_integrity_passed"] is True
    assert clean["output_closure_passed"] is True

    payload.write_text("abd", encoding="utf-8")
    tampered = verify_manifest_records(manifest_path, tmp_path)
    assert tampered["record_integrity_passed"] is False
    assert payload.resolve().as_posix() in tampered["hash_mismatches"]

    extra = tmp_path / "undeclared.csv"
    extra.write_text("x\n1\n", encoding="utf-8")
    omitted = verify_manifest_records(manifest_path, tmp_path)
    assert omitted["output_closure_passed"] is False
    assert extra.resolve().as_posix() in omitted["output_closure_omissions"]


@pytest.mark.skipif(
    not DEFAULT_REPORTED.exists(),
    reason="requires the generated quarterly_v2_execution_diagnostic_v1 artifact (not redistributed)",
)
def test_current_artifact_reconstructs_but_strict_lineage_fails() -> None:
    verification, _, _ = run_audit(
        source_directory=DEFAULT_SOURCE,
        reported_directory=DEFAULT_REPORTED,
        config_path=DEFAULT_CONFIG,
        prereg_path=DEFAULT_PREREG,
    )
    assert verification["calculation_integrity_passed"] is True
    assert verification["monthly_execution"]["passed"] is True
    assert verification["return_summary_and_hac"]["passed"] is True
    assert verification["capacity_summary"]["passed"] is True
    assert verification["trial_ledger"]["as_file_clock_order_passed"] is False
    assert verification["producer_manifest"]["strict_direct_read_closure_passed"] is False
    assert len(verification["producer_manifest"]["missing_descriptive_direct_inputs"]) == 1
    assert verification["strict_artifact_audit_passed"] is False
    assert verification["historical_backtest_verdict"] == "INVALID"
    assert verification["research_status"] == "NOT_VALIDATED"
    assert verification["portfolio_authorized"] is False
    assert verification["turnover"][
        "full_90_month_mean_total_one_way_traded_notional"
    ] == pytest.approx(1.379517376254258)
    assert verification["turnover"][
        "pre2026_84_month_mean_total_one_way_traded_notional"
    ] == pytest.approx(1.348129074178767)
    assert verification["pre2026_path_metrics"]["0"]["terminal_wealth"] == pytest.approx(
        1.3120565690968418
    )
    assert verification["pre2026_path_metrics"]["25"]["terminal_wealth"] == pytest.approx(
        0.9885610239049707
    )
    assert verification["pre2026_path_metrics"]["50"]["terminal_wealth"] == pytest.approx(
        0.7438466275045956
    )
