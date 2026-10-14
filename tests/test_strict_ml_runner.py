from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/run_liquid_momentum_top150_monotonic_gbdt_overlay.py"
)
SPEC = importlib.util.spec_from_file_location("strict_ml_runner_under_test", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)
_authorization_status = RUNNER._authorization_status
_resolve_candidate_count = RUNNER._resolve_candidate_count
_validate_source_panel = RUNNER._validate_source_panel


def _source_row() -> pd.DataFrame:
    signal = pd.Timestamp("2025-01-31", tz="UTC")
    return pd.DataFrame(
        {
            "signal_date": [signal],
            "entry_at": [signal + pd.Timedelta(days=1)],
            "exit_at": [signal + pd.Timedelta(days=29)],
            "target_return": [0.01],
            "ticker": ["A"],
            "eligible": [True],
            "price_available_at": [signal],
            "latest_input_available_at": [signal],
            "fundamental_available_at": [signal - pd.Timedelta(days=1)],
            "log_market_cap": [10.0],
        }
    )


def test_source_panel_validator_executes_and_fails_closed_on_future_data():
    panel = _source_row()
    evidence = _validate_source_panel(panel, ["log_market_cap"])

    assert evidence["rows"] == 1
    assert evidence["decision_before_entry_before_exit"] is True
    invalid = panel.copy()
    invalid["price_available_at"] = invalid["signal_date"] + pd.Timedelta(seconds=1)
    with pytest.raises(ValueError, match="price features were unavailable"):
        _validate_source_panel(invalid, ["log_market_cap"])


def test_fatal_lineage_failure_can_never_authorize_portfolio():
    rows = {
        "zero_security_overlap": True,
        "all_training_labels_exited_before_cutoff": True,
        "fold_local_target_scope": True,
        "strict_heldout_mean_incremental_rank_ic": True,
        "strict_heldout_incremental_ic_hac_tstat": True,
        "positive_security_folds": True,
        "upstream_membership_frozen_before_label_completion": True,
        "upstream_survivorship_free": False,
        "permanent_security_identifier": False,
    }
    gates = pd.DataFrame(
        [{"gate": gate, "passed": passed} for gate, passed in rows.items()]
    )

    status = _authorization_status(gates)

    assert status["statistical_ic_gate_pass"] is True
    assert status["fatal_lineage_gate_pass"] is False
    assert status["next_portfolio_diagnostic_eligible"] is False
    assert status["portfolio_authorized"] is False
    assert status["portfolio_stage"] == "diagnostic_only_blocked"


def test_runner_accepts_explicit_null_for_full_mother_universe_only():
    assert _resolve_candidate_count(
        {"universe": {"momentum_candidate_count": None}}
    ) is None
    assert _resolve_candidate_count(
        {"universe": {"momentum_candidate_count": 150}}
    ) == 150
    with pytest.raises(ValueError, match="positive or null"):
        _resolve_candidate_count({"universe": {"momentum_candidate_count": 0}})
