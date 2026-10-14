from copy import deepcopy
from datetime import UTC, datetime

import pandas as pd
import pytest

from alpha_factor_engine.research.governance import GovernanceViolation
from alpha_factor_engine.v2_pipeline import (
    _evaluate_v2_gates,
    _write_v2_governance_records,
    load_v2_config,
)


def _config() -> dict:
    return {
        "universe": {"mode": "licensed_pit", "permanent_id_column": "permno"},
        "reporting": {
            "benchmark": "equal_weight_universe",
            "champion": "m2",
        },
        "signal": {"score": "momentum_12_1", "entry": "next_session_open"},
        "candidates": {
            "m2": {"holdings": 10, "retention_rank": 15},
            "residual_overlay": {"momentum_weight": 0.8, "residual_ml_weight": 0.2},
        },
        "model": {"seed": 929, "target": "beta_size_residual_return_rank"},
        "costs": {"trading_cost_bps_one_way": 15},
        "gates": {"active_information_ratio_minimum": 0.5},
        "opened_trials": ["m2", "residual_overlay"],
        "prospective": {
            "strategy_version": "v2-test-seal",
            "governance_dir": "governance/v2-test-seal",
            "freeze_deadline": "2026-09-29T20:00:00-04:00",
            "first_signal_at": "2026-09-30T20:00:00-04:00",
            "first_entry_at": "2026-10-01T09:30:00-04:00",
            "minimum_months": 36,
            "champion": "m2",
            "challenger": "residual_overlay",
            "change_policy": "any_change_requires_new_version",
        },
    }


def _write_records(project_root, config, *, registered_at, source_hash="b" * 64):
    return _write_v2_governance_records(
        config,
        project_root=project_root,
        config_sha256="a" * 64,
        source_tree_sha256=source_hash,
        registered_at=registered_at,
    )


def _gate_config() -> dict:
    config = _config()
    config["costs"]["trading_cost_bps_one_way"] = 15
    config["candidates"]["m3_risk_bounded_momentum_top10"] = {
        "beta_maximum": 1.0
    }
    config["gates"] = {
        "active_information_ratio_minimum": 0.50,
        "active_annualized_return_minimum": 0.04,
        "active_newey_west_tstat_minimum_after_search": 3.00,
        "relative_max_drawdown_floor": -0.20,
        "maximum_single_name_gross_profit_share": 0.20,
        "positive_at_cost_bps": 25,
        "positive_at_double_base_cost": True,
        "maximum_adv_participation": 0.01,
        "residual_ml_incremental_return_minimum": 0.00,
        "residual_ml_bootstrap_sharpe_p05_minimum": 0.00,
        "residual_ml_newey_west_tstat_minimum_after_search": 3.00,
    }
    return config


def _gate_inputs():
    metrics = pd.DataFrame(
        [
            {
                "strategy": "m2_diversified_momentum",
                "active_annualized_return": 0.07,
                "active_information_ratio": 0.70,
                "active_newey_west_tstat": 3.50,
                "relative_max_drawdown": -0.10,
                "maximum_single_name_gross_profit_share": 0.15,
            }
        ]
    )
    cost_sensitivity = pd.DataFrame(
        {
            "strategy": ["m2_diversified_momentum"] * 2,
            "cost_bps": [25.0, 30.0],
            "active_annualized_return": [0.02, 0.01],
        }
    )
    capacity_summary = pd.DataFrame(
        [
            {
                "strategy": "m2_diversified_momentum",
                "maximum_adv_participation_observed": 0.005,
            }
        ]
    )
    residual_incremental = {
        "annualized_arithmetic_return": 0.02,
        "bootstrap_sharpe_p05": 0.10,
        "newey_west_tstat": 3.50,
    }
    constraint_summary = pd.DataFrame(
        [
            {
                "strategy": "m3_sector_beta_constrained",
                "maximum_absolute_beta_exposure": 0.90,
            }
        ]
    )
    production_checks = {
        "survivorship_free": True,
        "permanent_identifiers": True,
        "delisting_returns": True,
        "execution_grade_quotes": True,
        "historical_borrow": True,
        "historical_industry_classification": True,
        "alfred_vintages_used": True,
    }
    return {
        "metrics": metrics,
        "cost_sensitivity": cost_sensitivity,
        "capacity_summary": capacity_summary,
        "residual_incremental": residual_incremental,
        "constraint_summary": constraint_summary,
        "config": _gate_config(),
        "pit_audit": {"future_information_violations": 0},
        "production_data_checks": production_checks,
    }


@pytest.mark.parametrize("content", ["- one\n- two\n", "plain-scalar\n", "null\n"])
def test_load_v2_config_rejects_non_mapping_yaml(tmp_path, content):
    path = tmp_path / "invalid.yml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a mapping"):
        load_v2_config(path)


def test_v2_governance_creates_unopened_predeadline_seal_and_is_idempotent(tmp_path):
    config = _config()
    registered_at = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
    experiment, trial, forecasts = _write_records(
        tmp_path,
        config,
        registered_at=registered_at,
    )
    governance_dir = tmp_path / config["prospective"]["governance_dir"]
    paths = [
        governance_dir / "preregistration.json",
        governance_dir / "trial.json",
        governance_dir / "sealed_forecasts.jsonl",
    ]
    original_bytes = {path.name: path.read_bytes() for path in paths}

    assert experiment.opened_period is False
    assert pd.Timestamp(experiment.registered_at) < pd.Timestamp(
        config["prospective"]["freeze_deadline"]
    )
    assert experiment.first_decision_at == "2026-10-01T00:00:00+00:00"
    assert trial.opened_period is False
    assert forecasts.records == ()
    assert forecasts.decision_count == 0
    assert all(path.is_file() for path in paths)

    repeated = _write_records(
        tmp_path,
        config,
        registered_at=datetime(2026, 9, 2, 12, 0, tzinfo=UTC),
    )
    assert repeated == (experiment, trial, forecasts)
    assert {path.name: path.read_bytes() for path in paths} == original_bytes


def test_v2_governance_rejects_config_and_source_specification_drift(tmp_path):
    config = _config()
    registered_at = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
    config_root = tmp_path / "config-drift"
    _write_records(config_root, config, registered_at=registered_at)
    changed = deepcopy(config)
    changed["candidates"]["m2"]["holdings"] = 11

    with pytest.raises(GovernanceViolation, match="differs from preregistration"):
        _write_records(config_root, changed, registered_at=registered_at)

    source_root = tmp_path / "source-drift"
    _write_records(source_root, config, registered_at=registered_at)
    with pytest.raises(GovernanceViolation, match="differs from preregistration"):
        _write_records(
            source_root,
            config,
            registered_at=registered_at,
            source_hash="c" * 64,
        )


def test_v2_governance_rejects_first_registration_after_freeze(tmp_path):
    with pytest.raises(GovernanceViolation, match="freeze deadline has passed"):
        _write_records(
            tmp_path,
            _config(),
            registered_at=datetime(2026, 9, 30, 0, 0, 1, tzinfo=UTC),
        )


def test_v2_gates_fail_weak_m2_research_even_when_other_inputs_pass():
    inputs = _gate_inputs()
    inputs["metrics"].loc[0, "active_information_ratio"] = 0.10

    result = _evaluate_v2_gates(**inputs)

    assert result["stage_gates"]["m2_vs_equal_weight"][
        "active_information_ratio"
    ]["passed"] is False
    assert result["m2_research_pass"] is False
    assert result["m3_beta_constraint_pass"] is True
    assert result["residual_ml_incremental_pass"] is True
    assert result["research_status"] == "FAIL"
    assert all(result["production_data_checks"].values())
    assert result["production_status"] == "BLOCKED"


def test_v2_gates_fail_weak_residual_increment_while_m2_passes():
    inputs = _gate_inputs()
    inputs["residual_incremental"] = {
        "annualized_arithmetic_return": -0.01,
        "bootstrap_sharpe_p05": -0.10,
        "newey_west_tstat": 0.20,
    }

    result = _evaluate_v2_gates(**inputs)

    assert result["m2_research_pass"] is True
    assert result["residual_ml_incremental_pass"] is False
    assert all(
        check["passed"] is False
        for check in result["stage_gates"]["residual_ml_increment"].values()
    )
    assert result["research_status"] == "FAIL"
    assert all(result["production_data_checks"].values())
    assert result["production_status"] == "BLOCKED"
