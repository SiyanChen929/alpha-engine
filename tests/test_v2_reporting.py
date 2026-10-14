from pathlib import Path

import pandas as pd
import pytest

from alpha_factor_engine.v2_reporting import (
    V2_CHAMPION_ROLE,
    V2_ROLE_ORDER,
    create_v2_opened_diagnostic_report,
)


def _v2_inputs():
    dates = pd.date_range("2024-01-31", periods=18, freq="ME", tz="UTC")
    base_returns = {
        "m1_baseline": 0.008,
        "m2_diversified_momentum": 0.014,
        "m3_sector_beta_constrained": 0.012,
        "residual_ml_challenger": 0.011,
    }
    returns = {
        role: pd.DataFrame(
            {
                "signal_date": dates,
                "net_return": [
                    level + ((index % 4) - 1.5) * 0.003
                    for index in range(len(dates))
                ],
            }
        )
        for role, level in base_returns.items()
    }

    holdings = {}
    for role in V2_ROLE_ORDER:
        rows = []
        for date in dates:
            if role == "m3_sector_beta_constrained":
                weights = [0.18, 0.17, 0.16, 0.16, 0.17, 0.16]
                betas = [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5]
                sectors = ["A", "A", "B", "B", "C", "C"]
            else:
                weights = [1 / 6] * 6
                betas = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
                sectors = ["A", "A", "A", "B", "C", "D"]
            for index, (weight, beta, sector) in enumerate(
                zip(weights, betas, sectors, strict=True)
            ):
                rows.append(
                    {
                        "signal_date": date,
                        "ticker": f"T{index}",
                        "weight": weight,
                        "beta_12m": beta,
                        "sector": sector,
                    }
                )
        holdings[role] = pd.DataFrame(rows)

    metrics = pd.DataFrame(
        {
            "strategy": list(V2_ROLE_ORDER),
            "period": ["opened"] * 4,
            "annualized_return": [0.10, 0.18, 0.15, 0.14],
            "annualized_volatility": [0.14, 0.17, 0.13, 0.12],
            "sharpe": [0.71, 1.06, 1.15, 1.17],
            "max_drawdown": [-0.12, -0.13, -0.09, -0.10],
            "active_annualized_return": [0.0, 0.07, 0.045, 0.035],
            "active_information_ratio": [0.0, 0.70, 0.60, 0.55],
            "relative_max_drawdown": [0.0, -0.08, -0.06, -0.07],
        }
    )

    cost_rows = []
    for role, active_return, information_ratio in (
        ("m2_diversified_momentum", 0.07, 0.70),
        ("m3_sector_beta_constrained", 0.045, 0.60),
        ("residual_ml_challenger", 0.035, 0.55),
    ):
        for cost, haircut in ((10, 0.0), (20, 0.01), (40, 0.025)):
            cost_rows.append(
                {
                    "strategy": role,
                    "cost_bps": cost,
                    "annualized_return": active_return + 0.10 - haircut,
                    "sharpe": information_ratio + 0.30 - haircut * 5,
                    "active_annualized_return": active_return - haircut,
                    "active_information_ratio": information_ratio - haircut * 5,
                }
            )
    cost_sensitivity = pd.DataFrame(cost_rows)

    capacity_summary = pd.DataFrame(
        {
            "strategy": list(V2_ROLE_ORDER),
            "portfolio_aum_usd": [10_000_000] * 4,
            "maximum_adv_participation_limit": [0.01] * 4,
            "maximum_adv_participation_observed": [0.004, 0.008, 0.007, 0.009],
            "capacity_violations": [0] * 4,
            "capacity_passed": [True] * 4,
        }
    )
    predeclared_gates = {
        "historical_evidence_status": "OPENED_DIAGNOSTIC",
        "holdout_opened": True,
        "research_status": "PASS",
        "stage_gates": {
            "m2_vs_m1": {
                "active_information_ratio": {"value": 0.70, "minimum": 0.50},
                "active_annualized_return": {"value": 0.07, "minimum": 0.04},
            },
            "m3_constraints": {
                "maximum_absolute_beta": {"value": 0.03, "maximum": 0.10},
                "maximum_sector_share": {"value": 0.35, "maximum": 0.40},
            },
            "residual_ml_increment": {
                "incremental_bootstrap_p05": {"value": 0.001, "minimum": 0.0}
            },
        },
        # Even an erroneously supplied PASS cannot override the opened-test block.
        "production_status": "PASS",
        "production_data_checks": {
            "survivorship_free": False,
            "permanent_identifiers": False,
            "delisting_returns": False,
            "execution_grade_quotes": False,
            "historical_borrow": False,
            "historical_industry_classification": False,
            "alfred_vintages_used": False,
        },
    }
    data_quality = {
        "evidence_status": "opened_diagnostic",
        "universe_mode": "public_current_ticker_proxy",
        "survivorship_free": False,
        "permanent_identifiers": False,
        "delisting_returns": False,
        "execution_data": "adjusted-open proxy; no TAQ/NBBO",
        "historical_borrow": False,
        "historical_industry_classification": False,
        "alfred_vintages_used": False,
    }
    assumptions = {
        "annualization_factor": 12,
        "opened_start": "2024-01-01T00:00:00Z",
        "trading_cost_bps_one_way": 20,
        "annual_short_borrow_bps": 200,
        "portfolio_aum_usd": 10_000_000,
        "maximum_adv_participation": 0.01,
        "charge_initial_entry": True,
        "charge_terminal_liquidation": True,
    }
    return {
        "returns": returns,
        "holdings": holdings,
        "metrics": metrics,
        "cost_sensitivity": cost_sensitivity,
        "capacity_summary": capacity_summary,
        "predeclared_gates": predeclared_gates,
        "data_quality": data_quality,
        "assumptions": assumptions,
    }


def test_v2_report_is_fixed_role_opened_and_production_blocked(tmp_path):
    result = create_v2_opened_diagnostic_report(tmp_path, **_v2_inputs())

    assert result["champion_role"] == V2_CHAMPION_ROLE
    assert result["diagnostic_status"] == "PASS"
    assert result["production_status"] == "BLOCKED"
    assert result["production_decision"] == "NOT PRODUCTION-ELIGIBLE — DO NOT DEPLOY"
    assert result["opened_test"] is True
    assert result["current_ticker_limitation"] is True
    assert result["capacity_passed_at_configured_aum"] is True
    assert len(result["charts"]) == 6
    assert all(Path(path).is_file() for path in result["charts"])

    report = Path(result["report"]).read_text(encoding="utf-8")
    assert "Diagnostic decision: PASS" in report
    assert "Production decision: BLOCKED" in report
    assert "DO NOT DEPLOY" in report
    assert "M2 Diversified Momentum is the predeclared champion" in report
    assert "M1 Baseline" in report
    assert "M3 Risk-Bounded Momentum" in report
    assert "Residual-ML Challenger" in report
    assert "CURRENT-TICKER / survivorship-biased proxy" in report
    assert "20.0 bps one-way" in report
    assert "$10.00 million AUM" in report
    assert "1.00% maximum trailing-ADV participation" in report
    assert "200.0 bps annual short-borrow" in report
    assert "m2_vs_m1" in report
    assert "Production decision: PASS" not in report


def test_v2_report_rejects_role_drift_before_writing(tmp_path):
    inputs = _v2_inputs()
    inputs["returns"] = dict(inputs["returns"])
    inputs["returns"].pop("residual_ml_challenger")

    with pytest.raises(ValueError, match="exactly the canonical v2 roles"):
        create_v2_opened_diagnostic_report(tmp_path, **inputs)
    assert not (tmp_path / "v2_opened_diagnostic.md").exists()


def test_v2_report_fails_closed_when_predeclared_gate_fails(tmp_path):
    inputs = _v2_inputs()
    inputs["predeclared_gates"]["stage_gates"]["m2_vs_m1"][
        "active_information_ratio"
    ]["value"] = 0.2

    result = create_v2_opened_diagnostic_report(tmp_path, **inputs)

    assert result["diagnostic_status"] == "FAIL"
    assert result["failed_gates"] == ["m2_vs_m1.active_information_ratio"]
    assert result["production_status"] == "BLOCKED"
