from pathlib import Path

import pandas as pd

from alpha_factor_engine.champion_reporting import (
    _normalize_capacity_summary,
    _normalize_returns,
    create_champion_challenger_report,
)


def test_return_curves_use_realization_exit_timestamp():
    frame = pd.DataFrame(
        {
            "signal_date": [pd.Timestamp("2023-12-29T23:59:00Z")],
            "exit_at": [pd.Timestamp("2024-02-01T14:30:00Z")],
            "net_return": [0.01],
        }
    )
    normalized = _normalize_returns(
        {"candidate": frame},
        opened_start=pd.Timestamp("2024-01-01T00:00:00Z"),
    )

    assert normalized["candidate"].loc[0, "date"] == pd.Timestamp(
        "2024-02-01T14:30:00Z"
    )


def _inputs():
    dates = pd.date_range("2024-01-31", periods=18, freq="ME", tz="UTC")
    strategy_names = [
        "equal_weight_universe",
        "momentum_12_1_top6",
        "ml_full_top6",
        "ml_full_neutral_long_short",
    ]
    return_values = {
        "equal_weight_universe": 0.010,
        "momentum_12_1_top6": 0.014,
        "ml_full_top6": 0.012,
        "ml_full_neutral_long_short": 0.002,
    }
    returns = {
        name: pd.DataFrame(
            {
                "signal_date": dates,
                "net_return": [value + ((index % 3) - 1) * 0.006 for index in range(len(dates))],
            }
        )
        for name, value in return_values.items()
    }
    holdings = {}
    for name in strategy_names:
        rows = []
        for date in dates:
            for index in range(6):
                rows.append(
                    {
                        "signal_date": date,
                        "ticker": f"T{index}",
                        "weight": 1 / 6,
                    }
                )
        holdings[name] = pd.DataFrame(rows)
    metrics = pd.DataFrame(
        {
            "strategy": strategy_names,
            "portfolio_type": ["benchmark", "long_only", "long_only", "long_short"],
            "annualized_return": [0.12, 0.17, 0.14, 0.02],
            "annualized_volatility": [0.15, 0.18, 0.17, 0.08],
            "sharpe": [0.80, 0.94, 0.82, 0.25],
            "max_drawdown": [-0.12, -0.14, -0.15, -0.10],
            "active_annualized_return": [0.0, 0.05, 0.02, None],
            "active_information_ratio": [0.0, 0.60, 0.25, None],
            "relative_max_drawdown": [0.0, -0.08, -0.10, None],
        }
    )
    costs = pd.DataFrame(
        {
            "strategy": ["momentum_12_1_top6"] * 3 + ["ml_full_top6"] * 3,
            "cost_bps": [15, 25, 30, 15, 25, 30],
            "annualized_return": [0.17, 0.16, 0.15, 0.14, 0.13, 0.12],
            "sharpe": [0.94, 0.88, 0.82, 0.82, 0.75, 0.69],
            "active_annualized_return": [0.05, 0.04, 0.03, 0.02, 0.01, 0.005],
            "active_information_ratio": [0.60, 0.52, 0.45, 0.25, 0.15, 0.08],
        }
    )
    gates = {
        "historical_evidence_status": "OPENED_DIAGNOSTIC",
        "champion": "momentum_12_1_top6",
        "research_status": "PASS",
        "research_checks": {
            "active_information_ratio": {"value": 0.60, "minimum": 0.50},
        },
        "production_data_checks": {
            "survivorship_free": False,
            "delisting_returns": False,
            "execution_grade_quotes": False,
            "historical_borrow": False,
        },
        "production_status": "BLOCKED",
        "holdout_opened": True,
    }
    data_quality = {
        "universe_mode": "public_current_ticker_proxy",
        "evidence_status": "opened_diagnostic",
        "survivorship_free": False,
        "delisting_returns": False,
        "historical_borrow": False,
        "historical_industry_classification": False,
        "alfred_vintages_used": False,
        "execution_data": "Yahoo adjusted-open proxy; no TAQ/NBBO",
    }
    config = {
        "research": {
            "evaluation_entry_at_or_after": "2024-01-01T00:00:00Z",
            "evidence_status": "opened_diagnostic",
        },
        "reporting": {"champion": "momentum_12_1_top6", "annualization_factor": 12},
        "universe": {
            "mode": "public_current_ticker_proxy",
            "source": "dynamic_top30_from_current_ticker_proxy",
        },
        "costs": {
            "trading_cost_bps_one_way": 15,
            "portfolio_aum_usd": 10_000_000,
            "maximum_adv_participation": 0.01,
            "charge_initial_entry": True,
            "charge_terminal_liquidation": True,
        },
        "neutral_portfolio": {"annual_short_borrow_bps": 200},
    }
    capacity = pd.DataFrame(
        {
            "strategy": ["momentum_12_1_top6", "ml_full_top6"],
            "portfolio_aum_usd": [10_000_000, 10_000_000],
            "maximum_adv_participation_limit": [0.01, 0.01],
            "maximum_adv_participation_observed": [0.008, 0.012],
            "capacity_violations": [0, 2],
            "capacity_passed": [True, False],
        }
    )
    return returns, holdings, metrics, costs, gates, data_quality, config, capacity


def test_report_labels_opened_cost_and_capacity_assumptions(tmp_path):
    inputs = _inputs()
    result = create_champion_challenger_report(tmp_path, *inputs)

    report = Path(result["report"]).read_text()
    assert result["research_status"] == "PASS"
    assert result["production_status"] == "FAIL"
    assert result["production_decision"].startswith("NOT PRODUCTION-ELIGIBLE")
    assert result["capacity_passed_at_configured_aum"] is True
    assert "OPENED diagnostic" in report
    assert "CURRENT-TICKER / survivorship-biased universe" in report
    assert "15.0 bps one-way cost" in report
    assert "Initial entry: charged; terminal liquidation: charged" in report
    assert "Long-only candidates have no borrow charge" in report
    assert "200.0 bps annual short-borrow" in report
    assert "$10.00 million AUM" in report
    assert "1.00% of trailing ADV" in report
    assert "PRODUCTION GATE PASS" not in report
    assert len(result["charts"]) == 6
    assert all(Path(path).is_file() for path in result["charts"])


def test_capacity_pass_requires_the_configured_aum_and_limit(tmp_path):
    *_, data_quality, config, capacity = _inputs()
    capacity.loc[:, "portfolio_aum_usd"] = 5_000_000
    normalized, source = _normalize_capacity_summary(
        tmp_path,
        capacity,
        data_quality=data_quality,
        config=config,
        champion="momentum_12_1_top6",
    )
    assert source == "function input"
    assert not normalized["capacity_passed_at_configured_aum"].any()

    capacity.loc[:, "portfolio_aum_usd"] = 10_000_000
    capacity.to_csv(tmp_path / "capacity_summary.csv", index=False)
    from_file, source = _normalize_capacity_summary(
        tmp_path,
        None,
        data_quality=data_quality,
        config=config,
        champion="momentum_12_1_top6",
    )
    assert source == "capacity_summary.csv"
    assert from_file.loc[
        from_file["strategy"].eq("momentum_12_1_top6"),
        "capacity_passed_at_configured_aum",
    ].item()
