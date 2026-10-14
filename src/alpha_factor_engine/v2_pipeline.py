"""Diversified momentum and momentum-anchored residual-ML v2 protocol."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

from alpha_factor_engine.champion_pipeline import (
    _evaluation_cross_sections,
    _feature_sets,
    _filter_membership,
    _replace_with_exact_momentum,
)
from alpha_factor_engine.contracts import sha256_directory, sha256_file, stable_json_hash
from alpha_factor_engine.research.champion_challenger import (
    add_beta_size_residual_target,
    add_decision_time,
    assert_panel_is_point_in_time,
    build_ranked_long_only,
    capacity_diagnostics,
    long_only_evaluation,
    relative_return_frame,
    select_public_proxy_universe,
)
from alpha_factor_engine.research.governance import (
    ExperimentTrialLedger,
    GovernanceViolation,
    ImmutableExperimentSpec,
    SealedForecastLedger,
    TrialRecord,
    create_or_verify_preregistration,
    create_or_verify_trial_record,
    initialize_forecast_ledger,
    load_preregistration,
)
from alpha_factor_engine.research.metrics import (
    moving_block_bootstrap_sharpe,
    performance_summary,
)
from alpha_factor_engine.research.models import model_config_from_dict, walk_forward_predict
from alpha_factor_engine.research.diversified_momentum import (
    build_diversified_momentum_portfolio,
)
from alpha_factor_engine.research.residual_overlay import (
    build_momentum_residual_overlay,
)


def load_v2_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("v2 configuration must contain a mapping")
    return config


def _v2_governance_payload(
    config: dict[str, Any],
    *,
    config_sha256: str,
    source_tree_sha256: str,
) -> dict[str, Any]:
    prospective = config["prospective"]
    return {
        "strategy_version": prospective["strategy_version"],
        "champion": prospective["champion"],
        "challenger": prospective["challenger"],
        "reporting": config["reporting"],
        "universe": config["universe"],
        "signal": config["signal"],
        "candidates": config["candidates"],
        "model": config["model"],
        "costs": config["costs"],
        "gates": config["gates"],
        "opened_trials": config["opened_trials"],
        "freeze_deadline": prospective["freeze_deadline"],
        "first_signal_at": prospective["first_signal_at"],
        "first_entry_at": prospective["first_entry_at"],
        "minimum_months": prospective["minimum_months"],
        "change_policy": prospective["change_policy"],
        "config_sha256": config_sha256,
        "source_tree_sha256": source_tree_sha256,
        "production_data_requirement": (
            "survivorship_free_permanent_ids_delistings_compustat_pit_alfred_"
            "execution_quotes_and_historical_sector_classification"
        ),
    }


def _write_v2_governance_records(
    config: dict[str, Any],
    *,
    project_root: Path,
    config_sha256: str,
    source_tree_sha256: str,
    registered_at: datetime,
) -> tuple[ImmutableExperimentSpec, TrialRecord, SealedForecastLedger]:
    prospective = config["prospective"]
    freeze_deadline = pd.Timestamp(prospective["freeze_deadline"]).tz_convert("UTC")
    first_signal = pd.Timestamp(prospective["first_signal_at"]).tz_convert("UTC")
    first_entry = pd.Timestamp(prospective["first_entry_at"]).tz_convert("UTC")
    registration = pd.Timestamp(registered_at).tz_convert("UTC")
    if not freeze_deadline < first_signal < first_entry:
        raise GovernanceViolation(
            "v2 timing must satisfy freeze_deadline < first_signal_at < first_entry_at"
        )
    specification = _v2_governance_payload(
        config,
        config_sha256=config_sha256,
        source_tree_sha256=source_tree_sha256,
    )
    governance_dir = project_root / prospective["governance_dir"]
    governance_dir.mkdir(parents=True, exist_ok=True)
    preregistration_path = governance_dir / "preregistration.json"
    trial_path = governance_dir / "trial.json"
    forecast_path = governance_dir / "sealed_forecasts.jsonl"

    def validate_existing(experiment: ImmutableExperimentSpec) -> ImmutableExperimentSpec:
        if experiment.experiment_id != prospective["strategy_version"]:
            raise GovernanceViolation("existing v2 seal has a different strategy_version")
        if pd.Timestamp(experiment.first_decision_at) != first_signal:
            raise GovernanceViolation("existing v2 seal has a different first_signal_at")
        if experiment.opened_period:
            raise GovernanceViolation("v2 prospective seal cannot be opened_period")
        if pd.Timestamp(experiment.registered_at) > freeze_deadline:
            raise GovernanceViolation("existing v2 seal missed its freeze deadline")
        experiment.verify(specification)
        return experiment

    if preregistration_path.exists():
        experiment = validate_existing(load_preregistration(preregistration_path))
    else:
        if registration > freeze_deadline:
            raise GovernanceViolation(
                "v2 freeze deadline has passed; create a new strategy_version"
            )
        experiment = ImmutableExperimentSpec.create(
            experiment_id=prospective["strategy_version"],
            specification=specification,
            registered_at=registered_at,
            first_decision_at=prospective["first_signal_at"],
            opened_period=False,
        )
        try:
            experiment = create_or_verify_preregistration(preregistration_path, experiment)
        except GovernanceViolation:
            experiment = validate_existing(load_preregistration(preregistration_path))

    trial = TrialRecord.create(
        trial_id=f"{prospective['strategy_version']}:prospective",
        experiment=experiment,
        purpose="diversified_momentum_and_residual_ml_prospective_test",
    )
    trial = create_or_verify_trial_record(trial_path, trial)
    forecasts = initialize_forecast_ledger(
        forecast_path,
        SealedForecastLedger.for_trial(trial),
    )
    return experiment, trial, forecasts


def _metrics_rows(
    returns_by_strategy: dict[str, pd.DataFrame],
    holdings_by_strategy: dict[str, pd.DataFrame],
    *,
    benchmark_name: str,
    seed: int,
) -> pd.DataFrame:
    benchmark = returns_by_strategy[benchmark_name]
    rows: list[dict[str, Any]] = []
    for name, returns in returns_by_strategy.items():
        summary = long_only_evaluation(returns, holdings_by_strategy[name], benchmark)
        summary.update(moving_block_bootstrap_sharpe(returns["net_return"], seed=seed))
        active = relative_return_frame(returns, benchmark)
        active_bootstrap = moving_block_bootstrap_sharpe(active["active_return"], seed=seed)
        summary.update({f"active_{key}": value for key, value in active_bootstrap.items()})
        rows.append({"strategy": name, "portfolio_type": "long_only", **summary})
    return pd.DataFrame(rows)


def _subperiod_metrics(
    returns_by_strategy: dict[str, pd.DataFrame],
    *,
    benchmark_name: str,
) -> pd.DataFrame:
    benchmark = returns_by_strategy[benchmark_name].copy()
    benchmark["exit_at"] = pd.to_datetime(benchmark["exit_at"], utc=True)
    rows: list[dict[str, Any]] = []
    periods = {
        "2019_2021": (pd.Timestamp("2019-01-01", tz="UTC"), pd.Timestamp("2022-01-01", tz="UTC")),
        "2022_2024": (pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC")),
        "2025_2026": (pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2027-01-01", tz="UTC")),
    }
    for name, frame in returns_by_strategy.items():
        candidate = frame.copy()
        candidate["exit_at"] = pd.to_datetime(candidate["exit_at"], utc=True)
        for period, (start, end) in periods.items():
            candidate_period = candidate.loc[candidate["exit_at"].ge(start) & candidate["exit_at"].lt(end)]
            benchmark_period = benchmark.loc[benchmark["exit_at"].ge(start) & benchmark["exit_at"].lt(end)]
            if candidate_period.empty or len(candidate_period) != len(benchmark_period):
                continue
            total = performance_summary(candidate_period.assign(borrow_cost=0.0))
            active = performance_summary(relative_return_frame(candidate_period, benchmark_period))
            rows.append(
                {
                    "strategy": name,
                    "period": period,
                    "months": len(candidate_period),
                    "annualized_return": total.get("annualized_return", np.nan),
                    "sharpe": total.get("sharpe", np.nan),
                    "max_drawdown": total.get("max_drawdown", np.nan),
                    "active_annualized_return": active.get(
                        "annualized_arithmetic_return", np.nan
                    ),
                    "active_information_ratio": active.get("sharpe", np.nan),
                }
            )
    return pd.DataFrame(rows)


def _artifact_inventory(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        if path.name == "manifest.json":
            continue
        rows.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def _run_cost_grid(
    builders: dict[str, Callable[[float], tuple[pd.DataFrame, pd.DataFrame]]],
    *,
    benchmark_builder: Callable[[float], tuple[pd.DataFrame, pd.DataFrame]],
    costs: list[float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cost in sorted(set(float(value) for value in costs)):
        benchmark, _ = benchmark_builder(cost)
        for name, builder in builders.items():
            strategy, holdings = builder(cost)
            metrics = long_only_evaluation(strategy, holdings, benchmark)
            rows.append(
                {
                    "strategy": name,
                    "cost_bps": cost,
                    "annualized_return": metrics["annualized_return"],
                    "sharpe": metrics["sharpe"],
                    "active_annualized_return": metrics["active_annualized_return"],
                    "active_information_ratio": metrics["active_information_ratio"],
                }
            )
    return pd.DataFrame(rows)


def _json_write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _validate_v2_protocol_config(config: dict[str, Any]) -> None:
    expected_roles = {
        "m1_baseline": "m1_momentum_top6",
        "m2_diversified_momentum": "m2_diversified_momentum_top10",
        "m3_sector_beta_constrained": "m3_risk_bounded_momentum_top10",
        "residual_ml_challenger": "residual_ml_overlay_top10",
    }
    if config["reporting"].get("benchmark") != "equal_weight_universe":
        raise ValueError("v2 benchmark must be the frozen equal_weight_universe")
    if config["reporting"].get("role_mapping") != expected_roles:
        raise ValueError("v2 reporting.role_mapping differs from the frozen role contract")
    if config["model"].get("target") != "beta_size_residual_return_rank":
        raise ValueError("v2 model target must be beta_size_residual_return_rank")
    overlay = config["candidates"]["residual_ml_overlay_top10"]
    if not np.isclose(
        float(overlay["momentum_weight"]) + float(overlay["residual_ml_weight"]),
        1.0,
    ):
        raise ValueError("v2 overlay weights must sum to one")
    if list(overlay.get("orthogonalize_ml_to", [])) != [
        "momentum_12_1",
        "beta_12m",
        "log_market_cap",
    ]:
        raise ValueError("v2 residual ML must use the frozen orthogonalization exposures")
    costs = config["costs"]
    if not bool(costs.get("charge_initial_entry")):
        raise ValueError("v2 accounting requires the initial entry trade to be charged")
    if not bool(costs.get("charge_terminal_liquidation")):
        raise ValueError("v2 accounting requires terminal liquidation to be charged")
    if bool(config["universe"].get("permit_current_sector_for_constraints")):
        raise ValueError("current sector metadata cannot be permitted for v2 constraints")


def _label_strategy(
    returns: pd.DataFrame,
    holdings: pd.DataFrame,
    *,
    strategy: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labelled_returns = returns.copy()
    labelled_holdings = holdings.copy()
    labelled_returns["borrow_cost"] = 0.0
    labelled_returns["strategy"] = strategy
    labelled_holdings["strategy"] = strategy
    return labelled_returns, labelled_holdings


def _build_equal_weight_strategy(
    panel: pd.DataFrame,
    *,
    strategy: str,
    score_column: str,
    holdings_count: int,
    cost_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns, holdings = build_ranked_long_only(
        panel,
        score_column=score_column,
        holdings_count=holdings_count,
        trading_cost_bps_one_way=cost_bps,
    )
    return _label_strategy(returns, holdings, strategy=strategy)


def _build_diversified_strategy(
    panel: pd.DataFrame,
    *,
    strategy: str,
    score_column: str,
    candidate_config: dict[str, Any],
    cost_bps: float,
    beta_exposure_limit: float | None,
    sector_column: str | None = None,
    sector_available_at_column: str | None = None,
    max_sector_weight: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = panel.copy()
    volatility_column = str(candidate_config["volatility_column"])
    floor = float(candidate_config["volatility_floor"])
    raw_volatility = pd.to_numeric(candidates[volatility_column], errors="coerce")
    if raw_volatility.isna().any() or (~np.isfinite(raw_volatility)).any():
        raise ValueError(f"{strategy} has missing or non-finite {volatility_column}")
    if floor <= 0 or not np.isfinite(floor):
        raise ValueError(f"{strategy} volatility_floor must be finite and positive")
    candidates["portfolio_volatility"] = raw_volatility.clip(lower=floor)

    beta = pd.to_numeric(candidates["beta_12m"], errors="coerce")
    if beta.isna().any() or (~np.isfinite(beta)).any():
        raise ValueError(f"{strategy} has missing or non-finite beta_12m")
    if beta_exposure_limit is None:
        # A convex long-only portfolio cannot exceed the largest constituent
        # absolute beta. This deterministic bound leaves beta unconstrained.
        resolved_beta_limit = float(beta.abs().max()) + 1.0
    else:
        resolved_beta_limit = float(beta_exposure_limit)

    returns, holdings = build_diversified_momentum_portfolio(
        candidates,
        score_column=score_column,
        volatility_column="portfolio_volatility",
        beta_column="beta_12m",
        feature_available_at_column="price_available_at",
        top_n=int(candidate_config["holdings"]),
        retention_rank=int(candidate_config["retention_rank"]),
        max_name_weight=float(candidate_config["max_name_weight"]),
        beta_exposure_limit=resolved_beta_limit,
        sector_column=sector_column,
        sector_available_at_column=sector_available_at_column,
        max_sector_weight=max_sector_weight,
        trading_cost_bps_one_way=cost_bps,
        charge_terminal_liquidation=True,
    )
    returns["configured_beta_exposure_limit"] = (
        np.nan if beta_exposure_limit is None else resolved_beta_limit
    )
    returns["sector_constraint_enforced"] = sector_column is not None
    holdings["configured_beta_exposure_limit"] = (
        np.nan if beta_exposure_limit is None else resolved_beta_limit
    )
    holdings["sector_constraint_enforced"] = sector_column is not None
    return _label_strategy(returns, holdings, strategy=strategy)


def _constraint_summary(
    returns_by_strategy: dict[str, pd.DataFrame],
    holdings_by_strategy: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for strategy, holdings in holdings_by_strategy.items():
        returns = returns_by_strategy[strategy]
        maximum_name = (
            holdings.groupby("signal_at")["weight"].max().max()
            if not holdings.empty
            else np.nan
        )
        maximum_beta = (
            pd.to_numeric(returns.get("beta_exposure"), errors="coerce").abs().max()
            if "beta_exposure" in returns
            else np.nan
        )
        maximum_sector = (
            pd.to_numeric(returns.get("largest_sector_weight"), errors="coerce").max()
            if "largest_sector_weight" in returns
            else np.nan
        )
        rows.append(
            {
                "strategy": strategy,
                "maximum_name_weight": float(maximum_name),
                "maximum_absolute_beta_exposure": (
                    float(maximum_beta) if pd.notna(maximum_beta) else np.nan
                ),
                "maximum_sector_weight": (
                    float(maximum_sector) if pd.notna(maximum_sector) else np.nan
                ),
                "sector_constraint_enforced": bool(
                    holdings.get(
                        "sector_constraint_enforced",
                        pd.Series(False, index=holdings.index),
                    ).fillna(False).all()
                ),
                "average_monthly_turnover": float(returns["turnover"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _incremental_metrics(
    strategy: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    frame = relative_return_frame(strategy, benchmark)
    summary = performance_summary(frame, return_column="active_return")
    summary.update(
        moving_block_bootstrap_sharpe(frame["active_return"], seed=seed)
    )
    return frame, summary


def _evaluate_v2_gates(
    metrics: pd.DataFrame,
    cost_sensitivity: pd.DataFrame,
    capacity_summary: pd.DataFrame,
    residual_incremental: dict[str, float],
    constraint_summary: pd.DataFrame,
    config: dict[str, Any],
    pit_audit: dict[str, int],
    *,
    production_data_checks: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Evaluate the frozen v2 research gates without granting production status."""

    gates = config["gates"]
    champion_name = "m2_diversified_momentum"
    m2 = metrics.set_index("strategy").loc[champion_name]
    m2_capacity = capacity_summary.set_index("strategy").loc[champion_name]
    constraints = constraint_summary.set_index("strategy")
    m3 = constraints.loc["m3_sector_beta_constrained"]

    def minimum(value: Any, threshold: Any, *, strict: bool = False) -> dict[str, Any]:
        actual = float(value)
        required = float(threshold)
        passed = bool(
            np.isfinite(actual)
            and (actual > required if strict else actual >= required)
        )
        return {"value": actual, "minimum": required, "passed": passed}

    def maximum(value: Any, threshold: Any) -> dict[str, Any]:
        actual = float(value)
        required = float(threshold)
        return {
            "value": actual,
            "maximum": required,
            "passed": bool(np.isfinite(actual) and actual <= required),
        }

    base_cost = float(config["costs"]["trading_cost_bps_one_way"])
    required_costs = [float(gates["positive_at_cost_bps"])]
    if bool(gates["positive_at_double_base_cost"]):
        required_costs.append(2.0 * base_cost)
    cost_checks: dict[str, dict[str, Any]] = {}
    for cost in sorted(set(required_costs)):
        selected = cost_sensitivity.loc[
            cost_sensitivity["strategy"].eq(champion_name)
            & np.isclose(cost_sensitivity["cost_bps"].astype(float), cost)
        ]
        actual = (
            float(selected.iloc[0]["active_annualized_return"])
            if len(selected) == 1
            else np.nan
        )
        cost_checks[f"positive_active_return_at_{cost:g}bps"] = minimum(
            actual, 0.0, strict=True
        )

    m2_checks: dict[str, dict[str, Any]] = {
        "zero_pit_violations": maximum(
            pit_audit["future_information_violations"], 0
        ),
        "active_annualized_return": minimum(
            m2["active_annualized_return"],
            gates["active_annualized_return_minimum"],
        ),
        "active_information_ratio": minimum(
            m2["active_information_ratio"],
            gates["active_information_ratio_minimum"],
        ),
        "active_newey_west_tstat_after_search": minimum(
            m2["active_newey_west_tstat"],
            gates["active_newey_west_tstat_minimum_after_search"],
        ),
        "relative_max_drawdown": minimum(
            m2["relative_max_drawdown"], gates["relative_max_drawdown_floor"]
        ),
        "single_name_gross_profit_share": maximum(
            m2["maximum_single_name_gross_profit_share"],
            gates["maximum_single_name_gross_profit_share"],
        ),
        "maximum_adv_participation": maximum(
            m2_capacity["maximum_adv_participation_observed"],
            gates["maximum_adv_participation"],
        ),
        **cost_checks,
    }
    m3_checks = {
        "maximum_absolute_beta_exposure": maximum(
            m3["maximum_absolute_beta_exposure"],
            config["candidates"]["m3_risk_bounded_momentum_top10"][
                "beta_maximum"
            ],
        )
    }
    residual_checks = {
        "incremental_annualized_return": minimum(
            residual_incremental["annualized_arithmetic_return"],
            gates["residual_ml_incremental_return_minimum"],
            strict=True,
        ),
        "incremental_bootstrap_sharpe_p05": minimum(
            residual_incremental["bootstrap_sharpe_p05"],
            gates["residual_ml_bootstrap_sharpe_p05_minimum"],
            strict=True,
        ),
        "incremental_newey_west_tstat_after_search": minimum(
            residual_incremental["newey_west_tstat"],
            gates["residual_ml_newey_west_tstat_minimum_after_search"],
        ),
    }
    stage_gates = {
        "m2_vs_equal_weight": m2_checks,
        "m3_beta_constraint": m3_checks,
        "residual_ml_increment": residual_checks,
    }
    all_research_checks = [
        item
        for stage in stage_gates.values()
        for item in stage.values()
    ]
    controls = {
        "survivorship_free": False,
        "permanent_identifiers": False,
        "delisting_returns": False,
        "execution_grade_quotes": False,
        "historical_borrow": False,
        "historical_industry_classification": False,
        "alfred_vintages_used": False,
    }
    if production_data_checks is not None:
        controls.update({key: bool(value) for key, value in production_data_checks.items()})
    return {
        "historical_evidence_status": "OPENED_DIAGNOSTIC",
        "holdout_opened": True,
        "champion": champion_name,
        "challenger": "residual_ml_challenger",
        "stage_gates": stage_gates,
        "m2_research_pass": all(item["passed"] for item in m2_checks.values()),
        "m3_beta_constraint_pass": all(item["passed"] for item in m3_checks.values()),
        "residual_ml_incremental_pass": all(
            item["passed"] for item in residual_checks.values()
        ),
        "research_status": (
            "PASS" if all(item["passed"] for item in all_research_checks) else "FAIL"
        ),
        "production_data_checks": controls,
        "production_status": "BLOCKED",
        "production_reason": (
            "opened current-ticker diagnostic; production requires PIT permanent IDs, "
            "delisting returns, historical classifications, ALFRED vintages, and "
            "execution-grade quotes"
        ),
    }


def run_champion_challenger_v2(
    config_path: str | Path,
    *,
    seal_governance: bool = True,
    output_dir_override: str | Path | None = None,
) -> dict[str, Path]:
    """Run the v2 opened diagnostic and, by default, seal its future protocol.

    ``seal_governance=False`` exists for disposable integration checks. Such a
    run is marked unregistered in its manifest and cannot be used as prospective
    evidence.
    """

    config_path = Path(config_path).expanduser().resolve()
    config = load_v2_config(config_path)
    _validate_v2_protocol_config(config)
    project_root = config_path.parent.parent
    configured_output = project_root / config["research"]["output_dir"]
    output_dir = (
        Path(output_dir_override).expanduser().resolve()
        if output_dir_override is not None
        else configured_output
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    universe_mode = str(config["universe"]["mode"])
    public_mode = universe_mode == "public_current_ticker_proxy"
    if public_mode:
        source_panel_path = project_root / config["research"]["source_panel"]
    elif universe_mode == str(config["universe"]["production_mode"]):
        source_panel_path = project_root / config["research"]["licensed_research_panel"]
    else:
        raise ValueError(f"unsupported v2 universe mode: {universe_mode}")
    daily_prices_path = project_root / config["research"]["source_daily_prices"]
    if not source_panel_path.exists():
        label = "licensed PIT" if not public_mode else "public research"
        raise FileNotFoundError(
            f"{label} source panel is missing: {source_panel_path}"
        )
    if not daily_prices_path.exists():
        raise FileNotFoundError(
            f"daily price source is missing: {daily_prices_path}; run public-research first"
        )

    config_sha256 = sha256_file(config_path)
    source_tree_sha256, source_file_count = sha256_directory(project_root / "src")
    registered_at = datetime.now(UTC)
    experiment: ImmutableExperimentSpec | None = None
    prospective_trial: TrialRecord | None = None
    forecast_ledger: SealedForecastLedger | None = None
    if seal_governance:
        experiment, prospective_trial, forecast_ledger = _write_v2_governance_records(
            config,
            project_root=project_root,
            config_sha256=config_sha256,
            source_tree_sha256=source_tree_sha256,
            registered_at=registered_at,
        )

    panel = pd.read_parquet(source_panel_path)
    panel["signal_date"] = pd.to_datetime(panel["signal_date"], utc=True, errors="raise")
    panel = add_decision_time(
        panel,
        decision_time_et=config["signal"]["decision_time_et"],
    )
    panel, exact_momentum, momentum_decisions = _replace_with_exact_momentum(
        panel,
        pd.read_parquet(daily_prices_path),
        benchmark_ticker=config["research"]["benchmark_ticker"],
        formation_sessions=int(config["signal"]["formation_sessions"]),
        skip_sessions=int(config["signal"]["skip_sessions"]),
    )
    feature_columns = _feature_sets(panel)["full"]
    pit_audit = assert_panel_is_point_in_time(
        panel,
        feature_columns=feature_columns,
    )
    panel = add_beta_size_residual_target(panel)

    first_prediction = pd.Timestamp(config["model"]["first_prediction"])
    if first_prediction.tzinfo is None:
        first_prediction = first_prediction.tz_localize("UTC")
    else:
        first_prediction = first_prediction.tz_convert("UTC")
    if public_mode:
        universe = select_public_proxy_universe(
            panel.loc[panel["signal_date"].ge(first_prediction)].copy(),
            top_n=int(config["universe"]["top_n"]),
            ranking_column=config["universe"]["ranking_column"],
            minimum_cross_section=int(config["universe"]["minimum_cross_section"]),
        )
    else:
        permanent_id = str(config["universe"]["permanent_id_column"])
        required = {permanent_id, "eligible", "decision_time", "signal_date"}
        missing = sorted(required.difference(panel.columns))
        if missing:
            raise ValueError(f"licensed PIT panel missing v2 universe columns: {missing}")
        if panel[permanent_id].isna().any():
            raise ValueError("licensed PIT panel contains missing permanent identifiers")
        universe = panel.loc[
            panel["signal_date"].ge(first_prediction)
            & panel["eligible"].fillna(False)
        ].copy()
        universe["ticker"] = universe[permanent_id].astype(str)
        universe["universe_rank"] = universe.groupby("signal_date").cumcount() + 1
        universe["universe_evidence_status"] = "licensed_pit"
        minimum = int(config["universe"]["production_training_universe_minimum"])
        counts = universe.groupby("signal_date")["ticker"].nunique()
        if counts.empty or int(counts.min()) < minimum:
            raise ValueError(
                "licensed PIT universe does not meet the configured cross-section minimum"
            )

    membership = universe[["signal_date", "ticker", "universe_rank"]]
    model_config = model_config_from_dict(config["model"])
    full_prediction, feature_importance, fit_audit = walk_forward_predict(
        panel,
        feature_columns,
        first_prediction=config["model"]["first_prediction"],
        half_life_years=float(config["model"]["recency_half_life_years"]),
        minimum_training_rows=int(config["model"]["minimum_training_rows"]),
        target_column="residual_target_return",
        config=model_config,
    )
    full_prediction = _filter_membership(full_prediction, membership)

    evaluation_entry_start = pd.Timestamp(
        config["research"]["evaluation_entry_at_or_after"]
    ).tz_convert("UTC")
    evaluation_exit_end = pd.Timestamp(
        config["research"]["evaluation_exit_at_or_before"]
    ).tz_convert("UTC")
    universe_eval = _evaluation_cross_sections(
        universe,
        entry_at_or_after=evaluation_entry_start,
        exit_at_or_before=evaluation_exit_end,
    )
    evaluation_keys = universe_eval[["signal_date", "ticker"]]
    prediction_eval = full_prediction.merge(
        evaluation_keys,
        on=["signal_date", "ticker"],
        how="inner",
        validate="one_to_one",
    )
    expected_rows = len(universe_eval)
    if len(prediction_eval) != expected_rows:
        raise RuntimeError(
            "residual-ML predictions do not cover every frozen evaluation-universe row"
        )

    overlay_config = config["candidates"]["residual_ml_overlay_top10"]
    scoring_only = prediction_eval[
        [
            "signal_date",
            "ticker",
            "momentum_12_1",
            "ensemble_score",
            "beta_12m",
            "log_market_cap",
        ]
    ].rename(columns={"ensemble_score": "residual_ml_score"})
    overlay_scores, overlay_diagnostics = build_momentum_residual_overlay(
        scoring_only,
        momentum_weight=float(overlay_config["momentum_weight"]),
        orthogonalize_ml=True,
    )
    overlay_columns = [
        "signal_date",
        "ticker",
        "residual_ml_adjusted_score",
        "momentum_rank_score",
        "residual_ml_rank_score",
        "momentum_contribution",
        "residual_ml_contribution",
        "overlay_score",
        "overlay_rank",
    ]
    prediction_eval = prediction_eval.merge(
        overlay_scores[overlay_columns],
        on=["signal_date", "ticker"],
        how="left",
        validate="one_to_one",
    )
    if prediction_eval["overlay_score"].isna().any():
        raise RuntimeError("overlay score coverage is incomplete")

    base_cost = float(config["costs"]["trading_cost_bps_one_way"])
    universe_count = int(config["universe"]["top_n"])
    m1_config = config["candidates"]["m1_momentum_top6"]
    m2_config = config["candidates"]["m2_diversified_momentum_top10"]
    m3_config = config["candidates"]["m3_risk_bounded_momentum_top10"]
    historical_sector_column = config["universe"].get("historical_sector_column")
    sector_available_column = config["universe"].get(
        "historical_sector_available_at_column"
    )
    if historical_sector_column:
        historical_sector_column = str(historical_sector_column)
        if historical_sector_column == str(
            config["universe"].get("public_sector_metadata_column")
        ):
            raise ValueError("current sector metadata cannot satisfy a PIT sector constraint")
        if not sector_available_column:
            raise ValueError(
                "historical sector constraint requires a configured availability column"
            )
        sector_available_column = str(sector_available_column)
        if sector_available_column not in universe_eval:
            raise ValueError(
                "historical sector constraint requires its own availability timestamp"
            )

    constituent_beta_maximum = float(m3_config["constituent_beta_maximum"])
    m3_eval = universe_eval.loc[
        pd.to_numeric(universe_eval["beta_12m"], errors="coerce").le(
            constituent_beta_maximum
        )
    ].copy()
    m3_counts = m3_eval.groupby("signal_at")["ticker"].nunique()
    missing_m3_dates = sorted(
        set(universe_eval["signal_at"].unique()).difference(m3_counts.index)
    )
    insufficient_m3 = m3_counts.lt(int(m3_config["holdings"]))
    if missing_m3_dates or insufficient_m3.any():
        raise ValueError(
            "M3 constituent beta filter leaves fewer than the configured holdings "
            "on at least one decision date"
        )

    def benchmark_builder(cost: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        return _build_equal_weight_strategy(
            universe_eval.assign(equal_weight_score=0.0),
            strategy="equal_weight_universe",
            score_column="equal_weight_score",
            holdings_count=universe_count,
            cost_bps=cost,
        )

    def m1_builder(cost: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        return _build_equal_weight_strategy(
            universe_eval,
            strategy="m1_baseline",
            score_column=str(m1_config["score_column"]),
            holdings_count=int(m1_config["holdings"]),
            cost_bps=cost,
        )

    def m2_builder(cost: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        return _build_diversified_strategy(
            universe_eval,
            strategy="m2_diversified_momentum",
            score_column=str(m2_config["score_column"]),
            candidate_config=m2_config,
            cost_bps=cost,
            beta_exposure_limit=None,
        )

    def m3_builder(cost: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        return _build_diversified_strategy(
            m3_eval,
            strategy="m3_sector_beta_constrained",
            score_column=str(m3_config["score_column"]),
            candidate_config=m3_config,
            cost_bps=cost,
            beta_exposure_limit=float(m3_config["beta_maximum"]),
            sector_column=(
                str(historical_sector_column) if historical_sector_column else None
            ),
            sector_available_at_column=sector_available_column,
            max_sector_weight=(
                float(m3_config["sector_maximum"])
                if historical_sector_column
                else None
            ),
        )

    def overlay_builder(cost: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        return _build_diversified_strategy(
            prediction_eval,
            strategy="residual_ml_challenger",
            score_column="overlay_score",
            candidate_config=overlay_config,
            cost_bps=cost,
            beta_exposure_limit=None,
        )

    builders: dict[str, Callable[[float], tuple[pd.DataFrame, pd.DataFrame]]] = {
        "m1_baseline": m1_builder,
        "m2_diversified_momentum": m2_builder,
        "m3_sector_beta_constrained": m3_builder,
        "residual_ml_challenger": overlay_builder,
    }
    returns_by_strategy: dict[str, pd.DataFrame] = {}
    holdings_by_strategy: dict[str, pd.DataFrame] = {}
    benchmark_returns, benchmark_holdings = benchmark_builder(base_cost)
    returns_by_strategy["equal_weight_universe"] = benchmark_returns
    holdings_by_strategy["equal_weight_universe"] = benchmark_holdings
    for strategy, builder in builders.items():
        strategy_returns, strategy_holdings = builder(base_cost)
        returns_by_strategy[strategy] = strategy_returns
        holdings_by_strategy[strategy] = strategy_holdings

    capacity_rows: list[dict[str, Any]] = []
    capacity_trade_frames: list[pd.DataFrame] = []
    for strategy, holdings in holdings_by_strategy.items():
        capacity, trades = capacity_diagnostics(
            holdings,
            portfolio_aum_usd=float(config["costs"]["portfolio_aum_usd"]),
            maximum_adv_participation=float(
                config["costs"]["maximum_adv_participation"]
            ),
        )
        capacity_rows.append({"strategy": strategy, **capacity})
        trades["strategy"] = strategy
        capacity_trade_frames.append(trades)
    capacity_summary = pd.DataFrame(capacity_rows)
    capacity_trades = pd.concat(capacity_trade_frames, ignore_index=True)

    seed = int(config["research"]["seed"])
    metrics = _metrics_rows(
        returns_by_strategy,
        holdings_by_strategy,
        benchmark_name="equal_weight_universe",
        seed=seed,
    ).merge(capacity_summary, on="strategy", validate="one_to_one")
    sensitivity_costs = list(config["costs"]["sensitivity_bps"])
    sensitivity_costs.extend(
        [
            float(config["gates"]["positive_at_cost_bps"]),
            2.0 * base_cost,
        ]
    )
    cost_sensitivity = _run_cost_grid(
        builders,
        benchmark_builder=benchmark_builder,
        costs=sensitivity_costs,
    )
    m2_vs_m1_frame, m2_vs_m1 = _incremental_metrics(
        returns_by_strategy["m2_diversified_momentum"],
        returns_by_strategy["m1_baseline"],
        seed=seed,
    )
    residual_incremental_frame, residual_incremental = _incremental_metrics(
        returns_by_strategy["residual_ml_challenger"],
        returns_by_strategy["m2_diversified_momentum"],
        seed=seed,
    )
    constraints = _constraint_summary(returns_by_strategy, holdings_by_strategy)
    production_controls = {
        "survivorship_free": not public_mode,
        "permanent_identifiers": (
            not public_mode
            and str(config["universe"]["permanent_id_column"]) in panel
        ),
        "delisting_returns": (not public_mode and "delisting_return" in panel),
        "execution_grade_quotes": (
            not public_mode and {"bid", "ask"}.issubset(panel.columns)
        ),
        "historical_borrow": (not public_mode and "borrow_fee_bps" in panel),
        "historical_industry_classification": bool(historical_sector_column),
        "alfred_vintages_used": (
            not public_mode and "macro_vintage_at" in panel
        ),
    }
    gates = _evaluate_v2_gates(
        metrics,
        cost_sensitivity,
        capacity_summary,
        residual_incremental,
        constraints,
        config,
        pit_audit,
        production_data_checks=production_controls,
    )
    subperiod_metrics = _subperiod_metrics(
        returns_by_strategy,
        benchmark_name="equal_weight_universe",
    )

    training_name_count = int(panel.loc[panel["eligible"].fillna(False), "ticker"].nunique())
    data_quality = {
        "evidence_status": config["research"]["evidence_status"],
        "active_metric_benchmark": config["reporting"]["benchmark"],
        "universe_mode": universe_mode,
        "universe_source": config["universe"]["source"],
        "survivorship_free": production_controls["survivorship_free"],
        "permanent_identifiers": production_controls["permanent_identifiers"],
        "delisting_returns": production_controls["delisting_returns"],
        "execution_data": (
            "Yahoo adjusted-open proxy; no TAQ/NBBO"
            if public_mode
            else "licensed source columns; certification still required"
        ),
        "execution_grade_quotes": production_controls["execution_grade_quotes"],
        "alfred_vintages_used": production_controls["alfred_vintages_used"],
        "historical_borrow": production_controls["historical_borrow"],
        "historical_industry_classification": production_controls[
            "historical_industry_classification"
        ],
        "current_sector_metadata_present_but_not_used": bool(
            config["universe"].get("public_sector_metadata_column") in panel
        ),
        "m3_sector_constraint_enforced": bool(historical_sector_column),
        "m3_requested_sector_maximum": float(m3_config["sector_maximum"]),
        "m3_beta_constraint_enforced": True,
        "m3_beta_maximum": float(m3_config["beta_maximum"]),
        "m3_constituent_beta_filter_enforced": True,
        "m3_constituent_beta_maximum": constituent_beta_maximum,
        "m3_minimum_filtered_cross_section": int(m3_counts.min()),
        "m3_unfiltered_beta1_trial_status": "INFEASIBLE_DISCLOSED_OPENED_TRIAL",
        "m2_vs_m1_incremental": {
            "annualized_arithmetic_return": m2_vs_m1[
                "annualized_arithmetic_return"
            ],
            "newey_west_tstat": m2_vs_m1["newey_west_tstat"],
            "bootstrap_sharpe_p05": m2_vs_m1["bootstrap_sharpe_p05"],
        },
        "residual_ml_vs_m2_incremental": {
            "annualized_arithmetic_return": residual_incremental[
                "annualized_arithmetic_return"
            ],
            "newey_west_tstat": residual_incremental["newey_west_tstat"],
            "bootstrap_sharpe_p05": residual_incremental[
                "bootstrap_sharpe_p05"
            ],
        },
        "pit_audit": pit_audit,
        "source_rows": len(panel),
        "source_names": int(panel["ticker"].nunique()),
        "training_eligible_names": training_name_count,
        "production_training_universe_minimum": int(
            config["universe"]["production_training_universe_minimum"]
        ),
        "selected_names": int(universe["ticker"].nunique()),
        "selected_months": int(universe["signal_date"].nunique()),
        "evaluable_months": int(universe_eval["signal_date"].nunique()),
        "evaluation_entry_at_or_after": str(evaluation_entry_start),
        "evaluation_exit_at_or_before": str(evaluation_exit_end),
        "last_evaluable_signal": str(universe_eval["signal_date"].max()),
        "governance_sealed": bool(seal_governance),
    }

    combined_returns = pd.concat(
        returns_by_strategy.values(), ignore_index=True, sort=False
    )
    combined_holdings = pd.concat(
        holdings_by_strategy.values(), ignore_index=True, sort=False
    )
    combined_returns.to_csv(output_dir / "monthly_returns.csv", index=False)
    combined_holdings.to_parquet(
        output_dir / "portfolio_holdings.parquet", index=False
    )
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    subperiod_metrics.to_csv(output_dir / "subperiod_metrics.csv", index=False)
    cost_sensitivity.to_csv(output_dir / "cost_sensitivity.csv", index=False)
    capacity_summary.to_csv(output_dir / "capacity_summary.csv", index=False)
    capacity_trades.to_csv(output_dir / "capacity_trades.csv", index=False)
    constraints.to_csv(output_dir / "constraint_diagnostics.csv", index=False)
    feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)
    fit_audit.to_csv(output_dir / "fit_audit.csv", index=False)
    overlay_diagnostics.to_csv(output_dir / "overlay_diagnostics.csv", index=False)
    universe.to_parquet(output_dir / "universe_snapshots.parquet", index=False)
    exact_momentum.to_parquet(
        output_dir / "exact_momentum_signals.parquet", index=False
    )
    momentum_decisions.to_csv(
        output_dir / "momentum_decision_calendar.csv", index=False
    )
    score_columns = [
        "signal_date",
        "ticker",
        "ridge_score",
        "extra_trees_score",
        "histogram_gbdt_score",
        "ensemble_score",
        "residual_ml_adjusted_score",
        "momentum_rank_score",
        "residual_ml_rank_score",
        "overlay_score",
        "overlay_rank",
    ]
    prediction_eval[score_columns].to_parquet(
        output_dir / "model_scores.parquet", index=False
    )
    m2_vs_m1_frame.to_csv(output_dir / "m2_vs_m1_returns.csv", index=False)
    residual_incremental_frame.to_csv(
        output_dir / "residual_ml_incremental_returns.csv", index=False
    )
    _json_write(output_dir / "m2_vs_m1_metrics.json", m2_vs_m1)
    _json_write(
        output_dir / "residual_ml_incremental_metrics.json", residual_incremental
    )
    _json_write(output_dir / "gates.json", gates)
    _json_write(output_dir / "data_quality.json", data_quality)
    _json_write(output_dir / "pit_audit.json", pit_audit)
    (output_dir / "config_snapshot.yml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    opened_trials = ExperimentTrialLedger()
    candidate_lookup: dict[str, Any] = {
        "equal_weight_universe": {
            "holdings": universe_count,
            "weighting": "equal",
        },
        "m1_momentum_top6": m1_config,
        "m2_diversified_momentum_top10": m2_config,
        "m3_unfiltered_top10_beta1_infeasible": {
            "base_candidate": "m2_diversified_momentum_top10",
            "beta_maximum": 1.0,
            "constituent_beta_filter": None,
            "opened_result": "infeasible_on_at_least_one_frozen_cross_section",
            "selection_basis": "feasibility_only_not_return_performance",
        },
        "m3_risk_bounded_momentum_top10": m3_config,
        "residual_ml_overlay_top10": overlay_config,
    }
    for index, name in enumerate(config["opened_trials"]):
        opened_spec = ImmutableExperimentSpec.create(
            experiment_id=f"opened:v2:{name}",
            specification={
                "candidate": name,
                "candidate_specification": candidate_lookup[name],
                "config_sha256": config_sha256,
                "source_tree_sha256": source_tree_sha256,
            },
            registered_at=registered_at,
            first_decision_at=str(universe_eval["signal_at"].min()),
            opened_period=True,
        )
        opened_trials = opened_trials.append(
            TrialRecord.create(
                trial_id=f"opened-v2-{index + 1:02d}-{name}",
                experiment=opened_spec,
                purpose="opened_historical_diagnostic",
            )
        )
    _json_write(output_dir / "experiment_trial_ledger.json", opened_trials.to_dict())

    from alpha_factor_engine.v2_reporting import (
        V2_ROLE_ORDER,
        create_v2_opened_diagnostic_report,
    )

    reporting_returns = {
        role: returns_by_strategy[role] for role in V2_ROLE_ORDER
    }
    reporting_holdings = {
        role: holdings_by_strategy[role] for role in V2_ROLE_ORDER
    }
    reporting_metrics = metrics.loc[
        metrics["strategy"].isin(V2_ROLE_ORDER)
    ].copy()
    reporting_metrics["period"] = "opened"
    reporting_capacity = capacity_summary.loc[
        capacity_summary["strategy"].isin(V2_ROLE_ORDER)
    ].copy()
    report_output = create_v2_opened_diagnostic_report(
        output_dir,
        returns=reporting_returns,
        holdings=reporting_holdings,
        metrics=reporting_metrics,
        cost_sensitivity=cost_sensitivity,
        capacity_summary=reporting_capacity,
        predeclared_gates=gates,
        data_quality=data_quality,
        assumptions={
            "annualization_factor": int(config["reporting"]["annualization_factor"]),
            "opened_start": config["research"]["evaluation_entry_at_or_after"],
            "trading_cost_bps_one_way": base_cost,
            "annual_short_borrow_bps": 0.0,
            "portfolio_aum_usd": float(config["costs"]["portfolio_aum_usd"]),
            "maximum_adv_participation": float(
                config["costs"]["maximum_adv_participation"]
            ),
            "charge_initial_entry": bool(config["costs"]["charge_initial_entry"]),
            "charge_terminal_liquidation": bool(
                config["costs"]["charge_terminal_liquidation"]
            ),
        },
    )
    report = Path(report_output["report"])
    charts = [Path(path) for path in report_output["charts"]]
    _json_write(output_dir / "report_decision.json", report_output)

    if experiment is not None and prospective_trial is not None and forecast_ledger is not None:
        prospective_manifest: dict[str, Any] = {
            "registration_status": "SEALED",
            "experiment_id": experiment.experiment_id,
            "trial_id": prospective_trial.trial_id,
            "spec_sha256": experiment.spec_sha256,
            "forecast_ledger_sha256": forecast_ledger.ledger_sha256,
            "registered_at": experiment.registered_at,
            "first_decision_at": experiment.first_decision_at,
            "opened_period": experiment.opened_period,
            "sealed_decision_count": forecast_ledger.decision_count,
        }
    else:
        prospective_manifest = {
            "registration_status": "SKIPPED_DISPOSABLE_RUN",
            "experiment_id": config["prospective"]["strategy_version"],
            "spec_sha256": stable_json_hash(
                _v2_governance_payload(
                    config,
                    config_sha256=config_sha256,
                    source_tree_sha256=source_tree_sha256,
                )
            ),
            "opened_period": False,
            "sealed_decision_count": 0,
        }
    prospective_manifest.update(
        {
            "freeze_deadline": config["prospective"]["freeze_deadline"],
            "first_entry_at": config["prospective"]["first_entry_at"],
            "minimum_months": config["prospective"]["minimum_months"],
        }
    )
    manifest = {
        "run_id": (
            f"{registered_at.strftime('%Y%m%dT%H%M%SZ')}_"
            f"{stable_json_hash(config)[:8]}"
        ),
        "created_at": registered_at.isoformat(),
        "historical_evidence_status": "OPENED_DIAGNOSTIC",
        "production_status": "BLOCKED",
        "config_sha256": config_sha256,
        "source_panel": str(source_panel_path.relative_to(project_root)),
        "source_panel_sha256": sha256_file(source_panel_path),
        "daily_prices": str(daily_prices_path.relative_to(project_root)),
        "daily_prices_sha256": sha256_file(daily_prices_path),
        "source_tree_sha256": source_tree_sha256,
        "source_file_count": source_file_count,
        "resolved_model_config": asdict(model_config),
        "feature_columns": feature_columns,
        "fixed_role_mapping": {
            "m1_baseline": "equal-weight top-6 exact 12-1 momentum",
            "m2_diversified_momentum": (
                "top-10 exact 12-1 momentum, inverse-volatility, rank buffer"
            ),
            "m3_sector_beta_constrained": (
                "M2 plus beta bound; sector bound only with PIT classification"
            ),
            "residual_ml_challenger": (
                "80% momentum plus 20% orthogonalized residual-ML rank overlay"
            ),
        },
        "report_decision": {
            "diagnostic_status": report_output["diagnostic_status"],
            "production_status": report_output["production_status"],
            "failed_gates": report_output["failed_gates"],
        },
        "prospective": prospective_manifest,
        "charts": [path.relative_to(output_dir).as_posix() for path in charts],
        "artifacts": _artifact_inventory(output_dir),
    }
    manifest_path = output_dir / "manifest.json"
    _json_write(manifest_path, manifest)
    return {
        "report": report,
        "manifest": manifest_path,
        "output_dir": output_dir,
        "governance_dir": project_root / config["prospective"]["governance_dir"],
    }


__all__ = [
    "load_v2_config",
    "run_champion_challenger_v2",
]
