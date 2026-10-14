"""Opened diagnostics plus a separately frozen champion-challenger protocol."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from alpha_factor_engine.contracts import sha256_directory, sha256_file, stable_json_hash
from alpha_factor_engine.features.fundamentals import FUNDAMENTAL_FEATURES
from alpha_factor_engine.research.champion_challenger import (
    add_beta_size_residual_target,
    add_decision_time,
    assert_panel_is_point_in_time,
    build_ranked_long_only,
    capacity_diagnostics,
    evaluable_cross_sections,
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
from alpha_factor_engine.research.momentum import compute_12_1_momentum
from alpha_factor_engine.research.models import (
    FEATURE_GROUPS,
    model_config_from_dict,
    walk_forward_predict,
)
from alpha_factor_engine.research.portfolio import build_long_short_portfolio


def load_champion_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def _feature_sets(panel: pd.DataFrame) -> dict[str, list[str]]:
    full = [column for values in FEATURE_GROUPS.values() for column in values if column in panel]
    return {
        "momentum_only": ["momentum_12_1"],
        "fundamentals_only": [column for column in FUNDAMENTAL_FEATURES if column in panel],
        "without_momentum": [
            column for column in full if column not in {"momentum_12_1", "momentum_6_1"}
        ],
        "full": full,
    }


def _replace_with_exact_momentum(
    panel: pd.DataFrame,
    daily_prices: pd.DataFrame,
    *,
    benchmark_ticker: str,
    formation_sessions: int,
    skip_sessions: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Recompute 12-1 on canonical benchmark sessions and exact endpoints."""

    if formation_sessions <= skip_sessions or skip_sessions <= 0:
        raise ValueError("formation_sessions must exceed a positive skip_sessions")
    required_daily = {"ticker", "date", "adj_close", "available_at"}
    missing_daily = required_daily.difference(daily_prices.columns)
    if missing_daily:
        raise ValueError(f"daily price source missing columns: {sorted(missing_daily)}")
    required_panel = {"ticker", "date", "signal_at", "eligible"}
    missing_panel = required_panel.difference(panel.columns)
    if missing_panel:
        raise ValueError(f"monthly panel missing exact-momentum columns: {sorted(missing_panel)}")

    daily = daily_prices.copy()
    daily["ticker"] = daily["ticker"].astype(str)
    daily["date"] = pd.to_datetime(daily["date"], errors="raise").dt.normalize()
    daily["available_at"] = pd.to_datetime(daily["available_at"], utc=True, errors="raise")
    daily["adj_close"] = pd.to_numeric(daily["adj_close"], errors="coerce")
    duplicate = daily.duplicated(["ticker", "date"], keep=False)
    if duplicate.any():
        raise ValueError("daily price source contains duplicate ticker/date rows")
    calendar = (
        daily.loc[daily["ticker"].eq(benchmark_ticker), "date"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    if not calendar:
        raise ValueError(f"benchmark {benchmark_ticker} is absent from the daily trading calendar")
    calendar_index = {date: index for index, date in enumerate(calendar)}
    first_price_date = daily.groupby("ticker", sort=False)["date"].min().to_dict()
    monthly = panel.copy()
    monthly["date"] = pd.to_datetime(monthly["date"], errors="raise").dt.normalize()

    signal_frames: list[pd.DataFrame] = []
    decision_rows: list[dict[str, Any]] = []
    for formation_date, group in monthly.groupby("date", sort=True):
        position = calendar_index.get(formation_date)
        if position is None:
            raise ValueError(f"formation date {formation_date.date()} is absent from benchmark calendar")
        if position < formation_sessions:
            continue
        names = sorted(
            group.loc[group["eligible"].fillna(False), "ticker"].astype(str).unique().tolist()
        )
        if not names:
            continue
        signal_values = pd.to_datetime(group["signal_at"], utc=True).unique()
        if len(signal_values) != 1:
            raise ValueError(f"formation date {formation_date.date()} has inconsistent signal_at")
        decision = {
            "signal_at": pd.Timestamp(signal_values[0]),
            "formation_date": formation_date,
            "twelve_month_date": calendar[position - formation_sessions],
            "one_month_date": calendar[position - skip_sessions],
            "next_session_date": calendar[position + 1] if position + 1 < len(calendar) else pd.NaT,
        }
        observed_entries = pd.Series(dtype="datetime64[ns, UTC]")
        if "entry_at" in group:
            candidate_entries = pd.to_datetime(group["entry_at"], utc=True, errors="coerce")
            if candidate_entries.isna().any() and candidate_entries.notna().any():
                raise ValueError(
                    f"formation date {formation_date.date()} has partial entry_at coverage"
                )
            observed_entries = candidate_entries.dropna()
            if observed_entries.nunique() > 1:
                raise ValueError(
                    f"formation date {formation_date.date()} has inconsistent entry_at"
                )
        if not observed_entries.empty and pd.notna(decision["next_session_date"]):
            local_entry_dates = observed_entries.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
            if not local_entry_dates.eq(decision["next_session_date"]).all():
                raise ValueError(
                    f"{formation_date.date()} entry is not the next canonical benchmark session"
                )
        endpoint_dates = {
            decision["formation_date"],
            decision["twelve_month_date"],
            decision["one_month_date"],
        }
        price_subset = daily.loc[
            daily["ticker"].isin(names) & daily["date"].isin(endpoint_dates),
            ["ticker", "date", "adj_close", "available_at"],
        ].rename(columns={"adj_close": "total_return_index"})
        endpoint_counts = price_subset.groupby("ticker")["date"].nunique()
        complete_names = sorted(
            name for name in names if int(endpoint_counts.get(name, 0)) == len(endpoint_dates)
        )
        incomplete_established = sorted(
            name
            for name in names
            if name not in complete_names
            and name in first_price_date
            and first_price_date[name] <= decision["twelve_month_date"]
        )
        if incomplete_established:
            raise ValueError(
                "established securities are missing exact momentum endpoints at "
                f"{formation_date.date()}: {incomplete_established[:10]}"
            )
        if not complete_names:
            continue
        signals = compute_12_1_momentum(
            price_subset,
            pd.DataFrame([decision]),
            universe=complete_names,
        )
        signal_frames.append(signals)
        decision["generic_eligible_names"] = len(names)
        decision["momentum_history_eligible_names"] = len(complete_names)
        decision["insufficient_history_names"] = len(names) - len(complete_names)
        decision_rows.append(decision)
    if not signal_frames:
        raise RuntimeError("exact 12-1 momentum produced no eligible decision dates")
    exact = pd.concat(signal_frames, ignore_index=True)
    decisions = pd.DataFrame(decision_rows).sort_values("signal_at").reset_index(drop=True)
    output = monthly.merge(
        exact[["signal_at", "ticker", "momentum_12_1"]].rename(
            columns={"momentum_12_1": "exact_momentum_12_1"}
        ),
        on=["signal_at", "ticker"],
        how="left",
        validate="one_to_one",
    )
    output["generic_eligible_before_momentum_history"] = output["eligible"].fillna(False)
    output["momentum_history_eligible"] = output["exact_momentum_12_1"].notna()
    output["eligible"] = (
        output["generic_eligible_before_momentum_history"]
        & output["momentum_history_eligible"]
    )
    output["legacy_approx_momentum_12_1"] = output["momentum_12_1"]
    exact_available = output["momentum_history_eligible"]
    output.loc[exact_available, "momentum_12_1"] = output.loc[
        exact_available, "exact_momentum_12_1"
    ]
    return output, exact, decisions


def _filter_membership(
    predictions: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    keys = membership[["signal_date", "ticker", "universe_rank"]]
    output = predictions.merge(keys, on=["signal_date", "ticker"], how="inner", validate="one_to_one")
    return output.sort_values(["signal_date", "universe_rank"]).reset_index(drop=True)


def _evaluation_cross_sections(
    universe: pd.DataFrame,
    *,
    entry_at_or_after: pd.Timestamp,
    exit_at_or_before: pd.Timestamp,
) -> pd.DataFrame:
    """Select whole executable cross-sections; never shrink one row at a time."""

    selected_dates: list[pd.Timestamp] = []
    for signal_date, group in universe.groupby("signal_date", sort=True):
        for column in ["entry_at", "exit_at"]:
            timestamps = pd.to_datetime(group[column], utc=True, errors="coerce")
            if timestamps.isna().any() and timestamps.notna().any():
                raise RuntimeError(f"partial {column} coverage at {signal_date}")
            if timestamps.notna().any() and timestamps.nunique() != 1:
                raise RuntimeError(f"inconsistent {column} timestamps at {signal_date}")
        entries = pd.to_datetime(group["entry_at"], utc=True, errors="coerce").dropna()
        exits = pd.to_datetime(group["exit_at"], utc=True, errors="coerce").dropna()
        if entries.empty or exits.empty:
            continue
        if entries.iloc[0] >= entry_at_or_after and exits.iloc[0] <= exit_at_or_before:
            selected_dates.append(signal_date)
    if not selected_dates:
        raise RuntimeError("evaluation boundaries selected no complete cross-sections")
    selected = universe.loc[universe["signal_date"].isin(selected_dates)].copy()
    return evaluable_cross_sections(selected)


def _attach_long_exposures(
    returns: pd.DataFrame,
    holdings: pd.DataFrame,
) -> pd.DataFrame:
    exposure_rows: list[dict[str, float | pd.Timestamp]] = []
    for signal_at, group in holdings.groupby("signal_at", sort=True):
        beta = pd.to_numeric(group.get("beta_12m"), errors="coerce").fillna(0.0)
        size = pd.to_numeric(group.get("log_market_cap"), errors="coerce").fillna(0.0)
        exposure_rows.append(
            {
                "signal_at": signal_at,
                "beta_exposure": float((group["weight"] * beta).sum()),
                "size_exposure": float((group["weight"] * size).sum()),
            }
        )
    output = returns.merge(pd.DataFrame(exposure_rows), on="signal_at", validate="one_to_one")
    output["borrow_cost"] = 0.0
    return output


def _run_long_strategy(
    panel: pd.DataFrame,
    *,
    name: str,
    score_column: str,
    holdings_count: int,
    trading_cost_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns, holdings = build_ranked_long_only(
        panel,
        score_column=score_column,
        holdings_count=holdings_count,
        trading_cost_bps_one_way=trading_cost_bps,
    )
    returns = _attach_long_exposures(returns, holdings)
    returns["strategy"] = name
    holdings["strategy"] = name
    return returns, holdings


def _metrics_rows(
    returns_by_strategy: dict[str, pd.DataFrame],
    holdings_by_strategy: dict[str, pd.DataFrame],
    *,
    benchmark_name: str,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    benchmark = returns_by_strategy[benchmark_name]
    for name, returns in returns_by_strategy.items():
        if name == "ml_full_neutral_long_short":
            summary = performance_summary(returns)
            summary.update(moving_block_bootstrap_sharpe(returns["net_return"], seed=seed))
            summary["maximum_absolute_beta_exposure"] = float(
                pd.to_numeric(returns["beta_exposure"], errors="coerce").abs().max()
            )
            summary["average_absolute_beta_exposure"] = float(
                pd.to_numeric(returns["beta_exposure"], errors="coerce").abs().mean()
            )
            rows.append(
                {
                    "strategy": name,
                    "portfolio_type": "beta_size_bounded_long_short",
                    **summary,
                    "active_annualized_return": np.nan,
                    "active_information_ratio": np.nan,
                    "maximum_single_name_gross_profit_share": np.nan,
                }
            )
            continue
        summary = long_only_evaluation(returns, holdings_by_strategy[name], benchmark)
        summary.update(moving_block_bootstrap_sharpe(returns["net_return"], seed=seed))
        relative = relative_return_frame(returns, benchmark)
        active_bootstrap = moving_block_bootstrap_sharpe(relative["net_return"], seed=seed)
        summary.update({f"active_{key}": value for key, value in active_bootstrap.items()})
        rows.append(
            {
                "strategy": name,
                "portfolio_type": "long_only",
                **summary,
            }
        )
    return pd.DataFrame(rows)


def _cost_sensitivity(
    universe: pd.DataFrame,
    full_predictions: pd.DataFrame,
    *,
    holdings_count: int,
    universe_count: int,
    costs: list[float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cost in sorted(set(float(item) for item in costs)):
        equal = universe.assign(equal_weight_score=0.0)
        benchmark, _ = _run_long_strategy(
            equal,
            name="equal_weight_universe",
            score_column="equal_weight_score",
            holdings_count=universe_count,
            trading_cost_bps=cost,
        )
        for name, frame, score in [
            ("momentum_12_1_top6", universe, "momentum_12_1"),
            ("ml_full_top6", full_predictions, "ensemble_score"),
        ]:
            strategy, holdings = _run_long_strategy(
                frame,
                name=name,
                score_column=score,
                holdings_count=holdings_count,
                trading_cost_bps=cost,
            )
            result = long_only_evaluation(strategy, holdings, benchmark)
            rows.append(
                {
                    "strategy": name,
                    "cost_bps": cost,
                    "annualized_return": result["annualized_return"],
                    "sharpe": result["sharpe"],
                    "active_annualized_return": result["active_annualized_return"],
                    "active_information_ratio": result["active_information_ratio"],
                }
            )
    return pd.DataFrame(rows)


def _evaluate_gates(
    metrics: pd.DataFrame,
    cost_sensitivity: pd.DataFrame,
    capacity_summary: pd.DataFrame,
    incremental: dict[str, float],
    config: dict[str, Any],
    pit_audit: dict[str, int],
) -> dict[str, Any]:
    gates = config["gates"]
    momentum = metrics.set_index("strategy").loc["momentum_12_1_top6"]
    momentum_capacity = capacity_summary.set_index("strategy").loc[
        "momentum_12_1_top6"
    ]
    base_cost = float(config["costs"]["trading_cost_bps_one_way"])
    required_costs = [float(gates["positive_at_cost_bps"]), 2 * base_cost]
    cost_checks: dict[str, dict[str, Any]] = {}
    for cost in required_costs:
        values = cost_sensitivity.loc[
            cost_sensitivity["strategy"].eq("momentum_12_1_top6")
            & cost_sensitivity["cost_bps"].eq(cost)
        ]
        actual = float(values.iloc[0]["active_annualized_return"]) if not values.empty else np.nan
        cost_checks[f"positive_active_return_at_{cost:g}bps"] = {
            "actual": actual,
            "requirement": 0.0,
            "passed": bool(np.isfinite(actual) and actual > 0),
        }

    research_checks = {
        "zero_pit_violations": {
            "actual": pit_audit["future_information_violations"],
            "requirement": 0,
            "passed": pit_audit["future_information_violations"] == 0,
        },
        "active_annualized_return": {
            "actual": float(momentum["active_annualized_return"]),
            "requirement": float(gates["active_annualized_return_minimum"]),
            "passed": float(momentum["active_annualized_return"])
            >= float(gates["active_annualized_return_minimum"]),
        },
        "active_information_ratio": {
            "actual": float(momentum["active_information_ratio"]),
            "requirement": float(gates["active_information_ratio_minimum"]),
            "passed": float(momentum["active_information_ratio"])
            >= float(gates["active_information_ratio_minimum"]),
        },
        "active_newey_west_tstat_after_search": {
            "actual": float(momentum["active_newey_west_tstat"]),
            "requirement": float(gates["large_search_tstat_minimum"]),
            "passed": float(momentum["active_newey_west_tstat"])
            >= float(gates["large_search_tstat_minimum"]),
        },
        "relative_max_drawdown": {
            "actual": float(momentum["relative_max_drawdown"]),
            "requirement": float(gates["relative_max_drawdown_floor"]),
            "passed": float(momentum["relative_max_drawdown"])
            >= float(gates["relative_max_drawdown_floor"]),
        },
        "single_name_gross_profit_share": {
            "actual": float(momentum["maximum_single_name_gross_profit_share"]),
            "requirement": float(gates["maximum_single_name_gross_profit_share"]),
            "passed": bool(
                np.isfinite(momentum["maximum_single_name_gross_profit_share"])
                and float(momentum["maximum_single_name_gross_profit_share"])
                <= float(gates["maximum_single_name_gross_profit_share"])
            ),
        },
        "maximum_adv_participation": {
            "actual": float(momentum_capacity["maximum_adv_participation_observed"]),
            "requirement": float(config["costs"]["maximum_adv_participation"]),
            "passed": bool(momentum_capacity["capacity_passed"]),
        },
        **cost_checks,
    }
    ml_checks = {
        "incremental_annualized_return": {
            "actual": incremental["annualized_arithmetic_return"],
            "requirement": 0.0,
            "passed": incremental["annualized_arithmetic_return"] > 0,
        },
        "incremental_bootstrap_sharpe_p05": {
            "actual": incremental["bootstrap_sharpe_p05"],
            "requirement": float(gates["ml_incremental_bootstrap_p05_minimum"]),
            "passed": incremental["bootstrap_sharpe_p05"]
            > float(gates["ml_incremental_bootstrap_p05_minimum"]),
        },
        "incremental_newey_west_tstat_after_search": {
            "actual": incremental["newey_west_tstat"],
            "requirement": float(gates["large_search_tstat_minimum"]),
            "passed": incremental["newey_west_tstat"]
            >= float(gates["large_search_tstat_minimum"]),
        },
    }
    neutral = metrics.set_index("strategy").loc["ml_full_neutral_long_short"]
    neutral_checks = {
        "sharpe": {
            "actual": float(neutral["sharpe"]),
            "requirement": float(gates["neutral_sharpe_minimum"]),
            "passed": float(neutral["sharpe"])
            >= float(gates["neutral_sharpe_minimum"]),
        },
        "maximum_absolute_beta_exposure": {
            "actual": float(neutral["maximum_absolute_beta_exposure"]),
            "requirement": float(gates["neutral_absolute_beta_maximum"]),
            "passed": float(neutral["maximum_absolute_beta_exposure"])
            <= float(gates["neutral_absolute_beta_maximum"]),
        },
        "max_drawdown": {
            "actual": float(neutral["max_drawdown"]),
            "requirement": float(gates["neutral_max_drawdown_floor"]),
            "passed": float(neutral["max_drawdown"])
            >= float(gates["neutral_max_drawdown_floor"]),
        },
    }
    production_data_checks = {
        "survivorship_free": False,
        "permanent_identifiers": False,
        "delisting_returns": False,
        "historical_industry_classification": False,
        "alfred_vintages_used": False,
        "execution_grade_quotes": False,
        "historical_borrow": False,
    }
    research_pass = all(item["passed"] for item in research_checks.values())
    reporting_checks: dict[str, dict[str, Any]] = {}
    for name, item in research_checks.items():
        relation_is_maximum = name in {
            "zero_pit_violations",
            "single_name_gross_profit_share",
            "maximum_adv_participation",
        }
        reporting_checks[name] = {
            "value": item["actual"],
            ("maximum" if relation_is_maximum else "minimum"): item["requirement"],
        }
    return {
        "historical_evidence_status": "OPENED_DIAGNOSTIC",
        "champion": "momentum_12_1_top6",
        "research_status": "PASS" if research_pass else "FAIL",
        "research_checks": reporting_checks,
        "momentum_research_checks": research_checks,
        "momentum_research_pass": research_pass,
        "ml_incremental_checks": ml_checks,
        "ml_incremental_pass": all(item["passed"] for item in ml_checks.values()),
        "neutral_research_checks": neutral_checks,
        "neutral_research_pass": all(item["passed"] for item in neutral_checks.values()),
        "production_data_checks": production_data_checks,
        "production_status": "BLOCKED",
        "holdout_opened": True,
        "production_reason": "current-ticker public proxy lacks CRSP/TAQ/ALFRED/borrow evidence",
    }


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


def _governance_payload(
    config: dict[str, Any],
    *,
    config_sha256: str,
    source_tree_sha256: str,
) -> dict[str, Any]:
    return {
        "strategy_version": config["prospective"]["strategy_version"],
        "champion": config["prospective"]["champion"],
        "challenger": config["prospective"]["challenger"],
        "universe": config["universe"],
        "signal": config["signal"],
        "costs": config["costs"],
        "model": config["model"],
        "neutral_portfolio": config["neutral_portfolio"],
        "gates": config["gates"],
        "change_policy": config["prospective"]["change_policy"],
        "freeze_deadline": config["prospective"]["freeze_deadline"],
        "first_signal_at": config["prospective"]["first_signal_at"],
        "first_entry_at": config["prospective"]["first_entry_at"],
        "minimum_months": config["prospective"]["minimum_months"],
        "config_sha256": config_sha256,
        "source_tree_sha256": source_tree_sha256,
        "production_data_requirement": "crsp_pit_permanent_ids_delistings_plus_execution_grade_fills",
    }


def _write_governance_records(
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
        raise RuntimeError(
            "prospective timing must satisfy freeze_deadline < first_signal_at < first_entry_at"
        )
    specification = _governance_payload(
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
            raise RuntimeError("existing preregistration has a different strategy_version")
        if pd.Timestamp(experiment.first_decision_at) != first_signal:
            raise RuntimeError("existing preregistration has a different first_signal_at")
        if experiment.opened_period:
            raise RuntimeError("prospective preregistration cannot be marked opened_period")
        if pd.Timestamp(experiment.registered_at) > freeze_deadline:
            raise RuntimeError("existing preregistration was created after the freeze deadline")
        experiment.verify(specification)
        return experiment

    if preregistration_path.exists():
        experiment = validate_existing(load_preregistration(preregistration_path))
    else:
        if registration > freeze_deadline:
            raise RuntimeError(
                "freeze deadline has passed; create a new strategy_version and future protocol"
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
            # A concurrent first run may have published the same protocol with
            # its own earlier registration timestamp. Validate and reuse it.
            experiment = validate_existing(load_preregistration(preregistration_path))
    trial = TrialRecord.create(
        trial_id=f"{prospective['strategy_version']}:prospective",
        experiment=experiment,
        purpose="prospective_champion_challenger",
    )
    trial = create_or_verify_trial_record(trial_path, trial)
    ledger = initialize_forecast_ledger(
        forecast_path,
        SealedForecastLedger.for_trial(trial),
    )
    return experiment, trial, ledger


def run_champion_challenger(config_path: str | Path) -> dict[str, Path]:
    """Run opened diagnostics and create an immutable prospective registration."""

    config_path = Path(config_path)
    config = load_champion_config(config_path)
    project_root = config_path.resolve().parent.parent
    output_dir = project_root / config["research"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    source_panel_path = project_root / config["research"]["source_panel"]
    daily_prices_path = project_root / config["research"]["source_daily_prices"]
    if not source_panel_path.exists():
        raise FileNotFoundError(
            f"source panel is missing: {source_panel_path}; run public-research first"
        )
    if not daily_prices_path.exists():
        raise FileNotFoundError(
            f"daily price source is missing: {daily_prices_path}; run public-research first"
        )
    config_sha256 = sha256_file(config_path)
    source_tree_sha256, source_file_count = sha256_directory(project_root / "src")
    registered_at = datetime.now(UTC)
    experiment, prospective_trial, forecast_ledger = _write_governance_records(
        config,
        project_root=project_root,
        config_sha256=config_sha256,
        source_tree_sha256=source_tree_sha256,
        registered_at=registered_at,
    )

    panel = pd.read_parquet(source_panel_path)
    panel["signal_date"] = pd.to_datetime(panel["signal_date"], utc=True)
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
    feature_sets = _feature_sets(panel)
    pit_audit = assert_panel_is_point_in_time(
        panel,
        feature_columns=feature_sets["full"],
    )
    panel = add_beta_size_residual_target(panel)

    universe_start = pd.Timestamp(config["model"]["first_prediction"], tz="UTC")
    universe = select_public_proxy_universe(
        panel.loc[panel["signal_date"].ge(universe_start)].copy(),
        top_n=int(config["universe"]["top_n"]),
        ranking_column=config["universe"]["ranking_column"],
        minimum_cross_section=int(config["universe"]["minimum_cross_section"]),
    )
    membership = universe[["signal_date", "ticker", "universe_rank"]]
    model_config = model_config_from_dict(config["model"])
    predictions: dict[str, pd.DataFrame] = {}
    importance_frames: list[pd.DataFrame] = []
    fit_frames: list[pd.DataFrame] = []
    for name, features in feature_sets.items():
        predicted, importance, fit_audit = walk_forward_predict(
            panel,
            features,
            first_prediction=config["model"]["first_prediction"],
            half_life_years=float(config["model"]["recency_half_life_years"]),
            minimum_training_rows=int(config["model"]["minimum_training_rows"]),
            target_column="residual_target_return",
            config=model_config,
        )
        predictions[name] = _filter_membership(predicted, membership)
        importance["feature_set"] = name
        fit_audit["feature_set"] = name
        importance_frames.append(importance)
        fit_frames.append(fit_audit)

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
    prediction_eval = {
        name: evaluable_cross_sections(
            frame.merge(
                evaluation_keys,
                on=["signal_date", "ticker"],
                how="inner",
                validate="one_to_one",
            )
        )
        for name, frame in predictions.items()
    }
    cost = float(config["costs"]["trading_cost_bps_one_way"])
    top_n = int(config["signal"]["holdings"])
    universe_n = int(config["universe"]["top_n"])
    returns_by_strategy: dict[str, pd.DataFrame] = {}
    holdings_by_strategy: dict[str, pd.DataFrame] = {}

    strategy_inputs = {
        "equal_weight_universe": (universe_eval.assign(equal_weight_score=0.0), "equal_weight_score", universe_n),
        "momentum_12_1_top6": (universe_eval, "momentum_12_1", top_n),
        "ridge_full_top6": (prediction_eval["full"], "ridge_score", top_n),
        "ml_momentum_only_top6": (prediction_eval["momentum_only"], "ensemble_score", top_n),
        "ml_fundamentals_only_top6": (prediction_eval["fundamentals_only"], "ensemble_score", top_n),
        "ml_without_momentum_top6": (prediction_eval["without_momentum"], "ensemble_score", top_n),
        "ml_full_top6": (prediction_eval["full"], "ensemble_score", top_n),
    }
    for name, (frame, score, count) in strategy_inputs.items():
        returns, holdings = _run_long_strategy(
            frame,
            name=name,
            score_column=score,
            holdings_count=count,
            trading_cost_bps=cost,
        )
        returns_by_strategy[name] = returns
        holdings_by_strategy[name] = holdings

    neutral_config = config["neutral_portfolio"]
    neutral, neutral_holdings = build_long_short_portfolio(
        prediction_eval["full"],
        score_column="ensemble_score",
        long_quantile=float(neutral_config["long_quantile"]),
        short_quantile=float(neutral_config["short_quantile"]),
        gross_exposure=float(neutral_config["gross_exposure"]),
        max_name_weight=float(neutral_config["max_name_weight"]),
        trading_cost_bps_one_way=cost,
        annual_short_borrow_bps=float(neutral_config["annual_short_borrow_bps"]),
        standardized_exposure_tolerance=float(
            neutral_config["standardized_exposure_tolerance"]
        ),
    )
    neutral["signal_at"] = neutral["signal_date"].map(
        universe_eval.drop_duplicates("signal_date").set_index("signal_date")["signal_at"]
    )
    neutral["strategy"] = "ml_full_neutral_long_short"
    neutral_holdings["strategy"] = "ml_full_neutral_long_short"
    returns_by_strategy["ml_full_neutral_long_short"] = neutral
    holdings_by_strategy["ml_full_neutral_long_short"] = neutral_holdings

    capacity_rows: list[dict[str, Any]] = []
    capacity_trade_frames: list[pd.DataFrame] = []
    for name, holdings in holdings_by_strategy.items():
        capacity, capacity_trades = capacity_diagnostics(
            holdings,
            portfolio_aum_usd=float(config["costs"]["portfolio_aum_usd"]),
            maximum_adv_participation=float(
                config["costs"]["maximum_adv_participation"]
            ),
        )
        capacity_rows.append({"strategy": name, **capacity})
        capacity_trades["strategy"] = name
        capacity_trade_frames.append(capacity_trades)
    capacity_summary = pd.DataFrame(capacity_rows)
    capacity_trades = pd.concat(capacity_trade_frames, ignore_index=True)

    metrics = _metrics_rows(
        returns_by_strategy,
        holdings_by_strategy,
        benchmark_name="equal_weight_universe",
        seed=int(config["research"]["seed"]),
    )
    metrics = metrics.merge(capacity_summary, on="strategy", validate="one_to_one")
    sensitivity_costs = list(config["costs"]["sensitivity_bps"])
    sensitivity_costs.append(2 * cost)
    cost_sensitivity = _cost_sensitivity(
        universe_eval,
        prediction_eval["full"],
        holdings_count=top_n,
        universe_count=universe_n,
        costs=sensitivity_costs,
    )
    incremental_frame = relative_return_frame(
        returns_by_strategy["ml_full_top6"],
        returns_by_strategy["momentum_12_1_top6"],
    )
    incremental = performance_summary(incremental_frame)
    incremental.update(
        moving_block_bootstrap_sharpe(
            incremental_frame["net_return"],
            seed=int(config["research"]["seed"]),
        )
    )
    gates = _evaluate_gates(
        metrics,
        cost_sensitivity,
        capacity_summary,
        incremental,
        config,
        pit_audit,
    )

    champion_capacity = next(
        row for row in capacity_rows if row["strategy"] == "momentum_12_1_top6"
    )
    production_data_quality = {
        "universe_mode": config["universe"]["mode"],
        "evidence_status": config["research"]["evidence_status"],
        "survivorship_free": False,
        "permanent_identifiers": False,
        "delisting_returns": False,
        "execution_data": "Yahoo adjusted-open proxy; no TAQ/NBBO",
        "alfred_vintages_used": False,
        "historical_borrow": False,
        "historical_industry_classification": False,
        "capacity_scenario": {
            key: value for key, value in champion_capacity.items() if key != "strategy"
        },
        "pit_audit": pit_audit,
        "source_rows": len(panel),
        "source_names": int(panel["ticker"].nunique()),
        "selected_names": int(universe["ticker"].nunique()),
        "selected_months": int(universe["signal_date"].nunique()),
        "evaluable_months": int(universe_eval["signal_date"].nunique()),
        "evaluation_entry_at_or_after": str(evaluation_entry_start),
        "evaluation_exit_at_or_before": str(evaluation_exit_end),
        "last_evaluable_signal": str(universe_eval["signal_date"].max()),
    }

    combined_returns = pd.concat(returns_by_strategy.values(), ignore_index=True, sort=False)
    combined_holdings = pd.concat(holdings_by_strategy.values(), ignore_index=True, sort=False)
    combined_returns.to_csv(output_dir / "monthly_returns.csv", index=False)
    combined_holdings.to_parquet(output_dir / "portfolio_holdings.parquet", index=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    cost_sensitivity.to_csv(output_dir / "cost_sensitivity.csv", index=False)
    capacity_summary.to_csv(output_dir / "capacity_summary.csv", index=False)
    capacity_trades.to_csv(output_dir / "capacity_trades.csv", index=False)
    pd.concat(importance_frames, ignore_index=True).to_csv(
        output_dir / "feature_importance.csv", index=False
    )
    pd.concat(fit_frames, ignore_index=True).to_csv(output_dir / "fit_audit.csv", index=False)
    universe.to_parquet(output_dir / "universe_snapshots.parquet", index=False)
    exact_momentum.to_parquet(output_dir / "exact_momentum_signals.parquet", index=False)
    momentum_decisions.to_csv(output_dir / "momentum_decision_calendar.csv", index=False)
    score_panel = membership.copy()
    for name, frame in predictions.items():
        scores = frame[["signal_date", "ticker", "ridge_score", "ensemble_score"]].rename(
            columns={
                "ridge_score": f"{name}_ridge_score",
                "ensemble_score": f"{name}_ensemble_score",
            }
        )
        score_panel = score_panel.merge(scores, on=["signal_date", "ticker"], how="left", validate="one_to_one")
    score_panel.to_parquet(output_dir / "ablation_predictions.parquet", index=False)
    (output_dir / "incremental_metrics.json").write_text(
        json.dumps(incremental, indent=2, sort_keys=True, default=str)
    )
    (output_dir / "gates.json").write_text(
        json.dumps(gates, indent=2, sort_keys=True, default=str)
    )
    (output_dir / "data_quality.json").write_text(
        json.dumps(production_data_quality, indent=2, sort_keys=True, default=str)
    )
    (output_dir / "config_snapshot.yml").write_text(config_path.read_text())

    historical_trials = ExperimentTrialLedger()
    for index, name in enumerate(config["ablations"]):
        historical_spec = ImmutableExperimentSpec.create(
            experiment_id=f"opened:{name}",
            specification={"ablation": name, "config_sha256": config_sha256},
            registered_at=registered_at,
            first_decision_at=str(universe_eval["signal_at"].min()),
            opened_period=True,
        )
        historical_trials = historical_trials.append(
            TrialRecord.create(
                trial_id=f"opened-{index + 1:02d}-{name}",
                experiment=historical_spec,
                purpose="opened_historical_diagnostic",
            )
        )
    (output_dir / "experiment_trial_ledger.json").write_text(
        json.dumps(historical_trials.to_dict(), indent=2, sort_keys=True, default=str)
    )

    # Reporting is imported late so the research computation remains usable in
    # minimal environments and its interface can evolve independently.
    from alpha_factor_engine.champion_reporting import create_champion_challenger_report

    report_output = create_champion_challenger_report(
        output_dir,
        returns=returns_by_strategy,
        holdings=holdings_by_strategy,
        metrics=metrics,
        cost_sensitivity=cost_sensitivity,
        gates=gates,
        data_quality=production_data_quality,
        config=config,
        capacity_summary=capacity_summary,
    )
    report = Path(report_output["report"])
    charts = [Path(path) for path in report_output["charts"]]

    manifest = {
        "run_id": f"{registered_at.strftime('%Y%m%dT%H%M%SZ')}_{stable_json_hash(config)[:8]}",
        "created_at": registered_at.isoformat(),
        "historical_evidence_status": "OPENED_DIAGNOSTIC",
        "config_sha256": config_sha256,
        "source_panel": str(source_panel_path.relative_to(project_root)),
        "source_panel_sha256": sha256_file(source_panel_path),
        "daily_prices": str(daily_prices_path.relative_to(project_root)),
        "daily_prices_sha256": sha256_file(daily_prices_path),
        "source_tree_sha256": source_tree_sha256,
        "source_file_count": source_file_count,
        "resolved_model_config": asdict(model_config),
        "feature_sets": feature_sets,
        "prospective": {
            "experiment_id": experiment.experiment_id,
            "trial_id": prospective_trial.trial_id,
            "spec_sha256": experiment.spec_sha256,
            "forecast_ledger_sha256": forecast_ledger.ledger_sha256,
            "registered_at": experiment.registered_at,
            "freeze_deadline": config["prospective"]["freeze_deadline"],
            "first_decision_at": experiment.first_decision_at,
            "first_entry_at": config["prospective"]["first_entry_at"],
            "minimum_months": config["prospective"]["minimum_months"],
            "sealed_decision_count": forecast_ledger.decision_count,
            "opened_period": experiment.opened_period,
        },
        "charts": [path.relative_to(output_dir).as_posix() for path in charts],
        "artifacts": _artifact_inventory(output_dir),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return {
        "report": report,
        "manifest": manifest_path,
        "output_dir": output_dir,
        "governance_dir": project_root / config["prospective"]["governance_dir"],
    }


__all__ = ["load_champion_config", "run_champion_challenger"]
