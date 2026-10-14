"""Broad-universe v3 readiness and robustness runner."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from alpha_factor_engine.contracts import sha256_directory, sha256_file
from alpha_factor_engine.data.broad_universe import (
    BroadUniverseDataUnavailable,
    BroadUniverseRequirements,
    REQUIRED_COLUMNS,
    audit_broad_universe_panel,
    load_broad_universe_panel,
)
from alpha_factor_engine.research.universe_validation import matched_random_portfolio_null


def load_v3_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("v3 configuration must contain a mapping")
    return config


def _json_write(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _data_readiness(project_root: Path, config: dict[str, Any]) -> dict[str, object]:
    panel_path = project_root / config["research"]["broad_panel"]
    licensed_root = project_root / "data" / "licensed"
    licensed_files = sorted(
        path.relative_to(project_root).as_posix()
        for path in licensed_root.rglob("*")
        if path.is_file()
    ) if licensed_root.exists() else []
    checks = {
        "broad_pit_panel_present": panel_path.is_file(),
        "any_licensed_file_present": bool(licensed_files),
        "marketdata_credential_configured": bool(
            os.getenv("MARKETDATA_TOKEN") or os.getenv("MARKETDATA_API_KEY")
        ),
        "fred_credential_configured": bool(os.getenv("FRED_API_KEY")),
        "sec_user_agent_configured": bool(os.getenv("SEC_USER_AGENT")),
    }
    return {
        "status": "READY_FOR_BROAD_AUDIT" if panel_path.is_file() else "BLOCKED_NO_BROAD_PIT_DATA",
        "expected_broad_panel": panel_path.relative_to(project_root).as_posix(),
        "checks": checks,
        "licensed_files": licensed_files,
        "credential_values_recorded": False,
        "required_panel_lineage": [
            "CRSP historical name/share-code membership",
            "CRSP PERMNO/PERMCO and delisting returns",
            "Compustat PIT fundamentals linked with historical CCM",
            "MarketData unadjusted OHLCV/next-open crosscheck",
            "SEC filing timestamps and ALFRED vintage timestamps",
        ],
    }


def _manual_first_month_reconstruction(
    holdings: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    strategy: str,
    trading_cost_bps: float,
) -> dict[str, object]:
    strategy_returns = returns.loc[returns["strategy"].eq(strategy)].copy()
    strategy_returns["signal_at"] = pd.to_datetime(strategy_returns["signal_at"], utc=True)
    first = strategy_returns.sort_values("signal_at").iloc[0]
    strategy_holdings = holdings.loc[holdings["strategy"].eq(strategy)].copy()
    strategy_holdings["signal_at"] = pd.to_datetime(strategy_holdings["signal_at"], utc=True)
    month = strategy_holdings.loc[strategy_holdings["signal_at"].eq(first["signal_at"])]
    gross = float((month["weight"] * month["target_return"]).sum())
    turnover = float(month["weight"].abs().sum())
    trading_cost = turnover * trading_cost_bps / 10_000.0
    reconstructed_net = gross - trading_cost
    recorded_net = float(first["net_return"])
    difference = reconstructed_net - recorded_net
    if abs(difference) > 1e-12:
        raise RuntimeError(
            f"manual first-month reconstruction mismatch: {difference}"
        )
    return {
        "status": "PASS",
        "scope": "conditional_v2_current_ticker_proxy",
        "strategy": strategy,
        "signal_at": first["signal_at"].isoformat(),
        "holdings": int(len(month)),
        "gross_return_sum_weight_times_security_return": gross,
        "initial_turnover": turnover,
        "trading_cost": trading_cost,
        "reconstructed_net_return": reconstructed_net,
        "recorded_net_return": recorded_net,
        "absolute_difference": abs(difference),
    }


def _run_conditional_random_null(
    project_root: Path,
    output_dir: Path,
    config: dict[str, Any],
) -> tuple[dict[str, object], dict[str, object]]:
    diagnostic = config["conditional_diagnostic"]
    panel = pd.read_parquet(project_root / diagnostic["source_panel"])
    returns = pd.read_csv(project_root / diagnostic["source_returns"])
    holdings = pd.read_parquet(
        project_root / "artifacts/champion_challenger_v2/portfolio_holdings.parquet"
    )
    observed = returns.loc[returns["strategy"].eq(diagnostic["observed_strategy"])].copy()
    benchmark = returns.loc[returns["strategy"].eq(diagnostic["benchmark_strategy"])].copy()
    observed_holdings = holdings.loc[
        holdings["strategy"].eq(diagnostic["observed_strategy"])
    ].copy()
    result = matched_random_portfolio_null(
        panel,
        observed,
        benchmark,
        holdings_count=int(diagnostic["holdings"]),
        simulations=int(config["validation"]["random_portfolio_simulations"]),
        security_column="ticker",
        signal_column="signal_at",
        target_column="target_return",
        trading_cost_bps_one_way=float(config["portfolio"]["trading_cost_bps_one_way"]),
        seed=int(config["research"]["seed"]),
        observed_holdings=observed_holdings,
    )
    result.trials.to_csv(output_dir / "conditional_random_portfolio_trials.csv", index=False)
    _json_write(output_dir / "conditional_random_portfolio_summary.json", result.summary)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    chart_specs = [
        ("active_annualized_return", "Active annualized return"),
        ("active_information_ratio", "Active information ratio"),
    ]
    for axis, (column, title) in zip(axes, chart_specs):
        axis.hist(result.trials[column].dropna(), bins=50, color="#7895cb", alpha=0.85)
        observed_value = float(result.summary[f"observed_{column}"])
        axis.axvline(observed_value, color="#b5362f", linewidth=2.2, label="Observed momentum")
        axis.axvline(
            float(result.summary[f"null_{column}_p95"]),
            color="#222222",
            linestyle="--",
            linewidth=1.3,
            label="Random 95th percentile",
        )
        axis.set_title(title)
        axis.set_xlabel(column.replace("_", " "))
        axis.set_ylabel("random portfolios")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle("Conditional random-stock null: current-ticker proxy only", fontsize=13)
    fig.tight_layout()
    chart_path = output_dir / "conditional_random_portfolio_null.png"
    fig.savefig(chart_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    manual = _manual_first_month_reconstruction(
        holdings,
        returns,
        strategy=str(diagnostic["observed_strategy"]),
        trading_cost_bps=float(config["portfolio"]["trading_cost_bps_one_way"]),
    )
    _json_write(output_dir / "manual_trade_reconstruction.json", manual)
    return result.summary, manual


def _protocol_payload(config: dict[str, Any]) -> dict[str, object]:
    return {
        "strategy_version": config["research"]["strategy_version"],
        "registered_at": datetime.now(UTC).isoformat(),
        "status": "PROTOCOL_SNAPSHOT_BEFORE_BROAD_PIT_DATA_ACCESS",
        "universe": config["universe"],
        "data_roles": config["data_roles"],
        "features": config["features"],
        "model": config["model"],
        "portfolio": config["portfolio"],
        "validation": config["validation"],
        "gates": config["gates"],
        "interpretation": (
            "This lock limits researcher degrees of freedom before the licensed broad panel is present. "
            "It is not a prospective performance result."
        ),
    }


def _create_or_verify_design_lock(
    project_root: Path,
    config_file: Path,
    config: dict[str, Any],
) -> Path:
    """Freeze the v3 design before the licensed broad panel is supplied."""

    governance_dir = project_root / "governance" / config["research"]["strategy_version"]
    governance_dir.mkdir(parents=True, exist_ok=True)
    path = governance_dir / "design_lock.json"
    source_hash, source_files = sha256_directory(project_root / "src")
    immutable = {
        "strategy_version": config["research"]["strategy_version"],
        "config_sha256": sha256_file(config_file),
        "source_tree_sha256": source_hash,
        "source_file_count": source_files,
        "broad_panel_absent_at_registration": not (
            project_root / config["research"]["broad_panel"]
        ).is_file(),
        "opened_period": True,
        "purpose": "lock_methodology_before_first_access_to_the_licensed_broad_panel",
        "claim_limit": "methodology lock only; not a prospective performance registration",
    }
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        for key, value in immutable.items():
            if existing.get(key) != value:
                raise RuntimeError(
                    f"v3 design lock drift in {key}; create a new strategy version"
                )
        return path
    payload = {"registered_at": datetime.now(UTC).isoformat(), **immutable}
    _json_write(path, payload)
    return path


def _schema_payload(config: dict[str, Any]) -> dict[str, object]:
    return {
        "format": "parquet",
        "row_key": ["signal_at", "permno"],
        "required_columns": sorted(REQUIRED_COLUMNS),
        "required_model_features": config["features"]["columns"],
        "return_identity": "target_return=(1+base_period_return)*(1+delisting_return)-1",
        "timestamp_rule": (
            "membership_available_at, price_available_at, fundamental_available_at, "
            "and industry_available_at must be no later than signal_at"
        ),
        "execution_rule": "signal_at < entry_at < exit_at; target_return spans entry_at to exit_at",
        "identifier_rule": "PERMNO and PERMCO are mandatory; ticker is descriptive only",
        "membership_rule": "historical CRSP name/share-code records; no current-constituent backfill",
        "delisting_rule": "delisting_return_observed=true and an explicit numeric value, including zero",
    }


def _validation_matrix(readiness: dict[str, object]) -> pd.DataFrame:
    broad_ready = bool(readiness["checks"]["broad_pit_panel_present"])
    blocked = "PENDING_BROAD_PANEL_AUDIT" if broad_ready else "BLOCKED_NO_BROAD_PIT_DATA"
    return pd.DataFrame(
        [
            {"validation": "strict_point_in_time_and_delisting_contract", "status": blocked},
            {"validation": "chronological_walk_forward_oos", "status": blocked},
            {"validation": "five_fold_heldout_permno", "status": blocked},
            {"validation": "200_stratified_random_universes", "status": blocked},
            {"validation": "200_full_retrain_label_permutations", "status": blocked},
            {"validation": "10000_matched_random_portfolios_broad", "status": blocked},
            {"validation": "10000_matched_random_portfolios_current_proxy", "status": "COMPLETED_DIAGNOSTIC_ONLY"},
            {"validation": "factor_alpha_multiple_testing_cost_capacity", "status": blocked},
            {"validation": "36_month_prospective_test", "status": "NOT_STARTED"},
        ]
    )


def _write_report(
    output_dir: Path,
    *,
    readiness: dict[str, object],
    broad_audit: dict[str, object] | None,
    random_summary: dict[str, object],
    manual: dict[str, object],
) -> Path:
    active_p = float(random_summary["empirical_p_value_active_annualized_return"])
    ir_p = float(random_summary["empirical_p_value_active_information_ratio"])
    lines = [
        "# Broad-Universe Alpha Engine v3 — Validation Readiness",
        "",
        "## Verdict",
        "",
        "**NOT VALIDATED.** The v3 empirical run is blocked because the required survivorship-free broad PIT panel is absent.",
        "",
        f"- Data status: `{readiness['status']}`",
        "- The implemented gate requires at least 1,000 eligible securities in every tested month.",
        "- Current ticker lists, SEC's current registrant list, Yahoo, and MarketData snapshot symbols cannot replace CRSP permanent IDs and delisting returns.",
        "- The immutable v2 archive remains the historical reference. The working source now contains separate v3 files and must not be used to rerun v2.",
        "",
        "## What is implemented",
        "",
        "- Strict CRSP/Compustat PIT broad-panel contract with explicit delisting-return reconciliation.",
        "- Historical membership, one-security-per-company, price, market-cap, ADV, and industry gates.",
        "- Stable PERMNO five-fold stock holdouts, stratified random-universe draws, and within-month label permutations.",
        "- A cost-aware 10,000-portfolio random-stock null with drifted weights and terminal liquidation.",
        "- MarketData bulk snapshot ingestion restricted to OHLCV; it cannot define membership.",
        "",
        "## Conditional diagnostic on the existing current-ticker proxy",
        "",
        f"This test compares `{random_summary['holdings_count']}` randomly selected stocks per month with the existing momentum portfolio over `{random_summary['months']}` months. It is useful as a placebo, but it does not cure survivorship bias.",
        "",
        f"- Observed active annualized return: {float(random_summary['observed_active_annualized_return']):.2%}",
        f"- Random-null median: {float(random_summary['null_active_annualized_return_p50']):.2%}",
        f"- Random-null 95th percentile: {float(random_summary['null_active_annualized_return_p95']):.2%}",
        f"- Empirical p-value for active return: {active_p:.4f}",
        f"- Observed information ratio: {float(random_summary['observed_active_information_ratio']):.3f}",
        f"- Random-null 95th percentile information ratio: {float(random_summary['null_active_information_ratio_p95']):.3f}",
        f"- Empirical p-value for information ratio: {ir_p:.4f}",
        f"- Observed average monthly turnover: {float(random_summary['observed_average_monthly_turnover']):.3f}",
        f"- Random-null median monthly turnover after matching retention counts: {float(random_summary['null_average_monthly_turnover_p50']):.3f}",
        "",
        "![Conditional random-stock null](conditional_random_portfolio_null.png)",
        "",
        "## Accounting check",
        "",
        f"The first observed portfolio month was reconstructed from `{manual['holdings']}` constituent weights and returns. Absolute return difference: `{float(manual['absolute_difference']):.3e}` (`{manual['status']}`).",
        "",
        "## Data required to release the full run",
        "",
        f"Place the joined, timestamped parquet panel at `{readiness['expected_broad_panel']}`. Required lineage:",
        "",
        *[f"- {item}" for item in readiness["required_panel_lineage"]],
        "",
        "After the file arrives, its audit must pass before the blocked validation jobs are released. The validation matrix records which empirical jobs remain unavailable.",
        "",
    ]
    if broad_audit is not None:
        lines.extend(["## Broad-panel contract audit", "", f"Status: `{broad_audit['status']}`", ""])
    path = output_dir / "v3_validation_readiness.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _artifact_manifest(output_dir: Path, config_path: Path) -> dict[str, object]:
    files = []
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file() and item.name != "manifest.json"):
        files.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "strategy_version": "broad_universe_v3",
        "config_path": config_path.as_posix(),
        "config_sha256": sha256_file(config_path),
        "artifacts": files,
        "result_validity": "NOT VALIDATED",
    }


def run_broad_universe_v3(config_path: str | Path = "configs/broad_universe_v3.yml") -> dict[str, Path | str]:
    config_file = Path(config_path).resolve()
    config = load_v3_config(config_file)
    project_root = config_file.parent.parent
    output_dir = project_root / config["research"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    design_lock = _create_or_verify_design_lock(project_root, config_file, config)

    readiness = _data_readiness(project_root, config)
    _json_write(output_dir / "data_readiness.json", readiness)
    protocol = _protocol_payload(config)
    _json_write(output_dir / "validation_protocol.json", protocol)
    _json_write(output_dir / "broad_panel_schema.json", _schema_payload(config))
    _validation_matrix(readiness).to_csv(output_dir / "validation_matrix.csv", index=False)

    broad_audit: dict[str, object] | None = None
    try:
        broad = load_broad_universe_panel(project_root / config["research"]["broad_panel"])
        universe = config["universe"]
        broad_audit = audit_broad_universe_panel(
            broad,
            requirements=BroadUniverseRequirements(
                minimum_securities_per_month=int(universe["minimum_securities_per_month"]),
                minimum_price=float(universe["minimum_price"]),
                minimum_market_cap_usd=float(universe["minimum_market_cap_usd"]),
                minimum_adv_usd_3m=float(universe["minimum_adv_usd_3m"]),
                common_share_codes=tuple(universe["common_share_codes"]),
                exchange_codes=tuple(universe["exchange_codes"]),
                one_security_per_company=bool(universe["one_security_per_company"]),
            ),
            feature_columns=config["features"]["columns"],
        )
        _json_write(output_dir / "broad_panel_audit.json", broad_audit)
    except BroadUniverseDataUnavailable:
        pass

    random_summary, manual = _run_conditional_random_null(
        project_root,
        output_dir,
        config,
    )
    report = _write_report(
        output_dir,
        readiness=readiness,
        broad_audit=broad_audit,
        random_summary=random_summary,
        manual=manual,
    )
    manifest_path = output_dir / "manifest.json"
    _json_write(manifest_path, _artifact_manifest(output_dir, config_file))
    return {
        "status": "NOT VALIDATED",
        "report": report,
        "manifest": manifest_path,
        "output_dir": output_dir,
        "design_lock": design_lock,
    }


__all__ = ["load_v3_config", "run_broad_universe_v3"]
