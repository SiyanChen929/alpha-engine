"""Run the frozen security-held-out nonlinear proxy model and core diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from alpha_factor_engine.research.models import model_config_from_dict
from alpha_factor_engine.research.universe_validation import (
    out_of_sample_cross_section_metrics,
    walk_forward_security_holdout_predict,
)
from proxy_research.evaluate import (
    MODEL_FEATURES,
    build_equal_weight_portfolio,
    capacity_summary,
    factor_validation,
    manual_first_month_reconstruction,
    sanitize_panel,
    summaries_by_period,
)


def run(config_path: str | Path) -> Path:
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text())
    root = config_path.parent.parent
    output = root / config["data"]["output_dir"]
    data_dir = output / "data"
    output.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(data_dir / "monthly_feature_panel.parquet")
    panel, label_audit = sanitize_panel(
        panel,
        minimum_cross_section=int(config["universe"]["minimum_monthly_cross_section"]),
        first_prediction=config["research"]["first_prediction"],
    )
    label_audit.to_csv(output / "label_coverage_audit.csv", index=False)
    panel.to_parquet(data_dir / "model_eligible_panel.parquet", index=False)

    model_config = model_config_from_dict({**config["model"], "seed": config["research"]["seed"]})
    predictions, fit_audit = walk_forward_security_holdout_predict(
        panel,
        MODEL_FEATURES,
        first_prediction=config["research"]["first_prediction"],
        folds=int(config["validation"]["security_holdout_folds"]),
        security_column="ticker",
        minimum_training_rows=int(config["model"]["minimum_training_rows"]),
        half_life_years=float(config["model"]["recency_half_life_years"]),
        seed=int(config["research"]["seed"]),
        config=model_config,
    )
    if fit_audit["security_overlap"].max() != 0:
        raise RuntimeError("held-out-stock fit contains overlapping tickers")
    if not fit_audit.loc[fit_audit["status"].eq("ok"), "latest_label_exit_at"].lt(
        fit_audit.loc[fit_audit["status"].eq("ok"), "cutoff"]
    ).all():
        raise RuntimeError("walk-forward fit used labels not known at its cutoff")
    predictions.to_parquet(output / "heldout_stock_predictions.parquet", index=False)
    fit_audit.to_csv(output / "heldout_stock_fit_audit.csv", index=False)
    metrics, monthly_ic = out_of_sample_cross_section_metrics(predictions)
    monthly_ic.to_csv(output / "model_monthly_ic.csv", index=False)

    cost_results = []
    base_returns = base_holdings = benchmark = None
    for cost in config["portfolio"]["cost_sensitivity_bps"]:
        returns, holdings = build_equal_weight_portfolio(
            predictions,
            score_column="ensemble_score",
            holdings_count=int(config["portfolio"]["holdings"]),
            trading_cost_bps_one_way=float(cost),
        )
        benchmark_returns, _ = build_equal_weight_portfolio(
            predictions,
            score_column=None,
            holdings_count=None,
            trading_cost_bps_one_way=float(cost),
        )
        period = summaries_by_period(
            returns, benchmark_returns,
            holdout_start=config["research"]["untouched_holdout_start"],
        )
        for window, values in period.items():
            cost_results.append({"cost_bps": float(cost), "window": window, **values})
        if float(cost) == float(config["portfolio"]["trading_cost_bps_one_way"]):
            base_returns, base_holdings, benchmark = returns, holdings, benchmark_returns
    if base_returns is None:
        raise RuntimeError("base trading cost missing from sensitivity grid")
    base_returns.to_csv(output / "model_portfolio_returns.csv", index=False)
    base_holdings.to_parquet(output / "model_portfolio_holdings.parquet", index=False)
    benchmark.to_csv(output / "equal_weight_benchmark_returns.csv", index=False)
    pd.DataFrame(cost_results).to_csv(output / "cost_sensitivity.csv", index=False)

    factor_summary, factor_monthly = factor_validation(
        panel, start=config["research"]["untouched_holdout_start"]
    )
    factor_summary.to_csv(output / "factor_validation_holdout.csv", index=False)
    factor_monthly.to_csv(output / "factor_monthly_holdout.csv", index=False)
    reconstruction = manual_first_month_reconstruction(
        predictions, base_returns, base_holdings,
        cost_bps=float(config["portfolio"]["trading_cost_bps_one_way"]),
    )
    summary = {
        "evidence_status": config["research"]["evidence_status"],
        "audit_verdict": "NOT VALIDATED",
        "audit_reason": "current-survivor proxy lacks delisted stocks and historical membership",
        "model_cross_section_metrics": metrics,
        "performance": summaries_by_period(
            base_returns, benchmark,
            holdout_start=config["research"]["untouched_holdout_start"],
        ),
        "capacity_at_10m": capacity_summary(base_holdings),
        "manual_reconstruction": reconstruction,
        "security_holdout_max_overlap": int(fit_audit["security_overlap"].max()),
        "feature_count": len(MODEL_FEATURES),
        "prediction_rows": len(predictions),
        "prediction_months": predictions["signal_date"].nunique(),
    }
    path = output / "core_results.json"
    path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return path


__all__ = ["run"]
