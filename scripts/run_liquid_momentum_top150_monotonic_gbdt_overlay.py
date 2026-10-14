#!/usr/bin/env python3
"""Run the fail-first strict held-out stage of the frozen ML challenger.

The portfolio overlay remains gated.  This runner first asks whether one fixed
monotonic GBDT generalizes to securities whose labels never enter its fit.  It
does not combine fold predictions into a portfolio because separately fitted
fold scores are not cross-fold calibrated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alpha_factor_engine.research.strict_ml_holdout import (
    MonotonicGBDTConfig,
    select_top_momentum_candidates,
    strict_heldout_information_coefficient,
    walk_forward_strict_security_holdout_predict,
)
from alpha_factor_engine.research.metrics import newey_west_tstat


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=_json_default,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _runtime_versions() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "numpy": version("numpy"),
        "pandas": version("pandas"),
        "scipy": version("scipy"),
        "scikit_learn": version("scikit-learn"),
        "pyarrow": version("pyarrow"),
    }


def _resolve_candidate_count(config: dict[str, Any]) -> int | None:
    """Resolve fixed-size or full-mother-universe candidate selection."""

    raw_value = config["universe"].get("momentum_candidate_count", 150)
    if raw_value is None:
        return None
    candidate_count = int(raw_value)
    if candidate_count <= 0:
        raise ValueError("momentum_candidate_count must be positive or null")
    return candidate_count


def _candidate_universe_audit(
    panel: pd.DataFrame,
    *,
    candidate_count: int | None,
    security_column: str,
) -> pd.DataFrame:
    """Reconcile the source, eligible, evaluable, and selected monthly rows."""

    frame = panel.copy()
    frame["signal_date"] = pd.to_datetime(frame["signal_date"], utc=True, errors="raise")
    frame["_eligible"] = frame["eligible"].fillna(False).astype(bool)
    target = pd.to_numeric(frame["target_return"], errors="coerce")
    frame["_evaluable"] = frame["_eligible"] & target.notna() & np.isfinite(target)
    monthly = (
        frame.groupby("signal_date", sort=True)
        .agg(
            source_rows=(security_column, "size"),
            eligible_rows=("_eligible", "sum"),
            evaluable_rows=("_evaluable", "sum"),
        )
        .reset_index()
    )
    selected = select_top_momentum_candidates(
        frame.drop(columns=["_eligible", "_evaluable"]),
        candidate_count=candidate_count,
        security_column=security_column,
    )
    selected_monthly = (
        selected.groupby("signal_date", sort=True)
        .size()
        .rename("selected_rows")
        .reset_index()
    )
    monthly = monthly.merge(
        selected_monthly, on="signal_date", how="left", validate="one_to_one"
    )
    if monthly["selected_rows"].isna().any():
        raise ValueError("candidate selection silently omitted a source month")
    monthly["selected_rows"] = monthly["selected_rows"].astype(int)
    monthly["excluded_by_candidate_prefilter"] = (
        monthly["eligible_rows"] - monthly["selected_rows"]
    ).astype(int)
    monthly["selection_mode"] = (
        "all_eligible_rows" if candidate_count is None else f"top_{candidate_count}_momentum"
    )
    return monthly


def _validate_source_panel(panel: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    required = {
        "signal_date",
        "entry_at",
        "exit_at",
        "target_return",
        "ticker",
        "eligible",
        "price_available_at",
        "latest_input_available_at",
        "fundamental_available_at",
        *features,
    }
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"strict ML source panel missing columns: {missing}")
    frame = panel.copy()
    for column in [
        "signal_date",
        "entry_at",
        "exit_at",
        "price_available_at",
        "latest_input_available_at",
        "fundamental_available_at",
    ]:
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if frame.duplicated(["signal_date", "ticker"]).any():
        raise ValueError("strict ML source panel contains duplicate signal/ticker rows")
    if frame[["signal_date", "entry_at", "exit_at"]].isna().any().any():
        raise ValueError("strict ML source panel has missing decision or execution timestamps")
    if not frame["signal_date"].lt(frame["entry_at"]).all():
        raise ValueError("strict ML source panel contains entry_at at or before signal_date")
    if not frame["entry_at"].lt(frame["exit_at"]).all():
        raise ValueError("strict ML source panel contains exit_at at or before entry_at")
    if frame["target_return"].isna().any():
        raise ValueError("strict ML source panel contains incomplete realized labels")
    if (frame["price_available_at"] > frame["signal_date"]).any():
        raise ValueError("price features were unavailable at the decision clock")
    if (frame["latest_input_available_at"] > frame["signal_date"]).any():
        raise ValueError("residual-momentum inputs were unavailable at the decision clock")
    fundamental_features = [
        column
        for column in features
        if column
        in {
            "book_to_market",
            "earnings_yield",
            "roe",
            "roa",
            "gross_profitability",
            "debt_to_equity",
            "asset_growth",
            "sales_growth",
            "accruals",
            "free_cash_flow_yield",
            "log_market_cap",
        }
    ]
    populated_fundamental = frame[fundamental_features].notna().any(axis=1)
    invalid_fundamental_clock = populated_fundamental & (
        frame["fundamental_available_at"].isna()
        | frame["fundamental_available_at"].gt(frame["signal_date"])
    )
    if invalid_fundamental_clock.any():
        raise ValueError("populated fundamental features fail their availability clock")
    eligible = frame["eligible"].fillna(False).astype(bool)
    monthly_eligible = frame.loc[eligible].groupby("signal_date", sort=True).size()
    if monthly_eligible.empty:
        raise ValueError("strict ML source panel contains no eligible rows")
    return {
        "rows": int(len(frame)),
        "months": int(frame["signal_date"].nunique()),
        "securities": int(frame["ticker"].nunique()),
        "eligible_rows": int(eligible.sum()),
        "monthly_eligible_minimum": int(monthly_eligible.min()),
        "monthly_eligible_median": float(monthly_eligible.median()),
        "monthly_eligible_maximum": int(monthly_eligible.max()),
        "duplicate_signal_security_rows": 0,
        "missing_realized_labels": 0,
        "future_price_availability_rows": 0,
        "future_residual_input_rows": 0,
        "future_or_missing_fundamental_availability_rows": 0,
        "decision_before_entry_before_exit": True,
    }


def _validate_source_lineage(
    config: dict[str, Any], source_path: Path, panel: pd.DataFrame
) -> tuple[dict[str, Any], list[Path]]:
    """Fail closed if the frozen v2 evaluation panel or its audit drifts."""

    lineage = config["source_lineage"]
    expected_sha256 = str(lineage["expected_source_sha256"])
    source_sha256 = _sha256(source_path)
    if source_sha256 != expected_sha256:
        raise ValueError(
            f"source panel hash drifted: {source_sha256} != {expected_sha256}"
        )
    expected_rows = int(lineage["expected_source_rows"])
    expected_months = int(lineage["expected_source_months"])
    if len(panel) != expected_rows or panel["signal_date"].nunique() != expected_months:
        raise ValueError("source panel row/month counts differ from frozen v2 lineage")

    upstream_manifest_path = ROOT / lineage["upstream_artifact_manifest"]
    upstream_manifest_sha256 = _sha256(upstream_manifest_path)
    if upstream_manifest_sha256 != lineage["expected_upstream_manifest_sha256"]:
        raise ValueError("upstream artifact manifest hash drifted")
    upstream_manifest = json.loads(upstream_manifest_path.read_text(encoding="utf-8"))

    audit_path = ROOT / lineage["monthly_universe_audit"]
    audit_sha256 = _sha256(audit_path)
    if audit_sha256 != lineage["expected_monthly_universe_audit_sha256"]:
        raise ValueError("upstream monthly universe audit hash drifted")
    audit = pd.read_csv(audit_path)
    required_audit = {
        "signal_date",
        "selected_rows",
        "selected_missing_labels",
        "signal_status",
        "evaluation_status",
    }
    missing_audit = sorted(required_audit.difference(audit.columns))
    if missing_audit:
        raise ValueError(f"upstream universe audit missing columns: {missing_audit}")
    audit["signal_date"] = pd.to_datetime(audit["signal_date"], utc=True, errors="raise")
    evaluable = audit.loc[audit["evaluation_status"].eq("evaluable")].copy()
    if len(evaluable) != expected_months:
        raise ValueError("upstream evaluable-month count does not match source panel")
    if int(evaluable["selected_rows"].sum()) != expected_rows:
        raise ValueError("upstream evaluable selected-row count does not match source panel")
    if int(evaluable["selected_missing_labels"].sum()) != 0:
        raise ValueError("upstream evaluation panel contains missing selected labels")
    future_kept = audit.loc[audit["evaluation_status"].ne("evaluable")]
    label_independent_evidence = bool(
        not future_kept.empty
        and future_kept["signal_status"].eq("kept").all()
        and future_kept["selected_missing_labels"].gt(0).any()
    )
    if not bool(lineage["membership_frozen_before_label_completion"]):
        raise ValueError("config does not assert pre-label universe membership")
    if not label_independent_evidence:
        raise ValueError("upstream audit lacks observable label-independent selection evidence")

    ledger_path = ROOT / lineage["opened_search_space_ledger"]
    ledger_sha256 = _sha256(ledger_path)
    if ledger_sha256 != lineage["expected_opened_search_space_ledger_sha256"]:
        raise ValueError("upstream opened-search ledger hash drifted")
    ledger = pd.read_csv(ledger_path)
    expected_ledger_rows = int(lineage["expected_upstream_opened_trials"])
    if len(ledger) != expected_ledger_rows:
        raise ValueError("upstream opened-search ledger count drifted")
    disclosed_family = int(
        config["validation"]["disclosed_multiple_testing_family_size_minimum"]
    )
    added_trials = len(config["known_opened_trials_added_by_this_research"])
    if disclosed_family < expected_ledger_rows + added_trials:
        raise ValueError("disclosed multiple-testing family omits known opened trials")

    for evidence_path, evidence_sha256 in [
        (source_path, source_sha256),
        (audit_path, audit_sha256),
        (ledger_path, ledger_sha256),
    ]:
        records = [
            record
            for record in upstream_manifest.get("records", [])
            if Path(record["path"]).resolve() == evidence_path.resolve()
        ]
        if len(records) != 1 or records[0]["sha256"] != evidence_sha256:
            raise ValueError(
                f"{evidence_path.name} is not uniquely hash-matched in upstream manifest"
            )

    evidence = {
        "source_panel_sha256": source_sha256,
        "source_manifest_hash_match": True,
        "upstream_manifest_sha256": upstream_manifest_sha256,
        "upstream_audit_sha256": audit_sha256,
        "upstream_opened_ledger_sha256": ledger_sha256,
        "audit_evaluable_months": int(len(evaluable)),
        "audit_evaluable_selected_rows": int(evaluable["selected_rows"].sum()),
        "audit_evaluable_missing_selected_labels": 0,
        "audit_non_evaluable_signal_months_kept": int(len(future_kept)),
        "audit_missing_label_rows_kept": int(
            future_kept["selected_missing_labels"].sum()
        ),
        "membership_frozen_before_label_completion": True,
        "upstream_opened_trials": int(len(ledger)),
        "disclosed_multiple_testing_family_size_minimum": disclosed_family,
    }
    return evidence, [audit_path, upstream_manifest_path, ledger_path]


def _gate_table(
    audit: pd.DataFrame,
    fold_month: pd.DataFrame,
    summary: dict[str, float],
    config: dict[str, Any],
) -> pd.DataFrame:
    successful = audit.loc[audit["status"].eq("ok")]
    by_fold = fold_month.groupby("security_fold")["incremental_rank_ic"].mean()
    gates = config["gates"]
    rows = [
        {
            "gate": "zero_security_overlap",
            "passed": bool(not successful.empty and successful["security_overlap"].eq(0).all()),
            "evidence": f"maximum={int(successful['security_overlap'].max()) if not successful.empty else -1}",
        },
        {
            "gate": "all_training_labels_exited_before_cutoff",
            "passed": bool(
                not successful.empty
                and (
                    pd.to_datetime(successful["latest_label_exit_at"], utc=True)
                    < pd.to_datetime(successful["cutoff"], utc=True)
                ).all()
            ),
            "evidence": "required=true",
        },
        {
            "gate": "fold_local_target_scope",
            "passed": bool(
                not successful.empty
                and successful["train_target_scope"].eq("nonheldout_fold_only").all()
            ),
            "evidence": "required=nonheldout_fold_only",
        },
        {
            "gate": "strict_heldout_mean_incremental_rank_ic",
            "passed": bool(
                summary["mean_incremental_rank_ic"]
                > float(gates["strict_heldout_mean_incremental_rank_ic_minimum"])
            ),
            "evidence": (
                f"observed={summary['mean_incremental_rank_ic']:.6f}; "
                f"minimum>{float(gates['strict_heldout_mean_incremental_rank_ic_minimum']):.6f}"
            ),
        },
        {
            "gate": "strict_heldout_incremental_ic_hac_tstat",
            "passed": bool(
                summary["incremental_rank_ic_hac_tstat"]
                >= float(gates["strict_heldout_incremental_ic_hac_tstat_minimum"])
            ),
            "evidence": (
                f"observed={summary['incremental_rank_ic_hac_tstat']:.6f}; "
                f"minimum={float(gates['strict_heldout_incremental_ic_hac_tstat_minimum']):.6f}"
            ),
        },
        {
            "gate": "positive_security_folds",
            "passed": bool(
                int(by_fold.gt(0.0).sum()) >= int(gates["minimum_positive_security_folds"])
            ),
            "evidence": (
                f"observed={int(by_fold.gt(0.0).sum())}; "
                f"minimum={int(gates['minimum_positive_security_folds'])}"
            ),
        },
        {
            "gate": "upstream_membership_frozen_before_label_completion",
            "passed": bool(
                config["source_lineage"]["membership_frozen_before_label_completion"]
            ),
            "evidence": "v2 audit keeps non-evaluable signal months",
        },
        {
            "gate": "upstream_survivorship_free",
            "passed": bool(config["universe"]["upstream_survivorship_free"]),
            "evidence": "required=true",
        },
        {
            "gate": "permanent_security_identifier",
            "passed": bool(config["universe"]["permanent_security_identifier"]),
            "evidence": f"security_column={config['universe']['security_column']}",
        },
    ]
    return pd.DataFrame(rows)


def _evaluate_periods(
    predictions: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[
    dict[str, tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]],
    pd.DataFrame,
    pd.DataFrame,
]:
    """Report known regimes separately and reserve pre-2026 for the main gate."""

    entry = pd.to_datetime(predictions["entry_at"], utc=True, errors="raise")
    primary_end = pd.Timestamp(config["validation"]["primary_gate_end_exclusive"])
    primary_end = (
        primary_end.tz_localize("UTC")
        if primary_end.tzinfo is None
        else primary_end.tz_convert("UTC")
    )
    h1_end = pd.Timestamp(config["validation"]["known_2026h1_end_exclusive"])
    h1_end = h1_end.tz_localize("UTC") if h1_end.tzinfo is None else h1_end.tz_convert("UTC")
    masks = {
        "full": pd.Series(True, index=predictions.index),
        "pre2026_primary": entry.lt(primary_end),
        "2026h1_known_regime": entry.ge(primary_end) & entry.lt(h1_end),
    }
    evaluations: dict[
        str, tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]
    ] = {}
    period_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for period, mask in masks.items():
        subset = predictions.loc[mask].copy()
        if subset.empty:
            raise ValueError(f"period {period!r} has no predictions")
        fold_month, monthly, summary = strict_heldout_information_coefficient(
            subset,
            target_column=config["target"]["realized_return_column"],
            control_columns=list(config["target"]["controls"]),
            control_missing_policy=config["target"]["control_missing_policy"],
        )
        evaluations[period] = (fold_month, monthly, summary)
        period_rows.append(
            {
                "period": period,
                "entry_start": entry.loc[mask].min(),
                "entry_end": entry.loc[mask].max(),
                "prediction_rows": int(mask.sum()),
                **summary,
            }
        )
        for security_fold, group in fold_month.groupby("security_fold", sort=True):
            fold_rows.append(
                {
                    "period": period,
                    "security_fold": int(security_fold),
                    "months": int(len(group)),
                    "names": int(group["n"].sum()),
                    "mean_raw_rank_ic": float(group["raw_rank_ic"].mean()),
                    "mean_incremental_rank_ic": float(
                        group["incremental_rank_ic"].mean()
                    ),
                    "incremental_rank_ic_hac_tstat": float(
                        newey_west_tstat(group["incremental_rank_ic"])
                    ),
                }
            )
    return evaluations, pd.DataFrame(period_rows), pd.DataFrame(fold_rows)


def _authorization_status(gates: pd.DataFrame) -> dict[str, bool | str]:
    """Keep statistical evidence separate from fatal lineage authorization."""

    indexed = gates.set_index("gate")["passed"]
    statistical_names = {
        "zero_security_overlap",
        "all_training_labels_exited_before_cutoff",
        "fold_local_target_scope",
        "strict_heldout_mean_incremental_rank_ic",
        "strict_heldout_incremental_ic_hac_tstat",
        "positive_security_folds",
        "upstream_membership_frozen_before_label_completion",
    }
    fatal_lineage_names = {
        "upstream_survivorship_free",
        "permanent_security_identifier",
    }
    required = statistical_names | fatal_lineage_names
    missing = sorted(required.difference(indexed.index))
    if missing:
        raise ValueError(f"authorization gates missing rows: {missing}")
    statistical_pass = bool(indexed.loc[list(statistical_names)].astype(bool).all())
    fatal_lineage_pass = bool(
        indexed.loc[list(fatal_lineage_names)].astype(bool).all()
    )
    # Costs, capacity, and permutation retrains do not run in this stage.
    return {
        "statistical_ic_gate_pass": statistical_pass,
        "fatal_lineage_gate_pass": fatal_lineage_pass,
        "next_portfolio_diagnostic_eligible": statistical_pass
        and fatal_lineage_pass,
        "portfolio_authorized": False,
        "portfolio_stage": "diagnostic_only_blocked",
    }


def _write_report(
    output_dir: Path,
    period_summary: pd.DataFrame,
    gates: pd.DataFrame,
    config: dict[str, Any],
    data_quality: dict[str, Any],
) -> Path:
    strict_names = {
        "zero_security_overlap",
        "all_training_labels_exited_before_cutoff",
        "fold_local_target_scope",
        "strict_heldout_mean_incremental_rank_ic",
        "strict_heldout_incremental_ic_hac_tstat",
        "positive_security_folds",
        "upstream_membership_frozen_before_label_completion",
    }
    strict_pass = bool(gates.loc[gates["gate"].isin(strict_names), "passed"].all())
    authorization = _authorization_status(gates)
    gate_lines = ["| gate | passed | evidence |", "| --- | --- | --- |"]
    gate_lines.extend(
        f"| {row.gate} | {row.passed} | {row.evidence} |"
        for row in gates.itertuples(index=False)
    )
    period_lines = [
        "| period | months | mean raw IC | mean incremental IC | HAC t | positive months |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    period_lines.extend(
        "| {period} | {months:.0f} | {raw:.4f} | {incremental:.4f} | {hac:.2f} | {positive:.1%} |".format(
            period=row.period,
            months=row.months,
            raw=row.mean_raw_rank_ic,
            incremental=row.mean_incremental_rank_ic,
            hac=row.incremental_rank_ic_hac_tstat,
            positive=row.positive_incremental_month_fraction,
        )
        for row in period_summary.itertuples(index=False)
    )
    primary = period_summary.loc[
        period_summary["period"].eq("pre2026_primary")
    ].iloc[0]
    report_title = config["research"].get(
        "report_title", "Top-150 monotonic GBDT overlay — strict held-out stage"
    )
    candidate_count = _resolve_candidate_count(config)
    selection_description = (
        "all eligible/evaluable rows; no momentum candidate prefilter"
        if candidate_count is None
        else f"Top-{candidate_count} by signal-time momentum"
    )
    report = f"""# {report_title}

## Result

Statistical held-out IC gate: **{'PASS' if strict_pass else 'FAIL'}**.

Portfolio authorization: **BLOCKED**. Fatal lineage pass is
`{authorization['fatal_lineage_gate_pass']}`; this stage does not execute costs,
capacity, or label-permutation retrains.

- Primary gate period: entries before 2026-01-01
- Primary monthly mean raw IC: {primary['mean_raw_rank_ic']:.4f}
- Primary monthly mean incremental IC: {primary['mean_incremental_rank_ic']:.4f}
- Primary incremental IC HAC t-stat: {primary['incremental_rank_ic_hac_tstat']:.2f}
- Primary positive incremental-IC months: {primary['positive_incremental_month_fraction']:.1%}
- Primary evaluated months / fold-months: {int(primary['months'])} / {int(primary['fold_months'])}
- Candidate selection: {selection_description}
- Source mother-universe rows/months/current tickers: {data_quality['rows']:,} / {data_quality['months']} / {data_quality['securities']:,}
- Monthly eligible rows min/median/max: {data_quality['monthly_eligible_minimum']:,} / {data_quality['monthly_eligible_median']:,.0f} / {data_quality['monthly_eligible_maximum']:,}

## Period results

{chr(10).join(period_lines)}

The portfolio stage remains blocked unless every strict held-out gate passes. Fold
scores are evaluated inside their own held-out cross-sections and are not
stitched into a synthetic portfolio.

## Gates

{chr(10).join(gate_lines)}

## Research status

**INVALID** for a survivorship-free or deployable alpha claim. The source panel
uses current tickers/current survivors, lacks delisting returns, and all
historical periods are opened. A statistical pass would only authorize the next
diagnostic review under the frozen {config['research']['strategy_version']}
specification; it cannot authorize a portfolio.
"""
    path = output_dir / "report-source.md"
    path.write_text(report, encoding="utf-8")
    return path


def run(
    config_path: str | Path,
    *,
    invocation_path: str | Path | None = None,
) -> Path:
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not bool(config["research"].get("frozen")):
        raise ValueError("strict ML challenger config must be frozen before execution")
    source_path = ROOT / config["research"]["source_panel"]
    output_dir = ROOT / config["research"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(source_path)
    features = list(config["features"]["columns"])
    data_quality = _validate_source_panel(panel, features)
    lineage_quality, lineage_paths = _validate_source_lineage(config, source_path, panel)
    data_quality.update(lineage_quality)
    data_quality["runtime_versions"] = _runtime_versions()
    candidate_count = _resolve_candidate_count(config)
    candidate_audit = _candidate_universe_audit(
        panel,
        candidate_count=candidate_count,
        security_column=config["universe"]["security_column"],
    )
    data_quality.update(
        {
            "candidate_selection_mode": candidate_audit["selection_mode"].iloc[0],
            "selected_candidate_rows": int(candidate_audit["selected_rows"].sum()),
            "monthly_selected_minimum": int(candidate_audit["selected_rows"].min()),
            "monthly_selected_median": float(candidate_audit["selected_rows"].median()),
            "monthly_selected_maximum": int(candidate_audit["selected_rows"].max()),
            "all_eligible_rows_retained": bool(
                candidate_audit["selected_rows"].eq(candidate_audit["eligible_rows"]).all()
            ),
        }
    )

    model_values = config["model"]
    model_config = MonotonicGBDTConfig(
        max_iter=int(model_values["max_iter"]),
        learning_rate=float(model_values["learning_rate"]),
        max_leaf_nodes=int(model_values["max_leaf_nodes"]),
        l2_regularization=float(model_values["l2_regularization"]),
        min_samples_leaf=int(model_values["min_samples_leaf"]),
        seed=int(config["research"]["seed"]),
    )
    predictions, fit_audit = walk_forward_strict_security_holdout_predict(
        panel,
        features,
        first_prediction=config["research"]["first_prediction"],
        folds=int(config["validation"]["security_holdout_folds"]),
        seed=int(config["research"]["seed"]),
        candidate_count=candidate_count,
        security_column=config["universe"]["security_column"],
        target_column=config["target"]["realized_return_column"],
        control_columns=list(config["target"]["controls"]),
        control_missing_policy=config["target"]["control_missing_policy"],
        minimum_training_rows=int(model_values["minimum_training_rows"]),
        half_life_years=float(model_values["recency_half_life_years"]),
        monotonic_directions=dict(model_values["monotonic_directions"]),
        model_config=model_config,
    )
    evaluations, period_summary, fold_summary = _evaluate_periods(predictions, config)
    fold_month, monthly, full_summary = evaluations["full"]
    primary_fold_month, primary_monthly, primary_summary = evaluations[
        "pre2026_primary"
    ]
    gates = _gate_table(fit_audit, primary_fold_month, primary_summary, config)

    predictions.to_parquet(output_dir / "strict_heldout_predictions.parquet", index=False)
    candidate_audit.to_csv(output_dir / "candidate_universe_monthly.csv", index=False)
    fit_audit.to_csv(output_dir / "strict_heldout_fit_audit.csv", index=False)
    fold_month.to_csv(output_dir / "strict_heldout_fold_month_ic.csv", index=False)
    monthly.to_csv(output_dir / "strict_heldout_monthly_ic.csv", index=False)
    primary_fold_month.to_csv(
        output_dir / "strict_heldout_pre2026_fold_month_ic.csv", index=False
    )
    primary_monthly.to_csv(
        output_dir / "strict_heldout_pre2026_monthly_ic.csv", index=False
    )
    period_summary.to_csv(output_dir / "strict_heldout_period_summary.csv", index=False)
    fold_summary.to_csv(output_dir / "strict_heldout_fold_summary.csv", index=False)
    gates.to_csv(output_dir / "strict_heldout_gates.csv", index=False)
    _write_json(output_dir / "data_quality.json", data_quality)
    _write_json(
        output_dir / "strict_heldout_summary.json",
        {
            **primary_summary,
            "primary_period": "pre2026_primary",
            "primary_entry_end_exclusive": config["validation"][
                "primary_gate_end_exclusive"
            ],
            "full_period_metrics": full_summary,
        },
    )
    authorization = _authorization_status(gates)
    _write_json(
        output_dir / "run_summary.json",
        {
            "strategy_version": config["research"]["strategy_version"],
            "source_panel": config["research"]["source_panel"],
            "source_rows": data_quality["rows"],
            "source_months": data_quality["months"],
            "runtime_versions": data_quality["runtime_versions"],
            "primary_period": "pre2026_primary",
            "primary_mean_incremental_rank_ic": primary_summary[
                "mean_incremental_rank_ic"
            ],
            "primary_incremental_rank_ic_hac_tstat": primary_summary[
                "incremental_rank_ic_hac_tstat"
            ],
            **authorization,
            "diagnostic_classification": config["research"].get(
                "evidence_status", "opened_current_survivor_diagnostic"
            ),
            "historical_backtest_verdict": "INVALID",
            "deployability_verdict": "INVALID",
            "fatal_failed_gates": gates.loc[~gates["passed"], "gate"].tolist(),
        },
    )
    _write_json(
        output_dir / "opened_diagnostic_trial.json",
        {
            "trial_id": config["research"].get(
                "opened_trial_id", config["research"]["strategy_version"]
            ),
            "classification": "opened_diagnostic_trial",
            "source_universe": "current_survivor_proxy",
            "candidate_selection_mode": data_quality["candidate_selection_mode"],
            "historical_backtest_verdict": "INVALID",
            "deployability_verdict": "INVALID",
            "untouched_holdout": False,
        },
    )
    (output_dir / "config_snapshot.yml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    _write_report(output_dir, period_summary, gates, config, data_quality)

    records = []
    resolved_invocation_path = (
        Path(invocation_path).resolve() if invocation_path is not None else Path(__file__).resolve()
    )
    evidence_paths = list(dict.fromkeys([
        config_path,
        source_path,
        *lineage_paths,
        resolved_invocation_path,
        Path(__file__).resolve(),
        ROOT / "src/alpha_factor_engine/research/strict_ml_holdout.py",
        ROOT / "src/alpha_factor_engine/research/metrics.py",
        ROOT / "src/alpha_factor_engine/research/universe_validation.py",
    ]))
    input_paths = {config_path, source_path, *lineage_paths}
    for path in evidence_paths:
        records.append(
            {
                "role": "input" if path in input_paths else "code",
                "path": path.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.json":
            records.append(
                {
                    "role": "output",
                    "path": path.relative_to(ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    _write_json(
        output_dir / "artifact_manifest.json",
        {
            "generated_at": datetime.now(UTC),
            "strategy_version": config["research"]["strategy_version"],
            "records": records,
        },
    )
    return output_dir / "report-source.md"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/liquid_momentum_top150_monotonic_gbdt_overlay_v2.yml",
    )
    arguments = parser.parse_args()
    report = run(arguments.config)
    print(report)


if __name__ == "__main__":
    main()
