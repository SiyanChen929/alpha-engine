#!/usr/bin/env python3
"""Run the frozen full-mother-universe feature-family diagnostic."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow
import scipy
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alpha_factor_engine.research.feature_family_diagnostic import (  # noqa: E402
    FUNDAMENTAL_FEATURES,
    apply_primary_screen,
    compute_monthly_diagnostics,
    definitions_from_config,
    feature_redundancy_summary,
    period_feature_summary,
    validate_source_panel,
)
from alpha_factor_engine.research.strict_ml_holdout import (  # noqa: E402
    strict_heldout_information_coefficient,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"cannot JSON serialize {type(value).__name__}")


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


def _pinned_path(
    lineage: dict[str, Any], path_key: str, hash_key: str
) -> tuple[Path, str]:
    path = (ROOT / str(lineage[path_key])).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"pinned input is absent: {path}")
    observed = _sha256(path)
    expected = str(lineage[hash_key])
    if observed != expected:
        raise ValueError(f"pinned input drifted: {path.name}: {observed} != {expected}")
    return path, observed


def _resolve_manifest_record_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _verify_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"manifest contains no records: {path}")
    resolved_paths = [_resolve_manifest_record_path(str(record["path"])) for record in records]
    if len(resolved_paths) != len(set(resolved_paths)):
        raise ValueError(f"manifest contains duplicate paths: {path}")
    failures: list[dict[str, Any]] = []
    for record, record_path in zip(records, resolved_paths):
        if not record_path.is_file():
            failures.append({"path": record_path.as_posix(), "reason": "missing"})
            continue
        observed_bytes = record_path.stat().st_size
        observed_sha256 = _sha256(record_path)
        if observed_bytes != int(record["bytes"]) or observed_sha256 != str(record["sha256"]):
            failures.append(
                {
                    "path": record_path.as_posix(),
                    "reason": "bytes_or_sha256_mismatch",
                    "expected_bytes": int(record["bytes"]),
                    "observed_bytes": observed_bytes,
                    "expected_sha256": str(record["sha256"]),
                    "observed_sha256": observed_sha256,
                }
            )
    if failures:
        raise ValueError(
            f"manifest verification failed for {len(failures)} record(s): {failures[:3]}"
        )
    return {
        "path": path.as_posix(),
        "records": len(records),
        "verified_records": len(records),
        "failed_records": 0,
        "status": "verified",
    }


def _validate_lineage(config: dict[str, Any]) -> tuple[dict[str, Any], list[Path]]:
    lineage = config["source_lineage"]
    if str(lineage["source_panel"]) != str(config["research"]["source_panel"]):
        raise ValueError("research and lineage source-panel paths disagree")
    mappings = (
        ("source_panel", "expected_source_sha256"),
        ("upstream_manifest", "expected_upstream_manifest_sha256"),
        ("upstream_universe_audit", "expected_upstream_universe_audit_sha256"),
        ("upstream_opened_ledger", "expected_upstream_opened_ledger_sha256"),
        ("full_mother_predictions", "expected_full_mother_predictions_sha256"),
        ("full_mother_summary", "expected_full_mother_summary_sha256"),
        ("prior_public_factor_summary", "expected_prior_public_factor_summary_sha256"),
        ("monthly_feature_panel", "expected_monthly_feature_panel_sha256"),
        ("fundamental_source", "expected_fundamental_source_sha256"),
    )
    resolved: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for path_key, hash_key in mappings:
        resolved[path_key], hashes[path_key] = _pinned_path(lineage, path_key, hash_key)
    upstream_manifest_audit = _verify_manifest(resolved["upstream_manifest"])
    full_mother_manifest = resolved["full_mother_predictions"].parent / "artifact_manifest.json"
    if not full_mother_manifest.is_file():
        raise FileNotFoundError("full-mother diagnostic manifest is absent")
    expected_full_manifest = lineage.get("expected_full_mother_manifest_sha256")
    if expected_full_manifest is not None and _sha256(full_mother_manifest) != str(expected_full_manifest):
        raise ValueError("full-mother diagnostic manifest hash drifted")
    full_mother_manifest_audit = _verify_manifest(full_mother_manifest)

    upstream_manifest = json.loads(resolved["upstream_manifest"].read_text(encoding="utf-8"))
    source_records = [
        record
        for record in upstream_manifest["records"]
        if _resolve_manifest_record_path(str(record["path"])) == resolved["source_panel"]
    ]
    if len(source_records) != 1 or str(source_records[0]["sha256"]) != hashes["source_panel"]:
        raise ValueError("source panel is not uniquely hash-bound by upstream manifest")

    audit = pd.read_csv(resolved["upstream_universe_audit"])
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
    evaluable = audit.loc[audit["evaluation_status"].eq("evaluable")]
    if len(evaluable) != int(lineage["expected_source_months"]):
        raise ValueError("universe audit evaluable months drifted")
    if int(evaluable["selected_rows"].sum()) != int(lineage["expected_source_rows"]):
        raise ValueError("universe audit evaluable row count drifted")
    if int(evaluable["selected_missing_labels"].sum()) != 0:
        raise ValueError("universe audit reports missing labels in evaluation panel")
    future = audit.loc[~audit["evaluation_status"].eq("evaluable")]
    membership_before_labels = bool(
        not future.empty
        and future["signal_status"].eq("kept").all()
        and future["selected_missing_labels"].gt(0).any()
    )
    if not membership_before_labels:
        raise ValueError("upstream audit lacks pre-label membership evidence")

    opened = pd.read_csv(resolved["upstream_opened_ledger"])
    if len(opened) != 72:
        raise ValueError("upstream opened-search lower-bound ledger drifted")
    evidence = {
        "hashes": hashes,
        "upstream_manifest": upstream_manifest_audit,
        "full_mother_manifest": full_mother_manifest_audit,
        "source_uniquely_bound_in_upstream_manifest": True,
        "membership_selected_before_label_completion": True,
        "evaluable_months_reconciled": int(len(evaluable)),
        "evaluable_rows_reconciled": int(evaluable["selected_rows"].sum()),
        "future_nonevaluable_months_retained": int(len(future)),
        "upstream_opened_ledger_rows": int(len(opened)),
    }
    return evidence, [*resolved.values(), full_mother_manifest]


def _reconcile_full_mother_ml(
    source: pd.DataFrame,
    predictions_path: Path,
    summary_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    predictions = pd.read_parquet(predictions_path)
    lineage = config["source_lineage"]
    if len(predictions) != int(lineage["expected_full_mother_prediction_rows"]):
        raise ValueError("full-mother prediction row count drifted")
    predictions["signal_date"] = pd.to_datetime(
        predictions["signal_date"], utc=True, errors="raise"
    )
    if predictions["signal_date"].nunique() != int(
        lineage["expected_full_mother_prediction_months"]
    ):
        raise ValueError("full-mother prediction month count drifted")
    if predictions.duplicated(["signal_date", "ticker"]).any():
        raise ValueError("full-mother predictions duplicate decision/security keys")
    source_target = source[["signal_date", "ticker", "target_return"]].rename(
        columns={"target_return": "source_target_return"}
    )
    reconciled = predictions.merge(
        source_target,
        on=["signal_date", "ticker"],
        how="left",
        validate="one_to_one",
    )
    if reconciled["source_target_return"].isna().any():
        raise ValueError("full-mother prediction key is absent from source panel")
    target_error = np.abs(
        reconciled["target_return"].to_numpy(float)
        - reconciled["source_target_return"].to_numpy(float)
    )
    if float(target_error.max()) > 1e-12:
        raise ValueError("full-mother prediction labels differ from source panel")

    _, _, full = strict_heldout_information_coefficient(predictions)
    entry = pd.to_datetime(predictions["entry_at"], utc=True, errors="raise")
    pre = predictions.loc[entry.lt(pd.Timestamp("2026-01-01", tz="UTC"))].copy()
    _, _, primary = strict_heldout_information_coefficient(pre)
    saved = json.loads(summary_path.read_text(encoding="utf-8"))
    comparisons = {
        "mean_raw_rank_ic": (
            primary["mean_raw_rank_ic"],
            saved["mean_raw_rank_ic"],
        ),
        "mean_incremental_rank_ic": (
            primary["mean_incremental_rank_ic"],
            saved["mean_incremental_rank_ic"],
        ),
        "incremental_rank_ic_hac_tstat": (
            primary["incremental_rank_ic_hac_tstat"],
            saved["incremental_rank_ic_hac_tstat"],
        ),
        "full_mean_incremental_rank_ic": (
            full["mean_incremental_rank_ic"],
            saved["full_period_metrics"]["mean_incremental_rank_ic"],
        ),
    }
    maximum_error = max(abs(float(left) - float(right)) for left, right in comparisons.values())
    if maximum_error > 1e-12:
        raise ValueError("full-mother ML summary does not independently recompute")
    return {
        "prediction_rows": int(len(predictions)),
        "prediction_months": int(predictions["signal_date"].nunique()),
        "prediction_first_signal_date": predictions["signal_date"].min(),
        "prediction_last_signal_date": predictions["signal_date"].max(),
        "prediction_keys_subset_of_source": True,
        "target_return_max_abs_reconciliation_error": float(target_error.max()),
        "saved_summary_max_abs_recomputation_error": maximum_error,
        "pre2026_months": int(primary["months"]),
        "pre2026_mean_raw_rank_ic": float(primary["mean_raw_rank_ic"]),
        "pre2026_mean_incremental_rank_ic": float(primary["mean_incremental_rank_ic"]),
        "pre2026_incremental_rank_ic_hac_tstat": float(
            primary["incremental_rank_ic_hac_tstat"]
        ),
        "full_mean_incremental_rank_ic": float(full["mean_incremental_rank_ic"]),
        "reconciliation_status": "verified",
    }


def _direction_registry(definitions: Iterable[Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in definitions:
        rows.append(
            {
                "feature_id": item.feature_id,
                "family": item.family,
                "definition_kind": item.kind,
                "components": ";".join(
                    f"{name}:{direction:+d}" for name, direction in item.components
                ),
                "minimum_components": item.minimum_components,
                "direction_status": item.direction_status,
                "untouched_holdout": False,
            }
        )
    return pd.DataFrame(rows)


def _family_summary(screen: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    selected = summary.merge(
        screen[["feature_id", "statistical_screen_pass"]],
        on="feature_id",
        how="left",
        validate="many_to_one",
    )
    rows: list[dict[str, Any]] = []
    for (period, family), group in selected.groupby(["period", "family"], sort=True):
        rows.append(
            {
                "period": period,
                "family": family,
                "feature_definitions": int(group["feature_id"].nunique()),
                "median_mean_oriented_rank_ic": float(
                    group["mean_oriented_rank_ic"].median()
                ),
                "median_mean_momentum_orthogonal_rank_ic": float(
                    group["mean_momentum_orthogonal_rank_ic"].median()
                ),
                "maximum_mean_momentum_orthogonal_rank_ic": float(
                    group["mean_momentum_orthogonal_rank_ic"].max()
                ),
                "minimum_monthly_coverage": float(
                    group["minimum_monthly_coverage"].min()
                ),
                "primary_statistical_screen_pass_count": int(
                    group["statistical_screen_pass"].fillna(False).sum()
                )
                if period == "pre_2026_primary"
                else 0,
                "research_verdict": "NOT VALIDATED",
            }
        )
    return pd.DataFrame(rows)


def _regime_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "feature_id",
        "family",
        "mean_coverage",
        "mean_oriented_rank_ic",
        "oriented_rank_ic_hac_tstat",
        "mean_momentum_orthogonal_rank_ic",
        "momentum_orthogonal_rank_ic_hac_tstat",
        "mean_momentum_orthogonal_winsorized_decile_spread",
    ]
    pre = summary.loc[summary["period"].eq("pre_2026_primary"), columns].copy()
    h1 = summary.loc[summary["period"].eq("2026_h1_reporting_only"), columns].copy()
    merged = pre.merge(
        h1,
        on=["feature_id", "family"],
        suffixes=("_pre2026", "_2026h1"),
        validate="one_to_one",
    )
    merged["orthogonal_ic_sign_same"] = np.sign(
        merged["mean_momentum_orthogonal_rank_ic_pre2026"]
    ).eq(np.sign(merged["mean_momentum_orthogonal_rank_ic_2026h1"]))
    return merged


def _prior_public_comparison(
    prior_path: Path, period_summary: pd.DataFrame
) -> pd.DataFrame:
    prior = pd.read_csv(prior_path).rename(
        columns={
            "factor": "feature_id",
            "months": "prior_opened_months",
            "mean_coverage": "prior_opened_mean_coverage",
            "mean_oriented_ic": "prior_opened_mean_oriented_ic",
            "ic_newey_west_tstat": "prior_opened_ic_hac_tstat",
        }
    )
    current = period_summary.loc[
        period_summary["period"].eq("pre_2026_primary")
        & period_summary["feature_id"].isin(FUNDAMENTAL_FEATURES),
        [
            "feature_id",
            "calendar_months",
            "mean_coverage",
            "mean_oriented_rank_ic",
            "oriented_rank_ic_hac_tstat",
            "mean_momentum_orthogonal_rank_ic",
            "momentum_orthogonal_rank_ic_hac_tstat",
        ],
    ].copy()
    output = current.merge(prior, on="feature_id", how="left", validate="one_to_one")
    output["prior_window_note"] = "opened_41_month_public_holdout_including_2026h1_not_comparable"
    output["current_window_note"] = "opened_84_month_pre2026_full_mother_proxy"
    return output


def _charts(output_dir: Path, screen: pd.DataFrame, regime: pd.DataFrame) -> list[Path]:
    chart_dir = output_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "price": "#2f6f9f",
        "liquidity_size": "#4f9d69",
        "volatility_risk": "#d28b36",
        "fundamental": "#8b5fbf",
    }
    ordered = screen.sort_values("screen_mean_ic")
    fig, ax = plt.subplots(figsize=(11, 12))
    positions = np.arange(len(ordered))
    ax.barh(
        positions,
        ordered["mean_oriented_rank_ic"],
        color=[colors.get(family, "#777777") for family in ordered["family"]],
        alpha=0.35,
        label="raw oriented IC",
    )
    ax.scatter(
        ordered["screen_mean_ic"],
        positions,
        color=[colors.get(family, "#777777") for family in ordered["family"]],
        s=24,
        label="screen IC (momentum-orthogonal; raw for momentum benchmark)",
        zorder=3,
    )
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_yticks(positions, ordered["feature_id"])
    ax.set_xlabel("Mean monthly Rank IC, pre-2026")
    ax.set_title("Opened full-mother feature screen: raw versus momentum-orthogonal IC")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    ic_path = chart_dir / "pre2026_raw_vs_orthogonal_ic.png"
    fig.savefig(ic_path, dpi=180)
    plt.close(fig)

    fundamental = screen.loc[screen["family"].eq("fundamental")].sort_values(
        "minimum_monthly_coverage"
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    positions = np.arange(len(fundamental))
    ax.barh(positions, fundamental["mean_coverage"], color="#8b5fbf", alpha=0.8)
    ax.scatter(
        fundamental["minimum_monthly_coverage"], positions, color="black", s=20, label="minimum month"
    )
    ax.set_yticks(positions, fundamental["feature_id"])
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Signal-time coverage")
    ax.set_title("Pre-2026 fundamental coverage")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    coverage_path = chart_dir / "pre2026_fundamental_coverage.png"
    fig.savefig(coverage_path, dpi=180)
    plt.close(fig)

    regime_order = regime.sort_values("mean_momentum_orthogonal_rank_ic_pre2026")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        regime_order["mean_momentum_orthogonal_rank_ic_pre2026"],
        regime_order["mean_momentum_orthogonal_rank_ic_2026h1"],
        color=[colors.get(family, "#777777") for family in regime_order["family"]],
        alpha=0.85,
    )
    for row in regime_order.itertuples(index=False):
        ax.annotate(
            row.feature_id,
            (
                row.mean_momentum_orthogonal_rank_ic_pre2026,
                row.mean_momentum_orthogonal_rank_ic_2026h1,
            ),
            fontsize=6,
            alpha=0.8,
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Pre-2026 mean momentum-orthogonal IC (84 months)")
    ax.set_ylabel("2026H1 mean momentum-orthogonal IC (6 months)")
    ax.set_title("Regime comparison; 2026H1 is reporting-only")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    regime_path = chart_dir / "orthogonal_ic_regime_comparison.png"
    fig.savefig(regime_path, dpi=180)
    plt.close(fig)
    return [ic_path, coverage_path, regime_path]


def _report(
    output_dir: Path,
    config: dict[str, Any],
    source_quality: dict[str, Any],
    screen: pd.DataFrame,
    summary: pd.DataFrame,
    redundancy: pd.DataFrame,
    ml_reference: dict[str, Any],
) -> Path:
    display = screen.sort_values("screen_mean_ic", ascending=False)
    feature_lines = [
        "| feature | family | min coverage | raw IC | orthogonal IC | orthogonal HAC t | block p | BH q | Bonferroni p | robust decile spread | screen |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in display.itertuples(index=False):
        orthogonal = row.mean_momentum_orthogonal_rank_ic
        orthogonal_text = f"{orthogonal:.4f}" if np.isfinite(orthogonal) else "n/a"
        orthogonal_t = row.momentum_orthogonal_rank_ic_hac_tstat
        orthogonal_t_text = f"{orthogonal_t:.2f}" if np.isfinite(orthogonal_t) else "n/a"
        feature_lines.append(
            f"| {row.feature_id} | {row.family} | {row.minimum_monthly_coverage:.1%} | "
            f"{row.mean_oriented_rank_ic:.4f} | {orthogonal_text} | {orthogonal_t_text} | "
            f"{row.screen_month_block_pvalue:.3f} | {row.screen_bh_qvalue_28:.3f} | "
            f"{row.screen_family_bonferroni_pvalue_105:.3f} | "
            f"{row.screen_winsorized_decile_spread:.3%} | {bool(row.statistical_screen_pass)} |"
        )

    h1 = summary.loc[summary["period"].eq("2026_h1_reporting_only")].copy()
    comparison = screen[["feature_id", "mean_momentum_orthogonal_rank_ic"]].merge(
        h1[["feature_id", "mean_momentum_orthogonal_rank_ic"]],
        on="feature_id",
        suffixes=("_pre2026", "_2026h1"),
        validate="one_to_one",
    )
    comparable = comparison.dropna()
    same_sign = int(
        np.sign(comparable["mean_momentum_orthogonal_rank_ic_pre2026"]).eq(
            np.sign(comparable["mean_momentum_orthogonal_rank_ic_2026h1"])
        ).sum()
    )
    screened = display.loc[display["statistical_screen_pass"]]
    top_incremental = display.loc[
        ~display["feature_id"].eq(config["evaluation"]["momentum_control"])
    ].iloc[0]
    high_redundancy = redundancy.loc[
        redundancy["mean_absolute_monthly_rank_correlation"].ge(0.90)
    ]
    redundancy_lines = [
        "| feature pair | mean monthly rank correlation | months |",
        "| --- | ---: | ---: |",
    ]
    for row in redundancy.head(10).itertuples(index=False):
        redundancy_lines.append(
            f"| {row.feature_left} / {row.feature_right} | "
            f"{row.mean_monthly_rank_correlation:.3f} | {row.months} |"
        )

    report = f"""# {config['research']['report_title']}

## Verdict

Research verdict: **NOT VALIDATED**.

Survivorship-free alpha and deployability claims: **INVALID**. Portfolio
authorization is `false`.

The diagnostic covers {len(display)} frozen feature hypotheses across
{source_quality['rows']:,} stock-months, {source_quality['months']} decision
months, and {source_quality['securities']:,} current tickers. Pre-2026 contains
84 months; 2026H1 contains six reporting-only months. Directions are explicitly
opened and literature-signed. No target is accepted by the score-construction
function.

No feature passed the complete frozen screen after the disclosed family lower
bound of {config['multiple_testing']['current_disclosed_family_size_minimum']}
tests. Screened rows: {len(screened)}. The highest opened non-momentum conditional
mean IC was `{top_incremental.feature_id}` at {top_incremental.screen_mean_ic:.4f}
(HAC t={top_incremental.screen_hac_tstat:.2f}, block p={top_incremental.screen_month_block_pvalue:.3f},
family Bonferroni p={top_incremental.screen_family_bonferroni_pvalue_105:.3f}).
This ranking is descriptive and cannot select a new model.

The saved strict full-mother GBDT result independently reconciles: pre-2026
incremental IC {ml_reference['pre2026_mean_incremental_rank_ic']:.4f}, HAC t
{ml_reference['pre2026_incremental_rank_ic_hac_tstat']:.2f}; target and saved
summary maximum absolute errors are
{ml_reference['target_return_max_abs_reconciliation_error']:.1e} and
{ml_reference['saved_summary_max_abs_recomputation_error']:.1e}.

## Pre-2026 feature results

{chr(10).join(feature_lines)}

`orthogonal IC` is the monthly partial rank correlation after residualizing
both the oriented feature rank and realized-return rank on exact 12-1 momentum
inside that evaluation month. The momentum benchmark has no self-orthogonal
metric. Decile spreads are gross next-open to next-rebalance-open diagnostics;
the screen uses monthly 1/99 winsorized returns because the raw panel contains
{source_quality['target_return_above_100pct_rows']} observations above +100%,
including a maximum of {source_quality['target_return_maximum']:.1%}. Both raw
and winsorized values are retained in the evidence files.

![Raw and orthogonal IC](charts/pre2026_raw_vs_orthogonal_ic.png)

## Coverage and redundancy

All ten requested fundamental factors are present for all 90 months, but
coverage differs materially. The lowest minimum monthly coverage is visible in
the chart and `coverage_by_month.csv`. Missing values are never filled using
future or peer accounting values. The source uses annual XBRL records rather
than quarterly/LTM fundamentals. Median populated-record age is
{source_quality['fundamental_age_days_median']:.0f} days; the maximum is
{source_quality['fundamental_age_days_maximum']:.0f} days, and
{source_quality['fundamental_age_above_365_days_rows']} stock-months exceed 365
days. These observations remain visible in the audit and are not silently
dropped after outcomes are known.

![Fundamental coverage](charts/pre2026_fundamental_coverage.png)

There are {len(high_redundancy)} feature pairs with mean absolute monthly rank
correlation at least 0.90. This reduces the effective information breadth fed
to nonlinear models.

{chr(10).join(redundancy_lines)}

## Regime separation

Only {same_sign}/{len(comparable)} defined momentum-orthogonal feature results
have the same mean sign in pre-2026 and 2026H1. Six months are insufficient for
an independent validation claim.

![Regime comparison](charts/orthogonal_ic_regime_comparison.png)

## Integrity and limits

- Source membership was selected before checking future-label completion; all
  90 evaluation months have complete labels and exact next-open price
  reconciliation.
- Price, residual-momentum, and populated SEC fundamental availability times
  do not exceed the decision clock.
- The source is a current-registrant/current-ticker proxy, with no permanent
  security identifier or complete delisting return history.
- This diagnostic has no security-master reconstruction, transaction costs,
  borrow, capacity, position ledger, or tradable portfolio.
- The 28 directions and composites are opened hypotheses. BH across 28 and a
  conservative Bonferroni adjustment across the disclosed lower bound of 105
  are both reported.

Smallest next evidence step: rebuild the same frozen definitions on a
survivorship-free permanent-security panel, reserve new dates, and require
consistent positive momentum-orthogonal IC before any portfolio experiment.
"""
    path = output_dir / "report-source.md"
    path.write_text(report, encoding="utf-8")
    return path


def _manifest(
    output_dir: Path,
    config_path: Path,
    input_paths: Iterable[Path],
    invocation_path: Path,
) -> Path:
    code_paths = [
        invocation_path,
        Path(__file__).resolve(),
        ROOT / "src/alpha_factor_engine/research/feature_family_diagnostic.py",
        ROOT / "src/alpha_factor_engine/research/strict_ml_holdout.py",
        ROOT / "src/alpha_factor_engine/research/metrics.py",
        ROOT / "src/alpha_factor_engine/features/fundamentals.py",
        ROOT / "src/alpha_factor_engine/features/price.py",
    ]
    inputs = list(dict.fromkeys([config_path.resolve(), *(path.resolve() for path in input_paths)]))
    codes = list(dict.fromkeys(path.resolve() for path in code_paths))
    records: list[dict[str, Any]] = []
    for role, paths in (("input", inputs), ("code", codes)):
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"manifest evidence path is absent: {path}")
            records.append(
                {
                    "role": role,
                    "path": path.as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            records.append(
                {
                    "role": "output",
                    "path": path.relative_to(ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    manifest_path = output_dir / "artifact_manifest.json"
    _write_json(
        manifest_path,
        {
            "generated_at": datetime.now(UTC),
            "strategy_version": "liquid_full_mother_feature_family_diagnostic_v1",
            "research_verdict": "NOT VALIDATED",
            "survivorship_free_alpha_claim": "INVALID",
            "deployability_verdict": "INVALID",
            "portfolio_authorized": False,
            "records": records,
        },
    )
    return manifest_path


def run(config_path: str | Path, *, invocation_path: str | Path | None = None) -> Path:
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not bool(config["research"].get("frozen")):
        raise ValueError("feature-family diagnostic config must be frozen")
    if config["direction_policy"]["status"] != "opened_literature_signed_frozen_before_this_runner":
        raise ValueError("direction policy must disclose this opened diagnostic")
    if bool(config["direction_policy"].get("untouched_holdout")):
        raise ValueError("opened diagnostic cannot claim an untouched holdout")
    definitions = definitions_from_config(config)
    individual = [item for item in definitions if item.kind == "individual"]
    fundamental = [
        item for item in individual if item.feature_id in set(FUNDAMENTAL_FEATURES)
    ]
    if len(definitions) != int(config["multiple_testing"]["new_feature_hypotheses"]):
        raise ValueError("feature hypothesis count differs from multiplicity disclosure")
    if len(fundamental) != 10 or {item.feature_id for item in fundamental} != set(FUNDAMENTAL_FEATURES):
        raise ValueError("diagnostic must contain exactly the ten requested fundamentals")
    if int(config["multiple_testing"]["current_disclosed_family_size_minimum"]) < (
        int(config["multiple_testing"]["upstream_disclosed_family_size_minimum"])
        + len(definitions)
    ):
        raise ValueError("multiple-testing family omits new feature hypotheses")

    lineage, input_paths = _validate_lineage(config)
    source_path = (ROOT / config["research"]["source_panel"]).resolve()
    panel = pd.read_parquet(source_path)
    panel, source_quality = validate_source_panel(
        panel,
        definitions,
        expected_rows=int(config["source_lineage"]["expected_source_rows"]),
        expected_months=int(config["source_lineage"]["expected_source_months"]),
        expected_securities=int(config["source_lineage"]["expected_source_securities"]),
        security_column=config["universe"]["security_column"],
        target_column=config["clock"]["target_column"],
    )
    source_quality["runtime_versions"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
        "scipy": scipy.__version__,
    }
    source_quality["source_sha256"] = lineage["hashes"]["source_panel"]

    evaluation = config["evaluation"]
    monthly, coverage, deciles = compute_monthly_diagnostics(
        panel,
        definitions,
        momentum_column=evaluation["momentum_control"],
        security_column=config["universe"]["security_column"],
        target_column=config["clock"]["target_column"],
        minimum_cross_section=int(evaluation["minimum_cross_section"]),
        deciles=int(evaluation["deciles"]),
        target_winsor_lower=float(evaluation["target_winsor_lower"]),
        target_winsor_upper=float(evaluation["target_winsor_upper"]),
    )
    summary = period_feature_summary(
        monthly,
        evaluation["periods"],
        simulations=int(evaluation["month_block_simulations"]),
        base_seed=int(config["research"]["seed"]),
    )
    screen, gates = apply_primary_screen(
        summary,
        primary_period="pre_2026_primary",
        momentum_feature=evaluation["momentum_control"],
        gates=config["screen_gates"],
        family_size_minimum=int(
            config["multiple_testing"]["current_disclosed_family_size_minimum"]
        ),
    )
    redundancy = feature_redundancy_summary(
        panel,
        definitions,
        minimum_pair_names=int(evaluation["minimum_cross_section"]),
        security_column=config["universe"]["security_column"],
    )
    family = _family_summary(screen, summary)
    regime = _regime_comparison(summary)
    ml_reference = _reconcile_full_mother_ml(
        panel,
        (ROOT / config["source_lineage"]["full_mother_predictions"]).resolve(),
        (ROOT / config["source_lineage"]["full_mother_summary"]).resolve(),
        config,
    )
    prior = _prior_public_comparison(
        (ROOT / config["source_lineage"]["prior_public_factor_summary"]).resolve(),
        summary,
    )
    registry = _direction_registry(definitions)
    opened = registry.copy()
    opened.insert(
        0,
        "disclosed_family_index",
        np.arange(
            int(config["multiple_testing"]["upstream_disclosed_family_size_minimum"]) + 1,
            int(config["multiple_testing"]["current_disclosed_family_size_minimum"]) + 1,
        ),
    )
    opened["primary_metric"] = np.where(
        opened["feature_id"].eq(evaluation["momentum_control"]),
        "monthly_oriented_rank_ic",
        "monthly_momentum_orthogonal_rank_ic",
    )
    opened["outcome_status"] = "opened_diagnostic_observed"
    opened["research_verdict"] = "NOT VALIDATED"
    opened["portfolio_authorized"] = False

    outlier_columns = [
        "signal_date",
        "ticker",
        "entry_at",
        "exit_at",
        "entry_price",
        "exit_price",
        "target_return",
    ]
    target_outliers = panel.assign(
        absolute_target_return=panel["target_return"].abs()
    ).nlargest(100, "absolute_target_return")[outlier_columns + ["absolute_target_return"]]
    age = pd.to_numeric(panel["fundamental_age_days"], errors="coerce")
    populated = panel[list(FUNDAMENTAL_FEATURES) + ["log_market_cap"]].notna().any(axis=1)
    staleness_monthly = (
        panel.loc[populated]
        .assign(
            fundamental_age_days_numeric=age.loc[populated],
            above_365_days=age.loc[populated].gt(365),
            above_730_days=age.loc[populated].gt(730),
        )
        .groupby("signal_date", sort=True)
        .agg(
            populated_fundamental_rows=("ticker", "size"),
            median_age_days=("fundamental_age_days_numeric", "median"),
            p95_age_days=(
                "fundamental_age_days_numeric",
                lambda values: values.quantile(0.95),
            ),
            maximum_age_days=("fundamental_age_days_numeric", "max"),
            above_365_days_rows=("above_365_days", "sum"),
            above_730_days_rows=("above_730_days", "sum"),
        )
        .reset_index()
    )
    staleness_outliers = (
        panel.loc[populated]
        .assign(fundamental_age_days_numeric=age.loc[populated])
        .nlargest(100, "fundamental_age_days_numeric")
        [
            [
                "signal_date",
                "ticker",
                "fundamental_available_at",
                "fundamental_fiscal_end",
                "fundamental_age_days_numeric",
                "debt_definition",
            ]
        ]
    )

    output_dir = (ROOT / config["research"]["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_snapshot.yml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    monthly.to_parquet(output_dir / "monthly_feature_diagnostics.parquet", index=False)
    monthly.to_csv(output_dir / "monthly_feature_diagnostics.csv", index=False)
    coverage.to_csv(output_dir / "coverage_by_month.csv", index=False)
    deciles.to_parquet(output_dir / "decile_returns_monthly.parquet", index=False)
    summary.to_csv(output_dir / "period_feature_summary.csv", index=False)
    screen.to_csv(output_dir / "pre2026_primary_screen.csv", index=False)
    gates.to_csv(output_dir / "feature_screen_gates.csv", index=False)
    redundancy.to_csv(output_dir / "feature_redundancy.csv", index=False)
    family.to_csv(output_dir / "family_summary.csv", index=False)
    regime.to_csv(output_dir / "period_regime_comparison.csv", index=False)
    prior.to_csv(output_dir / "prior_public_factor_comparison.csv", index=False)
    registry.to_csv(output_dir / "direction_registry.csv", index=False)
    opened.to_csv(output_dir / "opened_feature_hypotheses.csv", index=False)
    target_outliers.to_csv(output_dir / "target_outlier_audit.csv", index=False)
    staleness_monthly.to_csv(
        output_dir / "fundamental_staleness_by_month.csv", index=False
    )
    staleness_outliers.to_csv(
        output_dir / "fundamental_staleness_outliers.csv", index=False
    )
    _write_json(output_dir / "source_lineage_audit.json", lineage)
    _write_json(output_dir / "data_quality.json", source_quality)
    _write_json(output_dir / "ml_reference_reconciliation.json", ml_reference)
    _charts(output_dir, screen, regime)
    _report(
        output_dir,
        config,
        source_quality,
        screen,
        summary,
        redundancy,
        ml_reference,
    )
    run_summary = {
        "strategy_version": config["research"]["strategy_version"],
        "research_verdict": "NOT VALIDATED",
        "survivorship_free_alpha_claim": "INVALID",
        "deployability_verdict": "INVALID",
        "portfolio_authorized": False,
        "direction_status": config["direction_policy"]["status"],
        "source_rows": source_quality["rows"],
        "source_months": source_quality["months"],
        "source_securities": source_quality["securities"],
        "feature_hypotheses": len(definitions),
        "requested_fundamental_features": len(fundamental),
        "disclosed_multiple_testing_family_size_minimum": int(
            config["multiple_testing"]["current_disclosed_family_size_minimum"]
        ),
        "pre2026_statistical_screen_pass_count": int(
            screen["statistical_screen_pass"].sum()
        ),
        "pre2026_statistical_screen_pass_features": screen.loc[
            screen["statistical_screen_pass"], "feature_id"
        ].tolist(),
        "full_mother_ml_reconciliation_status": ml_reference["reconciliation_status"],
        "limitations": [
            "opened_historical_diagnostic",
            "current_survivor_current_ticker_proxy",
            "no_complete_delisting_returns",
            "no_permanent_security_identifier",
            "no_cost_borrow_capacity_or_position_ledger",
        ],
    }
    _write_json(output_dir / "run_summary.json", run_summary)

    invocation = (
        Path(invocation_path).resolve()
        if invocation_path is not None
        else Path(__file__).resolve()
    )
    _manifest(output_dir, config_path, input_paths, invocation)
    return output_dir / "report-source.md"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/liquid_full_mother_feature_family_diagnostic_v1.yml"),
    )
    args = parser.parse_args()
    print(run(args.config, invocation_path=Path(__file__).resolve()))


if __name__ == "__main__":
    main()
