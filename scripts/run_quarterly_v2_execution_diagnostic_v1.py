#!/usr/bin/env python3
"""Run the frozen, opened execution diagnostic for quarterly-v2 deciles."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm, spearmanr


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alpha_factor_engine.research.metrics import newey_west_tstat
from alpha_factor_engine.research.portfolio_execution_diagnostic import (
    build_execution_months,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _period_masks(dates: pd.Series) -> dict[str, pd.Series]:
    values = pd.to_datetime(dates, utc=True)
    return {
        "full": pd.Series(True, index=dates.index),
        "pre_2026": values.lt(pd.Timestamp("2026-01-01", tz="UTC")),
        "2026_h1_reporting_only": values.ge(pd.Timestamp("2026-01-01", tz="UTC")),
    }


def _reconcile_gross(monthly: pd.DataFrame, decile_path: Path) -> float:
    expected = pd.read_csv(decile_path)
    expected = expected.loc[
        expected["feature"].eq("quarterly_fundamental_momentum_score")
        & expected["decile"].isin([1, 10])
    ].copy()
    expected["signal_date"] = pd.to_datetime(expected["signal_date"], utc=True)
    pivot = expected.pivot(
        index="signal_date", columns="decile", values="equal_weight_return"
    )
    comparison = monthly.set_index("signal_date")["gross_long_short_return"].to_frame(
        "observed"
    )
    comparison["expected"] = pivot[10] - pivot[1]
    if comparison["expected"].isna().any() or len(comparison) != len(pivot):
        raise ValueError("gross decile reconciliation has missing or extra months")
    error = float((comparison["observed"] - comparison["expected"]).abs().max())
    if error > 1e-12:
        raise ValueError(f"gross decile reconciliation failed: {error}")
    return error


def _summary(monthly: pd.DataFrame, costs_bps: list[int], trial_floor: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for period, mask in _period_masks(monthly["signal_date"]).items():
        sample = monthly.loc[mask].copy()
        if sample.empty:
            continue
        for bps in [0, *costs_bps]:
            column = "gross_long_short_return" if bps == 0 else f"net_return_{bps}bps"
            tstat = float(newey_west_tstat(sample[column]))
            raw_p = float(norm.sf(tstat))
            rows.append(
                {
                    "period": period,
                    "cost_bps_per_dollar_traded": bps,
                    "months": int(len(sample)),
                    "annualized_arithmetic_return": float(12 * sample[column].mean()),
                    "hac_tstat": tstat,
                    "one_sided_hac_pvalue": raw_p,
                    "global_bonferroni_pvalue_floor_112": (
                        min(1.0, raw_p * trial_floor) if bps > 0 else np.nan
                    ),
                    "mean_total_one_way_traded_notional": float(
                        sample["total_one_way_traded_notional"].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _capacity_summary(monthly: pd.DataFrame, participations: list[float]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for participation in participations:
        column = f"capacity_side_usd_at_{participation:.4f}".replace(".", "p")
        for period, mask in _period_masks(monthly["signal_date"]).items():
            values = pd.to_numeric(monthly.loc[mask, column], errors="coerce")
            rows.append(
                {
                    "period": period,
                    "adv_participation": participation,
                    "months": int(mask.sum()),
                    "months_with_complete_capacity": int(values.notna().sum()),
                    "minimum_side_aum_usd": float(values.min()) if values.notna().any() else np.nan,
                    "median_side_aum_usd": float(values.median()) if values.notna().any() else np.nan,
                    "maximum_side_aum_usd": float(values.max()) if values.notna().any() else np.nan,
                    "median_gross_notional_usd": float(2 * values.median()) if values.notna().any() else np.nan,
                }
            )
    return pd.DataFrame(rows)


def run(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    prereg_path = ROOT / "artifacts/quarterly_v2_execution_diagnostic_v1_preregistration.json"
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    if prereg.get("status") != "frozen_before_turnover_and_cost_evaluation":
        raise ValueError("execution diagnostic preregistration is not frozen")
    for record in prereg["records"]:
        path = Path(record["path"])
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file() or _sha256(path) != record["sha256"]:
            raise ValueError(f"preregistration drift: {path}")

    source = config["source"]
    for key, hash_key in [
        ("monthly_panel", "expected_monthly_panel_sha256"),
        ("decile_monthly", "expected_decile_monthly_sha256"),
        ("recovered_manifest", "expected_recovered_manifest_sha256"),
    ]:
        path = ROOT / source[key]
        if _sha256(path) != source[hash_key]:
            raise ValueError(f"source hash drift: {path}")
    panel_path = ROOT / source["monthly_panel"]
    decile_path = ROOT / source["decile_monthly"]
    panel = pd.read_parquet(panel_path)
    output = ROOT / config["outputs"]["directory"]
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)

    participations = [float(value) for value in config["capacity"]["participation_rates"]]
    monthly, holdings = build_execution_months(
        panel,
        score_column=config["portfolio"]["signal"],
        return_column="target_return",
        adv_column=config["capacity"]["adv_column"],
        participations=tuple(participations),
    )
    gross_error = _reconcile_gross(monthly, decile_path)
    costs_bps = [int(value) for value in config["costs"]["scenarios_bps"]]
    for bps in costs_bps:
        monthly[f"cost_{bps}bps"] = (
            monthly["total_one_way_traded_notional"] * bps / 10_000.0
        )
        monthly[f"net_return_{bps}bps"] = (
            monthly["gross_long_short_return"] - monthly[f"cost_{bps}bps"]
        )
    trial_floor = int(config["evaluation"]["resulting_global_trial_floor"])
    summary = _summary(monthly, costs_bps, trial_floor)
    capacity = _capacity_summary(monthly, participations)
    monthly.to_csv(output / "monthly_execution.csv", index=False)
    holdings.to_parquet(output / "holdings.parquet", index=False)
    summary.to_csv(output / "return_summary.csv", index=False)
    capacity.to_csv(output / "capacity_summary.csv", index=False)
    (output / "config_snapshot.yml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    pre = summary.loc[summary["period"].eq("pre_2026")].set_index(
        "cost_bps_per_dollar_traded"
    )
    decile_summary = pd.read_csv(
        ROOT / source["recovered_artifact"] / "decile_summary.csv"
    )
    decile_values = decile_summary.loc[
        decile_summary["feature"].eq(config["portfolio"]["signal"])
        & decile_summary["period"].eq("pre_2026")
        & decile_summary["decile"].astype(str).str.fullmatch(r"\d+")
    ].copy()
    monotonicity = float(
        spearmanr(
            pd.to_numeric(decile_values["decile"]),
            decile_values["annualized_arithmetic_return"],
        ).statistic
    )
    _write_json(
        output / "opened_trial_ledger.json",
        {
            "opened_at": datetime.now(UTC).isoformat(),
            "classification": config["research_classification"],
            "historical_period_already_opened_before_preregistration": True,
            "prior_global_trial_floor": config["evaluation"]["prior_global_trial_floor"],
            "cost_tests_added": len(costs_bps),
            "resulting_global_trial_floor": trial_floor,
            "scenarios_bps": costs_bps,
            "selection_rule": "all frozen scenarios retained; no cost or capacity threshold tuning",
        },
    )
    borrow_complete = bool(config["costs"]["borrow_fee_included"])
    lineage_complete = False
    gross_stat_pass = bool(
        pre.loc[0, "hac_tstat"] >= config["gates"]["net_hac_tstat_minimum"]
    )
    gate_rows = [
        ("gross_result_reconciles_to_frozen_deciles", gross_error <= 1e-12, f"maximum_error={gross_error:.3e}"),
        ("gross_signal_statistical_gate", gross_stat_pass, f"pre2026_HAC_t={pre.loc[0, 'hac_tstat']:.6f}"),
    ]
    for bps in costs_bps:
        row = pre.loc[bps]
        gate_rows.extend(
            [
                (f"net_{bps}bps_return_positive", bool(row["annualized_arithmetic_return"] > 0), f"annualized={row['annualized_arithmetic_return']:.6f}"),
                (f"net_{bps}bps_hac_tstat", bool(row["hac_tstat"] >= config["gates"]["net_hac_tstat_minimum"]), f"observed={row['hac_tstat']:.6f}"),
                (f"net_{bps}bps_global_p", bool(row["global_bonferroni_pvalue_floor_112"] < 0.05), f"observed={row['global_bonferroni_pvalue_floor_112']:.6f}"),
            ]
        )
    gate_rows.extend(
        [
            ("complete_positive_adv", bool(monthly["adv_complete"].all()), f"complete={int(monthly['adv_complete'].sum())}/{len(monthly)}"),
            ("permanent_security_lineage_and_delistings", lineage_complete, "upstream current-survivor ticker panel"),
            ("borrow_fee_and_availability", borrow_complete, "borrow data unavailable"),
            ("market_impact_model", bool(config["costs"]["market_impact_model_included"]), "linear explicit-cost scenarios only"),
            ("portfolio_authorized", False, "gross statistical, lineage, borrow, and impact gates are required"),
        ]
    )
    gates = pd.DataFrame(gate_rows, columns=["gate", "passed", "evidence"])
    gates.to_csv(output / "gates.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    dates = pd.to_datetime(monthly["signal_date"], utc=True)
    ax.plot(dates, (1 + monthly["gross_long_short_return"]).cumprod(), label="Gross", linewidth=2)
    for bps in costs_bps:
        ax.plot(dates, (1 + monthly[f"net_return_{bps}bps"]).cumprod(), label=f"{bps} bps / traded $", linewidth=1.8)
    ax.axhline(1, color="black", linewidth=0.8)
    ax.set_title("Quarterly-v2 D10-D1: opened execution-cost diagnostic")
    ax.set_ylabel("Cumulative wealth (2x gross convention)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "execution_cost_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary_payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "strategy_version": config["strategy_version"],
        "historical_backtest_verdict": "INVALID",
        "research_status": "NOT_VALIDATED",
        "portfolio_authorized": False,
        "months": int(len(monthly)),
        "pre2026_months": int((monthly["signal_date"] < pd.Timestamp("2026-01-01", tz="UTC")).sum()),
        "gross_reconciliation_maximum_error": gross_error,
        "pre2026_gross_annualized": float(pre.loc[0, "annualized_arithmetic_return"]),
        "pre2026_gross_hac_tstat": float(pre.loc[0, "hac_tstat"]),
        "mean_total_one_way_traded_notional": float(monthly["total_one_way_traded_notional"].mean()),
        "pre2026_decile_return_spearman": monotonicity,
        "cost_scenarios": {
            str(bps): {
                "annualized_return": float(pre.loc[bps, "annualized_arithmetic_return"]),
                "hac_tstat": float(pre.loc[bps, "hac_tstat"]),
                "global_bonferroni_pvalue": float(pre.loc[bps, "global_bonferroni_pvalue_floor_112"]),
            }
            for bps in costs_bps
        },
        "fatal_limitations": [
            "opened post-hoc execution diagnostic",
            "gross signal already failed its statistical gate",
            "current-survivor ticker lineage and incomplete delisting returns",
            "borrow fee and locate availability absent",
            "linear transaction-cost scenarios omit nonlinear market impact",
            "capacity is an ADV envelope, not executable order-book capacity",
        ],
    }
    _write_json(output / "run_summary.json", summary_payload)
    report = f"""# Quarterly-v2 execution diagnostic v1

## Verdict

Historical return claim: **INVALID**. Research status: **NOT VALIDATED**. Portfolio authorization: **BLOCKED**.

The signal had already failed before this post-hoc execution diagnostic. The exercise preserves the frozen D10/D1 memberships, includes initial formation, marks prior equal weights to market, and charges each absolute trade at the two preregistered one-way rates. It does not search for a cost assumption.

## Results

- Gross reconciliation maximum error: `{gross_error:.2e}`.
- Mean monthly total traded notional per `$1 long + $1 short`: `{monthly['total_one_way_traded_notional'].mean():.3f}`.
- Pre-2026 gross: `{pre.loc[0, 'annualized_arithmetic_return']:.2%}` annualized arithmetic; HAC t=`{pre.loc[0, 'hac_tstat']:.2f}`.
- 25 bps per traded dollar: `{pre.loc[25, 'annualized_arithmetic_return']:.2%}`; HAC t=`{pre.loc[25, 'hac_tstat']:.2f}`; global p=`{pre.loc[25, 'global_bonferroni_pvalue_floor_112']:.3f}`.
- 50 bps per traded dollar: `{pre.loc[50, 'annualized_arithmetic_return']:.2%}`; HAC t=`{pre.loc[50, 'hac_tstat']:.2f}`; global p=`{pre.loc[50, 'global_bonferroni_pvalue_floor_112']:.3f}`.
- Decile-return monotonicity Spearman: `{monotonicity:.3f}`.

![Cost curves](execution_cost_curves.png)

## Capacity and remaining blockers

`capacity_summary.csv` reports a mechanical one-day ADV envelope at 1% and 5% participation. It uses the strict minimum across both equal-weight sleeves. It excludes borrow availability, locate size, spread, nonlinear impact and security-continuity failures, so it is not deployable capacity. A positive net scenario cannot override the failed gross-statistical, permanent-ID, delisting, borrow and impact gates.
"""
    (output / "report-source.md").write_text(report, encoding="utf-8")

    inputs = [
        ("input", panel_path),
        ("input", decile_path),
        ("input", ROOT / source["recovered_manifest"]),
        ("input", prereg_path),
        ("config", config_path),
        ("code", Path(__file__)),
        ("code", ROOT / "src/alpha_factor_engine/research/portfolio_execution_diagnostic.py"),
        ("code", ROOT / "src/alpha_factor_engine/research/metrics.py"),
        ("test", ROOT / "tests/test_portfolio_execution_diagnostic.py"),
    ]
    records = [
        {"role": role, "path": path.resolve().as_posix(), "bytes": path.stat().st_size, "sha256": _sha256(path)}
        for role, path in inputs
    ]
    records.extend(
        {"role": "output", "path": path.resolve().as_posix(), "bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    _write_json(
        output / "artifact_manifest.json",
        {
            "generated_at": summary_payload["generated_at"],
            "historical_backtest_verdict": "INVALID",
            "portfolio_authorized": False,
            "records": records,
        },
    )
    return output / "report-source.md"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/quarterly_v2_execution_diagnostic_v1.yml",
    )
    args = parser.parse_args()
    print(run(args.config))


if __name__ == "__main__":
    main()
