#!/usr/bin/env python3
"""Run the frozen tradability-first post-outcome diagnostic."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alpha_factor_engine.research.champion_challenger import capacity_diagnostics
from alpha_factor_engine.research.sec_filing_change_stage_a import calendar_portfolio_returns


CONFIG = ROOT / "configs/sec_reporting_lag_tradable_stage_b_v1.json"
PLAN = ROOT / "artifacts/sec_reporting_lag_tradable_stage_b_v1/preoutcome_plan_v1"
DESIGN = ROOT / "artifacts/sec_reporting_lag_tradable_stage_b_v1/design_receipt_v1.json"
PRICES = ROOT / "artifacts/public_broad_proxy_v1/data/daily_prices.parquet"
OUTPUT = ROOT / "artifacts/sec_reporting_lag_tradable_stage_b_v1/run_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path, role: str) -> dict[str, object]:
    path = path.resolve()
    return {"role": role, "path": path.relative_to(ROOT.resolve()).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n", encoding="utf-8")


def verify_records(records: list[dict[str, object]]) -> None:
    for item in records:
        path = ROOT / item["path"]
        if not path.is_file() or path.stat().st_size != int(item["bytes"]) or sha256_file(path) != item["sha256"]:
            raise ValueError(f"frozen record mismatch: {path}")


def phase_summary(monthly: pd.DataFrame, holdings: pd.DataFrame) -> dict[str, object]:
    frame = monthly.loc[monthly["portfolio_scope"].eq("ALL")].sort_values("month")
    components = holdings.loc[holdings["portfolio_scope"].eq("ALL")]
    result: dict[str, object] = {
        "months": int(len(frame)), "signal_events": int(components["current_accession"].nunique()),
        "unique_cik": int(components["cik"].nunique()), "average_stocks": float(frame["stocks"].mean()),
        "average_turnover": float(frame["total_turnover"].mean()),
        "gross_mean": float(frame["q5_minus_q1_gross_return"].mean()),
        "net30_mean": float(frame["q5_minus_q1_net30_return"].mean()),
        "net50_mean": float(frame["q5_minus_q1_net50_return"].mean()),
        "gross_terminal_wealth": float((1 + frame["q5_minus_q1_gross_return"]).prod()),
        "net30_terminal_wealth": float((1 + frame["q5_minus_q1_net30_return"]).prod()),
        "positive_gross_months": int(frame["q5_minus_q1_gross_return"].gt(0).sum()),
        "positive_net30_months": int(frame["q5_minus_q1_net30_return"].gt(0).sum()),
        "maximum_absolute_target_weight": float(frame["maximum_absolute_target_weight"].max()),
        "inference_status": "DESCRIPTIVE_ONLY_POST_OUTCOME_METHOD_REPAIR",
        "p_value": None,
    }
    return result


def capacity_input(holdings: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    all_holdings = holdings.loc[holdings["portfolio_scope"].eq("ALL")].copy()
    prior = schedule[["current_accession", "prior_adv_usd"]]
    all_holdings = all_holdings.merge(prior, on="current_accession", validate="many_to_one")
    target = (
        all_holdings.groupby(["month", "entry_date", "ticker"], sort=True)
        .agg(weight=("target_weight_component", "sum"), target_return=("gross_return", "first"), adv_usd_3m=("prior_adv_usd", "first"))
        .reset_index()
    )
    target["signal_at"] = pd.to_datetime(target["entry_date"]).dt.tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30)
    return target[["signal_at", "ticker", "weight", "target_return", "adv_usd_3m"]]


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"non-overwriting output exists: {OUTPUT}")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    receipt = json.loads(DESIGN.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "sec_reporting_lag_tradable_stage_b_design_receipt_v1" or receipt.get("status") != "FROZEN_PRE_FORWARD_RETURN_JOIN":
        raise ValueError("design is not frozen")
    verify_records(receipt["records"])
    plan_manifest = json.loads((PLAN / "plan_manifest.json").read_text(encoding="utf-8"))
    if plan_manifest.get("outcome_join_performed") is not False:
        raise ValueError("plan already joined outcomes")
    verify_records(plan_manifest["inputs"] + plan_manifest["outputs"])
    schedule = pd.read_parquet(PLAN / "monthly_tail_schedule.parquet")
    sealed = schedule.loc[schedule["phase"].eq("sealed_2025")].copy()
    opened = schedule.loc[schedule["phase"].isin(["development", "opened_diagnostic"])].copy()
    if sealed.empty or not sealed["outcome_join_permitted"].eq(False).all():
        raise ValueError("sealed cohort contract failed")
    OUTPUT.mkdir(parents=True)
    sealed_path = OUTPUT / "sealed_2025_signal_only.parquet"
    sealed.to_parquet(sealed_path, index=False)
    request_path = OUTPUT / "frozen_outcome_request.json"
    write_json(request_path, {"schema_version": "sec_reporting_lag_tradable_stage_b_outcome_request_v1", "created_at": datetime.now(UTC).isoformat(), "research_role": config["research_role"], "opened_months": sorted(opened["formation_month"].unique()), "sealed_months_excluded": sorted(sealed["formation_month"].unique()), "forward_return_read_started": False})
    marker_path = OUTPUT / "pre_forward_return_marker.json"
    write_json(marker_path, {"schema_version": "sec_reporting_lag_tradable_stage_b_preread_v1", "created_at": datetime.now(UTC).isoformat(), "outcome_request_sha256": sha256_file(request_path), "price_file_sha256": sha256_file(PRICES), "forward_return_values_read": False})
    tickers = sorted(opened["ticker"].unique())
    prices = pd.read_parquet(PRICES, columns=["ticker", "date", "timestamp", "close", "volume", "adj_open"], filters=[("date", ">=", pd.Timestamp("2018-01-01").to_pydatetime()), ("date", "<=", pd.Timestamp(opened["calendar_exit_date"].max()).to_pydatetime()), ("ticker", "in", tickers + ["SPY"])])
    signal = config["signal_definition"]
    holdings_parts, monthly_parts, exclusion_parts = [], [], []
    summaries = {}
    for phase in ("development", "opened_diagnostic"):
        part = opened.loc[opened["phase"].eq(phase)]
        holdings, monthly, exclusions = calendar_portfolio_returns(part, prices, holding_months=1, minimum_prior_close=float(signal["minimum_prior_close"]), adv_window=int(signal["adv_window_sessions"]), minimum_prior_stock_sessions=int(signal["minimum_prior_stock_sessions"]), minimum_prior_adv=float(signal["minimum_trailing_adv_usd"]), primary_one_way_cost=.003, stress_one_way_cost=.005)
        holdings["phase"], monthly["phase"], exclusions["phase"] = phase, phase, phase
        summaries[phase] = phase_summary(monthly, holdings)
        holdings_parts.append(holdings); monthly_parts.append(monthly); exclusion_parts.append(exclusions)
    holdings = pd.concat(holdings_parts, ignore_index=True)
    monthly = pd.concat(monthly_parts, ignore_index=True)
    exclusions = pd.concat(exclusion_parts, ignore_index=True)
    diagnostic_holdings = holdings.loc[holdings["phase"].eq("opened_diagnostic")]
    diagnostic_schedule = opened.loc[opened["phase"].eq("opened_diagnostic")]
    cap_summary, cap_trades = capacity_diagnostics(capacity_input(diagnostic_holdings, diagnostic_schedule), portfolio_aum_usd=10_000_000, maximum_adv_participation=.01)
    result = summaries["opened_diagnostic"]
    decision = "DESCRIPTIVE_POSITIVE_REQUIRES_FUTURE_CONFIRMATION" if result["net30_mean"] > 0 and result["net30_terminal_wealth"] > 1 else "RETIRED_METHOD_REPAIR_DIAGNOSTIC"
    result_summary = {"schema_version": "sec_reporting_lag_tradable_stage_b_summary_v1", "created_at": datetime.now(UTC).isoformat(), "candidate_decision": decision, "research_role": config["research_role"], "historical_alpha_verdict": "INVALID", "model_alpha_verdict": "NOT VALIDATED", "deployment_verdict": "NOT AUTHORIZED", "development": summaries["development"], "opened_diagnostic": result, "capacity": cap_summary, "sealed_2025_signal_outcomes_joined": 0, "limitations": config["hard_claim_limits"]}
    holdings_path, monthly_path, exclusions_path, capacity_path = OUTPUT / "monthly_holdings.csv", OUTPUT / "monthly_portfolio_returns.csv", OUTPUT / "price_path_exclusions.csv", OUTPUT / "capacity_trades.csv"
    holdings.to_csv(holdings_path, index=False); monthly.to_csv(monthly_path, index=False); exclusions.to_csv(exclusions_path, index=False); cap_trades.to_csv(capacity_path, index=False)
    summary_path = OUTPUT / "summary_metrics.json"; write_json(summary_path, result_summary)
    frame = monthly.loc[(monthly["phase"].eq("opened_diagnostic")) & monthly["portfolio_scope"].eq("ALL")].sort_values("month").copy()
    frame["date"] = pd.PeriodIndex(frame["month"], freq="M").to_timestamp("M")
    fig, ax = plt.subplots(figsize=(7.5, 4.2)); ax.plot(frame["date"], (1 + frame["q5_minus_q1_gross_return"]).cumprod(), label="Gross"); ax.plot(frame["date"], (1 + frame["q5_minus_q1_net30_return"]).cumprod(), label="Net 30 bps"); ax.axhline(1, color="black", linewidth=.7); ax.set_title("Tradability-first reporting lag diagnostic"); ax.legend(frameon=False); fig.tight_layout()
    chart_path = OUTPUT / "opened_diagnostic_result.png"; fig.savefig(chart_path, dpi=170); plt.close(fig)
    report_path = OUTPUT / "report-source.md"
    report_path.write_text(f"""# Reporting-lag tradability-first Stage B

Research role: **POST-OUTCOME DIAGNOSTIC ONLY**. Candidate decision: **{decision}**. Historical alpha: **INVALID**. Deployment: **NOT AUTHORIZED**.

The opened 2022-2024 diagnostic has {result['months']} months, {result['signal_events']} retained signal-months, and {result['unique_cik']} issuers. Gross mean={result['gross_mean']:.6f}; net-30 mean={result['net30_mean']:.6f}; net-50 mean={result['net50_mean']:.6f}. Gross/net-30 terminal wealth={result['gross_terminal_wealth']:.4f}/{result['net30_terminal_wealth']:.4f}. Average turnover={result['average_turnover']:.4f}. No p-value is reported because the repair was selected after Stage A outcomes were known.

The repair applies backward-only price and ADV eligibility before reporting-lag quintile assignment. All 2025 signals remain signal-only; zero sealed outcomes were joined.
""", encoding="utf-8")
    outputs = [record(sealed_path, "sealed_2025_signal_only"), record(request_path, "frozen_outcome_request"), record(marker_path, "pre_forward_return_marker"), record(holdings_path, "monthly_holdings"), record(monthly_path, "monthly_portfolio_returns"), record(exclusions_path, "price_path_exclusions"), record(capacity_path, "capacity_trades"), record(summary_path, "summary_metrics"), record(chart_path, "result_chart"), record(report_path, "research_report")]
    write_json(OUTPUT / "run_manifest.json", {"schema_version": "sec_reporting_lag_tradable_stage_b_run_manifest_v1", "created_at": result_summary["created_at"], "candidate_decision": decision, "inputs": [record(DESIGN, "design_receipt"), record(CONFIG, "human_protocol"), record(PLAN / "plan_manifest.json", "plan_manifest"), record(PRICES, "retrospective_prices"), record(Path(__file__), "runner")], "outputs": outputs, "sealed_2025_signal_outcomes_joined": 0})
    print(report_path)


if __name__ == "__main__":
    main()
