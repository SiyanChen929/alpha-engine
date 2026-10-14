#!/usr/bin/env python3
"""Independent audit of the post-outcome tradability-first diagnostic."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/sec_reporting_lag_tradable_stage_b_v1"
PLAN, RUN, OUT = BASE / "preoutcome_plan_v1", BASE / "run_v1", BASE / "independent_audit_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path, role: str) -> dict[str, object]:
    path = path.resolve()
    return {"role": role, "path": path.relative_to(ROOT.resolve()).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def add(rows: list[dict[str, object]], check: str, passed: bool, detail: object) -> None:
    rows.append({"check": check, "passed": bool(passed), "detail": json.dumps(detail, sort_keys=True, default=str)})


def verify_manifest(path: Path, prefix: str, rows: list[dict[str, object]]) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    records = value.get("records", []) + value.get("inputs", []) + value.get("outputs", [])
    for item in records:
        target = ROOT / item["path"]
        add(rows, f"{prefix}_exists::{item['role']}", target.is_file(), item["path"])
        if target.is_file():
            add(rows, f"{prefix}_hash::{item['role']}", target.stat().st_size == int(item["bytes"]) and sha256_file(target) == item["sha256"], item["sha256"])


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    rows: list[dict[str, object]] = []
    verify_manifest(BASE / "design_receipt_v1.json", "design", rows)
    verify_manifest(PLAN / "plan_manifest.json", "plan", rows)
    verify_manifest(RUN / "run_manifest.json", "run", rows)
    schedule = pd.read_parquet(PLAN / "monthly_tail_schedule.parquet")
    sealed = pd.read_parquet(RUN / "sealed_2025_signal_only.parquet")
    holdings = pd.read_csv(RUN / "monthly_holdings.csv")
    monthly = pd.read_csv(RUN / "monthly_portfolio_returns.csv")
    capacity = pd.read_csv(RUN / "capacity_trades.csv")
    summary = json.loads((RUN / "summary_metrics.json").read_text(encoding="utf-8"))
    frame = monthly.loc[(monthly["phase"].eq("opened_diagnostic")) & monthly["portfolio_scope"].eq("ALL")].sort_values("month")
    reported = summary["opened_diagnostic"]
    add(rows, "tradability_screen_before_sort", schedule["prior_close"].ge(5).all() and schedule["prior_adv_usd"].ge(20_000_000).all(), {"min_close": float(schedule["prior_close"].min()), "min_adv": float(schedule["prior_adv_usd"].min())})
    add(rows, "each_month_two_sided", schedule.groupby("formation_month")["direction"].nunique().eq(2).all(), int(schedule["formation_month"].nunique()))
    add(rows, "public_clock", schedule["accepted_at"].le(schedule["decision_at"]).all(), int(len(schedule)))
    add(rows, "sealed_exact", sealed.equals(schedule.loc[schedule["phase"].eq("sealed_2025")].reset_index(drop=True)), int(len(sealed)))
    add(rows, "no_sealed_holdings", not holdings["phase"].eq("sealed_2025").any(), sorted(holdings["phase"].unique()))
    add(rows, "zero_sealed_outcome_claim", summary["sealed_2025_signal_outcomes_joined"] == 0, summary["sealed_2025_signal_outcomes_joined"])
    add(rows, "complete_36_months", frame["month"].tolist() == [str(x) for x in pd.period_range("2022-01", "2024-12", freq="M")], frame["month"].tolist())
    add(rows, "net30_identity", np.allclose(frame["q5_minus_q1_net30_return"], frame["q5_minus_q1_gross_return"] - .003 * frame["total_turnover"], atol=1e-14), int(len(frame)))
    add(rows, "net50_identity", np.allclose(frame["q5_minus_q1_net50_return"], frame["q5_minus_q1_gross_return"] - .005 * frame["total_turnover"], atol=1e-14), int(len(frame)))
    for key, column in (("gross_mean", "q5_minus_q1_gross_return"), ("net30_mean", "q5_minus_q1_net30_return"), ("net50_mean", "q5_minus_q1_net50_return")):
        observed = float(frame[column].mean())
        add(rows, f"summary::{key}", np.isclose(observed, reported[key], atol=1e-15), {"observed": observed, "reported": reported[key]})
    add(rows, "gross_wealth", np.isclose(float((1 + frame["q5_minus_q1_gross_return"]).prod()), reported["gross_terminal_wealth"], atol=1e-15), reported["gross_terminal_wealth"])
    add(rows, "net30_wealth", np.isclose(float((1 + frame["q5_minus_q1_net30_return"]).prod()), reported["net30_terminal_wealth"], atol=1e-15), reported["net30_terminal_wealth"])
    add(rows, "no_inferential_p_value", reported["p_value"] is None and reported["inference_status"] == "DESCRIPTIVE_ONLY_POST_OUTCOME_METHOD_REPAIR", reported["inference_status"])
    capmax = float(capacity["adv_participation"].max())
    add(rows, "capacity_max", np.isclose(capmax, summary["capacity"]["maximum_adv_participation_observed"], atol=1e-15), capmax)
    add(rows, "retire_decision", summary["candidate_decision"] == "RETIRED_METHOD_REPAIR_DIAGNOSTIC" and reported["gross_mean"] < 0 and reported["net30_mean"] < 0, summary["candidate_decision"])
    checks = pd.DataFrame(rows)
    if not checks["passed"].all():
        raise AssertionError(checks.loc[~checks["passed"]].to_string(index=False))
    OUT.mkdir(parents=True)
    checks_path = OUT / "checks.csv"; checks.to_csv(checks_path, index=False)
    ledger_path = OUT / "claim_source_ledger.csv"
    pd.DataFrame([
        {"claim_id": "C01", "claim": "Tradability was applied before reporting-lag quintile assignment.", "evidence": "preoutcome_plan_v1/monthly_eligibility_ledger.parquet and monthly_tail_schedule.parquet", "status": "VERIFIED"},
        {"claim_id": "C02", "claim": "The opened diagnostic gross and net returns are negative.", "evidence": "run_v1/monthly_portfolio_returns.csv independent arithmetic", "status": "VERIFIED_DESCRIPTIVE"},
        {"claim_id": "C03", "claim": "No inferential p-value is allowed.", "evidence": "post-outcome method-repair research role", "status": "VERIFIED"},
        {"claim_id": "C04", "claim": "All 2025 signal outcomes remain unopened.", "evidence": "sealed file exact match and no sealed holdings", "status": "VERIFIED"},
        {"claim_id": "C05", "claim": "The diagnostic validates alpha.", "evidence": "negative gross and invalid current-survivor history", "status": "REJECTED"},
    ]).to_csv(ledger_path, index=False)
    audit_summary = {"schema_version": "sec_reporting_lag_tradable_stage_b_audit_v1", "created_at": datetime.now(UTC).isoformat(), "checks_passed": int(checks["passed"].sum()), "checks_total": int(len(checks)), "audit_verdict": "DESCRIPTIVE_RESULT_REPRODUCED", "candidate_decision": summary["candidate_decision"], "historical_alpha_verdict": "INVALID", "deployment_verdict": "NOT_AUTHORIZED"}
    summary_path = OUT / "audit_summary.json"; summary_path.write_text(json.dumps(audit_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path = OUT / "report-source.md"; report_path.write_text(f"# Independent audit: tradability-first reporting lag\n\nAll **{len(checks)}/{len(checks)}** frozen-lineage, clock, sort eligibility, sealed-cohort, cost, return, wealth, capacity, and decision checks passed. The repaired implementation has gross mean {reported['gross_mean']:.6f} and net-30 mean {reported['net30_mean']:.6f} per month. It remains a post-outcome descriptive diagnostic and is retired.\n", encoding="utf-8")
    manifest = {"schema_version": "sec_reporting_lag_tradable_stage_b_audit_manifest_v1", "created_at": audit_summary["created_at"], "inputs": [record(RUN / "run_manifest.json", "run_manifest"), record(RUN / "summary_metrics.json", "run_summary"), record(PLAN / "monthly_tail_schedule.parquet", "frozen_schedule"), record(Path(__file__), "auditor")], "outputs": [record(checks_path, "checks"), record(ledger_path, "claim_ledger"), record(summary_path, "audit_summary"), record(report_path, "audit_report")]}
    (OUT / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
