#!/usr/bin/env python3
"""Independent arithmetic, lineage, clock, and sealed-cohort audit."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/sec_reporting_lag_stage_a_v1/run_v1"
PLAN = ROOT / "artifacts/sec_reporting_lag_stage_a_v1/preoutcome_plan_v2"
OUTPUT = ROOT / "artifacts/sec_reporting_lag_stage_a_v1/independent_audit_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path, role: str) -> dict[str, object]:
    path = path.resolve()
    return {"role": role, "path": path.relative_to(ROOT.resolve()).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def add(checks: list[dict[str, object]], name: str, passed: bool, detail: object) -> None:
    checks.append({"check": name, "passed": bool(passed), "detail": json.dumps(detail, sort_keys=True, default=str)})


def verify_manifest(path: Path, checks: list[dict[str, object]], prefix: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = list(payload.get("records", [])) + list(payload.get("inputs", [])) + list(payload.get("outputs", []))
    add(checks, f"{prefix}_records_nonempty", bool(records), len(records))
    for item in records:
        source = ROOT / item["path"]
        add(checks, f"{prefix}_exists::{item['role']}", source.is_file(), item["path"])
        if source.is_file():
            add(checks, f"{prefix}_bytes::{item['role']}", source.stat().st_size == int(item["bytes"]), source.stat().st_size)
            add(checks, f"{prefix}_sha256::{item['role']}", sha256_file(source) == item["sha256"], item["sha256"])


def independent_hac_t(values: np.ndarray, lags: int) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=float)
    centered = x - x.mean()
    n = len(x)
    long_run = float(centered @ centered / n)
    for lag in range(1, min(lags, n - 1) + 1):
        gamma = float(centered[lag:] @ centered[:-lag] / n)
        long_run += 2 * (1 - lag / (lags + 1)) * gamma
    se = float(np.sqrt(max(long_run, 0) / n))
    t = float(x.mean() / se) if se > 0 else np.nan
    return se, t, float(1 - norm.cdf(t))


def recompute_turnover(holdings: pd.DataFrame) -> pd.DataFrame:
    frame = holdings.loc[holdings["portfolio_scope"].eq("ALL")].copy()
    grouped = (
        frame.groupby(["month", "month_ordinal", "ticker"], sort=True)
        .agg(weight=("target_weight_component", "sum"), return_=("gross_return", "first"))
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    previous_weights: dict[str, float] = {}
    previous_returns: dict[str, float] = {}
    previous_growth: float | None = None
    previous_ordinal: int | None = None
    for month, group in grouped.groupby("month", sort=True):
        ordinal = int(group["month_ordinal"].iloc[0])
        target = dict(zip(group["ticker"].astype(str), group["weight"].astype(float)))
        if previous_ordinal is None or ordinal != previous_ordinal + 1:
            start = float(sum(abs(v) for v in target.values()))
        else:
            assert previous_growth is not None
            names = set(target) | set(previous_weights)
            start = float(sum(abs(target.get(name, 0) - previous_weights.get(name, 0) * (1 + previous_returns.get(name, 0)) / previous_growth) for name in names))
        gross = float((group["weight"] * group["return_"]).sum())
        rows.append({"month": month, "start_turnover_independent": start, "gross_independent": gross, "terminal_turnover_independent": 0.0})
        previous_weights = target
        previous_returns = dict(zip(group["ticker"].astype(str), group["return_"].astype(float)))
        previous_growth = 1 + gross
        previous_ordinal = ordinal
    if rows:
        rows[-1]["terminal_turnover_independent"] = float(sum(abs(previous_weights[name] * (1 + previous_returns[name])) for name in previous_weights))
    return pd.DataFrame(rows)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"non-overwriting audit exists: {OUTPUT}")
    checks: list[dict[str, object]] = []
    verify_manifest(ROOT / "artifacts/sec_reporting_lag_stage_a_v1/design_receipt_v1.json", checks, "design")
    verify_manifest(PLAN / "plan_manifest.json", checks, "plan")
    verify_manifest(RUN / "run_manifest.json", checks, "run")
    summary = json.loads((RUN / "summary_metrics.json").read_text(encoding="utf-8"))
    schedule = pd.read_parquet(PLAN / "monthly_tail_schedule.parquet")
    sealed = pd.read_parquet(RUN / "sealed_2025_signal_only.parquet")
    holdings = pd.read_csv(RUN / "monthly_holdings.csv", parse_dates=["entry_date", "exit_date", "accepted_at"])
    monthly = pd.read_csv(RUN / "monthly_portfolio_returns.csv", parse_dates=["entry_date", "exit_date"])
    gates = pd.read_csv(RUN / "validation_gates.csv")
    capacity = pd.read_csv(RUN / "capacity_trades.csv")
    confirmation = monthly.loc[(monthly["phase"].eq("confirmation")) & monthly["portfolio_scope"].eq("ALL")].sort_values("month")
    csum = summary["confirmation"]
    add(checks, "sealed_schedule_exact_match", sealed.equals(schedule.loc[schedule["phase"].eq("sealed_2025")].reset_index(drop=True)), len(sealed))
    add(checks, "sealed_outcome_join_forbidden", sealed["outcome_join_permitted"].eq(False).all(), int(len(sealed)))
    add(checks, "no_sealed_holdings", not holdings["phase"].eq("sealed_2025").any(), sorted(holdings["phase"].unique()))
    add(checks, "summary_reports_zero_sealed_joins", summary["sealed_2025_signal_outcomes_joined"] == 0, summary["sealed_2025_signal_outcomes_joined"])
    add(checks, "decision_clocks_public", schedule["accepted_at"].le(schedule["decision_at"]).all(), int((~schedule["accepted_at"].le(schedule["decision_at"])).sum()))
    add(checks, "lag_arithmetic", (pd.to_datetime(schedule["filing_date"]).dt.normalize() - pd.to_datetime(schedule["report_date"]).dt.normalize()).dt.days.eq(schedule["reporting_lag_days"]).all(), int(len(schedule)))
    add(checks, "confirmation_months_complete", confirmation["month"].tolist() == [str(x) for x in pd.period_range("2022-01", "2024-12", freq="M")], confirmation["month"].tolist())
    add(checks, "net30_cost_identity", np.allclose(confirmation["q5_minus_q1_net30_return"], confirmation["q5_minus_q1_gross_return"] - .003 * confirmation["total_turnover"], atol=1e-14), float(np.max(np.abs(confirmation["q5_minus_q1_net30_return"] - (confirmation["q5_minus_q1_gross_return"] - .003 * confirmation["total_turnover"])))))
    add(checks, "net50_cost_identity", np.allclose(confirmation["q5_minus_q1_net50_return"], confirmation["q5_minus_q1_gross_return"] - .005 * confirmation["total_turnover"], atol=1e-14), int(len(confirmation)))
    for name, column in (("gross", "q5_minus_q1_gross_return"), ("net30", "q5_minus_q1_net30_return"), ("net50", "q5_minus_q1_net50_return")):
        observed = float(confirmation[column].mean())
        reported = float(csum["tests"][name]["mean"])
        add(checks, f"summary_mean::{name}", np.isclose(observed, reported, atol=1e-15), {"observed": observed, "reported": reported})
    se, t, p = independent_hac_t(confirmation["q5_minus_q1_net30_return"].to_numpy(), 3)
    net30 = csum["tests"]["net30"]
    add(checks, "independent_hac_se", np.isclose(se, net30["standard_error"], atol=1e-15), {"observed": se, "reported": net30["standard_error"]})
    add(checks, "independent_hac_t", np.isclose(t, net30["t"], atol=1e-15), {"observed": t, "reported": net30["t"]})
    add(checks, "independent_one_sided_p", np.isclose(p, net30["one_sided_p"], atol=1e-15), {"observed": p, "reported": net30["one_sided_p"]})
    add(checks, "bonferroni_floor_132", np.isclose(min(1.0, 132 * p), net30["known_floor_adjusted_screening_p"], atol=1e-15), net30["known_floor_adjusted_screening_p"])
    independent = recompute_turnover(holdings.loc[holdings["phase"].eq("confirmation")])
    compare = confirmation.merge(independent, on="month", validate="one_to_one")
    add(checks, "independent_gross_returns", np.allclose(compare["q5_minus_q1_gross_return"], compare["gross_independent"], atol=1e-14), float(np.max(np.abs(compare["q5_minus_q1_gross_return"] - compare["gross_independent"]))))
    add(checks, "independent_start_turnover", np.allclose(compare["start_turnover"], compare["start_turnover_independent"], atol=1e-14), float(np.max(np.abs(compare["start_turnover"] - compare["start_turnover_independent"]))))
    add(checks, "independent_terminal_turnover", np.allclose(compare["terminal_turnover"], compare["terminal_turnover_independent"], atol=1e-14), float(np.max(np.abs(compare["terminal_turnover"] - compare["terminal_turnover_independent"]))))
    add(checks, "capacity_max", np.isclose(float(capacity["adv_participation"].max()), summary["capacity"]["maximum_adv_participation_observed"], atol=1e-15), float(capacity["adv_participation"].max()))
    add(checks, "capacity_gate", bool((capacity["adv_participation"] <= .01).all()) == bool(summary["capacity"]["capacity_passed"]), int((capacity["adv_participation"] > .01).sum()))
    failed = set(gates.loc[~gates["passed"].astype(bool), "gate"])
    expected_failed = {"primary_net30_mean_positive", "primary_net30_hac_t", "global_floor_adjusted_p", "stress_net50_mean_positive", "leave_one_cik_out_positive"}
    add(checks, "failed_gate_set", failed == expected_failed, sorted(failed))
    add(checks, "retire_decision", summary["candidate_decision"] == "RETIRE_STAGE_A" and not gates["passed"].astype(bool).all(), summary["candidate_decision"])
    frame = pd.DataFrame(checks)
    if not frame["passed"].all():
        raise AssertionError(frame.loc[~frame["passed"]].to_string(index=False))
    OUTPUT.mkdir(parents=True)
    checks_path = OUTPUT / "checks.csv"
    frame.to_csv(checks_path, index=False)
    ledger = pd.DataFrame([
        {"claim_id": "C01", "claim": "The frozen confirmation net-30 mean is negative and statistically unsupported.", "evidence": "run_v1/summary_metrics.json plus independent HAC recomputation", "status": "VERIFIED"},
        {"claim_id": "C02", "claim": "All 2025 reporting-lag signals remained signal-only.", "evidence": "sealed_2025_signal_only.parquet and no sealed holdings", "status": "VERIFIED"},
        {"claim_id": "C03", "claim": "The $10m capacity screen passes under the frozen ADV proxy.", "evidence": "capacity_trades.csv independent maximum", "status": "VERIFIED_PROXY_ONLY"},
        {"claim_id": "C04", "claim": "The experiment validates deployable alpha.", "evidence": "failed efficacy gates and current-survivor universe", "status": "REJECTED"},
        {"claim_id": "S01", "claim": "The paper abstract reports a premium for faster reporters.", "evidence": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3432446", "status": "PRIMARY_ABSTRACT_ONLY"},
        {"claim_id": "S02", "claim": "Specific portfolio mechanics were cross-checked through a secondary summary.", "evidence": "https://www.cxoadvisory.com/fundamental-valuation/timely-firms-have-higher-returns/", "status": "SECONDARY_SOURCE"},
    ])
    ledger_path = OUTPUT / "claim_source_ledger.csv"
    ledger.to_csv(ledger_path, index=False)
    audit_summary = {"schema_version": "sec_reporting_lag_independent_audit_v1", "created_at": datetime.now(UTC).isoformat(), "checks_passed": int(frame["passed"].sum()), "checks_total": int(len(frame)), "audit_verdict": "ARITHMETIC_AND_LINEAGE_REPRODUCED", "candidate_decision": "RETIRE_STAGE_A", "historical_alpha_verdict": "INVALID", "deployment_verdict": "NOT AUTHORIZED"}
    summary_path = OUTPUT / "audit_summary.json"
    summary_path.write_text(json.dumps(audit_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path = OUTPUT / "report-source.md"
    report_path.write_text(f"""# Independent audit: SEC reporting lag Stage A

All **{len(frame)}/{len(frame)}** lineage, sealed-cohort, clock, cost, return, HAC, turnover, capacity, gate, and decision checks passed.

The reproduced confirmation result is gross {csum['tests']['gross']['mean']:.6f} per month, net-30 {net30['mean']:.6f}, HAC t={net30['t']:.3f}, adjusted p={net30['known_floor_adjusted_screening_p']:.6f}. The frozen decision **RETIRE_STAGE_A** is correct. Historical alpha remains **INVALID** because the 1,500-stock universe is a 2026 current-survivor proxy; deployment remains **NOT AUTHORIZED**.
""", encoding="utf-8")
    manifest = {"schema_version": "sec_reporting_lag_audit_manifest_v1", "created_at": audit_summary["created_at"], "inputs": [record(RUN / "run_manifest.json", "run_manifest"), record(RUN / "summary_metrics.json", "run_summary"), record(PLAN / "monthly_tail_schedule.parquet", "frozen_schedule"), record(Path(__file__), "independent_auditor")], "outputs": [record(checks_path, "audit_checks"), record(ledger_path, "claim_source_ledger"), record(summary_path, "audit_summary"), record(report_path, "audit_report")]}
    (OUTPUT / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
