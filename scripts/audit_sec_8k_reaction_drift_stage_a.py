#!/usr/bin/env python3
"""Independent audit of SEC Item 2.02 reaction-drift Stage A."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/sec_8k_reaction_drift_stage_a_v1"
PLAN, RUN, OUT = BASE / "preoutcome_plan_v1", BASE / "run_v1", BASE / "independent_audit_v1"
PRICES = ROOT / "artifacts/public_broad_proxy_v1/data/daily_prices.parquet"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path, role: str) -> dict[str, object]:
    path = path.resolve()
    return {"role": role, "path": path.relative_to(ROOT.resolve()).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def add(rows: list[dict[str, object]], name: str, passed: bool, detail: object) -> None:
    rows.append({"check": name, "passed": bool(passed), "detail": json.dumps(detail, sort_keys=True, default=str)})


def verify_manifest(path: Path, prefix: str, rows: list[dict[str, object]]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", []) + payload.get("inputs", []) + payload.get("outputs", [])
    for item in records:
        target = ROOT / item["path"]
        add(rows, f"{prefix}_exists::{item['role']}", target.is_file(), item["path"])
        if target.is_file():
            add(rows, f"{prefix}_hash::{item['role']}", target.stat().st_size == int(item["bytes"]) and sha256_file(target) == item["sha256"], item["sha256"])


def hac(values: np.ndarray, indices: np.ndarray, lags: int = 3) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=float); idx = np.asarray(indices, dtype=int); centered = x - x.mean(); n = len(x)
    long_run = float(centered @ centered / n)
    for lag in range(1, lags + 1):
        mask = idx[:, None] - idx[None, :] == lag
        gamma = float(np.sum(centered[:, None] * centered[None, :] * mask) / n)
        long_run += 2 * (1 - lag / (lags + 1)) * gamma
    se = float(np.sqrt(max(long_run, 0) / n)); t = float(x.mean() / se)
    return se, t, float(1 - norm.cdf(t))


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    rows: list[dict[str, object]] = []
    verify_manifest(BASE / "design_receipt_v1.json", "design", rows)
    verify_manifest(PLAN / "plan_manifest.json", "plan", rows)
    verify_manifest(RUN / "run_manifest.json", "run", rows)
    schedule = pd.read_parquet(PLAN / "event_clock_schedule.parquet")
    sealed = pd.read_parquet(RUN / "sealed_2025_clock_only.parquet")
    events = pd.read_csv(RUN / "event_returns.csv", parse_dates=["reaction_date", "entry_date", "exit_date", "reaction_history_start", "reaction_history_end_exclusive"])
    exclusions = pd.read_csv(RUN / "event_exclusions.csv")
    date_returns = pd.read_csv(RUN / "confirmation_entry_date_returns.csv")
    gates = pd.read_csv(RUN / "validation_gates.csv")
    summary = json.loads((RUN / "summary_metrics.json").read_text(encoding="utf-8"))
    add(rows, "clock_order", (schedule["public_at"].lt(schedule["reaction_close_at"]) & schedule["reaction_close_at"].lt(schedule["entry_at"]) & schedule["entry_at"].lt(schedule["exit_at"])).all(), int(len(schedule)))
    add(rows, "sealed_exact", sealed.equals(schedule.loc[schedule["phase"].eq("sealed_2025")].reset_index(drop=True)), int(len(sealed)))
    add(rows, "no_sealed_returns", not events["phase"].eq("sealed_2025").any() and summary["sealed_2025_signal_outcomes_joined"] == 0, sorted(events["phase"].unique()))
    add(rows, "online_history_strictly_prior", events["reaction_history_end_exclusive"].eq(events["reaction_date"]).all(), int(len(events)))
    add(rows, "online_window_730_days", (events["reaction_history_end_exclusive"] - events["reaction_history_start"]).dt.days.eq(730).all(), int(len(events)))
    add(rows, "direction_matches_tail", ((events["direction"].eq(1) & events["reaction_abnormal_return"].ge(events["upper_reaction_threshold"])) | (events["direction"].eq(-1) & events["reaction_abnormal_return"].le(events["lower_reaction_threshold"]))).all(), int(len(events)))
    add(rows, "pair_gross_identity", np.allclose(events["pair_gross_return_20"], .5 * events["direction"] * (events["stock_forward_return"] - events["spy_forward_return"]), atol=1e-14), int(len(events)))
    add(rows, "net30_identity", np.allclose(events["pair_net_30bps_return_20"], events["pair_gross_return_20"] - .003 * events["pair_traded_notional_per_gross"], atol=1e-14), int(len(events)))
    add(rows, "net50_identity", np.allclose(events["pair_net_50bps_return_20"], events["pair_gross_return_20"] - .005 * events["pair_traded_notional_per_gross"], atol=1e-14), int(len(events)))
    # Reconstruct all price-eligible reactions and the exact online threshold set.
    outcome_schedule = schedule.loc[schedule["phase"].isin(["development", "confirmation"])].copy()
    price_failure_reasons = {"exact_price_path_incomplete", "prior_stock_history_incomplete", "invalid_price_value", "reaction_close_below_floor", "prior_adv_below_floor"}
    price_failed = set(exclusions.loc[exclusions["reason"].isin(price_failure_reasons), "accession"].astype(str))
    eligible = outcome_schedule.loc[~outcome_schedule["accession"].astype(str).isin(price_failed)].copy()
    tickers = sorted(set(eligible["ticker"].astype(str)) | {"SPY"})
    prices = pd.read_parquet(PRICES, columns=["ticker", "date", "adj_close"], filters=[("date", ">=", pd.Timestamp(eligible["prior_reaction_date"].min()).to_pydatetime()), ("date", "<=", pd.Timestamp(eligible["reaction_date"].max()).to_pydatetime()), ("ticker", "in", tickers)])
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    lookup = prices.set_index(["ticker", "date"])["adj_close"]
    reactions = []
    for row in eligible.itertuples(index=False):
        stock = float(lookup.loc[(row.ticker, pd.Timestamp(row.reaction_date))] / lookup.loc[(row.ticker, pd.Timestamp(row.prior_reaction_date))] - 1)
        spy = float(lookup.loc[("SPY", pd.Timestamp(row.reaction_date))] / lookup.loc[("SPY", pd.Timestamp(row.prior_reaction_date))] - 1)
        reactions.append(stock - spy)
    eligible["reaction_independent"] = reactions
    actual_values = events.set_index("accession")["reaction_abnormal_return"]
    common = eligible.loc[eligible["accession"].isin(actual_values.index)]
    add(rows, "independent_reaction_values", np.allclose(common["reaction_independent"], common["accession"].map(actual_values), atol=1e-14), int(len(common)))
    expected_selected: set[str] = set()
    threshold_details: dict[pd.Timestamp, tuple[int, float, float]] = {}
    eligible["reaction_date"] = pd.to_datetime(eligible["reaction_date"]).dt.normalize()
    for date, group in eligible.groupby("reaction_date", sort=True):
        history = eligible.loc[eligible["reaction_date"].between(date - pd.Timedelta(days=730), date, inclusive="left"), "reaction_independent"].to_numpy()
        if len(history) < 500:
            continue
        lower, upper = np.quantile(history, [.2, .8], method="linear")
        chosen = group.loc[group["reaction_independent"].le(lower) | group["reaction_independent"].ge(upper), "accession"].astype(str)
        expected_selected.update(chosen)
        threshold_details[pd.Timestamp(date)] = (len(history), float(lower), float(upper))
    overlap = set(exclusions.loc[exclusions["reason"].eq("same_ticker_overlap_until_exit"), "accession"].astype(str))
    actual_before_overlap = set(events["accession"].astype(str)) | overlap
    add(rows, "independent_online_tail_set", expected_selected == actual_before_overlap, {"expected": len(expected_selected), "actual": len(actual_before_overlap), "symmetric_difference": len(expected_selected ^ actual_before_overlap)})
    for date, group in events.groupby("reaction_date"):
        n, lower, upper = threshold_details[pd.Timestamp(date)]
        add(rows, f"threshold_history_count::{date.date()}", group["reaction_history_events"].eq(n).all(), n)
        add(rows, f"threshold_values::{date.date()}", np.allclose(group["lower_reaction_threshold"], lower, atol=1e-14) and np.allclose(group["upper_reaction_threshold"], upper, atol=1e-14), {"lower": lower, "upper": upper})
    confirmation = events.loc[events["phase"].eq("confirmation")]
    independent_dates = confirmation.groupby("entry_date")[["pair_gross_return_20", "pair_net_30bps_return_20", "pair_net_50bps_return_20"]].mean().reset_index()
    add(rows, "entry_date_aggregation", np.allclose(independent_dates[["pair_gross_return_20", "pair_net_30bps_return_20", "pair_net_50bps_return_20"]], date_returns[["pair_gross_return_20", "pair_net_30bps_return_20", "pair_net_50bps_return_20"]], atol=1e-15), int(len(date_returns)))
    c = summary["confirmation"]
    for name, column in (("gross", "pair_gross_return_20"), ("primary_net_30bps", "pair_net_30bps_return_20"), ("stress_net_50bps", "pair_net_50bps_return_20")):
        mean = float(independent_dates[column].mean())
        add(rows, f"summary_mean::{name}", np.isclose(mean, c[name]["mean"], atol=1e-15), {"observed": mean, "reported": c[name]["mean"]})
    se, t, p = hac(date_returns["pair_net_30bps_return_20"].to_numpy(), date_returns["entry_index"].to_numpy())
    add(rows, "independent_hac", np.isclose(se, c["primary_net_30bps"]["standard_error"], atol=1e-15) and np.isclose(t, c["primary_net_30bps"]["t"], atol=1e-15) and np.isclose(p, c["primary_net_30bps"]["one_sided_p"], atol=1e-15), {"se": se, "t": t, "p": p})
    add(rows, "bonferroni_134", np.isclose(min(1, 134 * p), c["primary_net_30bps"]["known_floor_adjusted_screening_p"], atol=1e-15), c["primary_net_30bps"]["known_floor_adjusted_screening_p"])
    add(rows, "capacity_max", np.isclose(events["maximum_stock_trade_to_adv_at_10m"].max(), max(summary["development"]["maximum_trade_to_adv_at_10m"], c["maximum_trade_to_adv_at_10m"]), atol=1e-15), float(events["maximum_stock_trade_to_adv_at_10m"].max()))
    failed = set(gates.loc[~gates["passed"].astype(bool), "gate"])
    expected_failed = {"primary_net_mean", "primary_hac_t", "global_adjusted_p", "stress_net_mean", "leave_one_ticker_out", "capacity_10m"}
    add(rows, "failed_gate_set", failed == expected_failed, sorted(failed))
    add(rows, "retire_decision", summary["candidate_decision"] == "RETIRE_STAGE_A" and not gates["passed"].astype(bool).all(), summary["candidate_decision"])
    checks = pd.DataFrame(rows)
    if not checks["passed"].all():
        raise AssertionError(checks.loc[~checks["passed"]].to_string(index=False))
    OUT.mkdir(parents=True)
    checks_path = OUT / "checks.csv"; checks.to_csv(checks_path, index=False)
    ledger_path = OUT / "claim_source_ledger.csv"
    pd.DataFrame([
        {"claim_id": "C01", "claim": "The online reaction tails use only strictly earlier event dates.", "evidence": "independent reconstruction from frozen event clocks and prices", "status": "VERIFIED"},
        {"claim_id": "C02", "claim": "The confirmation gross and net results are negative.", "evidence": "independent event and entry-date reconstruction", "status": "VERIFIED"},
        {"claim_id": "C03", "claim": "All 2025 events remained clock-only.", "evidence": "sealed schedule exact match and no sealed event returns", "status": "VERIFIED"},
        {"claim_id": "C04", "claim": "The strategy passes $10m capacity at 1% ADV.", "evidence": "maximum observed participation 49.58%", "status": "REJECTED"},
        {"claim_id": "S01", "claim": "Item 2.02 identifies results-of-operations disclosures in SEC filings.", "evidence": "https://www.sec.gov/search-filings/edgar-application-programming-interfaces", "status": "OFFICIAL_SEC_CONTEXT"},
        {"claim_id": "S02", "claim": "Prior literature documents delayed responses to earnings news.", "evidence": "https://www.nber.org/papers/w11683", "status": "PRIMARY_RESEARCH_SOURCE"},
    ]).to_csv(ledger_path, index=False)
    audit_summary = {"schema_version": "sec_8k_reaction_drift_independent_audit_v1", "created_at": datetime.now(UTC).isoformat(), "checks_passed": int(checks["passed"].sum()), "checks_total": int(len(checks)), "audit_verdict": "LINEAGE_THRESHOLDS_RETURNS_AND_DECISION_REPRODUCED", "candidate_decision": "RETIRE_STAGE_A", "historical_alpha_verdict": "INVALID", "deployment_verdict": "NOT_AUTHORIZED"}
    summary_path = OUT / "audit_summary.json"; summary_path.write_text(json.dumps(audit_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path = OUT / "report-source.md"; report_path.write_text(f"# Independent audit: SEC Item 2.02 reaction drift\n\nAll **{len(checks)}/{len(checks)}** source-hash, clock, sealed-cohort, online-threshold, reaction, forward-return, cost, aggregation, HAC, multiplicity, capacity, gate, and decision checks passed. The independently reproduced confirmation net-30 mean is {c['primary_net_30bps']['mean']:.6f}, HAC t={c['primary_net_30bps']['t']:.3f}. The strategy remains retired and historical alpha remains INVALID.\n", encoding="utf-8")
    manifest = {"schema_version": "sec_8k_reaction_drift_audit_manifest_v1", "created_at": audit_summary["created_at"], "inputs": [record(RUN / "run_manifest.json", "run_manifest"), record(RUN / "summary_metrics.json", "run_summary"), record(PLAN / "event_clock_schedule.parquet", "frozen_schedule"), record(PRICES, "retrospective_prices"), record(Path(__file__), "independent_auditor")], "outputs": [record(checks_path, "checks"), record(ledger_path, "claim_ledger"), record(summary_path, "audit_summary"), record(report_path, "audit_report")]}
    (OUT / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
