#!/usr/bin/env python3
"""Execute the frozen SEC Item 2.02 reaction-drift confirmation."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alpha_factor_engine.research.sec_8k_reaction_outcomes import (
    aggregate_entry_dates,
    attach_reaction_and_forward_returns,
    gate,
    summarize,
)


CONFIG = ROOT / "configs/sec_8k_reaction_drift_stage_a_v1.json"
PLAN = ROOT / "artifacts/sec_8k_reaction_drift_stage_a_v1/preoutcome_plan_v1"
DESIGN = ROOT / "artifacts/sec_8k_reaction_drift_stage_a_v1/design_receipt_v1.json"
PRICES = ROOT / "artifacts/public_broad_proxy_v1/data/daily_prices.parquet"
CALENDAR = ROOT / "artifacts/reference_calendars/xnys_v1/xnys_regular_sessions_2014_2028.csv"
OUTPUT = ROOT / "artifacts/sec_8k_reaction_drift_stage_a_v1/run_v1"


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


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    receipt = json.loads(DESIGN.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "sec_8k_reaction_drift_design_receipt_v1" or receipt.get("status") != "FROZEN_PRE_PRICE":
        raise ValueError("Item 2.02 design is not frozen")
    verify_records(receipt["records"])
    plan_manifest = json.loads((PLAN / "plan_manifest.json").read_text(encoding="utf-8"))
    if plan_manifest.get("price_values_read") is not False or plan_manifest.get("outcome_join_performed") is not False:
        raise ValueError("event plan is not pre-price")
    verify_records(plan_manifest["inputs"] + plan_manifest["outputs"])
    schedule = pd.read_parquet(PLAN / "event_clock_schedule.parquet")
    sealed = schedule.loc[schedule["phase"].eq("sealed_2025")].copy()
    outcome_schedule = schedule.loc[schedule["phase"].isin(["development", "confirmation"])].copy()
    if sealed.empty or not sealed["outcome_join_permitted"].eq(False).all():
        raise ValueError("sealed schedule contract failed")
    OUTPUT.mkdir(parents=True)
    sealed_path = OUTPUT / "sealed_2025_clock_only.parquet"; sealed.to_parquet(sealed_path, index=False)
    request_path = OUTPUT / "frozen_outcome_request.json"
    write_json(request_path, {"schema_version": "sec_8k_reaction_drift_outcome_request_v1", "created_at": datetime.now(UTC).isoformat(), "design_receipt_sha256": sha256_file(DESIGN), "trial_id": config["formal_tests"]["primary_trial_id"], "outcome_accessions": sorted(outcome_schedule["accession"].astype(str)), "sealed_accessions_excluded": sorted(sealed["accession"].astype(str)), "price_read_started": False})
    marker_path = OUTPUT / "pre_price_read_marker.json"
    write_json(marker_path, {"schema_version": "sec_8k_reaction_drift_preread_marker_v1", "created_at": datetime.now(UTC).isoformat(), "outcome_request_sha256": sha256_file(request_path), "price_file_sha256_verified": sha256_file(PRICES), "price_values_read": False})
    tickers = sorted(set(outcome_schedule["ticker"].astype(str)) | {"SPY"})
    start = pd.Timestamp(outcome_schedule["prior_reaction_date"].min()) - pd.Timedelta(days=400)
    end = pd.Timestamp(outcome_schedule["exit_date"].max())
    prices = pd.read_parquet(PRICES, columns=["ticker", "date", "timestamp", "close", "adj_close", "adj_open", "dollar_volume", "available_at"], filters=[("date", ">=", start.to_pydatetime()), ("date", "<=", end.to_pydatetime()), ("ticker", "in", tickers)])
    calendar = pd.read_csv(CALENDAR)
    definition = config["event_definition"]
    events, exclusions, counts = attach_reaction_and_forward_returns(outcome_schedule, prices, calendar, minimum_prior_close=float(definition["minimum_prior_close"]), minimum_prior_stock_sessions=int(definition["minimum_prior_stock_sessions"]), adv_window_sessions=int(definition["adv_window_sessions"]), minimum_prior_adv=float(definition["minimum_trailing_adv_usd"]), threshold_window_days=int(definition["historical_threshold_window_days"]), threshold_minimum_events=int(definition["historical_threshold_minimum_events"]), tail_probability=float(definition["tail_probability"]), portfolio_aum_usd=float(definition["portfolio_aum_usd"]))
    if events["phase"].eq("sealed_2025").any():
        raise ValueError("sealed event obtained an outcome")
    development = events.loc[events["phase"].eq("development")].copy()
    confirmation = events.loc[events["phase"].eq("confirmation")].copy()
    development_counts = counts["by_phase"]["development"]
    confirmation_counts = counts["by_phase"]["confirmation"]
    dev_summary = summarize(development, trial_floor=int(config["formal_tests"]["current_disclosed_trial_floor"]), scheduled_events=development_counts["scheduled_events"], price_eligible_events=development_counts["price_eligible_events"])
    confirmation_summary = summarize(confirmation, trial_floor=int(config["formal_tests"]["current_disclosed_trial_floor"]), scheduled_events=confirmation_counts["scheduled_events"], price_eligible_events=confirmation_counts["price_eligible_events"])
    gates = gate(confirmation_summary, config["confirmation_gates"])
    decision = "RETAIN_FOR_PROSPECTIVE_REPLICATION" if bool(gates["passed"].all()) else "RETIRE_STAGE_A"
    events_path, exclusions_path, dates_path, gates_path = OUTPUT / "event_returns.csv", OUTPUT / "event_exclusions.csv", OUTPUT / "confirmation_entry_date_returns.csv", OUTPUT / "validation_gates.csv"
    events.to_csv(events_path, index=False); exclusions.to_csv(exclusions_path, index=False); aggregate_entry_dates(confirmation).to_csv(dates_path, index=False); gates.to_csv(gates_path, index=False)
    summary = {"schema_version": "sec_8k_reaction_drift_summary_v1", "created_at": datetime.now(UTC).isoformat(), "candidate_decision": decision, "efficacy_gates_passed": bool(gates["passed"].all()), "historical_alpha_verdict": "INVALID", "model_alpha_verdict": "NOT VALIDATED", "deployment_verdict": "NOT AUTHORIZED", "all_counts": counts, "development": dev_summary, "confirmation": confirmation_summary, "sealed_2025_signal_outcomes_joined": 0, "limitations": config["hard_claim_limits"]}
    summary_path = OUTPUT / "summary_metrics.json"; write_json(summary_path, summary)
    date_returns = aggregate_entry_dates(confirmation); date_returns["year"] = pd.to_datetime(date_returns["entry_date"]).dt.year
    yearly = date_returns.groupby("year")[["pair_gross_return_20", "pair_net_30bps_return_20", "pair_net_50bps_return_20"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].bar(["Gross", "Net 30", "Net 50"], [confirmation_summary["gross"]["mean"], confirmation_summary["primary_net_30bps"]["mean"], confirmation_summary["stress_net_50bps"]["mean"]]); axes[0].axhline(0, color="black", linewidth=.7); axes[0].set_title("Mean 20-session signed pair return")
    yearly.plot(kind="bar", ax=axes[1]); axes[1].axhline(0, color="black", linewidth=.7); axes[1].set_title("Equal-entry-date mean by year"); axes[1].legend(["Gross", "Net 30", "Net 50"], frameon=False, fontsize=8); fig.tight_layout()
    chart_path = OUTPUT / "confirmation_result.png"; fig.savefig(chart_path, dpi=170); plt.close(fig)
    failed = gates.loc[~gates["passed"], "gate"].tolist(); c = confirmation_summary
    report_path = OUTPUT / "report-source.md"
    report_path.write_text(f"""# SEC Item 2.02 reaction-drift Stage A

Candidate decision: **{decision}**. Historical alpha: **INVALID**. Model alpha: **NOT VALIDATED**. Deployment: **NOT AUTHORIZED**.

The frozen 2022-2024 confirmation contains {c['events']} retained tail events across {c['tickers']} tickers and {c['entry_dates']} entry dates. Gross mean={c['gross']['mean']:.6f}; net-30 mean={c['primary_net_30bps']['mean']:.6f}, HAC t={c['primary_net_30bps']['t']:.3f}, adjusted p={c['primary_net_30bps']['known_floor_adjusted_screening_p']:.6f}; net-50 mean={c['stress_net_50bps']['mean']:.6f}. Failed gates: {', '.join(failed) or 'none'}.

The online 20/80% reaction thresholds use only eligible Item 2.02 reaction sessions in the preceding 730 calendar days and strictly exclude the current reaction date. Entry is the next regular open after the full reaction close. All 2025 event clocks remain signal-only; zero sealed outcomes were joined.

Item 2.02 may lag the issuer or newswire first-public earnings release, so this is a conservative SEC-filed-release drift test, not a complete replication of analyst-surprise PEAD.

## Sources

- [SEC EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)
- [DellaVigna and Pollet, Investor Inattention and Friday Earnings Announcements](https://www.nber.org/papers/w11683)
- [Livnat and Mendenhall, analyst versus time-series earnings surprises](https://doi.org/10.1111/j.1475-679X.2006.00196.x)
""", encoding="utf-8")
    outputs = [record(sealed_path, "sealed_2025_clock_only"), record(request_path, "frozen_outcome_request"), record(marker_path, "pre_price_read_marker"), record(events_path, "event_returns"), record(exclusions_path, "event_exclusions"), record(dates_path, "confirmation_entry_date_returns"), record(gates_path, "validation_gates"), record(summary_path, "summary_metrics"), record(chart_path, "result_chart"), record(report_path, "research_report")]
    manifest = {"schema_version": "sec_8k_reaction_drift_run_manifest_v1", "created_at": summary["created_at"], "candidate_decision": decision, "inputs": [record(DESIGN, "design_receipt"), record(CONFIG, "human_protocol"), record(PLAN / "plan_manifest.json", "plan_manifest"), record(PRICES, "retrospective_prices"), record(CALENDAR, "xnys_calendar"), record(Path(__file__), "outcome_runner"), record(SRC / "alpha_factor_engine/research/sec_8k_reaction_outcomes.py", "outcome_core")], "outputs": outputs, "sealed_2025_signal_outcomes_joined": 0}
    write_json(OUTPUT / "run_manifest.json", manifest)
    print(report_path)


if __name__ == "__main__":
    main()
