#!/usr/bin/env python3
"""Run the frozen SEC Form NT Stage A after verifying its design receipt."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alpha_factor_engine.research.sec_filing_change_stage_a import (
    reconcile_benchmark_to_exchange_calendar,
)
from alpha_factor_engine.research.sec_late_filing_outcomes import (
    aggregate_entry_date_returns,
    attach_late_filing_outcomes,
    build_late_filing_schedule,
    gate_late_filing_confirmation,
    summarize_late_filing_events,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path, *, role: str) -> dict[str, object]:
    path = path.resolve()
    return {
        "role": role,
        "path": path.relative_to(ROOT.resolve()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def verify_records(records: list[dict[str, object]]) -> None:
    seen: set[Path] = set()
    for item in records:
        path = (ROOT / str(item["path"])).resolve()
        path.relative_to(ROOT.resolve())
        if path in seen or not path.is_file():
            raise ValueError(f"design record is duplicate or absent: {path}")
        seen.add(path)
        if path.stat().st_size != int(item["bytes"]):
            raise ValueError(f"design record byte mismatch: {path}")
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"design record hash mismatch: {path}")


def verify_design_receipt(path: Path) -> dict[str, object]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "sec_late_filing_stage_a_design_receipt_v1":
        raise ValueError("unsupported SEC late-filing design receipt")
    if receipt.get("status") != "FROZEN_PRE_OUTCOME":
        raise ValueError("SEC late-filing design is not frozen pre-outcome")
    if receipt.get("outcome_join_performed") is not False:
        raise ValueError("design receipt does not preserve pre-outcome status")
    records = receipt.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("design receipt has no records")
    verify_records(records)
    return receipt


def verify_plan_manifest(path: Path) -> dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "sec_late_filing_stage_a_plan_manifest_v1":
        raise ValueError("unsupported SEC late-filing plan manifest")
    if manifest.get("outcome_inputs_read") is not False or manifest.get("outcome_join_performed") is not False:
        raise ValueError("plan manifest is not pre-outcome")
    verify_records(list(manifest["inputs"]) + list(manifest["outputs"]))
    return manifest


def read_filtered_prices(
    path: Path,
    *,
    tickers: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    columns = ["ticker", "date", "timestamp", "adj_open", "dollar_volume", "available_at"]
    return pd.read_parquet(
        path,
        columns=columns,
        filters=[
            ("date", ">=", start.to_pydatetime()),
            ("date", "<=", end.to_pydatetime()),
            ("ticker", "in", tickers),
        ],
    )


def metric(value: object) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.6f}"


def build_report(
    *,
    summary: dict[str, object],
    gates: pd.DataFrame,
    schedule_counts: dict[str, int],
    coverage: dict[str, object],
    decision: str,
) -> str:
    primary = summary["primary_net_30bps"]
    stress = summary["stress_net_50bps"]
    gross = summary["gross"]
    pre = summary["pre_event_placebo"]
    post_pre = summary["post_minus_pre"]
    bootstrap = summary["primary_centered_circular_block_bootstrap"]
    failed = gates.loc[~gates["passed"], "gate"].tolist()
    return f"""# SEC late-filing Stage A

## Direct result

Candidate decision: **{decision}**. Historical alpha: **INVALID**. Model alpha: **NOT VALIDATED**. Deployment: **NOT AUTHORIZED**.

The frozen confirmation cohort contains {schedule_counts['confirmation']} scheduled events before price-path checks and {summary['events']} retained outcomes across {summary['tickers']} tickers and {summary['entry_dates']} entry dates.

- Gross 20-session short-stock/long-SPY mean per gross dollar: {metric(gross.get('mean'))}; HAC t={metric(gross.get('t'))}.
- Primary net at 30 bps one-way: {metric(primary.get('mean'))}; HAC t={metric(primary.get('t'))}; raw one-sided p={metric(primary.get('one_sided_p'))}; global-floor adjusted p={metric(primary.get('known_floor_adjusted_screening_p'))}.
- Stress net at 50 bps one-way: {metric(stress.get('mean'))}; HAC t={metric(stress.get('t'))}.
- Signed pre-event short-pair mean: {metric(pre.get('mean'))}; HAC t={metric(pre.get('t'))}.
- Post-minus-pre mean: {metric(post_pre.get('mean'))}; HAC t={metric(post_pre.get('t'))}.
- Centered circular-block bootstrap sensitivity p={metric(bootstrap.get('one_sided_p'))}.
- Maximum $10m stock trade / trailing ADV: {metric(summary['maximum_trade_to_adv_at_10m'])}.

Failed frozen gates: {', '.join(failed) if failed else 'none'}.

## Interpretation

The hypothesis was fixed before any price value was read. Prior research finds that late expected earnings announcements and formal SEC late-filing notifications are associated with adverse news and negative abnormal returns. Penman reports roughly 1% pre-cost abnormal returns over 20 days in an early sample; Bagnoli, Kross, and Watts show that missing management's expected announcement date conveys negative earnings information; Bartov and Konchitchki report -2.44% momentum-adjusted returns around Form NT filings. These magnitudes are priors, not expected performance for this next-open implementation.

The experiment specifically asks whether a tradable residual remains after exact SEC acceptance, a three-minute processing buffer, next-open entry, a 20-session hold, both-leg turnover costs, overlap suppression, liquidity screening, date-level aggregation, HAC inference, global trial-floor screening, concentration, leave-one-ticker-out, and capacity gates.

## Evidence boundary

- Frozen XNYS/benchmark reconciliation: {coverage['benchmark_sessions']} exact sessions, date and open timestamps matched.
- Development outcomes are descriptive. Confirmation years 2022-2024 contain the single frozen primary test.
- All 2025 Form NT events remain signal-only; no 2025 event outcome is joined or reported.
- The universe is a 2026 current-survivor proxy. It lacks delisting returns and permanent security identifiers.
- Yahoo adjusted opens, historical borrow, financing, recall, and TAQ impact remain production blockers.

## Research sources

- [Penman (1984), Abnormal Returns to Investment Strategies Based on the Timing of Earnings Reports](https://business.columbia.edu/faculty/research/abnormal-returns-investment-strategies-based-timing-earnings-reports)
- [Bagnoli, Kross, and Watts (2002), A Day Late, a Penny Short](https://doi.org/10.1111/1475-679X.t01-1-00054)
- [Bartov and Konchitchki (2017), SEC Filings, Regulatory Deadlines, and Capital Market Consequences](https://faculty.haas.berkeley.edu/yaniv/files/Papers_Publications/SEC-Filings-Regulatory-Deadlines-Capital-Market-Consequences_Bartov-Konchitchki_AH.pdf)
- [SEC Financial Reporting Manual, periodic-report deadlines](https://www.sec.gov/about/divisions-offices/division-corporation-finance/financial-reporting-manual/frm-topic-1)
"""


def run(args: argparse.Namespace) -> Path:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"non-overwriting run directory exists: {output}")
    receipt = verify_design_receipt(args.design_receipt)
    verify_plan_manifest(args.plan_manifest)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != "sec_late_filing_stage_a_protocol_v1":
        raise ValueError("unsupported SEC late-filing protocol")

    plan_dir = args.plan_manifest.parent
    candidates = pd.read_parquet(plan_dir / "candidate_events.parquet")
    calendar = pd.read_csv(args.calendar)
    schedule, schedule_exclusions = build_late_filing_schedule(
        candidates,
        calendar,
        decision_lag_minutes=int(config["event_definition"]["decision_lag_minutes"]),
        holding_sessions=20,
    )
    sealed = schedule.loc[schedule["phase"].eq("sealed_2025")].copy()
    outcome_schedule = schedule.loc[~schedule["phase"].eq("sealed_2025")].copy()
    if sealed.empty or outcome_schedule.empty:
        raise ValueError("schedule must contain sealed and outcome cohorts")
    if not sealed["outcome_join_permitted"].eq(False).all():
        raise ValueError("sealed 2025 event unexpectedly permits outcomes")
    schedule_counts = {
        phase: int(schedule["phase"].eq(phase).sum())
        for phase in ("development", "confirmation", "sealed_2025")
    }

    output.mkdir(parents=True)
    schedule_path = output / "preoutcome_schedule.parquet"
    sealed_path = output / "sealed_2025_signal_only.parquet"
    schedule_exclusion_path = output / "schedule_exclusions.csv"
    schedule.to_parquet(schedule_path, index=False)
    sealed.to_parquet(sealed_path, index=False)
    schedule_exclusions.to_csv(schedule_exclusion_path, index=False)

    request = {
        "schema_version": "sec_late_filing_outcome_request_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "design_receipt_sha256": sha256_file(args.design_receipt),
        "candidate_schedule_sha256": sha256_file(schedule_path),
        "sealed_schedule_sha256": sha256_file(sealed_path),
        "formal_trial_id": "SEC_NT_COMBINED_CONFIRMATION_2022_2024_NET30",
        "outcome_accessions": sorted(outcome_schedule["accession"].astype(str).tolist()),
        "sealed_accessions_excluded": sorted(sealed["accession"].astype(str).tolist()),
        "price_value_read_started": False,
    }
    request_path = output / "frozen_outcome_request.json"
    write_json(request_path, request)
    marker = {
        "schema_version": "sec_late_filing_preread_marker_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "design_status": receipt["status"],
        "outcome_request_sha256": sha256_file(request_path),
        "price_file_sha256_verified": sha256_file(args.prices),
        "price_values_read": False,
    }
    marker_path = output / "pre_price_read_marker.json"
    write_json(marker_path, marker)

    start = pd.Timestamp(outcome_schedule["pre_date"].min()).normalize()
    end = pd.Timestamp(outcome_schedule["exit_date"].max()).normalize()
    tickers = sorted(set(outcome_schedule["ticker"].astype(str)) | {"SPY"})
    prices = read_filtered_prices(args.prices, tickers=tickers, start=start, end=end)
    benchmark_coverage = reconcile_benchmark_to_exchange_calendar(
        prices,
        calendar,
        required_start=start,
        required_end=end,
        benchmark="SPY",
    )
    event_definition = config["event_definition"]
    events, price_exclusions = attach_late_filing_outcomes(
        outcome_schedule,
        prices,
        calendar,
        cost_bps_one_way=event_definition["cost_bps_one_way"],
        adv_window_sessions=20,
        minimum_adv_usd=float(event_definition["minimum_trailing_adv_usd"]),
        portfolio_aum_usd=float(event_definition["portfolio_aum_usd"]),
    )
    if events["phase"].eq("sealed_2025").any():
        raise ValueError("sealed 2025 event outcome was joined")

    development = events.loc[events["phase"].eq("development")].copy()
    confirmation = events.loc[events["phase"].eq("confirmation")].copy()
    trial_floor = int(config["formal_tests"]["current_disclosed_trial_floor"])
    development_summary = summarize_late_filing_events(development, trial_floor=trial_floor)
    confirmation_summary = summarize_late_filing_events(confirmation, trial_floor=trial_floor)
    gates = gate_late_filing_confirmation(
        confirmation_summary,
        gates=config["confirmation_gates"],
        scheduled_events=schedule_counts["confirmation"],
    )
    efficacy_pass = bool(gates["passed"].all())
    decision = "RETAIN_FOR_PROSPECTIVE_REPLICATION" if efficacy_pass else "RETIRE_STAGE_A"

    events_path = output / "event_returns.csv"
    entry_path = output / "confirmation_entry_date_returns.csv"
    price_exclusion_path = output / "price_path_exclusions.csv"
    gate_path = output / "validation_gates.csv"
    summary_path = output / "summary_metrics.json"
    report_path = output / "report-source.md"
    events.to_csv(events_path, index=False)
    aggregate_entry_date_returns(confirmation).to_csv(entry_path, index=False)
    price_exclusions.to_csv(price_exclusion_path, index=False)
    gates.to_csv(gate_path, index=False)
    result_summary = {
        "schema_version": "sec_late_filing_stage_a_summary_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "candidate_decision": decision,
        "efficacy_gates_passed": efficacy_pass,
        "historical_alpha_verdict": "INVALID",
        "model_alpha_verdict": "NOT VALIDATED",
        "deployment_verdict": "NOT AUTHORIZED",
        "schedule_counts": schedule_counts,
        "development": development_summary,
        "confirmation": confirmation_summary,
        "benchmark_calendar_reconciliation": benchmark_coverage,
        "sealed_2025_event_outcomes_joined": 0,
        "hard_production_blocks": config["hard_claim_limits"],
    }
    write_json(summary_path, result_summary)
    report_path.write_text(
        build_report(
            summary=confirmation_summary,
            gates=gates,
            schedule_counts=schedule_counts,
            coverage=benchmark_coverage,
            decision=decision,
        ),
        encoding="utf-8",
    )

    inputs = [
        record(args.design_receipt, role="design_receipt"),
        record(args.config, role="human_protocol"),
        record(args.plan_manifest, role="preoutcome_plan_manifest"),
        record(args.prices, role="frozen_retrospective_prices"),
        record(args.calendar, role="frozen_xnys_calendar"),
        record(Path(__file__), role="outcome_runner"),
        record(SRC / "alpha_factor_engine/research/sec_late_filing_outcomes.py", role="outcome_core"),
    ]
    outputs = [
        record(schedule_path, role="preoutcome_schedule"),
        record(sealed_path, role="sealed_2025_signal_only"),
        record(schedule_exclusion_path, role="schedule_exclusions"),
        record(request_path, role="frozen_outcome_request"),
        record(marker_path, role="pre_price_read_marker"),
        record(events_path, role="event_returns"),
        record(entry_path, role="confirmation_entry_date_returns"),
        record(price_exclusion_path, role="price_path_exclusions"),
        record(gate_path, role="validation_gates"),
        record(summary_path, role="summary_metrics"),
        record(report_path, role="research_report"),
    ]
    manifest = {
        "schema_version": "sec_late_filing_stage_a_run_manifest_v1",
        "created_at": result_summary["created_at"],
        "candidate_decision": decision,
        "inputs": inputs,
        "outputs": outputs,
        "sealed_2025_event_outcomes_joined": 0,
    }
    write_json(output / "run_manifest.json", manifest)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sec_late_filing_stage_a_v1.json")
    parser.add_argument("--plan-manifest", type=Path, default=ROOT / "artifacts/sec_late_filing_stage_a_v1/preoutcome_plan_v2/plan_manifest.json")
    parser.add_argument("--design-receipt", type=Path, default=ROOT / "artifacts/sec_late_filing_stage_a_v1/design_receipt_v1.json")
    parser.add_argument("--prices", type=Path, default=ROOT / "artifacts/public_broad_proxy_v1/data/daily_prices.parquet")
    parser.add_argument("--calendar", type=Path, default=ROOT / "artifacts/reference_calendars/xnys_v1/xnys_regular_sessions_2014_2028.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/sec_late_filing_stage_a_v1/run_v1")
    args = parser.parse_args()
    print(run(args))


if __name__ == "__main__":
    main()
