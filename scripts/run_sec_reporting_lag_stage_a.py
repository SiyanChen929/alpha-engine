#!/usr/bin/env python3
"""Run the frozen monthly SEC reporting-lag experiment."""

from __future__ import annotations

import argparse
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
from alpha_factor_engine.research.sec_filing_change_stage_a import (
    calendar_portfolio_returns,
    reconcile_benchmark_to_exchange_calendar,
    summarize_monthly_portfolio,
)


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
    seen: set[Path] = set()
    for item in records:
        path = (ROOT / str(item["path"])).resolve()
        path.relative_to(ROOT.resolve())
        if path in seen or not path.is_file():
            raise ValueError(f"duplicate or absent frozen record: {path}")
        seen.add(path)
        if path.stat().st_size != int(item["bytes"]) or sha256_file(path) != item["sha256"]:
            raise ValueError(f"frozen record mismatch: {path}")


def verify_design(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "sec_reporting_lag_design_receipt_v1" or payload.get("status") != "FROZEN_PRE_OUTCOME":
        raise ValueError("reporting-lag design is not frozen")
    if payload.get("outcome_values_read") is not False:
        raise ValueError("design receipt already records outcome access")
    verify_records(payload["records"])
    return payload


def read_prices(path: Path, tickers: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return pd.read_parquet(
        path,
        columns=["ticker", "date", "timestamp", "close", "volume", "adj_open"],
        filters=[("date", ">=", start.to_pydatetime()), ("date", "<=", end.to_pydatetime()), ("ticker", "in", tickers)],
    )


def capacity_frame(holdings: pd.DataFrame, prices: pd.DataFrame, adv_window: int) -> pd.DataFrame:
    all_holdings = holdings.loc[holdings["portfolio_scope"].eq("ALL")].copy()
    target = (
        all_holdings.groupby(["month", "entry_date", "ticker"], sort=True)
        .agg(weight=("target_weight_component", "sum"), target_return=("gross_return", "first"))
        .reset_index()
    )
    panel = prices.copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel["dollar_volume"] = pd.to_numeric(panel["close"], errors="coerce") * pd.to_numeric(panel["volume"], errors="coerce")
    by_ticker = {ticker: group.sort_values("date") for ticker, group in panel.groupby("ticker", sort=False)}
    advs: list[float] = []
    signal_times: list[pd.Timestamp] = []
    for row in target.itertuples(index=False):
        history = by_ticker[str(row.ticker)]
        history = history.loc[history["date"].lt(pd.Timestamp(row.entry_date))].tail(adv_window)
        if len(history) != adv_window:
            raise ValueError(f"capacity ADV history incomplete for {row.ticker} {row.entry_date}")
        advs.append(float(history["dollar_volume"].median()))
        signal_times.append(pd.Timestamp(row.entry_date).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30))
    target["adv_usd_3m"] = advs
    target["signal_at"] = signal_times
    return target[["signal_at", "ticker", "weight", "target_return", "adv_usd_3m"]]


def gate_table(summary: dict[str, object], capacity: dict[str, object], config: dict[str, object]) -> pd.DataFrame:
    gates = config["confirmation_gates"]
    net30 = summary["tests"]["net30"]
    net50 = summary["tests"]["net50"]
    rows = [
        ("complete_36_month_calendar", summary["portfolio_months"] >= int(gates["minimum_portfolio_months"]) and summary["expected_month_coverage"] == 1.0, {"months": summary["portfolio_months"], "coverage": summary["expected_month_coverage"]}),
        ("minimum_unique_cik", summary["unique_cik"] >= int(gates["minimum_unique_cik"]), summary["unique_cik"]),
        ("primary_net30_mean_positive", net30["mean"] > float(gates["primary_net_mean_minimum"]), net30["mean"]),
        ("primary_net30_hac_t", net30["t"] >= float(gates["primary_newey_west_t_minimum"]), net30["t"]),
        ("global_floor_adjusted_p", net30["known_floor_adjusted_screening_p"] <= float(gates["global_adjusted_p_maximum"]), net30["known_floor_adjusted_screening_p"]),
        ("stress_net50_mean_positive", net50["mean"] > float(gates["stress_net_mean_minimum"]), net50["mean"]),
        ("month_contribution", summary["maximum_month_absolute_return_contribution_share"] <= float(gates["maximum_month_absolute_return_contribution_share"]), summary["maximum_month_absolute_return_contribution_share"]),
        ("single_cik_weight", summary["maximum_single_cik_gross_weight_share"] <= float(gates["maximum_single_cik_gross_weight_share"]), summary["maximum_single_cik_gross_weight_share"]),
        ("leave_one_cik_out_positive", bool(summary["leave_one_cik_out_complete"]) and summary["minimum_leave_one_cik_out_net30_mean"] > float(gates["minimum_leave_one_cik_out_net30_mean"]), summary["minimum_leave_one_cik_out_net30_mean"]),
        ("capacity_at_10m", bool(capacity["capacity_passed"]), capacity["maximum_adv_participation_observed"]),
    ]
    return pd.DataFrame([{"gate": name, "passed": bool(passed), "required": True, "detail": detail} for name, passed, detail in rows])


def plot_results(monthly: pd.DataFrame, schedule: pd.DataFrame, path: Path) -> None:
    frame = monthly.loc[monthly["portfolio_scope"].eq("ALL")].sort_values("month").copy()
    frame["date"] = pd.PeriodIndex(frame["month"], freq="M").to_timestamp("M")
    frame["gross_wealth"] = (1 + frame["q5_minus_q1_gross_return"]).cumprod()
    frame["net30_wealth"] = (1 + frame["q5_minus_q1_net30_return"]).cumprod()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(frame["date"], frame["gross_wealth"], label="Gross", linewidth=1.8)
    axes[0].plot(frame["date"], frame["net30_wealth"], label="Net 30 bps", linewidth=1.8)
    axes[0].axhline(1, color="black", linewidth=0.7)
    axes[0].set_title("Confirmation cumulative wealth")
    axes[0].legend(frameon=False)
    tails = schedule.groupby("direction")["reporting_lag_days"].apply(list)
    axes[1].hist(tails.get(1, []), bins=20, alpha=.65, label="Fastest quintile")
    axes[1].hist(tails.get(-1, []), bins=20, alpha=.65, label="Slowest quintile")
    axes[1].set_title("Frozen reporting-lag tails")
    axes[1].set_xlabel("Calendar days")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run(args: argparse.Namespace) -> Path:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"non-overwriting output exists: {output}")
    receipt = verify_design(args.design_receipt)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    plan_dir = args.plan_manifest.parent
    plan_manifest = json.loads(args.plan_manifest.read_text(encoding="utf-8"))
    if plan_manifest.get("outcome_inputs_read") is not False:
        raise ValueError("plan is not preoutcome")
    verify_records(list(plan_manifest["inputs"]) + list(plan_manifest["outputs"]))
    schedule = pd.read_parquet(plan_dir / "monthly_tail_schedule.parquet")
    sealed = schedule.loc[schedule["phase"].eq("sealed_2025")].copy()
    if sealed.empty or not sealed["outcome_join_permitted"].eq(False).all():
        raise ValueError("sealed cohort contract failed")
    outcome_schedule = schedule.loc[schedule["phase"].isin(["development", "confirmation"])].copy()
    output.mkdir(parents=True)
    sealed_path = output / "sealed_2025_signal_only.parquet"
    sealed.to_parquet(sealed_path, index=False)
    request = {
        "schema_version": "sec_reporting_lag_outcome_request_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "design_receipt_sha256": sha256_file(args.design_receipt),
        "formal_trial_id": config["formal_tests"]["primary_trial_id"],
        "outcome_formation_months": sorted(outcome_schedule["formation_month"].unique().tolist()),
        "sealed_formation_months_excluded": sorted(sealed["formation_month"].unique().tolist()),
        "price_value_read_started": False,
    }
    request_path = output / "frozen_outcome_request.json"
    write_json(request_path, request)
    marker_path = output / "pre_price_read_marker.json"
    write_json(marker_path, {"schema_version": "sec_reporting_lag_preread_marker_v1", "created_at": datetime.now(UTC).isoformat(), "design_status": receipt["status"], "outcome_request_sha256": sha256_file(request_path), "price_file_sha256_verified": sha256_file(args.prices), "price_values_read": False})

    start = pd.Timestamp("2018-01-01")
    end = pd.Timestamp(outcome_schedule["calendar_exit_date"].max()).normalize()
    tickers = sorted(set(outcome_schedule["ticker"].astype(str)) | {"SPY"})
    prices = read_prices(args.prices, tickers, start, end)
    calendar = pd.read_csv(args.calendar)
    coverage = reconcile_benchmark_to_exchange_calendar(prices, calendar, required_start="2019-02-01", required_end=end, benchmark="SPY")
    signal = config["signal_definition"]
    phase_outputs: dict[str, object] = {}
    holding_parts: list[pd.DataFrame] = []
    monthly_parts: list[pd.DataFrame] = []
    exclusion_parts: list[pd.DataFrame] = []
    for phase in ("development", "confirmation"):
        phase_schedule = outcome_schedule.loc[outcome_schedule["phase"].eq(phase)].copy()
        holdings, monthly, exclusions = calendar_portfolio_returns(
            phase_schedule,
            prices,
            holding_months=1,
            minimum_prior_close=float(signal["minimum_prior_close"]),
            adv_window=int(signal["adv_window_sessions"]),
            minimum_prior_stock_sessions=int(signal["minimum_prior_stock_sessions"]),
            minimum_prior_adv=float(signal["minimum_trailing_adv_usd"]),
            primary_one_way_cost=float(signal["primary_cost_bps_one_way"]) / 10000,
            stress_one_way_cost=float(signal["stress_cost_bps_one_way"]) / 10000,
        )
        holdings["phase"] = phase
        monthly["phase"] = phase
        exclusions["phase"] = phase
        spec = config["cohorts"][phase]
        summary = summarize_monthly_portfolio(
            phase_schedule,
            holdings,
            monthly,
            expected_month_start=spec["formation_month_start"],
            expected_month_end=spec["formation_month_end"],
            trial_floor=int(config["formal_tests"]["current_disclosed_trial_floor"]),
            hac_lags=3,
            primary_one_way_cost=float(signal["primary_cost_bps_one_way"]) / 10000,
            stress_one_way_cost=float(signal["stress_cost_bps_one_way"]) / 10000,
            inferential_test_names=("net30",) if phase == "confirmation" else (),
            formal_mean_gate_names=(),
            inference_allowed=phase == "confirmation",
        )
        phase_outputs[phase] = summary
        holding_parts.append(holdings)
        monthly_parts.append(monthly)
        exclusion_parts.append(exclusions)
    all_holdings = pd.concat(holding_parts, ignore_index=True)
    all_monthly = pd.concat(monthly_parts, ignore_index=True)
    all_exclusions = pd.concat(exclusion_parts, ignore_index=True)
    confirmation_holdings = all_holdings.loc[all_holdings["phase"].eq("confirmation")]
    confirmation_prices = prices.loc[prices["date"].between(pd.Timestamp("2021-01-01"), end)]
    cap_input = capacity_frame(confirmation_holdings, confirmation_prices, int(signal["adv_window_sessions"]))
    cap_summary, cap_trades = capacity_diagnostics(cap_input, portfolio_aum_usd=float(signal["portfolio_aum_usd"]), maximum_adv_participation=float(config["confirmation_gates"]["maximum_trade_to_trailing_adv_at_10m"]))
    gates = gate_table(phase_outputs["confirmation"], cap_summary, config)
    decision = "RETAIN_FOR_PROSPECTIVE_REPLICATION" if bool(gates["passed"].all()) else "RETIRE_STAGE_A"

    holdings_path = output / "monthly_holdings.csv"
    monthly_path = output / "monthly_portfolio_returns.csv"
    exclusions_path = output / "price_path_exclusions.csv"
    cap_path = output / "capacity_trades.csv"
    gates_path = output / "validation_gates.csv"
    chart_path = output / "confirmation_result.png"
    all_holdings.to_csv(holdings_path, index=False)
    all_monthly.to_csv(monthly_path, index=False)
    all_exclusions.to_csv(exclusions_path, index=False)
    cap_trades.to_csv(cap_path, index=False)
    gates.to_csv(gates_path, index=False)
    plot_results(all_monthly.loc[all_monthly["phase"].eq("confirmation")], outcome_schedule.loc[outcome_schedule["phase"].eq("confirmation")], chart_path)
    summary = {
        "schema_version": "sec_reporting_lag_stage_a_summary_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "candidate_decision": decision,
        "efficacy_gates_passed": bool(gates["passed"].all()),
        "historical_alpha_verdict": "INVALID",
        "model_alpha_verdict": "NOT VALIDATED",
        "deployment_verdict": "NOT AUTHORIZED",
        "development": phase_outputs["development"],
        "confirmation": phase_outputs["confirmation"],
        "capacity": cap_summary,
        "benchmark_calendar_reconciliation": coverage,
        "sealed_2025_signal_outcomes_joined": 0,
        "hard_production_blocks": config["hard_claim_limits"],
    }
    summary_path = output / "summary_metrics.json"
    write_json(summary_path, summary)
    c = phase_outputs["confirmation"]
    report_path = output / "report-source.md"
    report_path.write_text(f"""# SEC 10-K reporting-lag Stage A

## Result

Candidate decision: **{decision}**. Historical alpha: **INVALID**. Model alpha: **NOT VALIDATED**. Deployment: **NOT AUTHORIZED**.

The frozen confirmation test covers {c['portfolio_months']} monthly portfolios and {c['unique_cik']} issuers. Gross mean={c['tests']['gross']['mean']:.6f}; net-30 mean={c['tests']['net30']['mean']:.6f}, HAC t={c['tests']['net30']['t']:.3f}, global-floor adjusted p={c['tests']['net30']['known_floor_adjusted_screening_p']:.6f}; net-50 mean={c['tests']['net50']['mean']:.6f}. Maximum $10m trade/ADV={cap_summary['maximum_adv_participation_observed']:.6f}.

Failed frozen gates: {', '.join(gates.loc[~gates['passed'], 'gate'].tolist()) or 'none'}.

## Evidence boundary

- Signal construction used only public SEC header clocks and the frozen XNYS calendar before price access.
- 2025 signals remain sealed; zero 2025 signal outcomes were joined.
- The universe is a 2026 current-survivor proxy. Historical conclusions remain INVALID.
- The design is paper-inspired because the original PDF could not be directly inspected and accelerated-filer status is unavailable.

## Research sources

- [Filing, Fast and Slow: Reporting Lag and Stock Returns, public SSRN abstract](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3432446)
- [Secondary methodology summary from CXO Advisory](https://www.cxoadvisory.com/fundamental-valuation/timely-firms-have-higher-returns/)
""", encoding="utf-8")
    outputs = [
        record(sealed_path, "sealed_2025_signal_only"), record(request_path, "frozen_outcome_request"), record(marker_path, "pre_price_read_marker"),
        record(holdings_path, "monthly_holdings"), record(monthly_path, "monthly_portfolio_returns"), record(exclusions_path, "price_path_exclusions"),
        record(cap_path, "capacity_trades"), record(gates_path, "validation_gates"), record(chart_path, "result_chart"), record(summary_path, "summary_metrics"), record(report_path, "research_report"),
    ]
    write_json(output / "run_manifest.json", {"schema_version": "sec_reporting_lag_run_manifest_v1", "created_at": summary["created_at"], "candidate_decision": decision, "inputs": [record(args.design_receipt, "design_receipt"), record(args.config, "human_protocol"), record(args.plan_manifest, "preoutcome_plan_manifest"), record(args.prices, "frozen_retrospective_prices"), record(args.calendar, "frozen_xnys_calendar"), record(Path(__file__), "outcome_runner")], "outputs": outputs, "sealed_2025_signal_outcomes_joined": 0})
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sec_reporting_lag_stage_a_v1.json")
    parser.add_argument("--plan-manifest", type=Path, default=ROOT / "artifacts/sec_reporting_lag_stage_a_v1/preoutcome_plan_v2/plan_manifest.json")
    parser.add_argument("--design-receipt", type=Path, default=ROOT / "artifacts/sec_reporting_lag_stage_a_v1/design_receipt_v1.json")
    parser.add_argument("--prices", type=Path, default=ROOT / "artifacts/public_broad_proxy_v1/data/daily_prices.parquet")
    parser.add_argument("--calendar", type=Path, default=ROOT / "artifacts/reference_calendars/xnys_v1/xnys_regular_sessions_2014_2028.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/sec_reporting_lag_stage_a_v1/run_v1")
    print(run(parser.parse_args()))


if __name__ == "__main__":
    main()
