#!/usr/bin/env python3
"""Build the post-outcome, tradability-before-ranking diagnostic schedule."""

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

from alpha_factor_engine.research.sec_reporting_lag_tradable_stage_b import build_tradability_first_plan


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


def run(args: argparse.Namespace) -> Path:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"non-overwriting output exists: {output}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != "sec_reporting_lag_tradable_stage_b_protocol_v1":
        raise ValueError("unsupported protocol")
    sources = config["source_boundaries"]
    expected = {"headers": ROOT / sources["mapped_sec_headers"], "universe": ROOT / sources["universe"], "prices": ROOT / sources["price_panel"], "calendar": ROOT / sources["exchange_calendar"]}
    for name, expected_path in expected.items():
        if getattr(args, name).resolve() != expected_path.resolve():
            raise ValueError(f"{name} differs from protocol")
    universe = pd.read_csv(args.universe)
    if len(universe) != int(sources["expected_universe_names"]):
        raise ValueError("universe count differs from protocol")
    headers = pd.read_parquet(args.headers)
    calendar = pd.read_csv(args.calendar)
    prices = pd.read_parquet(args.prices, columns=["ticker", "date", "close", "volume"])
    signal = config["signal_definition"]
    schedule, eligibility = build_tradability_first_plan(
        headers, calendar, prices, cohorts=config["cohorts"],
        minimum_lag_days=int(signal["minimum_reporting_lag_days"]), maximum_lag_days=int(signal["maximum_reporting_lag_days"]),
        maximum_signal_age_days=int(signal["maximum_signal_age_days"]), minimum_prior_close=float(signal["minimum_prior_close"]),
        minimum_prior_stock_sessions=int(signal["minimum_prior_stock_sessions"]), adv_window_sessions=int(signal["adv_window_sessions"]),
        minimum_prior_adv=float(signal["minimum_trailing_adv_usd"]),
    )
    if schedule["outcome_join_performed"].any() or not schedule.loc[schedule["phase"].eq("sealed_2025"), "outcome_join_permitted"].eq(False).all():
        raise ValueError("outcome or sealed contract failed")
    output.mkdir(parents=True)
    schedule_path = output / "monthly_tail_schedule.parquet"
    eligibility_path = output / "monthly_eligibility_ledger.parquet"
    schedule.to_parquet(schedule_path, index=False)
    eligibility.to_parquet(eligibility_path, index=False)
    phases = []
    for phase in config["cohorts"]:
        part = schedule.loc[schedule["phase"].eq(phase)]
        eligible = eligibility.loc[(eligibility["phase"].eq(phase)) & eligibility["eligible_before_sort"]]
        phases.append({"phase": phase, "months": int(part["formation_month"].nunique()), "eligible_before_sort": int(len(eligible)), "tail_signals": int(len(part)), "issuers": int(part["cik"].nunique()), "long_signals": int(part["direction"].eq(1).sum()), "short_signals": int(part["direction"].eq(-1).sum())})
    summary = {
        "schema_version": "sec_reporting_lag_tradable_stage_b_plan_summary_v1", "created_at": datetime.now(UTC).isoformat(),
        "status": "FROZEN_POST_OUTCOME_METHOD_REPAIR_PRE_FORWARD_RETURN_JOIN", "research_role": config["research_role"],
        "historical_alpha_verdict": "INVALID", "model_alpha_verdict": "NOT VALIDATED", "deployment_verdict": "NOT AUTHORIZED",
        "universe_names": int(len(universe)), "phases": phases, "historical_price_predictors_read": True,
        "entry_exit_return_constructed": False, "outcome_join_performed": False, "sealed_2025_outcome_join_performed": False,
        "limitations": config["hard_claim_limits"],
    }
    summary_path = output / "plan_summary.json"
    write_json(summary_path, summary)
    manifest = {
        "schema_version": "sec_reporting_lag_tradable_stage_b_plan_manifest_v1", "created_at": summary["created_at"],
        "inputs": [record(args.config, "human_protocol"), record(args.headers, "mapped_sec_headers"), record(args.universe, "frozen_current_survivor_universe"), record(args.prices, "retrospective_price_predictors"), record(args.calendar, "frozen_xnys_calendar"), record(Path(__file__), "plan_builder"), record(SRC / "alpha_factor_engine/research/sec_reporting_lag_tradable_stage_b.py", "plan_core")],
        "outputs": [record(schedule_path, "frozen_tradability_first_schedule"), record(eligibility_path, "monthly_eligibility_ledger"), record(summary_path, "plan_summary")],
        "historical_price_predictors_read": True, "entry_exit_return_constructed": False, "outcome_join_performed": False,
    }
    write_json(output / "plan_manifest.json", manifest)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sec_reporting_lag_tradable_stage_b_v1.json")
    parser.add_argument("--headers", type=Path, default=ROOT / "artifacts/sec_filing_change_stage_a_v1/preoutcome_plan_v3/mapped_header_inventory.parquet")
    parser.add_argument("--universe", type=Path, default=ROOT / "artifacts/public_broad_proxy_v1/data/frozen_proxy_universe.csv")
    parser.add_argument("--prices", type=Path, default=ROOT / "artifacts/public_broad_proxy_v1/data/daily_prices.parquet")
    parser.add_argument("--calendar", type=Path, default=ROOT / "artifacts/reference_calendars/xnys_v1/xnys_regular_sessions_2014_2028.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/sec_reporting_lag_tradable_stage_b_v1/preoutcome_plan_v1")
    print(run(parser.parse_args()))


if __name__ == "__main__":
    main()
