#!/usr/bin/env python3
"""Freeze the header-only monthly SEC reporting-lag signal schedule."""

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

from alpha_factor_engine.research.sec_reporting_lag_stage_a import build_reporting_lag_plan


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path, role: str) -> dict[str, object]:
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


def run(args: argparse.Namespace) -> Path:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"non-overwriting output exists: {output}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != "sec_reporting_lag_stage_a_protocol_v1":
        raise ValueError("unsupported reporting-lag protocol")
    sources = config["source_boundaries"]
    expected = {
        "headers": ROOT / sources["mapped_sec_headers"],
        "universe": ROOT / sources["universe"],
        "calendar": ROOT / sources["exchange_calendar"],
    }
    for key, path in expected.items():
        if getattr(args, key).resolve() != path.resolve():
            raise ValueError(f"{key} differs from frozen protocol")
    universe = pd.read_csv(args.universe)
    if len(universe) != int(sources["expected_universe_names"]):
        raise ValueError("universe count differs from protocol")
    headers = pd.read_parquet(args.headers)
    calendar = pd.read_csv(args.calendar)
    signal = config["signal_definition"]
    schedule, inventory, exclusions = build_reporting_lag_plan(
        headers,
        calendar,
        cohorts=config["cohorts"],
        minimum_lag_days=int(signal["minimum_reporting_lag_days"]),
        maximum_lag_days=int(signal["maximum_reporting_lag_days"]),
        maximum_signal_age_days=int(signal["maximum_signal_age_days"]),
    )
    if not schedule.loc[schedule["phase"].eq("sealed_2025"), "outcome_join_permitted"].eq(False).all():
        raise ValueError("sealed schedule permits outcomes")
    if schedule["outcome_join_performed"].any():
        raise ValueError("preoutcome schedule contains outcome join")
    output.mkdir(parents=True)
    schedule_path = output / "monthly_tail_schedule.parquet"
    inventory_path = output / "monthly_latest_filing_inventory.parquet"
    exclusions_path = output / "header_exclusions.csv"
    schedule.to_parquet(schedule_path, index=False)
    inventory.to_parquet(inventory_path, index=False)
    exclusions.to_csv(exclusions_path, index=False)
    phase_rows = []
    for phase in config["cohorts"]:
        part = schedule.loc[schedule["phase"].eq(phase)]
        phase_rows.append({
            "phase": phase,
            "months": int(part["formation_month"].nunique()),
            "signals": int(len(part)),
            "issuers": int(part["cik"].nunique()),
            "long_signals": int(part["direction"].eq(1).sum()),
            "short_signals": int(part["direction"].eq(-1).sum()),
            "outcome_join_performed": False,
        })
    summary = {
        "schema_version": "sec_reporting_lag_plan_summary_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "FROZEN_HEADER_ONLY_PRE_OUTCOME_PLAN",
        "historical_alpha_verdict": "INVALID",
        "model_alpha_verdict": "NOT VALIDATED",
        "deployment_verdict": "NOT AUTHORIZED",
        "universe_names": int(len(universe)),
        "source_10k_headers": int(headers["form"].eq("10-K").sum()),
        "monthly_latest_filing_rows": int(len(inventory)),
        "tail_signals": int(len(schedule)),
        "phases": phase_rows,
        "outcome_inputs_read": False,
        "outcome_join_performed": False,
        "limitations": config["hard_claim_limits"],
    }
    summary_path = output / "plan_summary.json"
    write_json(summary_path, summary)
    manifest = {
        "schema_version": "sec_reporting_lag_plan_manifest_v1",
        "created_at": summary["created_at"],
        "inputs": [
            record(args.config, "human_protocol"),
            record(args.headers, "mapped_sec_header_inventory"),
            record(args.universe, "frozen_current_survivor_universe"),
            record(args.calendar, "frozen_xnys_calendar"),
            record(Path(__file__), "plan_builder"),
            record(SRC / "alpha_factor_engine/research/sec_reporting_lag_stage_a.py", "plan_core"),
        ],
        "outputs": [
            record(schedule_path, "frozen_monthly_tail_schedule"),
            record(inventory_path, "monthly_latest_filing_inventory"),
            record(exclusions_path, "header_exclusions"),
            record(summary_path, "plan_summary"),
        ],
        "outcome_inputs_read": False,
        "outcome_join_performed": False,
    }
    manifest_path = output / "plan_manifest.json"
    write_json(manifest_path, manifest)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sec_reporting_lag_stage_a_v1.json")
    parser.add_argument("--headers", type=Path, default=ROOT / "artifacts/sec_filing_change_stage_a_v1/preoutcome_plan_v3/mapped_header_inventory.parquet")
    parser.add_argument("--universe", type=Path, default=ROOT / "artifacts/public_broad_proxy_v1/data/frozen_proxy_universe.csv")
    parser.add_argument("--calendar", type=Path, default=ROOT / "artifacts/reference_calendars/xnys_v1/xnys_regular_sessions_2014_2028.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/sec_reporting_lag_stage_a_v1/preoutcome_plan_v1")
    print(run(parser.parse_args()))


if __name__ == "__main__":
    main()
