#!/usr/bin/env python3
"""Build a header-only Item 2.02 reaction-drift event-clock plan."""

from __future__ import annotations

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

from alpha_factor_engine.data.sec_8k_events import load_item_202_inventory
from alpha_factor_engine.research.sec_8k_reaction_drift import build_event_clock_plan


CONFIG = ROOT / "configs/sec_8k_reaction_drift_stage_a_v1.json"
UNIVERSE = ROOT / "artifacts/public_broad_proxy_v1/data/frozen_proxy_universe.csv"
RAW_MANIFEST = ROOT / "artifacts/sec_periodic_filing_inventory_v1/raw_manifest.json"
CACHE = ROOT / "data/cache/sec"
CALENDAR = ROOT / "artifacts/reference_calendars/xnys_v1/xnys_regular_sessions_2014_2028.csv"
OUTPUT = ROOT / "artifacts/sec_8k_reaction_drift_stage_a_v1/preoutcome_plan_v1"


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


def verify_raw_manifest() -> list[Path]:
    payload = json.loads(RAW_MANIFEST.read_text(encoding="utf-8"))
    paths = []
    for item in payload["records"]:
        path = Path(item["path"]).resolve(); path.relative_to(ROOT.resolve())
        if item["role"] != "raw_input" or not path.is_file() or path.stat().st_size != int(item["bytes"]) or sha256_file(path) != item["sha256"]:
            raise ValueError(f"raw SEC record mismatch: {path}")
        paths.append(path)
    return paths


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config.get("schema_version") != "sec_8k_reaction_drift_stage_a_protocol_v1":
        raise ValueError("unsupported protocol")
    raw_paths = verify_raw_manifest()
    universe = pd.read_csv(UNIVERSE)
    if len(universe) != config["source_boundaries"]["expected_universe_names"]:
        raise ValueError("universe count differs from protocol")
    definition = config["event_definition"]
    inventory = load_item_202_inventory(CACHE, issuer_ciks=universe["cik"].astype(int), start_date=definition["minimum_filing_date"], end_date=definition["maximum_filing_date"], allowed_source_paths=raw_paths)
    calendar = pd.read_csv(CALENDAR)
    schedule, exclusions, misses = build_event_clock_plan(inventory, universe, calendar, cohorts=config["cohorts"], reaction_delay_minutes=30, holding_sessions=20)
    if schedule["price_values_read"].any() or schedule["outcome_join_permitted"].isna().any():
        raise ValueError("pre-price schedule contract failed")
    sealed = schedule.loc[schedule["phase"].eq("sealed_2025")]
    if sealed.empty or not sealed["outcome_join_permitted"].eq(False).all():
        raise ValueError("sealed cohort contract failed")
    OUTPUT.mkdir(parents=True)
    inventory_path, schedule_path, exclusions_path, misses_path = OUTPUT / "item_202_inventory.parquet", OUTPUT / "event_clock_schedule.parquet", OUTPUT / "event_exclusions.csv", OUTPUT / "mapping_misses.csv"
    inventory.to_parquet(inventory_path, index=False); schedule.to_parquet(schedule_path, index=False); exclusions.to_csv(exclusions_path, index=False); misses.to_csv(misses_path, index=False)
    phases = []
    for phase in config["cohorts"]:
        part = schedule.loc[schedule["phase"].eq(phase)]
        phases.append({"phase": phase, "events": int(len(part)), "issuers": int(part["issuer_cik"].nunique()), "tickers": int(part["ticker"].nunique()), "reaction_dates": int(part["reaction_date"].nunique()), "price_values_read": False})
    date_delta = inventory["filing_acceptance_et_date_delta_days"].value_counts().sort_index()
    summary = {"schema_version": "sec_8k_reaction_drift_plan_summary_v1", "created_at": datetime.now(UTC).isoformat(), "status": "FROZEN_HEADER_AND_CLOCK_ONLY_PRE_PRICE_PLAN", "historical_alpha_verdict": "INVALID", "model_alpha_verdict": "NOT VALIDATED", "deployment_verdict": "NOT AUTHORIZED", "raw_sec_source_records_verified": len(raw_paths), "item_202_headers": int(len(inventory)), "item_202_issuers": int(inventory["issuer_cik"].nunique()), "event_clock_rows": int(len(schedule)), "same_day_and_clock_exclusions": int(len(exclusions)), "mapping_misses": int(len(misses)), "filing_minus_acceptance_et_date_days": {str(k): int(v) for k, v in date_delta.items()}, "phases": phases, "price_values_read": False, "outcome_join_performed": False, "limitations": config["hard_claim_limits"]}
    summary_path = OUTPUT / "plan_summary.json"; write_json(summary_path, summary)
    manifest = {"schema_version": "sec_8k_reaction_drift_plan_manifest_v1", "created_at": summary["created_at"], "inputs": [record(CONFIG, "human_protocol"), record(UNIVERSE, "frozen_current_survivor_universe"), record(RAW_MANIFEST, "sec_raw_manifest"), record(CALENDAR, "frozen_xnys_calendar"), record(Path(__file__), "plan_builder"), record(SRC / "alpha_factor_engine/data/sec_8k_events.py", "item_202_parser"), record(SRC / "alpha_factor_engine/research/sec_8k_reaction_drift.py", "event_clock_core")], "outputs": [record(inventory_path, "item_202_inventory"), record(schedule_path, "frozen_event_clock_schedule"), record(exclusions_path, "event_exclusions"), record(misses_path, "mapping_misses"), record(summary_path, "plan_summary")], "price_values_read": False, "outcome_join_performed": False}
    write_json(OUTPUT / "plan_manifest.json", manifest)
    print(summary_path)


if __name__ == "__main__":
    main()
