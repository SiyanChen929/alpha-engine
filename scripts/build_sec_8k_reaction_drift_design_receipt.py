#!/usr/bin/env python3
"""Freeze the Item 2.02 reaction-drift design before price access."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "artifacts/sec_8k_reaction_drift_stage_a_v1/preoutcome_plan_v1"
OUT = ROOT / "artifacts/sec_8k_reaction_drift_stage_a_v1/design_receipt_v1.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path, role: str) -> dict[str, object]:
    path = path.resolve()
    return {"role": role, "path": path.relative_to(ROOT.resolve()).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def verify(item: dict[str, object]) -> None:
    path = ROOT / item["path"]
    if not path.is_file() or path.stat().st_size != int(item["bytes"]) or sha256_file(path) != item["sha256"]:
        raise ValueError(f"plan record mismatch: {path}")


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    manifest = json.loads((PLAN / "plan_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("price_values_read") is not False or manifest.get("outcome_join_performed") is not False:
        raise ValueError("plan is not pre-price")
    for item in manifest["inputs"] + manifest["outputs"]:
        verify(item)
    summary = json.loads((PLAN / "plan_summary.json").read_text(encoding="utf-8"))
    records = [record(ROOT / "configs/sec_8k_reaction_drift_stage_a_v1.json", "human_protocol"), record(PLAN / "plan_manifest.json", "plan_manifest"), record(PLAN / "event_clock_schedule.parquet", "frozen_event_clock_schedule"), record(PLAN / "plan_summary.json", "plan_summary"), record(ROOT / "artifacts/public_broad_proxy_v1/data/daily_prices.parquet", "retrospective_prices"), record(ROOT / "artifacts/reference_calendars/xnys_v1/xnys_regular_sessions_2014_2028.csv", "xnys_calendar"), record(ROOT / "src/alpha_factor_engine/data/sec_8k_events.py", "item_202_parser"), record(ROOT / "src/alpha_factor_engine/research/sec_8k_reaction_drift.py", "event_clock_core"), record(ROOT / "src/alpha_factor_engine/research/sec_8k_reaction_outcomes.py", "outcome_core"), record(ROOT / "scripts/build_sec_8k_reaction_drift_plan.py", "plan_builder"), record(ROOT / "scripts/run_sec_8k_reaction_drift_stage_a.py", "outcome_runner"), record(Path(__file__), "design_receipt_builder")]
    confirmation = next(row for row in summary["phases"] if row["phase"] == "confirmation")
    sealed = next(row for row in summary["phases"] if row["phase"] == "sealed_2025")
    payload = {"schema_version": "sec_8k_reaction_drift_design_receipt_v1", "created_at": datetime.now(UTC).isoformat(), "status": "FROZEN_PRE_PRICE", "primary_trial_id": "SEC_8K_202_REACTION_DRIFT_CONFIRMATION_2022_2024_NET30", "current_disclosed_trial_floor": 134, "confirmation_scheduled_events": confirmation["events"], "sealed_2025_clock_events": sealed["events"], "price_values_read": False, "outcome_join_performed": False, "historical_alpha_verdict": "INVALID", "model_alpha_verdict": "NOT VALIDATED", "deployment_verdict": "NOT_AUTHORIZED", "records": records}
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
