#!/usr/bin/env python3
"""Freeze the Stage B diagnostic runner before forward-return construction."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "artifacts/sec_reporting_lag_tradable_stage_b_v1/preoutcome_plan_v1"


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
    output = ROOT / "artifacts/sec_reporting_lag_tradable_stage_b_v1/design_receipt_v1.json"
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((PLAN / "plan_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("outcome_join_performed") is not False or manifest.get("entry_exit_return_constructed") is not False:
        raise ValueError("plan is not pre-forward-return")
    for item in manifest["inputs"] + manifest["outputs"]:
        verify(item)
    summary = json.loads((PLAN / "plan_summary.json").read_text(encoding="utf-8"))
    records = [record(ROOT / "configs/sec_reporting_lag_tradable_stage_b_v1.json", "human_protocol"), record(PLAN / "plan_manifest.json", "plan_manifest"), record(PLAN / "monthly_tail_schedule.parquet", "frozen_schedule"), record(PLAN / "monthly_eligibility_ledger.parquet", "eligibility_ledger"), record(ROOT / "artifacts/public_broad_proxy_v1/data/daily_prices.parquet", "retrospective_prices"), record(ROOT / "artifacts/reference_calendars/xnys_v1/xnys_regular_sessions_2014_2028.csv", "xnys_calendar"), record(ROOT / "src/alpha_factor_engine/research/sec_reporting_lag_tradable_stage_b.py", "plan_core"), record(ROOT / "src/alpha_factor_engine/research/sec_filing_change_stage_a.py", "portfolio_dependency"), record(ROOT / "src/alpha_factor_engine/research/champion_challenger.py", "capacity_dependency"), record(ROOT / "scripts/build_sec_reporting_lag_tradable_stage_b_plan.py", "plan_builder"), record(ROOT / "scripts/run_sec_reporting_lag_tradable_stage_b.py", "outcome_runner"), record(Path(__file__), "design_receipt_builder")]
    opened = next(row for row in summary["phases"] if row["phase"] == "opened_diagnostic")
    payload = {"schema_version": "sec_reporting_lag_tradable_stage_b_design_receipt_v1", "created_at": datetime.now(UTC).isoformat(), "status": "FROZEN_PRE_FORWARD_RETURN_JOIN", "research_role": "POST_OUTCOME_DIAGNOSTIC_NO_CONFIRMATORY_CLAIM", "current_disclosed_trial_floor": 133, "inference_permitted": False, "opened_diagnostic_months": opened["months"], "opened_diagnostic_tail_signals": opened["tail_signals"], "sealed_outcome_join_performed": False, "records": records}
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
