#!/usr/bin/env python3
"""Freeze reporting-lag code, protocol, inputs, and runner before outcomes."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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
    path = (ROOT / str(item["path"])).resolve()
    if not path.is_file() or path.stat().st_size != int(item["bytes"]) or sha256_file(path) != item["sha256"]:
        raise ValueError(f"plan record mismatch: {path}")


def main() -> None:
    output = ROOT / "artifacts/sec_reporting_lag_stage_a_v1/design_receipt_v1.json"
    if output.exists():
        raise FileExistsError(f"non-overwriting design receipt exists: {output}")
    plan_dir = ROOT / "artifacts/sec_reporting_lag_stage_a_v1/preoutcome_plan_v2"
    manifest_path = plan_dir / "plan_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("outcome_inputs_read") is not False or manifest.get("outcome_join_performed") is not False:
        raise ValueError("plan is not preoutcome")
    for item in list(manifest["inputs"]) + list(manifest["outputs"]):
        verify(item)
    summary = json.loads((plan_dir / "plan_summary.json").read_text(encoding="utf-8"))
    records = [
        record(ROOT / "configs/sec_reporting_lag_stage_a_v1.json", "human_protocol"),
        record(manifest_path, "preoutcome_plan_manifest"),
        record(plan_dir / "monthly_tail_schedule.parquet", "frozen_monthly_tail_schedule"),
        record(plan_dir / "plan_summary.json", "preoutcome_plan_summary"),
        record(ROOT / "artifacts/public_broad_proxy_v1/data/frozen_proxy_universe.csv", "frozen_current_survivor_universe"),
        record(ROOT / "artifacts/public_broad_proxy_v1/data/daily_prices.parquet", "frozen_retrospective_prices"),
        record(ROOT / "artifacts/reference_calendars/xnys_v1/xnys_regular_sessions_2014_2028.csv", "frozen_xnys_calendar"),
        record(ROOT / "artifacts/reference_calendars/xnys_v1/calendar_receipt.json", "xnys_calendar_receipt"),
        record(ROOT / "src/alpha_factor_engine/research/sec_reporting_lag_stage_a.py", "signal_core"),
        record(ROOT / "src/alpha_factor_engine/research/sec_filing_change_stage_a.py", "calendar_inference_dependency"),
        record(ROOT / "src/alpha_factor_engine/research/champion_challenger.py", "capacity_dependency"),
        record(ROOT / "scripts/build_sec_reporting_lag_plan.py", "plan_builder"),
        record(ROOT / "scripts/run_sec_reporting_lag_stage_a.py", "outcome_runner"),
        record(Path(__file__), "design_receipt_builder"),
    ]
    payload = {
        "schema_version": "sec_reporting_lag_design_receipt_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "experiment": "sec_reporting_lag_stage_a_v1",
        "status": "FROZEN_PRE_OUTCOME",
        "historical_alpha_verdict": "INVALID",
        "model_alpha_verdict": "NOT VALIDATED",
        "deployment_verdict": "NOT AUTHORIZED",
        "primary_trial_id": "SEC_10K_REPORTING_LAG_CONFIRMATION_2022_2024_NET30",
        "current_disclosed_trial_floor": 132,
        "confirmation_months": 36,
        "confirmation_signals": next(row["signals"] for row in summary["phases"] if row["phase"] == "confirmation"),
        "sealed_2025_signals": next(row["signals"] for row in summary["phases"] if row["phase"] == "sealed_2025"),
        "outcome_values_read": False,
        "outcome_join_performed": False,
        "non_overwrite_policy": True,
        "research_priors": [
            {"claim": "Faster reporting firms earn a premium in the authors' public abstract.", "source": "Filing, Fast and Slow SSRN abstract", "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3432446"},
            {"claim": "The monthly shortest-minus-longest lag construction and published gross magnitudes are taken from a secondary methodology summary and are not treated as verified primary text.", "source": "CXO Advisory summary", "url": "https://www.cxoadvisory.com/fundamental-valuation/timely-firms-have-higher-returns/"},
        ],
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
