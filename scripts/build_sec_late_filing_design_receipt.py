#!/usr/bin/env python3
"""Freeze the SEC late-filing Stage A before any outcome values are read."""

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


def record(path: Path, *, role: str) -> dict[str, object]:
    resolved = path.resolve()
    return {
        "role": role,
        "path": resolved.relative_to(ROOT.resolve()).as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def verify_record(item: dict[str, object]) -> None:
    path = (ROOT / str(item["path"])).resolve()
    path.relative_to(ROOT.resolve())
    if not path.is_file() or path.stat().st_size != int(item["bytes"]):
        raise ValueError(f"record missing or byte-mismatched: {path}")
    if sha256_file(path) != item["sha256"]:
        raise ValueError(f"record hash mismatch: {path}")


def main() -> None:
    output = ROOT / "artifacts/sec_late_filing_stage_a_v1/design_receipt_v1.json"
    if output.exists():
        raise FileExistsError(f"non-overwriting design receipt exists: {output}")
    plan_dir = ROOT / "artifacts/sec_late_filing_stage_a_v1/preoutcome_plan_v2"
    plan_manifest_path = plan_dir / "plan_manifest.json"
    plan_manifest = json.loads(plan_manifest_path.read_text(encoding="utf-8"))
    if plan_manifest.get("outcome_inputs_read") is not False:
        raise ValueError("late-filing plan no longer has pre-outcome status")
    for item in list(plan_manifest["inputs"]) + list(plan_manifest["outputs"]):
        verify_record(item)
    summary = json.loads((plan_dir / "plan_summary.json").read_text(encoding="utf-8"))
    if summary.get("outcome_inputs_read") is not False or summary.get("sealed_2025_event_count") != 15:
        raise ValueError("late-filing plan summary is not the expected pre-outcome plan")

    records = [
        record(ROOT / "configs/sec_late_filing_stage_a_v1.json", role="human_protocol"),
        record(plan_manifest_path, role="preoutcome_plan_manifest"),
        record(plan_dir / "candidate_events.parquet", role="frozen_candidate_events"),
        record(plan_dir / "plan_summary.json", role="preoutcome_plan_summary"),
        record(ROOT / "artifacts/public_broad_proxy_v1/data/frozen_proxy_universe.csv", role="frozen_current_survivor_universe"),
        record(ROOT / "artifacts/sec_periodic_filing_inventory_v1/raw_manifest.json", role="sec_raw_source_manifest"),
        record(ROOT / "artifacts/public_broad_proxy_v1/data/daily_prices.parquet", role="frozen_retrospective_daily_prices"),
        record(ROOT / "artifacts/reference_calendars/xnys_v1/xnys_regular_sessions_2014_2028.csv", role="frozen_xnys_calendar"),
        record(ROOT / "artifacts/reference_calendars/xnys_v1/calendar_receipt.json", role="xnys_calendar_receipt"),
        record(ROOT / "src/alpha_factor_engine/data/sec_periodic_filings.py", role="sec_header_parser_dependency"),
        record(ROOT / "src/alpha_factor_engine/research/sec_filing_change_plan.py", role="mapping_dependency"),
        record(ROOT / "src/alpha_factor_engine/research/sec_filing_change_stage_a.py", role="inference_and_calendar_dependency"),
        record(ROOT / "src/alpha_factor_engine/research/sec_late_filing_stage_a.py", role="late_filing_plan_core"),
        record(ROOT / "src/alpha_factor_engine/research/sec_late_filing_outcomes.py", role="late_filing_outcome_core"),
        record(ROOT / "scripts/build_sec_late_filing_stage_a_plan.py", role="plan_builder"),
        record(ROOT / "scripts/run_sec_late_filing_stage_a.py", role="outcome_runner"),
        record(Path(__file__), role="design_receipt_builder"),
        record(ROOT / "pyproject.toml", role="package_contract"),
    ]
    if len({item["path"] for item in records}) != len(records):
        raise ValueError("design receipt contains duplicate paths")
    payload = {
        "schema_version": "sec_late_filing_stage_a_design_receipt_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "experiment": "sec_late_filing_stage_a_v1",
        "status": "FROZEN_PRE_OUTCOME",
        "historical_alpha_verdict": "INVALID",
        "model_alpha_verdict": "NOT VALIDATED",
        "deployment_verdict": "NOT AUTHORIZED",
        "primary_trial_id": "SEC_NT_COMBINED_CONFIRMATION_2022_2024_NET30",
        "development_years_et": [2018, 2019, 2020, 2021],
        "confirmation_years_et": [2022, 2023, 2024],
        "sealed_signal_only_years_et": [2025],
        "current_disclosed_trial_floor": 131,
        "eligible_events": int(summary["eligible_events"]),
        "confirmation_events_before_schedule": 95,
        "sealed_2025_events": 15,
        "outcome_values_read": False,
        "outcome_join_performed": False,
        "non_overwrite_policy": True,
        "research_priors": [
            {
                "claim": "Late expected reports are associated with adverse news and approximately one percent pre-cost abnormal return over 20 days in an early sample.",
                "source": "Penman 1984",
                "url": "https://business.columbia.edu/faculty/research/abnormal-returns-investment-strategies-based-timing-earnings-reports"
            },
            {
                "claim": "Missing management's expected earnings report date conveys adverse earnings information.",
                "source": "Bagnoli, Kross, and Watts 2002",
                "url": "https://doi.org/10.1111/1475-679X.t01-1-00054"
            },
            {
                "claim": "Form NT filings have negative momentum-adjusted returns around the notification date in the authors' sample.",
                "source": "Bartov and Konchitchki 2017",
                "url": "https://faculty.haas.berkeley.edu/yaniv/files/Papers_Publications/SEC-Filings-Regulatory-Deadlines-Capital-Market-Consequences_Bartov-Konchitchki_AH.pdf"
            },
            {
                "claim": "SEC periodic-report deadlines differ by filer status and form, with limited automatic extensions following Form 12b-25.",
                "source": "SEC Financial Reporting Manual",
                "url": "https://www.sec.gov/about/divisions-offices/division-corporation-finance/financial-reporting-manual/frm-topic-1"
            }
        ],
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
