#!/usr/bin/env python3
"""Build a non-overwriting, header-only SEC late-filing event plan."""

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

from alpha_factor_engine.research.sec_late_filing_stage_a import (
    build_late_filing_candidate_plan,
    load_late_filing_inventory,
)


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


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def verify_raw_manifest(path: Path) -> set[Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("SEC raw manifest has no records")
    paths: set[Path] = set()
    for item in records:
        source = Path(str(item.get("path", ""))).resolve()
        source.relative_to(ROOT.resolve())
        if source in paths:
            raise ValueError(f"duplicate raw SEC source: {source}")
        if item.get("role") != "raw_input" or not source.is_file():
            raise ValueError(f"invalid raw SEC source: {source}")
        if source.stat().st_size != int(item["bytes"]):
            raise ValueError(f"raw SEC source byte mismatch: {source}")
        if sha256_file(source) != item["sha256"]:
            raise ValueError(f"raw SEC source hash mismatch: {source}")
        paths.add(source)
    return paths


def run(args: argparse.Namespace) -> Path:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"non-overwriting output already exists: {output}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != "sec_late_filing_stage_a_protocol_v1":
        raise ValueError("unsupported SEC late-filing protocol")
    sources = config["source_boundaries"]
    expected_paths = {
        "universe": ROOT / sources["universe"],
        "raw_manifest": ROOT / sources["sec_raw_manifest"],
        "sec_cache": ROOT / sources["sec_cache"],
    }
    if args.universe.resolve() != expected_paths["universe"].resolve():
        raise ValueError("universe argument differs from frozen protocol")
    if args.raw_manifest.resolve() != expected_paths["raw_manifest"].resolve():
        raise ValueError("raw manifest argument differs from frozen protocol")
    if args.sec_cache.resolve() != expected_paths["sec_cache"].resolve():
        raise ValueError("SEC cache argument differs from frozen protocol")

    raw_paths = verify_raw_manifest(args.raw_manifest)
    universe = pd.read_csv(args.universe)
    if len(universe) != int(sources["expected_universe_names"]):
        raise ValueError("frozen universe count differs from protocol")
    inventory = load_late_filing_inventory(
        args.sec_cache,
        issuer_ciks=pd.to_numeric(universe["cik"], errors="raise").astype(int).tolist(),
        start_date=config["event_definition"]["minimum_filing_date"],
        end_date=config["event_definition"]["maximum_filing_date"],
        allowed_source_paths=raw_paths,
    )
    candidates, exclusions, mapping_misses = build_late_filing_candidate_plan(
        inventory,
        universe,
        lag_bounds=config["event_definition"]["form_report_lag_bounds_days"],
    )
    if candidates.empty:
        raise ValueError("late-filing plan has no eligible candidates")
    if candidates["phase"].eq("sealed_2025").sum() == 0:
        raise ValueError("late-filing plan has no sealed 2025 candidates")
    if candidates["outcome_join_performed"].any() or candidates["outcome_join_permitted"].any():
        raise ValueError("pre-outcome plan unexpectedly permits outcome access")

    output.mkdir(parents=True)
    inventory_path = output / "late_filing_header_inventory.parquet"
    candidate_path = output / "candidate_events.parquet"
    exclusion_path = output / "event_exclusions.csv"
    mapping_path = output / "mapping_misses.csv"
    inventory.to_parquet(inventory_path, index=False)
    candidates.to_parquet(candidate_path, index=False)
    exclusions.to_csv(exclusion_path, index=False)
    mapping_misses.to_csv(mapping_path, index=False)

    phase_rows = []
    for phase in ("development", "confirmation", "sealed_2025"):
        part = candidates.loc[candidates["phase"].eq(phase)]
        phase_rows.append(
            {
                "phase": phase,
                "events": int(len(part)),
                "issuers": int(part["issuer_cik"].nunique()),
                "tickers": int(part["ticker"].nunique()),
                "forms": {
                    key: int(value)
                    for key, value in part["form"].value_counts().sort_index().items()
                },
                "outcome_join_permitted": False,
            }
        )
    summary = {
        "schema_version": "sec_late_filing_stage_a_plan_summary_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "FROZEN_HEADER_ONLY_PRE_OUTCOME_PLAN",
        "historical_alpha_verdict": "INVALID",
        "model_alpha_verdict": "NOT VALIDATED",
        "deployment_verdict": "NOT AUTHORIZED",
        "universe_names": int(len(universe)),
        "raw_sec_source_records_verified": int(len(raw_paths)),
        "late_filing_headers_2018_2025": int(len(inventory)),
        "eligible_events": int(len(candidates)),
        "excluded_events": int(len(exclusions)),
        "mapped_issuers": int(candidates["issuer_cik"].nunique()),
        "mapping_misses": int(len(mapping_misses)),
        "phases": phase_rows,
        "outcome_inputs_read": False,
        "outcome_join_performed": False,
        "sealed_2025_event_count": int(candidates["phase"].eq("sealed_2025").sum()),
        "next_gate": "Freeze runner and design receipt before any price or outcome read",
        "limitations": config["hard_claim_limits"],
    }
    summary_path = output / "plan_summary.json"
    write_json(summary_path, summary)
    output_records = [
        record(inventory_path, role="late_filing_header_inventory"),
        record(candidate_path, role="frozen_candidate_events"),
        record(exclusion_path, role="event_exclusions"),
        record(mapping_path, role="mapping_misses"),
        record(summary_path, role="plan_summary"),
    ]
    manifest = {
        "schema_version": "sec_late_filing_stage_a_plan_manifest_v1",
        "created_at": summary["created_at"],
        "inputs": [
            record(args.config, role="human_protocol"),
            record(args.universe, role="frozen_current_survivor_universe"),
            record(args.raw_manifest, role="sec_raw_source_manifest"),
            record(Path(__file__), role="plan_builder"),
            record(
                SRC / "alpha_factor_engine/research/sec_late_filing_stage_a.py",
                role="plan_core",
            ),
            record(
                SRC / "alpha_factor_engine/data/sec_periodic_filings.py",
                role="sec_header_parser_dependency",
            ),
        ],
        "outputs": output_records,
        "outcome_inputs_read": False,
        "outcome_join_performed": False,
    }
    manifest_path = output / "plan_manifest.json"
    write_json(manifest_path, manifest)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/sec_late_filing_stage_a_v1.json",
    )
    parser.add_argument(
        "--universe",
        type=Path,
        default=ROOT / "artifacts/public_broad_proxy_v1/data/frozen_proxy_universe.csv",
    )
    parser.add_argument(
        "--raw-manifest",
        type=Path,
        default=ROOT / "artifacts/sec_periodic_filing_inventory_v1/raw_manifest.json",
    )
    parser.add_argument("--sec-cache", type=Path, default=ROOT / "data/cache/sec")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/sec_late_filing_stage_a_v1/preoutcome_plan_v1",
    )
    args = parser.parse_args()
    print(run(args))


if __name__ == "__main__":
    main()
