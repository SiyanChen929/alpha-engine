#!/usr/bin/env python3
"""Build a header-only, pre-outcome SEC filing-change acquisition plan."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alpha_factor_engine.data.sec_periodic_filings import load_periodic_filing_inventory
from alpha_factor_engine.research.sec_filing_change_plan import (
    acquisition_documents,
    assign_pair_download_phases,
    attach_current_proxy_mapping,
    build_rough_prior_year_pairs,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path, *, role: str) -> dict[str, object]:
    return {
        "role": role,
        "path": path.resolve().relative_to(ROOT.resolve()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _verify_raw_header_manifest(path: Path) -> set[Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("SEC raw header manifest has no records")
    seen: set[Path] = set()
    for record in records:
        source = Path(str(record.get("path", ""))).resolve()
        source.relative_to(ROOT.resolve())
        if source in seen:
            raise ValueError(f"duplicate SEC raw-manifest path: {source}")
        seen.add(source)
        if record.get("role") != "raw_input" or not source.is_file():
            raise ValueError(f"invalid SEC raw-manifest record: {source}")
        if source.stat().st_size != int(record["bytes"]) or sha256_file(source) != record["sha256"]:
            raise ValueError(f"SEC raw-manifest hash mismatch: {source}")
    return seen


def run(args: argparse.Namespace) -> Path:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"non-overwriting plan directory already exists: {output}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != "sec_filing_change_stage_a_protocol_v1":
        raise ValueError("unsupported filing-change protocol")
    frozen_sec_sources = _verify_raw_header_manifest(args.header_manifest)
    universe = pd.read_csv(args.universe)
    if len(universe) != int(config["sources"]["universe"]["expected_names"]):
        raise ValueError("frozen proxy universe count differs from protocol")
    ciks = pd.to_numeric(universe["cik"], errors="raise").astype(int).tolist()
    inventory = load_periodic_filing_inventory(
        args.sec_cache,
        issuer_ciks=ciks,
        start_date=config["filing_eligibility"]["minimum_header_date"],
        end_date="2025-12-31",
        allowed_source_paths=frozen_sec_sources,
    )
    mapped, mapping_misses = attach_current_proxy_mapping(inventory, universe)
    pairs, no_pair = build_rough_prior_year_pairs(
        mapped,
        signal_start=config["filing_eligibility"]["first_signal_date"],
        signal_end="2025-12-31",
        minimum_gap_days=int(config["filing_eligibility"]["minimum_report_date_gap_days"]),
        maximum_gap_days=int(config["filing_eligibility"]["maximum_report_date_gap_days"]),
    )
    documents = acquisition_documents(mapped, pairs)

    pairs = assign_pair_download_phases(
        pairs,
        development_release_month_end=config["cohorts"]["development"][
            "release_month_lte"
        ],
        confirmation_release_month_end=config["cohorts"]["confirmation"][
            "release_month_lte"
        ],
    )
    pair_phase = (
        pairs[["current_accession", "download_phase"]]
        .drop_duplicates()
        .set_index("current_accession")["download_phase"]
        .to_dict()
    )
    phase_priority = {
        "00_top25_engineering_smoke": 0,
        "10_full_development_and_confirmation": 1,
        "20_sealed_2025_signal_only": 2,
    }
    accession_phases: dict[str, set[str]] = {}
    for pair in pairs.itertuples(index=False):
        phase = pair_phase[str(pair.current_accession)]
        accession_phases.setdefault(str(pair.current_accession), set()).add(phase)
        accession_phases.setdefault(str(pair.prior_accession), set()).add(phase)
    documents["download_phase"] = documents["accession"].map(
        lambda accession: min(
            accession_phases[str(accession)], key=lambda value: phase_priority[value]
        )
    )
    documents["outcome_join_permitted_during_acquisition"] = False

    output.mkdir(parents=True)
    header_path = output / "mapped_header_inventory.parquet"
    pair_path = output / "rough_pairing_ledger.parquet"
    document_path = output / "acquisition_documents.parquet"
    no_pair_path = output / "header_pair_exclusions.csv"
    mapping_miss_path = output / "mapping_misses.csv"
    mapped.to_parquet(header_path, index=False)
    pairs.to_parquet(pair_path, index=False)
    documents.to_parquet(document_path, index=False)
    no_pair.to_csv(no_pair_path, index=False)
    mapping_misses.to_csv(mapping_miss_path, index=False)

    by_phase = []
    for phase in phase_priority:
        phase_pairs = pairs.loc[pairs["download_phase"].eq(phase)]
        phase_accessions = set(phase_pairs["current_accession"]) | set(
            phase_pairs["prior_accession"]
        )
        by_phase.append(
            {
                "phase": phase,
                "rough_pairs": int(len(phase_pairs)),
                "current_filings": int(phase_pairs["current_accession"].nunique()),
                "unique_documents": int(len(phase_accessions)),
                "issuers": int(phase_pairs["issuer_cik"].nunique()),
                "tickers": int(phase_pairs["ticker"].nunique()),
                "note": "Header-only acquisition superset; final DEI fiscal pairing may be smaller.",
            }
        )
    summary = {
        "schema_version": "sec_filing_change_acquisition_plan_summary_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "PRE_OUTCOME_ACQUISITION_PLAN_ONLY",
        "historical_alpha_verdict": "INVALID",
        "deployment_verdict": "NOT AUTHORIZED",
        "frozen_universe_names": int(len(universe)),
        "proxy_unique_cik": int(pd.to_numeric(universe["cik"], errors="raise").nunique()),
        "proxy_unique_ticker": int(universe["ticker"].astype("string").str.upper().nunique()),
        "proxy_cik_conflicts": 0,
        "proxy_ticker_conflicts": 0,
        "proxy_mapping_one_to_one": True,
        "periodic_headers_2018_2025": int(len(inventory)),
        "headers_with_current_proxy_mapping": int(len(mapped)),
        "header_mapping_misses": int(len(mapping_misses)),
        "rough_pairs_2019_2025": int(len(pairs)),
        "rough_pair_issuers": int(pairs["issuer_cik"].nunique()),
        "rough_pair_tickers": int(pairs["ticker"].nunique()),
        "unique_primary_documents": int(len(documents)),
        "unpaired_current_headers": int(len(no_pair)),
        "forms": {key: int(value) for key, value in pairs["form"].value_counts().sort_index().items()},
        "phases": by_phase,
        "outcome_join_performed": False,
        "sec_user_agent_present": bool(os.getenv("SEC_USER_AGENT")),
        "next_gate": "Real SEC_USER_AGENT plus engineering-smoke document acquisition",
        "limitations": [
            "Current-survivor 2026-08-31 ticker mapping is not point-in-time security lineage.",
            "Header gap matching is only a download superset; final pairing requires DEI fiscal identity.",
            "API acceptance timestamps are not yet reconciled to raw SGML acceptance headers.",
            "No issuer forward return was opened by this planner."
        ]
    }
    summary_path = output / "plan_summary.json"
    _write_json(summary_path, summary)
    output_records = [
        _record(header_path, role="mapped_header_inventory"),
        _record(pair_path, role="rough_pairing_ledger"),
        _record(document_path, role="acquisition_documents"),
        _record(no_pair_path, role="header_pair_exclusions"),
        _record(mapping_miss_path, role="mapping_misses"),
        _record(summary_path, role="plan_summary"),
    ]
    manifest = {
        "schema_version": "sec_filing_change_acquisition_plan_manifest_v1",
        "created_at": summary["created_at"],
        "inputs": [
            _record(args.config, role="human_protocol"),
            _record(args.universe, role="frozen_proxy_universe"),
            _record(args.header_manifest, role="sec_header_raw_manifest"),
            _record(Path(__file__), role="planner"),
            _record(SRC / "alpha_factor_engine/data/sec_periodic_filings.py", role="sec_periodic_data_code"),
            _record(SRC / "alpha_factor_engine/research/sec_filing_change_plan.py", role="pair_plan_code"),
        ],
        "outputs": output_records,
        "outcome_join_performed": False,
    }
    manifest_path = output / "plan_manifest.json"
    _write_json(manifest_path, manifest)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/sec_filing_change_stage_a_v1.json"
    )
    parser.add_argument(
        "--universe",
        type=Path,
        default=ROOT / "artifacts/public_broad_proxy_v1/data/frozen_proxy_universe.csv",
    )
    parser.add_argument("--sec-cache", type=Path, default=ROOT / "data/cache/sec")
    parser.add_argument(
        "--header-manifest",
        type=Path,
        default=ROOT / "artifacts/sec_periodic_filing_inventory_v1/raw_manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/sec_filing_change_stage_a_v1/preoutcome_plan_v3",
    )
    args = parser.parse_args()
    print(run(args))


if __name__ == "__main__":
    main()
