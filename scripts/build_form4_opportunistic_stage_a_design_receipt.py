#!/usr/bin/env python3
"""Create the non-overwriting Form 4 Stage A design freeze before prices."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alpha_factor_engine.research.form4_opportunistic_stage_a import (
    build_design_receipt,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the Stage A research design before Yahoo prices are acquired"
    )
    parser.add_argument(
        "--human-protocol",
        type=Path,
        default=ROOT / "configs/form4_opportunistic_stage_a_v1.json",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=(
            ROOT
            / "artifacts/form4_event_alpha_v1/"
            "candidate_preregistration_2020q1_2025q4_v2/"
            "candidate_accessions.parquet"
        ),
    )
    parser.add_argument(
        "--candidate-upstream-manifest",
        type=Path,
        default=(
            ROOT
            / "artifacts/form4_event_alpha_v1/"
            "candidate_preregistration_2020q1_2025q4_v2/"
            "artifact_manifest.json"
        ),
    )
    parser.add_argument(
        "--yahoo-adapter",
        type=Path,
        default=SRC / "alpha_factor_engine/data/yahoo.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/form4_opportunistic_stage_a_v1/design_receipt.json",
    )
    args = parser.parse_args()
    receipt = build_design_receipt(
        human_protocol_path=args.human_protocol,
        core_code_path=SRC
        / "alpha_factor_engine/research/form4_opportunistic_stage_a.py",
        outcome_runner_path=ROOT / "scripts/run_form4_opportunistic_stage_a.py",
        acquisition_script_path=ROOT
        / "scripts/acquire_form4_opportunistic_stage_a_prices.py",
        candidate_path=args.candidates,
        candidate_upstream_manifest_path=args.candidate_upstream_manifest,
        yahoo_adapter_path=args.yahoo_adapter,
        receipt_path=args.output,
        repository_root=ROOT,
    )
    print(receipt)


if __name__ == "__main__":
    main()
