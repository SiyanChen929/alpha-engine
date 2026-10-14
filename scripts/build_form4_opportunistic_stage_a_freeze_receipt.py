#!/usr/bin/env python3
"""Freeze Form 4 Stage A inputs after acquisition and before outcomes."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alpha_factor_engine.research.form4_opportunistic_stage_a import (
    CANONICAL_FREEZE_RECEIPT,
    CANONICAL_RUNTIME_CONFIG,
    build_freeze_receipt,
)


CORE = SRC / "alpha_factor_engine/research/form4_opportunistic_stage_a.py"
OUTCOME_RUNNER = ROOT / "scripts/run_form4_opportunistic_stage_a.py"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Hash the exact config, core, runner, candidates, prices, and price "
            "manifest before any Stage A outcome join"
        )
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / CANONICAL_RUNTIME_CONFIG
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / CANONICAL_FREEZE_RECEIPT
    )
    args = parser.parse_args()
    receipt = build_freeze_receipt(
        config_path=args.config,
        core_code_path=CORE,
        outcome_runner_path=OUTCOME_RUNNER,
        receipt_path=args.output,
        repository_root=ROOT,
    )
    print(receipt)


if __name__ == "__main__":
    main()
