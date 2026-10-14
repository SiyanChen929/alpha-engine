#!/usr/bin/env python3
"""Run the receipt-locked Form 4 opportunistic-purchase Stage A diagnostic."""

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
    CANONICAL_RUN_DIRECTORY,
    CANONICAL_RUNTIME_CONFIG,
    run_stage_a_diagnostic,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run only after a separate immutable freeze receipt exists. "
            "Historical alpha remains INVALID."
        )
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / CANONICAL_RUNTIME_CONFIG
    )
    parser.add_argument(
        "--freeze-receipt", type=Path, default=ROOT / CANONICAL_FREEZE_RECEIPT
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / CANONICAL_RUN_DIRECTORY
    )
    args = parser.parse_args()
    outputs = run_stage_a_diagnostic(
        config_path=args.config,
        freeze_receipt_path=args.freeze_receipt,
        outcome_runner_path=Path(__file__),
        repository_root=ROOT,
        output_dir=args.output_dir,
    )
    print(outputs["report"])


if __name__ == "__main__":
    main()
