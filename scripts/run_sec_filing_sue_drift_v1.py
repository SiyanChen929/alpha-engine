#!/usr/bin/env python3
"""Run the frozen SEC-filing seasonal earnings-change drift diagnostic."""

from __future__ import annotations

import argparse
from pathlib import Path

from alpha_factor_engine.research.sec_filing_sue_drift import run_diagnostic


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/sec_filing_sue_drift_v1_preregistration.json",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    outputs = run_diagnostic(
        config_path=args.config,
        repository_root=ROOT,
        output_dir=args.output_dir,
    )
    print(outputs["report"])


if __name__ == "__main__":
    main()
