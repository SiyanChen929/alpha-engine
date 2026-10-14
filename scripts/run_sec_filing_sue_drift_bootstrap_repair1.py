#!/usr/bin/env python3
"""Run the post-outcome bootstrap correction without touching attempt 1."""

from __future__ import annotations

from pathlib import Path

from alpha_factor_engine.research.sec_filing_sue_drift_bootstrap_repair import run_repair


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    outputs = run_repair(
        ROOT / "configs/sec_filing_sue_drift_v1_bootstrap_repair1.json",
        ROOT,
    )
    print(outputs["report"])
