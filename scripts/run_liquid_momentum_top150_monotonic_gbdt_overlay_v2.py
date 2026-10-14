#!/usr/bin/env python3
"""Run the frozen v2 strict held-out monotonic GBDT challenger."""

from __future__ import annotations

import argparse
from pathlib import Path

from run_liquid_momentum_top150_monotonic_gbdt_overlay import run


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(
            root / "configs/liquid_momentum_top150_monotonic_gbdt_overlay_v2.yml"
        ),
    )
    arguments = parser.parse_args()
    report = run(arguments.config, invocation_path=Path(__file__).resolve())
    print(report)


if __name__ == "__main__":
    main()
