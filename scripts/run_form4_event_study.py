#!/usr/bin/env python3
"""Run one immutable, manifest-verified Form 4 event diagnostic bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from alpha_factor_engine.research.form4_event_runner import run_form4_event_study


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    outputs = run_form4_event_study(
        args.input_manifest.resolve(), args.output_dir.resolve()
    )
    print(outputs["run_manifest"])


if __name__ == "__main__":
    main()
