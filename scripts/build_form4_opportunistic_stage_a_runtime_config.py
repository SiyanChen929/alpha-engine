#!/usr/bin/env python3
"""Write the exact Form 4 Stage A runtime config after price acquisition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alpha_factor_engine.research.form4_opportunistic_stage_a import (
    CANONICAL_DESIGN_RECEIPT,
    CANONICAL_HUMAN_PROTOCOL,
    CANONICAL_PRICE_MANIFEST,
    CANONICAL_PRICE_PANEL,
    CANONICAL_RUNTIME_CONFIG,
    Form4StageAError,
    _require_canonical_path,
    stage_a_config_template,
)


def _relative(path: Path) -> str:
    target = path.resolve()
    try:
        return target.relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise Form4StageAError(f"runtime config path is outside repository: {target}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the fixed Stage A runtime config")
    parser.add_argument("--human-protocol", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--candidate-upstream-manifest", type=Path, required=True)
    parser.add_argument("--yahoo-adapter", type=Path, required=True)
    parser.add_argument("--design-receipt", type=Path, required=True)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--price-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = _require_canonical_path(
        args.output,
        root=ROOT,
        canonical_relative=CANONICAL_RUNTIME_CONFIG,
        role="runtime config",
    )
    for path, canonical, role in (
        (args.human_protocol, CANONICAL_HUMAN_PROTOCOL, "human protocol"),
        (args.design_receipt, CANONICAL_DESIGN_RECEIPT, "design receipt"),
        (args.prices, CANONICAL_PRICE_PANEL, "price panel"),
        (args.price_manifest, CANONICAL_PRICE_MANIFEST, "price manifest"),
    ):
        _require_canonical_path(
            path,
            root=ROOT,
            canonical_relative=canonical,
            role=role,
        )
    payload = stage_a_config_template(
        human_protocol=_relative(args.human_protocol),
        candidate_accessions=_relative(args.candidates),
        candidate_upstream_manifest=_relative(args.candidate_upstream_manifest),
        yahoo_adapter=_relative(args.yahoo_adapter),
        design_receipt=_relative(args.design_receipt),
        price_panel=_relative(args.prices),
        price_manifest=_relative(args.price_manifest),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8") as handle:
            handle.write(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
                + "\n"
            )
    except FileExistsError as exc:
        raise Form4StageAError(f"runtime config already exists: {output}") from exc
    print(output)


if __name__ == "__main__":
    main()
