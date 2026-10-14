#!/usr/bin/env python3
"""Prepare, import, verify, or plan offline SEC Form 4 source acquisition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alpha_factor_engine.data.sec_insider_offline import (
    build_bulk_acquisition_plan,
    build_bulk_acquisition_plan_from_catalog,
    import_verified_offline_bundle,
    prepare_bulk_manual_import,
    verify_offline_import_receipt,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Offline-only SEC source manager; performs no network request and "
            "never creates an alpha-validity claim"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare-bulk", help="freeze one manually downloaded quarterly ZIP"
    )
    prepare.add_argument("--source-zip", required=True, type=Path)
    prepare.add_argument("--quarter", required=True)
    prepare.add_argument("--source-url", required=True)
    prepare.add_argument("--bundle-dir", required=True, type=Path)
    prepare.add_argument("--snapshot-id", required=True)

    import_parser = subparsers.add_parser(
        "import", help="verify an offline manifest and populate the cache"
    )
    import_parser.add_argument("--manifest", required=True, type=Path)
    import_parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "data/cache/sec/insider",
    )
    import_parser.add_argument("--receipt", required=True, type=Path)

    verify = subparsers.add_parser(
        "verify", help="re-hash the input bundle and imported cache"
    )
    verify.add_argument("--receipt", required=True, type=Path)

    plan = subparsers.add_parser(
        "plan", help="build a complete plan from a saved official landing page"
    )
    plan.add_argument("--landing-page-html", required=True, type=Path)
    plan.add_argument("--start-quarter", default="2020q1")
    plan.add_argument("--end-quarter", default="2025q4")
    plan.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "data/cache/sec/insider",
    )
    plan.add_argument("--output", required=True, type=Path)

    catalog_plan = subparsers.add_parser(
        "plan-catalog", help="build a plan from a frozen SEC quarter catalog"
    )
    catalog_plan.add_argument("--catalog", required=True, type=Path)
    catalog_plan.add_argument("--start-quarter", default="2020q1")
    catalog_plan.add_argument("--end-quarter", default="2025q4")
    catalog_plan.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "data/cache/sec/insider",
    )
    catalog_plan.add_argument("--output", required=True, type=Path)

    args = parser.parse_args()
    if args.command == "prepare-bulk":
        manifest = prepare_bulk_manual_import(
            args.source_zip,
            quarter=args.quarter,
            source_url=args.source_url,
            bundle_dir=args.bundle_dir,
            snapshot_id=args.snapshot_id,
        )
        print(manifest)
    elif args.command == "import":
        receipt = import_verified_offline_bundle(
            args.manifest,
            cache_dir=args.cache_dir,
            receipt_path=args.receipt,
        )
        print(receipt)
    elif args.command == "verify":
        print(json.dumps({"verified": verify_offline_import_receipt(args.receipt)}))
    elif args.command == "plan":
        html = args.landing_page_html.read_text(encoding="utf-8")
        payload = build_bulk_acquisition_plan(
            html,
            start_quarter=args.start_quarter,
            end_quarter=args.end_quarter,
            cache_dir=args.cache_dir,
        )
        _write_json(args.output.resolve(), payload)
        print(args.output.resolve())
    elif args.command == "plan-catalog":
        catalog = pd.read_csv(args.catalog, dtype={"quarter": "string"})
        payload = build_bulk_acquisition_plan_from_catalog(
            catalog,
            start_quarter=args.start_quarter,
            end_quarter=args.end_quarter,
            cache_dir=args.cache_dir,
        )
        _write_json(args.output.resolve(), payload)
        print(args.output.resolve())
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
