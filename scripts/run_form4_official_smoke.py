#!/usr/bin/env python3
"""Create a data-only audit artifact from one official SEC Form 3/4/5 ZIP.

This intentionally stops before event-time backtesting.  The SEC bulk files do
not contain the intraday acceptance timestamp required by the research clock.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alpha_factor_engine.data.sec_insider import parse_insider_zip
from alpha_factor_engine.research.form4 import Form4PurchaseFilter, filter_code_p_purchases


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(zip_path: Path, config_path: Path, quarter: str) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    parsed = parse_insider_zip(zip_path.read_bytes(), quarter=quarter)
    signal = config["signal"]
    purchase_filter = Form4PurchaseFilter(
        minimum_purchase_usd=float(signal["minimum_purchase_usd"]),
        qualifying_role_pattern="|".join(signal["qualifying_owner_roles"]),
        include_security_pattern=str(signal["include_security_regex"]),
        exclude_security_pattern=str(signal["exclude_security_regex"]),
        require_explicit_non_10b5_1=bool(signal["require_rule_10b5_1_false"]),
    )
    events = filter_code_p_purchases(
        parsed.submissions,
        parsed.reporting_owners,
        parsed.nonderivative_transactions,
        config=purchase_filter,
    )

    output_dir = ROOT / config["outputs"]["directory"] / f"official_{parsed.quarter}_smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "code_p_purchase_candidates.parquet"
    events.to_parquet(events_path, index=False)

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "research_status": "DATA_ONLY_NOT_VALIDATED",
        "quarter": parsed.quarter,
        "archive_path": zip_path.resolve().as_posix(),
        "archive_sha256": parsed.archive_sha256,
        "submissions": int(len(parsed.submissions)),
        "reporting_owners": int(len(parsed.reporting_owners)),
        "nonderivative_transactions": int(len(parsed.nonderivative_transactions)),
        "footnotes": int(len(parsed.footnotes)),
        "code_p_purchase_candidates": int(len(events)),
        "candidate_issuers": int(events["issuer_cik"].nunique()),
        "candidate_tickers": int(events["ticker"].nunique()),
        "minimum_purchase_usd": float(purchase_filter.minimum_purchase_usd),
        "acceptance_timestamp_enriched": False,
        "raw_filing_header_reconciled": False,
        "historical_security_master_joined": False,
        "event_return_backtest_run": False,
        "alpha_claim_permitted": False,
    }
    summary_path = output_dir / "source_audit.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = f"""# SEC Form 4 official {parsed.quarter} smoke test

## Verdict

**DATA_ONLY / NOT VALIDATED.** The official bulk archive parses and the
predeclared conservative code-P purchase filter produces a usable candidate
table. No alpha or event-return claim is permitted.

## Observed coverage

- Submissions: {summary['submissions']:,}
- Reporting owners: {summary['reporting_owners']:,}
- Non-derivative transactions: {summary['nonderivative_transactions']:,}
- Footnotes: {summary['footnotes']:,}
- Filtered code-P purchase candidates: {summary['code_p_purchase_candidates']:,}
- Candidate issuers / tickers: {summary['candidate_issuers']:,} / {summary['candidate_tickers']:,}

The filter requires original Form 4, transaction code P, acquired code A,
positive shares and price, purchase value at least ${summary['minimum_purchase_usd']:,.0f},
officer/director role, common/ordinary security, explicit non-swap, and explicit
non-10b5-1 status. SEC code P can still include purchases from another person,
so the candidate is not labelled as a pure open-market purchase.

## Missing before research eligibility

1. Enrich every accession with SEC `acceptanceDateTime` and reconcile it to the
   raw filing `<ACCEPTANCE-DATETIME>` header.
2. Map issuer/ticker to a historical permanent security identifier and include
   delisting outcomes.
3. Enter strictly after public availability, join confirmed prices, create PIT
   matched controls, and calculate 5/20/60-session abnormal returns.
4. Apply costs, capacity, clustered inference, pre-event placebo, 200 conditional
   random-event nulls, and multiple-testing correction.

The quarterly ZIP hash is `{summary['archive_sha256']}`.
"""
    report_path = output_dir / "report-source.md"
    report_path.write_text(report, encoding="utf-8")

    evidence = [
        ("input", zip_path),
        ("config", config_path),
        ("code", Path(__file__).resolve()),
        ("code", ROOT / "src/alpha_factor_engine/data/sec_insider.py"),
        ("code", ROOT / "src/alpha_factor_engine/research/form4.py"),
        ("output", events_path),
        ("output", summary_path),
        ("output", report_path),
    ]
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "research_status": "DATA_ONLY_NOT_VALIDATED",
        "records": [
            {
                "role": role,
                "path": path.resolve().as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for role, path in evidence
        ],
    }
    (output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True, type=Path)
    parser.add_argument("--quarter", default="2025q4")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/form4_event_alpha_v1.yml"
    )
    args = parser.parse_args()
    print(run(args.zip.resolve(), args.config.resolve(), args.quarter))


if __name__ == "__main__":
    main()
