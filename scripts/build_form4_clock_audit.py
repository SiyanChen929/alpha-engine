#!/usr/bin/env python3
"""Build a reproducible evidence pack for the SEC Form 4 clock audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def build(output_dir: Path) -> Path:
    sample_path = output_dir / "acceptance_clock_sample.csv"
    pac_path = output_dir / "pac_reindex_cases.csv"
    sample = pd.read_csv(
        sample_path,
        dtype={"accession": "string", "issuer_cik": "string", "form": "string"},
    )
    pac = pd.read_csv(pac_path, dtype={"accession": "string"})
    sample["filing_date"] = pd.to_datetime(sample["filing_date"], errors="raise")
    sample["json_utc"] = pd.to_datetime(sample["json_utc"], utc=True, errors="raise")
    sample["raw_utc"] = pd.to_datetime(sample["raw_utc"], utc=True, errors="raise")
    recomputed_delta = (sample["json_utc"] - sample["raw_utc"]).dt.total_seconds()
    if not recomputed_delta.eq(sample["delta_seconds"]).all():
        raise RuntimeError("clock-sample delta reconstruction failed")
    if len(sample) != 24 or sample["accession"].nunique() != 24:
        raise RuntimeError("clock audit must contain 24 unique deterministic samples")
    if sample["prefix_ne_issuer"].sum() != 20:
        raise RuntimeError("unexpected accession-prefix/issuer mismatch count")
    if sample["delta_seconds"].ne(0).sum() != 2:
        raise RuntimeError("unexpected JSON/raw conflict count")
    expected_form_counts = {"4": 18, "4/A": 6}
    if sample["form"].value_counts().to_dict() != expected_form_counts:
        raise RuntimeError("unexpected Form 4 / 4-A sample counts")
    month_counts = sample["filing_date"].dt.to_period("M").value_counts().to_dict()
    if sorted(month_counts.values()) != [8, 8, 8]:
        raise RuntimeError("sample must contain eight observations per Q4 month")
    if len(pac) != 4 or pac["accession"].nunique() != 4:
        raise RuntimeError("PAC/re-index evidence must contain four unique accessions")

    conflicts = sample.loc[sample["delta_seconds"].ne(0)].copy()
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "research_status": "NOT VALIDATED",
        "sample_selection": (
            "2025Q4 Form 4/4-A; within each month sort by SHA256(accession), "
            "take six Form 4 and two Form 4-A"
        ),
        "sample_rows": int(len(sample)),
        "form4_rows": int(sample["form"].eq("4").sum()),
        "form4a_rows": int(sample["form"].eq("4/A").sum()),
        "issuer_submissions_hits": 24,
        "raw_submission_http_200": 24,
        "json_raw_exact_matches": int(sample["delta_seconds"].eq(0).sum()),
        "json_raw_conflicts": int(sample["delta_seconds"].ne(0).sum()),
        "accession_prefix_not_issuer": int(sample["prefix_ne_issuer"].sum()),
        "pac_reindex_cases": int(len(pac)),
        "raw_payloads_retained_in_this_pack": False,
        "full_candidate_clock_coverage": False,
        "cross_quarter_daily_first_coverage": False,
        "alpha_claim_permitted": False,
        "strict_clock_rule": (
            "information_at=max(raw_sgml_acceptance_at, "
            "earliest_cross_quarter_daily_master_last_modified_at)"
        ),
    }
    _write_json(output_dir / "audit_summary.json", summary)

    gates = pd.DataFrame(
        [
            ("deterministic_sample_24", True, "24/24 unique accessions"),
            ("sample_submissions_coverage", True, "24/24 observed"),
            ("sample_raw_filing_coverage", True, "24/24 observed HTTP 200"),
            ("sample_json_raw_clock_exact", False, "22/24; two conflicts"),
            ("full_1434_candidate_raw_clock_coverage", False, "not run"),
            ("cross_quarter_earliest_daily_evidence", False, "not run for all candidates"),
            ("effective_dated_security_mapping", False, "not available"),
            ("event_return_test", False, "not authorized before clock/data gates"),
        ],
        columns=["gate", "passed", "evidence"],
    )
    gates.to_csv(output_dir / "validation_gates.csv", index=False)

    source_rows = [
        (
            "SEC insider quarterly dataset",
            "https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets",
            "official source and as-filed/PAC limitations",
        ),
        (
            "SEC EDGAR API documentation",
            "https://www.sec.gov/search-filings/edgar-application-programming-interfaces",
            "submissions schema, update timing, bulk archive",
        ),
        (
            "SEC accessing EDGAR",
            "https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data",
            "daily/full index revision policy, raw SGML, fair access",
        ),
        (
            "SEC Webmaster FAQ",
            "https://www.sec.gov/about/webmaster-frequently-asked-questions",
            "filing availability latency is typical rather than guaranteed",
        ),
    ]
    pd.DataFrame(source_rows, columns=["source", "url", "supports"]).to_csv(
        output_dir / "claim_source_ledger.csv", index=False
    )

    conflict_lines = "\n".join(
        f"- `{row.accession}`: JSON - raw = {int(row.delta_seconds):,} seconds"
        for row in conflicts.itertuples(index=False)
    )
    report = f"""# SEC Form 4 historical clock audit v1

## Verdict

**NOT VALIDATED.** The parser and clock contracts are now fail-closed, but the
full 1,434-event sample still lacks retained raw filings and cross-quarter
earliest-daily evidence. No event-return or alpha claim is authorized.

## What changed

`acceptanceDateTime + 5 minutes` is no longer the historical primary clock.
The strict historical rule is:

```text
accepted_at_raw = raw SGML acceptance wall-clock in America/New_York -> UTC
first_public_evidence_at = Last-Modified of earliest archived daily master containing accession
information_at = max(accepted_at_raw, first_public_evidence_at)
entry = first regular-session open strictly after information_at
```

The daily search must cross quarter boundaries. Four accessions seen again on
2025-12-09 were already present on 2025-09-23; the December rows are PAC/re-index
observations and cannot create new events.

## Deterministic 24-filing spot check

- 24/24 accessions appeared exactly once in issuer submissions `recent`.
- 24/24 issuer-path complete submissions returned HTTP 200 during the live audit.
- 20/24 accession prefixes differed from issuer CIK, so archive paths cannot use
  the accession prefix as issuer identity.
- 22/24 JSON timestamps agreed with raw SGML; two did not:

{conflict_lines}

The mismatches are 4 and 5 hours, consistent with an apparent timezone-encoding
problem in the JSON field for those records. Raw SGML remains required. This
pack retains the observed table, not the 24 raw payloads, so it is an audit
finding rather than a complete reproducible source snapshot.

## Remaining blockers

1. Fetch and hash raw SGML for every research accession.
2. Search archived daily indexes across adjacent quarters and freeze the true
   first occurrence plus HTTP `Last-Modified`, ETag and bytes hash.
3. Build effective-dated issuer/security/share-class mapping and delisting outcomes.
4. Adjudicate joint filings and duplicate economic transactions before measuring
   purchase size or insider clusters.
5. Only then run 5/20/60-session returns, same-industry controls, placebo,
   200 conditional nulls, costs, capacity and multiple-testing correction.

## Evidence limits

The quarterly DERA archive is a current flattened view and omits some filing
metadata. SEC daily indexes preserve historical appearances; current full and
quarterly indexes may reflect later post-acceptance corrections. The status
stays `NOT VALIDATED` until every fatal clock and security-lineage gate passes.
"""
    (output_dir / "report-source.md").write_text(report, encoding="utf-8")

    run_manifest = {
        "generated_at": summary["generated_at"],
        "research_status": "NOT VALIDATED",
        "sample_selection": summary["sample_selection"],
        "inputs": [
            {"path": sample_path.resolve().as_posix(), "sha256": _sha256(sample_path)},
            {"path": pac_path.resolve().as_posix(), "sha256": _sha256(pac_path)},
        ],
        "limitations": [
            "spot-check raw payloads not retained in artifact",
            "full candidate raw clock coverage absent",
            "cross-quarter daily-first evidence absent",
            "historical permanent security mapping absent",
        ],
    }
    _write_json(output_dir / "run_manifest.json", run_manifest)

    evidence_paths = [
        sample_path,
        pac_path,
        output_dir / "audit_summary.json",
        output_dir / "validation_gates.csv",
        output_dir / "claim_source_ledger.csv",
        output_dir / "report-source.md",
        output_dir / "run_manifest.json",
        ROOT / "configs/form4_event_alpha_v1.yml",
        ROOT / "src/alpha_factor_engine/data/sec_insider.py",
        ROOT / "src/alpha_factor_engine/research/form4.py",
        Path(__file__).resolve(),
    ]
    manifest = {
        "generated_at": summary["generated_at"],
        "research_status": "NOT VALIDATED",
        "records": [
            {
                "path": path.resolve().as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in evidence_paths
        ],
    }
    _write_json(output_dir / "artifact_manifest.json", manifest)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/form4_event_alpha_v1/clock_audit_v1",
    )
    args = parser.parse_args()
    print(build(args.output_dir.resolve()))


if __name__ == "__main__":
    main()
