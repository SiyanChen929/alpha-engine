#!/usr/bin/env python3
"""Build a hash-closed evidence pack for the SEC Form 3/4/5 bulk history.

This artifact validates acquisition and parsing only.  It deliberately does
not attach historical public timestamps, returns, or an alpha claim.
"""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
from xml.etree import ElementTree

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alpha_factor_engine.data.sec_insider_offline import (
    build_bulk_acquisition_plan_from_catalog,
    load_verified_offline_bundle,
    verify_offline_import_receipt,
)


OUTPUT = (
    ROOT
    / "artifacts/form4_event_alpha_v1/bulk_history_2020q1_2025q4_v1"
)
CATALOG = ROOT / "configs/form4_bulk_quarter_catalog_v1.csv"
CATALOG_META = ROOT / "configs/form4_bulk_quarter_catalog_v1.meta.json"
CACHE = ROOT / "data/cache/sec/insider"
LEGACY_Q4 = ROOT / "artifacts/form4_event_alpha_v1/manual_import_2025q4_v1"
HISTORY_IMPORT = ROOT / "artifacts/form4_event_alpha_v1/manual_import_history_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _receipt_path(quarter: str) -> Path:
    if quarter == "2025q4":
        return LEGACY_Q4 / "import_receipt.json"
    return HISTORY_IMPORT / quarter / "import_receipt.json"


def _test_counts(path: Path) -> dict[str, int]:
    root = ElementTree.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    if not suites:
        raise RuntimeError(f"test report has no test suites: {path}")
    return {
        key: sum(int(float(suite.attrib.get(key, "0"))) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }


def build(output_dir: Path = OUTPUT) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(UTC).isoformat()
    targeted_test_path = output_dir / "targeted-tests.xml"
    full_test_path = output_dir / "full-tests.xml"
    targeted_tests = _test_counts(targeted_test_path)
    full_tests = _test_counts(full_test_path)
    for name, counts in (
        ("targeted", targeted_tests),
        ("full", full_tests),
    ):
        if counts["tests"] <= 0 or counts["failures"] or counts["errors"]:
            raise RuntimeError(f"{name} regression suite did not pass: {counts}")
    catalog = pd.read_csv(CATALOG, dtype={"quarter": "string"})
    plan = build_bulk_acquisition_plan_from_catalog(
        catalog,
        start_quarter="2020q1",
        end_quarter="2025q4",
        cache_dir=CACHE,
    )
    expected_quarters = [
        f"{year}q{quarter}"
        for year in range(2020, 2026)
        for quarter in range(1, 5)
    ]
    records = plan.get("records")
    if not isinstance(records, list) or [row["quarter"] for row in records] != expected_quarters:
        raise RuntimeError("bulk plan is not the exact ordered 2020Q1-2025Q4 panel")
    if (
        plan.get("verified_cached_count") != 24
        or plan.get("manual_download_required_count") != 0
        or plan.get("blocking_cache_finding_count") != 0
    ):
        raise RuntimeError("bulk history cache is not complete and unambiguous")

    plan_path = output_dir / "bulk_quarter_plan.json"
    _write_json(plan_path, plan)
    inventory_rows: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []
    evidence: list[tuple[str, Path]] = [
        ("input", CATALOG),
        ("input", CATALOG_META),
        ("code", ROOT / "src/alpha_factor_engine/data/sec_insider.py"),
        ("code", ROOT / "src/alpha_factor_engine/data/sec_insider_offline.py"),
        ("code", ROOT / "scripts/manage_form4_offline_acquisition.py"),
        ("code", Path(__file__).resolve()),
        ("code", ROOT / "tests/test_sec_insider.py"),
        ("code", ROOT / "tests/test_sec_insider_offline.py"),
    ]

    for planned in records:
        quarter = str(planned["quarter"])
        receipt_path = _receipt_path(quarter)
        if not verify_offline_import_receipt(receipt_path):  # pragma: no cover
            raise RuntimeError(f"receipt did not verify: {quarter}")
        receipt = _read_json(receipt_path)
        manifest_path = Path(receipt["input_manifest_path"]).resolve()
        _, manifest_sha, imported = load_verified_offline_bundle(manifest_path)
        if len(imported) != 1 or imported[0].role != f"bulk:{quarter}":
            raise RuntimeError(f"offline manifest role mismatch: {quarter}")
        item = imported[0]
        if receipt.get("input_manifest_sha256") != manifest_sha:
            raise RuntimeError(f"receipt/manifest hash mismatch: {quarter}")
        cache_path = Path(str(planned["cache_path"])).resolve()
        receipt_rows = receipt.get("records")
        if not isinstance(receipt_rows, list) or len(receipt_rows) != 1:
            raise RuntimeError(f"receipt row mismatch: {quarter}")
        receipt_row = receipt_rows[0]
        if (
            item.sha256 != planned["sha256"]
            or item.byte_count != planned["bytes"]
            or Path(str(receipt_row["cache_path"])).resolve() != cache_path
            or item.sha256 != receipt_row["sha256"]
            or _sha256(cache_path) != item.sha256
            or cache_path.stat().st_size != item.byte_count
        ):
            raise RuntimeError(f"catalog/bundle/receipt/cache mismatch: {quarter}")
        validation = dict(item.validation)
        inventory_rows.append(
            {
                "quarter": quarter,
                "source_url": item.source_url,
                "bytes": item.byte_count,
                "sha256": item.sha256,
                "metadata_filename": validation["metadata_filename"],
                "submissions": validation["submissions"],
                "form4_submissions": validation["form4_submissions"],
                "original_form4_submissions": validation[
                    "original_form4_submissions"
                ],
                "reporting_owners": validation["reporting_owners"],
                "nonderivative_transactions": validation[
                    "nonderivative_transactions"
                ],
                "footnotes": validation["footnotes"],
                "normalized_cik_values": validation["normalized_cik_values"],
                "quarantined_optional_date_values": validation[
                    "quarantined_optional_date_values"
                ],
                "offline_manifest_sha256": manifest_sha,
                "cache_path": cache_path.as_posix(),
                "receipt_path": receipt_path.resolve().as_posix(),
                "transport_authenticity_independently_verified": False,
                "alpha_claim_permitted": False,
            }
        )
        for issue in validation["data_quality_issues"]:
            examples = issue.get("examples", [])
            quality_rows.append(
                {
                    "quarter": quarter,
                    "table": issue["table"],
                    "column": issue["column"],
                    "action": issue["action"],
                    "affected_count": int(
                        issue.get("invalid_count", issue.get("normalized_count", 0))
                    ),
                    "examples_json": json.dumps(
                        examples, sort_keys=True, allow_nan=False
                    ),
                    "raw_archive_sha256": item.sha256,
                }
            )
        evidence.extend(
            [
                ("input", receipt_path.resolve()),
                ("input", manifest_path),
                ("input", item.path),
                ("input", cache_path),
            ]
        )

    inventory = pd.DataFrame(inventory_rows).sort_values("quarter", kind="mergesort")
    if inventory["quarter"].tolist() != expected_quarters:
        raise RuntimeError("quarter inventory ordering or coverage drifted")
    if inventory["sha256"].duplicated().any():
        raise RuntimeError("distinct SEC quarters unexpectedly share archive bytes")
    inventory_path = output_dir / "quarter_inventory.csv"
    inventory.to_csv(inventory_path, index=False, lineterminator="\n")

    quality_columns = [
        "quarter",
        "table",
        "column",
        "action",
        "affected_count",
        "examples_json",
        "raw_archive_sha256",
    ]
    quality = pd.DataFrame(quality_rows, columns=quality_columns)
    if not quality.empty:
        quality = quality.sort_values(
            ["quarter", "table", "column", "action"], kind="mergesort"
        )
    quality_path = output_dir / "data_quality_issues.csv"
    quality.to_csv(quality_path, index=False, lineterminator="\n")

    total_bytes = int(inventory["bytes"].sum())
    total_submissions = int(inventory["submissions"].sum())
    total_form4 = int(inventory["form4_submissions"].sum())
    total_original_form4 = int(inventory["original_form4_submissions"].sum())
    total_transactions = int(inventory["nonderivative_transactions"].sum())
    normalized_ciks = int(inventory["normalized_cik_values"].sum())
    quarantined_dates = int(
        inventory["quarantined_optional_date_values"].sum()
    )
    metadata_counts = inventory["metadata_filename"].value_counts().to_dict()

    transport = {
        "schema_version": "sec_form4_bulk_transport_record_v1",
        "recorded_at": generated_at,
        "provider": "SEC",
        "requested_panel": "2020q1-2025q4",
        "newly_acquired_quarters": 23,
        "preexisting_verified_quarters": 1,
        "method": "serial_https_get_then_offline_fail_closed_import",
        "contact_identity_source": "local_git_user_email",
        "contact_value_retained": False,
        "minimum_inter_request_delay_seconds": 0.25,
        "concurrent_requests": 1,
        "retryable_status_policy": "bounded retries; no bypass",
        "transport_authenticity_independently_verified": False,
        "note": (
            "The artifact freezes exact bytes and official URL contracts, but "
            "does not treat an operator transport assertion as cryptographic provenance."
        ),
    }
    transport_path = output_dir / "transport_record.json"
    _write_json(transport_path, transport)

    gates = pd.DataFrame(
        [
            ("bulk_quarter_panel_exact", True, "2020Q1-2025Q4; 24/24 ordered quarters"),
            ("cache_complete_unambiguous", True, "24 verified; 0 missing; 0 ambiguous"),
            ("offline_receipts_reverified", True, "24/24 source/manifest/cache byte chains"),
            ("zip_crc_schema_keys_foreign_keys", True, "24/24 parsed fail-closed"),
            (
                "metadata_schema_versions_audited",
                True,
                json.dumps(metadata_counts, sort_keys=True),
            ),
            (
                "identifier_normalization_audited",
                True,
                f"{normalized_ciks} valid decimal CIK values left-zero-padded",
            ),
            (
                "malformed_optional_dates_quarantined",
                True,
                f"{quarantined_dates} values set to NaT without date repair",
            ),
            (
                "historical_public_clock_complete",
                False,
                "bulk ZIP filing dates are insufficient; raw SGML and earliest daily-index evidence pending",
            ),
            (
                "effective_dated_security_return_lineage_complete",
                False,
                "permanent IDs, corporate actions, delistings and event-window prices pending",
            ),
            (
                "event_alpha_inference_complete",
                False,
                "no matched-control returns, null retrains, costs or multiple-testing result run",
            ),
            (
                "regression_suites_pass",
                True,
                f"targeted={targeted_tests['tests']}; full={full_tests['tests']}",
            ),
            ("performance_claim_permitted", False, "data acquisition artifact only"),
        ],
        columns=["gate", "passed", "evidence"],
    )
    gates_path = output_dir / "validation_gates.csv"
    gates.to_csv(gates_path, index=False, lineterminator="\n")

    summary = {
        "schema_version": "sec_form4_bulk_history_evidence_v1",
        "generated_at": generated_at,
        "research_status": "NOT VALIDATED",
        "bulk_history_acquisition_complete": True,
        "alpha_claim_permitted": False,
        "performance_claim_permitted": False,
        "event_return_backtest_run": False,
        "quarter_start": "2020q1",
        "quarter_end": "2025q4",
        "quarter_count": 24,
        "verified_receipt_count": 24,
        "total_bytes": total_bytes,
        "total_submissions": total_submissions,
        "total_form4_and_amendment_submissions": total_form4,
        "total_original_form4_submissions": total_original_form4,
        "total_nonderivative_transactions": total_transactions,
        "metadata_filename_counts": metadata_counts,
        "normalized_cik_values": normalized_ciks,
        "quarantined_optional_date_values": quarantined_dates,
        "transport_authenticity_independently_verified": False,
        "targeted_regression_tests": targeted_tests,
        "full_regression_tests": full_tests,
        "remaining_blockers": [
            "raw complete-submission SGML acceptance headers",
            "earliest cross-quarter daily-index public evidence",
            "effective-dated permanent security identifiers",
            "corporate actions and delisting-complete returns",
            "matched controls, costs, conditional nulls and multiple-testing correction",
        ],
    }
    summary_path = output_dir / "acquisition_summary.json"
    _write_json(summary_path, summary)

    report = f"""# SEC Form 4 bulk history acquisition evidence

## Verdict

**Bulk acquisition: COMPLETE. Research status: NOT VALIDATED.** Exact official
quarter URLs for `2020Q1-2025Q4` now map to 24 unique, content-addressed ZIPs.
All bundle manifests, import receipts, and cache bytes re-hash to the same
quarter-specific SHA256. No event return or alpha claim is permitted.

## Verified coverage

- Quarters: `24/24`
- Exact bytes: `{total_bytes:,}`
- Submissions: `{total_submissions:,}`
- Form 4 / 4-A submissions: `{total_form4:,}`
- Original Form 4 submissions: `{total_original_form4:,}`
- Non-derivative transaction rows: `{total_transactions:,}`
- Metadata versions: `{json.dumps(metadata_counts, sort_keys=True)}`
- Regression suites: targeted `{targeted_tests['tests']}/{targeted_tests['tests']}`;
  full `{full_tests['tests']}/{full_tests['tests']}`

## Data-quality handling

The earlier parser incorrectly assumed the post-2022 metadata filename and
required every optional date in the official source to be valid. Versioned
metadata detection now accepts exactly one documented legacy/current metadata
file. `{normalized_ciks}` valid decimal CIK values were deterministically
left-zero-padded to 10 digits. `{quarantined_dates}` impossible optional dates
were set to `NaT`, recorded with accession/transaction examples, and are
ineligible wherever that date is required. No year was guessed or repaired.
`FILING_DATE`, ZIP CRC, required columns, primary keys, foreign keys and numeric
signal fields remain strict.

## What this does not establish

The quarterly DERA files do not provide the strict historical public clock.
We still need each selected accession's raw SGML acceptance header and earliest
cross-quarter daily-index evidence, then effective-dated security mapping,
corporate actions, delistings, prices, matched controls, costs and inference.
Consequently `event_return_backtest_run=false`, `alpha_claim_permitted=false`,
and the overall status remains `NOT VALIDATED`.

Official source: [SEC Insider Transactions Data Sets](https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets)
"""
    report_path = output_dir / "report-source.md"
    report_path.write_text(report, encoding="utf-8")

    output_files = [
        plan_path,
        inventory_path,
        quality_path,
        transport_path,
        gates_path,
        summary_path,
        report_path,
    ]
    output_files.extend([targeted_test_path, full_test_path])
    evidence.extend(("output", path) for path in output_files)
    resolved = [(role, path.resolve()) for role, path in evidence]
    paths = [path for _, path in resolved]
    if len(paths) != len(set(paths)):
        duplicates = sorted(
            {path.as_posix() for path in paths if paths.count(path) > 1}
        )
        raise RuntimeError(f"artifact evidence contains duplicate paths: {duplicates}")
    manifest = {
        "schema_version": "sec_form4_bulk_history_artifact_manifest_v1",
        "generated_at": generated_at,
        "research_status": "NOT VALIDATED",
        "alpha_claim_permitted": False,
        "records": [
            {
                "role": role,
                "path": path.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for role, path in resolved
        ],
    }
    _write_json(output_dir / "artifact_manifest.json", manifest)
    return output_dir


if __name__ == "__main__":
    print(build())
