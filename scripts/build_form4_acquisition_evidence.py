#!/usr/bin/env python3
"""Build a reproducible evidence pack for offline Form 4 acquisition."""

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
    ACQUISITION_PLAN_SCHEMA,
    OFFLINE_RECEIPT_SCHEMA,
    load_verified_offline_bundle,
    verify_offline_import_receipt,
)
from alpha_factor_engine.data.sec_insider import SUBMISSIONS_URL, filing_archive_urls


DEFAULT_OUTPUT = ROOT / "artifacts/form4_event_alpha_v1/acquisition_plan_v1"
DEFAULT_IMPORT = ROOT / "artifacts/form4_event_alpha_v1/manual_import_2025q4_v1"
DEFAULT_CATALOG = ROOT / "configs/form4_bulk_quarter_catalog_v1.csv"
DEFAULT_CATALOG_META = ROOT / "configs/form4_bulk_quarter_catalog_v1.meta.json"
DEFAULT_TEST_REPORT = DEFAULT_OUTPUT / "targeted-tests.xml"
DEFAULT_CANDIDATES = (
    ROOT
    / "artifacts/form4_event_alpha_v1/official_2025q4_smoke/code_p_purchase_candidates.parquet"
)
DEFAULT_SMOKE_MANIFEST = (
    ROOT / "artifacts/form4_event_alpha_v1/official_2025q4_smoke/artifact_manifest.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build(
    *,
    output_dir: Path = DEFAULT_OUTPUT,
    import_dir: Path = DEFAULT_IMPORT,
    catalog_path: Path = DEFAULT_CATALOG,
    catalog_meta_path: Path = DEFAULT_CATALOG_META,
    test_report_path: Path = DEFAULT_TEST_REPORT,
    candidates_path: Path = DEFAULT_CANDIDATES,
    smoke_manifest_path: Path = DEFAULT_SMOKE_MANIFEST,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "bulk_quarter_plan.json"
    receipt_path = import_dir / "import_receipt.json"
    input_manifest_path = import_dir / "input_bundle/offline_import_manifest.json"

    if not verify_offline_import_receipt(receipt_path):  # pragma: no cover
        raise RuntimeError("offline import receipt verification failed")
    _, input_manifest_sha, imported_records = load_verified_offline_bundle(
        input_manifest_path
    )
    receipt = _read_json(receipt_path)
    plan = _read_json(plan_path)
    catalog_meta = _read_json(catalog_meta_path)
    smoke_manifest = _read_json(smoke_manifest_path)
    smoke_records = smoke_manifest.get("records")
    if not isinstance(smoke_records, list):
        raise RuntimeError("official smoke manifest has no records")
    smoke_candidate_rows = [
        row
        for row in smoke_records
        if Path(str(row.get("path", ""))).resolve() == candidates_path.resolve()
    ]
    if len(smoke_candidate_rows) != 1:
        raise RuntimeError("candidate table is not unique in official smoke manifest")
    smoke_candidate = smoke_candidate_rows[0]
    if (
        candidates_path.stat().st_size != int(smoke_candidate["bytes"])
        or _sha256(candidates_path) != smoke_candidate["sha256"]
    ):
        raise RuntimeError("candidate table differs from official smoke manifest")
    test_root = ElementTree.parse(test_report_path).getroot()
    test_suite = test_root if test_root.tag == "testsuite" else test_root.find("testsuite")
    if test_suite is None:
        raise RuntimeError("targeted test report has no testsuite")
    test_count = int(test_suite.attrib.get("tests", "0"))
    test_failures = int(test_suite.attrib.get("failures", "0"))
    test_errors = int(test_suite.attrib.get("errors", "0"))
    test_skipped = int(test_suite.attrib.get("skipped", "0"))
    if test_count <= 0 or test_failures or test_errors:
        raise RuntimeError("targeted acquisition tests did not pass")
    if receipt.get("schema_version") != OFFLINE_RECEIPT_SCHEMA:
        raise RuntimeError("offline receipt schema drifted")
    if plan.get("schema_version") != ACQUISITION_PLAN_SCHEMA:
        raise RuntimeError("bulk acquisition plan schema drifted")
    if plan.get("quarter_count") != 24 or (
        plan.get("start_quarter"), plan.get("end_quarter")
    ) != ("2020q1", "2025q4"):
        raise RuntimeError("bulk acquisition plan coverage drifted")
    records = plan.get("records")
    if not isinstance(records, list) or len(records) != 24:
        raise RuntimeError("bulk acquisition plan rows drifted")
    status = pd.DataFrame(records)
    expected_quarters = [
        f"{year}q{quarter}" for year in range(2020, 2026) for quarter in range(1, 5)
    ]
    if status["quarter"].tolist() != expected_quarters:
        raise RuntimeError("bulk acquisition quarters are not complete and ordered")
    if status["quarter"].duplicated().any():
        raise RuntimeError("bulk acquisition quarters are duplicated")
    cached = status.loc[status["status"].eq("verified_cached")]
    missing = status.loc[status["status"].eq("manual_download_required")]
    blocking = status.loc[
        status["status"].isin(
            ["invalid_cached_asset", "ambiguous_multiple_cached_versions"]
        )
    ]
    if len(cached) != int(plan["verified_cached_count"]):
        raise RuntimeError("verified-cache count does not reconcile")
    if len(missing) != int(plan["manual_download_required_count"]):
        raise RuntimeError("missing-quarter count does not reconcile")
    if len(blocking) != int(plan["blocking_cache_finding_count"]):
        raise RuntimeError("blocking-cache count does not reconcile")
    if len(imported_records) != 1 or imported_records[0].role != "bulk:2025q4":
        raise RuntimeError("expected exactly the imported 2025Q4 bulk record")
    imported = imported_records[0]
    q4 = cached.loc[cached["quarter"].eq("2025q4")]
    if len(q4) != 1 or q4.iloc[0]["sha256"] != imported.sha256:
        raise RuntimeError("2025Q4 plan/cache/import hash mismatch")
    if receipt["input_manifest_sha256"] != input_manifest_sha:
        raise RuntimeError("receipt/input-manifest hash mismatch")
    if receipt["transport_authenticity_independently_verified"] is not False:
        raise RuntimeError("manual transport must not be presented as authenticated")

    candidates = pd.read_parquet(candidates_path)
    required_candidate_columns = {
        "event_id",
        "accession",
        "filing_date",
        "document_type",
        "issuer_cik",
        "ticker",
    }
    missing_candidate_columns = required_candidate_columns.difference(candidates.columns)
    if missing_candidate_columns:
        raise RuntimeError(
            f"candidate table missing columns: {sorted(missing_candidate_columns)}"
        )
    candidates["accession"] = candidates["accession"].astype("string")
    candidates["issuer_cik"] = candidates["issuer_cik"].astype("string").str.zfill(10)
    candidates["filing_date"] = pd.to_datetime(candidates["filing_date"], errors="raise")
    if candidates["accession"].duplicated().any():
        raise RuntimeError("candidate accessions must be unique before raw acquisition")
    if not candidates["accession"].str.fullmatch(r"\d{10}-\d{2}-\d{6}").all():
        raise RuntimeError("candidate table contains invalid accessions")
    if not candidates["issuer_cik"].str.fullmatch(r"\d{10}").all():
        raise RuntimeError("candidate table contains invalid issuer CIKs")
    if not candidates["document_type"].astype("string").isin(["4"]).all():
        raise RuntimeError("primary candidate table contains a non-Form-4 record")

    raw_plan = candidates[
        ["event_id", "accession", "issuer_cik", "filing_date", "ticker"]
    ].copy()
    raw_plan["source_url"] = [
        filing_archive_urls(issuer_cik=cik, accession=accession)[
            "complete_submission"
        ]
        for cik, accession in zip(raw_plan["issuer_cik"], raw_plan["accession"])
    ]
    raw_plan["offline_role"] = "raw_submission:" + raw_plan["accession"]
    raw_plan["status"] = "manual_download_required"
    raw_plan["required_clock_use"] = "raw_acceptance_header"
    raw_plan.to_csv(
        output_dir / "raw_submission_acquisition_plan.csv",
        index=False,
        lineterminator="\n",
    )

    issuer_plan = (
        candidates[["issuer_cik"]]
        .drop_duplicates()
        .sort_values("issuer_cik", kind="mergesort")
        .reset_index(drop=True)
    )
    issuer_plan["source_url"] = [
        SUBMISSIONS_URL.format(cik=int(cik)) for cik in issuer_plan["issuer_cik"]
    ]
    issuer_plan["offline_role"] = (
        "submissions:" + issuer_plan["issuer_cik"] + ":main"
    )
    issuer_plan["status"] = "manual_download_required"
    issuer_plan["clock_use"] = "provisional_crosscheck_only"
    issuer_plan.to_csv(
        output_dir / "submissions_api_acquisition_plan.csv",
        index=False,
        lineterminator="\n",
    )

    daily_directory_plan = pd.DataFrame(
        [
            ("2025q3", "https://www.sec.gov/Archives/edgar/daily-index/2025/QTR3/index.json"),
            ("2025q4", "https://www.sec.gov/Archives/edgar/daily-index/2025/QTR4/index.json"),
            ("2026q1", "https://www.sec.gov/Archives/edgar/daily-index/2026/QTR1/index.json"),
        ],
        columns=["quarter", "source_url"],
    )
    daily_directory_plan["status"] = "directory_index_required"
    daily_directory_plan["purpose"] = (
        "enumerate actual master.YYYYMMDD.idx files before downloading daily bytes"
    )
    daily_directory_plan.to_csv(
        output_dir / "daily_directory_acquisition_plan.csv",
        index=False,
        lineterminator="\n",
    )

    status_columns = [
        "quarter",
        "source_url",
        "displayed_size_mb",
        "status",
        "bytes",
        "sha256",
        "cache_path",
    ]
    status[status_columns].to_csv(
        output_dir / "bulk_quarter_status.csv", index=False, lineterminator="\n"
    )

    gates = pd.DataFrame(
        [
            (
                "offline_manifest_integrity",
                True,
                f"1/1 records; manifest SHA256 {input_manifest_sha}",
            ),
            (
                "bulk_archive_schema_and_quarter",
                True,
                (
                    f"2025Q4 parsed; {imported.validation['submissions']:,} submissions; "
                    f"{imported.validation['nonderivative_transactions']:,} non-derivative transactions"
                ),
            ),
            (
                "content_addressed_cache_reverified",
                True,
                f"2025Q4 {imported.byte_count:,} bytes / {imported.sha256}",
            ),
            (
                "bulk_history_2020q1_2025q4_complete",
                len(cached) == 24,
                f"{len(cached)}/24 verified; {len(missing)} manual downloads remain",
            ),
            (
                "manual_transport_independently_authenticated",
                False,
                "declared official URL and valid SEC shape; transport not independently authenticated",
            ),
            (
                "raw_sgml_clock_coverage",
                False,
                "0/1,434 current candidates retained in this acquisition bundle",
            ),
            (
                "daily_index_first_public_coverage",
                False,
                "no daily master assets imported in this bundle",
            ),
            (
                "alpha_claim_authorized",
                False,
                "data acquisition only; event return backtest remains blocked",
            ),
        ],
        columns=["gate", "passed", "evidence"],
    )
    gates.to_csv(output_dir / "validation_gates.csv", index=False, lineterminator="\n")

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "research_status": "NOT VALIDATED",
        "alpha_claim_permitted": False,
        "bulk_required_quarters": 24,
        "bulk_verified_cached_quarters": int(len(cached)),
        "bulk_manual_download_required_quarters": int(len(missing)),
        "bulk_blocking_cache_findings": int(len(blocking)),
        "catalog_displayed_total_mb": float(plan["displayed_total_mb"]),
        "imported_quarter": "2025q4",
        "imported_bytes": imported.byte_count,
        "imported_sha256": imported.sha256,
        "transport_authenticity_independently_verified": False,
        "raw_sgml_assets_imported": 0,
        "raw_sgml_assets_planned": int(len(raw_plan)),
        "submissions_main_assets_planned": int(len(issuer_plan)),
        "daily_directory_indexes_planned": int(len(daily_directory_plan)),
        "daily_directory_quarters": daily_directory_plan["quarter"].tolist(),
        "daily_master_assets_imported": 0,
        "event_return_backtest_run": False,
        "targeted_regression_tests": {
            "tests": test_count,
            "failures": test_failures,
            "errors": test_errors,
            "skipped": test_skipped,
        },
        "targeted_test_scope": (
            "tests/test_sec_insider_offline.py tests/test_sec_insider.py "
            "tests/test_form4_event_runner.py"
        ),
    }
    _write_json(output_dir / "acquisition_summary.json", summary)

    report = f"""# SEC Form 4 offline acquisition evidence v1

## Verdict

**NOT VALIDATED.** The existing 2025Q4 SEC insider bulk ZIP is now retained in
a content-addressed workspace cache and independently re-hashed against its
offline import bundle. This is data-lineage progress only; no return or alpha
claim is authorized.

## Completed

- Imported quarter: `2025q4`
- Exact bytes: `{imported.byte_count:,}`
- SHA256: `{imported.sha256}`
- Parsed submissions: `{imported.validation['submissions']:,}`
- Parsed non-derivative transactions: `{imported.validation['nonderivative_transactions']:,}`
- Receipt/source/cache byte comparison: pass
- Targeted regression suite: `{test_count} passed`, `{test_skipped} skipped`

The import checks the immutable manifest, path safety, exact bytes/hash,
official SEC URL structure, ZIP CRC, documented table schema, identifiers,
dates, numeric fields, primary keys, and foreign keys. The receipt keeps
`transport_authenticity_independently_verified=false`: a manually supplied
file with a plausible SEC shape and declared SEC URL is not cryptographic proof
of its transport origin.

## Acquisition plan

The frozen 2020Q1–2025Q4 catalog contains 24 exact official quarter links with
{plan['displayed_total_mb']:.2f} MB of displayed download size. Current status:

- Verified in cache: `{len(cached)}/24`
- Manual download required: `{len(missing)}/24`
- Invalid or ambiguous cache findings: `{len(blocking)}`

The clock-acquisition plan is now accession-specific: `{len(raw_plan):,}` raw
complete submissions, `{len(issuer_plan):,}` issuer submissions JSON files for
provisional clock cross-checking, and three daily-index directory catalogs
(`2025Q3`, `2025Q4`, `2026Q1`). The directory catalogs must be fetched first so
we download actual listed `master.YYYYMMDD.idx` files instead of probing every
calendar date.

Automated downloads remain disabled until the operator supplies a real declared
`SEC_USER_AGENT`. The alternative is one-by-one browser download followed by
`prepare-bulk`, `import`, and `verify`; the offline tool itself makes zero network
requests.

## Remaining research blockers

1. Acquire and freeze the remaining 23 quarterly archives for complete
   three-year routine/opportunistic insider classification history.
2. Retain every candidate accession's complete raw SGML and reconcile the raw
   `<ACCEPTANCE-DATETIME>` header.
3. Retain daily master bytes and HTTP `Last-Modified` metadata across quarter
   boundaries to find the earliest conservative public-evidence timestamp.
4. Complete effective-dated security mapping, prices, corporate actions,
   delisting outcomes, matched controls, costs, and inference before any event
   performance claim.

Official references: [SEC Insider Transactions Data Sets](https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets) and [Accessing EDGAR Data](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data).
"""
    (output_dir / "report-source.md").write_text(report, encoding="utf-8")

    evidence = [
        catalog_path,
        catalog_meta_path,
        plan_path,
        receipt_path,
        input_manifest_path,
        imported.path,
        Path(receipt["records"][0]["cache_path"]),
        output_dir / "bulk_quarter_status.csv",
        output_dir / "validation_gates.csv",
        output_dir / "acquisition_summary.json",
        output_dir / "report-source.md",
        output_dir / "raw_submission_acquisition_plan.csv",
        output_dir / "submissions_api_acquisition_plan.csv",
        output_dir / "daily_directory_acquisition_plan.csv",
        candidates_path,
        smoke_manifest_path,
        test_report_path,
        ROOT / "src/alpha_factor_engine/data/sec_insider.py",
        ROOT / "src/alpha_factor_engine/data/sec_insider_offline.py",
        ROOT / "scripts/manage_form4_offline_acquisition.py",
        Path(__file__).resolve(),
        ROOT / "tests/test_sec_insider_offline.py",
    ]
    normalized = [path.resolve() for path in evidence]
    if len(normalized) != len(set(normalized)):
        raise RuntimeError("acquisition evidence contains duplicate paths")
    manifest = {
        "generated_at": summary["generated_at"],
        "research_status": "NOT VALIDATED",
        "alpha_claim_permitted": False,
        "records": [
            {
                "path": path.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in normalized
        ],
    }
    _write_json(output_dir / "artifact_manifest.json", manifest)
    return output_dir


if __name__ == "__main__":
    print(build())
