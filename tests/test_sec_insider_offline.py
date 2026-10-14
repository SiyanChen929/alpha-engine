from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import io
import json
from pathlib import Path
import zipfile

import pandas as pd
import pytest

from alpha_factor_engine.data.sec_insider_offline import (
    ACQUISITION_PLAN_SCHEMA,
    OFFLINE_IMPORT_SCHEMA,
    OfflineImportError,
    TRANSPORT_ATTESTATION,
    build_bulk_acquisition_plan,
    build_bulk_acquisition_plan_from_catalog,
    import_verified_offline_bundle,
    load_verified_offline_bundle,
    prepare_bulk_manual_import,
    verify_offline_import_receipt,
)


ACCESSION = "0001225208-25-008847"
BULK_URL = (
    "https://www.sec.gov/files/structureddata/data/"
    "insider-transactions-data-sets/2025q4_form345.zip"
)


def _tsv(columns, rows):
    return pd.DataFrame(rows, columns=columns).to_csv(sep="\t", index=False)


def _bulk_archive() -> bytes:
    submission_columns = [
        "ACCESSION_NUMBER",
        "FILING_DATE",
        "PERIOD_OF_REPORT",
        "DATE_OF_ORIG_SUB",
        "DOCUMENT_TYPE",
        "ISSUERCIK",
        "ISSUERNAME",
        "ISSUERTRADINGSYMBOL",
        "AFF10B5ONE",
    ]
    owner_columns = [
        "ACCESSION_NUMBER",
        "RPTOWNERCIK",
        "RPTOWNERNAME",
        "RPTOWNER_RELATIONSHIP",
        "RPTOWNER_TITLE",
    ]
    transaction_columns = [
        "ACCESSION_NUMBER",
        "NONDERIV_TRANS_SK",
        "SECURITY_TITLE",
        "TRANS_DATE",
        "DEEMED_EXECUTION_DATE",
        "TRANS_FORM_TYPE",
        "TRANS_CODE",
        "EQUITY_SWAP_INVOLVED",
        "TRANS_TIMELINESS",
        "TRANS_SHARES",
        "TRANS_PRICEPERSHARE",
        "TRANS_ACQUIRED_DISP_CD",
        "SHRS_OWND_FOLWNG_TRANS",
        "VALU_OWND_FOLWNG_TRANS",
        "DIRECT_INDIRECT_OWNERSHIP",
    ]
    footnote_columns = ["ACCESSION_NUMBER", "FOOTNOTE_ID", "FOOTNOTE_TXT"]
    metadata = {
        "dialect": {"header": True, "delimiter": "\t"},
        "tables": [
            {
                "url": "SUBMISSION.tsv",
                "tableSchema": {
                    "primaryKey": "ACCESSION_NUMBER",
                    "columns": [{"name": value} for value in submission_columns],
                },
            },
            {
                "url": "REPORTINGOWNER.tsv",
                "tableSchema": {
                    "primaryKey": ["ACCESSION_NUMBER", "RPTOWNERCIK"],
                    "columns": [{"name": value} for value in owner_columns],
                },
            },
            {
                "url": "NONDERIV_TRANS.tsv",
                "tableSchema": {
                    "primaryKey": ["ACCESSION_NUMBER", "NONDERIV_TRANS_SK"],
                    "columns": [{"name": value} for value in transaction_columns],
                },
            },
            {
                "url": "FOOTNOTES.tsv",
                "tableSchema": {
                    "primaryKey": ["ACCESSION_NUMBER", "FOOTNOTE_ID"],
                    "columns": [{"name": value} for value in footnote_columns],
                },
            },
        ],
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "SUBMISSION.tsv",
            _tsv(
                submission_columns,
                [[
                    ACCESSION,
                    "31-OCT-2025",
                    "29-OCT-2025",
                    None,
                    "4",
                    "0000898293",
                    "JABIL INC",
                    "JBL",
                    "0",
                ]],
            ),
        )
        archive.writestr(
            "REPORTINGOWNER.tsv",
            _tsv(
                owner_columns,
                [[ACCESSION, "0001912920", "McKay Francis", "Officer", "CFO"]],
            ),
        )
        archive.writestr(
            "NONDERIV_TRANS.tsv",
            _tsv(
                transaction_columns,
                [[
                    ACCESSION,
                    "1",
                    "Common Stock",
                    "29-OCT-2025",
                    None,
                    "4",
                    "P",
                    "0",
                    None,
                    "1000",
                    "20",
                    "A",
                    "5000",
                    None,
                    "D",
                ]],
            ),
        )
        archive.writestr("FOOTNOTES.tsv", _tsv(footnote_columns, []))
        archive.writestr("FORM_345_metadata.json", json.dumps(metadata))
    return buffer.getvalue()


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _record(
    root: Path,
    *,
    role: str,
    relative: str,
    content: bytes,
    media_type: str,
    source_url: str,
    metadata: dict,
) -> dict:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {
        "role": role,
        "path": relative,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "media_type": media_type,
        "source_url": source_url,
        "metadata": metadata,
    }


def _manifest(root: Path, records: list[dict]) -> Path:
    payload = {
        "schema_version": OFFLINE_IMPORT_SCHEMA,
        "snapshot_id": "offline-test-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "transport_attestation": TRANSPORT_ATTESTATION,
        "records": records,
    }
    path = root / "offline_import_manifest.json"
    path.write_bytes(_json_bytes(payload))
    return path


def test_prepare_import_and_receipt_reverify_bulk_bytes(tmp_path):
    source = tmp_path / "manual.zip"
    source.write_bytes(_bulk_archive())
    bundle = tmp_path / "bundle"
    manifest = prepare_bulk_manual_import(
        source,
        quarter="2025Q4",
        source_url=BULK_URL,
        bundle_dir=bundle,
        snapshot_id="manual-2025q4-test",
    )

    payload, _, records = load_verified_offline_bundle(manifest)
    assert payload["transport_attestation"] == TRANSPORT_ATTESTATION
    assert records[0].validation["nonderivative_transactions"] == 1

    receipt = import_verified_offline_bundle(
        manifest,
        cache_dir=tmp_path / "cache",
        receipt_path=tmp_path / "receipt.json",
    )
    imported = json.loads(receipt.read_text(encoding="utf-8"))
    assert imported["research_status"] == "NOT VALIDATED"
    assert imported["alpha_claim_permitted"] is False
    assert imported["transport_authenticity_independently_verified"] is False
    assert Path(imported["records"][0]["cache_path"]).read_bytes() == source.read_bytes()
    assert verify_offline_import_receipt(receipt) is True
    # Re-importing the exact frozen snapshot is idempotent and preserves the receipt.
    before = receipt.read_bytes()
    second = import_verified_offline_bundle(
        manifest,
        cache_dir=tmp_path / "cache",
        receipt_path=receipt,
    )
    assert second == receipt
    assert receipt.read_bytes() == before


def test_prepare_resumes_exact_source_only_partial_bundle(tmp_path):
    content = _bulk_archive()
    digest = hashlib.sha256(content).hexdigest()
    source = tmp_path / "manual.zip"
    source.write_bytes(content)
    bundle = tmp_path / "bundle"
    partial = bundle / "sources" / f"2025q4_{digest}.zip"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(content)

    manifest = prepare_bulk_manual_import(
        source,
        quarter="2025q4",
        source_url=BULK_URL,
        bundle_dir=bundle,
        snapshot_id="manual-2025q4-resume-test",
    )

    assert manifest.is_file()
    _, _, records = load_verified_offline_bundle(manifest)
    assert len(records) == 1
    assert records[0].sha256 == digest


def test_manifest_byte_tamper_and_fake_sec_url_fail_closed(tmp_path):
    source = tmp_path / "manual.zip"
    source.write_bytes(_bulk_archive())
    manifest = prepare_bulk_manual_import(
        source,
        quarter="2025q4",
        source_url=BULK_URL,
        bundle_dir=tmp_path / "bundle",
        snapshot_id="manual-2025q4-test",
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["records"][0]["source_url"] = (
        "https://sec.example.com/2025q4_form345.zip"
    )
    manifest.write_bytes(_json_bytes(payload))
    with pytest.raises(OfflineImportError, match="allowed SEC HTTPS URL"):
        load_verified_offline_bundle(manifest)

    payload["records"][0]["source_url"] = BULK_URL
    manifest.write_bytes(_json_bytes(payload))
    source_in_bundle = manifest.parent / payload["records"][0]["path"]
    source_in_bundle.write_bytes(source_in_bundle.read_bytes() + b"tamper")
    with pytest.raises(OfflineImportError, match="manifest integrity mismatch"):
        load_verified_offline_bundle(manifest)


def test_raw_submission_import_uses_archive_cik_separately_from_issuer(tmp_path):
    root = tmp_path / "raw-bundle"
    accession = "0001214659-25-015477"
    archive_cik = "1214659"
    issuer_cik = "1280452"
    raw = (
        f"<SEC-DOCUMENT>{accession}.txt\n"
        f"<ACCESSION-NUMBER>{accession}\n"
        "<ACCEPTANCE-DATETIME>20251028213817\n"
    ).encode("ascii")
    record = _record(
        root,
        role=f"raw_submission:{accession}",
        relative=f"raw/{accession}.txt",
        content=raw,
        media_type="text/plain",
        source_url=(
            "https://www.sec.gov/Archives/edgar/data/1214659/"
            f"{accession.replace('-', '')}/{accession}.txt"
        ),
        metadata={
            "accession": accession,
            "archive_cik": archive_cik,
            "issuer_cik": issuer_cik,
        },
    )
    manifest = _manifest(root, [record])

    _, _, verified = load_verified_offline_bundle(manifest)

    assert verified[0].validation["archive_cik"] == "0001214659"
    assert verified[0].validation["issuer_cik"] == "0001280452"
    assert verified[0].validation["accepted_at_raw"] == "2025-10-29T01:38:17+00:00"


def test_daily_master_import_requires_canonical_url_and_last_modified(tmp_path):
    root = tmp_path / "daily-bundle"
    content = (
        "Description: Daily Index of EDGAR Dissemination Feed\n"
        "CIK|Company Name|Form Type|Date Filed|Filename\n"
        "------------------------------------------------------------\n"
        f"898293|JABIL INC|4|2025-10-31|edgar/data/898293/{ACCESSION}.txt\n"
    ).encode("latin-1")
    record = _record(
        root,
        role="daily_master:2025-10-31",
        relative="daily/master.20251031.idx",
        content=content,
        media_type="text/plain",
        source_url=(
            "https://www.sec.gov/Archives/edgar/daily-index/"
            "2025/QTR4/master.20251031.idx"
        ),
        metadata={
            "index_date": "2025-10-31",
            "last_modified_at": "2025-11-01T02:05:32Z",
            "etag": None,
        },
    )
    manifest = _manifest(root, [record])
    _, _, verified = load_verified_offline_bundle(manifest)
    assert verified[0].validation["form4_rows"] == 1

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["records"][0]["metadata"]["last_modified_at"] = "2025-11-01"
    manifest.write_bytes(_json_bytes(payload))
    with pytest.raises(OfflineImportError, match="timezone-aware"):
        load_verified_offline_bundle(manifest)


def test_submissions_json_offline_role_is_validated_and_counted(tmp_path):
    root = tmp_path / "submissions-bundle"
    payload = {
        "filings": {
            "recent": {
                "accessionNumber": [ACCESSION],
                "filingDate": ["2025-10-31"],
                "acceptanceDateTime": ["2025-10-31T20:37:07.000Z"],
                "form": ["4"],
                "primaryDocument": ["doc4.xml"],
            }
        }
    }
    record = _record(
        root,
        role="submissions:0000898293:main",
        relative="submissions/CIK0000898293.json",
        content=_json_bytes(payload),
        media_type="application/json",
        source_url="https://data.sec.gov/submissions/CIK0000898293.json",
        metadata={"cik": "0000898293", "part": "main"},
    )

    _, _, verified = load_verified_offline_bundle(_manifest(root, [record]))

    assert verified[0].validation["filing_rows"] == 1
    assert verified[0].validation["form4_rows"] == 1


def test_quarter_plan_is_complete_and_recognizes_content_addressed_cache(tmp_path):
    content = _bulk_archive()
    digest = hashlib.sha256(content).hexdigest()
    cached = tmp_path / "cache/bulk/2025q4" / f"{digest}.zip"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(content)
    html = (
        f'<a href="{BULK_URL}">2025 Q4</a>'
        '<a href="/files/structureddata/data/insider-transactions-data-sets/'
        '2025q3_form345.zip">2025 Q3</a>'
    )

    plan = build_bulk_acquisition_plan(
        html,
        start_quarter="2025q3",
        end_quarter="2025q4",
        cache_dir=tmp_path / "cache",
    )

    assert plan["schema_version"] == ACQUISITION_PLAN_SCHEMA
    assert plan["quarter_count"] == 2
    assert plan["verified_cached_count"] == 1
    assert plan["manual_download_required_count"] == 1
    statuses = {row["quarter"]: row["status"] for row in plan["records"]}
    assert statuses == {
        "2025q3": "manual_download_required",
        "2025q4": "verified_cached",
    }

    with pytest.raises(OfflineImportError, match="missing requested quarters"):
        build_bulk_acquisition_plan(
            html,
            start_quarter="2025q2",
            end_quarter="2025q4",
            cache_dir=tmp_path / "cache",
        )


def test_frozen_catalog_plan_validates_exact_quarter_urls(tmp_path):
    catalog = pd.DataFrame(
        {
            "quarter": ["2025q3", "2025q4"],
            "source_url": [
                BULK_URL.replace("2025q4", "2025q3"),
                BULK_URL,
            ],
            "displayed_size_mb": [7.59, 7.92],
        }
    )

    plan = build_bulk_acquisition_plan_from_catalog(
        catalog,
        start_quarter="2025q3",
        end_quarter="2025q4",
        cache_dir=tmp_path / "cache",
    )

    assert plan["quarter_count"] == 2
    assert plan["displayed_total_mb"] == pytest.approx(15.51)
    bad = catalog.copy()
    bad.loc[1, "source_url"] = BULK_URL.replace("2025q4", "2025q3")
    with pytest.raises(OfflineImportError, match="does not identify quarter 2025q4"):
        build_bulk_acquisition_plan_from_catalog(
            bad,
            start_quarter="2025q3",
            end_quarter="2025q4",
            cache_dir=tmp_path / "cache",
        )
