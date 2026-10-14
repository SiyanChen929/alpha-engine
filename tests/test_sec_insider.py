import io
import json
import zipfile

import pandas as pd
import pytest

from alpha_factor_engine.data.sec_insider import (
    AcceptanceClockError,
    InsiderDataSchemaError,
    SecInsiderBulkClient,
    attach_acceptance_clock,
    attach_first_public_clock,
    build_first_daily_public_evidence,
    build_acceptance_map,
    daily_master_url,
    discover_quarter_links,
    filing_archive_urls,
    parse_acceptance_header,
    parse_daily_master_index,
    parse_insider_zip,
    reconcile_acceptance_headers,
)


ACCESSION = "0001225208-25-008847"


def _tsv(columns, rows):
    return pd.DataFrame(rows, columns=columns).to_csv(sep="\t", index=False)


def _archive(
    *,
    omit_submission_column=None,
    metadata_override=None,
    metadata_filename="FORM_345_metadata.json",
    filing_date="31-OCT-2025",
    transaction_date="29-OCT-2025",
    issuer_cik="0000898293",
    owner_cik="0001912920",
):
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
    submission_row = [
        ACCESSION,
        filing_date,
        "29-OCT-2025",
        None,
        "4",
        issuer_cik,
        "JABIL INC",
        "JBL",
        "0",
    ]
    if omit_submission_column:
        index = submission_columns.index(omit_submission_column)
        submission_columns.pop(index)
        submission_row.pop(index)
    owners_columns = [
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
                    "columns": [{"name": value} for value in owners_columns],
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
    if metadata_override is not None:
        metadata = metadata_override
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("SUBMISSION.tsv", _tsv(submission_columns, [submission_row]))
        archive.writestr(
            "REPORTINGOWNER.tsv",
            _tsv(
                owners_columns,
                [[ACCESSION, owner_cik, "McKay Francis", "Officer", "CFO"]],
            ),
        )
        archive.writestr(
            "NONDERIV_TRANS.tsv",
            _tsv(
                transaction_columns,
                [
                    [
                        ACCESSION,
                        "1",
                        "Common Stock",
                        transaction_date,
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
                    ]
                ],
            ),
        )
        archive.writestr("FOOTNOTES.tsv", _tsv(footnote_columns, []))
        archive.writestr(metadata_filename, json.dumps(metadata))
    return buffer.getvalue()


def test_parse_insider_zip_preserves_identifiers_and_validates_schema():
    parsed = parse_insider_zip(_archive(), quarter="2025Q4")

    assert parsed.quarter == "2025q4"
    assert parsed.submissions.loc[0, "ISSUERCIK"] == "0000898293"
    assert parsed.reporting_owners.loc[0, "RPTOWNERCIK"] == "0001912920"
    assert parsed.nonderivative_transactions.loc[0, "TRANS_SHARES"] == 1000
    assert parsed.archive_sha256


def test_parse_insider_zip_fails_on_missing_documented_column():
    with pytest.raises(InsiderDataSchemaError, match="AFF10B5ONE"):
        parse_insider_zip(
            _archive(omit_submission_column="AFF10B5ONE"), quarter="2025q4"
        )


def test_pre_2023_archive_does_not_require_later_10b5_field():
    parsed = parse_insider_zip(
        _archive(omit_submission_column="AFF10B5ONE"), quarter="2022q4"
    )

    assert "AFF10B5ONE" not in parsed.submissions


def test_pre_2023_legacy_metadata_filename_is_versioned_and_accepted():
    parsed = parse_insider_zip(
        _archive(
            omit_submission_column="AFF10B5ONE",
            metadata_filename="insider_transactions_metadata.json",
        ),
        quarter="2022q4",
    )

    assert parsed.metadata_filename == "insider_transactions_metadata.json"
    assert parsed.data_quality_issues == ()


def test_official_malformed_optional_transaction_date_is_quarantined():
    parsed = parse_insider_zip(
        _archive(transaction_date="18-JUN-0025"),
        quarter="2025q2",
    )

    assert pd.isna(parsed.nonderivative_transactions.loc[0, "TRANS_DATE"])
    assert len(parsed.data_quality_issues) == 1
    issue = parsed.data_quality_issues[0]
    assert issue["table"] == "NONDERIV_TRANS.tsv"
    assert issue["column"] == "TRANS_DATE"
    assert issue["invalid_count"] == 1
    assert issue["examples"][0]["ACCESSION_NUMBER"] == ACCESSION


def test_legacy_unpadded_decimal_ciks_are_canonicalized_and_audited():
    parsed = parse_insider_zip(
        _archive(
            omit_submission_column="AFF10B5ONE",
            metadata_filename="insider_transactions_metadata.json",
            issuer_cik="898293",
            owner_cik="1912920",
        ),
        quarter="2022q4",
    )

    assert parsed.submissions.loc[0, "ISSUERCIK"] == "0000898293"
    assert parsed.reporting_owners.loc[0, "RPTOWNERCIK"] == "0001912920"
    normalization_issues = [
        issue
        for issue in parsed.data_quality_issues
        if issue["action"] == "left_zero_padded_valid_decimal_cik_to_10_digits"
    ]
    assert sum(issue["normalized_count"] for issue in normalization_issues) == 2


def test_malformed_mandatory_filing_date_remains_fail_closed():
    with pytest.raises(InsiderDataSchemaError, match="FILING_DATE has invalid dates"):
        parse_insider_zip(
            _archive(filing_date="31-OCT-0025"),
            quarter="2025q4",
        )


def test_parse_insider_zip_fails_on_invalid_metadata_contract():
    bad_metadata = {"dialect": {"header": True, "delimiter": "\t"}, "tables": []}

    with pytest.raises(InsiderDataSchemaError, match="no table definitions"):
        parse_insider_zip(
            _archive(metadata_override=bad_metadata), quarter="2025q4"
        )


def test_acceptance_map_includes_recent_and_historical_payloads():
    recent = {
        "filings": {
            "recent": {
                "accessionNumber": [ACCESSION],
                "filingDate": ["2025-10-31"],
                "acceptanceDateTime": ["2025-10-31T20:37:07.000Z"],
                "form": ["4"],
                "primaryDocument": ["xslF345X05/doc4.xml"],
            }
        }
    }
    old_accession = "0001225208-18-001303"
    old = {
        "accessionNumber": [old_accession],
        "filingDate": ["2018-01-19"],
        "acceptanceDateTime": ["2018-01-19T21:34:49.000Z"],
        "form": ["4"],
        "primaryDocument": ["xslF345X03/doc4.xml"],
    }

    clocks = build_acceptance_map([recent, old], sources=["recent", "history"])

    assert clocks["accession"].tolist() == [old_accession, ACCESSION]
    assert str(clocks["acceptance_at"].dtype) == "datetime64[ns, UTC]"


def test_acceptance_map_rejects_missing_schema_and_conflicting_clocks():
    with pytest.raises(InsiderDataSchemaError, match="missing columns"):
        build_acceptance_map([{}], sources=["bad"])

    base = {
        "accessionNumber": [ACCESSION],
        "filingDate": ["2025-10-31"],
        "acceptanceDateTime": ["2025-10-31T20:37:07.000Z"],
        "form": ["4"],
        "primaryDocument": ["doc4.xml"],
    }
    conflict = {**base, "acceptanceDateTime": ["2025-10-31T20:38:07.000Z"]}
    with pytest.raises(AcceptanceClockError, match="conflicting"):
        build_acceptance_map([base, conflict], sources=["one", "two"])


def test_attach_acceptance_clock_is_fail_closed_and_adds_safety_delay():
    clocks = pd.DataFrame(
        {
            "accession": [ACCESSION],
            "filing_date": [pd.Timestamp("2025-10-31")],
            "acceptance_at": [pd.Timestamp("2025-10-31T20:37:07Z")],
            "form": ["4"],
        }
    )
    events = pd.DataFrame(
        {
            "accession": [ACCESSION],
            "filing_date": [pd.Timestamp("2025-10-31")],
            "document_type": ["4"],
        }
    )

    with pytest.raises(AcceptanceClockError, match="not public-time evidence"):
        attach_acceptance_clock(events, clocks, safety_delay="5min")

    joined = attach_acceptance_clock(
        events,
        clocks,
        safety_delay="5min",
        allow_acceptance_only_diagnostic=True,
    )

    assert joined.loc[0, "information_at"] == pd.Timestamp("2025-10-31T20:42:07Z")
    assert joined.loc[0, "clock_claim_status"].startswith("acceptance_only")
    with pytest.raises(AcceptanceClockError, match="missing acceptanceDateTime"):
        attach_acceptance_clock(
            events,
            clocks.iloc[0:0],
            allow_acceptance_only_diagnostic=True,
        )


def test_raw_header_and_archive_urls_are_deterministic():
    accepted = parse_acceptance_header("<ACCEPTANCE-DATETIME>20251031163707\n")
    urls = filing_archive_urls(
        issuer_cik="0000898293",
        accession=ACCESSION,
        primary_document="xslF345X05/doc4.xml",
    )

    assert accepted == pd.Timestamp("2025-10-31T20:37:07Z")
    assert urls["complete_submission"].endswith(
        "/898293/000122520825008847/0001225208-25-008847.txt"
    )
    assert urls["primary_document"].endswith("/doc4.xml")


def test_acceptance_reconciliation_fails_closed_on_json_timezone_conflict():
    clocks = pd.DataFrame(
        {
            "accession": [ACCESSION],
            "acceptance_at_raw": ["2025-10-31T16:37:07.000Z"],
            "acceptance_at": [pd.Timestamp("2025-10-31T16:37:07Z")],
        }
    )
    raw = {ACCESSION: b"<ACCEPTANCE-DATETIME>20251031163707\n"}

    with pytest.raises(AcceptanceClockError, match="conflicts with raw SGML"):
        reconcile_acceptance_headers(clocks, raw)

    diagnostic = reconcile_acceptance_headers(
        clocks, raw, require_exact_api_match=False
    )
    assert bool(diagnostic.loc[0, "json_clock_conflict"])
    assert diagnostic.loc[0, "api_raw_delta_seconds"] == -4 * 60 * 60
    assert diagnostic.loc[0, "acceptance_at"] == pd.Timestamp("2025-10-31T20:37:07Z")


def test_acceptance_reconciliation_requires_complete_raw_coverage():
    clocks = build_acceptance_map(
        [
            {
                "accessionNumber": [ACCESSION],
                "filingDate": ["2025-10-31"],
                "acceptanceDateTime": ["2025-10-31T20:37:07.000Z"],
                "form": ["4"],
                "primaryDocument": ["doc4.xml"],
            }
        ]
    )

    with pytest.raises(AcceptanceClockError, match="coverage mismatch"):
        reconcile_acceptance_headers(clocks, {})


def _daily_master(rows):
    header = """Description: Daily Index of EDGAR Dissemination Feed\n\nCIK|Company Name|Form Type|Date Filed|Filename\n--------------------------------------------------------------------------------\n"""
    return (header + "\n".join("|".join(row) for row in rows) + "\n").encode(
        "latin-1"
    )


def test_daily_master_url_matches_official_uncompressed_archive_name():
    assert daily_master_url("2025-09-23") == (
        "https://www.sec.gov/Archives/edgar/daily-index/"
        "2025/QTR3/master.20250923.idx"
    )


def test_daily_master_public_evidence_searches_across_quarters_and_collapses_rows():
    pac_accession = "0001193125-25-212761"
    q3_rows = [
        (
            "1030894",
            "CELESTICA INC",
            "4",
            "2025-04-02",
            f"edgar/data/1030894/{pac_accession}.txt",
        ),
        (
            "2082884",
            "WILSON ROBERT",
            "4",
            "2025-04-02",
            f"edgar/data/1030894/{pac_accession}.txt",
        ),
    ]
    q4_rows = [
        (
            "1030894",
            "CELESTICA INC",
            "4",
            "2025-04-02",
            f"edgar/data/1030894/{pac_accession}.txt",
        )
    ]
    first = parse_daily_master_index(
        _daily_master(q3_rows),
        index_date="2025-09-23",
        source_url=daily_master_url("2025-09-23"),
        last_modified_at="2025-09-24T02:05:32Z",
    )
    repeated = parse_daily_master_index(
        _daily_master(q4_rows),
        index_date="2025-12-09",
        source_url=daily_master_url("2025-12-09"),
        last_modified_at="2025-12-10T03:04:12Z",
    )

    evidence = build_first_daily_public_evidence([repeated, first])

    assert evidence.loc[0, "first_index_date"] == pd.Timestamp("2025-09-23")
    assert evidence.loc[0, "first_daily_master_last_modified_at"] == pd.Timestamp(
        "2025-09-24T02:05:32Z"
    )
    assert evidence.loc[0, "first_day_index_row_count"] == 2
    assert evidence.loc[0, "indexed_ciks"] == "0001030894|0002082884"
    assert daily_master_url("2025-12-09").endswith(
        "/2025/QTR4/master.20251209.idx"
    )


def test_first_public_clock_uses_later_of_raw_acceptance_and_daily_evidence():
    pac_accession = "0001193125-25-212761"
    snapshot = parse_daily_master_index(
        _daily_master(
            [
                (
                    "1030894",
                    "CELESTICA INC",
                    "4",
                    "2025-04-02",
                    f"edgar/data/1030894/{pac_accession}.txt",
                )
            ]
        ),
        index_date="2025-09-23",
        source_url=daily_master_url("2025-09-23"),
        last_modified_at="2025-09-24T02:05:32Z",
    )
    evidence = build_first_daily_public_evidence([snapshot])
    clocks = pd.DataFrame(
        {
            "accession": [pac_accession],
            "raw_header_acceptance_at": [pd.Timestamp("2025-09-23T16:30:53Z")],
            "acceptance_clock_source": ["raw_edgar_sgml_header"],
        }
    )
    events = pd.DataFrame(
        {"accession": [pac_accession], "document_type": ["4"]}
    )

    joined = attach_first_public_clock(events, clocks, evidence)

    assert joined.loc[0, "information_at"] == pd.Timestamp("2025-09-24T02:05:32Z")
    assert joined.loc[0, "accepted_at_raw"] < joined.loc[0, "information_at"]


def test_discovery_does_not_assume_one_sec_directory_and_user_agent_fails_closed(tmp_path):
    html = """
    <a href="/files/structureddata/data/insider-transactions-data-sets/2025q4_form345.zip">Q4</a>
    <a href="/files/datastandardsinnovation/data/insider-transactions-data-sets/2026q2_form345.zip">Q2</a>
    """
    links = discover_quarter_links(html)

    assert "structureddata" in links["2025q4"]
    assert "datastandardsinnovation" in links["2026q2"]
    with pytest.raises(RuntimeError, match="real contact email"):
        SecInsiderBulkClient(cache_dir=tmp_path, user_agent="bot@example.invalid")
