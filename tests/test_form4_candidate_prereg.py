import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from alpha_factor_engine.research.form4_candidate_prereg import (
    Form4CandidateError,
    Form4CandidateSpec,
    build_clock_acquisition_plans,
    classify_candidate_events,
    extract_quarter_candidates,
    validate_global_lineage,
    verify_closed_manifest,
)


def _accession(number: int, year: int = 23) -> str:
    return f"0000000001-{year:02d}-{number:06d}"


def _quarter_tables():
    accessions = [_accession(number) for number in range(1, 8)]
    submissions = pd.DataFrame(
        {
            "ACCESSION_NUMBER": accessions,
            "FILING_DATE": ["2023-05-05"] * 7,
            "DATE_OF_ORIG_SUB": [None, None, "2023-05-04", None, None, None, None],
            "DOCUMENT_TYPE": ["4", "4", "4/A", "4", "4", "4", "4"],
            "ISSUERCIK": [f"{100 + number:010d}" for number in range(1, 8)],
            "ISSUERNAME": [f"Issuer {number}" for number in range(1, 8)],
            "ISSUERTRADINGSYMBOL": [f"T{number}" for number in range(1, 8)],
            "AFF10B5ONE": ["0", "0", "0", "1", "0", "0", "false"],
        }
    )
    owners = pd.DataFrame(
        {
            "ACCESSION_NUMBER": accessions,
            "RPTOWNERCIK": [f"{1000 + number:010d}" for number in range(1, 8)],
            "RPTOWNERNAME": [f"Owner {number}" for number in range(1, 8)],
            "RPTOWNER_RELATIONSHIP": ["Officer"] * 6 + ["Director"],
            "RPTOWNER_TITLE": ["CFO"] * 6 + [""],
        }
    )
    rows = [
        [_accession(1), "1", "Common Stock", "2023-05-03", "4", "P", "0", "", 2000, 20, "A", "D"],
        [_accession(2), "1", "Common Stock", "2023-05-03", "4", "P", "0", "", 5000, 20, "A", "I"],
        [_accession(3), "1", "Common Stock", "2023-05-03", "4", "P", "0", "", 5000, 20, "A", "D"],
        [_accession(4), "1", "Common Stock", "2023-05-03", "4", "P", "0", "", 2000, 20, "A", "D"],
        [_accession(5), "1", "Convertible Preferred Stock", "2023-05-03", "4", "P", "0", "", 5000, 20, "A", "D"],
        [_accession(6), "1", "Common Stock", "2023-04-01", "4", "P", "0", "L", 2000, 20, "A", "D"],
        [_accession(7), "1", "Class A Common Stock", "2023-05-03", "4", "P", "false", "", 1500, 20, "A", "D"],
        [_accession(7), "2", "Class A Common Stock", "2023-05-03", "4", "P", "false", "", 1000, 20, "A", "I"],
    ]
    transactions = pd.DataFrame(
        rows,
        columns=[
            "ACCESSION_NUMBER",
            "NONDERIV_TRANS_SK",
            "SECURITY_TITLE",
            "TRANS_DATE",
            "TRANS_FORM_TYPE",
            "TRANS_CODE",
            "EQUITY_SWAP_INVOLVED",
            "TRANS_TIMELINESS",
            "TRANS_SHARES",
            "TRANS_PRICEPERSHARE",
            "TRANS_ACQUIRED_DISP_CD",
            "DIRECT_INDIRECT_OWNERSHIP",
        ],
    )
    return submissions, owners, transactions


def test_extractor_enforces_original_direct_common_and_retains_indirect_audit():
    candidates, history, audit = extract_quarter_candidates(
        *_quarter_tables(), quarter="2023q2"
    )
    by_accession = candidates.set_index("accession")

    assert set(by_accession.index) == {_accession(1), _accession(4), _accession(6), _accession(7)}
    assert by_accession.loc[_accession(7), "direct_purchase_usd"] == 30_000
    assert by_accession.loc[_accession(7), "indirect_purchase_usd"] == 20_000
    assert by_accession.loc[_accession(7), "direct_transaction_rows"] == 1
    assert by_accession.loc[_accession(7), "indirect_transaction_rows"] == 1
    assert _accession(2) not in set(history["accession"])
    assert audit["amendment_form4a_rows"] == 1


def test_post_effective_10b5_and_timeliness_are_fail_closed():
    candidates, _, _ = extract_quarter_candidates(*_quarter_tables(), quarter="2023q2")
    indexed = candidates.set_index("accession")

    assert bool(indexed.loc[_accession(1), "preclock_evaluation_candidate"])
    assert not bool(indexed.loc[_accession(4), "preclock_evaluation_candidate"])
    assert "10b5_true_or_missing" in indexed.loc[_accession(4), "blocking_reasons"]
    assert not bool(indexed.loc[_accession(6), "preclock_evaluation_candidate"])
    assert "late_or_stale_filing" in indexed.loc[_accession(6), "blocking_reasons"]
    assert not candidates["research_eligible"].any()


def test_indirect_value_cannot_lift_direct_value_over_threshold():
    submissions, owners, transactions = _quarter_tables()
    transactions.loc[
        transactions["ACCESSION_NUMBER"].eq(_accession(7))
        & transactions["DIRECT_INDIRECT_OWNERSHIP"].eq("D"),
        "TRANS_SHARES",
    ] = 500
    candidates, _, _ = extract_quarter_candidates(
        submissions, owners, transactions, quarter="2023q2"
    )

    assert _accession(7) not in set(candidates["accession"])


def test_aff10b5_schema_is_required_from_2023_but_older_history_is_allowed():
    submissions, owners, transactions = _quarter_tables()
    submissions = submissions.drop(columns=["AFF10B5ONE"])
    with pytest.raises(Form4CandidateError, match="AFF10B5ONE"):
        extract_quarter_candidates(submissions, owners, transactions, quarter="2023q2")

    submissions["FILING_DATE"] = "2022-05-05"
    candidates, _, _ = extract_quarter_candidates(
        submissions, owners, transactions.assign(TRANS_DATE="2022-05-03"), quarter="2022q2"
    )
    assert candidates["evaluation_phase"].eq("history_only").all()
    assert not candidates["preclock_evaluation_candidate"].any()


def test_transaction_after_filing_is_quarantined_without_date_repair():
    submissions, owners, transactions = _quarter_tables()
    transactions.loc[
        transactions["ACCESSION_NUMBER"].eq(_accession(1)), "TRANS_DATE"
    ] = "2023-05-06"
    candidates, history, audit = extract_quarter_candidates(
        submissions, owners, transactions, quarter="2023q2"
    )

    assert _accession(1) not in set(candidates["accession"])
    assert _accession(1) not in set(history["accession"])
    assert audit["chronology_invalid_accessions"] == 1
    assert _accession(1) in audit["chronology_invalid_examples_json"]


def _history_rows():
    rows = []
    dates = {
        "R": ["2020-03-01", "2021-03-01", "2022-03-01", "2023-03-01", "2024-03-01"],
        "O": ["2020-01-01", "2021-02-01", "2022-03-01", "2023-04-01", "2024-05-01"],
        "U": ["2022-06-01", "2024-06-01"],
    }
    number = 100
    for owner, owner_dates in dates.items():
        for value in owner_dates:
            number += 1
            date = pd.Timestamp(value)
            rows.append(
                {
                    "accession": _accession(number, year=date.year % 100),
                    "source_quarter": f"{date.year}q1",
                    "filing_date": date + pd.Timedelta(days=2),
                    "provisional_information_at": (
                        date.tz_localize("America/New_York")
                        + pd.Timedelta(days=3)
                        - pd.Timedelta(nanoseconds=1)
                    ).tz_convert("UTC"),
                    "issuer_cik": "0000000100",
                    "ticker": "AAA",
                    "owner_cik": owner,
                    "transaction_date": date,
                    "clock_status": "filing_date_eod_proxy_not_research_eligible",
                }
            )
    return pd.DataFrame(rows)


def test_three_year_classification_and_joint_owner_rule_are_causal():
    candidates = pd.DataFrame(
        {
            "accession": [_accession(900), _accession(901), _accession(902)],
            "filing_date": pd.to_datetime(["2023-06-01", "2023-06-01", "2025-06-01"]),
            "qualifying_owner_ciks": ["R", "O|R", "O"],
            "preclock_evaluation_candidate": [True, True, True],
            "blocking_reasons": ["exact_public_clock_missing"] * 3,
            "research_eligible": [False] * 3,
        }
    )
    result, labels, event_owners = classify_candidate_events(candidates, _history_rows())
    indexed = result.set_index("accession")

    assert indexed.loc[_accession(900), "provisional_event_classification"] == "routine"
    assert indexed.loc[_accession(901), "provisional_event_classification"] == "mixed"
    assert indexed.loc[_accession(902), "provisional_event_classification"] == "opportunistic"
    assert bool(indexed.loc[_accession(902), "preliminary_primary_candidate"])
    assert not labels["research_eligible"].any()
    assert len(event_owners.loc[event_owners["accession"].eq(_accession(901))]) == 2


def test_classification_does_not_use_disclosure_after_january_cutoff():
    history = _history_rows()
    mask = history["owner_cik"].eq("R") & history["transaction_date"].eq(pd.Timestamp("2022-03-01"))
    history.loc[mask, "provisional_information_at"] = pd.Timestamp("2023-01-02T20:00:00Z")
    candidates = pd.DataFrame(
        {
            "accession": [_accession(910)],
            "filing_date": [pd.Timestamp("2023-06-01")],
            "qualifying_owner_ciks": ["R"],
            "preclock_evaluation_candidate": [True],
            "blocking_reasons": ["exact_public_clock_missing"],
            "research_eligible": [False],
        }
    )
    result, _, _ = classify_candidate_events(candidates, history)

    assert result.loc[0, "provisional_event_classification"] == "unclassified"


def test_clock_plan_includes_eval_and_owner_history_and_cross_quarter_panel():
    candidates = pd.DataFrame(
        {
            "accession": [_accession(1, 25), _accession(2, 24)],
            "issuer_cik": ["0000000101", "0000000102"],
            "ticker": ["AAA", "BBB"],
            "filing_date": pd.to_datetime(["2025-01-03", "2024-12-30"]),
            "preclock_evaluation_candidate": [True, False],
            "qualifying_owner_ciks": ["O", "X"],
        }
    )
    history = pd.DataFrame(
        {
            "accession": [_accession(3, 22), _accession(4, 24)],
            "issuer_cik": ["0000000103", "0000000104"],
            "ticker": ["CCC", "DDD"],
            "filing_date": pd.to_datetime(["2022-01-04", "2024-12-31"]),
            "owner_cik": ["O", "X"],
        }
    )
    raw, daily, windows = build_clock_acquisition_plans(candidates, history)

    assert set(raw["accession"]) == {_accession(1, 25), _accession(3, 22)}
    assert raw["source_url"].str.startswith("https://www.sec.gov/Archives/edgar/data/").all()
    assert daily["index_date"].min() < pd.Timestamp("2022-01-04")
    assert daily["index_date"].max() > pd.Timestamp("2025-01-03")
    assert windows["cross_quarter_required"].all()


def test_global_lineage_rejects_cross_quarter_duplicate_accession():
    candidates = pd.DataFrame(
        {
            "accession": ["0000000001-23-000001", "0000000001-23-000001"],
            "source_quarter": ["2023q1", "2023q2"],
        }
    )
    history = pd.DataFrame(
        {
            "accession": ["0000000001-23-000001"],
            "owner_cik": ["O"],
            "source_quarter": ["2023q1"],
        }
    )
    with pytest.raises(Exception, match="duplicate"):
        validate_global_lineage(candidates, history)


def test_closed_manifest_detects_unrecorded_output(tmp_path: Path):
    output = tmp_path / "a.txt"
    output.write_text("a\n", encoding="utf-8")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = tmp_path / "artifact_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "form4_candidate_prereg_artifact_manifest_v1",
                "records": [
                    {
                        "role": "output",
                        "path": str(output),
                        "bytes": output.stat().st_size,
                        "sha256": digest,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert verify_closed_manifest(manifest)

    (tmp_path / "unrecorded.csv").write_text("x\n", encoding="utf-8")
    with pytest.raises(Form4CandidateError, match="closure"):
        verify_closed_manifest(manifest)
