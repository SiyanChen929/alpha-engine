import pandas as pd
import pytest

from alpha_factor_engine.research.form4 import (
    Form4PurchaseFilter,
    Form4ResearchError,
    build_event_schedule,
    build_filing_date_quarantine_schedule,
    build_trailing_purchase_clusters,
    classify_routine_opportunistic,
    collapse_entry_session_events,
    filter_code_p_purchases,
    match_event_controls,
)


def _accession(number):
    return f"0000000001-25-{number:06d}"


def _form4_tables():
    submissions = pd.DataFrame(
        [
            [_accession(1), "2025-01-03", "4", "0000000101", "Issuer 1", "AAA", "0"],
            [_accession(2), "2025-01-03", "4/A", "0000000102", "Issuer 2", "BBB", "0"],
            [_accession(3), "2025-01-03", "4", "0000000103", "Issuer 3", "CCC", "1"],
            [_accession(4), "2025-01-03", "4", "0000000104", "Issuer 4", "DDD", "false"],
            [_accession(5), "2025-01-03", "4", "0000000105", "Issuer 5", "EEE", "false"],
            [_accession(6), "2025-01-03", "4", "0000000106", "Issuer 6", "FFF", "false"],
        ],
        columns=[
            "ACCESSION_NUMBER",
            "FILING_DATE",
            "DOCUMENT_TYPE",
            "ISSUERCIK",
            "ISSUERNAME",
            "ISSUERTRADINGSYMBOL",
            "AFF10B5ONE",
        ],
    )
    owners = pd.DataFrame(
        [
            [_accession(1), "0000001001", "Owner 1", "Officer", "CFO"],
            [_accession(1), "0000001999", "Passive Owner", "TenPercentOwner", ""],
            [_accession(2), "0000001002", "Owner 2", "Officer", "CEO"],
            [_accession(3), "0000001003", "Owner 3", "Director", ""],
            [_accession(4), "0000001004", "Owner 4", "TenPercentOwner", ""],
            [_accession(5), "0000001005", "Owner 5", "Director", ""],
            [_accession(6), "0000001006", "Owner 6", "Officer", "COO"],
        ],
        columns=[
            "ACCESSION_NUMBER",
            "RPTOWNERCIK",
            "RPTOWNERNAME",
            "RPTOWNER_RELATIONSHIP",
            "RPTOWNER_TITLE",
        ],
    )
    rows = [
        [_accession(1), "1", "Common Stock", "2025-01-02", "4", "P", "0", "", 1000, 20, "A", "D"],
        [_accession(1), "2", "Class A Common Stock", "2025-01-02", "4", "P", "false", "L", 500, 20, "A", "I"],
        [_accession(1), "3", "Common Stock", "2025-01-02", "4", "A", "0", "", 10000, 20, "A", "D"],
        [_accession(2), "1", "Common Stock", "2025-01-02", "4", "P", "0", "", 2000, 20, "A", "D"],
        [_accession(3), "1", "Common Stock", "2025-01-02", "4", "P", "0", "", 2000, 20, "A", "D"],
        [_accession(4), "1", "Common Stock", "2025-01-02", "4", "P", "0", "", 2000, 20, "A", "D"],
        [_accession(5), "1", "Convertible Preferred Stock", "2025-01-02", "4", "P", "0", "", 2000, 20, "A", "D"],
        [_accession(6), "1", "Common Stock", "2025-01-02", "4", "P", "0", "", 10, 20, "A", "D"],
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


def test_code_p_filter_is_conservative_aggregates_rows_and_excludes_amendments():
    events = filter_code_p_purchases(*_form4_tables())

    assert events["accession"].tolist() == [_accession(1)]
    assert events.loc[0, "purchase_usd"] == 30_000
    assert events.loc[0, "qualifying_transaction_rows"] == 2
    assert bool(events.loc[0, "any_direct_ownership"])
    assert bool(events.loc[0, "any_late_transaction"])
    assert events.loc[0, "signal_type"] == "form4_code_p_purchase"
    assert events.loc[0, "owner_ciks"] == "0000001001"
    assert events.loc[0, "reporting_owner_count"] == 2


def test_code_p_filter_does_not_treat_missing_10b5_indicator_as_false():
    submissions, owners, transactions = _form4_tables()
    submissions.loc[submissions["ACCESSION_NUMBER"].eq(_accession(1)), "AFF10B5ONE"] = None

    events = filter_code_p_purchases(
        submissions,
        owners,
        transactions,
        config=Form4PurchaseFilter(minimum_purchase_usd=0),
    )

    assert _accession(1) not in set(events["accession"])


def test_pre2023_history_can_be_extracted_only_with_explicit_primary_opt_out():
    submissions, owners, transactions = _form4_tables()
    submissions = submissions.drop(columns=["AFF10B5ONE"])

    with pytest.raises(Form4ResearchError, match="AFF10B5ONE"):
        filter_code_p_purchases(submissions, owners, transactions)

    history = filter_code_p_purchases(
        submissions,
        owners,
        transactions,
        config=Form4PurchaseFilter(
            minimum_purchase_usd=0,
            require_explicit_non_10b5_1=False,
        ),
    )

    assert _accession(1) in set(history["accession"])


def test_routine_classification_uses_only_three_complete_prior_years():
    trades = pd.DataFrame(
        {
            "owner_cik": ["R", "R", "R", "O", "O", "O", "U"],
            "transaction_date": [
                "2021-03-05",
                "2022-03-10",
                "2023-03-15",
                "2021-01-05",
                "2022-02-10",
                "2023-03-15",
                "2023-04-01",
            ],
            "information_at": [
                "2021-03-07T20:00:00Z",
                "2022-03-12T20:00:00Z",
                "2023-03-17T20:00:00Z",
                "2021-01-07T20:00:00Z",
                "2022-02-12T20:00:00Z",
                "2023-03-17T20:00:00Z",
                "2023-04-03T20:00:00Z",
            ],
        }
    )

    result = classify_routine_opportunistic(trades, ["2024-01-01"])
    labels = result.set_index("owner_cik")["classification"].to_dict()

    assert labels == {"O": "opportunistic", "R": "routine", "U": "unclassified"}
    assert result.set_index("owner_cik").loc["R", "routine_months"] == "3"


def test_routine_classification_excludes_prior_year_trade_disclosed_after_cutoff():
    trades = pd.DataFrame(
        {
            "owner_cik": ["R", "R", "R"],
            "transaction_date": ["2021-03-05", "2022-03-10", "2023-03-31"],
            "information_at": [
                "2021-03-07T20:00:00Z",
                "2022-03-12T20:00:00Z",
                "2024-01-02T20:00:00Z",
            ],
        }
    )

    result = classify_routine_opportunistic(trades, ["2024-01-01"])

    assert result.loc[0, "classification"] == "unclassified"
    assert result.loc[0, "active_years"] == 2
    assert result.loc[0, "known_trade_count"] == 2


def test_routine_classification_requires_information_clock():
    trades = pd.DataFrame(
        {"owner_cik": ["R"], "transaction_date": ["2023-03-31"]}
    )

    with pytest.raises(Form4ResearchError, match="information_at"):
        classify_routine_opportunistic(trades, ["2024-01-01"])


def test_event_schedule_uses_first_open_strictly_after_information_time():
    opens = pd.to_datetime(
        [
            "2025-01-02T14:30:00Z",
            "2025-01-03T14:30:00Z",
            "2025-01-06T14:30:00Z",
            "2025-01-07T14:30:00Z",
            "2025-01-08T14:30:00Z",
        ],
        utc=True,
    )
    events = pd.DataFrame(
        {
            "event_id": ["premarket", "intraday"],
            "information_at": [
                pd.Timestamp("2025-01-02T13:00:00Z"),
                pd.Timestamp("2025-01-02T15:00:00Z"),
            ],
        }
    )

    scheduled = build_event_schedule(events, opens, holding_sessions=2).set_index("event_id")

    assert scheduled.loc["premarket", "entry_at"] == opens[0]
    assert scheduled.loc["premarket", "exit_at"] == opens[2]
    assert scheduled.loc["intraday", "entry_at"] == opens[1]
    assert scheduled.loc["intraday", "exit_at"] == opens[3]


def test_filing_date_quarantine_waits_one_complete_exchange_session():
    opens = pd.to_datetime(
        [
            "2025-01-03T14:30:00Z",
            "2025-01-06T14:30:00Z",
            "2025-01-07T14:30:00Z",
            "2025-01-08T14:30:00Z",
            "2025-01-09T14:30:00Z",
        ],
        utc=True,
    )
    events = pd.DataFrame(
        {"event_id": ["friday"], "filing_date": ["2025-01-03"]}
    )

    scheduled = build_filing_date_quarantine_schedule(
        events,
        opens,
        quarantine_full_sessions=1,
        holding_sessions=2,
    )

    assert scheduled.loc[0, "information_at"] == pd.Timestamp(
        "2025-01-07T04:59:59.999999999Z"
    )
    assert scheduled.loc[0, "entry_at"] == opens[2]
    assert scheduled.loc[0, "exit_at"] == opens[4]
    assert scheduled.loc[0, "information_at"] < scheduled.loc[0, "entry_at"]


def test_filing_date_schedule_without_extra_quarantine_still_never_enters_same_day():
    opens = pd.to_datetime(
        [
            "2025-01-03T14:30:00Z",
            "2025-01-06T14:30:00Z",
            "2025-01-07T14:30:00Z",
        ],
        utc=True,
    )
    events = pd.DataFrame(
        {"event_id": ["friday"], "filing_date": ["2025-01-03"]}
    )

    scheduled = build_filing_date_quarantine_schedule(
        events,
        opens,
        quarantine_full_sessions=0,
        holding_sessions=1,
    )

    assert scheduled.loc[0, "entry_at"] == opens[1]


def test_trailing_cluster_is_causal_and_never_reads_later_filings():
    events = pd.DataFrame(
        {
            "event_id": ["e1", "e2", "e3"],
            "accession": ["a1", "a2", "a3"],
            "issuer_cik": ["1", "1", "1"],
            "information_at": pd.to_datetime(
                ["2025-01-01T20:00Z", "2025-01-04T20:00Z", "2025-01-09T20:00Z"],
                utc=True,
            ),
            "purchase_usd": [25_000, 50_000, 75_000],
            "owner_ciks": ["o1", "o2", "o3"],
        }
    )

    result = build_trailing_purchase_clusters(events, window_days=5).set_index("event_id")

    assert result.loc["e1", "cluster_accessions"] == "a1"
    assert result.loc["e2", "cluster_purchase_usd"] == 75_000
    assert result.loc["e2", "cluster_owner_count"] == 2
    assert result.loc["e3", "cluster_accessions"] == "a2|a3"


def test_cluster_rejects_nonfinite_purchase_amounts():
    events = pd.DataFrame(
        {
            "event_id": ["e1"],
            "accession": ["a1"],
            "issuer_cik": ["1"],
            "information_at": ["2025-01-01T20:00Z"],
            "purchase_usd": [float("inf")],
            "owner_ciks": ["o1"],
        }
    )

    with pytest.raises(Form4ResearchError, match="non-numeric"):
        build_trailing_purchase_clusters(events)


def test_entry_session_collapse_requires_one_exit_and_preserves_causal_totals():
    entry = pd.Timestamp("2025-01-03T14:30:00Z")
    exit_at = pd.Timestamp("2025-02-03T14:30:00Z")
    events = pd.DataFrame(
        {
            "event_id": ["e1", "e2"],
            "accession": ["a1", "a2"],
            "issuer_cik": ["1", "1"],
            "ticker": ["AAA", "AAA"],
            "information_at": pd.to_datetime(
                ["2025-01-02T20:00Z", "2025-01-02T21:00Z"], utc=True
            ),
            "entry_at": [entry, entry],
            "exit_at": [exit_at, exit_at],
            "purchase_usd": [25_000, 50_000],
            "owner_ciks": ["o1", "o2"],
        }
    )

    collapsed = collapse_entry_session_events(events)

    assert collapsed.loc[0, "purchase_usd"] == 75_000
    assert collapsed.loc[0, "information_at"] == pd.Timestamp("2025-01-02T21:00Z")
    assert collapsed.loc[0, "owner_ciks"] == "o1|o2"
    conflicting = events.copy()
    conflicting.loc[1, "exit_at"] = exit_at + pd.Timedelta(days=1)
    with pytest.raises(Form4ResearchError, match="conflicting ticker or exit"):
        collapse_entry_session_events(conflicting)


def test_matched_controls_enforce_availability_industry_and_nearest_distance():
    entry = pd.Timestamp("2025-01-03T14:30:00Z")
    information = pd.Timestamp("2025-01-02T21:00:00Z")
    events = pd.DataFrame(
        {
            "event_id": ["e1"],
            "issuer_cik": ["1"],
            "ticker": ["EVT"],
            "information_at": [information],
            "entry_at": [entry],
            "sic2": [35],
            "size": [10.0],
            "momentum": [0.2],
        }
    )
    candidates = pd.DataFrame(
        {
            "issuer_cik": ["2", "3", "4", "5"],
            "ticker": ["NEAR", "FAR", "FUTURE", "OTHER_SIC"],
            "entry_at": [entry] * 4,
            "available_at": [
                information - pd.Timedelta(hours=1),
                information - pd.Timedelta(hours=1),
                information + pd.Timedelta(minutes=1),
                information - pd.Timedelta(hours=1),
            ],
            "eligible": [True] * 4,
            "sic2": [35, 35, 35, 36],
            "size": [10.1, 20.0, 10.0, 10.0],
            "momentum": [0.19, -0.5, 0.2, 0.2],
        }
    )

    matches = match_event_controls(
        events,
        candidates,
        feature_columns=["size", "momentum"],
        controls_per_event=1,
    )

    assert matches.loc[0, "control_ticker"] == "NEAR"
    assert matches.loc[0, "control_available_at"] <= information
    with pytest.raises(Form4ResearchError, match="requires 3"):
        match_event_controls(
            events,
            candidates,
            feature_columns=["size", "momentum"],
            controls_per_event=3,
        )


def test_matched_controls_parse_boolean_strings_and_exclude_known_events_only():
    entry = pd.Timestamp("2025-01-03T14:30:00Z")
    information = pd.Timestamp("2025-01-02T21:00:00Z")
    events = pd.DataFrame(
        {
            "event_id": ["e1"],
            "issuer_cik": ["1"],
            "ticker": ["EVT"],
            "information_at": [information],
            "entry_at": [entry],
            "sic2": [35],
            "size": [10.0],
        }
    )
    candidates = pd.DataFrame(
        {
            "issuer_cik": ["2", "3", "4"],
            "ticker": ["STRING_FALSE", "KNOWN_EVENT", "FUTURE_EVENT"],
            "entry_at": [entry] * 3,
            "available_at": [information - pd.Timedelta(hours=1)] * 3,
            "known_event_at": [
                pd.NaT,
                information - pd.Timedelta(minutes=1),
                information + pd.Timedelta(days=1),
            ],
            "eligible": ["False", "True", "True"],
            "sic2": [35, 35, 35],
            "size": [10.0, 10.1, 10.2],
        }
    )

    matches = match_event_controls(
        events,
        candidates,
        feature_columns=["size"],
        controls_per_event=1,
    )

    assert matches.loc[0, "control_ticker"] == "FUTURE_EVENT"
