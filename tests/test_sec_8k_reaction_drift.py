import pandas as pd

from alpha_factor_engine.research.sec_8k_reaction_drift import build_event_clock_plan


def test_event_clock_waits_for_full_close_then_next_open():
    dates = pd.bdate_range("2021-12-20", "2022-03-31")
    calendar = pd.DataFrame({"session_date": dates, "market_open_utc": pd.to_datetime(dates.astype(str) + " 14:30", utc=True), "market_close_utc": pd.to_datetime(dates.astype(str) + " 21:00", utc=True)})
    inventory = pd.DataFrame([{"issuer_cik": "0000000001", "accession": "0000000001-22-000001", "filing_date": "2022-01-03", "acceptance_at": "2022-01-03T14:00:00Z", "items": "2.02,9.01"}])
    universe = pd.DataFrame([{"cik": 1, "ticker": "AAA"}])
    cohorts = {"confirmation": {"public_years_et": [2022]}}
    schedule, exclusions, misses = build_event_clock_plan(inventory, universe, calendar, cohorts=cohorts)
    assert exclusions.empty and misses.empty
    row = schedule.iloc[0]
    assert row["reaction_date"] == pd.Timestamp("2022-01-03")
    assert row["entry_date"] == pd.Timestamp("2022-01-04")
    assert row["public_at"] < row["reaction_close_at"] < row["entry_at"] < row["exit_at"]


def test_later_filing_date_clock_delays_reaction():
    dates = pd.bdate_range("2021-12-20", "2022-03-31")
    calendar = pd.DataFrame({"session_date": dates, "market_open_utc": pd.to_datetime(dates.astype(str) + " 14:30", utc=True), "market_close_utc": pd.to_datetime(dates.astype(str) + " 21:00", utc=True)})
    inventory = pd.DataFrame([{"issuer_cik": "1", "accession": "0000000001-22-000001", "filing_date": "2022-01-04", "acceptance_at": "2022-01-03T23:00:00Z", "items": "2.02"}])
    schedule, _, _ = build_event_clock_plan(inventory, pd.DataFrame([{"cik": 1, "ticker": "AAA"}]), calendar, cohorts={"confirmation": {"public_years_et": [2022]}})
    assert schedule.iloc[0]["reaction_date"] == pd.Timestamp("2022-01-04")


def test_same_ticker_day_keeps_earliest_public_clock():
    dates = pd.bdate_range("2021-12-20", "2022-03-31")
    calendar = pd.DataFrame({"session_date": dates, "market_open_utc": pd.to_datetime(dates.astype(str) + " 14:30", utc=True), "market_close_utc": pd.to_datetime(dates.astype(str) + " 21:00", utc=True)})
    inventory = pd.DataFrame([
        {"issuer_cik": "1", "accession": "0000000001-22-000001", "filing_date": "2022-01-03", "acceptance_at": "2022-01-03T13:00:00Z", "items": "2.02"},
        {"issuer_cik": "1", "accession": "0000000001-22-000002", "filing_date": "2022-01-03", "acceptance_at": "2022-01-03T14:00:00Z", "items": "2.02"},
    ])
    schedule, exclusions, _ = build_event_clock_plan(inventory, pd.DataFrame([{"cik": 1, "ticker": "AAA"}]), calendar, cohorts={"confirmation": {"public_years_et": [2022]}})
    assert schedule["accession"].tolist() == ["0000000001-22-000001"]
    assert exclusions["reason"].tolist() == ["same_ticker_filing_date_duplicate"]
