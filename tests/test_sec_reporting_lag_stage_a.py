import pandas as pd

from alpha_factor_engine.research.sec_reporting_lag_stage_a import (
    build_reporting_lag_plan,
    prepare_annual_headers,
)


def _calendar():
    dates = pd.bdate_range("2021-12-01", "2022-04-10")
    return pd.DataFrame({
        "session_date": dates,
        "market_open_utc": pd.to_datetime(dates.astype(str) + " 14:30", utc=True),
        "market_close_utc": pd.to_datetime(dates.astype(str) + " 21:00", utc=True),
    })


def _headers():
    rows = []
    for index, lag in enumerate([35, 40, 45, 50, 55, 60, 65, 70, 75, 80]):
        report = pd.Timestamp("2021-11-30")
        rows.append({
            "issuer_cik": index + 1,
            "ticker": f"T{index:02d}",
            "accession": f"A{index:02d}",
            "form": "10-K",
            "filing_date": report + pd.Timedelta(days=lag),
            "report_date": report,
            "acceptance_at": (report + pd.Timedelta(days=lag)).tz_localize("UTC"),
        })
    return pd.DataFrame(rows)


def test_prepare_headers_calculates_calendar_lag():
    annual, exclusions = prepare_annual_headers(_headers())
    assert exclusions.empty
    assert annual["reporting_lag_days"].tolist() == [35, 40, 45, 50, 55, 60, 65, 70, 75, 80]


def test_plan_uses_public_data_and_deterministic_tail_quintiles():
    cohorts = {
        "confirmation": {"formation_month_start": "2022-03", "formation_month_end": "2022-03"}
    }
    schedule, _, _ = build_reporting_lag_plan(_headers(), _calendar(), cohorts=cohorts)
    assert set(schedule["direction"]) == {-1, 1}
    assert set(schedule.loc[schedule["direction"].eq(1), "reporting_lag_days"]) == {35, 40}
    assert set(schedule.loc[schedule["direction"].eq(-1), "reporting_lag_days"]) == {75, 80}
    assert schedule["accepted_at"].le(schedule["decision_at"]).all()
    assert schedule["outcome_join_performed"].eq(False).all()


def test_sealed_phase_forbids_outcomes():
    cohorts = {
        "sealed_2025": {"formation_month_start": "2022-03", "formation_month_end": "2022-03"}
    }
    schedule, _, _ = build_reporting_lag_plan(_headers(), _calendar(), cohorts=cohorts)
    assert schedule["outcome_join_permitted"].eq(False).all()
