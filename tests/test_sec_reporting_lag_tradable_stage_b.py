import pandas as pd

from alpha_factor_engine.research.sec_reporting_lag_tradable_stage_b import build_tradability_first_plan


def test_tradability_screen_precedes_quintile_sort():
    dates = pd.bdate_range("2020-12-01", "2022-04-08")
    calendar = pd.DataFrame({
        "session_date": dates,
        "market_open_utc": pd.to_datetime(dates.astype(str) + " 14:30", utc=True),
        "market_close_utc": pd.to_datetime(dates.astype(str) + " 21:00", utc=True),
    })
    headers = []
    prices = []
    lags = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    for i, lag in enumerate(lags):
        report = pd.Timestamp("2021-10-31")
        headers.append({"issuer_cik": i + 1, "ticker": f"T{i:02d}", "accession": f"A{i:02d}", "form": "10-K", "filing_date": report + pd.Timedelta(days=lag), "report_date": report, "acceptance_at": (report + pd.Timedelta(days=lag, hours=15)).tz_localize("UTC")})
        close = 4.0 if lag == 80 else 20.0
        for date in dates:
            prices.append({"ticker": f"T{i:02d}", "date": date, "close": close, "volume": 2_000_000})
    cohorts = {"opened_diagnostic": {"formation_month_start": "2022-03", "formation_month_end": "2022-03"}}
    schedule, eligibility = build_tradability_first_plan(pd.DataFrame(headers), calendar, pd.DataFrame(prices), cohorts=cohorts)
    assert not schedule["ticker"].eq("T10").any()
    assert eligibility.loc[eligibility["ticker"].eq("T10"), "eligibility_reason"].eq("prior_close_below_floor").all()
    assert set(schedule["direction"]) == {-1, 1}
    assert schedule["accepted_at"].le(schedule["decision_at"]).all()
