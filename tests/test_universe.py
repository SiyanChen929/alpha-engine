import pandas as pd
import pytest

from alpha_factor_engine.contracts import PointInTimeViolation
from alpha_factor_engine.data.universe import (
    PointInTimeUniverse,
    UniverseQualityViolation,
    UniverseScreens,
    delisting_coverage_report,
    select_rolling_universe,
    select_universe_at,
    validate_security_master,
)


def _security_master() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "permno": [10001, 10001, 10002, 10003, 10004, 10005, 10006, 10007],
            "valid_from": pd.to_datetime(
                [
                    "2019-01-01",
                    "2021-01-01",
                    "2019-01-01",
                    "2019-01-01",
                    "2019-01-01",
                    "2019-01-01",
                    "2019-01-01",
                    "2019-01-01",
                ],
                utc=True,
            ),
            "valid_to": pd.to_datetime(
                ["2021-01-01", None, None, None, None, None, None, None], utc=True
            ),
            "available_at": pd.to_datetime(
                [
                    "2019-01-02",
                    "2021-01-02",
                    "2019-01-02",
                    "2019-01-02",
                    "2019-01-02",
                    "2019-01-02",
                    "2019-01-02",
                    "2020-02-01",
                ],
                utc=True,
            ),
            "active": [True, False, True, True, True, True, True, True],
            "shrcd": [10, 10, 11, 10, 12, 10, 10, 10],
            "exchcd": [1, 1, 2, 3, 1, 4, 1, 1],
            "price": [20.0, 20.0, 30.0, 25.0, 40.0, 50.0, 4.0, 100.0],
            "adv": [2e6, 2e6, 3e6, 3e6, 3e6, 3e6, 3e6, 10e6],
            "market_cap": [500e6, 500e6, 900e6, 900e6, 2e9, 2e9, 2e9, 9e9],
            "dlret": [None, -0.35, None, None, None, None, None, None],
        }
    )


def test_security_master_requires_permanent_id_and_valid_intervals():
    missing_permno = _security_master().drop(columns="permno")
    with pytest.raises(PointInTimeViolation, match="permno"):
        validate_security_master(missing_permno)

    ticker_identity = _security_master().rename(columns={"permno": "ticker"})
    with pytest.raises(PointInTimeViolation, match="permanent ID"):
        validate_security_master(ticker_identity, permanent_id_column="ticker")

    invalid = _security_master()
    invalid.loc[0, "valid_to"] = pd.Timestamp("2018-12-31", tz="UTC")
    with pytest.raises(PointInTimeViolation, match="validity intervals"):
        validate_security_master(invalid)

    malformed = _security_master().astype({"valid_to": "object"})
    malformed.loc[0, "valid_to"] = "not-a-date"
    with pytest.raises(PointInTimeViolation, match="valid_to"):
        validate_security_master(malformed)


def test_snapshot_uses_only_known_records_and_all_required_screens():
    screens = UniverseScreens(
        min_price=5.0,
        min_adv=1_000_000.0,
        min_market_cap=100_000_000.0,
    )
    result = select_universe_at(
        _security_master(), "2020-01-31T23:59:00Z", screens=screens, top_n=3
    )

    # 10004 is not a common share, 10005 is on an excluded exchange, 10006
    # fails price, and high-cap 10007 was not yet available.
    assert result["permno"].tolist() == [10002, 10003, 10001]
    assert result["universe_rank"].tolist() == [1, 2, 3]
    assert (result["available_at"] <= result["decision_time"]).all()


def test_top_n_ties_are_deterministic_by_adv_then_permno():
    frame = _security_master().query("permno in [10002, 10003]").copy()
    frame.loc[:, "adv"] = 3_000_000.0
    first = select_universe_at(frame, "2020-01-31", top_n=1)
    second = select_universe_at(frame.sample(frac=1, random_state=7), "2020-01-31", top_n=1)
    assert first["permno"].tolist() == [10002]
    assert second["permno"].tolist() == [10002]


def test_rolling_history_includes_later_inactive_name_only_while_active():
    engine = PointInTimeUniverse(_security_master())
    rolling = engine.rolling(["2022-01-31", "2020-01-31"], top_n=10)

    earlier = rolling.loc[rolling["decision_time"].eq(pd.Timestamp("2020-01-31", tz="UTC"))]
    later = rolling.loc[rolling["decision_time"].eq(pd.Timestamp("2022-01-31", tz="UTC"))]
    assert 10001 in earlier["permno"].tolist()
    assert 10001 not in later["permno"].tolist()
    assert rolling["decision_time"].drop_duplicates().is_monotonic_increasing

    audit_snapshot = engine.snapshot("2022-01-31", top_n=10, include_inactive=True)
    inactive = audit_snapshot.loc[audit_snapshot["permno"].eq(10001)].iloc[0]
    assert not inactive["active"]
    assert inactive["dlret"] == pytest.approx(-0.35)


def test_delisting_quality_report_fails_closed_on_missing_return():
    incomplete = _security_master()
    incomplete.loc[incomplete["permno"].eq(10001), "dlret"] = None

    report = delisting_coverage_report(incomplete, fail_closed=False)
    assert not report.passed
    assert report.coverage == 0.0
    assert report.missing_permanent_ids == (10001,)
    with pytest.raises(UniverseQualityViolation) as exc_info:
        delisting_coverage_report(incomplete)
    assert exc_info.value.report == report


def test_delisting_quality_report_rejects_vacuous_current_only_history():
    current_only = _security_master().loc[lambda value: value["active"]].copy()
    with pytest.raises(UniverseQualityViolation, match="current-survivor-only"):
        delisting_coverage_report(current_only)

    complete = delisting_coverage_report(_security_master())
    assert complete.passed
    assert complete.coverage == 1.0
    assert complete.required_delisting_ids == 1


def test_custom_permanent_id_is_supported_but_ticker_is_not_identity():
    frame = _security_master().rename(columns={"permno": "security_id"})
    result = select_rolling_universe(
        frame,
        ["2020-01-31", "2022-01-31"],
        permanent_id_column="security_id",
        top_n=2,
    )
    assert "security_id" in result
    assert len(result) == 4
