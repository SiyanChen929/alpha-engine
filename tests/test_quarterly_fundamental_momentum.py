import numpy as np
import pandas as pd

from alpha_factor_engine.research.quarterly_fundamental_momentum import (
    align_quarterly_features,
    extract_quarterly_release_features,
    filing_day_end_utc,
)


def _duration(current, prior, tag="NetIncomeLoss"):
    return {
        "units": {
            "USD": [
                {
                    "start": "2024-01-01",
                    "end": "2024-03-31",
                    "val": prior,
                    "accn": "0001",
                    "fy": 2025,
                    "fp": "Q1",
                    "form": "10-Q",
                    "filed": "2025-05-01",
                },
                {
                    "start": "2025-01-01",
                    "end": "2025-03-31",
                    "val": current,
                    "accn": "0001",
                    "fy": 2025,
                    "fp": "Q1",
                    "form": "10-Q",
                    "filed": "2025-05-01",
                },
                {
                    "start": "2025-01-01",
                    "end": "2025-06-30",
                    "val": current * 2,
                    "accn": "0001",
                    "fy": 2025,
                    "fp": "Q2",
                    "form": "10-Q",
                    "filed": "2025-08-01",
                },
            ]
        }
    }


def _payload():
    return {
        "facts": {
            "us-gaap": {
                "NetIncomeLoss": _duration(15.0, 10.0),
                "RevenueFromContractWithCustomerExcludingAssessedTax": _duration(
                    120.0, 100.0
                ),
                "GrossProfit": _duration(48.0, 35.0),
                "Assets": {
                    "units": {
                        "USD": [
                            {
                                "end": "2025-03-31",
                                "val": 500.0,
                                "accn": "0001",
                                "fy": 2025,
                                "fp": "Q1",
                                "form": "10-Q",
                                "filed": "2025-05-01",
                            }
                        ]
                    }
                },
            }
        }
    }


def test_extracts_same_accession_quarterly_changes_and_ignores_ytd():
    releases = extract_quarterly_release_features("ABC", 1, _payload())
    assert len(releases) == 1
    row = releases.iloc[0]
    assert row["net_income_current"] == 15.0
    assert row["net_income_prior_year"] == 10.0
    assert np.isclose(row["earnings_change_on_assets"], 0.01)
    assert np.isclose(row["sales_growth_yoy"], 0.20)
    assert np.isclose(row["gross_margin_change_yoy"], 0.05)
    assert row["clock_source"] == "conservative_filing_day_end_et"
    assert row["available_at"] == filing_day_end_utc("2025-05-01")


def test_excludes_malformed_period_ending_after_filing():
    payload = _payload()
    payload["facts"]["us-gaap"]["NetIncomeLoss"]["units"]["USD"].append(
        {
            "start": "2025-04-01",
            "end": "2025-06-30",
            "val": 9_999.0,
            "accn": "0001",
            "fy": 2025,
            "fp": "Q1",
            "form": "10-Q",
            "filed": "2025-05-01",
        }
    )

    releases = extract_quarterly_release_features("ABC", 1, payload)

    assert len(releases) == 1
    assert releases.iloc[0]["fiscal_end"] == pd.Timestamp("2025-03-31")
    assert releases.iloc[0]["net_income_current"] == 15.0


def test_exact_acceptance_clock_overrides_conservative_fallback():
    exact = pd.Timestamp("2025-05-01 21:42:00Z")
    releases = extract_quarterly_release_features(
        "ABC", 1, _payload(), {"0001": exact}
    )
    assert releases.iloc[0]["available_at"] == exact
    assert releases.iloc[0]["clock_source"] == "submissions_acceptance_datetime"


def test_monthly_alignment_is_asof_fresh_and_label_independent():
    release = extract_quarterly_release_features("ABC", 1, _payload())
    release2 = release.copy()
    release2["ticker"] = "XYZ"
    releases = pd.concat([release, release2], ignore_index=True)
    panel = pd.DataFrame(
        {
            "ticker": ["ABC", "XYZ", "ABC", "XYZ"],
            "signal_date": pd.to_datetime(
                ["2025-05-01T20:00Z", "2025-05-01T20:00Z", "2025-05-03T20:00Z", "2025-05-03T20:00Z"]
            ),
            "target_return": [99.0, -99.0, -0.5, 0.5],
        }
    )
    first = align_quarterly_features(panel, releases, max_age_days=180)
    changed = panel.copy()
    changed["target_return"] = [-1_000, 1_000, 1_000, -1_000]
    second = align_quarterly_features(changed, releases, max_age_days=180)
    assert first.loc[:1, "quarterly_feature_eligible"].eq(False).all()
    assert first.loc[2:, "quarterly_feature_eligible"].eq(True).all()
    pd.testing.assert_series_equal(
        first["quarterly_fundamental_momentum_score"],
        second["quarterly_fundamental_momentum_score"],
    )


def test_stale_quarterly_release_is_not_carried_indefinitely():
    releases = extract_quarterly_release_features("ABC", 1, _payload())
    panel = pd.DataFrame(
        {
            "ticker": ["ABC"],
            "signal_date": pd.to_datetime(["2026-01-01T20:00Z"]),
        }
    )
    aligned = align_quarterly_features(panel, releases, max_age_days=180)
    assert not bool(aligned.iloc[0]["quarterly_feature_eligible"])
    assert pd.isna(aligned.iloc[0]["earnings_change_on_assets"])


def test_alignment_preserves_utc_when_a_ticker_has_no_release():
    releases = extract_quarterly_release_features("ABC", 1, _payload())
    panel = pd.DataFrame(
        {
            "ticker": ["ABC", "NOFACT"],
            "signal_date": pd.to_datetime(
                ["2025-05-03T20:00Z", "2025-05-03T20:00Z"]
            ),
            "available_at": pd.to_datetime(
                ["2025-05-03T19:00Z", "2025-05-03T19:00Z"]
            ),
        }
    )

    aligned = align_quarterly_features(panel, releases, max_age_days=180)

    assert str(aligned["quarterly_available_at"].dtype) == "datetime64[ns, UTC]"
    pd.testing.assert_series_equal(
        aligned["available_at"], panel["available_at"], check_names=False
    )
    assert bool(aligned.loc[aligned["ticker"].eq("ABC"), "quarterly_feature_eligible"].iloc[0])
    assert not bool(
        aligned.loc[aligned["ticker"].eq("NOFACT"), "quarterly_feature_eligible"].iloc[0]
    )
