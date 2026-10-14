import pandas as pd
import pytest

from alpha_factor_engine.contracts import PointInTimeViolation
from alpha_factor_engine.data.broad_universe import (
    BroadUniverseDataUnavailable,
    BroadUniverseRequirements,
    audit_broad_universe_panel,
    load_broad_universe_panel,
    validate_broad_universe_panel,
)


def _panel() -> pd.DataFrame:
    rows = []
    for month in ["2020-01-31", "2020-02-28"]:
        signal = pd.Timestamp(month, tz="America/New_York") + pd.Timedelta(hours=20)
        for security in range(4):
            base_return = 0.01 * (security + 1)
            delisting_return = -0.5 if month == "2020-02-28" and security == 3 else 0.0
            rows.append(
                {
                    "permno": 10_000 + security,
                    "permco": 20_000 + security,
                    "ticker": f"S{security}",
                    "signal_at": signal,
                    "entry_at": signal + pd.Timedelta(hours=13, minutes=30),
                    "exit_at": signal + pd.Timedelta(days=29, hours=13, minutes=30),
                    "membership_available_at": signal - pd.Timedelta(days=1),
                    "price_available_at": signal - pd.Timedelta(minutes=30),
                    "fundamental_available_at": signal - pd.Timedelta(days=10),
                    "industry_available_at": signal - pd.Timedelta(days=10),
                    "shrcd": 10,
                    "exchcd": 1,
                    "prc": 20.0,
                    "market_cap_usd": 500_000_000.0,
                    "adv_usd_3m": 5_000_000.0,
                    "historical_sector": "TECH",
                    "eligible": True,
                    "base_period_return": base_return,
                    "delisting_return": delisting_return,
                    "delisting_return_observed": True,
                    "target_return": (1 + base_return) * (1 + delisting_return) - 1,
                    "book_to_market": 0.4,
                }
            )
    return pd.DataFrame(rows)


def _requirements() -> BroadUniverseRequirements:
    return BroadUniverseRequirements(
        minimum_securities_per_month=4,
        minimum_price=5.0,
        minimum_market_cap_usd=100_000_000,
        minimum_adv_usd_3m=1_000_000,
    )


def test_broad_panel_passes_strict_contract():
    audit = validate_broad_universe_panel(
        _panel(),
        requirements=_requirements(),
        feature_columns=["book_to_market"],
    )

    assert audit["status"] == "PASS"
    assert audit["minimum_securities_observed"] == 4
    assert audit["unique_permnos"] == 4


@pytest.mark.parametrize(
    "mutation,failed_check",
    [
        (lambda frame: frame.assign(price_available_at=frame["signal_at"] + pd.Timedelta(seconds=1)), "price_available_at_is_point_in_time"),
        (lambda frame: frame.assign(delisting_return_observed=False), "delisting_return_is_explicitly_included"),
        (lambda frame: frame.assign(target_return=0.0), "delisting_return_is_explicitly_included"),
        (lambda frame: frame.assign(permco=1), "one_security_per_company"),
    ],
)
def test_broad_panel_fails_closed(mutation, failed_check):
    audit = audit_broad_universe_panel(
        mutation(_panel()),
        requirements=_requirements(),
        feature_columns=["book_to_market"],
    )

    assert audit["status"] == "FAIL"
    failed = {item["check"] for item in audit["checks"] if not item["passed"]}
    assert failed_check in failed
    with pytest.raises(PointInTimeViolation):
        validate_broad_universe_panel(
            mutation(_panel()),
            requirements=_requirements(),
            feature_columns=["book_to_market"],
        )


def test_broad_panel_rejects_missing_requested_feature():
    audit = audit_broad_universe_panel(
        _panel(),
        requirements=_requirements(),
        feature_columns=["roe"],
    )

    assert audit["status"] == "FAIL"
    feature_check = next(item for item in audit["checks"] if item["check"] == "required_model_features")
    assert feature_check["passed"] is False


def test_missing_broad_panel_has_actionable_error(tmp_path):
    with pytest.raises(BroadUniverseDataUnavailable, match="CRSP-security-master"):
        load_broad_universe_panel(tmp_path / "missing.parquet")

