import pandas as pd
import pytest

from alpha_factor_engine.contracts import PointInTimeViolation, point_in_time_join


def test_point_in_time_join_never_uses_future_revision():
    decisions = pd.DataFrame(
        {
            "ticker": ["A", "A", "B"],
            "signal_date": pd.to_datetime(
                ["2024-02-01T00:00Z", "2024-04-01T00:00Z", "2024-04-01T00:00Z"]
            ),
        }
    )
    facts = pd.DataFrame(
        {
            "ticker": ["A", "A", "B"],
            "available_at": pd.to_datetime(
                ["2024-01-15T00:00Z", "2024-03-15T00:00Z", "2024-05-01T00:00Z"]
            ),
            "value": [1.0, 2.0, 99.0],
        }
    )
    result = point_in_time_join(decisions, facts, by="ticker")
    assert result.loc[result["ticker"].eq("A"), "value"].tolist() == [1.0, 2.0]
    assert pd.isna(result.loc[result["ticker"].eq("B"), "value"]).all()


def test_point_in_time_contract_rejects_future_availability():
    from alpha_factor_engine.contracts import assert_no_future_information

    frame = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2024-01-01T00:00Z"]),
            "available_at": pd.to_datetime(["2024-01-02T00:00Z"]),
        }
    )
    with pytest.raises(PointInTimeViolation):
        assert_no_future_information(frame)

