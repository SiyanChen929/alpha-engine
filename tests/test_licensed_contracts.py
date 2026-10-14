import pandas as pd
import pytest

from alpha_factor_engine.contracts import PointInTimeViolation
from alpha_factor_engine.data.licensed import CONTRACTS, ParquetPITAdapter


def test_compustat_contract_fails_closed_on_missing_knowledge_time(tmp_path):
    frame = pd.DataFrame({"gvkey": ["1"], "datadate": ["2023-12-31"], "at": [100.0]})
    path = tmp_path / "bad.parquet"
    frame.to_parquet(path)
    with pytest.raises(PointInTimeViolation):
        ParquetPITAdapter(CONTRACTS["compustat_pit"]).load(path)


def test_crsp_contract_accepts_multiple_dates_for_one_permno(tmp_path):
    frame = pd.DataFrame(
        {
            "permno": [10001, 10001],
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"], utc=True),
            "ret": [0.01, -0.02],
            "dlret": [None, None],
            "prc": [10.0, 9.8],
            "shrout": [1000, 1000],
            "shrcd": [10, 10],
            "exchcd": [1, 1],
            "available_at": pd.to_datetime(["2024-01-03", "2024-01-04"], utc=True),
        }
    )
    path = tmp_path / "crsp.parquet"
    frame.to_parquet(path)
    result = ParquetPITAdapter(CONTRACTS["crsp"]).load(path)
    assert len(result) == 2
    assert "effective_at" in result


def test_future_event_may_be_known_before_effective_date(tmp_path):
    frame = pd.DataFrame(
        {
            "event_id": ["earnings:ABC:2024Q1"],
            "event_type": ["earnings"],
            "effective_at": pd.to_datetime(["2024-05-01T20:15:00Z"]),
            "announced_at": pd.to_datetime(["2024-04-01T12:00:00Z"]),
            "available_at": pd.to_datetime(["2024-04-01T12:01:00Z"]),
            "revision_id": ["v1"],
        }
    )
    path = tmp_path / "events.parquet"
    frame.to_parquet(path)
    result = ParquetPITAdapter(CONTRACTS["events"]).load(path)
    assert result.loc[0, "available_at"] < result.loc[0, "effective_at"]
