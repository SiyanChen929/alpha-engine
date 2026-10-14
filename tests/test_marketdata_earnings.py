from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from alpha_factor_engine.data.marketdata_earnings import MarketDataEarningsClient, MarketDataEarningsError


class _Response:
    status_code = 200

    def __init__(self, payload: dict[str, object]) -> None:
        self.content = json.dumps(payload).encode()

    def raise_for_status(self) -> None:
        return None


def _payload() -> dict[str, object]:
    return {
        "s": "ok",
        "symbol": ["MSFT"],
        "fiscalYear": [2024],
        "fiscalQuarter": [2],
        "date": [1703998800],
        "reportDate": [1706245200],
        "reportTime": ["after market close"],
        "currency": ["USD"],
        "reportedEPS": [2.93],
        "estimatedEPS": [2.78],
        "surpriseEPS": [0.15],
        "surpriseEPSpct": [0.0539568],
        "updated": [1706500000],
    }


def test_fetch_preserves_raw_and_marks_non_pit(tmp_path: Path) -> None:
    client = MarketDataEarningsClient(token="x", cache_dir=tmp_path)
    client.session.get = lambda *args, **kwargs: _Response(_payload())
    frame, record = client.fetch("MSFT", start="2024-01-01", end="2024-12-31")
    assert frame.loc[0, "report_date"] == pd.Timestamp("2024-01-26")
    assert frame.loc[0, "historical_estimate_vintage_proven"] == False
    assert frame.loc[0, "exact_first_public_timestamp_proven"] == False
    assert record.rows == 1
    assert record.finality == "current_vendor_snapshot"
    assert Path(record.notes["cache_path"]).exists()


def test_cache_hit_does_not_call_network(tmp_path: Path) -> None:
    client = MarketDataEarningsClient(token="x", cache_dir=tmp_path)
    client.session.get = lambda *args, **kwargs: _Response(_payload())
    first, _ = client.fetch("MSFT", start="2024-01-01", end="2024-12-31")
    client.session.get = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network called"))
    second, _ = client.fetch("MSFT", start="2024-01-01", end="2024-12-31")
    pd.testing.assert_frame_equal(first, second)


def test_rejects_symbol_mismatch(tmp_path: Path) -> None:
    payload = _payload()
    payload["symbol"] = ["AAPL"]
    client = MarketDataEarningsClient(token="x", cache_dir=tmp_path)
    client.session.get = lambda *args, **kwargs: _Response(payload)
    with pytest.raises(MarketDataEarningsError, match="symbol mismatch"):
        client.fetch("MSFT", start="2024-01-01", end="2024-12-31")


def test_non_demo_requires_token(tmp_path: Path) -> None:
    client = MarketDataEarningsClient(token=None, cache_dir=tmp_path)
    with pytest.raises(RuntimeError, match="MARKETDATA_TOKEN"):
        client.fetch("MSFT", start="2024-01-01", end="2024-12-31")
