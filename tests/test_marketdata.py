import json

import pandas as pd
import pytest

from alpha_factor_engine.data.marketdata import MarketDataClient


class _Response:
    def __init__(self, payload):
        self.content = json.dumps(payload).encode()
        self.status_code = 200

    def raise_for_status(self):
        return None


class _Session:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append((url, params, timeout))
        return _Response(self.payload)


def _payload(timestamp):
    return {
        "s": "ok",
        "t": [int(timestamp.timestamp())],
        "o": [100.0],
        "h": [102.0],
        "l": [99.0],
        "c": [101.0],
        "v": [1000],
    }


def test_daily_timestamp_is_session_label_with_post_close_availability(tmp_path):
    vendor_label = pd.Timestamp("2024-01-02 00:00", tz="America/New_York")
    client = MarketDataClient(token="test", cache_dir=tmp_path)
    client.session = _Session(_payload(vendor_label))

    frame, record = client.fetch_candles(
        "TEST", "2024-01-02", "2024-01-02", resolution="D", refresh=True
    )

    expected_close = pd.Timestamp("2024-01-02 16:00", tz="America/New_York").tz_convert("UTC")
    assert frame.loc[0, "date"] == pd.Timestamp("2024-01-02")
    assert frame.loc[0, "effective_at"] == expected_close
    assert frame.loc[0, "available_at"] == expected_close + pd.Timedelta(minutes=30)
    assert record.notes["availability_latency_seconds"] == 1800


@pytest.mark.parametrize("day_offset", [0, 1])
def test_confirmed_daily_request_rejects_current_or_future_end(tmp_path, day_offset):
    requested_end = (
        pd.Timestamp.now(tz="America/New_York").normalize() + pd.Timedelta(days=day_offset)
    ).date()
    client = MarketDataClient(token="test", cache_dir=tmp_path)

    with pytest.raises(ValueError, match="end date before the current New York session date"):
        client.fetch_candles(
            "TEST",
            "2020-01-01",
            requested_end.isoformat(),
            resolution="D",
            finality="confirmed",
        )


def test_confirmed_intraday_request_rejects_current_session(tmp_path):
    today = pd.Timestamp.now(tz="America/New_York").date().isoformat()
    client = MarketDataClient(token="test", cache_dir=tmp_path)

    with pytest.raises(ValueError, match="end date before the current New York session date"):
        client.fetch_candles(
            "TEST", today, today, resolution="15", finality="confirmed"
        )


def test_intraday_effective_time_is_candle_end_plus_latency(tmp_path):
    candle_start = pd.Timestamp("2024-01-02 09:30", tz="America/New_York")
    client = MarketDataClient(token="test", cache_dir=tmp_path)
    client.session = _Session(_payload(candle_start))

    frame, record = client.fetch_candles(
        "TEST", "2024-01-02", "2024-01-02", resolution="15", refresh=True
    )

    expected_end = candle_start.tz_convert("UTC") + pd.Timedelta(minutes=15)
    assert frame.loc[0, "effective_at"] == expected_end
    assert frame.loc[0, "available_at"] == expected_end + pd.Timedelta(minutes=1)
    assert record.notes["timestamp_semantics"] == "candle start; effective_at is candle end"


def test_unsupported_resolution_is_rejected_before_request(tmp_path):
    client = MarketDataClient(token="test", cache_dir=tmp_path)

    with pytest.raises(ValueError, match="unsupported candle resolution"):
        client.fetch_candles(
            "TEST", "2024-01-01", "2024-01-31", resolution="W", refresh=True
        )
