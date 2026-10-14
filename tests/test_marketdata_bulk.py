import json

from alpha_factor_engine.data.marketdata_bulk import MarketDataBulkSnapshotClient


class _Response:
    status_code = 200

    def __init__(self, payload):
        self.content = json.dumps(payload).encode()

    def raise_for_status(self):
        return None


class _Session:
    def __init__(self, payload):
        self.payload = payload
        self.params = None

    def get(self, url, params, timeout):
        self.params = params
        return _Response(self.payload)


def test_marketdata_bulk_snapshot_is_ohlcv_only_and_unadjusted(tmp_path):
    payload = {
        "s": "ok",
        "symbol": ["AAA", "BBB"],
        "o": [10.0, 20.0],
        "h": [11.0, 21.0],
        "l": [9.0, 19.0],
        "c": [10.5, 20.5],
        "v": [1000, 2000],
        "t": [1577854800, 1577854800],
    }
    client = MarketDataBulkSnapshotClient(token="test", cache_dir=tmp_path)
    session = _Session(payload)
    client.session = session

    frame, record = client.fetch_snapshot("2020-01-02")

    assert session.params["snapshot"] == "true"
    assert session.params["adjustsplits"] == "false"
    assert frame["membership_role"].eq("none_ohlcv_only").all()
    assert record.notes["membership_role"].startswith("none")
    assert record.finality == "confirmed"

