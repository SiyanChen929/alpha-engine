import json

import pytest

from alpha_factor_engine.data.fred import FredClient


class _Response:
    content = json.dumps(
        {
            "count": 1,
            "observations": [
                {
                    "date": "2020-01-01",
                    "value": "1.0",
                    "realtime_start": "2020-02-01",
                    "realtime_end": "2020-02-28",
                }
            ]
        }
    ).encode()

    def raise_for_status(self):
        return None


class _Session:
    def __init__(self):
        self.params = None

    def get(self, url, params, timeout):
        self.params = params
        return _Response()


def test_alfred_requests_all_vintage_observations(tmp_path):
    client = FredClient(cache_dir=tmp_path)
    session = _Session()
    client.session = session
    frame, record = client.fetch_alfred_vintages(
        "TEST", "2020-01-01", "2020-12-31", api_key="a" * 32
    )
    assert session.params["output_type"] == 2
    assert len(frame) == 1
    assert record.finality == "vintage_complete"


class _PagedResponse:
    def __init__(self, payload):
        self.content = json.dumps(payload).encode()

    def raise_for_status(self):
        return None


class _PagedSession:
    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append(dict(params))
        return _PagedResponse(self.pages[params["offset"]])


def _observation(date, value, vintage):
    return {
        "date": date,
        "value": value,
        "realtime_start": vintage,
        "realtime_end": "9999-12-31",
    }


def test_alfred_collects_every_page_and_records_completeness(tmp_path):
    pages = {
        0: {
            "count": 3,
            "observations": [
                _observation("2020-01-01", "1.0", "2020-02-01"),
                _observation("2020-02-01", "2.0", "2020-03-01"),
            ],
        },
        2: {
            "count": 3,
            "observations": [
                _observation("2020-01-01", "1.1", "2020-03-01"),
            ],
        },
    }
    client = FredClient(cache_dir=tmp_path)
    session = _PagedSession(pages)
    client.session = session

    frame, record = client.fetch_alfred_vintages(
        "TEST", "2020-01-01", "2020-12-31", api_key="a" * 32
    )

    assert len(frame) == 3
    assert [call["offset"] for call in session.calls] == [0, 2]
    assert record.notes["api_count"] == 3
    assert record.notes["pages"] == 2


def test_alfred_fails_closed_when_pagination_is_truncated(tmp_path):
    pages = {
        0: {
            "count": 3,
            "observations": [
                _observation("2020-01-01", "1.0", "2020-02-01"),
                _observation("2020-02-01", "2.0", "2020-03-01"),
            ],
        },
        2: {"count": 3, "observations": []},
    }
    client = FredClient(cache_dir=tmp_path)
    client.session = _PagedSession(pages)

    with pytest.raises(RuntimeError, match="stopped at 2 of 3"):
        client.fetch_alfred_vintages(
            "TEST", "2020-01-01", "2020-12-31", api_key="a" * 32
        )
