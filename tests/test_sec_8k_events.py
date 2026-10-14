import gzip
import json

from alpha_factor_engine.data.sec_8k_events import load_item_202_inventory


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps(value).encode()))


def _payload(items="2.02,9.01", form="8-K"):
    table = {"accessionNumber": ["0000000001-22-000001"], "filingDate": ["2022-02-01"], "reportDate": ["2022-02-01"], "acceptanceDateTime": ["2022-02-01T21:05:00.000Z"], "form": [form], "items": [items], "primaryDocument": ["event.htm"]}
    return {"cik": "1", "filings": {"recent": table, "files": []}}


def test_loads_exact_item_202_and_excludes_amendment(tmp_path):
    cache = tmp_path / "sec"
    _write(cache / "submissions/CIK0000000001.json.gz", _payload())
    frame = load_item_202_inventory(cache, issuer_ciks=[1], start_date="2022-01-01", end_date="2022-12-31")
    assert frame["accession"].tolist() == ["0000000001-22-000001"]
    _write(cache / "submissions/CIK0000000001.json.gz", _payload(form="8-K/A"))
    assert load_item_202_inventory(cache, issuer_ciks=[1]).empty


def test_item_token_match_is_exact(tmp_path):
    cache = tmp_path / "sec"
    _write(cache / "submissions/CIK0000000001.json.gz", _payload(items="12.02,9.01"))
    assert load_item_202_inventory(cache, issuer_ciks=[1]).empty


def test_preserves_filing_date_clock_mismatch(tmp_path):
    cache = tmp_path / "sec"
    value = _payload(); value["filings"]["recent"]["filingDate"] = ["2022-02-02"]
    _write(cache / "submissions/CIK0000000001.json.gz", value)
    frame = load_item_202_inventory(cache, issuer_ciks=[1])
    assert frame["filing_acceptance_et_date_delta_days"].tolist() == [1]
