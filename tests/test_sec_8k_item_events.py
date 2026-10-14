from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from alpha_factor_engine.data.sec_8k_item_events import load_item_inventory
from alpha_factor_engine.data.sec_periodic_filings import PeriodicFilingDataError


def _write(path: Path) -> None:
    path.parent.mkdir(parents=True)
    recent = {
        "accessionNumber": ["0000000001-24-000001", "0000000001-24-000002", "0000000001-24-000003"],
        "filingDate": ["2024-02-01", "2024-02-02", "2024-02-03"],
        "reportDate": ["2023-12-31", "", "2023-12-31"],
        "acceptanceDateTime": ["2024-02-01T21:30:00.000Z", "2024-02-02T17:00:00.000Z", "2024-02-03T17:00:00.000Z"],
        "form": ["8-K", "8-K/A", "8-K"],
        "items": ["2.02,4.02", "4.02", "14.02"],
        "primaryDocument": ["a.htm", "b.htm", "c.htm"],
    }
    payload = {"cik": "1", "filings": {"recent": recent, "files": []}}
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


def test_exact_item_and_nonamended_form(tmp_path: Path) -> None:
    source = tmp_path / "submissions" / "CIK0000000001.json.gz"
    _write(source)
    frame = load_item_inventory(tmp_path, required_item="4.02")
    assert frame["accession"].tolist() == ["0000000001-24-000001"]
    assert frame.loc[0, "required_item"] == "4.02"
    assert frame.loc[0, "filing_acceptance_et_date_delta_days"] == 0


@pytest.mark.parametrize("token", ["", " 4.02", "4.02,5.02"])
def test_rejects_noncanonical_item_token(tmp_path: Path, token: str) -> None:
    with pytest.raises(PeriodicFilingDataError):
        load_item_inventory(tmp_path, required_item=token)
