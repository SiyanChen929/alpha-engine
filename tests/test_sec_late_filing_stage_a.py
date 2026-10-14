from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

from alpha_factor_engine.research.sec_late_filing_stage_a import (
    LateFilingPlanError,
    build_late_filing_candidate_plan,
    load_late_filing_inventory,
)


def _universe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cik": [1, 2],
            "ticker": ["AAA", "BBB"],
            "proxy_universe_rank": [1, 2],
            "selection_as_of": ["2026-08-31", "2026-08-31"],
            "evidence_status": ["current_survivor_proxy", "current_survivor_proxy"],
        }
    )


def _inventory() -> pd.DataFrame:
    rows = [
        ("0000000001", "a", "NT 10-K", "2021-03-02", "2020-12-31", "2021-03-02T21:01:00Z"),
        ("0000000001", "b", "NT 10-Q", "2023-05-12", "2023-03-31", "2023-05-12T20:01:00Z"),
        ("0000000002", "c", "NT 10-K", "2025-03-03", "2024-12-31", "2025-03-03T12:01:00Z"),
        ("0000000002", "d", "NT 10-Q", "2024-04-01", "2024-03-31", "2024-04-01T12:01:00Z"),
    ]
    result = []
    for cik, accession, form, filed, report, accepted in rows:
        result.append(
            {
                "issuer_cik": cik,
                "accession": f"0000000001-24-00000{ord(accession) - 96}",
                "form": form,
                "filing_date": pd.Timestamp(filed),
                "report_date": pd.Timestamp(report),
                "acceptance_at_raw": accepted,
                "acceptance_at": pd.Timestamp(accepted),
                "primary_document": f"{accession}.htm",
                "primary_document_url": f"https://www.sec.gov/{accession}.htm",
                "source_kind": "recent",
                "source_path": f"/{accession}.json.gz",
                "source_file_sha256": "1" * 64,
                "source_content_sha256": "2" * 64,
            }
        )
    return pd.DataFrame(result)


def test_candidate_plan_freezes_phases_and_excludes_bad_report_lag() -> None:
    candidates, exclusions, misses = build_late_filing_candidate_plan(
        _inventory(),
        _universe(),
        lag_bounds={"NT 10-K": [45, 110], "NT 10-Q": [25, 65]},
    )
    assert len(candidates) == 3
    assert len(exclusions) == 1
    assert misses.empty
    assert set(candidates["phase"]) == {"development", "confirmation", "sealed_2025"}
    assert not candidates["outcome_join_performed"].any()
    assert not candidates["outcome_join_permitted"].any()
    assert exclusions.iloc[0]["eligibility_reason"] == "report_lag_outside_frozen_form_bounds"


def test_candidate_plan_rejects_missing_cik_mapping() -> None:
    with pytest.raises(LateFilingPlanError, match="failed to map"):
        build_late_filing_candidate_plan(
            _inventory(),
            _universe().iloc[[0]],
            lag_bounds={"NT 10-K": [45, 110], "NT 10-Q": [25, 65]},
        )


def test_late_inventory_reads_only_target_forms_from_frozen_source(tmp_path: Path) -> None:
    cache = tmp_path / "sec"
    main_path = cache / "submissions" / "CIK0000000001.json.gz"
    main_path.parent.mkdir(parents=True)
    table = {
        "accessionNumber": ["0000000001-24-000001", "0000000001-24-000002"],
        "filingDate": ["2024-03-01", "2024-03-02"],
        "reportDate": ["2023-12-31", "2023-12-31"],
        "acceptanceDateTime": ["2024-03-01T20:00:00Z", "2024-03-02T20:00:00Z"],
        "form": ["NT 10-K", "10-K"],
        "primaryDocument": ["notice.htm", "annual.htm"],
    }
    payload = {"cik": "1", "filings": {"recent": table, "files": []}}
    main_path.write_bytes(gzip.compress(json.dumps(payload).encode("utf-8")))

    frame = load_late_filing_inventory(
        cache,
        issuer_ciks=[1],
        start_date="2018-01-01",
        end_date="2025-12-31",
        allowed_source_paths=[main_path],
    )
    assert len(frame) == 1
    assert frame.iloc[0]["form"] == "NT 10-K"
    assert frame.iloc[0]["acceptance_at"] == pd.Timestamp("2024-03-01T20:00:00Z")


def test_late_inventory_fails_if_main_source_not_in_manifest(tmp_path: Path) -> None:
    cache = tmp_path / "sec"
    main_path = cache / "submissions" / "CIK0000000001.json.gz"
    main_path.parent.mkdir(parents=True)
    main_path.write_bytes(b"unused")
    with pytest.raises(LateFilingPlanError, match="outside the frozen manifest"):
        load_late_filing_inventory(
            cache,
            issuer_ciks=[1],
            start_date="2018-01-01",
            end_date="2025-12-31",
            allowed_source_paths=[tmp_path / "different.json.gz"],
        )
