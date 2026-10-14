import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

from alpha_factor_engine.data.prospective_universe import (
    ProspectiveUniverseViolation,
    ProspectiveUniverseRules,
    build_prospective_universe,
    require_snapshot_use,
    validate_prospective_snapshot,
    verify_source_manifest,
)


def _write_gzip_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps(value).encode("utf-8"), mtime=0))


def _fixture(tmp_path: Path, count: int = 3) -> tuple[Path, Path]:
    sec = {
        str(index): {"cik_str": 1000 + index, "ticker": f"S{index}", "title": f"Issuer {index}"}
        for index in range(count)
    }
    sec_path = tmp_path / "sec.json.gz"
    _write_gzip_json(sec_path, sec)
    yahoo = tmp_path / "yahoo"
    dates = pd.date_range("2023-01-03 09:30", periods=70, freq="B", tz="America/New_York")
    timestamps = [int(value.timestamp()) for value in dates]
    for index in range(count):
        payload = {
            "chart": {
                "error": None,
                "result": [
                    {
                        "meta": {
                            "symbol": f"S{index}",
                            "instrumentType": "EQUITY",
                            "exchangeName": "NMS",
                            "fullExchangeName": "NasdaqGS",
                            "currency": "USD",
                        },
                        "timestamp": timestamps,
                        "indicators": {
                            "quote": [
                                {
                                    "open": [10.0 + index] * len(dates),
                                    "high": [11.0 + index] * len(dates),
                                    "low": [9.0 + index] * len(dates),
                                    "close": [10.0 + index] * len(dates),
                                    "volume": [1_000_000] * len(dates),
                                }
                            ],
                            "adjclose": [{"adjclose": [10.0 + index] * len(dates)}],
                        },
                    }
                ],
            }
        }
        _write_gzip_json(yahoo / f"S{index}_2023-01-01_2023-04-30.json.gz", payload)
    return sec_path, yahoo


def _build(tmp_path: Path):
    sec, yahoo = _fixture(tmp_path)
    return build_prospective_universe(
        root=tmp_path,
        sec_cache=sec,
        yahoo_cache_dir=yahoo,
        yahoo_start="2023-01-01",
        yahoo_end="2023-04-30",
        cutoff_date="2023-04-10",
        freeze_at="2023-04-11T12:00:00Z",
        rules=ProspectiveUniverseRules(
            candidate_count=3,
            minimum_members=2,
            maximum_members=4,
            minimum_price=5,
            minimum_adv_usd_63d=1_000_000,
            minimum_price_observations=63,
        ),
    )


def test_builds_label_free_prospective_snapshot_with_raw_hashes(tmp_path):
    snapshot, candidates, sources, qa = _build(tmp_path)
    assert len(snapshot) == 3
    assert candidates["selected_member"].all()
    assert len(sources) == 4
    assert snapshot["prospective_observation_eligible"].all()
    assert not snapshot["historical_backtest_eligible"].any()
    assert not snapshot["deployable_eligible"].any()
    assert snapshot["permanent_security_id"].isna().all()
    assert snapshot["share_class"].eq("UNKNOWN").all()
    assert snapshot["identity_source_effective_at"].eq("").all()
    assert snapshot["identity_effective_time_status"].eq(
        "UNAVAILABLE_IN_SOURCE_PAYLOAD"
    ).all()
    assert qa["snapshot_integrity_status"] == "PASS"
    assert qa["historical_backtest_status"] == "NOT VALIDATED"
    assert verify_source_manifest(sources, root=tmp_path)["passed"]


def test_membership_is_deterministic_and_does_not_admit_future_bar(tmp_path):
    first = _build(tmp_path)[0]
    second = _build(tmp_path)[0]
    pd.testing.assert_frame_equal(first, second)

    sec, yahoo = _fixture(tmp_path / "future")
    with pytest.raises(ProspectiveUniverseViolation, match="cutoff_date must precede"):
        build_prospective_universe(
            root=tmp_path / "future",
            sec_cache=sec,
            yahoo_cache_dir=yahoo,
            yahoo_start="2023-01-01",
            yahoo_end="2023-04-30",
            cutoff_date="2023-04-11",
            freeze_at="2023-04-11T12:00:00Z",
            rules=ProspectiveUniverseRules(candidate_count=3, minimum_members=2, maximum_members=4),
        )


def test_consumer_blocks_pre_freeze_historical_and_deployment_use(tmp_path):
    snapshot = _build(tmp_path)[0]
    with pytest.raises(ProspectiveUniverseViolation, match="precedes membership"):
        require_snapshot_use(
            snapshot, purpose="prospective_monitoring", decision_time="2023-04-11T11:59:59Z"
        )
    require_snapshot_use(
        snapshot, purpose="prospective_monitoring", decision_time="2023-04-11T12:00:00Z"
    )
    with pytest.raises(ProspectiveUniverseViolation, match="historical_backtest"):
        require_snapshot_use(
            snapshot, purpose="historical_backtest", decision_time="2023-04-12T12:00:00Z"
        )
    with pytest.raises(ProspectiveUniverseViolation, match="deployment"):
        require_snapshot_use(
            snapshot, purpose="deployment", decision_time="2023-04-12T12:00:00Z"
        )


def test_validator_rejects_label_column_and_tampered_hash(tmp_path):
    snapshot = _build(tmp_path)[0]
    with_label = snapshot.assign(target_return=0.1)
    with pytest.raises(ProspectiveUniverseViolation, match="no_future_label_columns"):
        validate_prospective_snapshot(
            with_label,
            freeze_at="2023-04-11T12:00:00Z",
            minimum_members=2,
            maximum_members=4,
        )

    _, _, sources, _ = _build(tmp_path)
    raw_path = tmp_path / sources.loc[sources["provider"].eq("SEC EDGAR"), "relative_path"].iloc[0]
    raw_path.write_bytes(raw_path.read_bytes() + b"tamper")
    result = verify_source_manifest(sources, root=tmp_path)
    assert not result["passed"]
    assert result["errors"]
