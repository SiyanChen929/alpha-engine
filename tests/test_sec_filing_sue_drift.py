import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alpha_factor_engine.research.sec_filing_sue_drift import (
    SueDriftError,
    aggregate_entry_sessions,
    attach_event_outcomes,
    hac_mean_test,
    manual_reconstruction,
    moving_block_null_p,
    prepare_release_signals,
    verify_frozen_inputs,
)


def _config():
    return {
        "release_eligibility": {
            "forms": ["10-Q"],
            "release_kinds": ["reported_10q"],
            "clock_sources": ["submissions_acceptance_plus_3m"],
            "header_inventory_verified": True,
            "supersession_only": False,
            "net_income_same_tag_comparator": True,
            "required_non_null": [
                "ticker",
                "cik",
                "accession",
                "fiscal_end",
                "available_at",
                "earnings_change_on_assets",
            ],
            "deduplication_key": ["accession"],
        },
        "signal": {
            "base_value": "earnings_change_on_assets",
            "history_window_events": 8,
            "minimum_history_events": 8,
            "history_shift_events": 1,
            "development_window": {
                "information_at_gte": "2018-01-01T00:00:00Z",
                "information_at_lt": "2023-01-01T00:00:00Z",
            },
            "tail_calibration": {"lower_quantile": 0.2, "upper_quantile": 0.8},
        },
        "universe_and_clock": {
            "evaluation_information_at_gte": "2023-01-01T00:00:00Z",
            "evaluation_information_at_lt": "2026-01-01T00:00:00Z",
            "required_exit_open_lte": "2025-12-31T23:59:59Z",
            "benchmark": "SPY",
            "minimum_prior_close_usd": 5.0,
            "adv_window_sessions": 3,
            "minimum_prior_adv_usd": 1000.0,
        },
        "estimands": {
            "primary_horizon_sessions": 2,
            "primary_round_trip_cost_per_gross_dollar": 0.006,
            "stress_round_trip_cost_per_gross_dollar": 0.010,
        },
    }


def _releases():
    rows = []
    for ticker, multiplier in (("AAA", 1.0), ("BBB", -1.0)):
        for index, date in enumerate(pd.date_range("2018-01-02", periods=13, freq="180D")):
            rows.append(
                {
                    "ticker": ticker,
                    "cik": 1 if ticker == "AAA" else 2,
                    "accession": f"{ticker}-{index:02d}",
                    "form": "10-Q",
                    "fiscal_end": date - pd.Timedelta(days=35),
                    "available_at": pd.Timestamp(date, tz="UTC") + pd.Timedelta(hours=12),
                    "release_kind": "reported_10q",
                    "clock_source": "submissions_acceptance_plus_3m",
                    "header_inventory_verified": True,
                    "supersession_only": False,
                    "net_income_same_tag_comparator": True,
                    "earnings_change_on_assets": multiplier * (index + 1) / 1000,
                }
            )
    return pd.DataFrame(rows)


def test_signal_uses_shifted_history_and_development_only_thresholds():
    config = _config()
    releases = _releases()
    selected, thresholds, _ = prepare_release_signals(releases, config)
    assert thresholds.development_events > 0
    assert set(selected["direction"]).issubset({-1, 1})
    assert (selected["available_at"] >= pd.Timestamp("2023-01-01", tz="UTC")).all()
    changed = releases.copy()
    evaluation = pd.to_datetime(changed["available_at"], utc=True) >= pd.Timestamp(
        "2023-01-01", tz="UTC"
    )
    changed.loc[evaluation, "earnings_change_on_assets"] *= 1_000_000
    _, thresholds_changed, _ = prepare_release_signals(changed, config)
    assert thresholds == thresholds_changed
    first_complete = (
        prepare_release_signals(releases, config)[0]
        .sort_values(["ticker", "available_at"])
        .iloc[0]
    )
    assert first_complete["prior_signal_std"] > 0


def _prices():
    dates = pd.bdate_range("2022-12-01", periods=90)
    opens = (dates + pd.Timedelta(hours=9, minutes=30)).tz_localize(
        "America/New_York"
    ).tz_convert("UTC")
    rows = []
    for ticker, base, step in (("AAA", 10.0, 0.1), ("SPY", 100.0, 0.2)):
        for i, (date, open_at) in enumerate(zip(dates, opens, strict=True)):
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "timestamp": open_at,
                    "close": base + i * step,
                    "adj_open": base + i * step,
                    "dollar_volume": 10_000.0 + i,
                    "available_at": open_at + pd.Timedelta(hours=7),
                }
            )
    return pd.DataFrame(rows)


def test_event_clock_is_strict_and_liquidity_excludes_entry_session():
    config = _config()
    prices = _prices()
    spy = prices[prices["ticker"].eq("SPY")].sort_values("timestamp").reset_index(drop=True)
    chosen = spy.iloc[30]
    selected = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "cik": [1],
            "accession": ["a"],
            "available_at": [chosen["timestamp"]],
            "clock_source": ["submissions_acceptance_plus_3m"],
            "sue_like": [2.0],
            "direction": [1],
        }
    )
    events, exclusions = attach_event_outcomes(selected, prices, config)
    assert exclusions.empty
    row = events.iloc[0]
    assert row["entry_at"] > row["available_at"]
    assert row["entry_date"] == spy.iloc[31]["date"]
    aaa = prices[prices["ticker"].eq("AAA")].sort_values("date").reset_index(drop=True)
    expected_adv = aaa.loc[28:30, "dollar_volume"].mean()
    assert row["entry_prior_adv_usd"] == expected_adv
    assert row["entry_prior_feature_available_at"] < row["entry_at"]
    expected_gross = 0.5 * (row["stock_return_20"] - row["spy_return_20"])
    assert row["pair_gross_return_20"] == pytest.approx(expected_gross)
    assert row["pair_net_30bps_return_20"] == pytest.approx(expected_gross - 0.006)


def test_entry_session_aggregation_and_manual_reconstruction():
    config = _config()
    prices = _prices()
    spy = prices[prices["ticker"].eq("SPY")].sort_values("timestamp").reset_index(drop=True)
    selected = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "cik": [1, 1],
            "accession": ["a", "b"],
            "available_at": [spy.iloc[30]["timestamp"], spy.iloc[30]["timestamp"]],
            "clock_source": ["submissions_acceptance_plus_3m"] * 2,
            "sue_like": [2.0, 3.0],
            "direction": [1, 1],
        }
    )
    events, exclusions = attach_event_outcomes(selected, prices, config)
    assert exclusions.empty
    assert len(events) == 1
    assert events.iloc[0]["collapsed_accession_count"] == 2
    sessions = aggregate_entry_sessions(events)
    assert len(sessions) == 1
    reconstruction = manual_reconstruction(events)
    assert reconstruction["passed"]
    assert reconstruction["maximum_absolute_error"] <= 1e-12


def test_inference_is_deterministic_and_cost_can_fail():
    values = np.linspace(-0.03, 0.01, 80)
    test = hac_mean_test(values, lags=5)
    assert test["mean"] < 0
    assert test["t"] < 0
    first = moving_block_null_p(values, draws=100, block_length=8, seed=929)
    second = moving_block_null_p(values, draws=100, block_length=8, seed=929)
    assert first == second
    assert first["one_sided_empirical_p"] > 0.5


def test_input_verification_detects_mutation(tmp_path: Path):
    data = tmp_path / "data.bin"
    manifest = tmp_path / "manifest.json"
    data.write_bytes(b"abc")
    manifest.write_text("{}")
    data_hash = hashlib.sha256(data.read_bytes()).hexdigest()
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    declaration = {
        "path": data.name,
        "bytes": 3,
        "sha256": data_hash,
        "upstream_manifest_path": manifest.name,
        "upstream_manifest_sha256": manifest_hash,
    }
    config = {
        "sources": {
            "quarterly_release_features": copy.deepcopy(declaration),
            "daily_prices": copy.deepcopy(declaration),
        }
    }
    assert verify_frozen_inputs(config, tmp_path)["passed"]
    data.write_bytes(b"abd")
    result = verify_frozen_inputs(config, tmp_path)
    assert not result["passed"]
    assert set(result["failures"]) == {
        "quarterly_release_features:data",
        "daily_prices:data",
    }


def test_conflicting_same_session_directions_are_excluded():
    config = _config()
    prices = _prices()
    spy = prices[prices["ticker"].eq("SPY")].sort_values("timestamp").reset_index(drop=True)
    selected = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "cik": [1, 1],
            "accession": ["a", "b"],
            "available_at": [spy.iloc[30]["timestamp"], spy.iloc[30]["timestamp"]],
            "clock_source": ["submissions_acceptance_plus_3m"] * 2,
            "sue_like": [-2.0, 2.0],
            "direction": [-1, 1],
        }
    )
    with pytest.raises(SueDriftError, match="no eligible tail events"):
        attach_event_outcomes(selected, prices, config)
