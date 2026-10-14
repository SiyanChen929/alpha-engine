from __future__ import annotations

import gzip
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import alpha_factor_engine.research.form4_opportunistic_stage_a as form4_stage_a
from alpha_factor_engine.data.yahoo import YahooFinanceClient
from alpha_factor_engine.research.form4_opportunistic_stage_a import (
    ACQUISITION_END,
    ACQUISITION_START,
    CANONICAL_ACQUISITION_DIRECTORY,
    CANONICAL_ACQUISITION_SCRIPT,
    CANONICAL_CORE_CODE,
    CANONICAL_DESIGN_RECEIPT,
    CANONICAL_FREEZE_RECEIPT,
    CANONICAL_HUMAN_PROTOCOL,
    CANONICAL_OUTCOME_RUNNER,
    CANONICAL_PRICE_MANIFEST,
    CANONICAL_PRICE_PANEL,
    CANONICAL_RUN_DIRECTORY,
    CANONICAL_RUNTIME_CONFIG,
    PRICE_MANIFEST_SCHEMA,
    UNAMBIGUOUS_TICKER_PATTERN,
    FreezeReceiptError,
    Form4StageAError,
    aggregate_entry_dates,
    attach_stage_a_outcomes,
    build_stage_a_gates,
    build_acquisition_universe,
    build_design_receipt,
    build_freeze_receipt,
    centered_circular_mbb,
    hac_mean_test,
    run_stage_a_diagnostic,
    scheduled_event_coverage_ledger,
    select_candidate_events,
    sha256_file,
    stage_a_config_template,
    summarize_stage_a,
    verify_freeze_receipt,
    verify_design_receipt,
    yahoo_alias,
)


def _candidates(rows: list[tuple[str, str, str, bool, str]]) -> pd.DataFrame:
    frame = pd.DataFrame(
        rows,
        columns=[
            "accession",
            "filing_date",
            "ticker",
            "preclock_evaluation_candidate",
            "provisional_event_classification",
        ],
    )
    frame["signal_type"] = "form4_code_p_purchase_candidate"
    frame["direct_purchase_usd"] = 50_000.0
    frame["timeliness_eligible"] = True
    frame["aff10b5_eligible"] = True
    return frame


def _prices(tickers: tuple[str, ...] = ("AAA", "SPY")) -> pd.DataFrame:
    dates = pd.bdate_range("2021-12-01", "2025-03-31")
    opens = (
        dates
        + pd.Timedelta(hours=9, minutes=30)
    ).tz_localize("America/New_York").tz_convert("UTC")
    market_returns = 0.0002 + 0.003 * np.sin(np.arange(len(dates)) / 7.0)
    rows: list[dict[str, object]] = []
    for ticker in tickers:
        if ticker == "SPY":
            returns = market_returns
            start = 100.0
            alias = "SPY"
        elif ticker == "BRK.B":
            returns = 0.9 * market_returns + 0.001 * np.cos(np.arange(len(dates)) / 11.0)
            start = 300.0
            alias = "BRK-B"
        elif ticker == "LOW":
            returns = 1.1 * market_returns
            start = 4.0
            alias = "LOW"
        else:
            returns = 1.4 * market_returns + 0.0015 * np.cos(np.arange(len(dates)) / 13.0)
            start = 50.0
            alias = ticker
        adjusted_open = start * np.cumprod(1.0 + returns)
        for index, date in enumerate(dates):
            close = float(adjusted_open[index] * 1.001)
            volume = 1_000_000.0
            rows.append(
                {
                    "source_ticker": ticker,
                    "yahoo_ticker": alias,
                    "date": date,
                    "timestamp": opens[index],
                    "adj_open": float(adjusted_open[index]),
                    "adj_close": close,
                    "close": close,
                    "volume": volume,
                }
            )
    return pd.DataFrame(rows)


def test_ticker_rule_development_selection_and_holdout_seal():
    candidates = _candidates(
        [
            ("a", "2023-04-01", "aaa", True, "opportunistic"),
            ("b", "2024-12-31", "BRK.B", True, "routine"),
            ("bad", "2024-01-02", "ABC/WS", True, "opportunistic"),
            ("not-candidate", "2024-01-02", "ZZZ", False, "opportunistic"),
            ("holdout", "2025-06-02", "AAA", True, "opportunistic"),
        ]
    )
    selected, exclusions, audit = select_candidate_events(candidates)
    assert selected["accession"].tolist() == ["a", "b"]
    assert selected.set_index("source_ticker").loc["BRK.B", "yahoo_ticker"] == "BRK-B"
    assert exclusions["accession"].tolist() == ["bad"]
    assert audit["holdout_metadata_rows_present_but_not_selected"] == 1
    assert audit["holdout_event_outcomes_opened"] is False
    assert yahoo_alias("BRK.B") == "BRK-B"
    assert yahoo_alias("ABC/WS") is None
    aliases, _, acquisition = build_acquisition_universe(candidates)
    assert aliases.iloc[0]["source_ticker"] == "SPY"
    assert set(aliases["yahoo_ticker"]) == {"SPY", "AAA", "BRK-B"}
    assert acquisition["outcome_join_performed"] is False


def test_missing_ticker_is_excluded_without_nullable_boolean_error():
    candidates = _candidates(
        [
            ("valid", "2024-01-02", "AAA", True, "opportunistic"),
            ("missing", "2024-01-02", pd.NA, True, "opportunistic"),
        ]
    )
    selected, exclusions, _ = select_candidate_events(candidates)
    assert selected["accession"].tolist() == ["valid"]
    assert exclusions["accession"].tolist() == ["missing"]


def test_clock_placebo_beta_cost_and_exact_endpoint_rules():
    candidates = _candidates(
        [("a", "2023-06-03", "AAA", True, "opportunistic")]
    )
    selected, _, _ = select_candidate_events(candidates)
    prices = _prices()
    outcomes, exclusions = attach_stage_a_outcomes(selected, prices)
    assert exclusions.empty
    row = outcomes.iloc[0]
    spy_dates = (
        prices.loc[prices["source_ticker"].eq("SPY"), "date"]
        .sort_values()
        .reset_index(drop=True)
    )
    first_strict = spy_dates.loc[spy_dates.gt(pd.Timestamp("2023-06-03"))].index[0]
    assert row["entry_date"] == spy_dates.iloc[first_strict + 1]
    assert row["exit_date"] == spy_dates.iloc[first_strict + 21]
    placebo_end_index = spy_dates.loc[spy_dates.lt(pd.Timestamp("2023-06-03"))].index[-1]
    assert row["placebo_end_date"] == spy_dates.iloc[placebo_end_index]
    assert row["placebo_start_date"] == spy_dates.iloc[placebo_end_index - 20]
    assert row["beta_sample_end_date"] <= row["placebo_start_date"]
    assert row["beta_observations"] == 252

    ws = row["stock_weight"]
    wm = row["benchmark_weight"]
    assert ws == pytest.approx(1 / (1 + abs(row["beta"])))
    assert wm == pytest.approx(-row["beta"] / (1 + abs(row["beta"])))
    expected_gross = ws * row["stock_return_20"] + wm * row["spy_return_20"]
    expected_cost = 0.003 * (
        abs(ws)
        + abs(wm)
        + abs(ws * (1 + row["stock_return_20"]))
        + abs(wm * (1 + row["spy_return_20"]))
    )
    assert row["hedged_gross_return_20"] == pytest.approx(expected_gross)
    assert row["traded_notional_cost_30bps"] == pytest.approx(expected_cost)
    assert row["hedged_net_return_30bps"] == pytest.approx(expected_gross - expected_cost)

    missing = prices.loc[
        ~(
            prices["source_ticker"].eq("AAA")
            & prices["date"].eq(row["exit_date"])
        )
    ]
    with pytest.raises(Form4StageAError, match="no events pass"):
        attach_stage_a_outcomes(selected, missing)


def test_collapse_precedes_same_class_overlap_and_class_control_is_independent():
    candidates = _candidates(
        [
            ("a", "2023-06-03", "AAA", True, "opportunistic"),
            ("b", "2023-06-04", "AAA", True, "opportunistic"),
            ("c", "2023-06-10", "AAA", True, "opportunistic"),
            ("d", "2023-06-10", "AAA", True, "routine"),
        ]
    )
    selected, _, _ = select_candidate_events(candidates)
    outcomes, exclusions = attach_stage_a_outcomes(selected, _prices())
    opportunistic = outcomes.loc[outcomes["event_class"].eq("opportunistic")]
    routine = outcomes.loc[outcomes["event_class"].eq("routine")]
    assert len(opportunistic) == 1
    assert opportunistic.iloc[0]["accession_ids"] == "a|b"
    assert opportunistic.iloc[0]["collapsed_accession_count"] == 2
    assert len(routine) == 1
    assert exclusions.loc[
        exclusions["reason"].eq("overlap_same_class_until_prior_exit"),
        "accession_ids",
    ].tolist() == ["c"]
    aggregated = aggregate_entry_dates(outcomes)
    assert len(aggregated) == 2
    assert set(aggregated["event_class"]) == {"opportunistic", "routine"}


def test_missing_future_exit_cannot_promote_later_overlapping_signal():
    candidates = _candidates(
        [
            ("good", "2023-06-03", "BBB", True, "opportunistic"),
            ("early", "2024-12-20", "AAA", True, "opportunistic"),
            ("later", "2024-12-26", "AAA", True, "opportunistic"),
        ]
    )
    selected, _, _ = select_candidate_events(candidates)
    truncated = _prices(("AAA", "BBB", "SPY"))
    truncated = truncated.loc[truncated["date"].le(pd.Timestamp("2025-01-10"))]
    outcomes, exclusions = attach_stage_a_outcomes(selected, truncated)
    assert outcomes["accession_ids"].tolist() == ["good"]
    reasons = exclusions.set_index("accession_ids")["reason"].to_dict()
    assert reasons["early"] == "exit_spy_open_unavailable"
    assert reasons["later"] == "overlap_same_class_until_prior_exit"


def test_cross_class_same_ticker_entry_is_removed_before_outcome_lookup():
    candidates = _candidates(
        [
            ("kept", "2023-06-03", "BBB", True, "opportunistic"),
            ("opp-conflict", "2023-08-05", "AAA", True, "opportunistic"),
            ("routine-conflict", "2023-08-05", "AAA", True, "routine"),
        ]
    )
    selected, _, _ = select_candidate_events(candidates)
    outcomes, exclusions = attach_stage_a_outcomes(
        selected, _prices(("AAA", "BBB", "SPY"))
    )
    assert outcomes["accession_ids"].tolist() == ["kept"]
    conflicts = exclusions.loc[
        exclusions["reason"].eq("cross_class_same_ticker_entry_conflict")
    ]
    assert set(conflicts["accession_ids"]) == {"opp-conflict", "routine-conflict"}


def test_spy_clock_requires_exact_regular_open_timestamp():
    selected, _, _ = select_candidate_events(
        _candidates([("a", "2023-06-03", "AAA", True, "opportunistic")])
    )
    prices = _prices()
    prices.loc[prices["source_ticker"].eq("SPY"), "timestamp"] += pd.Timedelta(
        minutes=1
    )
    with pytest.raises(Form4StageAError, match="exact 09:30"):
        attach_stage_a_outcomes(selected, prices)


def test_prior_adv_and_history_observation_dates_are_strictly_pre_entry():
    candidates = _candidates(
        [
            ("good", "2023-06-03", "AAA", True, "opportunistic"),
            ("low", "2023-06-03", "LOW", True, "opportunistic"),
        ]
    )
    selected, _, _ = select_candidate_events(candidates)
    outcomes, exclusions = attach_stage_a_outcomes(selected, _prices(("AAA", "LOW", "SPY")))
    assert outcomes["accession_ids"].tolist() == ["good"]
    assert exclusions.loc[exclusions["accession_ids"].eq("low"), "reason"].iloc[0] == "prior_median_dollar_volume_below_20m"
    row = outcomes.iloc[0]
    assert row["prior_history_end_date"] < row["entry_date"]
    assert row["adv_window_end_date"] < row["entry_date"]
    assert row["prior_observed_sessions"] >= 252
    assert row["prior_63_session_median_dollar_volume"] >= 20_000_000


def test_centered_circular_bootstrap_is_deterministic_and_centered():
    values = 0.002 + np.sin(np.arange(80) / 5.0) / 100
    first = centered_circular_mbb(values, draws=500, block_length=20, seed=929)
    second = centered_circular_mbb(values, draws=500, block_length=20, seed=929)
    assert first == second
    assert first["variant"] == "centered_circular_all_start_moving_blocks"
    monte_carlo_se_of_mean = first["null_std"] / np.sqrt(first["draws"])
    assert abs(first["null_mean"]) <= 4 * monte_carlo_se_of_mean


def test_degenerate_hac_and_bootstrap_fail_closed():
    indices = np.arange(40)
    hac = hac_mean_test(np.repeat(0.01, 40), session_indices=indices, lags=20)
    mbb = centered_circular_mbb(np.repeat(0.01, 40))
    assert hac["status"] == "DEGENERATE_VARIANCE"
    assert hac["t"] is None
    assert hac["one_sided_p"] is None
    assert mbb["status"] == "DEGENERATE_VARIANCE"
    assert mbb["one_sided_empirical_p"] is None


def test_hac_uses_actual_spy_session_distance():
    result = hac_mean_test(
        [1.0, 1.0, -1.0, -1.0],
        session_indices=[0, 100, 200, 300],
        lags=20,
    )
    assert result["lag_unit"] == "SPY_session_distance"
    assert result["standard_error"] == pytest.approx(0.5)


def test_exact_four_formal_tests_global_floor_and_stage_a_gates():
    dates = pd.bdate_range("2023-06-01", periods=100)
    rows: list[dict[str, object]] = []
    for index, date in enumerate(dates):
        for event_class, prefix, net30, net50, placebo in (
            (
                "opportunistic",
                "O",
                0.010 + 0.002 * np.sin(index / 4.0),
                0.005 + 0.001 * np.cos(index / 5.0),
                -0.001 + 0.001 * np.sin(index / 3.0),
            ),
            (
                "routine",
                "R",
                0.001 + 0.001 * np.cos(index / 6.0),
                -0.004 + 0.001 * np.sin(index / 7.0),
                0.0005 * np.cos(index / 8.0),
            ),
        ):
            rows.append(
                {
                    "event_class": event_class,
                    "entry_date": date,
                    "filing_date": date - pd.Timedelta(days=2),
                    "quarantined_date": date - pd.Timedelta(days=1),
                    "quarantined_index": index + 9,
                    "entry_index": index + 10,
                    "exit_index": index + 30,
                    "placebo_start_date": date - pd.Timedelta(days=32),
                    "placebo_end_date": date - pd.Timedelta(days=3),
                    "beta_sample_end_date": date - pd.Timedelta(days=32),
                    "prior_history_end_date": date - pd.Timedelta(days=1),
                    "adv_window_end_date": date - pd.Timedelta(days=1),
                    "source_ticker": f"{prefix}{index:03d}",
                    "schedule_id": f"{event_class}|{prefix}{index:03d}|{date.date().isoformat()}",
                    "entry_at": pd.Timestamp(date).tz_localize("America/New_York")
                    + pd.Timedelta(hours=9, minutes=30),
                    "exit_at": pd.Timestamp(date).tz_localize("America/New_York")
                    + pd.Timedelta(days=30, hours=9, minutes=30),
                    "hedged_gross_return_20": net30 + 0.006,
                    "hedged_net_return_30bps": net30,
                    "hedged_net_return_50bps": net50,
                    "placebo_hedged_gross_return_20": placebo,
                    "stock_return_20": net30,
                    "spy_return_20": 0.0,
                    "beta": 1.0,
                    "stock_weight": 0.5,
                    "benchmark_weight": -0.5,
                }
            )
    events = pd.DataFrame(rows)
    entry_dates = aggregate_entry_dates(events)
    exclusions = pd.DataFrame(
        columns=[
            "accession_ids",
            "filing_date",
            "source_ticker",
            "event_class",
            "entry_date",
            "reason",
            "schedule_id",
        ]
    )
    summary = summarize_stage_a(
        events,
        entry_dates,
        exclusions,
        {
            "selected_development_rows": 200,
            "preclock_development_rows_in_two_classes": 200,
            "opportunistic_rows": 100,
        },
    )
    assert summary["formal_test_count"] == 4
    assert {
        test["disclosed_trial_floor_screening_multiplier"]
        for test in summary["formal_tests"].values()
    } == {124}
    assert summary["scheduled_opportunistic_event_price_path_coverage"] == 1.0
    assert summary["opportunistic_benchmark_clock_coverage"] == 1.0
    assert summary["maximum_single_ticker_gross_absolute_contribution_share"] < 0.02
    assert summary["minimum_leave_one_ticker_out_net30_mean"] > 0
    gates, decision = build_stage_a_gates(summary)
    assert gates.loc[gates["stage_a_decision_required"], "passed"].all()
    assert decision.startswith("ADVANCE_TO_STAGE_B")

    clock_failure = pd.DataFrame(
        [
            {
                "accession_ids": "clock-missing",
                "filing_date": pd.Timestamp("2024-12-31"),
                "source_ticker": "CLOCK",
                "event_class": "opportunistic",
                "entry_date": pd.NaT,
                "schedule_id": pd.NA,
                "reason": "exit_spy_open_unavailable",
            }
        ]
    )
    clock_summary = summarize_stage_a(
        events,
        entry_dates,
        clock_failure,
        {
            "selected_development_rows": 201,
            "preclock_development_rows_in_two_classes": 201,
            "opportunistic_rows": 101,
        },
    )
    assert clock_summary["opportunistic_benchmark_clock_coverage"] == pytest.approx(
        100 / 101
    )
    clock_gates, clock_decision = build_stage_a_gates(clock_summary)
    assert not clock_gates.set_index("gate").loc[
        "complete_opportunistic_benchmark_clock_coverage", "passed"
    ]
    assert clock_decision == "RETIRE_STAGE_A_PROXY"

    mapping_summary = summarize_stage_a(
        events,
        entry_dates,
        exclusions,
        {
            "selected_development_rows": 200,
            "preclock_development_rows_in_two_classes": 211,
            "opportunistic_rows": 100,
        },
    )
    mapping_gates, _ = build_stage_a_gates(mapping_summary)
    assert not mapping_gates.set_index("gate").loc[
        "minimum_candidate_ticker_mapping_coverage", "passed"
    ]


def test_scheduled_event_coverage_uses_frozen_schedule_ids():
    events = pd.DataFrame(
        [
            {
                "schedule_id": "opportunistic|AAA|2024-01-03",
                "event_class": "opportunistic",
                "source_ticker": "AAA",
                "entry_date": pd.Timestamp("2024-01-03"),
            }
        ]
    )
    exclusions = pd.DataFrame(
        [
            {
                "schedule_id": "opportunistic|LOW|2024-01-03",
                "event_class": "opportunistic",
                "source_ticker": "LOW",
                "entry_date": pd.Timestamp("2024-01-03"),
                "reason": "prior_median_dollar_volume_below_20m",
            },
            {
                "schedule_id": "opportunistic|MISS|2024-01-03",
                "event_class": "opportunistic",
                "source_ticker": "MISS",
                "entry_date": pd.Timestamp("2024-01-03"),
                "reason": "ticker_price_history_unavailable",
            },
            {
                "schedule_id": pd.NA,
                "event_class": "opportunistic",
                "source_ticker": "BAD/WS",
                "entry_date": pd.NaT,
                "reason": "ambiguous_or_unsupported_ticker",
            },
        ]
    )
    ledger = scheduled_event_coverage_ledger(events, exclusions)
    assert len(ledger) == 3
    assert ledger["price_path_complete"].sum() == 2


def _write_price_manifest(
    root: Path,
    candidate: Path,
    candidate_upstream_manifest: Path,
    yahoo_adapter: Path,
    prices: Path,
    acquisition_script: Path,
    design_receipt: Path,
) -> Path:
    acquisition_dir = prices.parent
    acquisition_dir.mkdir(parents=True, exist_ok=True)
    audit = acquisition_dir / "download_audit.csv"
    aliases = acquisition_dir / "ticker_aliases.csv"
    exclusions = acquisition_dir / "ticker_exclusions.csv"
    candidate_frame = pd.read_parquet(candidate)
    alias_frame, exclusion_frame, candidate_audit = build_acquisition_universe(
        candidate_frame
    )
    alias_frame.to_csv(aliases, index=False)
    exclusion_frame.to_csv(exclusions, index=False)

    panel = _prices(tuple(alias_frame["source_ticker"].astype(str)))
    panel = panel.loc[
        panel["date"].between(
            pd.Timestamp(ACQUISITION_START), pd.Timestamp(ACQUISITION_END)
        )
    ].copy()
    panel.to_parquet(prices, index=False)

    vendor_records: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    raw_dir = acquisition_dir / "raw"
    raw_dir.mkdir()
    design_frozen_at = pd.Timestamp(
        json.loads(design_receipt.read_text(encoding="utf-8"))["frozen_at"]
    )
    retrieved_at = (design_frozen_at + pd.Timedelta(seconds=1)).isoformat()
    for row in alias_frame.itertuples(index=False):
        source_ticker = str(row.source_ticker)
        yahoo_ticker = str(row.yahoo_ticker)
        sample = panel.loc[panel["source_ticker"].eq(source_ticker)].sort_values(
            "date"
        )
        raw_payload = json.dumps(
            {
                "chart": {
                    "error": None,
                    "result": [
                        {
                            "meta": {
                                "symbol": yahoo_ticker,
                                "exchangeTimezoneName": "America/New_York",
                            },
                            "timestamp": [
                                int(pd.Timestamp(value).timestamp())
                                for value in sample["timestamp"]
                            ],
                            "indicators": {
                                "quote": [
                                    {
                                        "open": sample["adj_open"].astype(float).tolist(),
                                        "close": sample["close"].astype(float).tolist(),
                                        "volume": sample["volume"].astype(float).tolist(),
                                    }
                                ],
                                "adjclose": [
                                    {
                                        "adjclose": sample["adj_close"]
                                        .astype(float)
                                        .tolist()
                                    }
                                ],
                            },
                        }
                    ],
                }
            },
            separators=(",", ":"),
        ).encode()
        safe = yahoo_ticker.replace("/", "_").replace(".", "_").replace("^", "index_")
        raw_sha = hashlib.sha256(raw_payload).hexdigest()
        raw_cache = raw_dir / f"{safe}_{raw_sha}.json.gz"
        raw_cache.write_bytes(gzip.compress(raw_payload))
        audit_rows.append(
            {"ticker": yahoo_ticker, "status": "ok", "rows": len(sample)}
        )
        vendor_records.append(
            {
                "provider": "Yahoo Finance chart API",
                "dataset": f"daily_prices:{yahoo_ticker}",
                "retrieved_at": retrieved_at,
                "source_url": f"https://query1.finance.yahoo.com/{yahoo_ticker}",
                "sha256": raw_sha,
                "rows": len(sample),
                "finality": "confirmed",
                "yahoo_ticker": yahoo_ticker,
                "raw_cache_path": raw_cache.relative_to(root).as_posix(),
                "raw_cache_bytes": raw_cache.stat().st_size,
                "raw_cache_sha256": sha256_file(raw_cache),
                "raw_payload_sha256": raw_sha,
            }
        )
    pd.DataFrame(audit_rows).sort_values("ticker").to_csv(audit, index=False)

    def record(path: Path) -> dict[str, object]:
        return {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    payload = {
        "schema_version": PRICE_MANIFEST_SCHEMA,
        "created_at": (design_frozen_at + pd.Timedelta(seconds=2)).isoformat(),
        "provider": "Yahoo Finance chart API",
        "source_class": "current_ticker_yahoo_retrospective",
        "historical_alpha_verdict": "INVALID",
        "design_receipt": record(design_receipt),
        "candidate_file": record(candidate),
        "candidate_upstream_manifest": record(candidate_upstream_manifest),
        "yahoo_adapter": record(yahoo_adapter),
        "price_file": record(prices),
        "acquisition_script": record(acquisition_script),
        "support_files": {
            "download_audit": record(audit),
            "ticker_aliases": record(aliases),
            "ticker_exclusions": record(exclusions),
        },
        "acquisition": {
            "request_start": ACQUISITION_START,
            "request_end": ACQUISITION_END,
            "outcome_join_performed": False,
            "holdout_return_outcomes_joined": False,
            "design_receipt_verified_before_download": True,
            "fresh_network_retrieval_after_design_freeze": True,
            "candidate_classes": ["opportunistic", "routine"],
            "benchmark": "SPY",
            "ticker_regex": UNAMBIGUOUS_TICKER_PATTERN.pattern,
            "alias_rule": "replace_period_with_hyphen",
        },
        "ticker_aliases": alias_frame.to_dict("records"),
        "vendor_records": vendor_records,
        "limitations": {
            "permanent_security_identifiers_complete": False,
            "delisting_returns_complete": False,
            "survivorship_free": False,
            "downloaded_panel_rows": int(len(panel)),
            "retained_panel_rows": int(len(panel)),
            "removed_panel_rows": 0,
            "stage_a_candidate_audit": candidate_audit,
            "benchmark_clock_preflight": form4_stage_a.benchmark_clock_preflight(
                candidate_frame, panel
            ),
        },
    }
    manifest = root / CANONICAL_PRICE_MANIFEST
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest


def test_freeze_receipt_closes_ten_roles_and_detects_mutation(tmp_path: Path):
    root = tmp_path
    candidate = (
        root
        / "artifacts/form4_event_alpha_v1/"
        "candidate_preregistration_2020q1_2025q4_v2/"
        "candidate_accessions.parquet"
    )
    prices = root / CANONICAL_PRICE_PANEL
    core = root / CANONICAL_CORE_CODE
    runner = root / CANONICAL_OUTCOME_RUNNER
    acquisition = root / CANONICAL_ACQUISITION_SCRIPT
    design_receipt = root / CANONICAL_DESIGN_RECEIPT
    candidate_upstream_manifest = candidate.parent / "artifact_manifest.json"
    yahoo_adapter = root / "src/alpha_factor_engine/data/yahoo.py"
    human = root / CANONICAL_HUMAN_PROTOCOL
    freeze_receipt = root / CANONICAL_FREEZE_RECEIPT
    for path in (
        candidate,
        prices,
        core,
        runner,
        acquisition,
        design_receipt,
        yahoo_adapter,
        human,
        freeze_receipt,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    for path, content in (
        (core, b"core"),
        (runner, b"runner"),
        (acquisition, b"acquisition"),
        (yahoo_adapter, b"adapter"),
    ):
        path.write_bytes(content)
    _candidates(
        [("a", "2023-06-03", "AAA", True, "opportunistic")]
    ).to_parquet(candidate, index=False)
    candidate_upstream_manifest.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "path": str(candidate.resolve()),
                        "bytes": candidate.stat().st_size,
                        "sha256": sha256_file(candidate),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    human.write_bytes(
        Path("configs/form4_opportunistic_stage_a_v1.json").read_bytes()
    )
    human_payload = json.loads(human.read_text(encoding="utf-8"))
    human_payload["sources"]["candidate_accessions"].update(
        {
            "path": candidate.relative_to(root).as_posix(),
            "bytes": candidate.stat().st_size,
            "sha256": sha256_file(candidate),
            "upstream_manifest_path": candidate_upstream_manifest.relative_to(
                root
            ).as_posix(),
            "upstream_manifest_bytes": candidate_upstream_manifest.stat().st_size,
            "upstream_manifest_sha256": sha256_file(candidate_upstream_manifest),
        }
    )
    human_payload["sources"]["price_provider"].update(
        {
            "adapter_path": yahoo_adapter.relative_to(root).as_posix(),
            "adapter_sha256_at_protocol_authoring": sha256_file(yahoo_adapter),
        }
    )
    human.write_text(json.dumps(human_payload), encoding="utf-8")
    build_design_receipt(
        human_protocol_path=human,
        core_code_path=core,
        outcome_runner_path=runner,
        acquisition_script_path=acquisition,
        candidate_path=candidate,
        candidate_upstream_manifest_path=candidate_upstream_manifest,
        yahoo_adapter_path=yahoo_adapter,
        receipt_path=design_receipt,
        repository_root=root,
    )
    with pytest.raises(FreezeReceiptError, match="design receipt already exists"):
        build_design_receipt(
            human_protocol_path=human,
            core_code_path=core,
            outcome_runner_path=runner,
            acquisition_script_path=acquisition,
            candidate_path=candidate,
            candidate_upstream_manifest_path=candidate_upstream_manifest,
            yahoo_adapter_path=yahoo_adapter,
            receipt_path=design_receipt,
            repository_root=root,
        )
    manifest = _write_price_manifest(
        root,
        candidate,
        candidate_upstream_manifest,
        yahoo_adapter,
        prices,
        acquisition,
        design_receipt,
    )
    config_payload = stage_a_config_template(
        human_protocol=CANONICAL_HUMAN_PROTOCOL,
        candidate_accessions=candidate.relative_to(root).as_posix(),
        candidate_upstream_manifest=candidate_upstream_manifest.relative_to(
            root
        ).as_posix(),
        yahoo_adapter=yahoo_adapter.relative_to(root).as_posix(),
        design_receipt=CANONICAL_DESIGN_RECEIPT,
        price_panel=CANONICAL_PRICE_PANEL,
        price_manifest=CANONICAL_PRICE_MANIFEST,
    )
    config = root / CANONICAL_RUNTIME_CONFIG
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(json.dumps(config_payload), encoding="utf-8")

    original_manifest = manifest.read_bytes()
    bad_manifest = json.loads(original_manifest)
    bad_manifest["ticker_aliases"] = [
        row
        for row in bad_manifest["ticker_aliases"]
        if row["source_ticker"] == "SPY"
    ]
    manifest.write_text(json.dumps(bad_manifest), encoding="utf-8")
    with pytest.raises(
        FreezeReceiptError, match="universe recomputed from frozen candidates"
    ):
        build_freeze_receipt(
            config_path=config,
            core_code_path=core,
            outcome_runner_path=runner,
            receipt_path=freeze_receipt,
            repository_root=root,
        )
    manifest.write_bytes(original_manifest)

    original_prices = prices.read_bytes()
    tampered_panel = pd.read_parquet(prices)
    tampered_panel.loc[tampered_panel.index[0], "adj_open"] *= 1.01
    tampered_panel.to_parquet(prices, index=False)
    bad_panel_manifest = json.loads(original_manifest)
    bad_panel_manifest["price_file"] = {
        "path": prices.relative_to(root).as_posix(),
        "bytes": prices.stat().st_size,
        "sha256": sha256_file(prices),
    }
    manifest.write_text(json.dumps(bad_panel_manifest), encoding="utf-8")
    with pytest.raises(
        FreezeReceiptError, match="independent raw Yahoo reconstruction"
    ):
        build_freeze_receipt(
            config_path=config,
            core_code_path=core,
            outcome_runner_path=runner,
            receipt_path=freeze_receipt,
            repository_root=root,
        )
    prices.write_bytes(original_prices)
    manifest.write_bytes(original_manifest)

    inconsistent_hash_manifest = json.loads(original_manifest)
    inconsistent_hash_manifest["vendor_records"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(inconsistent_hash_manifest), encoding="utf-8")
    with pytest.raises(
        FreezeReceiptError, match="vendor and archived raw hashes differ"
    ):
        build_freeze_receipt(
            config_path=config,
            core_code_path=core,
            outcome_runner_path=runner,
            receipt_path=freeze_receipt,
            repository_root=root,
        )
    manifest.write_bytes(original_manifest)

    receipt = build_freeze_receipt(
        config_path=config,
        core_code_path=core,
        outcome_runner_path=runner,
        receipt_path=freeze_receipt,
        repository_root=root,
    )
    with pytest.raises(FreezeReceiptError, match="freeze receipt already exists"):
        build_freeze_receipt(
            config_path=config,
            core_code_path=core,
            outcome_runner_path=runner,
            receipt_path=receipt,
            repository_root=root,
        )
    expected = {
        "config": config,
        "human_protocol": human,
        "core_code": core,
        "outcome_runner": runner,
        "candidate_accessions": candidate,
        "candidate_upstream_manifest": candidate_upstream_manifest,
        "yahoo_adapter": yahoo_adapter,
        "design_receipt": design_receipt,
        "price_panel": prices,
        "price_manifest": manifest,
    }
    verified = verify_freeze_receipt(receipt, expected_paths=expected, repository_root=root)
    assert (
        verified["candidate_issuer_price_or_return_join_before_freeze"] is False
    )
    assert {row["role"] for row in verified["records"]} == set(expected)
    prices.write_bytes(b"changed")
    with pytest.raises(FreezeReceiptError, match="frozen byte count mismatch"):
        verify_freeze_receipt(receipt, expected_paths=expected, repository_root=root)


def test_yahoo_adapter_and_raw_audit_reject_symbol_redirect():
    redirected = {
        "meta": {
            "symbol": "BBB",
            "exchangeTimezoneName": "America/New_York",
        },
        "timestamp": [],
        "indicators": {"quote": [{}], "adjclose": [{}]},
    }
    with pytest.raises(RuntimeError, match="differs from the requested security"):
        YahooFinanceClient._parse_result("AAA", redirected)

    raw = json.dumps(
        {"chart": {"error": None, "result": [redirected]}}
    ).encode()
    with pytest.raises(FreezeReceiptError, match="differs from requested ticker"):
        form4_stage_a._reconstruct_clean_price_panel_from_raw(
            {"AAA": raw},
            alias_pairs={("AAA", "AAA")},
            expected_vendor_rows={"AAA": 0},
        )


def test_benchmark_preflight_rejects_truncated_spy_before_freeze():
    candidates = _candidates(
        [("late", "2024-12-31", "AAA", True, "opportunistic")]
    )
    prices = _prices()
    truncated = prices.loc[
        ~prices["source_ticker"].eq("SPY")
        | prices["date"].le(pd.Timestamp("2024-12-31"))
    ].copy()
    with pytest.raises(Form4StageAError, match="fixed acquisition end"):
        form4_stage_a.benchmark_clock_preflight(candidates, truncated)


def test_alternate_run_directory_is_rejected_before_any_parquet_read(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(
        form4_stage_a, "__file__", str(tmp_path / CANONICAL_CORE_CODE)
    )

    def forbidden(*args, **kwargs):  # pragma: no cover
        raise AssertionError("no parquet read is permitted for an alternate run path")

    monkeypatch.setattr(pd, "read_parquet", forbidden)
    with pytest.raises(Form4StageAError, match="single canonical path"):
        run_stage_a_diagnostic(
            config_path=tmp_path / CANONICAL_RUNTIME_CONFIG,
            freeze_receipt_path=tmp_path / CANONICAL_FREEZE_RECEIPT,
            outcome_runner_path=tmp_path / CANONICAL_OUTCOME_RUNNER,
            repository_root=tmp_path,
            output_dir=tmp_path / "artifacts/form4_opportunistic_stage_a_v1/run2",
        )


def test_outcome_path_checks_receipt_before_any_parquet_read(tmp_path: Path, monkeypatch):
    human = tmp_path / CANONICAL_HUMAN_PROTOCOL
    human.parent.mkdir(parents=True, exist_ok=True)
    human.write_bytes(Path("configs/form4_opportunistic_stage_a_v1.json").read_bytes())
    config_payload = stage_a_config_template(
        human_protocol=CANONICAL_HUMAN_PROTOCOL,
        candidate_accessions="missing-candidates.parquet",
        candidate_upstream_manifest="missing-upstream.json",
        yahoo_adapter="missing-yahoo.py",
        design_receipt=CANONICAL_DESIGN_RECEIPT,
        price_panel=CANONICAL_PRICE_PANEL,
        price_manifest=CANONICAL_PRICE_MANIFEST,
    )
    config = tmp_path / CANONICAL_RUNTIME_CONFIG
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(json.dumps(config_payload), encoding="utf-8")
    monkeypatch.setattr(
        form4_stage_a, "__file__", str(tmp_path / CANONICAL_CORE_CODE)
    )

    def forbidden(*args, **kwargs):  # pragma: no cover - called only on failure
        raise AssertionError("outcome-bearing parquet was opened before receipt validation")

    monkeypatch.setattr(pd, "read_parquet", forbidden)
    with pytest.raises(FreezeReceiptError, match="cannot read required input"):
        run_stage_a_diagnostic(
            config_path=config,
            freeze_receipt_path=tmp_path / CANONICAL_FREEZE_RECEIPT,
            outcome_runner_path=tmp_path / CANONICAL_OUTCOME_RUNNER,
            repository_root=tmp_path,
            output_dir=tmp_path / CANONICAL_RUN_DIRECTORY,
        )


def test_existing_output_directory_blocks_before_receipt_or_outcome_read(
    tmp_path: Path, monkeypatch
):
    config_payload = stage_a_config_template(
        human_protocol=CANONICAL_HUMAN_PROTOCOL,
        candidate_accessions="missing-candidates.parquet",
        candidate_upstream_manifest="missing-upstream.json",
        yahoo_adapter="missing-yahoo.py",
        design_receipt=CANONICAL_DESIGN_RECEIPT,
        price_panel=CANONICAL_PRICE_PANEL,
        price_manifest=CANONICAL_PRICE_MANIFEST,
    )
    config = tmp_path / CANONICAL_RUNTIME_CONFIG
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(json.dumps(config_payload), encoding="utf-8")
    output = tmp_path / CANONICAL_RUN_DIRECTORY
    output.mkdir(parents=True)
    monkeypatch.setattr(
        form4_stage_a, "__file__", str(tmp_path / CANONICAL_CORE_CODE)
    )

    def forbidden(*args, **kwargs):  # pragma: no cover
        raise AssertionError("no parquet read is permitted for an existing output")

    monkeypatch.setattr(pd, "read_parquet", forbidden)
    with pytest.raises(Form4StageAError, match="output directory already exists"):
        run_stage_a_diagnostic(
            config_path=config,
            freeze_receipt_path=tmp_path / CANONICAL_FREEZE_RECEIPT,
            outcome_runner_path=tmp_path / CANONICAL_OUTCOME_RUNNER,
            repository_root=tmp_path,
            output_dir=output,
        )
