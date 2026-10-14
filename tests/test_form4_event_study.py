import hashlib

import numpy as np
import pandas as pd
import pytest

from alpha_factor_engine.contracts import PointInTimeViolation

from alpha_factor_engine.research.form4_event_study import (
    DIAGNOSTIC_ONLY,
    Form4EventStudyError,
    RESEARCH_CANDIDATE,
    build_prior_close_eligibility_snapshot,
    canonical_frame_sha256,
    compute_open_to_open_returns,
    compute_pre_event_placebo,
    exchange_calendar_sha256,
    match_liquidity_risk_controls,
    suppress_causal_overlaps,
    validate_event_entry_clocks,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _april_exchange_opens():
    return pd.to_datetime(
        [
            "2025-04-09T13:30:00Z",
            "2025-04-10T13:30:00Z",
            "2025-04-11T13:30:00Z",
            "2025-04-14T13:30:00Z",
            "2025-04-15T13:30:00Z",
        ],
        utc=True,
    )


def _open_prices():
    sessions = pd.to_datetime(
        [
            "2025-01-02",
            "2025-01-03",
            "2025-01-06",
            "2025-01-07",
            "2025-01-08",
            "2025-01-09",
            "2025-01-10",
        ]
    )
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "ticker": "EVT",
                    "security_id": "perm-1",
                    "date": sessions,
                    "open": np.arange(10.0, 17.0),
                }
            ),
            pd.DataFrame(
                {
                    "ticker": "SPY",
                    "security_id": "perm-spy",
                    "date": sessions,
                    "open": np.arange(100.0, 107.0),
                }
            ),
        ],
        ignore_index=True,
    )


def _exchange_opens():
    return pd.to_datetime(
        [
            "2025-01-02T14:30:00Z",
            "2025-01-03T14:30:00Z",
            "2025-01-06T14:30:00Z",
            "2025-01-07T14:30:00Z",
            "2025-01-08T14:30:00Z",
            "2025-01-09T14:30:00Z",
            "2025-01-10T14:30:00Z",
        ],
        utc=True,
    )


def _long_exchange_opens():
    local_midnights = pd.bdate_range("2025-01-02", periods=60)
    local_opens = (local_midnights + pd.Timedelta(hours=9, minutes=30)).tz_localize(
        "America/New_York"
    )
    return local_opens.tz_convert("UTC")


def test_prior_close_snapshot_excludes_entry_session_and_labels_missing_id():
    decisions = pd.DataFrame(
        {
            "event_id": ["e1"],
            "ticker": ["evt"],
            "information_at": ["2025-04-14T21:00:00Z"],
            "entry_at": ["2025-04-15T13:30:00Z"],
            "security_master_known_at": ["2020-01-01T00:00:00Z"],
            "security_master_effective_at": ["2020-01-01T00:00:00Z"],
        }
    )
    prices = pd.DataFrame(
        {
            "ticker": ["EVT"] * 4,
            "date": ["2025-04-10", "2025-04-11", "2025-04-14", "2025-04-15"],
            "close": [10.0, 11.0, 12.0, 1000.0],
            "volume": [100.0, 200.0, 300.0, 1_000_000.0],
            "available_at": [
                "2025-04-10T20:05:00Z",
                "2025-04-11T20:05:00Z",
                "2025-04-14T20:05:00Z",
                "2025-04-15T20:05:00Z",
            ],
        }
    )

    result = build_prior_close_eligibility_snapshot(
        decisions,
        prices,
        exchange_sessions=_april_exchange_opens(),
        calendar_authoritative=True,
        calendar_provenance_sha256=exchange_calendar_sha256(
            _april_exchange_opens()
        ),
        adv_window=2,
        volatility_window=2,
        minimum_price=5.0,
        minimum_adv_dollars=1_000.0,
        price_available_column="available_at",
    )

    assert result.loc[0, "prior_close"] == 12.0
    assert result.loc[0, "adv_dollars"] == (11.0 * 200.0 + 12.0 * 300.0) / 2.0
    assert result.loc[0, "feature_max_session"] == pd.Timestamp("2025-04-14").date()
    assert result.loc[0, "feature_available_at"] < result.loc[0, "entry_at"]
    assert result.loc[0, "feature_available_at"] <= result.loc[0, "information_at"]
    assert result.loc[0, "eligibility_status"] == "eligible"
    assert bool(result.loc[0, "eligible"])
    assert result.loc[0, "research_scope"] == DIAGNOSTIC_ONLY
    assert "missing_permanent_security_id" in result.loc[0, "scope_reasons"]
    assert "price_history_not_keyed_by_permanent_id" in result.loc[0, "scope_reasons"]


def test_prior_close_snapshot_with_permanent_id_is_research_candidate():
    decisions = pd.DataFrame(
        {
            "event_id": ["e1"],
            "security_id": ["perm-1"],
            "ticker": ["EVT"],
            "research_scope": [RESEARCH_CANDIDATE],
            "information_at": ["2025-04-14T21:00:00Z"],
            "entry_at": ["2025-04-15T13:30:00Z"],
            "security_master_known_at": ["2020-01-01T00:00:00Z"],
            "security_master_effective_at": ["2020-01-01T00:00:00Z"],
        }
    )
    prices = pd.DataFrame(
        {
            "ticker": ["EVT"] * 3,
            "security_id": ["perm-1"] * 3,
            "date": ["2025-04-10", "2025-04-11", "2025-04-14"],
            "close": [10.0, 11.0, 12.0],
            "volume": [100.0, 200.0, 300.0],
            "available_at": [
                "2025-04-10T20:05:00Z",
                "2025-04-11T20:05:00Z",
                "2025-04-14T20:05:00Z",
            ],
        }
    )

    corporate_action_evidence = b"corporate-action-snapshot-v1"
    delisting_evidence = b"delisting-snapshot-v1"
    result = build_prior_close_eligibility_snapshot(
        decisions,
        prices,
        exchange_sessions=_april_exchange_opens(),
        calendar_authoritative=True,
        calendar_provenance_sha256=exchange_calendar_sha256(
            _april_exchange_opens()
        ),
        adv_window=2,
        volatility_window=2,
        minimum_adv_dollars=0,
        price_available_column="available_at",
        price_basis="split_dividend_adjusted_ohlc",
        price_provenance_sha256=canonical_frame_sha256(prices),
        corporate_action_complete=True,
        corporate_action_provenance_sha256=_sha256(corporate_action_evidence),
        corporate_action_evidence=corporate_action_evidence,
        delisting_complete=True,
        delisting_provenance_sha256=_sha256(delisting_evidence),
        delisting_evidence=delisting_evidence,
        delisting_payoff_included=True,
        security_master_clock_complete=True,
    )

    assert result.loc[0, "research_scope"] == DIAGNOSTIC_ONLY
    assert "delisting_payoff_not_verified_in_return_ledger" in result.loc[
        0, "scope_reasons"
    ]
    unproven = build_prior_close_eligibility_snapshot(
        decisions,
        prices,
        exchange_sessions=_april_exchange_opens(),
        calendar_authoritative=True,
        calendar_provenance_sha256=exchange_calendar_sha256(
            _april_exchange_opens()
        ),
        adv_window=2,
        volatility_window=2,
        minimum_adv_dollars=0,
        price_available_column="available_at",
        price_basis="split_dividend_adjusted_ohlc",
        corporate_action_complete=True,
        delisting_complete=True,
        security_master_clock_complete=True,
    ).iloc[0]
    assert unproven["research_scope"] == DIAGNOSTIC_ONLY
    assert "corporate_action_history_not_proven_complete" in unproven["scope_reasons"]
    assert "delisting_payoff_not_accounted" in unproven["scope_reasons"]


def test_open_to_open_returns_use_exact_spy_sessions_and_apply_costs():
    event = pd.DataFrame(
        {
            "event_id": ["e1"],
            "security_id": ["perm-1"],
            "ticker": ["EVT"],
            "information_at": ["2025-01-02T21:00:00Z"],
            "entry_at": ["2025-01-03T14:30:00Z"],
            "clock_verified": [True],
            "eligible": [True],
            "selected_after_overlap": [True],
        }
    )

    result = compute_open_to_open_returns(
        event,
        _open_prices(),
        exchange_sessions=_exchange_opens(),
        calendar_authoritative=True,
        horizons=[2],
        one_way_cost_bps=15,
        price_basis="split_dividend_adjusted_ohlc",
        corporate_action_complete=True,
        delisting_complete=True,
    ).iloc[0]

    expected_gross = 13.0 / 11.0 - 1.0
    expected_benchmark = 103.0 / 101.0 - 1.0
    assert result["exit_session"] == pd.Timestamp("2025-01-07").date()
    assert np.isclose(result["event_gross_return"], expected_gross)
    assert np.isclose(
        result["event_net_return_after_event_cost_only"], expected_gross - 0.003
    )
    assert np.isclose(
        result["market_adjusted_return_gross"], expected_gross - expected_benchmark
    )
    assert result["cost_scope"] == "event_leg_only_diagnostic_no_hedge_execution_cost"
    assert not bool(result["execution_cost_complete"])
    assert result["research_scope"] == DIAGNOSTIC_ONLY
    assert "benchmark_is_nontraded_diagnostic_adjustment" in result["scope_reasons"]
    assert "delisting_payoff_not_accounted" in result["scope_reasons"]
    assert result["return_status"] == "ok"


def test_open_to_open_returns_do_not_compress_missing_security_bars():
    event = pd.DataFrame(
        {
            "event_id": ["e1"],
            "ticker": ["EVT"],
            "information_at": ["2025-01-02T21:00:00Z"],
            "entry_at": ["2025-01-03T14:30:00Z"],
            "clock_verified": [True],
            "eligible": [True],
            "selected_after_overlap": [True],
        }
    )
    prices = _open_prices()
    prices = prices.loc[
        ~(prices["ticker"].eq("EVT") & prices["date"].eq(pd.Timestamp("2025-01-07")))
    ]

    result = compute_open_to_open_returns(
        event,
        prices,
        exchange_sessions=_exchange_opens(),
        calendar_authoritative=True,
        horizons=[2],
    ).iloc[0]

    assert result["return_status"] == "missing_exact_exit_open"
    assert pd.isna(result["event_gross_return"])
    assert result["research_scope"] == DIAGNOSTIC_ONLY
    assert "missing_exact_exit_price" in result["scope_reasons"]


def test_pre_event_placebo_ends_before_true_entry():
    event = pd.DataFrame(
        {
            "event_id": ["e1"],
            "ticker": ["EVT"],
            "information_at": ["2025-01-08T21:00:00Z"],
            "entry_at": ["2025-01-09T14:30:00Z"],
            "clock_verified": [True],
        }
    )

    result = compute_pre_event_placebo(
        event,
        _open_prices(),
        exchange_sessions=_exchange_opens(),
        calendar_authoritative=True,
        horizons=[2],
        gap_sessions=1,
    ).iloc[0]

    assert result["placebo_start_session"] == pd.Timestamp("2025-01-06").date()
    assert result["placebo_end_session"] == pd.Timestamp("2025-01-08").date()
    assert result["placebo_end_session"] < result["true_entry_session"]
    assert result["placebo_end_open_at"] < pd.Timestamp(event.loc[0, "information_at"])
    assert np.isclose(result["placebo_return"], 14.0 / 12.0 - 1.0)
    assert result["analysis_role"] == "pre_event_placebo_only"


def test_causal_overlap_suppression_does_not_extend_from_suppressed_event():
    sessions = _long_exchange_opens()
    entries = [sessions[1], sessions[2], sessions[21]]
    events = pd.DataFrame(
        {
            "event_id": ["e1", "e2", "e3"],
            "issuer_cik": ["1", "1", "1"],
            "ticker": ["EVT", "EVT", "EVT"],
            "information_at": [value - pd.Timedelta(hours=1) for value in entries],
            "entry_at": entries,
            "exit_at": [value + pd.Timedelta(minutes=1) for value in entries],
        }
    )
    calendar_digest = exchange_calendar_sha256(sessions)
    horizon_artifact = pd.DataFrame(
        {
            "event_id": events["event_id"],
            "horizon_sessions": [20, 20, 20],
            "entry_at": events["entry_at"],
            "exit_open_at": [sessions[21], sessions[22], sessions[41]],
            "return_basis": ["exact_independent_exchange_session_open_to_open"] * 3,
            "calendar_authoritative": [True] * 3,
            "calendar_provenance_sha256": [calendar_digest] * 3,
        }
    )
    result = suppress_causal_overlaps(
        events,
        exchange_sessions=sessions,
        calendar_authoritative=True,
        calendar_provenance_sha256=calendar_digest,
        horizon_artifact=horizon_artifact,
        horizon_artifact_sha256=canonical_frame_sha256(horizon_artifact),
    ).set_index("event_id")

    assert result.loc["e1", "overlap_status"] == "kept"
    assert result.loc["e2", "overlap_status"] == "suppressed_active_position"
    assert result.loc["e2", "suppressed_by_event_id"] == "e1"
    assert result.loc["e3", "overlap_status"] == "kept"
    assert result.loc["e3", "research_scope"] == DIAGNOSTIC_ONLY
    assert result.loc["e3", "overlap_identity_basis"] == "issuer_cik"
    assert result.loc["e3", "issuer_cik"] == "0000000001"


def test_overlap_does_not_trust_entry_plus_one_minute_exit_without_horizon_artifact():
    sessions = _long_exchange_opens()
    entry = sessions[1]
    second_entry = sessions[5]
    events = pd.DataFrame(
        {
            "event_id": ["e1", "e2"],
            "security_id": ["PERM", "PERM"],
            "issuer_cik": ["1", "1"],
            "ticker": ["EVT", "EVT"],
            "information_at": [
                entry - pd.Timedelta(hours=1),
                second_entry - pd.Timedelta(hours=1),
            ],
            "entry_at": [entry, second_entry],
            # This caller-supplied value would falsely retain e2 for a 20-session hold.
            "exit_at": [
                entry + pd.Timedelta(minutes=1),
                second_entry + pd.Timedelta(minutes=1),
            ],
        }
    )

    result = suppress_causal_overlaps(events)

    assert result["overlap_status"].eq("blocked_unverified_horizon").all()
    assert not result["selected_after_overlap"].any()
    assert result["research_scope"].eq(DIAGNOSTIC_ONLY).all()

    calendar_digest = exchange_calendar_sha256(sessions)
    authoritative_horizon = pd.DataFrame(
        {
            "event_id": ["e1", "e2"],
            "horizon_sessions": [20, 20],
            "entry_at": events["entry_at"],
            "exit_open_at": [sessions[21], sessions[25]],
            "return_basis": ["exact_independent_exchange_session_open_to_open"] * 2,
            "calendar_authoritative": [True, True],
            "calendar_provenance_sha256": [calendar_digest, calendar_digest],
        }
    )
    artifact_result = suppress_causal_overlaps(
        events,
        exchange_sessions=sessions,
        calendar_authoritative=True,
        calendar_provenance_sha256=calendar_digest,
        horizon_artifact=authoritative_horizon,
        horizon_artifact_sha256=canonical_frame_sha256(authoritative_horizon),
    ).set_index("event_id")
    assert artifact_result.loc["e1", "overlap_status"] == "kept"
    assert artifact_result.loc["e2", "overlap_status"] == "suppressed_active_position"


def test_liquidity_risk_controls_match_nearest_same_sic_and_report_unmatched():
    entry = pd.Timestamp("2025-01-03T14:30:00Z")
    information = pd.Timestamp("2025-01-02T21:00:00Z")
    events = pd.DataFrame(
        {
            "event_id": ["e1", "e2"],
            "security_id": ["event-1", "event-2"],
            "issuer_cik": ["1", "9"],
            "ticker": ["EVT", "NONE"],
            "information_at": [information, information],
            "entry_at": [entry, entry],
            "feature_available_at": [information - pd.Timedelta(hours=1)] * 2,
            "eligible": [True, True],
            "clock_verified": [True, True],
            "sic2": [35, 99],
            "security_master_known_at": ["2020-01-01T00:00Z"] * 2,
            "security_master_effective_at": ["2020-01-01T00:00Z"] * 2,
            "sic_known_at": ["2020-01-01T00:00Z"] * 2,
            "sic_effective_at": ["2020-01-01T00:00Z"] * 2,
            "log_adv_dollars": [10.0, 10.0],
            "realized_volatility": [0.20, 0.20],
            "research_scope": [RESEARCH_CANDIDATE] * 2,
        }
    )
    candidates = pd.DataFrame(
        {
            "security_id": ["event-1", "near", "far"],
            "issuer_cik": ["1", "2", "3"],
            "ticker": ["EVT", "NEAR", "FAR"],
            "entry_at": [entry] * 3,
            "feature_available_at": [information - pd.Timedelta(hours=1)] * 3,
            "eligible": [True] * 3,
            "sic2": [35, 35, 35],
            "security_master_known_at": ["2020-01-01T00:00Z"] * 3,
            "security_master_effective_at": ["2020-01-01T00:00Z"] * 3,
            "sic_known_at": ["2020-01-01T00:00Z"] * 3,
            "sic_effective_at": ["2020-01-01T00:00Z"] * 3,
            "known_event_at": [pd.NaT, pd.NaT, pd.NaT],
            "log_adv_dollars": [10.0, 10.1, 20.0],
            "realized_volatility": [0.20, 0.21, 0.80],
            "research_scope": [RESEARCH_CANDIDATE] * 3,
        }
    )

    result = match_liquidity_risk_controls(
        events,
        candidates,
        controls_per_event=1,
        security_master_clock_complete=True,
        industry_clock_complete=True,
        known_event_clock_complete=True,
        known_event_provenance_sha256=canonical_frame_sha256(candidates),
        exchange_sessions=_exchange_opens(),
        calendar_authoritative=True,
        calendar_provenance_sha256=exchange_calendar_sha256(_exchange_opens()),
    ).set_index("event_id")

    assert result.loc["e1", "control_ticker"] == "NEAR"
    assert result.loc["e1", "match_status"] == "matched"
    assert result.loc["e1", "research_scope"] == DIAGNOSTIC_ONLY
    assert result.loc["e2", "match_status"] == "unmatched_no_same_sic_candidates"
    assert result.loc["e2", "matched_control_count"] == 0
    unproven = match_liquidity_risk_controls(
        events,
        candidates,
        controls_per_event=1,
        security_master_clock_complete=True,
        industry_clock_complete=True,
        known_event_clock_complete=True,
        exchange_sessions=_exchange_opens(),
        calendar_authoritative=True,
        calendar_provenance_sha256=exchange_calendar_sha256(_exchange_opens()),
    ).set_index("event_id")
    assert unproven.loc["e1", "research_scope"] == DIAGNOSTIC_ONLY
    assert "known_event_clock_not_proven_complete" in unproven.loc[
        "e1", "scope_reasons"
    ]


def test_controls_without_sic_or_permanent_ids_are_explicitly_diagnostic():
    entry = pd.Timestamp("2025-01-03T14:30:00Z")
    information = pd.Timestamp("2025-01-02T21:00:00Z")
    events = pd.DataFrame(
        {
            "event_id": ["e1"],
            "issuer_cik": ["1"],
            "ticker": ["EVT"],
            "information_at": [information],
            "entry_at": [entry],
            "feature_available_at": [information - pd.Timedelta(hours=1)],
            "eligible": [True],
            "log_adv_dollars": [10.0],
            "realized_volatility": [0.20],
            "research_scope": [RESEARCH_CANDIDATE],
        }
    )
    candidates = pd.DataFrame(
        {
            "issuer_cik": ["2", "3"],
            "ticker": ["NEAR", "FAR"],
            "entry_at": [entry, entry],
            "feature_available_at": [information - pd.Timedelta(hours=1)] * 2,
            "eligible": [True, True],
            "log_adv_dollars": [10.1, 20.0],
            "realized_volatility": [0.21, 0.80],
        }
    )

    result = match_liquidity_risk_controls(
        events,
        candidates,
        controls_per_event=1,
        industry_column=None,
        permanent_id_column=None,
    ).iloc[0]

    assert result["control_ticker"] == "NEAR"
    assert result["match_status"] == "matched"
    assert result["research_scope"] == DIAGNOSTIC_ONLY
    assert "missing_permanent_security_id" in result["scope_reasons"]
    assert "missing_same_sic_control_dimension" in result["scope_reasons"]
    assert "known_event_clock_not_proven_complete" in result["scope_reasons"]
    assert "security_master_clock_not_proven_complete" in result["scope_reasons"]
    assert result["industry_match_basis"] == "cross_industry_diagnostic_fallback"


def test_control_match_reports_post_information_features_instead_of_using_them():
    entry = pd.Timestamp("2025-01-03T14:30:00Z")
    information = pd.Timestamp("2025-01-02T20:00:00Z")
    events = pd.DataFrame(
        {
            "event_id": ["e1"],
            "issuer_cik": ["1"],
            "ticker": ["EVT"],
            "information_at": [information],
            "entry_at": [entry],
            "feature_available_at": [information + pd.Timedelta(minutes=1)],
            "eligible": [True],
            "log_adv_dollars": [10.0],
            "realized_volatility": [0.2],
        }
    )
    candidates = pd.DataFrame(
        {
            "issuer_cik": ["2"],
            "ticker": ["CTL"],
            "entry_at": [entry],
            "feature_available_at": [information - pd.Timedelta(hours=1)],
            "eligible": [True],
            "log_adv_dollars": [10.1],
            "realized_volatility": [0.21],
        }
    )

    result = match_liquidity_risk_controls(
        events,
        candidates,
        controls_per_event=1,
        industry_column=None,
        permanent_id_column=None,
    ).iloc[0]

    assert result["match_status"] == "unmatched_event_features_after_information"
    assert result["matched_control_count"] == 0


def test_control_match_independently_rejects_intraday_entry_despite_true_flag():
    information = pd.Timestamp("2025-01-02T21:00:00Z")
    illegal_entry = pd.Timestamp("2025-01-03T15:00:00Z")
    event = pd.DataFrame(
        {
            "event_id": ["e1"],
            "ticker": ["EVT"],
            "information_at": [information],
            "entry_at": [illegal_entry],
            "clock_verified": [True],
            "feature_available_at": [information - pd.Timedelta(hours=1)],
            "eligible": [True],
            "log_adv_dollars": [10.0],
            "realized_volatility": [0.20],
        }
    )
    candidates = pd.DataFrame(
        {
            "ticker": ["CTL"],
            "entry_at": [illegal_entry],
            "feature_available_at": [information - pd.Timedelta(hours=1)],
            "eligible": [True],
            "log_adv_dollars": [10.1],
            "realized_volatility": [0.21],
        }
    )

    with pytest.raises(PointInTimeViolation, match="09:30"):
        match_liquidity_risk_controls(
            event,
            candidates,
            controls_per_event=1,
            industry_column=None,
        )


def test_entry_clock_uses_first_dst_aware_regular_open_from_independent_calendar():
    sessions = pd.to_datetime(
        [
            "2025-03-07T14:30:00Z",
            "2025-03-10T13:30:00Z",
            "2025-03-11T13:30:00Z",
        ],
        utc=True,
    )
    event = pd.DataFrame(
        {
            "event_id": ["dst"],
            "information_at": ["2025-03-07T22:00:00Z"],
            "entry_at": ["2025-03-10T13:30:00Z"],
        }
    )

    verified = validate_event_entry_clocks(
        event,
        sessions,
        calendar_authoritative=True,
        calendar_provenance_sha256=exchange_calendar_sha256(sessions),
    )

    assert bool(verified.loc[0, "clock_verified"])
    assert verified.loc[0, "clock_status"] == "verified_authoritative_calendar"
    unhashed = validate_event_entry_clocks(
        event, sessions, calendar_authoritative=True
    )
    assert not bool(unhashed.loc[0, "clock_verified"])
    assert unhashed.loc[0, "research_scope"] == DIAGNOSTIC_ONLY
    wrong = event.copy()
    wrong["information_at"] = "2025-03-10T12:00:00Z"
    wrong["entry_at"] = "2025-03-11T13:30:00Z"
    with pytest.raises(PointInTimeViolation, match="first exchange open"):
        validate_event_entry_clocks(
            wrong,
            sessions,
            calendar_authoritative=True,
            calendar_provenance_sha256=exchange_calendar_sha256(sessions),
        )


def test_entry_clock_without_independent_calendar_cannot_be_verified():
    event = pd.DataFrame(
        {
            "event_id": ["e1"],
            "information_at": ["2025-01-02T21:00:00Z"],
            "entry_at": ["2025-01-03T14:30:00Z"],
        }
    )

    result = validate_event_entry_clocks(event, None)

    assert not bool(result.loc[0, "clock_verified"])
    assert result.loc[0, "research_scope"] == DIAGNOSTIC_ONLY
    assert result.loc[0, "clock_status"] == "missing_independent_exchange_calendar"


def test_eligibility_does_not_compress_missing_exchange_session_in_window():
    event = pd.DataFrame(
        {
            "event_id": ["e1"],
            "security_id": ["perm-1"],
            "ticker": ["EVT"],
            "information_at": ["2025-04-14T21:00:00Z"],
            "entry_at": ["2025-04-15T13:30:00Z"],
        }
    )
    prices = pd.DataFrame(
        {
            "security_id": ["PERM-1"] * 4,
            "ticker": ["EVT"] * 4,
            "date": ["2025-04-09", "2025-04-10", "2025-04-11", "2025-04-14"],
            "close": [9.0, 10.0, 1_000.0, 12.0],
            "volume": [100.0, 200.0, 1_000_000.0, 300.0],
            "available_at": [
                "2025-04-09T20:05:00Z",
                "2025-04-10T20:05:00Z",
                "2025-04-14T22:00:00Z",  # delayed past information_at
                "2025-04-14T20:05:00Z",
            ],
        }
    )

    result = build_prior_close_eligibility_snapshot(
        event,
        prices,
        exchange_sessions=_april_exchange_opens(),
        calendar_authoritative=True,
        calendar_provenance_sha256=exchange_calendar_sha256(
            _april_exchange_opens()
        ),
        price_available_column="available_at",
        adv_window=2,
        volatility_window=2,
        minimum_adv_dollars=0,
        price_basis="split_dividend_adjusted_ohlc",
        corporate_action_complete=True,
        delisting_complete=True,
    ).iloc[0]

    assert result["prior_close"] == 12.0
    assert result["eligibility_status"] == "missing_exact_exchange_session_coverage"
    assert not bool(result["eligible"])
    assert pd.isna(result["adv_dollars"])


def test_missing_whole_spy_session_is_not_compressed_out_of_horizon():
    event = pd.DataFrame(
        {
            "event_id": ["e1"],
            "security_id": ["perm-1"],
            "ticker": ["EVT"],
            "information_at": ["2025-01-02T21:00:00Z"],
            "entry_at": ["2025-01-03T14:30:00Z"],
            "clock_verified": [True],
            "eligible": [True],
            "selected_after_overlap": [True],
        }
    )
    prices = _open_prices()
    prices = prices.loc[
        ~(prices["ticker"].eq("SPY") & prices["date"].eq(pd.Timestamp("2025-01-06")))
    ]

    result = compute_open_to_open_returns(
        event,
        prices,
        exchange_sessions=_exchange_opens(),
        calendar_authoritative=True,
        horizons=[1],
    ).iloc[0]

    assert result["exit_session"] == pd.Timestamp("2025-01-06").date()
    assert result["return_status"] == "missing_benchmark_exit_open"


def test_returns_default_gates_block_unverified_events():
    event = pd.DataFrame(
        {
            "event_id": ["e1"],
            "ticker": ["EVT"],
            "information_at": ["2025-01-02T21:00:00Z"],
            "entry_at": ["2025-01-03T14:30:00Z"],
        }
    )

    result = compute_open_to_open_returns(
        event,
        _open_prices(),
        exchange_sessions=_exchange_opens(),
        calendar_authoritative=True,
        horizons=[1],
    ).iloc[0]

    assert result["return_status"] == "blocked_clock_not_verified"
    assert pd.isna(result["event_gross_return"])


def test_control_exclusion_canonicalizes_security_id_and_cik():
    entry = pd.Timestamp("2025-01-03T14:30:00Z")
    information = pd.Timestamp("2025-01-02T21:00:00Z")
    event = pd.DataFrame(
        {
            "event_id": ["e1"],
            "security_id": [" sec-a "],
            "issuer_cik": ["1"],
            "ticker": ["EVT"],
            "information_at": [information],
            "entry_at": [entry],
            "feature_available_at": [information - pd.Timedelta(hours=1)],
            "eligible": [True],
            "log_adv_dollars": [10.0],
            "realized_volatility": [0.20],
            "research_scope": [RESEARCH_CANDIDATE],
        }
    )
    candidates = pd.DataFrame(
        {
            "security_id": ["SEC-A", "sec-b", "sec-c"],
            "issuer_cik": ["2", "0000000001", "3"],
            "ticker": ["SAME_SECURITY", "SAME_ISSUER", "VALID"],
            "entry_at": [entry] * 3,
            "feature_available_at": [information - pd.Timedelta(hours=1)] * 3,
            "eligible": [True] * 3,
            "log_adv_dollars": [10.0, 10.0, 11.0],
            "realized_volatility": [0.20, 0.20, 0.30],
        }
    )

    result = match_liquidity_risk_controls(
        event,
        candidates,
        controls_per_event=1,
        industry_column=None,
    ).iloc[0]

    assert result["control_ticker"] == "VALID"
    assert result["control_permanent_id"] == "SEC-C"
    assert result["control_issuer_cik"] == "0000000003"


def test_control_candidate_uniqueness_is_checked_by_canonical_permanent_id():
    entry = pd.Timestamp("2025-01-03T14:30:00Z")
    information = pd.Timestamp("2025-01-02T21:00:00Z")
    event = pd.DataFrame(
        {
            "event_id": ["e1"],
            "ticker": ["EVT"],
            "information_at": [information],
            "entry_at": [entry],
            "feature_available_at": [information - pd.Timedelta(hours=1)],
            "eligible": [True],
            "log_adv_dollars": [10.0],
            "realized_volatility": [0.2],
        }
    )
    candidates = pd.DataFrame(
        {
            "security_id": [" duplicate ", "DUPLICATE"],
            "ticker": ["A", "B"],
            "entry_at": [entry, entry],
            "feature_available_at": [information - pd.Timedelta(hours=1)] * 2,
            "eligible": [True, True],
            "log_adv_dollars": [10.1, 10.2],
            "realized_volatility": [0.21, 0.22],
        }
    )

    with pytest.raises(ValueError, match="permanent security id"):
        match_liquidity_risk_controls(
            event, candidates, controls_per_event=1, industry_column=None
        )


def test_selected_diagnostic_control_propagates_scope():
    entry = pd.Timestamp("2025-01-03T14:30:00Z")
    information = pd.Timestamp("2025-01-02T21:00:00Z")
    clock = "2020-01-01T00:00:00Z"
    event = pd.DataFrame(
        {
            "event_id": ["e1"],
            "security_id": ["EVENT"],
            "issuer_cik": ["1"],
            "ticker": ["EVT"],
            "information_at": [information],
            "entry_at": [entry],
            "feature_available_at": [information - pd.Timedelta(hours=1)],
            "eligible": [True],
            "sic2": [35],
            "clock_verified": [True],
            "security_master_known_at": [clock],
            "security_master_effective_at": [clock],
            "sic_known_at": [clock],
            "sic_effective_at": [clock],
            "log_adv_dollars": [10.0],
            "realized_volatility": [0.20],
            "research_scope": [RESEARCH_CANDIDATE],
        }
    )
    candidates = pd.DataFrame(
        {
            "security_id": ["CONTROL"],
            "issuer_cik": ["2"],
            "ticker": ["CTL"],
            "entry_at": [entry],
            "feature_available_at": [information - pd.Timedelta(hours=1)],
            "eligible": [True],
            "sic2": [35],
            "security_master_known_at": [clock],
            "security_master_effective_at": [clock],
            "sic_known_at": [clock],
            "sic_effective_at": [clock],
            "known_event_at": [pd.NaT],
            "research_scope": [DIAGNOSTIC_ONLY],
            "scope_reasons": ["current_survivor_panel"],
            "log_adv_dollars": [10.1],
            "realized_volatility": [0.21],
        }
    )

    result = match_liquidity_risk_controls(
        event,
        candidates,
        controls_per_event=1,
        security_master_clock_complete=True,
        industry_clock_complete=True,
        known_event_clock_complete=True,
        known_event_provenance_sha256=canonical_frame_sha256(candidates),
        exchange_sessions=_exchange_opens(),
        calendar_authoritative=True,
        calendar_provenance_sha256=exchange_calendar_sha256(_exchange_opens()),
    ).iloc[0]

    assert result["research_scope"] == DIAGNOSTIC_ONLY
    assert result["selected_control_research_scope"] == DIAGNOSTIC_ONLY
    assert "selected_control_diagnostic_only" in result["scope_reasons"]
    assert "current_survivor_panel" in result["scope_reasons"]


def test_future_security_master_candidate_has_explicit_unmatched_status():
    entry = pd.Timestamp("2025-01-03T14:30:00Z")
    information = pd.Timestamp("2025-01-02T21:00:00Z")
    old_clock = "2020-01-01T00:00:00Z"
    event = pd.DataFrame(
        {
            "event_id": ["e1"],
            "security_id": ["EVENT"],
            "issuer_cik": ["1"],
            "ticker": ["EVT"],
            "information_at": [information],
            "entry_at": [entry],
            "clock_verified": [True],
            "feature_available_at": [information - pd.Timedelta(hours=1)],
            "eligible": [True],
            "sic2": [35],
            "security_master_known_at": [old_clock],
            "security_master_effective_at": [old_clock],
            "sic_known_at": [old_clock],
            "sic_effective_at": [old_clock],
            "log_adv_dollars": [10.0],
            "realized_volatility": [0.20],
            "research_scope": [RESEARCH_CANDIDATE],
        }
    )
    candidates = pd.DataFrame(
        {
            "security_id": ["CONTROL"],
            "issuer_cik": ["2"],
            "ticker": ["CTL"],
            "entry_at": [entry],
            "feature_available_at": [information - pd.Timedelta(hours=1)],
            "eligible": [True],
            "sic2": [35],
            "security_master_known_at": [information + pd.Timedelta(minutes=1)],
            "security_master_effective_at": [old_clock],
            "sic_known_at": [old_clock],
            "sic_effective_at": [old_clock],
            "known_event_at": [pd.NaT],
            "log_adv_dollars": [10.1],
            "realized_volatility": [0.21],
            "research_scope": [RESEARCH_CANDIDATE],
        }
    )

    result = match_liquidity_risk_controls(
        event,
        candidates,
        controls_per_event=1,
        security_master_clock_complete=True,
        industry_clock_complete=True,
        known_event_clock_complete=True,
        known_event_provenance_sha256=canonical_frame_sha256(candidates),
        exchange_sessions=_exchange_opens(),
        calendar_authoritative=True,
        calendar_provenance_sha256=exchange_calendar_sha256(_exchange_opens()),
    ).iloc[0]

    assert result["match_status"] == "unmatched_no_pit_security_master_candidates"
    assert result["matched_control_count"] == 0


def test_calendar_digest_is_recomputed_and_fake_hash_fails_closed():
    sessions = _exchange_opens()
    event = pd.DataFrame(
        {
            "event_id": ["e1"],
            "information_at": ["2025-01-02T21:00:00Z"],
            "entry_at": ["2025-01-03T14:30:00Z"],
        }
    )

    with pytest.raises(Form4EventStudyError, match="calendar snapshot digest mismatch"):
        validate_event_entry_clocks(
            event,
            sessions,
            calendar_authoritative=True,
            calendar_provenance_sha256=HASH_A,
        )


def test_horizon_artifact_digest_and_exact_session_exit_are_recomputed():
    sessions = _long_exchange_opens()
    calendar_digest = exchange_calendar_sha256(sessions)
    event = pd.DataFrame(
        {
            "event_id": ["e1"],
            "security_id": ["SEC-1"],
            "issuer_cik": ["1"],
            "ticker": ["EVT"],
            "information_at": [sessions[1] - pd.Timedelta(hours=1)],
            "entry_at": [sessions[1]],
        }
    )
    artifact = pd.DataFrame(
        {
            "event_id": ["e1"],
            "horizon_sessions": [20],
            "entry_at": [sessions[1]],
            "exit_open_at": [sessions[21]],
            "return_basis": ["exact_independent_exchange_session_open_to_open"],
            "calendar_authoritative": [True],
            "calendar_provenance_sha256": [calendar_digest],
        }
    )
    declared = canonical_frame_sha256(artifact)
    tampered = artifact.copy()
    tampered.loc[0, "exit_open_at"] = sessions[21] + pd.Timedelta(minutes=1)

    with pytest.raises(Form4EventStudyError, match="artifact digest mismatch"):
        suppress_causal_overlaps(
            event,
            exchange_sessions=sessions,
            calendar_authoritative=True,
            calendar_provenance_sha256=calendar_digest,
            horizon_artifact=tampered,
            horizon_artifact_sha256=declared,
        )
    with pytest.raises(Form4EventStudyError, match="artifact digest mismatch"):
        suppress_causal_overlaps(
            event,
            exchange_sessions=sessions,
            calendar_authoritative=True,
            calendar_provenance_sha256=calendar_digest,
            horizon_artifact=artifact,
            horizon_artifact_sha256=HASH_B,
        )
    with pytest.raises(PointInTimeViolation, match="exactly entry_at"):
        suppress_causal_overlaps(
            event,
            exchange_sessions=sessions,
            calendar_authoritative=True,
            calendar_provenance_sha256=calendar_digest,
            horizon_artifact=tampered,
            horizon_artifact_sha256=canonical_frame_sha256(tampered),
        )


def test_price_basis_is_closed_enum_and_boolean_evidence_cannot_promote():
    event = pd.DataFrame(
        {
            "event_id": ["e1"],
            "ticker": ["EVT"],
            "information_at": ["2025-01-02T21:00:00Z"],
            "entry_at": ["2025-01-03T14:30:00Z"],
            "clock_verified": [True],
            "eligible": [True],
            "selected_after_overlap": [True],
        }
    )
    with pytest.raises(Form4EventStudyError, match="price_basis must be one of"):
        compute_open_to_open_returns(
            event,
            _open_prices(),
            exchange_sessions=_exchange_opens(),
            horizons=[2],
            price_basis="BANANA",
        )

    result = compute_open_to_open_returns(
        event,
        _open_prices(),
        exchange_sessions=_exchange_opens(),
        calendar_authoritative=True,
        calendar_provenance_sha256=exchange_calendar_sha256(_exchange_opens()),
        horizons=[2],
        price_basis="split_dividend_adjusted_ohlc",
        corporate_action_complete=True,
        delisting_complete=True,
        delisting_payoff_included=True,
    ).iloc[0]
    assert result["research_scope"] == DIAGNOSTIC_ONLY
    assert not bool(result["corporate_action_complete"])
    assert not bool(result["delisting_complete"])
    assert not bool(result["delisting_payoff_included"])
    assert "delisting_payoff_not_accounted" in result["scope_reasons"]


def test_verified_delisting_payoff_is_actually_entered_in_return():
    event = pd.DataFrame(
        {
            "event_id": ["e1"],
            "security_id": ["perm-1"],
            "ticker": ["EVT"],
            "information_at": ["2025-01-02T21:00:00Z"],
            "entry_at": ["2025-01-03T14:30:00Z"],
            "clock_verified": [True],
            "eligible": [True],
            "selected_after_overlap": [True],
        }
    )
    prices = _open_prices()
    prices = prices.loc[
        ~(prices["ticker"].eq("EVT") & prices["date"].eq(pd.Timestamp("2025-01-07")))
    ].copy()
    payoff_ledger = pd.DataFrame(
        {
            "event_id": ["e1"],
            "horizon_sessions": [2],
            "event_total_return_including_delisting": [-0.75],
        }
    )
    delisting_evidence = b"delisting-source-and-method-v1"

    result = compute_open_to_open_returns(
        event,
        prices,
        exchange_sessions=_exchange_opens(),
        calendar_authoritative=True,
        calendar_provenance_sha256=exchange_calendar_sha256(_exchange_opens()),
        horizons=[2],
        price_basis="split_dividend_adjusted_ohlc",
        price_provenance_sha256=canonical_frame_sha256(prices),
        delisting_complete=True,
        delisting_evidence=delisting_evidence,
        delisting_provenance_sha256=_sha256(delisting_evidence),
        delisting_payoff_included=True,
        delisting_payoffs=payoff_ledger,
        delisting_payoff_provenance_sha256=canonical_frame_sha256(payoff_ledger),
    ).iloc[0]

    assert result["return_status"] == "ok_delisting_payoff"
    assert bool(result["delisting_payoff_applied"])
    assert result["event_gross_return"] == -0.75
    assert result["research_scope"] == DIAGNOSTIC_ONLY
