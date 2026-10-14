from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_factor_engine.research.feature_family_diagnostic import (
    FeatureDefinition,
    apply_primary_screen,
    benjamini_hochberg,
    build_oriented_scores,
    compute_monthly_diagnostics,
    deterministic_deciles,
    feature_redundancy_summary,
    moving_block_bootstrap_mean,
    validate_source_panel,
)


def _definition(name: str, direction: int = 1, family: str = "test") -> FeatureDefinition:
    return FeatureDefinition(
        feature_id=name,
        family=family,
        kind="individual",
        components=((name, direction),),
        minimum_components=1,
        direction_status="opened_literature_signed_frozen_before_this_runner",
    )


def _validation_panel(rows: int = 4) -> pd.DataFrame:
    signal = pd.Timestamp("2024-01-31 22:00:00", tz="UTC")
    entry = pd.Timestamp("2024-02-01 14:30:00", tz="UTC")
    exit_at = pd.Timestamp("2024-03-01 14:30:00", tz="UTC")
    entry_price = np.linspace(10.0, 13.0, rows)
    exit_price = entry_price * np.linspace(0.95, 1.08, rows)
    return pd.DataFrame(
        {
            "ticker": [f"T{number:03d}" for number in range(rows)],
            "signal_date": [signal] * rows,
            "entry_at": [entry] * rows,
            "exit_at": [exit_at] * rows,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "target_return": exit_price / entry_price - 1.0,
            "eligible": [True] * rows,
            "price_available_at": [signal - pd.Timedelta(minutes=30)] * rows,
            "latest_input_available_at": [signal - pd.Timedelta(days=20)] * rows,
            "fundamental_available_at": [signal - pd.Timedelta(days=10)] * rows,
            "fundamental_fiscal_end": [signal - pd.Timedelta(days=100)] * rows,
            "fundamental_age_days": [10.0] * rows,
            "debt_definition": ["reported_long_term_debt"] * rows,
            "exact_total_momentum_12_1": np.linspace(-0.2, 0.3, rows),
            "roe": np.linspace(-0.1, 0.2, rows),
        }
    )


def test_feature_definition_rejects_invalid_direction() -> None:
    with pytest.raises(ValueError, match="directions"):
        _definition("alpha", direction=0)


def test_validate_source_panel_reconciles_clock_and_prices() -> None:
    panel = _validation_panel()
    validated, quality = validate_source_panel(
        panel,
        (_definition("exact_total_momentum_12_1"), _definition("roe")),
        expected_rows=4,
        expected_months=1,
        expected_securities=4,
    )
    assert len(validated) == 4
    assert quality["target_return_reconstruction_max_abs_error"] == 0.0
    assert quality["partial_label_months"] == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda frame: frame.assign(
                price_available_at=frame["signal_date"] + pd.Timedelta(seconds=1)
            ),
            "price feature availability",
        ),
        (
            lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True),
            "duplicate decision/security",
        ),
        (
            lambda frame: frame.assign(
                target_return=frame["target_return"].mask(frame.index == 0)
            ),
            "incomplete or non-finite labels",
        ),
        (
            lambda frame: frame.assign(
                target_return=frame["target_return"] + 0.01
            ),
            "does not reconcile",
        ),
        (
            lambda frame: frame.assign(
                fundamental_age_days=frame["fundamental_age_days"] + 1
            ),
            "fundamental age does not reconcile",
        ),
    ],
)
def test_validate_source_panel_fails_closed(mutation, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_source_panel(
            mutation(_validation_panel()),
            (_definition("exact_total_momentum_12_1"), _definition("roe")),
        )


def test_signal_constructor_refuses_a_future_label() -> None:
    frame = _validation_panel()[["ticker", "roe", "target_return"]]
    with pytest.raises(ValueError, match="must not receive target_return"):
        build_oriented_scores(frame, (_definition("roe"),))


def test_composite_uses_only_available_signal_components() -> None:
    definition = FeatureDefinition(
        feature_id="quality",
        family="fundamental",
        kind="composite",
        components=(("roe", 1), ("accruals", -1)),
        minimum_components=2,
        direction_status="opened_literature_signed_frozen_before_this_runner",
    )
    signal = pd.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "roe": [1.0, 2.0, 3.0],
            "accruals": [0.3, np.nan, 0.1],
        }
    )
    result = build_oriented_scores(signal, (definition,)).set_index("ticker")
    assert np.isnan(result.loc["B", "oriented_score"])
    assert result.loc["C", "oriented_score"] > result.loc["A", "oriented_score"]


def test_deterministic_deciles_use_security_tie_breaker() -> None:
    scores = pd.Series([1.0] * 20, index=np.arange(20))
    securities = pd.Series([f"T{number:02d}" for number in range(19, -1, -1)])
    first = deterministic_deciles(scores, securities, bins=10)
    second = deterministic_deciles(scores.sample(frac=1.0, random_state=7), securities, bins=10)
    by_security_first = pd.DataFrame({"ticker": securities, "decile": first}).set_index("ticker")
    by_security_second = pd.DataFrame(
        {"ticker": securities, "decile": second}
    ).set_index("ticker")
    pd.testing.assert_series_equal(
        by_security_first["decile"].sort_index(),
        by_security_second["decile"].sort_index(),
    )


def test_monthly_diagnostic_finds_increment_after_momentum_residualization() -> None:
    rng = np.random.default_rng(929)
    names = 200
    momentum = np.linspace(-1.0, 1.0, names)
    independent = rng.normal(size=names)
    signal = momentum + independent
    target = 0.6 * momentum + 0.8 * independent + rng.normal(scale=0.15, size=names)
    panel = pd.DataFrame(
        {
            "ticker": [f"T{number:03d}" for number in range(names)],
            "signal_date": [pd.Timestamp("2024-01-31", tz="UTC")] * names,
            "exact_total_momentum_12_1": momentum,
            "test_signal": signal,
            "target_return": target,
        }
    )
    monthly, _, deciles = compute_monthly_diagnostics(
        panel,
        (_definition("exact_total_momentum_12_1"), _definition("test_signal")),
        minimum_cross_section=100,
    )
    test_row = monthly.loc[monthly["feature_id"].eq("test_signal")].iloc[0]
    momentum_row = monthly.loc[
        monthly["feature_id"].eq("exact_total_momentum_12_1")
    ].iloc[0]
    assert test_row["momentum_orthogonal_rank_ic"] > 0.7
    assert np.isnan(momentum_row["momentum_orthogonal_rank_ic"])
    assert set(deciles.loc[deciles["feature_id"].eq("test_signal"), "score_basis"]) == {
        "oriented_raw",
        "momentum_orthogonal",
    }


def test_block_bootstrap_is_deterministic_and_requires_two_blocks() -> None:
    values = pd.Series(np.linspace(-0.03, 0.06, 36))
    first = moving_block_bootstrap_mean(
        values, block_length=6, simulations=300, seed=929
    )
    second = moving_block_bootstrap_mean(
        values, block_length=6, simulations=300, seed=929
    )
    assert first == second
    insufficient = moving_block_bootstrap_mean(
        values.iloc[:5], block_length=3, simulations=300, seed=929
    )
    assert np.isnan(insufficient["block_one_sided_pvalue"])


def test_benjamini_hochberg_is_monotone_in_sorted_order() -> None:
    pvalues = pd.Series([0.04, 0.001, 0.03, np.nan])
    adjusted = benjamini_hochberg(pvalues)
    assert adjusted.iloc[1] == pytest.approx(0.003)
    assert adjusted.iloc[2] <= adjusted.iloc[0]
    assert np.isnan(adjusted.iloc[3])


def test_primary_screen_never_authorizes_a_portfolio() -> None:
    summary = pd.DataFrame(
        {
            "period": ["pre_2026_primary"],
            "feature_id": ["test_signal"],
            "family": ["test"],
            "definition_kind": ["individual"],
            "direction_status": ["opened"],
            "minimum_monthly_coverage": [1.0],
            "mean_oriented_rank_ic": [0.10],
            "oriented_rank_ic_hac_tstat": [5.0],
            "oriented_rank_ic_hac_one_sided_pvalue": [1e-7],
            "oriented_rank_ic_block_one_sided_pvalue": [0.001],
            "mean_oriented_winsorized_decile_spread": [0.02],
            "mean_momentum_orthogonal_rank_ic": [0.09],
            "momentum_orthogonal_rank_ic_hac_tstat": [4.0],
            "momentum_orthogonal_rank_ic_hac_one_sided_pvalue": [1e-6],
            "momentum_orthogonal_rank_ic_block_one_sided_pvalue": [0.002],
            "mean_momentum_orthogonal_winsorized_decile_spread": [0.015],
        }
    )
    screen, _ = apply_primary_screen(
        summary,
        primary_period="pre_2026_primary",
        momentum_feature="exact_total_momentum_12_1",
        gates={
            "minimum_monthly_coverage": 0.3,
            "minimum_mean_oriented_ic": 0.0,
            "minimum_hac_tstat": 2.0,
            "maximum_month_block_pvalue": 0.05,
            "maximum_bh_qvalue": 0.05,
            "maximum_family_bonferroni_pvalue": 0.05,
            "minimum_winsorized_decile_spread": 0.0,
        },
        family_size_minimum=105,
    )
    assert bool(screen.loc[0, "statistical_screen_pass"])
    assert screen.loc[0, "research_verdict"] == "NOT VALIDATED"
    assert not bool(screen.loc[0, "portfolio_authorized"])


def test_redundancy_summary_detects_duplicate_rank_feature() -> None:
    names = 120
    values = np.linspace(-1.0, 1.0, names)
    panel = pd.DataFrame(
        {
            "ticker": [f"T{number:03d}" for number in range(names)],
            "signal_date": [pd.Timestamp("2024-01-31", tz="UTC")] * names,
            "left": values,
            "right": values + 7.0,
        }
    )
    result = feature_redundancy_summary(
        panel, (_definition("left"), _definition("right")), minimum_pair_names=100
    )
    assert result.loc[0, "mean_monthly_rank_correlation"] == pytest.approx(1.0)
