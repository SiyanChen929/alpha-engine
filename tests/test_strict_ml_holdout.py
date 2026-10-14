from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_factor_engine.contracts import PointInTimeViolation
from alpha_factor_engine.research.strict_ml_holdout import (
    CONTROL_MISSING_POLICY,
    MonotonicGBDTConfig,
    build_incremental_target,
    prepare_cross_sectional_features,
    prepare_strict_fold_local_split,
    select_top_momentum_candidates,
    strict_heldout_information_coefficient,
    walk_forward_strict_security_holdout_predict,
)
from alpha_factor_engine.research.universe_validation import assign_security_folds


FEATURES = [
    "exact_total_momentum_12_1",
    "residual_momentum_score",
    "beta_12m",
    "log_market_cap",
    "log_dollar_volume_3m",
    "roe",
    "accruals",
]


def _panel(*, months: int = 36, names: int = 60) -> pd.DataFrame:
    signal_dates = pd.date_range("2020-01-31", periods=months, freq="ME", tz="UTC")
    rows: list[dict[str, object]] = []
    for month_number, signal_date in enumerate(signal_dates):
        for name_number in range(names):
            momentum = (
                0.01 * name_number
                + 0.08 * np.sin((month_number + name_number) / 5.0)
            )
            beta = 0.7 + 0.01 * (name_number % 25)
            size = 8.0 + 0.04 * name_number
            log_adv = 15.0 + 0.03 * ((name_number * 7) % names)
            roe = -0.1 + 0.01 * (name_number % 30)
            accruals = -0.08 + 0.004 * ((name_number * 11) % 40)
            target = (
                0.002 * name_number
                + 0.015 * roe
                - 0.01 * accruals
                + 0.02 * np.sin((month_number * 3 + name_number) / 7.0)
            )
            rows.append(
                {
                    "signal_date": signal_date,
                    "exit_at": signal_date + pd.Timedelta(days=25),
                    "ticker": f"S{name_number:03d}",
                    "eligible": True,
                    "target_return": target,
                    "exact_total_momentum_12_1": momentum,
                    "residual_momentum_score": momentum - 0.2 * beta,
                    "beta_12m": beta,
                    "log_market_cap": size,
                    "log_dollar_volume_3m": log_adv,
                    "roe": roe,
                    "accruals": accruals,
                }
            )
    return pd.DataFrame(rows)


def test_top_candidate_selection_is_deterministic_and_label_independent():
    panel = _panel(months=2, names=20)
    first = select_top_momentum_candidates(panel, candidate_count=12)
    revised = panel.sample(frac=1.0, random_state=7).copy()
    revised["target_return"] = np.arange(len(revised), dtype=float) * 100.0
    second = select_top_momentum_candidates(revised, candidate_count=12)

    keys = ["signal_date", "momentum_candidate_rank", "ticker"]
    pd.testing.assert_frame_equal(first[keys].reset_index(drop=True), second[keys].reset_index(drop=True))
    assert first.groupby("signal_date")["ticker"].size().eq(12).all()


def test_none_candidate_count_keeps_all_eligible_rows_with_stable_order():
    panel = _panel(months=2, names=20)
    panel.loc[
        panel["ticker"].isin({"S003", "S017"}), "eligible"
    ] = False
    first = select_top_momentum_candidates(panel, candidate_count=None)

    revised = panel.sample(frac=1.0, random_state=29).copy()
    revised["target_return"] = np.arange(len(revised), dtype=float) * -1_000.0
    second = select_top_momentum_candidates(revised, candidate_count=None)

    keys = ["signal_date", "momentum_candidate_rank", "ticker"]
    pd.testing.assert_frame_equal(
        first[keys].reset_index(drop=True),
        second[keys].reset_index(drop=True),
    )
    assert first.groupby("signal_date")["ticker"].size().eq(18).all()
    assert first.groupby("signal_date")["momentum_candidate_rank"].max().eq(18).all()
    assert not first["ticker"].isin({"S003", "S017"}).any()


def test_none_candidate_count_runs_strict_fold_local_walk_forward():
    panel = _panel(months=36, names=60)
    predictions, audit = walk_forward_strict_security_holdout_predict(
        panel,
        FEATURES,
        first_prediction="2022-01-01",
        folds=3,
        candidate_count=None,
        minimum_training_rows=500,
        monotonic_directions={"roe": 1, "accruals": -1},
        model_config=MonotonicGBDTConfig(
            max_iter=5,
            learning_rate=0.05,
            max_leaf_nodes=3,
            l2_regularization=1.0,
            min_samples_leaf=10,
            seed=929,
        ),
    )

    successful = audit.loc[audit["status"].eq("ok")]
    assert successful["security_overlap"].eq(0).all()
    assert predictions.groupby("signal_date")["ticker"].nunique().eq(60).all()
    assert predictions["strict_ml_score"].notna().all()


def test_incremental_target_is_orthogonal_to_fold_local_controls():
    panel = _panel(months=1, names=40)
    result = build_incremental_target(panel)
    controls = [
        "exact_total_momentum_12_1",
        "beta_12m",
        "log_market_cap",
        "log_dollar_volume_3m",
    ]
    ranked = result[controls].rank(method="average", pct=True) * 2.0 - 1.0
    design = np.column_stack([np.ones(len(result)), ranked.to_numpy(float)])

    assert np.allclose(
        design.T @ result["incremental_target"].to_numpy(float),
        0.0,
        atol=1e-10,
    )
    assert "incremental_target" not in panel


def test_incremental_target_rejects_partial_future_labels():
    panel = _panel(months=1, names=20)
    panel.loc[0, "target_return"] = np.nan

    with pytest.raises(PointInTimeViolation, match="training target is incomplete"):
        build_incremental_target(panel)


def test_incremental_target_handles_missing_controls_inside_supplied_fold_only():
    panel = _panel(months=1, names=40)
    panel.loc[panel.index[::4], "log_market_cap"] = np.nan
    result = build_incremental_target(panel)
    design_parts = [np.ones(len(result))]
    for column in [
        "exact_total_momentum_12_1",
        "beta_12m",
        "log_market_cap",
        "log_dollar_volume_3m",
    ]:
        values = result[column]
        design_parts.append(
            (values.rank(method="average", pct=True) * 2.0 - 1.0)
            .fillna(0.0)
            .to_numpy(float)
        )
        design_parts.append(values.isna().astype(float).to_numpy())
    design = np.column_stack(design_parts)

    assert result["incremental_target"].notna().all()
    assert np.allclose(
        design.T @ result["incremental_target"].to_numpy(float),
        0.0,
        atol=1e-10,
    )
    with pytest.raises(ValueError, match="unsupported incremental-target"):
        build_incremental_target(panel, control_missing_policy="global_median")
    assert CONTROL_MISSING_POLICY == (
        "fold_local_centered_rank_zero_plus_missing_indicator"
    )


def test_feature_preparation_does_not_require_or_read_return_labels():
    panel = _panel(months=2, names=20).drop(columns="target_return")
    prepared, columns = prepare_cross_sectional_features(panel, FEATURES)

    assert len(columns) == 2 * len(FEATURES)
    assert prepared[list(columns)].notna().all().all()
    assert not any("target" in column for column in columns)


def test_heldout_label_changes_cannot_change_fold_local_training_inputs():
    panel = _panel()
    folds = assign_security_folds(panel, security_column="ticker", folds=3, seed=929)
    heldout_names = set(folds.loc[folds["security_fold"].eq(1), "ticker"])
    original = prepare_strict_fold_local_split(
        panel,
        FEATURES,
        security_fold=1,
        test_year=2022,
        folds=3,
        candidate_count=45,
    )
    revised_panel = panel.copy()
    revised_panel.loc[
        revised_panel["ticker"].isin(heldout_names), "target_return"
    ] *= -10_000.0
    revised = prepare_strict_fold_local_split(
        revised_panel,
        FEATURES,
        security_fold=1,
        test_year=2022,
        folds=3,
        candidate_count=45,
    )

    compare_columns = [
        "signal_date",
        "ticker",
        "incremental_target",
        *original.feature_columns,
    ]
    left = original.train[compare_columns].sort_values(["signal_date", "ticker"]).reset_index(drop=True)
    right = revised.train[compare_columns].sort_values(["signal_date", "ticker"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)
    assert original.audit["security_overlap"] == 0
    assert original.audit["latest_label_exit_at"] < original.audit["cutoff"]


def test_strict_walk_forward_produces_zero_overlap_and_fold_local_ic():
    panel = _panel()
    predictions, audit = walk_forward_strict_security_holdout_predict(
        panel,
        FEATURES,
        first_prediction="2022-01-01",
        folds=3,
        candidate_count=45,
        minimum_training_rows=500,
        monotonic_directions={"roe": 1, "accruals": -1},
        model_config=MonotonicGBDTConfig(
            max_iter=5,
            learning_rate=0.05,
            max_leaf_nodes=3,
            l2_regularization=1.0,
            min_samples_leaf=10,
            seed=929,
        ),
    )
    fold_month, monthly, summary = strict_heldout_information_coefficient(
        predictions, minimum_cross_section=8
    )

    successful = audit.loc[audit["status"].eq("ok")]
    assert successful["security_overlap"].eq(0).all()
    assert (successful["latest_label_exit_at"] < successful["cutoff"]).all()
    assert successful["train_target_scope"].eq("nonheldout_fold_only").all()
    assert predictions["strict_ml_score"].notna().all()
    assert fold_month["security_fold"].nunique() == 3
    assert monthly["folds"].eq(3).all()
    assert summary["months"] == len(monthly)
