import numpy as np
import pandas as pd
import pytest

from alpha_factor_engine.contracts import PointInTimeViolation
from alpha_factor_engine.research.residual_overlay import (
    ResidualOverlayConfig,
    build_momentum_residual_overlay,
)


def _scoring_frame(*, dates: int = 1, names: int = 8) -> pd.DataFrame:
    rows = []
    signal_dates = pd.date_range("2024-01-31", periods=dates, freq="ME", tz="UTC")
    for date_index, signal_date in enumerate(signal_dates):
        for index in range(names):
            momentum = float(index - names / 2)
            beta = 0.7 + 0.04 * ((index * 3) % names)
            size = 8.0 + 0.2 * ((index * 5) % names)
            residual_noise = float(((index * 7 + date_index * 3) % names) - names / 2) / 10.0
            rows.append(
                {
                    "signal_date": signal_date,
                    "ticker": f"T{index:02d}",
                    "momentum_12_1": momentum,
                    "residual_ml_score": 1.8 * momentum - 2.5 * beta + 0.7 * size + residual_noise,
                    "beta_12m": beta,
                    "log_market_cap": size,
                }
            )
    return pd.DataFrame(rows)


def test_default_rank_blend_has_fixed_80_20_contributions_and_diagnostics():
    scores, diagnostics = build_momentum_residual_overlay(_scoring_frame())

    assert ResidualOverlayConfig().residual_ml_weight == pytest.approx(0.2)
    assert scores["momentum_weight"].eq(0.8).all()
    assert scores["residual_ml_weight"].eq(0.2).all()
    assert np.allclose(
        scores["overlay_score"],
        0.8 * scores["momentum_rank_score"] + 0.2 * scores["residual_ml_rank_score"],
    )
    assert np.allclose(
        scores["momentum_contribution"], 0.8 * scores["momentum_rank_score"]
    )
    assert np.allclose(
        scores["residual_ml_contribution"], 0.2 * scores["residual_ml_rank_score"]
    )
    assert diagnostics.loc[0, "cross_section_size"] == 8
    assert np.isfinite(diagnostics.loc[0, "momentum_ml_rank_correlation"])
    assert diagnostics.loc[0, "residual_ml_absolute_contribution_share"] == pytest.approx(0.2)


def test_missing_or_nonfinite_scores_and_return_labels_fail_closed():
    missing = _scoring_frame()
    missing.loc[0, "residual_ml_score"] = np.nan
    with pytest.raises(ValueError, match="missing or non-finite"):
        build_momentum_residual_overlay(missing)

    infinite = _scoring_frame()
    infinite.loc[1, "momentum_12_1"] = np.inf
    with pytest.raises(ValueError, match="missing or non-finite"):
        build_momentum_residual_overlay(infinite)

    labeled = _scoring_frame().assign(target_return=0.01)
    with pytest.raises(PointInTimeViolation, match="return-label"):
        build_momentum_residual_overlay(labeled)


def test_score_and_final_order_ties_are_deterministic_under_row_shuffling():
    frame = _scoring_frame(names=6)
    frame["momentum_12_1"] = [2.0, 2.0, 1.0, 1.0, 0.0, 0.0]
    frame["residual_ml_score"] = 5.0

    first, _ = build_momentum_residual_overlay(frame.sample(frac=1.0, random_state=1))
    second, _ = build_momentum_residual_overlay(frame.sample(frac=1.0, random_state=929))
    columns = [
        "ticker",
        "momentum_rank_score",
        "residual_ml_rank_score",
        "overlay_score",
        "overlay_rank",
    ]

    pd.testing.assert_frame_equal(first[columns], second[columns])
    assert first.loc[first["momentum_12_1"].eq(2.0), "momentum_rank_score"].nunique() == 1
    assert first.loc[first["momentum_12_1"].eq(0.0), "momentum_rank_score"].nunique() == 1
    tied_top = first.loc[first["momentum_12_1"].eq(2.0), "ticker"].tolist()
    assert tied_top == sorted(tied_top)


def test_orthogonalization_is_contemporaneous_and_removes_linear_exposures():
    frame = _scoring_frame(dates=2, names=12)
    scores, diagnostics = build_momentum_residual_overlay(frame, orthogonalize_ml=True)

    for _, group in scores.groupby("signal_date"):
        design = np.column_stack(
            [
                np.ones(len(group)),
                group["momentum_12_1"],
                group["beta_12m"],
                group["log_market_cap"],
            ]
        )
        assert np.allclose(
            design.T @ group["residual_ml_adjusted_score"].to_numpy(),
            0.0,
            atol=1e-10,
        )
    assert diagnostics["orthogonalized_ml"].all()
    assert diagnostics["orthogonalization_r_squared"].gt(0.9).all()

    changed_future = frame.copy()
    last_date = changed_future["signal_date"].max()
    changed_future.loc[changed_future["signal_date"].eq(last_date), "residual_ml_score"] *= -100.0
    changed_future.loc[changed_future["signal_date"].eq(last_date), "beta_12m"] += 50.0
    changed_scores, _ = build_momentum_residual_overlay(changed_future, orthogonalize_ml=True)
    first_date = frame["signal_date"].min()
    comparison_columns = [
        "ticker",
        "residual_ml_adjusted_score",
        "residual_ml_rank_score",
        "overlay_score",
        "overlay_rank",
    ]
    original_first = scores.loc[scores["signal_date"].eq(first_date), comparison_columns].reset_index(
        drop=True
    )
    changed_first = changed_scores.loc[
        changed_scores["signal_date"].eq(first_date), comparison_columns
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(original_first, changed_first)


def test_orthogonalization_requires_complete_contemporaneous_exposures():
    frame = _scoring_frame()
    frame.loc[0, "beta_12m"] = np.nan

    with pytest.raises(ValueError, match="orthogonalization exposure.*missing or non-finite"):
        build_momentum_residual_overlay(frame, orthogonalize_ml=True)


def test_custom_weight_is_fixed_across_dates_and_has_complementary_ml_weight():
    scores, diagnostics = build_momentum_residual_overlay(
        _scoring_frame(dates=2), momentum_weight=0.7
    )

    assert scores["momentum_weight"].eq(0.7).all()
    assert scores["residual_ml_weight"].eq(0.3).all()
    assert diagnostics["momentum_weight"].eq(0.7).all()
    assert diagnostics["residual_ml_weight"].eq(0.3).all()

    with pytest.raises(ValueError, match="strictly between"):
        build_momentum_residual_overlay(_scoring_frame(), momentum_weight=1.0)
