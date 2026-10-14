import numpy as np
import pandas as pd

from alpha_factor_engine.research.models import (
    ModelConfig,
    prepare_model_panel,
    rank_and_combine_components,
)


def _macro_panel(last_macro_value: float = 15.0) -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2020-01-31", periods=15, freq="ME", tz="UTC")
    for date_index, signal_date in enumerate(dates, start=1):
        macro_value = float(date_index if date_index < 15 else last_macro_value)
        for stock_index, ticker in enumerate(["A", "B", "C"], start=1):
            rows.append(
                {
                    "signal_date": signal_date,
                    "ticker": ticker,
                    "momentum_12_1": float(stock_index),
                    "fed_funds": macro_value,
                    "target_return": float(stock_index - 2) / 100,
                }
            )
    return pd.DataFrame(rows)


def test_macro_zscore_is_causal_time_varying_and_constant_within_date():
    transformed, _ = prepare_model_panel(
        _macro_panel(), ["momentum_12_1", "fed_funds"]
    )
    by_date = transformed.groupby("signal_date")

    assert by_date["x_fed_funds"].nunique().max() == 1
    assert by_date["x_momentum_12_1"].nunique().min() == 3

    usable_macro = by_date["x_fed_funds"].first().iloc[12:]
    assert usable_macro.nunique() > 1

    first_twelve = np.arange(1.0, 13.0)
    expected_thirteenth = (13.0 - first_twelve.mean()) / first_twelve.std(ddof=0)
    assert np.isclose(usable_macro.iloc[0], expected_thirteenth)

    changed_future, _ = prepare_model_panel(
        _macro_panel(last_macro_value=10_000.0), ["momentum_12_1", "fed_funds"]
    )
    original_earlier = transformed.loc[
        transformed["signal_date"].lt(transformed["signal_date"].max()), "x_fed_funds"
    ]
    changed_earlier = changed_future.loc[
        changed_future["signal_date"].lt(changed_future["signal_date"].max()), "x_fed_funds"
    ]
    assert np.allclose(original_earlier, changed_earlier)


def test_ranked_component_ensemble_is_scale_invariant():
    dates = pd.Series(
        pd.to_datetime(
            [
                "2024-01-31T23:59:00Z",
                "2024-01-31T23:59:00Z",
                "2024-01-31T23:59:00Z",
                "2024-02-29T23:59:00Z",
                "2024-02-29T23:59:00Z",
                "2024-02-29T23:59:00Z",
            ]
        )
    )
    components = pd.DataFrame(
        {
            "ridge_score": [1.0, 3.0, 2.0, 9.0, 7.0, 8.0],
            "extra_trees_score": [4.0, 2.0, 6.0, 1.0, 5.0, 3.0],
            "histogram_gbdt_score": [8.0, 9.0, 7.0, 4.0, 2.0, 6.0],
        }
    )
    scaled = components.copy()
    scaled["extra_trees_score"] = scaled["extra_trees_score"] * 100 + 1_000

    original = rank_and_combine_components(components, dates, ModelConfig())
    transformed = rank_and_combine_components(scaled, dates, ModelConfig())

    assert np.allclose(original, transformed)


def test_prepare_model_panel_supports_a_separate_residual_target():
    panel = _macro_panel()
    panel["residual_target"] = -panel["target_return"]

    transformed, _ = prepare_model_panel(
        panel,
        ["momentum_12_1"],
        target_column="residual_target",
    )

    first_date = transformed["signal_date"].min()
    observed = transformed.loc[transformed["signal_date"].eq(first_date)].sort_values("ticker")
    assert observed["target_rank"].is_monotonic_decreasing
    assert observed["model_target_column"].eq("residual_target").all()
