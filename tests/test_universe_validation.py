import numpy as np
import pandas as pd

from alpha_factor_engine.research.universe_validation import (
    assign_security_folds,
    draw_stratified_security_universes,
    matched_random_portfolio_null,
    permute_labels_within_month,
)


def test_security_folds_are_stable_and_keep_permno_together():
    frame = pd.DataFrame(
        {
            "permno": np.repeat(np.arange(100, 140), 3),
            "value": np.arange(120),
        }
    )
    first = assign_security_folds(frame, folds=5, seed=929)
    second = assign_security_folds(frame.sample(frac=1, random_state=1), folds=5, seed=929)

    pd.testing.assert_frame_equal(first, second)
    assert first["permno"].is_unique
    assert set(first["security_fold"]).issubset(set(range(5)))


def test_label_permutation_preserves_each_month_distribution():
    frame = pd.DataFrame(
        {
            "signal_at": np.repeat(pd.to_datetime(["2020-01-31", "2020-02-28"], utc=True), 8),
            "permno": np.tile(np.arange(8), 2),
            "target_return": np.arange(16, dtype=float),
        }
    )
    permuted = permute_labels_within_month(frame, seed=929)

    for month, group in frame.groupby("signal_at"):
        expected = sorted(group["target_return"])
        actual = sorted(permuted.loc[permuted["signal_at"].eq(month), "target_return"])
        assert actual == expected
    assert not permuted["target_return"].equals(frame["target_return"])


def test_stratified_random_universe_has_exact_size_and_is_reproducible():
    rows = []
    for permno in range(100):
        for month in range(3):
            rows.append(
                {
                    "permno": permno,
                    "historical_sector": f"S{permno % 5}",
                    "market_cap_usd": float(permno + 1) * (month + 1),
                }
            )
    frame = pd.DataFrame(rows)
    first = draw_stratified_security_universes(
        frame, sizes=[25, 50], simulations=4, size_bins=5, seed=7
    )
    second = draw_stratified_security_universes(
        frame, sizes=[25, 50], simulations=4, size_bins=5, seed=7
    )

    pd.testing.assert_frame_equal(first, second)
    counts = first.groupby(["requested_size", "simulation"])["permno"].nunique()
    assert (counts.index.get_level_values("requested_size") == counts.to_numpy()).all()


def test_random_portfolio_null_detects_dominant_observed_stock():
    months = pd.date_range("2020-01-31", periods=36, freq="ME", tz="UTC")
    rows = []
    observed_rows = []
    benchmark_rows = []
    for month_number, month in enumerate(months):
        returns = []
        for security in range(20):
            value = 0.04 + 0.005 * np.sin(month_number) if security == 0 else 0.0
            returns.append(value)
            rows.append(
                {"signal_at": month, "ticker": f"S{security}", "target_return": value}
            )
        observed_rows.append({"signal_at": month, "net_return": returns[0]})
        benchmark_rows.append({"signal_at": month, "net_return": float(np.mean(returns))})
    result = matched_random_portfolio_null(
        pd.DataFrame(rows),
        pd.DataFrame(observed_rows),
        pd.DataFrame(benchmark_rows),
        holdings_count=1,
        simulations=2_000,
        trading_cost_bps_one_way=0,
        seed=929,
    )

    assert len(result.trials) == 2_000
    assert result.summary["empirical_p_value_active_annualized_return"] < 0.01
    assert result.summary["observed_active_annualized_return"] > result.summary[
        "null_active_annualized_return_p95"
    ]

