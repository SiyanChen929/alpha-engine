import numpy as np
import pytest

from alpha_factor_engine.research.sec_filing_sue_drift_bootstrap_repair import (
    circular_mbb,
    noncircular_design_expectation,
    repaired_noncircular_mbb,
)


def test_design_expectation_matches_large_monte_carlo_mean():
    values = np.linspace(-0.04, 0.06, 53) ** 2
    centered = values - values.mean()
    expected = noncircular_design_expectation(centered, 8)
    result, _ = repaired_noncircular_mbb(values, draws=40_000, block_length=8, seed=929)
    monte_carlo_se = result["corrected_null_std"] / np.sqrt(result["draws"])
    assert result["analytic_design_expected_mean"] == pytest.approx(expected)
    assert abs(result["corrected_null_mean"]) <= 4 * monte_carlo_se


def test_repair_is_deterministic_and_preserves_observed_mean():
    values = np.sin(np.linspace(0, 9, 80)) / 100
    first, first_draws = repaired_noncircular_mbb(values, draws=500, block_length=9, seed=929)
    second, second_draws = repaired_noncircular_mbb(values, draws=500, block_length=9, seed=929)
    assert first == second
    assert first["observed_mean"] == pytest.approx(values.mean())
    assert first_draws.equals(second_draws)


def test_circular_all_start_design_is_centered():
    values = np.arange(31, dtype=float) ** 2
    result, _ = circular_mbb(values, draws=20_000, block_length=7, seed=929)
    monte_carlo_se = result["null_std"] / np.sqrt(result["draws"])
    assert abs(result["null_mean"]) <= 4 * monte_carlo_se
