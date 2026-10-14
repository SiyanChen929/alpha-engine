"""Research metrics with HAC and block-bootstrap uncertainty."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from alpha_factor_engine.research.portfolio import neutralize_score


def max_drawdown(returns: pd.Series) -> float:
    wealth = pd.concat(
        [pd.Series([1.0], dtype=float), (1 + returns.fillna(0)).cumprod().reset_index(drop=True)],
        ignore_index=True,
    )
    drawdown = wealth / wealth.cummax() - 1
    return float(drawdown.min()) if len(drawdown) else np.nan


def newey_west_tstat(returns: pd.Series, lags: int = 3) -> float:
    values = returns.dropna().to_numpy(dtype=float)
    n = len(values)
    if n < lags + 3:
        return np.nan
    demeaned = values - values.mean()
    long_run_variance = float(np.dot(demeaned, demeaned) / n)
    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)
        covariance = float(np.dot(demeaned[lag:], demeaned[:-lag]) / n)
        long_run_variance += 2 * weight * covariance
    standard_error = np.sqrt(max(long_run_variance, 0) / n)
    return float(values.mean() / standard_error) if standard_error > 0 else np.nan


def performance_summary(returns: pd.DataFrame, return_column: str = "net_return") -> dict[str, float]:
    series = returns[return_column].dropna()
    months = len(series)
    if months == 0:
        return {}
    annualized_return = float((1 + series).prod() ** (12 / months) - 1)
    annualized_volatility = float(series.std(ddof=1) * np.sqrt(12))
    annualized_arithmetic_return = float(series.mean() * 12)
    sharpe = annualized_arithmetic_return / annualized_volatility if annualized_volatility > 0 else np.nan
    downside_deviation = float(np.sqrt(np.mean(np.minimum(series.to_numpy(), 0.0) ** 2)) * np.sqrt(12))
    return {
        "months": float(months),
        "annualized_return": annualized_return,
        "annualized_arithmetic_return": annualized_arithmetic_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": float(sharpe),
        "sortino": (
            float(annualized_arithmetic_return / downside_deviation)
            if downside_deviation > 0
            else np.nan
        ),
        "max_drawdown": max_drawdown(series),
        "positive_month_fraction": float(series.gt(0).mean()),
        "average_monthly_turnover": float(returns["turnover"].mean()),
        "average_trading_cost": float(returns["trading_cost"].mean()),
        "average_borrow_cost": float(returns["borrow_cost"].mean()),
        "newey_west_tstat": newey_west_tstat(series),
        "best_month": float(series.max()),
        "worst_month": float(series.min()),
    }


def information_coefficient(
    predictions: pd.DataFrame,
    score_column: str = "ensemble_score",
    *,
    neutralize: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for signal_date, group in predictions.groupby("signal_date", sort=True):
        missing = group["target_return"].isna()
        if missing.all():
            continue
        if missing.any():
            raise ValueError(
                f"partial target coverage at {signal_date}; IC would condition on future availability"
            )
        score = neutralize_score(group, score_column) if neutralize else group[score_column]
        valid = pd.DataFrame({"score": score, "target_return": group["target_return"]}).dropna()
        if len(valid) < 8:
            continue
        score_rank = rankdata(valid["score"])
        return_rank = rankdata(valid["target_return"])
        ic = float(np.corrcoef(score_rank, return_rank)[0, 1])
        rows.append(
            {
                "signal_date": signal_date,
                "ic": ic,
                "n": len(valid),
                "score_basis": "beta_size_neutralized" if neutralize else "raw",
            }
        )
    return pd.DataFrame(rows)


def moving_block_bootstrap_sharpe(
    returns: pd.Series,
    *,
    block_length: int = 6,
    simulations: int = 3000,
    seed: int = 929,
) -> dict[str, float]:
    values = returns.dropna().to_numpy(dtype=float)
    n = len(values)
    if n < block_length * 2:
        return {"bootstrap_sharpe_p05": np.nan, "bootstrap_sharpe_p50": np.nan, "bootstrap_sharpe_p95": np.nan}
    if np.std(values, ddof=1) <= 0:
        return {
            "bootstrap_sharpe_p05": np.nan,
            "bootstrap_sharpe_p50": np.nan,
            "bootstrap_sharpe_p95": np.nan,
        }
    rng = np.random.default_rng(seed)
    starts = np.arange(0, n - block_length + 1)
    draws: list[float] = []
    block_count = int(np.ceil(n / block_length))
    for _ in range(simulations):
        sampled_starts = rng.choice(starts, size=block_count, replace=True)
        sample = np.concatenate([values[start : start + block_length] for start in sampled_starts])[:n]
        volatility = sample.std(ddof=1)
        draws.append(sample.mean() / volatility * np.sqrt(12) if volatility > 0 else np.nan)
    quantiles = np.nanquantile(draws, [0.05, 0.50, 0.95])
    return {
        "bootstrap_sharpe_p05": float(quantiles[0]),
        "bootstrap_sharpe_p50": float(quantiles[1]),
        "bootstrap_sharpe_p95": float(quantiles[2]),
    }
