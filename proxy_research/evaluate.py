"""Evaluation primitives for the frozen public broad-universe proxy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from alpha_factor_engine.research.metrics import newey_west_tstat, performance_summary


PRICE_FEATURES = [
    "momentum_12_1", "momentum_6_1", "reversal_1m", "realized_vol_3m",
    "downside_vol_3m", "beta_12m", "idio_vol_12m", "illiquidity_3m",
    "log_dollar_volume_3m", "price_to_52w_high", "max_drawdown_12m", "skew_3m",
]
FUNDAMENTAL_DIRECTIONS = {
    "book_to_market": 1,
    "earnings_yield": 1,
    "roe": 1,
    "roa": 1,
    "gross_profitability": 1,
    "debt_to_equity": -1,
    "asset_growth": -1,
    "sales_growth": 1,
    "accruals": -1,
    "free_cash_flow_yield": 1,
}
MACRO_FEATURES = [
    "fed_funds", "yield_curve", "vix", "baa_yield", "fed_funds_change_3m",
    "yield_curve_change_3m", "vix_change_3m", "baa_yield_change_3m",
]
MODEL_FEATURES = PRICE_FEATURES + list(FUNDAMENTAL_DIRECTIONS) + [
    "log_market_cap", "fundamental_age_days",
] + MACRO_FEATURES


def sanitize_panel(
    panel: pd.DataFrame,
    *,
    minimum_cross_section: int,
    first_prediction: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Freeze eligible rows and remove whole months with partial future labels."""

    frame = panel.copy()
    frame["signal_date"] = pd.to_datetime(frame["signal_date"], utc=True)
    frame["entry_at"] = pd.to_datetime(frame["entry_at"], utc=True, errors="coerce")
    frame["exit_at"] = pd.to_datetime(frame["exit_at"], utc=True, errors="coerce")
    frame = frame.loc[frame["eligible"].fillna(False)].copy()
    audit = []
    keep_dates = []
    first = pd.Timestamp(first_prediction)
    first = first.tz_localize("UTC") if first.tzinfo is None else first.tz_convert("UTC")
    for signal_date, group in frame.groupby("signal_date", sort=True):
        missing = int(group["target_return"].isna().sum())
        if missing == len(group):
            status = "future_exit_unavailable"
        elif missing:
            status = "dropped_entire_partial_label_month"
        elif signal_date >= first and len(group) < minimum_cross_section:
            status = "below_minimum_cross_section"
        else:
            status = "kept"
            keep_dates.append(signal_date)
        audit.append({
            "signal_date": signal_date,
            "eligible_rows": len(group),
            "missing_labels": missing,
            "status": status,
        })
    clean = frame.loc[frame["signal_date"].isin(keep_dates)].copy()
    if clean.duplicated(["signal_date", "ticker"]).any():
        raise RuntimeError("duplicate ticker-months remain after sanitization")
    if clean["target_return"].isna().any():
        raise RuntimeError("sanitized panel retains missing labels")
    return clean.sort_values(["signal_date", "ticker"]), pd.DataFrame(audit)


def build_equal_weight_portfolio(
    predictions: pd.DataFrame,
    *,
    score_column: str | None,
    holdings_count: int | None,
    trading_cost_bps_one_way: float,
    retention_rank: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a next-open-to-next-open long-only portfolio with drift accounting.

    When ``retention_rank`` is supplied, an existing holding is retained while
    its current deterministic full-universe rank remains within that cutoff.
    Vacated slots are filled from the highest-ranked remaining names.  The
    buffer uses prior membership only and never uses a realized return label.
    """

    if retention_rank is not None:
        if score_column is None or holdings_count is None:
            raise ValueError("retention_rank requires a scored fixed-size portfolio")
        if retention_rank < holdings_count:
            raise ValueError("retention_rank must be at least holdings_count")

    holdings = []
    previous: dict[str, float] = {}
    returns = []
    for signal_date, group in predictions.groupby("signal_date", sort=True):
        if group["target_return"].isna().any():
            raise RuntimeError(f"partial label coverage at {signal_date}")
        month = group.copy()
        if score_column is not None:
            month = month.sort_values(
                [score_column, "ticker"], ascending=[False, True], kind="mergesort"
            )
        else:
            month = month.sort_values("ticker")
        if holdings_count is not None:
            if len(month) < holdings_count:
                continue
            if retention_rank is None:
                month = month.head(holdings_count)
            else:
                ranked = month.copy()
                ranked["selection_rank"] = np.arange(1, len(ranked) + 1)
                retained = ranked.loc[
                    ranked["ticker"].astype(str).isin(previous)
                    & ranked["selection_rank"].le(retention_rank)
                ]
                additions = ranked.loc[
                    ~ranked["ticker"].astype(str).isin(retained["ticker"].astype(str))
                ].head(holdings_count - len(retained))
                month = pd.concat([retained, additions], ignore_index=True).sort_values(
                    [score_column, "ticker"], ascending=[False, True], kind="mergesort"
                )
                if len(month) != holdings_count:
                    raise RuntimeError("retention buffer did not produce the fixed holdings count")
        weight = 1.0 / len(month)
        month["weight"] = weight
        current = dict(zip(month["ticker"].astype(str), month["weight"]))
        names = set(previous) | set(current)
        trade_weight = {name: current.get(name, 0.0) - previous.get(name, 0.0) for name in names}
        turnover = float(sum(abs(value) for value in trade_weight.values()))
        month["trade_weight"] = month["ticker"].astype(str).map(trade_weight).fillna(0.0)
        gross = float((month["weight"] * month["target_return"]).sum())
        trading_cost = turnover * trading_cost_bps_one_way / 10_000.0
        entry_values = month["entry_at"].dropna().unique()
        exit_values = month["exit_at"].dropna().unique()
        if len(entry_values) != 1 or len(exit_values) != 1:
            raise RuntimeError(f"inconsistent execution timestamps at {signal_date}")
        returns.append({
            "signal_date": signal_date,
            "entry_at": entry_values[0],
            "exit_at": exit_values[0],
            "gross_return": gross,
            "turnover": turnover,
            "trading_cost": trading_cost,
            "borrow_cost": 0.0,
            "net_return": gross - trading_cost,
            "terminal_liquidation_turnover": 0.0,
            "terminal_liquidation_cost": 0.0,
            "holdings_count": len(month),
        })
        holdings.append(month)
        denominator = 1.0 + gross
        if denominator <= 0:
            raise RuntimeError(f"non-positive portfolio NAV at {signal_date}")
        previous = {
            str(row.ticker): weight * (1.0 + float(row.target_return)) / denominator
            for row in month.itertuples(index=False)
        }
    if not returns:
        raise RuntimeError("portfolio has no evaluable months")
    liquidation_turnover = float(sum(abs(value) for value in previous.values()))
    liquidation_cost = liquidation_turnover * trading_cost_bps_one_way / 10_000.0
    returns[-1]["turnover"] += liquidation_turnover
    returns[-1]["trading_cost"] += liquidation_cost
    returns[-1]["terminal_liquidation_turnover"] = liquidation_turnover
    returns[-1]["terminal_liquidation_cost"] = liquidation_cost
    returns[-1]["net_return"] -= liquidation_cost
    return pd.DataFrame(returns), pd.concat(holdings, ignore_index=True)


def active_summary(strategy: pd.DataFrame, benchmark: pd.DataFrame) -> dict[str, float]:
    aligned = strategy.merge(
        benchmark[["signal_date", "net_return"]], on="signal_date",
        suffixes=("_strategy", "_benchmark"), validate="one_to_one",
    )
    active = aligned["net_return_strategy"] - aligned["net_return_benchmark"]
    volatility = active.std(ddof=1) * np.sqrt(12)
    return {
        "months": float(len(active)),
        "active_annualized_arithmetic_return": float(active.mean() * 12),
        "active_annualized_volatility": float(volatility),
        "information_ratio": float(active.mean() * 12 / volatility) if volatility > 0 else np.nan,
        "active_newey_west_tstat": newey_west_tstat(active),
        "positive_active_month_fraction": float(active.gt(0).mean()),
    }


def factor_validation(
    panel: pd.DataFrame,
    *,
    start: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = pd.Timestamp(start)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    sample = panel.loc[panel["signal_date"].ge(cutoff)].copy()
    monthly_rows = []
    for factor, direction in FUNDAMENTAL_DIRECTIONS.items():
        for signal_date, group in sample.groupby("signal_date", sort=True):
            valid = group[[factor, "target_return"]].dropna()
            if len(valid) < 100:
                continue
            ranks = valid[factor].rank(method="first")
            decile = pd.qcut(ranks, 10, labels=False)
            raw_ic = float(np.corrcoef(rankdata(valid[factor]), rankdata(valid["target_return"]))[0, 1])
            means = valid.groupby(decile)["target_return"].mean()
            monthly_rows.append({
                "factor": factor,
                "signal_date": signal_date,
                "n": len(valid),
                "coverage": float(group[factor].notna().mean()),
                "oriented_ic": direction * raw_ic,
                "oriented_top_minus_bottom": direction * float(means.iloc[-1] - means.iloc[0]),
                **{f"decile_{int(index)+1}": value for index, value in means.items()},
            })
    monthly = pd.DataFrame(monthly_rows)
    summary_rows = []
    for factor, group in monthly.groupby("factor", sort=False):
        tstat = newey_west_tstat(group["oriented_ic"])
        pvalue = float(1 - norm.cdf(tstat)) if np.isfinite(tstat) else np.nan
        summary_rows.append({
            "factor": factor,
            "months": len(group),
            "mean_coverage": group["coverage"].mean(),
            "mean_oriented_ic": group["oriented_ic"].mean(),
            "ic_newey_west_tstat": tstat,
            "one_sided_p_value": pvalue,
            "mean_oriented_top_minus_bottom": group["oriented_top_minus_bottom"].mean(),
        })
    summary = pd.DataFrame(summary_rows)
    summary["bh_q_value"] = benjamini_hochberg(summary["one_sided_p_value"])
    return summary, monthly


def benjamini_hochberg(pvalues: pd.Series) -> pd.Series:
    values = pd.to_numeric(pvalues, errors="coerce")
    output = pd.Series(np.nan, index=pvalues.index, dtype=float)
    valid = values.dropna().sort_values()
    if valid.empty:
        return output
    adjusted = valid.to_numpy() * len(valid) / np.arange(1, len(valid) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output.loc[valid.index] = np.minimum(adjusted, 1.0)
    return output


def summaries_by_period(
    returns: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    holdout_start: str,
) -> dict[str, dict[str, float]]:
    cutoff = pd.Timestamp(holdout_start)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    output = {}
    for label, mask in {
        "full_oos": returns["signal_date"].notna(),
        "untouched_holdout": returns["signal_date"].ge(cutoff),
    }.items():
        chosen = returns.loc[mask].copy()
        bench = benchmark.loc[benchmark["signal_date"].isin(chosen["signal_date"])].copy()
        output[label] = {**performance_summary(chosen), **active_summary(chosen, bench)}
    return output


def capacity_summary(holdings: pd.DataFrame, *, capital_usd: float = 10_000_000) -> dict[str, float]:
    rows = []
    previous: dict[str, float] = {}
    previous_adv: dict[str, float] = {}
    for signal_date, group in holdings.groupby("signal_date", sort=True):
        current = dict(zip(group["ticker"].astype(str), group["weight"].astype(float)))
        current_adv = dict(zip(group["ticker"].astype(str), group["adv_usd_3m"].astype(float)))
        for ticker in set(previous) | set(current):
            trade_weight = current.get(ticker, 0.0) - previous.get(ticker, 0.0)
            adv = current_adv.get(ticker, previous_adv.get(ticker, np.nan))
            rows.append({
                "signal_date": signal_date,
                "ticker": ticker,
                "trade_weight": trade_weight,
                "adv_usd": adv,
                "trade_type": "rebalance",
            })
        gross = float((group["weight"] * group["target_return"]).sum())
        denominator = 1.0 + gross
        previous = {
            str(row.ticker): float(row.weight) * (1.0 + float(row.target_return)) / denominator
            for row in group.itertuples(index=False)
        }
        previous_adv = current_adv
    for ticker, weight in previous.items():
        rows.append({
            "signal_date": holdings["signal_date"].max(),
            "ticker": ticker,
            "trade_weight": -weight,
            "adv_usd": previous_adv.get(ticker, np.nan),
            "trade_type": "terminal_liquidation",
        })
    trades = pd.DataFrame(rows)
    trades["trade_dollars"] = trades["trade_weight"].abs() * capital_usd
    trades["adv_participation"] = trades["trade_dollars"] / trades["adv_usd"].replace(0, np.nan)
    return {
        "capital_usd": capital_usd,
        "trade_records": len(trades),
        "median_adv_participation": float(trades["adv_participation"].median()),
        "p95_adv_participation": float(trades["adv_participation"].quantile(0.95)),
        "maximum_adv_participation": float(trades["adv_participation"].max()),
    }


def manual_first_month_reconstruction(
    predictions: pd.DataFrame,
    returns: pd.DataFrame,
    holdings: pd.DataFrame,
    *,
    cost_bps: float,
) -> dict[str, float | str | bool]:
    signal_date = returns.iloc[0]["signal_date"]
    chosen = holdings.loc[holdings["signal_date"].eq(signal_date)].copy()
    gross = float(np.mean(chosen["target_return"]))
    turnover = float(chosen["weight"].abs().sum())
    cost = turnover * cost_bps / 10_000.0
    reported = returns.iloc[0]
    return {
        "signal_date": str(signal_date),
        "holdings_count": len(chosen),
        "manual_gross_return": gross,
        "reported_gross_return": float(reported["gross_return"]),
        "manual_turnover": turnover,
        "reported_turnover": float(reported["turnover"]),
        "manual_trading_cost": cost,
        "reported_trading_cost": float(reported["trading_cost"]),
        "exact_match": bool(
            np.isclose(gross, reported["gross_return"], atol=1e-14)
            and np.isclose(turnover, reported["turnover"], atol=1e-14)
            and np.isclose(cost, reported["trading_cost"], atol=1e-14)
        ),
    }


__all__ = [
    "FUNDAMENTAL_DIRECTIONS", "MODEL_FEATURES", "active_summary",
    "benjamini_hochberg", "build_equal_weight_portfolio", "capacity_summary",
    "factor_validation", "manual_first_month_reconstruction", "sanitize_panel",
    "summaries_by_period",
]
