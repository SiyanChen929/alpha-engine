#!/usr/bin/env python3
"""Run the opened liquid-universe residual-momentum development experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from numba import njit
from scipy.stats import norm, spearmanr


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alpha_factor_engine.contracts import sha256_directory, sha256_file
from alpha_factor_engine.research.metrics import (
    max_drawdown,
    moving_block_bootstrap_sharpe,
    newey_west_tstat,
    performance_summary,
)
from alpha_factor_engine.research.residual_momentum import (
    compute_market_residual_momentum,
    neutralize_cross_sectional_score,
)
from proxy_research.evaluate import (
    active_summary,
    build_equal_weight_portfolio,
    capacity_summary,
    manual_first_month_reconstruction,
)


STRATEGIES = {
    "total_momentum_top50": "exact_total_momentum_12_1",
    "residual_momentum_top50": "residual_momentum_score",
    "cross_section_neutralized_score_top50": "neutralized_residual_momentum_score",
}


@njit(cache=False)
def _simulate_overlap_matched_paths(
    codes_by_month: np.ndarray,
    names_per_month: np.ndarray,
    returns_by_month_code: np.ndarray,
    desired_overlap: np.ndarray,
    benchmark_net: np.ndarray,
    *,
    simulations: int,
    holdings_count: int,
    cost_rate: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fast exact-overlap random portfolios; called through a documented wrapper."""

    np.random.seed(seed)
    months = len(names_per_month)
    annual_returns = np.empty(simulations, dtype=np.float64)
    information_ratios = np.empty(simulations, dtype=np.float64)
    average_turnovers = np.empty(simulations, dtype=np.float64)
    for trial in range(simulations):
        previous_codes = np.full(holdings_count, -1, dtype=np.int64)
        previous_weights = np.zeros(holdings_count, dtype=np.float64)
        net_returns = np.empty(months, dtype=np.float64)
        turnover_sum = 0.0
        for month in range(months):
            eligible_locations = np.empty(holdings_count, dtype=np.int64)
            eligible_count = 0
            for location in range(holdings_count):
                code = previous_codes[location]
                if code >= 0 and np.isfinite(returns_by_month_code[month, code]):
                    eligible_locations[eligible_count] = location
                    eligible_count += 1
            retain_count = desired_overlap[month]
            if retain_count > eligible_count:
                retain_count = eligible_count
            # Partial Fisher-Yates selection gives a uniform retained subset.
            for location in range(retain_count):
                swap_location = location + np.random.randint(eligible_count - location)
                temporary = eligible_locations[location]
                eligible_locations[location] = eligible_locations[swap_location]
                eligible_locations[swap_location] = temporary

            selected = np.full(holdings_count, -1, dtype=np.int64)
            for location in range(retain_count):
                selected[location] = previous_codes[eligible_locations[location]]
            selected_count = retain_count
            while selected_count < holdings_count:
                candidate = codes_by_month[
                    month, np.random.randint(names_per_month[month])
                ]
                # Exclude every prior holding so overlap equals the retained
                # count rather than increasing accidentally through re-draws.
                rejected = False
                for location in range(holdings_count):
                    if previous_codes[location] == candidate:
                        rejected = True
                        break
                if rejected:
                    continue
                for location in range(selected_count):
                    if selected[location] == candidate:
                        rejected = True
                        break
                if rejected:
                    continue
                selected[selected_count] = candidate
                selected_count += 1

            gross_return = 0.0
            for location in range(holdings_count):
                gross_return += returns_by_month_code[month, selected[location]]
            gross_return /= holdings_count
            previous_weight_sum = previous_weights.sum()
            overlap_minimum_weight = 0.0
            for location in range(retain_count):
                prior_weight = previous_weights[eligible_locations[location]]
                overlap_minimum_weight += min(1.0 / holdings_count, prior_weight)
            turnover = 1.0 + previous_weight_sum - 2.0 * overlap_minimum_weight
            turnover_sum += turnover
            net_returns[month] = gross_return - turnover * cost_rate
            denominator = 1.0 + gross_return
            for location in range(holdings_count):
                previous_codes[location] = selected[location]
                previous_weights[location] = (
                    (1.0 / holdings_count)
                    * (1.0 + returns_by_month_code[month, selected[location]])
                    / denominator
                )
        terminal_turnover = previous_weights.sum()
        net_returns[-1] -= terminal_turnover * cost_rate
        turnover_sum += terminal_turnover
        active_mean = 0.0
        for month in range(months):
            active_mean += net_returns[month] - benchmark_net[month]
        active_mean /= months
        active_variance = 0.0
        for month in range(months):
            difference = net_returns[month] - benchmark_net[month] - active_mean
            active_variance += difference * difference
        active_variance /= max(1, months - 1)
        annual_returns[trial] = active_mean * 12.0
        annual_volatility = np.sqrt(active_variance * 12.0)
        information_ratios[trial] = (
            annual_returns[trial] / annual_volatility
            if annual_volatility > 0
            else np.nan
        )
        average_turnovers[trial] = turnover_sum / months
    return annual_returns, information_ratios, average_turnovers


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"cannot JSON serialize {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _utc(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _period_slice(frame: pd.DataFrame, specification: dict[str, Any]) -> pd.DataFrame:
    output = frame.copy()
    start = _utc(specification.get("entry_at_or_after"))
    end = _utc(specification.get("entry_before"))
    if start is not None:
        output = output.loc[output["entry_at"].ge(start)]
    if end is not None:
        output = output.loc[output["entry_at"].lt(end)]
    return output.copy()


def _relative_returns(strategy: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    aligned = strategy.merge(
        benchmark[["signal_date", "net_return"]],
        on="signal_date",
        suffixes=("_strategy", "_benchmark"),
        validate="one_to_one",
    )
    aligned["active_return"] = aligned["net_return_strategy"] - aligned["net_return_benchmark"]
    return aligned


def _relative_summary(strategy: pd.DataFrame, benchmark: pd.DataFrame) -> dict[str, float]:
    aligned = _relative_returns(strategy, benchmark)
    active = aligned["active_return"]
    relative_wealth = (
        (1.0 + aligned["net_return_strategy"]).cumprod()
        / (1.0 + aligned["net_return_benchmark"]).cumprod()
    )
    relative_drawdown = relative_wealth / relative_wealth.cummax() - 1.0
    annual_return = float(active.mean() * 12)
    annual_volatility = float(active.std(ddof=1) * np.sqrt(12))
    return {
        "months": float(len(active)),
        "active_annualized_arithmetic_return": annual_return,
        "active_annualized_volatility": annual_volatility,
        "information_ratio": annual_return / annual_volatility if annual_volatility > 0 else np.nan,
        "active_newey_west_tstat": newey_west_tstat(active),
        "relative_max_drawdown": float(relative_drawdown.min()),
        "positive_active_month_fraction": float(active.gt(0).mean()),
    }


def _complete_liquid_universe(
    panel: pd.DataFrame,
    scores: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build signals without labels, then fail closed at evaluation time.

    Membership uses only information available at the decision. Future returns
    and execution timestamps are inspected after membership is frozen. A
    partially realized month is removed as a whole, never name by name.
    """
    frame = panel.copy()
    for column in ["signal_date", "entry_at", "exit_at", "price_available_at"]:
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame = frame.loc[frame["signal_date"].ge(_utc(config["research"]["first_signal_date"]))]
    frame = frame.loc[frame["eligible"].fillna(False)].copy()
    score_columns = [
        "formation_date",
        "ticker",
        "signal_at",
        "market_beta_signal",
        "beta_observations",
        "formation_observations",
        "residual_return_sum_12_1",
        "residual_volatility_daily_12_1",
        "residual_momentum_score",
        "exact_total_momentum_12_1",
        "latest_input_available_at",
    ]
    merged = frame.merge(
        scores[score_columns].rename(columns={"formation_date": "date", "signal_at": "residual_signal_at"}),
        on=["date", "ticker"],
        how="left",
        validate="one_to_one",
    )

    audit_rows: list[dict[str, Any]] = []
    chosen: list[pd.DataFrame] = []
    required_exposures = config["development_family"]["neutralization_exposures"]
    monthly_size = int(config["universe"]["monthly_size"])
    minimum_cross_section = int(config["universe"]["minimum_complete_cross_section"])
    minimum_price = float(config["universe"]["minimum_price"])
    minimum_adv = float(config["universe"]["minimum_adv_usd"])
    for signal_date, month in merged.groupby("signal_date", sort=True):
        missing_labels_before_selection = int(month["target_return"].isna().sum())
        signal_status = "kept"
        usable = month.loc[
            month["pit_close"].ge(minimum_price)
            & month["adv_usd_3m"].ge(minimum_adv)
            & np.isfinite(month["residual_momentum_score"])
            & np.isfinite(month["exact_total_momentum_12_1"])
        ].copy()
        selected = usable.iloc[0:0].copy()
        if len(usable) < minimum_cross_section:
            signal_status = "below_signal_cross_section"
        if signal_status == "kept":
            if (usable["price_available_at"] > signal_date).any():
                raise RuntimeError(f"price feature availability exceeds signal at {signal_date}")
            if (usable["latest_input_available_at"] > signal_date).any():
                raise RuntimeError(f"residual input availability exceeds signal at {signal_date}")
            selected = usable.sort_values(
                ["adv_usd_3m", "ticker"],
                ascending=[False, True],
                kind="mergesort",
            ).head(monthly_size)
            selected["universe_rank"] = np.arange(1, len(selected) + 1)
            chosen.append(selected)
            cutoff_adv = float(selected["adv_usd_3m"].min())
        else:
            cutoff_adv = np.nan
        selected_missing_labels = int(selected["target_return"].isna().sum())
        selected_complete_execution = bool(
            len(selected)
            and selected["entry_at"].notna().all()
            and selected["exit_at"].notna().all()
        )
        if len(selected) == 0:
            evaluation_status = "not_selected"
        elif selected_missing_labels == len(selected):
            evaluation_status = "future_exit_unavailable"
        elif selected_missing_labels:
            evaluation_status = "partial_labels_excluded"
        elif not selected_complete_execution:
            evaluation_status = "incomplete_execution_excluded"
        elif not selected["entry_at"].gt(signal_date).all() or not selected["exit_at"].gt(selected["entry_at"]).all():
            evaluation_status = "invalid_execution_clock_excluded"
        else:
            evaluation_status = "evaluable"
        exposure_complete = (
            selected[required_exposures]
            .apply(lambda column: np.isfinite(pd.to_numeric(column, errors="coerce")))
            .all(axis=1)
            if len(selected)
            else pd.Series(dtype=bool)
        )
        audit_rows.append(
            {
                "signal_date": signal_date,
                "eligible_rows": len(month),
                "missing_labels_before_selection": missing_labels_before_selection,
                "complete_signal_rows": len(usable),
                "selected_rows": len(selected),
                "selected_missing_labels": selected_missing_labels,
                "neutralization_complete_rows": int(exposure_complete.sum()),
                "monthly_adv_cutoff_usd": cutoff_adv,
                "signal_status": signal_status,
                "evaluation_status": evaluation_status,
                "status": signal_status,
            }
        )
    if not chosen:
        raise RuntimeError("no complete liquid-universe months survived")
    universe = pd.concat(chosen, ignore_index=True).sort_values(["signal_date", "ticker"])
    counts = universe.groupby("signal_date")["ticker"].size()
    if not (counts.ge(minimum_cross_section) & counts.le(monthly_size)).all():
        raise RuntimeError("liquid universe violates its monthly minimum or maximum size")
    if universe.duplicated(["signal_date", "ticker"]).any():
        raise RuntimeError("liquid universe contains duplicate ticker-months")
    universe = neutralize_cross_sectional_score(
        universe,
        score_column="residual_momentum_score",
        exposure_columns=list(required_exposures),
        group_column="signal_date",
        output_column="neutralized_residual_momentum_score",
    )
    audit = pd.DataFrame(audit_rows)
    neutral_counts = universe.groupby("signal_date")[
        "neutralized_residual_momentum_score"
    ].count()
    if (neutral_counts < minimum_cross_section).any():
        failed = neutral_counts.loc[neutral_counts < minimum_cross_section]
        raise RuntimeError(
            "risk-neutralized candidate lacks the minimum complete cross-section: "
            f"{failed.head().to_dict()}"
        )
    evaluable_dates = set(
        audit.loc[audit["evaluation_status"].eq("evaluable"), "signal_date"]
    )
    evaluation = universe.loc[universe["signal_date"].isin(evaluable_dates)].copy()
    return (
        universe.reset_index(drop=True),
        evaluation.reset_index(drop=True),
        audit,
    )


def _monthly_ic(universe: pd.DataFrame, score_column: str, strategy: str) -> pd.DataFrame:
    rows = []
    for signal_date, month in universe.groupby("signal_date", sort=True):
        values = month[[score_column, "target_return"]].dropna()
        rows.append(
            {
                "strategy": strategy,
                "signal_date": signal_date,
                "n": len(values),
                "rank_ic": float(values[score_column].corr(values["target_return"], method="spearman")),
            }
        )
    return pd.DataFrame(rows)


def _decile_results(
    universe: pd.DataFrame,
    periods: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly_rows: list[dict[str, Any]] = []
    for period_name, period_spec in periods.items():
        sample = _period_slice(universe, period_spec)
        for strategy, score_column in STRATEGIES.items():
            for signal_date, month in sample.groupby("signal_date", sort=True):
                ranked = month[score_column].rank(method="first")
                decile = pd.qcut(ranked, 10, labels=False) + 1
                means = month.assign(decile=decile).groupby("decile")["target_return"].mean()
                for number, value in means.items():
                    monthly_rows.append(
                        {
                            "period": period_name,
                            "strategy": strategy,
                            "signal_date": signal_date,
                            "decile": int(number),
                            "mean_forward_return": float(value),
                        }
                    )
    monthly = pd.DataFrame(monthly_rows)
    summary_rows: list[dict[str, Any]] = []
    for (period, strategy), group in monthly.groupby(["period", "strategy"], sort=True):
        means = group.groupby("decile")["mean_forward_return"].mean().sort_index()
        monotonicity = float(spearmanr(means.index.to_numpy(float), means.to_numpy(float)).statistic)
        summary_rows.append(
            {
                "period": period,
                "strategy": strategy,
                "months": group["signal_date"].nunique(),
                "decile_1_annualized_return": float(means.iloc[0] * 12),
                "decile_10_annualized_return": float(means.iloc[-1] * 12),
                "decile_10_minus_1_annualized": float((means.iloc[-1] - means.iloc[0]) * 12),
                "decile_rank_monotonicity": monotonicity,
                **{f"decile_{int(index)}_annualized": float(value * 12) for index, value in means.items()},
            }
        )
    return monthly, pd.DataFrame(summary_rows)


def _build_label_independent_target_holdings(
    signal_universe: pd.DataFrame,
    *,
    score_column: str,
    holdings_count: int,
    retention_rank: int,
) -> pd.DataFrame:
    """Create target membership using scores and prior membership only."""

    rows: list[pd.DataFrame] = []
    previous: set[str] = set()
    for signal_date, group in signal_universe.groupby("signal_date", sort=True):
        ranked = (
            group.loc[np.isfinite(pd.to_numeric(group[score_column], errors="coerce"))]
            .sort_values([score_column, "ticker"], ascending=[False, True], kind="mergesort")
            .copy()
        )
        ranked["selection_rank"] = np.arange(1, len(ranked) + 1)
        retained = ranked.loc[
            ranked["ticker"].astype(str).isin(previous)
            & ranked["selection_rank"].le(retention_rank)
        ]
        additions = ranked.loc[
            ~ranked["ticker"].astype(str).isin(retained["ticker"].astype(str))
        ].head(holdings_count - len(retained))
        chosen = pd.concat([retained, additions], ignore_index=True).sort_values(
            [score_column, "ticker"], ascending=[False, True], kind="mergesort"
        )
        if len(chosen) != holdings_count:
            raise RuntimeError(
                f"label-independent target membership has {len(chosen)} names at {signal_date}"
            )
        chosen["target_weight"] = 1.0 / holdings_count
        chosen["retained_from_prior_signal"] = chosen["ticker"].astype(str).isin(previous)
        rows.append(
            chosen[
                [
                    "signal_date",
                    "ticker",
                    score_column,
                    "selection_rank",
                    "target_weight",
                    "retained_from_prior_signal",
                    "adv_usd_3m",
                    "pit_close",
                    "price_available_at",
                    "latest_input_available_at",
                ]
            ].copy()
        )
        previous = set(chosen["ticker"].astype(str))
    if not rows:
        raise RuntimeError("no label-independent target memberships produced")
    return pd.concat(rows, ignore_index=True)


def _opened_search_space_ledger() -> pd.DataFrame:
    """Conservative lower-bound ledger implied by disclosed opened axes."""

    rows = []
    for number, values in enumerate(
        product(
            [500, 1000],
            [20_000_000, 25_000_000, 50_000_000],
            [
                "total_momentum",
                "market_residual_momentum",
                "score_neutralized_residual_momentum",
            ],
            [None, 75],
            [15, 25],
        ),
        start=1,
    ):
        universe_size, adv_floor, signal, retention_rank, cost_bps = values
        rows.append(
            {
                "search_id": f"opened-{number:03d}",
                "universe_size": universe_size,
                "minimum_adv_usd": adv_floor,
                "signal": signal,
                "retention_rank": retention_rank,
                "cost_bps_one_way": cost_bps,
                "outcome_available": False,
                "ledger_status": "declared_search_space_lower_bound; historical outcomes incomplete",
            }
        )
    return pd.DataFrame(rows)


def _run_periods(
    universe: pd.DataFrame,
    config: dict[str, Any],
    *,
    cost_bps: float,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    holdings_count = int(config["portfolio"]["holdings"])
    retention_rank = int(config["portfolio"]["retention_rank"])
    metric_rows: list[dict[str, Any]] = []
    full_returns: dict[str, pd.DataFrame] = {}
    full_holdings: dict[str, pd.DataFrame] = {}
    for period_name, period_spec in config["validation"]["periods"].items():
        sample = _period_slice(universe, period_spec)
        if sample.empty:
            raise RuntimeError(f"period {period_name} contains no observations")
        benchmark, benchmark_holdings = build_equal_weight_portfolio(
            sample,
            score_column=None,
            holdings_count=None,
            trading_cost_bps_one_way=cost_bps,
        )
        if period_name == "full":
            full_returns["equal_weight_liquid_universe"] = benchmark
            full_holdings["equal_weight_liquid_universe"] = benchmark_holdings
        for strategy, score_column in STRATEGIES.items():
            returns, holdings = build_equal_weight_portfolio(
                sample,
                score_column=score_column,
                holdings_count=holdings_count,
                trading_cost_bps_one_way=cost_bps,
                retention_rank=retention_rank,
            )
            if period_name == "full":
                full_returns[strategy] = returns
                full_holdings[strategy] = holdings
            active = _relative_summary(returns, benchmark)
            active_series = _relative_returns(returns, benchmark)["active_return"]
            bootstrap = moving_block_bootstrap_sharpe(
                active_series,
                block_length=int(config["validation"]["block_length_months"]),
                simulations=int(config["validation"]["block_bootstrap_simulations"]),
                seed=int(config["research"]["seed"]),
            )
            metric_rows.append(
                {
                    "period": period_name,
                    "strategy": strategy,
                    "cost_bps_one_way": cost_bps,
                    **performance_summary(returns),
                    **active,
                    **bootstrap,
                }
            )
    return pd.DataFrame(metric_rows), full_returns, full_holdings


def _incremental_metrics(
    universe: pd.DataFrame,
    config: dict[str, Any],
    *,
    cost_bps: float,
) -> pd.DataFrame:
    rows = []
    holdings_count = int(config["portfolio"]["holdings"])
    retention_rank = int(config["portfolio"]["retention_rank"])
    for period_name, period_spec in config["validation"]["periods"].items():
        sample = _period_slice(universe, period_spec)
        total, _ = build_equal_weight_portfolio(
            sample,
            score_column=STRATEGIES["total_momentum_top50"],
            holdings_count=holdings_count,
            trading_cost_bps_one_way=cost_bps,
            retention_rank=retention_rank,
        )
        for challenger in ["residual_momentum_top50", "cross_section_neutralized_score_top50"]:
            returns, _ = build_equal_weight_portfolio(
                sample,
                score_column=STRATEGIES[challenger],
                holdings_count=holdings_count,
                trading_cost_bps_one_way=cost_bps,
                retention_rank=retention_rank,
            )
            summary = _relative_summary(returns, total)
            relative = _relative_returns(returns, total)["active_return"]
            rows.append(
                {
                    "period": period_name,
                    "challenger": challenger,
                    "baseline": "total_momentum_top50",
                    **summary,
                    **moving_block_bootstrap_sharpe(
                        relative,
                        block_length=int(config["validation"]["block_length_months"]),
                        simulations=int(config["validation"]["block_bootstrap_simulations"]),
                        seed=int(config["research"]["seed"]) + 1,
                    ),
                }
            )
    return pd.DataFrame(rows)


def _cost_sensitivity(universe: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    holdings_count = int(config["portfolio"]["holdings"])
    retention_rank = int(config["portfolio"]["retention_rank"])
    for cost in config["portfolio"]["cost_sensitivity_bps"]:
        for period_name in ["full", "pre_2026"]:
            sample = _period_slice(universe, config["validation"]["periods"][period_name])
            benchmark, _ = build_equal_weight_portfolio(
                sample,
                score_column=None,
                holdings_count=None,
                trading_cost_bps_one_way=float(cost),
            )
            for strategy, score_column in STRATEGIES.items():
                returns, _ = build_equal_weight_portfolio(
                    sample,
                    score_column=score_column,
                    holdings_count=holdings_count,
                    trading_cost_bps_one_way=float(cost),
                    retention_rank=retention_rank,
                )
                rows.append(
                    {
                        "period": period_name,
                        "strategy": strategy,
                        "cost_bps_one_way": float(cost),
                        **_relative_summary(returns, benchmark),
                        "net_annualized_return": performance_summary(returns)["annualized_return"],
                    }
                )
    return pd.DataFrame(rows)


def _exposure_summary(
    universe: pd.DataFrame,
    holdings_by_strategy: dict[str, pd.DataFrame],
    exposure_columns: list[str],
) -> pd.DataFrame:
    ranked = universe.copy()
    percentile_columns = []
    for column in exposure_columns:
        percentile = f"{column}_cross_section_percentile"
        ranked[percentile] = ranked.groupby("signal_date")[column].rank(pct=True)
        percentile_columns.append(percentile)
    lookup = ranked[["signal_date", "ticker", *percentile_columns]]
    rows = []
    for strategy, holdings in holdings_by_strategy.items():
        if strategy == "equal_weight_liquid_universe":
            source = holdings.merge(lookup, on=["signal_date", "ticker"], validate="one_to_one")
        else:
            source = holdings.merge(lookup, on=["signal_date", "ticker"], validate="one_to_one")
        for column, percentile in zip(exposure_columns, percentile_columns):
            monthly_raw = source.assign(weighted=source["weight"] * source[column]).groupby("signal_date")["weighted"].sum()
            monthly_percentile = source.assign(weighted=source["weight"] * source[percentile]).groupby("signal_date")["weighted"].sum()
            rows.append(
                {
                    "strategy": strategy,
                    "exposure": column,
                    "mean_weighted_raw_exposure": float(monthly_raw.mean()),
                    "mean_weighted_cross_section_percentile": float(monthly_percentile.mean()),
                    "percentile_tilt_from_universe_midpoint": float(monthly_percentile.mean() - 0.5),
                }
            )
    return pd.DataFrame(rows)


def _capacity_table(
    holdings_by_strategy: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> pd.DataFrame:
    rows = []
    for strategy, holdings in holdings_by_strategy.items():
        if strategy == "equal_weight_liquid_universe":
            continue
        for capital in config["portfolio"]["aum_sensitivity_usd"]:
            rows.append(
                {
                    "strategy": strategy,
                    **capacity_summary(holdings, capital_usd=float(capital)),
                }
            )
    return pd.DataFrame(rows)


def _overlap_matched_random_null(
    universe: pd.DataFrame,
    config: dict[str, Any],
    *,
    simulations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cost_bps = float(config["portfolio"]["trading_cost_bps_one_way"])
    holdings_count = int(config["portfolio"]["holdings"])
    retention_rank = int(config["portfolio"]["retention_rank"])
    primary_score = STRATEGIES[config["development_family"]["primary_candidate"]]
    trial_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    master_rng = np.random.default_rng(int(config["research"]["seed"]))
    for period_index, period_name in enumerate(["full", "pre_2026"]):
        sample = _period_slice(universe, config["validation"]["periods"][period_name])
        benchmark, _ = build_equal_weight_portfolio(
            sample,
            score_column=None,
            holdings_count=None,
            trading_cost_bps_one_way=cost_bps,
        )
        observed, observed_holdings = build_equal_weight_portfolio(
            sample,
            score_column=primary_score,
            holdings_count=holdings_count,
            trading_cost_bps_one_way=cost_bps,
            retention_rank=retention_rank,
        )
        benchmark = benchmark.sort_values("signal_date")
        observed = observed.sort_values("signal_date")
        benchmark_net = benchmark["net_return"].to_numpy(float)
        observed_active = observed["net_return"].to_numpy(float) - benchmark_net
        observed_ann = float(observed_active.mean() * 12)
        observed_vol = float(observed_active.std(ddof=1) * np.sqrt(12))
        observed_ir = observed_ann / observed_vol if observed_vol > 0 else np.nan

        all_tickers = sorted(sample["ticker"].astype(str).unique())
        ticker_codes = {ticker: code for code, ticker in enumerate(all_tickers)}
        grouped_months = list(sample.groupby("signal_date", sort=True))
        maximum_names = max(len(month) for _, month in grouped_months)
        codes_by_month = np.full(
            (len(grouped_months), maximum_names), -1, dtype=np.int64
        )
        names_per_month = np.empty(len(grouped_months), dtype=np.int64)
        returns_by_month_code = np.full(
            (len(grouped_months), len(all_tickers)), np.nan, dtype=np.float64
        )
        observed_sets = []
        for month_index, (signal_date, month) in enumerate(grouped_months):
            codes = np.asarray(
                [ticker_codes[value] for value in month["ticker"].astype(str)],
                dtype=np.int64,
            )
            forward_returns = month["target_return"].to_numpy(float)
            names_per_month[month_index] = len(codes)
            codes_by_month[month_index, : len(codes)] = codes
            returns_by_month_code[month_index, codes] = forward_returns
            names = observed_holdings.loc[
                observed_holdings["signal_date"].eq(signal_date), "ticker"
            ].astype(str)
            observed_sets.append({ticker_codes[value] for value in names})
        desired_overlap = np.asarray(
            [0]
            + [
                len(observed_sets[index].intersection(observed_sets[index - 1]))
                for index in range(1, len(observed_sets))
            ],
            dtype=np.int64,
        )
        annual_returns, information_ratios, average_turnovers = _simulate_overlap_matched_paths(
            codes_by_month,
            names_per_month,
            returns_by_month_code,
            desired_overlap,
            benchmark_net,
            simulations=simulations,
            holdings_count=holdings_count,
            cost_rate=cost_bps / 10_000.0,
            seed=int(master_rng.integers(0, 2**31 - 1)) + period_index,
        )
        for trial in range(simulations):
            trial_rows.append(
                {
                    "period": period_name,
                    "trial": trial,
                    "active_annualized_arithmetic_return": annual_returns[trial],
                    "information_ratio": information_ratios[trial],
                    "average_monthly_turnover": average_turnovers[trial],
                }
            )
        period_trials = pd.DataFrame(trial_rows).loc[lambda value: value["period"].eq(period_name)]
        summary_rows.append(
            {
                "period": period_name,
                "simulations": simulations,
                "observed_active_annualized_arithmetic_return": observed_ann,
                "observed_information_ratio": observed_ir,
                "observed_average_monthly_turnover": float(observed["turnover"].mean()),
                "null_active_return_p50": float(period_trials["active_annualized_arithmetic_return"].median()),
                "null_active_return_p95": float(period_trials["active_annualized_arithmetic_return"].quantile(0.95)),
                "empirical_p_active_return": float(
                    (1 + period_trials["active_annualized_arithmetic_return"].ge(observed_ann).sum())
                    / (simulations + 1)
                ),
                "empirical_p_information_ratio": float(
                    (1 + period_trials["information_ratio"].ge(observed_ir).sum())
                    / (simulations + 1)
                ),
                "interpretation_limit": "exact name-overlap-matched null inside opened current-survivor panel; realized turnover is diagnostic only",
            }
        )
    return pd.DataFrame(trial_rows), pd.DataFrame(summary_rows)


def _multiple_testing(metrics: pd.DataFrame, family_size: int) -> pd.DataFrame:
    chosen = metrics.loc[metrics["period"].eq("pre_2026")].copy()
    chosen["one_sided_hac_p_value"] = 1.0 - norm.cdf(chosen["active_newey_west_tstat"])
    chosen["bonferroni_p_value"] = np.minimum(
        chosen["one_sided_hac_p_value"] * family_size, 1.0
    )
    return chosen[
        [
            "strategy",
            "active_newey_west_tstat",
            "one_sided_hac_p_value",
            "bonferroni_p_value",
        ]
    ].reset_index(drop=True)


def _manual_signal_reconstruction(
    daily: pd.DataFrame,
    scores: pd.DataFrame,
    audit: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    sample = scores.sort_values(["formation_date", "ticker"]).iloc[0]
    decision = audit.loc[audit["formation_date"].eq(sample["formation_date"])].iloc[0]
    tickers = [config["research"]["benchmark_ticker"], sample["ticker"]]
    prices = daily.loc[daily["ticker"].isin(tickers)].pivot(
        index="date", columns="ticker", values="adj_close"
    )
    benchmark = config["research"]["benchmark_ticker"]
    calendar = prices[benchmark].dropna().index
    prices = prices.reindex(calendar)
    returns = prices.pct_change(fill_method=None)
    position = int(calendar.get_loc(sample["formation_date"]))
    formation_start = position - int(config["signal"]["formation_sessions"])
    formation_end = position - int(config["signal"]["skip_sessions"])
    beta_start = max(0, formation_end - int(config["signal"]["beta_window_sessions"]) + 1)
    beta_sample = returns.iloc[beta_start : formation_end + 1][
        [sample["ticker"], benchmark]
    ].dropna()
    beta = float(np.cov(beta_sample[sample["ticker"]], beta_sample[benchmark], ddof=1)[0, 1] / np.var(beta_sample[benchmark], ddof=1))
    formation = returns.iloc[formation_start + 1 : formation_end + 1]
    residual = formation[sample["ticker"]] - beta * formation[benchmark]
    residual = residual.dropna()
    residual_sum = float(residual.sum())
    residual_vol = float(residual.std(ddof=1))
    score = residual_sum / residual_vol
    exact_total = float(
        prices.iloc[formation_end][sample["ticker"]]
        / prices.iloc[formation_start][sample["ticker"]]
        - 1.0
    )
    return {
        "formation_date": sample["formation_date"],
        "ticker": sample["ticker"],
        "beta_window_start": decision["beta_window_start"],
        "formation_start": decision["formation_start"],
        "formation_end": decision["formation_end"],
        "manual_beta": beta,
        "reported_beta": float(sample["market_beta_signal"]),
        "manual_residual_sum": residual_sum,
        "reported_residual_sum": float(sample["residual_return_sum_12_1"]),
        "manual_residual_volatility": residual_vol,
        "reported_residual_volatility": float(sample["residual_volatility_daily_12_1"]),
        "manual_score": score,
        "reported_score": float(sample["residual_momentum_score"]),
        "manual_exact_total_momentum": exact_total,
        "reported_exact_total_momentum": float(sample["exact_total_momentum_12_1"]),
        "exact_match": bool(
            np.isclose(beta, sample["market_beta_signal"], atol=1e-12)
            and np.isclose(residual_sum, sample["residual_return_sum_12_1"], atol=1e-12)
            and np.isclose(residual_vol, sample["residual_volatility_daily_12_1"], atol=1e-12)
            and np.isclose(score, sample["residual_momentum_score"], atol=1e-10)
            and np.isclose(exact_total, sample["exact_total_momentum_12_1"], atol=1e-12)
        ),
    }


def _yearly_results(
    returns_by_strategy: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    combined = []
    benchmark = returns_by_strategy["equal_weight_liquid_universe"].copy()
    benchmark["year"] = benchmark["entry_at"].dt.tz_convert("America/New_York").dt.year
    for strategy, returns in returns_by_strategy.items():
        frame = returns.copy()
        frame["year"] = frame["entry_at"].dt.tz_convert("America/New_York").dt.year
        for year, group in frame.groupby("year"):
            bench = benchmark.loc[benchmark["year"].eq(year)]
            active = group["net_return"].to_numpy(float) - bench["net_return"].to_numpy(float)
            combined.append(
                {
                    "strategy": strategy,
                    "year": int(year),
                    "months": len(group),
                    "compounded_net_return": float((1.0 + group["net_return"]).prod() - 1.0),
                    "active_arithmetic_return": float(active.sum()),
                }
            )
    return pd.DataFrame(combined)


def _gate_table(
    metrics: pd.DataFrame,
    incremental: pd.DataFrame,
    costs: pd.DataFrame,
    deciles: pd.DataFrame,
    capacity: pd.DataFrame,
    random_null: pd.DataFrame,
    universe_audit: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    gate = config["gates"]
    primary = config["development_family"]["primary_candidate"]
    primary_pre = metrics.loc[
        metrics["period"].eq("pre_2026") & metrics["strategy"].eq(primary)
    ].iloc[0]
    incremental_pre = incremental.loc[
        incremental["period"].eq("pre_2026") & incremental["challenger"].eq(primary)
    ].iloc[0]
    cost_row = costs.loc[
        costs["period"].eq("pre_2026")
        & costs["strategy"].eq(primary)
        & costs["cost_bps_one_way"].eq(float(gate["positive_active_at_cost_bps"]))
    ].iloc[0]
    decile_row = deciles.loc[
        deciles["period"].eq("pre_2026") & deciles["strategy"].eq(primary)
    ].iloc[0]
    capacity_row = capacity.loc[
        capacity["strategy"].eq(primary) & capacity["capital_usd"].eq(10_000_000.0)
    ].iloc[0]
    null_row = random_null.loc[random_null["period"].eq("pre_2026")].iloc[0]
    rows = [
        (
            "all_cross_sections_complete",
            bool(
                universe_audit.loc[
                    universe_audit["evaluation_status"].eq("evaluable"), "selected_rows"
                ].between(
                    int(config["universe"]["minimum_complete_cross_section"]),
                    int(config["universe"]["monthly_size"]),
                ).all()
            ),
            "required=true",
        ),
        ("signal_inputs_available_before_decision", True, "required=true; enforced fail-closed"),
        ("signal_generation_independent_of_future_labels", True, "required=true; membership frozen before evaluation labels are inspected"),
        ("upstream_survivorship_free", bool(config["universe"]["upstream_survivorship_free"]), "required=true"),
        ("upstream_delisting_complete", bool(config["universe"]["upstream_delisting_complete"]), "required=true"),
        ("complete_historical_trial_ledger", bool(config["validation"].get("trial_ledger_complete", False)), "required=true for corrected historical inference"),
        ("pre_2026_active_return", primary_pre["active_annualized_arithmetic_return"] >= float(gate["pre_2026_active_return_minimum"]), f"observed={primary_pre['active_annualized_arithmetic_return']:.6f}; minimum={float(gate['pre_2026_active_return_minimum']):.6f}"),
        ("pre_2026_information_ratio", primary_pre["information_ratio"] >= float(gate["pre_2026_information_ratio_minimum"]), f"observed={primary_pre['information_ratio']:.6f}; minimum={float(gate['pre_2026_information_ratio_minimum']):.6f}"),
        ("hac_tstat_after_disclosed_search", primary_pre["active_newey_west_tstat"] >= float(gate["pre_2026_hac_tstat_after_search_minimum"]), f"observed={primary_pre['active_newey_west_tstat']:.6f}; minimum={float(gate['pre_2026_hac_tstat_after_search_minimum']):.6f}"),
        ("residual_beats_total_momentum", incremental_pre["active_annualized_arithmetic_return"] >= float(gate["residual_minus_total_return_minimum"]), f"observed={incremental_pre['active_annualized_arithmetic_return']:.6f}"),
        ("bootstrap_active_sharpe_p05", primary_pre["bootstrap_sharpe_p05"] >= float(gate["bootstrap_active_sharpe_p05_minimum"]), f"observed={primary_pre['bootstrap_sharpe_p05']:.6f}"),
        ("positive_active_at_50bps", cost_row["active_annualized_arithmetic_return"] > 0, f"observed={cost_row['active_annualized_arithmetic_return']:.6f}"),
        ("relative_drawdown", primary_pre["relative_max_drawdown"] >= float(gate["maximum_relative_drawdown"]), f"observed={primary_pre['relative_max_drawdown']:.6f}; floor={float(gate['maximum_relative_drawdown']):.6f}"),
        ("decile_monotonicity", decile_row["decile_rank_monotonicity"] >= float(gate["minimum_decile_rank_monotonicity"]), f"observed={decile_row['decile_rank_monotonicity']:.6f}"),
        ("random_null_active_pvalue", null_row["empirical_p_active_return"] <= 0.05, f"observed={null_row['empirical_p_active_return']:.6f}"),
        ("capacity_10m_p95", capacity_row["p95_adv_participation"] <= float(gate["maximum_adv_participation"]), f"observed={capacity_row['p95_adv_participation']:.6f}; maximum={float(gate['maximum_adv_participation']):.6f}"),
        ("prospective_months", False, f"observed=0; minimum={int(gate['prospective_months_minimum'])}"),
    ]
    return pd.DataFrame(
        [{"gate": name, "passed": bool(passed), "evidence": evidence} for name, passed, evidence in rows]
    )


def _charts(
    output_dir: Path,
    returns_by_strategy: dict[str, pd.DataFrame],
    decile_summary: pd.DataFrame,
    exposure_summary: pd.DataFrame,
) -> list[Path]:
    chart_dir = output_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, ax = plt.subplots(figsize=(11, 6))
    for strategy, returns in returns_by_strategy.items():
        wealth = (1.0 + returns.sort_values("signal_date")["net_return"]).cumprod()
        label = strategy.replace("_", " ")
        ax.plot(returns.sort_values("signal_date")["entry_at"], wealth, label=label, linewidth=1.8)
    ax.set_yscale("log")
    ax.set_title("Liquid-universe next-open net wealth (base costs)")
    ax.set_ylabel("Growth of $1, log scale")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = chart_dir / "cumulative_net_wealth.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    benchmark = returns_by_strategy["equal_weight_liquid_universe"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for strategy in STRATEGIES:
        aligned = _relative_returns(returns_by_strategy[strategy], benchmark)
        active_wealth = (1.0 + aligned["active_return"]).cumprod()
        ax.plot(aligned["entry_at"], active_wealth, label=strategy.replace("_", " "))
    ax.axvline(pd.Timestamp("2026-01-01", tz="UTC"), color="black", linestyle="--", alpha=0.5)
    ax.set_title("Cumulative active return versus liquid-universe benchmark")
    ax.set_ylabel("Growth of $1 active-return series")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = chart_dir / "cumulative_active_return.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    pre = decile_summary.loc[decile_summary["period"].eq("pre_2026")]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for _, row in pre.iterrows():
        values = [row[f"decile_{number}_annualized"] for number in range(1, 11)]
        ax.plot(range(1, 11), values, marker="o", label=row["strategy"].replace("_", " "))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(1, 11))
    ax.set_title("Pre-2026 factor deciles: annualized next-month return")
    ax.set_xlabel("Signal decile (10 = high)")
    ax.set_ylabel("Annualized arithmetic return")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = chart_dir / "pre2026_factor_deciles.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    pivot = exposure_summary.pivot(
        index="exposure", columns="strategy", values="percentile_tilt_from_universe_midpoint"
    )
    fig, ax = plt.subplots(figsize=(11, 5.5))
    pivot.plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Average holdings exposure tilt versus universe midpoint")
    ax.set_ylabel("Cross-sectional percentile tilt")
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = chart_dir / "holdings_exposure_tilts.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)
    return paths


def _markdown_table(frame: pd.DataFrame, columns: list[str], formats: dict[str, str] | None = None) -> str:
    formats = formats or {}
    labels = [column.replace("_", " ") for column in columns]
    lines = ["| " + " | ".join(labels) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame[columns].itertuples(index=False, name=None):
        values = []
        for column, value in zip(columns, row):
            if pd.isna(value):
                values.append("NA")
            elif column in formats:
                values.append(format(value, formats[column]))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_report(
    output_dir: Path,
    config: dict[str, Any],
    metrics: pd.DataFrame,
    incremental: pd.DataFrame,
    deciles: pd.DataFrame,
    costs: pd.DataFrame,
    gates: pd.DataFrame,
    null_summary: pd.DataFrame,
    universe_audit: pd.DataFrame,
    scores: pd.DataFrame,
) -> Path:
    primary = config["development_family"]["primary_candidate"]
    key = metrics.loc[
        metrics["strategy"].eq(primary) & metrics["period"].isin(["pre_2026", "year_2026_h1", "full"])
    ].copy()
    metric_table = _markdown_table(
        key,
        [
            "period",
            "months",
            "active_annualized_arithmetic_return",
            "information_ratio",
            "active_newey_west_tstat",
            "relative_max_drawdown",
            "bootstrap_sharpe_p05",
        ],
        {
            "months": ".0f",
            "active_annualized_arithmetic_return": ".2%",
            "information_ratio": ".2f",
            "active_newey_west_tstat": ".2f",
            "relative_max_drawdown": ".2%",
            "bootstrap_sharpe_p05": ".2f",
        },
    )
    compare = incremental.loc[
        incremental["challenger"].eq(primary) & incremental["period"].isin(["pre_2026", "full"])
    ]
    compare_table = _markdown_table(
        compare,
        ["period", "active_annualized_arithmetic_return", "information_ratio", "active_newey_west_tstat"],
        {
            "active_annualized_arithmetic_return": ".2%",
            "information_ratio": ".2f",
            "active_newey_west_tstat": ".2f",
        },
    )
    gate_table = _markdown_table(gates, ["gate", "passed", "evidence"])
    kept = universe_audit.loc[universe_audit["signal_status"].eq("kept")]
    evaluable = universe_audit.loc[
        universe_audit["evaluation_status"].eq("evaluable")
    ]
    failed = gates.loc[~gates["passed"], "gate"].tolist()
    strategy_version = config["research"]["strategy_version"]
    report = f"""# {strategy_version} — opened development report

## Verdict

**INVALID** for a survivorship-free or deployable alpha claim. The upstream panel was selected from current SEC registrants using 2026-08-31 liquidity, excludes delisted names, and the historical period has already been opened. Signal, clock, execution, cost, and capacity checks below remain useful development evidence.

The primary candidate is **{primary}** in a contemporaneous ADV-at-least-${config['universe']['minimum_adv_usd']:,.0f}, maximum-{config['universe']['monthly_size']} universe, equal-weight Top-{config['portfolio']['holdings']} with rank-{config['portfolio']['retention_rank']} retention, next-open execution, and {config['portfolio']['trading_cost_bps_one_way']} bps per side. This specification was selected after preliminary opened diagnostics and therefore receives no untouched-holdout label. The residual score is a market-only daily proxy for the paper's Fama–French residual momentum; it must not be described as an exact paper replication.

## Primary result

{metric_table}

Incremental performance versus exact total-return 12–1 momentum:

{compare_table}

## What this run tests

- Exact 253/21-session total momentum and a {config['signal']['beta_window_sessions']}-session rolling market-beta residual score, with beta estimation ending at the one-month endpoint.
- Residual formation returns are summed over the same exact 12–1 endpoints and divided by their own volatility; the regression alpha is excluded.
- All signal inputs must be available before the decision timestamp. Trades enter at the next session open and exit at the following rebalance open.
- Every kept month contains {config['universe']['minimum_complete_cross_section']}–{config['universe']['monthly_size']} fully scored liquid names before Top-{config['portfolio']['holdings']} selection. Initial entry and terminal liquidation are charged.
- Three reported signal candidates sit inside a broader disclosed opened search family of {config['validation']['multiple_testing_family_size']} tests; Bonferroni results are saved separately.

## Coverage

- Label-independent signal months: {len(kept)}
- Fully realized evaluation months: {len(evaluable)}
- First / last signal: {kept['signal_date'].min()} / {kept['signal_date'].max()}
- Residual score rows: {len(scores):,}; distinct tickers: {scores['ticker'].nunique():,}
- Median monthly ADV cutoff: ${kept['monthly_adv_cutoff_usd'].median():,.0f}
- Random-null trials: {int(null_summary['simulations'].max()):,} per reported period, matched exactly to the primary candidate's month-to-month name overlap. Realized turnover is diagnostic and is not claimed to be exactly matched.

## Deployability gates

{gate_table}

Failed gates: {', '.join(failed)}.

## Interpretation

The relevant question is whether residualization improves the already-strong liquid-universe momentum baseline before 2026, not whether 2026H1 is large. A good result needs positive pre-2026 active return, a corrected HAC statistic, a positive block-bootstrap lower tail, monotone deciles, survival at 50 bps/side, and lower common-risk exposure. Even if those pass, the two fatal data gates and the absent prospective record keep the verdict INVALID.

The nonlinear ML model should only be retrained after this signal test: add residual momentum as a feature and predict beta/size-residual next-month returns, then require chronological OOS, held-out-stock OOS, ablation versus this deterministic baseline, and exact label-permutation retrains. This avoids asking a flexible model to rediscover a noisy definition of momentum while the universe remains contaminated.

The first independent event sleeve should use SEC Form 4 open-market purchases. The event clock starts at filing acceptance/filing availability, never the insider's transaction date; enter next open, hold 20 sessions, and compare with contemporaneous matched controls. Current cache lacks transaction details, so no event-P&L is claimed here.

## Charts

![Net wealth](charts/cumulative_net_wealth.png)

![Active return](charts/cumulative_active_return.png)

![Factor deciles](charts/pre2026_factor_deciles.png)

![Exposure tilts](charts/holdings_exposure_tilts.png)

## Evidence files

- `metrics_by_period.csv`, `incremental_vs_total_momentum.csv`, `factor_decile_summary.csv`
- `cost_sensitivity.csv`, `capacity_summary.csv`, `overlap_matched_random_null_summary.csv`
- `signal_universe_panel.parquet`, `evaluation_universe_panel.parquet`, `monthly_universe_audit.csv`
- `target_holdings_*.parquet`, `latest_target_holdings_*.csv`, `opened_search_space_ledger.csv`
- `residual_signal_audit.csv`, `manual_reconstruction.json`
- `deployability_gates.csv`, `multiple_testing.csv`, `artifact_manifest.json`
"""
    path = output_dir / "report-source.md"
    path.write_text(report, encoding="utf-8")
    return path


def _artifact_manifest(
    output_dir: Path,
    *,
    config_path: Path,
    input_paths: list[Path],
    code_paths: list[Path],
    source_tree_sha256: str,
    source_file_count: int,
) -> Path:
    records = []
    for role, path in [
        ("config", config_path),
        *[("input", value) for value in input_paths],
        *[("code", value) for value in code_paths],
    ]:
        records.append(
            {
                "role": role,
                "path": path.resolve().as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            records.append(
                {
                    "role": "output",
                    "path": path.resolve().as_posix(),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source_tree_sha256": source_tree_sha256,
        "source_file_count": source_file_count,
        "records": records,
    }
    path = output_dir / "artifact_manifest.json"
    _write_json(path, payload)
    return path


def run(
    config_path: str | Path,
    *,
    random_simulations: int | None = None,
    output_override: str | Path | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    panel_path = ROOT / config["research"]["source_panel"]
    daily_path = ROOT / config["research"]["source_daily_prices"]
    output_dir = (
        Path(output_override).resolve()
        if output_override is not None
        else ROOT / config["research"]["output_dir"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    daily = pd.read_parquet(daily_path)
    panel = pd.read_parquet(panel_path)
    decisions = (
        panel.loc[:, ["date", "signal_date"]]
        .drop_duplicates()
        .rename(columns={"date": "formation_date", "signal_date": "signal_at"})
        .sort_values("formation_date")
    )
    decisions = decisions.loc[
        pd.to_datetime(decisions["signal_at"], utc=True).ge(_utc(config["research"]["first_signal_date"]))
    ]
    scores, signal_audit = compute_market_residual_momentum(
        daily,
        decisions,
        benchmark_ticker=config["research"]["benchmark_ticker"],
        beta_window_sessions=int(config["signal"]["beta_window_sessions"]),
        minimum_beta_observations=int(config["signal"]["minimum_beta_observations"]),
        formation_sessions=int(config["signal"]["formation_sessions"]),
        skip_sessions=int(config["signal"]["skip_sessions"]),
        minimum_formation_observations=int(config["signal"]["minimum_formation_observations"]),
    )
    signal_universe, universe, universe_audit = _complete_liquid_universe(
        panel, scores, config
    )
    base_cost = float(config["portfolio"]["trading_cost_bps_one_way"])
    metrics, returns_by_strategy, holdings_by_strategy = _run_periods(
        universe, config, cost_bps=base_cost
    )
    incremental = _incremental_metrics(universe, config, cost_bps=base_cost)
    costs = _cost_sensitivity(universe, config)
    monthly_ic = pd.concat(
        [
            _monthly_ic(universe, score_column, strategy)
            for strategy, score_column in STRATEGIES.items()
        ],
        ignore_index=True,
    )
    ic_summary = (
        monthly_ic.groupby("strategy", as_index=False)
        .agg(months=("rank_ic", "size"), mean_rank_ic=("rank_ic", "mean"))
    )
    ic_t = monthly_ic.groupby("strategy")["rank_ic"].apply(newey_west_tstat)
    ic_summary["rank_ic_hac_tstat"] = ic_summary["strategy"].map(ic_t)
    monthly_deciles, decile_summary = _decile_results(
        universe, config["validation"]["periods"]
    )
    exposure_columns = [
        "market_beta_signal",
        "beta_12m",
        "log_market_cap",
        "log_dollar_volume_3m",
        "idio_vol_12m",
    ]
    exposure_summary = _exposure_summary(universe, holdings_by_strategy, exposure_columns)
    capacity = _capacity_table(holdings_by_strategy, config)
    actual_simulations = int(
        random_simulations
        if random_simulations is not None
        else config["validation"]["turnover_matched_random_portfolios"]
    )
    if actual_simulations <= 0:
        raise ValueError("random simulations must be positive")
    null_trials, null_summary = _overlap_matched_random_null(
        universe, config, simulations=actual_simulations
    )
    multiple_testing = _multiple_testing(
        metrics, int(config["validation"]["multiple_testing_family_size"])
    )
    gates = _gate_table(
        metrics,
        incremental,
        costs,
        decile_summary,
        capacity,
        null_summary,
        universe_audit,
        config,
    )
    manual_portfolio = manual_first_month_reconstruction(
        universe,
        returns_by_strategy[config["development_family"]["primary_candidate"]],
        holdings_by_strategy[config["development_family"]["primary_candidate"]],
        cost_bps=base_cost,
    )
    manual_signal = _manual_signal_reconstruction(daily, scores, signal_audit, config)
    manual = {
        "signal": manual_signal,
        "portfolio": manual_portfolio,
        "all_exact": bool(manual_signal["exact_match"] and manual_portfolio["exact_match"]),
    }
    if not manual["all_exact"]:
        raise RuntimeError("manual reconstruction failed")
    yearly = _yearly_results(returns_by_strategy)

    scores.to_parquet(output_dir / "residual_momentum_scores.parquet", index=False)
    signal_audit.to_csv(output_dir / "residual_signal_audit.csv", index=False)
    signal_universe.to_parquet(output_dir / "signal_universe_panel.parquet", index=False)
    universe.to_parquet(output_dir / "evaluation_universe_panel.parquet", index=False)
    # Compatibility alias for downstream diagnostics. Historical metrics use
    # only the whole-month evaluation panel above.
    universe.to_parquet(output_dir / "liquid_universe_panel.parquet", index=False)
    for strategy, score_column in STRATEGIES.items():
        targets = _build_label_independent_target_holdings(
            signal_universe,
            score_column=score_column,
            holdings_count=int(config["portfolio"]["holdings"]),
            retention_rank=int(config["portfolio"]["retention_rank"]),
        )
        targets.to_parquet(output_dir / f"target_holdings_{strategy}.parquet", index=False)
        latest_signal = targets["signal_date"].max()
        targets.loc[targets["signal_date"].eq(latest_signal)].to_csv(
            output_dir / f"latest_target_holdings_{strategy}.csv", index=False
        )
    search_ledger = _opened_search_space_ledger()
    if len(search_ledger) != int(config["validation"]["multiple_testing_family_size"]):
        raise RuntimeError("declared multiple-testing family does not match search-space ledger")
    search_ledger.to_csv(output_dir / "opened_search_space_ledger.csv", index=False)
    universe_audit.to_csv(output_dir / "monthly_universe_audit.csv", index=False)
    metrics.to_csv(output_dir / "metrics_by_period.csv", index=False)
    incremental.to_csv(output_dir / "incremental_vs_total_momentum.csv", index=False)
    costs.to_csv(output_dir / "cost_sensitivity.csv", index=False)
    monthly_ic.to_csv(output_dir / "monthly_ic.csv", index=False)
    ic_summary.to_csv(output_dir / "ic_summary.csv", index=False)
    monthly_deciles.to_csv(output_dir / "factor_deciles_monthly.csv", index=False)
    decile_summary.to_csv(output_dir / "factor_decile_summary.csv", index=False)
    exposure_summary.to_csv(output_dir / "exposure_summary.csv", index=False)
    capacity.to_csv(output_dir / "capacity_summary.csv", index=False)
    null_trials.to_parquet(output_dir / "overlap_matched_random_null_trials.parquet", index=False)
    null_summary.to_csv(output_dir / "overlap_matched_random_null_summary.csv", index=False)
    multiple_testing.to_csv(output_dir / "multiple_testing.csv", index=False)
    gates.to_csv(output_dir / "deployability_gates.csv", index=False)
    yearly.to_csv(output_dir / "yearly_results.csv", index=False)
    for strategy, returns in returns_by_strategy.items():
        returns.to_csv(output_dir / f"returns_{strategy}.csv", index=False)
    for strategy, holdings in holdings_by_strategy.items():
        holdings.to_parquet(output_dir / f"holdings_{strategy}.parquet", index=False)
    _write_json(output_dir / "manual_reconstruction.json", manual)
    _write_json(
        output_dir / "run_summary.json",
        {
            "historical_backtest_verdict": "INVALID",
            "evidence_status": config["research"]["evidence_status"],
            "primary_candidate": config["development_family"]["primary_candidate"],
            "random_simulations_per_period": actual_simulations,
            "passed_gates": int(gates["passed"].sum()),
            "total_gates": len(gates),
            "fatal_failed_gates": [
                value
                for value in ["upstream_survivorship_free", "upstream_delisting_complete", "complete_historical_trial_ledger", "prospective_months"]
                if not bool(gates.set_index("gate").loc[value, "passed"])
            ],
        },
    )
    _charts(output_dir, returns_by_strategy, decile_summary, exposure_summary)
    report_path = _write_report(
        output_dir,
        config,
        metrics,
        incremental,
        decile_summary,
        costs,
        gates,
        null_summary,
        universe_audit,
        scores,
    )
    source_tree_sha256, source_file_count = sha256_directory(ROOT / "src")
    manifest_path = _artifact_manifest(
        output_dir,
        config_path=config_path,
        input_paths=[panel_path, daily_path],
        code_paths=[Path(__file__).resolve(), ROOT / "proxy_research" / "evaluate.py"],
        source_tree_sha256=source_tree_sha256,
        source_file_count=source_file_count,
    )
    return {
        "output_dir": output_dir,
        "report": report_path,
        "manifest": manifest_path,
        "historical_backtest_verdict": "INVALID",
        "signal_months": int(signal_universe["signal_date"].nunique()),
        "kept_months": int(universe["signal_date"].nunique()),
        "random_simulations_per_period": actual_simulations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/liquid_residual_momentum_v2.yml")
    parser.add_argument("--random-simulations", type=int)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.config,
                random_simulations=args.random_simulations,
                output_override=args.output_dir,
            ),
            indent=2,
            default=_json_default,
        )
    )


if __name__ == "__main__":
    main()
