"""Robustness tests for broad cross-sectional alpha research."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from alpha_factor_engine.research.models import (
    ModelConfig,
    NonlinearEnsemble,
    prepare_model_panel,
    rank_and_combine_components,
)


def _stable_bucket(value: object, *, buckets: int, seed: int) -> int:
    payload = f"{seed}|{value}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % buckets


def assign_security_folds(
    frame: pd.DataFrame,
    *,
    security_column: str = "permno",
    folds: int = 5,
    seed: int = 929,
) -> pd.DataFrame:
    """Assign each permanent security identifier to exactly one stable fold."""

    if folds < 2:
        raise ValueError("folds must be at least 2")
    if security_column not in frame:
        raise ValueError(f"missing security column: {security_column}")
    if frame[security_column].isna().any():
        raise ValueError(f"{security_column} contains missing identifiers")
    securities = pd.Series(frame[security_column].unique(), name=security_column)
    mapping = pd.DataFrame({security_column: securities})
    mapping["security_fold"] = mapping[security_column].map(
        lambda value: _stable_bucket(value, buckets=folds, seed=seed)
    )
    return mapping.sort_values(["security_fold", security_column]).reset_index(drop=True)


def permute_labels_within_month(
    frame: pd.DataFrame,
    *,
    target_column: str = "target_return",
    signal_column: str = "signal_at",
    seed: int = 929,
) -> pd.DataFrame:
    """Permute labels within month for a full retraining placebo test."""

    missing = {target_column, signal_column}.difference(frame.columns)
    if missing:
        raise ValueError(f"label permutation missing columns: {sorted(missing)}")
    output = frame.copy()
    rng = np.random.default_rng(seed)
    permuted = output[target_column].copy()
    for _, index in output.groupby(signal_column, sort=True).groups.items():
        values = output.loc[index, target_column].to_numpy(copy=True)
        if pd.isna(values).any() and not pd.isna(values).all():
            raise ValueError("partial label coverage cannot be permuted")
        if not pd.isna(values).all():
            permuted.loc[index] = values[rng.permutation(len(values))]
    output[target_column] = permuted
    output["label_permutation_seed"] = seed
    return output


def _security_strata(
    frame: pd.DataFrame,
    *,
    security_column: str,
    sector_column: str,
    size_column: str,
    size_bins: int,
) -> pd.DataFrame:
    required = {security_column, sector_column, size_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"stratification missing columns: {sorted(missing)}")
    summary = (
        frame.groupby(security_column, sort=True)
        .agg(
            median_size=(size_column, "median"),
            sector=(sector_column, lambda values: values.dropna().astype(str).mode().iloc[0] if values.notna().any() else "MISSING"),
        )
        .reset_index()
    )
    valid_size = pd.to_numeric(summary["median_size"], errors="coerce")
    if valid_size.notna().sum() < size_bins:
        raise ValueError("too few securities with valid size for requested stratification")
    rank = valid_size.rank(method="first")
    summary["size_bin"] = pd.qcut(rank, q=size_bins, labels=False, duplicates="drop")
    summary["stratum"] = summary["sector"].astype(str) + "|" + summary["size_bin"].astype(str)
    return summary


def draw_stratified_security_universes(
    frame: pd.DataFrame,
    *,
    sizes: Iterable[int] = (500, 1_000, 2_000),
    simulations: int = 200,
    security_column: str = "permno",
    sector_column: str = "historical_sector",
    size_column: str = "market_cap_usd",
    size_bins: int = 10,
    seed: int = 929,
) -> pd.DataFrame:
    """Draw stable, sector/size-stratified permanent-ID universes."""

    if simulations <= 0:
        raise ValueError("simulations must be positive")
    strata = _security_strata(
        frame,
        security_column=security_column,
        sector_column=sector_column,
        size_column=size_column,
        size_bins=size_bins,
    )
    population = len(strata)
    requested_sizes = sorted(set(int(value) for value in sizes))
    if not requested_sizes or requested_sizes[0] <= 0:
        raise ValueError("sizes must contain positive integers")
    if requested_sizes[-1] > population:
        raise ValueError(
            f"requested universe size {requested_sizes[-1]} exceeds population {population}"
        )
    groups = {key: group.copy() for key, group in strata.groupby("stratum", sort=True)}
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for requested_size in requested_sizes:
        proportions = {key: len(group) / population for key, group in groups.items()}
        for simulation in range(simulations):
            allocation = {key: min(len(groups[key]), int(np.floor(requested_size * share))) for key, share in proportions.items()}
            remaining = requested_size - sum(allocation.values())
            order = sorted(
                groups,
                key=lambda key: (requested_size * proportions[key] - allocation[key], key),
                reverse=True,
            )
            while remaining:
                progressed = False
                for key in order:
                    if allocation[key] < len(groups[key]):
                        allocation[key] += 1
                        remaining -= 1
                        progressed = True
                        if remaining == 0:
                            break
                if not progressed:
                    raise RuntimeError("stratified allocation could not reach requested size")
            for key, count in allocation.items():
                if count == 0:
                    continue
                candidates = groups[key][security_column].to_numpy()
                selected = rng.choice(candidates, size=count, replace=False)
                rows.extend(
                    {
                        "requested_size": requested_size,
                        "simulation": simulation,
                        security_column: security,
                        "stratum": key,
                    }
                    for security in selected
                )
    return pd.DataFrame(rows).sort_values(
        ["requested_size", "simulation", "stratum", security_column]
    ).reset_index(drop=True)


@dataclass(frozen=True)
class RandomPortfolioNullResult:
    trials: pd.DataFrame
    summary: dict[str, float | int | str]


def matched_random_portfolio_null(
    panel: pd.DataFrame,
    observed_returns: pd.DataFrame,
    benchmark_returns: pd.DataFrame,
    *,
    holdings_count: int,
    simulations: int = 10_000,
    security_column: str = "ticker",
    signal_column: str = "signal_at",
    target_column: str = "target_return",
    trading_cost_bps_one_way: float = 15.0,
    seed: int = 929,
    observed_holdings: pd.DataFrame | None = None,
    batch_size: int = 500,
) -> RandomPortfolioNullResult:
    """Compare an observed equal-weight top-N strategy with random top-N picks.

    Each random portfolio uses the same number of positions, rebalance dates,
    drift-aware turnover accounting, one-way cost, and terminal liquidation.
    When observed holdings are supplied, the random paths also retain the same
    count of names from one month to the next.
    """

    if simulations <= 0 or holdings_count <= 0 or batch_size <= 0:
        raise ValueError("simulations, holdings_count, and batch_size must be positive")
    required = {security_column, signal_column, target_column}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"random null panel missing columns: {sorted(missing)}")
    for label, frame in [("observed", observed_returns), ("benchmark", benchmark_returns)]:
        missing_return = {signal_column, "net_return"}.difference(frame.columns)
        if missing_return:
            raise ValueError(f"{label} returns missing columns: {sorted(missing_return)}")

    candidate = panel.copy()
    candidate[signal_column] = pd.to_datetime(candidate[signal_column], utc=True, errors="raise")
    observed_columns = [signal_column, "net_return"]
    if "turnover" in observed_returns:
        observed_columns.append("turnover")
    observed = observed_returns[observed_columns].copy()
    if "turnover" in observed:
        observed = observed.rename(columns={"turnover": "turnover_observed"})
    benchmark = benchmark_returns[[signal_column, "net_return"]].copy()
    observed[signal_column] = pd.to_datetime(observed[signal_column], utc=True, errors="raise")
    benchmark[signal_column] = pd.to_datetime(benchmark[signal_column], utc=True, errors="raise")
    aligned = observed.merge(benchmark, on=signal_column, suffixes=("_observed", "_benchmark"), validate="one_to_one")
    candidate = candidate.loc[candidate[signal_column].isin(aligned[signal_column])].copy()
    if aligned.empty:
        raise ValueError("observed and benchmark returns have no common periods")

    retention_counts: dict[pd.Timestamp, int] = {}
    if observed_holdings is not None:
        holding_missing = {signal_column, security_column}.difference(observed_holdings.columns)
        if holding_missing:
            raise ValueError(
                f"observed holdings missing columns: {sorted(holding_missing)}"
            )
        holding_frame = observed_holdings[[signal_column, security_column]].copy()
        holding_frame[signal_column] = pd.to_datetime(
            holding_frame[signal_column], utc=True, errors="raise"
        )
        prior_names: set[str] | None = None
        for signal_at in aligned[signal_column]:
            names = set(
                holding_frame.loc[
                    holding_frame[signal_column].eq(signal_at), security_column
                ].astype(str)
            )
            if len(names) != holdings_count:
                raise ValueError(
                    f"observed holdings at {signal_at} contain {len(names)} names; "
                    f"expected {holdings_count}"
                )
            retention_counts[signal_at] = 0 if prior_names is None else len(names & prior_names)
            prior_names = names

    securities = sorted(candidate[security_column].astype(str).unique())
    security_index = {security: index for index, security in enumerate(securities)}
    rng = np.random.default_rng(seed)
    grouped = {key: group for key, group in candidate.groupby(signal_column, sort=True)}
    monthly_inputs: list[tuple[pd.Timestamp, np.ndarray, np.ndarray]] = []
    for month_index, signal_at in enumerate(aligned[signal_column]):
        if signal_at not in grouped:
            raise ValueError(f"candidate panel missing observed signal month {signal_at}")
        month = grouped[signal_at].copy()
        labels = pd.to_numeric(month[target_column], errors="coerce")
        if labels.isna().any():
            raise ValueError(f"partial or missing target coverage at {signal_at}")
        if len(month) < holdings_count:
            raise ValueError(
                f"{signal_at} has {len(month)} candidates for {holdings_count} holdings"
            )
        ids = month[security_column].astype(str).map(security_index).to_numpy(int)
        if len(np.unique(ids)) != len(ids):
            raise ValueError(f"duplicate {security_column} values at {signal_at}")
        values = labels.to_numpy(float)
        monthly_inputs.append((signal_at, ids, values))

    benchmark_values = aligned["net_return_benchmark"].to_numpy(float)
    months = len(monthly_inputs)
    trial_frames: list[pd.DataFrame] = []
    for batch_start in range(0, simulations, batch_size):
        batch_count = min(batch_size, simulations - batch_start)
        previous = np.zeros((batch_count, len(securities)), dtype=float)
        previous_selected = np.zeros((batch_count, len(securities)), dtype=bool)
        monthly_net = np.empty((batch_count, months), dtype=float)
        monthly_turnover = np.empty_like(monthly_net)
        final_gross = np.zeros(batch_count, dtype=float)
        final_entry_cost = np.zeros(batch_count, dtype=float)
        for month_index, (signal_at, ids, values) in enumerate(monthly_inputs):
            month_size = len(ids)
            if observed_holdings is None or month_index == 0:
                random_keys = rng.random((batch_count, month_size))
                selected = np.argpartition(
                    random_keys, holdings_count - 1, axis=1
                )[:, :holdings_count]
                selected_local = np.zeros((batch_count, month_size), dtype=bool)
                np.put_along_axis(selected_local, selected, True, axis=1)
            else:
                desired_retained = retention_counts[signal_at]
                available_prior = previous_selected[:, ids]
                retained = np.zeros((batch_count, month_size), dtype=bool)
                if desired_retained:
                    retain_keys = rng.random((batch_count, month_size))
                    retain_keys[~available_prior] = 2.0
                    retain_positions = np.argpartition(
                        retain_keys,
                        min(desired_retained - 1, month_size - 1),
                        axis=1,
                    )[:, :desired_retained]
                    np.put_along_axis(retained, retain_positions, True, axis=1)
                    retained &= available_prior
                entrants_needed = holdings_count - retained.sum(axis=1)
                entrant_keys = rng.random((batch_count, month_size))
                entrant_keys[available_prior] = 2.0
                entrant_positions = np.argpartition(
                    entrant_keys, holdings_count - 1, axis=1
                )[:, :holdings_count]
                entrant_mask = np.arange(holdings_count)[None, :] < entrants_needed[:, None]
                entrants = np.zeros((batch_count, month_size), dtype=bool)
                np.put_along_axis(entrants, entrant_positions, entrant_mask, axis=1)
                entrants &= ~available_prior
                selected_local = retained | entrants
                if not np.all(selected_local.sum(axis=1) == holdings_count):
                    raise RuntimeError(
                        "turnover-matched random selection did not fill all holdings"
                    )
            current_local = selected_local.astype(float) / holdings_count
            current = np.zeros_like(previous)
            current[:, ids] = current_local
            turnover = np.abs(current - previous).sum(axis=1)
            gross_return = current_local @ values
            if np.any(1.0 + gross_return <= 0):
                raise ValueError(f"random portfolio wealth became non-positive at {signal_at}")
            entry_cost = turnover * trading_cost_bps_one_way / 10_000.0
            monthly_net[:, month_index] = gross_return - entry_cost
            monthly_turnover[:, month_index] = turnover
            previous = np.zeros_like(previous)
            previous[:, ids] = current_local * (1.0 + values) / (1.0 + gross_return[:, None])
            previous_selected = np.zeros_like(previous_selected)
            previous_selected[:, ids] = selected_local
            final_gross = gross_return
            final_entry_cost = entry_cost

        liquidation_turnover = np.abs(previous).sum(axis=1)
        pre_liquidation_nav = np.maximum(0.0, 1.0 + final_gross - final_entry_cost)
        liquidation_cost = (
            liquidation_turnover
            * trading_cost_bps_one_way
            / 10_000.0
            * pre_liquidation_nav
        )
        monthly_net[:, -1] -= liquidation_cost
        monthly_turnover[:, -1] += liquidation_turnover
        active = monthly_net - benchmark_values[None, :]
        annualized_return = np.prod(1.0 + monthly_net, axis=1) ** (12.0 / months) - 1.0
        volatility = monthly_net.std(axis=1, ddof=1) * np.sqrt(12.0)
        sharpe = np.divide(
            monthly_net.mean(axis=1) * 12.0,
            volatility,
            out=np.full(batch_count, np.nan),
            where=volatility > 0,
        )
        active_volatility = active.std(axis=1, ddof=1) * np.sqrt(12.0)
        active_return = active.mean(axis=1) * 12.0
        information_ratio = np.divide(
            active_return,
            active_volatility,
            out=np.full(batch_count, np.nan),
            where=active_volatility > 0,
        )
        trial_frames.append(
            pd.DataFrame(
                {
                    "simulation": np.arange(batch_start, batch_start + batch_count),
                    "annualized_return": annualized_return,
                    "sharpe": sharpe,
                    "active_annualized_return": active_return,
                    "active_information_ratio": information_ratio,
                    "average_monthly_turnover": monthly_turnover.mean(axis=1),
                }
            )
        )
    trials = pd.concat(trial_frames, ignore_index=True)

    observed_values = aligned["net_return_observed"].to_numpy(float)
    observed_active = observed_values - benchmark_values
    observed_vol = observed_values.std(ddof=1) * np.sqrt(12.0)
    observed_active_vol = observed_active.std(ddof=1) * np.sqrt(12.0)
    observed_metrics = {
        "annualized_return": float(np.prod(1.0 + observed_values) ** (12.0 / months) - 1.0),
        "sharpe": float(observed_values.mean() * 12.0 / observed_vol),
        "active_annualized_return": float(observed_active.mean() * 12.0),
        "active_information_ratio": float(
            observed_active.mean() * 12.0 / observed_active_vol
        ),
    }
    summary: dict[str, float | int | str] = {
        "status": (
            "CONDITIONAL_CURRENT_TICKER_PROXY_TURNOVER_STRUCTURE_MATCHED"
            if observed_holdings is not None
            else "CONDITIONAL_CURRENT_TICKER_PROXY_ONLY"
        ),
        "simulations": simulations,
        "months": months,
        "holdings_count": holdings_count,
        "candidate_security_count": len(securities),
        "seed": seed,
        "trading_cost_bps_one_way": float(trading_cost_bps_one_way),
        "retention_count_matched": observed_holdings is not None,
    }
    if "turnover_observed" in aligned:
        summary["observed_average_monthly_turnover"] = float(
            aligned["turnover_observed"].mean()
        )
    summary["null_average_monthly_turnover_p50"] = float(
        trials["average_monthly_turnover"].median()
    )
    for metric, observed_value in observed_metrics.items():
        null_values = trials[metric].to_numpy(float)
        summary[f"observed_{metric}"] = observed_value
        summary[f"null_{metric}_p50"] = float(np.nanquantile(null_values, 0.50))
        summary[f"null_{metric}_p95"] = float(np.nanquantile(null_values, 0.95))
        summary[f"empirical_p_value_{metric}"] = float(
            (1 + np.sum(null_values >= observed_value)) / (simulations + 1)
        )
    return RandomPortfolioNullResult(trials=trials, summary=summary)


def walk_forward_security_holdout_predict(
    panel: pd.DataFrame,
    feature_columns: list[str],
    *,
    first_prediction: str,
    folds: int = 5,
    security_column: str = "permno",
    target_column: str = "target_return",
    minimum_training_rows: int = 10_000,
    half_life_years: float = 5.0,
    seed: int = 929,
    config: ModelConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict securities whose permanent IDs never appear in that fold's fit."""

    prepared, x_columns = prepare_model_panel(
        panel,
        feature_columns,
        target_column=target_column,
    )
    if security_column not in prepared:
        raise ValueError(f"missing security column: {security_column}")
    fold_map = assign_security_folds(
        prepared,
        security_column=security_column,
        folds=folds,
        seed=seed,
    )
    prepared = prepared.merge(fold_map, on=security_column, validate="many_to_one")
    prepared["signal_date"] = pd.to_datetime(prepared["signal_date"], utc=True, errors="raise")
    prepared["exit_at"] = pd.to_datetime(prepared["exit_at"], utc=True, errors="raise")
    first = pd.Timestamp(first_prediction)
    if first.tzinfo is None:
        first = first.tz_localize("UTC")
    else:
        first = first.tz_convert("UTC")
    resolved = config or ModelConfig(seed=seed)
    predictions: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    for fold in range(folds):
        test_fold = prepared.loc[
            prepared["security_fold"].eq(fold)
            & prepared["signal_date"].ge(first)
            & prepared["eligible"].fillna(False)
        ]
        for year, test in test_fold.groupby(test_fold["signal_date"].dt.year, sort=True):
            cutoff = test["signal_date"].min()
            train = prepared.loc[
                prepared["security_fold"].ne(fold)
                & prepared["exit_at"].lt(cutoff)
                & prepared["target_rank"].notna()
                & prepared["eligible"].fillna(False)
            ].copy()
            row = {
                "security_fold": fold,
                "year": int(year),
                "cutoff": cutoff,
                "train_rows": len(train),
                "test_rows": len(test),
                "train_security_count": train[security_column].nunique(),
                "test_security_count": test[security_column].nunique(),
                "security_overlap": len(
                    set(train[security_column]).intersection(set(test[security_column]))
                ),
                "latest_label_exit_at": train["exit_at"].max() if len(train) else pd.NaT,
            }
            if len(train) < minimum_training_rows:
                row["status"] = "insufficient_history"
                audit_rows.append(row)
                continue
            ages = (cutoff - train["exit_at"]).dt.total_seconds().div(86_400).clip(lower=0)
            weights = np.power(0.5, ages.to_numpy() / (365.25 * half_life_years))
            model = NonlinearEnsemble(resolved).fit(
                train[x_columns], train["target_rank"], sample_weight=weights
            )
            components = rank_and_combine_components(
                model.predict_components(test[x_columns]), test["signal_date"], resolved
            )
            predicted = test.copy()
            for column in components:
                predicted[column] = components[column]
            predictions.append(predicted)
            row["status"] = "ok"
            audit_rows.append(row)
    if not predictions:
        raise RuntimeError("security-holdout walk-forward produced no predictions")
    output = pd.concat(predictions, ignore_index=True).sort_values(
        ["signal_date", "security_fold", security_column]
    )
    return output, pd.DataFrame(audit_rows)


def out_of_sample_cross_section_metrics(
    predictions: pd.DataFrame,
    *,
    score_column: str = "ensemble_score",
    target_column: str = "target_return",
    signal_column: str = "signal_date",
) -> tuple[dict[str, float], pd.DataFrame]:
    required = {score_column, target_column, signal_column}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"OOS metrics missing columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    squared_error = 0.0
    squared_target = 0.0
    for signal_at, group in predictions.groupby(signal_column, sort=True):
        valid = group[[score_column, target_column]].dropna()
        if len(valid) < 10:
            continue
        score_rank = rankdata(valid[score_column])
        return_rank = rankdata(valid[target_column])
        ic = float(np.corrcoef(score_rank, return_rank)[0, 1])
        ranked_score = pd.Series(score_rank, index=valid.index).rank(pct=True) * 2 - 1
        ranked_target = pd.Series(return_rank, index=valid.index).rank(pct=True) * 2 - 1
        squared_error += float(np.square(ranked_target - ranked_score).sum())
        squared_target += float(np.square(ranked_target).sum())
        decile = pd.qcut(
            pd.Series(score_rank, index=valid.index).rank(method="first"),
            10,
            labels=False,
        )
        decile_returns = valid.groupby(decile)[target_column].mean()
        rows.append(
            {
                "signal_at": signal_at,
                "n": len(valid),
                "rank_ic": ic,
                "top_minus_bottom": float(decile_returns.iloc[-1] - decile_returns.iloc[0]),
                "decile_monotonic": bool(decile_returns.is_monotonic_increasing),
            }
        )
    monthly = pd.DataFrame(rows)
    metrics = {
        "months": float(len(monthly)),
        "oos_rank_r2": float(1.0 - squared_error / squared_target) if squared_target else np.nan,
        "mean_rank_ic": float(monthly["rank_ic"].mean()) if len(monthly) else np.nan,
        "rank_ic_tstat_iid": float(
            monthly["rank_ic"].mean()
            / (monthly["rank_ic"].std(ddof=1) / np.sqrt(len(monthly)))
        )
        if len(monthly) > 1 and monthly["rank_ic"].std(ddof=1) > 0
        else np.nan,
        "mean_top_minus_bottom": float(monthly["top_minus_bottom"].mean()) if len(monthly) else np.nan,
        "monotonic_month_fraction": float(monthly["decile_monotonic"].mean()) if len(monthly) else np.nan,
    }
    return metrics, monthly


__all__ = [
    "RandomPortfolioNullResult",
    "assign_security_folds",
    "draw_stratified_security_universes",
    "matched_random_portfolio_null",
    "out_of_sample_cross_section_metrics",
    "permute_labels_within_month",
    "walk_forward_security_holdout_predict",
]
