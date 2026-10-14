"""Random-universe and full-retrain placebo tests for the public proxy."""

from __future__ import annotations

import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yaml

from alpha_factor_engine.research.models import (
    NonlinearEnsemble,
    model_config_from_dict,
    prepare_model_panel,
    rank_and_combine_components,
)
from alpha_factor_engine.research.universe_validation import out_of_sample_cross_section_metrics
from proxy_research.evaluate import (
    MODEL_FEATURES,
    active_summary,
    build_equal_weight_portfolio,
)


def _load(config_path: str | Path):
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text())
    root = config_path.parent.parent
    output = root / config["data"]["output_dir"]
    return config, output


def _stratified_sample(
    universe: pd.DataFrame,
    *,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    frame = universe.copy()
    liquidity_rank = frame["trailing_63d_median_dollar_volume"].rank(method="first")
    frame["liquidity_decile"] = pd.qcut(liquidity_rank, 10, labels=False)
    frame["stratum"] = frame["exchange_code"].fillna("MISSING").astype(str) + "|" + frame["liquidity_decile"].astype(str)
    groups = {key: group for key, group in frame.groupby("stratum", sort=True)}
    exact = {key: size * len(group) / len(frame) for key, group in groups.items()}
    allocation = {key: min(len(groups[key]), int(np.floor(value))) for key, value in exact.items()}
    remaining = size - sum(allocation.values())
    order = sorted(groups, key=lambda key: (exact[key] - allocation[key], key), reverse=True)
    while remaining:
        for key in order:
            if allocation[key] < len(groups[key]):
                allocation[key] += 1
                remaining -= 1
                if remaining == 0:
                    break
    selected = []
    for key, count in allocation.items():
        selected.extend(rng.choice(groups[key]["ticker"].to_numpy(), size=count, replace=False))
    return np.asarray(selected)


def run_random_universes(config_path: str | Path) -> Path:
    config, output = _load(config_path)
    predictions = pd.read_parquet(output / "heldout_stock_predictions.parquet")
    universe = pd.read_csv(output / "data/frozen_proxy_universe.csv")
    simulations = int(config["validation"]["random_universe_simulations"])
    seed = int(config["validation"]["random_seed"])
    cost = float(config["portfolio"]["trading_cost_bps_one_way"])
    holdings = int(config["portfolio"]["holdings"])
    rng = np.random.default_rng(seed)
    rows = []
    selections = []
    for size in config["validation"]["random_universe_sizes"]:
        for simulation in range(simulations):
            tickers = _stratified_sample(universe, size=int(size), rng=rng)
            sample = predictions.loc[predictions["ticker"].isin(tickers)].copy()
            counts = sample.groupby("signal_date")["ticker"].size()
            valid_dates = counts[counts.ge(holdings)].index
            sample = sample.loc[sample["signal_date"].isin(valid_dates)]
            strategy, _ = build_equal_weight_portfolio(
                sample, score_column="ensemble_score", holdings_count=holdings,
                trading_cost_bps_one_way=cost,
            )
            benchmark, _ = build_equal_weight_portfolio(
                sample, score_column=None, holdings_count=None,
                trading_cost_bps_one_way=cost,
            )
            metrics = active_summary(strategy, benchmark)
            rows.append({
                "requested_size": int(size), "simulation": simulation,
                "sampled_tickers": len(tickers), "months": len(strategy), **metrics,
            })
            selections.extend(
                {"requested_size": int(size), "simulation": simulation, "ticker": ticker}
                for ticker in tickers
            )
            if (simulation + 1) % 25 == 0:
                print(f"random universe size={size}: {simulation + 1}/{simulations}", flush=True)
    trials = pd.DataFrame(rows)
    trials.to_csv(output / "random_universe_trials.csv", index=False)
    pd.DataFrame(selections).to_parquet(output / "random_universe_members.parquet", index=False)
    summary = (
        trials.groupby("requested_size")
        .agg(
            trials=("simulation", "size"),
            median_active_return=("active_annualized_arithmetic_return", "median"),
            p05_active_return=("active_annualized_arithmetic_return", lambda x: x.quantile(0.05)),
            positive_active_fraction=("active_annualized_arithmetic_return", lambda x: x.gt(0).mean()),
            median_information_ratio=("information_ratio", "median"),
        )
        .reset_index()
    )
    summary.to_csv(output / "random_universe_summary.csv", index=False)
    return output / "random_universe_summary.csv"


def _fit_fixed_cut(
    train: pd.DataFrame,
    test: pd.DataFrame,
    x_columns: list[str],
    *,
    target: pd.Series,
    cutoff: pd.Timestamp,
    half_life_years: float,
    model_config,
    single_thread_trees: bool = False,
) -> tuple[pd.DataFrame, NonlinearEnsemble]:
    ages = (cutoff - train["exit_at"]).dt.total_seconds().div(86_400).clip(lower=0)
    weights = np.power(0.5, ages.to_numpy() / (365.25 * half_life_years))
    model = NonlinearEnsemble(model_config)
    if single_thread_trees:
        model.extra_trees.set_params(n_jobs=1)
    model.fit(train[x_columns], target, sample_weight=weights)
    components = rank_and_combine_components(
        model.predict_components(test[x_columns]), test["signal_date"], model_config
    )
    return components, model


def run_label_permutations(config_path: str | Path) -> Path:
    config, output = _load(config_path)
    panel = pd.read_parquet(output / "data/model_eligible_panel.parquet")
    prepared, x_columns = prepare_model_panel(panel, MODEL_FEATURES)
    prepared["signal_date"] = pd.to_datetime(prepared["signal_date"], utc=True)
    prepared["exit_at"] = pd.to_datetime(prepared["exit_at"], utc=True)
    cutoff = pd.Timestamp(config["research"]["untouched_holdout_start"], tz="UTC")
    train = prepared.loc[
        prepared["exit_at"].lt(cutoff) & prepared["target_rank"].notna()
    ].copy()
    test = prepared.loc[prepared["signal_date"].ge(cutoff)].copy()
    model_config = model_config_from_dict({**config["model"], "seed": config["research"]["seed"]})
    half_life = float(config["model"]["recency_half_life_years"])
    observed_components, observed_model = _fit_fixed_cut(
        train, test, x_columns, target=train["target_rank"], cutoff=cutoff,
        half_life_years=half_life, model_config=model_config,
    )
    observed = test.copy()
    observed["ensemble_score"] = observed_components["ensemble_score"]
    observed_metrics, observed_monthly = out_of_sample_cross_section_metrics(observed)
    observed_monthly.to_csv(output / "fixed_cut_observed_monthly_ic.csv", index=False)
    cost = float(config["portfolio"]["trading_cost_bps_one_way"])
    holding_count = int(config["portfolio"]["holdings"])
    observed_returns, _ = build_equal_weight_portfolio(
        observed, score_column="ensemble_score", holdings_count=holding_count,
        trading_cost_bps_one_way=cost,
    )
    fixed_benchmark, _ = build_equal_weight_portfolio(
        observed, score_column=None, holdings_count=None,
        trading_cost_bps_one_way=cost,
    )
    observed_active = active_summary(observed_returns, fixed_benchmark)
    observed_returns.to_csv(output / "fixed_cut_model_returns.csv", index=False)
    fixed_benchmark.to_csv(output / "fixed_cut_benchmark_returns.csv", index=False)
    observed_model.feature_importance(x_columns).to_csv(
        output / "fixed_cut_feature_importance.csv", index=False
    )

    path = output / "label_permutation_trials.csv"
    if path.is_file():
        trials = pd.read_csv(path)
        for column in ["active_annualized_arithmetic_return", "information_ratio"]:
            if column not in trials:
                trials[column] = np.nan
    else:
        trials = pd.DataFrame(columns=[
            "simulation", "seed", "mean_rank_ic",
            "active_annualized_arithmetic_return", "information_ratio",
        ])
    completed = set(
        pd.to_numeric(
            trials.loc[trials["active_annualized_arithmetic_return"].notna(), "simulation"],
            errors="coerce",
        ).dropna().astype(int)
    )
    total = int(config["validation"]["full_retrain_label_permutations"])
    base_seed = int(config["validation"]["random_seed"])
    groups = [indices.to_numpy() for _, indices in train.groupby("signal_date", sort=True).groups.items()]
    remaining = [simulation for simulation in range(total) if simulation not in completed]

    def run_trial(simulation: int) -> dict[str, float | int]:
        rng = np.random.default_rng(base_seed + 10_000 + simulation)
        permuted = train["target_rank"].copy()
        for indices in groups:
            permuted.loc[indices] = permuted.loc[indices].to_numpy()[rng.permutation(len(indices))]
        components, _ = _fit_fixed_cut(
            train, test, x_columns, target=permuted, cutoff=cutoff,
            half_life_years=half_life, model_config=model_config,
            single_thread_trees=True,
        )
        placebo = test.copy()
        placebo["ensemble_score"] = components["ensemble_score"]
        metrics, _ = out_of_sample_cross_section_metrics(placebo)
        placebo_returns, _ = build_equal_weight_portfolio(
            placebo, score_column="ensemble_score", holdings_count=holding_count,
            trading_cost_bps_one_way=cost,
        )
        portfolio_metrics = active_summary(placebo_returns, fixed_benchmark)
        return {
            "simulation": simulation,
            "seed": base_seed + 10_000 + simulation,
            "mean_rank_ic": metrics["mean_rank_ic"],
            "active_annualized_arithmetic_return": portfolio_metrics["active_annualized_arithmetic_return"],
            "information_ratio": portfolio_metrics["information_ratio"],
        }

    finished = len(completed)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(run_trial, simulation): simulation for simulation in remaining}
        for future in as_completed(futures):
            result = future.result()
            trials = trials.loc[~pd.to_numeric(trials["simulation"], errors="coerce").eq(result["simulation"])]
            trials = pd.concat([trials, pd.DataFrame([result])], ignore_index=True)
            finished += 1
            if finished % 5 == 0 or finished == total:
                trials.sort_values("simulation").to_csv(path, index=False)
                print(f"label permutation retrains: {finished}/{total}", flush=True)
    null = trials["mean_rank_ic"].astype(float)
    null_active = trials["active_annualized_arithmetic_return"].astype(float)
    observed_ic = float(observed_metrics["mean_rank_ic"])
    summary = {
        "test_design": "one_fixed_2023_cut_full_ensemble_retrain_per_permutation",
        "permutations": len(trials),
        "observed_mean_rank_ic": observed_ic,
        "null_mean_rank_ic_p50": float(null.median()),
        "null_mean_rank_ic_p95": float(null.quantile(0.95)),
        "empirical_p_value": float((1 + null.ge(observed_ic).sum()) / (len(null) + 1)),
        "observed_active_annualized_arithmetic_return": float(
            observed_active["active_annualized_arithmetic_return"]
        ),
        "null_active_return_p50": float(null_active.median()),
        "null_active_return_p95": float(null_active.quantile(0.95)),
        "empirical_p_value_active_return": float(
            (1 + null_active.ge(observed_active["active_annualized_arithmetic_return"]).sum())
            / (len(null_active) + 1)
        ),
        "train_rows": len(train),
        "test_rows": len(test),
    }
    (output / "label_permutation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return output / "label_permutation_summary.json"


__all__ = ["run_label_permutations", "run_random_universes"]
