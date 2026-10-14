"""Cross-sectionally neutralized long/short portfolios with explicit costs."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, linprog, minimize


def neutralize_score(group: pd.DataFrame, score_column: str) -> pd.Series:
    score = pd.to_numeric(group[score_column], errors="coerce")
    exposures: list[np.ndarray] = [np.ones(len(group))]
    for column in ["beta_12m", "log_market_cap"]:
        if column in group:
            values = pd.to_numeric(group[column], errors="coerce")
            values = values.fillna(values.median())
            standard_deviation = values.std(ddof=0)
            if standard_deviation > 1e-12:
                exposures.append(((values - values.mean()) / standard_deviation).to_numpy())
    if "sic_division" in group and group["sic_division"].notna().sum() >= 5:
        dummies = pd.get_dummies(group["sic_division"], drop_first=True, dtype=float)
        exposures.extend(dummies[column].to_numpy() for column in dummies)
    design = np.column_stack(exposures)
    valid = score.notna().to_numpy() & np.isfinite(design).all(axis=1)
    residual = pd.Series(np.nan, index=group.index, dtype=float)
    if valid.sum() <= design.shape[1] + 2:
        residual.loc[score.notna()] = score.loc[score.notna()] - score.loc[score.notna()].mean()
        return residual
    coefficients, *_ = np.linalg.lstsq(design[valid], score.to_numpy()[valid], rcond=None)
    residual.iloc[np.flatnonzero(valid)] = score.to_numpy()[valid] - design[valid] @ coefficients
    return residual


def _weights_for_month(
    group: pd.DataFrame,
    score_column: str,
    long_quantile: float,
    short_quantile: float,
    gross_exposure: float,
    max_name_weight: float,
    standardized_exposure_tolerance: float,
) -> pd.Series:
    scores = neutralize_score(group, score_column)
    usable = scores.dropna().sort_values()
    n_long = max(2, int(np.floor(len(usable) * long_quantile)))
    n_short = max(2, int(np.floor(len(usable) * short_quantile)))
    minimum_per_side = int(np.ceil((gross_exposure / 2) / max_name_weight - 1e-12))
    if n_long < minimum_per_side or n_short < minimum_per_side:
        raise ValueError(
            "requested quantiles cannot meet gross exposure without violating max_name_weight"
        )
    if len(usable) < n_long + n_short:
        return pd.Series(0.0, index=group.index)
    full_beta = pd.to_numeric(group["beta_12m"], errors="coerce")
    full_size = pd.to_numeric(group["log_market_cap"], errors="coerce")
    full_beta = full_beta.fillna(full_beta.median())
    full_size = full_size.fillna(full_size.median())
    beta_scale = full_beta.std(ddof=0)
    size_scale = full_size.std(ddof=0)
    beta = (
        (full_beta - full_beta.mean()) / beta_scale
        if beta_scale > 1e-12
        else full_beta * 0
    )
    size = (
        (full_size - full_size.mean()) / size_scale
        if size_scale > 1e-12
        else full_size * 0
    )
    side_targets = np.asarray([gross_exposure / 2, gross_exposure / 2])
    maximum_side_count = len(usable) // 2
    feasibility = None
    for expansion in range(maximum_side_count + 1):
        long_count = min(n_long + expansion, maximum_side_count)
        short_count = min(n_short + expansion, maximum_side_count)
        if long_count + short_count > len(usable):
            continue
        short_names = usable.index[:short_count]
        long_names = usable.index[-long_count:]
        beta_long = beta.loc[long_names].to_numpy()
        beta_short = beta.loc[short_names].to_numpy()
        size_long = size.loc[long_names].to_numpy()
        size_short = size.loc[short_names].to_numpy()
        side_matrix = np.vstack(
            [
                np.r_[np.ones(long_count), np.zeros(short_count)],
                np.r_[np.zeros(long_count), np.ones(short_count)],
            ]
        )
        exposure_rows = [
            row
            for row in [
                np.r_[beta_long, -beta_short],
                np.r_[size_long, -size_short],
            ]
            if np.linalg.norm(row) > 1e-10
        ]
        base = np.r_[
            np.repeat((gross_exposure / 2) / long_count, long_count),
            np.repeat((gross_exposure / 2) / short_count, short_count),
        ]
        exposure_matrix = np.vstack(exposure_rows) if exposure_rows else None
        feasibility = linprog(
            np.zeros_like(base),
            A_ub=(
                np.vstack([exposure_matrix, -exposure_matrix])
                if exposure_matrix is not None
                else None
            ),
            b_ub=(
                standardized_exposure_tolerance * np.ones(2 * len(exposure_rows))
                if exposure_matrix is not None
                else None
            ),
            A_eq=side_matrix,
            b_eq=side_targets,
            bounds=[(0.0, max_name_weight)] * (long_count + short_count),
            method="highs",
        )
        if feasibility.success:
            break
        if long_count == maximum_side_count and short_count == maximum_side_count:
            break
    if feasibility is None or not feasibility.success:
        message = "no disjoint candidate expansion satisfies the fixed constraints"
        if feasibility is not None:
            message = feasibility.message
        raise RuntimeError(f"beta/size-neutral weight feasibility failed: {message}")
    allocation = minimize(
        lambda values: 0.5 * float(np.square(values - base).sum()),
        feasibility.x,
        jac=lambda values: values - base,
        constraints=[
            LinearConstraint(side_matrix, side_targets, side_targets),
            *(
                [
                    LinearConstraint(
                        exposure_matrix,
                        -standardized_exposure_tolerance * np.ones(len(exposure_rows)),
                        standardized_exposure_tolerance * np.ones(len(exposure_rows)),
                    )
                ]
                if exposure_rows
                else []
            ),
        ],
        bounds=[(0.0, max_name_weight)] * (long_count + short_count),
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 1_000},
    )
    solved = allocation.x if allocation.success else feasibility.x
    if np.max(np.abs(side_matrix @ solved - side_targets)) > 1e-7:
        raise RuntimeError(f"beta/size-neutral weight solve failed: {allocation.message}")
    if exposure_rows and np.max(np.abs(exposure_matrix @ solved)) > standardized_exposure_tolerance + 1e-6:
        raise RuntimeError("weight solve violated beta/size exposure tolerance")
    weights = pd.Series(0.0, index=group.index)
    weights.loc[long_names] = solved[:long_count]
    weights.loc[short_names] = -solved[long_count:]
    if weights.abs().max() > max_name_weight + 1e-9:
        raise RuntimeError("weight solve violated max_name_weight")
    return weights


def build_long_short_portfolio(
    predictions: pd.DataFrame,
    *,
    score_column: str = "ensemble_score",
    long_quantile: float = 0.20,
    short_quantile: float = 0.20,
    gross_exposure: float = 1.0,
    max_name_weight: float = 0.10,
    trading_cost_bps_one_way: float = 15.0,
    annual_short_borrow_bps: float = 200.0,
    standardized_exposure_tolerance: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    holdings: list[pd.DataFrame] = []
    for signal_date, group in predictions.groupby("signal_date", sort=True):
        missing_labels = group["target_return"].isna()
        if missing_labels.all():
            # The final one or more signal months cannot be evaluated until the
            # next rebalance open exists. Skip the whole cross-section.
            continue
        if missing_labels.any():
            missing_names = group.loc[missing_labels, "ticker"].astype(str).tolist()
            raise RuntimeError(
                f"partial target coverage at {signal_date}; refusing to condition the portfolio "
                f"on future availability for {missing_names[:10]}"
            )
        month = group.copy()
        if len(month) < 10:
            continue
        month["weight"] = _weights_for_month(
            month,
            score_column,
            long_quantile,
            short_quantile,
            gross_exposure,
            max_name_weight,
            standardized_exposure_tolerance,
        )
        month["neutralized_score"] = neutralize_score(month, score_column)
        holdings.append(month)
    if not holdings:
        raise RuntimeError(f"no portfolios formed for {score_column}")
    holding_frame = pd.concat(holdings, ignore_index=True)
    previous: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    for signal_date, group in holding_frame.groupby("signal_date", sort=True):
        current = dict(zip(group["ticker"], group["weight"]))
        all_names = set(previous) | set(current)
        turnover = sum(abs(current.get(name, 0.0) - previous.get(name, 0.0)) for name in all_names)
        gross_return = float((group["weight"] * group["target_return"]).sum())
        short_gross = float(-group["weight"].clip(upper=0).sum())
        entry_values = pd.to_datetime(group["entry_at"].dropna(), utc=True).unique()
        exit_values = pd.to_datetime(group["exit_at"].dropna(), utc=True).unique()
        if len(entry_values) != 1 or len(exit_values) != 1:
            raise RuntimeError(
                f"inconsistent execution timestamps at {signal_date}: "
                f"{len(entry_values)} entries, {len(exit_values)} exits"
            )
        entry_at = pd.Timestamp(entry_values[0])
        exit_at = pd.Timestamp(exit_values[0])
        holding_days = max(1.0, (exit_at - entry_at).total_seconds() / 86_400)
        trading_cost = turnover * trading_cost_bps_one_way / 10_000
        borrow_cost = short_gross * annual_short_borrow_bps / 10_000 * holding_days / 365.25
        standardized_exposures: dict[str, float] = {}
        for exposure_name, column in [
            ("beta_standardized_exposure", "beta_12m"),
            ("size_standardized_exposure", "log_market_cap"),
        ]:
            values = pd.to_numeric(group[column], errors="coerce")
            values = values.fillna(values.median())
            scale = values.std(ddof=0)
            standardized = (values - values.mean()) / scale if scale > 1e-12 else values * 0
            standardized_exposures[exposure_name] = float((group["weight"] * standardized).sum())
        rows.append(
            {
                "signal_date": signal_date,
                "entry_at": entry_at,
                "exit_at": exit_at,
                "entry_date": entry_at.tz_convert("America/New_York").date().isoformat(),
                "exit_date": exit_at.tz_convert("America/New_York").date().isoformat(),
                "gross_return": gross_return,
                "trading_cost": trading_cost,
                "terminal_liquidation_turnover": 0.0,
                "terminal_liquidation_cost": 0.0,
                "borrow_cost": borrow_cost,
                "net_return": gross_return - trading_cost - borrow_cost,
                "turnover": turnover,
                "long_count": int(group["weight"].gt(0).sum()),
                "short_count": int(group["weight"].lt(0).sum()),
                "gross_exposure": float(group["weight"].abs().sum()),
                "net_exposure": float(group["weight"].sum()),
                "beta_exposure": float((group["weight"] * group["beta_12m"].fillna(0)).sum()),
                "size_exposure": float(
                    (group["weight"] * group["log_market_cap"].fillna(0)).sum()
                ),
                **standardized_exposures,
                "score_column": score_column,
            }
        )
        denominator = 1 + gross_return
        if denominator <= 0:
            raise RuntimeError(f"portfolio NAV is non-positive at {signal_date}")
        realized_returns = dict(zip(group["ticker"], group["target_return"]))
        previous = {
            name: weight * (1 + float(realized_returns[name])) / denominator
            for name, weight in current.items()
            if abs(weight) > 0
        }
    # The finite research window ends with a closeout, so charge the omitted
    # final one-way liquidation rather than assuming a costless disappearance.
    liquidation_turnover = sum(abs(weight) for weight in previous.values())
    pre_liquidation_nav = max(
        0.0,
        1.0
        + float(rows[-1]["gross_return"])
        - float(rows[-1]["trading_cost"])
        - float(rows[-1]["borrow_cost"]),
    )
    liquidation_cost = (
        liquidation_turnover
        * trading_cost_bps_one_way
        / 10_000
        * pre_liquidation_nav
    )
    rows[-1]["turnover"] = float(rows[-1]["turnover"]) + liquidation_turnover
    rows[-1]["trading_cost"] = float(rows[-1]["trading_cost"]) + liquidation_cost
    rows[-1]["terminal_liquidation_turnover"] = liquidation_turnover
    rows[-1]["terminal_liquidation_cost"] = liquidation_cost
    rows[-1]["net_return"] = float(rows[-1]["net_return"]) - liquidation_cost
    returns = pd.DataFrame(rows).sort_values("signal_date").reset_index(drop=True)
    return returns, holding_frame
