"""Diversified, point-in-time momentum portfolio construction.

This module intentionally stays separate from the v1 champion.  It ranks a
frozen cross-section deterministically, targets inverse-volatility weights,
and solves the smallest adjustment needed to satisfy hard portfolio limits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, linprog, minimize


BASE_COLUMNS = {
    "signal_at",
    "ticker",
    "entry_at",
    "exit_at",
    "entry_price",
    "exit_price",
}


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _aware_utc(values: pd.Series, label: str) -> pd.Series:
    converted: list[pd.Timestamp] = []
    for index, value in values.items():
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} has an invalid timestamp at row {index}: {value!r}") from error
        if pd.isna(timestamp):
            raise ValueError(f"{label} has a missing timestamp at row {index}")
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError(f"{label} must be timezone-aware at row {index}")
        converted.append(timestamp.tz_convert("UTC"))
    return pd.Series(converted, index=values.index, dtype="datetime64[ns, UTC]")


def _validate_open_clock(values: pd.Series, label: str) -> None:
    local = values.dt.tz_convert("America/New_York")
    invalid = ~(
        local.dt.hour.eq(9)
        & local.dt.minute.eq(30)
        & local.dt.second.eq(0)
        & local.dt.microsecond.eq(0)
        & local.dt.nanosecond.eq(0)
    )
    if invalid.any():
        example = values.loc[invalid].iloc[0]
        raise ValueError(f"{label} must use the exact 09:30 America/New_York open; got {example}")


def _solve_constrained_inverse_volatility(
    selected: pd.DataFrame,
    *,
    volatility_column: str,
    beta_column: str,
    max_name_weight: float,
    beta_exposure_limit: float,
    sector_column: str | None,
    max_sector_weight: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    volatility = selected[volatility_column].to_numpy(float)
    inverse_volatility = 1.0 / volatility
    base = inverse_volatility / inverse_volatility.sum()
    beta = selected[beta_column].to_numpy(float)
    count = len(selected)

    sector_matrix: np.ndarray | None = None
    if sector_column is not None:
        sectors = sorted(selected[sector_column].astype(str).unique())
        sector_matrix = np.vstack(
            [selected[sector_column].astype(str).eq(sector).to_numpy(float) for sector in sectors]
        )

    base_name_ok = bool(base.max() <= max_name_weight + 1e-12)
    base_beta_ok = bool(abs(float(beta @ base)) <= beta_exposure_limit + 1e-12)
    base_sector_ok = bool(
        sector_matrix is None
        or np.max(sector_matrix @ base) <= float(max_sector_weight) + 1e-12
    )
    if base_name_ok and base_beta_ok and base_sector_ok:
        return base, base

    inequality_rows: list[np.ndarray] = [beta, -beta]
    inequality_limits: list[float] = [beta_exposure_limit, beta_exposure_limit]
    if sector_matrix is not None:
        inequality_rows.extend(sector_matrix)
        inequality_limits.extend([float(max_sector_weight)] * len(sector_matrix))
    inequality = np.vstack(inequality_rows)
    limits = np.asarray(inequality_limits, dtype=float)
    equality = np.ones((1, count), dtype=float)
    feasibility = linprog(
        np.zeros(count, dtype=float),
        A_ub=inequality,
        b_ub=limits,
        A_eq=equality,
        b_eq=np.ones(1),
        bounds=[(0.0, max_name_weight)] * count,
        method="highs",
    )
    if not feasibility.success:
        raise ValueError(
            "diversified momentum constraints are infeasible for the selected top-N names: "
            f"{feasibility.message}"
        )

    constraints = [
        LinearConstraint(equality, np.ones(1), np.ones(1)),
        LinearConstraint(inequality, -np.inf * np.ones(len(limits)), limits),
    ]
    optimized = minimize(
        lambda weights: 0.5 * float(np.square(weights - base).sum()),
        feasibility.x,
        jac=lambda weights: weights - base,
        constraints=constraints,
        bounds=[(0.0, max_name_weight)] * count,
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 2_000},
    )
    if not optimized.success:
        raise RuntimeError(f"inverse-volatility weight optimization failed: {optimized.message}")
    weights = np.asarray(optimized.x, dtype=float)
    if abs(weights.sum() - 1.0) > 1e-7:
        raise RuntimeError("optimized diversified momentum weights do not sum to one")
    if weights.max() > max_name_weight + 1e-7:
        raise RuntimeError("optimized diversified momentum weights violate max_name_weight")
    if abs(float(beta @ weights)) > beta_exposure_limit + 1e-7:
        raise RuntimeError("optimized diversified momentum weights violate beta_exposure_limit")
    if sector_matrix is not None and np.max(sector_matrix @ weights) > float(max_sector_weight) + 1e-7:
        raise RuntimeError("optimized diversified momentum weights violate max_sector_weight")
    return weights, base


def build_diversified_momentum_portfolio(
    candidates: pd.DataFrame,
    *,
    score_column: str = "momentum_12_1",
    volatility_column: str = "realized_vol_3m",
    beta_column: str = "beta_12m",
    feature_available_at_column: str = "price_available_at",
    top_n: int = 12,
    retention_rank: int | None = None,
    max_name_weight: float = 0.12,
    beta_exposure_limit: float = 1.10,
    sector_column: str | None = None,
    sector_available_at_column: str | None = None,
    max_sector_weight: float | None = None,
    trading_cost_bps_one_way: float = 15.0,
    charge_terminal_liquidation: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a constrained inverse-volatility top-N momentum portfolio.

    ``feature_available_at_column`` must timestamp the price-derived score,
    volatility, and beta observations.  Sector constraints are enabled only
    when a sector label and its own point-in-time availability timestamp are
    both supplied.  Costs use absolute traded weight, drifted pretrade weights,
    and an ending-NAV terminal liquidation, matching the v1 accounting.

    ``retention_rank`` enables a deterministic rank buffer.  Existing names
    are retained while their current full-universe rank is at or above that
    cutoff; vacated slots are filled from the best-ranked non-retained names.
    Selection always contains exactly ``top_n`` names and depends only on prior
    membership, never prior weights or return drift.  ``None`` is equivalent
    to ``top_n`` (an unbuffered top-N portfolio).
    """

    required = BASE_COLUMNS | {
        score_column,
        volatility_column,
        beta_column,
        feature_available_at_column,
    }
    sector_arguments = [sector_column, sector_available_at_column, max_sector_weight]
    sector_enabled = any(item is not None for item in sector_arguments)
    if sector_enabled and not all(item is not None for item in sector_arguments):
        raise ValueError(
            "sector constraints require sector_column, sector_available_at_column, "
            "and max_sector_weight together"
        )
    if sector_enabled:
        required |= {str(sector_column), str(sector_available_at_column)}
    _require_columns(candidates, required, "candidates")

    if not isinstance(top_n, int) or top_n <= 0:
        raise ValueError("top_n must be a positive integer")
    effective_retention_rank = top_n if retention_rank is None else retention_rank
    if not isinstance(effective_retention_rank, int) or effective_retention_rank < top_n:
        raise ValueError("retention_rank must be an integer greater than or equal to top_n")
    if not np.isfinite(max_name_weight) or not 0 < max_name_weight <= 1:
        raise ValueError("max_name_weight must lie in (0, 1]")
    if top_n * max_name_weight < 1 - 1e-12:
        raise ValueError("top_n and max_name_weight cannot support 100% invested exposure")
    if not np.isfinite(beta_exposure_limit) or beta_exposure_limit < 0:
        raise ValueError("beta_exposure_limit must be finite and non-negative")
    if sector_enabled and (
        not np.isfinite(float(max_sector_weight)) or not 0 < float(max_sector_weight) <= 1
    ):
        raise ValueError("max_sector_weight must lie in (0, 1]")
    if sector_enabled and str(sector_column).strip().casefold() == "sic_division_current":
        raise ValueError(
            "sic_division_current is non-PIT metadata and cannot be used for sector constraints"
        )
    if not np.isfinite(trading_cost_bps_one_way) or trading_cost_bps_one_way < 0:
        raise ValueError("trading_cost_bps_one_way must be finite and non-negative")

    panel = candidates.copy()
    panel["ticker"] = panel["ticker"].astype(str)
    if panel["ticker"].str.strip().eq("").any():
        raise ValueError("candidates.ticker contains an empty identifier")
    for column in ["signal_at", "entry_at", "exit_at", feature_available_at_column]:
        panel[column] = _aware_utc(panel[column], f"candidates.{column}")
    if sector_enabled:
        panel[str(sector_available_at_column)] = _aware_utc(
            panel[str(sector_available_at_column)],
            f"candidates.{sector_available_at_column}",
        )
        sector_values = panel[str(sector_column)]
        if sector_values.isna().any() or sector_values.astype(str).str.strip().eq("").any():
            raise ValueError("PIT sector labels must be complete and non-empty")

    feature_future = panel[feature_available_at_column].gt(panel["signal_at"])
    if feature_future.any():
        example = panel.loc[feature_future, ["ticker", "signal_at", feature_available_at_column]].iloc[0]
        raise ValueError(
            f"price-derived features were unavailable at signal_at for {example['ticker']}: "
            f"{example[feature_available_at_column]} > {example['signal_at']}"
        )
    if sector_enabled:
        sector_future = panel[str(sector_available_at_column)].gt(panel["signal_at"])
        if sector_future.any():
            example = panel.loc[
                sector_future,
                ["ticker", "signal_at", str(sector_available_at_column)],
            ].iloc[0]
            raise ValueError(
                f"PIT sector label was unavailable at signal_at for {example['ticker']}"
            )

    invalid_order = ~(
        panel["signal_at"].lt(panel["entry_at"])
        & panel["entry_at"].lt(panel["exit_at"])
    )
    if invalid_order.any():
        example = panel.loc[invalid_order, ["signal_at", "entry_at", "exit_at"]].iloc[0]
        raise ValueError(
            "execution timestamps must satisfy signal_at < entry_at < exit_at; "
            f"got {example.to_dict()}"
        )
    _validate_open_clock(panel["entry_at"], "candidates.entry_at")
    _validate_open_clock(panel["exit_at"], "candidates.exit_at")

    for column in [score_column, volatility_column, beta_column, "entry_price", "exit_price"]:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
        if (~np.isfinite(panel[column])).any():
            raise ValueError(f"candidates.{column} contains missing or non-finite values")
    if panel[volatility_column].le(0).any():
        raise ValueError(f"candidates.{volatility_column} must be strictly positive")
    if panel[["entry_price", "exit_price"]].le(0).any(axis=None):
        raise ValueError("entry_price and exit_price must be strictly positive")
    duplicates = panel.duplicated(["signal_at", "ticker"], keep=False)
    if duplicates.any():
        example = panel.loc[duplicates, ["signal_at", "ticker"]].iloc[0]
        raise ValueError(f"duplicate signal/ticker candidate: {example.to_dict()}")

    schedule_rows: list[dict[str, pd.Timestamp]] = []
    holdings_frames: list[pd.DataFrame] = []
    prior_selected_tickers: set[str] | None = None
    for signal_at, group in panel.groupby("signal_at", sort=True):
        if len(group) < top_n:
            raise ValueError(
                f"signal_at={signal_at} has {len(group)} candidates; top-{top_n} requires at least {top_n}"
            )
        if group["entry_at"].nunique() != 1 or group["exit_at"].nunique() != 1:
            raise ValueError(f"signal_at={signal_at} has inconsistent entry_at or exit_at timestamps")
        ordered = group.sort_values(
            [score_column, "ticker"],
            ascending=[False, True],
            kind="mergesort",
        ).copy()
        ordered["universe_rank"] = np.arange(1, len(ordered) + 1)
        if prior_selected_tickers is None:
            selected = ordered.head(top_n).copy()
            selected["selection_reason"] = "initial_top_n"
        else:
            retain_mask = (
                ordered["ticker"].isin(prior_selected_tickers)
                & ordered["universe_rank"].le(effective_retention_rank)
            )
            retained = ordered.loc[retain_mask].copy()
            retained["selection_reason"] = "retained_within_buffer"
            slots = top_n - len(retained)
            entrants = ordered.loc[~retain_mask].head(slots).copy()
            entrants["selection_reason"] = "entered_top_n"
            selected = pd.concat([retained, entrants], ignore_index=False).sort_values(
                ["universe_rank", "ticker"],
                kind="mergesort",
            )
        if len(selected) != top_n or selected["ticker"].nunique() != top_n:
            raise RuntimeError("rank-buffer selection failed to produce exactly top_n unique names")
        selected["selection_rank"] = np.arange(1, top_n + 1)
        selected["retention_rank_threshold"] = effective_retention_rank
        prior_selected_tickers = set(selected["ticker"])
        weights, inverse_volatility_base = _solve_constrained_inverse_volatility(
            selected,
            volatility_column=volatility_column,
            beta_column=beta_column,
            max_name_weight=max_name_weight,
            beta_exposure_limit=beta_exposure_limit,
            sector_column=str(sector_column) if sector_enabled else None,
            max_sector_weight=float(max_sector_weight) if sector_enabled else None,
        )
        selected["inverse_volatility_base_weight"] = inverse_volatility_base
        selected["weight"] = weights
        selected["target_return"] = selected["exit_price"] / selected["entry_price"] - 1.0
        selected["strategy_score_column"] = score_column
        holdings_frames.append(selected)
        schedule_rows.append(
            {
                "signal_at": signal_at,
                "entry_at": selected["entry_at"].iloc[0],
                "exit_at": selected["exit_at"].iloc[0],
            }
        )

    schedule = pd.DataFrame(schedule_rows).sort_values("entry_at").reset_index(drop=True)
    if len(schedule) > 1:
        contiguous = schedule["exit_at"].iloc[:-1].reset_index(drop=True).eq(
            schedule["entry_at"].iloc[1:].reset_index(drop=True)
        )
        if not contiguous.all():
            index = int(np.flatnonzero(~contiguous.to_numpy())[0])
            raise ValueError(
                "holding periods must be contiguous: prior exit_at must equal next entry_at; "
                f"got {schedule.loc[index, 'exit_at']} then {schedule.loc[index + 1, 'entry_at']}"
            )

    holdings = pd.concat(holdings_frames, ignore_index=True).sort_values(
        ["entry_at", "selection_rank", "ticker"]
    )
    previous: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    for signal_at, group in holdings.groupby("signal_at", sort=True):
        current = dict(zip(group["ticker"], group["weight"]))
        all_names = set(previous) | set(current)
        turnover = float(
            sum(abs(current.get(name, 0.0) - previous.get(name, 0.0)) for name in all_names)
        )
        gross_return = float((group["weight"] * group["target_return"]).sum())
        if 1 + gross_return <= 0:
            raise ValueError(f"portfolio wealth became non-positive at signal_at={signal_at}")
        trading_cost = turnover * trading_cost_bps_one_way / 10_000
        beta_exposure = float((group["weight"] * group[beta_column]).sum())
        largest_sector_weight = (
            float(group.groupby(str(sector_column))["weight"].sum().max())
            if sector_enabled
            else np.nan
        )
        rows.append(
            {
                "signal_at": signal_at,
                "entry_at": group["entry_at"].iloc[0],
                "exit_at": group["exit_at"].iloc[0],
                "gross_return": gross_return,
                "trading_cost": trading_cost,
                "terminal_liquidation_turnover": 0.0,
                "terminal_liquidation_cost": 0.0,
                "net_return": gross_return - trading_cost,
                "turnover": turnover,
                "long_count": top_n,
                "gross_exposure": float(group["weight"].abs().sum()),
                "net_exposure": float(group["weight"].sum()),
                "beta_exposure": beta_exposure,
                "largest_sector_weight": largest_sector_weight,
            }
        )
        realized = dict(zip(group["ticker"], group["target_return"]))
        denominator = 1.0 + gross_return
        previous = {
            ticker: weight * (1.0 + float(realized[ticker])) / denominator
            for ticker, weight in current.items()
        }

    if charge_terminal_liquidation:
        liquidation_turnover = float(sum(abs(weight) for weight in previous.values()))
        pre_liquidation_nav = max(
            0.0,
            1.0 + float(rows[-1]["gross_return"]) - float(rows[-1]["trading_cost"]),
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

    returns = pd.DataFrame(rows).sort_values("entry_at").reset_index(drop=True)
    return returns, holdings.reset_index(drop=True)


__all__ = ["build_diversified_momentum_portfolio"]
