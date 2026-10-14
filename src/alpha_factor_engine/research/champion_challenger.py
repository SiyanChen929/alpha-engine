"""Champion-challenger helpers for leakage-audited monthly equity research."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from alpha_factor_engine.contracts import PointInTimeViolation
from alpha_factor_engine.research.metrics import performance_summary
from alpha_factor_engine.research.momentum import build_momentum_champion


AVAILABILITY_GROUPS: dict[str, tuple[str, ...]] = {
    "price_available_at": (
        "momentum_12_1",
        "momentum_6_1",
        "reversal_1m",
        "realized_vol_3m",
        "downside_vol_3m",
        "beta_12m",
        "idio_vol_12m",
        "illiquidity_3m",
        "log_dollar_volume_3m",
        "price_to_52w_high",
        "max_drawdown_12m",
        "skew_3m",
    ),
    "fundamental_available_at": (
        "book_to_market",
        "earnings_yield",
        "roe",
        "roa",
        "gross_profitability",
        "debt_to_equity",
        "asset_growth",
        "sales_growth",
        "accruals",
        "free_cash_flow_yield",
        "log_market_cap",
        "fundamental_age_days",
        "gross_margin",
        "cash_flow_to_assets",
    ),
    "macro_available_at": (
        "fed_funds",
        "yield_curve",
        "vix",
        "baa_yield",
        "fed_funds_change_3m",
        "yield_curve_change_3m",
        "vix_change_3m",
        "baa_yield_change_3m",
    ),
}


def add_decision_time(
    panel: pd.DataFrame,
    *,
    decision_time_et: str = "20:00",
) -> pd.DataFrame:
    """Attach a timezone-aware decision clock to each formation session."""

    try:
        hour_text, minute_text = decision_time_et.split(":", maxsplit=1)
        hour, minute = int(hour_text), int(minute_text)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("decision_time_et must use HH:MM") from error
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ValueError("decision_time_et must use a valid 24-hour clock")
    if "date" not in panel and "signal_date" not in panel:
        raise ValueError("panel requires date or signal_date to form decision_time")

    output = panel.copy()
    if "date" in output:
        dates = pd.to_datetime(output["date"], errors="raise").dt.date
    else:
        dates = pd.to_datetime(output["signal_date"], utc=True, errors="raise").dt.tz_convert(
            "America/New_York"
        ).dt.date
    local_midnight = pd.to_datetime(dates).dt.tz_localize(
        "America/New_York", nonexistent="raise", ambiguous="raise"
    )
    output["decision_time"] = (
        local_midnight + pd.Timedelta(hours=hour, minutes=minute)
    ).dt.tz_convert("UTC")
    output["signal_at"] = output["decision_time"]
    return output


def assert_panel_is_point_in_time(
    panel: pd.DataFrame,
    *,
    decision_column: str = "decision_time",
    feature_columns: Iterable[str] | None = None,
) -> dict[str, int]:
    """Fail when a populated feature was unavailable at the decision clock."""

    if decision_column not in panel:
        raise PointInTimeViolation(f"panel missing decision column: {decision_column}")
    decision = pd.to_datetime(panel[decision_column], utc=True, errors="raise")
    requested = set(feature_columns or [item for values in AVAILABILITY_GROUPS.values() for item in values])
    checked_rows = 0
    violations = 0
    missing_timestamps = 0
    for available_column, group_columns in AVAILABILITY_GROUPS.items():
        used_columns = [column for column in group_columns if column in panel and column in requested]
        if not used_columns:
            continue
        if available_column not in panel:
            raise PointInTimeViolation(
                f"panel uses {used_columns[:3]} but lacks {available_column}"
            )
        populated = panel[used_columns].notna().any(axis=1)
        available = pd.to_datetime(panel[available_column], utc=True, errors="coerce")
        missing = populated & available.isna()
        future = populated & available.gt(decision)
        checked_rows += int(populated.sum())
        missing_timestamps += int(missing.sum())
        violations += int(future.sum())
        if missing.any() or future.any():
            examples = panel.loc[
                missing | future,
                [column for column in ["ticker", "signal_date", decision_column, available_column] if column in panel],
            ].head(5)
            raise PointInTimeViolation(
                f"{available_column} failed PIT availability: missing={int(missing.sum())}, "
                f"future={int(future.sum())}, examples={examples.to_dict('records')}"
            )

    for column in ["entry_at", "exit_at"]:
        if column not in panel:
            raise PointInTimeViolation(f"panel missing execution timestamp: {column}")
    entry = pd.to_datetime(panel["entry_at"], utc=True, errors="coerce")
    exit_at = pd.to_datetime(panel["exit_at"], utc=True, errors="coerce")
    evaluable = panel.get("target_return", pd.Series(np.nan, index=panel.index)).notna()
    invalid_execution = evaluable & (~decision.lt(entry) | ~entry.lt(exit_at))
    if invalid_execution.any():
        raise PointInTimeViolation(
            "evaluable rows must satisfy decision_time < entry_at < exit_at"
        )
    return {
        "checked_feature_rows": checked_rows,
        "missing_availability_timestamps": missing_timestamps,
        "future_information_violations": violations,
    }


def add_beta_size_residual_target(
    panel: pd.DataFrame,
    *,
    target_column: str = "target_return",
    output_column: str = "residual_target_return",
    eligible_column: str = "eligible",
) -> pd.DataFrame:
    """Create a training-only return residual using contemporaneous exposures."""

    required = {"signal_date", target_column, "beta_12m", "log_market_cap"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"residual target panel missing columns: {sorted(missing)}")
    output = panel.copy()
    output[output_column] = np.nan
    for signal_date, group in output.groupby("signal_date", sort=True):
        target = pd.to_numeric(group[target_column], errors="coerce")
        eligible = (
            group[eligible_column].fillna(False).astype(bool)
            if eligible_column in group
            else pd.Series(True, index=group.index)
        )
        usable = eligible & target.notna()
        if not usable.any():
            continue
        if target.loc[eligible].isna().any():
            raise PointInTimeViolation(
                f"partial target coverage at {signal_date}; residualization would condition on future availability"
            )
        beta = pd.to_numeric(group.loc[usable, "beta_12m"], errors="coerce")
        size = pd.to_numeric(group.loc[usable, "log_market_cap"], errors="coerce")
        beta = beta.fillna(beta.median())
        size = size.fillna(size.median())
        design_parts: list[np.ndarray] = [np.ones(int(usable.sum()))]
        for values in [beta, size]:
            scale = values.std(ddof=0)
            if pd.notna(scale) and scale > 1e-12:
                design_parts.append(((values - values.mean()) / scale).to_numpy())
        design = np.column_stack(design_parts)
        y = target.loc[usable].to_numpy(float)
        if len(y) <= design.shape[1] + 2:
            residual = y - y.mean()
        else:
            coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
            residual = y - design @ coefficients
        output.loc[group.index[usable], output_column] = residual
    return output


def select_public_proxy_universe(
    panel: pd.DataFrame,
    *,
    top_n: int = 30,
    ranking_column: str = "log_market_cap",
    minimum_cross_section: int = 20,
) -> pd.DataFrame:
    """Select a deterministic diagnostic universe while preserving its bias label."""

    required = {"signal_date", "ticker", "eligible", ranking_column}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"public proxy universe missing columns: {sorted(missing)}")
    if top_n <= 0 or minimum_cross_section <= 0:
        raise ValueError("top_n and minimum_cross_section must be positive")
    if "decision_time" not in panel:
        raise PointInTimeViolation("public proxy universe requires decision_time")
    assert_panel_is_point_in_time(panel, feature_columns=[ranking_column])
    snapshots: list[pd.DataFrame] = []
    for signal_date, group in panel.groupby("signal_date", sort=True):
        candidates = group.loc[
            group["eligible"].fillna(False)
            & pd.to_numeric(group[ranking_column], errors="coerce").notna()
        ].copy()
        if len(candidates) < minimum_cross_section:
            raise PointInTimeViolation(
                f"{signal_date} has only {len(candidates)} eligible securities; "
                f"minimum_cross_section={minimum_cross_section}"
            )
        candidates[ranking_column] = pd.to_numeric(candidates[ranking_column], errors="raise")
        candidates = candidates.sort_values(
            [ranking_column, "ticker"],
            ascending=[False, True],
            kind="mergesort",
        ).head(top_n)
        candidates["universe_rank"] = np.arange(1, len(candidates) + 1)
        candidates["universe_evidence_status"] = "current_ticker_proxy_opened_diagnostic"
        snapshots.append(candidates)
    return pd.concat(snapshots, ignore_index=True).sort_values(
        ["signal_date", "universe_rank"]
    ).reset_index(drop=True)


def evaluable_cross_sections(panel: pd.DataFrame) -> pd.DataFrame:
    """Keep wholly realized months and reject partial future-label coverage."""

    keep: list[pd.DataFrame] = []
    for signal_date, group in panel.groupby("signal_date", sort=True):
        missing = group["target_return"].isna()
        if missing.all():
            continue
        if missing.any():
            raise PointInTimeViolation(
                f"partial target coverage at {signal_date}; refusing an availability-selected portfolio"
            )
        keep.append(group)
    if not keep:
        raise ValueError("no fully evaluable cross-sections")
    return pd.concat(keep, ignore_index=True)


def build_ranked_long_only(
    panel: pd.DataFrame,
    *,
    score_column: str,
    holdings_count: int,
    trading_cost_bps_one_way: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the champion's identical execution and cost logic to any score."""

    required = {
        "signal_at",
        "ticker",
        score_column,
        "entry_at",
        "exit_at",
        "entry_price",
        "exit_price",
    }
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"ranked long-only panel missing columns: {sorted(missing)}")
    candidates = panel.copy()
    candidates["momentum_12_1"] = pd.to_numeric(candidates[score_column], errors="coerce")
    missing_scores = candidates["momentum_12_1"].isna()
    if missing_scores.any():
        examples = candidates.loc[missing_scores, ["signal_at", "ticker"]].head(5)
        raise PointInTimeViolation(
            f"{score_column} is missing for {int(missing_scores.sum())} frozen-universe rows; "
            f"examples={examples.to_dict('records')}"
        )
    returns, holdings = build_momentum_champion(
        candidates,
        top_n=holdings_count,
        trading_cost_bps_one_way=trading_cost_bps_one_way,
        charge_terminal_liquidation=True,
    )
    returns["strategy_score"] = score_column
    holdings["strategy_score"] = score_column
    holdings["original_score"] = holdings[score_column]
    return returns, holdings


def relative_return_frame(
    strategy: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate investable period-by-period relative returns."""

    left = strategy[["signal_at", "entry_at", "exit_at", "net_return"]].rename(
        columns={"net_return": "strategy_return"}
    )
    right = benchmark[["signal_at", "entry_at", "exit_at", "net_return"]].rename(
        columns={"net_return": "benchmark_return"}
    )
    merged = left.merge(
        right,
        on=["signal_at", "entry_at", "exit_at"],
        validate="one_to_one",
    )
    if len(merged) != len(left) or len(merged) != len(right):
        raise PointInTimeViolation(
            "strategy and benchmark must have identical signal, entry, and exit timestamps"
        )
    merged["active_return"] = merged["strategy_return"] - merged["benchmark_return"]
    merged["relative_wealth_return"] = (
        (1 + merged["strategy_return"]) / (1 + merged["benchmark_return"]) - 1
    )
    # Information ratio and active-return inference use arithmetic active
    # returns.  Relative wealth is retained separately for compounded wealth.
    merged["net_return"] = merged["active_return"]
    merged["turnover"] = 0.0
    merged["trading_cost"] = 0.0
    merged["borrow_cost"] = 0.0
    return merged


def maximum_single_name_gross_profit_share(holdings: pd.DataFrame) -> float:
    """Approximate concentration based on uncompounded gross contributions."""

    contribution = (
        holdings.assign(contribution=holdings["weight"] * holdings["target_return"])
        .groupby("ticker")["contribution"]
        .sum()
    )
    total = float(contribution.sum())
    if total <= 0 or contribution.empty:
        return np.nan
    return float(contribution.max() / total)


def capacity_diagnostics(
    holdings: pd.DataFrame,
    *,
    portfolio_aum_usd: float,
    maximum_adv_participation: float,
) -> tuple[dict[str, float | int | bool], pd.DataFrame]:
    """Measure rebalance and liquidation trade dollars against trailing ADV."""

    if portfolio_aum_usd <= 0 or not np.isfinite(portfolio_aum_usd):
        raise ValueError("portfolio_aum_usd must be finite and positive")
    if not 0 < maximum_adv_participation <= 1:
        raise ValueError("maximum_adv_participation must lie in (0, 1]")
    required = {"signal_at", "ticker", "weight", "target_return", "adv_usd_3m"}
    missing = required.difference(holdings.columns)
    if missing:
        raise ValueError(f"capacity holdings missing columns: {sorted(missing)}")
    previous: dict[str, float] = {}
    previous_adv: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    last_signal: pd.Timestamp | None = None
    for signal_at, group in holdings.groupby("signal_at", sort=True):
        current = dict(zip(group["ticker"].astype(str), group["weight"].astype(float)))
        current_adv = dict(
            zip(group["ticker"].astype(str), pd.to_numeric(group["adv_usd_3m"], errors="coerce"))
        )
        all_names = set(previous) | set(current)
        for name in sorted(all_names):
            delta = current.get(name, 0.0) - previous.get(name, 0.0)
            if abs(delta) <= 1e-14:
                continue
            adv = current_adv.get(name, previous_adv.get(name, np.nan))
            if not np.isfinite(adv) or adv <= 0:
                raise PointInTimeViolation(f"missing positive ADV for capacity trade {name} at {signal_at}")
            participation = portfolio_aum_usd * abs(delta) / float(adv)
            rows.append(
                {
                    "signal_at": signal_at,
                    "ticker": name,
                    "trade_type": "rebalance",
                    "trade_weight": delta,
                    "trade_dollars": portfolio_aum_usd * abs(delta),
                    "adv_usd": float(adv),
                    "adv_participation": participation,
                    "limit": maximum_adv_participation,
                    "passed": participation <= maximum_adv_participation,
                }
            )
        gross_return = float((group["weight"] * group["target_return"]).sum())
        if 1 + gross_return <= 0:
            raise ValueError(f"portfolio wealth became non-positive at {signal_at}")
        realized = dict(zip(group["ticker"].astype(str), group["target_return"].astype(float)))
        previous = {
            name: weight * (1 + realized[name]) / (1 + gross_return)
            for name, weight in current.items()
            if abs(weight) > 0
        }
        previous_adv = {name: float(current_adv[name]) for name in previous}
        last_signal = pd.Timestamp(signal_at)
    if last_signal is None:
        raise ValueError("capacity holdings are empty")
    for name, weight in sorted(previous.items()):
        adv = previous_adv[name]
        participation = portfolio_aum_usd * abs(weight) / adv
        rows.append(
            {
                "signal_at": last_signal,
                "ticker": name,
                "trade_type": "terminal_liquidation",
                "trade_weight": -weight,
                "trade_dollars": portfolio_aum_usd * abs(weight),
                "adv_usd": adv,
                "adv_participation": participation,
                "limit": maximum_adv_participation,
                "passed": participation <= maximum_adv_participation,
            }
        )
    trades = pd.DataFrame(rows)
    return (
        {
            "portfolio_aum_usd": float(portfolio_aum_usd),
            "maximum_adv_participation_limit": float(maximum_adv_participation),
            "maximum_adv_participation_observed": float(trades["adv_participation"].max()),
            "capacity_violations": int((~trades["passed"]).sum()),
            "capacity_passed": bool(trades["passed"].all()),
        },
        trades,
    )


def long_only_evaluation(
    strategy_returns: pd.DataFrame,
    strategy_holdings: pd.DataFrame,
    benchmark_returns: pd.DataFrame,
) -> dict[str, float]:
    """Return total and same-universe benchmark-relative diagnostics."""

    total = performance_summary(
        strategy_returns.assign(borrow_cost=0.0), return_column="net_return"
    )
    benchmark = performance_summary(
        benchmark_returns.assign(borrow_cost=0.0), return_column="net_return"
    )
    relative_frame = relative_return_frame(strategy_returns, benchmark_returns)
    active = performance_summary(relative_frame, return_column="active_return")
    relative_wealth = performance_summary(
        relative_frame,
        return_column="relative_wealth_return",
    )
    return {
        **total,
        "active_annualized_return": active.get("annualized_arithmetic_return", np.nan),
        "active_information_ratio": active.get("sharpe", np.nan),
        "active_newey_west_tstat": active.get("newey_west_tstat", np.nan),
        "relative_wealth_cagr": relative_wealth.get("annualized_return", np.nan),
        "relative_max_drawdown": relative_wealth.get("max_drawdown", np.nan),
        "benchmark_annualized_return": benchmark.get("annualized_return", np.nan),
        "benchmark_sharpe": benchmark.get("sharpe", np.nan),
        "benchmark_max_drawdown": benchmark.get("max_drawdown", np.nan),
        "active_max_drawdown_gap": total.get("max_drawdown", np.nan)
        - benchmark.get("max_drawdown", np.nan),
        "maximum_single_name_gross_profit_share": maximum_single_name_gross_profit_share(
            strategy_holdings
        ),
    }


__all__ = [
    "AVAILABILITY_GROUPS",
    "add_beta_size_residual_target",
    "add_decision_time",
    "assert_panel_is_point_in_time",
    "build_ranked_long_only",
    "capacity_diagnostics",
    "evaluable_cross_sections",
    "long_only_evaluation",
    "maximum_single_name_gross_profit_share",
    "relative_return_frame",
    "select_public_proxy_universe",
]
