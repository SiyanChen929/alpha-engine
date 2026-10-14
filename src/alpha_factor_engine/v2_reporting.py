"""Fail-closed reporting for the v2 opened diagnostic.

The strategy roles are fixed before results are supplied.  In particular, M2
is always the predeclared champion; this module never promotes a strategy from
opened-period performance.

Public API
----------
``create_v2_opened_diagnostic_report`` writes six PNG charts and one Markdown
report, then returns the artifact paths and gate decisions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alpha_factor_engine.champion_reporting import (
    _bool,
    _cost_assumptions,
    _first_present,
    _nested,
    _normalize_capacity_summary,
    _normalize_cost_sensitivity,
    _normalize_returns,
    _save,
    _style,
    _summary_table,
)


V2_ROLE_ORDER: tuple[str, ...] = (
    "m1_baseline",
    "m2_diversified_momentum",
    "m3_sector_beta_constrained",
    "residual_ml_challenger",
)
V2_DISPLAY_NAMES: dict[str, str] = {
    "m1_baseline": "M1 Baseline",
    "m2_diversified_momentum": "M2 Diversified Momentum",
    "m3_sector_beta_constrained": "M3 Risk-Bounded Momentum",
    "residual_ml_challenger": "Residual-ML Challenger",
}
V2_CHAMPION_ROLE = "m2_diversified_momentum"
_COLORS: dict[str, str] = {
    "m1_baseline": "#667085",
    "m2_diversified_momentum": "#155EEF",
    "m3_sector_beta_constrained": "#17B26A",
    "residual_ml_challenger": "#7A5AF8",
}
_CRITICAL_PRODUCTION_CONTROLS = (
    "survivorship_free",
    "permanent_identifiers",
    "delisting_returns",
    "execution_grade_quotes",
    "historical_borrow",
    "historical_industry_classification",
    "alfred_vintages_used",
)


def _require_role_mapping(name: str, value: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping keyed by the canonical v2 roles")
    supplied = {str(key) for key in value}
    required = set(V2_ROLE_ORDER)
    if supplied != required:
        missing = sorted(required - supplied)
        extra = sorted(supplied - required)
        raise ValueError(
            f"{name} must contain exactly the canonical v2 roles; "
            f"missing={missing}, extra={extra}"
        )


def _as_float(value: Any, label: str, *, lower: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not np.isfinite(result) or (lower is not None and result < lower):
        raise ValueError(f"{label} is outside its valid range")
    return result


def _assumption_payload(raw: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise TypeError("assumptions must be a mapping")
    inherited = _cost_assumptions(raw)

    def choose(direct: Sequence[str], nested_value: Any) -> Any:
        for key in direct:
            if key in raw:
                return raw[key]
        return nested_value

    trading_cost = choose(
        ("trading_cost_bps_one_way", "base_cost_bps_one_way", "one_way_cost_bps"),
        inherited["trading_cost_bps_one_way"],
    )
    aum = choose(("portfolio_aum_usd", "aum_usd"), inherited["portfolio_aum_usd"])
    adv_limit = choose(
        ("maximum_adv_participation", "adv_participation_limit"),
        inherited["maximum_adv_participation"],
    )
    if trading_cost is None or aum is None or adv_limit is None:
        raise ValueError(
            "assumptions must declare one-way trading cost, portfolio AUM, "
            "and maximum ADV participation"
        )
    annualization = choose(
        ("annualization_factor",),
        _nested(raw, ("reporting", "annualization_factor")),
    )
    annualization = 12 if annualization is None else int(annualization)
    if annualization <= 0:
        raise ValueError("annualization_factor must be positive")
    opened_start = choose(
        ("opened_start", "holdout_start", "evaluation_entry_at_or_after"),
        _nested(raw, ("research", "evaluation_entry_at_or_after")),
    )
    if opened_start in (None, ""):
        normalized_start = None
    else:
        normalized_start = pd.Timestamp(opened_start)
        if normalized_start.tzinfo is None:
            normalized_start = normalized_start.tz_localize("UTC")
        else:
            normalized_start = normalized_start.tz_convert("UTC")

    borrow = choose(
        ("annual_short_borrow_bps",), inherited["annual_short_borrow_bps"]
    )
    entry = choose(("charge_initial_entry",), inherited["charge_initial_entry"])
    liquidation = choose(
        ("charge_terminal_liquidation",),
        inherited["charge_terminal_liquidation"],
    )
    return {
        "trading_cost_bps_one_way": _as_float(
            trading_cost, "trading_cost_bps_one_way", lower=0.0
        ),
        "annual_short_borrow_bps": (
            None
            if borrow is None
            else _as_float(borrow, "annual_short_borrow_bps", lower=0.0)
        ),
        "portfolio_aum_usd": _as_float(aum, "portfolio_aum_usd", lower=1e-12),
        "maximum_adv_participation": _as_float(
            adv_limit, "maximum_adv_participation", lower=1e-12
        ),
        "charge_initial_entry": _bool(entry),
        "charge_terminal_liquidation": _bool(liquidation),
        "annualization_factor": annualization,
        "opened_start": normalized_start,
    }


def _validate_opened(predeclared_gates: Mapping[str, Any], data_quality: Mapping[str, Any]) -> None:
    explicit = _bool(
        predeclared_gates.get(
            "holdout_opened", predeclared_gates.get("opened_period")
        )
    )
    if explicit is False:
        raise ValueError(
            "this API is only for opened diagnostics; holdout_opened cannot be false"
        )
    status = str(
        predeclared_gates.get("historical_evidence_status")
        or data_quality.get("evidence_status")
        or "opened_diagnostic"
    ).lower()
    if status and not any(token in status for token in ("opened", "diagnostic")):
        raise ValueError("evidence status must identify an opened diagnostic")


def _metric_role_names(metrics: pd.DataFrame) -> pd.Series:
    if not isinstance(metrics, pd.DataFrame) or metrics.empty:
        raise ValueError("metrics must be a non-empty DataFrame")
    name_column = _first_present(metrics.columns, ("strategy", "model", "sleeve", "name"))
    if name_column is None:
        raise ValueError("metrics needs a strategy-name column")
    frame = metrics
    if "period" in frame:
        opened = frame["period"].astype(str).str.lower().isin(
            ("opened", "opened_test", "holdout", "test", "current_test")
        )
        if opened.any():
            frame = frame.loc[opened]
    names = frame[name_column].astype(str)
    supplied = set(names)
    required = set(V2_ROLE_ORDER)
    if supplied != required or len(names) != len(V2_ROLE_ORDER):
        raise ValueError(
            "metrics must have exactly one opened row for each canonical v2 role"
        )
    return names


def _aligned_active_metrics(
    returns: Mapping[str, pd.DataFrame], annualization: int
) -> dict[str, dict[str, float]]:
    baseline = returns[V2_ROLE_ORDER[0]]
    dates = baseline["date"].reset_index(drop=True)
    benchmark = baseline["net_return"].to_numpy(float)
    result: dict[str, dict[str, float]] = {}
    for role in V2_ROLE_ORDER:
        frame = returns[role]
        if len(frame) != len(baseline) or not frame["date"].reset_index(drop=True).equals(dates):
            raise ValueError(
                "all v2 return series must use the identical opened-period timestamps"
            )
        values = frame["net_return"].to_numpy(float)
        if role == V2_ROLE_ORDER[0]:
            result[role] = {
                "active_annualized_return": 0.0,
                "active_information_ratio": 0.0,
                "relative_max_drawdown": 0.0,
            }
            continue
        active = (1.0 + values) / (1.0 + benchmark) - 1.0
        relative_wealth = np.cumprod(1.0 + active)
        active_return = float(relative_wealth[-1] ** (annualization / len(active)) - 1.0)
        tracking_error = float(np.std(active, ddof=1)) if len(active) > 1 else np.nan
        information_ratio = (
            float(np.mean(active) / tracking_error * np.sqrt(annualization))
            if np.isfinite(tracking_error) and tracking_error > 0
            else np.nan
        )
        relative_peak = np.maximum.accumulate(relative_wealth)
        result[role] = {
            "active_annualized_return": active_return,
            "active_information_ratio": information_ratio,
            "relative_max_drawdown": float(
                np.min(relative_wealth / relative_peak - 1.0)
            ),
        }
    return result


def _opened_summary(
    returns: Mapping[str, pd.DataFrame], metrics: pd.DataFrame, annualization: int
) -> pd.DataFrame:
    _metric_role_names(metrics)
    summary = _summary_table(returns, metrics, annualization=annualization)
    active = _aligned_active_metrics(returns, annualization)
    for column in (
        "active_annualized_return",
        "active_information_ratio",
        "relative_max_drawdown",
    ):
        if column not in summary:
            summary[column] = np.nan
        for role in V2_ROLE_ORDER:
            index = summary.index[summary["strategy"].eq(role)]
            if len(index) != 1:
                raise ValueError(f"metrics/returns did not resolve one row for {role}")
            # Precomputed pipeline metrics are used when present.  The curve-derived
            # value fills only absent metrics and preserves compatibility with richer
            # benchmark models.
            if pd.isna(summary.loc[index[0], column]):
                summary.loc[index[0], column] = active[role][column]
    summary["_order"] = summary["strategy"].map(
        {role: index for index, role in enumerate(V2_ROLE_ORDER)}
    )
    summary = summary.sort_values("_order", kind="mergesort").drop(columns="_order")
    summary["display_name"] = summary["strategy"].map(V2_DISPLAY_NAMES)
    return summary.reset_index(drop=True)


def _holding_exposures(
    holdings: Mapping[str, pd.DataFrame], opened_start: pd.Timestamp | None
) -> pd.DataFrame:
    _require_role_mapping("holdings", holdings)
    rows: list[dict[str, Any]] = []
    for role in V2_ROLE_ORDER:
        raw = holdings[role]
        if not isinstance(raw, pd.DataFrame) or raw.empty:
            raise ValueError(f"holdings[{role!r}] must be a non-empty DataFrame")
        date_column = _first_present(
            raw.columns, ("signal_date", "date", "entry_at", "timestamp")
        )
        weight_column = _first_present(
            raw.columns, ("weight", "target_weight", "position_weight")
        )
        if date_column is None or weight_column is None:
            raise ValueError(f"holdings[{role!r}] needs date and weight columns")
        beta_column = _first_present(raw.columns, ("beta_12m", "beta", "market_beta"))
        sector_column = _first_present(
            raw.columns, ("sector", "gics_sector", "sic_division", "sector_code")
        )
        frame = raw.copy()
        frame[date_column] = pd.to_datetime(frame[date_column], utc=True, errors="coerce")
        frame[weight_column] = pd.to_numeric(frame[weight_column], errors="coerce")
        if frame[[date_column, weight_column]].isna().any(axis=None):
            raise ValueError(f"holdings[{role!r}] has invalid dates or weights")
        if opened_start is not None:
            frame = frame.loc[frame[date_column] >= opened_start]
        if frame.empty:
            raise ValueError(f"holdings[{role!r}] has no rows in the opened period")
        if beta_column is not None:
            frame[beta_column] = pd.to_numeric(frame[beta_column], errors="coerce")

        daily: list[dict[str, float]] = []
        for _, group in frame.groupby(date_column, sort=True):
            weights = group[weight_column].to_numpy(float)
            gross = float(np.abs(weights).sum())
            if gross <= 0:
                raise ValueError(f"holdings[{role!r}] contains a zero-gross portfolio")
            nonzero = np.abs(weights) > 1e-12
            beta_value = np.nan
            if beta_column is not None and group[beta_column].notna().all():
                beta_value = abs(float(np.dot(weights, group[beta_column].to_numpy(float))))
            sector_share = np.nan
            if sector_column is not None and group[sector_column].notna().all():
                sector_gross = (
                    group.assign(_gross=np.abs(weights))
                    .groupby(sector_column, dropna=False)["_gross"]
                    .sum()
                )
                sector_share = float(sector_gross.max() / gross)
            daily.append(
                {
                    "gross_exposure": gross,
                    "absolute_net_exposure": abs(float(weights.sum())),
                    "active_positions": float(nonzero.sum()),
                    "largest_gross_weight": float(np.abs(weights).max() / gross),
                    "absolute_beta_exposure": beta_value,
                    "maximum_sector_gross_share": sector_share,
                }
            )
        daily_frame = pd.DataFrame(daily)
        rows.append(
            {
                "strategy": role,
                "display_name": V2_DISPLAY_NAMES[role],
                "median_gross_exposure": float(daily_frame["gross_exposure"].median()),
                "median_absolute_net_exposure": float(
                    daily_frame["absolute_net_exposure"].median()
                ),
                "median_active_positions": float(daily_frame["active_positions"].median()),
                "maximum_largest_gross_weight": float(
                    daily_frame["largest_gross_weight"].max()
                ),
                "median_absolute_beta_exposure": float(
                    daily_frame["absolute_beta_exposure"].median()
                ),
                "maximum_absolute_beta_exposure": float(
                    daily_frame["absolute_beta_exposure"].max()
                ),
                "median_maximum_sector_gross_share": float(
                    daily_frame["maximum_sector_gross_share"].median()
                ),
                "maximum_sector_gross_share": float(
                    daily_frame["maximum_sector_gross_share"].max()
                ),
            }
        )
    return pd.DataFrame(rows)


def _gate_rows(predeclared_gates: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def collect(group: str, checks: Mapping[str, Any]) -> None:
        for raw_name, raw_specification in checks.items():
            if not isinstance(raw_specification, Mapping):
                continue
            specification = raw_specification
            value = specification.get("value", specification.get("actual"))
            supplied_pass = _bool(specification.get("passed"))
            relation: str | None = None
            threshold: Any = None
            passed: bool | None = None
            if "minimum" in specification and value is not None:
                relation, threshold = ">=", specification["minimum"]
                passed = float(value) >= float(threshold)
            elif "maximum" in specification and value is not None:
                relation, threshold = "<=", specification["maximum"]
                passed = float(value) <= float(threshold)
            elif "equals" in specification and value is not None:
                relation, threshold = "==", specification["equals"]
                passed = value == threshold
            elif "required" in specification and value is not None:
                relation, threshold = "==", specification["required"]
                required_bool = _bool(threshold)
                actual_bool = _bool(value)
                passed = (
                    actual_bool == required_bool
                    if actual_bool is not None and required_bool is not None
                    else value == threshold
                )
            elif supplied_pass is not None:
                relation, threshold, passed = "predeclared", "pass", supplied_pass
            if passed is None:
                continue
            if supplied_pass is not None:
                passed = bool(passed and supplied_pass)
            rows.append(
                {
                    "group": group,
                    "name": str(raw_name),
                    "value": value,
                    "relation": relation,
                    "threshold": threshold,
                    "passed": bool(passed),
                }
            )

    containers = (
        ("stage_gates", "stage"),
        ("research_checks", "research"),
        ("checks", "diagnostic"),
    )
    for key, default_group in containers:
        raw = predeclared_gates.get(key)
        if not isinstance(raw, Mapping):
            continue
        if key == "stage_gates":
            for stage_name, stage_checks in raw.items():
                if isinstance(stage_checks, Mapping):
                    collect(str(stage_name), stage_checks)
        else:
            collect(default_group, raw)
    return rows


def _diagnostic_status(
    predeclared_gates: Mapping[str, Any], checks: Sequence[Mapping[str, Any]]
) -> str:
    supplied = str(
        predeclared_gates.get("diagnostic_status")
        or predeclared_gates.get("research_status")
        or ""
    ).upper()
    computed = (
        "NOT_EVALUATED"
        if not checks
        else "PASS"
        if all(bool(row["passed"]) for row in checks)
        else "FAIL"
    )
    if supplied in {"FAIL", "BLOCKED"} or computed == "FAIL":
        return "FAIL"
    if computed == "NOT_EVALUATED":
        return "NOT_EVALUATED"
    return "PASS"


def _production_controls(
    predeclared_gates: Mapping[str, Any], data_quality: Mapping[str, Any]
) -> dict[str, bool]:
    supplied = predeclared_gates.get("production_data_checks")
    controls: dict[str, bool] = {}
    if isinstance(supplied, Mapping):
        for name, value in supplied.items():
            converted = _bool(value)
            if converted is not None:
                controls[str(name)] = converted
    for name in _CRITICAL_PRODUCTION_CONTROLS:
        if name not in controls:
            converted = _bool(data_quality.get(name))
            controls[name] = False if converted is None else converted
    if "execution_grade_quotes" not in (supplied or {}):
        execution = str(data_quality.get("execution_data", "")).lower()
        if execution:
            controls["execution_grade_quotes"] = not any(
                marker in execution for marker in ("proxy", "no taq", "no nbbo")
            )
    return controls


def _current_ticker_limitation(data_quality: Mapping[str, Any]) -> bool:
    source = " ".join(
        str(data_quality.get(key, "")).lower()
        for key in ("universe_mode", "universe_source", "source")
    )
    survivorship_free = _bool(data_quality.get("survivorship_free"))
    permanent_ids = _bool(data_quality.get("permanent_identifiers"))
    return (
        "current" in source
        or "proxy" in source
        or survivorship_free is not True
        or permanent_ids is not True
    )


def _normalize_capacity(
    root: Path,
    capacity_summary: pd.DataFrame,
    data_quality: Mapping[str, Any],
    assumptions: Mapping[str, Any],
) -> pd.DataFrame:
    if not isinstance(capacity_summary, pd.DataFrame) or capacity_summary.empty:
        raise ValueError("capacity_summary must be a non-empty DataFrame")
    config = {
        "costs": {
            "portfolio_aum_usd": assumptions["portfolio_aum_usd"],
            "maximum_adv_participation": assumptions[
                "maximum_adv_participation"
            ],
        }
    }
    normalized, _ = _normalize_capacity_summary(
        root,
        capacity_summary,
        data_quality=data_quality,
        config=config,
        champion=V2_CHAMPION_ROLE,
    )
    unknown = sorted(set(normalized["strategy"]) - set(V2_ROLE_ORDER))
    if unknown:
        raise ValueError(f"capacity_summary contains unknown v2 roles: {unknown}")
    if V2_CHAMPION_ROLE not in set(normalized["strategy"]):
        raise ValueError("capacity_summary must include the predeclared M2 champion")
    return normalized


def _normalize_costs(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = _normalize_cost_sensitivity(frame, champion=V2_CHAMPION_ROLE)
    unknown = sorted(set(normalized["strategy"].astype(str)) - set(V2_ROLE_ORDER))
    if unknown:
        raise ValueError(f"cost_sensitivity contains unknown v2 roles: {unknown}")
    required = set(V2_ROLE_ORDER[1:])
    missing = sorted(required - set(normalized["strategy"].astype(str)))
    if missing:
        raise ValueError(
            "cost_sensitivity must include M2, M3, and residual-ML; "
            f"missing={missing}"
        )
    return normalized


def _fmt_number(value: Any, *, percentage: bool = False) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    numeric = float(value)
    return f"{numeric:.2%}" if percentage else f"{numeric:.3f}"


def _save_charts(
    chart_dir: Path,
    returns: Mapping[str, pd.DataFrame],
    summary: pd.DataFrame,
    exposures: pd.DataFrame,
    cost_sensitivity: pd.DataFrame,
    capacity: pd.DataFrame,
    assumptions: Mapping[str, Any],
    limitation_label: str,
) -> list[Path]:
    _style()
    chart_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(10.4, 5.6))
    for role in V2_ROLE_ORDER:
        frame = returns[role]
        wealth = np.cumprod(1.0 + frame["net_return"].to_numpy(float))
        ax.plot(frame["date"], wealth, label=V2_DISPLAY_NAMES[role], color=_COLORS[role], lw=2)
    ax.set_title("Opened-period net cumulative wealth")
    ax.set_ylabel("Growth of $1")
    ax.legend(ncol=2, frameon=False)
    path = chart_dir / "01_opened_cumulative_wealth.png"
    _save(fig, path, limitation_label)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(10.4, 5.6))
    baseline = returns[V2_ROLE_ORDER[0]]["net_return"].to_numpy(float)
    for role in V2_ROLE_ORDER[1:]:
        frame = returns[role]
        active = (1.0 + frame["net_return"].to_numpy(float)) / (1.0 + baseline) - 1.0
        ax.plot(
            frame["date"],
            np.cumprod(1.0 + active),
            label=V2_DISPLAY_NAMES[role],
            color=_COLORS[role],
            lw=2,
        )
    ax.axhline(1.0, color="#98A2B3", lw=1)
    ax.set_title("Opened cumulative wealth relative to M1 baseline")
    ax.set_ylabel("Wealth relative to M1")
    ax.legend(ncol=2, frameon=False)
    path = chart_dir / "02_benchmark_relative_wealth.png"
    _save(fig, path, limitation_label)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(10.4, 5.6))
    for role in V2_ROLE_ORDER:
        frame = returns[role]
        wealth = np.cumprod(1.0 + frame["net_return"].to_numpy(float))
        drawdown = wealth / np.maximum.accumulate(wealth) - 1.0
        ax.plot(frame["date"], drawdown, label=V2_DISPLAY_NAMES[role], color=_COLORS[role], lw=1.8)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.set_title("Opened-period drawdowns")
    ax.set_ylabel("Drawdown")
    ax.legend(ncol=2, frameon=False)
    path = chart_dir / "03_opened_drawdowns.png"
    _save(fig, path, limitation_label)
    paths.append(path)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.8))
    labels = [f"M{index + 1}" if index < 3 else "Residual ML" for index in range(4)]
    metric_specs = (
        ("active_annualized_return", "Active annualized return", True),
        ("active_information_ratio", "Information ratio", False),
        ("max_drawdown", "Maximum drawdown", True),
    )
    for ax, (column, title, percentage) in zip(axes, metric_specs, strict=True):
        values = summary[column].to_numpy(float)
        ax.bar(labels, values, color=[_COLORS[role] for role in V2_ROLE_ORDER])
        ax.set_title(title)
        if percentage:
            ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
        ax.tick_params(axis="x", rotation=22)
        ax.axhline(0.0, color="#98A2B3", lw=1)
    fig.suptitle("Opened metric comparison (active metrics use the declared benchmark)")
    path = chart_dir / "04_opened_metric_comparison.png"
    _save(fig, path, limitation_label)
    paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8))
    for role in V2_ROLE_ORDER:
        subset = cost_sensitivity.loc[cost_sensitivity["strategy"].eq(role)]
        if subset.empty:
            continue
        axes[0].plot(
            subset["cost_bps"], subset["comparison_return"], marker="o",
            label=V2_DISPLAY_NAMES[role], color=_COLORS[role]
        )
        axes[1].plot(
            subset["cost_bps"], subset["comparison_ratio"], marker="o",
            label=V2_DISPLAY_NAMES[role], color=_COLORS[role]
        )
    axes[0].set_title(str(cost_sensitivity.attrs.get("return_label", "Net return")))
    axes[0].set_ylabel("Annualized return")
    axes[0].yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axes[1].set_title(str(cost_sensitivity.attrs.get("ratio_label", "Risk-adjusted ratio")))
    for ax in axes:
        ax.set_xlabel("One-way cost (bps)")
        ax.axvline(
            assumptions["trading_cost_bps_one_way"], color="#D92D20", ls="--", lw=1,
            label="Base assumption" if ax is axes[0] else None,
        )
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Opened cost sensitivity")
    path = chart_dir / "05_cost_sensitivity.png"
    _save(fig, path, limitation_label)
    paths.append(path)

    merged = exposures.merge(capacity, on="strategy", how="left")
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.8))
    colors = [_COLORS[role] for role in V2_ROLE_ORDER]
    x = np.arange(len(V2_ROLE_ORDER))
    beta = merged["maximum_absolute_beta_exposure"].to_numpy(float)
    sector = merged["maximum_sector_gross_share"].to_numpy(float)
    axes[0].bar(x, np.nan_to_num(beta, nan=0.0), color=colors)
    axes[0].set_title("Maximum |beta exposure|")
    axes[1].bar(x, np.nan_to_num(sector, nan=0.0), color=colors)
    axes[1].set_title("Maximum sector gross share")
    axes[1].yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    observed = merged["maximum_adv_participation_observed"].to_numpy(float)
    limits = merged["maximum_adv_participation_limit"].to_numpy(float)
    axes[2].bar(x, np.nan_to_num(observed, nan=0.0), color=colors, label="Observed")
    axes[2].scatter(x, limits, color="#D92D20", marker="_", s=180, label="Limit")
    axes[2].set_title("Maximum ADV participation")
    axes[2].yaxis.set_major_formatter(lambda value, _: f"{value:.1%}")
    axes[2].legend(frameon=False, fontsize=8)
    for index, ax in enumerate(axes):
        ax.set_xticks(x, labels, rotation=22)
        values = (beta, sector, observed)[index]
        for position, value in enumerate(values):
            if not np.isfinite(value):
                ax.text(position, 0.01, "N/A", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Risk constraints and capacity evidence")
    path = chart_dir / "06_risk_capacity_diagnostics.png"
    _save(fig, path, limitation_label)
    paths.append(path)
    return paths


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _write_report(
    path: Path,
    returns: Mapping[str, pd.DataFrame],
    summary: pd.DataFrame,
    exposures: pd.DataFrame,
    cost_sensitivity: pd.DataFrame,
    capacity: pd.DataFrame,
    gate_rows: Sequence[Mapping[str, Any]],
    diagnostic_status: str,
    production_controls: Mapping[str, bool],
    data_quality: Mapping[str, Any],
    assumptions: Mapping[str, Any],
    current_ticker: bool,
    blockers: Sequence[str],
    chart_paths: Sequence[Path],
) -> None:
    first_date = returns[V2_ROLE_ORDER[0]]["date"].min().date().isoformat()
    last_date = returns[V2_ROLE_ORDER[0]]["date"].max().date().isoformat()
    universe_label = (
        "CURRENT-TICKER / survivorship-biased proxy"
        if current_ticker
        else "point-in-time permanent-ID universe"
    )
    metric_rows = []
    for _, row in summary.iterrows():
        metric_rows.append(
            [
                str(row["display_name"]),
                _fmt_number(row["annualized_return"], percentage=True),
                _fmt_number(row["sharpe"]),
                _fmt_number(row["max_drawdown"], percentage=True),
                _fmt_number(row["active_annualized_return"], percentage=True),
                _fmt_number(row["active_information_ratio"]),
                _fmt_number(row["relative_max_drawdown"], percentage=True),
            ]
        )
    gate_table_rows = [
        [
            str(row["group"]),
            str(row["name"]),
            str(row["value"]),
            f"{row['relation']} {row['threshold']}",
            "PASS" if row["passed"] else "FAIL",
        ]
        for row in gate_rows
    ]
    capacity_lookup = capacity.set_index("strategy")
    capacity_rows = []
    for role in V2_ROLE_ORDER:
        if role not in capacity_lookup.index:
            capacity_rows.append([V2_DISPLAY_NAMES[role], "N/A", "N/A", "N/A", "N/A", "MISSING"])
            continue
        row = capacity_lookup.loc[role]
        capacity_rows.append(
            [
                V2_DISPLAY_NAMES[role],
                f"${float(row['portfolio_aum_usd']) / 1_000_000:.2f}m",
                _fmt_number(row["maximum_adv_participation_observed"], percentage=True),
                _fmt_number(row["maximum_adv_participation_limit"], percentage=True),
                str(int(row["capacity_violations"])),
                "PASS" if bool(row["capacity_passed_at_configured_aum"]) else "FAIL",
            ]
        )
    exposure_rows = []
    for _, row in exposures.iterrows():
        exposure_rows.append(
            [
                str(row["display_name"]),
                _fmt_number(row["median_gross_exposure"]),
                _fmt_number(row["median_absolute_net_exposure"]),
                _fmt_number(row["maximum_absolute_beta_exposure"]),
                _fmt_number(row["maximum_sector_gross_share"], percentage=True),
                _fmt_number(row["maximum_largest_gross_weight"], percentage=True),
            ]
        )
    entry = assumptions["charge_initial_entry"]
    liquidation = assumptions["charge_terminal_liquidation"]
    entry_text = "charged" if entry is True else "not charged" if entry is False else "unspecified"
    liquidation_text = (
        "charged" if liquidation is True else "not charged" if liquidation is False else "unspecified"
    )
    borrow = assumptions["annual_short_borrow_bps"]
    borrow_text = (
        "unspecified"
        if borrow is None
        else f"{borrow:.1f} bps annual short-borrow, applied only to actual short notional"
    )
    control_rows = [
        [name, "PASS" if passed else "FAIL"]
        for name, passed in sorted(production_controls.items())
    ]
    incremental_rows = []
    for label, key in (
        ("M2 minus M1", "m2_vs_m1_incremental"),
        ("Residual ML minus M2", "residual_ml_vs_m2_incremental"),
    ):
        values = data_quality.get(key, {})
        if not isinstance(values, Mapping):
            values = {}
        incremental_rows.append(
            [
                label,
                _fmt_number(
                    values.get("annualized_arithmetic_return"), percentage=True
                ),
                _fmt_number(values.get("newey_west_tstat")),
                _fmt_number(values.get("bootstrap_sharpe_p05")),
            ]
        )
    lines = [
        "# Alpha Factor Engine v2 — Opened Diagnostic",
        "",
        f"**Diagnostic decision: {diagnostic_status}.** This decision uses only the predeclared opened-test gates listed below.",
        "",
        "**Production decision: BLOCKED — NOT PRODUCTION-ELIGIBLE; DO NOT DEPLOY.**",
        "",
        "M2 Diversified Momentum is the predeclared champion. Opened results cannot promote M3 or the residual-ML challenger.",
        "",
        "## Evidence boundary",
        "",
        f"- Period: {first_date} through {last_date}; the test is OPENED.",
        f"- Universe: {universe_label}.",
        f"- Evidence status: {data_quality.get('evidence_status', 'opened_diagnostic')}.",
        f"- Active-return benchmark: {data_quality.get('active_metric_benchmark', 'declared benchmark')}; M1 is the concentrated momentum baseline.",
        "- Curves use supplied net returns. Cost and capacity evidence remains scenario-based, not realized fills.",
        "",
        "## M1–M3 and residual-ML results",
        "",
        _markdown_table(
            ["Role", "Ann. return", "Sharpe", "Max DD", "Active return", "IR", "Relative max DD"],
            metric_rows,
        ),
        "",
        "## Incremental comparisons",
        "",
        _markdown_table(
            ["Comparison", "Ann. arithmetic increment", "Newey-West t", "Bootstrap Sharpe p05"],
            incremental_rows,
        ),
        "",
        "## Predeclared diagnostic gates",
        "",
        _markdown_table(["Gate group", "Check", "Value", "Requirement", "Result"], gate_table_rows)
        if gate_table_rows
        else "No machine-readable predeclared checks were supplied; diagnostic status is NOT_EVALUATED.",
        "",
        "## Constraint evidence",
        "",
        _markdown_table(
            ["Role", "Median gross", "Median abs(net)", "Max abs(beta)", "Max sector gross", "Max name gross"],
            exposure_rows,
        ),
        "",
        "Missing beta or sector classifications are shown as N/A and cannot support an M3 constraint claim.",
        "",
        (
            "The first opened M3 attempt (unfiltered top-10 with a 1.00 beta cap) was infeasible. "
            "The disclosed M3 diagnostic filters constituents on point-in-time beta before ranking; "
            "this feasibility change was not selected on return performance."
            if data_quality.get("m3_unfiltered_beta1_trial_status")
            else "No prior M3 feasibility attempt was disclosed."
        ),
        "",
        "## Costs and capacity",
        "",
        f"- Base trading cost: {assumptions['trading_cost_bps_one_way']:.1f} bps one-way.",
        f"- Initial entry: {entry_text}; terminal liquidation: {liquidation_text}.",
        f"- Borrow: {borrow_text}. Long-only exposure has no borrow charge.",
        f"- Capacity scenario: ${assumptions['portfolio_aum_usd'] / 1_000_000:.2f} million AUM and {assumptions['maximum_adv_participation']:.2%} maximum trailing-ADV participation.",
        "",
        _markdown_table(
            ["Role", "AUM", "Observed max ADV", "ADV limit", "Violations", "At configured scenario"],
            capacity_rows,
        ),
        "",
        f"Cost sensitivity contains {len(cost_sensitivity):d} supplied strategy/cost observations. It is an opened diagnostic, not an estimate of live slippage.",
        "",
        "## Production controls",
        "",
        _markdown_table(["Control", "Result"], control_rows),
        "",
        "## Binding production blockers",
        "",
        *[f"- {blocker}" for blocker in blockers],
        "",
        "## Charts",
        "",
        *[f"- [{chart.name}](charts/{chart.name})" for chart in chart_paths],
        "",
        "These artifacts support research diagnosis only. They are not a live-trading recommendation or production approval.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def create_v2_opened_diagnostic_report(
    output_dir: str | Path,
    *,
    returns: Mapping[str, pd.Series | pd.DataFrame],
    holdings: Mapping[str, pd.DataFrame],
    metrics: pd.DataFrame,
    cost_sensitivity: pd.DataFrame,
    capacity_summary: pd.DataFrame,
    predeclared_gates: Mapping[str, Any],
    data_quality: Mapping[str, Any],
    assumptions: Mapping[str, Any],
) -> dict[str, Any]:
    """Write the fixed-role v2 opened-diagnostic report.

    Parameters are keyword-only after ``output_dir``. ``returns`` and
    ``holdings`` must be keyed by exactly ``V2_ROLE_ORDER``.  M2 is fixed as
    champion.  ``predeclared_gates`` should contain machine-readable
    ``stage_gates``, ``research_checks``, or ``checks`` specifications with a
    value and minimum/maximum/required threshold.

    Returns a dictionary with report/chart paths, the fixed role map,
    diagnostic status, production status, failed gates, production blockers,
    and configured-scenario capacity status.
    """
    if not isinstance(predeclared_gates, Mapping):
        raise TypeError("predeclared_gates must be a mapping")
    if not isinstance(data_quality, Mapping):
        raise TypeError("data_quality must be a mapping")
    _validate_opened(predeclared_gates, data_quality)
    _require_role_mapping("returns", returns)
    normalized_assumptions = _assumption_payload(assumptions)
    normalized_returns = _normalize_returns(
        returns, opened_start=normalized_assumptions["opened_start"]
    )
    annualization = normalized_assumptions["annualization_factor"]
    summary = _opened_summary(normalized_returns, metrics, annualization)
    exposures = _holding_exposures(holdings, normalized_assumptions["opened_start"])
    normalized_costs = _normalize_costs(cost_sensitivity)

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    normalized_capacity = _normalize_capacity(
        root, capacity_summary, data_quality, normalized_assumptions
    )
    checks = _gate_rows(predeclared_gates)
    diagnostic_status = _diagnostic_status(predeclared_gates, checks)
    production_controls = _production_controls(predeclared_gates, data_quality)
    current_ticker = _current_ticker_limitation(data_quality)
    capacity_lookup = normalized_capacity.set_index("strategy")
    champion_capacity_pass = bool(
        capacity_lookup.loc[V2_CHAMPION_ROLE, "capacity_passed_at_configured_aum"]
    )
    all_capacity_roles_present = set(normalized_capacity["strategy"]) == set(V2_ROLE_ORDER)

    m3 = exposures.loc[exposures["strategy"].eq("m3_sector_beta_constrained")].iloc[0]
    blockers = ["The holdout/test results have been opened."]
    if current_ticker:
        blockers.append(
            "The universe is a current-ticker/survivorship-biased proxy rather than a PIT permanent-ID panel."
        )
    blockers.extend(
        f"Production data control failed: {name}."
        for name, passed in sorted(production_controls.items())
        if not passed
    )
    if not champion_capacity_pass:
        blockers.append("M2 does not pass capacity at the configured AUM/ADV scenario.")
    if not all_capacity_roles_present:
        blockers.append("Capacity evidence is not supplied for all four v2 roles.")
    if pd.isna(m3["maximum_absolute_beta_exposure"]):
        blockers.append("M3 beta-exposure evidence is missing.")
    if pd.isna(m3["maximum_sector_gross_share"]):
        blockers.append("M3 point-in-time sector-classification evidence is missing.")
    if normalized_assumptions["charge_initial_entry"] is None:
        blockers.append("Initial-entry cost treatment is unspecified.")
    if normalized_assumptions["charge_terminal_liquidation"] is None:
        blockers.append("Terminal-liquidation cost treatment is unspecified.")
    if normalized_assumptions["annual_short_borrow_bps"] is None:
        blockers.append("Short-borrow cost is unspecified for any short sleeve.")

    limitation_label = (
        "OPENED diagnostic | "
        + (
            "CURRENT-TICKER / survivorship-biased universe"
            if current_ticker
            else "PIT permanent-ID universe"
        )
        + " | production BLOCKED"
    )
    chart_paths = _save_charts(
        root / "charts",
        normalized_returns,
        summary,
        exposures,
        normalized_costs,
        normalized_capacity,
        normalized_assumptions,
        limitation_label,
    )
    report_path = root / "v2_opened_diagnostic.md"
    _write_report(
        report_path,
        normalized_returns,
        summary,
        exposures,
        normalized_costs,
        normalized_capacity,
        checks,
        diagnostic_status,
        production_controls,
        data_quality,
        normalized_assumptions,
        current_ticker,
        blockers,
        chart_paths,
    )
    failed_gates = [
        f"{row['group']}.{row['name']}" for row in checks if not bool(row["passed"])
    ]
    return {
        "report": str(report_path),
        "charts": [str(path) for path in chart_paths],
        "strategy_roles": dict(V2_DISPLAY_NAMES),
        "champion_role": V2_CHAMPION_ROLE,
        "champion": V2_DISPLAY_NAMES[V2_CHAMPION_ROLE],
        "diagnostic_status": diagnostic_status,
        "production_status": "BLOCKED",
        "production_decision": "NOT PRODUCTION-ELIGIBLE — DO NOT DEPLOY",
        "opened_test": True,
        "current_ticker_limitation": current_ticker,
        "failed_gates": failed_gates,
        "production_blockers": blockers,
        "capacity_passed_at_configured_aum": champion_capacity_pass,
        "all_capacity_roles_present": all_capacity_roles_present,
    }


__all__ = [
    "V2_CHAMPION_ROLE",
    "V2_DISPLAY_NAMES",
    "V2_ROLE_ORDER",
    "create_v2_opened_diagnostic_report",
]
