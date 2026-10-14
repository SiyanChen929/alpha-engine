"""Champion-challenger diagnostic reporting.

This module deliberately treats the supplied champion as predeclared.  It
never chooses a winner from opened-test returns, which would turn a comparison
chart into an additional model-selection step.
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


_COLORS = (
    "#155EEF",
    "#17B26A",
    "#7A5AF8",
    "#F79009",
    "#D92D20",
    "#0BA5EC",
)
# Performance becomes observable at the holding-period exit.  Prefer that
# common execution timestamp so long-only and long-short curves align exactly.
_DATE_COLUMNS = ("exit_at", "exit_date", "signal_date", "date", "timestamp")
_RETURN_COLUMNS = ("net_return", "return", "returns", "strategy_return")
_NAME_COLUMNS = ("model", "strategy", "sleeve", "name")
_OPENED_PERIODS = ("opened", "opened_test", "holdout", "test", "current_test")
_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "annualized_return": ("annualized_return", "annual_return", "cagr"),
    "annualized_volatility": ("annualized_volatility", "volatility", "annual_volatility"),
    "sharpe": ("sharpe", "sharpe_ratio"),
    "max_drawdown": ("max_drawdown", "max_dd"),
    "active_annualized_return": ("active_annualized_return", "annualized_active_return"),
    "active_information_ratio": ("active_information_ratio", "information_ratio"),
    "relative_max_drawdown": ("relative_max_drawdown", "active_max_drawdown"),
    "newey_west_tstat": ("newey_west_tstat", "hac_tstat", "t_stat", "tstat"),
    "bootstrap_sharpe_p05": ("bootstrap_sharpe_p05", "sharpe_p05"),
    "average_monthly_turnover": ("average_monthly_turnover", "turnover"),
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "font.size": 10,
        }
    )


def _display_name(value: Any) -> str:
    return str(value).removesuffix("_score").replace("_", " ").title()


def _first_present(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    return next((column for column in candidates if column in columns), None)


def _nested(mapping: Mapping[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        value: Any = mapping
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                break
            value = value[key]
        else:
            return value
    return None


def _bool(value: Any) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized in {"PASS", "TRUE", "YES"}:
            return True
        if normalized in {"FAIL", "FALSE", "NO"}:
            return False
    return None


def _opened_start(config: Mapping[str, Any]) -> pd.Timestamp | None:
    value = _nested(
        config,
        ("research", "holdout_start"),
        ("research", "opened_start"),
        ("research", "evaluation_entry_at_or_after"),
        ("holdout_start",),
        ("opened_start",),
    )
    if value in (None, ""):
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _normalize_returns(
    returns: Mapping[str, pd.Series | pd.DataFrame],
    *,
    opened_start: pd.Timestamp | None,
) -> dict[str, pd.DataFrame]:
    if not isinstance(returns, Mapping) or not returns:
        raise ValueError("returns must be a non-empty mapping of strategy names to returns")
    normalized: dict[str, pd.DataFrame] = {}
    for raw_name, raw_values in returns.items():
        name = str(raw_name)
        if isinstance(raw_values, pd.Series):
            values = pd.DataFrame(
                {"date": raw_values.index, "net_return": raw_values.to_numpy(copy=True)}
            )
        elif isinstance(raw_values, pd.DataFrame):
            date_column = _first_present(raw_values.columns, _DATE_COLUMNS)
            return_column = _first_present(raw_values.columns, _RETURN_COLUMNS)
            if date_column is None or return_column is None:
                raise ValueError(
                    f"returns[{name!r}] needs a date column {_DATE_COLUMNS} and "
                    f"return column {_RETURN_COLUMNS}"
                )
            values = raw_values[[date_column, return_column]].rename(
                columns={date_column: "date", return_column: "net_return"}
            )
        else:
            raise TypeError(f"returns[{name!r}] must be a pandas Series or DataFrame")
        values["date"] = pd.to_datetime(values["date"], utc=True, errors="coerce")
        values["net_return"] = pd.to_numeric(values["net_return"], errors="coerce")
        if values.isna().any(axis=None):
            raise ValueError(f"returns[{name!r}] contains missing or invalid dates/returns")
        if not np.isfinite(values["net_return"].to_numpy(float)).all():
            raise ValueError(f"returns[{name!r}] contains non-finite returns")
        if values["net_return"].lt(-1.0).any():
            raise ValueError(f"returns[{name!r}] contains a return below -100%")
        if values["date"].duplicated().any():
            raise ValueError(f"returns[{name!r}] contains duplicate dates")
        values = values.sort_values("date", kind="mergesort")
        if opened_start is not None:
            values = values.loc[values["date"] >= opened_start]
        if values.empty:
            raise ValueError(f"returns[{name!r}] has no observations in the opened period")
        normalized[name] = values.reset_index(drop=True)
    return normalized


def _annualization_factor(config: Mapping[str, Any]) -> int:
    configured = _nested(
        config,
        ("reporting", "annualization_factor"),
        ("research", "annualization_factor"),
        ("annualization_factor",),
    )
    if configured is None:
        rebalance = str(_nested(config, ("research", "rebalance")) or "monthly").lower()
        return 252 if "day" in rebalance else 52 if "week" in rebalance else 12
    factor = int(configured)
    if factor <= 0:
        raise ValueError("annualization_factor must be positive")
    return factor


def _return_statistics(values: pd.Series, annualization: int) -> dict[str, float]:
    array = values.to_numpy(float)
    months = len(array)
    wealth = np.cumprod(1.0 + array)
    annualized_return = (
        float(wealth[-1] ** (annualization / months) - 1.0)
        if wealth[-1] > 0 and months
        else -1.0
    )
    volatility = float(np.std(array, ddof=1) * np.sqrt(annualization)) if months > 1 else np.nan
    sharpe = (
        float(np.mean(array) / np.std(array, ddof=1) * np.sqrt(annualization))
        if months > 1 and np.std(array, ddof=1) > 0
        else np.nan
    )
    running_peak = np.maximum.accumulate(wealth)
    max_drawdown = float(np.min(wealth / running_peak - 1.0))
    return {
        "months": float(months),
        "annualized_return": annualized_return,
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


def _opened_metric_rows(metrics: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    if not isinstance(metrics, pd.DataFrame) or metrics.empty:
        raise ValueError("metrics must be a non-empty DataFrame")
    name_column = _first_present(metrics.columns, _NAME_COLUMNS)
    if name_column is None:
        if metrics.index.name in _NAME_COLUMNS or not isinstance(metrics.index, pd.RangeIndex):
            result = metrics.reset_index().rename(columns={metrics.index.name or "index": "model"})
            name_column = "model"
        else:
            raise ValueError(f"metrics needs a strategy-name column {_NAME_COLUMNS}")
    else:
        result = metrics.copy()
    if "period" in result:
        period = result["period"].astype(str).str.lower()
        opened = result.loc[period.isin(_OPENED_PERIODS)].copy()
        if not opened.empty:
            result = opened
    result[name_column] = result[name_column].astype(str)
    if result[name_column].duplicated().any():
        raise ValueError("metrics contains more than one opened row per strategy")
    return result, name_column


def _summary_table(
    normalized_returns: Mapping[str, pd.DataFrame],
    metrics: pd.DataFrame,
    *,
    annualization: int,
) -> pd.DataFrame:
    rows = []
    opened_metrics, name_column = _opened_metric_rows(metrics)
    metric_lookup = opened_metrics.set_index(name_column) if name_column else pd.DataFrame()
    for name, values in normalized_returns.items():
        row: dict[str, Any] = {"strategy": name}
        row.update(_return_statistics(values["net_return"], annualization))
        if name in metric_lookup.index:
            supplied = metric_lookup.loc[name]
            for canonical, aliases in _METRIC_ALIASES.items():
                column = _first_present(supplied.index, aliases)
                if column is not None and pd.notna(supplied[column]):
                    row[canonical] = float(supplied[column])
        rows.append(row)
    return pd.DataFrame(rows)


def _champion_name(
    returns: Mapping[str, pd.DataFrame],
    gates: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[str, str]:
    explicit = _nested(
        config,
        ("reporting", "champion"),
        ("champion",),
    ) or _nested(gates, ("champion",))
    names = list(returns)
    if explicit is not None:
        explicit = str(explicit)
        if explicit not in returns:
            raise ValueError(f"predeclared champion {explicit!r} is absent from returns")
        return explicit, "explicit configuration"
    if "champion" in returns:
        return "champion", "champion mapping key"
    if "ensemble_score" in returns:
        return "ensemble_score", "ensemble convention"
    if len(names) == 1:
        return names[0], "only supplied candidate"
    return names[0], "input mapping order; configure reporting.champion for formal use"


def _holding_statistics(
    holdings: Mapping[str, pd.DataFrame],
    *,
    opened_start: pd.Timestamp | None,
) -> pd.DataFrame:
    if not isinstance(holdings, Mapping) or not holdings:
        raise ValueError("holdings must be a non-empty mapping")
    rows: list[dict[str, Any]] = []
    for raw_name, raw_frame in holdings.items():
        name = str(raw_name)
        if not isinstance(raw_frame, pd.DataFrame) or raw_frame.empty:
            raise ValueError(f"holdings[{name!r}] must be a non-empty DataFrame")
        weight_column = _first_present(raw_frame.columns, ("weight", "target_weight", "position_weight"))
        if weight_column is None:
            raise ValueError(f"holdings[{name!r}] needs a weight column")
        date_column = _first_present(raw_frame.columns, ("signal_date", "date", "entry_at", "timestamp"))
        values = raw_frame.copy()
        if date_column is None:
            values["_holding_date"] = pd.Timestamp("1970-01-01", tz="UTC")
            date_column = "_holding_date"
        else:
            values[date_column] = pd.to_datetime(values[date_column], utc=True, errors="coerce")
            if values[date_column].isna().any():
                raise ValueError(f"holdings[{name!r}] contains invalid dates")
            if opened_start is not None:
                values = values.loc[values[date_column] >= opened_start]
        values[weight_column] = pd.to_numeric(values[weight_column], errors="coerce")
        if values[weight_column].isna().any() or not np.isfinite(
            values[weight_column].to_numpy(float)
        ).all():
            raise ValueError(f"holdings[{name!r}] contains invalid weights")
        if values.empty:
            continue
        daily_rows = []
        for _, group in values.groupby(date_column, sort=True):
            weights = group[weight_column].to_numpy(float)
            weights = weights[np.abs(weights) > 1e-12]
            gross = float(np.abs(weights).sum())
            net = float(weights.sum())
            effective_n = float(gross**2 / np.square(weights).sum()) if len(weights) and gross else 0.0
            daily_rows.append(
                {
                    "active_positions": float(len(weights)),
                    "effective_n": effective_n,
                    "gross_exposure": gross,
                    "absolute_net_exposure": abs(net),
                    "largest_gross_weight": float(np.max(np.abs(weights)) / gross)
                    if len(weights) and gross
                    else 0.0,
                }
            )
        daily = pd.DataFrame(daily_rows)
        rows.append(
            {
                "strategy": name,
                **{f"median_{column}": float(daily[column].median()) for column in daily},
            }
        )
    return pd.DataFrame(rows)


def _normalize_cost_sensitivity(
    frame: pd.DataFrame,
    *,
    champion: str,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("cost_sensitivity must be a non-empty DataFrame")
    cost_column = _first_present(frame.columns, ("cost_bps", "trading_cost_bps", "one_way_cost_bps"))
    return_column = _first_present(frame.columns, _METRIC_ALIASES["annualized_return"])
    sharpe_column = _first_present(frame.columns, _METRIC_ALIASES["sharpe"])
    if cost_column is None or return_column is None or sharpe_column is None:
        raise ValueError("cost_sensitivity needs cost_bps, annualized return, and Sharpe columns")
    name_column = _first_present(frame.columns, _NAME_COLUMNS)
    result = frame.copy().rename(
        columns={
            cost_column: "cost_bps",
            return_column: "annualized_return",
            sharpe_column: "sharpe",
        }
    )
    if name_column is None:
        result["strategy"] = champion
    else:
        result = result.rename(columns={name_column: "strategy"})
    for column in ("cost_bps", "annualized_return", "sharpe"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[["cost_bps", "annualized_return", "sharpe"]].isna().any(axis=None):
        raise ValueError("cost_sensitivity contains invalid numeric values")
    has_active = {
        "active_annualized_return",
        "active_information_ratio",
    }.issubset(result.columns)
    if has_active:
        for column in ("active_annualized_return", "active_information_ratio"):
            result[column] = pd.to_numeric(result[column], errors="coerce")
        if result[["active_annualized_return", "active_information_ratio"]].isna().any(
            axis=None
        ):
            raise ValueError("cost_sensitivity contains invalid active-return values")
        result["comparison_return"] = result["active_annualized_return"]
        result["comparison_ratio"] = result["active_information_ratio"]
        result.attrs["return_label"] = "Benchmark-relative active return"
        result.attrs["ratio_label"] = "Active information ratio"
    else:
        result["comparison_return"] = result["annualized_return"]
        result["comparison_ratio"] = result["sharpe"]
        result.attrs["return_label"] = "Net total return"
        result.attrs["ratio_label"] = "Sharpe ratio"
    return result.sort_values(["strategy", "cost_bps"], kind="mergesort")


def _cost_assumptions(config: Mapping[str, Any]) -> dict[str, Any]:
    trading_cost = _nested(
        config,
        ("costs", "trading_cost_bps_one_way"),
        ("portfolio", "trading_cost_bps_one_way"),
    )
    borrow_cost = _nested(
        config,
        ("neutral_portfolio", "annual_short_borrow_bps"),
        ("portfolio", "annual_short_borrow_bps"),
    )
    return {
        "trading_cost_bps_one_way": None if trading_cost is None else float(trading_cost),
        "annual_short_borrow_bps": None if borrow_cost is None else float(borrow_cost),
        "portfolio_aum_usd": _nested(config, ("costs", "portfolio_aum_usd")),
        "maximum_adv_participation": _nested(
            config, ("costs", "maximum_adv_participation")
        ),
        "charge_initial_entry": _bool(
            _nested(config, ("costs", "charge_initial_entry"))
        ),
        "charge_terminal_liquidation": _bool(
            _nested(config, ("costs", "charge_terminal_liquidation"))
        ),
    }


def _capacity_payload_from_quality(data_quality: Mapping[str, Any]) -> pd.DataFrame | None:
    raw = _nested(
        data_quality,
        ("capacity_summary",),
        ("capacity_by_strategy",),
        ("capacity",),
        ("capacity_scenario",),
    )
    if raw is None:
        scalar_keys = {
            "portfolio_aum_usd",
            "maximum_adv_participation_limit",
            "maximum_adv_participation_observed",
            "capacity_violations",
            "capacity_passed",
        }
        if "maximum_adv_participation_observed" in data_quality:
            raw = {key: data_quality[key] for key in scalar_keys if key in data_quality}
    if raw is None:
        return None
    if isinstance(raw, pd.DataFrame):
        return raw.copy()
    if isinstance(raw, list):
        return pd.DataFrame(raw)
    if isinstance(raw, Mapping):
        if raw and all(isinstance(value, Mapping) for value in raw.values()):
            return pd.DataFrame(
                [{"strategy": strategy, **dict(values)} for strategy, values in raw.items()]
            )
        return pd.DataFrame([dict(raw)])
    raise TypeError("data_quality capacity evidence must be tabular or a mapping")


def _normalize_capacity_summary(
    root: Path,
    capacity_summary: pd.DataFrame | None,
    *,
    data_quality: Mapping[str, Any],
    config: Mapping[str, Any],
    champion: str,
) -> tuple[pd.DataFrame, str]:
    source = "not supplied"
    frame: pd.DataFrame | None
    if capacity_summary is not None:
        if not isinstance(capacity_summary, pd.DataFrame):
            raise TypeError("capacity_summary must be a DataFrame or None")
        frame = capacity_summary.copy()
        source = "function input"
    else:
        path = root / "capacity_summary.csv"
        if path.exists():
            frame = pd.read_csv(path)
            source = path.name
        else:
            frame = _capacity_payload_from_quality(data_quality)
            if frame is not None:
                source = "data_quality"
    if frame is None or frame.empty:
        return pd.DataFrame(), source

    if {"metric", "value"}.issubset(frame.columns):
        if "strategy" not in frame:
            frame["strategy"] = champion
        frame = (
            frame.pivot_table(
                index="strategy", columns="metric", values="value", aggfunc="first"
            )
            .reset_index()
            .rename_axis(columns=None)
        )
    name_column = _first_present(frame.columns, _NAME_COLUMNS)
    if name_column is None:
        frame["strategy"] = champion
    elif name_column != "strategy":
        frame = frame.rename(columns={name_column: "strategy"})

    aliases = {
        "portfolio_aum_usd": ("portfolio_aum_usd", "aum_usd"),
        "maximum_adv_participation_limit": (
            "maximum_adv_participation_limit",
            "maximum_adv_participation",
            "adv_participation_limit",
        ),
        "maximum_adv_participation_observed": (
            "maximum_adv_participation_observed",
            "max_adv_participation_observed",
            "observed_adv_participation",
        ),
        "capacity_violations": ("capacity_violations", "violations"),
        "capacity_passed": ("capacity_passed", "passed"),
    }
    for canonical, candidates in aliases.items():
        column = _first_present(frame.columns, candidates)
        if column is not None and column != canonical:
            frame = frame.rename(columns={column: canonical})

    assumptions = _cost_assumptions(config)
    configured_aum = assumptions["portfolio_aum_usd"]
    configured_limit = assumptions["maximum_adv_participation"]
    if "portfolio_aum_usd" not in frame:
        frame["portfolio_aum_usd"] = configured_aum
    if "maximum_adv_participation_limit" not in frame:
        frame["maximum_adv_participation_limit"] = configured_limit
    required = {
        "portfolio_aum_usd",
        "maximum_adv_participation_limit",
        "maximum_adv_participation_observed",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"capacity_summary missing columns: {sorted(missing)}")

    numeric_columns = [
        "portfolio_aum_usd",
        "maximum_adv_participation_limit",
        "maximum_adv_participation_observed",
    ]
    if "capacity_violations" in frame:
        numeric_columns.append("capacity_violations")
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[numeric_columns].isna().any(axis=None) or not np.isfinite(
        frame[numeric_columns].to_numpy(float)
    ).all():
        raise ValueError("capacity_summary contains invalid numeric values")
    if (
        frame["portfolio_aum_usd"].le(0).any()
        or frame["maximum_adv_participation_limit"].le(0).any()
        or frame["maximum_adv_participation_observed"].lt(0).any()
        or (
            "capacity_violations" in frame
            and (
                frame["capacity_violations"].lt(0).any()
                or not np.allclose(
                    frame["capacity_violations"],
                    np.round(frame["capacity_violations"]),
                )
            )
        )
    ):
        raise ValueError("capacity AUM, participation, or violation values are outside valid ranges")

    observed_pass = frame["maximum_adv_participation_observed"].le(
        frame["maximum_adv_participation_limit"]
    )
    if "capacity_violations" not in frame:
        frame["capacity_violations"] = (~observed_pass).astype(int)
    if "capacity_passed" in frame:
        converted = frame["capacity_passed"].map(_bool)
        if converted.isna().any():
            raise ValueError("capacity_passed must contain booleans")
        frame["capacity_passed"] = converted.astype(bool)
    else:
        frame["capacity_passed"] = observed_pass & frame["capacity_violations"].eq(0)

    if configured_aum is None or configured_limit is None:
        frame["configured_aum_matches"] = False
        frame["configured_limit_matches"] = False
    else:
        configured_aum = float(configured_aum)
        configured_limit = float(configured_limit)
        frame["configured_aum_matches"] = np.isclose(
            frame["portfolio_aum_usd"], configured_aum, rtol=1e-9, atol=0
        )
        frame["configured_limit_matches"] = np.isclose(
            frame["maximum_adv_participation_limit"],
            configured_limit,
            rtol=1e-9,
            atol=1e-12,
        )
    frame["capacity_passed_at_configured_aum"] = (
        frame["capacity_passed"]
        & frame["configured_aum_matches"]
        & frame["configured_limit_matches"]
        & observed_pass
        & frame["capacity_violations"].eq(0)
    )
    frame["strategy"] = frame["strategy"].astype(str)
    if frame["strategy"].duplicated().any():
        raise ValueError("capacity_summary contains duplicate strategy rows")
    frame["_champion_first"] = frame["strategy"].ne(champion)
    frame = frame.sort_values(["_champion_first", "strategy"], kind="mergesort").drop(
        columns="_champion_first"
    )
    return frame.reset_index(drop=True), source


def _research_checks(
    gates: Mapping[str, Any],
    summary: pd.DataFrame,
    champion: str,
) -> tuple[str, list[dict[str, Any]]]:
    supplied_status = str(gates.get("research_status", "")).upper()
    checks: list[dict[str, Any]] = []
    supplied_checks = gates.get("research_checks")
    if isinstance(supplied_checks, Mapping):
        for name, specification in supplied_checks.items():
            if not isinstance(specification, Mapping) or "value" not in specification:
                continue
            value = float(specification["value"])
            if "minimum" in specification:
                threshold = float(specification["minimum"])
                passed = value >= threshold
                relation = ">="
            elif "maximum" in specification:
                threshold = float(specification["maximum"])
                passed = value <= threshold
                relation = "<="
            else:
                continue
            checks.append(
                {
                    "name": str(name),
                    "value": value,
                    "relation": relation,
                    "threshold": threshold,
                    "passed": passed,
                }
            )
    if not checks:
        row = summary.loc[summary["strategy"].eq(champion)].iloc[0]
        configured = (
            ("holdout_sharpe", "sharpe", "holdout_sharpe_minimum", ">="),
            ("newey_west_tstat", "newey_west_tstat", "newey_west_tstat_minimum", ">="),
            (
                "bootstrap_sharpe_p05",
                "bootstrap_sharpe_p05",
                "bootstrap_sharpe_p05_minimum",
                ">=",
            ),
            ("max_drawdown", "max_drawdown", "maximum_drawdown_floor", ">="),
        )
        for label, metric, gate_name, relation in configured:
            if gate_name not in gates:
                continue
            value = float(row[metric]) if metric in row and pd.notna(row[metric]) else np.nan
            threshold = float(gates[gate_name])
            checks.append(
                {
                    "name": label,
                    "value": value,
                    "relation": relation,
                    "threshold": threshold,
                    "passed": bool(np.isfinite(value) and value >= threshold),
                }
            )
    computed = "PASS" if checks and all(item["passed"] for item in checks) else "FAIL" if checks else "NOT_EVALUATED"
    if supplied_status == "FAIL" or computed == "FAIL":
        return "FAIL", checks
    if supplied_status == "PASS" and computed in {"PASS", "NOT_EVALUATED"}:
        return "PASS", checks
    return computed, checks


def _production_checks(
    gates: Mapping[str, Any], data_quality: Mapping[str, Any]
) -> tuple[str, dict[str, bool]]:
    critical = (
        "survivorship_free",
        "delisting_returns",
        "historical_borrow",
        "historical_industry_classification",
        "alfred_vintages_used",
        "execution_grade_quotes",
    )
    raw_checks = gates.get("production_data_checks")
    checks: dict[str, bool] = {}
    if isinstance(raw_checks, Mapping):
        for name, value in raw_checks.items():
            converted = _bool(value)
            if converted is not None:
                checks[str(name)] = converted
    for name in critical:
        if name not in checks:
            converted = _bool(data_quality.get(name))
            checks[name] = False if converted is None else converted
    execution_check_supplied = isinstance(raw_checks, Mapping) and (
        "execution_grade_quotes" in raw_checks
    )
    if not execution_check_supplied:
        execution = str(data_quality.get("execution_data", "")).lower()
        if execution:
            checks["execution_grade_quotes"] = not any(
                marker in execution for marker in ("proxy", "no taq", "no nbbo")
            )
    supplied_status = str(gates.get("production_status", "")).upper()
    if supplied_status == "FAIL" or any(not passed for passed in checks.values()):
        return "FAIL", checks
    if supplied_status == "PASS" and checks and all(checks.values()):
        return "PASS", checks
    return ("PASS" if checks and all(checks.values()) else "NOT_EVALUATED"), checks


def _limitation_label(
    gates: Mapping[str, Any], data_quality: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[str, bool, bool]:
    opened_value = _bool(gates.get("holdout_opened"))
    if opened_value is None:
        opened_value = _bool(_nested(config, ("research", "holdout_opened")))
    evidence_status = str(
        gates.get("historical_evidence_status")
        or data_quality.get("evidence_status")
        or _nested(config, ("research", "evidence_status"))
        or ""
    ).lower()
    if "opened" in evidence_status:
        opened_value = True
    opened = True if opened_value is None else opened_value
    survivorship_free = _bool(data_quality.get("survivorship_free"))
    source = " ".join(
        str(value or "").lower()
        for value in (
            _nested(config, ("universe", "source")),
            _nested(config, ("universe", "mode")),
            data_quality.get("universe_mode"),
        )
    )
    current_ticker = survivorship_free is not True or "current" in source or "proxy" in source
    test_text = "OPENED diagnostic" if opened else "sealed diagnostic"
    universe_text = "CURRENT-TICKER / survivorship-biased universe" if current_ticker else "PIT universe"
    return f"{test_text} | {universe_text} | not production evidence", opened, current_ticker


def _save(fig: plt.Figure, path: Path, limitation_label: str) -> None:
    fig.text(0.5, 0.008, limitation_label, ha="center", va="bottom", fontsize=8, color="#667085")
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _create_charts(
    chart_dir: Path,
    normalized_returns: Mapping[str, pd.DataFrame],
    summary: pd.DataFrame,
    holding_summary: pd.DataFrame,
    cost_sensitivity: pd.DataFrame,
    capacity_summary: pd.DataFrame,
    cost_assumptions: Mapping[str, Any],
    *,
    champion: str,
    annualization: int,
    limitation_label: str,
) -> list[Path]:
    _style()
    chart_dir.mkdir(parents=True, exist_ok=True)
    names = [champion, *[name for name in normalized_returns if name != champion]]
    colors = {name: _COLORS[index % len(_COLORS)] for index, name in enumerate(names)}
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(10.4, 5.6))
    for name in names:
        values = normalized_returns[name]
        wealth = (1.0 + values["net_return"]).cumprod()
        ax.plot(
            values["date"],
            wealth,
            lw=2.4 if name == champion else 1.7,
            color=colors[name],
            label=_display_name(name),
        )
    ax.axhline(1.0, color="#667085", lw=1)
    ax.set(title="Opened champion-challenger net wealth", xlabel="Evaluation date", ylabel="Growth of $1")
    ax.legend(frameon=False, ncol=2)
    path = chart_dir / "opened_cumulative_wealth.png"
    _save(fig, path, limitation_label)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(10.4, 5.2))
    for name in names:
        values = normalized_returns[name]
        wealth = (1.0 + values["net_return"]).cumprod()
        drawdown = wealth / wealth.cummax() - 1.0
        ax.plot(
            values["date"],
            drawdown * 100,
            lw=2.2 if name == champion else 1.5,
            color=colors[name],
            label=_display_name(name),
        )
    ax.axhline(0, color="#667085", lw=1)
    ax.set(title="Opened-test drawdown", xlabel="Evaluation date", ylabel="Drawdown (%)")
    ax.legend(frameon=False, ncol=2)
    path = chart_dir / "opened_drawdowns.png"
    _save(fig, path, limitation_label)
    paths.append(path)

    window = annualization
    minimum = max(3, window // 2)
    fig, ax = plt.subplots(figsize=(10.4, 5.2))
    for name in names:
        values = normalized_returns[name]
        rolling_mean = values["net_return"].rolling(window, min_periods=minimum).mean()
        rolling_std = values["net_return"].rolling(window, min_periods=minimum).std(ddof=1)
        rolling_sharpe = rolling_mean.div(rolling_std.replace(0, np.nan)) * np.sqrt(annualization)
        ax.plot(
            values["date"],
            rolling_sharpe,
            lw=2.2 if name == champion else 1.5,
            color=colors[name],
            label=_display_name(name),
        )
    ax.axhline(0, color="#667085", lw=1)
    ax.set(
        title=f"Rolling {window}-observation Sharpe",
        xlabel="Evaluation date",
        ylabel="Annualized Sharpe",
    )
    ax.legend(frameon=False, ncol=2)
    path = chart_dir / "opened_rolling_sharpe.png"
    _save(fig, path, limitation_label)
    paths.append(path)

    ordered = summary.set_index("strategy").reindex(names).reset_index()
    display_names = [_display_name(name) for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.4), sharey=True)
    palette = [colors[name] for name in names]
    active_available = (
        "active_annualized_return" in ordered
        and "active_information_ratio" in ordered
        and ordered["active_annualized_return"].notna().any()
        and ordered["active_information_ratio"].notna().any()
    )
    metric_names = names
    metric_palette = palette
    metric_labels = display_names
    metric_rows = ordered
    if active_available:
        eligible = ordered[["active_annualized_return", "active_information_ratio"]].notna().all(
            axis=1
        )
        metric_rows = ordered.loc[eligible].copy()
        metric_names = metric_rows["strategy"].tolist()
        metric_labels = [_display_name(name) for name in metric_names]
        metric_palette = [colors[name] for name in metric_names]
    left_column = "active_annualized_return" if active_available else "annualized_return"
    right_column = "active_information_ratio" if active_available else "sharpe"
    metric_positions = np.arange(len(metric_labels))
    axes[0].barh(metric_positions, metric_rows[left_column] * 100, color=metric_palette)
    axes[0].axvline(0, color="#667085", lw=1)
    axes[0].set_yticks(metric_positions, metric_labels)
    axes[0].invert_yaxis()
    axes[0].set(
        title="Opened benchmark-relative active return"
        if active_available
        else "Opened annualized net return",
        xlabel="Percent",
    )
    axes[1].barh(metric_positions, metric_rows[right_column], color=metric_palette)
    axes[1].axvline(0, color="#667085", lw=1)
    axes[1].set(
        title="Opened active information ratio" if active_available else "Opened Sharpe ratio",
        xlabel="Information ratio" if active_available else "Sharpe",
    )
    path = chart_dir / "opened_metric_comparison.png"
    _save(fig, path, limitation_label)
    paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    cost_return_label = cost_sensitivity.attrs.get("return_label", "Net total return")
    cost_ratio_label = cost_sensitivity.attrs.get("ratio_label", "Sharpe ratio")
    for name, values in cost_sensitivity.groupby("strategy", sort=False):
        color = colors.get(str(name), "#667085")
        axes[0].plot(
            values["cost_bps"],
            values["comparison_return"] * 100,
            marker="o",
            color=color,
            label=_display_name(name),
        )
        axes[1].plot(
            values["cost_bps"],
            values["comparison_ratio"],
            marker="o",
            color=color,
            label=_display_name(name),
        )
    axes[0].axhline(0, color="#667085", lw=1)
    axes[0].set(
        title=f"Cost sensitivity: {cost_return_label.lower()}",
        xlabel="Assumed one-way cost (bps)",
        ylabel=f"{cost_return_label} (%)",
    )
    axes[1].axhline(0, color="#667085", lw=1)
    axes[1].set(
        title=f"Cost sensitivity: {cost_ratio_label.lower()}",
        xlabel="Assumed one-way cost (bps)",
        ylabel=cost_ratio_label,
    )
    for ax in axes:
        ax.legend(frameon=False)
        ax.set_xticks(sorted(cost_sensitivity["cost_bps"].unique()))
    path = chart_dir / "opened_cost_sensitivity.png"
    _save(fig, path, limitation_label)
    paths.append(path)

    holdings = holding_summary.set_index("strategy").reindex(names)
    positions = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(14.2, max(5.2, 0.48 * len(names))))
    axes[0].barh(positions, holdings["median_effective_n"], color=palette)
    axes[0].set_yticks(positions, display_names)
    axes[0].invert_yaxis()
    axes[0].set(title="Median effective N", xlabel="Effective positions")
    if not capacity_summary.empty:
        capacity_names = [name for name in names if name in set(capacity_summary["strategy"])]
        capacity_positions = np.arange(len(capacity_names))
        capacity_labels = [_display_name(name) for name in capacity_names]
        capacity_palette = [colors[name] for name in capacity_names]
        capacity = capacity_summary.set_index("strategy").reindex(capacity_names)
        observed = capacity["maximum_adv_participation_observed"] * 100
        limits = capacity["maximum_adv_participation_limit"] * 100
        axes[1].barh(capacity_positions, observed, color=capacity_palette)
        axes[1].scatter(
            limits,
            capacity_positions,
            marker="|",
            s=260,
            linewidths=2.2,
            color="#D92D20",
            label="Limit",
        )
        axes[1].set_yticks(capacity_positions, capacity_labels)
        axes[1].invert_yaxis()
        axes[1].set(title="Maximum trade / ADV", xlabel="Percent of trailing ADV")
        axes[1].legend(frameon=False)
        axes[2].barh(capacity_positions, capacity["capacity_violations"], color=capacity_palette)
        axes[2].set_yticks(capacity_positions, capacity_labels)
        axes[2].invert_yaxis()
        axes[2].set(title="Capacity-limit violations", xlabel="Trade count")
        aum = cost_assumptions.get("portfolio_aum_usd")
        if aum is not None:
            fig.suptitle(f"Holdings and capacity at ${float(aum) / 1_000_000:.2f} million AUM")
    else:
        axes[1].barh(positions, holdings["median_active_positions"], color=palette)
        axes[1].set_yticks(positions, display_names)
        axes[1].invert_yaxis()
        axes[1].set(title="Median active positions", xlabel="Count")
        axes[2].barh(positions, holdings["median_gross_exposure"], label="Gross", color="#155EEF")
        axes[2].barh(
            positions,
            holdings["median_absolute_net_exposure"],
            label="|Net|",
            color="#F79009",
        )
        axes[2].set_yticks(positions, display_names)
        axes[2].invert_yaxis()
        axes[2].set(title="Median exposure", xlabel="Portfolio exposure")
        axes[2].legend(frameon=False)
    path = chart_dir / "opened_holdings_diagnostics.png"
    _save(fig, path, limitation_label)
    paths.append(path)
    return paths


def _percent(value: Any) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{float(value):.2%}"


def _number(value: Any) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{float(value):.2f}"


def _metric_markdown(summary: pd.DataFrame, names: Sequence[str]) -> str:
    indexed = summary.set_index("strategy")
    lines = [
        "| Candidate | Months | Ann. net return | Active return | Sharpe | Active IR | Max drawdown | Relative max drawdown |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        row = indexed.loc[name]
        lines.append(
            f"| {name} | {int(row['months'])} | {_percent(row['annualized_return'])} | "
            f"{_percent(row.get('active_annualized_return'))} | {_number(row['sharpe'])} | "
            f"{_number(row.get('active_information_ratio'))} | {_percent(row['max_drawdown'])} | "
            f"{_percent(row.get('relative_max_drawdown'))} |"
        )
    return "\n".join(lines)


def _gate_markdown(checks: Sequence[Mapping[str, Any]]) -> str:
    if not checks:
        return "No machine-readable research thresholds were supplied; status is diagnostic only."
    lines = ["| Research check | Observed | Requirement | Result |", "|---|---:|---:|---|"]
    for check in checks:
        lines.append(
            f"| {check['name']} | {float(check['value']):.3f} | "
            f"{check['relation']} {float(check['threshold']):.3f} | "
            f"{'PASS' if check['passed'] else 'FAIL'} |"
        )
    return "\n".join(lines)


def _supplemental_gate_markdown(checks: Any) -> str:
    if not isinstance(checks, Mapping) or not checks:
        return "No supplemental checks were supplied."
    lines = ["| Check | Observed | Requirement | Result |", "|---|---:|---:|---|"]
    for name, raw in checks.items():
        if not isinstance(raw, Mapping):
            continue
        value = pd.to_numeric(pd.Series([raw.get("actual")]), errors="coerce").iloc[0]
        threshold = pd.to_numeric(pd.Series([raw.get("requirement")]), errors="coerce").iloc[0]
        relation = "<=" if "maximum_absolute" in str(name) else ">="
        lines.append(
            f"| {name} | {float(value):.3f} | {relation} {float(threshold):.3f} | "
            f"{'PASS' if bool(raw.get('passed')) else 'FAIL'} |"
        )
    return "\n".join(lines) if len(lines) > 2 else "No supplemental checks were supplied."


def _capacity_markdown(capacity: pd.DataFrame) -> str:
    if capacity.empty:
        return (
            "No capacity summary was supplied. Configured AUM and ADV limits remain an "
            "untested scenario, so capacity cannot support production promotion."
        )
    lines = [
        "| Candidate | Evaluated AUM | Max trade / ADV | ADV limit | Violations | At configured AUM |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for _, row in capacity.iterrows():
        lines.append(
            f"| {row['strategy']} | ${row['portfolio_aum_usd'] / 1_000_000:.2f}m | "
            f"{row['maximum_adv_participation_observed']:.2%} | "
            f"{row['maximum_adv_participation_limit']:.2%} | "
            f"{int(row['capacity_violations'])} | "
            f"{'PASS' if row['capacity_passed_at_configured_aum'] else 'FAIL'} |"
        )
    return "\n".join(lines)


def create_champion_challenger_report(
    output_dir: str | Path,
    returns: Mapping[str, pd.Series | pd.DataFrame],
    holdings: Mapping[str, pd.DataFrame],
    metrics: pd.DataFrame,
    cost_sensitivity: pd.DataFrame,
    gates: Mapping[str, Any],
    data_quality: Mapping[str, Any],
    config: Mapping[str, Any],
    capacity_summary: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Write six charts and a decision-led champion-challenger report.

    ``returns`` and ``holdings`` must be keyed by the same predeclared candidate
    names.  If no explicit champion is configured, mapping order is used; the
    opened performance ranking is never used for champion selection.
    """

    for name, value in (("gates", gates), ("data_quality", data_quality), ("config", config)):
        if not isinstance(value, Mapping):
            raise TypeError(f"{name} must be a mapping")
    root = Path(output_dir)
    chart_dir = root / "charts"
    root.mkdir(parents=True, exist_ok=True)

    opened_start = _opened_start(config)
    normalized_returns = _normalize_returns(returns, opened_start=opened_start)
    champion, champion_source = _champion_name(normalized_returns, gates, config)
    names = [champion, *[name for name in normalized_returns if name != champion]]
    annualization = _annualization_factor(config)
    summary = _summary_table(
        normalized_returns,
        metrics,
        annualization=annualization,
    )
    holding_summary = _holding_statistics(holdings, opened_start=opened_start)
    missing_holdings = set(names).difference(holding_summary["strategy"])
    if missing_holdings:
        raise ValueError(f"opened holdings are missing for candidates: {sorted(missing_holdings)}")
    normalized_costs = _normalize_cost_sensitivity(cost_sensitivity, champion=champion)
    cost_assumptions = _cost_assumptions(config)
    normalized_capacity, capacity_source = _normalize_capacity_summary(
        root,
        capacity_summary,
        data_quality=data_quality,
        config=config,
        champion=champion,
    )

    research_status, research_checks = _research_checks(gates, summary, champion)
    production_status, production_checks = _production_checks(gates, data_quality)
    limitation_label, opened, current_ticker = _limitation_label(gates, data_quality, config)
    champion_capacity = normalized_capacity.loc[
        normalized_capacity.get("strategy", pd.Series(dtype=str)).eq(champion)
    ]
    capacity_passed = bool(
        len(champion_capacity) == 1
        and champion_capacity.iloc[0]["capacity_passed_at_configured_aum"]
    )
    production_checks["capacity_at_configured_aum"] = capacity_passed
    production_checks["sealed_evaluation"] = not opened
    production_checks["point_in_time_universe"] = not current_ticker
    if opened or current_ticker or not capacity_passed:
        production_status = "FAIL"
    diagnostic_decision = (
        "REJECT CURRENT CHAMPION"
        if research_status == "FAIL"
        else "RESEARCH GATE PASS; REQUIRE A NEW SEALED TEST"
        if research_status == "PASS" and opened
        else "CONTINUE CHAMPION"
        if research_status == "PASS"
        else "DIAGNOSTIC ONLY"
    )
    production_decision = (
        "NOT PRODUCTION-ELIGIBLE — DO NOT DEPLOY"
        if production_status != "PASS"
        else "PRODUCTION GATE PASS"
    )

    chart_paths = _create_charts(
        chart_dir,
        normalized_returns,
        summary,
        holding_summary,
        normalized_costs,
        normalized_capacity,
        cost_assumptions,
        champion=champion,
        annualization=annualization,
        limitation_label=limitation_label,
    )

    all_dates = pd.concat([values["date"] for values in normalized_returns.values()])
    failed_production = [name for name, passed in production_checks.items() if not passed]
    universe_source = _nested(config, ("universe", "source")) or "unspecified"
    trading_cost = cost_assumptions["trading_cost_bps_one_way"]
    borrow_cost = cost_assumptions["annual_short_borrow_bps"]
    portfolio_aum = cost_assumptions["portfolio_aum_usd"]
    adv_limit = cost_assumptions["maximum_adv_participation"]
    initial_entry = cost_assumptions["charge_initial_entry"]
    terminal_liquidation = cost_assumptions["charge_terminal_liquidation"]
    lines = [
        "# Champion-Challenger Diagnostic",
        "",
        "## Decision first",
        "",
        f"**Diagnostic decision: {diagnostic_decision}.**  ",
        f"**Production decision: {production_decision}.**",
        "",
        f"Champion: `{champion}` ({champion_source}). Research gate: **{research_status}**; "
        f"production gate: **{production_status}**.",
        "",
        f"> {limitation_label}. The period and candidates have been inspected; these charts cannot "
        "serve as a newly sealed promotion test.",
        "",
        "## Opened-test comparison",
        "",
        _metric_markdown(summary, names),
        "",
        f"Evaluation range: {all_dates.min().date()} through {all_dates.max().date()}. "
        f"Universe source: `{universe_source}`.",
        "",
        "## Acceptance gates",
        "",
        _gate_markdown(research_checks),
        "",
        "## Nonlinear challenger verdict",
        "",
        f"Incremental ML gate: **{'PASS' if bool(gates.get('ml_incremental_pass')) else 'FAIL'}**. "
        "This compares the full nonlinear model directly with the preregistered momentum champion.",
        "",
        _supplemental_gate_markdown(gates.get("ml_incremental_checks")),
        "",
        "## Neutral sleeve diagnostic",
        "",
        f"Neutral long-short gate: **{'PASS' if bool(gates.get('neutral_research_pass')) else 'FAIL'}**.",
        "",
        _supplemental_gate_markdown(gates.get("neutral_research_checks")),
        "",
        "Production checks: "
        + (
            ", ".join(
                f"`{name}`={'PASS' if passed else 'FAIL'}"
                for name, passed in production_checks.items()
            )
            if production_checks
            else "not supplied"
        )
        + ".",
        "",
        "## Economic implementation",
        "",
        f"The return series charge an assumed flat **{trading_cost if trading_cost is not None else 'n/a'} "
        "bps one-way cost per traded notional**. "
        f"Initial entry: {'charged' if initial_entry is True else 'not charged' if initial_entry is False else 'unspecified'}; "
        f"terminal liquidation: {'charged' if terminal_liquidation is True else 'not charged' if terminal_liquidation is False else 'unspecified'}. "
        f"The supplied sensitivity spans {normalized_costs['cost_bps'].min():.0f}-"
        f"{normalized_costs['cost_bps'].max():.0f} bps.",
        "",
        "This flat-bps input is an assumption, not observed TAQ/NBBO spread, slippage, or market "
        "impact. Long-only candidates have no borrow charge. The neutral long-short comparator uses "
        f"a generic **{borrow_cost if borrow_cost is not None else 'n/a'} bps annual short-borrow** "
        "assumption; it is not historical locate or fee evidence.",
        "The single-name gross-profit-share gate is an uncompounded contribution diagnostic, not "
        "a full path-dependent P&L attribution.",
        "",
        "## AUM and ADV capacity diagnostic",
        "",
        f"Configured scenario: **${float(portfolio_aum) / 1_000_000:.2f} million AUM**"
        if portfolio_aum is not None
        else "Configured AUM: **not supplied**",
        f"Maximum permitted participation: **{float(adv_limit):.2%} of trailing ADV**."
        if adv_limit is not None
        else "Maximum permitted ADV participation: **not supplied**.",
        f"Capacity evidence source: `{capacity_source}`.",
        "",
        _capacity_markdown(normalized_capacity),
        "",
        "## Binding limitations",
        "",
        f"- Opened evaluation: {'yes' if opened else 'no'}; a redesign requires a new sealed period.",
        f"- Current-ticker/survivorship limitation: {'yes' if current_ticker else 'no'}.",
        "- Failed production data checks: "
        + (", ".join(failed_production) if failed_production else "none explicitly reported")
        + ".",
        "- Challenger comparisons are diagnostics; choosing a winner from this period would add an unrecorded trial.",
        "",
        "## Charts",
        "",
        *[f"- [{path.name}](charts/{path.name})" for path in chart_paths],
        "",
    ]
    report_path = root / "champion_challenger_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "report": str(report_path),
        "charts": [str(path) for path in chart_paths],
        "champion": champion,
        "challengers": [name for name in names if name != champion],
        "champion_source": champion_source,
        "research_status": research_status,
        "production_status": production_status,
        "diagnostic_decision": diagnostic_decision,
        "production_decision": production_decision,
        "opened_test": opened,
        "current_ticker_limitation": current_ticker,
        "capacity_evidence": not normalized_capacity.empty,
        "capacity_passed_at_configured_aum": capacity_passed,
        "capacity_source": capacity_source,
        "failed_production_checks": failed_production,
    }


write_champion_challenger_report = create_champion_challenger_report


__all__ = ["create_champion_challenger_report", "write_champion_challenger_report"]
