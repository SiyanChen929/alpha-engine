"""Leakage-safe 12-1 momentum signals and a concentrated long portfolio.

The signal builder deliberately requires exact calendar endpoints.  The caller
must derive ``twelve_month_date`` and ``one_month_date`` from its canonical
trading calendar before this module sees the data; no per-security row shifting
is allowed to silently turn a missing session into a different lookback.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


PRICE_COLUMNS = {"ticker", "date", "total_return_index", "available_at"}
DECISION_COLUMNS = {
    "signal_at",
    "formation_date",
    "twelve_month_date",
    "one_month_date",
}
EXECUTION_COLUMNS = {
    "signal_at",
    "ticker",
    "entry_at",
    "exit_at",
    "entry_price",
    "exit_price",
}
PORTFOLIO_COLUMNS = EXECUTION_COLUMNS | {"momentum_12_1"}


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _aware_utc(values: pd.Series, label: str) -> pd.Series:
    converted: list[pd.Timestamp] = []
    for index, value in values.items():
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} contains an invalid timestamp at row {index}: {value!r}") from error
        if pd.isna(timestamp):
            raise ValueError(f"{label} contains a missing timestamp at row {index}")
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError(f"{label} must be timezone-aware at row {index}: {value!r}")
        converted.append(timestamp.tz_convert("UTC"))
    return pd.Series(converted, index=values.index, dtype="datetime64[ns, UTC]")


def _exact_dates(values: pd.Series, label: str) -> pd.Series:
    converted: list[pd.Timestamp] = []
    for index, value in values.items():
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} contains an invalid date at row {index}: {value!r}") from error
        if pd.isna(timestamp):
            raise ValueError(f"{label} contains a missing date at row {index}")
        if timestamp.tzinfo is not None or timestamp != timestamp.normalize():
            raise ValueError(
                f"{label} must contain timezone-naive exact dates without clock times; "
                f"row {index} is {value!r}"
            )
        converted.append(timestamp.normalize())
    return pd.Series(converted, index=values.index, dtype="datetime64[ns]")


def compute_12_1_momentum(
    prices: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    universe: Sequence[str],
) -> pd.DataFrame:
    """Compute 12-1 momentum from explicit, point-in-time price endpoints.

    ``prices`` must contain a positive total-return index and the exact time at
    which each observation became available.  Each decision supplies the
    formation date plus the trading-calendar dates conventionally denoted
    ``t-k`` and ``t-21`` (the pipeline freezes ``k=253`` to match the attached
    paper's displayed inclusive-endpoint equation).  The returned signal is
    ``TRI(one_month_date) / TRI(twelve_month_date) - 1``.
    """

    _require_columns(prices, PRICE_COLUMNS, "prices")
    _require_columns(decisions, DECISION_COLUMNS, "decisions")
    names = [str(name) for name in universe]
    if not names:
        raise ValueError("universe must contain at least one permanent security identifier")
    if len(names) != len(set(names)):
        raise ValueError("universe contains duplicate security identifiers")

    history = prices.copy()
    history["ticker"] = history["ticker"].astype(str)
    history["date"] = _exact_dates(history["date"], "prices.date")
    history["available_at"] = _aware_utc(history["available_at"], "prices.available_at")
    history["total_return_index"] = pd.to_numeric(history["total_return_index"], errors="coerce")
    duplicates = history.duplicated(["ticker", "date"], keep=False)
    if duplicates.any():
        example = history.loc[duplicates, ["ticker", "date"]].iloc[0]
        raise ValueError(
            "prices must have one row per ticker/date; duplicate found for "
            f"{example['ticker']} on {example['date'].date()}"
        )
    missing_names = sorted(set(names).difference(history["ticker"].unique()))
    if missing_names:
        raise ValueError(f"universe identifiers absent from prices: {missing_names}")

    schedule = decisions.copy()
    schedule["signal_at"] = _aware_utc(schedule["signal_at"], "decisions.signal_at")
    for column in ["formation_date", "twelve_month_date", "one_month_date"]:
        schedule[column] = _exact_dates(schedule[column], f"decisions.{column}")
    if schedule["signal_at"].duplicated().any():
        raise ValueError("decisions must contain exactly one row per signal_at")
    invalid_order = ~(
        schedule["twelve_month_date"].lt(schedule["one_month_date"])
        & schedule["one_month_date"].le(schedule["formation_date"])
    )
    if invalid_order.any():
        row = schedule.loc[invalid_order].iloc[0]
        raise ValueError(
            "12-1 endpoints must satisfy twelve_month_date < one_month_date <= formation_date; "
            f"invalid signal_at={row['signal_at']}"
        )
    local_signal_dates = schedule["signal_at"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    if ~local_signal_dates.eq(schedule["formation_date"]).all():
        raise ValueError("each signal_at must fall on its exact formation_date in America/New_York")

    indexed = history.set_index(["ticker", "date"]).sort_index()
    rows: list[dict[str, object]] = []
    for decision in schedule.sort_values("signal_at").itertuples(index=False):
        endpoint_dates = {
            "formation": decision.formation_date,
            "twelve_month": decision.twelve_month_date,
            "one_month": decision.one_month_date,
        }
        for ticker in sorted(names):
            points: dict[str, pd.Series] = {}
            for endpoint, endpoint_date in endpoint_dates.items():
                key = (ticker, endpoint_date)
                if key not in indexed.index:
                    raise ValueError(
                        f"missing exact {endpoint} price for {ticker} on {endpoint_date.date()}"
                    )
                point = indexed.loc[key]
                if isinstance(point, pd.DataFrame):
                    raise ValueError(f"ambiguous duplicate price for {ticker} on {endpoint_date.date()}")
                value = float(point["total_return_index"])
                if not np.isfinite(value) or value <= 0:
                    raise ValueError(
                        f"non-positive or non-finite total_return_index for {ticker} "
                        f"on {endpoint_date.date()}"
                    )
                if point["available_at"] > decision.signal_at:
                    raise ValueError(
                        f"{endpoint} price for {ticker} on {endpoint_date.date()} was unavailable "
                        f"at signal_at={decision.signal_at}"
                    )
                points[endpoint] = point
            start_value = float(points["twelve_month"]["total_return_index"])
            end_value = float(points["one_month"]["total_return_index"])
            rows.append(
                {
                    "signal_at": decision.signal_at,
                    "formation_date": decision.formation_date,
                    "ticker": ticker,
                    "twelve_month_date": decision.twelve_month_date,
                    "one_month_date": decision.one_month_date,
                    "twelve_month_value": start_value,
                    "one_month_value": end_value,
                    "twelve_month_available_at": points["twelve_month"]["available_at"],
                    "one_month_available_at": points["one_month"]["available_at"],
                    "momentum_12_1": end_value / start_value - 1.0,
                }
            )
    return pd.DataFrame(rows).sort_values(["signal_at", "ticker"]).reset_index(drop=True)


def build_momentum_champion(
    candidates: pd.DataFrame,
    *,
    top_n: int = 6,
    trading_cost_bps_one_way: float = 15.0,
    charge_terminal_liquidation: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Form a top-six equal-weight portfolio and calculate executable returns.

    Costs are charged per dollar of absolute traded weight.  Turnover compares
    target weights with weights drifted by the preceding holding-period returns.
    Exact execution semantics are enforced: signal < entry < exit, and one
    period's exit must be the following period's entry.
    """

    _require_columns(candidates, PORTFOLIO_COLUMNS, "candidates")
    if not isinstance(top_n, int) or top_n <= 0:
        raise ValueError("top_n must be a positive integer")
    if not np.isfinite(trading_cost_bps_one_way) or trading_cost_bps_one_way < 0:
        raise ValueError("trading_cost_bps_one_way must be finite and non-negative")

    panel = candidates.copy()
    panel["ticker"] = panel["ticker"].astype(str)
    if panel["ticker"].str.strip().eq("").any():
        raise ValueError("candidates.ticker contains an empty identifier")
    for column in ["signal_at", "entry_at", "exit_at"]:
        panel[column] = _aware_utc(panel[column], f"candidates.{column}")
    invalid_order = ~(
        panel["signal_at"].lt(panel["entry_at"])
        & panel["entry_at"].lt(panel["exit_at"])
    )
    if invalid_order.any():
        example = panel.loc[invalid_order, ["signal_at", "entry_at", "exit_at"]].iloc[0]
        raise ValueError(
            "execution timestamps must satisfy signal_at < entry_at < exit_at; "
            f"got {example['signal_at']}, {example['entry_at']}, {example['exit_at']}"
        )
    clock_violations: dict[str, pd.Series] = {}
    for column in ["entry_at", "exit_at"]:
        local = panel[column].dt.tz_convert("America/New_York")
        clock_violations[column] = ~(
            local.dt.hour.eq(9)
            & local.dt.minute.eq(30)
            & local.dt.second.eq(0)
        )
        if clock_violations[column].all():
            raise ValueError(f"candidates.{column} must use the 09:30 America/New_York open")
    execution_schedule = (
        panel.groupby("signal_at", sort=True)
        .agg(entry_at=("entry_at", "first"), exit_at=("exit_at", "first"))
        .sort_values("entry_at")
        .reset_index()
    )
    if len(execution_schedule) > 1:
        contiguous = execution_schedule["exit_at"].iloc[:-1].reset_index(drop=True).eq(
            execution_schedule["entry_at"].iloc[1:].reset_index(drop=True)
        )
        if not contiguous.all():
            index = int(np.flatnonzero(~contiguous.to_numpy())[0])
            raise ValueError(
                "holding periods must be contiguous: prior exit_at must equal next entry_at; "
                f"got {execution_schedule.loc[index, 'exit_at']} then "
                f"{execution_schedule.loc[index + 1, 'entry_at']}"
            )
    for column, invalid_clock in clock_violations.items():
        if invalid_clock.any():
            raise ValueError(f"candidates.{column} must use the 09:30 America/New_York open")
    for column in ["momentum_12_1", "entry_price", "exit_price"]:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
        invalid = ~np.isfinite(panel[column])
        if invalid.any():
            raise ValueError(f"candidates.{column} contains missing or non-finite values")
    if panel[["entry_price", "exit_price"]].le(0).any(axis=None):
        raise ValueError("entry_price and exit_price must be strictly positive")
    duplicates = panel.duplicated(["signal_at", "ticker"], keep=False)
    if duplicates.any():
        example = panel.loc[duplicates, ["signal_at", "ticker"]].iloc[0]
        raise ValueError(
            "candidates must have one row per signal/ticker; duplicate found for "
            f"{example['ticker']} at {example['signal_at']}"
        )

    selected_frames: list[pd.DataFrame] = []
    schedule_rows: list[dict[str, pd.Timestamp]] = []
    for signal_at, group in panel.groupby("signal_at", sort=True):
        if len(group) < top_n:
            raise ValueError(
                f"signal_at={signal_at} has {len(group)} candidates; top-{top_n} requires at least {top_n}"
            )
        if group["entry_at"].nunique() != 1 or group["exit_at"].nunique() != 1:
            raise ValueError(f"signal_at={signal_at} has inconsistent entry_at or exit_at timestamps")
        entry_at = group["entry_at"].iloc[0]
        exit_at = group["exit_at"].iloc[0]
        if not signal_at < entry_at < exit_at:
            raise ValueError(
                "execution timestamps must satisfy signal_at < entry_at < exit_at; "
                f"got {signal_at}, {entry_at}, {exit_at}"
            )
        ordered = group.sort_values(
            ["momentum_12_1", "ticker"],
            ascending=[False, True],
            kind="mergesort",
        )
        selected = ordered.head(top_n).copy()
        selected["momentum_rank"] = np.arange(1, top_n + 1)
        selected["weight"] = 1.0 / top_n
        selected["target_return"] = selected["exit_price"] / selected["entry_price"] - 1.0
        selected_frames.append(selected)
        schedule_rows.append({"signal_at": signal_at, "entry_at": entry_at, "exit_at": exit_at})

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

    holdings = pd.concat(selected_frames, ignore_index=True).sort_values(
        ["entry_at", "momentum_rank", "ticker"]
    )
    previous: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    for signal_at, group in holdings.groupby("signal_at", sort=True):
        current = dict(zip(group["ticker"], group["weight"]))
        all_names = set(previous) | set(current)
        turnover = float(sum(abs(current.get(name, 0.0) - previous.get(name, 0.0)) for name in all_names))
        gross_return = float((group["weight"] * group["target_return"]).sum())
        if 1.0 + gross_return <= 0:
            raise ValueError(f"portfolio wealth became non-positive at signal_at={signal_at}")
        trading_cost = turnover * trading_cost_bps_one_way / 10_000
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
                "gross_exposure": 1.0,
                "net_exposure": 1.0,
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


def run_momentum_champion(
    prices: pd.DataFrame,
    decisions: pd.DataFrame,
    executions: pd.DataFrame,
    *,
    universe: Sequence[str],
    top_n: int = 6,
    trading_cost_bps_one_way: float = 15.0,
    charge_terminal_liquidation: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute PIT signals, attach exact executions, and run the portfolio."""

    _require_columns(executions, EXECUTION_COLUMNS, "executions")
    signals = compute_12_1_momentum(prices, decisions, universe=universe)
    fills = executions.copy()
    fills["signal_at"] = _aware_utc(fills["signal_at"], "executions.signal_at")
    fills["ticker"] = fills["ticker"].astype(str)
    if fills.duplicated(["signal_at", "ticker"]).any():
        raise ValueError("executions must have one row per signal_at/ticker")
    candidates = signals.merge(
        fills,
        on=["signal_at", "ticker"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    missing = candidates["_merge"].ne("both")
    if missing.any():
        example = candidates.loc[missing, ["signal_at", "ticker"]].iloc[0]
        raise ValueError(
            f"missing exact execution prices for {example['ticker']} at {example['signal_at']}"
        )
    candidates = candidates.drop(columns="_merge")
    returns, holdings = build_momentum_champion(
        candidates,
        top_n=top_n,
        trading_cost_bps_one_way=trading_cost_bps_one_way,
        charge_terminal_liquidation=charge_terminal_liquidation,
    )
    return returns, holdings, signals


__all__ = [
    "build_momentum_champion",
    "compute_12_1_momentum",
    "run_momentum_champion",
]
