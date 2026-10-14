"""Deterministic execution diagnostics for frozen equal-weight decile sleeves."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def assign_equal_count_deciles(
    month: pd.DataFrame,
    *,
    score_column: str,
    issuer_column: str = "issuer_cik",
    ticker_column: str = "ticker",
    buckets: int = 10,
) -> pd.DataFrame:
    """Reproduce stable producer buckets without using the return label."""

    if buckets < 2:
        raise ValueError("buckets must be at least two")
    required = {score_column, issuer_column, ticker_column}
    missing = sorted(required.difference(month.columns))
    if missing:
        raise ValueError(f"missing decile inputs: {missing}")
    valid = month.loc[pd.to_numeric(month[score_column], errors="coerce").notna()].copy()
    if valid.empty:
        return valid.assign(decile=pd.Series(dtype="int64"))
    valid["_score"] = pd.to_numeric(valid[score_column], errors="raise")
    valid["_issuer"] = pd.to_numeric(valid[issuer_column], errors="coerce").fillna(np.inf)
    valid = valid.sort_values(
        ["_score", "_issuer", ticker_column], kind="mergesort"
    )
    valid["decile"] = (
        np.floor(np.arange(len(valid)) * buckets / len(valid)).astype(int) + 1
    ).clip(1, buckets)
    return valid.drop(columns=["_score", "_issuer"])


def drifted_equal_weights(
    tickers: list[str], returns: Mapping[str, float]
) -> dict[str, float]:
    """Mark equal absolute notionals to market and renormalize sleeve gross to one."""

    if not tickers:
        raise ValueError("cannot drift an empty sleeve")
    starting = 1.0 / len(tickers)
    notionals: dict[str, float] = {}
    for ticker in tickers:
        value = float(returns[ticker])
        if not np.isfinite(value) or value <= -1.0:
            raise ValueError(f"invalid holding return for {ticker}: {value}")
        notionals[ticker] = starting * (1.0 + value)
    total = float(sum(notionals.values()))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("invalid marked sleeve notional")
    return {ticker: value / total for ticker, value in notionals.items()}


def one_way_traded_notional(
    target_tickers: list[str], prior_drifted_weights: Mapping[str, float] | None
) -> float:
    """Dollar turnover: sum absolute trades for a unit-gross sleeve."""

    if not target_tickers:
        raise ValueError("cannot target an empty sleeve")
    target_weight = 1.0 / len(target_tickers)
    target = {ticker: target_weight for ticker in target_tickers}
    prior = dict(prior_drifted_weights or {})
    names = set(target).union(prior)
    return float(sum(abs(target.get(name, 0.0) - prior.get(name, 0.0)) for name in names))


def side_capacity_from_adv(
    sleeve: pd.DataFrame,
    *,
    adv_column: str,
    participation: float,
) -> float:
    """Maximum one-side AUM if every equal-weight order is capped by ADV share."""

    if not 0 < participation <= 1:
        raise ValueError("participation must lie in (0, 1]")
    if sleeve.empty:
        raise ValueError("capacity requires a nonempty sleeve")
    adv = pd.to_numeric(sleeve[adv_column], errors="coerce").to_numpy(float)
    if not np.isfinite(adv).all() or np.any(adv <= 0):
        return float("nan")
    return float(np.min(adv * participation * len(sleeve)))


def build_execution_months(
    panel: pd.DataFrame,
    *,
    score_column: str,
    return_column: str,
    adv_column: str,
    participations: tuple[float, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build frozen long-short returns, turnover, capacity, and holdings."""

    required = {
        "signal_date",
        "ticker",
        "issuer_cik",
        score_column,
        return_column,
        adv_column,
    }
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"panel missing execution inputs: {missing}")
    work = panel.copy()
    work["signal_date"] = pd.to_datetime(work["signal_date"], utc=True, errors="raise")
    if work.duplicated(["signal_date", "ticker"]).any():
        raise ValueError("panel has duplicate ticker-month rows")

    rows: list[dict[str, object]] = []
    holding_rows: list[dict[str, object]] = []
    prior_long: list[str] | None = None
    prior_short: list[str] | None = None
    prior_returns: dict[str, float] | None = None
    for signal_date, month in work.groupby("signal_date", sort=True):
        ranked = assign_equal_count_deciles(month, score_column=score_column)
        if ranked.empty:
            continue
        short = ranked.loc[ranked["decile"].eq(1)].copy()
        long = ranked.loc[ranked["decile"].eq(10)].copy()
        if short.empty or long.empty:
            raise ValueError(f"missing extreme sleeve at {signal_date}")
        for sleeve_name, sleeve, sign in (("short", short, -1), ("long", long, 1)):
            weight = 1.0 / len(sleeve)
            for record in sleeve.itertuples(index=False):
                holding_rows.append(
                    {
                        "signal_date": signal_date,
                        "sleeve": sleeve_name,
                        "ticker": getattr(record, "ticker"),
                        "issuer_cik": getattr(record, "issuer_cik"),
                        "decile": getattr(record, "decile"),
                        "target_absolute_weight": weight,
                        "signed_weight": sign * weight,
                        "target_return": getattr(record, return_column),
                        "adv_usd_3m": getattr(record, adv_column),
                    }
                )

        long_names = long["ticker"].astype(str).tolist()
        short_names = short["ticker"].astype(str).tolist()
        if prior_long is None:
            long_turnover = 1.0
            short_turnover = 1.0
        else:
            if prior_returns is None or prior_short is None:
                raise RuntimeError("inconsistent prior sleeve state")
            long_turnover = one_way_traded_notional(
                long_names, drifted_equal_weights(prior_long, prior_returns)
            )
            short_turnover = one_way_traded_notional(
                short_names, drifted_equal_weights(prior_short, prior_returns)
            )
        long_return = float(pd.to_numeric(long[return_column], errors="raise").mean())
        short_return = float(pd.to_numeric(short[return_column], errors="raise").mean())
        row: dict[str, object] = {
            "signal_date": signal_date,
            "long_names": int(len(long)),
            "short_names": int(len(short)),
            "long_return": long_return,
            "short_return": short_return,
            "gross_long_short_return": long_return - short_return,
            "long_one_way_traded_notional": long_turnover,
            "short_one_way_traded_notional": short_turnover,
            "total_one_way_traded_notional": long_turnover + short_turnover,
            "adv_complete": bool(
                pd.to_numeric(pd.concat([long[adv_column], short[adv_column]]), errors="coerce")
                .gt(0)
                .all()
            ),
        }
        for participation in participations:
            label = f"capacity_side_usd_at_{participation:.4f}".replace(".", "p")
            capacities = [
                side_capacity_from_adv(
                    sleeve, adv_column=adv_column, participation=participation
                )
                for sleeve in (long, short)
            ]
            row[label] = float(np.min(capacities)) if np.isfinite(capacities).all() else np.nan
        rows.append(row)
        prior_long = long_names
        prior_short = short_names
        prior_returns = {
            str(record.ticker): float(getattr(record, return_column))
            for record in ranked.itertuples(index=False)
        }
    return pd.DataFrame(rows), pd.DataFrame(holding_rows)


__all__ = [
    "assign_equal_count_deciles",
    "build_execution_months",
    "drifted_equal_weights",
    "one_way_traded_notional",
    "side_capacity_from_adv",
]
