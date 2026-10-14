"""Independent trade-leg capacity reconstruction for the quarterly-v2 audit."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def _utc_dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="raise")


def _drift_prior_sleeve(prior: pd.DataFrame) -> dict[str, float]:
    weights = pd.to_numeric(prior["target_absolute_weight"], errors="raise").to_numpy(float)
    returns = pd.to_numeric(prior["target_return"], errors="raise").to_numpy(float)
    if not np.isfinite(weights).all() or not np.isfinite(returns).all():
        raise ValueError("prior sleeve contains non-finite weights or returns")
    if np.any(weights < 0) or np.any(returns <= -1):
        raise ValueError("prior sleeve contains invalid weights or returns")
    marked = weights * (1.0 + returns)
    total = float(marked.sum())
    if total <= 0 or not np.isfinite(total):
        raise ValueError("prior sleeve has invalid marked notional")
    return dict(zip(prior["ticker"].astype(str), marked / total, strict=True))


def reconstruct_trade_leg_capacity(
    panel: pd.DataFrame,
    holdings: pd.DataFrame,
    monthly_execution: pd.DataFrame,
    *,
    participations: Iterable[float] = (0.01, 0.05),
    tolerance: float = 1e-14,
) -> pd.DataFrame:
    """Reconstruct capacity from every actual rebalance trade leg.

    For a one-side AUM of ``A``, a trade with absolute sleeve-weight change
    ``|dw|`` consumes ``A * |dw|`` dollars.  Its ADV bound is therefore
    ``A <= participation * ADV / |dw|``.  The monthly bound is the minimum
    across long and short entries, resizes, and exits.  A missing ADV on any
    nonzero trade leg makes that month's trade-based capacity unavailable.
    """

    participations = tuple(float(value) for value in participations)
    if not participations or any(value <= 0 or value > 1 for value in participations):
        raise ValueError("participations must be in (0, 1]")

    panel_required = {"signal_date", "ticker", "adv_usd_3m"}
    holdings_required = {
        "signal_date",
        "sleeve",
        "ticker",
        "target_absolute_weight",
        "target_return",
    }
    monthly_required = {"signal_date", "total_one_way_traded_notional", "adv_complete"}
    for label, frame, required in (
        ("panel", panel, panel_required),
        ("holdings", holdings, holdings_required),
        ("monthly_execution", monthly_execution, monthly_required),
    ):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{label} missing columns: {missing}")

    panel = panel.copy()
    holdings = holdings.copy()
    monthly_execution = monthly_execution.copy()
    for frame in (panel, holdings, monthly_execution):
        frame["signal_date"] = _utc_dates(frame["signal_date"])
        frame["ticker"] = frame["ticker"].astype(str) if "ticker" in frame else None

    if panel.duplicated(["signal_date", "ticker"]).any():
        raise ValueError("panel has duplicate ticker-month rows")
    if holdings.duplicated(["signal_date", "sleeve", "ticker"]).any():
        raise ValueError("holdings have duplicate sleeve ticker-month rows")
    if monthly_execution.duplicated(["signal_date"]).any():
        raise ValueError("monthly execution has duplicate months")

    adv_lookup = panel.set_index(["signal_date", "ticker"])["adv_usd_3m"]
    monthly_lookup = monthly_execution.set_index("signal_date")
    prior_by_sleeve: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []

    for signal_date, month in holdings.groupby("signal_date", sort=True):
        if signal_date not in monthly_lookup.index:
            raise ValueError(f"holdings month missing from monthly execution: {signal_date}")
        month_sleeves = set(month["sleeve"].astype(str))
        if month_sleeves != {"long", "short"}:
            raise ValueError(f"unexpected sleeves at {signal_date}: {sorted(month_sleeves)}")

        trade_legs: list[dict[str, object]] = []
        next_prior: dict[str, pd.DataFrame] = {}
        for sleeve_name, sleeve in month.groupby("sleeve", sort=True):
            sleeve_name = str(sleeve_name)
            weights = pd.to_numeric(sleeve["target_absolute_weight"], errors="raise")
            if not np.isfinite(weights).all() or not np.isclose(weights.sum(), 1.0, atol=1e-10):
                raise ValueError(f"target weights do not sum to one: {signal_date} {sleeve_name}")
            target = dict(zip(sleeve["ticker"].astype(str), weights.astype(float), strict=True))
            prior = (
                _drift_prior_sleeve(prior_by_sleeve[sleeve_name])
                if sleeve_name in prior_by_sleeve
                else {}
            )
            for ticker in sorted(set(target).union(prior)):
                delta = float(target.get(ticker, 0.0) - prior.get(ticker, 0.0))
                if abs(delta) <= tolerance:
                    continue
                key = (signal_date, ticker)
                adv_value = float(adv_lookup.loc[key]) if key in adv_lookup.index else np.nan
                valid_adv = bool(np.isfinite(adv_value) and adv_value > 0)
                trade_legs.append(
                    {
                        "sleeve": sleeve_name,
                        "ticker": ticker,
                        "absolute_delta_weight": abs(delta),
                        "adv_usd_3m": adv_value,
                        "adv_available": valid_adv,
                    }
                )
            next_prior[sleeve_name] = sleeve.copy()

        trade_leg_count = len(trade_legs)
        complete_count = sum(bool(leg["adv_available"]) for leg in trade_legs)
        missing = [str(leg["ticker"]) for leg in trade_legs if not leg["adv_available"]]
        reconstructed_turnover = float(
            sum(float(leg["absolute_delta_weight"]) for leg in trade_legs)
        )
        producer_turnover = float(monthly_lookup.loc[signal_date, "total_one_way_traded_notional"])
        row: dict[str, object] = {
            "signal_date": signal_date,
            "trade_leg_count": trade_leg_count,
            "complete_adv_trade_legs": complete_count,
            "missing_adv_trade_legs": len(missing),
            "actual_trade_leg_adv_complete": len(missing) == 0,
            "missing_adv_tickers": "|".join(sorted(missing)),
            "reconstructed_total_one_way_traded_notional": reconstructed_turnover,
            "producer_total_one_way_traded_notional": producer_turnover,
            "turnover_absolute_error": abs(reconstructed_turnover - producer_turnover),
            "producer_target_holdings_adv_complete": bool(
                monthly_lookup.loc[signal_date, "adv_complete"]
            ),
        }
        for participation in participations:
            suffix = f"{participation:.4f}".replace(".", "p")
            capacity_name = f"trade_leg_capacity_side_aum_usd_at_{suffix}"
            producer_name = f"capacity_side_usd_at_{suffix}"
            if missing:
                row[capacity_name] = np.nan
            else:
                row[capacity_name] = float(
                    min(
                        participation
                        * float(leg["adv_usd_3m"])
                        / float(leg["absolute_delta_weight"])
                        for leg in trade_legs
                    )
                )
            row[f"producer_target_holdings_capacity_side_aum_usd_at_{suffix}"] = (
                float(monthly_lookup.loc[signal_date, producer_name])
                if producer_name in monthly_lookup.columns
                else np.nan
            )
        rows.append(row)
        prior_by_sleeve = next_prior

    result = pd.DataFrame(rows)
    if len(result) != len(monthly_execution):
        raise ValueError(
            f"month count mismatch: reconstructed={len(result)} producer={len(monthly_execution)}"
        )
    return result


__all__ = ["reconstruct_trade_leg_capacity"]
