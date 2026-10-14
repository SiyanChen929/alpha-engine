"""Annual SEC XBRL factors aligned to filing availability."""

from __future__ import annotations

import numpy as np
import pandas as pd


FUNDAMENTAL_FEATURES = [
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
]


def _safe_divide(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return np.nan
    return numerator / denominator


def _value(row: pd.Series, name: str) -> float:
    value = row.get(name, np.nan)
    return float(value) if pd.notna(value) else np.nan


def _split_product(
    splits: pd.DataFrame,
    start_exclusive: pd.Timestamp,
    end_inclusive: pd.Timestamp,
) -> float:
    mask = splits["date"].gt(start_exclusive) & splits["date"].le(end_inclusive)
    if not mask.any():
        return 1.0
    return float(splits.loc[mask, "split_ratio"].fillna(1.0).prod())


def build_fundamental_features(
    price_panel: pd.DataFrame,
    annual_concepts: pd.DataFrame,
    daily_prices: pd.DataFrame,
) -> pd.DataFrame:
    """Create the requested ten fundamental factors without report-date leakage."""

    base_columns = ["ticker", "signal_date", "date", "pit_close"]
    output: list[dict[str, object]] = []
    concepts = annual_concepts.copy()
    concepts["available_at"] = pd.to_datetime(concepts["available_at"], utc=True)
    concepts["fiscal_end"] = pd.to_datetime(concepts["fiscal_end"])
    prices = daily_prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    for ticker, decisions in price_panel[base_columns].groupby("ticker", sort=True):
        company = concepts.loc[concepts["ticker"].eq(ticker)].copy()
        split_events = prices.loc[
            prices["ticker"].eq(ticker) & prices["split_ratio"].ne(1.0), ["date", "split_ratio"]
        ].copy()
        for decision in decisions.itertuples(index=False):
            known = company.loc[company["available_at"].le(decision.signal_date)].copy()
            record: dict[str, object] = {
                "ticker": ticker,
                "signal_date": decision.signal_date,
                "fundamental_available_at": pd.NaT,
                "fundamental_fiscal_end": pd.NaT,
                "debt_definition": "reported_long_term_debt",
            }
            if known.empty:
                output.append(record)
                continue
            known = known.sort_values("available_at").drop_duplicates(
                ["concept", "fiscal_end"], keep="last"
            )
            pivot = known.pivot_table(index="fiscal_end", columns="concept", values="value", aggfunc="last")
            pivot = pivot.sort_index()
            eligible_ends = pivot.index[pivot.get("assets", pd.Series(index=pivot.index)).notna()]
            if len(eligible_ends) == 0:
                output.append(record)
                continue
            current_end = pd.Timestamp(eligible_ends[-1])
            current = pivot.loc[current_end]
            previous_candidates = pivot.loc[pivot.index < current_end - pd.Timedelta(days=250)]
            previous = previous_candidates.iloc[-1] if not previous_candidates.empty else pd.Series(dtype=float)
            current_rows = known.loc[known["fiscal_end"].eq(current_end)]
            available_at = current_rows["available_at"].max()
            assets = _value(current, "assets")
            prior_assets = _value(previous, "assets")
            equity = _value(current, "equity")
            prior_equity = _value(previous, "equity")
            revenue = _value(current, "revenue")
            prior_revenue = _value(previous, "revenue")
            net_income = _value(current, "net_income")
            gross_profit = _value(current, "gross_profit")
            cfo = _value(current, "operating_cash_flow")
            capex = _value(current, "capital_expenditure")
            shares = _value(current, "diluted_shares")
            debt = _value(current, "debt")
            if not np.isfinite(debt):
                debt = _value(current, "liabilities")
                record["debt_definition"] = "total_liabilities_fallback"
            signal_day = pd.Timestamp(decision.date)
            share_split_factor = _split_product(split_events, current_end, signal_day)
            shares_at_signal = shares * share_split_factor if np.isfinite(shares) else np.nan
            market_cap = float(decision.pit_close) * shares_at_signal
            average_assets = np.nanmean([assets, prior_assets]) if np.isfinite(prior_assets) else assets
            average_equity = np.nanmean([equity, prior_equity]) if np.isfinite(prior_equity) else equity
            record.update(
                {
                    "fundamental_available_at": available_at,
                    "fundamental_fiscal_end": current_end,
                    "fundamental_age_days": (decision.signal_date - available_at).days,
                    "log_market_cap": np.log(market_cap) if market_cap > 0 else np.nan,
                    "book_to_market": _safe_divide(equity, market_cap),
                    "earnings_yield": _safe_divide(net_income, market_cap),
                    "roe": _safe_divide(net_income, average_equity),
                    "roa": _safe_divide(net_income, average_assets),
                    "gross_profitability": _safe_divide(gross_profit, average_assets),
                    "debt_to_equity": _safe_divide(debt, equity),
                    "asset_growth": _safe_divide(assets, prior_assets) - 1 if prior_assets > 0 else np.nan,
                    "sales_growth": _safe_divide(revenue, prior_revenue) - 1 if prior_revenue > 0 else np.nan,
                    "accruals": _safe_divide(net_income - cfo, average_assets),
                    "free_cash_flow_yield": _safe_divide(cfo - capex, market_cap),
                    "gross_margin": _safe_divide(gross_profit, revenue),
                    "cash_flow_to_assets": _safe_divide(cfo, average_assets),
                }
            )
            output.append(record)
    result = pd.DataFrame(output)
    result["fundamental_available_at"] = pd.to_datetime(
        result["fundamental_available_at"], utc=True, errors="coerce"
    )
    known = result["fundamental_available_at"].notna()
    if (
        result.loc[known, "fundamental_available_at"]
        > pd.to_datetime(result.loc[known, "signal_date"], utc=True)
    ).any():
        raise ValueError("future SEC filing leaked into fundamental features")
    return result.sort_values(["signal_date", "ticker"]).reset_index(drop=True)

