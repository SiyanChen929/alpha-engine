"""Daily-price signals and next-session executable monthly labels."""

from __future__ import annotations

import numpy as np
import pandas as pd


PRICE_FEATURES = [
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
]


def _rolling_beta(stock: pd.Series, market: pd.Series, window: int = 252) -> tuple[pd.Series, pd.Series]:
    covariance = stock.rolling(window, min_periods=126).cov(market)
    market_variance = market.rolling(window, min_periods=126).var()
    beta = covariance / market_variance.replace(0, np.nan)
    stock_variance = stock.rolling(window, min_periods=126).var()
    idio_variance = (stock_variance - beta.pow(2) * market_variance).clip(lower=0)
    idio_vol = np.sqrt(idio_variance * 252)
    return beta, idio_vol


def _future_split_factor(split_ratio: pd.Series) -> pd.Series:
    """Product of splits strictly after each row, used to undo vendor back-adjustment."""

    future = split_ratio.shift(-1).fillna(1.0)
    return future.iloc[::-1].cumprod().iloc[::-1]


def build_monthly_price_panel(
    daily_prices: pd.DataFrame,
    *,
    benchmark: str = "SPY",
    minimum_price: float = 5.0,
    minimum_adv_usd: float = 20_000_000,
) -> pd.DataFrame:
    required = {"ticker", "date", "open", "close", "adj_open", "adj_close", "volume", "split_ratio"}
    missing = required.difference(daily_prices.columns)
    if missing:
        raise ValueError(f"daily prices missing columns: {sorted(missing)}")
    prices = daily_prices.copy().sort_values(["ticker", "date"])
    prices["date"] = pd.to_datetime(prices["date"])
    prices["return_1d"] = prices.groupby("ticker")["adj_close"].pct_change(fill_method=None)
    market = (
        prices.loc[prices["ticker"].eq(benchmark), ["date", "return_1d"]]
        .rename(columns={"return_1d": "market_return_1d"})
        .drop_duplicates("date")
    )
    prices = prices.merge(market, on="date", how="left")
    output: list[pd.DataFrame] = []
    for ticker, group in prices.groupby("ticker", sort=True):
        group = group.sort_values("date").copy()
        adjusted = group["adj_close"]
        returns = group["return_1d"]
        group["momentum_12_1"] = adjusted.shift(21) / adjusted.shift(252) - 1
        group["momentum_6_1"] = adjusted.shift(21) / adjusted.shift(126) - 1
        group["reversal_1m"] = adjusted / adjusted.shift(21) - 1
        group["realized_vol_3m"] = returns.rolling(63, min_periods=42).std() * np.sqrt(252)
        group["downside_vol_3m"] = (
            returns.clip(upper=0).pow(2).rolling(63, min_periods=42).mean().pow(0.5)
            * np.sqrt(252)
        )
        beta, idio_vol = _rolling_beta(returns, group["market_return_1d"])
        group["beta_12m"] = beta
        group["idio_vol_12m"] = idio_vol
        dollar_volume = group["close"] * group["volume"]
        group["illiquidity_3m"] = (
            (returns.abs() / dollar_volume.replace(0, np.nan)).rolling(63, min_periods=42).mean() * 1e8
        )
        group["log_dollar_volume_3m"] = np.log(
            dollar_volume.rolling(63, min_periods=42).median().clip(lower=1)
        )
        rolling_high = adjusted.rolling(252, min_periods=126).max()
        group["price_to_52w_high"] = adjusted / rolling_high
        group["max_drawdown_12m"] = adjusted / rolling_high - 1
        group["skew_3m"] = returns.rolling(63, min_periods=42).skew()
        group["adv_usd_3m"] = dollar_volume.rolling(63, min_periods=42).mean()
        group["future_split_factor"] = _future_split_factor(group["split_ratio"].fillna(1.0))
        group["pit_close"] = group["close"] * group["future_split_factor"]
        group["next_entry_price"] = group["adj_open"].shift(-1)
        group["next_entry_date"] = group["date"].shift(-1)
        formation = group.groupby(group["date"].dt.to_period("M"), sort=True).tail(1).copy()
        formation["entry_price"] = formation["next_entry_price"]
        formation["entry_date"] = formation["next_entry_date"]
        formation["exit_price"] = formation["entry_price"].shift(-1)
        formation["exit_date"] = formation["entry_date"].shift(-1)
        # Yahoo's daily open is a session-level proxy.  Preserve the exact
        # assumed execution clock so label availability is compared with
        # 09:30 New York, not an accidentally UTC-localized midnight.
        formation["entry_at"] = (
            pd.to_datetime(formation["entry_date"])
            .dt.tz_localize("America/New_York", nonexistent="shift_forward", ambiguous="raise")
            .add(pd.Timedelta(hours=9, minutes=30))
            .dt.tz_convert("UTC")
        )
        formation["exit_at"] = (
            pd.to_datetime(formation["exit_date"])
            .dt.tz_localize("America/New_York", nonexistent="shift_forward", ambiguous="raise")
            .add(pd.Timedelta(hours=9, minutes=30))
            .dt.tz_convert("UTC")
        )
        formation["target_return"] = formation["exit_price"] / formation["entry_price"] - 1
        formation["signal_date"] = (
            pd.to_datetime(formation["date"].dt.date).dt.tz_localize("UTC")
            + pd.Timedelta(hours=23, minutes=59)
        )
        formation["price_available_at"] = pd.to_datetime(
            formation["available_at"], utc=True, errors="coerce"
        )
        formation["eligible"] = (
            formation["pit_close"].ge(minimum_price)
            & formation["adv_usd_3m"].ge(minimum_adv_usd)
            & formation[PRICE_FEATURES].notna().sum(axis=1).ge(8)
        )
        output.append(formation)
    panel = pd.concat(output, ignore_index=True)
    panel = panel.loc[~panel["ticker"].eq(benchmark)].copy()
    panel["entry_date"] = pd.to_datetime(panel["entry_date"])
    panel["exit_date"] = pd.to_datetime(panel["exit_date"])
    panel["entry_at"] = pd.to_datetime(panel["entry_at"], utc=True)
    panel["exit_at"] = pd.to_datetime(panel["exit_at"], utc=True)
    return panel.sort_values(["signal_date", "ticker"]).reset_index(drop=True)
