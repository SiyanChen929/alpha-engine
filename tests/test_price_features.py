from datetime import timedelta

import numpy as np
import pandas as pd

from alpha_factor_engine.features.price import build_monthly_price_panel


def _daily(ticker: str, periods: int = 500) -> pd.DataFrame:
    dates = pd.bdate_range("2022-01-03", periods=periods)
    close = 100 * np.cumprod(np.full(periods, 1.0004))
    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": dates,
            "open": close,
            "close": close,
            "adj_open": close,
            "adj_close": close,
            "volume": 1_000_000,
            "split_ratio": 1.0,
            "available_at": pd.to_datetime(dates, utc=True) + timedelta(hours=22),
        }
    )


def test_monthly_target_enters_after_signal_date():
    prices = pd.concat([_daily("ABC"), _daily("SPY")], ignore_index=True)
    panel = build_monthly_price_panel(prices, benchmark="SPY", minimum_adv_usd=0)
    valid = panel.dropna(subset=["entry_date", "exit_date", "target_return"])
    assert (valid["entry_date"] > valid["date"]).all()
    assert (valid["exit_date"] > valid["entry_date"]).all()


def test_downside_volatility_counts_nonnegative_days_as_zero():
    periods = 180
    dates = pd.bdate_range("2022-01-03", periods=periods)
    daily_returns = np.resize(np.array([0.01, -0.02, 0.015, -0.005]), periods)
    close = 100 * np.cumprod(1 + daily_returns)

    def frame(ticker: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ticker": ticker,
                "date": dates,
                "open": close,
                "close": close,
                "adj_open": close,
                "adj_close": close,
                "volume": 1_000_000,
                "split_ratio": 1.0,
                "available_at": pd.to_datetime(dates, utc=True) + timedelta(hours=22),
            }
        )

    prices = pd.concat([frame("ABC"), frame("SPY")], ignore_index=True)
    panel = build_monthly_price_panel(prices, benchmark="SPY", minimum_adv_usd=0)
    stock_panel = panel.loc[panel["ticker"].eq("ABC")].set_index("date")

    realized = pd.Series(close, index=dates).pct_change(fill_method=None)
    expected = (
        realized.clip(upper=0).pow(2).rolling(63, min_periods=42).mean().pow(0.5)
        * np.sqrt(252)
    )
    comparable_dates = stock_panel.index.intersection(expected.dropna().index)

    assert len(comparable_dates) > 0
    assert np.allclose(
        stock_panel.loc[comparable_dates, "downside_vol_3m"],
        expected.loc[comparable_dates],
    )
