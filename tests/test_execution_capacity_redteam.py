from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_factor_engine.research.execution_capacity_redteam import (
    reconstruct_trade_leg_capacity,
)


def _fixtures(*, include_outgoing_adv: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(["2024-01-01", "2024-02-01"], utc=True)
    holdings_rows = []
    first = {"long": ["A", "B"], "short": ["C", "D"]}
    second = {"long": ["A", "E"], "short": ["C", "F"]}
    for date, sleeves in zip(dates, [first, second], strict=True):
        for sleeve, tickers in sleeves.items():
            for ticker in tickers:
                holdings_rows.append(
                    {
                        "signal_date": date,
                        "sleeve": sleeve,
                        "ticker": ticker,
                        "target_absolute_weight": 0.5,
                        "target_return": 0.0,
                    }
                )
    holdings = pd.DataFrame(holdings_rows)
    panel_rows = []
    for ticker in ["A", "B", "C", "D"]:
        panel_rows.append({"signal_date": dates[0], "ticker": ticker, "adv_usd_3m": 100.0})
    second_names = ["A", "C", "E", "F"] + (["B", "D"] if include_outgoing_adv else [])
    for ticker in second_names:
        panel_rows.append({"signal_date": dates[1], "ticker": ticker, "adv_usd_3m": 100.0})
    panel = pd.DataFrame(panel_rows)
    monthly = pd.DataFrame(
        {
            "signal_date": dates,
            "total_one_way_traded_notional": [2.0, 2.0],
            "adv_complete": [True, True],
            "capacity_side_usd_at_0p0100": [2.0, 2.0],
            "capacity_side_usd_at_0p0500": [10.0, 10.0],
        }
    )
    return panel, holdings, monthly


def test_missing_outgoing_adv_fails_month_closed() -> None:
    panel, holdings, monthly = _fixtures(include_outgoing_adv=False)
    result = reconstruct_trade_leg_capacity(panel, holdings, monthly)
    second = result.iloc[1]
    assert second["missing_adv_trade_legs"] == 2
    assert not second["actual_trade_leg_adv_complete"]
    assert np.isnan(second["trade_leg_capacity_side_aum_usd_at_0p0100"])
    assert second["reconstructed_total_one_way_traded_notional"] == pytest.approx(2.0)


def test_actual_delta_weight_capacity_includes_exits() -> None:
    panel, holdings, monthly = _fixtures(include_outgoing_adv=True)
    result = reconstruct_trade_leg_capacity(panel, holdings, monthly)
    second = result.iloc[1]
    assert second["missing_adv_trade_legs"] == 0
    assert second["actual_trade_leg_adv_complete"]
    # Every nonzero trade is half a unit: 1% * $100 / 0.5 = $2 side AUM.
    assert second["trade_leg_capacity_side_aum_usd_at_0p0100"] == pytest.approx(2.0)
    assert second["turnover_absolute_error"] == pytest.approx(0.0)


def test_duplicate_panel_keys_are_rejected() -> None:
    panel, holdings, monthly = _fixtures(include_outgoing_adv=True)
    panel = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate ticker-month"):
        reconstruct_trade_leg_capacity(panel, holdings, monthly)
