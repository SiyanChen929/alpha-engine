from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_factor_engine.research.portfolio_execution_diagnostic import (
    assign_equal_count_deciles,
    build_execution_months,
    drifted_equal_weights,
    one_way_traded_notional,
    side_capacity_from_adv,
)


def test_stable_deciles_do_not_use_label() -> None:
    frame = pd.DataFrame(
        {
            "ticker": [f"T{i:02d}" for i in range(20)][::-1],
            "issuer_cik": list(range(20))[::-1],
            "score": [1.0] * 20,
            "future": np.arange(20) * 1000.0,
        }
    )
    first = assign_equal_count_deciles(frame, score_column="score")
    second = assign_equal_count_deciles(
        frame.assign(future=-frame["future"]), score_column="score"
    )
    assert first[["ticker", "decile"]].equals(second[["ticker", "decile"]])
    assert first.iloc[0]["ticker"] == "T00"
    assert first.iloc[-1]["ticker"] == "T19"


def test_drift_and_turnover() -> None:
    drifted = drifted_equal_weights(["A", "B"], {"A": 0.10, "B": -0.10})
    assert np.isclose(drifted["A"], 0.55)
    assert np.isclose(drifted["B"], 0.45)
    assert np.isclose(one_way_traded_notional(["A", "B"], drifted), 0.10)
    assert one_way_traded_notional(["C", "D"], drifted) == 2.0
    assert one_way_traded_notional(["A", "B"], None) == 1.0


def test_capacity_is_fail_closed() -> None:
    sleeve = pd.DataFrame({"adv": [1_000_000.0, 2_000_000.0]})
    assert side_capacity_from_adv(sleeve, adv_column="adv", participation=0.01) == 20_000.0
    assert np.isnan(
        side_capacity_from_adv(
            pd.DataFrame({"adv": [1_000_000.0, np.nan]}),
            adv_column="adv",
            participation=0.01,
        )
    )


def test_month_builder_includes_initial_formation_and_full_replacement() -> None:
    rows = []
    for month, offset in [("2024-01-31", 0), ("2024-02-29", 20)]:
        for index in range(20):
            rows.append(
                {
                    "signal_date": month,
                    "ticker": f"T{index + offset:02d}",
                    "issuer_cik": index + offset,
                    "score": float(index),
                    "target_return": 0.01 * (index % 2),
                    "adv": 1_000_000.0 + index,
                }
            )
    monthly, holdings = build_execution_months(
        pd.DataFrame(rows),
        score_column="score",
        return_column="target_return",
        adv_column="adv",
        participations=(0.01,),
    )
    assert monthly["total_one_way_traded_notional"].tolist() == [2.0, 4.0]
    assert len(holdings) == 8
    assert monthly["adv_complete"].all()
