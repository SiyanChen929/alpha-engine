import numpy as np
import pandas as pd

from proxy_research.evaluate import (
    benjamini_hochberg,
    build_equal_weight_portfolio,
    sanitize_panel,
)


def _row(date, ticker, target, eligible=True):
    return {
        "signal_date": pd.Timestamp(date, tz="UTC"),
        "entry_at": pd.Timestamp(date, tz="UTC") + pd.Timedelta(days=1),
        "exit_at": pd.Timestamp(date, tz="UTC") + pd.Timedelta(days=31),
        "ticker": ticker,
        "target_return": target,
        "eligible": eligible,
        "score": {"A": 2, "B": 1, "C": 0}.get(ticker, -1),
    }


def test_sanitize_drops_entire_partial_label_month():
    panel = pd.DataFrame([
        _row("2020-01-01", "A", 0.1), _row("2020-01-01", "B", np.nan),
        _row("2020-02-01", "A", 0.2), _row("2020-02-01", "B", 0.0),
    ])
    clean, audit = sanitize_panel(panel, minimum_cross_section=2, first_prediction="2020-01-01")
    assert clean["signal_date"].nunique() == 1
    assert clean["signal_date"].iloc[0] == pd.Timestamp("2020-02-01", tz="UTC")
    assert "dropped_entire_partial_label_month" in set(audit["status"])


def test_equal_weight_portfolio_charges_entry_and_terminal_exit():
    panel = pd.DataFrame([
        _row("2020-01-01", "A", 0.10), _row("2020-01-01", "B", 0.00),
        _row("2020-02-01", "A", 0.00), _row("2020-02-01", "B", 0.10),
    ])
    panel.loc[panel["signal_date"].eq(pd.Timestamp("2020-02-01", tz="UTC")), "score"] *= -1
    returns, holdings = build_equal_weight_portfolio(
        panel, score_column="score", holdings_count=1, trading_cost_bps_one_way=10,
    )
    assert np.isclose(returns.iloc[0]["gross_return"], 0.10)
    assert np.isclose(returns.iloc[0]["turnover"], 1.0)
    assert np.isclose(returns.iloc[-1]["turnover"], 3.0)
    assert np.isclose(returns["trading_cost"].sum(), 0.004)
    assert len(holdings) == 2


def test_equal_weight_retention_buffer_uses_current_rank_and_prior_membership():
    rows = []
    for date, scores in [
        ("2020-01-01", {"A": 4, "B": 3, "C": 2, "D": 1}),
        ("2020-02-01", {"A": 2, "B": 4, "C": 3, "D": 1}),
    ]:
        for ticker, score in scores.items():
            row = _row(date, ticker, 0.0)
            row["score"] = score
            rows.append(row)
    panel = pd.DataFrame(rows)
    _, holdings = build_equal_weight_portfolio(
        panel,
        score_column="score",
        holdings_count=2,
        retention_rank=3,
        trading_cost_bps_one_way=10,
    )
    second = holdings.loc[
        holdings["signal_date"].eq(pd.Timestamp("2020-02-01", tz="UTC")), "ticker"
    ]
    # A falls to rank three but remains; C cannot displace it through the buffer.
    assert set(second) == {"A", "B"}


def test_benjamini_hochberg_is_monotone_in_pvalue_order():
    p = pd.Series([0.04, 0.001, 0.03, np.nan])
    q = benjamini_hochberg(p)
    order = p.dropna().sort_values().index
    assert q.loc[order].is_monotonic_increasing
    assert np.isnan(q.iloc[-1])
