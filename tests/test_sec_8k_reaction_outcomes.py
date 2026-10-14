import pandas as pd

from alpha_factor_engine.research.sec_8k_reaction_outcomes import assign_rolling_reaction_signals


def test_rolling_threshold_excludes_same_day_events():
    frame = pd.DataFrame([
        {"accession": "A", "ticker": "A", "reaction_date": "2022-01-01", "reaction_abnormal_return": -0.10},
        {"accession": "B", "ticker": "B", "reaction_date": "2022-01-02", "reaction_abnormal_return": 0.10},
        {"accession": "C", "ticker": "C", "reaction_date": "2022-01-03", "reaction_abnormal_return": 9.00},
        {"accession": "D", "ticker": "D", "reaction_date": "2022-01-03", "reaction_abnormal_return": -9.00},
    ])
    selected, _ = assign_rolling_reaction_signals(frame, window_days=365, minimum_history_events=2, tail_probability=.25)
    day = selected.loc[selected["reaction_date"].eq(pd.Timestamp("2022-01-03"))]
    assert set(day["direction"]) == {-1, 1}
    assert day["reaction_history_events"].eq(2).all()
    assert day["reaction_history_end_exclusive"].eq(pd.Timestamp("2022-01-03")).all()


def test_middle_reaction_is_not_selected():
    history = [{"accession": f"H{i}", "ticker": f"H{i}", "reaction_date": f"2021-01-{i+1:02d}", "reaction_abnormal_return": value} for i, value in enumerate([-2, -1, 1, 2])]
    history.append({"accession": "M", "ticker": "M", "reaction_date": "2021-02-01", "reaction_abnormal_return": 0.0})
    selected, exclusions = assign_rolling_reaction_signals(pd.DataFrame(history), window_days=365, minimum_history_events=4, tail_probability=.2)
    assert "M" not in set(selected["accession"])
    assert exclusions.loc[exclusions["accession"].eq("M"), "reason"].eq("reaction_not_in_online_tail").all()
