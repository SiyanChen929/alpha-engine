from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_xnys_snapshot_contains_known_one_off_and_rule_holidays():
    calendar = pd.read_csv(
        ROOT
        / "artifacts/reference_calendars/xnys_v1/xnys_regular_sessions_2014_2028.csv"
    )
    sessions = set(calendar["session_date"])
    assert "2018-12-05" not in sessions  # George H. W. Bush national day of mourning
    assert "2022-06-20" not in sessions  # first observed Juneteenth closure
    assert "2024-06-19" not in sessions
    assert "2025-01-09" not in sessions  # Jimmy Carter national day of mourning
    assert "2018-12-06" in sessions
    assert "2025-01-10" in sessions
