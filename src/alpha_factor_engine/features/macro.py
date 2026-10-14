"""Release-aware macro feature joins."""

from __future__ import annotations

import pandas as pd


def build_macro_features(
    decisions: pd.DataFrame,
    series: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    unique_dates = decisions[["signal_date"]].drop_duplicates().sort_values("signal_date").copy()
    unique_dates["signal_date"] = pd.to_datetime(unique_dates["signal_date"], utc=True)
    result = unique_dates
    availability_columns: list[str] = []
    for name, observations in series.items():
        values = observations[["available_at", "value"]].copy()
        values["available_at"] = pd.to_datetime(values["available_at"], utc=True)
        values = values.sort_values("available_at").drop_duplicates("available_at", keep="last")
        values = values.rename(columns={"value": name, "available_at": f"{name}_available_at"})
        result = pd.merge_asof(
            result.sort_values("signal_date"),
            values.sort_values(f"{name}_available_at"),
            left_on="signal_date",
            right_on=f"{name}_available_at",
            direction="backward",
        )
        availability_columns.append(f"{name}_available_at")
        result[f"{name}_change_3m"] = result[name] - result[name].shift(3)
        known = result[f"{name}_available_at"].notna()
        if (result.loc[known, f"{name}_available_at"] > result.loc[known, "signal_date"]).any():
            raise ValueError(f"future FRED observation leaked for {name}")
    result["macro_available_at"] = result[availability_columns].max(axis=1)
    return result

