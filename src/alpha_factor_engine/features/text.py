"""Contract for LLM/embedding features derived from filings, calls, or news."""

from __future__ import annotations

import pandas as pd

from alpha_factor_engine.contracts import PointInTimeViolation, validate_unique


REQUIRED_TEXT_COLUMNS = {
    "document_id",
    "ticker",
    "published_at",
    "available_at",
    "source_sha256",
    "model_id",
    "prompt_version",
}


def validate_text_feature_store(frame: pd.DataFrame) -> None:
    missing = REQUIRED_TEXT_COLUMNS.difference(frame.columns)
    if missing:
        raise PointInTimeViolation(f"text feature store missing columns: {sorted(missing)}")
    published = pd.to_datetime(frame["published_at"], utc=True)
    available = pd.to_datetime(frame["available_at"], utc=True)
    if (available < published).any():
        raise PointInTimeViolation("text feature available_at precedes document publication")
    feature_columns = [
        column
        for column in frame.columns
        if column not in REQUIRED_TEXT_COLUMNS and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not feature_columns:
        raise ValueError("text feature store has no numeric features")
    validate_unique(frame, ["document_id", "model_id", "prompt_version"], "text feature store")


def join_latest_text_features(decisions: pd.DataFrame, store: pd.DataFrame) -> pd.DataFrame:
    validate_text_feature_store(store)
    left = decisions.copy()
    right = store.copy()
    left["signal_date"] = pd.to_datetime(left["signal_date"], utc=True)
    right["available_at"] = pd.to_datetime(right["available_at"], utc=True)
    return pd.merge_asof(
        left.sort_values(["ticker", "signal_date"]),
        right.sort_values(["ticker", "available_at"]),
        by="ticker",
        left_on="signal_date",
        right_on="available_at",
        direction="backward",
    )

