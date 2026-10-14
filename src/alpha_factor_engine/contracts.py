"""Shared point-in-time data contracts and lineage helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


class PointInTimeViolation(ValueError):
    """Raised when a dataset exposes information before it was knowable."""


@dataclass(frozen=True)
class DatasetRecord:
    provider: str
    dataset: str
    retrieved_at: str
    source_url: str
    sha256: str
    rows: int
    min_effective_at: str | None = None
    max_effective_at: str | None = None
    min_available_at: str | None = None
    max_available_at: str | None = None
    finality: str = "confirmed"
    revision_policy: str = "append_only"
    notes: dict[str, Any] = field(default_factory=dict)


class RunManifest:
    """Append-only manifest for every external dataset used by a run."""

    def __init__(self, run_id: str, config_sha256: str) -> None:
        self.run_id = run_id
        self.config_sha256 = config_sha256
        self.created_at = datetime.now(UTC).isoformat()
        self.records: list[DatasetRecord] = []

    def add(self, record: DatasetRecord) -> None:
        self.records.append(record)

    def write(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        payload = {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "config_sha256": self.config_sha256,
            "datasets": [asdict(item) for item in self.records],
            **(extra or {}),
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: str | Path) -> tuple[str, int]:
    """Hash file names and contents for a deterministic source-tree fingerprint."""

    root = Path(path)
    files = sorted(
        item
        for item in root.rglob("*")
        if item.is_file() and "__pycache__" not in item.parts and item.suffix != ".pyc"
    )
    digest = hashlib.sha256()
    for item in files:
        relative = item.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        content = item.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest(), len(files)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def ensure_pit_columns(
    frame: pd.DataFrame,
    *,
    name: str = "dataset",
    allow_available_before_effective: bool = False,
) -> None:
    required = {"effective_at", "available_at"}
    missing = required.difference(frame.columns)
    if missing:
        raise PointInTimeViolation(f"{name} missing PIT columns: {sorted(missing)}")
    effective = pd.to_datetime(frame["effective_at"], utc=True, errors="raise")
    available = pd.to_datetime(frame["available_at"], utc=True, errors="raise")
    invalid = available < effective
    if invalid.any() and not allow_available_before_effective:
        examples = frame.loc[invalid].head(3).to_dict("records")
        raise PointInTimeViolation(f"{name} has available_at < effective_at: {examples}")


def assert_no_future_information(
    frame: pd.DataFrame,
    *,
    decision_col: str = "signal_date",
    available_col: str = "available_at",
    name: str = "feature panel",
) -> None:
    decision = pd.to_datetime(frame[decision_col], utc=True, errors="raise")
    available = pd.to_datetime(frame[available_col], utc=True, errors="raise")
    invalid = available > decision
    if invalid.any():
        examples = frame.loc[invalid, [decision_col, available_col]].head(5).to_dict("records")
        raise PointInTimeViolation(f"{name} contains future observations: {examples}")


def point_in_time_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    by: str | list[str],
    decision_col: str = "signal_date",
    available_col: str = "available_at",
    suffix: str = "_pit",
) -> pd.DataFrame:
    """Join each decision to the latest value that was available at that time."""

    keys = [by] if isinstance(by, str) else list(by)
    left_sorted = left.copy()
    right_sorted = right.copy()
    left_sorted[decision_col] = pd.to_datetime(left_sorted[decision_col], utc=True)
    right_sorted[available_col] = pd.to_datetime(right_sorted[available_col], utc=True)
    left_sorted = left_sorted.sort_values([*keys, decision_col])
    right_sorted = right_sorted.sort_values([*keys, available_col])
    joined = pd.merge_asof(
        left_sorted,
        right_sorted,
        left_on=decision_col,
        right_on=available_col,
        by=keys,
        direction="backward",
        allow_exact_matches=True,
        suffixes=("", suffix),
    )
    observed = joined[available_col].notna()
    assert_no_future_information(
        joined.loc[observed],
        decision_col=decision_col,
        available_col=available_col,
        name="point-in-time join",
    )
    return joined


def frame_bounds(frame: pd.DataFrame, column: str) -> tuple[str | None, str | None]:
    if column not in frame or frame.empty:
        return None, None
    values = pd.to_datetime(frame[column], utc=True, errors="coerce").dropna()
    if values.empty:
        return None, None
    return values.min().isoformat(), values.max().isoformat()


def stable_json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return sha256_bytes(encoded)


def validate_unique(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    columns = list(columns)
    duplicated = frame.duplicated(columns, keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, columns].head(5).to_dict("records")
        raise ValueError(f"{name} duplicate keys {columns}: {examples}")
