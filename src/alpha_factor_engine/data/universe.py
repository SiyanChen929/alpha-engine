"""Point-in-time security-master and investable-universe contracts.

The selector operates on permanent identifiers and half-open validity intervals.
It retains inactive securities in the input history so a later-inactive company
can still enter earlier snapshots.  Tickers are deliberately not accepted as
the identity spine.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from alpha_factor_engine.contracts import PointInTimeViolation, validate_unique


REQUIRED_SECURITY_MASTER_COLUMNS: tuple[str, ...] = (
    "valid_from",
    "valid_to",
    "available_at",
    "active",
    "shrcd",
    "exchcd",
    "price",
    "adv",
    "market_cap",
    "dlret",
)


class UniverseQualityViolation(PointInTimeViolation):
    """Raised when security-master quality is insufficient for research use."""

    def __init__(self, message: str, report: DelistingCoverageReport | None = None) -> None:
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class UniverseScreens:
    """Predeclared eligibility screens applied at every decision time."""

    common_share_codes: tuple[int, ...] = (10, 11)
    exchange_codes: tuple[int, ...] = (1, 2, 3)
    min_price: float = 5.0
    min_adv: float = 1_000_000.0
    min_market_cap: float = 100_000_000.0

    def __post_init__(self) -> None:
        if not self.common_share_codes:
            raise ValueError("common_share_codes cannot be empty")
        if not self.exchange_codes:
            raise ValueError("exchange_codes cannot be empty")
        for name in ("min_price", "min_adv", "min_market_cap"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class DelistingCoverageReport:
    """Security-master delisting-return completeness at the permanent-ID level."""

    permanent_id_column: str
    total_permanent_ids: int
    active_permanent_ids: int
    inactive_permanent_ids: int
    required_delisting_ids: int
    covered_delisting_ids: int
    coverage: float
    minimum_coverage: float
    missing_permanent_ids: tuple[Any, ...]
    issues: tuple[str, ...]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def require_pass(self) -> DelistingCoverageReport:
        if not self.passed:
            detail = ", ".join(self.issues) or "unknown quality failure"
            raise UniverseQualityViolation(
                f"security-master delisting coverage failed closed: {detail}", self
            )
        return self


def _as_utc_timestamp(value: str | pd.Timestamp, *, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise PointInTimeViolation(f"{name} cannot be missing")
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def validate_security_master(
    frame: pd.DataFrame,
    *,
    permanent_id_column: str = "permno",
) -> pd.DataFrame:
    """Validate and normalize a bitemporal security-master history.

    ``valid_from`` and ``valid_to`` describe the half-open economic interval
    ``[valid_from, valid_to)``.  ``available_at`` is the knowledge timestamp.
    Multiple corrections to an interval are allowed when their availability
    timestamps differ; the latest known correction wins at selection time.
    """

    required = {permanent_id_column, *REQUIRED_SECURITY_MASTER_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise PointInTimeViolation(
            f"security master missing required columns: {sorted(missing)}"
        )
    if permanent_id_column.lower() in {"ticker", "symbol"}:
        raise PointInTimeViolation(
            "security master identity must be a permanent ID such as PERMNO, not a ticker"
        )

    result = frame.copy()
    identifiers = result[permanent_id_column]
    invalid_identifier = identifiers.isna() | identifiers.astype(str).str.strip().eq("")
    if invalid_identifier.any():
        raise PointInTimeViolation("security master contains missing permanent IDs")

    raw_valid_to_present = result["valid_to"].notna()
    for column in ("valid_from", "valid_to", "available_at"):
        result[column] = pd.to_datetime(result[column], utc=True, errors="coerce")
    if result[["valid_from", "available_at"]].isna().any(axis=None):
        raise PointInTimeViolation("valid_from and available_at must be valid timestamps")
    if (raw_valid_to_present & result["valid_to"].isna()).any():
        raise PointInTimeViolation("valid_to must be a valid timestamp or null")
    invalid_interval = result["valid_to"].notna() & (
        result["valid_to"] <= result["valid_from"]
    )
    if invalid_interval.any():
        examples = result.loc[
            invalid_interval, [permanent_id_column, "valid_from", "valid_to"]
        ].head(5)
        raise PointInTimeViolation(
            f"security master has invalid half-open validity intervals: "
            f"{examples.to_dict('records')}"
        )

    if result["active"].isna().any() or not pd.api.types.is_bool_dtype(result["active"]):
        raise PointInTimeViolation("security master active column must be boolean")

    for column in ("shrcd", "exchcd", "price", "adv", "market_cap", "dlret"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    active = result["active"]
    active_missing = result.loc[
        active, ["shrcd", "exchcd", "price", "adv", "market_cap"]
    ].isna()
    if active_missing.any(axis=None):
        bad_columns = active_missing.columns[active_missing.any()].tolist()
        raise PointInTimeViolation(
            f"active security-master records lack screen values: {bad_columns}"
        )
    screen_values = result.loc[active, ["price", "adv", "market_cap"]].to_numpy(float)
    if not np.isfinite(screen_values).all():
        raise PointInTimeViolation("active security-master screen values must be finite")
    dlret_values = result["dlret"].dropna().to_numpy(float)
    if not np.isfinite(dlret_values).all():
        raise PointInTimeViolation("delisting returns must be finite when present")

    validate_unique(
        result,
        [permanent_id_column, "valid_from", "valid_to", "available_at"],
        "security master",
    )
    return result.sort_values(
        [permanent_id_column, "available_at", "valid_from"], kind="mergesort"
    ).reset_index(drop=True)


def delisting_coverage_report(
    frame: pd.DataFrame,
    *,
    permanent_id_column: str = "permno",
    minimum_coverage: float = 1.0,
    require_inactive_history: bool = True,
    fail_closed: bool = True,
) -> DelistingCoverageReport:
    """Report delisting-return coverage and block incomplete histories by default.

    Every permanent ID with an inactive history record requires at least one
    finite ``dlret``.  Requiring inactive history prevents a current-survivor-
    only file from receiving a vacuous 100% coverage score.
    """

    if not 0 <= minimum_coverage <= 1:
        raise ValueError("minimum_coverage must lie in [0, 1]")
    validated = validate_security_master(frame, permanent_id_column=permanent_id_column)

    all_ids = set(validated[permanent_id_column].unique().tolist())
    active_ids = set(
        validated.loc[validated["active"], permanent_id_column].unique().tolist()
    )
    inactive_ids = set(
        validated.loc[~validated["active"], permanent_id_column].unique().tolist()
    )
    covered_ids = set(
        validated.loc[
            validated[permanent_id_column].isin(inactive_ids) & validated["dlret"].notna(),
            permanent_id_column,
        ].unique().tolist()
    )
    missing_ids = tuple(sorted(inactive_ids.difference(covered_ids), key=str))
    coverage = len(covered_ids) / len(inactive_ids) if inactive_ids else 0.0

    issues: list[str] = []
    if require_inactive_history and not inactive_ids:
        issues.append("no inactive permanent IDs; current-survivor-only history cannot be certified")
    if inactive_ids and coverage < minimum_coverage:
        issues.append(
            f"delisting-return coverage {coverage:.3f} is below {minimum_coverage:.3f}"
        )
    passed = not issues
    report = DelistingCoverageReport(
        permanent_id_column=permanent_id_column,
        total_permanent_ids=len(all_ids),
        active_permanent_ids=len(active_ids),
        inactive_permanent_ids=len(inactive_ids),
        required_delisting_ids=len(inactive_ids),
        covered_delisting_ids=len(covered_ids),
        coverage=coverage,
        minimum_coverage=minimum_coverage,
        missing_permanent_ids=missing_ids,
        issues=tuple(issues),
        passed=passed,
    )
    if fail_closed:
        report.require_pass()
    return report


def select_universe_at(
    frame: pd.DataFrame,
    decision_time: str | pd.Timestamp,
    *,
    screens: UniverseScreens | None = None,
    top_n: int | None = None,
    permanent_id_column: str = "permno",
    include_inactive: bool = False,
) -> pd.DataFrame:
    """Select one deterministic point-in-time universe snapshot.

    Inactive histories remain in ``frame``.  They are excluded from an
    investable snapshot unless ``include_inactive`` is explicitly requested
    for audit or terminal-return workflows.
    """

    if top_n is not None and (not isinstance(top_n, int) or isinstance(top_n, bool) or top_n <= 0):
        raise ValueError("top_n must be a positive integer or None")
    rules = screens or UniverseScreens()
    decision = _as_utc_timestamp(decision_time, name="decision_time")
    validated = validate_security_master(frame, permanent_id_column=permanent_id_column)

    known = validated["available_at"] <= decision
    economically_valid = (validated["valid_from"] <= decision) & (
        validated["valid_to"].isna() | (decision < validated["valid_to"])
    )
    candidates = validated.loc[known & economically_valid].copy()
    candidates = candidates.sort_values(
        [permanent_id_column, "available_at", "valid_from"], kind="mergesort"
    ).drop_duplicates(permanent_id_column, keep="last")
    if not include_inactive:
        candidates = candidates.loc[candidates["active"]].copy()

    eligible = (
        candidates["shrcd"].isin(rules.common_share_codes)
        & candidates["exchcd"].isin(rules.exchange_codes)
        & candidates["price"].abs().ge(rules.min_price)
        & candidates["adv"].ge(rules.min_adv)
        & candidates["market_cap"].ge(rules.min_market_cap)
    )
    candidates = candidates.loc[eligible].copy()
    candidates = candidates.sort_values(
        ["market_cap", "adv", permanent_id_column],
        ascending=[False, False, True],
        kind="mergesort",
    )
    if top_n is not None:
        candidates = candidates.head(top_n)
    candidates.insert(0, "decision_time", decision)
    candidates.insert(1, "universe_rank", np.arange(1, len(candidates) + 1, dtype=int))

    if (candidates["available_at"] > candidates["decision_time"]).any():
        raise PointInTimeViolation("universe snapshot contains future security-master records")
    return candidates.reset_index(drop=True)


def select_rolling_universe(
    frame: pd.DataFrame,
    decision_times: Iterable[str | pd.Timestamp],
    *,
    screens: UniverseScreens | None = None,
    top_n: int | None = None,
    permanent_id_column: str = "permno",
    include_inactive: bool = False,
) -> pd.DataFrame:
    """Build deterministic snapshots for sorted, unique decision times."""

    decisions = sorted(
        {
            _as_utc_timestamp(value, name="decision_time")
            for value in decision_times
        }
    )
    if not decisions:
        raise ValueError("decision_times cannot be empty")
    snapshots = [
        select_universe_at(
            frame,
            decision,
            screens=screens,
            top_n=top_n,
            permanent_id_column=permanent_id_column,
            include_inactive=include_inactive,
        )
        for decision in decisions
    ]
    return pd.concat(snapshots, ignore_index=True)


class PointInTimeUniverse:
    """Validated convenience wrapper for repeated universe formation."""

    def __init__(self, frame: pd.DataFrame, *, permanent_id_column: str = "permno") -> None:
        self.permanent_id_column = permanent_id_column
        self.frame = validate_security_master(
            frame, permanent_id_column=permanent_id_column
        )

    def snapshot(
        self,
        decision_time: str | pd.Timestamp,
        *,
        screens: UniverseScreens | None = None,
        top_n: int | None = None,
        include_inactive: bool = False,
    ) -> pd.DataFrame:
        return select_universe_at(
            self.frame,
            decision_time,
            screens=screens,
            top_n=top_n,
            permanent_id_column=self.permanent_id_column,
            include_inactive=include_inactive,
        )

    def rolling(
        self,
        decision_times: Iterable[str | pd.Timestamp],
        *,
        screens: UniverseScreens | None = None,
        top_n: int | None = None,
        include_inactive: bool = False,
    ) -> pd.DataFrame:
        return select_rolling_universe(
            self.frame,
            decision_times,
            screens=screens,
            top_n=top_n,
            permanent_id_column=self.permanent_id_column,
            include_inactive=include_inactive,
        )

    def quality_report(
        self,
        *,
        minimum_coverage: float = 1.0,
        require_inactive_history: bool = True,
        fail_closed: bool = True,
    ) -> DelistingCoverageReport:
        return delisting_coverage_report(
            self.frame,
            permanent_id_column=self.permanent_id_column,
            minimum_coverage=minimum_coverage,
            require_inactive_history=require_inactive_history,
            fail_closed=fail_closed,
        )


__all__ = [
    "DelistingCoverageReport",
    "PointInTimeUniverse",
    "REQUIRED_SECURITY_MASTER_COLUMNS",
    "UniverseQualityViolation",
    "UniverseScreens",
    "delisting_coverage_report",
    "select_rolling_universe",
    "select_universe_at",
    "validate_security_master",
]
