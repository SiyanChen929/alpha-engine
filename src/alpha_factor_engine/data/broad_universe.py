"""Fail-closed contract for a survivorship-free broad equity research panel.

The contract is intentionally stricter than the public-data diagnostic.  A
large current-ticker download is not accepted as a historical universe.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from alpha_factor_engine.contracts import PointInTimeViolation


class BroadUniverseDataUnavailable(FileNotFoundError):
    """Raised when the licensed PIT panel required by v3 is unavailable."""


@dataclass(frozen=True)
class BroadUniverseRequirements:
    minimum_securities_per_month: int = 1_000
    minimum_price: float = 5.0
    minimum_market_cap_usd: float = 100_000_000.0
    minimum_adv_usd_3m: float = 1_000_000.0
    common_share_codes: tuple[int, ...] = (10, 11)
    exchange_codes: tuple[int, ...] = (1, 2, 3)
    one_security_per_company: bool = True


REQUIRED_COLUMNS = {
    "permno",
    "permco",
    "ticker",
    "signal_at",
    "entry_at",
    "exit_at",
    "membership_available_at",
    "price_available_at",
    "fundamental_available_at",
    "industry_available_at",
    "shrcd",
    "exchcd",
    "prc",
    "market_cap_usd",
    "adv_usd_3m",
    "historical_sector",
    "eligible",
    "base_period_return",
    "delisting_return",
    "delisting_return_observed",
    "target_return",
}

FUNDAMENTAL_FEATURES = {
    "book_to_market",
    "earnings_yield",
    "roe",
    "roa",
    "gross_profitability",
    "debt_to_equity",
    "asset_growth",
    "sales_growth",
    "accruals",
    "free_cash_flow_yield",
}


def load_broad_universe_panel(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise BroadUniverseDataUnavailable(
            "broad PIT panel is unavailable; expected a CRSP-security-master/"
            f"delisting-backed parquet file at {source}"
        )
    return pd.read_parquet(source)


def _check(
    checks: list[dict[str, object]],
    name: str,
    passed: bool,
    detail: str,
    *,
    critical: bool = True,
) -> None:
    checks.append(
        {
            "check": name,
            "passed": bool(passed),
            "critical": critical,
            "detail": detail,
        }
    )


def audit_broad_universe_panel(
    frame: pd.DataFrame,
    *,
    requirements: BroadUniverseRequirements | None = None,
    feature_columns: Iterable[str] = (),
) -> dict[str, object]:
    """Return a structured audit without silently repairing invalid input."""

    req = requirements or BroadUniverseRequirements()
    checks: list[dict[str, object]] = []
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    _check(checks, "required_columns", not missing, f"missing={missing}")
    if missing:
        return {
            "status": "FAIL",
            "requirements": asdict(req),
            "checks": checks,
            "rows": int(len(frame)),
        }

    panel = frame.copy()
    for column in [
        "signal_at",
        "entry_at",
        "exit_at",
        "membership_available_at",
        "price_available_at",
        "fundamental_available_at",
        "industry_available_at",
    ]:
        panel[column] = pd.to_datetime(panel[column], utc=True, errors="coerce")

    duplicate_security = panel.duplicated(["signal_at", "permno"], keep=False)
    _check(
        checks,
        "unique_security_month",
        not duplicate_security.any(),
        f"duplicate_rows={int(duplicate_security.sum())}",
    )
    if req.one_security_per_company:
        duplicate_company = panel.loc[panel["eligible"].fillna(False).astype(bool)].duplicated(
            ["signal_at", "permco"], keep=False
        )
        _check(
            checks,
            "one_security_per_company",
            not duplicate_company.any(),
            f"duplicate_eligible_rows={int(duplicate_company.sum())}",
        )

    permanent_ids = panel["permno"].notna() & panel["permco"].notna()
    _check(
        checks,
        "permanent_identifier_coverage",
        permanent_ids.all(),
        f"missing_rows={int((~permanent_ids).sum())}",
    )

    eligible = panel["eligible"].fillna(False).astype(bool)
    allowed_security = panel["shrcd"].isin(req.common_share_codes) & panel["exchcd"].isin(
        req.exchange_codes
    )
    _check(
        checks,
        "historical_common_share_exchange_filter",
        bool(allowed_security.loc[eligible].all()),
        f"invalid_eligible_rows={int((eligible & ~allowed_security).sum())}",
    )

    numeric_filters = {
        "price": pd.to_numeric(panel["prc"], errors="coerce").abs().ge(req.minimum_price),
        "market_cap": pd.to_numeric(panel["market_cap_usd"], errors="coerce").ge(
            req.minimum_market_cap_usd
        ),
        "adv": pd.to_numeric(panel["adv_usd_3m"], errors="coerce").ge(req.minimum_adv_usd_3m),
    }
    for label, mask in numeric_filters.items():
        _check(
            checks,
            f"eligible_{label}_filter",
            bool(mask.loc[eligible].all()),
            f"invalid_eligible_rows={int((eligible & ~mask).sum())}",
        )

    information_timestamps = [
        "membership_available_at",
        "price_available_at",
        "industry_available_at",
    ]
    for column in information_timestamps:
        valid = panel[column].notna() & panel[column].le(panel["signal_at"])
        _check(
            checks,
            f"{column}_is_point_in_time",
            bool(valid.loc[eligible].all()),
            f"invalid_eligible_rows={int((eligible & ~valid).sum())}",
        )

    requested_features = set(feature_columns)
    missing_features = sorted(requested_features.difference(panel.columns))
    _check(
        checks,
        "required_model_features",
        not missing_features,
        f"missing={missing_features}",
    )
    fundamental_used = sorted(
        requested_features.intersection(FUNDAMENTAL_FEATURES).intersection(panel.columns)
    )
    if fundamental_used:
        populated = panel[fundamental_used].notna().any(axis=1)
        valid_fundamental_time = panel["fundamental_available_at"].notna() & panel[
            "fundamental_available_at"
        ].le(panel["signal_at"])
        relevant = eligible & populated
        _check(
            checks,
            "fundamental_features_are_point_in_time",
            bool(valid_fundamental_time.loc[relevant].all()),
            f"features={fundamental_used}; invalid_populated_rows={int((relevant & ~valid_fundamental_time).sum())}",
        )

    execution_order = (
        panel["signal_at"].notna()
        & panel["entry_at"].notna()
        & panel["exit_at"].notna()
        & panel["signal_at"].lt(panel["entry_at"])
        & panel["entry_at"].lt(panel["exit_at"])
    )
    _check(
        checks,
        "signal_entry_exit_order",
        bool(execution_order.loc[eligible].all()),
        f"invalid_eligible_rows={int((eligible & ~execution_order).sum())}",
    )

    explicit_delisting = panel["delisting_return_observed"].fillna(False).astype(bool)
    base = pd.to_numeric(panel["base_period_return"], errors="coerce")
    delist = pd.to_numeric(panel["delisting_return"], errors="coerce")
    target = pd.to_numeric(panel["target_return"], errors="coerce")
    expected_target = (1.0 + base) * (1.0 + delist) - 1.0
    return_complete = (
        explicit_delisting
        & base.notna()
        & delist.notna()
        & target.notna()
        & np.isclose(target, expected_target, atol=1e-10, rtol=1e-8)
    )
    _check(
        checks,
        "delisting_return_is_explicitly_included",
        bool(return_complete.loc[eligible].all()),
        f"invalid_eligible_rows={int((eligible & ~return_complete).sum())}",
    )

    sector_complete = panel["historical_sector"].notna() & panel[
        "historical_sector"
    ].astype(str).str.strip().ne("")
    _check(
        checks,
        "historical_sector_complete",
        bool(sector_complete.loc[eligible].all()),
        f"missing_eligible_rows={int((eligible & ~sector_complete).sum())}",
    )

    counts = panel.loc[eligible].groupby("signal_at")["permno"].nunique().sort_index()
    minimum_observed = int(counts.min()) if len(counts) else 0
    _check(
        checks,
        "minimum_monthly_cross_section",
        minimum_observed >= req.minimum_securities_per_month,
        f"minimum_observed={minimum_observed}; required={req.minimum_securities_per_month}; months={len(counts)}",
    )

    critical_failures = [item for item in checks if item["critical"] and not item["passed"]]
    return {
        "status": "PASS" if not critical_failures else "FAIL",
        "requirements": asdict(req),
        "checks": checks,
        "rows": int(len(panel)),
        "eligible_rows": int(eligible.sum()),
        "months": int(len(counts)),
        "minimum_securities_observed": minimum_observed,
        "unique_permnos": int(panel.loc[eligible, "permno"].nunique()),
    }


def validate_broad_universe_panel(
    frame: pd.DataFrame,
    *,
    requirements: BroadUniverseRequirements | None = None,
    feature_columns: Iterable[str] = (),
) -> dict[str, object]:
    audit = audit_broad_universe_panel(
        frame,
        requirements=requirements,
        feature_columns=feature_columns,
    )
    if audit["status"] != "PASS":
        failures = [
            str(item["check"])
            for item in audit["checks"]
            if item["critical"] and not item["passed"]
        ]
        raise PointInTimeViolation(
            "broad-universe validation failed closed: " + ", ".join(failures)
        )
    return audit


__all__ = [
    "BroadUniverseDataUnavailable",
    "BroadUniverseRequirements",
    "FUNDAMENTAL_FEATURES",
    "REQUIRED_COLUMNS",
    "audit_broad_universe_panel",
    "load_broad_universe_panel",
    "validate_broad_universe_panel",
]
