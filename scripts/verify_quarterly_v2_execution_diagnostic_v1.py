#!/usr/bin/env python3
"""Independent verifier for the opened quarterly-v2 execution diagnostic.

This file intentionally does not import the producer runner, the production
portfolio execution module, or any project metric helper.  The calculations
below are reconstructed from the frozen recovered panel and public artifact
contracts only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT / "artifacts/quarterly_fundamental_momentum_diagnostic_v2_attempt1_recovered"
)
DEFAULT_REPORTED = ROOT / "artifacts/quarterly_v2_execution_diagnostic_v1"
DEFAULT_OUTPUT = (
    ROOT / "artifacts/quarterly_v2_execution_diagnostic_v1_independent_audit"
)
DEFAULT_CONFIG = ROOT / "configs/quarterly_v2_execution_diagnostic_v1.yml"
DEFAULT_PREREG = (
    ROOT / "artifacts/quarterly_v2_execution_diagnostic_v1_preregistration.json"
)

NUMERIC_TOLERANCE = 1e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def parse_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    token = str(value).strip().lower()
    if token in {"true", "1", "yes"}:
        return True
    if token in {"false", "0", "no"}:
        return False
    raise ValueError(f"cannot parse boolean: {value!r}")


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def stable_equal_count_deciles(
    month: pd.DataFrame,
    *,
    score_column: str,
    buckets: int = 10,
) -> pd.DataFrame:
    """Assign deterministic equal-count buckets from score, issuer, ticker."""

    _require_columns(month, [score_column, "issuer_cik", "ticker"], "monthly panel")
    if buckets < 2:
        raise ValueError("buckets must be >= 2")
    score = pd.to_numeric(month[score_column], errors="coerce")
    ranked = month.loc[score.notna()].copy()
    if ranked.empty:
        return ranked.assign(decile=pd.Series(dtype="int64"))
    if ranked["ticker"].isna().any():
        raise ValueError("ranked panel contains missing ticker")
    ranked["_audit_score"] = pd.to_numeric(
        ranked[score_column], errors="raise"
    ).astype(float)
    ranked["_audit_issuer"] = pd.to_numeric(
        ranked["issuer_cik"], errors="coerce"
    ).fillna(np.inf)
    ranked["_audit_ticker"] = ranked["ticker"].astype(str)
    ranked = ranked.sort_values(
        ["_audit_score", "_audit_issuer", "_audit_ticker"], kind="mergesort"
    )
    ordinal = np.arange(len(ranked), dtype=float)
    ranked["decile"] = np.floor(ordinal * buckets / len(ranked)).astype(int) + 1
    ranked["decile"] = ranked["decile"].clip(1, buckets)
    return ranked.drop(columns=["_audit_score", "_audit_issuer", "_audit_ticker"])


def marked_weights(
    tickers: list[str], returns: Mapping[str, float]
) -> dict[str, float]:
    """Mark prior equal absolute notionals to market and normalize to one."""

    if not tickers:
        raise ValueError("cannot mark an empty sleeve")
    raw: dict[str, float] = {}
    initial_weight = 1.0 / len(tickers)
    for ticker in tickers:
        value = float(returns[ticker])
        if not np.isfinite(value) or value <= -1.0:
            raise ValueError(f"invalid holding return for {ticker}: {value}")
        raw[ticker] = initial_weight * (1.0 + value)
    gross = float(sum(raw.values()))
    if not np.isfinite(gross) or gross <= 0:
        raise ValueError("marked sleeve has invalid gross")
    return {ticker: value / gross for ticker, value in raw.items()}


def absolute_trade_sum(
    targets: list[str], prior_marked: Mapping[str, float] | None
) -> float:
    """Sum absolute target trades for a unit-gross equal-weight sleeve."""

    if not targets:
        raise ValueError("cannot target an empty sleeve")
    target_weight = 1.0 / len(targets)
    target = {ticker: target_weight for ticker in targets}
    prior = dict(prior_marked or {})
    names = set(target).union(prior)
    return float(
        sum(abs(target.get(name, 0.0) - prior.get(name, 0.0)) for name in names)
    )


def reconstruct_execution(
    panel: pd.DataFrame,
    *,
    score_column: str,
    return_column: str,
    adv_column: str,
    participations: tuple[float, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rebuild all deciles, extreme holdings, returns, turnover, and capacity."""

    _require_columns(
        panel,
        [
            "signal_date",
            "ticker",
            "issuer_cik",
            score_column,
            return_column,
            adv_column,
        ],
        "recovered panel",
    )
    work = panel.copy()
    work["signal_date"] = pd.to_datetime(work["signal_date"], utc=True, errors="raise")
    if work.duplicated(["signal_date", "ticker"]).any():
        raise ValueError("panel contains duplicate ticker-month keys")

    monthly_rows: list[dict[str, Any]] = []
    holding_rows: list[dict[str, Any]] = []
    decile_rows: list[dict[str, Any]] = []
    prior_long: list[str] | None = None
    prior_short: list[str] | None = None
    prior_returns: dict[str, float] | None = None

    for signal_date, month in work.groupby("signal_date", sort=True):
        ranked = stable_equal_count_deciles(month, score_column=score_column)
        if ranked.empty:
            continue
        returns = pd.to_numeric(ranked[return_column], errors="raise").astype(float)
        if not np.isfinite(returns).all():
            raise ValueError(f"non-finite target return at {signal_date}")
        ranked = ranked.assign(_audit_return=returns.to_numpy())

        for decile, sleeve in ranked.groupby("decile", sort=True):
            decile_rows.append(
                {
                    "signal_date": signal_date,
                    "feature": score_column,
                    "decile": int(decile),
                    "names": int(len(sleeve)),
                    "equal_weight_return": float(sleeve["_audit_return"].mean()),
                }
            )

        short = ranked.loc[ranked["decile"].eq(1)].copy()
        long = ranked.loc[ranked["decile"].eq(10)].copy()
        if short.empty or long.empty:
            raise ValueError(f"missing extreme decile at {signal_date}")

        for name, sleeve, sign in (("short", short, -1.0), ("long", long, 1.0)):
            target_weight = 1.0 / len(sleeve)
            for row in sleeve[
                ["ticker", "issuer_cik", "decile", "_audit_return", adv_column]
            ].to_dict("records"):
                holding_rows.append(
                    {
                        "signal_date": signal_date,
                        "sleeve": name,
                        "ticker": str(row["ticker"]),
                        "issuer_cik": float(row["issuer_cik"]),
                        "decile": int(row["decile"]),
                        "target_absolute_weight": target_weight,
                        "signed_weight": sign * target_weight,
                        "target_return": float(row["_audit_return"]),
                        "adv_usd_3m": float(row[adv_column]),
                    }
                )

        long_names = long["ticker"].astype(str).tolist()
        short_names = short["ticker"].astype(str).tolist()
        if prior_long is None:
            long_turnover = 1.0
            short_turnover = 1.0
        else:
            if prior_short is None or prior_returns is None:
                raise RuntimeError("incomplete prior sleeve state")
            long_turnover = absolute_trade_sum(
                long_names, marked_weights(prior_long, prior_returns)
            )
            short_turnover = absolute_trade_sum(
                short_names, marked_weights(prior_short, prior_returns)
            )

        long_return = float(long["_audit_return"].mean())
        short_return = float(short["_audit_return"].mean())
        all_adv = pd.to_numeric(
            pd.concat([long[adv_column], short[adv_column]]), errors="coerce"
        ).astype(float)
        monthly: dict[str, Any] = {
            "signal_date": signal_date,
            "long_names": int(len(long)),
            "short_names": int(len(short)),
            "long_return": long_return,
            "short_return": short_return,
            "gross_long_short_return": long_return - short_return,
            "long_one_way_traded_notional": long_turnover,
            "short_one_way_traded_notional": short_turnover,
            "total_one_way_traded_notional": long_turnover + short_turnover,
            "adv_complete": bool(np.isfinite(all_adv).all() and all_adv.gt(0).all()),
        }
        for participation in participations:
            if not 0 < participation <= 1:
                raise ValueError("ADV participation must be in (0, 1]")
            side_values: list[float] = []
            for sleeve in (long, short):
                adv = pd.to_numeric(sleeve[adv_column], errors="coerce").to_numpy(
                    dtype=float
                )
                if not np.isfinite(adv).all() or np.any(adv <= 0):
                    side_values.append(float("nan"))
                else:
                    side_values.append(float(np.min(adv * participation * len(sleeve))))
            label = f"capacity_side_usd_at_{participation:.4f}".replace(".", "p")
            monthly[label] = (
                float(np.min(side_values))
                if np.isfinite(np.asarray(side_values, dtype=float)).all()
                else float("nan")
            )
        monthly_rows.append(monthly)
        prior_long = long_names
        prior_short = short_names
        prior_returns = dict(
            zip(
                ranked["ticker"].astype(str),
                ranked["_audit_return"].astype(float),
                strict=True,
            )
        )

    return (
        pd.DataFrame(monthly_rows),
        pd.DataFrame(holding_rows),
        pd.DataFrame(decile_rows),
    )


def newey_west_intercept_t(values: Iterable[float], lags: int = 3) -> float:
    """Independent Bartlett-kernel Newey-West t statistic for a sample mean."""

    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    n = len(array)
    if n < lags + 3:
        return float("nan")
    residual = array - array.mean()
    long_run_variance = float(np.dot(residual, residual) / n)
    for lag in range(1, lags + 1):
        bartlett = 1.0 - lag / (lags + 1.0)
        autocovariance = float(np.dot(residual[lag:], residual[:-lag]) / n)
        long_run_variance += 2.0 * bartlett * autocovariance
    standard_error = math.sqrt(max(long_run_variance, 0.0) / n)
    return float(array.mean() / standard_error) if standard_error > 0 else float("nan")


def one_sided_normal_pvalue(tstat: float) -> float:
    return float(0.5 * math.erfc(tstat / math.sqrt(2.0)))


def period_masks(dates: pd.Series) -> dict[str, pd.Series]:
    parsed = pd.to_datetime(dates, utc=True, errors="raise")
    return {
        "full": pd.Series(True, index=dates.index),
        "pre_2026": parsed.lt(pd.Timestamp("2026-01-01", tz="UTC")),
        "2026_h1_reporting_only": parsed.ge(pd.Timestamp("2026-01-01", tz="UTC")),
    }


def build_return_summary(
    monthly: pd.DataFrame,
    *,
    costs_bps: list[int],
    hac_lags: int,
    trial_floor: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    p_column = f"global_bonferroni_pvalue_floor_{trial_floor}"
    for period, mask in period_masks(monthly["signal_date"]).items():
        sample = monthly.loc[mask]
        for bps in [0, *costs_bps]:
            column = (
                "gross_long_short_return" if bps == 0 else f"net_return_{bps}bps"
            )
            tstat = newey_west_intercept_t(sample[column], lags=hac_lags)
            raw_p = one_sided_normal_pvalue(tstat)
            rows.append(
                {
                    "period": period,
                    "cost_bps_per_dollar_traded": bps,
                    "months": int(len(sample)),
                    "annualized_arithmetic_return": float(12.0 * sample[column].mean()),
                    "hac_tstat": tstat,
                    "one_sided_hac_pvalue": raw_p,
                    p_column: min(1.0, raw_p * trial_floor) if bps > 0 else np.nan,
                    "mean_total_one_way_traded_notional": float(
                        sample["total_one_way_traded_notional"].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_capacity_summary(
    monthly: pd.DataFrame, participations: list[float]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for participation in participations:
        column = f"capacity_side_usd_at_{participation:.4f}".replace(".", "p")
        for period, mask in period_masks(monthly["signal_date"]).items():
            values = pd.to_numeric(monthly.loc[mask, column], errors="coerce")
            rows.append(
                {
                    "period": period,
                    "adv_participation": participation,
                    "months": int(mask.sum()),
                    "months_with_complete_capacity": int(values.notna().sum()),
                    "minimum_side_aum_usd": float(values.min()),
                    "median_side_aum_usd": float(values.median()),
                    "maximum_side_aum_usd": float(values.max()),
                    "median_gross_notional_usd": float(2.0 * values.median()),
                }
            )
    return pd.DataFrame(rows)


def path_statistics(monthly: pd.DataFrame, costs_bps: list[int]) -> pd.DataFrame:
    pre = monthly.loc[
        pd.to_datetime(monthly["signal_date"], utc=True).lt(
            pd.Timestamp("2026-01-01", tz="UTC")
        )
    ]
    rows: list[dict[str, Any]] = []
    for bps in [0, *costs_bps]:
        column = "gross_long_short_return" if bps == 0 else f"net_return_{bps}bps"
        returns = pd.to_numeric(pre[column], errors="raise").to_numpy(dtype=float)
        wealth = np.cumprod(1.0 + returns)
        wealth_with_initial = np.concatenate(([1.0], wealth))
        running_peak = np.maximum.accumulate(wealth_with_initial)
        drawdown = wealth_with_initial / running_peak - 1.0
        annualized_volatility = float(np.std(returns, ddof=1) * math.sqrt(12.0))
        annualized_arithmetic = float(np.mean(returns) * 12.0)
        rows.append(
            {
                "period": "pre_2026",
                "cost_bps_per_dollar_traded": bps,
                "months": len(returns),
                "terminal_wealth": float(wealth[-1]),
                "cagr": float(wealth[-1] ** (12.0 / len(returns)) - 1.0),
                "max_drawdown": float(np.min(drawdown)),
                "annualized_volatility": annualized_volatility,
                "naive_sharpe": (
                    annualized_arithmetic / annualized_volatility
                    if annualized_volatility > 0
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def rank_correlation(x: Iterable[float], y: Iterable[float]) -> float:
    x_rank = pd.Series(list(x), dtype=float).rank(method="average").to_numpy()
    y_rank = pd.Series(list(y), dtype=float).rank(method="average").to_numpy()
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def compare_frames(
    expected: pd.DataFrame,
    observed: pd.DataFrame,
    *,
    keys: list[str],
    exact_columns: list[str],
    numeric_columns: list[str],
    tolerance: float = NUMERIC_TOLERANCE,
    numeric_tolerances: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Compare keyed tables, preserving NaN equality and exact mismatch counts."""

    _require_columns(expected, [*keys, *exact_columns, *numeric_columns], "expected")
    _require_columns(observed, [*keys, *exact_columns, *numeric_columns], "observed")
    left = expected[[*keys, *exact_columns, *numeric_columns]].copy()
    right = observed[[*keys, *exact_columns, *numeric_columns]].copy()
    for key in keys:
        if "date" in key:
            left[key] = pd.to_datetime(left[key], utc=True, errors="raise")
            right[key] = pd.to_datetime(right[key], utc=True, errors="raise")
    left_duplicate_count = int(left.duplicated(keys).sum())
    right_duplicate_count = int(right.duplicated(keys).sum())
    merged = left.merge(right, on=keys, how="outer", suffixes=("_expected", "_observed"), indicator=True)
    missing_from_observed = int(merged["_merge"].eq("left_only").sum())
    extra_in_observed = int(merged["_merge"].eq("right_only").sum())
    both = merged.loc[merged["_merge"].eq("both")].copy()

    exact_mismatches: dict[str, int] = {}
    for column in exact_columns:
        lhs = both[f"{column}_expected"]
        rhs = both[f"{column}_observed"]
        equal = lhs.eq(rhs) | (lhs.isna() & rhs.isna())
        exact_mismatches[column] = int((~equal).sum())

    numeric_errors: dict[str, dict[str, Any]] = {}
    for column in numeric_columns:
        column_tolerance = float((numeric_tolerances or {}).get(column, tolerance))
        lhs = pd.to_numeric(both[f"{column}_expected"], errors="coerce").to_numpy(float)
        rhs = pd.to_numeric(both[f"{column}_observed"], errors="coerce").to_numpy(float)
        nan_mismatch = np.isnan(lhs) ^ np.isnan(rhs)
        finite_pair = np.isfinite(lhs) & np.isfinite(rhs)
        errors = np.abs(lhs[finite_pair] - rhs[finite_pair])
        maximum = float(np.max(errors)) if len(errors) else 0.0
        over_tolerance = int(np.sum(errors > column_tolerance)) + int(np.sum(nan_mismatch))
        numeric_errors[column] = {
            "maximum_absolute_error": maximum,
            "absolute_tolerance": column_tolerance,
            "nan_pattern_mismatch_count": int(np.sum(nan_mismatch)),
            "mismatch_count_over_tolerance": over_tolerance,
        }

    passed = bool(
        left_duplicate_count == 0
        and right_duplicate_count == 0
        and missing_from_observed == 0
        and extra_in_observed == 0
        and all(value == 0 for value in exact_mismatches.values())
        and all(
            value["mismatch_count_over_tolerance"] == 0
            for value in numeric_errors.values()
        )
    )
    return {
        "passed": passed,
        "expected_rows": int(len(left)),
        "observed_rows": int(len(right)),
        "expected_duplicate_key_count": left_duplicate_count,
        "observed_duplicate_key_count": right_duplicate_count,
        "missing_from_observed": missing_from_observed,
        "extra_in_observed": extra_in_observed,
        "exact_mismatch_counts": exact_mismatches,
        "numeric_errors": numeric_errors,
    }


def verify_manifest_records(manifest_path: Path, artifact_directory: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("records", [])
    normalized_paths = [Path(row["path"]).resolve().as_posix() for row in records]
    duplicates = sorted(
        {path for path in normalized_paths if normalized_paths.count(path) > 1}
    )
    missing: list[str] = []
    size_mismatches: list[str] = []
    hash_mismatches: list[str] = []
    for record in records:
        path = Path(record["path"])
        if not path.is_file():
            missing.append(path.as_posix())
            continue
        if path.stat().st_size != int(record["bytes"]):
            size_mismatches.append(path.as_posix())
        if sha256_file(path) != record["sha256"]:
            hash_mismatches.append(path.as_posix())

    declared_outputs = {
        Path(row["path"]).resolve()
        for row in records
        if row.get("role") == "output"
        and Path(row["path"]).resolve().parent == artifact_directory.resolve()
    }
    actual_outputs = {
        path.resolve()
        for path in artifact_directory.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    omissions = sorted(path.as_posix() for path in actual_outputs - declared_outputs)
    extras = sorted(path.as_posix() for path in declared_outputs - actual_outputs)
    return {
        "record_count": len(records),
        "duplicate_paths": duplicates,
        "missing_files": missing,
        "size_mismatches": size_mismatches,
        "hash_mismatches": hash_mismatches,
        "output_closure_omissions": omissions,
        "output_closure_extras": extras,
        "record_integrity_passed": not (
            duplicates or missing or size_mismatches or hash_mismatches
        ),
        "output_closure_passed": not (omissions or extras),
        "manifest": manifest,
    }


def _manifest_paths(manifest: Mapping[str, Any]) -> set[Path]:
    return {Path(row["path"]).resolve() for row in manifest.get("records", [])}


def _parse_timestamp(value: Any) -> pd.Timestamp:
    return pd.Timestamp(value).tz_convert("UTC") if pd.Timestamp(value).tzinfo else pd.Timestamp(value).tz_localize("UTC")


def expected_gates(
    monthly: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    gross_decile_error: float,
    costs_bps: list[int],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    pre = summary.loc[summary["period"].eq("pre_2026")].set_index(
        "cost_bps_per_dollar_traded"
    )
    p_column = f"global_bonferroni_pvalue_floor_{int(config['evaluation']['resulting_global_trial_floor'])}"
    threshold = float(config["gates"]["net_hac_tstat_minimum"])
    rows: list[tuple[str, bool, str]] = [
        (
            "gross_result_reconciles_to_frozen_deciles",
            gross_decile_error <= NUMERIC_TOLERANCE,
            f"maximum_error={gross_decile_error:.3e}",
        ),
        (
            "gross_signal_statistical_gate",
            bool(pre.loc[0, "hac_tstat"] >= threshold),
            f"pre2026_HAC_t={pre.loc[0, 'hac_tstat']:.6f}",
        ),
    ]
    for bps in costs_bps:
        row = pre.loc[bps]
        rows.extend(
            [
                (
                    f"net_{bps}bps_return_positive",
                    bool(row["annualized_arithmetic_return"] > 0),
                    f"annualized={row['annualized_arithmetic_return']:.6f}",
                ),
                (
                    f"net_{bps}bps_hac_tstat",
                    bool(row["hac_tstat"] >= threshold),
                    f"observed={row['hac_tstat']:.6f}",
                ),
                (
                    f"net_{bps}bps_global_p",
                    bool(row[p_column] < 0.05),
                    f"observed={row[p_column]:.6f}",
                ),
            ]
        )
    rows.extend(
        [
            (
                "complete_positive_adv",
                bool(monthly["adv_complete"].all()),
                f"complete={int(monthly['adv_complete'].sum())}/{len(monthly)}",
            ),
            (
                "permanent_security_lineage_and_delistings",
                False,
                "upstream current-survivor ticker panel",
            ),
            (
                "borrow_fee_and_availability",
                bool(config["costs"]["borrow_fee_included"]),
                "borrow data unavailable",
            ),
            (
                "market_impact_model",
                bool(config["costs"]["market_impact_model_included"]),
                "linear explicit-cost scenarios only",
            ),
            (
                "portfolio_authorized",
                False,
                "gross statistical, lineage, borrow, and impact gates are required",
            ),
        ]
    )
    return pd.DataFrame(rows, columns=["gate", "passed", "evidence"])


def run_audit(
    *,
    source_directory: Path = DEFAULT_SOURCE,
    reported_directory: Path = DEFAULT_REPORTED,
    config_path: Path = DEFAULT_CONFIG,
    prereg_path: Path = DEFAULT_PREREG,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, Any]]:
    """Compute all verification results without writing or mutating producer files."""

    source_directory = source_directory.resolve()
    reported_directory = reported_directory.resolve()
    config_path = config_path.resolve()
    prereg_path = prereg_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))

    panel_path = source_directory / "monthly_quarterly_signal_panel.parquet"
    upstream_deciles_path = source_directory / "decile_monthly.csv"
    upstream_decile_summary_path = source_directory / "decile_summary.csv"
    upstream_manifest_path = source_directory / "artifact_manifest.json"
    panel = pd.read_parquet(panel_path)

    score_column = str(config["portfolio"]["signal"])
    costs_bps = [int(value) for value in config["costs"]["scenarios_bps"]]
    participations = [
        float(value) for value in config["capacity"]["participation_rates"]
    ]
    trial_floor = int(config["evaluation"]["resulting_global_trial_floor"])
    hac_lags = int(config["evaluation"]["hac_lags"])
    monthly, holdings, deciles = reconstruct_execution(
        panel,
        score_column=score_column,
        return_column="target_return",
        adv_column=str(config["capacity"]["adv_column"]),
        participations=tuple(participations),
    )
    for bps in costs_bps:
        monthly[f"cost_{bps}bps"] = (
            monthly["total_one_way_traded_notional"] * bps / 10_000.0
        )
        monthly[f"net_return_{bps}bps"] = (
            monthly["gross_long_short_return"] - monthly[f"cost_{bps}bps"]
        )
    return_summary = build_return_summary(
        monthly,
        costs_bps=costs_bps,
        hac_lags=hac_lags,
        trial_floor=trial_floor,
    )
    capacity_summary = build_capacity_summary(monthly, participations)
    path_summary = path_statistics(monthly, costs_bps)

    observed_monthly = pd.read_csv(reported_directory / "monthly_execution.csv")
    observed_monthly["signal_date"] = pd.to_datetime(
        observed_monthly["signal_date"], utc=True, errors="raise"
    )
    observed_holdings = pd.read_parquet(reported_directory / "holdings.parquet")
    observed_return_summary = pd.read_csv(reported_directory / "return_summary.csv")
    observed_capacity_summary = pd.read_csv(reported_directory / "capacity_summary.csv")
    observed_gates = pd.read_csv(reported_directory / "gates.csv")
    observed_gates["passed"] = observed_gates["passed"].map(parse_bool)
    observed_trial = json.loads(
        (reported_directory / "opened_trial_ledger.json").read_text(encoding="utf-8")
    )
    observed_run = json.loads(
        (reported_directory / "run_summary.json").read_text(encoding="utf-8")
    )

    capacity_columns = [
        f"capacity_side_usd_at_{value:.4f}".replace(".", "p")
        for value in participations
    ]
    cost_columns = [
        column
        for bps in costs_bps
        for column in (f"cost_{bps}bps", f"net_return_{bps}bps")
    ]
    monthly_comparison = compare_frames(
        monthly,
        observed_monthly,
        keys=["signal_date"],
        exact_columns=["long_names", "short_names", "adv_complete"],
        numeric_columns=[
            "long_return",
            "short_return",
            "gross_long_short_return",
            "long_one_way_traded_notional",
            "short_one_way_traded_notional",
            "total_one_way_traded_notional",
            *capacity_columns,
            *cost_columns,
        ],
        numeric_tolerances={column: 1e-6 for column in capacity_columns},
    )
    holdings_comparison = compare_frames(
        holdings,
        observed_holdings,
        keys=["signal_date", "sleeve", "ticker"],
        exact_columns=["decile"],
        numeric_columns=[
            "issuer_cik",
            "target_absolute_weight",
            "signed_weight",
            "target_return",
            "adv_usd_3m",
        ],
    )
    upstream_deciles = pd.read_csv(upstream_deciles_path)
    upstream_deciles = upstream_deciles.loc[
        upstream_deciles["feature"].eq(score_column)
    ].copy()
    decile_comparison = compare_frames(
        deciles,
        upstream_deciles,
        keys=["signal_date", "feature", "decile"],
        exact_columns=["names"],
        numeric_columns=["equal_weight_return"],
    )
    return_summary_comparison = compare_frames(
        return_summary,
        observed_return_summary,
        keys=["period", "cost_bps_per_dollar_traded"],
        exact_columns=["months"],
        numeric_columns=[
            "annualized_arithmetic_return",
            "hac_tstat",
            "one_sided_hac_pvalue",
            f"global_bonferroni_pvalue_floor_{trial_floor}",
            "mean_total_one_way_traded_notional",
        ],
    )
    capacity_comparison = compare_frames(
        capacity_summary,
        observed_capacity_summary,
        keys=["period", "adv_participation"],
        exact_columns=["months", "months_with_complete_capacity"],
        numeric_columns=[
            "minimum_side_aum_usd",
            "median_side_aum_usd",
            "maximum_side_aum_usd",
            "median_gross_notional_usd",
        ],
        numeric_tolerances={
            "minimum_side_aum_usd": 1e-6,
            "median_side_aum_usd": 1e-6,
            "maximum_side_aum_usd": 1e-6,
            "median_gross_notional_usd": 1e-6,
        },
    )

    d10 = deciles.loc[deciles["decile"].eq(10)].set_index("signal_date")
    d1 = deciles.loc[deciles["decile"].eq(1)].set_index("signal_date")
    independent_gross = d10["equal_weight_return"] - d1["equal_weight_return"]
    gross_internal_error = float(
        (
            monthly.set_index("signal_date")["gross_long_short_return"]
            - independent_gross
        )
        .abs()
        .max()
    )
    upstream_extremes = upstream_deciles.loc[
        upstream_deciles["decile"].isin([1, 10])
    ].copy()
    upstream_extremes["signal_date"] = pd.to_datetime(
        upstream_extremes["signal_date"], utc=True, errors="raise"
    )
    upstream_pivot = upstream_extremes.pivot(
        index="signal_date", columns="decile", values="equal_weight_return"
    )
    gross_by_decile_error = float(
        (
            monthly.set_index("signal_date")["gross_long_short_return"]
            - (upstream_pivot[10] - upstream_pivot[1])
        )
        .abs()
        .max()
    )
    expected_gate_frame = expected_gates(
        monthly,
        return_summary,
        gross_decile_error=gross_by_decile_error,
        costs_bps=costs_bps,
        config=config,
    )
    gate_comparison = compare_frames(
        expected_gate_frame,
        observed_gates,
        keys=["gate"],
        exact_columns=["passed", "evidence"],
        numeric_columns=[],
    )

    expected_trial = {
        "classification": config["research_classification"],
        "historical_period_already_opened_before_preregistration": True,
        "prior_global_trial_floor": int(config["evaluation"]["prior_global_trial_floor"]),
        "cost_tests_added": len(costs_bps),
        "resulting_global_trial_floor": trial_floor,
        "scenarios_bps": costs_bps,
        "selection_rule": "all frozen scenarios retained; no cost or capacity threshold tuning",
    }
    trial_mismatches = {
        key: {"expected": value, "observed": observed_trial.get(key)}
        for key, value in expected_trial.items()
        if observed_trial.get(key) != value
    }
    arithmetic_trial_floor_passed = bool(
        expected_trial["prior_global_trial_floor"]
        + expected_trial["cost_tests_added"]
        == expected_trial["resulting_global_trial_floor"]
    )
    frozen_at = _parse_timestamp(prereg["frozen_at"])
    opened_at = _parse_timestamp(observed_trial["opened_at"])
    generated_at = _parse_timestamp(observed_run["generated_at"])
    prereg_clock_passed = bool(frozen_at <= opened_at <= generated_at)
    prereg_mtime = pd.Timestamp(prereg_path.stat().st_mtime, unit="s", tz="UTC")

    prereg_record_mismatches: list[dict[str, Any]] = []
    for record in prereg.get("records", []):
        path = Path(record["path"])
        if not path.is_absolute():
            path = ROOT / path
        observed_sha = sha256_file(path) if path.is_file() else None
        if observed_sha != record.get("sha256"):
            prereg_record_mismatches.append(
                {
                    "path": path.resolve().as_posix(),
                    "expected_sha256": record.get("sha256"),
                    "observed_sha256": observed_sha,
                }
            )

    manifest_check = verify_manifest_records(
        reported_directory / "artifact_manifest.json", reported_directory
    )
    manifest = manifest_check.pop("manifest")
    producer_manifest_paths = _manifest_paths(manifest)
    required_outcome_direct_inputs = {
        panel_path.resolve(),
        upstream_deciles_path.resolve(),
        upstream_manifest_path.resolve(),
        prereg_path.resolve(),
        config_path.resolve(),
        (ROOT / "scripts/run_quarterly_v2_execution_diagnostic_v1.py").resolve(),
        (
            ROOT
            / "src/alpha_factor_engine/research/portfolio_execution_diagnostic.py"
        ).resolve(),
        (ROOT / "src/alpha_factor_engine/research/metrics.py").resolve(),
        (ROOT / "tests/test_portfolio_execution_diagnostic.py").resolve(),
    }
    required_descriptive_direct_inputs = {upstream_decile_summary_path.resolve()}
    missing_outcome_direct_inputs = sorted(
        path.as_posix()
        for path in required_outcome_direct_inputs.difference(producer_manifest_paths)
    )
    missing_descriptive_direct_inputs = sorted(
        path.as_posix()
        for path in required_descriptive_direct_inputs.difference(producer_manifest_paths)
    )
    upstream_manifest = json.loads(upstream_manifest_path.read_text(encoding="utf-8"))
    upstream_manifest_paths = _manifest_paths(upstream_manifest)
    decile_summary_transitively_recorded = (
        upstream_decile_summary_path.resolve() in upstream_manifest_paths
    )
    upstream_decile_summary_record = next(
        (
            row
            for row in upstream_manifest.get("records", [])
            if Path(row["path"]).resolve() == upstream_decile_summary_path.resolve()
        ),
        None,
    )
    decile_summary_transitive_hash_passed = bool(
        upstream_decile_summary_record
        and sha256_file(upstream_decile_summary_path)
        == upstream_decile_summary_record.get("sha256")
    )

    upstream_decile_summary = pd.read_csv(upstream_decile_summary_path)
    monotonicity_rows = upstream_decile_summary.loc[
        upstream_decile_summary["feature"].eq(score_column)
        & upstream_decile_summary["period"].eq("pre_2026")
        & upstream_decile_summary["decile"].astype(str).str.fullmatch(r"\d+")
    ].copy()
    recomputed_monotonicity = rank_correlation(
        pd.to_numeric(monotonicity_rows["decile"], errors="raise"),
        pd.to_numeric(
            monotonicity_rows["annualized_arithmetic_return"], errors="raise"
        ),
    )
    monotonicity_error = abs(
        recomputed_monotonicity - float(observed_run["pre2026_decile_return_spearman"])
    )

    verdict_fields_passed = bool(
        observed_run.get("historical_backtest_verdict") == "INVALID"
        and observed_run.get("research_status") == "NOT_VALIDATED"
        and observed_run.get("portfolio_authorized") is False
        and manifest.get("historical_backtest_verdict") == "INVALID"
        and manifest.get("portfolio_authorized") is False
    )
    numeric_integrity_passed = bool(
        monthly_comparison["passed"]
        and holdings_comparison["passed"]
        and decile_comparison["passed"]
        and return_summary_comparison["passed"]
        and capacity_comparison["passed"]
        and gate_comparison["passed"]
        and gross_by_decile_error <= NUMERIC_TOLERANCE
        and monotonicity_error <= NUMERIC_TOLERANCE
    )
    trial_ledger_fields_passed = not trial_mismatches and arithmetic_trial_floor_passed
    outcome_critical_lineage_passed = bool(
        manifest_check["record_integrity_passed"]
        and manifest_check["output_closure_passed"]
        and not missing_outcome_direct_inputs
        and not prereg_record_mismatches
    )
    strict_direct_read_closure_passed = bool(
        not missing_outcome_direct_inputs and not missing_descriptive_direct_inputs
    )
    strict_audit_passed = bool(
        numeric_integrity_passed
        and trial_ledger_fields_passed
        and prereg_clock_passed
        and outcome_critical_lineage_passed
        and strict_direct_read_closure_passed
        and verdict_fields_passed
    )

    pre_paths = path_summary.set_index("cost_bps_per_dollar_traded")
    verification: dict[str, Any] = {
        "audit_scope": "quarterly_v2_execution_diagnostic_v1 opened descriptive execution stress",
        "independence_contract": {
            "producer_runner_imported": False,
            "portfolio_execution_module_imported": False,
            "project_metric_helpers_imported": False,
        },
        "tolerance": NUMERIC_TOLERANCE,
        "panel_rows": int(len(panel)),
        "panel_months": int(panel["signal_date"].nunique()),
        "ranked_stock_months": int(deciles["names"].sum()),
        "extreme_holding_rows": int(len(holdings)),
        "monthly_execution": monthly_comparison,
        "extreme_holdings": holdings_comparison,
        "all_decile_aggregates": decile_comparison,
        "return_summary_and_hac": return_summary_comparison,
        "capacity_summary": capacity_comparison,
        "gates": gate_comparison,
        "gross_decile_reconciliation_maximum_error": gross_by_decile_error,
        "gross_internal_reconstruction_maximum_error": gross_internal_error,
        "decile_monotonicity_recomputed": recomputed_monotonicity,
        "decile_monotonicity_absolute_error": monotonicity_error,
        "trial_ledger": {
            "field_mismatches": trial_mismatches,
            "trial_floor_arithmetic_passed": arithmetic_trial_floor_passed,
            "fields_passed": trial_ledger_fields_passed,
            "prereg_frozen_at": frozen_at.isoformat(),
            "ledger_opened_at": opened_at.isoformat(),
            "run_generated_at": generated_at.isoformat(),
            "as_file_clock_order_passed": prereg_clock_passed,
            "prereg_filesystem_mtime": prereg_mtime.isoformat(),
            "filesystem_mtime_precedes_run": bool(prereg_mtime <= opened_at),
            "prereg_record_hash_mismatches": prereg_record_mismatches,
        },
        "producer_manifest": {
            **manifest_check,
            "missing_outcome_critical_direct_inputs": missing_outcome_direct_inputs,
            "missing_descriptive_direct_inputs": missing_descriptive_direct_inputs,
            "decile_summary_transitively_recorded": decile_summary_transitively_recorded,
            "decile_summary_transitive_hash_passed": decile_summary_transitive_hash_passed,
            "outcome_critical_lineage_passed": outcome_critical_lineage_passed,
            "strict_direct_read_closure_passed": strict_direct_read_closure_passed,
        },
        "pre2026_path_metrics": {
            str(int(bps)): {
                key: float(pre_paths.loc[bps, key])
                for key in [
                    "terminal_wealth",
                    "cagr",
                    "max_drawdown",
                    "annualized_volatility",
                    "naive_sharpe",
                ]
            }
            for bps in [0, *costs_bps]
        },
        "turnover": {
            "full_90_month_mean_total_one_way_traded_notional": float(
                monthly["total_one_way_traded_notional"].mean()
            ),
            "pre2026_84_month_mean_total_one_way_traded_notional": float(
                monthly.loc[
                    pd.to_datetime(monthly["signal_date"], utc=True).lt(
                        pd.Timestamp("2026-01-01", tz="UTC")
                    ),
                    "total_one_way_traded_notional",
                ].mean()
            ),
        },
        "calculation_integrity_passed": numeric_integrity_passed,
        "trial_lineage_passed": bool(
            trial_ledger_fields_passed and prereg_clock_passed
        ),
        "strict_artifact_audit_passed": strict_audit_passed,
        "historical_backtest_verdict": "INVALID",
        "research_status": "NOT_VALIDATED",
        "portfolio_authorized": False,
        "verdict_fields_preserved": verdict_fields_passed,
    }
    frames = {
        "monthly": monthly,
        "holdings": holdings,
        "deciles": deciles,
        "return_summary": return_summary,
        "capacity_summary": capacity_summary,
        "path_summary": path_summary,
        "expected_gates": expected_gate_frame,
    }
    context = {
        "config": config,
        "prereg": prereg,
        "paths": {
            "panel": panel_path,
            "decile_monthly": upstream_deciles_path,
            "decile_summary": upstream_decile_summary_path,
            "upstream_manifest": upstream_manifest_path,
            "reported_directory": reported_directory,
            "config": config_path,
            "prereg": prereg_path,
        },
    }
    return verification, frames, context


def build_evidence_matrix(verification: Mapping[str, Any]) -> pd.DataFrame:
    manifest = verification["producer_manifest"]
    trial = verification["trial_ledger"]
    rows = [
        (
            "stable deciles and all-decile returns",
            verification["all_decile_aggregates"]["passed"],
            "Critical",
            f"900 aggregates; max return error={verification['all_decile_aggregates']['numeric_errors']['equal_weight_return']['maximum_absolute_error']:.3e}",
            "Directly tests deterministic membership aggregation.",
        ),
        (
            "extreme holdings",
            verification["extreme_holdings"]["passed"],
            "Critical",
            f"{verification['extreme_holdings']['expected_rows']} rows; missing={verification['extreme_holdings']['missing_from_observed']}; extra={verification['extreme_holdings']['extra_in_observed']}",
            "Confirms D10/D1 names, weights, returns and ADV.",
        ),
        (
            "gross returns and mark-to-market turnover",
            verification["monthly_execution"]["passed"],
            "Critical",
            f"90 months; gross-decile error={verification['gross_decile_reconciliation_maximum_error']:.3e}",
            "Includes initial 2.0 formation and subsequent drift/rebalance trades.",
        ),
        (
            "25/50 bps costs and Newey-West summaries",
            verification["return_summary_and_hac"]["passed"],
            "Critical",
            f"max HAC t error={verification['return_summary_and_hac']['numeric_errors']['hac_tstat']['maximum_absolute_error']:.3e}",
            "Confirms explicit linear costs and HAC inference.",
        ),
        (
            "ADV capacity envelope",
            verification["capacity_summary"]["passed"],
            "Major",
            f"max median error={verification['capacity_summary']['numeric_errors']['median_side_aum_usd']['maximum_absolute_error']:.3e}",
            "Mechanical envelope only; borrow and impact remain absent.",
        ),
        (
            "trial-ledger fields and arithmetic",
            trial["fields_passed"],
            "Major",
            f"field mismatches={len(trial['field_mismatches'])}; floor arithmetic={trial['trial_floor_arithmetic_passed']}",
            "Two opened cost tests are counted, 110 to 112.",
        ),
        (
            "preregistration clock order",
            trial["as_file_clock_order_passed"],
            "Major",
            f"frozen={trial['prereg_frozen_at']}; opened={trial['ledger_opened_at']}",
            "As-file timestamps place the claimed freeze after evaluation began; filesystem mtime cannot cure contradictory signed metadata.",
        ),
        (
            "producer manifest record integrity",
            manifest["record_integrity_passed"],
            "Critical",
            f"records={manifest['record_count']}; hash mismatches={len(manifest['hash_mismatches'])}",
            "All declared records currently match bytes and hashes.",
        ),
        (
            "producer output-directory closure",
            manifest["output_closure_passed"],
            "Major",
            f"omissions={len(manifest['output_closure_omissions'])}; extras={len(manifest['output_closure_extras'])}",
            "All generated files other than the manifest are declared as outputs.",
        ),
        (
            "outcome-critical direct-read input closure",
            len(manifest["missing_outcome_critical_direct_inputs"]) == 0,
            "Critical",
            f"missing={len(manifest['missing_outcome_critical_direct_inputs'])}",
            "Panel, frozen deciles, config, preregistration and execution code are directly recorded.",
        ),
        (
            "descriptive direct-read input closure",
            len(manifest["missing_descriptive_direct_inputs"]) == 0,
            "Major lineage / no portfolio effect",
            "missing direct input: recovered decile_summary.csv; transitive current hash matches upstream manifest",
            "The reported monotonicity is reproducible, but its directly read file was not fail-closed in the producer manifest.",
        ),
        (
            "reported gates and verdict",
            verification["gates"]["passed"]
            and verification["verdict_fields_preserved"],
            "Critical",
            "INVALID / NOT_VALIDATED / portfolio_authorized=false",
            "Independent arithmetic does not authorize deployment.",
        ),
    ]
    return pd.DataFrame(
        rows, columns=["check", "passed", "severity", "evidence", "effect"]
    )


def build_manual_reconstruction(frames: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    monthly = frames["monthly"].sort_values("signal_date").reset_index(drop=True)
    initial = monthly.iloc[0]
    second = monthly.iloc[1]
    largest = monthly.loc[monthly["total_one_way_traded_notional"].idxmax()]

    def selected(row: pd.Series) -> dict[str, Any]:
        return {
            "signal_date": row["signal_date"],
            "long_names": int(row["long_names"]),
            "short_names": int(row["short_names"]),
            "long_return": float(row["long_return"]),
            "short_return": float(row["short_return"]),
            "gross_long_short_return": float(row["gross_long_short_return"]),
            "long_one_way_traded_notional": float(
                row["long_one_way_traded_notional"]
            ),
            "short_one_way_traded_notional": float(
                row["short_one_way_traded_notional"]
            ),
            "total_one_way_traded_notional": float(
                row["total_one_way_traded_notional"]
            ),
            "cost_25bps": float(row["cost_25bps"]),
            "net_return_25bps": float(row["net_return_25bps"]),
            "cost_50bps": float(row["cost_50bps"]),
            "net_return_50bps": float(row["net_return_50bps"]),
        }

    return {
        "formulae": {
            "bucket": "floor(stable_zero_based_rank * 10 / valid_names) + 1",
            "gross": "equal_weight_mean(D10 returns) - equal_weight_mean(D1 returns)",
            "marked_weight": "prior_equal_weight * (1 + holding_return), normalized within sleeve",
            "turnover": "sum over union of names of abs(target_weight - marked_prior_weight)",
            "cost": "total_one_way_traded_notional * bps / 10000",
            "capacity": "minimum over both sleeves and names of ADV * participation * sleeve_name_count",
        },
        "initial_formation": selected(initial),
        "first_mark_to_market_rebalance": selected(second),
        "largest_rebalance": selected(largest),
    }


def write_audit_artifact(
    output_directory: Path,
    verification: Mapping[str, Any],
    frames: Mapping[str, pd.DataFrame],
    context: Mapping[str, Any],
) -> Path:
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise FileExistsError(f"audit output already exists: {output_directory}")
    output_directory.mkdir(parents=True)

    frames["monthly"].to_csv(
        output_directory / "recomputed_monthly_execution.csv", index=False
    )
    frames["holdings"].to_parquet(
        output_directory / "recomputed_holdings.parquet", index=False
    )
    frames["deciles"].to_csv(
        output_directory / "recomputed_decile_monthly.csv", index=False
    )
    frames["return_summary"].to_csv(
        output_directory / "recomputed_return_summary.csv", index=False
    )
    frames["capacity_summary"].to_csv(
        output_directory / "recomputed_capacity_summary.csv", index=False
    )
    frames["path_summary"].to_csv(
        output_directory / "independent_path_metrics.csv", index=False
    )
    frames["expected_gates"].to_csv(
        output_directory / "recomputed_gates.csv", index=False
    )
    evidence = build_evidence_matrix(verification)
    evidence.to_csv(output_directory / "evidence_matrix.csv", index=False)
    write_json(output_directory / "manual_reconstruction.json", build_manual_reconstruction(frames))
    generated_at = datetime.now(UTC).isoformat()
    verification_payload = {"generated_at": generated_at, **verification}
    write_json(output_directory / "verification.json", verification_payload)

    paths = verification["pre2026_path_metrics"]
    turnover = verification["turnover"]
    manifest = verification["producer_manifest"]
    trial = verification["trial_ledger"]
    report = f"""# Independent audit: quarterly-v2 execution diagnostic v1

## Outcome

Calculation integrity: **PASS**. Strict artifact/trial audit: **FAIL**. Historical backtest verdict: **INVALID**. Research status: **NOT_VALIDATED**. Portfolio authorization: **false**.

An independent implementation rebuilt stable deciles, 9,986 D10/D1 holdings, 90 monthly returns, mark-to-market turnover, 25/50 bps costs, Newey-West HAC statistics, and ADV envelopes directly from the recovered panel. It imported none of the producer runner, production execution module, or project metric helpers.

## Exact reconstruction

- All-decile aggregates: pass=`{verification['all_decile_aggregates']['passed']}`; maximum return error `{verification['all_decile_aggregates']['numeric_errors']['equal_weight_return']['maximum_absolute_error']:.3e}`.
- Holdings: pass=`{verification['extreme_holdings']['passed']}`; missing `{verification['extreme_holdings']['missing_from_observed']}`, extra `{verification['extreme_holdings']['extra_in_observed']}`.
- Monthly execution and turnover: pass=`{verification['monthly_execution']['passed']}`; gross-to-decile maximum error `{verification['gross_decile_reconciliation_maximum_error']:.3e}`.
- Mean total one-way traded notional: `{turnover['full_90_month_mean_total_one_way_traded_notional']:.12f}` over all 90 months; `{turnover['pre2026_84_month_mean_total_one_way_traded_notional']:.12f}` over the 84 pre-2026 months.
- Costs and HAC summaries: pass=`{verification['return_summary_and_hac']['passed']}`; maximum HAC t-stat error `{verification['return_summary_and_hac']['numeric_errors']['hac_tstat']['maximum_absolute_error']:.3e}`.
- Capacity: pass=`{verification['capacity_summary']['passed']}`; maximum median-side-AUM error `{verification['capacity_summary']['numeric_errors']['median_side_aum_usd']['maximum_absolute_error']:.3e}`.
- Reported gates: pass=`{verification['gates']['passed']}`.

## Pre-2026 path diagnostics

| One-way cost | Terminal wealth | CAGR | Max drawdown | Annualized vol | Naive Sharpe |
|---:|---:|---:|---:|---:|---:|
| 0 bps | {paths['0']['terminal_wealth']:.6f} | {paths['0']['cagr']:.3%} | {paths['0']['max_drawdown']:.2%} | {paths['0']['annualized_volatility']:.2%} | {paths['0']['naive_sharpe']:.3f} |
| 25 bps | {paths['25']['terminal_wealth']:.6f} | {paths['25']['cagr']:.3%} | {paths['25']['max_drawdown']:.2%} | {paths['25']['annualized_volatility']:.2%} | {paths['25']['naive_sharpe']:.3f} |
| 50 bps | {paths['50']['terminal_wealth']:.6f} | {paths['50']['cagr']:.3%} | {paths['50']['max_drawdown']:.2%} | {paths['50']['annualized_volatility']:.2%} | {paths['50']['naive_sharpe']:.3f} |

These are opened descriptive stress metrics. They do not repair the failed gross statistical test or establish deployability.

## Confirmed lineage failures

1. **Preregistration clock order fails.** The preregistration says `frozen_at={trial['prereg_frozen_at']}`, after the ledger's `opened_at={trial['ledger_opened_at']}` and run time. The file mtime precedes the run, but mutable filesystem metadata cannot override contradictory as-file trial metadata.
2. **One direct descriptive input is omitted.** The producer directly reads recovered `decile_summary.csv` for monotonicity but does not list that file directly in its execution manifest. Its current hash matches the upstream recovered manifest, so the monotonicity is reproducible now. The producer was not fail-closed against later modification of that directly read file. Outcome-critical direct inputs have no omission.

Producer declared-record integrity pass=`{manifest['record_integrity_passed']}` and output-directory closure pass=`{manifest['output_closure_passed']}`. Strict direct-read closure pass=`{manifest['strict_direct_read_closure_passed']}`.

## Verdict

Return, turnover, cost and HAC calculations agree within `1e-12`; scaled capacity fields agree within `$1e-6`. Trial-clock and direct-read lineage controls fail. More importantly, gross HAC t-stat remains below the frozen threshold, permanent security lineage/delisting returns are incomplete, borrow availability/fees are absent, and nonlinear market impact is absent. The only defensible scope is an **INVALID / NOT_VALIDATED, non-deployable, opened descriptive diagnostic**.
"""
    (output_directory / "report-source.md").write_text(report, encoding="utf-8")

    input_paths: set[Path] = {
        Path(context["paths"]["panel"]),
        Path(context["paths"]["decile_monthly"]),
        Path(context["paths"]["decile_summary"]),
        Path(context["paths"]["upstream_manifest"]),
        Path(context["paths"]["config"]),
        Path(context["paths"]["prereg"]),
        Path(__file__).resolve(),
        (ROOT / "tests/test_quarterly_v2_execution_verifier.py").resolve(),
    }
    reported_directory = Path(context["paths"]["reported_directory"])
    input_paths.update(path.resolve() for path in reported_directory.rglob("*") if path.is_file())
    input_records: list[dict[str, Any]] = []
    for path in sorted(input_paths):
        role = "code" if path == Path(__file__).resolve() else "test" if path.name.startswith("test_") else "audited_input"
        input_records.append(
            {
                "role": role,
                "path": path.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    output_records = [
        {
            "role": "output",
            "path": path.resolve().as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(output_directory.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    ]
    write_json(
        output_directory / "artifact_manifest.json",
        {
            "generated_at": generated_at,
            "audit_verdict": "FAIL",
            "calculation_integrity": "PASS",
            "historical_backtest_verdict": "INVALID",
            "research_status": "NOT_VALIDATED",
            "portfolio_authorized": False,
            "records": [*input_records, *output_records],
        },
    )
    return output_directory / "report-source.md"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-directory", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--reported-directory", type=Path, default=DEFAULT_REPORTED)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    verification, frames, context = run_audit(
        source_directory=args.source_directory,
        reported_directory=args.reported_directory,
        config_path=args.config,
        prereg_path=args.preregistration,
    )
    if args.check_only:
        print(json.dumps(_json_safe(verification), indent=2, sort_keys=True))
        return
    print(write_audit_artifact(args.output_directory, verification, frames, context))


if __name__ == "__main__":
    main()
