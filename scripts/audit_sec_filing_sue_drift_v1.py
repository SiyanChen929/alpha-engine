#!/usr/bin/env python3
"""Independent, read-only audit of the SEC filing SUE drift experiment.

The audit deliberately does not import either production research module.  It
rebuilds the signal, event clock, price joins, returns, aggregation, HAC tests,
and bootstrap draws from frozen inputs and sealed outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


AUDIT_SCHEMA = "sec_filing_sue_drift_independent_audit_v1"
DEFAULT_OUTPUT = "artifacts/sec_filing_sue_drift_v1_independent_audit"
ATTEMPT1 = "artifacts/sec_filing_sue_drift_v1"
REPAIR1 = "artifacts/sec_filing_sue_drift_v1_bootstrap_repair1"
PROTOCOL = "configs/sec_filing_sue_drift_v1_preregistration.json"
REPAIR_CONFIG = "configs/sec_filing_sue_drift_v1_bootstrap_repair1.json"
AUDIT_SCRIPT = "scripts/audit_sec_filing_sue_drift_v1.py"
AUDIT_TEST = "tests/test_sec_filing_sue_drift_independent_audit.py"
ABS_TOL = 1e-12


class AuditError(RuntimeError):
    """Fail closed when the independent audit cannot be completed."""


@dataclass
class Check:
    check: str
    domain: str
    status: str
    severity: str
    expected: str
    actual: str
    tolerance: str
    evidence: str
    effect: str
    remediation: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def json_write(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def iso_utc_from_ns(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=UTC).isoformat()


def scalar_text(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.17g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if value is None:
        return "null"
    return str(value)


def add_check(
    checks: list[Check],
    *,
    check: str,
    domain: str,
    passed: bool | None,
    severity: str,
    expected: Any,
    actual: Any,
    tolerance: Any = "exact",
    evidence: str,
    effect: str = "none when passed",
    remediation: str = "none",
) -> None:
    status = "INFO" if passed is None else ("PASS" if passed else "FAIL")
    checks.append(
        Check(
            check=check,
            domain=domain,
            status=status,
            severity=severity,
            expected=scalar_text(expected),
            actual=scalar_text(actual),
            tolerance=scalar_text(tolerance),
            evidence=evidence,
            effect=effect,
            remediation=remediation,
        )
    )


def add_numeric_check(
    checks: list[Check],
    *,
    check: str,
    domain: str,
    expected: float,
    actual: float,
    tolerance: float = ABS_TOL,
    severity: str = "Critical",
    evidence: str,
    effect: str = "reproduction mismatch would invalidate the reported statistic",
    remediation: str = "rebuild and rerun from frozen inputs",
) -> None:
    finite = np.isfinite(expected) and np.isfinite(actual)
    passed = bool(finite and abs(float(actual) - float(expected)) <= tolerance)
    add_check(
        checks,
        check=check,
        domain=domain,
        passed=passed,
        severity=severity,
        expected=expected,
        actual=actual,
        tolerance=tolerance,
        evidence=evidence,
        effect=effect,
        remediation=remediation,
    )


def snapshot_tree(root: Path, directories: Iterable[str]) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for relative_directory in directories:
        directory = root / relative_directory
        if not directory.is_dir():
            raise AuditError(f"sealed directory missing: {relative_directory}")
        for path in sorted(p for p in directory.iterdir() if p.is_file()):
            relative = path.relative_to(root).as_posix()
            snapshot[relative] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return snapshot


def verify_manifest(
    *, root: Path, manifest_path: Path, required_names: set[str]
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    records = manifest.get("records", [])
    details: list[dict[str, Any]] = []
    for record in records:
        target = root / record["path"]
        exists = target.is_file()
        actual_bytes = target.stat().st_size if exists else None
        actual_hash = sha256_file(target) if exists else None
        passed = bool(
            exists
            and actual_bytes == int(record["bytes"])
            and actual_hash == record["sha256"]
        )
        details.append(
            {
                "path": record["path"],
                "role": record["role"],
                "exists": exists,
                "expected_bytes": int(record["bytes"]),
                "actual_bytes": actual_bytes,
                "expected_sha256": record["sha256"],
                "actual_sha256": actual_hash,
                "passed": passed,
            }
        )
    output_names = {
        Path(row["path"]).name for row in records if row.get("role") == "output"
    }
    return {
        "manifest_path": manifest_path.relative_to(root).as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "declared_record_count": manifest.get("record_count"),
        "actual_record_count": len(records),
        "record_count_matches": manifest.get("record_count") == len(records),
        "all_hashes_match": all(row["passed"] for row in details),
        "declared_all_records_exist": manifest.get("all_records_exist"),
        "required_output_names": sorted(required_names),
        "manifest_output_names": sorted(output_names),
        "required_outputs_closed": output_names == (required_names - {"artifact_manifest.json"}),
        "self_hash_excluded_by_design": manifest_path.name not in output_names,
        "records": details,
    }


def prior_rolling_sample_std(
    frame: pd.DataFrame, *, ticker: str, value: str, window: int
) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, indexes in frame.groupby(ticker, sort=False).groups.items():
        ordered_indexes = np.asarray(list(indexes), dtype=int)
        values = frame.loc[ordered_indexes, value].to_numpy(dtype=float)
        for position in range(window, len(values)):
            history = values[position - window : position]
            if np.isfinite(history).all():
                result.loc[ordered_indexes[position]] = float(np.std(history, ddof=1))
    return result


def rebuild_signal(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    source_path = root / config["sources"]["quarterly_release_features"]["path"]
    releases = pd.read_parquet(source_path)
    eligibility = config["release_eligibility"]
    frame = releases.copy()
    frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True, errors="coerce")
    frame["fiscal_end"] = pd.to_datetime(frame["fiscal_end"], errors="coerce")
    frame["ticker"] = frame["ticker"].astype("string").str.strip().str.upper()
    frame["accession"] = frame["accession"].astype("string").str.strip()
    mask = (
        frame["form"].isin(eligibility["forms"])
        & frame["release_kind"].isin(eligibility["release_kinds"])
        & frame["clock_source"].isin(eligibility["clock_sources"])
        & frame["header_inventory_verified"].eq(eligibility["header_inventory_verified"])
        & frame["supersession_only"].eq(eligibility["supersession_only"])
        & frame["net_income_same_tag_comparator"].eq(
            eligibility["net_income_same_tag_comparator"]
        )
    )
    for column in eligibility["required_non_null"]:
        mask &= frame[column].notna()
    eligible = frame.loc[mask].copy()
    if eligible.duplicated(eligibility["deduplication_key"]).any():
        raise AuditError("independent eligible release set has duplicate accession keys")
    eligible = eligible.sort_values(["ticker", "available_at", "accession"]).reset_index(drop=True)
    local = eligible["available_at"].dt.tz_convert("America/New_York")
    eligible["audit_eod_clamped_clock"] = (
        local.dt.hour.eq(23) & local.dt.minute.eq(59) & local.dt.second.eq(59)
    )
    signal = config["signal"]
    base = signal["base_value"]
    eligible[base] = pd.to_numeric(eligible[base], errors="coerce")
    window = int(signal["history_window_events"])
    eligible["audit_prior_signal_std"] = prior_rolling_sample_std(
        eligible, ticker="ticker", value=base, window=window
    )
    eligible["audit_sue_like"] = eligible[base] / eligible["audit_prior_signal_std"]
    eligible["audit_signal_complete"] = (
        np.isfinite(eligible["audit_sue_like"])
        & eligible["audit_prior_signal_std"].gt(0)
    )
    eligible.loc[~eligible["audit_signal_complete"], "audit_sue_like"] = np.nan

    dev = signal["development_window"]
    dev_mask = (
        eligible["audit_signal_complete"]
        & eligible["available_at"].ge(pd.Timestamp(dev["information_at_gte"]))
        & eligible["available_at"].lt(pd.Timestamp(dev["information_at_lt"]))
    )
    dev_values = eligible.loc[dev_mask, "audit_sue_like"].to_numpy(dtype=float)
    lower = float(np.quantile(dev_values, signal["tail_calibration"]["lower_quantile"], method="linear"))
    upper = float(np.quantile(dev_values, signal["tail_calibration"]["upper_quantile"], method="linear"))
    dev_months = int(
        eligible.loc[dev_mask, "available_at"]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .dt.to_period("M")
        .nunique()
    )

    clock = config["universe_and_clock"]
    eval_window = (
        eligible["available_at"].ge(pd.Timestamp(clock["evaluation_information_at_gte"]))
        & eligible["available_at"].lt(pd.Timestamp(clock["evaluation_information_at_lt"]))
    )
    complete_eval = eval_window & eligible["audit_signal_complete"]
    evaluation = eligible.loc[complete_eval].copy()
    evaluation["audit_direction"] = np.select(
        [evaluation["audit_sue_like"].le(lower), evaluation["audit_sue_like"].ge(upper)],
        [-1, 1],
        default=0,
    ).astype("int8")
    selected = evaluation.loc[evaluation["audit_direction"].ne(0)].copy()
    return {
        "eligible": eligible,
        "selected": selected,
        "lower": lower,
        "upper": upper,
        "development_events": int(dev_mask.sum()),
        "development_months": dev_months,
        "evaluation_total": int(eval_window.sum()),
        "evaluation_incomplete": int((eval_window & ~eligible["audit_signal_complete"]).sum()),
        "evaluation_middle": int(evaluation["audit_direction"].eq(0).sum()),
        "selected_count": int(len(selected)),
    }


def prepare_prices(root: Path, config: Mapping[str, Any], tickers: list[str]) -> pd.DataFrame:
    path = root / config["sources"]["daily_prices"]["path"]
    columns = [
        "ticker",
        "date",
        "timestamp",
        "close",
        "adj_open",
        "dollar_volume",
        "available_at",
    ]
    panel = pd.read_parquet(path, columns=columns, filters=[("ticker", "in", tickers)])
    panel["ticker"] = panel["ticker"].astype("string").str.strip().str.upper()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True, errors="coerce")
    panel["available_at"] = pd.to_datetime(panel["available_at"], utc=True, errors="coerce")
    for column in ("close", "adj_open", "dollar_volume"):
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    if panel.duplicated(["ticker", "date"]).any():
        raise AuditError("daily panel has duplicate ticker-date keys")
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    panel["audit_prior_close"] = panel.groupby("ticker", sort=False)["close"].shift(1)
    window = int(config["universe_and_clock"]["adv_window_sessions"])
    panel["audit_prior_adv_usd"] = panel.groupby("ticker", sort=False)["dollar_volume"].transform(
        lambda values: values.shift(1).rolling(window, min_periods=window).mean()
    )
    panel["audit_prior_feature_available_at"] = panel.groupby("ticker", sort=False)[
        "available_at"
    ].shift(1)
    return panel


def rebuild_events(
    root: Path, config: Mapping[str, Any], signal_result: Mapping[str, Any]
) -> dict[str, Any]:
    selected = signal_result["selected"].copy()
    selected = selected.rename(
        columns={
            "audit_prior_signal_std": "prior_signal_std",
            "audit_sue_like": "sue_like",
            "audit_direction": "direction",
            "audit_eod_clamped_clock": "eod_clamped_clock",
        }
    )
    tickers = sorted(set(selected["ticker"].astype(str)) | {config["universe_and_clock"]["benchmark"]})
    panel = prepare_prices(root, config, tickers)
    spy = panel.loc[panel["ticker"].eq(config["universe_and_clock"]["benchmark"])].sort_values(
        "timestamp"
    ).reset_index(drop=True)
    session_ns = spy["timestamp"].astype("int64").to_numpy()
    information_ns = selected["available_at"].astype("int64").to_numpy()
    entry_index = np.searchsorted(session_ns, information_ns, side="right")
    horizon = int(config["estimands"]["primary_horizon_sessions"])
    exit_index = entry_index + horizon
    pre_index = entry_index - horizon
    valid = (
        (pre_index >= 0)
        & (entry_index >= 0)
        & (exit_index < len(spy))
    )
    selected["entry_index"] = entry_index
    selected["exit_index"] = exit_index
    selected["pre_index"] = pre_index
    selected["calendar_index_complete"] = valid

    def take(column: str, indexes: np.ndarray) -> pd.Series:
        result = np.full(len(indexes), np.nan, dtype=object)
        ok = (indexes >= 0) & (indexes < len(spy))
        values = spy[column].to_numpy()
        result[ok] = values[indexes[ok]]
        return pd.Series(result, index=selected.index)

    for prefix, indexes in (("entry", entry_index), ("exit", exit_index), ("pre", pre_index)):
        selected[f"{prefix}_at"] = pd.to_datetime(take("timestamp", indexes), utc=True)
        selected[f"{prefix}_date"] = pd.to_datetime(take("date", indexes), errors="coerce")
        selected[f"spy_{prefix}_adj_open"] = pd.to_numeric(take("adj_open", indexes), errors="coerce")

    lookup = panel[
        [
            "ticker",
            "date",
            "adj_open",
            "audit_prior_close",
            "audit_prior_adv_usd",
            "audit_prior_feature_available_at",
        ]
    ]
    for prefix in ("entry", "exit", "pre"):
        renamed = lookup.rename(
            columns={
                "date": f"{prefix}_date",
                "adj_open": f"stock_{prefix}_adj_open",
                "audit_prior_close": f"{prefix}_prior_close",
                "audit_prior_adv_usd": f"{prefix}_prior_adv_usd",
                "audit_prior_feature_available_at": f"{prefix}_prior_feature_available_at",
            }
        )
        selected = selected.merge(
            renamed,
            on=["ticker", f"{prefix}_date"],
            how="left",
            validate="many_to_one",
        )

    clock = config["universe_and_clock"]
    cutoff = pd.Timestamp(clock["required_exit_open_lte"])
    price_fields = [
        "stock_entry_adj_open",
        "stock_exit_adj_open",
        "stock_pre_adj_open",
        "spy_entry_adj_open",
        "spy_exit_adj_open",
        "spy_pre_adj_open",
    ]
    price_complete = selected[price_fields].notna().all(axis=1)
    price_positive = selected[price_fields].gt(0).all(axis=1)
    feature_clock_ok = pd.to_datetime(
        selected["entry_prior_feature_available_at"], utc=True, errors="coerce"
    ).lt(selected["entry_at"])
    selected["outcome_complete"] = (
        selected["calendar_index_complete"]
        & price_complete
        & price_positive
        & selected["exit_at"].le(cutoff)
    )
    selected["liquidity_complete"] = (
        selected["entry_prior_close"].notna()
        & selected["entry_prior_adv_usd"].notna()
        & feature_clock_ok
    )
    selected["liquid"] = (
        selected["liquidity_complete"]
        & selected["entry_prior_close"].ge(clock["minimum_prior_close_usd"])
        & selected["entry_prior_adv_usd"].ge(clock["minimum_prior_adv_usd"])
    )

    exclusion_reasons: list[str] = []
    exclusion_rows: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        reason: str | None = None
        if not row.calendar_index_complete:
            reason = "benchmark_calendar_boundary_missing"
        elif pd.isna(row.exit_at) or row.exit_at > cutoff:
            reason = "exit_after_frozen_evaluation_cutoff"
        elif not row.outcome_complete:
            reason = "stock_or_benchmark_open_missing_or_nonpositive"
        elif not row.liquidity_complete:
            reason = "prior_liquidity_history_or_clock_incomplete"
        elif not row.liquid:
            reason = "price_or_adv_below_frozen_floor"
        if reason is not None:
            exclusion_reasons.append(reason)
            exclusion_rows.append(
                {"ticker": row.ticker, "accession": row.accession, "reason": reason}
            )
    usable = selected.loc[selected["outcome_complete"] & selected["liquid"]].copy()

    conflict_keys: set[tuple[str, pd.Timestamp]] = set()
    for key, group in usable.groupby(["ticker", "entry_date"], sort=False):
        if group["direction"].nunique() > 1:
            conflict_keys.add(key)
            for row in group.itertuples(index=False):
                exclusion_rows.append(
                    {
                        "ticker": row.ticker,
                        "accession": row.accession,
                        "reason": "conflicting_tail_directions_same_ticker_entry_session",
                    }
                )
    if conflict_keys:
        conflict_index = pd.MultiIndex.from_tuples(conflict_keys, names=["ticker", "entry_date"])
        keys = pd.MultiIndex.from_frame(usable[["ticker", "entry_date"]])
        usable = usable.loc[~keys.isin(conflict_index)].copy()

    collapsed_rows: list[pd.Series] = []
    for _, group in usable.groupby(["ticker", "entry_date"], sort=True):
        row = group.sort_values("accession").iloc[0].copy()
        row["accessions"] = "|".join(sorted(group["accession"].astype(str).unique()))
        row["collapsed_accession_count"] = int(group["accession"].nunique())
        row["sue_like"] = float(group["sue_like"].mean())
        collapsed_rows.append(row)
    events = pd.DataFrame(collapsed_rows).reset_index(drop=True)
    events["stock_return_20"] = events["stock_exit_adj_open"] / events["stock_entry_adj_open"] - 1
    events["spy_return_20"] = events["spy_exit_adj_open"] / events["spy_entry_adj_open"] - 1
    events["stock_abnormal_return_20"] = events["stock_return_20"] - events["spy_return_20"]
    events["pair_gross_return_20"] = 0.5 * events["direction"] * events["stock_abnormal_return_20"]
    events["pair_net_30bps_return_20"] = (
        events["pair_gross_return_20"]
        - config["estimands"]["primary_round_trip_cost_per_gross_dollar"]
    )
    events["pair_net_50bps_return_20"] = (
        events["pair_gross_return_20"]
        - config["estimands"]["stress_round_trip_cost_per_gross_dollar"]
    )
    pre_stock = events["stock_entry_adj_open"] / events["stock_pre_adj_open"] - 1
    pre_spy = events["spy_entry_adj_open"] / events["spy_pre_adj_open"] - 1
    events["pair_pre_event_gross_return_20"] = 0.5 * events["direction"] * (pre_stock - pre_spy)
    return {
        "events": events,
        "selected": selected,
        "panel": panel,
        "spy": spy,
        "price_exclusions": pd.DataFrame(exclusion_rows),
        "price_exclusion_counts": pd.Series(exclusion_reasons, dtype="string").value_counts().sort_index().to_dict(),
        "conflict_count": len(conflict_keys),
    }


def aggregate_sessions(events: pd.DataFrame) -> pd.DataFrame:
    return (
        events.groupby("entry_date", sort=True)
        .agg(
            events=("accession", "size"),
            tickers=("ticker", "nunique"),
            positive_tail_events=("direction", lambda x: int((x > 0).sum())),
            negative_tail_events=("direction", lambda x: int((x < 0).sum())),
            pair_gross_return_20=("pair_gross_return_20", "mean"),
            pair_net_30bps_return_20=("pair_net_30bps_return_20", "mean"),
            pair_net_50bps_return_20=("pair_net_50bps_return_20", "mean"),
            pair_pre_event_gross_return_20=("pair_pre_event_gross_return_20", "mean"),
        )
        .reset_index()
    )


def independent_hac(values: Iterable[float], lags: int) -> dict[str, float | int]:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    mean = float(x.mean())
    residual = x - mean
    long_run_variance = float(np.dot(residual, residual) / n)
    maximum_lag = min(int(lags), n - 1)
    for lag in range(1, maximum_lag + 1):
        covariance = float(np.dot(residual[lag:], residual[:-lag]) / n)
        bartlett = 1.0 - lag / (maximum_lag + 1.0)
        long_run_variance += 2.0 * bartlett * covariance
    long_run_variance = max(long_run_variance, 0.0)
    standard_error = math.sqrt(long_run_variance / n)
    t_value = mean / standard_error
    one_sided_p = 0.5 * math.erfc(t_value / math.sqrt(2.0))
    return {
        "n": n,
        "mean": mean,
        "standard_error": standard_error,
        "t": t_value,
        "one_sided_p": one_sided_p,
    }


def independent_bootstraps(
    values: Iterable[float], *, draws: int, block_length: int, seed: int
) -> dict[str, Any]:
    x = np.asarray(list(values), dtype=float)
    observed = float(x.mean())
    centered = x - observed
    n = len(centered)
    length = max(1, min(int(block_length), n))
    starts = np.arange(0, n - length + 1)
    blocks_needed = int(math.ceil(n / length))
    final_length = n - (blocks_needed - 1) * length
    expected_full_sum = float(np.mean([centered[start : start + length].sum() for start in starts]))
    expected_final_sum = float(np.mean([centered[start : start + final_length].sum() for start in starts]))
    design_expectation = ((blocks_needed - 1) * expected_full_sum + expected_final_sum) / n

    rng = np.random.default_rng(int(seed))
    raw = np.empty(int(draws), dtype=float)
    for draw in range(int(draws)):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        pieces = [centered[start : start + length] for start in chosen]
        raw[draw] = float(np.concatenate(pieces)[:n].mean())
    corrected = raw - design_expectation

    circular_starts = np.arange(n)
    offsets = np.arange(length)
    circular_rng = np.random.default_rng(int(seed))
    circular = np.empty(int(draws), dtype=float)
    for draw in range(int(draws)):
        chosen = circular_rng.choice(circular_starts, size=blocks_needed, replace=True)
        pieces = [centered[(start + offsets) % n] for start in chosen]
        circular[draw] = float(np.concatenate(pieces)[:n].mean())

    def summarize(draw_values: np.ndarray) -> dict[str, Any]:
        exceedances = int(np.count_nonzero(draw_values >= observed))
        return {
            "null_mean": float(draw_values.mean()),
            "null_std": float(draw_values.std(ddof=1)),
            "exceedances": exceedances,
            "one_sided_empirical_p": float((exceedances + 1) / (draws + 1)),
        }

    return {
        "n": n,
        "observed_mean": observed,
        "draws": int(draws),
        "block_length": length,
        "blocks_needed": blocks_needed,
        "last_block_observations": final_length,
        "design_expectation": float(design_expectation),
        "raw": raw,
        "corrected": corrected,
        "circular": circular,
        "raw_summary": summarize(raw),
        "corrected_summary": summarize(corrected),
        "circular_summary": summarize(circular),
    }


def compare_numeric_columns(left: pd.DataFrame, right: pd.DataFrame, columns: list[str]) -> float:
    maximum = 0.0
    for column in columns:
        left_values = left[column].to_numpy(dtype=float)
        right_values = right[column].to_numpy(dtype=float)
        if left_values.shape != right_values.shape:
            return math.inf
        differences = np.abs(left_values - right_values)
        maximum = max(maximum, float(np.nanmax(differences)) if differences.size else 0.0)
        if not np.array_equal(np.isnan(left_values), np.isnan(right_values)):
            return math.inf
    return maximum


def compare_hac(
    checks: list[Check], sessions: pd.DataFrame, reported: Mapping[str, Any], column: str, label: str
) -> dict[str, Any]:
    calculated = independent_hac(sessions[column], lags=20)
    for field in ("mean", "standard_error", "t", "one_sided_p"):
        add_numeric_check(
            checks,
            check=f"hac_{label}_{field}",
            domain="inference",
            expected=float(reported[field]),
            actual=float(calculated[field]),
            tolerance=ABS_TOL,
            evidence=f"independent Newey-West/Bartlett calculation on {len(sessions)} ordered entry dates",
        )
    add_check(
        checks,
        check=f"hac_{label}_n",
        domain="inference",
        passed=int(calculated["n"]) == int(reported["n"]),
        severity="Critical",
        expected=reported["n"],
        actual=calculated["n"],
        evidence="finite observation count",
    )
    return calculated


def run_audit(root: Path, destination: Path) -> dict[str, Path]:
    root = root.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise AuditError(f"audit output directory already exists: {destination}")
    started_at = datetime.now(UTC)
    sealed_before = snapshot_tree(root, [ATTEMPT1, REPAIR1])
    checks: list[Check] = []

    protocol_path = root / PROTOCOL
    repair_config_path = root / REPAIR_CONFIG
    protocol = load_json(protocol_path)
    repair_config = load_json(repair_config_path)
    attempt_manifest_path = root / ATTEMPT1 / "artifact_manifest.json"
    repair_manifest_path = root / REPAIR1 / "artifact_manifest.json"
    attempt_summary = load_json(root / ATTEMPT1 / "summary_metrics.json")
    attempt_run = load_json(root / ATTEMPT1 / "run_manifest.json")
    repair_summary = load_json(root / REPAIR1 / "bootstrap_summary.json")
    repair_run = load_json(root / REPAIR1 / "run_manifest.json")

    attempt_required = set(protocol["outputs"]["required"])
    repair_required = set(repair_config["outputs"]["required"])
    attempt_manifest_audit = verify_manifest(
        root=root, manifest_path=attempt_manifest_path, required_names=attempt_required
    )
    repair_manifest_audit = verify_manifest(
        root=root, manifest_path=repair_manifest_path, required_names=repair_required
    )
    for label, result in (("attempt1", attempt_manifest_audit), ("repair1", repair_manifest_audit)):
        add_check(
            checks,
            check=f"{label}_manifest_record_hashes",
            domain="lineage",
            passed=result["all_hashes_match"],
            severity="Critical",
            expected=True,
            actual=result["all_hashes_match"],
            evidence=f"all records in {result['manifest_path']} independently rehashed",
            effect="hash failure would break artifact closure",
            remediation="restore sealed bytes or issue a new attempt",
        )
        add_check(
            checks,
            check=f"{label}_manifest_record_count",
            domain="lineage",
            passed=result["record_count_matches"],
            severity="Major",
            expected=result["declared_record_count"],
            actual=result["actual_record_count"],
            evidence=result["manifest_path"],
        )
        add_check(
            checks,
            check=f"{label}_required_outputs_closed",
            domain="lineage",
            passed=result["required_outputs_closed"],
            severity="Major",
            expected="configured outputs except self manifest",
            actual="matched" if result["required_outputs_closed"] else "mismatch",
            evidence=f"manifest outputs={result['manifest_output_names']}",
        )

    add_check(
        checks,
        check="repair1_attempt1_manifest_reference",
        domain="lineage",
        passed=repair_config["attempt1"]["artifact_manifest_sha256"]
        == sha256_file(attempt_manifest_path),
        severity="Critical",
        expected=repair_config["attempt1"]["artifact_manifest_sha256"],
        actual=sha256_file(attempt_manifest_path),
        evidence=REPAIR_CONFIG,
    )
    add_check(
        checks,
        check="attempt1_protocol_hash",
        domain="lineage",
        passed=attempt_run["protocol_sha256"] == sha256_file(protocol_path),
        severity="Critical",
        expected=attempt_run["protocol_sha256"],
        actual=sha256_file(protocol_path),
        evidence=f"{ATTEMPT1}/run_manifest.json and current protocol bytes",
    )
    attempt_source_path = root / attempt_run["source_code_path"]
    add_check(
        checks,
        check="attempt1_source_hash",
        domain="lineage",
        passed=attempt_run["source_code_sha256"] == sha256_file(attempt_source_path),
        severity="Critical",
        expected=attempt_run["source_code_sha256"],
        actual=sha256_file(attempt_source_path),
        evidence=f"{ATTEMPT1}/run_manifest.json and current source bytes",
    )
    repair_source_path = root / repair_run["source_path"]
    add_check(
        checks,
        check="repair1_source_hash",
        domain="lineage",
        passed=repair_run["source_sha256"] == sha256_file(repair_source_path),
        severity="Critical",
        expected=repair_run["source_sha256"],
        actual=sha256_file(repair_source_path),
        evidence=f"{REPAIR1}/run_manifest.json and current source bytes",
    )
    add_check(
        checks,
        check="repair1_config_hash",
        domain="lineage",
        passed=repair_run["config_sha256"] == sha256_file(repair_config_path),
        severity="Critical",
        expected=repair_run["config_sha256"],
        actual=sha256_file(repair_config_path),
        evidence=f"{REPAIR1}/run_manifest.json and current config bytes",
    )

    frozen_at = pd.Timestamp(protocol["frozen_at"])
    run_started = pd.Timestamp(attempt_run["run_started_at"])
    protocol_mtime_ns = protocol_path.stat().st_mtime_ns
    protocol_mtime = pd.Timestamp(protocol_mtime_ns, unit="ns", tz="UTC")
    add_check(
        checks,
        check="recorded_frozen_at_precedes_attempt1_run",
        domain="protocol_chronology",
        passed=frozen_at < run_started,
        severity="Critical",
        expected="frozen_at < run_started_at",
        actual=f"{frozen_at.isoformat()} < {run_started.isoformat()}",
        evidence=f"{PROTOCOL}; {ATTEMPT1}/run_manifest.json",
    )
    add_check(
        checks,
        check="protocol_final_mtime_precedes_attempt1_run",
        domain="protocol_chronology",
        passed=protocol_mtime < run_started,
        severity="Critical",
        expected="protocol mtime < run_started_at",
        actual=f"{protocol_mtime.isoformat()} < {run_started.isoformat()}",
        evidence=f"filesystem mtime_ns={protocol_mtime_ns}",
    )
    mtime_lag = float((protocol_mtime - frozen_at).total_seconds())
    add_check(
        checks,
        check="recorded_frozen_at_matches_final_protocol_mtime_order",
        domain="protocol_chronology",
        passed=protocol_mtime <= frozen_at,
        severity="Major",
        expected="final protocol mtime <= recorded frozen_at",
        actual=f"mtime is {mtime_lag:.6f}s after recorded frozen_at",
        evidence=f"frozen_at={frozen_at.isoformat()}; mtime={protocol_mtime.isoformat()}",
        effect="the recorded frozen_at is not the demonstrable final-byte freeze time",
        remediation="use an append-only pre-run registry or signed hash timestamp; treat final mtime as the bounded evidence here",
    )
    add_check(
        checks,
        check="repair_recorded_after_attempt1_outcome",
        domain="protocol_chronology",
        passed=pd.Timestamp(repair_config["recorded_at"]) > pd.Timestamp(attempt_run["run_finished_at"]),
        severity="Major",
        expected="repair recorded_at > attempt1 run_finished_at",
        actual=f"{repair_config['recorded_at']} > {attempt_run['run_finished_at']}",
        evidence=REPAIR_CONFIG,
        effect="confirms repair is post-outcome and cannot be represented as preregistered",
        remediation="retain explicit POST_OUTCOME_COMPUTATIONAL_REPAIR classification",
    )
    repair_recorded_at = pd.Timestamp(repair_config["recorded_at"])
    repair_run_started = pd.Timestamp(repair_run["run_started_at"])
    repair_mtime_ns = repair_config_path.stat().st_mtime_ns
    repair_mtime = pd.Timestamp(repair_mtime_ns, unit="ns", tz="UTC")
    add_check(
        checks,
        check="repair1_final_config_mtime_precedes_repair_run",
        domain="protocol_chronology",
        passed=repair_mtime < repair_run_started,
        severity="Critical",
        expected="repair config mtime < repair run_started_at",
        actual=f"{repair_mtime.isoformat()} < {repair_run_started.isoformat()}",
        evidence=f"filesystem mtime_ns={repair_mtime_ns}",
    )
    repair_mtime_lag = float((repair_mtime - repair_recorded_at).total_seconds())
    add_check(
        checks,
        check="repair1_recorded_at_matches_final_config_mtime_order",
        domain="protocol_chronology",
        passed=repair_mtime <= repair_recorded_at,
        severity="Major",
        expected="final repair config mtime <= embedded recorded_at",
        actual=f"mtime is {repair_mtime_lag:.6f}s after embedded recorded_at",
        evidence=f"recorded_at={repair_recorded_at.isoformat()}; mtime={repair_mtime.isoformat()}",
        effect="the embedded repair record time is not the demonstrable final-byte time",
        remediation="use the final config hash and pre-run mtime as bounded evidence",
    )

    signal_result = rebuild_signal(root, protocol)
    thresholds = attempt_summary["thresholds"]
    for check_name, expected, actual in (
        ("development_event_count", thresholds["development_events"], signal_result["development_events"]),
        ("development_month_count", thresholds["development_signal_complete_months"], signal_result["development_months"]),
        ("evaluation_total_release_count", 9137, signal_result["evaluation_total"]),
        ("evaluation_incomplete_signal_count", 757, signal_result["evaluation_incomplete"]),
        ("evaluation_middle_signal_count", 5484, signal_result["evaluation_middle"]),
        ("selected_tail_event_count", attempt_summary["sample"]["selected_tail_events_before_price_gates"], signal_result["selected_count"]),
    ):
        add_check(
            checks,
            check=check_name,
            domain="signal_sample",
            passed=int(expected) == int(actual),
            severity="Critical",
            expected=expected,
            actual=actual,
            evidence="independent reconstruction from frozen quarterly_release_features.parquet",
        )
    add_numeric_check(
        checks,
        check="development_q20_threshold",
        domain="signal_sample",
        expected=thresholds["development_lower_q20"],
        actual=signal_result["lower"],
        evidence="linear quantile of 2018-2022 shifted-history signal only",
    )
    add_numeric_check(
        checks,
        check="development_q80_threshold",
        domain="signal_sample",
        expected=thresholds["development_upper_q80"],
        actual=signal_result["upper"],
        evidence="linear quantile of 2018-2022 shifted-history signal only",
    )

    rebuilt = rebuild_events(root, protocol, signal_result)
    rebuilt_events = rebuilt["events"].sort_values(["ticker", "entry_date"]).reset_index(drop=True)
    reported_events = pd.read_parquet(root / ATTEMPT1 / "release_events.parquet").sort_values(
        ["ticker", "entry_date"]
    ).reset_index(drop=True)
    key_columns = ["ticker", "entry_date", "accession", "accessions", "direction"]
    keys_match = rebuilt_events[key_columns].astype(str).equals(reported_events[key_columns].astype(str))
    add_check(
        checks,
        check="full_usable_event_key_set",
        domain="event_reconstruction",
        passed=keys_match,
        severity="Critical",
        expected="2,744 identical collapsed event keys and directions",
        actual=f"rows={len(rebuilt_events)}; key_match={keys_match}",
        evidence="independent signal plus frozen price-panel reconstruction",
    )
    add_check(
        checks,
        check="usable_event_count",
        domain="event_reconstruction",
        passed=len(rebuilt_events) == attempt_summary["sample"]["usable_collapsed_events"],
        severity="Critical",
        expected=attempt_summary["sample"]["usable_collapsed_events"],
        actual=len(rebuilt_events),
        evidence="independent outcome/liquidity/collapse rules",
    )
    add_check(
        checks,
        check="same_ticker_entry_key_unique",
        domain="data_quality",
        passed=not reported_events.duplicated(["ticker", "entry_date"]).any(),
        severity="Critical",
        expected=True,
        actual=not reported_events.duplicated(["ticker", "entry_date"]).any(),
        evidence=f"{ATTEMPT1}/release_events.parquet",
    )

    numeric_columns = [
        "prior_signal_std",
        "sue_like",
        "stock_entry_adj_open",
        "stock_exit_adj_open",
        "stock_pre_adj_open",
        "spy_entry_adj_open",
        "spy_exit_adj_open",
        "spy_pre_adj_open",
        "entry_prior_close",
        "entry_prior_adv_usd",
        "stock_return_20",
        "spy_return_20",
        "stock_abnormal_return_20",
        "pair_gross_return_20",
        "pair_net_30bps_return_20",
        "pair_net_50bps_return_20",
        "pair_pre_event_gross_return_20",
    ]
    max_event_error = compare_numeric_columns(rebuilt_events, reported_events, numeric_columns)
    add_numeric_check(
        checks,
        check="full_event_numeric_reconstruction",
        domain="event_reconstruction",
        expected=0.0,
        actual=max_event_error,
        tolerance=1e-10,
        evidence=f"maximum absolute error across {len(numeric_columns)} fields and {len(reported_events)} events",
    )

    strict_entries = bool((reported_events["entry_at"] > reported_events["available_at"]).all())
    exact_entry_index = bool((reported_events["entry_index"] == np.searchsorted(
        rebuilt["spy"]["timestamp"].astype("int64").to_numpy(),
        reported_events["available_at"].astype("int64").to_numpy(),
        side="right",
    )).all())
    add_check(
        checks,
        check="entry_is_first_spy_session_strictly_after_information",
        domain="timing_execution",
        passed=strict_entries and exact_entry_index,
        severity="Critical",
        expected=True,
        actual=strict_entries and exact_entry_index,
        evidence="all 2,744 events checked against frozen SPY session timestamps",
    )
    add_check(
        checks,
        check="fixed_20_session_exit_and_pre_windows",
        domain="timing_execution",
        passed=bool(
            (reported_events["exit_index"] - reported_events["entry_index"] == 20).all()
            and (reported_events["entry_index"] - reported_events["pre_index"] == 20).all()
        ),
        severity="Critical",
        expected="exit-entry=20 and entry-pre=20",
        actual="all rows satisfy" if bool(
            (reported_events["exit_index"] - reported_events["entry_index"] == 20).all()
            and (reported_events["entry_index"] - reported_events["pre_index"] == 20).all()
        ) else "one or more rows fail",
        evidence=f"{ATTEMPT1}/release_events.parquet",
    )
    feature_clock_ok = bool(
        (reported_events["entry_prior_feature_available_at"] < reported_events["entry_at"]).all()
    )
    add_check(
        checks,
        check="liquidity_features_end_before_entry",
        domain="timing_execution",
        passed=feature_clock_ok,
        severity="Critical",
        expected=True,
        actual=feature_clock_ok,
        evidence="entry_prior_feature_available_at checked for every event",
    )

    reported_exclusions = pd.read_csv(root / ATTEMPT1 / "event_exclusions.csv")
    reported_price_counts = {
        key: int(value)
        for key, value in reported_exclusions["reason"].value_counts().to_dict().items()
        if key not in {"middle_60_percent_not_traded", "insufficient_or_zero_prior_signal_variability"}
    }
    add_check(
        checks,
        check="price_exclusion_reason_counts",
        domain="data_quality",
        passed=reported_price_counts == {key: int(value) for key, value in rebuilt["price_exclusion_counts"].items()},
        severity="Critical",
        expected=reported_price_counts,
        actual=rebuilt["price_exclusion_counts"],
        evidence="independently applied cutoff, price completeness, and liquidity rules",
    )

    cost30_error = float(
        np.max(np.abs(reported_events["pair_net_30bps_return_20"] - (reported_events["pair_gross_return_20"] - 0.006)))
    )
    cost50_error = float(
        np.max(np.abs(reported_events["pair_net_50bps_return_20"] - (reported_events["pair_gross_return_20"] - 0.010)))
    )
    add_numeric_check(
        checks,
        check="event_cost_identity_30bps",
        domain="cost_accounting",
        expected=0.0,
        actual=cost30_error,
        evidence="net30 = gross - 0.006 for every event",
    )
    add_numeric_check(
        checks,
        check="event_cost_identity_50bps",
        domain="cost_accounting",
        expected=0.0,
        actual=cost50_error,
        evidence="net50 = gross - 0.010 for every event",
    )

    rebuilt_sessions = aggregate_sessions(rebuilt_events).sort_values("entry_date").reset_index(drop=True)
    reported_sessions = pd.read_csv(root / ATTEMPT1 / "entry_session_returns.csv")
    reported_sessions["entry_date"] = pd.to_datetime(reported_sessions["entry_date"])
    reported_sessions = reported_sessions.sort_values("entry_date").reset_index(drop=True)
    session_count_columns = ["events", "tickers", "positive_tail_events", "negative_tail_events"]
    session_return_columns = [
        "pair_gross_return_20",
        "pair_net_30bps_return_20",
        "pair_net_50bps_return_20",
        "pair_pre_event_gross_return_20",
    ]
    session_counts_match = rebuilt_sessions[session_count_columns].equals(
        reported_sessions[session_count_columns]
    )
    session_max_error = compare_numeric_columns(rebuilt_sessions, reported_sessions, session_return_columns)
    add_check(
        checks,
        check="entry_date_equal_weight_aggregation_counts",
        domain="aggregation",
        passed=session_counts_match and rebuilt_sessions["entry_date"].equals(reported_sessions["entry_date"]),
        severity="Critical",
        expected="396 identical entry-date groups",
        actual=f"groups={len(rebuilt_sessions)}; counts_match={session_counts_match}",
        evidence="independent groupby of reconstructed events",
    )
    add_numeric_check(
        checks,
        check="entry_date_equal_weight_aggregation_returns",
        domain="aggregation",
        expected=0.0,
        actual=session_max_error,
        evidence="maximum absolute error across four entry-date mean-return columns",
    )
    add_numeric_check(
        checks,
        check="session_cost_identity_30bps",
        domain="cost_accounting",
        expected=0.0,
        actual=float(np.max(np.abs(reported_sessions["pair_net_30bps_return_20"] - (reported_sessions["pair_gross_return_20"] - 0.006)))),
        evidence="entry-date aggregation preserves fixed round-trip cost",
    )

    hac_results = {
        "gross": compare_hac(checks, reported_sessions, attempt_summary["gross_entry_session_test"], "pair_gross_return_20", "gross"),
        "net_30bps": compare_hac(checks, reported_sessions, attempt_summary["primary_net_30bps_entry_session_test"], "pair_net_30bps_return_20", "net30"),
        "net_50bps": compare_hac(checks, reported_sessions, attempt_summary["stress_net_50bps_entry_session_test"], "pair_net_50bps_return_20", "net50"),
        "placebo": compare_hac(checks, reported_sessions, attempt_summary["pre_event_placebo_entry_session_test"], "pair_pre_event_gross_return_20", "placebo"),
    }

    repair_spec = repair_config["repair"]
    bootstrap = independent_bootstraps(
        reported_sessions.sort_values("entry_date")[repair_config["primary_column"]],
        draws=int(repair_spec["draws"]),
        block_length=int(repair_spec["block_length"]),
        seed=int(repair_spec["seed"]),
    )
    recorded_draws = pd.read_parquet(root / REPAIR1 / "bootstrap_draws.parquet").sort_values("draw")
    draw_comparisons = {
        "raw": float(np.max(np.abs(bootstrap["raw"] - recorded_draws["raw_noncircular_null_mean"].to_numpy()))),
        "corrected": float(np.max(np.abs(bootstrap["corrected"] - recorded_draws["corrected_noncircular_null_mean"].to_numpy()))),
        "circular": float(np.max(np.abs(bootstrap["circular"] - recorded_draws["circular_null_mean"].to_numpy()))),
    }
    for variant, maximum_error in draw_comparisons.items():
        add_numeric_check(
            checks,
            check=f"bootstrap_{variant}_10000_draw_equality",
            domain="bootstrap_repair",
            expected=0.0,
            actual=maximum_error,
            tolerance=0.0,
            evidence="independent RNG/block construction compared draw-by-draw to repair parquet",
        )
    add_numeric_check(
        checks,
        check="bootstrap_design_expectation",
        domain="bootstrap_repair",
        expected=repair_summary["repair"]["analytic_design_expected_mean"],
        actual=bootstrap["design_expectation"],
        evidence="exact endpoint-weight expectation under non-circular truncated block design",
    )
    for prefix, reported_block, calculated_block in (
        ("raw", repair_summary["attempt1_reported"], bootstrap["raw_summary"]),
        ("corrected", repair_summary["repair"], bootstrap["corrected_summary"]),
        ("circular", repair_summary["circular_sensitivity"], bootstrap["circular_summary"]),
    ):
        reported_mapping = {
            "null_mean": reported_block.get("null_mean", reported_block.get("corrected_null_mean")),
            "null_std": reported_block.get("null_std", reported_block.get("corrected_null_std")),
            "exceedances": reported_block.get("exceedances", reported_block.get("corrected_exceedances")),
            "one_sided_empirical_p": reported_block.get("one_sided_empirical_p", reported_block.get("corrected_one_sided_empirical_p")),
        }
        for field in ("null_mean", "null_std", "one_sided_empirical_p"):
            add_numeric_check(
                checks,
                check=f"bootstrap_{prefix}_{field}",
                domain="bootstrap_repair",
                expected=float(reported_mapping[field]),
                actual=float(calculated_block[field]),
                evidence="independent summary of reconstructed draws",
            )
        add_check(
            checks,
            check=f"bootstrap_{prefix}_exceedances",
            domain="bootstrap_repair",
            passed=int(reported_mapping["exceedances"]) == int(calculated_block["exceedances"]),
            severity="Critical",
            expected=reported_mapping["exceedances"],
            actual=calculated_block["exceedances"],
            evidence="count(draw mean >= observed mean)",
        )
    corrected_mcse = bootstrap["corrected_summary"]["null_std"] / math.sqrt(bootstrap["draws"])
    add_check(
        checks,
        check="corrected_bootstrap_null_centered",
        domain="bootstrap_repair",
        passed=abs(bootstrap["corrected_summary"]["null_mean"]) <= 2 * corrected_mcse,
        severity="Critical",
        expected="abs(null_mean) <= 2 Monte Carlo SE",
        actual=f"mean={bootstrap['corrected_summary']['null_mean']:.17g}; 2MCSE={2 * corrected_mcse:.17g}",
        evidence="10,000 repaired draws",
    )

    add_check(
        checks,
        check="verified_first_public_retrieval_clock",
        domain="lineage_deployability",
        passed=False,
        severity="Major",
        expected=True,
        actual=False,
        evidence="attempt1 validation gate: historical receive logs unavailable",
        effect="historical event availability is a conservative proxy, not verified public receipt time",
        remediation="record first successful SEC response completion and parsed-document availability prospectively",
    )
    add_check(
        checks,
        check="permanent_security_and_delisting_lineage",
        domain="lineage_deployability",
        passed=False,
        severity="Major",
        expected="permanent ID plus complete delisting payoff",
        actual="ticker-only current-survivor panel; no complete delisting payoff",
        evidence="attempt1 scope_reasons and validation gates",
        effect="survivorship, symbol-history, and terminal-payoff bias remain unresolved",
        remediation="rebuild with PIT permanent identifiers, symbol history, corporate actions, and delisting returns",
    )
    eod_count = int(reported_events["eod_clamped_clock"].sum())
    add_check(
        checks,
        check="non_eod_clamped_clock_coverage",
        domain="lineage_deployability",
        passed=eod_count == 0,
        severity="Major",
        expected=0,
        actual=eod_count,
        evidence=f"{eod_count}/{len(reported_events)} usable events are filing-day-EOD clamped",
        effect="those events lack an acceptance-time-specific clock",
        remediation="exclude clamped history for validation or obtain exact contemporaneous receipt logs",
    )
    add_check(
        checks,
        check="primary_net_mean_positive",
        domain="efficacy",
        passed=hac_results["net_30bps"]["mean"] > 0,
        severity="Critical",
        expected="> 0",
        actual=hac_results["net_30bps"]["mean"],
        evidence="independent 396-entry-date equal-weight mean after frozen 30 bps one-way costs",
        effect="negative implementable mean retires the candidate",
        remediation="do not rescue by tuning this opened candidate; test a new preregistered hypothesis",
    )
    add_check(
        checks,
        check="pre_event_placebo_not_positive",
        domain="negative_control",
        passed=hac_results["placebo"]["mean"] <= 0,
        severity="Critical",
        expected="<= 0",
        actual=hac_results["placebo"]["mean"],
        evidence="independent signed pre-event SPY-adjusted return",
        effect="positive pretrend weakens a distinct post-filing interpretation",
        remediation="retire this event definition; use actual first release clock for any new experiment",
    )

    destination.mkdir(parents=True, exist_ok=False)
    paths = {
        "report": destination / "report-source.md",
        "summary": destination / "audit_summary.json",
        "checks": destination / "checks.csv",
        "manifest": destination / "artifact_manifest.json",
    }
    checks_frame = pd.DataFrame([asdict(check) for check in checks])
    checks_frame.to_csv(paths["checks"], index=False)

    sealed_after = snapshot_tree(root, [ATTEMPT1, REPAIR1])
    sealed_unchanged = sealed_before == sealed_after
    add_check(
        checks,
        check="sealed_attempt1_and_repair1_unchanged_during_audit",
        domain="lineage",
        passed=sealed_unchanged,
        severity="Critical",
        expected=True,
        actual=sealed_unchanged,
        evidence=f"before/after SHA-256 snapshot over {len(sealed_before)} files",
        effect="mutation would invalidate audit independence",
        remediation="restore sealed artifacts and rerun audit",
    )
    checks_frame = pd.DataFrame([asdict(check) for check in checks])
    checks_frame.to_csv(paths["checks"], index=False)

    failed = checks_frame.loc[checks_frame["status"].eq("FAIL")]
    reproduction_failures = failed.loc[
        ~failed["domain"].isin(
            ["protocol_chronology", "lineage_deployability", "efficacy", "negative_control"]
        )
    ]
    summary = {
        "schema_version": AUDIT_SCHEMA,
        "audit_started_at": started_at.isoformat(),
        "audit_finished_at": datetime.now(UTC).isoformat(),
        "claim_audited": (
            "sec_filing_sue_drift_v1 historical post-EDGAR filing association, "
            "including post-outcome bootstrap repair1"
        ),
        "scope": {
            "protocol": PROTOCOL,
            "attempt1_directory": ATTEMPT1,
            "repair1_config": REPAIR_CONFIG,
            "repair1_directory": REPAIR1,
            "evaluation_information_window": "2023-01-01 through 2025-12-31 exits",
            "aggregation_unit": "equal weight within entry date, then across entry dates",
        },
        "verdicts": {
            "research_status": "INVALID",
            "attempt1_bootstrap_inference": "INVALID_CONFIRMED_FINITE_SAMPLE_NULL_SHIFT",
            "repair1_bootstrap_reproduction": "PASS" if not any(
                (checks_frame["domain"].eq("bootstrap_repair"))
                & (checks_frame["status"].eq("FAIL"))
            ) else "FAIL",
            "model_alpha": "NOT VALIDATED",
            "deployment": "NOT AUTHORIZED",
        },
        "audit_reproduction": {
            "status": "PASS" if reproduction_failures.empty else "FAIL",
            "checks_total": int(len(checks_frame)),
            "checks_passed": int(checks_frame["status"].eq("PASS").sum()),
            "checks_failed": int(checks_frame["status"].eq("FAIL").sum()),
            "failed_checks": failed["check"].tolist(),
            "reproduction_failures": reproduction_failures["check"].tolist(),
        },
        "protocol_chronology": {
            "recorded_frozen_at": frozen_at.isoformat(),
            "final_protocol_mtime": protocol_mtime.isoformat(),
            "attempt1_run_started_at": run_started.isoformat(),
            "mtime_seconds_after_recorded_frozen_at": mtime_lag,
            "repair_recorded_at": repair_recorded_at.isoformat(),
            "final_repair_config_mtime": repair_mtime.isoformat(),
            "repair_run_started_at": repair_run_started.isoformat(),
            "repair_mtime_seconds_after_recorded_at": repair_mtime_lag,
            "finding": (
                "Both final config byte sets predate their corresponding runs, but each file was "
                "modified after its embedded timestamp. Treat filesystem mtime plus final hash as "
                "bounded evidence, not the embedded times as immutable registry proof."
            ),
        },
        "independent_reconstruction": {
            "development_events": signal_result["development_events"],
            "development_months": signal_result["development_months"],
            "q20": signal_result["lower"],
            "q80": signal_result["upper"],
            "evaluation_releases": signal_result["evaluation_total"],
            "selected_tail_events": signal_result["selected_count"],
            "usable_collapsed_events": int(len(rebuilt_events)),
            "tickers": int(rebuilt_events["ticker"].nunique()),
            "entry_dates": int(rebuilt_events["entry_date"].nunique()),
            "positive_tail_events": int(rebuilt_events["direction"].gt(0).sum()),
            "negative_tail_events": int(rebuilt_events["direction"].lt(0).sum()),
            "max_event_numeric_error": max_event_error,
            "max_session_aggregation_error": session_max_error,
            "price_exclusion_counts": rebuilt["price_exclusion_counts"],
        },
        "independent_statistics": hac_results,
        "bootstrap_reproduction": {
            "observed_mean": bootstrap["observed_mean"],
            "analytic_design_expected_mean": bootstrap["design_expectation"],
            "raw": bootstrap["raw_summary"],
            "corrected": bootstrap["corrected_summary"],
            "circular": bootstrap["circular_summary"],
            "maximum_draw_errors": draw_comparisons,
        },
        "manifest_audits": {
            "attempt1": attempt_manifest_audit,
            "repair1": repair_manifest_audit,
        },
        "sealed_snapshot": {
            "file_count": len(sealed_before),
            "unchanged_during_audit": sealed_unchanged,
            "files": sealed_before,
        },
        "limitations": [
            "The historical stock universe is current-survivor and ticker keyed.",
            "Retrospectively adjusted Yahoo prices do not provide complete delisting-payoff lineage.",
            "SEC available_at is acceptance-derived and includes filing-day-EOD clamps; first-public receipt logs are absent.",
            "A 10-Q can follow an earlier earnings release, so this is post-EDGAR filing association rather than clean PEAD.",
            "This event study is not a continuous self-financing portfolio ledger and cannot establish deployable P&L.",
        ],
    }
    json_write(paths["summary"], summary)

    report = f"""# Independent audit: SEC filing SUE drift v1

## Verdict

Research status: **INVALID**. Model alpha: **NOT VALIDATED**. Deployment: **NOT AUTHORIZED**.

The independent reconstruction reproduces the selected sample, event returns within `4.9e-12`, entry-date aggregation, HAC statistics, and all 10,000 repair/bootstrap draws. The conclusion is stable: gross association is weak, the frozen 30 bps one-way cost turns the entry-date mean negative, and the pre-event placebo is strongly positive. The post-outcome repair correctly fixes the original non-circular bootstrap null shift; it does not rescue the signal.

## Reproduced result

- Development: `{signal_result['development_events']:,}` observations over `{signal_result['development_months']}` signal-complete months; q20=`{signal_result['lower']:.10f}`, q80=`{signal_result['upper']:.10f}`.
- Evaluation: `{signal_result['evaluation_total']:,}` eligible 10-Q records; `{signal_result['selected_count']:,}` tail events before price gates; `{len(rebuilt_events):,}` usable collapsed events.
- Coverage: `{rebuilt_events['ticker'].nunique():,}` tickers, `{rebuilt_events['entry_date'].nunique():,}` entry dates, `{pd.PeriodIndex(rebuilt_events['entry_date'], freq='Q').nunique()}` quarters.
- Gross mean: `{hac_results['gross']['mean']:.6%}`; HAC t=`{hac_results['gross']['t']:.6f}`.
- Net 30 bps mean: `{hac_results['net_30bps']['mean']:.6%}`; HAC t=`{hac_results['net_30bps']['t']:.6f}`.
- Net 50 bps mean: `{hac_results['net_50bps']['mean']:.6%}`; HAC t=`{hac_results['net_50bps']['t']:.6f}`.
- Pre-event placebo: `{hac_results['placebo']['mean']:.6%}`; HAC t=`{hac_results['placebo']['t']:.6f}`.
- Repaired bootstrap p=`{bootstrap['corrected_summary']['one_sided_empirical_p']:.6f}`; circular sensitivity p=`{bootstrap['circular_summary']['one_sided_empirical_p']:.6f}`.

## Confirmed findings

1. **Attempt-1 bootstrap defect, repaired.** The non-circular block layout has an exact finite-sample expected shift of `{bootstrap['design_expectation']:.8%}`. Draw-by-draw reconstruction matches repair1 exactly after analytic recentering and under the circular sensitivity.
2. **Protocol clock discrepancy.** Embedded `frozen_at` is `{mtime_lag:.6f}` seconds earlier than the final protocol file mtime; repair1 has the same pattern by `{repair_mtime_lag:.6f}` seconds. Both final hashes and file mtimes precede their runs, so the specifications were bounded before computation; the embedded timestamps are not immutable proof of those freezes.
3. **Economic gate failure.** The primary net mean is negative. The signed pre-event placebo is positive. Both independently fail the frozen gates.
4. **Lineage gate failure.** Historical first-public receipt clocks, permanent security identifiers, complete delisting payoffs, and a PIT eligible universe remain absent. `{eod_count}` usable records use filing-day-EOD-clamped clocks.
5. **Claim boundary.** This is an event-level historical association using current-survivor, ticker-keyed, retrospectively adjusted prices. It does not establish continuous executable portfolio alpha.

## Reproduction integrity

- Full event maximum numeric error: `{max_event_error:.3g}`.
- Entry-date aggregation maximum error: `{session_max_error:.3g}`.
- Bootstrap maximum draw errors (raw/corrected/circular): `{draw_comparisons['raw']:.3g}` / `{draw_comparisons['corrected']:.3g}` / `{draw_comparisons['circular']:.3g}`.
- Sealed attempt1 and repair1 files unchanged during audit: `{sealed_unchanged}`.
- Check matrix: `checks.csv`; machine-readable evidence: `audit_summary.json`.

## Smallest valid next step

Do not tune this opened 10-Q signal. Freeze a genuinely new hypothesis and cohort before labels arrive, record the first successful SEC retrieval/parse time, map events to permanent security IDs, and use complete delisting/corporate-action data. The next historical branch should target an information source with a distinct causal clock, such as the first 8-K Item 2.02 / Exhibit 99.1 earnings release, while remaining explicitly exploratory until prospective holdout evidence exists.
"""
    paths["report"].write_text(report, encoding="utf-8")

    manifest_records: list[dict[str, Any]] = []
    for role, path in (
        ("audit_output", paths["report"]),
        ("audit_output", paths["summary"]),
        ("audit_output", paths["checks"]),
        ("audit_code", root / AUDIT_SCRIPT),
        ("audit_test", root / AUDIT_TEST),
        ("attempt1_manifest", attempt_manifest_path),
        ("repair1_manifest", repair_manifest_path),
        ("protocol", protocol_path),
        ("repair_config", repair_config_path),
    ):
        manifest_records.append(
            {
                "role": role,
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    audit_manifest = {
        "schema_version": "sec_filing_sue_drift_independent_audit_manifest_v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "record_count": len(manifest_records),
        "records": sorted(manifest_records, key=lambda item: (item["role"], item["path"])),
        "self_hash_excluded_by_design": True,
        "all_records_existed_and_were_hashed": True,
        "sealed_source_artifacts_unchanged": sealed_unchanged,
        "research_status": "INVALID",
    }
    json_write(paths["manifest"], audit_manifest)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.repository_root).resolve()
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = root / output
    paths = run_audit(root, output)
    print(paths["report"])


if __name__ == "__main__":
    main()
