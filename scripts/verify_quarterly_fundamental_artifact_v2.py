#!/usr/bin/env python3
"""Independent verifier for quarterly-fundamental diagnostic v2.

The verifier deliberately does not import the producer, its feature module, or
the project's metric helpers.  It reconstructs the persisted primary test from
first principles and audits the filing-state/provenance invariants needed for
the narrow artifact-integrity claim.  A passing result does not upgrade the
producer's ``INVALID`` historical/deployability verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm, rankdata


PRIMARY = "quarterly_fundamental_momentum_score"
PRIMARY_COMPONENTS = ("earnings_change_on_assets", "sales_growth_yoy")
RELEASE_FEATURES = (*PRIMARY_COMPONENTS, "gross_margin_change_yoy")
CONCEPTS = ("net_income", "revenue", "gross_profit")
DEFAULT_TOLERANCE = 1e-12
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
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
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _finite_number(value: object) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="coerce")


def _naive_dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.tz_localize(None)


def _centered_rank(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    return (numeric.rank(method="average", pct=True) * 2.0 - 1.0).to_numpy()


def _hac_tstat(values: Iterable[object], lags: int = 3) -> float:
    array = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().to_numpy(float)
    count = len(array)
    if count < lags + 3:
        return float("nan")
    centered = array - array.mean()
    long_run_variance = float(centered @ centered / count)
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        long_run_variance += 2.0 * weight * float(
            centered[lag:] @ centered[:-lag] / count
        )
    standard_error = np.sqrt(max(long_run_variance, 0.0) / count)
    return float(array.mean() / standard_error) if standard_error > 0 else float("nan")


def _equal_float(left: object, right: object, tolerance: float) -> bool:
    a = _finite_number(left)
    b = _finite_number(right)
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tolerance


def _max_abs_error(left: pd.Series, right: pd.Series) -> float:
    a = pd.to_numeric(left, errors="coerce").to_numpy(float)
    b = pd.to_numeric(right, errors="coerce").to_numpy(float)
    both_nan = np.isnan(a) & np.isnan(b)
    one_nan = np.isnan(a) ^ np.isnan(b)
    if one_nan.any():
        return float("inf")
    finite = ~both_nan
    return float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def _period_masks(dates: pd.Series) -> dict[str, pd.Series]:
    stamps = _utc(dates)
    return {
        "full": pd.Series(True, index=dates.index),
        "pre_2026": stamps.lt(pd.Timestamp("2026-01-01", tz="UTC")),
        "2026_h1_reporting_only": stamps.ge(pd.Timestamp("2026-01-01", tz="UTC")),
    }


def reconstruct_monthly_statistics(
    panel: pd.DataFrame,
    *,
    controls: list[str],
    minimum_names: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute the frozen primary IC and deterministic issuer deciles."""

    _require_columns(
        panel,
        ["signal_date", "ticker", "issuer_cik", "target_return", PRIMARY, *controls],
        "monthly panel",
    )
    work = panel.copy()
    work["signal_date"] = _utc(work["signal_date"])
    rows: list[dict[str, object]] = []
    deciles: list[dict[str, object]] = []
    for signal_date, month in work.groupby("signal_date", sort=True):
        valid = month.loc[pd.to_numeric(month[PRIMARY], errors="coerce").notna()].copy()
        if len(valid) < minimum_names:
            continue
        target = pd.to_numeric(valid["target_return"], errors="coerce")
        if target.isna().any() or not np.isfinite(target.to_numpy(float)).all():
            raise ValueError(f"partial or non-finite target coverage at {signal_date}")

        ranked_controls: list[np.ndarray] = []
        for control in controls:
            values = pd.to_numeric(valid[control], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            ranked_controls.extend(
                [
                    np.nan_to_num(_centered_rank(values), nan=0.0),
                    values.isna().astype(float).to_numpy(),
                ]
            )
        target_rank = _centered_rank(target)
        design = np.column_stack(
            [np.ones(len(valid), dtype=float), *ranked_controls]
        )
        effective_rank = int(np.linalg.matrix_rank(design))
        if len(valid) <= effective_rank + 2:
            raise ValueError(
                f"too few rows for independent residualization at {signal_date}: "
                f"rows={len(valid)}, rank={effective_rank}"
            )
        coefficients = np.linalg.lstsq(design, target_rank, rcond=None)[0]
        residual = target_rank - design @ coefficients
        score = pd.to_numeric(valid[PRIMARY], errors="raise").to_numpy(float)
        target_array = target.to_numpy(float)
        rows.append(
            {
                "signal_date": signal_date,
                "feature": PRIMARY,
                "names": int(len(valid)),
                "raw_rank_ic": float(
                    np.corrcoef(rankdata(score), rankdata(target_array))[0, 1]
                ),
                "incremental_rank_ic": float(
                    np.corrcoef(rankdata(score), rankdata(residual))[0, 1]
                ),
            }
        )

        ordered = valid.assign(
            _score=score,
            _cik=pd.to_numeric(valid["issuer_cik"], errors="coerce").fillna(np.inf),
        ).sort_values(["_score", "_cik", "ticker"], kind="mergesort")
        ordered["decile"] = (
            np.floor(np.arange(len(ordered)) * 10 / len(ordered)).astype(int) + 1
        ).clip(1, 10)
        for decile, members in ordered.groupby("decile", sort=True):
            deciles.append(
                {
                    "signal_date": signal_date,
                    "feature": PRIMARY,
                    "decile": int(decile),
                    "names": int(len(members)),
                    "equal_weight_return": float(members["target_return"].mean()),
                }
            )
    return (
        pd.DataFrame(
            rows,
            columns=[
                "signal_date",
                "feature",
                "names",
                "raw_rank_ic",
                "incremental_rank_ic",
            ],
        ),
        pd.DataFrame(
            deciles,
            columns=[
                "signal_date",
                "feature",
                "decile",
                "names",
                "equal_weight_return",
            ],
        ),
    )


def summarize_ic(monthly: pd.DataFrame, trial_floor: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for period, mask in _period_masks(monthly["signal_date"]).items():
        sample = monthly.loc[mask].copy()
        if sample.empty:
            continue
        incremental_t = _hac_tstat(sample["incremental_rank_ic"])
        yearly = (
            sample.assign(year=_utc(sample["signal_date"]).dt.year)
            .groupby("year", sort=True)["incremental_rank_ic"]
            .mean()
        )
        raw_p = float(norm.sf(incremental_t))
        rows.append(
            {
                "feature": PRIMARY,
                "period": period,
                "months": int(len(sample)),
                "minimum_names": int(sample["names"].min()),
                "median_names": float(sample["names"].median()),
                "maximum_names": int(sample["names"].max()),
                "mean_raw_rank_ic": float(sample["raw_rank_ic"].mean()),
                "raw_ic_hac_tstat": _hac_tstat(sample["raw_rank_ic"]),
                "mean_incremental_rank_ic": float(sample["incremental_rank_ic"].mean()),
                "incremental_ic_hac_tstat": incremental_t,
                "positive_month_fraction": float(
                    sample["incremental_rank_ic"].gt(0).mean()
                ),
                "positive_year_fraction": float(yearly.gt(0).mean()),
                "one_sided_hac_pvalue": raw_p,
                # With one primary test the producer's legacy BH column equals p.
                "bh_qvalue_pre2026_family": raw_p if period == "pre_2026" else np.nan,
                "global_bonferroni_pvalue_floor_110": (
                    min(1.0, raw_p * trial_floor)
                    if period == "pre_2026"
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_deciles(monthly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for decile, evidence in monthly.groupby("decile", sort=True):
        for period, mask in _period_masks(evidence["signal_date"]).items():
            sample = evidence.loc[mask]
            if sample.empty:
                continue
            rows.append(
                {
                    "feature": PRIMARY,
                    "period": period,
                    "decile": int(decile),
                    "months": int(len(sample)),
                    "annualized_arithmetic_return": float(
                        12.0 * sample["equal_weight_return"].mean()
                    ),
                    "hac_tstat": _hac_tstat(sample["equal_weight_return"]),
                }
            )
    pivot = monthly.pivot(
        index="signal_date", columns="decile", values="equal_weight_return"
    )
    if 1 in pivot and 10 in pivot:
        spread = (pivot[10] - pivot[1]).dropna()
        spread_frame = pd.DataFrame(
            {"signal_date": spread.index, "equal_weight_return": spread.to_numpy()}
        )
        for period, mask in _period_masks(spread_frame["signal_date"]).items():
            sample = spread_frame.loc[mask, "equal_weight_return"]
            if sample.empty:
                continue
            rows.append(
                {
                    "feature": PRIMARY,
                    "period": period,
                    "decile": "D10-D1",
                    "months": int(len(sample)),
                    "annualized_arithmetic_return": float(12.0 * sample.mean()),
                    "hac_tstat": _hac_tstat(sample),
                }
            )
    return pd.DataFrame(rows)


def _compare_table(
    observed: pd.DataFrame,
    reconstructed: pd.DataFrame,
    *,
    keys: list[str],
    numeric: list[str],
    label: str,
) -> dict[str, object]:
    left = observed.copy()
    right = reconstructed.copy()
    missing_left_keys = sorted(set(keys).difference(left.columns))
    missing_right_keys = sorted(set(keys).difference(right.columns))
    if missing_left_keys or missing_right_keys:
        return {
            "label": label,
            "keys_match": False,
            "columns_match": False,
            "missing_observed_key_columns": missing_left_keys,
            "missing_reconstructed_key_columns": missing_right_keys,
            "maximum_absolute_error": float("inf"),
        }
    missing_left_numeric = sorted(set(numeric).difference(left.columns))
    missing_right_numeric = sorted(set(numeric).difference(right.columns))
    comparable_numeric = [
        column
        for column in numeric
        if column in left.columns and column in right.columns
    ]
    for frame in (left, right):
        if "signal_date" in keys:
            frame["signal_date"] = _utc(frame["signal_date"])
        if "decile" in keys:
            frame["decile"] = frame["decile"].astype(str)
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        return {
            "label": label,
            "keys_match": False,
            "columns_match": not missing_left_numeric and not missing_right_numeric,
            "maximum_absolute_error": float("inf"),
            "reason": "duplicate keys",
        }
    left_keys = set(map(tuple, left[keys].itertuples(index=False, name=None)))
    right_keys = set(map(tuple, right[keys].itertuples(index=False, name=None)))
    if left_keys != right_keys:
        return {
            "label": label,
            "keys_match": False,
            "columns_match": not missing_left_numeric and not missing_right_numeric,
            "observed_rows": int(len(left)),
            "reconstructed_rows": int(len(right)),
            "missing_keys": int(len(right_keys - left_keys)),
            "extra_keys": int(len(left_keys - right_keys)),
            "maximum_absolute_error": float("inf"),
        }
    merged = left.merge(
        right, on=keys, how="inner", suffixes=("_observed", "_reconstructed"), validate="one_to_one"
    )
    errors = {
        column: _max_abs_error(
            merged[f"{column}_observed"], merged[f"{column}_reconstructed"]
        )
        for column in comparable_numeric
    }
    return {
        "label": label,
        "keys_match": True,
        "columns_match": not missing_left_numeric and not missing_right_numeric,
        "missing_observed_numeric_columns": missing_left_numeric,
        "missing_reconstructed_numeric_columns": missing_right_numeric,
        "observed_rows": int(len(left)),
        "reconstructed_rows": int(len(right)),
        "maximum_absolute_error": max(errors.values(), default=0.0),
        "column_maximum_absolute_errors": errors,
    }


def _file_inventory(artifact: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    aggregate = hashlib.sha256()
    for path in sorted(item for item in artifact.rglob("*") if item.is_file()):
        relative = path.relative_to(artifact).as_posix()
        digest = _sha256(path)
        size = path.stat().st_size
        rows.append({"path": relative, "bytes": size, "sha256": digest})
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    return {
        "file_count": len(rows),
        "total_bytes": int(sum(int(row["bytes"]) for row in rows)),
        "aggregate_inventory_sha256": aggregate.hexdigest(),
        "records": rows,
    }


def _enrich_partial_panel(
    panel: pd.DataFrame,
    releases: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Reconstruct audit-only columns omitted by the interrupted v1 writer."""

    result = panel.copy()
    result["signal_date"] = _utc(result["signal_date"])
    required_audit_columns = [
        "decision_at",
        "adv_usd_3m",
        "issuer_cik",
        "issuer_representative",
        "form",
        "release_kind",
        "supersession_only",
        "header_inventory_verified",
        "barrier_inventory_source",
        "context_validation_level",
    ]
    missing_persisted = sorted(set(required_audit_columns).difference(result.columns))
    derivations: list[str] = []
    source_panel_hash_matches: bool | None = None

    if "decision_at" not in result:
        result["entry_at"] = _utc(result["entry_at"])
        result["decision_at"] = result["entry_at"] - pd.Timedelta(
            minutes=int(config["clock"]["operational_latency_minutes_before_entry"])
        )
        derivations.append("decision_at_from_entry_at_minus_frozen_latency")

    if "adv_usd_3m" not in result:
        source_name = (config.get("source") or {}).get("monthly_panel")
        if source_name is not None:
            source_path = Path(str(source_name))
            if not source_path.is_absolute():
                source_path = PROJECT_ROOT / source_path
            if source_path.is_file():
                expected_hash = (config.get("source") or {}).get(
                    "expected_monthly_panel_sha256"
                )
                source_panel_hash_matches = (
                    expected_hash is None or _sha256(source_path) == expected_hash
                )
                source_panel = pd.read_parquet(
                    source_path, columns=["ticker", "signal_date", "adv_usd_3m"]
                )
                source_panel["signal_date"] = _utc(source_panel["signal_date"])
                if source_panel.duplicated(["ticker", "signal_date"]).any():
                    raise ValueError("upstream panel has duplicate ticker/month keys")
                result = result.merge(
                    source_panel,
                    on=["ticker", "signal_date"],
                    how="left",
                    validate="one_to_one",
                )
                derivations.append("adv_usd_3m_from_hash_checked_frozen_source_panel")
        if "adv_usd_3m" not in result:
            result["adv_usd_3m"] = np.nan

    release_fields = {
        "issuer_cik": "cik",
        "form": "form",
        "release_kind": "release_kind",
        "supersession_only": "supersession_only",
        "header_inventory_verified": "header_inventory_verified",
        "barrier_inventory_source": "barrier_inventory_source",
        "context_validation_level": "context_validation_level",
    }
    missing_release_fields = {
        output: source
        for output, source in release_fields.items()
        if output not in result
    }
    if missing_release_fields:
        lookup_columns = ["ticker", "accession", *set(missing_release_fields.values())]
        lookup = releases[lookup_columns].rename(
            columns={source: output for output, source in missing_release_fields.items()}
        )
        if lookup.duplicated(["ticker", "accession"]).any():
            raise ValueError("release lookup has duplicate ticker/accession keys")
        result = result.merge(
            lookup,
            on=["ticker", "accession"],
            how="left",
            validate="many_to_one",
        )
        derivations.append("release_audit_columns_from_persisted_release_table")

    if "issuer_representative" not in result:
        result["issuer_representative"] = True
        has_issuer = result["issuer_cik"].notna()
        if has_issuer.any():
            candidates = result.loc[
                has_issuer, ["signal_date", "ticker", "issuer_cik", "adv_usd_3m"]
            ].copy()
            candidates["_adv"] = pd.to_numeric(
                candidates["adv_usd_3m"], errors="coerce"
            ).fillna(-np.inf)
            selected = (
                candidates.sort_values(
                    ["signal_date", "issuer_cik", "_adv", "ticker"],
                    ascending=[True, True, False, True],
                    kind="mergesort",
                )
                .drop_duplicates(["signal_date", "issuer_cik"], keep="first")
                .index
            )
            result.loc[has_issuer & ~result.index.isin(selected), "issuer_representative"] = False
        derivations.append("issuer_representative_from_frozen_adv_cik_ticker_rule")

    return result, {
        "passed_without_enrichment": not missing_persisted,
        "missing_persisted_columns": missing_persisted,
        "audit_only_derivations": derivations,
        "source_panel_hash_matches": source_panel_hash_matches,
    }


def _manifest_audit(artifact: Path) -> dict[str, object]:
    manifest_path = artifact / "artifact_manifest.json"
    if not manifest_path.is_file():
        return {
            "passed": False,
            "reason": "artifact_manifest.json is missing",
            "records": 0,
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("records")
    if not isinstance(records, list):
        return {"passed": False, "reason": "manifest records is not a list", "records": 0}
    bad: list[str] = []
    duplicate_paths: list[str] = []
    seen: set[str] = set()
    output_paths: set[str] = set()
    for record in records:
        raw_path = Path(str(record.get("path", "")))
        path = raw_path if raw_path.is_absolute() else (artifact / raw_path).resolve()
        key = path.resolve().as_posix()
        if key in seen:
            duplicate_paths.append(key)
        seen.add(key)
        if record.get("role") == "output":
            output_paths.add(key)
        if (
            not path.is_file()
            or path == manifest_path
            or path.stat().st_size != int(record.get("bytes", -1))
            or _sha256(path) != record.get("sha256")
        ):
            bad.append(key)
    expected_outputs = {
        path.resolve().as_posix()
        for path in artifact.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    omissions = expected_outputs - output_paths
    extras = output_paths - expected_outputs
    verdict_ok = manifest.get("historical_backtest_verdict") == "INVALID"
    authorization_ok = manifest.get("portfolio_authorized") is False
    return {
        "passed": not bad
        and not duplicate_paths
        and not omissions
        and not extras
        and verdict_ok
        and authorization_ok,
        "records": int(len(records)),
        "bad_records": int(len(bad)),
        "duplicate_paths": int(len(duplicate_paths)),
        "output_omissions": int(len(omissions)),
        "extra_output_records": int(len(extras)),
        "historical_backtest_verdict": manifest.get("historical_backtest_verdict"),
        "portfolio_authorized": manifest.get("portfolio_authorized"),
        "verdict_and_authorization_consistent": verdict_ok and authorization_ok,
        "bad_record_examples": bad[:5],
        "omission_examples": sorted(omissions)[:5],
        "extra_examples": sorted(extras)[:5],
    }


def _recovery_lineage_audit(
    artifact: Path,
    run_summary: dict[str, Any],
) -> dict[str, object]:
    """Audit the disclosed reporting-only recovery, when one is present."""

    status = str(run_summary.get("execution_status") or "")
    if not status.startswith("RECOVERED_"):
        return {"applicable": False, "passed": True}
    amendment_path = artifact / "execution_amendment.json"
    reconciliation_path = artifact / "source_reconciliation.json"
    manifest_path = artifact / "artifact_manifest.json"
    if not all(path.is_file() for path in (amendment_path, reconciliation_path, manifest_path)):
        return {
            "applicable": True,
            "passed": False,
            "reason": "recovery disclosure, reconciliation, or manifest is missing",
        }
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    reconciliation = json.loads(reconciliation_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_by_path = {
        Path(str(record["path"])).resolve().as_posix(): record
        for record in manifest.get("records", [])
    }

    checkpoint_manifest_path = Path(str(amendment.get("checkpoint_manifest", "")))
    if not checkpoint_manifest_path.is_absolute():
        checkpoint_manifest_path = PROJECT_ROOT / checkpoint_manifest_path
    checkpoint_ok = checkpoint_manifest_path.is_file()
    checkpoint = (
        json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
        if checkpoint_ok
        else {}
    )
    snapshot_root = Path(str(checkpoint.get("artifact", "")))
    if not snapshot_root.is_absolute():
        snapshot_root = PROJECT_ROOT / snapshot_root
    checkpoint_bad = 0
    checkpoint_records = checkpoint.get("records", [])
    changed_recovered_files: list[str] = []
    unexpected_recovered_changes: list[str] = []
    allowed_changed = {
        "monthly_quarterly_signal_panel.parquet",
        "feature_period_summary.csv",
        "gates.csv",
    }
    for record in checkpoint_records:
        relative = Path(str(record.get("path", "")))
        snapshotted = snapshot_root / relative
        if (
            not snapshotted.is_file()
            or snapshotted.stat().st_size != int(record.get("bytes", -1))
            or _sha256(snapshotted) != record.get("sha256")
        ):
            checkpoint_bad += 1
        recovered = artifact / relative
        if recovered.is_file() and _sha256(recovered) != record.get("sha256"):
            changed_recovered_files.append(relative.as_posix())
            if relative.as_posix() not in allowed_changed:
                unexpected_recovered_changes.append(relative.as_posix())

    frozen_columns = list(
        (reconciliation.get("attempt1_alignment_replay") or {})
        .get("column_checks", {})
        .keys()
    )
    column_checks: dict[str, bool] = {}
    panel_key_match = False
    if frozen_columns:
        checkpoint_panel = snapshot_root / "monthly_quarterly_signal_panel.parquet"
        recovered_panel = artifact / "monthly_quarterly_signal_panel.parquet"
        if checkpoint_panel.is_file() and recovered_panel.is_file():
            old = pd.read_parquet(checkpoint_panel)
            new = pd.read_parquet(recovered_panel)
            for frame in (old, new):
                frame["signal_date"] = _utc(frame["signal_date"])
            old = old.sort_values(["signal_date", "ticker"], kind="mergesort").reset_index(drop=True)
            new = new.sort_values(["signal_date", "ticker"], kind="mergesort").reset_index(drop=True)
            panel_key_match = old[["signal_date", "ticker"]].equals(
                new[["signal_date", "ticker"]]
            )
            for column in frozen_columns:
                column_checks[column] = bool(
                    column in old.columns
                    and column in new.columns
                    and old[column].equals(new[column])
                )

    raw_records = [
        record for record in manifest.get("records", []) if record.get("role") == "raw_input"
    ]
    companyfacts_records = sum(
        "/companyfacts/" in str(record.get("path")) for record in raw_records
    )
    inventory_records = sum(
        "/submissions" in str(record.get("path")) for record in raw_records
    )
    disclosure_records_ok = all(
        manifest_by_path.get(path.resolve().as_posix(), {}).get("role") == "output"
        and manifest_by_path[path.resolve().as_posix()].get("sha256") == _sha256(path)
        for path in (amendment_path, reconciliation_path)
    )
    amendment_ok = bool(
        amendment.get("classification")
        == "reporting_only_execution_amendment_after_opened_partial_trial"
        and amendment.get("producer_rerun_performed") is False
        and amendment.get("ic_recomputed_by_recovery") is False
        and amendment.get("decile_returns_recomputed_by_recovery") is False
        and amendment.get("all_replayed_frozen_values_unchanged") is True
        and int(
            (amendment.get("trial_accounting") or {}).get(
                "reporting_recovery_tests_added", -1
            )
        )
        == 0
    )
    reconciliation_ok = bool(
        reconciliation.get("all_extracted_companyfacts_payloads_reconciled") is True
        and int(reconciliation.get("mismatched_companyfacts_payloads", -1)) == 0
        and int(reconciliation.get("missing_companyfacts_payloads", -1)) == 0
        and int(reconciliation.get("checked_companyfacts_payloads", -1))
        == companyfacts_records
        and int(reconciliation.get("inventory_manifest_records_verified", -1))
        == inventory_records
        and int(reconciliation.get("attempt1_checkpoint_records_verified", -1))
        == len(checkpoint_records)
        and (reconciliation.get("attempt1_alignment_replay") or {}).get(
            "all_frozen_values_unchanged"
        )
        is True
    )
    passed = bool(
        checkpoint_ok
        and len(checkpoint_records) == 13
        and checkpoint_bad == 0
        and not unexpected_recovered_changes
        and panel_key_match
        and column_checks
        and all(column_checks.values())
        and disclosure_records_ok
        and amendment_ok
        and reconciliation_ok
    )
    return {
        "applicable": True,
        "passed": passed,
        "execution_status": status,
        "checkpoint_manifest": checkpoint_manifest_path.resolve().as_posix(),
        "checkpoint_records": len(checkpoint_records),
        "checkpoint_bad_records": checkpoint_bad,
        "changed_recovered_checkpoint_files": sorted(changed_recovered_files),
        "unexpected_recovered_checkpoint_changes": sorted(
            unexpected_recovered_changes
        ),
        "panel_key_match": panel_key_match,
        "frozen_value_columns_checked": len(column_checks),
        "frozen_value_column_mismatches": sorted(
            column for column, matched in column_checks.items() if not matched
        ),
        "manifest_raw_companyfacts_records": companyfacts_records,
        "manifest_raw_inventory_records": inventory_records,
        "execution_amendment_manifested_and_consistent": disclosure_records_ok
        and amendment_ok,
        "source_reconciliation_manifested_and_consistent": disclosure_records_ok
        and reconciliation_ok,
        "original_attempt_post_run_toctou_completed": False,
        "scope_note": (
            "Current payload reconciliation plus a hash-locked checkpoint; "
            "the interrupted producer's unobserved interval is not certified."
        ),
    }
def _fiscal_and_clock_audit(
    releases: pd.DataFrame, panel: pd.DataFrame, *, max_age_days: int, latency_minutes: int
) -> dict[str, object]:
    _require_columns(
        releases,
        [
            "ticker",
            "accession",
            "available_at",
            "fiscal_end",
            "filed",
            "release_kind",
            "supersession_only",
            "header_inventory_verified",
            "barrier_inventory_source",
            *PRIMARY_COMPONENTS,
        ],
        "release table",
    )
    _require_columns(
        panel,
        [
            "ticker",
            "signal_date",
            "entry_at",
            "decision_at",
            "quarterly_available_at",
            "accession",
            "quarterly_feature_age_days",
            "quarterly_feature_eligible",
            PRIMARY,
            *PRIMARY_COMPONENTS,
        ],
        "monthly panel",
    )
    r = releases.copy()
    p = panel.copy()
    r["available_at"] = _utc(r["available_at"])
    r["fiscal_end"] = _naive_dates(r["fiscal_end"])
    r["filed"] = _naive_dates(r["filed"])
    p["entry_at"] = _utc(p["entry_at"])
    p["decision_at"] = _utc(p["decision_at"])
    p["quarterly_available_at"] = _utc(p["quarterly_available_at"])

    duplicate_release_keys = int(
        r.duplicated(["ticker", "available_at", "accession"]).sum()
    )
    duplicate_release_accessions = int(r.duplicated(["ticker", "accession"]).sum())
    duplicate_panel_keys = int(p.duplicated(["signal_date", "ticker"]).sum())
    missing_release_clocks = int(r["available_at"].isna().sum() + r["filed"].isna().sum())
    missing_panel_clocks = int(p["entry_at"].isna().sum() + p["decision_at"].isna().sum())
    fiscal_regressions = 0
    for _, company in r.sort_values(
        ["ticker", "available_at", "fiscal_end", "accession"], kind="mergesort"
    ).groupby("ticker", sort=True):
        running: pd.Timestamp | None = None
        for value in company["fiscal_end"]:
            if pd.isna(value):
                continue
            value = pd.Timestamp(value)
            if running is not None and value < running:
                fiscal_regressions += 1
            running = value if running is None else max(running, value)

    known_fiscal = r["fiscal_end"].notna()
    release_before_period_end = int(
        (
            r.loc[known_fiscal, "available_at"]
            <= r.loc[known_fiscal, "fiscal_end"].dt.tz_localize("UTC")
        ).sum()
    )
    lag = (r["filed"] - r["fiscal_end"]).dt.days
    lag_error = int((known_fiscal & pd.to_numeric(r["release_lag_days"], errors="coerce").ne(lag)).sum()) if "release_lag_days" in r else int(known_fiscal.sum())
    negative_release_lags = int((known_fiscal & lag.lt(0)).sum())
    available_before_filed_date = int(
        (
            r["available_at"].notna()
            & r["filed"].notna()
            & r["available_at"].lt(r["filed"].dt.tz_localize("UTC"))
        ).sum()
    )
    provenance_future_ends = 0
    for column in [name for name in r if name.endswith("_end") and name != "fiscal_end"]:
        values = _naive_dates(r[column])
        provenance_future_ends += int((values.notna() & values.gt(r["filed"])).sum())

    expected_decision = p["entry_at"] - pd.Timedelta(minutes=latency_minutes)
    decision_clock_mismatches = int(p["decision_at"].ne(expected_decision).sum())
    valid_decisions = p["decision_at"].notna() & expected_decision.notna()
    decision_error = (
        float(
            np.max(
                np.abs(
                    p.loc[valid_decisions, "decision_at"].astype("int64").to_numpy()
                    - expected_decision.loc[valid_decisions].astype("int64").to_numpy()
                )
            )
        )
        if valid_decisions.any()
        else float("inf")
    )
    known = p["quarterly_available_at"].notna()
    future_panel_rows = int(
        (p.loc[known, "quarterly_available_at"] >= p.loc[known, "decision_at"]).sum()
    )
    computed_age = (
        p["decision_at"] - p["quarterly_available_at"]
    ).dt.total_seconds() / 86_400.0
    age_error = _max_abs_error(
        p.loc[known, "quarterly_feature_age_days"], computed_age.loc[known]
    )

    # Independently select the latest strictly public state for every decision.
    asof_mismatches = 0
    for ticker, decisions in p.groupby("ticker", sort=True):
        company = r.loc[r["ticker"].eq(ticker)].sort_values(
            ["available_at", "accession"], kind="mergesort"
        )
        release_times = (
            company["available_at"].astype("int64").to_numpy()
            if not company.empty
            else np.array([], dtype=np.int64)
        )
        for index, row in decisions.iterrows():
            position = int(
                np.searchsorted(
                    release_times, pd.Timestamp(row["decision_at"]).value, side="left"
                )
                - 1
            )
            if position >= 0:
                selected = company.iloc[position]
                if (
                    pd.isna(row["quarterly_available_at"])
                    or str(row["accession"]) != str(selected["accession"])
                    or pd.Timestamp(row["quarterly_available_at"])
                    != pd.Timestamp(selected["available_at"])
                ):
                    asof_mismatches += 1
            elif pd.notna(row["quarterly_available_at"]):
                asof_mismatches += 1

    barrier_accessions = set(
        map(
            tuple,
            r.loc[r["supersession_only"].fillna(True), ["ticker", "accession"]]
            .astype(str)
            .itertuples(index=False, name=None),
        )
    )
    aligned_to_barrier = p.apply(
        lambda row: (str(row["ticker"]), str(row["accession"])) in barrier_accessions,
        axis=1,
    ) & known
    barrier_score_leaks = int(p.loc[aligned_to_barrier, PRIMARY].notna().sum())
    complete_release = r[list(PRIMARY_COMPONENTS)].notna().all(axis=1)
    bad_supersession_flag = int(
        r["supersession_only"].fillna(True).ne(~complete_release).sum()
    )
    named_barrier = r["release_kind"].astype(str).str.contains("barrier")
    named_barrier_not_null = int(
        (named_barrier & ~r["supersession_only"].fillna(True)).sum()
    )
    header_failures = int((~r["header_inventory_verified"].fillna(False)).sum())
    header_source_failures = int(
        r["barrier_inventory_source"].astype(str).ne("submissions_header").sum()
    )
    stale_score_leaks = int(
        (
            known
            & computed_age.gt(max_age_days)
            & p[PRIMARY].notna()
        ).sum()
    )
    return {
        "passed": all(
            value == 0
            for value in (
                duplicate_release_keys,
                duplicate_release_accessions,
                duplicate_panel_keys,
                missing_release_clocks,
                missing_panel_clocks,
                fiscal_regressions,
                release_before_period_end,
                lag_error,
                negative_release_lags,
                available_before_filed_date,
                provenance_future_ends,
                future_panel_rows,
                asof_mismatches,
                barrier_score_leaks,
                bad_supersession_flag,
                named_barrier_not_null,
                header_failures,
                header_source_failures,
                stale_score_leaks,
                decision_clock_mismatches,
            )
        )
        and decision_error == 0.0
        and age_error <= DEFAULT_TOLERANCE,
        "duplicate_release_keys": duplicate_release_keys,
        "duplicate_release_accessions": duplicate_release_accessions,
        "duplicate_panel_signal_ticker_keys": duplicate_panel_keys,
        "missing_release_clocks": missing_release_clocks,
        "missing_panel_clocks": missing_panel_clocks,
        "fiscal_regressions": fiscal_regressions,
        "release_not_after_fiscal_end": release_before_period_end,
        "release_lag_mismatches": lag_error,
        "negative_release_lags": negative_release_lags,
        "available_before_filed_date": available_before_filed_date,
        "provenance_fact_ends_after_filing": provenance_future_ends,
        "decision_clock_error_nanoseconds": decision_error,
        "decision_clock_mismatches": decision_clock_mismatches,
        "future_or_simultaneous_panel_states": future_panel_rows,
        "maximum_feature_age_error_days": age_error,
        "asof_state_mismatches": asof_mismatches,
        "barrier_score_leaks": barrier_score_leaks,
        "supersession_flag_mismatches": bad_supersession_flag,
        "named_barrier_complete_rows": named_barrier_not_null,
        "unverified_header_rows": header_failures,
        "non_submissions_header_rows": header_source_failures,
        "stale_score_leaks": stale_score_leaks,
    }


def _compatible_context_audit(releases: pd.DataFrame) -> dict[str, object]:
    r = releases.copy()
    r["fiscal_end"] = _naive_dates(r["fiscal_end"])
    r["filed"] = _naive_dates(r["filed"])
    failures: list[str] = []
    q4_arithmetic_errors: list[float] = []
    checked_pairs = 0
    checked_q4_pairs = 0
    context_level_failures = int(
        releases["context_validation_level"]
        .astype(str)
        .ne("companyfacts_compatible_no_contextref")
        .sum()
    ) if "context_validation_level" in releases else len(releases)

    for row_index, row in r.iterrows():
        row_id = f"{row.get('ticker')}:{row.get('accession')}:{row_index}"
        for concept in CONCEPTS:
            current = _finite_number(row.get(f"{concept}_current"))
            prior = _finite_number(row.get(f"{concept}_prior_year"))
            if current is None and prior is None:
                continue
            checked_pairs += 1
            if current is None or prior is None:
                failures.append(f"{row_id}:{concept}:one-sided-pair")
                continue
            tag = row.get(f"{concept}_source_tag")
            prior_tag = row.get(f"{concept}_prior_source_tag")
            if (
                not bool(row.get(f"{concept}_same_tag_comparator"))
                or not isinstance(tag, str)
                or tag != prior_tag
            ):
                failures.append(f"{row_id}:{concept}:cross-tag")

            form = str(row.get("form", ""))
            if form.startswith("10-Q"):
                accessions = {
                    str(row.get(f"{concept}_current_fact_accession")),
                    str(row.get(f"{concept}_prior_fact_accession")),
                }
                units = {
                    str(row.get(f"{concept}_current_fact_unit")),
                    str(row.get(f"{concept}_prior_fact_unit")),
                }
                if accessions != {str(row.get("accession"))} or units != {"USD"}:
                    failures.append(f"{row_id}:{concept}:10q-provenance")
                current_start = pd.to_datetime(
                    row.get(f"{concept}_current_fact_start"), errors="coerce"
                )
                current_end = pd.to_datetime(
                    row.get(f"{concept}_current_fact_end"), errors="coerce"
                )
                prior_start = pd.to_datetime(
                    row.get(f"{concept}_prior_fact_start"), errors="coerce"
                )
                prior_end = pd.to_datetime(
                    row.get(f"{concept}_prior_fact_end"), errors="coerce"
                )
                if (
                    pd.isna(current_start)
                    or pd.isna(current_end)
                    or pd.isna(prior_start)
                    or pd.isna(prior_end)
                    or current_end != row["fiscal_end"]
                    or not 70 <= (current_end - current_start).days <= 120
                    or not 350 <= (current_end - prior_end).days <= 380
                    or abs(
                        (current_end - current_start).days
                        - (prior_end - prior_start).days
                    )
                    > 8
                ):
                    failures.append(f"{row_id}:{concept}:10q-period")
                continue

            if not form.startswith("10-K"):
                failures.append(f"{row_id}:{concept}:unexpected-form")
                continue
            checked_q4_pairs += 1
            method = row.get(f"{concept}_q4_method")
            if method == "direct_10k":
                accessions = {
                    str(row.get(f"{concept}_current_fact_accession")),
                    str(row.get(f"{concept}_prior_fact_accession")),
                }
                units = {
                    str(row.get(f"{concept}_current_fact_unit")),
                    str(row.get(f"{concept}_prior_fact_unit")),
                }
                starts = [
                    pd.to_datetime(row.get(f"{concept}_{leg}_start"), errors="coerce")
                    for leg in ("current_fact", "prior_fact")
                ]
                ends = [
                    pd.to_datetime(row.get(f"{concept}_{leg}_end"), errors="coerce")
                    for leg in ("current_fact", "prior_fact")
                ]
                durations = [
                    (ends[index] - starts[index]).days
                    if pd.notna(ends[index]) and pd.notna(starts[index])
                    else None
                    for index in range(2)
                ]
                if (
                    accessions != {str(row.get("accession"))}
                    or units != {"USD"}
                    or any(value is None or not 70 <= value <= 120 for value in durations)
                    or ends[0] != row["fiscal_end"]
                    or pd.isna(ends[1])
                    or not 350 <= (ends[0] - ends[1]).days <= 380
                    or abs(int(durations[0]) - int(durations[1])) > 5
                    or _finite_number(row.get(f"{concept}_q4_days_current"))
                    != durations[0]
                    or _finite_number(row.get(f"{concept}_q4_days_prior"))
                    != durations[1]
                    or _finite_number(row.get(f"{concept}_annual")) is not None
                    or _finite_number(row.get(f"{concept}_q3_ytd")) is not None
                ):
                    failures.append(f"{row_id}:{concept}:direct-q4-provenance")
            elif method == "same_10k_derived":
                legs = ("annual_current", "annual_prior", "q3_ytd_current", "q3_ytd_prior")
                accessions = {
                    str(row.get(f"{concept}_{leg}_accession")) for leg in legs
                }
                units = {str(row.get(f"{concept}_{leg}_unit")) for leg in legs}
                starts = {
                    leg: pd.to_datetime(
                        row.get(f"{concept}_{leg}_start"), errors="coerce"
                    )
                    for leg in legs
                }
                ends = {
                    leg: pd.to_datetime(
                        row.get(f"{concept}_{leg}_end"), errors="coerce"
                    )
                    for leg in legs
                }
                annual = _finite_number(row.get(f"{concept}_annual"))
                annual_prior = _finite_number(row.get(f"{concept}_annual_prior_year"))
                ytd = _finite_number(row.get(f"{concept}_q3_ytd"))
                ytd_prior = _finite_number(row.get(f"{concept}_q3_ytd_prior_year"))
                values_complete = all(
                    value is not None for value in (annual, annual_prior, ytd, ytd_prior)
                )
                if values_complete:
                    current_error = abs(current - (float(annual) - float(ytd)))
                    prior_error = abs(prior - (float(annual_prior) - float(ytd_prior)))
                    q4_arithmetic_errors.extend([current_error, prior_error])
                periods_valid = False
                if all(pd.notna(starts[leg]) and pd.notna(ends[leg]) for leg in legs):
                    annual_days = (ends["annual_current"] - starts["annual_current"]).days
                    annual_prior_days = (
                        ends["annual_prior"] - starts["annual_prior"]
                    ).days
                    ytd_days = (ends["q3_ytd_current"] - starts["q3_ytd_current"]).days
                    ytd_prior_days = (
                        ends["q3_ytd_prior"] - starts["q3_ytd_prior"]
                    ).days
                    q4_days = (
                        ends["annual_current"] - ends["q3_ytd_current"]
                    ).days
                    prior_q4_days = (
                        ends["annual_prior"] - ends["q3_ytd_prior"]
                    ).days
                    periods_valid = bool(
                        ends["annual_current"] == row["fiscal_end"]
                        and 350 <= annual_days <= 380
                        and 350 <= annual_prior_days <= 380
                        and abs(annual_days - annual_prior_days) <= 8
                        and 230 <= ytd_days <= 310
                        and 230 <= ytd_prior_days <= 310
                        and abs(ytd_days - ytd_prior_days) <= 8
                        and 350
                        <= (ends["annual_current"] - ends["annual_prior"]).days
                        <= 380
                        and 350
                        <= (starts["annual_current"] - starts["annual_prior"]).days
                        <= 380
                        and abs(
                            (starts["annual_current"] - starts["q3_ytd_current"]).days
                        )
                        <= 1
                        and abs(
                            (starts["annual_prior"] - starts["q3_ytd_prior"]).days
                        )
                        <= 1
                        and 350
                        <= (ends["q3_ytd_current"] - ends["q3_ytd_prior"]).days
                        <= 380
                        and 350
                        <= (starts["q3_ytd_current"] - starts["q3_ytd_prior"]).days
                        <= 380
                        and 70 <= q4_days <= 120
                        and 70 <= prior_q4_days <= 120
                        and abs(q4_days - prior_q4_days) <= 5
                        and _finite_number(row.get(f"{concept}_q4_days_current"))
                        == q4_days
                        and _finite_number(row.get(f"{concept}_q4_days_prior"))
                        == prior_q4_days
                    )
                available_at = pd.to_datetime(row.get("available_at"), utc=True, errors="coerce")
                q3_available_at = pd.to_datetime(
                    row.get(f"{concept}_q3_ytd_available_at"),
                    utc=True,
                    errors="coerce",
                )
                annual_available_at = pd.to_datetime(
                    row.get("annual_available_at"), utc=True, errors="coerce"
                )
                if (
                    accessions != {str(row.get("accession"))}
                    or str(row.get(f"{concept}_q3_ytd_accession"))
                    != str(row.get("accession"))
                    or units != {"USD"}
                    or not values_complete
                    or not periods_valid
                    or pd.isna(available_at)
                    or q3_available_at != available_at
                    or annual_available_at != available_at
                ):
                    failures.append(f"{row_id}:{concept}:derived-q4-provenance")
            else:
                failures.append(f"{row_id}:{concept}:missing-q4-method")

    # Independently recompute the persisted accounting features.
    feature_errors: dict[str, float] = {}
    assets = pd.to_numeric(r["assets_current"], errors="coerce")
    ni_now = pd.to_numeric(r["net_income_current"], errors="coerce")
    ni_prior = pd.to_numeric(r["net_income_prior_year"], errors="coerce")
    rev_now = pd.to_numeric(r["revenue_current"], errors="coerce")
    rev_prior = pd.to_numeric(r["revenue_prior_year"], errors="coerce")
    gp_now = pd.to_numeric(r["gross_profit_current"], errors="coerce")
    gp_prior = pd.to_numeric(r["gross_profit_prior_year"], errors="coerce")
    expected_earnings = ((ni_now - ni_prior) / assets).where(
        ni_now.notna() & ni_prior.notna() & assets.gt(0)
    )
    expected_sales = (rev_now / rev_prior - 1.0).where(
        rev_now.notna() & rev_prior.gt(0)
    )
    expected_margin = (gp_now / rev_now - gp_prior / rev_prior).where(
        gp_now.notna() & gp_prior.notna() & rev_now.gt(0) & rev_prior.gt(0)
    )
    for column, expected in (
        ("earnings_change_on_assets", expected_earnings),
        ("sales_growth_yoy", expected_sales),
        ("gross_margin_change_yoy", expected_margin),
    ):
        feature_errors[column] = _max_abs_error(r[column], expected)
    maximum_q4_error = max(q4_arithmetic_errors, default=0.0)
    maximum_feature_error = max(feature_errors.values(), default=0.0)
    return {
        "passed": not failures
        and context_level_failures == 0
        and maximum_q4_error <= DEFAULT_TOLERANCE
        and maximum_feature_error <= DEFAULT_TOLERANCE,
        "checked_concept_pairs": checked_pairs,
        "checked_q4_concept_pairs": checked_q4_pairs,
        "provenance_failures": len(failures),
        "context_validation_level_failures": context_level_failures,
        "failure_examples": failures[:10],
        "maximum_q4_arithmetic_error": maximum_q4_error,
        "maximum_feature_formula_error": maximum_feature_error,
        "feature_formula_errors": feature_errors,
    }


def _issuer_and_score_audit(panel: pd.DataFrame) -> dict[str, object]:
    p = panel.copy()
    p["signal_date"] = _utc(p["signal_date"])
    _require_columns(
        p,
        [
            "signal_date",
            "ticker",
            "issuer_cik",
            "issuer_representative",
            "adv_usd_3m",
            PRIMARY,
            *PRIMARY_COMPONENTS,
        ],
        "monthly panel",
    )
    expected_representative = pd.Series(True, index=p.index)
    has_issuer = p["issuer_cik"].notna()
    if has_issuer.any():
        candidates = p.loc[
            has_issuer, ["signal_date", "ticker", "issuer_cik", "adv_usd_3m"]
        ].copy()
        candidates["_adv"] = pd.to_numeric(
            candidates["adv_usd_3m"], errors="coerce"
        ).fillna(-np.inf)
        selected = (
            candidates.sort_values(
                ["signal_date", "issuer_cik", "_adv", "ticker"],
                ascending=[True, True, False, True],
                kind="mergesort",
            )
            .drop_duplicates(["signal_date", "issuer_cik"], keep="first")
            .index
        )
        expected_representative.loc[has_issuer & ~p.index.isin(selected)] = False
    observed_representative = p["issuer_representative"].fillna(False).astype(bool)
    representative_mismatches = int(
        observed_representative.ne(expected_representative).sum()
    )
    nonrep_feature_leaks = int(
        p.loc[~expected_representative, list(RELEASE_FEATURES)].notna().any(axis=1).sum()
    )

    complete = p[list(PRIMARY_COMPONENTS)].notna().all(axis=1)
    expected_score = pd.Series(np.nan, index=p.index, dtype=float)
    for _, month in p.loc[complete].groupby("signal_date", sort=True):
        earnings_rank = month[PRIMARY_COMPONENTS[0]].rank(method="average", pct=True)
        sales_rank = month[PRIMARY_COMPONENTS[1]].rank(method="average", pct=True)
        expected_score.loc[month.index] = (earnings_rank + sales_rank) / 2.0
    score_error = _max_abs_error(p[PRIMARY], expected_score)
    eligibility_mismatches = int(
        p["quarterly_feature_eligible"].fillna(False).astype(bool).ne(complete).sum()
    ) if "quarterly_feature_eligible" in p else len(p)
    duplicated_eligible_issuers = int(
        p.loc[p[PRIMARY].notna() & p["issuer_cik"].notna()]
        .duplicated(["signal_date", "issuer_cik"])
        .sum()
    )
    return {
        "passed": representative_mismatches == 0
        and nonrep_feature_leaks == 0
        and score_error <= DEFAULT_TOLERANCE
        and eligibility_mismatches == 0
        and duplicated_eligible_issuers == 0,
        "representative_selection_mismatches": representative_mismatches,
        "nonrepresentative_feature_leaks": nonrep_feature_leaks,
        "maximum_common_intersection_score_error": score_error,
        "eligibility_flag_mismatches": eligibility_mismatches,
        "duplicated_eligible_issuer_months": duplicated_eligible_issuers,
    }


def verify(artifact: Path, *, tolerance: float = DEFAULT_TOLERANCE) -> dict[str, object]:
    artifact = artifact.resolve()
    full_required_files = [
        "artifact_manifest.json",
        "config_snapshot.yml",
        "quarterly_release_features.parquet",
        "monthly_quarterly_signal_panel.parquet",
        "monthly_feature_ic.csv",
        "feature_period_summary.csv",
        "decile_monthly.csv",
        "decile_summary.csv",
        "run_summary.json",
        "opened_trial_ledger.json",
        "toctou_check.json",
        "release_consistency_summary.json",
        "report-source.md",
        "source_audit.csv",
        "feature_coverage_by_month.csv",
        "gates.csv",
        "manual_reconstruction.json",
        "charts/incremental_ic.png",
        "charts/pre2026_deciles.png",
    ]
    reconstruction_required_files = [
        "config_snapshot.yml",
        "quarterly_release_features.parquet",
        "monthly_quarterly_signal_panel.parquet",
        "monthly_feature_ic.csv",
        "feature_period_summary.csv",
        "decile_monthly.csv",
        "decile_summary.csv",
    ]
    missing_full = [
        name for name in full_required_files if not (artifact / name).is_file()
    ]
    missing_reconstruction = [
        name
        for name in reconstruction_required_files
        if not (artifact / name).is_file()
    ]
    inventory = _file_inventory(artifact) if artifact.is_dir() else {
        "file_count": 0,
        "total_bytes": 0,
        "aggregate_inventory_sha256": None,
        "records": [],
    }
    if missing_reconstruction:
        return {
            "artifact": artifact.as_posix(),
            "verification_passed": False,
            "artifact_verification_status": "FAIL",
            "partial_artifact": True,
            "missing_full_artifact_files": missing_full,
            "fatal_errors": [
                f"missing reconstruction-critical files: {missing_reconstruction}"
            ],
            "research_verdict_unchanged": "NOT VALIDATED",
            "artifact_file_inventory": inventory,
        }

    config = yaml.safe_load((artifact / "config_snapshot.yml").read_text(encoding="utf-8"))
    controls = list(config["evaluation"]["controls"])
    minimum_names = int(config["evaluation"]["minimum_names_per_month"])
    trial_floor = int(config["evaluation"]["resulting_global_trial_floor"])
    max_age_days = int(config["clock"]["maximum_feature_age_days"])
    latency_minutes = int(config["clock"]["operational_latency_minutes_before_entry"])
    persisted_panel = pd.read_parquet(
        artifact / "monthly_quarterly_signal_panel.parquet"
    )
    releases = pd.read_parquet(artifact / "quarterly_release_features.parquet")
    panel, panel_enrichment = _enrich_partial_panel(
        persisted_panel, releases, config
    )

    manifest = _manifest_audit(artifact)
    fiscal_clock = _fiscal_and_clock_audit(
        releases,
        panel,
        max_age_days=max_age_days,
        latency_minutes=latency_minutes,
    )
    compatible_context = _compatible_context_audit(releases)
    issuer_score = _issuer_and_score_audit(panel)
    reconstructed_ic, reconstructed_deciles = reconstruct_monthly_statistics(
        panel, controls=controls, minimum_names=minimum_names
    )
    reconstructed_summary = summarize_ic(reconstructed_ic, trial_floor)
    reconstructed_decile_summary = summarize_deciles(reconstructed_deciles)

    monthly_ic_comparison = _compare_table(
        pd.read_csv(artifact / "monthly_feature_ic.csv"),
        reconstructed_ic,
        keys=["signal_date", "feature", "names"],
        numeric=["raw_rank_ic", "incremental_rank_ic"],
        label="monthly IC",
    )
    monthly_decile_comparison = _compare_table(
        pd.read_csv(artifact / "decile_monthly.csv"),
        reconstructed_deciles,
        keys=["signal_date", "feature", "decile", "names"],
        numeric=["equal_weight_return"],
        label="monthly deciles",
    )
    observed_summary = pd.read_csv(artifact / "feature_period_summary.csv")
    summary_columns = [
        "months",
        "minimum_names",
        "median_names",
        "maximum_names",
        "mean_raw_rank_ic",
        "raw_ic_hac_tstat",
        "mean_incremental_rank_ic",
        "incremental_ic_hac_tstat",
        "positive_month_fraction",
        "positive_year_fraction",
        "one_sided_hac_pvalue",
        "bh_qvalue_pre2026_family",
        "global_bonferroni_pvalue_floor_110",
    ]
    summary_comparison = _compare_table(
        observed_summary.loc[observed_summary["feature"].eq(PRIMARY)],
        reconstructed_summary,
        keys=["feature", "period"],
        numeric=summary_columns,
        label="IC period summary and HAC",
    )
    decile_summary_comparison = _compare_table(
        pd.read_csv(artifact / "decile_summary.csv"),
        reconstructed_decile_summary,
        keys=["feature", "period", "decile"],
        numeric=["months", "annualized_arithmetic_return", "hac_tstat"],
        label="decile period summary and HAC",
    )
    metric_comparisons = [
        monthly_ic_comparison,
        monthly_decile_comparison,
        summary_comparison,
        decile_summary_comparison,
    ]
    metrics_passed = all(
        comparison.get("keys_match") is True
        and comparison.get("columns_match") is True
        and float(comparison.get("maximum_absolute_error", float("inf"))) <= tolerance
        for comparison in metric_comparisons
    )

    run_summary_path = artifact / "run_summary.json"
    run_summary = (
        json.loads(run_summary_path.read_text(encoding="utf-8"))
        if run_summary_path.is_file()
        else {}
    )
    primary_pre = reconstructed_summary.loc[
        reconstructed_summary["period"].eq("pre_2026")
    ]
    run_summary_errors: dict[str, float] = {}
    if len(primary_pre) == 1:
        row = primary_pre.iloc[0]
        mapping = {
            "primary_pre2026_months": "months",
            "primary_mean_raw_rank_ic": "mean_raw_rank_ic",
            "primary_mean_incremental_rank_ic": "mean_incremental_rank_ic",
            "primary_incremental_ic_hac_tstat": "incremental_ic_hac_tstat",
            "primary_raw_one_sided_pvalue": "one_sided_hac_pvalue",
            "primary_global_bonferroni_pvalue": "global_bonferroni_pvalue_floor_110",
        }
        for output_name, reconstructed_name in mapping.items():
            if output_name not in run_summary:
                run_summary_errors[output_name] = float("inf")
            else:
                left = _finite_number(run_summary[output_name])
                right = _finite_number(row[reconstructed_name])
                run_summary_errors[output_name] = (
                    abs(float(left) - float(right))
                    if left is not None and right is not None
                    else (0.0 if left is None and right is None else float("inf"))
                )
    else:
        run_summary_errors["primary_pre2026_row_count"] = float("inf")
    run_summary_passed = (
        run_summary_path.is_file()
        and bool(run_summary_errors)
        and max(run_summary_errors.values(), default=0.0) <= tolerance
        and run_summary.get("historical_backtest_verdict") == "INVALID"
        and run_summary.get("portfolio_authorized") is False
        and int(run_summary.get("global_trial_floor", -1)) == trial_floor
    )
    recovery_lineage = _recovery_lineage_audit(artifact, run_summary)

    completeness_passed = not missing_full
    checks = [
        {
            "check": "complete_producer_artifact",
            "passed": completeness_passed,
            "evidence": {
                "required_files": len(full_required_files),
                "missing_files": missing_full,
            },
        },
        {
            "check": "persisted_v2_panel_audit_columns",
            "passed": bool(panel_enrichment["passed_without_enrichment"]),
            "evidence": panel_enrichment,
        },
        {
            "check": "manifest_hashes_and_output_closure",
            "passed": bool(manifest["passed"]),
            "evidence": manifest,
        },
        {
            "check": "reporting_recovery_lineage",
            "passed": bool(recovery_lineage["passed"]),
            "evidence": recovery_lineage,
        },
        {
            "check": "filing_state_clock_and_null_barriers",
            "passed": bool(fiscal_clock["passed"]),
            "evidence": fiscal_clock,
        },
        {
            "check": "same_tag_and_q4_compatible_context_provenance",
            "passed": bool(compatible_context["passed"]),
            "evidence": compatible_context,
        },
        {
            "check": "common_issuer_intersection_ranking",
            "passed": bool(issuer_score["passed"]),
            "evidence": issuer_score,
        },
        {
            "check": "independent_ic_decile_hac_reconstruction",
            "passed": metrics_passed,
            "evidence": metric_comparisons,
        },
        {
            "check": "run_summary_reconciliation",
            "passed": run_summary_passed,
            "evidence": {
                "maximum_absolute_error": max(
                    run_summary_errors.values(), default=0.0
                ),
                "field_errors": run_summary_errors,
                "historical_backtest_verdict": run_summary.get(
                    "historical_backtest_verdict"
                ),
                "portfolio_authorized": run_summary.get("portfolio_authorized"),
            },
        },
    ]
    passed = all(bool(check["passed"]) for check in checks)
    primary_evidence: dict[str, object] = {}
    if len(primary_pre) == 1:
        primary_evidence = {
            key: (
                value.item() if isinstance(value, np.generic) else value
            )
            for key, value in primary_pre.iloc[0].to_dict().items()
        }
    research_verdict = (
        run_summary.get("historical_backtest_verdict")
        if run_summary_path.is_file()
        else "NOT VALIDATED"
    )
    return {
        "artifact": artifact.as_posix(),
        "verification_passed": passed,
        "artifact_verification_status": "PASS" if passed else "FAIL",
        "partial_artifact": bool(missing_full),
        "missing_full_artifact_files": missing_full,
        "tolerance": tolerance,
        "research_verdict_unchanged": research_verdict,
        "portfolio_authorized": run_summary.get("portfolio_authorized"),
        "release_rows": int(len(releases)),
        "panel_rows": int(len(persisted_panel)),
        "independently_reconstructed_monthly_ic_rows": int(len(reconstructed_ic)),
        "independently_reconstructed_monthly_decile_rows": int(
            len(reconstructed_deciles)
        ),
        "primary_pre2026_reconstruction": primary_evidence,
        "checks": checks,
        "artifact_file_inventory": inventory,
        "limitations": [
            "This verifies persisted compatible-context evidence, not raw XBRL contextRef or dimensions.",
            "It cannot repair the current-survivor universe, ticker lineage, or missing delisting returns.",
            "It does not validate execution costs, capacity, borrow, or deployability.",
            "The historical period remains opened; reconstruction is not a new out-of-sample test.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("artifacts/quarterly_fundamental_momentum_diagnostic_v2"),
    )
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.artifact, tolerance=args.tolerance)
    serialized = json.dumps(
        _json_safe(result),
        indent=2,
        sort_keys=True,
        default=str,
        allow_nan=False,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    raise SystemExit(0 if result["verification_passed"] else 1)


if __name__ == "__main__":
    main()
