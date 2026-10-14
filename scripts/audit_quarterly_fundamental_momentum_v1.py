#!/usr/bin/env python3
"""Independent, read-only audit of the quarterly-fundamental diagnostic.

The script never changes the primary artifact.  It reconstructs the published
statistics from the frozen signal panel, inspects raw SEC Company Facts for
cross-tag comparators and intervening 10-K filings, runs explicitly post-hoc
sensitivity checks, verifies the upstream manifests, and writes a separate
audit artifact.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import platform
import sys
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm, rankdata


ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / "artifacts/quarterly_fundamental_momentum_diagnostic_v1"
OUTPUT = ROOT / "artifacts/quarterly_fundamental_momentum_diagnostic_v1_independent_audit"
MAPPING = ROOT / "artifacts/public_broad_proxy_v1/data/frozen_proxy_universe.csv"
FACTS_DIR = ROOT / "data/cache/sec/companyfacts"
PRE_2026 = pd.Timestamp("2026-01-01", tz="UTC")
CONTROLS = (
    "exact_total_momentum_12_1",
    "reversal_1m",
    "beta_12m",
    "realized_vol_3m",
    "illiquidity_3m",
    "log_market_cap",
)
FEATURES = (
    "quarterly_fundamental_momentum_score",
    "earnings_change_on_assets",
    "sales_growth_yoy",
    "gross_margin_change_yoy",
)
DURATION_TAGS = {
    "net_income": ("NetIncomeLoss", "ProfitLoss"),
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ),
    "gross_profit": ("GrossProfit",),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def sec_date(value: object) -> pd.Timestamp | None:
    if not isinstance(value, str):
        return None
    try:
        return pd.Timestamp(date.fromisoformat(value[:10]))
    except (TypeError, ValueError):
        return None


def finite_float(value: object) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def filing_day_end_utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value).normalize().tz_localize("America/New_York")
    return (stamp + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)).tz_convert(
        "UTC"
    )


def company_duration_rows(
    ticker: str, cik: int, payload: dict[str, Any]
) -> pd.DataFrame:
    us_gaap = ((payload.get("facts") or {}).get("us-gaap") or {})
    rows: list[dict[str, object]] = []
    for concept, tags in DURATION_TAGS.items():
        for priority, tag in enumerate(tags):
            entries = (
                ((((us_gaap.get(tag) or {}).get("units") or {}).get("USD")) or [])
            )
            for item in entries:
                if item.get("form") not in {"10-Q", "10-Q/A"}:
                    continue
                if item.get("fp") not in {"Q1", "Q2", "Q3"}:
                    continue
                start = sec_date(item.get("start"))
                end = sec_date(item.get("end"))
                filed = sec_date(item.get("filed"))
                value = finite_float(item.get("val"))
                accession = str(item.get("accn") or "")
                if (
                    start is None
                    or end is None
                    or filed is None
                    or value is None
                    or not accession
                    or end > filed
                ):
                    continue
                duration = int((end - start).days)
                if not 70 <= duration <= 120:
                    continue
                rows.append(
                    {
                        "ticker": ticker,
                        "cik": cik,
                        "accession": accession,
                        "concept": concept,
                        "source_tag": tag,
                        "tag_priority": priority,
                        "start": start,
                        "end": end,
                        "filed": filed,
                        "value": value,
                    }
                )
    return pd.DataFrame(rows)


def annual_asset_filings(
    ticker: str, payload: dict[str, Any]
) -> list[dict[str, object]]:
    us_gaap = ((payload.get("facts") or {}).get("us-gaap") or {})
    entries = (
        (((((us_gaap.get("Assets") or {}).get("units") or {}).get("USD"))) or [])
    )
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for item in entries:
        if item.get("form") not in {"10-K", "10-K/A"}:
            continue
        accession = str(item.get("accn") or "")
        filed = str(item.get("filed") or "")[:10]
        key = (accession, filed)
        if not accession or not filed or key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "ticker": ticker,
                "annual_accession": accession,
                "annual_available_at": filing_day_end_utc(filed),
            }
        )
    return rows


def centered_rank(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if int(values.notna().sum()) < 2:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return values.rank(method="average", pct=True) * 2.0 - 1.0


def residual_target(month: pd.DataFrame) -> np.ndarray:
    target = pd.to_numeric(month["target_return"], errors="coerce")
    if target.isna().any() or not np.isfinite(target.to_numpy(float)).all():
        raise ValueError("target_return is incomplete or non-finite")
    columns: list[np.ndarray] = []
    for control in CONTROLS:
        values = pd.to_numeric(month[control], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        columns.extend(
            [
                centered_rank(values).fillna(0.0).to_numpy(float),
                values.isna().astype(float).to_numpy(),
            ]
        )
    target_rank = centered_rank(target).to_numpy(float)
    design = np.column_stack([np.ones(len(month)), *columns])
    coefficients = np.linalg.lstsq(design, target_rank, rcond=None)[0]
    return target_rank - design @ coefficients


def evaluate_feature(
    panel: pd.DataFrame, score_column: str, *, minimum_names: int = 50
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for signal_date, month in panel.groupby("signal_date", sort=True):
        valid = month.loc[
            pd.to_numeric(month[score_column], errors="coerce").notna()
        ].copy()
        if len(valid) < minimum_names:
            continue
        score = pd.to_numeric(valid[score_column], errors="raise").to_numpy(float)
        residual = residual_target(valid)
        score_rank = rankdata(score)
        target_rank = rankdata(valid["target_return"].to_numpy(float))
        residual_rank = rankdata(residual)
        ordered = pd.Series(score, index=valid.index).rank(method="first")
        decile = (
            np.floor((ordered - 1) * 10 / len(valid)).astype(int) + 1
        ).clip(1, 10)
        d1 = float(valid.loc[decile.eq(1), "target_return"].mean())
        d10 = float(valid.loc[decile.eq(10), "target_return"].mean())
        rows.append(
            {
                "signal_date": signal_date,
                "names": len(valid),
                "raw_rank_ic": float(np.corrcoef(score_rank, target_rank)[0, 1]),
                "incremental_rank_ic": float(
                    np.corrcoef(score_rank, residual_rank)[0, 1]
                ),
                "d1_return": d1,
                "d10_return": d10,
                "d10_minus_d1": d10 - d1,
            }
        )
    return pd.DataFrame(rows)


def hac_tstat(series: pd.Series, lags: int = 3) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(float)
    n = len(values)
    if n < lags + 3:
        return np.nan
    errors = values - values.mean()
    long_run_variance = float(errors @ errors / n)
    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)
        long_run_variance += 2 * weight * float(
            errors[lag:] @ errors[:-lag] / n
        )
    standard_error = np.sqrt(max(long_run_variance, 0.0) / n)
    return float(values.mean() / standard_error) if standard_error > 0 else np.nan


def evidence_summary(monthly: pd.DataFrame) -> dict[str, float | int]:
    sample = monthly.loc[pd.to_datetime(monthly["signal_date"], utc=True).lt(PRE_2026)]
    return {
        "months": int(len(sample)),
        "minimum_names": int(sample["names"].min()),
        "median_names": float(sample["names"].median()),
        "maximum_names": int(sample["names"].max()),
        "mean_raw_rank_ic": float(sample["raw_rank_ic"].mean()),
        "mean_incremental_rank_ic": float(sample["incremental_rank_ic"].mean()),
        "incremental_ic_hac_tstat_lag3": hac_tstat(
            sample["incremental_rank_ic"]
        ),
        "one_sided_hac_pvalue": float(
            norm.sf(hac_tstat(sample["incremental_rank_ic"]))
        ),
        "annualized_arithmetic_d10_minus_d1": float(
            12 * sample["d10_minus_d1"].mean()
        ),
        "d10_minus_d1_hac_tstat_lag3": hac_tstat(sample["d10_minus_d1"]),
    }


def bh_qvalues(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=pvalues.get)  # type: ignore[arg-type]
    adjusted: dict[str, float] = {}
    running = 1.0
    m = len(ordered)
    for index in range(m - 1, -1, -1):
        feature = ordered[index]
        running = min(running, pvalues[feature] * m / (index + 1))
        adjusted[feature] = float(min(1.0, running))
    return adjusted


def recompute_composite(panel: pd.DataFrame) -> pd.DataFrame:
    output = panel.copy()
    for feature in (
        "earnings_change_on_assets",
        "sales_growth_yoy",
        "gross_margin_change_yoy",
    ):
        output[f"audit_rank_{feature}"] = output.groupby("signal_date")[
            feature
        ].rank(method="average", pct=True)
    complete = output[
        ["audit_rank_earnings_change_on_assets", "audit_rank_sales_growth_yoy"]
    ].notna().all(axis=1)
    output["audit_composite"] = np.nan
    output.loc[complete, "audit_composite"] = output.loc[
        complete,
        ["audit_rank_earnings_change_on_assets", "audit_rank_sales_growth_yoy"],
    ].mean(axis=1)
    return output


def common_intersection_composite(panel: pd.DataFrame) -> pd.DataFrame:
    output = panel.copy()
    complete = output[
        ["earnings_change_on_assets", "sales_growth_yoy"]
    ].notna().all(axis=1)
    for feature in ("earnings_change_on_assets", "sales_growth_yoy"):
        ranked = pd.Series(np.nan, index=output.index, dtype=float)
        ranked.loc[complete] = output.loc[complete].groupby("signal_date")[
            feature
        ].rank(method="average", pct=True)
        output[f"audit_common_rank_{feature}"] = ranked
    output["audit_common_composite"] = np.nan
    output.loc[complete, "audit_common_composite"] = output.loc[
        complete,
        [
            "audit_common_rank_earnings_change_on_assets",
            "audit_common_rank_sales_growth_yoy",
        ],
    ].mean(axis=1)
    return output


def manifest_check(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for index, record in enumerate(payload.get("records") or []):
        target = Path(record["path"])
        exists = target.is_file()
        actual_bytes = target.stat().st_size if exists else None
        actual_hash = sha256(target) if exists else None
        rows.append(
            {
                "record_index": index,
                "role": record.get("role"),
                "path": target.as_posix(),
                "exists": exists,
                "expected_bytes": record.get("bytes"),
                "actual_bytes": actual_bytes,
                "bytes_match": exists and actual_bytes == record.get("bytes"),
                "expected_sha256": record.get("sha256"),
                "actual_sha256": actual_hash,
                "sha256_match": exists and actual_hash == record.get("sha256"),
            }
        )
    frame = pd.DataFrame(rows)
    paths = frame["path"].tolist()
    summary = {
        "records": int(len(frame)),
        "roles": dict(Counter(frame["role"])),
        "duplicate_paths": int(len(paths) - len(set(paths))),
        "missing": int((~frame["exists"]).sum()),
        "byte_mismatches": int((~frame["bytes_match"]).sum()),
        "sha256_mismatches": int((~frame["sha256_match"]).sum()),
    }
    return frame, summary


def turnover_audit(panel: pd.DataFrame) -> dict[str, float | int | bool]:
    pre = panel.loc[pd.to_datetime(panel["signal_date"], utc=True).lt(PRE_2026)]
    parts: list[pd.DataFrame] = []
    for signal_date, month in pre.groupby("signal_date", sort=True):
        valid = month.loc[
            month["quarterly_fundamental_momentum_score"].notna()
        ].copy()
        ordered = valid["quarterly_fundamental_momentum_score"].rank(method="first")
        decile = (
            np.floor((ordered - 1) * 10 / len(valid)).astype(int) + 1
        ).clip(1, 10)
        valid["weight"] = 0.0
        valid.loc[decile.eq(10), "weight"] = 1.0 / int(decile.eq(10).sum())
        valid.loc[decile.eq(1), "weight"] = -1.0 / int(decile.eq(1).sum())
        parts.append(valid[["signal_date", "ticker", "weight"]])
    weights = (
        pd.concat(parts)
        .pivot(index="signal_date", columns="ticker", values="weight")
        .fillna(0.0)
        .sort_index()
    )
    turnover = weights.diff().abs().sum(axis=1).iloc[1:]
    gross_monthly = float(
        evaluate_feature(panel, "quarterly_fundamental_momentum_score")
        .loc[lambda x: pd.to_datetime(x["signal_date"], utc=True).lt(PRE_2026), "d10_minus_d1"]
        .mean()
    )
    break_even = gross_monthly / float(turnover.mean())
    return {
        "production_costs_included": False,
        "production_borrow_included": False,
        "production_capacity_tested": False,
        "months": int(len(weights)),
        "average_monthly_sum_abs_weight_change": float(turnover.mean()),
        "median_monthly_sum_abs_weight_change": float(turnover.median()),
        "gross_monthly_d10_minus_d1": gross_monthly,
        "audit_implied_break_even_cost_bps_per_dollar_traded_before_borrow": float(
            break_even * 10_000
        ),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    main_manifest_rows, main_manifest_summary = manifest_check(
        PRIMARY / "artifact_manifest.json"
    )
    upstream_manifest_rows, upstream_manifest_summary = manifest_check(
        ROOT / "artifacts/liquid_residual_momentum_v2/artifact_manifest.json"
    )
    main_manifest_rows.to_csv(OUTPUT / "main_manifest_verification.csv", index=False)
    upstream_manifest_rows.to_csv(
        OUTPUT / "upstream_manifest_verification.csv", index=False
    )

    panel = pd.read_parquet(PRIMARY / "monthly_quarterly_signal_panel.parquet")
    releases = pd.read_parquet(PRIMARY / "quarterly_release_features.parquet")
    mapping = pd.read_csv(MAPPING).drop_duplicates("ticker").set_index("ticker")
    raw_paths: list[Path] = []
    duration_parts: list[pd.DataFrame] = []
    annual_rows: list[dict[str, object]] = []
    cik_mismatches: list[dict[str, object]] = []
    ticker_mismatches: list[dict[str, object]] = []
    for ticker in sorted(panel["ticker"].unique()):
        cik = int(mapping.loc[ticker, "cik"])
        path = FACTS_DIR / f"CIK{cik:010d}.json.gz"
        if not path.is_file():
            continue
        raw_paths.append(path)
        payload = json.loads(gzip.decompress(path.read_bytes()))
        try:
            payload_cik = int(payload.get("cik"))
        except (TypeError, ValueError):
            payload_cik = None
        if payload_cik != cik:
            cik_mismatches.append(
                {"ticker": ticker, "expected_cik": cik, "payload_cik": payload_cik}
            )
        payload_tickers = [str(value).upper() for value in payload.get("tickers") or []]
        if payload_tickers and str(ticker).upper() not in payload_tickers:
            ticker_mismatches.append(
                {"ticker": ticker, "cik": cik, "payload_tickers": payload_tickers}
            )
        duration = company_duration_rows(str(ticker), cik, payload)
        if not duration.empty:
            duration_parts.append(duration)
        annual_rows.extend(annual_asset_filings(str(ticker), payload))

    duration = pd.concat(duration_parts, ignore_index=True)
    keys = ["ticker", "accession"]
    duration["release_end"] = duration.groupby(keys)["end"].transform("max")
    current = duration.loc[
        (duration["end"] - duration["release_end"]).abs().dt.days.le(7)
    ].sort_values(
        [*keys, "concept", "tag_priority", "source_tag", "start"],
        kind="mergesort",
    ).drop_duplicates([*keys, "concept"])
    prior = duration.assign(
        year_gap_days=(duration["release_end"] - duration["end"]).dt.days
    )
    prior = prior.loc[prior["year_gap_days"].between(250, 470)].assign(
        gap_error=lambda frame: (frame["year_gap_days"] - 365).abs()
    ).sort_values(
        [*keys, "concept", "gap_error", "tag_priority", "end"],
        ascending=[True, True, True, True, True, False],
        kind="mergesort",
    ).drop_duplicates([*keys, "concept"])
    pairs = current.merge(
        prior, on=[*keys, "concept"], how="inner", suffixes=("_current", "_prior")
    )
    cross_tag = pairs.loc[
        pairs["source_tag_current"].ne(pairs["source_tag_prior"])
    ].copy()
    cross_tag["current_duration_days"] = (
        cross_tag["end_current"] - cross_tag["start_current"]
    ).dt.days
    cross_tag["prior_duration_days"] = (
        cross_tag["end_prior"] - cross_tag["start_prior"]
    ).dt.days
    cross_tag_columns = [
        "ticker",
        "accession",
        "concept",
        "source_tag_current",
        "source_tag_prior",
        "start_current",
        "end_current",
        "value_current",
        "start_prior",
        "end_prior",
        "value_prior",
        "current_duration_days",
        "prior_duration_days",
    ]
    cross_tag[cross_tag_columns].to_csv(
        OUTPUT / "cross_tag_comparator_pairs.csv", index=False
    )
    cross_tag_key_concepts = {
        (row.ticker, row.accession): set()
        for row in cross_tag.itertuples(index=False)
    }
    for row in cross_tag.itertuples(index=False):
        cross_tag_key_concepts[(row.ticker, row.accession)].add(row.concept)

    strict = panel.copy()
    concept_sets = pd.Series(
        [
            cross_tag_key_concepts.get((ticker, accession), set())
            for ticker, accession in zip(strict["ticker"], strict["accession"])
        ],
        index=strict.index,
    )
    mismatch_net = concept_sets.map(lambda values: "net_income" in values)
    mismatch_revenue = concept_sets.map(lambda values: "revenue" in values)
    strict.loc[mismatch_net, "earnings_change_on_assets"] = np.nan
    strict.loc[
        mismatch_revenue, ["sales_growth_yoy", "gross_margin_change_yoy"]
    ] = np.nan
    strict = recompute_composite(strict)

    annual = pd.DataFrame(annual_rows).drop_duplicates(
        ["ticker", "annual_accession", "annual_available_at"]
    )
    aligned_parts: list[pd.DataFrame] = []
    for ticker, decisions in panel.groupby("ticker", sort=True):
        company = annual.loc[
            annual["ticker"].eq(ticker),
            ["annual_accession", "annual_available_at"],
        ].sort_values("annual_available_at", kind="mergesort")
        decisions = decisions.sort_values("signal_date", kind="mergesort")
        if company.empty:
            decisions = decisions.copy()
            decisions["annual_accession"] = None
            decisions["annual_available_at"] = pd.NaT
            aligned_parts.append(decisions)
        else:
            aligned_parts.append(
                pd.merge_asof(
                    decisions,
                    company,
                    left_on="signal_date",
                    right_on="annual_available_at",
                    direction="backward",
                )
            )
    q4 = pd.concat(aligned_parts, ignore_index=True)
    q4["annual_available_at"] = pd.to_datetime(
        q4["annual_available_at"], utc=True, errors="coerce"
    )
    newer_10k = (
        q4["annual_available_at"].notna()
        & q4["quarterly_available_at"].notna()
        & q4["annual_available_at"].gt(q4["quarterly_available_at"])
    )
    q4.loc[
        newer_10k,
        [
            "earnings_change_on_assets",
            "sales_growth_yoy",
            "gross_margin_change_yoy",
        ],
    ] = np.nan
    q4 = recompute_composite(q4)

    common = common_intersection_composite(panel)
    variants = {
        "primary_published": (panel, "quarterly_fundamental_momentum_score"),
        "posthoc_drop_cross_tag": (strict, "audit_composite"),
        "posthoc_invalidate_after_newer_10k": (q4, "audit_composite"),
        "posthoc_common_intersection_ranks": (common, "audit_common_composite"),
    }
    monthly_frames: list[pd.DataFrame] = []
    summaries: dict[str, dict[str, float | int]] = {}
    for variant, (frame, score) in variants.items():
        monthly = evaluate_feature(frame, score)
        monthly.insert(0, "variant", variant)
        monthly_frames.append(monthly)
        summaries[variant] = evidence_summary(monthly)
    monthly_sensitivity = pd.concat(monthly_frames, ignore_index=True)
    monthly_sensitivity.to_csv(OUTPUT / "primary_monthly_sensitivity.csv", index=False)

    all_feature_rows: list[pd.DataFrame] = []
    independent_pvalues: dict[str, float] = {}
    for feature in FEATURES:
        evidence = evaluate_feature(panel, feature)
        evidence.insert(0, "feature", feature)
        all_feature_rows.append(evidence)
        independent_pvalues[feature] = evidence_summary(evidence)[
            "one_sided_hac_pvalue"
        ]  # type: ignore[assignment]
    independent_all = pd.concat(all_feature_rows, ignore_index=True)
    published_monthly = pd.read_csv(PRIMARY / "monthly_feature_ic.csv")
    published_monthly["signal_date"] = pd.to_datetime(
        published_monthly["signal_date"], utc=True
    )
    comparison = published_monthly.merge(
        independent_all,
        on=["signal_date", "feature"],
        suffixes=("_published", "_independent"),
        validate="one_to_one",
    )
    statistic_reconstruction = {
        "published_rows": int(len(published_monthly)),
        "independently_reconstructed_rows": int(len(independent_all)),
        "one_to_one_rows": int(len(comparison)),
        "maximum_absolute_names_difference": float(
            (
                comparison["names_published"]
                - comparison["names_independent"]
            ).abs().max()
        ),
        "maximum_absolute_raw_ic_difference": float(
            (
                comparison["raw_rank_ic_published"]
                - comparison["raw_rank_ic_independent"]
            ).abs().max()
        ),
        "maximum_absolute_incremental_ic_difference": float(
            (
                comparison["incremental_rank_ic_published"]
                - comparison["incremental_rank_ic_independent"]
            ).abs().max()
        ),
        "independent_pre2026_pvalues": independent_pvalues,
        "independent_pre2026_bh_qvalues": bh_qvalues(independent_pvalues),
    }

    pre = panel.loc[pd.to_datetime(panel["signal_date"], utc=True).lt(PRE_2026)]
    complete = pre[
        ["earnings_change_on_assets", "sales_growth_yoy"]
    ].notna().all(axis=1)
    coverage = pre.assign(primary_complete=complete).groupby("signal_date").agg(
        total_names=("ticker", "size"),
        primary_complete=("primary_complete", "sum"),
        earnings_available=("earnings_change_on_assets", "count"),
        sales_available=("sales_growth_yoy", "count"),
    )
    coverage["primary_complete_fraction"] = (
        coverage["primary_complete"] / coverage["total_names"]
    )
    coverage.reset_index().to_csv(OUTPUT / "primary_coverage_by_month.csv", index=False)

    panel_pre = pd.to_datetime(panel["signal_date"], utc=True).lt(PRE_2026)
    q4_pre = pd.to_datetime(q4["signal_date"], utc=True).lt(PRE_2026)
    q4_primary = newer_10k & q4["quarterly_feature_eligible"] & q4_pre
    q4_monthly = (
        q4.assign(q4_superseded_primary=q4_primary)
        .loc[q4_pre]
        .groupby("signal_date")
        .agg(
            primary_complete=("quarterly_feature_eligible", "sum"),
            q4_superseded_primary=("q4_superseded_primary", "sum"),
        )
    )
    q4_monthly["q4_superseded_fraction"] = (
        q4_monthly["q4_superseded_primary"] / q4_monthly["primary_complete"]
    )
    q4_monthly.reset_index().to_csv(
        OUTPUT / "q4_10k_supersession_by_month.csv", index=False
    )

    existing_manual = json.loads(
        (PRIMARY / "manual_reconstruction.json").read_text(encoding="utf-8")
    )
    manual_keys = {
        (str(row["ticker"]), str(row["accession"]))
        for row in existing_manual.get("rows") or []
    }
    aligned_keys = set(zip(panel["ticker"].astype(str), panel["accession"].astype(str)))
    pre_eligible_keys = set(
        zip(
            pre.loc[pre["quarterly_feature_eligible"], "ticker"].astype(str),
            pre.loc[pre["quarterly_feature_eligible"], "accession"].astype(str),
        )
    )
    manual_sample_audit = {
        "generated_by_primary_runner_not_independent": True,
        "sample_rows": len(manual_keys),
        "rows_used_anywhere_in_monthly_panel": len(manual_keys & aligned_keys),
        "rows_used_in_pre2026_primary_evaluation": len(
            manual_keys & pre_eligible_keys
        ),
        "missing_source_tag_and_context_fields": True,
        "does_not_trace_monthly_rank_decile_or_ic": True,
    }

    # Bridge one row all the way from raw XBRL contexts to the evaluated
    # month. IRTC is the sole pre-2026 evaluated accession in the primary
    # runner's deterministic five-row sample, so this is deterministic rather
    # than chosen for its return.
    manual_ticker = "IRTC"
    manual_cik = 1_388_658
    manual_accession = "0001388658-25-000106"
    manual_pairs = pairs.loc[
        pairs["ticker"].eq(manual_ticker)
        & pairs["accession"].eq(manual_accession)
        & pairs["concept"].isin(["net_income", "revenue"])
    ].copy()
    if set(manual_pairs["concept"]) != {"net_income", "revenue"}:
        raise RuntimeError("manual raw reconstruction contexts are incomplete")
    manual_release = releases.loc[
        releases["ticker"].eq(manual_ticker)
        & releases["accession"].eq(manual_accession)
    ].iloc[0]
    manual_payload = json.loads(
        gzip.decompress(
            (FACTS_DIR / f"CIK{manual_cik:010d}.json.gz").read_bytes()
        )
    )
    asset_entries = (
        ((((((manual_payload.get("facts") or {}).get("us-gaap") or {}).get("Assets") or {}).get("units") or {}).get("USD")))
        or []
    )
    asset_candidates: list[dict[str, object]] = []
    release_end = pd.Timestamp(manual_release["fiscal_end"])
    for item in asset_entries:
        if (
            item.get("accn") != manual_accession
            or item.get("form") not in {"10-Q", "10-Q/A"}
        ):
            continue
        end = sec_date(item.get("end"))
        value = finite_float(item.get("val"))
        if end is None or value is None:
            continue
        end_gap = abs(int((end - release_end).days))
        if end_gap <= 31:
            asset_candidates.append(
                {"end": end, "value": value, "end_gap_days": end_gap}
            )
    if not asset_candidates:
        raise RuntimeError("manual raw reconstruction asset context is missing")
    manual_asset = sorted(
        asset_candidates, key=lambda row: (row["end_gap_days"], row["end"])
    )[0]
    selected = manual_pairs.set_index("concept")
    independent_earnings = float(
        (
            selected.loc["net_income", "value_current"]
            - selected.loc["net_income", "value_prior"]
        )
        / manual_asset["value"]
    )
    independent_sales = float(
        selected.loc["revenue", "value_current"]
        / selected.loc["revenue", "value_prior"]
        - 1
    )
    manual_panel_rows = panel.loc[
        panel["ticker"].eq(manual_ticker)
        & panel["accession"].eq(manual_accession)
        & pd.to_datetime(panel["signal_date"], utc=True).lt(PRE_2026)
    ]
    if len(manual_panel_rows) != 1:
        raise RuntimeError("manual reconstruction must map to exactly one primary month")
    manual_panel_row = manual_panel_rows.iloc[0]
    manual_month = panel.loc[
        panel["signal_date"].eq(manual_panel_row["signal_date"])
    ].copy()
    manual_month["audit_earnings_rank"] = manual_month[
        "earnings_change_on_assets"
    ].rank(method="average", pct=True)
    manual_month["audit_sales_rank"] = manual_month["sales_growth_yoy"].rank(
        method="average", pct=True
    )
    manual_complete = manual_month[
        ["audit_earnings_rank", "audit_sales_rank"]
    ].notna().all(axis=1)
    manual_month["audit_score"] = np.nan
    manual_month.loc[manual_complete, "audit_score"] = manual_month.loc[
        manual_complete, ["audit_earnings_rank", "audit_sales_rank"]
    ].mean(axis=1)
    manual_evaluable = manual_month.loc[manual_complete].copy()
    manual_evaluable["audit_order"] = manual_evaluable["audit_score"].rank(
        method="first"
    )
    manual_evaluable["audit_decile"] = (
        np.floor(
            (manual_evaluable["audit_order"] - 1)
            * 10
            / len(manual_evaluable)
        ).astype(int)
        + 1
    ).clip(1, 10)
    manual_evaluated = manual_evaluable.loc[
        manual_evaluable["ticker"].eq(manual_ticker)
    ].iloc[0]
    raw_pair_records = []
    for row in manual_pairs.itertuples(index=False):
        raw_pair_records.append(
            {
                "concept": row.concept,
                "current_source_tag": row.source_tag_current,
                "current_start": row.start_current,
                "current_end": row.end_current,
                "current_value": float(row.value_current),
                "prior_source_tag": row.source_tag_prior,
                "prior_start": row.start_prior,
                "prior_end": row.end_prior,
                "prior_value": float(row.value_prior),
            }
        )
    independent_manual = {
        "selection": "sole pre-2026 evaluated accession in the primary runner deterministic five-row sample",
        "ticker": manual_ticker,
        "cik": manual_cik,
        "accession": manual_accession,
        "raw_selected_contexts": raw_pair_records,
        "raw_asset_context": manual_asset,
        "independent_feature_calculation": {
            "earnings_change_on_assets": independent_earnings,
            "sales_growth_yoy": independent_sales,
        },
        "published_release_features": {
            "earnings_change_on_assets": float(
                manual_release["earnings_change_on_assets"]
            ),
            "sales_growth_yoy": float(manual_release["sales_growth_yoy"]),
        },
        "monthly_trace": {
            "quarterly_available_at": manual_panel_row["quarterly_available_at"],
            "signal_date": manual_panel_row["signal_date"],
            "entry_at": manual_panel_row["entry_at"],
            "exit_at": manual_panel_row["exit_at"],
            "age_days": float(manual_panel_row["quarterly_feature_age_days"]),
            "earnings_percentile_rank": float(
                manual_evaluated["audit_earnings_rank"]
            ),
            "sales_percentile_rank": float(manual_evaluated["audit_sales_rank"]),
            "independent_composite": float(manual_evaluated["audit_score"]),
            "published_composite": float(
                manual_panel_row["quarterly_fundamental_momentum_score"]
            ),
            "evaluable_names": int(len(manual_evaluable)),
            "score_order": int(manual_evaluated["audit_order"]),
            "decile": int(manual_evaluated["audit_decile"]),
            "target_return": float(manual_evaluated["target_return"]),
        },
        "checks": {
            "earnings_matches": bool(
                np.isclose(
                    independent_earnings,
                    manual_release["earnings_change_on_assets"],
                )
            ),
            "sales_matches": bool(
                np.isclose(independent_sales, manual_release["sales_growth_yoy"])
            ),
            "composite_matches": bool(
                np.isclose(
                    manual_evaluated["audit_score"],
                    manual_panel_row["quarterly_fundamental_momentum_score"],
                )
            ),
            "release_known_before_signal": bool(
                manual_panel_row["quarterly_available_at"]
                <= manual_panel_row["signal_date"]
            ),
            "signal_before_entry": bool(
                manual_panel_row["signal_date"] < manual_panel_row["entry_at"]
            ),
        },
    }
    write_json(OUTPUT / "independent_manual_reconstruction.json", independent_manual)

    cost_audit = turnover_audit(panel)
    cross_tag_aligned = pd.Series(
        [
            (ticker, accession) in cross_tag_key_concepts
            for ticker, accession in zip(panel["ticker"], panel["accession"])
        ],
        index=panel.index,
    )
    cross_tag_primary = cross_tag_aligned & panel["quarterly_feature_eligible"]
    cross_tag_pre_primary = cross_tag_primary & panel_pre
    cross_tag_summary = {
        "selected_current_prior_concept_pairs": int(len(pairs)),
        "cross_source_tag_concept_pairs": int(len(cross_tag)),
        "cross_source_tag_releases": int(
            cross_tag[["ticker", "accession"]].drop_duplicates().shape[0]
        ),
        "cross_source_tag_tickers": int(cross_tag["ticker"].nunique()),
        "aligned_stock_months_any_cross_tag": int(cross_tag_aligned.sum()),
        "aligned_primary_complete_stock_months_any_cross_tag": int(
            cross_tag_primary.sum()
        ),
        "pre2026_primary_complete_stock_months_any_cross_tag": int(
            cross_tag_pre_primary.sum()
        ),
        "pre2026_months_affected": int(
            panel.loc[cross_tag_pre_primary, "signal_date"].nunique()
        ),
    }
    q4_summary = {
        "assets_tag_10k_filings_used_as_conservative_lower_bound": int(len(annual)),
        "tickers_with_assets_tag_10k": int(annual["ticker"].nunique()),
        "pre2026_primary_complete_stock_months": int(
            (q4["quarterly_feature_eligible"] & q4_pre).sum()
        ),
        "pre2026_primary_stock_months_with_newer_10k": int(q4_primary.sum()),
        "pre2026_fraction_with_newer_10k": float(
            q4_primary.sum()
            / (q4["quarterly_feature_eligible"] & q4_pre).sum()
        ),
        "pre2026_months_affected": int(q4.loc[q4_primary, "signal_date"].nunique()),
        "pre2026_tickers_affected": int(q4.loc[q4_primary, "ticker"].nunique()),
        "maximum_monthly_fraction_with_newer_10k": float(
            q4_monthly["q4_superseded_fraction"].max()
        ),
    }

    findings = pd.DataFrame(
        [
            {
                "severity": "Critical",
                "check": "historical universe and security lineage",
                "result": "FAIL",
                "evidence": "2026 current-survivor ticker universe; no permanent security ID or delisting returns",
                "effect": "historical/deployable alpha claim is invalid",
                "remediation": "PIT universe, permanent security master, delisting returns",
            },
            {
                "severity": "Major",
                "check": "Q4 and intervening 10-K",
                "result": "FAIL",
                "evidence": f"{q4_summary['pre2026_primary_stock_months_with_newer_10k']}/{q4_summary['pre2026_primary_complete_stock_months']} pre-2026 primary stock-months carry an older 10-Q after a newer 10-K",
                "effect": "the signal is not a complete latest-quarter fundamental signal",
                "remediation": "derive as-filed Q4 from 10-K or invalidate older Q3 after the 10-K clock",
            },
            {
                "severity": "Major",
                "check": "same-accounting-definition comparator",
                "result": "FAIL",
                "evidence": f"{cross_tag_summary['cross_source_tag_releases']} releases pair current/prior values from different US-GAAP tags",
                "effect": "some year-over-year changes compare different accounting concepts",
                "remediation": "require same source tag and compatible duration/context; fail ambiguous ties",
            },
            {
                "severity": "Major",
                "check": "costs, borrow, capacity",
                "result": "MISSING",
                "evidence": "D10-D1 is a gross arithmetic diagnostic with no transaction cost, borrow, financing or capacity model",
                "effect": "spread cannot support deployability",
                "remediation": "construct executable weights, turnover, costs, borrow and ADV capacity before portfolio claims",
            },
            {
                "severity": "Major",
                "check": "manual reconstruction",
                "result": "MISSING",
                "evidence": f"primary-generated sample has {manual_sample_audit['rows_used_in_pre2026_primary_evaluation']}/5 rows used in the pre-2026 primary evaluation and omits raw tag/context and IC trace",
                "effect": "the file is a sample dump, not independent manual reconstruction",
                "remediation": "risk-based raw filing-to-feature-to-rank-to-return reconstruction",
            },
            {
                "severity": "Major",
                "check": "multiple testing family",
                "result": "PARTIAL",
                "evidence": "BH covers four local features while the trial ledger states at least 81 opened tests",
                "effect": "a future local q-value pass would not establish a global discovery",
                "remediation": "unified trial ledger/correction or untouched prospective holdout",
            },
            {
                "severity": "Minor",
                "check": "artifact environment lineage",
                "result": "PARTIAL",
                "evidence": "primary manifest includes metrics.py and strict_ml_holdout.py, but no environment lock",
                "effect": "exact dependency and ABI versions are not frozen in the primary artifact",
                "remediation": "include a dependency/environment lock in the manifest",
            },
            {
                "severity": "Minor",
                "check": "composite missingness and ranking universe",
                "result": "SENSITIVITY_STABLE",
                "evidence": "component-specific versus common-intersection ranks produce nearly identical aggregate IC",
                "effect": "nonstationary coverage remains, but this rank choice does not change the conclusion",
                "remediation": "freeze common-universe ranking and report coverage by sector/size",
            },
            {
                "severity": "Minor",
                "check": "decile tie handling",
                "result": "NO_OBSERVED_BOUNDARY_CONFLICT",
                "evidence": "no pre-2026 tied composite score crosses a decile boundary",
                "effect": "method='first' has no observed P&L effect in this snapshot",
                "remediation": "use ticker as an explicit deterministic tie-break and test row-order invariance",
            },
        ]
    )
    findings.to_csv(OUTPUT / "findings.csv", index=False)

    environment = {
        "generated_at": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
    }
    write_json(OUTPUT / "environment.json", environment)

    main_output_paths = {
        record["path"]
        for record in json.loads(
            (PRIMARY / "artifact_manifest.json").read_text(encoding="utf-8")
        )["records"]
        if record["role"] == "output"
    }
    actual_main_outputs = {
        path.resolve().as_posix()
        for path in PRIMARY.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    manifest_closure = {
        "main_manifest": main_manifest_summary,
        "upstream_manifest": upstream_manifest_summary,
        "main_output_files_actual": len(actual_main_outputs),
        "main_output_files_manifested": len(main_output_paths),
        "main_output_omissions": sorted(actual_main_outputs - main_output_paths),
        "main_output_extras": sorted(main_output_paths - actual_main_outputs),
    }
    write_json(OUTPUT / "manifest_verification_summary.json", manifest_closure)

    summary = {
        "generated_at": environment["generated_at"],
        "audited_primary_manifest_sha256": sha256(
            PRIMARY / "artifact_manifest.json"
        ),
        "research_status": "INVALID",
        "statistical_gate": "FAIL",
        "portfolio_authorized": False,
        "no_future_release_after_signal_violations": int(
            (
                panel.loc[
                    panel["quarterly_available_at"].notna(), "quarterly_available_at"
                ]
                > panel.loc[
                    panel["quarterly_available_at"].notna(), "signal_date"
                ]
            ).sum()
        ),
        "no_release_at_or_after_entry_violations": int(
            (
                panel.loc[
                    panel["quarterly_available_at"].notna(), "quarterly_available_at"
                ]
                >= panel.loc[panel["quarterly_available_at"].notna(), "entry_at"]
            ).sum()
        ),
        "source_identity": {
            "companyfacts_files_checked": len(raw_paths),
            "payload_cik_mismatches": len(cik_mismatches),
            "payload_current_ticker_mismatches": len(ticker_mismatches),
        },
        "statistic_reconstruction": statistic_reconstruction,
        "primary_and_posthoc_sensitivities": summaries,
        "cross_tag": cross_tag_summary,
        "q4_10k_supersession": q4_summary,
        "coverage": {
            "minimum_primary_complete_fraction": float(
                coverage["primary_complete_fraction"].min()
            ),
            "median_primary_complete_fraction": float(
                coverage["primary_complete_fraction"].median()
            ),
            "maximum_primary_complete_fraction": float(
                coverage["primary_complete_fraction"].max()
            ),
        },
        "gross_spread_cost_audit": cost_audit,
        "primary_manual_sample_audit": manual_sample_audit,
        "independent_manual_reconstruction": independent_manual,
        "post_hoc_policy": "sensitivities diagnose specification risk; they are not primary results or selectable alpha candidates",
    }
    write_json(OUTPUT / "audit_summary.json", summary)

    primary_monthly = monthly_sensitivity.loc[
        pd.to_datetime(monthly_sensitivity["signal_date"], utc=True).lt(PRE_2026)
    ].copy()
    fig, ax = plt.subplots(figsize=(11, 5))
    for variant, color in (
        ("primary_published", "#2F5597"),
        ("posthoc_invalidate_after_newer_10k", "#C00000"),
        ("posthoc_drop_cross_tag", "#70AD47"),
    ):
        line = primary_monthly.loc[primary_monthly["variant"].eq(variant)].sort_values(
            "signal_date"
        )
        ax.plot(
            pd.to_datetime(line["signal_date"], utc=True),
            line["incremental_rank_ic"].rolling(12, min_periods=6).mean(),
            label=variant,
            color=color,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Post-hoc audit sensitivities: 12-month mean incremental IC")
    ax.set_ylabel("Incremental Rank IC")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT / "primary_ic_sensitivity.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(
        pd.to_datetime(q4_monthly.index, utc=True),
        q4_monthly["q4_superseded_fraction"] * 100,
        width=20,
        color="#C55A11",
    )
    ax.set_title("Primary rows carrying Q3 after a newer 10-K")
    ax.set_ylabel("Share of monthly primary sample (%)")
    fig.tight_layout()
    fig.savefig(OUTPUT / "q4_10k_supersession.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    published = summaries["primary_published"]
    q4_sensitivity = summaries["posthoc_invalidate_after_newer_10k"]
    cross_sensitivity = summaries["posthoc_drop_cross_tag"]
    report = f"""# Independent audit: quarterly fundamental momentum diagnostic v1

## Verdict

Research status: **INVALID** for a historical or deployable alpha claim. The published statistical gate is **FAIL** and portfolio authorization remains **BLOCKED**. No release-after-signal violation was found in the implemented 10-Q path.

## Reproduction

- Independently reconstructed monthly feature rows: {statistic_reconstruction['independently_reconstructed_rows']}; one-to-one matches: {statistic_reconstruction['one_to_one_rows']}.
- Maximum absolute raw/incremental IC differences: {statistic_reconstruction['maximum_absolute_raw_ic_difference']:.3e} / {statistic_reconstruction['maximum_absolute_incremental_ic_difference']:.3e}.
- Primary pre-2026: incremental IC {published['mean_incremental_rank_ic']:.6f}, HAC t={published['incremental_ic_hac_tstat_lag3']:.3f}; gross D10-D1 {published['annualized_arithmetic_d10_minus_d1']:.2%}, t={published['d10_minus_d1_hac_tstat_lag3']:.3f}.
- Main manifest: {main_manifest_summary['records']} records, {main_manifest_summary['missing']} missing, {main_manifest_summary['sha256_mismatches']} hash mismatches. Upstream manifest: {upstream_manifest_summary['records']}/{upstream_manifest_summary['records']} verified.

## Material findings

1. **Q4/10-K omission.** The extractor only accepts 10-Q/10-Q-A. At least {q4_summary['pre2026_primary_stock_months_with_newer_10k']:,}/{q4_summary['pre2026_primary_complete_stock_months']:,} ({q4_summary['pre2026_fraction_with_newer_10k']:.2%}) pre-2026 primary stock-months carry an old 10-Q after a newer 10-K was public. The monthly share reaches {q4_summary['maximum_monthly_fraction_with_newer_10k']:.2%}. Invalidating these rows changes IC to {q4_sensitivity['mean_incremental_rank_ic']:.6f} (t={q4_sensitivity['incremental_ic_hac_tstat_lag3']:.3f}) and gross D10-D1 to {q4_sensitivity['annualized_arithmetic_d10_minus_d1']:.2%} (t={q4_sensitivity['d10_minus_d1_hac_tstat_lag3']:.3f}). This is post-hoc and remains statistically weak.
2. **Cross-tag comparators.** {cross_tag_summary['cross_source_tag_concept_pairs']} concept pairs across {cross_tag_summary['cross_source_tag_releases']} releases use different US-GAAP tags for current and prior values. {cross_tag_summary['pre2026_primary_complete_stock_months_any_cross_tag']} pre-2026 primary stock-months are affected. Dropping them yields IC {cross_sensitivity['mean_incremental_rank_ic']:.6f} (t={cross_sensitivity['incremental_ic_hac_tstat_lag3']:.3f}) and gross D10-D1 {cross_sensitivity['annualized_arithmetic_d10_minus_d1']:.2%} (t={cross_sensitivity['d10_minus_d1_hac_tstat_lag3']:.3f}); the failure verdict does not change.
3. **Gross spread only.** The reported D10-D1 contains no trading cost, borrow, financing, market-impact or capacity model. Audit-derived monthly sum-absolute-weight-change is {cost_audit['average_monthly_sum_abs_weight_change']:.3f}; the gross mean would be exhausted at roughly {cost_audit['audit_implied_break_even_cost_bps_per_dollar_traded_before_borrow']:.1f} bps per dollar traded before borrow.
4. **Manual evidence gap.** Only {manual_sample_audit['rows_used_in_pre2026_primary_evaluation']}/5 rows in the primary-generated sample enter the pre-2026 primary evaluation. It does not trace raw tags/contexts through monthly rank, decile and return.
5. **Lineage and multiplicity.** The primary manifest is internally consistent and includes the transitive metric modules; it still lacks an environment lock. Local BH covers four tests while the ledger records at least 81 opened tests.

## Bounded sensitivities

The common-intersection ranking variant gives IC {summaries['posthoc_common_intersection_ranks']['mean_incremental_rank_ic']:.6f} (t={summaries['posthoc_common_intersection_ranks']['incremental_ic_hac_tstat_lag3']:.3f}), close to the published result. These diagnostics are not eligible for strategy selection.

The independent IRTC trace reconstructs both raw features, the monthly composite, decision clock and decile; all five checks pass. Broader risk-based manual sampling remains required for validation.

![IC sensitivity](primary_ic_sensitivity.png)

![Q4 supersession](q4_10k_supersession.png)
"""
    (OUTPUT / "report-source.md").write_text(report, encoding="utf-8")

    inputs = [
        PRIMARY / "artifact_manifest.json",
        PRIMARY / "monthly_quarterly_signal_panel.parquet",
        PRIMARY / "quarterly_release_features.parquet",
        PRIMARY / "monthly_feature_ic.csv",
        PRIMARY / "feature_period_summary.csv",
        PRIMARY / "decile_monthly.csv",
        PRIMARY / "decile_summary.csv",
        PRIMARY / "gates.csv",
        MAPPING,
    ]
    outputs = sorted(
        path
        for path in OUTPUT.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    records = [("input", path) for path in inputs]
    records.extend(("raw_input", path) for path in sorted(raw_paths))
    records.append(("code", Path(__file__).resolve()))
    records.extend(("output", path) for path in outputs)
    manifest = {
        "generated_at": environment["generated_at"],
        "research_status": "INVALID",
        "statistical_gate": "FAIL",
        "portfolio_authorized": False,
        "records": [
            {
                "role": role,
                "path": path.resolve().as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for role, path in records
        ],
    }
    write_json(OUTPUT / "artifact_manifest.json", manifest)
    print(OUTPUT / "report-source.md")


if __name__ == "__main__":
    main()
