#!/usr/bin/env python3
"""Run the frozen quarterly fundamental-momentum diagnostic.

This is an opened, current-survivor factor study.  It evaluates information
coefficients and decile returns, but it never authorizes portfolio deployment.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm, rankdata


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alpha_factor_engine.research.metrics import newey_west_tstat
from alpha_factor_engine.research.quarterly_fundamental_momentum import (
    RELEASE_FEATURES,
    align_quarterly_features,
    extract_quarterly_release_features,
)
from alpha_factor_engine.research.strict_ml_holdout import build_incremental_target


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash_sample(
    frame: pd.DataFrame, *, column: str, count: int
) -> pd.DataFrame:
    sampled = frame.assign(
        selection_key=frame[column].map(
            lambda value: hashlib.sha256(str(value).encode()).hexdigest()
        )
    )
    return sampled.sort_values("selection_key", kind="mergesort").head(count)


def _read_gzip_json(path: Path) -> tuple[dict[str, Any], str]:
    compressed = path.read_bytes()
    raw = gzip.decompress(compressed)
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _json_safe(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {str(key): _json_safe(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_json_safe(value) for value in payload]
    if isinstance(payload, np.generic):
        payload = payload.item()
    if isinstance(payload, float) and not np.isfinite(payload):
        return None
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(
            _json_safe(payload),
            indent=2,
            sort_keys=True,
            default=str,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _acceptance_map(payload: dict[str, Any]) -> dict[str, pd.Timestamp]:
    recent = ((payload.get("filings") or {}).get("recent") or {})
    accessions = recent.get("accessionNumber") or []
    accepted = pd.to_datetime(
        pd.Series(recent.get("acceptanceDateTime") or [], dtype="string"),
        utc=True,
        errors="coerce",
    )
    return {
        str(accession): stamp
        for accession, stamp in zip(accessions, accepted)
        if pd.notna(stamp)
    }


def _bh_qvalues(pvalues: pd.Series) -> pd.Series:
    values = pd.to_numeric(pvalues, errors="coerce").to_numpy(float)
    output = np.full(len(values), np.nan)
    valid = np.isfinite(values)
    if not valid.any():
        return pd.Series(output, index=pvalues.index)
    valid_positions = np.flatnonzero(valid)
    ordered_positions = valid_positions[np.argsort(values[valid], kind="mergesort")]
    m = len(ordered_positions)
    adjusted = np.empty(m, dtype=float)
    running = 1.0
    for reverse_rank in range(m - 1, -1, -1):
        position = ordered_positions[reverse_rank]
        rank = reverse_rank + 1
        running = min(running, values[position] * m / rank)
        adjusted[reverse_rank] = min(1.0, running)
    for ordered_index, position in enumerate(ordered_positions):
        output[position] = adjusted[ordered_index]
    return pd.Series(output, index=pvalues.index)


def _period_masks(dates: pd.Series) -> dict[str, pd.Series]:
    stamps = pd.to_datetime(dates, utc=True)
    return {
        "full": pd.Series(True, index=dates.index),
        "pre_2026": stamps.lt(pd.Timestamp("2026-01-01", tz="UTC")),
        "2026_h1_reporting_only": stamps.ge(pd.Timestamp("2026-01-01", tz="UTC")),
    }


def _monthly_feature_evidence(
    panel: pd.DataFrame,
    features: list[str],
    *,
    controls: list[str],
    minimum_names: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ic_rows: list[dict[str, object]] = []
    decile_rows: list[dict[str, object]] = []
    for signal_date, month in panel.groupby("signal_date", sort=True):
        for feature in features:
            valid = month.loc[
                pd.to_numeric(month[feature], errors="coerce").notna()
            ].copy()
            if len(valid) < minimum_names:
                continue
            if valid["target_return"].isna().any():
                raise RuntimeError(
                    f"partial target coverage at {signal_date} for {feature}"
                )
            evaluated = build_incremental_target(
                valid,
                target_column="target_return",
                control_columns=controls,
                output_column="incremental_target",
            )
            score = pd.to_numeric(evaluated[feature], errors="raise").to_numpy()
            target = pd.to_numeric(evaluated["target_return"], errors="raise").to_numpy()
            incremental = pd.to_numeric(
                evaluated["incremental_target"], errors="raise"
            ).to_numpy()
            raw_ic = float(np.corrcoef(rankdata(score), rankdata(target))[0, 1])
            incremental_ic = float(
                np.corrcoef(rankdata(score), rankdata(incremental))[0, 1]
            )
            ic_rows.append(
                {
                    "signal_date": signal_date,
                    "feature": feature,
                    "names": int(len(evaluated)),
                    "raw_rank_ic": raw_ic,
                    "incremental_rank_ic": incremental_ic,
                }
            )
            ordered_rank = pd.Series(score, index=evaluated.index).rank(
                method="first"
            )
            decile = np.floor((ordered_rank - 1) * 10 / len(evaluated)).astype(int) + 1
            decile = decile.clip(1, 10)
            evaluated = evaluated.assign(decile=decile)
            for bucket, group in evaluated.groupby("decile", sort=True):
                decile_rows.append(
                    {
                        "signal_date": signal_date,
                        "feature": feature,
                        "decile": int(bucket),
                        "names": int(len(group)),
                        "equal_weight_return": float(group["target_return"].mean()),
                    }
                )
    return (
        pd.DataFrame(
            ic_rows,
            columns=[
                "signal_date",
                "feature",
                "names",
                "raw_rank_ic",
                "incremental_rank_ic",
            ],
        ),
        pd.DataFrame(
            decile_rows,
            columns=[
                "signal_date",
                "feature",
                "decile",
                "names",
                "equal_weight_return",
            ],
        ),
    )


def _summarize_ic(monthly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature, evidence in monthly.groupby("feature", sort=True):
        masks = _period_masks(evidence["signal_date"])
        for period, mask in masks.items():
            sample = evidence.loc[mask].copy()
            if sample.empty:
                continue
            tstat = float(newey_west_tstat(sample["incremental_rank_ic"]))
            yearly = (
                sample.assign(
                    year=pd.to_datetime(sample["signal_date"], utc=True).dt.year
                )
                .groupby("year")["incremental_rank_ic"]
                .mean()
            )
            rows.append(
                {
                    "feature": feature,
                    "period": period,
                    "months": int(len(sample)),
                    "minimum_names": int(sample["names"].min()),
                    "median_names": float(sample["names"].median()),
                    "maximum_names": int(sample["names"].max()),
                    "mean_raw_rank_ic": float(sample["raw_rank_ic"].mean()),
                    "raw_ic_hac_tstat": float(
                        newey_west_tstat(sample["raw_rank_ic"])
                    ),
                    "mean_incremental_rank_ic": float(
                        sample["incremental_rank_ic"].mean()
                    ),
                    "incremental_ic_hac_tstat": tstat,
                    "positive_month_fraction": float(
                        sample["incremental_rank_ic"].gt(0).mean()
                    ),
                    "positive_year_fraction": float(yearly.gt(0).mean()),
                    "one_sided_hac_pvalue": float(norm.sf(tstat)),
                }
            )
    summary = pd.DataFrame(rows)
    primary = summary["period"].eq("pre_2026")
    summary["bh_qvalue_pre2026_family"] = np.nan
    summary.loc[primary, "bh_qvalue_pre2026_family"] = _bh_qvalues(
        summary.loc[primary, "one_sided_hac_pvalue"]
    )
    return summary


def _summarize_deciles(monthly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (feature, decile), evidence in monthly.groupby(
        ["feature", "decile"], sort=True
    ):
        masks = _period_masks(evidence["signal_date"])
        for period, mask in masks.items():
            sample = evidence.loc[mask]
            if sample.empty:
                continue
            rows.append(
                {
                    "feature": feature,
                    "period": period,
                    "decile": int(decile),
                    "months": int(len(sample)),
                    "annualized_arithmetic_return": float(
                        12 * sample["equal_weight_return"].mean()
                    ),
                    "hac_tstat": float(
                        newey_west_tstat(sample["equal_weight_return"])
                    ),
                }
            )
    result = pd.DataFrame(rows)
    spread_rows: list[dict[str, object]] = []
    for feature, evidence in monthly.groupby("feature", sort=True):
        pivot = evidence.pivot(
            index="signal_date", columns="decile", values="equal_weight_return"
        )
        if 1 not in pivot or 10 not in pivot:
            continue
        spread = (pivot[10] - pivot[1]).dropna()
        masks = _period_masks(pd.Series(spread.index, index=spread.index))
        for period, mask in masks.items():
            sample = spread.loc[mask.to_numpy()]
            if sample.empty:
                continue
            spread_rows.append(
                {
                    "feature": feature,
                    "period": period,
                    "decile": "D10-D1",
                    "months": int(len(sample)),
                    "annualized_arithmetic_return": float(12 * sample.mean()),
                    "hac_tstat": float(newey_west_tstat(sample)),
                }
            )
    return pd.concat([result, pd.DataFrame(spread_rows)], ignore_index=True)


def _charts(
    monthly_ic: pd.DataFrame,
    decile_summary: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    chart_dir = output_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    primary = monthly_ic.loc[
        monthly_ic["feature"].eq("quarterly_fundamental_momentum_score")
    ].copy()
    primary["signal_date"] = pd.to_datetime(primary["signal_date"], utc=True)
    primary["rolling_12m"] = primary["incremental_rank_ic"].rolling(
        12, min_periods=6
    ).mean()
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(
        primary["signal_date"],
        primary["incremental_rank_ic"],
        width=20,
        color=np.where(primary["incremental_rank_ic"].ge(0), "#70AD47", "#C00000"),
        alpha=0.55,
    )
    ax.plot(primary["signal_date"], primary["rolling_12m"], color="#2F5597", linewidth=2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Quarterly fundamental momentum: incremental monthly Rank IC")
    ax.set_ylabel("Rank IC after price/risk controls")
    fig.tight_layout()
    path = chart_dir / "incremental_ic.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    deciles = decile_summary.loc[
        decile_summary["feature"].eq("quarterly_fundamental_momentum_score")
        & decile_summary["period"].eq("pre_2026")
        & decile_summary["decile"].astype(str).str.fullmatch(r"\d+")
    ].copy()
    deciles["decile"] = pd.to_numeric(deciles["decile"])
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(
        deciles["decile"],
        deciles["annualized_arithmetic_return"] * 100,
        color="#4472C4",
    )
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Signal decile")
    ax.set_ylabel("Annualized arithmetic return (%)")
    ax.set_title("Pre-2026 next-open return by quarterly-fundamental decile")
    fig.tight_layout()
    path = chart_dir / "pre2026_deciles.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def run(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = ROOT / config["outputs"]["directory"]
    output_dir.mkdir(parents=True, exist_ok=True)
    source = config["source"]
    panel_path = ROOT / source["monthly_panel"]
    mapping_path = ROOT / source["current_universe_mapping"]
    if _sha256(panel_path) != source["expected_monthly_panel_sha256"]:
        raise ValueError("monthly panel hash drifted")
    if _sha256(mapping_path) != source["expected_universe_mapping_sha256"]:
        raise ValueError("current universe mapping hash drifted")
    panel = pd.read_parquet(panel_path)
    if (
        len(panel) != int(source["expected_rows"])
        or panel["signal_date"].nunique() != int(source["expected_months"])
        or panel["ticker"].nunique() != int(source["expected_current_tickers"])
    ):
        raise ValueError("monthly panel counts drifted")
    panel["signal_date"] = pd.to_datetime(panel["signal_date"], utc=True)
    if panel.duplicated(["signal_date", "ticker"]).any():
        raise ValueError("monthly panel contains duplicate signal/ticker rows")

    mapping = pd.read_csv(mapping_path)
    conflicts = mapping.groupby("ticker")["cik"].nunique()
    if conflicts.gt(1).any():
        raise ValueError("current ticker maps to multiple CIKs")
    cik_by_ticker = (
        mapping.drop_duplicates("ticker").set_index("ticker")["cik"].astype(int).to_dict()
    )
    facts_dir = ROOT / source["companyfacts_cache"]
    submissions_dir = ROOT / source["submissions_cache"]
    release_parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    raw_paths: list[Path] = []
    panel_tickers = sorted(panel["ticker"].unique())
    for ticker_index, ticker in enumerate(panel_tickers, start=1):
        if ticker_index == 1 or ticker_index % 100 == 0:
            print(
                f"SEC quarterly extraction {ticker_index}/{len(panel_tickers)}",
                flush=True,
            )
        cik = cik_by_ticker.get(ticker)
        if cik is None:
            audit_rows.append(
                {"ticker": ticker, "cik": pd.NA, "status": "missing_current_cik", "releases": 0}
            )
            continue
        facts_path = facts_dir / f"CIK{int(cik):010d}.json.gz"
        if not facts_path.is_file():
            audit_rows.append(
                {"ticker": ticker, "cik": cik, "status": "missing_companyfacts_cache", "releases": 0}
            )
            continue
        payload, raw_sha = _read_gzip_json(facts_path)
        raw_paths.append(facts_path)
        acceptance: dict[str, pd.Timestamp] = {}
        submissions_path = submissions_dir / f"CIK{int(cik):010d}.json.gz"
        if submissions_path.is_file():
            submissions, _ = _read_gzip_json(submissions_path)
            acceptance = _acceptance_map(submissions)
            raw_paths.append(submissions_path)
        releases = extract_quarterly_release_features(
            str(ticker), int(cik), payload, acceptance
        )
        if not releases.empty:
            release_parts.append(releases)
        audit_rows.append(
            {
                "ticker": ticker,
                "cik": cik,
                "status": "ok" if not releases.empty else "no_usable_quarterly_release",
                "companyfacts_raw_sha256": raw_sha,
                "submissions_acceptance_accessions": int(len(acceptance)),
                "releases": int(len(releases)),
                "primary_complete_releases": int(
                    releases[list(config["features"]["primary_components"])]
                    .notna()
                    .all(axis=1)
                    .sum()
                )
                if not releases.empty
                else 0,
            }
        )
    if not release_parts:
        raise RuntimeError("no quarterly release features were extracted")
    releases = pd.concat(release_parts, ignore_index=True).sort_values(
        ["available_at", "ticker", "accession"], kind="mergesort"
    )
    audit = pd.DataFrame(audit_rows).sort_values("ticker")
    # Persist deterministic extraction evidence before downstream statistics so
    # a fail-closed coverage result remains inspectable.
    releases.to_parquet(output_dir / "quarterly_release_features.parquet", index=False)
    audit.to_csv(output_dir / "source_audit.csv", index=False)
    aligned = align_quarterly_features(
        panel,
        releases,
        max_age_days=int(config["clock"]["maximum_feature_age_days"]),
    )
    features = [config["features"]["primary"], *RELEASE_FEATURES]
    feature_coverage = (
        aligned.groupby("signal_date", sort=True)[features]
        .count()
        .reset_index()
    )
    feature_coverage.to_csv(output_dir / "feature_coverage_by_month.csv", index=False)
    print(
        "Maximum monthly quarterly-feature coverage: "
        + ", ".join(
            f"{feature}={int(feature_coverage[feature].max())}"
            for feature in features
        ),
        flush=True,
    )
    monthly_ic, decile_monthly = _monthly_feature_evidence(
        aligned,
        features,
        controls=list(config["evaluation"]["controls"]),
        minimum_names=int(config["evaluation"]["minimum_names_per_month"]),
    )
    if monthly_ic.empty:
        raise RuntimeError(
            "no quarterly feature-month met the frozen minimum-names gate; "
            "see feature_coverage_by_month.csv"
        )
    summary = _summarize_ic(monthly_ic)
    decile_summary = _summarize_deciles(decile_monthly)
    primary = summary.loc[
        summary["feature"].eq(config["features"]["primary"])
        & summary["period"].eq(config["evaluation"]["primary_period"])
    ].iloc[0]
    gate_cfg = config["gates"]
    statistical_gate = bool(
        primary["mean_incremental_rank_ic"]
        > float(gate_cfg["mean_incremental_rank_ic_minimum"])
        and primary["incremental_ic_hac_tstat"]
        >= float(gate_cfg["incremental_ic_hac_tstat_minimum"])
        and primary["bh_qvalue_pre2026_family"]
        < float(gate_cfg["bh_qvalue_maximum"])
        and primary["positive_year_fraction"]
        >= float(gate_cfg["positive_year_fraction_minimum"])
    )
    gates = pd.DataFrame(
        [
            ("primary_mean_incremental_rank_ic", primary["mean_incremental_rank_ic"] > 0, f"observed={primary['mean_incremental_rank_ic']:.6f}; minimum>0"),
            ("primary_incremental_ic_hac_tstat", primary["incremental_ic_hac_tstat"] >= 2, f"observed={primary['incremental_ic_hac_tstat']:.6f}; minimum=2"),
            ("primary_bh_qvalue", primary["bh_qvalue_pre2026_family"] < 0.05, f"observed={primary['bh_qvalue_pre2026_family']:.6f}; maximum<0.05"),
            ("primary_positive_year_fraction", primary["positive_year_fraction"] >= 0.60, f"observed={primary['positive_year_fraction']:.6f}; minimum=0.60"),
            ("upstream_survivorship_free", False, "source is a 2026 current-ticker/current-survivor universe"),
            ("permanent_security_identifier", False, "security key is ticker"),
            ("complete_delisting_returns", False, "not available in source panel"),
            ("exact_sec_acceptance_for_every_release", bool(releases["clock_source"].eq("submissions_acceptance_datetime").all()), f"exact={releases['clock_source'].eq('submissions_acceptance_datetime').sum()}/{len(releases)}"),
            ("portfolio_authorized", False, "factor diagnostic only; no borrow-aware executable portfolio"),
        ],
        columns=["gate", "passed", "evidence"],
    )

    aligned_columns = [
        "ticker", "signal_date", "entry_at", "exit_at", "target_return", "eligible",
        "exact_total_momentum_12_1", "reversal_1m", "beta_12m", "realized_vol_3m",
        "illiquidity_3m", "log_market_cap", "accession", "fiscal_end", "filed",
        "quarterly_available_at", "clock_source", "quarterly_feature_age_days",
        *RELEASE_FEATURES, "quarterly_fundamental_momentum_score",
        "quarterly_feature_eligible",
    ]
    aligned[aligned_columns].to_parquet(
        output_dir / "monthly_quarterly_signal_panel.parquet", index=False
    )
    monthly_ic.to_csv(output_dir / "monthly_feature_ic.csv", index=False)
    summary.to_csv(output_dir / "feature_period_summary.csv", index=False)
    decile_monthly.to_csv(output_dir / "decile_monthly.csv", index=False)
    decile_summary.to_csv(output_dir / "decile_summary.csv", index=False)
    gates.to_csv(output_dir / "gates.csv", index=False)
    (output_dir / "config_snapshot.yml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    sample = _stable_hash_sample(releases, column="accession", count=5)
    _write_json(
        output_dir / "manual_reconstruction.json",
        {
            "selection": "five smallest SHA256(accession)",
            "rows": sample[
                [
                    "ticker", "cik", "accession", "fiscal_end", "filed",
                    "available_at", "clock_source", "net_income_current",
                    "net_income_prior_year", "assets_current",
                    "earnings_change_on_assets", "revenue_current",
                    "revenue_prior_year", "sales_growth_yoy",
                ]
            ].to_dict(orient="records"),
        },
    )
    charts = _charts(monthly_ic, decile_summary, output_dir)
    local_trials = int(config["evaluation"]["search_family_tests_added"])
    _write_json(
        output_dir / "opened_trial_ledger.json",
        {
            "opened_at": datetime.now(UTC).isoformat(),
            "classification": config["research_classification"],
            "prior_disclosed_family_minimum": 77,
            "tests_added_here": local_trials,
            "new_family_minimum_before_other_concurrent_work": 77 + local_trials,
            "features": features,
            "selection_rule": "no winner selected; all disclosed feature tests retained",
        },
    )
    run_summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "strategy_version": config["strategy_version"],
        "historical_backtest_verdict": "INVALID",
        "statistical_gate_pass": statistical_gate,
        "portfolio_authorized": False,
        "source_rows": int(len(panel)),
        "source_months": int(panel["signal_date"].nunique()),
        "source_current_tickers": int(panel["ticker"].nunique()),
        "companyfacts_files_read": int(sum(audit["companyfacts_raw_sha256"].notna())),
        "release_rows": int(len(releases)),
        "release_tickers": int(releases["ticker"].nunique()),
        "exact_clock_rows": int(releases["clock_source"].eq("submissions_acceptance_datetime").sum()),
        "fallback_clock_rows": int(releases["clock_source"].eq("conservative_filing_day_end_et").sum()),
        "primary_pre2026_months": int(primary["months"]),
        "primary_mean_incremental_rank_ic": float(primary["mean_incremental_rank_ic"]),
        "primary_incremental_ic_hac_tstat": float(primary["incremental_ic_hac_tstat"]),
        "primary_bh_qvalue": float(primary["bh_qvalue_pre2026_family"]),
        "fatal_limitations": [
            "current-ticker/current-survivor universe",
            "ticker is not a permanent security identifier",
            "delisting returns are absent",
            "most SEC clocks use conservative filed-day-end rather than retained raw acceptance",
            "historical period is opened and reused",
        ],
    }
    _write_json(output_dir / "run_summary.json", run_summary)

    primary_spread = decile_summary.loc[
        decile_summary["feature"].eq(config["features"]["primary"])
        & decile_summary["period"].eq("pre_2026")
        & decile_summary["decile"].astype(str).eq("D10-D1")
    ].iloc[0]
    report = f"""# Quarterly fundamental momentum diagnostic v1

## Verdict

Historical/deployable claim: **INVALID**. Statistical gate: **{'PASS' if statistical_gate else 'FAIL'}**. Portfolio authorization: **BLOCKED**.

This is an opened current-survivor diagnostic over {len(panel):,} stock-months and {panel['signal_date'].nunique()} monthly states. It adds four disclosed tests to the trial family. It cannot repair ticker lineage or missing delisting returns.

## Primary result

- Signal: equal-weight rank of quarterly earnings change on assets and year-over-year sales growth.
- Pre-2026 months: {int(primary['months'])}
- Monthly names, min/median/max: {int(primary['minimum_names'])} / {primary['median_names']:.0f} / {int(primary['maximum_names'])}
- Mean raw Rank IC: {primary['mean_raw_rank_ic']:.4f}
- Mean incremental Rank IC after momentum/reversal/beta/volatility/liquidity/size controls: {primary['mean_incremental_rank_ic']:.4f}
- Incremental IC HAC t-stat: {primary['incremental_ic_hac_tstat']:.2f}
- BH q-value across four disclosed feature tests: {primary['bh_qvalue_pre2026_family']:.3f}
- Gross D10-D1 annualized arithmetic return: {primary_spread['annualized_arithmetic_return']:.2%}; HAC t={primary_spread['hac_tstat']:.2f}

![Incremental IC](charts/incremental_ic.png)

![Deciles](charts/pre2026_deciles.png)

## Data and clock

- Parsed {len(raw_paths):,} cached SEC source objects for {len(audit):,} current tickers.
- Extracted {len(releases):,} 10-Q/10-Q-A releases across {releases['ticker'].nunique():,} tickers.
- Exact submissions acceptance clocks: {releases['clock_source'].eq('submissions_acceptance_datetime').sum():,}/{len(releases):,}; remaining releases use the conservative end of the SEC filing calendar date in New York.
- Current and prior-year quarterly values must both appear in the same accession; 6/9-month YTD observations are excluded.
- Signals older than {config['clock']['maximum_feature_age_days']} days are discarded.
- Frozen v1 excludes 10-K/Q4 derivation. A later 10-K does not yet invalidate an older Q3 signal; the independent audit treats this as a required v2 repair.
- A small subset pairs synonymous canonical concepts across different XBRL source tags inside the same accession; a same-tag sensitivity is reported separately.

## Interpretation

The exercise asks whether quarterly accounting change contributes information beyond the existing price/risk controls. Positive decile returns alone are insufficient: the reported D10-D1 series is gross and has no turnover, spread, impact or borrow deduction. The frozen primary test also requires positive incremental IC, HAC t>=2, BH q<0.05 and at least 60% positive years. Even a statistical pass would remain `INVALID` for a clean historical or deployable claim because the upstream universe is selected from 2026 survivors and lacks permanent security IDs/delistings.

Research priors: [Livnat and Mendenhall (2006)](https://doi.org/10.1111/j.1475-679X.2006.00196.x), [Jegadeesh and Livnat (2006)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=903767), [Novy-Marx (2015)](https://www.nber.org/papers/w20984).
"""
    (output_dir / "report-source.md").write_text(report, encoding="utf-8")

    evidence = [
        ("input", panel_path),
        ("input", mapping_path),
        ("input", ROOT / "artifacts/liquid_residual_momentum_v2/artifact_manifest.json"),
        ("config", config_path),
        ("code", Path(__file__).resolve()),
        ("code", ROOT / "src/alpha_factor_engine/research/quarterly_fundamental_momentum.py"),
        ("code", ROOT / "src/alpha_factor_engine/research/metrics.py"),
        ("code", ROOT / "src/alpha_factor_engine/research/strict_ml_holdout.py"),
        *[("raw_input", path) for path in sorted(set(raw_paths))],
        *[
            ("output", path)
            for path in sorted(output_dir.rglob("*"))
            if path.is_file()
            and path.name != "artifact_manifest.json"
            and "independent_audit" not in path.relative_to(output_dir).parts
        ],
    ]
    manifest = {
        "generated_at": run_summary["generated_at"],
        "historical_backtest_verdict": "INVALID",
        "portfolio_authorized": False,
        "records": [
            {
                "role": role,
                "path": path.resolve().as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for role, path in evidence
        ],
    }
    _write_json(output_dir / "artifact_manifest.json", manifest)
    return output_dir / "report-source.md"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/quarterly_fundamental_momentum_diagnostic_v1.yml",
    )
    args = parser.parse_args()
    print(run(args.config))


if __name__ == "__main__":
    main()
