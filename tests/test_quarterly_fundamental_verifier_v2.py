import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = ROOT / "scripts/verify_quarterly_fundamental_artifact_v2.py"
SPEC = importlib.util.spec_from_file_location("quarterly_v2_independent_verifier", VERIFIER_PATH)
assert SPEC is not None and SPEC.loader is not None
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _concept_defaults(row: dict[str, object], concept: str) -> None:
    for field in (
        "current",
        "prior_year",
        "annual",
        "annual_prior_year",
        "q3_ytd",
        "q3_ytd_prior_year",
        "q4_days_current",
        "q4_days_prior",
    ):
        row[f"{concept}_{field}"] = np.nan
    row[f"{concept}_source_tag"] = None
    row[f"{concept}_prior_source_tag"] = None
    row[f"{concept}_same_tag_comparator"] = False
    row[f"{concept}_q3_ytd_accession"] = None
    row[f"{concept}_q3_ytd_available_at"] = pd.NaT
    row[f"{concept}_q4_method"] = None
    for leg in (
        "current_fact",
        "prior_fact",
        "annual_current",
        "annual_prior",
        "q3_ytd_current",
        "q3_ytd_prior",
    ):
        row[f"{concept}_{leg}_accession"] = None
        row[f"{concept}_{leg}_start"] = pd.NaT
        row[f"{concept}_{leg}_end"] = pd.NaT
        row[f"{concept}_{leg}_unit"] = None


def _base_release(ticker: str, cik: int) -> dict[str, object]:
    row: dict[str, object] = {
        "ticker": ticker,
        "cik": cik,
        "accession": f"q-{ticker}",
        "form": "10-Q",
        "fiscal_year": 2024,
        "fiscal_period": "Q3",
        "fiscal_end": pd.Timestamp("2024-09-30"),
        "filed": pd.Timestamp("2024-11-01"),
        "release_lag_days": 32,
        "available_at": pd.Timestamp("2024-11-01T20:03:00Z"),
        "clock_source": "submissions_acceptance_plus_3m",
        "release_kind": "reported_10q",
        "supersession_marker": True,
        "supersession_only": False,
        "context_validation_level": "companyfacts_compatible_no_contextref",
        "barrier_inventory_source": "submissions_header",
        "header_inventory_verified": True,
        "annual_available_at": pd.Timestamp("2024-11-01T20:03:00Z"),
        "assets_current": float(1000 + cik),
    }
    for concept in VERIFIER.CONCEPTS:
        _concept_defaults(row, concept)
    scale = float(cik - 999)
    values = {
        "net_income": (40.0 + scale, 30.0),
        "revenue": (200.0 + 2.0 * scale, 180.0),
    }
    for concept, (current, prior) in values.items():
        row[f"{concept}_current"] = current
        row[f"{concept}_prior_year"] = prior
        row[f"{concept}_source_tag"] = f"{concept}_tag"
        row[f"{concept}_prior_source_tag"] = f"{concept}_tag"
        row[f"{concept}_same_tag_comparator"] = True
        for leg, start, end in (
            ("current_fact", "2024-07-01", "2024-09-30"),
            ("prior_fact", "2023-07-01", "2023-09-30"),
        ):
            row[f"{concept}_{leg}_accession"] = row["accession"]
            row[f"{concept}_{leg}_start"] = pd.Timestamp(start)
            row[f"{concept}_{leg}_end"] = pd.Timestamp(end)
            row[f"{concept}_{leg}_unit"] = "USD"
    row["earnings_change_on_assets"] = (
        float(row["net_income_current"]) - float(row["net_income_prior_year"])
    ) / float(row["assets_current"])
    row["sales_growth_yoy"] = (
        float(row["revenue_current"]) / float(row["revenue_prior_year"]) - 1.0
    )
    row["gross_margin_change_yoy"] = np.nan
    return row


def _direct_q4(row: dict[str, object]) -> None:
    row.update(
        {
            "accession": f"k-{row['ticker']}",
            "form": "10-K",
            "fiscal_period": "Q4",
            "fiscal_end": pd.Timestamp("2023-12-31"),
            "filed": pd.Timestamp("2024-02-15"),
            "release_lag_days": 46,
            "available_at": pd.Timestamp("2024-02-15T20:03:00Z"),
            "annual_available_at": pd.Timestamp("2024-02-15T20:03:00Z"),
            "release_kind": "q4_10k_strict",
        }
    )
    for concept in ("net_income", "revenue"):
        for leg, start, end in (
            ("current_fact", "2023-10-01", "2023-12-31"),
            ("prior_fact", "2022-10-01", "2022-12-31"),
        ):
            row[f"{concept}_{leg}_accession"] = row["accession"]
            row[f"{concept}_{leg}_start"] = pd.Timestamp(start)
            row[f"{concept}_{leg}_end"] = pd.Timestamp(end)
            row[f"{concept}_{leg}_unit"] = "USD"
        row[f"{concept}_q4_method"] = "direct_10k"
        row[f"{concept}_q4_days_current"] = 91
        row[f"{concept}_q4_days_prior"] = 91


def _derived_q4(row: dict[str, object]) -> None:
    _direct_q4(row)
    for concept in ("net_income", "revenue"):
        current = float(row[f"{concept}_current"])
        prior = float(row[f"{concept}_prior_year"])
        ytd = 90.0 if concept == "net_income" else 900.0
        ytd_prior = 80.0 if concept == "net_income" else 800.0
        row[f"{concept}_annual"] = current + ytd
        row[f"{concept}_annual_prior_year"] = prior + ytd_prior
        row[f"{concept}_q3_ytd"] = ytd
        row[f"{concept}_q3_ytd_prior_year"] = ytd_prior
        row[f"{concept}_q4_method"] = "same_10k_derived"
        row[f"{concept}_q3_ytd_accession"] = row["accession"]
        row[f"{concept}_q3_ytd_available_at"] = row["available_at"]
        row[f"{concept}_q4_days_current"] = 92
        row[f"{concept}_q4_days_prior"] = 92
        for leg, start, end in (
            ("annual_current", "2023-01-01", "2023-12-31"),
            ("annual_prior", "2022-01-01", "2022-12-31"),
            ("q3_ytd_current", "2023-01-01", "2023-09-30"),
            ("q3_ytd_prior", "2022-01-01", "2022-09-30"),
        ):
            row[f"{concept}_{leg}_accession"] = row["accession"]
            row[f"{concept}_{leg}_start"] = pd.Timestamp(start)
            row[f"{concept}_{leg}_end"] = pd.Timestamp(end)
            row[f"{concept}_{leg}_unit"] = "USD"
        # Direct provenance is absent when the derived path is used.
        for leg in ("current_fact", "prior_fact"):
            row[f"{concept}_{leg}_accession"] = None
            row[f"{concept}_{leg}_start"] = pd.NaT
            row[f"{concept}_{leg}_end"] = pd.NaT
            row[f"{concept}_{leg}_unit"] = None


def _seal(artifact: Path) -> None:
    records = []
    for path in sorted(artifact.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        records.append(
            {
                "role": "output",
                "path": path.resolve().as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    (artifact / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "historical_backtest_verdict": "INVALID",
                "portfolio_authorized": False,
                "records": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _build_artifact(artifact: Path) -> None:
    artifact.mkdir()
    config = {
        "clock": {
            "maximum_feature_age_days": 1000,
            "operational_latency_minutes_before_entry": 30,
        },
        "evaluation": {
            "controls": [
                "exact_total_momentum_12_1",
                "reversal_1m",
                "beta_12m",
                "realized_vol_3m",
                "illiquidity_3m",
                "log_market_cap",
            ],
            "minimum_names_per_month": 50,
            "resulting_global_trial_floor": 110,
        },
    }
    (artifact / "config_snapshot.yml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    releases = [_base_release(f"A{number:03d}", 1000 + number) for number in range(60)]
    _direct_q4(releases[0])
    _derived_q4(releases[1])
    duplicate = _base_release("Z000", 1000)
    releases.append(duplicate)
    release_frame = pd.DataFrame(releases)
    release_frame.to_parquet(artifact / "quarterly_release_features.parquet", index=False)

    rows: list[dict[str, object]] = []
    dates = pd.date_range("2025-01-01", periods=8, freq="MS", tz="UTC")
    by_ticker = release_frame.set_index("ticker")
    for month_index, signal_date in enumerate(dates):
        entry = signal_date + pd.Timedelta(days=2, hours=14, minutes=30)
        decision = entry - pd.Timedelta(minutes=30)
        month_rows: list[dict[str, object]] = []
        for ticker in release_frame["ticker"]:
            release = by_ticker.loc[ticker]
            representative = ticker != "Z000"
            earnings = release["earnings_change_on_assets"] if representative else np.nan
            sales = release["sales_growth_yoy"] if representative else np.nan
            month_rows.append(
                {
                    "ticker": ticker,
                    "signal_date": signal_date,
                    "entry_at": entry,
                    "decision_at": decision,
                    "exit_at": entry + pd.offsets.MonthBegin(1),
                    "target_return": (
                        0.001 * (int(ticker[1:]) + 1)
                        + 0.03
                        * np.sin(
                            (int(ticker[1:]) + 1) * (month_index + 2) * 0.37
                        )
                        if ticker != "Z000"
                        else -0.25
                    ),
                    "adv_usd_3m": 1_000_000 - int(ticker[1:]) * 1000 - (500_000 if ticker == "Z000" else 0),
                    "issuer_cik": release["cik"],
                    "issuer_representative": representative,
                    "accession": release["accession"],
                    "form": release["form"],
                    "fiscal_end": release["fiscal_end"],
                    "filed": release["filed"],
                    "quarterly_available_at": release["available_at"],
                    "clock_source": release["clock_source"],
                    "release_kind": release["release_kind"],
                    "supersession_only": False,
                    "header_inventory_verified": True,
                    "barrier_inventory_source": "submissions_header",
                    "context_validation_level": "companyfacts_compatible_no_contextref",
                    "quarterly_feature_age_days": (
                        decision - release["available_at"]
                    ).total_seconds()
                    / 86_400,
                    "earnings_change_on_assets": earnings,
                    "sales_growth_yoy": sales,
                    "gross_margin_change_yoy": np.nan,
                    "exact_total_momentum_12_1": int(ticker[1:]) / 60,
                    "reversal_1m": ((int(ticker[1:]) + month_index) % 11) / 11,
                    "beta_12m": 0.8 + (int(ticker[1:]) % 5) / 10,
                    "realized_vol_3m": 0.1 + (int(ticker[1:]) % 13) / 100,
                    "illiquidity_3m": np.nan if int(ticker[1:]) % 17 == 0 else int(ticker[1:]) / 1000,
                    "log_market_cap": 8 + np.log1p(int(ticker[1:])),
                }
            )
        month = pd.DataFrame(month_rows)
        complete = month[list(VERIFIER.PRIMARY_COMPONENTS)].notna().all(axis=1)
        earnings_rank = month.loc[complete, "earnings_change_on_assets"].rank(
            method="average", pct=True
        )
        sales_rank = month.loc[complete, "sales_growth_yoy"].rank(
            method="average", pct=True
        )
        month[VERIFIER.PRIMARY] = np.nan
        month.loc[complete, VERIFIER.PRIMARY] = (earnings_rank + sales_rank) / 2
        month["quarterly_feature_eligible"] = complete
        rows.extend(month.to_dict(orient="records"))
    panel = pd.DataFrame(rows)
    panel.to_parquet(artifact / "monthly_quarterly_signal_panel.parquet", index=False)

    monthly_ic, monthly_deciles = VERIFIER.reconstruct_monthly_statistics(
        panel,
        controls=config["evaluation"]["controls"],
        minimum_names=config["evaluation"]["minimum_names_per_month"],
    )
    summary = VERIFIER.summarize_ic(monthly_ic, 110)
    decile_summary = VERIFIER.summarize_deciles(monthly_deciles)
    monthly_ic.to_csv(artifact / "monthly_feature_ic.csv", index=False)
    monthly_deciles.to_csv(artifact / "decile_monthly.csv", index=False)
    summary.to_csv(artifact / "feature_period_summary.csv", index=False)
    decile_summary.to_csv(artifact / "decile_summary.csv", index=False)
    primary = summary.loc[summary["period"].eq("pre_2026")].iloc[0]
    (artifact / "run_summary.json").write_text(
        json.dumps(
            {
                "historical_backtest_verdict": "INVALID",
                "portfolio_authorized": False,
                "global_trial_floor": 110,
                "primary_pre2026_months": int(primary["months"]),
                "primary_mean_raw_rank_ic": float(primary["mean_raw_rank_ic"]),
                "primary_mean_incremental_rank_ic": float(
                    primary["mean_incremental_rank_ic"]
                ),
                "primary_incremental_ic_hac_tstat": float(
                    primary["incremental_ic_hac_tstat"]
                ),
                "primary_raw_one_sided_pvalue": float(
                    primary["one_sided_hac_pvalue"]
                ),
                "primary_global_bonferroni_pvalue": float(
                    primary["global_bonferroni_pvalue_floor_110"]
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for name, content in {
        "opened_trial_ledger.json": "{}\n",
        "toctou_check.json": "{}\n",
        "release_consistency_summary.json": "{}\n",
        "report-source.md": "# Synthetic verifier fixture\n",
        "source_audit.csv": "ticker,status\nA000,ok\n",
        "feature_coverage_by_month.csv": "signal_date,names\n2025-01-01,60\n",
        "gates.csv": "gate,passed\nsynthetic,False\n",
        "manual_reconstruction.json": "{}\n",
    }.items():
        (artifact / name).write_text(content, encoding="utf-8")
    chart_dir = artifact / "charts"
    chart_dir.mkdir()
    (chart_dir / "incremental_ic.png").write_bytes(b"synthetic-png")
    (chart_dir / "pre2026_deciles.png").write_bytes(b"synthetic-png")
    _seal(artifact)


def _check(result: dict[str, object], name: str) -> dict[str, object]:
    return next(item for item in result["checks"] if item["check"] == name)


def test_independent_verifier_accepts_valid_synthetic_fixture(tmp_path):
    source = VERIFIER_PATH.read_text(encoding="utf-8")
    assert "from alpha_factor_engine" not in source
    assert "import alpha_factor_engine" not in source
    artifact = tmp_path / "artifact"
    _build_artifact(artifact)
    result = VERIFIER.verify(artifact)
    assert result["verification_passed"]
    assert result["research_verdict_unchanged"] == "INVALID"
    assert result["independently_reconstructed_monthly_ic_rows"] == 8
    assert result["independently_reconstructed_monthly_decile_rows"] == 80
    assert all(check["passed"] for check in result["checks"])


def test_verifier_rejects_future_state_and_barrier_score_leak(tmp_path):
    artifact = tmp_path / "artifact"
    _build_artifact(artifact)
    releases = pd.read_parquet(artifact / "quarterly_release_features.parquet")
    releases.loc[releases["ticker"].eq("A002"), "supersession_only"] = True
    releases.to_parquet(artifact / "quarterly_release_features.parquet", index=False)
    panel = pd.read_parquet(artifact / "monthly_quarterly_signal_panel.parquet")
    target = panel["ticker"].eq("A003") & panel["signal_date"].eq(panel["signal_date"].min())
    panel.loc[target, "quarterly_available_at"] = panel.loc[target, "decision_at"]
    panel.loc[target, "quarterly_feature_age_days"] = 0.0
    panel.to_parquet(artifact / "monthly_quarterly_signal_panel.parquet", index=False)
    _seal(artifact)
    result = VERIFIER.verify(artifact)
    evidence = _check(result, "filing_state_clock_and_null_barriers")["evidence"]
    assert not result["verification_passed"]
    assert evidence["future_or_simultaneous_panel_states"] == 1
    assert evidence["barrier_score_leaks"] >= 1


def test_verifier_rejects_q4_arithmetic_and_metric_drift(tmp_path):
    artifact = tmp_path / "artifact"
    _build_artifact(artifact)
    releases = pd.read_parquet(artifact / "quarterly_release_features.parquet")
    derived = releases["ticker"].eq("A001")
    releases.loc[derived, "net_income_current"] += 1.0
    releases.to_parquet(artifact / "quarterly_release_features.parquet", index=False)
    monthly = pd.read_csv(artifact / "monthly_feature_ic.csv")
    monthly.loc[0, "incremental_rank_ic"] += 1e-5
    monthly.to_csv(artifact / "monthly_feature_ic.csv", index=False)
    _seal(artifact)
    result = VERIFIER.verify(artifact)
    provenance = _check(
        result, "same_tag_and_q4_compatible_context_provenance"
    )["evidence"]
    statistics = _check(
        result, "independent_ic_decile_hac_reconstruction"
    )["evidence"]
    assert not result["verification_passed"]
    assert provenance["maximum_q4_arithmetic_error"] == 1.0
    assert statistics[0]["maximum_absolute_error"] >= 1e-5 - 1e-14


def test_verifier_rejects_manifest_hash_and_output_omission(tmp_path):
    artifact = tmp_path / "artifact"
    _build_artifact(artifact)
    (artifact / "unrecorded.txt").write_text("extra output\n", encoding="utf-8")
    result = VERIFIER.verify(artifact)
    evidence = _check(result, "manifest_hashes_and_output_closure")["evidence"]
    assert not result["verification_passed"]
    assert evidence["output_omissions"] == 1


def test_verifier_rejects_fiscal_state_regression(tmp_path):
    artifact = tmp_path / "artifact"
    _build_artifact(artifact)
    releases = pd.read_parquet(artifact / "quarterly_release_features.parquet")
    regressed = _base_release("A004", 1004)
    regressed.update(
        {
            "accession": "late-old-A004",
            "fiscal_end": pd.Timestamp("2024-06-30"),
            "filed": pd.Timestamp("2024-08-01"),
            "available_at": pd.Timestamp("2024-12-01T20:03:00Z"),
        }
    )
    for concept in ("net_income", "revenue"):
        for leg, start, end in (
            ("current_fact", "2024-04-01", "2024-06-30"),
            ("prior_fact", "2023-04-01", "2023-06-30"),
        ):
            regressed[f"{concept}_{leg}_accession"] = regressed["accession"]
            regressed[f"{concept}_{leg}_start"] = pd.Timestamp(start)
            regressed[f"{concept}_{leg}_end"] = pd.Timestamp(end)
    releases = pd.concat([releases, pd.DataFrame([regressed])], ignore_index=True)
    releases.to_parquet(artifact / "quarterly_release_features.parquet", index=False)
    _seal(artifact)
    result = VERIFIER.verify(artifact)
    evidence = _check(result, "filing_state_clock_and_null_barriers")["evidence"]
    assert not result["verification_passed"]
    assert evidence["fiscal_regressions"] == 1
