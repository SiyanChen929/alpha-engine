from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from scripts import run_sec_filing_change_stage_a as runner


def _record(path: Path, role: str) -> dict[str, object]:
    return {
        "role": role,
        "path": path.resolve().relative_to(runner.ROOT.resolve()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": runner.sha256_file(path),
    }


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, (dict, list)):
        path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    else:
        path.write_text(str(value), encoding="utf-8")
    return path


def _prices() -> pd.DataFrame:
    dates = pd.bdate_range("2019-01-02", "2023-12-29")
    timestamps = (dates + pd.Timedelta(hours=9, minutes=30)).tz_localize(
        "America/New_York"
    ).tz_convert("UTC")
    market = 0.0001 + 0.001 * np.sin(np.arange(len(dates)) / 13.0)
    rows: list[dict[str, object]] = []
    tickers = ["SPY", "AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"]
    for number, ticker in enumerate(tickers):
        extra = 0.0 if ticker == "SPY" else (0.00015 if number % 2 else -0.0001)
        values = (100 + number) * np.cumprod(1.0 + market + extra)
        for index, date in enumerate(dates):
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "timestamp": timestamps[index],
                    "close": values[index],
                    "volume": 1_000_000.0,
                    "adj_open": values[index],
                }
            )
    return pd.DataFrame(rows)


def _scores() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specifications = (
        ("2021-01-15T20:00:00Z", ("AAA", "BBB", "CCC", "DDD")),
        ("2022-01-15T20:00:00Z", ("EEE", "FFF", "GGG", "HHH")),
    )
    for year_index, (accepted_at, tickers) in enumerate(specifications):
        for index, ticker in enumerate(tickers):
            quintile = 1 if index % 2 == 0 else 5
            rows.append(
                {
                    "current_accession": f"signal-{year_index}-{index}",
                    "cik": 100 + year_index * 10 + index,
                    "ticker": ticker,
                    "form": "10-K" if index < 2 else "10-Q",
                    "current_accepted_at": accepted_at,
                    "jaccard_quintile": quintile,
                    "jaccard_sort_eligible": True,
                    "cosine_quintile": quintile,
                    "cosine_sort_eligible": True,
                }
            )
    return pd.DataFrame(rows)


def test_runner_executes_one_shot_development_bundle_end_to_end():
    artifact_root = runner.ROOT / "artifacts"
    artifact_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".stage-a-runner-test-", dir=artifact_root) as raw:
        base = Path(raw)
        config = json.loads(
            (runner.ROOT / "configs/sec_filing_change_stage_a_v1.json").read_text(
                encoding="utf-8"
            )
        )
        config["cohorts"]["development"].update(
            {
                "release_month_gte": "2021-01",
                "release_month_lte": "2021-12",
                "formal_evaluation_month_gte": "2021-02",
                "formal_evaluation_month_lte": "2021-04",
            }
        )
        config["cohorts"]["confirmation"].update(
            {
                "release_month_gte": "2022-01",
                "release_month_lte": "2022-12",
                "formal_evaluation_month_gte": "2022-02",
                "formal_evaluation_month_lte": "2022-04",
            }
        )
        for cohort in ("development_minimums", "confirmation_minimums"):
            config["gates"][cohort].update(
                {
                    "signed_events": 1,
                    "unique_cik": 1,
                    "portfolio_months": 1,
                    "each_form_direction": 1,
                    "each_form_portfolio_months": 1,
                }
            )
        config["gates"]["maximum_absolute_calendar_portfolio_target_weight"] = 1.0
        config["gates"]["maximum_single_cik_absolute_contribution_share"] = 1.0
        config["gates"]["maximum_entry_date_absolute_contribution_share"] = 1.0
        config_path = _write(base / "protocol.json", config)
        prices_path = base / "prices.parquet"
        scores_path = base / "scores.parquet"
        _prices().to_parquet(prices_path, index=False)
        _scores().to_parquet(scores_path, index=False)

        price_frame = _prices()
        spy_clock = price_frame.loc[price_frame["ticker"].eq("SPY")]
        calendar_path = base / "xnys.csv"
        pd.DataFrame(
            {
                "session_date": spy_clock["date"].dt.strftime("%Y-%m-%d"),
                "market_open_utc": spy_clock["timestamp"],
                "market_close_utc": spy_clock["timestamp"]
                + pd.Timedelta(hours=6, minutes=30),
            }
        ).to_csv(calendar_path, index=False)
        calendar_generator = _write(base / "build-xnys.py", "fixture generator")
        calendar_receipt_payload = {
            "schema_version": "xnys_calendar_snapshot_receipt_v1",
            "calendar": "XNYS",
            "source_status": "INDEPENDENT_VERSIONED_COMMUNITY_RULESET_NOT_EXCHANGE_LICENSED_DATA",
            "records": [
                _record(calendar_generator, "calendar_snapshot_generator"),
                _record(calendar_path, "xnys_regular_sessions"),
            ],
        }
        calendar_receipt = _write(
            base / "xnys-receipt.json", calendar_receipt_payload
        )
        config["sources"]["daily_prices"].update(
            {"start": "2019-01-02", "end": "2023-12-29"}
        )
        config["sources"]["exchange_calendar"].update(
            {
                "path": calendar_path.relative_to(runner.ROOT).as_posix(),
                "receipt": calendar_receipt.relative_to(runner.ROOT).as_posix(),
                "generator": calendar_generator.relative_to(runner.ROOT).as_posix(),
            }
        )
        config_path = _write(base / "protocol.json", config)

        role_paths = {
            "human_protocol": config_path,
            "preoutcome_plan_manifest": _write(base / "plan-manifest.json", {}),
            "frozen_acquisition_documents": _write(base / "documents.bin", "documents"),
            "header_only_rough_pairs": _write(base / "rough.bin", "rough"),
            "frozen_current_survivor_universe": _write(base / "universe.bin", "universe"),
            "sec_submissions_raw_manifest": _write(base / "sec-manifest.bin", "sec"),
            "frozen_retrospective_daily_prices": prices_path,
            "xnys_regular_sessions": calendar_path,
            "xnys_calendar_receipt": calendar_receipt,
            "calendar_snapshot_generator": calendar_generator,
            "sec_periodic_data_code": runner.SRC / "alpha_factor_engine/data/sec_periodic_filings.py",
            "filing_change_feature_code": runner.SRC / "alpha_factor_engine/features/filing_change.py",
            "preoutcome_plan_code": runner.SRC / "alpha_factor_engine/research/sec_filing_change_plan.py",
            "outcome_core_code": runner.SRC / "alpha_factor_engine/research/sec_filing_change_stage_a.py",
            "plan_builder": runner.ROOT / "scripts/build_sec_filing_change_acquisition_plan.py",
            "document_acquirer": runner.ROOT / "scripts/acquire_sec_filing_change_documents.py",
            "score_builder": runner.ROOT / "scripts/build_sec_filing_change_scores.py",
            "outcome_runner": Path(runner.__file__),
            "design_receipt_builder": runner.ROOT / "scripts/build_sec_filing_change_design_receipt.py",
        }
        receipt = {
            "schema_version": "sec_filing_change_design_receipt_v2",
            "created_at": datetime.now(UTC).isoformat(),
            "experiment": "sec_filing_change_stage_a_v1",
            "status": "FROZEN_PRE_OUTCOME",
            "records": [_record(path, role) for role, path in role_paths.items()],
            "primary_signal": "jaccard_similarity",
            "secondary_signal": "cosine_similarity",
            "current_disclosed_trial_floor": 130,
            "outcome_join_performed": False,
            "sealed_2025_signal_return_joined": False,
            "historical_alpha_verdict": "INVALID",
            "deployment_verdict": "NOT AUTHORIZED",
            "non_overwrite_policy": True,
        }
        receipt_path = _write(base / "design-receipt.json", receipt)

        score_inputs = {
            "design_receipt": receipt_path,
            "human_protocol": config_path,
            "acquisition_manifest": _write(base / "acquisition-manifest.bin", "a"),
            "acquisition_summary": _write(base / "acquisition-summary.bin", "a2"),
            "acquisition_audit": _write(base / "acquisition-audit.bin", "b"),
            "acquisition_failures": _write(base / "acquisition-failures.bin", "b2"),
            "dataset_records": _write(base / "dataset-records.bin", "b3"),
            "frozen_acquisition_documents": role_paths["frozen_acquisition_documents"],
            "mapped_header_inventory": _write(base / "mapped.bin", "c"),
            "rough_pairing_ledger": role_paths["header_only_rough_pairs"],
            "mapping_misses": _write(base / "mapping-misses.bin", "c2"),
            "score_builder": runner.ROOT / "scripts/build_sec_filing_change_scores.py",
            "feature_code": runner.SRC / "alpha_factor_engine/features/filing_change.py",
        }
        score_outputs = {
            "document_parse_audit": _write(base / "parse.bin", "d"),
            "strict_fiscal_pairs": _write(base / "pairs.bin", "e"),
            "strict_pair_exclusions": _write(base / "pair-exclusions.bin", "f"),
            "raw_similarity_scores": _write(base / "raw-scores.bin", "g"),
            "sorted_similarity_scores": scores_path,
            "monthly_cutoff_ledger": _write(base / "cutoffs.bin", "h"),
            "score_summary": _write(
                base / "score-summary.json",
                {
                    "schema_version": "sec_filing_change_score_summary_v2",
                    "status": "DETERMINISTIC_SIGNAL_BUILD_ONLY",
                    "outcome_join_performed": False,
                    "upstream_coverage": {
                        "status": "PASS_UPSTREAM_SIGNAL_BUILD_COVERAGE",
                        "all_signal_build_gates_passed": True,
                    },
                },
            ),
        }
        score_manifest = {
            "schema_version": "sec_filing_change_score_manifest_v2",
            "created_at": datetime.now(UTC).isoformat(),
            "inputs": [_record(path, role) for role, path in score_inputs.items()],
            "outputs": [_record(path, role) for role, path in score_outputs.items()],
            "outcome_join_performed": False,
            "upstream_coverage_status": "PASS_UPSTREAM_SIGNAL_BUILD_COVERAGE",
            "all_signal_build_coverage_gates_passed": True,
            "benchmark_clock_deferred_to_outcome_runner": True,
        }
        score_manifest_path = _write(base / "score-manifest.json", score_manifest)
        output = base / "run"
        result = runner.run(
            argparse.Namespace(
                config=config_path,
                design_receipt=receipt_path,
                score_manifest=score_manifest_path,
                scores=scores_path,
                prices=prices_path,
                xnys_regular_sessions=calendar_path,
                xnys_calendar_receipt=calendar_receipt,
                calendar_snapshot_generator=calendar_generator,
                output=output,
                trial_registry=base / "trial-registry",
            )
        )
        summary = json.loads(result.read_text(encoding="utf-8"))
        assert summary["historical_alpha_verdict"] == "INVALID"
        assert summary["deployment_verdict"] == "NOT AUTHORIZED"
        assert summary["sealed_2025_signal_return_joined"] is False
        assert (output / "development_jaccard_calendar_monthly.csv").is_file()
        assert len(list((base / "trial-registry").glob("development_*.json"))) == 1
