#!/usr/bin/env python3
"""Run the frozen SEC filing-change development and gated confirmation tests."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alpha_factor_engine.research.sec_filing_change_stage_a import (
    RESEARCH_VERDICT,
    aggregate_entry_dates,
    attach_event_outcomes,
    build_calendar_preoutcome_schedule,
    build_preoutcome_schedule,
    calendar_portfolio_returns,
    monthly_portfolio_gate_table,
    reconcile_benchmark_to_exchange_calendar,
    summarize_event_sample,
    summarize_monthly_portfolio,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path, role: str) -> dict[str, object]:
    return {
        "role": role,
        "path": path.resolve().relative_to(ROOT.resolve()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _verify_records(payload: dict[str, object]) -> None:
    groups = [payload.get("records"), payload.get("inputs"), payload.get("outputs")]
    combined = [
        record
        for group in groups
        if isinstance(group, list)
        for record in group
    ]
    if not combined:
        raise ValueError("frozen JSON has no file records")
    if any(
        not isinstance(record, dict)
        or set(record) != {"role", "path", "bytes", "sha256"}
        for record in combined
    ):
        raise ValueError("frozen JSON has malformed file records")
    roles = [record.get("role") for record in combined if isinstance(record, dict)]
    if len(roles) != len(combined) or len(roles) != len(set(roles)):
        raise ValueError("frozen JSON has duplicate or invalid roles")
    paths: list[Path] = []
    for record in combined:
        raw_path = record["path"]
        size = record["bytes"]
        digest = record["sha256"]
        if (
            not isinstance(raw_path, str)
            or not raw_path
            or Path(raw_path).is_absolute()
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"malformed frozen record: {record['role']}")
        candidate = ROOT / raw_path
        if candidate.is_symlink():
            raise ValueError(f"frozen file may not be a symlink: {record['role']}")
        path = candidate.resolve()
        path.relative_to(ROOT.resolve())
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"frozen file absent or changed size: {record['role']}")
        if sha256_file(path) != digest:
            raise ValueError(f"frozen file hash mismatch: {record['role']}")
        paths.append(path)
    if len(paths) != len(set(paths)):
        raise ValueError("frozen JSON assigns multiple roles to one file")


def _verify_json(path: Path, schema: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"frozen JSON is not an object: {path}")
    if payload.get("schema_version") != schema:
        raise ValueError(f"unsupported frozen JSON: {path}")
    _verify_records(payload)
    return payload


def _path_for_role(payload: dict[str, object], role: str) -> Path:
    combined = [
        record
        for key in ("records", "inputs", "outputs")
        for record in payload.get(key, [])
        if record.get("role") == role
    ]
    if len(combined) != 1:
        raise ValueError(f"frozen JSON must contain exactly one {role!r} record")
    return (ROOT / str(combined[0]["path"])).resolve()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _candidate_scores(scores: pd.DataFrame, candidate: str) -> pd.DataFrame:
    quintile = f"{candidate}_quintile"
    eligible = f"{candidate}_sort_eligible"
    missing = {quintile, eligible}.difference(scores.columns)
    if missing:
        raise ValueError(f"scores missing candidate columns: {sorted(missing)}")
    frame = scores.copy()
    frame["similarity_quintile"] = frame[quintile]
    frame["sort_eligible"] = frame[eligible]
    return frame


def _freeze_sample_schedules(
    *,
    name: str,
    signals: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
    output: Path,
    design_receipt: Path,
    score_manifest: Path,
    config_path: Path,
    minimum_event_entry_after: pd.Timestamp | None = None,
    minimum_calendar_entry_at_or_after: pd.Timestamp | None = None,
) -> dict[str, Path]:
    event_schedule, event_exclusions = build_preoutcome_schedule(
        signals, benchmark_prices
    )
    calendar_schedule, calendar_exclusions = build_calendar_preoutcome_schedule(
        signals, benchmark_prices, holding_months=3
    )
    if minimum_event_entry_after is not None and not event_schedule.empty:
        purged = event_schedule.loc[
            event_schedule["entry_date"].le(pd.Timestamp(minimum_event_entry_after))
        ]
        if not purged.empty:
            purge_rows = purged[["current_accession", "ticker"]].copy()
            purge_rows["reason"] = "event_cohort_overlap_purge"
            event_exclusions = pd.concat([event_exclusions, purge_rows], ignore_index=True)
            event_schedule = event_schedule.loc[
                event_schedule["entry_date"].gt(pd.Timestamp(minimum_event_entry_after))
            ].copy()
    if minimum_calendar_entry_at_or_after is not None and not calendar_schedule.empty:
        purged = calendar_schedule.loc[
            calendar_schedule["entry_date"].lt(
                pd.Timestamp(minimum_calendar_entry_at_or_after)
            )
        ]
        if not purged.empty:
            purge_rows = purged[["current_accession", "ticker"]].copy()
            purge_rows["reason"] = "calendar_cohort_overlap_purge"
            calendar_exclusions = pd.concat(
                [calendar_exclusions, purge_rows], ignore_index=True
            )
            calendar_schedule = calendar_schedule.loc[
                calendar_schedule["entry_date"].ge(
                    pd.Timestamp(minimum_calendar_entry_at_or_after)
                )
            ].copy()
    paths = {
        "event_schedule": output / f"{name}_event_preoutcome_schedule.parquet",
        "event_exclusions": output / f"{name}_event_schedule_exclusions.csv",
        "calendar_schedule": output / f"{name}_calendar_preoutcome_schedule.parquet",
        "calendar_exclusions": output / f"{name}_calendar_schedule_exclusions.csv",
    }
    event_schedule.to_parquet(paths["event_schedule"], index=False)
    event_exclusions.to_csv(paths["event_exclusions"], index=False)
    calendar_schedule.to_parquet(paths["calendar_schedule"], index=False)
    calendar_exclusions.to_csv(paths["calendar_exclusions"], index=False)
    schedule_receipt = {
        "schema_version": "sec_filing_change_preoutcome_schedule_receipt_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "sample": name,
        "outcome_join_performed": False,
        "sealed_2025_signal_return_joined": False,
        "records": [
            _record(paths["event_schedule"], f"{name}_event_preoutcome_schedule"),
            _record(paths["event_exclusions"], f"{name}_event_schedule_exclusions"),
            _record(paths["calendar_schedule"], f"{name}_calendar_preoutcome_schedule"),
            _record(paths["calendar_exclusions"], f"{name}_calendar_schedule_exclusions"),
            _record(design_receipt, "design_receipt"),
            _record(score_manifest, "score_manifest"),
            _record(config_path, "human_protocol"),
            _record(Path(__file__), "outcome_runner"),
            _record(
                SRC / "alpha_factor_engine/research/sec_filing_change_stage_a.py",
                "outcome_core",
            ),
        ],
    }
    receipt_path = output / f"{name}_schedule_receipt.json"
    _write_json(receipt_path, schedule_receipt)
    _verify_json(receipt_path, "sec_filing_change_preoutcome_schedule_receipt_v1")
    paths["receipt"] = receipt_path
    return paths


def _reserve_trial_bundle(
    *,
    registry: Path,
    bundle: str,
    trial_ids: list[str],
    design_receipt: Path,
    score_manifest: Path,
    output: Path,
) -> Path:
    registry = registry.resolve()
    registry.relative_to(ROOT.resolve())
    registry.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(
        (
            sha256_file(design_receipt)
            + sha256_file(score_manifest)
            + bundle
            + "|".join(trial_ids)
        ).encode("utf-8")
    ).hexdigest()
    path = registry / f"{bundle}_{identity}.json"
    payload = {
        "schema_version": "sec_filing_change_trial_reservation_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "bundle": bundle,
        "trial_ids": trial_ids,
        "design_receipt_sha256": sha256_file(design_receipt),
        "score_manifest_sha256": sha256_file(score_manifest),
        "output": output.relative_to(ROOT).as_posix(),
        "status": "CONSUMED_ON_FIRST_OUTCOME_READ",
        "deletion_or_replacement_invalidates_one_shot_claim": True,
    }
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return path


def _evaluate_sample(
    *,
    name: str,
    frozen: dict[str, Path],
    prices: pd.DataFrame,
    output: Path,
    trial_floor: int,
    gate_minimums: dict[str, int],
    config: dict[str, object],
    expected_month_start: str,
    expected_month_end: str,
    event_inference_required: bool,
) -> tuple[dict[str, object], pd.DataFrame]:
    event_schedule = pd.read_parquet(frozen["event_schedule"])
    calendar_schedule = pd.read_parquet(frozen["calendar_schedule"])
    events, outcome_exclusions = attach_event_outcomes(
        event_schedule,
        prices,
        minimum_prior_close=float(config["eligibility"]["minimum_prior_unadjusted_close_usd"]),
        adv_window=int(config["eligibility"]["adv_window_sessions"]),
        minimum_prior_stock_sessions=int(config["eligibility"]["minimum_prior_stock_sessions"]),
        minimum_prior_adv=float(config["eligibility"]["minimum_prior_adv_usd"]),
        beta_window=int(config["risk_and_return"]["beta_window_matched_sessions"]),
        minimum_beta_observations=int(config["risk_and_return"]["minimum_beta_observations"]),
        primary_one_way_cost=float(config["risk_and_return"]["primary_one_way_cost_rate"]),
        stress_one_way_cost=float(config["risk_and_return"]["stress_one_way_cost_rate"]),
    )
    event_path = output / f"{name}_event_outcomes.parquet"
    outcome_exclusion_path = output / f"{name}_outcome_exclusions.csv"
    entry_path = output / f"{name}_entry_date_returns.csv"
    events.to_parquet(event_path, index=False)
    outcome_exclusions.to_csv(outcome_exclusion_path, index=False)
    entry_dates = aggregate_entry_dates(events) if not events.empty else pd.DataFrame()
    entry_dates.to_csv(entry_path, index=False)
    event_scheduled = len(event_schedule)
    event_path_reasons = {
        "ticker_price_history_missing",
        "exact_stock_price_path_missing",
        "invalid_exact_stock_price",
    }
    event_path_missing = outcome_exclusions.loc[
        outcome_exclusions.get("reason", pd.Series(dtype=str)).isin(event_path_reasons)
    ]
    event_ex_ante = outcome_exclusions.loc[
        ~outcome_exclusions.index.isin(event_path_missing.index)
    ]
    event_ex_ante_count = int(event_ex_ante["schedule_id"].nunique()) if not event_ex_ante.empty else 0
    event_path_missing_count = int(event_path_missing["schedule_id"].nunique()) if not event_path_missing.empty else 0
    event_eligible = max(0, event_scheduled - event_ex_ante_count)
    event_coverage = (
        (event_eligible - event_path_missing_count) / event_eligible
        if event_eligible
        else 0.0
    )
    event_summary = summarize_event_sample(
        events,
        trial_floor=trial_floor,
        inferential_test_names=("pre_placebo",) if event_inference_required else (),
        formal_mean_gate_names=("post_minus_pre",) if event_inference_required else (),
        inference_allowed=(
            event_coverage
            >= float(config["gates"]["coverage"]["event_exact_price_path"])
        ),
    )
    holdings, monthly, calendar_outcome_exclusions = calendar_portfolio_returns(
        calendar_schedule,
        prices,
        holding_months=3,
        minimum_prior_close=float(config["eligibility"]["minimum_prior_unadjusted_close_usd"]),
        adv_window=int(config["eligibility"]["adv_window_sessions"]),
        minimum_prior_stock_sessions=int(config["eligibility"]["minimum_prior_stock_sessions"]),
        minimum_prior_adv=float(config["eligibility"]["minimum_prior_adv_usd"]),
        primary_one_way_cost=float(config["risk_and_return"]["primary_one_way_cost_rate"]),
        stress_one_way_cost=float(config["risk_and_return"]["stress_one_way_cost_rate"]),
    )
    holdings_path = output / f"{name}_calendar_holdings.parquet"
    monthly_path = output / f"{name}_calendar_monthly.csv"
    calendar_outcome_exclusion_path = output / f"{name}_calendar_outcome_exclusions.csv"
    holdings.to_parquet(holdings_path, index=False)
    monthly.to_csv(monthly_path, index=False)
    calendar_outcome_exclusions.to_csv(calendar_outcome_exclusion_path, index=False)
    path_missing_reasons = {
        "ticker_price_history_missing",
        "complete_calendar_clock_path_unavailable",
        "exact_monthly_stock_price_path_missing",
        "invalid_exact_monthly_stock_price",
    }
    path_missing = calendar_outcome_exclusions.loc[
        calendar_outcome_exclusions.get("reason", pd.Series(dtype=str)).isin(path_missing_reasons)
    ]
    cohort_feasibility = calendar_outcome_exclusions.loc[
        calendar_outcome_exclusions.get("reason", pd.Series(dtype=str)).eq(
            "single_sided_calendar_cohort"
        )
    ]
    ex_ante = calendar_outcome_exclusions.loc[
        ~calendar_outcome_exclusions.index.isin(
            path_missing.index.union(cohort_feasibility.index)
        )
    ]
    ex_ante_accessions = set(ex_ante.get("current_accession", pd.Series(dtype=str)).astype(str))
    path_missing_accessions = set(
        path_missing.get("current_accession", pd.Series(dtype=str)).astype(str)
    )
    eligible_calendar = max(0, len(calendar_schedule) - len(ex_ante_accessions))
    calendar_path_coverage = (
        (eligible_calendar - len(path_missing_accessions)) / eligible_calendar
        if eligible_calendar
        else 0.0
    )
    eligible_accessions = set(calendar_schedule["current_accession"].astype(str)) - set(
        calendar_outcome_exclusions["current_accession"].astype(str)
        if not calendar_outcome_exclusions.empty
        else []
    )
    eligible_schedule = calendar_schedule.loc[
        calendar_schedule["current_accession"].astype(str).isin(eligible_accessions)
    ]
    portfolio_summary = summarize_monthly_portfolio(
        eligible_schedule,
        holdings,
        monthly,
        expected_month_start=expected_month_start,
        expected_month_end=expected_month_end,
        trial_floor=trial_floor,
        hac_lags=3,
        primary_one_way_cost=float(config["risk_and_return"]["primary_one_way_cost_rate"]),
        stress_one_way_cost=float(config["risk_and_return"]["stress_one_way_cost_rate"]),
        inferential_test_names=("net30",),
        formal_mean_gate_names=("net50",) if event_inference_required else (),
        inference_allowed=calendar_path_coverage == 1.0,
    )
    portfolio_summary["return_estimate_status"] = (
        "COMPLETE_PROXY_PATH" if calendar_path_coverage == 1.0 else "INVALID_INCOMPLETE_PATH"
    )
    gates = monthly_portfolio_gate_table(
        portfolio_summary,
        signed_events_minimum=int(gate_minimums["signed_events"]),
        unique_cik_minimum=int(gate_minimums["unique_cik"]),
        portfolio_months_minimum=int(gate_minimums["portfolio_months"]),
        each_form_direction_minimum=int(gate_minimums["each_form_direction"]),
        each_form_portfolio_months_minimum=int(
            gate_minimums["each_form_portfolio_months"]
        ),
        maximum_absolute_target_weight=float(
            config["gates"]["maximum_absolute_calendar_portfolio_target_weight"]
        ),
        maximum_single_cik_gross_weight_share=float(
            config["gates"]["maximum_single_cik_absolute_contribution_share"]
        ),
        maximum_month_absolute_return_contribution_share=float(
            config["gates"]["maximum_entry_date_absolute_contribution_share"]
        ),
        minimum_leave_one_cik_out_net30_mean=float(
            config["gates"]["minimum_leave_one_cik_out_net30_mean"]
        ),
    )
    tests = event_summary.get("tests", {})
    placebo = tests.get("pre_placebo", {})
    post_minus_pre = tests.get("post_minus_pre", {})
    placebo_pass = not (
        placebo.get("mean") is not None
        and float(placebo["mean"]) > 0
        and placebo.get("one_sided_p") is not None
        and float(placebo["one_sided_p"]) <= 0.05
    )
    diagnostic_rows = [
        {
            "gate": "minimum_event_exact_price_path_coverage",
            "passed": event_coverage >= float(config["gates"]["coverage"]["event_exact_price_path"]),
            "required": event_inference_required,
            "detail": event_coverage,
        },
        {
            "gate": "complete_formal_calendar_price_paths",
            "passed": calendar_path_coverage == float(config["gates"]["coverage"]["formal_calendar_exact_price_path"]),
            "required": True,
            "detail": calendar_path_coverage,
        },
        {
            "gate": "pre_event_placebo_falsification",
            "passed": placebo_pass,
            "required": event_inference_required,
            "detail": {"mean": placebo.get("mean"), "p": placebo.get("one_sided_p")},
        },
        {
            "gate": "post_minus_pre_positive",
            "passed": post_minus_pre.get("mean") is not None and post_minus_pre["mean"] > 0,
            "required": event_inference_required,
            "detail": post_minus_pre.get("mean"),
        },
        {
            "gate": "secondary_event_maximum_single_cik_contribution",
            "passed": event_summary.get("maximum_single_cik_absolute_contribution_share") is not None
            and event_summary["maximum_single_cik_absolute_contribution_share"]
            <= float(config["gates"]["maximum_single_cik_absolute_contribution_share"]),
            "required": False,
            "detail": event_summary.get("maximum_single_cik_absolute_contribution_share"),
        },
        {
            "gate": "secondary_event_maximum_entry_date_contribution",
            "passed": event_summary.get("maximum_entry_date_absolute_contribution_share") is not None
            and event_summary["maximum_entry_date_absolute_contribution_share"]
            <= float(config["gates"]["maximum_entry_date_absolute_contribution_share"]),
            "required": False,
            "detail": event_summary.get("maximum_entry_date_absolute_contribution_share"),
        },
        {
            "gate": "secondary_event_leave_one_cik_out_positive",
            "passed": event_summary.get("minimum_leave_one_cik_out_net30_mean") is not None
            and event_summary["minimum_leave_one_cik_out_net30_mean"]
            > float(config["gates"]["minimum_leave_one_cik_out_net30_mean"]),
            "required": False,
            "detail": event_summary.get("minimum_leave_one_cik_out_net30_mean"),
        },
    ]
    gates = pd.concat([gates, pd.DataFrame(diagnostic_rows)], ignore_index=True)
    gates_path = output / f"{name}_gates.csv"
    gates.to_csv(gates_path, index=False)
    summary = {
        "sample": name,
        "formal_calendar_portfolio": portfolio_summary,
        "secondary_event_diagnostics": event_summary,
        "scheduled_event_diagnostics": event_scheduled,
        "event_ex_ante_exclusions": event_ex_ante_count,
        "event_future_path_exclusions": event_path_missing_count,
        "event_exact_price_path_coverage": event_coverage,
        "scheduled_calendar_signals": int(len(calendar_schedule)),
        "calendar_ex_ante_exclusions": int(len(ex_ante_accessions)),
        "calendar_future_path_exclusions": int(len(path_missing_accessions)),
        "calendar_cohort_feasibility_exclusions": int(
            cohort_feasibility.get(
                "current_accession", pd.Series(dtype=str)
            ).astype(str).nunique()
        ),
        "formal_calendar_exact_price_path_coverage": calendar_path_coverage,
        "stage_a_observed_gates_passed": bool(
            gates.loc[gates["required"], "passed"].all()
        ),
        "deployability_gates_passed": False,
        "artifacts": {
            "event_schedule": frozen["event_schedule"].relative_to(ROOT).as_posix(),
            "calendar_schedule": frozen["calendar_schedule"].relative_to(ROOT).as_posix(),
            "schedule_receipt": frozen["receipt"].relative_to(ROOT).as_posix(),
            "events": event_path.relative_to(ROOT).as_posix(),
            "entry_dates": entry_path.relative_to(ROOT).as_posix(),
            "calendar_holdings": holdings_path.relative_to(ROOT).as_posix(),
            "calendar_monthly": monthly_path.relative_to(ROOT).as_posix(),
            "calendar_outcome_exclusions": calendar_outcome_exclusion_path.relative_to(ROOT).as_posix(),
            "gates": gates_path.relative_to(ROOT).as_posix(),
        },
    }
    return summary, gates


def _chart(output: Path, samples: dict[str, dict[str, object]]) -> list[Path]:
    paths: list[Path] = []
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    for name, summary in samples.items():
        if summary.get("formal_calendar_exact_price_path_coverage") != 1.0:
            continue
        monthly_path = ROOT / str(summary["artifacts"]["calendar_monthly"])
        if not monthly_path.is_file() or monthly_path.stat().st_size == 0:
            continue
        frame = pd.read_csv(monthly_path)
        if frame.empty or "q5_minus_q1_net30_return" not in frame:
            continue
        frame = frame.loc[frame["portfolio_scope"].eq("ALL")].copy()
        frame["entry_date"] = pd.to_datetime(frame["entry_date"])
        ax.plot(
            frame["entry_date"],
            (1.0 + frame["q5_minus_q1_net30_return"]).cumprod() - 1.0,
            label=name,
        )
        plotted = True
    if plotted:
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("Filing-change Q5-Q1 calendar portfolio after 30bp one-way costs")
        ax.set_ylabel("Cumulative return")
        ax.legend()
        fig.tight_layout()
        path = output / "calendar_cumulative_net30.png"
        fig.savefig(path, dpi=180)
        paths.append(path)
    plt.close(fig)
    return paths


def run(args: argparse.Namespace) -> Path:
    design = _verify_json(args.design_receipt, "sec_filing_change_design_receipt_v2")
    required_design_fields = {
        "schema_version", "created_at", "experiment", "status", "records",
        "primary_signal", "secondary_signal", "current_disclosed_trial_floor",
        "outcome_join_performed", "sealed_2025_signal_return_joined",
        "historical_alpha_verdict", "deployment_verdict", "non_overwrite_policy",
    }
    if set(design) != required_design_fields:
        raise ValueError("design receipt fields differ from the frozen contract")
    required_design_roles = {
        "human_protocol", "preoutcome_plan_manifest", "frozen_acquisition_documents",
        "header_only_rough_pairs", "frozen_current_survivor_universe",
        "sec_submissions_raw_manifest", "frozen_retrospective_daily_prices",
        "xnys_regular_sessions", "xnys_calendar_receipt",
        "calendar_snapshot_generator",
        "sec_periodic_data_code", "filing_change_feature_code", "preoutcome_plan_code",
        "outcome_core_code", "plan_builder", "document_acquirer", "score_builder",
        "outcome_runner", "design_receipt_builder",
    }
    design_roles = {record["role"] for record in design["records"]}
    if design_roles != required_design_roles or len(design["records"]) != len(required_design_roles):
        raise ValueError("design receipt roles differ from the frozen contract")
    if (
        design.get("experiment") != "sec_filing_change_stage_a_v1"
        or design.get("status") != "FROZEN_PRE_OUTCOME"
        or design.get("outcome_join_performed") is not False
        or design.get("sealed_2025_signal_return_joined") is not False
        or design.get("non_overwrite_policy") is not True
        or design.get("historical_alpha_verdict") != "INVALID"
        or design.get("deployment_verdict") != "NOT AUTHORIZED"
    ):
        raise ValueError("design receipt top-level constraints are invalid")
    score_manifest = _verify_json(
        args.score_manifest, "sec_filing_change_score_manifest_v2"
    )
    if set(score_manifest) != {
        "schema_version", "created_at", "inputs", "outputs", "outcome_join_performed",
        "upstream_coverage_status", "all_signal_build_coverage_gates_passed",
        "benchmark_clock_deferred_to_outcome_runner",
    }:
        raise ValueError("score manifest fields differ from the frozen contract")
    score_input_roles = {record["role"] for record in score_manifest["inputs"]}
    score_output_roles = {record["role"] for record in score_manifest["outputs"]}
    if score_input_roles != {
        "design_receipt", "human_protocol", "acquisition_manifest",
        "acquisition_summary", "acquisition_audit", "acquisition_failures",
        "dataset_records", "frozen_acquisition_documents",
        "mapped_header_inventory", "rough_pairing_ledger", "mapping_misses",
        "score_builder", "feature_code",
    } or score_output_roles != {
        "document_parse_audit", "strict_fiscal_pairs", "strict_pair_exclusions",
        "raw_similarity_scores", "sorted_similarity_scores",
        "monthly_cutoff_ledger", "score_summary",
    }:
        raise ValueError("score manifest roles differ from the frozen contract")
    if (
        score_manifest.get("outcome_join_performed") is not False
        or score_manifest.get("upstream_coverage_status")
        != "PASS_UPSTREAM_SIGNAL_BUILD_COVERAGE"
        or score_manifest.get("all_signal_build_coverage_gates_passed") is not True
        or score_manifest.get("benchmark_clock_deferred_to_outcome_runner") is not True
    ):
        raise ValueError("score manifest is blocked or contains an outcome join")
    if _path_for_role(score_manifest, "sorted_similarity_scores") != args.scores.resolve():
        raise ValueError("scores path differs from its frozen manifest")
    if _path_for_role(score_manifest, "design_receipt") != args.design_receipt.resolve():
        raise ValueError("score manifest was built under a different design receipt")
    if _path_for_role(score_manifest, "human_protocol") != args.config.resolve():
        raise ValueError("score manifest was built under a different protocol")
    score_summary_path = _path_for_role(score_manifest, "score_summary")
    score_summary = json.loads(score_summary_path.read_text(encoding="utf-8"))
    if (
        score_summary.get("schema_version") != "sec_filing_change_score_summary_v2"
        or score_summary.get("status") != "DETERMINISTIC_SIGNAL_BUILD_ONLY"
        or score_summary.get("outcome_join_performed") is not False
        or score_summary.get("upstream_coverage", {}).get(
            "all_signal_build_gates_passed"
        )
        is not True
        or score_summary.get("upstream_coverage", {}).get("status")
        != "PASS_UPSTREAM_SIGNAL_BUILD_COVERAGE"
    ):
        raise ValueError("score summary failed upstream data-coverage gates")
    if _path_for_role(design, "frozen_retrospective_daily_prices") != args.prices.resolve():
        raise ValueError("daily prices differ from the frozen design receipt")
    if _path_for_role(design, "xnys_regular_sessions") != args.xnys_regular_sessions.resolve():
        raise ValueError("XNYS sessions differ from the frozen design receipt")
    if _path_for_role(design, "xnys_calendar_receipt") != args.xnys_calendar_receipt.resolve():
        raise ValueError("XNYS receipt differs from the frozen design receipt")
    if _path_for_role(design, "calendar_snapshot_generator") != args.calendar_snapshot_generator.resolve():
        raise ValueError("XNYS generator differs from the frozen design receipt")
    if _path_for_role(design, "human_protocol") != args.config.resolve():
        raise ValueError("protocol path differs from the frozen design receipt")
    if _path_for_role(design, "outcome_runner") != Path(__file__).resolve():
        raise ValueError("outcome runner differs from the frozen design receipt")
    if _path_for_role(design, "outcome_core_code") != (
        SRC / "alpha_factor_engine/research/sec_filing_change_stage_a.py"
    ).resolve():
        raise ValueError("outcome core differs from the frozen design receipt")
    if _path_for_role(score_manifest, "score_builder") != (
        ROOT / "scripts/build_sec_filing_change_scores.py"
    ).resolve():
        raise ValueError("score builder differs from its score manifest")
    if _path_for_role(score_manifest, "feature_code") != (
        SRC / "alpha_factor_engine/features/filing_change.py"
    ).resolve():
        raise ValueError("feature code differs from its score manifest")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != "sec_filing_change_stage_a_protocol_v1":
        raise ValueError("unsupported filing-change protocol")
    exchange_source = config.get("sources", {}).get("exchange_calendar", {})
    canonical_calendar_paths = {
        "path": args.xnys_regular_sessions,
        "receipt": args.xnys_calendar_receipt,
        "generator": args.calendar_snapshot_generator,
    }
    if not isinstance(exchange_source, dict) or any(
        (ROOT / str(exchange_source.get(key, ""))).resolve() != supplied.resolve()
        for key, supplied in canonical_calendar_paths.items()
    ):
        raise ValueError("XNYS inputs differ from protocol canonical paths")
    calendar_receipt = _verify_json(
        args.xnys_calendar_receipt, "xnys_calendar_snapshot_receipt_v1"
    )
    if (
        calendar_receipt.get("calendar") != "XNYS"
        or calendar_receipt.get("source_status")
        != "INDEPENDENT_VERSIONED_COMMUNITY_RULESET_NOT_EXCHANGE_LICENSED_DATA"
        or _path_for_role(calendar_receipt, "xnys_regular_sessions")
        != args.xnys_regular_sessions.resolve()
        or _path_for_role(calendar_receipt, "calendar_snapshot_generator")
        != args.calendar_snapshot_generator.resolve()
    ):
        raise ValueError("XNYS calendar receipt constraints are invalid")
    output = args.output.resolve()
    output.relative_to(ROOT.resolve())
    if output.exists():
        raise FileExistsError(f"non-overwriting Stage A output exists: {output}")
    scores = pd.read_parquet(args.scores)
    scores["current_accepted_at"] = pd.to_datetime(scores["current_accepted_at"], utc=True)
    local_release = (
        scores["current_accepted_at"]
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
        .dt.to_period("M")
    )
    if "release_month" in scores.columns:
        supplied_release = pd.Series(
            pd.PeriodIndex(scores["release_month"].astype(str), freq="M"),
            index=scores.index,
        )
        if not supplied_release.equals(local_release):
            raise ValueError("score release_month differs from New York acceptance clock")
    scores["release_month"] = local_release
    benchmark_prices = pd.read_parquet(
        args.prices,
        columns=["ticker", "date", "timestamp"],
        filters=[("ticker", "==", "SPY")],
    )
    exchange_calendar = pd.read_csv(args.xnys_regular_sessions)
    benchmark_calendar_reconciliation = reconcile_benchmark_to_exchange_calendar(
        benchmark_prices,
        exchange_calendar,
        required_start=config["sources"]["daily_prices"]["start"],
        required_end=config["sources"]["daily_prices"]["end"],
    )
    output.mkdir(parents=True)
    _write_json(
        output / "benchmark_calendar_reconciliation.json",
        benchmark_calendar_reconciliation,
    )
    cohorts = config["cohorts"]
    development_start = pd.Period(
        cohorts["development"]["release_month_gte"], freq="M"
    )
    development_end = pd.Period(
        cohorts["development"]["release_month_lte"], freq="M"
    )
    development = scores.loc[
        scores["release_month"].between(development_start, development_end)
    ]
    if development.empty:
        raise ValueError("development cohort has no scored filings")
    confirmation_start = pd.Period(
        cohorts["confirmation"]["release_month_gte"], freq="M"
    )
    confirmation_end = pd.Period(
        cohorts["confirmation"]["release_month_lte"], freq="M"
    )
    confirmation = scores.loc[
        scores["release_month"].between(confirmation_start, confirmation_end)
    ].copy()
    if confirmation.empty:
        raise ValueError("confirmation cohort has no scored filings")
    trial_floor = int(config["inference"]["current_disclosed_trial_floor"])
    declared_trials = set(config["inference"]["new_formal_trials_before_2025"])
    implemented_trials = {
        "D_JACCARD_MONTHLY_NET30",
        "D_COSINE_MONTHLY_NET30",
        "C_JACCARD_MONTHLY_NET30",
        "C_JACCARD_MONTHLY_NET50",
        "C_JACCARD_EVENT_PREPLACEBO_63",
        "C_JACCARD_EVENT_POST_MINUS_PRE_63",
    }
    if declared_trials != implemented_trials:
        raise ValueError("implemented trial IDs differ from the frozen protocol")
    if set(config["inference"]["p_value_emitting_trials"]) != {
        "D_JACCARD_MONTHLY_NET30",
        "D_COSINE_MONTHLY_NET30",
        "C_JACCARD_MONTHLY_NET30",
        "C_JACCARD_EVENT_PREPLACEBO_63",
    } or set(config["inference"]["mean_gate_only_trials_without_p_value"]) != {
        "C_JACCARD_MONTHLY_NET50",
        "C_JACCARD_EVENT_POST_MINUS_PRE_63",
    }:
        raise ValueError("formal inference roles differ from the frozen protocol")
    frozen: dict[str, dict[str, Path]] = {}
    for candidate in ("jaccard", "cosine"):
        name = f"development_{candidate}"
        frozen[name] = _freeze_sample_schedules(
            name=name,
            signals=_candidate_scores(development, candidate),
            benchmark_prices=benchmark_prices,
            output=output,
            design_receipt=args.design_receipt,
            score_manifest=args.score_manifest,
            config_path=args.config,
        )
    development_event_exits: list[pd.Timestamp] = []
    development_calendar_exits: list[pd.Timestamp] = []
    for name in ("development_jaccard", "development_cosine"):
        event_schedule = pd.read_parquet(frozen[name]["event_schedule"])
        calendar_schedule = pd.read_parquet(frozen[name]["calendar_schedule"])
        if not event_schedule.empty:
            development_event_exits.append(pd.to_datetime(event_schedule["exit_date"]).max())
        if not calendar_schedule.empty:
            development_calendar_exits.append(
                pd.to_datetime(calendar_schedule["calendar_exit_date"]).max()
            )
    if not development_event_exits or not development_calendar_exits:
        raise ValueError("development schedules cannot establish cohort purge boundaries")
    frozen["confirmation_jaccard"] = _freeze_sample_schedules(
        name="confirmation_jaccard",
        signals=_candidate_scores(confirmation, "jaccard"),
        benchmark_prices=benchmark_prices,
        output=output,
        design_receipt=args.design_receipt,
        score_manifest=args.score_manifest,
        config_path=args.config,
        minimum_event_entry_after=max(development_event_exits),
        minimum_calendar_entry_at_or_after=max(development_calendar_exits),
    )

    development_reservation = _reserve_trial_bundle(
        registry=args.trial_registry,
        bundle="development",
        trial_ids=["D_JACCARD_MONTHLY_NET30", "D_COSINE_MONTHLY_NET30"],
        design_receipt=args.design_receipt,
        score_manifest=args.score_manifest,
        output=output,
    )

    def outcome_cutoff(names: tuple[str, ...]) -> pd.Timestamp:
        candidates: list[pd.Timestamp] = []
        for sample_name in names:
            event = pd.read_parquet(frozen[sample_name]["event_schedule"])
            calendar = pd.read_parquet(frozen[sample_name]["calendar_schedule"])
            if not event.empty:
                candidates.append(pd.to_datetime(event["exit_date"]).max())
            if not calendar.empty:
                candidates.append(pd.to_datetime(calendar["calendar_exit_date"]).max())
        if not candidates:
            raise ValueError(f"no scheduled outcomes for {names}")
        return max(candidates)

    price_columns = ["ticker", "date", "timestamp", "close", "volume", "adj_open"]
    development_cutoff = outcome_cutoff(("development_jaccard", "development_cosine"))
    development_prices = pd.read_parquet(
        args.prices,
        columns=price_columns,
        filters=[("date", "<=", development_cutoff.to_pydatetime())],
    )
    samples: dict[str, dict[str, object]] = {}
    gate_frames: list[pd.DataFrame] = []
    for candidate in ("jaccard", "cosine"):
        name = f"development_{candidate}"
        summary, gates = _evaluate_sample(
            name=name,
            frozen=frozen[name],
            prices=development_prices,
            output=output,
            trial_floor=trial_floor,
            gate_minimums=config["gates"]["development_minimums"],
            config=config,
            expected_month_start=cohorts["development"]["formal_evaluation_month_gte"],
            expected_month_end=cohorts["development"]["formal_evaluation_month_lte"],
            event_inference_required=False,
        )
        samples[name] = summary
        gates.insert(0, "sample", name)
        gate_frames.append(gates)

    primary_passed = bool(
        samples["development_jaccard"]["stage_a_observed_gates_passed"]
    )
    development_data_complete = all(
        samples[f"development_{candidate}"]["formal_calendar_exact_price_path_coverage"]
        == 1.0
        for candidate in ("jaccard", "cosine")
    )
    confirmation_joined = False
    confirmation_reservation: Path | None = None
    if primary_passed and development_data_complete:
        confirmation_reservation = _reserve_trial_bundle(
            registry=args.trial_registry,
            bundle="confirmation",
            trial_ids=[
                "C_JACCARD_MONTHLY_NET30",
                "C_JACCARD_MONTHLY_NET50",
                "C_JACCARD_EVENT_PREPLACEBO_63",
                "C_JACCARD_EVENT_POST_MINUS_PRE_63",
            ],
            design_receipt=args.design_receipt,
            score_manifest=args.score_manifest,
            output=output,
        )
        confirmation_cutoff = outcome_cutoff(("confirmation_jaccard",))
        confirmation_prices = pd.read_parquet(
            args.prices,
            columns=price_columns,
            filters=[("date", "<=", confirmation_cutoff.to_pydatetime())],
        )
        confirmation_summary, confirmation_gates = _evaluate_sample(
            name="confirmation_jaccard",
            frozen=frozen["confirmation_jaccard"],
            prices=confirmation_prices,
            output=output,
            trial_floor=trial_floor,
            gate_minimums=config["gates"]["confirmation_minimums"],
            config=config,
            expected_month_start=cohorts["confirmation"]["formal_evaluation_month_gte"],
            expected_month_end=cohorts["confirmation"]["formal_evaluation_month_lte"],
            event_inference_required=True,
        )
        samples["confirmation_jaccard"] = confirmation_summary
        confirmation_gates.insert(0, "sample", "confirmation_jaccard")
        gate_frames.append(confirmation_gates)
        confirmation_joined = True

    all_gates = pd.concat(gate_frames, ignore_index=True)
    all_gates_path = output / "all_gates.csv"
    all_gates.to_csv(all_gates_path, index=False)
    chart_paths = _chart(output, samples)
    if not development_data_complete:
        decision = "BLOCKED_DATA_COVERAGE"
    elif not primary_passed:
        decision = "RETIRE_DETERMINISTIC_FILING_CHANGE_STAGE_A"
    elif not confirmation_joined:
        decision = "BLOCKED_CONFIRMATION_NOT_JOINED"
    elif (
        samples["confirmation_jaccard"]["formal_calendar_exact_price_path_coverage"]
        < 1.0
        or samples["confirmation_jaccard"]["event_exact_price_path_coverage"]
        < float(config["gates"]["coverage"]["event_exact_price_path"])
    ):
        decision = "BLOCKED_DATA_COVERAGE"
    elif not samples["confirmation_jaccard"]["stage_a_observed_gates_passed"]:
        decision = "RETIRE_DETERMINISTIC_FILING_CHANGE_STAGE_A"
    else:
        decision = "AWAIT_FAMILY_PIPELINE_PERMUTATION_AND_PERMANENT_SECURITY_STAGE_B"
    summary = {
        "schema_version": "sec_filing_change_stage_a_summary_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "decision": decision,
        "historical_alpha_verdict": RESEARCH_VERDICT,
        "model_verdict": "NOT VALIDATED",
        "deployment_verdict": "NOT AUTHORIZED",
        "primary_signal": "jaccard_similarity",
        "secondary_signal": "cosine_similarity",
        "development_primary_passed": primary_passed,
        "confirmation_signal_return_joined": confirmation_joined,
        "sealed_2025_signal_return_joined": False,
        "true_prospective_outer_signal_return_joined": False,
        "family_pipeline_permutation_completed": False,
        "deployability_gates_passed": False,
        "benchmark_calendar_reconciliation": benchmark_calendar_reconciliation,
        "samples": samples,
        "trial_reservations": [
            development_reservation.relative_to(ROOT).as_posix(),
            *(
                [confirmation_reservation.relative_to(ROOT).as_posix()]
                if confirmation_reservation is not None
                else []
            ),
        ],
        "limitations": [
            "Current-survivor ticker histories omit complete delisting and historical membership lineage.",
            "Historical borrow, locate failures, and nonlinear market impact are unavailable.",
            "Family-pipeline permutation remains a promotion gate even after positive confirmation."
        ],
        "design_receipt_verified": design.get("status") == "FROZEN_PRE_OUTCOME",
    }
    summary_path = output / "stage_a_summary.json"
    _write_json(summary_path, summary)
    output_records = [
        _record(path, f"stage_a_output::{path.name}")
        for path in sorted(output.iterdir())
        if path.is_file()
    ] + [
        _record(development_reservation, "development_trial_reservation"),
        *(
            [_record(confirmation_reservation, "confirmation_trial_reservation")]
            if confirmation_reservation is not None
            else []
        ),
    ]
    manifest = {
        "schema_version": "sec_filing_change_stage_a_manifest_v1",
        "created_at": summary["created_at"],
        "inputs": [
            _record(args.design_receipt, "design_receipt"),
            _record(args.score_manifest, "score_manifest"),
            _record(args.scores, "sorted_similarity_scores"),
            _record(args.prices, "daily_prices"),
            _record(args.xnys_regular_sessions, "xnys_regular_sessions"),
            _record(args.xnys_calendar_receipt, "xnys_calendar_receipt"),
            _record(Path(__file__), "outcome_runner"),
            _record(SRC / "alpha_factor_engine/research/sec_filing_change_stage_a.py", "outcome_core"),
        ],
        "outputs": output_records,
        "sealed_2025_signal_return_joined": False,
    }
    _write_json(output / "stage_a_manifest.json", manifest)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sec_filing_change_stage_a_v1.json")
    parser.add_argument(
        "--design-receipt", type=Path, default=ROOT / "artifacts/sec_filing_change_stage_a_v1/design_receipt_v2.json"
    )
    parser.add_argument(
        "--score-manifest", type=Path, default=ROOT / "artifacts/sec_filing_change_stage_a_v1/scores_full_v2/score_manifest.json"
    )
    parser.add_argument(
        "--scores", type=Path, default=ROOT / "artifacts/sec_filing_change_stage_a_v1/scores_full_v2/sorted_similarity_scores.parquet"
    )
    parser.add_argument(
        "--prices", type=Path, default=ROOT / "artifacts/public_broad_proxy_v1/data/daily_prices.parquet"
    )
    parser.add_argument(
        "--xnys-regular-sessions",
        type=Path,
        default=ROOT / "artifacts/reference_calendars/xnys_v1/xnys_regular_sessions_2014_2028.csv",
    )
    parser.add_argument(
        "--xnys-calendar-receipt",
        type=Path,
        default=ROOT / "artifacts/reference_calendars/xnys_v1/calendar_receipt.json",
    )
    parser.add_argument(
        "--calendar-snapshot-generator",
        type=Path,
        default=ROOT / "scripts/build_xnys_calendar_snapshot.py",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/sec_filing_change_stage_a_v1/run_v2"
    )
    parser.add_argument(
        "--trial-registry",
        type=Path,
        default=ROOT / "artifacts/sec_filing_change_stage_a_v1/trial_reservations_v2",
    )
    args = parser.parse_args()
    print(run(args))


if __name__ == "__main__":
    main()
