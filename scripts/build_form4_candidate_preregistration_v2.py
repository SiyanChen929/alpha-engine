#!/usr/bin/env python3
"""Build a return-blind Form 4 candidate census and frozen evaluation plan."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from xml.etree import ElementTree

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alpha_factor_engine.data.sec_insider import parse_insider_zip
from alpha_factor_engine.research.form4_candidate_prereg import (
    Form4CandidateSpec,
    build_clock_acquisition_plans,
    classify_candidate_events,
    extract_quarter_candidates,
    sha256_file,
    validate_global_lineage,
    verify_closed_manifest,
)


DEFAULT_CONFIG = ROOT / "configs/form4_event_preregistration_v2.yml"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _test_counts(path: Path) -> dict[str, int] | None:
    if not path.is_file():
        return None
    root = ElementTree.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    if not suites:
        raise RuntimeError(f"test report has no suites: {path}")
    return {
        key: sum(int(float(suite.attrib.get(key, "0"))) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }


def _build_spec(config: dict) -> Form4CandidateSpec:
    filt = config["candidate_filter"]
    split = config["sample_split"]
    return Form4CandidateSpec(
        minimum_direct_purchase_usd=float(filt["minimum_direct_purchase_usd"]),
        qualifying_role_pattern=str(filt["qualifying_owner_regex"]),
        include_security_pattern=str(filt["include_security_regex"]),
        exclude_security_pattern=str(filt["exclude_security_regex"]),
        aff10b5_effective_date=str(filt["aff10b5_effective_date"]),
        maximum_filing_lag_calendar_days=int(
            filt["maximum_filing_lag_calendar_days"]
        ),
        development_start=str(split["development_start"]),
        holdout_start=str(split["frozen_holdout_start"]),
        holdout_end=str(split["frozen_holdout_end"]),
        routine_lookback_years=int(
            config["owner_classification"]["lookback_complete_calendar_years"]
        ),
    )


def _ensure_preregistration(
    path: Path,
    *,
    config: dict,
    config_path: Path,
    inventory: pd.DataFrame,
) -> dict:
    design_files = [
        config_path,
        ROOT / "src/alpha_factor_engine/research/form4_candidate_prereg.py",
        Path(__file__).resolve(),
    ]
    design_hashes = {item.relative_to(ROOT).as_posix(): sha256_file(item) for item in design_files}
    source_snapshot = [
        {
            "quarter": str(row.quarter),
            "bytes": int(row.bytes),
            "sha256": str(row.sha256),
            "cache_path": str(Path(row.cache_path).resolve()),
        }
        for row in inventory.itertuples(index=False)
    ]
    invariant = {
        "schema_version": "form4_event_return_blind_preregistration_v2",
        "research_status": "NOT VALIDATED",
        "return_data_loaded": False,
        "event_return_backtest_run": False,
        "alpha_claim_permitted": False,
        "design_hashes": design_hashes,
        "source_snapshot": source_snapshot,
        "specification": config,
    }
    if path.is_file():
        existing = _read_json(path)
        comparable = {key: existing.get(key) for key in invariant}
        if comparable != invariant:
            raise RuntimeError(
                "frozen Form 4 preregistration conflicts with current design or source bytes"
            )
        return existing
    payload = {
        **invariant,
        "frozen_at": datetime.now(UTC).isoformat(),
        "freeze_boundary": (
            "Candidate counts and schemas were visible; no price, return, IC, "
            "portfolio, null, or holdout outcome was loaded."
        ),
        "holdout_reveal_policy": (
            "2025 outcomes may be opened once only after every fatal clock, "
            "security, return-lineage, controls, and cost gate passes."
        ),
    }
    _write_json(path, payload)
    return payload


def build(config_path: Path = DEFAULT_CONFIG) -> Path:
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != "form4_event_preregistration_v2":
        raise RuntimeError("unsupported Form 4 preregistration config")
    # PyYAML materializes unquoted ISO dates as ``datetime.date``.  Normalize
    # once so the frozen preregistration is strict, portable JSON.
    config = json.loads(json.dumps(config, default=str))
    output_dir = (ROOT / config["outputs"]["directory"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(UTC).isoformat()

    inventory_path = (ROOT / config["source_panel"]["inventory"]).resolve()
    upstream_manifest = (ROOT / config["source_panel"]["upstream_manifest"]).resolve()
    inventory = pd.read_csv(inventory_path, dtype={"quarter": "string", "sha256": "string"})
    expected = [f"{year}q{quarter}" for year in range(2020, 2026) for quarter in range(1, 5)]
    if inventory["quarter"].tolist() != expected or len(inventory) != 24:
        raise RuntimeError("source inventory is not the exact ordered 2020Q1-2025Q4 panel")
    if inventory["sha256"].duplicated().any():
        raise RuntimeError("source inventory contains duplicate archive hashes")

    spec = _build_spec(config)
    candidate_parts: list[pd.DataFrame] = []
    history_parts: list[pd.DataFrame] = []
    audit_rows: list[dict] = []
    schema_rows: list[dict] = []
    zip_paths: list[Path] = []
    for sequence, row in enumerate(inventory.itertuples(index=False), start=1):
        archive = Path(str(row.cache_path)).resolve()
        if not archive.is_file() or archive.is_symlink():
            raise RuntimeError(f"source archive missing or unsafe: {archive}")
        if archive.stat().st_size != int(row.bytes) or sha256_file(archive) != str(row.sha256):
            raise RuntimeError(f"source archive does not match inventory: {row.quarter}")
        parsed = parse_insider_zip(archive.read_bytes(), quarter=str(row.quarter))
        if parsed.archive_sha256 != str(row.sha256):
            raise RuntimeError(f"parser hash mismatch: {row.quarter}")
        candidates, history, audit = extract_quarter_candidates(
            parsed.submissions,
            parsed.reporting_owners,
            parsed.nonderivative_transactions,
            quarter=str(row.quarter),
            spec=spec,
        )
        candidate_parts.append(candidates)
        history_parts.append(history)
        audit_rows.append(dict(audit))
        schema_rows.extend(
            [
                {
                    "quarter": str(row.quarter),
                    "table": "SUBMISSION",
                    "column_count": int(len(parsed.submissions.columns)),
                    "columns": "|".join(parsed.submissions.columns),
                    "rows": int(len(parsed.submissions)),
                    "metadata_filename": parsed.metadata_filename,
                    "archive_sha256": parsed.archive_sha256,
                },
                {
                    "quarter": str(row.quarter),
                    "table": "REPORTINGOWNER",
                    "column_count": int(len(parsed.reporting_owners.columns)),
                    "columns": "|".join(parsed.reporting_owners.columns),
                    "rows": int(len(parsed.reporting_owners)),
                    "metadata_filename": parsed.metadata_filename,
                    "archive_sha256": parsed.archive_sha256,
                },
                {
                    "quarter": str(row.quarter),
                    "table": "NONDERIV_TRANS",
                    "column_count": int(len(parsed.nonderivative_transactions.columns)),
                    "columns": "|".join(parsed.nonderivative_transactions.columns),
                    "rows": int(len(parsed.nonderivative_transactions)),
                    "metadata_filename": parsed.metadata_filename,
                    "archive_sha256": parsed.archive_sha256,
                },
            ]
        )
        zip_paths.append(archive)
        print(f"candidate extraction {sequence}/24 {row.quarter}: {len(candidates):,}", flush=True)

    candidates = pd.concat(candidate_parts, ignore_index=True)
    history = pd.concat(history_parts, ignore_index=True)
    validate_global_lineage(candidates, history)
    candidates, owner_labels, event_owner_labels = classify_candidate_events(
        candidates,
        history,
        classification_years=(2023, 2024, 2025),
        lookback_years=spec.routine_lookback_years,
    )
    raw_plan, daily_plan, clock_windows = build_clock_acquisition_plans(
        candidates,
        history,
        daily_search_business_days_after=int(
            config["clock"]["daily_search_business_days_after"]
        ),
    )

    candidate_path = output_dir / "candidate_accessions.parquet"
    history_path = output_dir / "owner_history_events.parquet"
    owner_labels_path = output_dir / "provisional_owner_classifications.parquet"
    event_owner_path = output_dir / "candidate_owner_classifications.parquet"
    candidates.to_parquet(candidate_path, index=False)
    history.to_parquet(history_path, index=False)
    owner_labels.to_parquet(owner_labels_path, index=False)
    event_owner_labels.to_parquet(event_owner_path, index=False)

    audits = pd.DataFrame(audit_rows).sort_values("quarter", kind="mergesort")
    audit_path = output_dir / "filter_funnel_by_quarter.csv"
    audits.to_csv(audit_path, index=False, lineterminator="\n")
    schemas = pd.DataFrame(schema_rows).sort_values(["quarter", "table"], kind="mergesort")
    schema_path = output_dir / "source_schema_audit.csv"
    schemas.to_csv(schema_path, index=False, lineterminator="\n")

    raw_plan_path = output_dir / "raw_sgml_acquisition_plan.csv"
    daily_plan_path = output_dir / "daily_index_acquisition_plan.csv"
    window_path = output_dir / "accession_daily_search_windows.csv"
    raw_plan.to_csv(raw_plan_path, index=False, lineterminator="\n")
    daily_plan.to_csv(daily_plan_path, index=False, lineterminator="\n")
    clock_windows.to_csv(window_path, index=False, lineterminator="\n")

    candidates["year"] = pd.to_datetime(candidates["filing_date"]).dt.year
    year_rows: list[dict] = []
    for year, frame in candidates.groupby("year", sort=True):
        owners: set[str] = set()
        for value in frame["qualifying_owner_ciks"]:
            owners.update(token for token in str(value).split("|") if token)
        year_rows.append(
            {
                "year": int(year),
                "phase": (
                    "holdout_2025"
                    if year == 2025
                    else "development_2023_2024"
                    if year in (2023, 2024)
                    else "history_only"
                ),
                "structural_candidate_accessions": int(len(frame)),
                "candidate_issuers": int(frame["issuer_cik"].nunique()),
                "candidate_tickers": int(frame["ticker"].nunique()),
                "qualifying_owners": int(len(owners)),
                "preclock_evaluation_candidates": int(
                    frame["preclock_evaluation_candidate"].sum()
                ),
                "provisional_opportunistic_accessions": int(
                    frame["provisional_event_classification"].eq("opportunistic").sum()
                ),
                "preliminary_primary_candidates": int(
                    frame["preliminary_primary_candidate"].sum()
                ),
                "research_eligible_accessions": 0,
            }
        )
    year_counts = pd.DataFrame(year_rows)
    year_counts_path = output_dir / "candidate_counts_by_year.csv"
    year_counts.to_csv(year_counts_path, index=False, lineterminator="\n")

    class_counts = (
        owner_labels.groupby(["classification_year", "classification"], sort=True)
        .size()
        .rename("owner_count")
        .reset_index()
    )
    class_counts_path = output_dir / "owner_classification_counts.csv"
    class_counts.to_csv(class_counts_path, index=False, lineterminator="\n")

    source_ledger = pd.DataFrame(
        [
            {
                "source_id": "SEC_FORM4_INSTRUCTIONS",
                "claim": "Code P covers a purchase on an exchange or from another person.",
                "source_type": "official_regulator_form_instructions",
                "url": "https://www.sec.gov/files/form4.pdf",
                "use": "candidate label is code-P purchase, not confirmed open-market purchase",
            },
            {
                "source_id": "SEC_INSIDER_DATASETS",
                "claim": "Quarterly Form 3/4/5 files are flattened as-filed research datasets.",
                "source_type": "official_regulator_documentation",
                "url": "https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets",
                "use": "source panel and as-filed amendment policy",
            },
            {
                "source_id": "SEC_EDGAR_ACCESS",
                "claim": "Raw filing and daily/full-index archives require fair-access compliant acquisition.",
                "source_type": "official_regulator_documentation",
                "url": "https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data",
                "use": "raw SGML and archived daily-index clock plan",
            },
            {
                "source_id": "COHEN_MALLOY_POMORSKI_2012",
                "claim": "A prior-three-year same-calendar-month rule separates routine from opportunistic insider trades.",
                "source_type": "primary_academic_paper",
                "url": "https://www.nber.org/papers/w16454",
                "use": "owner classification design; original result is not treated as transferred alpha",
            },
        ]
    )
    source_ledger_path = output_dir / "claim_source_ledger.csv"
    source_ledger.to_csv(source_ledger_path, index=False, lineterminator="\n")

    prereg_path = output_dir / "preregistration.json"
    prereg = _ensure_preregistration(
        prereg_path,
        config=config,
        config_path=config_path,
        inventory=inventory,
    )
    targeted_tests = _test_counts(output_dir / "targeted-tests.xml")
    if targeted_tests is not None and (
        targeted_tests["tests"] <= 0
        or targeted_tests["failures"]
        or targeted_tests["errors"]
    ):
        raise RuntimeError(f"targeted tests did not pass: {targeted_tests}")

    evaluation = candidates.loc[candidates["preclock_evaluation_candidate"]]
    preliminary = candidates.loc[candidates["preliminary_primary_candidate"]]
    preliminary_holdout = preliminary.loc[preliminary["evaluation_phase"].eq("holdout_2025")]
    unique_candidate_owners = {
        token
        for value in candidates["qualifying_owner_ciks"]
        for token in str(value).split("|")
        if token
    }
    summary = {
        "schema_version": "form4_candidate_preregistration_evidence_v2",
        "generated_at": generated_at,
        "preregistration_frozen_at": prereg["frozen_at"],
        "research_status": "NOT VALIDATED",
        "data_artifact_status": "RETURN_BLIND_CANDIDATE_CENSUS_COMPLETE",
        "return_data_loaded": False,
        "event_return_backtest_run": False,
        "holdout_returns_opened": False,
        "performance_claim_permitted": False,
        "alpha_claim_permitted": False,
        "source_quarters": int(len(inventory)),
        "source_zip_bytes": int(inventory["bytes"].sum()),
        "source_archives_verified": True,
        "structural_candidate_accessions": int(len(candidates)),
        "candidate_issuers": int(candidates["issuer_cik"].nunique()),
        "candidate_tickers": int(candidates["ticker"].nunique()),
        "candidate_qualifying_owners": int(len(unique_candidate_owners)),
        "owner_history_event_rows": int(len(history)),
        "preclock_evaluation_candidates": int(len(evaluation)),
        "preliminary_primary_opportunistic_candidates": int(len(preliminary)),
        "preliminary_holdout_primary_candidates": int(len(preliminary_holdout)),
        "research_eligible_accessions": 0,
        "raw_sgml_assets_required": int(len(raw_plan)),
        "raw_sgml_bytes_unknown_until_acquisition": True,
        "daily_index_weekday_assets_planned": int(len(daily_plan)),
        "daily_index_bytes_unknown_until_acquisition": True,
        "total_planned_sec_gets_before_extensions": int(len(raw_plan) + len(daily_plan)),
        "raw_clock_coverage": 0.0,
        "permanent_security_mapping_coverage": 0.0,
        "targeted_tests": targeted_tests,
        "primary_holdout": "2025-01-01 through 2025-12-31; unopened",
        "candidate_label_limitation": (
            "SEC code P may be an exchange purchase or a purchase from another person"
        ),
    }
    summary_path = output_dir / "run_summary.json"
    _write_json(summary_path, summary)

    gates = pd.DataFrame(
        [
            ("bulk_source_panel_verified", True, "24/24 SHA256-verified SEC quarterly archives"),
            ("actual_schema_audited", True, "72 quarter-table schema records; AFF10B5ONE transition retained"),
            ("candidate_keys_globally_unique", True, f"{len(candidates):,} unique accessions"),
            ("return_blind_preregistration_frozen", True, prereg["frozen_at"]),
            ("exact_raw_sgml_acceptance_coverage", False, f"0/{len(raw_plan):,}"),
            ("earliest_cross_quarter_daily_evidence", False, f"0/{len(daily_plan):,} planned weekday assets"),
            ("effective_dated_permanent_security_mapping", False, "0% mapped; ticker is not identity"),
            ("corporate_action_delisting_return_lineage", False, "no return data loaded"),
            ("matched_controls_costs_nulls_multiplicity", False, "protocol frozen; outcomes not run"),
            ("2025_holdout_revealed", False, "sealed until fatal data gates pass"),
            ("performance_claim_permitted", False, "candidate census only"),
        ],
        columns=["gate", "passed", "evidence"],
    )
    gates_path = output_dir / "validation_gates.csv"
    gates.to_csv(gates_path, index=False, lineterminator="\n")

    report = f"""# Form 4 opportunistic-purchase candidate preregistration v2

## Verdict

**Candidate census complete; research status: NOT VALIDATED.** The 24 official
quarterly archives are parsed and a return-blind 2025 holdout protocol is
frozen. No prices, returns, ICs, portfolios, or holdout outcomes were loaded.

The source contains `{len(candidates):,}` structural direct-purchase candidate
accessions, `{candidates['issuer_cik'].nunique():,}` issuers and
`{len(unique_candidate_owners):,}` qualifying officer/director CIKs. After the
predeclared timeliness and post-April-2023 explicit non-10b5-1 screens,
`{len(evaluation):,}` development/holdout accessions require clock enrichment.
Using the filing-date end-of-day proxy only, `{len(preliminary):,}` are
provisionally opportunistic and `{len(preliminary_holdout):,}` fall in the
unopened 2025 holdout. These counts are acquisition diagnostics, not alpha.

## Why the label is conservative

SEC transaction code `P` covers a purchase on an exchange **or from another
person**. The package therefore says `code_p_purchase_candidate`; it cannot
prove a pure open-market execution from the flattened table alone. The primary
candidate uses original Form 4, non-derivative code P, acquired code A,
explicit non-swap, positive shares and price, common/ordinary equity, an
officer/director, direct ownership, at least $25,000, a non-late filing within
five calendar days, and explicit `AFF10B5ONE=false` after 2023-04-01. Indirect
ownership is retained for audit and excluded from primary selection. Form 4/A
never creates a second event or retroactively changes an earlier signal.

## Routine/opportunistic protocol

For a January 1 classification, an owner must have qualifying direct purchases
in each of the prior three complete calendar years. A common transaction month
in all three years is `routine`; complete history with no common month is
`opportunistic`; incomplete history is `unclassified`. Joint filings qualify as
opportunistic only when every qualifying owner is opportunistic. The current
labels use filing-date end-of-day as a conservative acquisition proxy and must
be recomputed from exact clocks before returns.

This mirrors the ex-ante classification idea in [Cohen, Malloy and Pomorski,
*Decoding Inside Information*](https://www.nber.org/papers/w16454), while making
no portability claim for the paper's historical performance.

## Exact data still required

- `{len(raw_plan):,}` raw complete-submission SGML files, each with a hashed
  `<ACCEPTANCE-DATETIME>` header.
- `{len(daily_plan):,}` cross-quarter weekday daily-index requests from the
  first search boundary through the last 2025 buffer. Retain bytes, SHA256,
  `Last-Modified`, ETag and retrieval time; document holidays/404s. For each
  accession, `information_at=max(raw acceptance, earliest archived daily-index
  Last-Modified)` and entry is the first regular open strictly later.
- A 100%-covered effective-dated CIK/share-class/permanent-security map. Ticker
  is a cross-check only; ambiguous share classes fail. Corporate actions,
  delistings and confirmed open prices must share the same lineage.

The source panel is the [SEC Insider Transactions Data Sets](https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets),
and acquisition must follow [SEC EDGAR access guidance](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data).

## Frozen outcome test

The single primary hypothesis is the 2025 opportunistic direct code-P cohort's
equal-weight 20-session next-open matched abnormal return. Each event requires
three same-entry-session, same-SIC2 controls selected on PIT ranks of market
cap, ADV, residual momentum, beta and idiosyncratic volatility. Secondary
horizons are 5 and 60 sessions. Costs are 15/30/50 bps one-way per traded leg;
30 bps is the primary deployability gate, with actual borrow and financing
required for a short control book. Two hundred conditional random-event nulls
preserve date/SIC2/liquidity strata. Inference is two-way clustered by issuer
and entry session, with Romano-Wolf stepdown and a conservative global
Bonferroni denominator of at least 111 trials.

The holdout opens once. An exact-clock or mapping gap blocks P&L. Fewer than 100
holdout events or 30 issuers blocks inference. A non-positive net estimate,
non-positive clustered lower 95% bound, adjusted significance failure,
conditional-null failure, positive placebo, 30-bps cost failure, or material
single-name/industry dependence retires this sleeve without retuning 2025.
"""
    report_path = output_dir / "report-source.md"
    report_path.write_text(report, encoding="utf-8")

    code_paths = [
        ROOT / "src/alpha_factor_engine/data/sec_insider.py",
        ROOT / "src/alpha_factor_engine/research/form4.py",
        ROOT / "src/alpha_factor_engine/research/form4_candidate_prereg.py",
        Path(__file__).resolve(),
        ROOT / "tests/test_form4_candidate_prereg.py",
    ]
    input_paths = [config_path, inventory_path, upstream_manifest, *zip_paths]
    output_paths = sorted(
        [path for path in output_dir.iterdir() if path.is_file() and path.name != "artifact_manifest.json"],
        key=lambda path: path.name,
    )
    records = []
    for role, paths in [("input", input_paths), ("code", code_paths), ("output", output_paths)]:
        for path in paths:
            records.append(
                {
                    "role": role,
                    "path": path.resolve().as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    manifest = {
        "schema_version": "form4_candidate_prereg_artifact_manifest_v1",
        "generated_at": generated_at,
        "research_status": "NOT VALIDATED",
        "performance_claim_permitted": False,
        "records": records,
        "output_file_count": len(output_paths),
        "output_record_count": sum(row["role"] == "output" for row in records),
        "output_omission_count": 0,
    }
    manifest_path = output_dir / "artifact_manifest.json"
    _write_json(manifest_path, manifest)
    if not verify_closed_manifest(manifest_path):  # pragma: no cover
        raise RuntimeError("artifact manifest failed verification")
    print(output_dir)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    build(args.config)


if __name__ == "__main__":
    main()
