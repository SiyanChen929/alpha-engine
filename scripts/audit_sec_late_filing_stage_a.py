#!/usr/bin/env python3
"""Independent reconstruction audit for the frozen SEC Form NT Stage A."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path, *, role: str) -> dict[str, object]:
    path = path.resolve()
    return {
        "role": role,
        "path": path.relative_to(ROOT.resolve()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def independent_hac(values: np.ndarray, indices: np.ndarray, lags: int = 3) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    idx = np.asarray(indices, dtype=int)
    mean = float(x.mean())
    centered = x - mean
    variance_sum = float(centered @ centered)
    for right in range(1, len(x)):
        gaps = idx[right] - idx[:right]
        mask = gaps <= lags
        if mask.any():
            weights = 1.0 - gaps[mask] / (lags + 1.0)
            variance_sum += float(2.0 * np.dot(weights, centered[right] * centered[:right][mask]))
    se = math.sqrt(variance_sum / (len(x) * len(x)))
    t_value = mean / se
    return {"mean": mean, "standard_error": se, "t": t_value, "one_sided_p": float(norm.sf(t_value))}


def check(name: str, passed: bool, detail: str, rows: list[dict[str, object]]) -> None:
    rows.append({"check": name, "passed": bool(passed), "detail": detail})


def run(args: argparse.Namespace) -> Path:
    run_dir = args.run_dir.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"non-overwriting audit directory exists: {output}")
    output.mkdir(parents=True)
    checks: list[dict[str, object]] = []

    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_records = list(manifest["inputs"]) + list(manifest["outputs"])
    for item in manifest_records:
        path = ROOT / str(item["path"])
        passed = path.is_file() and path.stat().st_size == int(item["bytes"]) and sha256_file(path) == item["sha256"]
        check(f"run_manifest:{item['role']}", passed, str(item["path"]), checks)

    design_path = ROOT / "artifacts/sec_late_filing_stage_a_v1/design_receipt_v1.json"
    design = json.loads(design_path.read_text(encoding="utf-8"))
    for item in design["records"]:
        path = ROOT / str(item["path"])
        passed = path.is_file() and path.stat().st_size == int(item["bytes"]) and sha256_file(path) == item["sha256"]
        check(f"design_receipt:{item['role']}", passed, str(item["path"]), checks)

    events = pd.read_csv(run_dir / "event_returns.csv")
    schedule = pd.read_parquet(run_dir / "preoutcome_schedule.parquet")
    sealed = pd.read_parquet(run_dir / "sealed_2025_signal_only.parquet")
    summary = json.loads((run_dir / "summary_metrics.json").read_text(encoding="utf-8"))
    confirmation = events.loc[events["phase"].eq("confirmation")].copy()
    check("sealed_count", len(sealed) == 15, f"observed={len(sealed)}", checks)
    overlap = set(sealed["accession"].astype(str)) & set(events["accession"].astype(str))
    check("sealed_outcomes_absent", not overlap, f"overlap={sorted(overlap)}", checks)
    check("confirmation_retained_count", len(confirmation) == 50, f"observed={len(confirmation)}", checks)
    check("confirmation_scheduled_count", int(schedule["phase"].eq("confirmation").sum()) == 95, "expected=95", checks)

    numeric_checks = {
        "stock_return_20": confirmation["stock_exit_adj_open"] / confirmation["stock_entry_adj_open"] - 1.0,
        "spy_return_20": confirmation["spy_exit_adj_open"] / confirmation["spy_entry_adj_open"] - 1.0,
    }
    numeric_checks["pair_gross_return_20"] = 0.5 * (
        numeric_checks["spy_return_20"] - numeric_checks["stock_return_20"]
    )
    traded = 1.0 + 0.5 * (1.0 + numeric_checks["stock_return_20"]) + 0.5 * (1.0 + numeric_checks["spy_return_20"])
    numeric_checks["pair_traded_notional_per_gross"] = traded
    numeric_checks["pair_net_30bps_return_20"] = numeric_checks["pair_gross_return_20"] - traded * 0.003
    numeric_checks["pair_net_50bps_return_20"] = numeric_checks["pair_gross_return_20"] - traded * 0.005
    for column, rebuilt in numeric_checks.items():
        difference = float(np.max(np.abs(rebuilt.to_numpy(dtype=float) - confirmation[column].to_numpy(dtype=float))))
        check(f"rebuild:{column}", difference <= 5e-12, f"max_abs_difference={difference:.3e}", checks)

    dates = (
        confirmation.groupby(["entry_date", "entry_index"], sort=True)[
            [
                "pair_gross_return_20",
                "pair_net_30bps_return_20",
                "pair_net_50bps_return_20",
                "pair_pre_event_gross_return_20",
                "post_minus_pre_return_20",
            ]
        ]
        .mean()
        .reset_index()
    )
    stored_dates = pd.read_csv(run_dir / "confirmation_entry_date_returns.csv")
    for column in [
        "pair_gross_return_20",
        "pair_net_30bps_return_20",
        "pair_net_50bps_return_20",
        "pair_pre_event_gross_return_20",
        "post_minus_pre_return_20",
    ]:
        diff = float(np.max(np.abs(dates[column].to_numpy() - stored_dates[column].to_numpy())))
        check(f"entry_date_aggregate:{column}", diff <= 5e-12, f"max_abs_difference={diff:.3e}", checks)

    metric_map = {
        "gross": "pair_gross_return_20",
        "primary_net_30bps": "pair_net_30bps_return_20",
        "stress_net_50bps": "pair_net_50bps_return_20",
        "pre_event_placebo": "pair_pre_event_gross_return_20",
        "post_minus_pre": "post_minus_pre_return_20",
    }
    audit_stats: dict[str, dict[str, float]] = {}
    for key, column in metric_map.items():
        rebuilt = independent_hac(dates[column].to_numpy(), dates["entry_index"].to_numpy())
        audit_stats[key] = rebuilt
        stored = summary["confirmation"][key]
        for field in ("mean", "standard_error", "t", "one_sided_p"):
            diff = abs(float(rebuilt[field]) - float(stored[field]))
            check(f"hac:{key}:{field}", diff <= 5e-12, f"difference={diff:.3e}", checks)

    run_decision = summary["candidate_decision"]
    gates = pd.read_csv(run_dir / "validation_gates.csv")
    check("decision_matches_gates", run_decision == "RETIRE_STAGE_A" and not gates["passed"].all(), f"decision={run_decision}", checks)
    check("primary_mean_negative", audit_stats["primary_net_30bps"]["mean"] < 0, f"mean={audit_stats['primary_net_30bps']['mean']:.8f}", checks)
    check("global_adjusted_p_one", float(summary["confirmation"]["primary_net_30bps"]["known_floor_adjusted_screening_p"]) == 1.0, "expected=1.0", checks)

    checks_frame = pd.DataFrame(checks)
    checks_path = output / "checks.csv"
    checks_frame.to_csv(checks_path, index=False)
    if not checks_frame["passed"].all():
        raise AssertionError(checks_frame.loc[~checks_frame["passed"]].to_dict("records"))

    sources = pd.DataFrame(
        [
            {
                "claim_id": "timing_strategy_prior",
                "claim": "Late expected earnings reports were associated with about one percent pre-cost abnormal return over 20 days in an early sample.",
                "source_title": "Abnormal Returns to Investment Strategies Based on the Timing of Earnings Reports",
                "author_or_publisher": "Stephen Penman; Journal of Accounting and Economics; Columbia Business School record",
                "publication_date": "1984-12-01",
                "url": "https://business.columbia.edu/faculty/research/abnormal-returns-investment-strategies-based-timing-earnings-reports",
                "access_note": "Abstract and citation accessed 2026-09-03; pre-cost historical prior, not a replication target.",
            },
            {
                "claim_id": "expected_date_information",
                "claim": "Missing management's expected earnings-report date conveys adverse earnings information.",
                "source_title": "The Information in Management's Expected Earnings Report Date: A Day Late, a Penny Short",
                "author_or_publisher": "Bagnoli, Kross, and Watts; Journal of Accounting Research",
                "publication_date": "2002-12",
                "url": "https://doi.org/10.1111/1475-679X.t01-1-00054",
                "access_note": "Publisher abstract accessed 2026-09-03; expected earnings date differs from SEC Form NT acceptance.",
            },
            {
                "claim_id": "form_nt_event_prior",
                "claim": "The authors report negative momentum-adjusted returns around Form NT filing dates, including -2.44 percent for combined NT 10-Q and NT 10-K.",
                "source_title": "SEC Filings, Regulatory Deadlines, and Capital Market Consequences",
                "author_or_publisher": "Eli Bartov and Yaniv Konchitchki; Accounting Horizons",
                "publication_date": "2017",
                "url": "https://faculty.haas.berkeley.edu/yaniv/files/Papers_Publications/SEC-Filings-Regulatory-Deadlines-Capital-Market-Consequences_Bartov-Konchitchki_AH.pdf",
                "access_note": "Full paper Table 5 accessed 2026-09-03; reported window is -2 to +2 and is not fully tradable after acceptance.",
            },
            {
                "claim_id": "sec_deadlines",
                "claim": "10-K and 10-Q deadlines differ by filer status; Form 12b-25 can provide limited automatic extensions.",
                "source_title": "Financial Reporting Manual, Topic 1",
                "author_or_publisher": "U.S. Securities and Exchange Commission",
                "publication_date": "page current when accessed",
                "url": "https://www.sec.gov/about/divisions-offices/division-corporation-finance/financial-reporting-manual/frm-topic-1",
                "access_note": "Official SEC guidance accessed 2026-09-03; motivates using the Form NT event instead of one universal calendar-lag threshold.",
            },
        ]
    )
    sources_path = output / "claim_source_ledger.csv"
    sources.to_csv(sources_path, index=False)

    audit_summary = {
        "schema_version": "sec_late_filing_independent_audit_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "checks": int(len(checks_frame)),
        "checks_passed": int(checks_frame["passed"].sum()),
        "independent_reconstruction_passed": True,
        "candidate_decision": run_decision,
        "confirmation_events": int(len(confirmation)),
        "confirmation_entry_dates": int(confirmation["entry_date"].nunique()),
        "sealed_2025_outcomes_joined": 0,
        "primary_net_30bps": audit_stats["primary_net_30bps"],
        "interpretation": "The paper prior does not survive the frozen next-open, liquid, costed implementation; the negative result is not an unexplained accounting mismatch.",
    }
    summary_path = output / "audit_summary.json"
    write_json(summary_path, audit_summary)
    report = f"""# Independent audit — SEC late-filing Stage A

## Verdict

The run is internally reproducible and the conservative retirement decision is supported. All {len(checks_frame)} independent lineage, arithmetic, aggregation, HAC, sealed-cohort, and decision checks passed.

- Frozen confirmation: 95 scheduled, 50 retained, 39 tickers, 33 entry dates.
- Gross short-stock/long-SPY mean: {audit_stats['gross']['mean']:.4%}; HAC t={audit_stats['gross']['t']:.3f}.
- Net at 30 bps one-way: {audit_stats['primary_net_30bps']['mean']:.4%}; HAC t={audit_stats['primary_net_30bps']['t']:.3f}.
- Global-floor adjusted p: 1.000.
- 2025 sealed events with joined outcomes: 0.

The literature result is not replicated by this tradable clock. The paper evidence includes pre-event and event-date reactions; this implementation waits for exact public SEC acceptance plus three minutes and enters at the next available regular open. The remaining 20-session return is negative for the predeclared short direction, before borrow and impact.

Coverage and capacity also fail. Only 52.6% of scheduled confirmation events survive exact-price and liquidity checks. At $10m, median stock trade participation is 8.7% of trailing ADV and the maximum is 312.2%; this is not institutional capacity.

Historical alpha remains `INVALID`, model alpha `NOT VALIDATED`, and deployment `NOT AUTHORIZED`. No sign reversal, form split, year split, liquidity threshold, or holding-period change may rescue this opened candidate; each would require a new trial and new untouched data.
"""
    report_path = output / "report-source.md"
    report_path.write_text(report, encoding="utf-8")
    outputs = [
        record(checks_path, role="independent_checks"),
        record(sources_path, role="claim_source_ledger"),
        record(summary_path, role="audit_summary"),
        record(report_path, role="audit_report"),
    ]
    manifest_out = {
        "schema_version": "sec_late_filing_independent_audit_manifest_v1",
        "created_at": audit_summary["created_at"],
        "inputs": [record(manifest_path, role="canonical_run_manifest"), record(Path(__file__), role="independent_auditor")],
        "outputs": outputs,
    }
    write_json(output / "artifact_manifest.json", manifest_out)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=ROOT / "artifacts/sec_late_filing_stage_a_v1/run_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/sec_late_filing_stage_a_v1/independent_audit_v1")
    args = parser.parse_args()
    print(run(args))


if __name__ == "__main__":
    main()
