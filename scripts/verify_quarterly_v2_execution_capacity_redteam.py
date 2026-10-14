#!/usr/bin/env python3
"""Build an independent trade-leg capacity audit without changing the producer."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alpha_factor_engine.research.execution_capacity_redteam import (  # noqa: E402
    reconstruct_trade_leg_capacity,
)


PANEL = ROOT / "artifacts/quarterly_fundamental_momentum_diagnostic_v2_attempt1_recovered/monthly_quarterly_signal_panel.parquet"
PRODUCER = ROOT / "artifacts/quarterly_v2_execution_diagnostic_v1"
HOLDINGS = PRODUCER / "holdings.parquet"
MONTHLY = PRODUCER / "monthly_execution.csv"
PRODUCER_MANIFEST = PRODUCER / "artifact_manifest.json"
OUTPUT = ROOT / "artifacts/quarterly_v2_execution_diagnostic_v1_capacity_redteam"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_producer_inputs() -> None:
    manifest = json.loads(PRODUCER_MANIFEST.read_text(encoding="utf-8"))
    expected = {Path(record["path"]).resolve(): record["sha256"] for record in manifest["records"]}
    for path in (PANEL, HOLDINGS, MONTHLY):
        if path.resolve() not in expected:
            raise ValueError(f"producer manifest omits required audit input: {path}")
        if sha256(path) != expected[path.resolve()]:
            raise ValueError(f"producer artifact drift: {path}")


def period_summary(frame: pd.DataFrame, mask: pd.Series) -> dict[str, object]:
    sample = frame.loc[mask].copy()
    complete = sample.loc[sample["actual_trade_leg_adv_complete"]].copy()
    producer_capacity = sample["producer_target_holdings_capacity_side_aum_usd_at_0p0100"]
    trade_capacity = complete["trade_leg_capacity_side_aum_usd_at_0p0100"]
    return {
        "months": int(len(sample)),
        "producer_target_holdings_adv_complete_months": int(
            sample["producer_target_holdings_adv_complete"].sum()
        ),
        "actual_trade_leg_adv_complete_months": int(
            sample["actual_trade_leg_adv_complete"].sum()
        ),
        "missing_adv_trade_legs": int(sample["missing_adv_trade_legs"].sum()),
        "trade_legs": int(sample["trade_leg_count"].sum()),
        "producer_target_holdings_capacity_1pct_minimum_usd": float(producer_capacity.min()),
        "producer_target_holdings_capacity_1pct_median_usd": float(producer_capacity.median()),
        "trade_leg_capacity_1pct_minimum_usd_complete_months_only": (
            float(trade_capacity.min()) if len(trade_capacity) else None
        ),
        "trade_leg_capacity_1pct_median_usd_complete_months_only": (
            float(trade_capacity.median()) if len(trade_capacity) else None
        ),
    }


def build(junit_path: Path | None) -> Path:
    if OUTPUT.exists():
        raise FileExistsError(f"audit output already exists: {OUTPUT}")
    verify_producer_inputs()
    panel = pd.read_parquet(PANEL)
    holdings = pd.read_parquet(HOLDINGS)
    monthly = pd.read_csv(MONTHLY)
    per_month = reconstruct_trade_leg_capacity(panel, holdings, monthly)
    if float(per_month["turnover_absolute_error"].max()) > 1e-12:
        raise ValueError("independent trade weights do not reconcile to producer turnover")

    OUTPUT.mkdir(parents=True)
    per_month.to_csv(OUTPUT / "per_month_trade_leg_coverage.csv", index=False)
    dates = pd.to_datetime(per_month["signal_date"], utc=True)
    primary_mask = dates.lt(pd.Timestamp("2026-01-01", tz="UTC"))
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "audit_scope": "capacity input coverage and deterministic ADV envelope only",
        "producer_artifact": PRODUCER.as_posix(),
        "producer_verdict_preserved": "INVALID",
        "research_status_preserved": "NOT_VALIDATED",
        "portfolio_authorized": False,
        "reconstruction_rule": "for each actual long/short entry resize or exit: side_AUM <= participation * current_signal_month_lagged_ADV / absolute_delta_weight; any missing trade-leg ADV makes monthly capacity unavailable",
        "maximum_turnover_reconciliation_error": float(
            per_month["turnover_absolute_error"].max()
        ),
        "full": period_summary(per_month, pd.Series(True, index=per_month.index)),
        "pre_2026": period_summary(per_month, primary_mask),
        "finding": "Producer complete_positive_adv covers target holdings, not every actual trade leg. Its capacity figures are holding-entry envelopes and cannot be used as executable rebalance capacity.",
    }
    write_json(OUTPUT / "summary.json", summary)

    pre = summary["pre_2026"]
    report = f"""# Quarterly-v2 execution capacity — independent red-team audit

## Verdict

The upstream historical claim remains **INVALID**, research remains **NOT VALIDATED**, and portfolio authorization remains **false**. This audit changes no producer or upstream file.

## Finding

The producer reports positive ADV for current D10/D1 target holdings in all `{pre['months']}` pre-2026 months. Actual execution also includes outgoing and resized positions. Reconstructing every nonzero trade from the frozen holdings and marked prior weights finds:

- actual-trade-leg ADV complete months: `{pre['actual_trade_leg_adv_complete_months']}/{pre['months']}`;
- missing outgoing/resized ADV trade legs: `{pre['missing_adv_trade_legs']}`;
- maximum turnover reconciliation error: `{summary['maximum_turnover_reconciliation_error']:.3e}`;
- producer 1% ADV holding-envelope minimum/median: `${pre['producer_target_holdings_capacity_1pct_minimum_usd']:,.0f}` / `${pre['producer_target_holdings_capacity_1pct_median_usd']:,.0f}`.

Only months with every actual trade leg covered receive a reconstructed capacity. Those conditional figures do not repair the other `{pre['months'] - pre['actual_trade_leg_adv_complete_months']}` months and are not deployable capacity.

## Reconstruction

For each sleeve and month, prior equal weights are marked by the prior holding return and renormalized to unit gross. For every entry, resize, and exit, `absolute_delta_weight` is independently reconstructed. At participation `p`, each trade requires `side_AUM <= p * current-month lagged ADV / absolute_delta_weight`; the monthly limit is the minimum over both sleeves. Any missing/nonpositive ADV on a nonzero trade leg fails that month closed.

Remaining blockers include ticker-only security identity, delistings, borrow/locates, financing, fill prices, spread, nonlinear impact, and an order ledger.
"""
    (OUTPUT / "report-source.md").write_text(report, encoding="utf-8")

    if junit_path is not None:
        if not junit_path.is_file():
            raise FileNotFoundError(junit_path)
        shutil.copy2(junit_path, OUTPUT / "targeted-tests.xml")

    record_paths = [
        ("input", PANEL),
        ("input", HOLDINGS),
        ("input", MONTHLY),
        ("input", PRODUCER_MANIFEST),
        ("code", Path(__file__)),
        ("code", ROOT / "src/alpha_factor_engine/research/execution_capacity_redteam.py"),
        ("test", ROOT / "tests/test_execution_capacity_redteam.py"),
    ]
    record_paths.extend(
        ("output", path)
        for path in sorted(OUTPUT.iterdir())
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    records = [
        {
            "role": role,
            "path": path.resolve().as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for role, path in record_paths
    ]
    write_json(
        OUTPUT / "artifact_manifest.json",
        {
            "generated_at": summary["generated_at"],
            "manifest_records": len(records),
            "producer_verdict_preserved": "INVALID",
            "research_status_preserved": "NOT_VALIDATED",
            "records": records,
        },
    )
    return OUTPUT / "report-source.md"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--junit", type=Path)
    args = parser.parse_args()
    print(build(args.junit))


if __name__ == "__main__":
    main()
