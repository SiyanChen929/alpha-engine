#!/usr/bin/env python3
"""Build and optionally run a small synthetic Form 4 evidence bundle.

The bundle exists only to exercise clocks, horizons, matching, costs, ledger
reconciliation, and manifests.  It contains no observed security return.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_factor_engine.research.form4_event_runner import (
    INPUT_SCHEMA,
    SYNTHETIC_SOURCE,
    run_form4_event_study,
)


ACCESSION = "0001000000-24-000001"
ISSUER_CIK = "0001000000"


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _write(root: Path, relative: str, content: bytes) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _record(
    root: Path,
    role: str,
    relative: str,
    content: bytes,
    media_type: str,
    metadata: dict[str, object],
) -> dict[str, object]:
    _write(root, relative, content)
    return {
        "role": role,
        "path": relative,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "media_type": media_type,
        "metadata": metadata,
    }


def _synthetic_dates() -> pd.DatetimeIndex:
    dates = pd.bdate_range("2024-08-01", "2025-05-30")
    holidays = pd.to_datetime(
        [
            "2024-09-02",
            "2024-11-28",
            "2024-12-25",
            "2025-01-01",
            "2025-01-20",
            "2025-02-17",
            "2025-04-18",
            "2025-05-26",
        ]
    )
    return dates[~dates.isin(holidays)]


def _daily_master_content(*, include_event: bool) -> bytes:
    lines = [
        "Description: synthetic fixture; not SEC data",
        "CIK|Company Name|Form Type|Date Filed|Filename",
        "------------------------------------------------------------",
    ]
    if include_event:
        lines.append(
            "1000000|Synthetic Event Corp|4|2024-12-31|"
            f"edgar/data/1000000/{ACCESSION}.txt"
        )
    return ("\n".join(lines) + "\n").encode("latin-1")


def build_synthetic_bundle(bundle_dir: str | Path) -> Path:
    root = Path(bundle_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"fixture directory must be new or empty: {root}")
    root.mkdir(parents=True, exist_ok=True)

    dates = _synthetic_dates()
    local_opens = (
        dates + pd.Timedelta(hours=9, minutes=30)
    ).tz_localize("America/New_York")
    opens = local_opens.tz_convert("UTC")
    calendar = pd.DataFrame({"open_at": opens.astype("string")})

    master = pd.DataFrame(
        [
            ("SEC-EVT", ISSUER_CIK, "EVT", "35", False),
            ("SEC-CTLA", "0002000001", "CTLA", "35", False),
            ("SEC-CTLB", "0002000002", "CTLB", "35", False),
            ("SEC-CTLC", "0002000003", "CTLC", "35", False),
            ("SEC-SPY", "0000886982", "SPY", "67", True),
        ],
        columns=["security_id", "issuer_cik", "ticker", "sic2", "is_benchmark"],
    )
    master["known_at"] = "2020-01-01T00:00:00Z"
    master["effective_from"] = "2020-01-01T00:00:00Z"
    master["effective_to"] = ""
    master["is_primary_listing"] = True
    master = master[
        [
            "security_id",
            "issuer_cik",
            "ticker",
            "sic2",
            "known_at",
            "effective_from",
            "effective_to",
            "is_primary_listing",
            "is_benchmark",
        ]
    ]

    entry_date = pd.Timestamp("2025-01-02")
    entry_position = int(np.flatnonzero(dates == entry_date)[0])
    rows: list[dict[str, object]] = []
    definitions = [
        ("SEC-EVT", "EVT", 50.0, 0.00045, 2_100_000.0),
        ("SEC-CTLA", "CTLA", 49.0, 0.00040, 2_000_000.0),
        ("SEC-CTLB", "CTLB", 51.0, 0.00035, 2_200_000.0),
        ("SEC-CTLC", "CTLC", 65.0, 0.00010, 4_000_000.0),
        ("SEC-SPY", "SPY", 500.0, 0.00030, 80_000_000.0),
    ]
    for security_id, ticker, base, slope, volume in definitions:
        for position, (session, local_open) in enumerate(zip(dates, local_opens)):
            growth = (1.0 + slope) ** position
            if ticker == "EVT" and position > entry_position:
                growth *= (1.0 + 0.0015) ** (position - entry_position)
            open_price = base * growth
            end_of_day = (
                pd.Timestamp(session)
                .tz_localize("America/New_York")
                + pd.offsets.DateOffset(days=1)
                - pd.Timedelta(nanoseconds=1)
            ).tz_convert("UTC")
            rows.append(
                {
                    "security_id": security_id,
                    "ticker": ticker,
                    "date": session.date().isoformat(),
                    "open": round(float(open_price), 8),
                    "close": round(float(open_price * (1.0 + slope / 2.0)), 8),
                    "volume": volume,
                    "available_at": end_of_day.isoformat(),
                }
            )
    prices = pd.DataFrame(rows)

    events = pd.DataFrame(
        [
            {
                "event_id": "synthetic-event-1",
                "accession": ACCESSION,
                "filing_date": "2024-12-31",
                "document_type": "4",
                "issuer_cik": ISSUER_CIK,
                "ticker": "EVT",
                "security_id": "SEC-EVT",
                "sic2": "35",
                "transaction_code": "P",
                "acquired_disposed_code": "A",
                "equity_swap_involved": False,
                "rule_10b5_1": False,
                "qualifying_owner_role": "Director",
                "purchase_usd": 100_000.0,
            }
        ]
    )

    coverage_start = dates.min().date().isoformat()
    coverage_end = dates.max().date().isoformat()
    records: list[dict[str, object]] = []
    records.append(
        _record(
            root,
            "events",
            "tables/events.csv",
            _csv_bytes(events),
            "text/csv",
            {
                "authoritative": True,
                "signal_filter_complete": True,
                "economic_event_dedup_complete": True,
                "amendments_retained_and_primary_excluded": True,
                "source": "synthetic_fixture",
            },
        )
    )
    records.append(
        _record(
            root,
            "exchange_calendar",
            "tables/exchange_calendar.csv",
            _csv_bytes(calendar),
            "text/csv",
            {
                "authoritative": True,
                "exchange": "SYNTHETIC_XNYS",
                "source": "synthetic_fixture",
            },
        )
    )
    records.append(
        _record(
            root,
            "security_master",
            "tables/security_master.csv",
            _csv_bytes(master),
            "text/csv",
            {
                "authoritative": True,
                "point_in_time_complete": True,
                "issuer_cik_complete": True,
                "sic_history_complete": True,
                "source": "synthetic_fixture",
            },
        )
    )
    records.append(
        _record(
            root,
            "daily_prices",
            "tables/daily_prices.csv",
            _csv_bytes(prices),
            "text/csv",
            {
                "authoritative": True,
                "complete": True,
                "price_basis": "split_dividend_adjusted_ohlc",
                "source": "synthetic_fixture",
            },
        )
    )
    corporate = pd.DataFrame(
        columns=[
            "security_id",
            "effective_at",
            "known_at",
            "action_type",
            "split_factor",
            "cash_dividend",
        ]
    )
    records.append(
        _record(
            root,
            "corporate_actions",
            "tables/corporate_actions.csv",
            _csv_bytes(corporate),
            "text/csv",
            {
                "authoritative": True,
                "complete": True,
                "coverage_start": coverage_start,
                "coverage_end": coverage_end,
                "source": "synthetic_fixture",
            },
        )
    )
    delistings = pd.DataFrame(
        columns=["security_id", "delisting_at", "known_at", "outcome_status"]
    )
    records.append(
        _record(
            root,
            "delistings",
            "tables/delistings.csv",
            _csv_bytes(delistings),
            "text/csv",
            {
                "authoritative": True,
                "complete": True,
                "coverage_start": coverage_start,
                "coverage_end": coverage_end,
                "source": "synthetic_fixture",
            },
        )
    )
    payoff = pd.DataFrame(
        columns=[
            "security_id",
            "entry_at",
            "horizon_sessions",
            "event_total_return_including_delisting",
        ]
    )
    records.append(
        _record(
            root,
            "delisting_payoffs",
            "tables/delisting_payoffs.csv",
            _csv_bytes(payoff),
            "text/csv",
            {
                "authoritative": True,
                "complete": True,
                "source": "synthetic_fixture",
            },
        )
    )

    cost_model = {
        "model_version": "synthetic_cost_v1",
        "two_leg_complete": True,
        "event_one_way_bps": 15.0,
        "control_one_way_bps": 20.0,
        "control_borrow_annual_bps": 100.0,
        "event_financing_annual_bps": 25.0,
        "control_financing_annual_bps": 25.0,
        "calendar_days_per_year": 365,
        "included_components": [
            "commission",
            "bid_ask_spread",
            "slippage",
            "market_impact",
            "borrow",
            "financing",
        ],
    }
    records.append(
        _record(
            root,
            "cost_model",
            "spec/cost_model.json",
            _json_bytes(cost_model),
            "application/json",
            {"source": "synthetic_fixture"},
        )
    )
    run_spec = {
        "benchmark_ticker": "SPY",
        "benchmark_security_id": "SEC-SPY",
        "horizons": [5, 20, 60],
        "primary_horizon_sessions": 20,
        "controls_per_event": 2,
        "adv_window_sessions": 20,
        "volatility_window_sessions": 20,
        "minimum_history_sessions": 21,
        "minimum_price_usd": 5.0,
        "minimum_adv_dollars": 20_000_000.0,
        "maximum_staleness_calendar_days": 7,
        "minimum_purchase_usd": 25_000.0,
        "clock_search_start": "2024-12-31",
        "clock_search_end": "2025-01-03",
        "require_cross_quarter_daily_search": True,
        "price_basis": "split_dividend_adjusted_ohlc",
    }
    records.append(
        _record(
            root,
            "run_spec",
            "spec/run_spec.json",
            _json_bytes(run_spec),
            "application/json",
            {"frozen": True, "source": "synthetic_fixture"},
        )
    )

    raw_submission = (
        "<SEC-DOCUMENT>synthetic-test-only\n"
        "<SEC-HEADER>\n"
        "<ACCEPTANCE-DATETIME>20241231160000\n"
        "</SEC-HEADER>\n"
    ).encode("utf-8")
    records.append(
        _record(
            root,
            f"raw_submission:{ACCESSION}",
            f"sec/raw/{ACCESSION}.txt",
            raw_submission,
            "text/plain",
            {
                "accession": ACCESSION,
                "issuer_cik": ISSUER_CIK,
                "authoritative": True,
                "source_url": "synthetic://sec-complete-submission",
            },
        )
    )

    daily_definitions = [
        ("2024-12-31", "2024-12-31T23:00:00Z", False),
        ("2025-01-02", "2025-01-02T13:00:00Z", True),
        ("2025-01-03", "2025-01-03T23:00:00Z", True),
    ]
    for index_date, modified, include_event in daily_definitions:
        content = _daily_master_content(include_event=include_event)
        records.append(
            _record(
                root,
                f"daily_master:{index_date}",
                f"sec/daily/master.{index_date.replace('-', '')}.idx",
                content,
                "text/plain",
                {
                    "index_date": index_date,
                    "last_modified_at": modified,
                    "source_url": f"synthetic://sec-daily/{index_date}",
                    "etag": f"synthetic-{index_date}",
                    "retrieved_at": "2025-06-01T00:00:00Z",
                    "authoritative": True,
                },
            )
        )

    manifest = {
        "schema_version": INPUT_SCHEMA,
        "snapshot_id": "synthetic-form4-runner-v1",
        "source_class": SYNTHETIC_SOURCE,
        "created_at": datetime.now(UTC).isoformat(),
        "records": records,
    }
    manifest_path = root / "input_manifest.json"
    manifest_path.write_bytes(_json_bytes(manifest))
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    manifest = build_synthetic_bundle(args.bundle_dir)
    print(manifest)
    if args.output_dir is not None:
        result = run_form4_event_study(manifest, args.output_dir)
        print(result["run_manifest"])


if __name__ == "__main__":
    main()
