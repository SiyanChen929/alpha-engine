#!/usr/bin/env python3
"""Build a versioned XNYS regular-session reference snapshot.

This is a one-time provenance utility.  The research runner consumes only the
frozen CSV and receipt, so its runtime does not depend on exchange_calendars.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import exchange_calendars


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path, role: str) -> dict[str, object]:
    return {
        "role": role,
        "path": path.resolve().relative_to(ROOT.resolve()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2014-01-02")
    parser.add_argument("--end", default="2028-12-31")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "reference_calendars" / "xnys_v1",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.relative_to(ROOT.resolve())
    output.mkdir(parents=True, exist_ok=True)

    calendar = exchange_calendars.get_calendar(
        "XNYS", start=args.start, end=args.end
    )
    schedule = calendar.schedule.loc[:, ["open", "close"]].copy()
    schedule.insert(0, "session_date", schedule.index.strftime("%Y-%m-%d"))
    schedule = schedule.rename(
        columns={"open": "market_open_utc", "close": "market_close_utc"}
    ).reset_index(drop=True)
    csv_path = output / "xnys_regular_sessions_2014_2028.csv"
    schedule.to_csv(csv_path, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")

    generator = Path(__file__).resolve()
    payload = {
        "schema_version": "xnys_calendar_snapshot_receipt_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "calendar": "XNYS",
        "calendar_semantics": "regular exchange sessions and scheduled open/close timestamps",
        "first_session": str(schedule["session_date"].iloc[0]),
        "last_session": str(schedule["session_date"].iloc[-1]),
        "sessions": int(len(schedule)),
        "exchange_calendars_version": version("exchange_calendars"),
        "tzdata_version": version("tzdata"),
        "source_status": "INDEPENDENT_VERSIONED_COMMUNITY_RULESET_NOT_EXCHANGE_LICENSED_DATA",
        "crosscheck_source": "https://www.nyse.com/trade/hours-calendars",
        "package_source": "https://pypi.org/project/exchange-calendars/4.11.2/",
        "records": [
            record(generator, "calendar_snapshot_generator"),
            record(csv_path, "xnys_regular_sessions"),
        ],
    }
    receipt = output / "calendar_receipt.json"
    receipt.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
