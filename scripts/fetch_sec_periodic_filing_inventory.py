#!/usr/bin/env python3
"""Resumably cache SEC submissions headers needed for fiscal-state barriers."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alpha_factor_engine.data.sec import SecCompanyFactsClient


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(gzip.decompress(path.read_bytes()))


def _periodic_count(payload: dict[str, Any], start: pd.Timestamp) -> int:
    forms = ((payload.get("filings") or {}).get("recent") or {}).get("form") or []
    filed = ((payload.get("filings") or {}).get("recent") or {}).get("filingDate") or []
    return sum(
        form in {"10-Q", "10-Q/A", "10-K", "10-K/A"}
        and pd.notna(stamp)
        and stamp >= start
        for form, stamp in zip(
            forms, pd.to_datetime(pd.Series(filed, dtype="string"), errors="coerce")
        )
    )


def run(args: argparse.Namespace) -> Path:
    user_agent = args.user_agent or os.getenv("SEC_USER_AGENT")
    if not user_agent:
        raise ValueError("provide --user-agent or SEC_USER_AGENT with a real contact email")
    if not re.search(r"[^\s@]+@[^\s@]+\.[^\s@]+", user_agent):
        raise ValueError("SEC user agent lacks a contact email")
    mapping = pd.read_csv(args.mapping)
    ciks = sorted(pd.to_numeric(mapping["cik"], errors="raise").astype(int).unique())
    client = SecCompanyFactsClient(
        cache_dir=args.cache_dir,
        user_agent=user_agent,
        timeout=args.timeout,
        min_interval_seconds=args.minimum_interval_seconds,
    )
    start = pd.Timestamp(args.start_date)
    rows: list[dict[str, object]] = []
    raw_paths: list[Path] = []
    for index, cik in enumerate(ciks, start=1):
        main_path = args.cache_dir / "submissions" / f"CIK{cik:010d}.json.gz"
        status = "cached" if main_path.is_file() else "downloaded"
        payload, _ = client._get_json(
            client.submissions_url.format(cik=cik), main_path, refresh=False
        )
        raw_paths.append(main_path)
        archive_selected = 0
        archive_periodic = 0
        for archive in (payload.get("filings") or {}).get("files") or []:
            filing_to = pd.to_datetime(archive.get("filingTo"), errors="coerce")
            filing_from = pd.to_datetime(archive.get("filingFrom"), errors="coerce")
            if pd.isna(filing_to) or filing_to < start:
                continue
            name = str(archive.get("name") or "")
            if not name.startswith(f"CIK{cik:010d}-submissions-") or not name.endswith(".json"):
                raise ValueError(f"unexpected SEC archive name for CIK {cik}: {name}")
            archive_path = (
                args.cache_dir
                / "submissions_archive"
                / f"CIK{cik:010d}"
                / f"{name}.gz"
            )
            archive_payload, _ = client._get_json(
                f"https://data.sec.gov/submissions/{name}",
                archive_path,
                refresh=False,
            )
            raw_paths.append(archive_path)
            archive_selected += 1
            # Archive files use flat filing arrays rather than filings.recent.
            forms = archive_payload.get("form") or []
            dates = pd.to_datetime(
                pd.Series(archive_payload.get("filingDate") or [], dtype="string"),
                errors="coerce",
            )
            archive_periodic += sum(
                form in {"10-Q", "10-Q/A", "10-K", "10-K/A"}
                and pd.notna(filed)
                and filed >= start
                for form, filed in zip(forms, dates)
            )
            if pd.notna(filing_from) and filing_from > filing_to:
                raise ValueError(f"reversed archive dates for {name}")
        rows.append(
            {
                "cik": cik,
                "main_status": status,
                "main_sha256": _sha256(main_path),
                "recent_periodic_since_start": _periodic_count(payload, start),
                "archive_files_selected": archive_selected,
                "archive_periodic_since_start": archive_periodic,
            }
        )
        if index == 1 or index % 100 == 0 or index == len(ciks):
            print(f"SEC submissions inventory {index}/{len(ciks)}", flush=True)

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    audit = pd.DataFrame(rows)
    audit.to_csv(output / "issuer_inventory_receipt.csv", index=False)
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source": "SEC EDGAR submissions headers",
        "start_date": args.start_date,
        "issuers_requested": len(ciks),
        "issuers_cached": len(audit),
        "main_files": int(len(audit)),
        "selected_archive_files": int(audit["archive_files_selected"].sum()),
        "periodic_headers_since_start": int(
            audit["recent_periodic_since_start"].sum()
            + audit["archive_periodic_since_start"].sum()
        ),
        "claim": "inventory acquisition only; no alpha or performance claim",
    }
    (output / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "generated_at": summary["generated_at"],
        "records": [
            {
                "role": "raw_input",
                "path": path.resolve().as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(set(raw_paths))
        ],
    }
    (output / "raw_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output / "run_summary.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mapping",
        type=Path,
        default=ROOT / "artifacts/public_broad_proxy_v1/data/frozen_proxy_universe.csv",
    )
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/cache/sec")
    parser.add_argument("--start-date", default="2018-01-01")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/sec_periodic_filing_inventory_v1",
    )
    parser.add_argument("--user-agent")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--minimum-interval-seconds", type=float, default=0.13)
    args = parser.parse_args()
    print(run(args))


if __name__ == "__main__":
    main()
