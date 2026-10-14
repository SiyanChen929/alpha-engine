#!/usr/bin/env python3
"""Acquire the fixed Yahoo price bundle for Form 4 Stage A.

This script reads candidate metadata (including the presence of 2025 rows),
downloads only the tickers selected from the 2023-04-01 through 2024-12-31
cohort plus SPY, and writes price lineage. It never joins 2025 events to prices
or computes a 2025 event return.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alpha_factor_engine.data.yahoo import YahooFinanceClient
from alpha_factor_engine.research.form4_opportunistic_stage_a import (
    ACQUISITION_END,
    ACQUISITION_START,
    BENCHMARK,
    CANONICAL_ACQUISITION_DIRECTORY,
    CANONICAL_ACQUISITION_SCRIPT,
    CANONICAL_DESIGN_RECEIPT,
    CANONICAL_HUMAN_PROTOCOL,
    EVENT_CLASSES,
    PRICE_MANIFEST_SCHEMA,
    UNAMBIGUOUS_TICKER_PATTERN,
    Form4StageAError,
    _normalize_price_panel,
    _require_canonical_path,
    benchmark_clock_preflight,
    build_acquisition_universe,
    sha256_file,
    verify_design_receipt,
)


DEFAULT_CANDIDATES = (
    ROOT
    / "artifacts/form4_event_alpha_v1/"
    "candidate_preregistration_2020q1_2025q4_v2/"
    "candidate_accessions.parquet"
)
DEFAULT_CANDIDATE_UPSTREAM_MANIFEST = DEFAULT_CANDIDATES.parent / "artifact_manifest.json"
DEFAULT_YAHOO_ADAPTER = SRC / "alpha_factor_engine/data/yahoo.py"
DEFAULT_HUMAN_PROTOCOL = ROOT / "configs/form4_opportunistic_stage_a_v1.json"
DEFAULT_DESIGN_RECEIPT = (
    ROOT / "artifacts/form4_opportunistic_stage_a_v1/design_receipt.json"
)
DEFAULT_OUTPUT_DIR = ROOT / CANONICAL_ACQUISITION_DIRECTORY
DEFAULT_CORE = SRC / "alpha_factor_engine/research/form4_opportunistic_stage_a.py"
DEFAULT_OUTCOME_RUNNER = ROOT / "scripts/run_form4_opportunistic_stage_a.py"


def _relative(path: Path) -> str:
    target = path.resolve()
    try:
        return target.relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise Form4StageAError(f"acquisition path is outside repository: {target}") from exc


def _file_record(path: Path) -> dict[str, object]:
    return {
        "path": _relative(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=str)
        + "\n",
        encoding="utf-8",
    )


def acquire(
    *,
    candidate_path: Path,
    candidate_upstream_manifest: Path,
    yahoo_adapter: Path,
    human_protocol: Path,
    design_receipt: Path,
    output_dir: Path,
    cache_dir: Path,
    workers: int,
) -> dict[str, Path]:
    candidate_path = candidate_path.resolve()
    candidate_upstream_manifest = candidate_upstream_manifest.resolve()
    yahoo_adapter = yahoo_adapter.resolve()
    human_protocol = human_protocol.resolve()
    design_receipt = design_receipt.resolve()
    output_dir = output_dir.resolve()
    cache_dir = cache_dir.resolve()
    _require_canonical_path(
        Path(__file__),
        root=ROOT,
        canonical_relative=CANONICAL_ACQUISITION_SCRIPT,
        role="acquisition script",
    )
    _require_canonical_path(
        human_protocol,
        root=ROOT,
        canonical_relative=CANONICAL_HUMAN_PROTOCOL,
        role="human protocol",
    )
    _require_canonical_path(
        design_receipt,
        root=ROOT,
        canonical_relative=CANONICAL_DESIGN_RECEIPT,
        role="design receipt",
    )
    _require_canonical_path(
        output_dir,
        root=ROOT,
        canonical_relative=CANONICAL_ACQUISITION_DIRECTORY,
        role="acquisition directory",
    )
    _relative(candidate_path)
    _relative(candidate_upstream_manifest)
    _relative(yahoo_adapter)
    _relative(human_protocol)
    _relative(design_receipt)
    _relative(output_dir)
    _relative(cache_dir)
    if output_dir.exists():
        raise Form4StageAError(f"acquisition output directory already exists: {output_dir}")
    if workers < 1 or workers > 16:
        raise Form4StageAError("workers must be between 1 and 16")

    verify_design_receipt(
        design_receipt,
        expected_paths={
            "human_protocol": human_protocol,
            "core_code": DEFAULT_CORE,
            "outcome_runner": DEFAULT_OUTCOME_RUNNER,
            "acquisition_script": Path(__file__),
            "candidate_accessions": candidate_path,
            "candidate_upstream_manifest": candidate_upstream_manifest,
            "yahoo_adapter": yahoo_adapter,
        },
        repository_root=ROOT,
    )

    # This read uses labels and dates only.  There is no price/return join.
    candidates = pd.read_parquet(candidate_path)
    aliases, ticker_exclusions, audit_summary = build_acquisition_universe(candidates)
    requested = aliases["yahoo_ticker"].astype(str).tolist()
    client = YahooFinanceClient(cache_dir=cache_dir)
    downloaded, records, download_audit = client.fetch_many(
        requested,
        ACQUISITION_START,
        ACQUISITION_END,
        workers=workers,
        refresh=True,
    )
    failed = download_audit.loc[~download_audit["status"].eq("ok")]
    observed = set(downloaded["ticker"].astype(str))
    missing = set(requested).difference(observed)
    if BENCHMARK in missing or BENCHMARK in set(failed["ticker"].astype(str)):
        detail = failed[["ticker", "status"]].to_dict("records")
        raise Form4StageAError(
            f"essential SPY acquisition failed; missing={sorted(missing)}; failures={detail}"
        )
    for record in records:
        yahoo_ticker = record.dataset.rsplit(":", 1)[-1]
        if (
            record.provider != "Yahoo Finance chart API"
            or record.dataset != f"daily_prices:{yahoo_ticker}"
            or record.finality != "confirmed"
        ):
            raise Form4StageAError(
                f"unconfirmed or malformed Yahoo vendor record: {record.dataset}"
            )

    alias_lookup = aliases.rename(columns={"yahoo_ticker": "ticker"})
    panel = downloaded.merge(
        alias_lookup[["ticker", "source_ticker"]],
        on="ticker",
        how="left",
        validate="many_to_one",
    ).rename(columns={"ticker": "yahoo_ticker"})
    if panel["source_ticker"].isna().any():
        raise Form4StageAError("downloaded Yahoo ticker lacks a frozen source alias")
    columns = [
        "source_ticker",
        "yahoo_ticker",
        "date",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "adj_close",
        "adj_open",
        "dollar_volume",
        "effective_at",
        "available_at",
        "split_ratio",
        "dividend",
    ]
    missing_columns = set(columns).difference(panel.columns)
    if missing_columns:
        raise Form4StageAError(
            f"Yahoo adapter output missing columns: {sorted(missing_columns)}"
        )
    panel = panel[columns].copy()
    downloaded_panel_rows = int(len(panel))
    usable = panel["timestamp"].notna()
    for column in ("adj_open", "adj_close", "close", "volume"):
        numeric = pd.to_numeric(panel[column], errors="coerce")
        panel[column] = numeric
        usable &= np.isfinite(numeric)
    usable &= panel["adj_open"].gt(0)
    usable &= panel["adj_close"].gt(0)
    usable &= panel["close"].gt(0)
    usable &= panel["volume"].gt(0)
    panel = panel.loc[usable].sort_values(
        ["source_ticker", "date"], kind="mergesort"
    )
    if panel.duplicated(["source_ticker", "date"]).any():
        raise Form4StageAError("Yahoo price panel contains duplicate ticker dates")
    normalized_preflight = _normalize_price_panel(
        panel,
        expected_source_tickers=set(aliases["source_ticker"].astype(str)),
    )
    spy_preflight = normalized_preflight.loc[
        normalized_preflight["source_ticker"].eq(BENCHMARK)
    ]
    if len(spy_preflight) < 275:
        raise Form4StageAError("clean SPY history is too short for Stage A")
    if spy_preflight["date"].min() < pd.Timestamp(
        ACQUISITION_START
    ) or spy_preflight["date"].max() > pd.Timestamp(ACQUISITION_END):
        raise Form4StageAError("clean SPY dates fall outside the fixed request range")
    benchmark_preflight_audit = benchmark_clock_preflight(candidates, panel)

    output_dir.mkdir(parents=True, exist_ok=False)
    raw_archive_dir = output_dir / "raw"
    raw_archive_dir.mkdir()
    price_path = output_dir / "prices.parquet"
    audit_path = output_dir / "download_audit.csv"
    alias_path = output_dir / "ticker_aliases.csv"
    exclusion_path = output_dir / "ticker_exclusions.csv"
    manifest_path = output_dir / "price_manifest.json"
    panel.to_parquet(price_path, index=False)
    download_audit.sort_values("ticker", kind="mergesort").to_csv(audit_path, index=False)
    aliases.to_csv(alias_path, index=False)
    ticker_exclusions.to_csv(exclusion_path, index=False)

    vendor_records: list[dict[str, object]] = []
    by_dataset = {record.dataset.rsplit(":", 1)[-1]: record for record in records}
    if len(by_dataset) != len(records):
        raise Form4StageAError("duplicate Yahoo DatasetRecord ticker")
    for yahoo_ticker, record in sorted(by_dataset.items()):
        safe = YahooFinanceClient._safe_symbol(yahoo_ticker)
        download_cache = cache_dir / f"{safe}_{ACQUISITION_START}_{ACQUISITION_END}.json.gz"
        if not download_cache.is_file() or download_cache.is_symlink():
            raise Form4StageAError(f"missing Yahoo download cache: {download_cache}")
        try:
            raw_payload = gzip.decompress(download_cache.read_bytes())
        except (OSError, EOFError) as exc:
            raise Form4StageAError(
                f"invalid Yahoo download cache for {yahoo_ticker}"
            ) from exc
        if hashlib.sha256(raw_payload).hexdigest() != record.sha256:
            raise Form4StageAError(
                f"Yahoo cache payload changed after retrieval for {yahoo_ticker}"
            )
        raw_cache = raw_archive_dir / f"{safe}_{record.sha256}.json.gz"
        archived = gzip.compress(raw_payload, compresslevel=6, mtime=0)
        try:
            with raw_cache.open("xb") as handle:
                handle.write(archived)
        except FileExistsError as exc:
            raise Form4StageAError(
                f"content-addressed Yahoo archive already exists: {raw_cache}"
            ) from exc
        vendor_records.append(
            {
                **asdict(record),
                "yahoo_ticker": yahoo_ticker,
                "raw_cache_path": _relative(raw_cache),
                "raw_cache_bytes": raw_cache.stat().st_size,
                "raw_cache_sha256": sha256_file(raw_cache),
                "raw_payload_sha256": record.sha256,
            }
        )
    manifest = {
        "schema_version": PRICE_MANIFEST_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "provider": "Yahoo Finance chart API",
        "source_class": "current_ticker_yahoo_retrospective",
        "historical_alpha_verdict": "INVALID",
        "design_receipt": _file_record(design_receipt),
        "candidate_file": _file_record(candidate_path),
        "candidate_upstream_manifest": _file_record(candidate_upstream_manifest),
        "yahoo_adapter": _file_record(yahoo_adapter),
        "price_file": _file_record(price_path),
        "acquisition_script": _file_record(Path(__file__)),
        "support_files": {
            "download_audit": _file_record(audit_path),
            "ticker_aliases": _file_record(alias_path),
            "ticker_exclusions": _file_record(exclusion_path),
        },
        "acquisition": {
            "request_start": ACQUISITION_START,
            "request_end": ACQUISITION_END,
            "outcome_join_performed": False,
            "holdout_return_outcomes_joined": False,
            "design_receipt_verified_before_download": True,
            "fresh_network_retrieval_after_design_freeze": True,
            "candidate_classes": list(EVENT_CLASSES),
            "benchmark": BENCHMARK,
            "ticker_regex": UNAMBIGUOUS_TICKER_PATTERN.pattern,
            "alias_rule": "replace_period_with_hyphen",
        },
        "ticker_aliases": aliases.to_dict("records"),
        "vendor_records": vendor_records,
        "limitations": {
            "permanent_security_identifiers_complete": False,
            "delisting_returns_complete": False,
            "survivorship_free": False,
            "vendor_adjustments_may_be_revised": True,
            "candidate_clock_is_exact_first_public": False,
            "price_panel_cleaning": (
                "drop rows with missing timestamp, non-finite/non-positive "
                "adjusted/open/close prices, or non-positive volume"
            ),
            "raw_archive_policy": (
                "fresh retrieval after design freeze; deterministic gzip copy in "
                "the non-overwriting acquisition/raw directory named by payload sha256"
            ),
            "downloaded_panel_rows": downloaded_panel_rows,
            "retained_panel_rows": int(len(panel)),
            "removed_panel_rows": int(downloaded_panel_rows - len(panel)),
            "stage_a_candidate_audit": audit_summary,
            "benchmark_clock_preflight": benchmark_preflight_audit,
        },
    }
    _write_json(manifest_path, manifest)
    return {
        "prices": price_path,
        "download_audit": audit_path,
        "ticker_aliases": alias_path,
        "ticker_exclusions": exclusion_path,
        "price_manifest": manifest_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Acquire Stage A Yahoo history only; no event/outcome join and no "
            "2025 holdout event evaluation"
        )
    )
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument(
        "--candidate-upstream-manifest",
        type=Path,
        default=DEFAULT_CANDIDATE_UPSTREAM_MANIFEST,
    )
    parser.add_argument(
        "--yahoo-adapter", type=Path, default=DEFAULT_YAHOO_ADAPTER
    )
    parser.add_argument(
        "--human-protocol", type=Path, default=DEFAULT_HUMAN_PROTOCOL
    )
    parser.add_argument(
        "--design-receipt", type=Path, default=DEFAULT_DESIGN_RECEIPT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "data/cache/yahoo/form4_opportunistic_stage_a",
    )
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    outputs = acquire(
        candidate_path=args.candidates,
        candidate_upstream_manifest=args.candidate_upstream_manifest,
        yahoo_adapter=args.yahoo_adapter,
        human_protocol=args.human_protocol,
        design_receipt=args.design_receipt,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        workers=args.workers,
    )
    print(outputs["price_manifest"])


if __name__ == "__main__":
    main()
