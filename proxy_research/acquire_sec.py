"""Resumable SEC Company Facts acquisition for the frozen proxy universe."""

from __future__ import annotations

import concurrent.futures
import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from alpha_factor_engine.contracts import sha256_file
from alpha_factor_engine.data.sec import ANNUAL_CONCEPTS, SecCompanyFactsClient
from proxy_research.acquire_prices import load_config


def _normalize_companyfacts_fast(
    ticker: str,
    cik: int,
    payload: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    facts = ((payload.get("facts") or {}).get("us-gaap") or {})
    for canonical, spec in ANNUAL_CONCEPTS.items():
        for priority, tag in enumerate(spec.tags):
            entries = (((facts.get(tag) or {}).get("units") or {}).get(spec.unit) or [])
            for item in entries:
                if item.get("form") not in {"10-K", "10-K/A"} or item.get("fp") not in {"FY", None}:
                    continue
                try:
                    end = date.fromisoformat(str(item["end"]))
                    filed = date.fromisoformat(str(item["filed"]))
                    if spec.kind == "duration":
                        start = date.fromisoformat(str(item["start"]))
                        if not 300 <= (end - start).days <= 430:
                            continue
                    else:
                        start = None
                    value = float(item["val"])
                except (KeyError, TypeError, ValueError, OverflowError):
                    continue
                if not pd.notna(value):
                    continue
                rows.append(
                    {
                        "ticker": ticker,
                        "cik": cik,
                        "concept": canonical,
                        "value": value,
                        "unit": spec.unit,
                        "fiscal_start": start,
                        "fiscal_end": end,
                        "effective_at": end,
                        "filed": filed,
                        "available_at": filed,
                        "form": item.get("form"),
                        "fiscal_year": item.get("fy"),
                        "accession": str(item.get("accn", "")),
                        "source_tag": tag,
                        "tag_priority": priority,
                    }
                )
    if not rows:
        return pd.DataFrame(
            columns=[
                "ticker", "cik", "concept", "value", "unit", "fiscal_start",
                "fiscal_end", "effective_at", "filed", "available_at", "form",
                "fiscal_year", "accession", "source_tag",
            ]
        )
    frame = pd.DataFrame(rows)
    for column in ["fiscal_start", "fiscal_end", "effective_at", "filed"]:
        frame[column] = pd.to_datetime(frame[column])
    frame["effective_at"] = frame["effective_at"].dt.tz_localize("UTC")
    frame["available_at"] = pd.to_datetime(frame["available_at"]).dt.tz_localize("UTC") + pd.Timedelta(days=1)
    frame = frame.sort_values(
        ["concept", "fiscal_end", "available_at", "tag_priority"],
        ascending=[True, True, True, True],
    ).drop_duplicates(["ticker", "concept", "fiscal_end", "available_at"], keep="first")
    return frame.drop(columns="tag_priority").reset_index(drop=True)


def acquire_sec(config_path: str | Path) -> dict[str, Path | int]:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    root = config_path.parent.parent
    output = root / config["data"]["output_dir"]
    data_dir = output / "data"
    chunk_dir = data_dir / "sec_facts_only_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    universe_path = data_dir / "frozen_proxy_universe.csv"
    if not universe_path.is_file():
        raise FileNotFoundError("run public broad proxy price acquisition first")
    universe = pd.read_csv(universe_path)
    client = SecCompanyFactsClient(
        root / "data/cache/sec",
        user_agent=os.getenv("SEC_USER_AGENT"),
        min_interval_seconds=0.15,
    )
    audits: list[pd.DataFrame] = []
    concept_paths: list[Path] = []
    metadata_paths: list[Path] = []
    chunk_size = 50
    for chunk_number, start in enumerate(range(0, len(universe), chunk_size)):
        block = universe.iloc[start : start + chunk_size]
        concept_path = chunk_dir / f"concepts_{chunk_number:04d}.parquet"
        metadata_path = chunk_dir / f"metadata_{chunk_number:04d}.parquet"
        audit_path = chunk_dir / f"audit_{chunk_number:04d}.csv"
        if concept_path.is_file() and metadata_path.is_file() and audit_path.is_file():
            concept_paths.append(concept_path)
            metadata_paths.append(metadata_path)
            audits.append(pd.read_csv(audit_path))
            continue
        concepts: list[pd.DataFrame] = []
        metadata: list[dict[str, Any]] = []
        audit_rows: list[dict[str, object]] = []
        def fetch_one(row: object) -> tuple[pd.DataFrame, dict[str, object]]:
            ticker = str(row.ticker)
            cik = int(row.cik)
            url = client.companyfacts_url.format(cik=cik)
            payload, _ = client._get_json(
                url,
                client.cache_dir / "companyfacts" / f"CIK{cik:010d}.json.gz",
                refresh=False,
            )
            facts = _normalize_companyfacts_fast(ticker, cik, payload)
            return facts, {
                "ticker": ticker,
                "cik": cik,
                "entity_name": payload.get("entityName") or row.title,
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(fetch_one, row): row for row in block.itertuples(index=False)}
            for future in concurrent.futures.as_completed(futures):
                row = futures[future]
                try:
                    facts, meta = future.result()
                    concepts.append(facts)
                    metadata.append(meta)
                    audit_rows.append(
                        {"ticker": row.ticker, "cik": int(row.cik), "status": "ok", "rows": len(facts)}
                    )
                except Exception as error:
                    audit_rows.append(
                        {
                            "ticker": row.ticker,
                            "cik": int(row.cik),
                            "status": f"error: {error}"[:400],
                            "rows": 0,
                        }
                    )
        concept_frame = (
            pd.concat(concepts, ignore_index=True)
            if concepts
            else pd.DataFrame(columns=["ticker", "cik", "concept", "value", "available_at"])
        )
        concept_frame.to_parquet(concept_path, index=False)
        pd.DataFrame(metadata).to_parquet(metadata_path, index=False)
        audit = pd.DataFrame(audit_rows)
        audit.to_csv(audit_path, index=False)
        concept_paths.append(concept_path)
        metadata_paths.append(metadata_path)
        audits.append(audit)
        print(
            f"sec_chunk={chunk_number + 1}/{(len(universe) + chunk_size - 1) // chunk_size} "
            f"ok={int(audit['status'].eq('ok').sum())} errors={int((~audit['status'].eq('ok')).sum())} "
            f"concept_rows={len(concept_frame)}",
            flush=True,
        )

    all_concepts = pd.concat(
        [pd.read_parquet(path) for path in concept_paths], ignore_index=True
    )
    all_metadata = pd.concat(
        [pd.read_parquet(path) for path in metadata_paths], ignore_index=True
    ).drop_duplicates("ticker", keep="last")
    all_audit = pd.concat(audits, ignore_index=True).drop_duplicates("ticker", keep="last")
    concepts_path = data_dir / "sec_annual_concepts.parquet"
    metadata_path = data_dir / "company_metadata.parquet"
    audit_path = output / "sec_download_audit.csv"
    all_concepts.to_parquet(concepts_path, index=False)
    all_metadata.to_parquet(metadata_path, index=False)
    all_audit.to_csv(audit_path, index=False)
    manifest = {
        "strategy_version": config["research"]["strategy_version"],
        "universe_sha256": sha256_file(universe_path),
        "sec_user_agent_configured": bool(os.getenv("SEC_USER_AGENT")),
        "requests_used_cached_raw_when_present": True,
        "mode": "companyfacts_only",
        "tickers": len(universe),
        "ok": int(all_audit["status"].eq("ok").sum()),
        "errors": int((~all_audit["status"].eq("ok")).sum()),
        "tickers_with_any_normalized_us_gaap_fact": int(
            all_audit.loc[all_audit["rows"].gt(0), "ticker"].nunique()
        ),
        "concept_rows": len(all_concepts),
        "concepts_sha256": sha256_file(concepts_path),
        "metadata_sha256": sha256_file(metadata_path),
        "availability_rule": "filed date plus one day for every observation",
        "taxonomy": "us-gaap",
    }
    (output / "sec_data_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return {
        "tickers": len(universe),
        "concept_rows": len(all_concepts),
        "concepts": concepts_path,
        "metadata": metadata_path,
    }


__all__ = ["acquire_sec"]
