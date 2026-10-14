"""Resumable Yahoo acquisition for the frozen public broad proxy universe."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from alpha_factor_engine.contracts import sha256_file
from alpha_factor_engine.data.yahoo import YahooFinanceClient


def load_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("proxy configuration must contain a mapping")
    return value


def _load_sec_candidates(path: Path, count: int) -> pd.DataFrame:
    payload = json.loads(gzip.decompress(path.read_bytes()))
    rows = list(payload.values())[:count]
    frame = pd.DataFrame(rows).rename(columns={"cik_str": "cik"})
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.replace(".", "-", regex=False)
    frame["candidate_order"] = range(1, len(frame) + 1)
    frame = frame.drop_duplicates("ticker", keep="first")
    return frame[["candidate_order", "ticker", "cik", "title"]]


def _yahoo_meta(cache_path: Path) -> dict[str, object]:
    payload = json.loads(gzip.decompress(cache_path.read_bytes()))
    result = (payload.get("chart", {}).get("result") or [{}])[0]
    meta = result.get("meta") or {}
    return {
        "ticker": str(meta.get("symbol", "")).upper(),
        "instrument_type": meta.get("instrumentType"),
        "exchange_code": meta.get("exchangeName"),
        "exchange_name": meta.get("fullExchangeName"),
        "currency": meta.get("currency"),
        "regular_market_price": meta.get("regularMarketPrice"),
    }


def acquire_prices(config_path: str | Path) -> dict[str, Path | int]:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    root = config_path.parent.parent
    data_config = config["data"]
    output = root / data_config["output_dir"]
    data_dir = output / "data"
    chunk_dir = data_dir / "price_chunks"
    output.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    candidate_path = data_dir / "frozen_sec_candidates.csv"
    if candidate_path.is_file():
        candidates = pd.read_csv(candidate_path)
    else:
        candidates = _load_sec_candidates(
            root / data_config["sec_ticker_cache"], int(data_config["candidate_count"])
        )
        candidates.to_csv(candidate_path, index=False)
    if len(candidates) != int(data_config["candidate_count"]):
        raise RuntimeError("frozen candidate count differs from configuration")

    tickers = candidates["ticker"].astype(str).tolist()
    benchmark = str(data_config["benchmark"])
    download_tickers = tickers + ([benchmark] if benchmark not in tickers else [])
    chunk_size = int(data_config["yahoo_chunk_size"])
    client = YahooFinanceClient(root / data_config["yahoo_cache_dir"])
    audits: list[pd.DataFrame] = []
    chunk_paths: list[Path] = []
    for chunk_number, start_index in enumerate(range(0, len(download_tickers), chunk_size)):
        names = download_tickers[start_index : start_index + chunk_size]
        chunk_path = chunk_dir / f"prices_{chunk_number:04d}.parquet"
        audit_path = chunk_dir / f"audit_{chunk_number:04d}.csv"
        if chunk_path.is_file() and audit_path.is_file():
            chunk_paths.append(chunk_path)
            audits.append(pd.read_csv(audit_path))
            continue
        prices, _, audit = client.fetch_many(
            names,
            config["research"]["start"],
            config["research"]["end"],
            workers=int(data_config["yahoo_workers"]),
            refresh=False,
        )
        prices.to_parquet(chunk_path, index=False)
        audit.to_csv(audit_path, index=False)
        chunk_paths.append(chunk_path)
        audits.append(audit)
        print(
            f"chunk={chunk_number + 1}/{(len(download_tickers) + chunk_size - 1) // chunk_size} "
            f"ok={int(audit['status'].eq('ok').sum())} errors={int((~audit['status'].eq('ok')).sum())}",
            flush=True,
        )

    audit = pd.concat(audits, ignore_index=True).drop_duplicates("ticker", keep="last")
    audit.to_csv(output / "yahoo_download_audit.csv", index=False)
    ok = set(audit.loc[audit["status"].eq("ok"), "ticker"].astype(str))
    meta_rows: list[dict[str, object]] = []
    for ticker in sorted(ok):
        safe = client._safe_symbol(ticker)
        cache_path = (
            root
            / data_config["yahoo_cache_dir"]
            / f"{safe}_{config['research']['start']}_{config['research']['end']}.json.gz"
        )
        if cache_path.is_file():
            meta_rows.append(_yahoo_meta(cache_path))
    metadata = pd.DataFrame(meta_rows).drop_duplicates("ticker", keep="last")
    metadata.to_csv(data_dir / "yahoo_security_metadata.csv", index=False)

    eligible_meta = metadata.loc[
        metadata["instrument_type"].eq(data_config["instrument_type"])
        & metadata["currency"].eq(data_config["currency"])
        & metadata["exchange_code"].isin(data_config["allowed_exchange_codes"])
    ].copy()
    eligible_names = set(eligible_meta["ticker"])
    liquidity_rows: list[dict[str, object]] = []
    for chunk_path in chunk_paths:
        chunk = pd.read_parquet(chunk_path, columns=["ticker", "date", "close", "volume"])
        chunk = chunk.loc[chunk["ticker"].isin(eligible_names)]
        chunk["dollar_volume"] = chunk["close"] * chunk["volume"]
        for ticker, group in chunk.groupby("ticker", sort=False):
            recent = group.sort_values("date").tail(63)
            liquidity_rows.append(
                {
                    "ticker": ticker,
                    "trailing_63d_median_dollar_volume": float(recent["dollar_volume"].median()),
                    "history_rows": int(len(group)),
                    "first_price_date": group["date"].min(),
                    "last_price_date": group["date"].max(),
                }
            )
    liquidity = pd.DataFrame(liquidity_rows)
    ranked = (
        candidates.merge(eligible_meta, on="ticker", how="inner", validate="one_to_one")
        .merge(liquidity, on="ticker", how="inner", validate="one_to_one")
        .sort_values(
            ["trailing_63d_median_dollar_volume", "ticker"],
            ascending=[False, True],
            kind="mergesort",
        )
    )
    selected = ranked.head(int(data_config["target_universe_count"])).copy()
    selected["proxy_universe_rank"] = range(1, len(selected) + 1)
    selected["selection_as_of"] = config["research"]["end"]
    selected["evidence_status"] = "current_survivor_proxy"
    if len(selected) < int(data_config["minimum_downloaded_universe"]):
        raise RuntimeError(
            f"only {len(selected)} eligible Yahoo equities; minimum is "
            f"{data_config['minimum_downloaded_universe']}"
        )
    selected_path = data_dir / "frozen_proxy_universe.csv"
    selected.to_csv(selected_path, index=False)

    selected_names = set(selected["ticker"]) | {benchmark}
    selected_chunks = []
    for chunk_path in chunk_paths:
        chunk = pd.read_parquet(chunk_path)
        keep = chunk.loc[chunk["ticker"].isin(selected_names)]
        if len(keep):
            selected_chunks.append(keep)
    daily = pd.concat(selected_chunks, ignore_index=True).sort_values(["ticker", "date"])
    daily_path = data_dir / "daily_prices.parquet"
    daily.to_parquet(daily_path, index=False)

    manifest = {
        "strategy_version": config["research"]["strategy_version"],
        "config_sha256": sha256_file(config_path),
        "sec_ticker_cache_sha256": sha256_file(root / data_config["sec_ticker_cache"]),
        "candidate_count": len(candidates),
        "download_ok": int(audit["status"].eq("ok").sum()),
        "download_errors": int((~audit["status"].eq("ok")).sum()),
        "eligible_metadata_count": len(eligible_meta),
        "selected_count": len(selected),
        "daily_rows": len(daily),
        "daily_prices_sha256": sha256_file(daily_path),
        "selected_universe_sha256": sha256_file(selected_path),
        "survivorship_free": False,
        "selection_uses_end_of_sample_liquidity": True,
        "claim_limit": "exploratory_current_survivor_proxy",
    }
    (output / "price_data_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return {
        "selected_count": len(selected),
        "daily_rows": len(daily),
        "daily_prices": daily_path,
        "universe": selected_path,
    }


__all__ = ["acquire_prices", "load_config"]

