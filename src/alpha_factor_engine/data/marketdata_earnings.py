"""MarketData earnings snapshots with explicit non-PIT lineage."""

from __future__ import annotations

import gzip
import json
import os
import random
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

from alpha_factor_engine.contracts import DatasetRecord, frame_bounds, sha256_bytes


REQUIRED_ARRAYS = (
    "symbol",
    "fiscalYear",
    "fiscalQuarter",
    "date",
    "reportDate",
    "reportTime",
    "currency",
    "reportedEPS",
    "estimatedEPS",
    "surpriseEPS",
    "surpriseEPSpct",
    "updated",
)


class MarketDataEarningsError(RuntimeError):
    """Raised when an earnings snapshot violates its source contract."""


class MarketDataEarningsClient:
    endpoint = "https://api.marketdata.app/v1/stocks/earnings"

    def __init__(
        self,
        token: str | None = None,
        cache_dir: str | Path = "data/cache/marketdata/earnings",
        timeout: int = 45,
    ) -> None:
        self.token = token or os.getenv("MARKETDATA_TOKEN") or os.getenv("MARKETDATA_API_KEY")
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        self.session = requests.Session()
        if self.token:
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})
        self.session.headers.update({"Accept": "application/json", "User-Agent": "AlphaFactorEngine/0.1"})

    def fetch(
        self,
        ticker: str,
        *,
        start: str,
        end: str,
        refresh: bool = False,
        retries: int = 5,
    ) -> tuple[pd.DataFrame, DatasetRecord]:
        symbol = str(ticker).strip().upper()
        if not symbol:
            raise ValueError("ticker is empty")
        if not self.token and symbol != "AAPL":
            raise RuntimeError("MARKETDATA_TOKEN is not set")
        start_date = pd.Timestamp(start).date()
        end_date = pd.Timestamp(end).date()
        if end_date < start_date:
            raise ValueError("end precedes start")
        safe = symbol.replace("/", "_").replace(".", "_")
        path = self.cache_dir / "current_snapshot" / f"{safe}_{start_date}_{end_date}.json.gz"
        url = f"{self.endpoint}/{quote(symbol, safe='.-')}/"
        params = {"from": str(start_date), "to": str(end_date)}
        if path.exists() and not refresh:
            raw = gzip.decompress(path.read_bytes())
            cache_status = "cache_hit"
        else:
            last_error: Exception | None = None
            for attempt in range(retries):
                try:
                    response = self.session.get(url, params=params, timeout=self.timeout)
                    if response.status_code == 429:
                        time.sleep((2**attempt) + random.random())
                        continue
                    response.raise_for_status()
                    raw = response.content
                    break
                except Exception as exc:
                    last_error = exc
                    time.sleep((attempt + 1) * 0.5)
            else:
                raise MarketDataEarningsError(f"MarketData earnings request failed for {symbol}: {last_error}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(gzip.compress(raw, compresslevel=6))
            cache_status = "downloaded"
        retrieved_at = pd.Timestamp(path.stat().st_mtime_ns, unit="ns", tz="UTC")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise MarketDataEarningsError(f"MarketData earnings returned invalid JSON for {symbol}") from exc
        if payload.get("s") == "no_data":
            frame = pd.DataFrame(columns=[*REQUIRED_ARRAYS, "period_end", "report_date", "updated_at"])
        elif payload.get("s") != "ok":
            raise MarketDataEarningsError(
                f"MarketData earnings returned {payload.get('s')}: {payload.get('errmsg', '')}"
            )
        else:
            missing = [name for name in REQUIRED_ARRAYS if name not in payload]
            if missing:
                raise MarketDataEarningsError(f"MarketData earnings arrays missing for {symbol}: {missing}")
            if any(not isinstance(payload[name], list) for name in REQUIRED_ARRAYS):
                raise MarketDataEarningsError(f"MarketData earnings fields are not arrays for {symbol}")
            lengths = {name: len(payload[name]) for name in REQUIRED_ARRAYS}
            if len(set(lengths.values())) != 1:
                raise MarketDataEarningsError(f"MarketData earnings array lengths differ for {symbol}: {lengths}")
            frame = pd.DataFrame({name: payload[name] for name in REQUIRED_ARRAYS})
            if not frame["symbol"].astype(str).str.upper().eq(symbol).all():
                raise MarketDataEarningsError(f"MarketData earnings symbol mismatch for {symbol}")
            frame["period_end"] = pd.to_datetime(frame["date"], unit="s", utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
            frame["report_date"] = pd.to_datetime(frame["reportDate"], unit="s", utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
            frame["updated_at"] = pd.to_datetime(frame["updated"], unit="s", utc=True, errors="coerce")
            frame["snapshot_retrieved_at"] = pd.Timestamp(retrieved_at)
            frame["historical_estimate_vintage_proven"] = False
            frame["exact_first_public_timestamp_proven"] = False
            frame["evidence_status"] = "CURRENT_VENDOR_SNAPSHOT_NOT_PIT"
            if frame.duplicated(["symbol", "fiscalYear", "fiscalQuarter"]).any():
                raise MarketDataEarningsError(f"duplicate fiscal periods for {symbol}")
        effective_bounds = frame_bounds(frame, "report_date") if not frame.empty else (None, None)
        available_bounds = frame_bounds(frame, "snapshot_retrieved_at") if not frame.empty else (None, None)
        record = DatasetRecord(
            provider="MarketData.app",
            dataset=f"stock_earnings_current_snapshot:{symbol}",
            retrieved_at=retrieved_at,
            source_url=url,
            sha256=sha256_bytes(raw),
            rows=len(frame),
            min_effective_at=effective_bounds[0],
            max_effective_at=effective_bounds[1],
            min_available_at=available_bounds[0],
            max_available_at=available_bounds[1],
            finality="current_vendor_snapshot",
            revision_policy="raw response is immutable after first cache write; provider historical estimate vintages are unavailable",
            notes={
                "cache_status": cache_status,
                "cache_path": path.as_posix(),
                "report_date_semantics": "historical calendar date after report; future dates may be estimates",
                "report_time_semantics": "categorical before market open, after market close, or during market hours",
                "historical_estimate_vintage_proven": False,
                "exact_first_public_timestamp_proven": False,
            },
        )
        return frame, record


__all__ = ["MarketDataEarningsClient", "MarketDataEarningsError"]
