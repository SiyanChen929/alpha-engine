"""Confirmed MarketData bulk daily snapshots for v3 universe enrichment.

Snapshots provide OHLCV only.  They never substitute for CRSP membership,
permanent identifiers, or delisting returns.
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import pandas as pd
import requests

from alpha_factor_engine.contracts import DatasetRecord, frame_bounds, sha256_bytes, utc_now


class MarketDataBulkSnapshotClient:
    endpoint = "https://api.marketdata.app/v1/stocks/bulkcandles/D/"

    def __init__(
        self,
        token: str | None = None,
        *,
        cache_dir: str | Path = "data/cache/marketdata/bulk_snapshots",
        timeout: int = 60,
    ) -> None:
        self.token = token or os.getenv("MARKETDATA_TOKEN") or os.getenv("MARKETDATA_API_KEY")
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        self.session = requests.Session()
        if self.token:
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})
        self.session.headers.update({"Accept": "application/json", "User-Agent": "AlphaFactorEngine/0.1"})

    def fetch_snapshot(
        self,
        date: str,
        *,
        symbols: list[str] | None = None,
        refresh: bool = False,
    ) -> tuple[pd.DataFrame, DatasetRecord]:
        if not self.token:
            raise RuntimeError("MARKETDATA_TOKEN is not set")
        session_date = pd.Timestamp(date).date()
        today = pd.Timestamp.now(tz="America/New_York").date()
        if session_date >= today:
            raise ValueError("confirmed bulk snapshots require a date before today in New York")
        if symbols is not None and not symbols:
            raise ValueError("symbols must be non-empty when supplied")

        symbol_key = "all" if symbols is None else sha256_bytes(",".join(sorted(symbols)).encode())[:12]
        path = self.cache_dir / "confirmed" / f"{session_date}_{symbol_key}.json.gz"
        params: dict[str, str] = {
            "date": session_date.isoformat(),
            "adjustsplits": "false",
        }
        if symbols is None:
            params["snapshot"] = "true"
        else:
            params["symbols"] = ",".join(symbols)

        if path.is_file() and not refresh:
            raw = gzip.decompress(path.read_bytes())
        else:
            response = self.session.get(self.endpoint, params=params, timeout=self.timeout)
            response.raise_for_status()
            raw = response.content
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(gzip.compress(raw, compresslevel=6))
        payload = json.loads(raw)
        if payload.get("s") != "ok":
            raise RuntimeError(
                f"MarketData bulk snapshot returned {payload.get('s')}: {payload.get('errmsg', '')}"
            )
        frame = pd.DataFrame(
            {
                "ticker": payload["symbol"],
                "open": payload["o"],
                "high": payload["h"],
                "low": payload["l"],
                "close": payload["c"],
                "volume": payload["v"],
                "timestamp": pd.to_datetime(payload["t"], unit="s", utc=True),
            }
        )
        frame["session_date"] = pd.Timestamp(session_date)
        local_midnight = pd.Timestamp(session_date, tz="America/New_York")
        effective_at = (local_midnight + pd.Timedelta(hours=16)).tz_convert("UTC")
        frame["effective_at"] = effective_at
        frame["available_at"] = effective_at + pd.Timedelta(minutes=30)
        frame["finality"] = "confirmed"
        frame["membership_role"] = "none_ohlcv_only"
        eb = frame_bounds(frame, "effective_at")
        ab = frame_bounds(frame, "available_at")
        record = DatasetRecord(
            provider="MarketData.app",
            dataset=f"bulk_daily_snapshot:{session_date}:{symbol_key}",
            retrieved_at=utc_now(),
            source_url=self.endpoint,
            sha256=sha256_bytes(raw),
            rows=len(frame),
            min_effective_at=eb[0],
            max_effective_at=eb[1],
            min_available_at=ab[0],
            max_available_at=ab[1],
            finality="confirmed",
            revision_policy="immutable date-keyed confirmed cache",
            notes={
                "adjust_splits": False,
                "membership_role": "none; CRSP historical security master remains mandatory",
                "delisting_role": "none; CRSP delisting returns remain mandatory",
            },
        )
        return frame.sort_values("ticker").reset_index(drop=True), record


__all__ = ["MarketDataBulkSnapshotClient"]
