"""Small, dependency-light Yahoo chart API adapter for public research.

Yahoo is a convenience source, not a production security master. The adapter
therefore preserves ticker, retrieval time, raw payload hash, and split events,
and never claims delisting-complete or survivorship-free coverage.
"""

from __future__ import annotations

import concurrent.futures
import gzip
import json
import time
from datetime import UTC, datetime, time as wall_time
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from alpha_factor_engine.contracts import DatasetRecord, frame_bounds, sha256_bytes, utc_now


class YahooFinanceClient:
    base_url = "https://query1.finance.yahoo.com/v8/finance/chart"

    def __init__(self, cache_dir: str | Path = "data/cache/yahoo", timeout: int = 45) -> None:
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0 AlphaFactorEngine/0.1"})

    @staticmethod
    def _epoch(value: str | pd.Timestamp) -> int:
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("UTC")
        return int(stamp.timestamp())

    @staticmethod
    def _safe_symbol(ticker: str) -> str:
        return ticker.replace("/", "_").replace(".", "_").replace("^", "index_")

    def fetch_history(
        self,
        ticker: str,
        start: str,
        end: str,
        *,
        refresh: bool = False,
    ) -> tuple[pd.DataFrame, DatasetRecord]:
        cache_path = self.cache_dir / f"{self._safe_symbol(ticker)}_{start}_{end}.json.gz"
        url = f"{self.base_url}/{quote(ticker, safe='.-^')}"
        params = {
            "period1": self._epoch(start),
            "period2": self._epoch(pd.Timestamp(end) + pd.Timedelta(days=1)),
            "interval": "1d",
            "events": "div,splits",
            "includeAdjustedClose": "true",
        }
        if cache_path.exists() and not refresh:
            raw = gzip.decompress(cache_path.read_bytes())
        else:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            raw = response.content
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(gzip.compress(raw, compresslevel=6))
            time.sleep(0.04)
        payload = json.loads(raw)
        error = payload.get("chart", {}).get("error")
        result = (payload.get("chart", {}).get("result") or [None])[0]
        if error or result is None:
            raise RuntimeError(f"Yahoo returned no usable data for {ticker}: {error}")
        frame = self._parse_result(ticker, result)
        effective_bounds = frame_bounds(frame, "effective_at")
        available_bounds = frame_bounds(frame, "available_at")
        finality = "confirmed" if self._is_confirmed_end(pd.Timestamp(end)) else "provisional"
        record = DatasetRecord(
            provider="Yahoo Finance chart API",
            dataset=f"daily_prices:{ticker}",
            retrieved_at=utc_now(),
            source_url=url,
            sha256=sha256_bytes(raw),
            rows=len(frame),
            min_effective_at=effective_bounds[0],
            max_effective_at=effective_bounds[1],
            min_available_at=available_bounds[0],
            max_available_at=available_bounds[1],
            finality=finality,
            revision_policy="cache_raw_payload; vendor may back-adjust history",
            notes={
                "adjustment": "adj_close supplied by vendor; adjusted_open derived by adj_close/close",
                "coverage_warning": "ticker history is not a survivorship-free security master",
            },
        )
        return frame, record

    @staticmethod
    def _is_confirmed_end(end: pd.Timestamp) -> bool:
        eastern = ZoneInfo("America/New_York")
        now = datetime.now(eastern)
        end_date = end.date()
        if end_date < now.date():
            return True
        return end_date == now.date() and now.time() >= wall_time(18, 0)

    @staticmethod
    def _parse_result(ticker: str, result: dict) -> pd.DataFrame:
        returned_symbol = (result.get("meta") or {}).get("symbol")
        if returned_symbol != ticker:
            raise RuntimeError(
                "Yahoo response symbol differs from the requested security: "
                f"requested={ticker!r}, returned={returned_symbol!r}"
            )
        timestamps = result.get("timestamp") or []
        quote_rows = ((result.get("indicators") or {}).get("quote") or [{}])[0]
        adjusted = ((result.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose")
        size = len(timestamps)

        def values(key: str) -> list[float | None]:
            found = quote_rows.get(key)
            return found if found is not None else [None] * size

        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(timestamps, unit="s", utc=True),
                "open": values("open"),
                "high": values("high"),
                "low": values("low"),
                "close": values("close"),
                "volume": values("volume"),
                "adj_close": adjusted if adjusted is not None else values("close"),
            }
        )
        for column in ["open", "high", "low", "close", "volume", "adj_close"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=["close", "adj_close"]).copy()
        exchange_tz = result.get("meta", {}).get("exchangeTimezoneName", "America/New_York")
        local_date = frame["timestamp"].dt.tz_convert(exchange_tz).dt.date
        frame.insert(0, "date", pd.to_datetime(local_date))
        frame.insert(0, "ticker", ticker)
        adjustment = frame["adj_close"] / frame["close"].replace(0, np.nan)
        frame["adj_open"] = frame["open"] * adjustment
        frame["dollar_volume"] = frame["close"] * frame["volume"]
        close_local = pd.to_datetime(frame["date"].astype(str) + " 16:00").dt.tz_localize(
            exchange_tz, ambiguous="NaT", nonexistent="shift_forward"
        )
        frame["effective_at"] = close_local.dt.tz_convert("UTC")
        frame["available_at"] = frame["effective_at"] + pd.Timedelta(minutes=30)
        frame["split_ratio"] = 1.0
        frame["dividend"] = 0.0
        events = result.get("events") or {}
        for event in (events.get("splits") or {}).values():
            date = pd.to_datetime(event["date"], unit="s", utc=True).tz_convert(exchange_tz).date()
            numerator = float(event.get("numerator", 1.0))
            denominator = float(event.get("denominator", 1.0))
            frame.loc[frame["date"].dt.date.eq(date), "split_ratio"] = numerator / denominator
        for event in (events.get("dividends") or {}).values():
            date = pd.to_datetime(event["date"], unit="s", utc=True).tz_convert(exchange_tz).date()
            frame.loc[frame["date"].dt.date.eq(date), "dividend"] = float(event.get("amount", 0.0))
        return frame.sort_values("date").reset_index(drop=True)

    def fetch_many(
        self,
        tickers: list[str],
        start: str,
        end: str,
        *,
        workers: int = 6,
        refresh: bool = False,
    ) -> tuple[pd.DataFrame, list[DatasetRecord], pd.DataFrame]:
        frames: list[pd.DataFrame] = []
        records: list[DatasetRecord] = []
        audit: list[dict[str, object]] = []

        def run(ticker: str) -> tuple[str, pd.DataFrame, DatasetRecord]:
            frame, record = self.fetch_history(ticker, start, end, refresh=refresh)
            return ticker, frame, record

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run, ticker): ticker for ticker in tickers}
            for future in concurrent.futures.as_completed(futures):
                ticker = futures[future]
                try:
                    _, frame, record = future.result()
                    frames.append(frame)
                    records.append(record)
                    audit.append({"ticker": ticker, "status": "ok", "rows": len(frame)})
                except Exception as exc:
                    audit.append({"ticker": ticker, "status": f"error: {exc}", "rows": 0})
        if not frames:
            raise RuntimeError("No Yahoo histories downloaded")
        combined = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"])
        return combined, records, pd.DataFrame(audit).sort_values("ticker")
