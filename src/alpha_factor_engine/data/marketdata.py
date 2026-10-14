"""MarketData.app stock candle adapter with confirmed/provisional separation."""

from __future__ import annotations

import gzip
import json
import os
import random
import re
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

from alpha_factor_engine.contracts import DatasetRecord, frame_bounds, sha256_bytes, utc_now


_NEW_YORK = "America/New_York"
_DAILY_RESOLUTIONS = {"D", "DAILY", "1D"}


def _resolution_minutes(resolution: str) -> tuple[str, int | None]:
    """Classify the resolutions whose completion time we can model safely."""

    normalized = str(resolution).strip().upper()
    if normalized in _DAILY_RESOLUTIONS:
        return "daily", None
    if normalized == "MINUTELY":
        return "intraday", 1
    if normalized in {"H", "HOURLY"}:
        return "intraday", 60
    minute_match = re.fullmatch(r"([1-9]\d*)", normalized)
    if minute_match:
        return "intraday", int(minute_match.group(1))
    hour_match = re.fullmatch(r"([1-9]\d*)H", normalized)
    if hour_match:
        return "intraday", int(hour_match.group(1)) * 60
    raise ValueError(
        f"unsupported candle resolution {resolution!r}; use D/1D/daily or a minute/hour intraday resolution"
    )


def _new_york_date(value: str) -> object:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(_NEW_YORK)
    return timestamp.date()


class MarketDataClient:
    base_url = "https://api.marketdata.app/v1/stocks/candles"
    option_chain_url = "https://api.marketdata.app/v1/options/chain"

    def __init__(
        self,
        token: str | None = None,
        cache_dir: str | Path = "data/cache/marketdata",
        timeout: int = 45,
    ) -> None:
        self.token = token or os.getenv("MARKETDATA_TOKEN") or os.getenv("MARKETDATA_API_KEY")
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        self.session = requests.Session()
        if self.token:
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})
        self.session.headers.update({"Accept": "application/json", "User-Agent": "AlphaFactorEngine/0.1"})

    def fetch_candles(
        self,
        ticker: str,
        start: str,
        end: str,
        *,
        resolution: str = "D",
        finality: str = "confirmed",
        refresh: bool = False,
        retries: int = 5,
    ) -> tuple[pd.DataFrame, DatasetRecord]:
        if not self.token:
            raise RuntimeError("MARKETDATA_TOKEN is not set")
        if finality not in {"confirmed", "provisional"}:
            raise ValueError("finality must be confirmed or provisional")
        resolution_kind, intraday_minutes = _resolution_minutes(resolution)
        if finality == "confirmed":
            requested_end = _new_york_date(end)
            current_new_york_date = pd.Timestamp.now(tz=_NEW_YORK).date()
            if requested_end >= current_new_york_date:
                raise ValueError(
                    "confirmed candles require an end date before the current New York session date"
                )
        safe = ticker.replace("/", "_").replace(".", "_")
        namespace = "confirmed" if finality == "confirmed" else "provisional"
        path = self.cache_dir / namespace / f"{safe}_{resolution}_{start}_{end}.json.gz"
        url = f"{self.base_url}/{resolution}/{quote(ticker, safe='.-')}/"
        params = {
            "from": start,
            "to": end,
            "extended": "false",
            "adjustsplits": "true",
            "adjustdividends": "false",
        }
        if path.exists() and not refresh:
            raw = gzip.decompress(path.read_bytes())
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
                raise RuntimeError(f"MarketData request failed for {ticker}: {last_error}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(gzip.compress(raw, compresslevel=6))
        payload = json.loads(raw)
        if payload.get("s") != "ok":
            raise RuntimeError(f"MarketData returned {payload.get('s')}: {payload.get('errmsg', '')}")
        timestamp = pd.to_datetime(payload["t"], unit="s", utc=True)
        frame = pd.DataFrame(
            {
                "ticker": ticker,
                "timestamp": timestamp,
                "open": payload["o"],
                "high": payload["h"],
                "low": payload["l"],
                "close": payload["c"],
                "volume": payload["v"],
            }
        )
        local_timestamp = frame["timestamp"].dt.tz_convert(_NEW_YORK)
        frame["date"] = local_timestamp.dt.tz_localize(None).dt.normalize()
        if resolution_kind == "daily":
            # MarketData documents the daily timestamp as midnight New York
            # time: it is a session-date label, not the close's event time.
            session_midnight = local_timestamp.dt.normalize()
            frame["effective_at"] = (session_midnight + pd.Timedelta(hours=16)).dt.tz_convert("UTC")
            availability_latency = pd.Timedelta(minutes=30)
        else:
            # The intraday timestamp is the candle start. The candle cannot be
            # used until its full interval has elapsed, plus a delivery buffer.
            frame["effective_at"] = frame["timestamp"] + pd.Timedelta(minutes=intraday_minutes)
            availability_latency = pd.Timedelta(minutes=1)
        frame["available_at"] = frame["effective_at"] + availability_latency
        frame["finality"] = finality
        eb = frame_bounds(frame, "effective_at")
        ab = frame_bounds(frame, "available_at")
        record = DatasetRecord(
            provider="MarketData.app",
            dataset=f"stock_candles:{ticker}:{resolution}",
            retrieved_at=utc_now(),
            source_url=url,
            sha256=sha256_bytes(raw),
            rows=len(frame),
            min_effective_at=eb[0],
            max_effective_at=eb[1],
            min_available_at=ab[0],
            max_available_at=ab[1],
            finality=finality,
            revision_policy="confirmed and provisional namespaces are physically separate",
            notes={
                "adjust_splits": True,
                "adjust_dividends": False,
                "timestamp_semantics": (
                    "session-date label; effective_at is 16:00 America/New_York"
                    if resolution_kind == "daily"
                    else "candle start; effective_at is candle end"
                ),
                "availability_latency_seconds": int(availability_latency.total_seconds()),
            },
        )
        return frame, record

    def fetch_option_chain(
        self,
        underlying: str,
        *,
        quote_date: str | None = None,
        expiration: str | None = None,
        side: str | None = None,
        refresh: bool = False,
        **filters: str | int | float,
    ) -> tuple[pd.DataFrame, DatasetRecord]:
        """Fetch a current or historical EOD chain.

        This source is suitable for chain research and screening. It is not a
        substitute for historical OPRA NBBO/trades in an executable backtest.
        """

        if not self.token and underlying.upper() != "AAPL":
            raise RuntimeError("MARKETDATA_TOKEN is not set")
        if side is not None and side not in {"call", "put"}:
            raise ValueError("side must be call or put")
        today = pd.Timestamp.now(tz="America/New_York").date()
        requested_date = pd.Timestamp(quote_date).date() if quote_date else today
        finality = "confirmed" if quote_date and requested_date < today else "provisional"
        namespace = "confirmed" if finality == "confirmed" else "provisional"
        date_key = quote_date or "current"
        filter_hash = sha256_bytes(
            json.dumps(
                {"expiration": expiration, "side": side, **filters}, sort_keys=True, default=str
            ).encode()
        )[:10]
        safe = underlying.replace("/", "_").replace(".", "_")
        path = self.cache_dir / "options" / namespace / f"{safe}_{date_key}_{filter_hash}.json.gz"
        url = f"{self.option_chain_url}/{quote(underlying, safe='.-')}/"
        params: dict[str, str | int | float] = dict(filters)
        if quote_date:
            params["date"] = quote_date
        if expiration:
            params["expiration"] = expiration
        if side:
            params["side"] = side
        if path.exists() and not refresh:
            raw = gzip.decompress(path.read_bytes())
        else:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            raw = response.content
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(gzip.compress(raw, compresslevel=6))
        payload = json.loads(raw)
        if payload.get("s") != "ok":
            raise RuntimeError(f"MarketData option chain returned {payload.get('s')}: {payload.get('errmsg', '')}")
        count = len(payload.get("optionSymbol") or [])
        columns = {
            key: value
            for key, value in payload.items()
            if isinstance(value, list) and len(value) == count
        }
        frame = pd.DataFrame(columns)
        if frame.empty:
            raise RuntimeError(f"MarketData returned an empty option chain for {underlying}")
        frame = frame.rename(
            columns={
                "optionSymbol": "option_id",
                "underlying": "underlier",
                "side": "call_put",
                "askSize": "ask_size",
                "bidSize": "bid_size",
                "firstTraded": "first_traded",
                "openInterest": "open_interest",
                "underlyingPrice": "underlying_price",
            }
        )
        for column in ["expiration", "first_traded", "updated"]:
            if column in frame:
                frame[column] = pd.to_datetime(frame[column], unit="s", utc=True, errors="coerce")
        retrieved_at = pd.Timestamp.now(tz="UTC")
        if "updated" in frame and frame["updated"].notna().any():
            frame["effective_at"] = frame["updated"]
        elif quote_date:
            frame["effective_at"] = pd.Timestamp(quote_date, tz="UTC") + pd.Timedelta(hours=23, minutes=59)
        else:
            frame["effective_at"] = retrieved_at
        if finality == "confirmed":
            frame["available_at"] = pd.Timestamp(quote_date, tz="UTC") + pd.Timedelta(days=1, hours=12)
        else:
            frame["available_at"] = retrieved_at
        frame["quote_date"] = pd.Timestamp(requested_date)
        frame["finality"] = finality
        eb = frame_bounds(frame, "effective_at")
        ab = frame_bounds(frame, "available_at")
        record = DatasetRecord(
            provider="MarketData.app",
            dataset=f"option_chain:{underlying}:{date_key}",
            retrieved_at=utc_now(),
            source_url=url,
            sha256=sha256_bytes(raw),
            rows=len(frame),
            min_effective_at=eb[0],
            max_effective_at=eb[1],
            min_available_at=ab[0],
            max_available_at=ab[1],
            finality=finality,
            revision_policy="confirmed historical and current/provisional caches are separate",
            notes={
                "execution_grade": False,
                "warning": "historical chains are EOD snapshots, not OPRA NBBO/trade replay",
                "corporate_actions": "vendor documents historical chains as as-traded and unadjusted",
            },
        )
        return frame, record
