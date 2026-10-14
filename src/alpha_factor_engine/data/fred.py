"""FRED and ALFRED adapters with explicit vintage semantics."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pandas as pd
import requests

from alpha_factor_engine.contracts import DatasetRecord, frame_bounds, sha256_bytes, utc_now


class FredClient:
    graph_url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    api_url = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, cache_dir: str | Path = "data/cache/fred", timeout: int = 45) -> None:
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "AlphaFactorEngine/0.1"})

    def fetch_latest_series(
        self,
        series_id: str,
        start: str,
        end: str,
        *,
        availability_lag_days: int = 1,
        refresh: bool = False,
    ) -> tuple[pd.DataFrame, DatasetRecord]:
        path = self.cache_dir / f"{series_id}_{start}_{end}_latest.csv"
        params = {"id": series_id, "cosd": start, "coed": end}
        if path.exists() and not refresh:
            raw = path.read_bytes()
        else:
            response = self.session.get(self.graph_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            raw = response.content
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        frame = pd.read_csv(io.BytesIO(raw))
        frame.columns = ["observation_date", "value"]
        frame["observation_date"] = pd.to_datetime(frame["observation_date"])
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        frame = frame.dropna(subset=["value"]).copy()
        frame["series_id"] = series_id
        frame["effective_at"] = pd.to_datetime(frame["observation_date"], utc=True)
        frame["available_at"] = frame["effective_at"] + pd.Timedelta(days=availability_lag_days)
        frame["vintage_start"] = pd.NaT
        frame["vintage_end"] = pd.NaT
        eb = frame_bounds(frame, "effective_at")
        ab = frame_bounds(frame, "available_at")
        record = DatasetRecord(
            provider="Federal Reserve Bank of St. Louis FRED",
            dataset=series_id,
            retrieved_at=utc_now(),
            source_url=self.graph_url,
            sha256=sha256_bytes(raw),
            rows=len(frame),
            min_effective_at=eb[0],
            max_effective_at=eb[1],
            min_available_at=ab[0],
            max_available_at=ab[1],
            finality="latest_vintage",
            revision_policy="latest observation only; use fetch_alfred_vintages for revised macro series",
            notes={"availability_lag_days": availability_lag_days},
        )
        return frame, record

    def fetch_alfred_vintages(
        self,
        series_id: str,
        observation_start: str,
        observation_end: str,
        *,
        realtime_start: str = "1776-07-04",
        realtime_end: str = "9999-12-31",
        api_key: str | None = None,
    ) -> tuple[pd.DataFrame, DatasetRecord]:
        key = api_key or os.getenv("FRED_API_KEY")
        if not key:
            raise RuntimeError("FRED_API_KEY is required for ALFRED vintage retrieval")
        base_params = {
            "series_id": series_id,
            "api_key": key,
            "file_type": "json",
            "observation_start": observation_start,
            "observation_end": observation_end,
            "realtime_start": realtime_start,
            "realtime_end": realtime_end,
            # 2 = observations by vintage date, all observations.  Output 4
            # would return initial releases only and silently omit revisions.
            "output_type": 2,
            "limit": 100_000,
        }
        observations: list[dict[str, object]] = []
        raw_pages: list[bytes] = []
        offset = 0
        expected_count: int | None = None
        while expected_count is None or len(observations) < expected_count:
            params = {**base_params, "offset": offset}
            response = self.session.get(self.api_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            raw_pages.append(response.content)
            payload = json.loads(response.content)
            page = payload.get("observations", [])
            if not isinstance(page, list):
                raise RuntimeError("ALFRED observations payload is not a list")
            if "count" not in payload:
                raise RuntimeError("ALFRED response omitted the total observation count")
            try:
                count = int(payload["count"])
            except (TypeError, ValueError) as exc:
                raise RuntimeError("ALFRED returned an invalid total observation count") from exc
            if count < 0:
                raise RuntimeError("ALFRED returned a negative total observation count")
            if expected_count is None:
                expected_count = count
            elif count != expected_count:
                raise RuntimeError(
                    f"ALFRED count changed during pagination: {expected_count} -> {count}"
                )
            if not page and len(observations) < expected_count:
                raise RuntimeError(
                    f"ALFRED pagination stopped at {len(observations)} of {expected_count} observations"
                )
            if len(observations) + len(page) > expected_count:
                raise RuntimeError(
                    f"ALFRED pagination exceeded declared count {expected_count}"
                )
            observations.extend(page)
            offset += len(page)
        if expected_count is None or len(observations) != expected_count:
            raise RuntimeError(
                f"ALFRED completeness check failed: received {len(observations)}, expected {expected_count}"
            )
        if not observations:
            raise RuntimeError(f"ALFRED returned no observations for {series_id}")
        raw = b"\n".join(raw_pages)
        frame = pd.DataFrame(observations)
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        frame["effective_at"] = pd.to_datetime(frame["date"], utc=True)
        frame["available_at"] = pd.to_datetime(frame["realtime_start"], utc=True)
        frame["vintage_start"] = pd.to_datetime(frame["realtime_start"], utc=True)
        frame["vintage_end"] = pd.to_datetime(
            frame["realtime_end"], format="%Y-%m-%d", utc=True, errors="coerce"
        )
        frame["series_id"] = series_id
        frame = frame.dropna(subset=["value"])
        eb = frame_bounds(frame, "effective_at")
        ab = frame_bounds(frame, "available_at")
        record = DatasetRecord(
            provider="ALFRED",
            dataset=series_id,
            retrieved_at=utc_now(),
            source_url=self.api_url,
            sha256=sha256_bytes(raw),
            rows=len(frame),
            min_effective_at=eb[0],
            max_effective_at=eb[1],
            min_available_at=ab[0],
            max_available_at=ab[1],
            finality="vintage_complete",
            revision_policy="retain every realtime_start/realtime_end vintage",
            notes={"api_count": expected_count, "pages": len(raw_pages), "output_type": 2},
        )
        return frame, record
