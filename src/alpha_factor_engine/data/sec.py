"""SEC EDGAR Company Facts adapter with filing-time point-in-time alignment."""

from __future__ import annotations

import gzip
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from alpha_factor_engine.contracts import DatasetRecord, frame_bounds, sha256_bytes, utc_now


@dataclass(frozen=True)
class ConceptSpec:
    tags: tuple[str, ...]
    kind: str  # "instant" or "duration"
    unit: str


ANNUAL_CONCEPTS: dict[str, ConceptSpec] = {
    "assets": ConceptSpec(("Assets",), "instant", "USD"),
    "equity": ConceptSpec(
        (
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ),
        "instant",
        "USD",
    ),
    "liabilities": ConceptSpec(("Liabilities",), "instant", "USD"),
    "debt": ConceptSpec(
        (
            "LongTermDebtAndFinanceLeaseObligations",
            "LongTermDebt",
            "LongTermDebtAndFinanceLeaseObligationsNoncurrent",
            "LongTermDebtNoncurrent",
        ),
        "instant",
        "USD",
    ),
    "revenue": ConceptSpec(
        (
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues",
            "SalesRevenueNet",
        ),
        "duration",
        "USD",
    ),
    "gross_profit": ConceptSpec(("GrossProfit",), "duration", "USD"),
    "net_income": ConceptSpec(("NetIncomeLoss", "ProfitLoss"), "duration", "USD"),
    "operating_cash_flow": ConceptSpec(
        ("NetCashProvidedByUsedInOperatingActivities",), "duration", "USD"
    ),
    "capital_expenditure": ConceptSpec(
        (
            "PaymentsToAcquirePropertyPlantAndEquipment",
            "PaymentsForAdditionsToPropertyPlantAndEquipment",
        ),
        "duration",
        "USD",
    ),
    "diluted_shares": ConceptSpec(
        ("WeightedAverageNumberOfDilutedSharesOutstanding",), "duration", "shares"
    ),
}


class SecCompanyFactsClient:
    ticker_map_url = "https://www.sec.gov/files/company_tickers.json"
    companyfacts_url = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
    submissions_url = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

    def __init__(
        self,
        cache_dir: str | Path = "data/cache/sec",
        user_agent: str | None = None,
        timeout: int = 60,
        min_interval_seconds: float = 0.12,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.user_agent = user_agent or os.getenv("SEC_USER_AGENT")
        if (
            not self.user_agent
            or ".invalid" in self.user_agent.lower()
            or "example.com" in self.user_agent.lower()
            or not re.search(r"[^\s@]+@[^\s@]+\.[^\s@]+", self.user_agent)
        ):
            raise ValueError(
                "SEC_USER_AGENT must identify the application and contain a real contact email"
            )
        self.timeout = timeout
        self.min_interval_seconds = min_interval_seconds
        self._last_request = 0.0
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"}
        )

    def _get_json(self, url: str, path: Path, refresh: bool = False) -> tuple[dict[str, Any], bytes]:
        if path.exists() and not refresh:
            raw = gzip.decompress(path.read_bytes())
            return json.loads(raw), raw
        delay = self.min_interval_seconds - (time.monotonic() - self._last_request)
        if delay > 0:
            time.sleep(delay)
        response: requests.Response | None = None
        for attempt in range(4):
            response = self.session.get(url, timeout=self.timeout)
            self._last_request = time.monotonic()
            if response.status_code not in {403, 404, 429, 500, 502, 503, 504}:
                break
            time.sleep(0.5 * (2**attempt))
        assert response is not None
        response.raise_for_status()
        raw = response.content
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(raw, compresslevel=6))
        return json.loads(raw), raw

    def ticker_map(self, refresh: bool = False) -> tuple[pd.DataFrame, DatasetRecord]:
        payload, raw = self._get_json(
            self.ticker_map_url, self.cache_dir / "company_tickers.json.gz", refresh=refresh
        )
        frame = pd.DataFrame(payload.values()).rename(columns={"cik_str": "cik"})
        frame["ticker"] = frame["ticker"].str.upper()
        frame["cik"] = pd.to_numeric(frame["cik"], errors="raise").astype(int)
        record = DatasetRecord(
            provider="SEC EDGAR",
            dataset="company_tickers",
            retrieved_at=utc_now(),
            source_url=self.ticker_map_url,
            sha256=sha256_bytes(raw),
            rows=len(frame),
            finality="current_mapping",
            revision_policy="raw response retained by content hash in run manifest",
        )
        return frame, record

    def fetch_company(
        self, ticker: str, cik: int, *, refresh: bool = False
    ) -> tuple[pd.DataFrame, dict[str, Any], list[DatasetRecord]]:
        facts_url = self.companyfacts_url.format(cik=cik)
        submissions_url = self.submissions_url.format(cik=cik)
        facts_payload, facts_raw = self._get_json(
            facts_url, self.cache_dir / "companyfacts" / f"CIK{cik:010d}.json.gz", refresh=refresh
        )
        submissions_payload, submissions_raw = self._get_json(
            submissions_url,
            self.cache_dir / "submissions" / f"CIK{cik:010d}.json.gz",
            refresh=refresh,
        )
        acceptance = self._acceptance_map(submissions_payload)
        concepts = self.normalize_annual_concepts(ticker, cik, facts_payload, acceptance)
        metadata = {
            "ticker": ticker,
            "cik": cik,
            "entity_name": submissions_payload.get("name") or facts_payload.get("entityName"),
            "sic": submissions_payload.get("sic"),
            "sic_description": submissions_payload.get("sicDescription"),
            "exchanges": submissions_payload.get("exchanges", []),
            "former_names": submissions_payload.get("formerNames", []),
        }
        eb = frame_bounds(concepts, "effective_at")
        ab = frame_bounds(concepts, "available_at")
        records = [
            DatasetRecord(
                provider="SEC EDGAR XBRL Company Facts",
                dataset=f"companyfacts:{ticker}",
                retrieved_at=utc_now(),
                source_url=facts_url,
                sha256=sha256_bytes(facts_raw),
                rows=len(concepts),
                min_effective_at=eb[0],
                max_effective_at=eb[1],
                min_available_at=ab[0],
                max_available_at=ab[1],
                finality="filed",
                revision_policy="amendments retained as later available_at rows",
                notes={"taxonomy": "us-gaap", "forms": ["10-K", "10-K/A"]},
            ),
            DatasetRecord(
                provider="SEC EDGAR Submissions",
                dataset=f"submissions:{ticker}",
                retrieved_at=utc_now(),
                source_url=submissions_url,
                sha256=sha256_bytes(submissions_raw),
                rows=len(acceptance),
                finality="filed",
                revision_policy="raw response retained; acceptance timestamps map by accession",
            ),
        ]
        return concepts, metadata, records

    @staticmethod
    def _acceptance_map(payload: dict[str, Any]) -> dict[str, pd.Timestamp]:
        recent = (payload.get("filings") or {}).get("recent") or {}
        accessions = recent.get("accessionNumber") or []
        accepted = recent.get("acceptanceDateTime") or []
        if not accessions or not accepted:
            return {}
        parsed = pd.to_datetime(pd.Series(accepted, dtype="string"), utc=True, errors="coerce")
        return {
            str(accession): stamp
            for accession, stamp in zip(accessions, parsed)
            if pd.notna(stamp)
        }

    @staticmethod
    def normalize_annual_concepts(
        ticker: str,
        cik: int,
        payload: dict[str, Any],
        acceptance: dict[str, pd.Timestamp] | None = None,
    ) -> pd.DataFrame:
        acceptance = acceptance or {}
        us_gaap = ((payload.get("facts") or {}).get("us-gaap") or {})
        rows: list[dict[str, Any]] = []
        for canonical, spec in ANNUAL_CONCEPTS.items():
            for priority, tag in enumerate(spec.tags):
                concept = us_gaap.get(tag) or {}
                units = concept.get("units") or {}
                entries = units.get(spec.unit) or []
                for item in entries:
                    if item.get("form") not in {"10-K", "10-K/A"}:
                        continue
                    if item.get("fp") not in {"FY", None}:
                        continue
                    end = pd.to_datetime(item.get("end"), errors="coerce")
                    filed = pd.to_datetime(item.get("filed"), errors="coerce")
                    if pd.isna(end) or pd.isna(filed):
                        continue
                    start = pd.to_datetime(item.get("start"), errors="coerce")
                    if spec.kind == "duration":
                        if pd.isna(start):
                            continue
                        duration = (end - start).days
                        if not 300 <= duration <= 430:
                            continue
                    accession = str(item.get("accn", ""))
                    available = acceptance.get(accession)
                    if available is None:
                        available = pd.Timestamp(filed).tz_localize("UTC") + pd.Timedelta(days=1)
                    value = pd.to_numeric(item.get("val"), errors="coerce")
                    if pd.isna(value):
                        continue
                    rows.append(
                        {
                            "ticker": ticker,
                            "cik": cik,
                            "concept": canonical,
                            "value": float(value),
                            "unit": spec.unit,
                            "fiscal_start": start,
                            "fiscal_end": end,
                            "effective_at": pd.Timestamp(end).tz_localize("UTC"),
                            "filed": filed,
                            "available_at": available,
                            "form": item.get("form"),
                            "fiscal_year": item.get("fy"),
                            "accession": accession,
                            "source_tag": tag,
                            "tag_priority": priority,
                        }
                    )
        if not rows:
            return pd.DataFrame(
                columns=[
                    "ticker",
                    "cik",
                    "concept",
                    "value",
                    "unit",
                    "fiscal_start",
                    "fiscal_end",
                    "effective_at",
                    "filed",
                    "available_at",
                    "form",
                    "fiscal_year",
                    "accession",
                    "source_tag",
                ]
            )
        frame = pd.DataFrame(rows)
        frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True)
        frame = frame.sort_values(
            ["concept", "fiscal_end", "available_at", "tag_priority"],
            ascending=[True, True, True, True],
        )
        frame = frame.drop_duplicates(
            ["ticker", "concept", "fiscal_end", "available_at"], keep="first"
        )
        return frame.drop(columns="tag_priority").reset_index(drop=True)

    def fetch_universe(
        self,
        tickers: list[str],
        *,
        refresh: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[DatasetRecord], pd.DataFrame]:
        mapping, map_record = self.ticker_map(refresh=refresh)
        lookup = mapping.set_index("ticker")["cik"].to_dict()
        concept_frames: list[pd.DataFrame] = []
        metadata: list[dict[str, Any]] = []
        records: list[DatasetRecord] = [map_record]
        audit: list[dict[str, Any]] = []
        for ticker in tickers:
            cik = lookup.get(ticker)
            if cik is None:
                audit.append({"ticker": ticker, "status": "missing_cik", "rows": 0})
                continue
            try:
                concepts, meta, company_records = self.fetch_company(ticker, int(cik), refresh=refresh)
                concept_frames.append(concepts)
                metadata.append(meta)
                records.extend(company_records)
                audit.append({"ticker": ticker, "status": "ok", "rows": len(concepts)})
            except Exception as exc:
                audit.append({"ticker": ticker, "status": f"error: {exc}", "rows": 0})
        if not concept_frames:
            raise RuntimeError("No SEC company facts downloaded")
        return (
            pd.concat(concept_frames, ignore_index=True),
            pd.DataFrame(metadata),
            records,
            pd.DataFrame(audit).sort_values("ticker"),
        )
