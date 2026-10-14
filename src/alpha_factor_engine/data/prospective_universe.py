"""Immutable prospective equity-universe snapshots from public raw caches.

This module deliberately solves a narrow problem: freeze the securities that
are observable *from the freeze time forward*.  SEC CIK is an issuer ID and a
ticker is only a versioned symbol, so neither is promoted to a permanent
security ID.  Consumers requesting historical-backtest or deployment use are
blocked until authoritative security-level lineage, share-class and delisting
evidence have been attached.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from alpha_factor_engine.contracts import PointInTimeViolation, sha256_bytes, sha256_file
from alpha_factor_engine.data.yahoo import YahooFinanceClient


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_PURPOSES = {"prospective_monitoring", "historical_backtest", "deployment"}
_FORBIDDEN_LABEL_COLUMNS = {
    "target_return",
    "forward_return",
    "next_month_return",
    "future_return",
    "label",
    "base_period_return",
    "delisting_return",
}


class ProspectiveUniverseViolation(PointInTimeViolation):
    """Raised when a prospective snapshot is malformed or misused."""


@dataclass(frozen=True)
class ProspectiveUniverseRules:
    """Predeclared inputs and screens for a single prospective cohort."""

    candidate_count: int = 2_500
    minimum_members: int = 1_000
    maximum_members: int = 4_000
    minimum_price: float = 5.0
    minimum_adv_usd_63d: float = 5_000_000.0
    minimum_price_observations: int = 63
    instrument_type: str = "EQUITY"
    currency: str = "USD"
    allowed_exchange_codes: tuple[str, ...] = ("NYQ", "NMS", "NGM", "NCM", "ASE")

    def __post_init__(self) -> None:
        if self.candidate_count <= 0:
            raise ValueError("candidate_count must be positive")
        if not 0 < self.minimum_members <= self.maximum_members:
            raise ValueError("member bounds must satisfy 0 < minimum <= maximum")
        if self.minimum_price < 0 or self.minimum_adv_usd_63d < 0:
            raise ValueError("price and ADV screens must be nonnegative")
        if self.minimum_price_observations <= 0:
            raise ValueError("minimum_price_observations must be positive")


REQUIRED_SNAPSHOT_COLUMNS: tuple[str, ...] = (
    "cohort_id",
    "freeze_at",
    "source_effective_at",
    "source_available_at",
    "ingested_at",
    "membership_effective_at",
    "membership_available_at",
    "identity_source_effective_at",
    "identity_source_available_at",
    "identity_ingested_at",
    "identity_effective_time_status",
    "universe_rank",
    "issuer_id",
    "cik",
    "ticker",
    "exchange_code",
    "exchange_name",
    "instrument_type",
    "currency",
    "share_class",
    "share_class_status",
    "permanent_security_id",
    "permanent_id_status",
    "snapshot_security_key",
    "lineage_status",
    "price_source_effective_at",
    "price_source_available_at",
    "last_unadjusted_close",
    "trailing_63d_median_dollar_volume",
    "sec_raw_payload_sha256",
    "yahoo_raw_payload_sha256",
    "prospective_observation_eligible",
    "historical_backtest_eligible",
    "deployable_eligible",
)


def _utc(value: str | pd.Timestamp, *, name: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if pd.isna(result):
        raise ProspectiveUniverseViolation(f"{name} cannot be missing")
    if result.tzinfo is None:
        result = result.tz_localize("UTC")
    return result.tz_convert("UTC")


def _normalize_ticker(value: object) -> str:
    return str(value).strip().upper().replace(".", "-")


def _snapshot_key(cik: int, ticker: str, exchange_code: str) -> str:
    # Snapshot-local bridge only.  It is never advertised as a permanent ID.
    value = f"sec-cik={int(cik):010d}|ticker={ticker}|exchange={exchange_code}"
    return "snapshot:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_sec_candidate_cache(path: str | Path, *, candidate_count: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the ordered current SEC issuer/ticker map without filling gaps."""

    source = Path(path)
    compressed = source.read_bytes()
    raw = gzip.decompress(compressed)
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ProspectiveUniverseViolation("SEC ticker cache must contain a JSON object")
    rows = list(payload.values())[:candidate_count]
    if len(rows) != candidate_count:
        raise ProspectiveUniverseViolation(
            f"SEC ticker cache has {len(rows)} requested candidates; expected {candidate_count}"
        )
    frame = pd.DataFrame(rows).rename(columns={"cik_str": "cik"})
    required = {"cik", "ticker", "title"}
    missing = required.difference(frame.columns)
    if missing:
        raise ProspectiveUniverseViolation(f"SEC ticker payload missing fields: {sorted(missing)}")
    frame["cik"] = pd.to_numeric(frame["cik"], errors="raise").astype("int64")
    frame["ticker"] = frame["ticker"].map(_normalize_ticker)
    frame.insert(0, "candidate_order", np.arange(1, len(frame) + 1, dtype=int))
    if frame["ticker"].eq("").any() or frame["cik"].le(0).any():
        raise ProspectiveUniverseViolation("SEC ticker payload contains blank ticker or invalid CIK")
    if frame.duplicated("ticker").any():
        raise ProspectiveUniverseViolation("ordered SEC candidate slice contains duplicate tickers")
    return frame, {
        "path": source,
        "file_bytes": len(compressed),
        "file_sha256": sha256_bytes(compressed),
        "raw_bytes": len(raw),
        "raw_sha256": sha256_bytes(raw),
        "payload_rows": len(payload),
    }


def parse_yahoo_cache(
    path: str | Path,
    *,
    expected_ticker: str,
    cutoff_date: str | pd.Timestamp,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Parse one exact Yahoo response and summarize only bars through cutoff."""

    source = Path(path)
    compressed = source.read_bytes()
    raw = gzip.decompress(compressed)
    payload = json.loads(raw)
    error = (payload.get("chart") or {}).get("error")
    result = ((payload.get("chart") or {}).get("result") or [None])[0]
    if error or result is None:
        raise ProspectiveUniverseViolation(f"Yahoo payload unusable for {expected_ticker}: {error}")
    meta = result.get("meta") or {}
    returned_ticker = _normalize_ticker(meta.get("symbol", expected_ticker))
    if returned_ticker != expected_ticker:
        raise ProspectiveUniverseViolation(
            f"Yahoo symbol mismatch: requested={expected_ticker}, returned={returned_ticker}"
        )
    prices = YahooFinanceClient._parse_result(expected_ticker, result)
    cutoff = pd.Timestamp(cutoff_date).normalize()
    usable = prices.loc[prices["date"].le(cutoff)].copy()
    usable = usable.loc[
        pd.to_numeric(usable["close"], errors="coerce").gt(0)
        & pd.to_numeric(usable["volume"], errors="coerce").ge(0)
    ].sort_values("date", kind="mergesort")
    if usable.empty:
        raise ProspectiveUniverseViolation(f"Yahoo payload has no price through cutoff for {expected_ticker}")
    recent = usable.tail(63)
    dollar_volume = pd.to_numeric(recent["close"], errors="coerce") * pd.to_numeric(
        recent["volume"], errors="coerce"
    )
    effective_at = pd.to_datetime(usable["effective_at"], utc=True).max()
    summary = {
        "ticker": expected_ticker,
        "instrument_type": str(meta.get("instrumentType") or ""),
        "exchange_code": str(meta.get("exchangeName") or ""),
        "exchange_name": str(meta.get("fullExchangeName") or ""),
        "currency": str(meta.get("currency") or ""),
        "last_unadjusted_close": float(usable.iloc[-1]["close"]),
        "trailing_63d_median_dollar_volume": float(dollar_volume.median()),
        "history_rows_through_cutoff": int(len(usable)),
        "recent_price_observations": int(len(recent)),
        "first_price_date": usable["date"].min().date().isoformat(),
        "last_price_date": usable["date"].max().date().isoformat(),
        "price_source_effective_at": effective_at.isoformat(),
        "price_source_available_at": (effective_at + pd.Timedelta(minutes=30)).isoformat(),
    }
    evidence = {
        "path": source,
        "file_bytes": len(compressed),
        "file_sha256": sha256_bytes(compressed),
        "raw_bytes": len(raw),
        "raw_sha256": sha256_bytes(raw),
    }
    return summary, evidence


def _gzip_evidence(path: Path) -> dict[str, Any]:
    """Hash both stored and decompressed bytes, even when semantic parsing fails."""

    compressed = path.read_bytes()
    raw = gzip.decompress(compressed)
    return {
        "file_bytes": len(compressed),
        "file_sha256": sha256_bytes(compressed),
        "raw_bytes": len(raw),
        "raw_sha256": sha256_bytes(raw),
    }


def _source_root_hash(source_rows: Iterable[dict[str, Any]]) -> str:
    rows = [
        {
            "relative_path": str(row["relative_path"]),
            "file_bytes": int(row["file_bytes"]),
            "file_sha256": str(row["file_sha256"]),
            "raw_bytes": int(row["raw_bytes"]),
            "raw_sha256": str(row["raw_sha256"]),
        }
        for row in source_rows
    ]
    encoded = json.dumps(sorted(rows, key=lambda item: item["relative_path"]), separators=(",", ":"), sort_keys=True).encode()
    return sha256_bytes(encoded)


def build_prospective_universe(
    *,
    root: str | Path,
    sec_cache: str | Path,
    yahoo_cache_dir: str | Path,
    yahoo_start: str,
    yahoo_end: str,
    cutoff_date: str,
    freeze_at: str | pd.Timestamp,
    rules: ProspectiveUniverseRules | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build one deterministic, all-rows-reconciled prospective cohort."""

    project_root = Path(root).resolve()
    rule = rules or ProspectiveUniverseRules()
    frozen = _utc(freeze_at, name="freeze_at")
    cutoff = pd.Timestamp(cutoff_date).normalize()
    if cutoff.date() >= frozen.date():
        raise ProspectiveUniverseViolation(
            "cutoff_date must precede freeze date so only completed sessions enter membership"
        )

    sec_path = (project_root / sec_cache).resolve() if not Path(sec_cache).is_absolute() else Path(sec_cache)
    yahoo_dir = (
        (project_root / yahoo_cache_dir).resolve()
        if not Path(yahoo_cache_dir).is_absolute()
        else Path(yahoo_cache_dir)
    )
    candidates, sec_evidence = load_sec_candidate_cache(
        sec_path, candidate_count=rule.candidate_count
    )
    sec_relative = sec_path.relative_to(project_root).as_posix()
    source_rows: list[dict[str, Any]] = [
        {
            "provider": "SEC EDGAR",
            "dataset": "company_tickers_current_mapping",
            "relative_path": sec_relative,
            "source_url": "https://www.sec.gov/files/company_tickers.json",
            "source_effective_at": "",
            "source_available_at": "",
            "ingested_at": frozen.isoformat(),
            "ingested_at_basis": "conservative_snapshot_freeze_bound",
            "file_bytes": sec_evidence["file_bytes"],
            "file_sha256": sec_evidence["file_sha256"],
            "raw_bytes": sec_evidence["raw_bytes"],
            "raw_sha256": sec_evidence["raw_sha256"],
        }
    ]

    audit_rows: list[dict[str, Any]] = []
    for item in candidates.itertuples(index=False):
        ticker = str(item.ticker)
        safe = YahooFinanceClient._safe_symbol(ticker)
        cache_path = yahoo_dir / f"{safe}_{yahoo_start}_{yahoo_end}.json.gz"
        base: dict[str, Any] = {
            "candidate_order": int(item.candidate_order),
            "cik": int(item.cik),
            "ticker": ticker,
            "title": str(item.title),
            "cache_status": "missing",
            "screen_pass": False,
            "screen_reasons": "missing_yahoo_raw_cache",
        }
        if not cache_path.is_file():
            audit_rows.append(base)
            continue
        raw_evidence: dict[str, Any] | None = None
        try:
            raw_evidence = _gzip_evidence(cache_path)
            summary, evidence = parse_yahoo_cache(
                cache_path, expected_ticker=ticker, cutoff_date=cutoff
            )
            base.update(summary)
            base.update(
                {
                    "cache_status": "parsed",
                    "yahoo_file_sha256": evidence["file_sha256"],
                    "yahoo_raw_payload_sha256": evidence["raw_sha256"],
                }
            )
            reasons: list[str] = []
            if summary["instrument_type"] != rule.instrument_type:
                reasons.append("instrument_type")
            if summary["currency"] != rule.currency:
                reasons.append("currency")
            if summary["exchange_code"] not in rule.allowed_exchange_codes:
                reasons.append("exchange")
            if summary["last_unadjusted_close"] < rule.minimum_price:
                reasons.append("price")
            if summary["trailing_63d_median_dollar_volume"] < rule.minimum_adv_usd_63d:
                reasons.append("adv")
            if summary["recent_price_observations"] < rule.minimum_price_observations:
                reasons.append("history")
            if _utc(summary["price_source_available_at"], name="price_source_available_at") > frozen:
                reasons.append("future_availability")
            base["screen_pass"] = not reasons
            base["screen_reasons"] = "included" if not reasons else "|".join(reasons)

            source_rows.append(
                {
                    "provider": "Yahoo Finance chart API",
                    "dataset": f"daily_prices_and_current_metadata:{ticker}",
                    "relative_path": cache_path.relative_to(project_root).as_posix(),
                    "source_url": f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
                    "source_effective_at": summary["price_source_effective_at"],
                    "source_available_at": summary["price_source_available_at"],
                    "ingested_at": frozen.isoformat(),
                    "ingested_at_basis": "conservative_snapshot_freeze_bound",
                    "file_bytes": evidence["file_bytes"],
                    "file_sha256": evidence["file_sha256"],
                    "raw_bytes": evidence["raw_bytes"],
                    "raw_sha256": evidence["raw_sha256"],
                }
            )
        except Exception as exc:
            base["cache_status"] = "parse_error"
            base["screen_reasons"] = f"parse_error:{type(exc).__name__}:{exc}"
            if raw_evidence is not None:
                base.update(
                    {
                        "yahoo_file_sha256": raw_evidence["file_sha256"],
                        "yahoo_raw_payload_sha256": raw_evidence["raw_sha256"],
                    }
                )
                source_rows.append(
                    {
                        "provider": "Yahoo Finance chart API",
                        "dataset": f"unusable_raw_payload:{ticker}",
                        "relative_path": cache_path.relative_to(project_root).as_posix(),
                        "source_url": f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
                        "source_effective_at": "",
                        "source_available_at": "",
                        "ingested_at": frozen.isoformat(),
                        "ingested_at_basis": "conservative_snapshot_freeze_bound",
                        "file_bytes": raw_evidence["file_bytes"],
                        "file_sha256": raw_evidence["file_sha256"],
                        "raw_bytes": raw_evidence["raw_bytes"],
                        "raw_sha256": raw_evidence["raw_sha256"],
                    }
                )
        audit_rows.append(base)

    audit = pd.DataFrame(audit_rows).sort_values("candidate_order", kind="mergesort")
    eligible = audit.loc[audit["screen_pass"].fillna(False).astype(bool)].copy()
    # One snapshot security proxy per issuer.  The current SEC map is issuer
    # oriented and may omit secondary classes; keeping duplicates would create
    # unjustified double exposure.  Ties are deterministic and label-free.
    eligible = eligible.sort_values(
        ["cik", "trailing_63d_median_dollar_volume", "ticker"],
        ascending=[True, False, True],
        kind="mergesort",
    ).drop_duplicates("cik", keep="first")
    eligible = eligible.sort_values(
        ["trailing_63d_median_dollar_volume", "ticker"],
        ascending=[False, True],
        kind="mergesort",
    )
    if len(eligible) > rule.maximum_members:
        eligible = eligible.head(rule.maximum_members).copy()
    if len(eligible) < rule.minimum_members:
        raise ProspectiveUniverseViolation(
            f"only {len(eligible)} candidates pass frozen screens; minimum={rule.minimum_members}"
        )

    source_manifest = pd.DataFrame(source_rows).sort_values("relative_path", kind="mergesort")
    source_root_sha256 = _source_root_hash(source_rows)
    eligible = eligible.reset_index(drop=True)
    eligible["universe_rank"] = np.arange(1, len(eligible) + 1, dtype=int)
    cohort_payload = {
        "freeze_at": frozen.isoformat(),
        "cutoff_date": cutoff.date().isoformat(),
        "rules": asdict(rule),
        "source_root_sha256": source_root_sha256,
        "members": eligible[
            ["universe_rank", "cik", "ticker", "exchange_code", "yahoo_raw_payload_sha256"]
        ].to_dict("records"),
    }
    cohort_id = "prospective:" + sha256_bytes(
        json.dumps(cohort_payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    )

    snapshot = pd.DataFrame(
        {
            "cohort_id": cohort_id,
            "freeze_at": frozen.isoformat(),
            "source_effective_at": eligible["price_source_effective_at"],
            # SEC mapping publication time is absent from the raw payload.  The
            # freeze time is a conservative system-known bound for the combined
            # membership decision.
            "source_available_at": frozen.isoformat(),
            "ingested_at": frozen.isoformat(),
            "membership_effective_at": frozen.isoformat(),
            "membership_available_at": frozen.isoformat(),
            # Neither current-mapping payload contains a bitemporal identity
            # interval.  Leave effective time null and expose that limitation,
            # while conservatively recording when the mapping became usable by
            # this system.
            "identity_source_effective_at": "",
            "identity_source_available_at": frozen.isoformat(),
            "identity_ingested_at": frozen.isoformat(),
            "identity_effective_time_status": "UNAVAILABLE_IN_SOURCE_PAYLOAD",
            "selection_cutoff_date": cutoff.date().isoformat(),
            "universe_rank": eligible["universe_rank"].astype(int),
            "issuer_id": eligible["cik"].map(lambda value: f"SEC_CIK{int(value):010d}"),
            "cik": eligible["cik"].astype("int64"),
            "ticker": eligible["ticker"].astype(str),
            "issuer_name": eligible["title"].astype(str),
            "exchange_code": eligible["exchange_code"].astype(str),
            "exchange_name": eligible["exchange_name"].astype(str),
            "cik_ticker_source": "SEC_EDGAR_COMPANY_TICKERS_CURRENT_MAPPING",
            "exchange_source": "YAHOO_CHART_CURRENT_METADATA",
            "exchange_metadata_status": "CURRENT_AT_INGESTION_NO_EFFECTIVE_INTERVAL",
            "instrument_type": eligible["instrument_type"].astype(str),
            "currency": eligible["currency"].astype(str),
            "share_class": "UNKNOWN",
            "share_class_status": "UNVERIFIED_PUBLIC_SOURCE",
            "permanent_security_id": pd.Series([pd.NA] * len(eligible), dtype="string"),
            "permanent_id_status": "UNAVAILABLE_PUBLIC_SOURCE",
            "snapshot_security_key": [
                _snapshot_key(row.cik, row.ticker, row.exchange_code)
                for row in eligible.itertuples(index=False)
            ],
            "lineage_status": "PROSPECTIVE_ONLY_FROM_FREEZE_AT",
            "valid_from": frozen.isoformat(),
            "valid_to": "",
            "price_source_effective_at": eligible["price_source_effective_at"],
            "price_source_available_at": eligible["price_source_available_at"],
            "last_price_date": eligible["last_price_date"],
            "last_unadjusted_close": eligible["last_unadjusted_close"].astype(float),
            "trailing_63d_median_dollar_volume": eligible[
                "trailing_63d_median_dollar_volume"
            ].astype(float),
            "history_rows_through_cutoff": eligible["history_rows_through_cutoff"].astype(int),
            "recent_price_observations": eligible["recent_price_observations"].astype(int),
            "sec_raw_payload_sha256": sec_evidence["raw_sha256"],
            "yahoo_raw_payload_sha256": eligible["yahoo_raw_payload_sha256"].astype(str),
            "source_root_sha256": source_root_sha256,
            "selection_rule": "current_SEC_ordered_candidates_then_frozen_public_screens_no_labels",
            "prospective_observation_eligible": True,
            "historical_backtest_eligible": False,
            "deployable_eligible": False,
        }
    )
    audit["selected_member"] = audit["ticker"].isin(set(snapshot["ticker"]))
    audit["rejection_is_explicit"] = ~audit["selected_member"]
    qa = validate_prospective_snapshot(
        snapshot,
        freeze_at=frozen,
        minimum_members=rule.minimum_members,
        maximum_members=rule.maximum_members,
    )
    qa.update(
        {
            "cohort_id": cohort_id,
            "rules": asdict(rule),
            "candidate_rows": int(len(audit)),
            "parsed_yahoo_rows": int(audit["cache_status"].eq("parsed").sum()),
            "screen_pass_rows_before_issuer_dedup": int(audit["screen_pass"].sum()),
            "member_rows": int(len(snapshot)),
            "source_records": int(len(source_manifest)),
            "raw_candidate_caches_missing": int(audit["cache_status"].eq("missing").sum()),
            "raw_candidate_caches_hashed": int(
                audit["yahoo_raw_payload_sha256"].notna().sum()
                if "yahoo_raw_payload_sha256" in audit
                else 0
            ),
            "source_root_sha256": source_root_sha256,
            "sec_payload_rows": int(sec_evidence["payload_rows"]),
            "marketdata_cache_used": False,
            "marketdata_token_required_for_build": False,
            "historical_survivorship_free": False,
            "claim_limit": "prospective_membership_observation_from_freeze_at_only",
            "research_verdict_historical_alpha": "NOT VALIDATED",
        }
    )
    return snapshot, audit, source_manifest, qa


def validate_prospective_snapshot(
    frame: pd.DataFrame,
    *,
    freeze_at: str | pd.Timestamp,
    minimum_members: int = 1_000,
    maximum_members: int = 4_000,
) -> dict[str, Any]:
    """Validate integrity and explicitly retain historical/deployment blocks."""

    frozen = _utc(freeze_at, name="freeze_at")
    checks: list[dict[str, Any]] = []

    def add(scope: str, check: str, passed: bool, detail: str, severity: str) -> None:
        checks.append(
            {
                "scope": scope,
                "check": check,
                "passed": bool(passed),
                "severity": severity,
                "detail": detail,
            }
        )

    missing = sorted(set(REQUIRED_SNAPSHOT_COLUMNS).difference(frame.columns))
    add("snapshot_integrity", "required_columns", not missing, f"missing={missing}", "Critical")
    if missing:
        raise ProspectiveUniverseViolation(f"prospective snapshot missing columns: {missing}")
    forbidden = sorted(_FORBIDDEN_LABEL_COLUMNS.intersection(frame.columns))
    add("snapshot_integrity", "no_future_label_columns", not forbidden, f"present={forbidden}", "Critical")
    duplicate_key = frame.duplicated("snapshot_security_key", keep=False)
    duplicate_ticker = frame.duplicated("ticker", keep=False)
    duplicate_cik = frame.duplicated("cik", keep=False)
    add("snapshot_integrity", "unique_snapshot_security_key", not duplicate_key.any(), f"duplicates={int(duplicate_key.sum())}", "Critical")
    add("snapshot_integrity", "unique_ticker", not duplicate_ticker.any(), f"duplicates={int(duplicate_ticker.sum())}", "Critical")
    add("snapshot_integrity", "one_proxy_per_issuer", not duplicate_cik.any(), f"duplicates={int(duplicate_cik.sum())}", "Critical")
    size_ok = minimum_members <= len(frame) <= maximum_members
    add("snapshot_integrity", "member_count_bounds", size_ok, f"members={len(frame)}; bounds=[{minimum_members},{maximum_members}]", "Critical")

    date_columns = [
        "freeze_at",
        "source_effective_at",
        "source_available_at",
        "ingested_at",
        "membership_effective_at",
        "membership_available_at",
        "identity_source_available_at",
        "identity_ingested_at",
        "price_source_effective_at",
        "price_source_available_at",
        "valid_from",
    ]
    parsed = {name: pd.to_datetime(frame[name], utc=True, errors="coerce") for name in date_columns}
    timestamps_complete = all(value.notna().all() for value in parsed.values())
    add("snapshot_integrity", "timestamps_complete", timestamps_complete, f"columns={date_columns}", "Critical")
    known_by_freeze = (
        parsed["source_effective_at"].le(parsed["source_available_at"])
        & parsed["price_source_effective_at"].le(parsed["price_source_available_at"])
        & parsed["price_source_available_at"].le(frozen)
        & parsed["source_available_at"].eq(frozen)
        & parsed["ingested_at"].eq(frozen)
        & parsed["membership_available_at"].eq(frozen)
        & parsed["membership_effective_at"].eq(frozen)
        & parsed["identity_source_available_at"].eq(frozen)
        & parsed["identity_ingested_at"].eq(frozen)
        & parsed["valid_from"].eq(frozen)
    )
    add("snapshot_integrity", "information_clock", bool(known_by_freeze.all()), f"invalid_rows={int((~known_by_freeze).sum())}", "Critical")
    identity_clock_explicit = (
        frame["identity_source_effective_at"].fillna("").astype(str).str.strip().eq("")
        & frame["identity_effective_time_status"].eq("UNAVAILABLE_IN_SOURCE_PAYLOAD")
    )
    add(
        "snapshot_integrity",
        "unknown_identity_effective_time_is_explicit",
        bool(identity_clock_explicit.all()),
        f"invalid_rows={int((~identity_clock_explicit).sum())}",
        "Critical",
    )
    hashes = pd.concat(
        [frame["sec_raw_payload_sha256"], frame["yahoo_raw_payload_sha256"], frame["source_root_sha256"]],
        ignore_index=True,
    ).astype(str)
    hash_ok = hashes.map(lambda value: bool(_SHA256.fullmatch(value))).all()
    add("snapshot_integrity", "raw_payload_hashes", bool(hash_ok), f"hash_values={len(hashes)}", "Critical")
    prospective = frame["prospective_observation_eligible"].eq(True)
    add("snapshot_integrity", "prospective_members_explicit", bool(prospective.all()), f"ineligible_rows={int((~prospective).sum())}", "Critical")

    permanent = frame["permanent_security_id"].notna() & frame["permanent_security_id"].astype(str).str.strip().ne("")
    share_class = frame["share_class_status"].eq("VERIFIED")
    historical_blocked = frame["historical_backtest_eligible"].eq(False)
    deployment_blocked = frame["deployable_eligible"].eq(False)
    add("historical_claim", "authoritative_permanent_security_id", bool(permanent.all()), f"coverage={permanent.mean():.6f}", "Critical")
    add("historical_claim", "verified_share_class", bool(share_class.all()), f"coverage={share_class.mean():.6f}", "Major")
    add("historical_claim", "inactive_and_delisting_history", False, "current prospective cohort has no prior inactive-security/delisting panel", "Critical")
    add("fail_closed", "historical_use_blocked", bool(historical_blocked.all()), f"unblocked_rows={int((~historical_blocked).sum())}", "Critical")
    add("fail_closed", "deployment_use_blocked", bool(deployment_blocked.all()), f"unblocked_rows={int((~deployment_blocked).sum())}", "Critical")

    integrity_failures = [item for item in checks if item["scope"] == "snapshot_integrity" and not item["passed"]]
    fail_closed_failures = [item for item in checks if item["scope"] == "fail_closed" and not item["passed"]]
    if integrity_failures or fail_closed_failures:
        names = [item["check"] for item in integrity_failures + fail_closed_failures]
        raise ProspectiveUniverseViolation("prospective snapshot failed closed: " + ", ".join(names))
    return {
        "snapshot_integrity_status": "PASS",
        "prospective_monitoring_status": "PASS",
        "historical_backtest_status": "NOT VALIDATED",
        "deployment_status": "NOT VALIDATED",
        "checks": checks,
    }


def require_snapshot_use(
    frame: pd.DataFrame,
    *,
    purpose: str,
    decision_time: str | pd.Timestamp,
) -> None:
    """Fail closed when a prospective-only snapshot is used outside scope."""

    if purpose not in _ALLOWED_PURPOSES:
        raise ValueError(f"purpose must be one of {sorted(_ALLOWED_PURPOSES)}")
    decision = _utc(decision_time, name="decision_time")
    available = pd.to_datetime(frame["membership_available_at"], utc=True, errors="raise")
    if available.gt(decision).any():
        raise ProspectiveUniverseViolation("decision_time precedes membership availability")
    if purpose == "prospective_monitoring":
        if not frame["prospective_observation_eligible"].eq(True).all():
            raise ProspectiveUniverseViolation("snapshot contains prospective-ineligible rows")
        return
    eligibility_column = (
        "historical_backtest_eligible" if purpose == "historical_backtest" else "deployable_eligible"
    )
    if not frame[eligibility_column].eq(True).all():
        raise ProspectiveUniverseViolation(
            f"snapshot is prospective-only; {purpose} is blocked by {eligibility_column}"
        )


def verify_source_manifest(frame: pd.DataFrame, *, root: str | Path) -> dict[str, Any]:
    """Rehash every listed compressed payload and its decompressed bytes."""

    project_root = Path(root).resolve()
    required = {"relative_path", "file_bytes", "file_sha256", "raw_bytes", "raw_sha256"}
    missing = required.difference(frame.columns)
    if missing:
        raise ProspectiveUniverseViolation(f"source manifest missing columns: {sorted(missing)}")
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    for item in frame.sort_values("relative_path", kind="mergesort").itertuples(index=False):
        path = (project_root / str(item.relative_path)).resolve()
        within_root = project_root == path or project_root in path.parents
        if not within_root or not path.is_file():
            errors.append(f"missing_or_outside_root:{item.relative_path}")
            continue
        compressed = path.read_bytes()
        try:
            raw = gzip.decompress(compressed)
        except Exception as exc:
            errors.append(f"decompression_error:{item.relative_path}:{type(exc).__name__}")
            continue
        passed = (
            len(compressed) == int(item.file_bytes)
            and sha256_bytes(compressed) == str(item.file_sha256)
            and len(raw) == int(item.raw_bytes)
            and sha256_bytes(raw) == str(item.raw_sha256)
        )
        rows.append({"relative_path": str(item.relative_path), "passed": bool(passed)})
        if not passed:
            errors.append(f"hash_or_size_mismatch:{item.relative_path}")
    observed_root = _source_root_hash(frame.to_dict("records"))
    return {
        "passed": not errors,
        "records": int(len(frame)),
        "verified_records": int(sum(row["passed"] for row in rows)),
        "source_root_sha256": observed_root,
        "errors": errors,
    }


def build_artifact_manifest(
    *,
    root: str | Path,
    cohort_id: str,
    freeze_at: str,
    source_root_sha256: str,
    inputs: Iterable[str | Path],
    outputs: Iterable[str | Path],
    verdict: str = "NOT VALIDATED",
) -> dict[str, Any]:
    """Build a byte-level manifest; the manifest itself is intentionally excluded."""

    project_root = Path(root).resolve()

    def record(path_value: str | Path) -> dict[str, Any]:
        path = Path(path_value).resolve()
        return {
            "path": path.relative_to(project_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    input_rows = sorted((record(path) for path in inputs), key=lambda item: item["path"])
    output_rows = sorted((record(path) for path in outputs), key=lambda item: item["path"])
    return {
        "schema_version": 1,
        "cohort_id": cohort_id,
        "created_at": freeze_at,
        "source_root_sha256": source_root_sha256,
        "snapshot_scope": "prospective_only_from_freeze_at",
        "historical_survivorship_free": False,
        "research_verdict_historical_alpha": verdict,
        "inputs": input_rows,
        "outputs": output_rows,
    }


def verify_artifact_manifest(manifest: dict[str, Any], *, root: str | Path) -> dict[str, Any]:
    project_root = Path(root).resolve()
    errors: list[str] = []
    checked = 0
    for section in ("inputs", "outputs"):
        for record in manifest.get(section, []):
            checked += 1
            path = (project_root / str(record["path"])).resolve()
            if project_root not in path.parents or not path.is_file():
                errors.append(f"missing_or_outside_root:{record['path']}")
                continue
            if path.stat().st_size != int(record["bytes"]) or sha256_file(path) != str(record["sha256"]):
                errors.append(f"hash_or_size_mismatch:{record['path']}")
    return {"passed": not errors, "checked_records": checked, "errors": errors}


__all__ = [
    "ProspectiveUniverseRules",
    "ProspectiveUniverseViolation",
    "REQUIRED_SNAPSHOT_COLUMNS",
    "build_artifact_manifest",
    "build_prospective_universe",
    "load_sec_candidate_cache",
    "parse_yahoo_cache",
    "require_snapshot_use",
    "validate_prospective_snapshot",
    "verify_artifact_manifest",
    "verify_source_manifest",
]
