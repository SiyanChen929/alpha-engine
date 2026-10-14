"""Strict schemas for licensed and specialist point-in-time datasets.

These adapters deliberately fail closed. A path or credential alone does not
make a dataset point-in-time safe; each row must carry its knowledge timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from alpha_factor_engine.contracts import PointInTimeViolation, ensure_pit_columns, validate_unique


@dataclass(frozen=True)
class VendorContract:
    name: str
    entity_keys: tuple[str, ...]
    effective_column: str
    available_column: str
    required_columns: tuple[str, ...]
    notes: str
    availability_may_precede_effective: bool = False


CONTRACTS: dict[str, VendorContract] = {
    "crsp": VendorContract(
        "CRSP stock file + delisting + name history",
        ("permno", "date"),
        "date",
        "available_at",
        ("permno", "date", "ret", "dlret", "prc", "shrout", "shrcd", "exchcd", "available_at"),
        "Security master and delisting returns are mandatory; build universes from historical name/share-code files.",
    ),
    "compustat_pit": VendorContract(
        "Compustat point-in-time fundamentals",
        ("gvkey", "datadate", "available_at"),
        "datadate",
        "available_at",
        ("gvkey", "datadate", "fyear", "fyr", "at", "ceq", "sale", "ni", "oancf", "capx", "available_at"),
        "available_at must be derived from the vendor PIT timestamp or filing/publication time, never datadate.",
    ),
    "ibes": VendorContract(
        "I/B/E/S detail or summary estimates",
        ("ibes_id", "statpers", "fpedats", "available_at"),
        "fpedats",
        "available_at",
        ("ibes_id", "statpers", "fpedats", "measure", "value", "available_at"),
        "Use estimate announcement timestamps and split-adjustment mappings; retain revisions and withdrawals.",
        True,
    ),
    "taq": VendorContract(
        "NYSE TAQ trades and NBBO quotes",
        ("symbol", "sip_timestamp", "sequence_number"),
        "exchange_timestamp",
        "available_at",
        ("symbol", "exchange_timestamp", "sip_timestamp", "sequence_number", "condition", "available_at"),
        "Preserve exchange and SIP clocks, conditions, corrections, and sequence ordering.",
    ),
    "options": VendorContract(
        "OPRA/OCC option quotes and contract adjustments",
        ("option_id", "quote_timestamp"),
        "quote_timestamp",
        "available_at",
        ("option_id", "underlier_id", "expiry", "strike", "call_put", "bid", "ask", "quote_timestamp", "available_at"),
        "Backtests require historical tradable NBBO, OCC adjustment history, open interest publication lag, and conservative fills.",
    ),
    "borrow": VendorContract(
        "Historical locate/borrow inventory",
        ("security_id", "quote_timestamp"),
        "quote_timestamp",
        "available_at",
        ("security_id", "fee_bps", "shares_available", "quote_timestamp", "available_at"),
        "A generic borrow-rate assumption is only a sensitivity test; hard-to-borrow names require time-stamped inventory.",
    ),
    "events": VendorContract(
        "Corporate/index/macro event tape",
        ("event_id", "revision_id"),
        "effective_at",
        "available_at",
        ("event_id", "event_type", "effective_at", "announced_at", "available_at", "revision_id"),
        "Retain original announcement, revisions, cancellations, and vendor receipt timestamps.",
        True,
    ),
}


class ParquetPITAdapter:
    def __init__(self, contract: VendorContract) -> None:
        self.contract = contract

    def load(self, path: str | Path) -> pd.DataFrame:
        frame = pd.read_parquet(path)
        missing = set(self.contract.required_columns).difference(frame.columns)
        if missing:
            raise PointInTimeViolation(f"{self.contract.name} missing columns: {sorted(missing)}")
        validate_unique(frame, list(self.contract.entity_keys), self.contract.name)
        frame = frame.rename(
            columns={
                self.contract.effective_column: "effective_at",
                self.contract.available_column: "available_at",
            }
        )
        ensure_pit_columns(
            frame,
            name=self.contract.name,
            allow_available_before_effective=self.contract.availability_may_precede_effective,
        )
        if self.contract.name == CONTRACTS["events"].name:
            announced = pd.to_datetime(frame["announced_at"], utc=True, errors="raise")
            available = pd.to_datetime(frame["available_at"], utc=True, errors="raise")
            invalid = available < announced
            if invalid.any():
                raise PointInTimeViolation("event available_at precedes announced_at")
        return frame

    @staticmethod
    def as_of(frame: pd.DataFrame, decision_time: str | pd.Timestamp) -> pd.DataFrame:
        timestamp = pd.Timestamp(decision_time)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        available = pd.to_datetime(frame["available_at"], utc=True)
        return frame.loc[available <= timestamp].copy()


class CRSPAdapter(ParquetPITAdapter):
    def __init__(self) -> None:
        super().__init__(CONTRACTS["crsp"])


class CompustatPITAdapter(ParquetPITAdapter):
    def __init__(self) -> None:
        super().__init__(CONTRACTS["compustat_pit"])


class IBESAdapter(ParquetPITAdapter):
    def __init__(self) -> None:
        super().__init__(CONTRACTS["ibes"])


class TAQAdapter(ParquetPITAdapter):
    def __init__(self) -> None:
        super().__init__(CONTRACTS["taq"])


class OPRAOptionsAdapter(ParquetPITAdapter):
    def __init__(self) -> None:
        super().__init__(CONTRACTS["options"])


class BorrowAdapter(ParquetPITAdapter):
    def __init__(self) -> None:
        super().__init__(CONTRACTS["borrow"])


class EventTapeAdapter(ParquetPITAdapter):
    def __init__(self) -> None:
        super().__init__(CONTRACTS["events"])
