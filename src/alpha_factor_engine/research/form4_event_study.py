"""Fail-closed primitives for diagnostic SEC Form 4 event studies.

The functions in this module separate five clocks that are easy to conflate:

* the filing's ``information_at`` timestamp;
* the first executable ``entry_at``;
* the last complete daily close available before entry;
* the exact entry and exit opens used for returns; and
* the pre-event window used only as a placebo diagnostic.

Outputs are labelled ``DIAGNOSTIC_ONLY`` when permanent security identifiers
or same-SIC controls are unavailable.  A diagnostic label is deliberately not
a research-validity verdict.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from alpha_factor_engine.contracts import PointInTimeViolation, validate_unique


class Form4EventStudyError(ValueError):
    """Raised when an event-study input cannot be used without repair."""


RESEARCH_CANDIDATE = "RESEARCH_CANDIDATE"
DIAGNOSTIC_ONLY = "DIAGNOSTIC_ONLY"
PRICE_BASES = frozenset(
    {
        "split_dividend_adjusted_ohlc",
        "raw_ohlc_with_corporate_action_ledger",
    }
)


def _valid_sha256(value: object) -> bool:
    return bool(
        pd.notna(value)
        and re.fullmatch(r"[0-9a-fA-F]{64}", str(value).strip())
    )


def _digest_value(value: object) -> object:
    """Return a deterministic, JSON-safe representation of a scalar."""

    if value is None or value is pd.NA:
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            return None
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert("UTC")
        return {"timestamp": timestamp.isoformat()}
    if isinstance(value, date):
        return {"date": value.isoformat()}
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def canonical_frame_sha256(
    frame: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> str:
    """Hash a canonical tabular snapshot rather than trusting a hash-shaped string.

    Row order is intentionally ignored. Column order is explicit and retained.
    Callers can use this helper to declare the digest that the primitives then
    recompute from the supplied frame.
    """

    selected = list(frame.columns if columns is None else columns)
    missing = set(selected).difference(frame.columns)
    if missing:
        raise Form4EventStudyError(
            f"canonical frame missing columns: {sorted(missing)}"
        )
    records = [
        [_digest_value(value) for value in row]
        for row in frame.loc[:, selected].itertuples(index=False, name=None)
    ]
    records.sort(
        key=lambda row: json.dumps(
            row, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
    )
    payload = json.dumps(
        {"columns": selected, "records": records},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def exchange_calendar_sha256(
    exchange_sessions: Sequence[str | pd.Timestamp] | pd.Series | pd.DatetimeIndex,
) -> str:
    """Hash validated regular-open timestamps from an exchange snapshot."""

    opens = _prepare_exchange_opens(exchange_sessions)
    payload = "\n".join(value.isoformat() for value in opens).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _verify_digest(actual: str, declared: str | None, *, name: str) -> bool:
    if declared is None or not str(declared).strip():
        return False
    if not _valid_sha256(declared) or str(declared).lower() != actual.lower():
        raise Form4EventStudyError(f"{name} digest mismatch")
    return True


def _content_sha256(content: bytes | bytearray | memoryview | str | Path) -> str:
    if isinstance(content, (bytes, bytearray, memoryview)):
        payload = bytes(content)
    else:
        path = Path(content)
        if not path.is_file():
            raise Form4EventStudyError(f"evidence path is not a file: {path}")
        payload = path.read_bytes()
    return hashlib.sha256(payload).hexdigest()


def _verify_content(
    content: bytes | bytearray | memoryview | str | Path | None,
    declared: str | None,
    *,
    name: str,
) -> bool:
    if content is None:
        if declared is not None and str(declared).strip():
            raise Form4EventStudyError(
                f"{name} digest supplied without the evidence content"
            )
        return False
    return _verify_digest(_content_sha256(content), declared, name=name)


def _price_basis(value: str | None) -> str:
    if value is None or not str(value).strip():
        return "UNSPECIFIED"
    normalized = str(value).strip()
    if normalized not in PRICE_BASES:
        raise Form4EventStudyError(
            f"price_basis must be one of {sorted(PRICE_BASES)}; got {normalized!r}"
        )
    return normalized


def _require(frame: pd.DataFrame, columns: Sequence[str], *, name: str) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise Form4EventStudyError(f"{name} missing columns: {sorted(missing)}")


def _utc(values: pd.Series, *, name: str, allow_missing: bool = False) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    invalid = parsed.isna()
    if allow_missing:
        present = values.notna() & values.astype("string").str.strip().ne("")
        invalid &= present
    if invalid.any():
        raise PointInTimeViolation(
            f"{name} contains missing or invalid timestamps: "
            f"{values.loc[invalid].head(5).tolist()}"
        )
    return parsed


def _number(
    values: pd.Series,
    *,
    name: str,
    allow_missing: bool = False,
) -> pd.Series:
    parsed = pd.to_numeric(values, errors="coerce")
    present = values.notna() & values.astype("string").str.strip().ne("")
    invalid = present & (parsed.isna() | ~np.isfinite(parsed))
    if not allow_missing:
        invalid |= ~present
    if invalid.any():
        raise Form4EventStudyError(
            f"{name} contains missing or non-finite values: "
            f"{values.loc[invalid].head(5).tolist()}"
        )
    return parsed.astype(float)


def _boolean(values: pd.Series, *, name: str) -> pd.Series:
    text = values.astype("string").str.strip().str.lower()
    output = pd.Series(pd.NA, index=values.index, dtype="boolean")
    output.loc[text.isin({"1", "true", "t", "yes", "y"})] = True
    output.loc[text.isin({"0", "false", "f", "no", "n"})] = False
    invalid = values.isna() | output.isna()
    if invalid.any():
        raise Form4EventStudyError(
            f"{name} contains missing or invalid booleans: "
            f"{values.loc[invalid].head(5).tolist()}"
        )
    return output


def _ticker(values: pd.Series, *, name: str) -> pd.Series:
    output = values.astype("string").str.strip().str.upper()
    invalid = values.isna() | output.eq("")
    if invalid.any():
        raise Form4EventStudyError(
            f"{name} contains missing tickers: {values.loc[invalid].head(5).tolist()}"
        )
    return output


def _security_ids(
    values: pd.Series, *, name: str, allow_missing: bool = True
) -> pd.Series:
    output = values.astype("string").str.strip().str.upper()
    missing = values.isna() | output.eq("")
    if missing.any() and not allow_missing:
        raise Form4EventStudyError(
            f"{name} contains missing security identifiers: "
            f"{values.loc[missing].head(5).tolist()}"
        )
    output.loc[missing] = pd.NA
    return output


def _ciks(values: pd.Series, *, name: str, allow_missing: bool = True) -> pd.Series:
    output = values.astype("string").str.strip()
    missing = values.isna() | output.eq("")
    invalid = ~missing & ~output.str.fullmatch(r"\d+").fillna(False)
    if invalid.any():
        raise Form4EventStudyError(
            f"{name} contains non-digit CIK values: {values.loc[invalid].head(5).tolist()}"
        )
    if missing.any() and not allow_missing:
        raise Form4EventStudyError(
            f"{name} contains missing CIK values: {values.loc[missing].head(5).tolist()}"
        )
    output.loc[missing] = pd.NA
    output.loc[~missing] = output.loc[~missing].str.lstrip("0").replace("", "0").str.zfill(10)
    return output


def _session_date(value: object) -> date | None:
    if value is None or pd.isna(value):
        return None
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("America/New_York")
    return timestamp.date()


def _session_dates(values: pd.Series, *, name: str) -> pd.Series:
    output = values.map(_session_date)
    if output.isna().any():
        raise Form4EventStudyError(
            f"{name} contains missing or invalid sessions: "
            f"{values.loc[output.isna()].head(5).tolist()}"
        )
    return output


def _end_of_session_day_utc(session_dates: pd.Series) -> pd.Series:
    local = pd.DatetimeIndex(pd.to_datetime(session_dates.astype(str))).tz_localize(
        "America/New_York"
    )
    # Calendar-day offset preserves local midnight through DST transitions.
    end = local + pd.offsets.DateOffset(days=1) - pd.Timedelta(nanoseconds=1)
    return pd.Series(end.tz_convert("UTC"), index=session_dates.index)


def _scope(existing: object, reasons: Sequence[str]) -> tuple[str, str]:
    reason_set = {str(reason) for reason in reasons if str(reason).strip()}
    if pd.notna(existing) and str(existing).strip() == DIAGNOSTIC_ONLY:
        reason_set.add("upstream_diagnostic_only")
    if pd.isna(existing) or str(existing).strip() != RESEARCH_CANDIDATE:
        reason_set.add("upstream_research_scope_not_established")
    # These reusable primitives cannot verify a runner's frozen manifests,
    # source files, delisting accounting, or execution ledger end to end.
    # Promotion is therefore reserved for a separate manifest-verifying runner.
    reason_set.add("primitive_requires_manifest_verified_runner_promotion")
    return DIAGNOSTIC_ONLY, "|".join(sorted(reason_set))


def _downgrade_scope(row: dict[str, object], *reasons: str) -> None:
    existing_reasons = {
        token
        for token in str(row.get("scope_reasons", "")).split("|")
        if token and token != "nan"
    }
    existing_reasons.update(reason for reason in reasons if reason)
    row["research_scope"] = DIAGNOSTIC_ONLY
    row["scope_reasons"] = "|".join(sorted(existing_reasons))


def _has_identifier(row: pd.Series | dict[str, Any], column: str | None) -> bool:
    if not column or column not in row:
        return False
    value = row[column]
    return pd.notna(value) and str(value).strip() != ""


def _prepare_exchange_opens(
    exchange_sessions: Sequence[str | pd.Timestamp] | pd.Series | pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    opens = pd.DatetimeIndex(
        pd.to_datetime(pd.Series(list(exchange_sessions)), utc=True, errors="coerce")
    )
    if opens.empty or opens.isna().any():
        raise Form4EventStudyError("exchange_sessions must contain valid open timestamps")
    opens = opens.sort_values()
    if opens.has_duplicates:
        raise Form4EventStudyError("exchange_sessions contains duplicate opens")
    local = opens.tz_convert("America/New_York")
    regular_open = (
        (local.hour == 9)
        & (local.minute == 30)
        & (local.second == 0)
        & (local.microsecond == 0)
        & (local.nanosecond == 0)
    )
    if not regular_open.all():
        examples = [value.isoformat() for value in opens[~regular_open][:5]]
        raise PointInTimeViolation(
            "exchange_sessions must be DST-aware 09:30 America/New_York regular opens: "
            f"{examples}"
        )
    local_dates = pd.Index(local.date)
    if local_dates.has_duplicates:
        raise Form4EventStudyError("exchange_sessions has multiple regular opens per local date")
    return opens


def validate_event_entry_clocks(
    events: pd.DataFrame,
    exchange_sessions: Sequence[str | pd.Timestamp] | pd.Series | pd.DatetimeIndex | None,
    *,
    calendar_authoritative: bool = False,
    calendar_provenance_sha256: str | None = None,
    id_column: str = "event_id",
    information_column: str = "information_at",
    entry_column: str = "entry_at",
) -> pd.DataFrame:
    """Verify entry is the first independent regular open strictly after filing info.

    With no authoritative calendar, syntactically valid 09:30 entries remain
    available for diagnostics, but ``clock_verified`` is always false.
    """

    _require(events, [id_column, information_column, entry_column], name="events")
    frame = events.copy()
    validate_unique(frame, [id_column], "entry-clock events")
    frame[information_column] = _utc(
        frame[information_column], name=f"events.{information_column}"
    )
    frame[entry_column] = _utc(frame[entry_column], name=f"events.{entry_column}")
    if (~frame[information_column].lt(frame[entry_column])).any():
        raise PointInTimeViolation("entry clock requires information_at < entry_at")
    local_entry = frame[entry_column].dt.tz_convert("America/New_York")
    syntactic_regular_open = (
        local_entry.dt.hour.eq(9)
        & local_entry.dt.minute.eq(30)
        & local_entry.dt.second.eq(0)
        & local_entry.dt.microsecond.eq(0)
        & local_entry.dt.nanosecond.eq(0)
    )
    if (~syntactic_regular_open).any():
        raise PointInTimeViolation(
            "entry_at must be a DST-aware 09:30 America/New_York regular open"
        )

    if exchange_sessions is None:
        frame["expected_entry_at"] = pd.NaT
        frame["clock_verified"] = False
        frame["clock_status"] = "missing_independent_exchange_calendar"
        scopes = [
            _scope(value, ["missing_authoritative_exchange_calendar"])
            for value in frame.get(
                "research_scope", pd.Series(pd.NA, index=frame.index)
            )
        ]
    else:
        opens = _prepare_exchange_opens(exchange_sessions)
        actual_calendar_sha256 = exchange_calendar_sha256(opens)
        calendar_digest_verified = _verify_digest(
            actual_calendar_sha256,
            calendar_provenance_sha256,
            name="exchange calendar snapshot",
        )
        positions = opens.searchsorted(frame[information_column], side="right")
        if (positions >= len(opens)).any():
            raise PointInTimeViolation("exchange calendar has no open after an event")
        expected = pd.Series(opens.take(positions), index=frame.index)
        mismatch = ~frame[entry_column].eq(expected)
        if mismatch.any():
            examples = frame.loc[mismatch, [id_column, information_column, entry_column]].copy()
            examples["expected_entry_at"] = expected.loc[mismatch]
            raise PointInTimeViolation(
                "entry_at is not the first exchange open strictly after information_at: "
                f"{examples.head(5).to_dict('records')}"
            )
        frame["expected_entry_at"] = expected
        calendar_lineage_verified = bool(
            calendar_authoritative and calendar_digest_verified
        )
        frame["clock_verified"] = calendar_lineage_verified
        if calendar_lineage_verified:
            frame["clock_status"] = "verified_authoritative_calendar"
            reasons = []
        elif calendar_authoritative:
            frame["clock_status"] = "authoritative_calendar_missing_provenance_hash"
            reasons = ["missing_calendar_provenance_hash"]
        else:
            frame["clock_status"] = "matched_non_authoritative_calendar"
            reasons = ["exchange_calendar_not_authoritative"]
        scopes = [
            _scope(value, reasons)
            for value in frame.get(
                "research_scope", pd.Series(pd.NA, index=frame.index)
            )
        ]
    frame["research_scope"] = [item[0] for item in scopes]
    frame["scope_reasons"] = [item[1] for item in scopes]
    frame["calendar_provenance_sha256"] = calendar_provenance_sha256 or ""
    frame["calendar_snapshot_sha256"] = (
        exchange_calendar_sha256(exchange_sessions)
        if exchange_sessions is not None
        else ""
    )
    return frame


def _prepare_daily_features(
    daily_prices: pd.DataFrame,
    *,
    ticker_column: str,
    session_column: str,
    close_column: str,
    volume_column: str,
    available_column: str | None,
    permanent_id_column: str | None,
) -> pd.DataFrame:
    _require(
        daily_prices,
        [ticker_column, session_column, close_column, volume_column],
        name="daily prices",
    )
    selected_columns = [ticker_column, session_column, close_column, volume_column]
    if available_column:
        selected_columns.append(available_column)
    if permanent_id_column and permanent_id_column in daily_prices:
        selected_columns.append(permanent_id_column)
    frame = daily_prices[selected_columns].copy()
    frame["_ticker"] = _ticker(frame[ticker_column], name=f"daily prices.{ticker_column}")
    frame["_session"] = _session_dates(
        frame[session_column], name=f"daily prices.{session_column}"
    )
    frame["_close"] = _number(frame[close_column], name=f"daily prices.{close_column}")
    frame["_volume"] = _number(frame[volume_column], name=f"daily prices.{volume_column}")
    if frame["_close"].le(0).any():
        raise Form4EventStudyError("daily prices contain non-positive closes")
    if frame["_volume"].lt(0).any():
        raise Form4EventStudyError("daily prices contain negative volume")
    validate_unique(frame, ["_ticker", "_session"], "daily prices")
    if permanent_id_column and permanent_id_column in frame:
        frame["_permanent_id"] = _security_ids(
            frame[permanent_id_column],
            name=f"daily prices.{permanent_id_column}",
        )
        identified = frame.loc[frame["_permanent_id"].notna()]
        validate_unique(
            identified,
            ["_permanent_id", "_session"],
            "daily prices by permanent security id",
        )
    if available_column:
        frame["_available_at"] = _utc(
            frame[available_column], name=f"daily prices.{available_column}"
        )
        frame["_availability_basis"] = "source_available_at"
    else:
        frame["_available_at"] = _end_of_session_day_utc(frame["_session"])
        frame["_availability_basis"] = "session_date_end_of_day_conservative"
    return frame.sort_values(["_ticker", "_session"], kind="mergesort").reset_index(
        drop=True
    )


def build_prior_close_eligibility_snapshot(
    decisions: pd.DataFrame,
    daily_prices: pd.DataFrame,
    *,
    exchange_sessions: Sequence[str | pd.Timestamp] | pd.Series | pd.DatetimeIndex | None = None,
    calendar_authoritative: bool = False,
    calendar_provenance_sha256: str | None = None,
    id_column: str = "event_id",
    ticker_column: str = "ticker",
    information_column: str = "information_at",
    entry_column: str = "entry_at",
    session_column: str = "date",
    close_column: str = "close",
    volume_column: str = "volume",
    price_available_column: str | None = None,
    permanent_id_column: str | None = "security_id",
    security_master_known_column: str = "security_master_known_at",
    security_master_effective_column: str = "security_master_effective_at",
    security_master_clock_complete: bool = False,
    adv_window: int = 63,
    volatility_window: int = 63,
    minimum_history_sessions: int | None = None,
    minimum_price: float = 5.0,
    minimum_adv_dollars: float = 20_000_000.0,
    maximum_staleness_calendar_days: int = 7,
    price_basis: str | None = None,
    price_provenance_sha256: str | None = None,
    corporate_action_complete: bool = False,
    corporate_action_provenance_sha256: str | None = None,
    corporate_action_evidence: bytes | bytearray | memoryview | str | Path | None = None,
    delisting_complete: bool = False,
    delisting_provenance_sha256: str | None = None,
    delisting_evidence: bytes | bytearray | memoryview | str | Path | None = None,
    delisting_payoff_included: bool = False,
) -> pd.DataFrame:
    """Attach entry eligibility built from prior sessions known by filing time.

    Every ADV and volatility input must be available by ``information_at``;
    a same-day close published between the filing and entry is excluded. When
    the source has no timestamp column, a session is conservatively
    treated as available at 23:59:59.999999999 America/New_York.
    """

    if adv_window <= 0:
        raise ValueError("adv_window must be positive")
    if volatility_window < 2:
        raise ValueError("volatility_window must be at least 2")
    if minimum_price <= 0 or minimum_adv_dollars < 0:
        raise ValueError("minimum_price must be positive and minimum_adv_dollars non-negative")
    if maximum_staleness_calendar_days < 1:
        raise ValueError("maximum_staleness_calendar_days must be positive")
    normalized_price_basis = _price_basis(price_basis)
    price_digest_verified = _verify_digest(
        canonical_frame_sha256(daily_prices),
        price_provenance_sha256,
        name="daily price snapshot",
    )
    corporate_action_evidence_verified = _verify_content(
        corporate_action_evidence,
        corporate_action_provenance_sha256,
        name="corporate-action evidence",
    )
    delisting_evidence_verified = _verify_content(
        delisting_evidence,
        delisting_provenance_sha256,
        name="delisting evidence",
    )
    required_history = max(adv_window, volatility_window + 1)
    if minimum_history_sessions is not None:
        if minimum_history_sessions <= 0:
            raise ValueError("minimum_history_sessions must be positive")
        required_history = max(required_history, minimum_history_sessions)

    _require(
        decisions,
        [id_column, ticker_column, information_column, entry_column],
        name="decisions",
    )
    output = decisions.copy()
    output = validate_event_entry_clocks(
        output,
        exchange_sessions,
        calendar_authoritative=calendar_authoritative,
        calendar_provenance_sha256=calendar_provenance_sha256,
        id_column=id_column,
        information_column=information_column,
        entry_column=entry_column,
    )
    output[ticker_column] = _ticker(output[ticker_column], name=f"decisions.{ticker_column}")
    if permanent_id_column and permanent_id_column in output:
        output[permanent_id_column] = _security_ids(
            output[permanent_id_column], name=f"decisions.{permanent_id_column}"
        )
    output[information_column] = _utc(
        output[information_column], name=f"decisions.{information_column}"
    )
    output[entry_column] = _utc(output[entry_column], name=f"decisions.{entry_column}")
    security_clock_columns_present = all(
        column in output
        for column in [security_master_known_column, security_master_effective_column]
    )
    if security_clock_columns_present:
        output[security_master_known_column] = _utc(
            output[security_master_known_column],
            name=f"decisions.{security_master_known_column}",
            allow_missing=True,
        )
        output[security_master_effective_column] = _utc(
            output[security_master_effective_column],
            name=f"decisions.{security_master_effective_column}",
            allow_missing=True,
        )
    if (~output[information_column].lt(output[entry_column])).any():
        raise PointInTimeViolation("eligibility decisions require information_at < entry_at")

    prices = _prepare_daily_features(
        daily_prices,
        ticker_column=ticker_column,
        session_column=session_column,
        close_column=close_column,
        volume_column=volume_column,
        available_column=price_available_column,
        permanent_id_column=permanent_id_column,
    )
    by_ticker = {ticker: group for ticker, group in prices.groupby("_ticker", sort=False)}
    calendar_dates = (
        list(
            _prepare_exchange_opens(exchange_sessions)
            .tz_convert("America/New_York")
            .date
        )
        if exchange_sessions is not None
        else []
    )
    calendar_positions = {
        session: position for position, session in enumerate(calendar_dates)
    }
    by_permanent_id = (
        {
            identifier: group
            for identifier, group in prices.loc[prices["_permanent_id"].notna()].groupby(
                "_permanent_id", sort=False
            )
        }
        if "_permanent_id" in prices
        else {}
    )
    additions: list[dict[str, object]] = []
    for _, decision in output.iterrows():
        entry_at = decision[entry_column]
        entry_session = entry_at.tz_convert("America/New_York").date()
        reasons: list[str] = [
            token
            for token in str(decision.get("scope_reasons", "")).split("|")
            if token and token != "nan"
        ]
        decision_has_permanent_id = _has_identifier(decision, permanent_id_column)
        if decision_has_permanent_id and "_permanent_id" in prices:
            permanent_value = str(decision[permanent_id_column]).strip().upper()
            history = by_permanent_id.get(permanent_value)
            price_identity_basis = "permanent_security_id"
            if history is None:
                reasons.append("permanent_id_price_history_not_found")
        else:
            history = by_ticker.get(decision[ticker_column])
            price_identity_basis = "ticker_diagnostic_fallback"
            reasons.append("price_history_not_keyed_by_permanent_id")
        if history is None:
            eligible_history = prices.iloc[0:0]
        else:
            eligible_history = history.loc[
                history["_session"].lt(entry_session)
                & history["_available_at"].le(decision[information_column])
            ]
        if not decision_has_permanent_id:
            reasons.append("missing_permanent_security_id")
        security_clock_failure = ""
        if not security_master_clock_complete or not security_clock_columns_present:
            reasons.append("security_master_clock_not_proven_complete")
        if security_clock_columns_present:
            if pd.isna(decision[security_master_known_column]) or pd.isna(
                decision[security_master_effective_column]
            ):
                reasons.append("missing_event_security_master_clock")
                if security_master_clock_complete:
                    security_clock_failure = "missing_security_master_clock"
            elif decision[security_master_known_column] > decision[information_column]:
                security_clock_failure = "security_master_known_after_information"
            elif decision[security_master_effective_column] > decision[entry_column]:
                security_clock_failure = "security_master_effective_after_entry"
        if price_available_column is None:
            reasons.append("missing_source_price_availability_timestamp")
        if normalized_price_basis == "UNSPECIFIED":
            reasons.append("missing_explicit_price_basis")
        if not price_digest_verified:
            reasons.append("missing_price_provenance_hash")
        if not corporate_action_complete or not corporate_action_evidence_verified:
            reasons.append("corporate_action_history_not_proven_complete")
        if not delisting_complete or not delisting_evidence_verified:
            reasons.append("delisting_history_not_proven_complete")
        if not delisting_payoff_included:
            reasons.append("delisting_payoff_not_accounted")
        else:
            reasons.append("delisting_payoff_not_verified_in_return_ledger")
        scope, scope_reasons = _scope(decision.get("research_scope"), reasons)

        row: dict[str, object] = {
            "entry_session": entry_session,
            "prior_close": np.nan,
            "adv_dollars": np.nan,
            "log_adv_dollars": np.nan,
            "realized_volatility": np.nan,
            "history_sessions": int(len(eligible_history)),
            "feature_window_start_session": pd.NaT,
            "feature_max_session": pd.NaT,
            "feature_available_at": pd.NaT,
            "price_availability_basis": "",
            "price_identity_basis": price_identity_basis,
            "security_master_clock_status": (
                security_clock_failure
                or (
                    "verified"
                    if security_master_clock_complete and security_clock_columns_present
                    else "not_proven_complete"
                )
            ),
            "price_basis": normalized_price_basis,
            "price_provenance_sha256": price_provenance_sha256 or "",
            "price_snapshot_digest_verified": bool(price_digest_verified),
            "corporate_action_complete": bool(
                corporate_action_complete and corporate_action_evidence_verified
            ),
            "corporate_action_evidence_verified": bool(
                corporate_action_evidence_verified
            ),
            "corporate_action_provenance_sha256": (
                corporate_action_provenance_sha256 or ""
            ),
            "delisting_complete": bool(
                delisting_complete and delisting_evidence_verified
            ),
            "delisting_evidence_verified": bool(delisting_evidence_verified),
            "delisting_provenance_sha256": delisting_provenance_sha256 or "",
            "delisting_payoff_included": bool(delisting_payoff_included),
            "eligible": False,
            "eligibility_status": "missing_ticker_history",
            "eligibility_rule": (
                f"prior_close>={minimum_price:g};adv{adv_window}>="
                f"{minimum_adv_dollars:g};history>={required_history};"
                f"staleness<={maximum_staleness_calendar_days}d"
            ),
            "adv_window_sessions": adv_window,
            "volatility_window_sessions": volatility_window,
            "volatility_annualization": 252,
            "research_scope": scope,
            "scope_reasons": scope_reasons,
        }
        if security_clock_failure:
            row["eligibility_status"] = security_clock_failure
            additions.append(row)
            continue
        if eligible_history.empty:
            additions.append(row)
            continue

        latest = eligible_history.iloc[-1]
        row["prior_close"] = float(latest["_close"])
        row["feature_max_session"] = latest["_session"]
        row["feature_available_at"] = eligible_history.tail(required_history)[
            "_available_at"
        ].max()
        row["price_availability_basis"] = latest["_availability_basis"]
        row["feature_window_start_session"] = eligible_history.iloc[
            -min(len(eligible_history), required_history)
        ]["_session"]
        staleness = (entry_session - latest["_session"]).days
        if len(eligible_history) < required_history:
            row["eligibility_status"] = "insufficient_history"
            additions.append(row)
            continue

        if calendar_dates:
            last_position = calendar_positions.get(latest["_session"])
            actual_sessions = eligible_history.tail(required_history)["_session"].tolist()
            if last_position is None or last_position + 1 < required_history:
                row["eligibility_status"] = "missing_exact_exchange_session_coverage"
                additions.append(row)
                continue
            expected_sessions = calendar_dates[
                last_position - required_history + 1 : last_position + 1
            ]
            if actual_sessions != expected_sessions:
                row["eligibility_status"] = "missing_exact_exchange_session_coverage"
                additions.append(row)
                continue

        adv_sample = eligible_history.tail(adv_window)
        adv = float((adv_sample["_close"] * adv_sample["_volume"]).mean())
        vol_prices = eligible_history.tail(volatility_window + 1)["_close"]
        feature_inputs = pd.concat(
            [adv_sample, eligible_history.tail(volatility_window + 1)]
        ).loc[lambda value: ~value.index.duplicated(keep="first")]
        feature_available_at = feature_inputs["_available_at"].max()
        if feature_available_at > decision[information_column]:
            raise PointInTimeViolation(
                "eligibility feature input became available after information_at"
            )
        row["feature_available_at"] = feature_available_at
        returns = vol_prices.pct_change(fill_method=None).dropna()
        volatility = float(returns.std(ddof=1) * np.sqrt(252.0))
        if not np.isfinite(adv) or not np.isfinite(volatility):
            row["eligibility_status"] = "nonfinite_derived_feature"
            additions.append(row)
            continue
        row["adv_dollars"] = adv
        row["log_adv_dollars"] = float(np.log(adv)) if adv > 0 else np.nan
        row["realized_volatility"] = volatility
        if staleness > maximum_staleness_calendar_days:
            row["eligibility_status"] = "stale_prior_close"
        elif row["prior_close"] < minimum_price:
            row["eligibility_status"] = "below_minimum_price"
        elif adv < minimum_adv_dollars:
            row["eligibility_status"] = "below_minimum_adv"
        else:
            row["eligible"] = True
            row["eligibility_status"] = "eligible"
        additions.append(row)

    addition_frame = pd.DataFrame(additions, index=output.index)
    for column in addition_frame:
        output[column] = addition_frame[column]
    return output


def _prepare_open_prices(
    daily_prices: pd.DataFrame,
    *,
    ticker_column: str,
    session_column: str,
    open_column: str,
    permanent_id_column: str | None,
) -> pd.DataFrame:
    _require(daily_prices, [ticker_column, session_column, open_column], name="daily prices")
    selected_columns = [ticker_column, session_column, open_column]
    if permanent_id_column and permanent_id_column in daily_prices:
        selected_columns.append(permanent_id_column)
    frame = daily_prices[selected_columns].copy()
    frame["_ticker"] = _ticker(frame[ticker_column], name=f"daily prices.{ticker_column}")
    frame["_session"] = _session_dates(
        frame[session_column], name=f"daily prices.{session_column}"
    )
    frame["_open"] = _number(
        frame[open_column], name=f"daily prices.{open_column}", allow_missing=True
    )
    if frame["_open"].dropna().le(0).any():
        raise Form4EventStudyError("daily prices contain non-positive opens")
    validate_unique(frame, ["_ticker", "_session"], "daily open prices")
    if permanent_id_column and permanent_id_column in frame:
        frame["_permanent_id"] = _security_ids(
            frame[permanent_id_column],
            name=f"daily prices.{permanent_id_column}",
        )
        validate_unique(
            frame.loc[frame["_permanent_id"].notna()],
            ["_permanent_id", "_session"],
            "daily open prices by permanent security id",
        )
    return frame.sort_values(["_ticker", "_session"], kind="mergesort").reset_index(
        drop=True
    )


def _event_frame(
    events: pd.DataFrame,
    *,
    id_column: str,
    ticker_column: str,
    information_column: str,
    entry_column: str,
    permanent_id_column: str | None = "security_id",
    issuer_column: str = "issuer_cik",
) -> pd.DataFrame:
    _require(
        events,
        [id_column, ticker_column, information_column, entry_column],
        name="events",
    )
    frame = events.copy()
    validate_unique(frame, [id_column], "events")
    frame[ticker_column] = _ticker(frame[ticker_column], name=f"events.{ticker_column}")
    if permanent_id_column and permanent_id_column in frame:
        frame[permanent_id_column] = _security_ids(
            frame[permanent_id_column], name=f"events.{permanent_id_column}"
        )
    if issuer_column in frame:
        frame[issuer_column] = _ciks(frame[issuer_column], name=f"events.{issuer_column}")
    frame[information_column] = _utc(
        frame[information_column], name=f"events.{information_column}"
    )
    frame[entry_column] = _utc(frame[entry_column], name=f"events.{entry_column}")
    if (~frame[information_column].lt(frame[entry_column])).any():
        raise PointInTimeViolation("event returns require information_at < entry_at")
    return frame


def _open_lookups(
    prices: pd.DataFrame, benchmark_ticker: str
) -> tuple[dict[tuple[str, date], float], dict[tuple[str, date], float], set[str]]:
    benchmark = benchmark_ticker.strip().upper()
    benchmark_rows = prices.loc[prices["_ticker"].eq(benchmark)]
    if benchmark_rows.empty:
        raise Form4EventStudyError(f"benchmark {benchmark} is absent from daily prices")
    ticker_lookup = {
        (ticker, session): float(open_price)
        for ticker, session, open_price in prices.loc[
            prices["_open"].notna(), ["_ticker", "_session", "_open"]
        ].itertuples(index=False, name=None)
    }
    if "_permanent_id" in prices:
        permanent_rows = prices.loc[
            prices["_open"].notna() & prices["_permanent_id"].notna(),
            ["_permanent_id", "_session", "_open"],
        ]
        permanent_lookup = {
            (str(identifier), session): float(open_price)
            for identifier, session, open_price in permanent_rows.itertuples(
                index=False, name=None
            )
        }
        permanent_ids = set(prices["_permanent_id"].dropna().astype(str))
    else:
        permanent_lookup = {}
        permanent_ids = set()
    return ticker_lookup, permanent_lookup, permanent_ids


def _event_price_identity(
    event: pd.Series,
    *,
    permanent_id_column: str | None,
    price_permanent_ids: set[str],
    prices_have_permanent_ids: bool,
    ticker_column: str,
) -> tuple[str, str, list[str], bool]:
    if _has_identifier(event, permanent_id_column):
        identifier = str(event[permanent_id_column]).strip()
        if prices_have_permanent_ids:
            reasons = [] if identifier in price_permanent_ids else [
                "permanent_id_price_history_not_found"
            ]
            return identifier, "permanent_security_id", reasons, True
        return (
            str(event[ticker_column]),
            "ticker_diagnostic_fallback",
            ["price_history_not_keyed_by_permanent_id"],
            False,
        )
    return (
        str(event[ticker_column]),
        "ticker_diagnostic_fallback",
        ["missing_permanent_security_id", "price_history_not_keyed_by_permanent_id"],
        False,
    )


def compute_open_to_open_returns(
    events: pd.DataFrame,
    daily_prices: pd.DataFrame,
    *,
    exchange_sessions: Sequence[str | pd.Timestamp] | pd.Series | pd.DatetimeIndex | None = None,
    calendar_authoritative: bool = False,
    calendar_provenance_sha256: str | None = None,
    horizons: Sequence[int] = (5, 20, 60),
    benchmark_ticker: str = "SPY",
    one_way_cost_bps: float = 0.0,
    id_column: str = "event_id",
    ticker_column: str = "ticker",
    information_column: str = "information_at",
    entry_column: str = "entry_at",
    session_column: str = "date",
    open_column: str = "open",
    permanent_id_column: str | None = "security_id",
    issuer_column: str = "issuer_cik",
    require_clock_verified: bool = True,
    require_eligible: bool = True,
    require_overlap_selected: bool = True,
    price_basis: str | None = None,
    price_provenance_sha256: str | None = None,
    corporate_action_complete: bool = False,
    corporate_action_provenance_sha256: str | None = None,
    corporate_action_evidence: bytes | bytearray | memoryview | str | Path | None = None,
    delisting_complete: bool = False,
    delisting_provenance_sha256: str | None = None,
    delisting_evidence: bytes | bytearray | memoryview | str | Path | None = None,
    delisting_payoff_included: bool = False,
    delisting_payoffs: pd.DataFrame | None = None,
    delisting_payoff_provenance_sha256: str | None = None,
) -> pd.DataFrame:
    """Compute exact-calendar open-to-open event and SPY returns.

    The caller-supplied exchange calendar defines horizons; SPY is only the
    return benchmark. Missing SPY or security rows are reported and never
    compressed. Costs cover the event leg only. Market-adjusted output is a
    diagnostic subtraction, not executable hedged portfolio P&L.
    """

    normalized_horizons = sorted({int(value) for value in horizons})
    if not normalized_horizons or normalized_horizons[0] <= 0:
        raise ValueError("horizons must contain positive integers")
    if one_way_cost_bps < 0:
        raise ValueError("one_way_cost_bps must be non-negative")
    normalized_price_basis = _price_basis(price_basis)
    price_digest_verified = _verify_digest(
        canonical_frame_sha256(daily_prices),
        price_provenance_sha256,
        name="daily price snapshot",
    )
    corporate_action_evidence_verified = _verify_content(
        corporate_action_evidence,
        corporate_action_provenance_sha256,
        name="corporate-action evidence",
    )
    delisting_evidence_verified = _verify_content(
        delisting_evidence,
        delisting_provenance_sha256,
        name="delisting evidence",
    )
    payoff_digest_verified = False
    payoff_lookup: dict[tuple[object, int], float] = {}
    if delisting_payoffs is not None:
        payoff_columns = [
            id_column,
            "horizon_sessions",
            "event_total_return_including_delisting",
        ]
        _require(delisting_payoffs, payoff_columns, name="delisting payoff ledger")
        payoff_digest_verified = _verify_digest(
            canonical_frame_sha256(delisting_payoffs),
            delisting_payoff_provenance_sha256,
            name="delisting payoff ledger",
        )
        payoff_frame = delisting_payoffs.copy()
        payoff_frame["horizon_sessions"] = _number(
            payoff_frame["horizon_sessions"],
            name="delisting payoff ledger.horizon_sessions",
        )
        payoff_frame["event_total_return_including_delisting"] = _number(
            payoff_frame["event_total_return_including_delisting"],
            name="delisting payoff ledger.event_total_return_including_delisting",
        )
        validate_unique(
            payoff_frame,
            [id_column, "horizon_sessions"],
            "delisting payoff ledger",
        )
        payoff_lookup = {
            (event_id, int(horizon)): float(total_return)
            for event_id, horizon, total_return in payoff_frame[
                [
                    id_column,
                    "horizon_sessions",
                    "event_total_return_including_delisting",
                ]
            ].itertuples(index=False, name=None)
        }
    elif delisting_payoff_provenance_sha256:
        raise Form4EventStudyError(
            "delisting payoff ledger digest supplied without the ledger"
        )
    frame = _event_frame(
        events,
        id_column=id_column,
        ticker_column=ticker_column,
        information_column=information_column,
        entry_column=entry_column,
        permanent_id_column=permanent_id_column,
        issuer_column=issuer_column,
    )
    for gate_column in ["clock_verified", "eligible", "selected_after_overlap"]:
        if gate_column in frame:
            frame[gate_column] = _boolean(frame[gate_column], name=f"events.{gate_column}")
    prices = _prepare_open_prices(
        daily_prices,
        ticker_column=ticker_column,
        session_column=session_column,
        open_column=open_column,
        permanent_id_column=permanent_id_column,
    )
    ticker_lookup, permanent_lookup, price_permanent_ids = _open_lookups(
        prices, benchmark_ticker
    )
    if exchange_sessions is None:
        calendar_opens = pd.DatetimeIndex([])
        calendar_dates: list[date] = []
        positions: dict[date, int] = {}
    else:
        calendar_opens = _prepare_exchange_opens(exchange_sessions)
        calendar_digest_verified = _verify_digest(
            exchange_calendar_sha256(calendar_opens),
            calendar_provenance_sha256,
            name="exchange calendar snapshot",
        )
        calendar_dates = list(calendar_opens.tz_convert("America/New_York").date)
        positions = {session: position for position, session in enumerate(calendar_dates)}
    benchmark = benchmark_ticker.strip().upper()
    round_trip_cost = 2.0 * one_way_cost_bps / 10_000.0
    rows: list[dict[str, object]] = []
    for _, event in frame.iterrows():
        entry_session = event[entry_column].tz_convert("America/New_York").date()
        identity_value, identity_basis, reasons, use_permanent_id = _event_price_identity(
            event,
            permanent_id_column=permanent_id_column,
            price_permanent_ids=price_permanent_ids,
            prices_have_permanent_ids="_permanent_id" in prices,
            ticker_column=ticker_column,
        )
        if exchange_sessions is None:
            reasons.append("missing_authoritative_exchange_calendar")
        elif not calendar_authoritative:
            reasons.append("exchange_calendar_not_authoritative")
        elif not calendar_digest_verified:
            reasons.append("missing_calendar_provenance_hash")
        if normalized_price_basis == "UNSPECIFIED":
            reasons.append("missing_explicit_price_basis")
        if not price_digest_verified:
            reasons.append("missing_price_provenance_hash")
        if not corporate_action_complete or not corporate_action_evidence_verified:
            reasons.append("corporate_action_history_not_proven_complete")
        if not delisting_complete or not delisting_evidence_verified:
            reasons.append("delisting_history_not_proven_complete")
        payoff_accounting_verified = bool(
            delisting_payoff_included and payoff_digest_verified
        )
        if not payoff_accounting_verified:
            reasons.append("delisting_payoff_not_accounted")
        # Benchmark subtraction is intentionally never labelled executable hedge P&L.
        reasons.append("benchmark_is_nontraded_diagnostic_adjustment")
        scope, scope_reasons = _scope(event.get("research_scope"), reasons)
        for horizon in normalized_horizons:
            row: dict[str, object] = {
                id_column: event[id_column],
                ticker_column: event[ticker_column],
                "information_at": event[information_column],
                "entry_at": event[entry_column],
                "entry_session": entry_session,
                "horizon_sessions": horizon,
                "exit_session": pd.NaT,
                "exit_open_at": pd.NaT,
                "entry_open": np.nan,
                "exit_open": np.nan,
                "benchmark_entry_open": np.nan,
                "benchmark_exit_open": np.nan,
                "event_gross_return": np.nan,
                "event_net_return_after_event_cost_only": np.nan,
                "benchmark_return": np.nan,
                "market_adjusted_return_gross": np.nan,
                "market_adjusted_return_after_event_cost_only": np.nan,
                "event_one_way_cost_bps": float(one_way_cost_bps),
                "event_round_trip_cost_return": float(round_trip_cost),
                "benchmark_execution_cost": np.nan,
                "execution_cost_complete": False,
                "cost_scope": "event_leg_only_diagnostic_no_hedge_execution_cost",
                "return_status": "entry_not_exchange_session",
                "return_basis": "exact_independent_exchange_session_open_to_open",
                "return_interpretation": (
                    "diagnostic_market_adjustment_not_tradeable_hedge_pnl"
                ),
                "price_identity_basis": identity_basis,
                "price_basis": normalized_price_basis,
                "price_provenance_sha256": price_provenance_sha256 or "",
                "price_snapshot_digest_verified": bool(price_digest_verified),
                "corporate_action_complete": bool(
                    corporate_action_complete and corporate_action_evidence_verified
                ),
                "corporate_action_evidence_verified": bool(
                    corporate_action_evidence_verified
                ),
                "corporate_action_provenance_sha256": (
                    corporate_action_provenance_sha256 or ""
                ),
                "delisting_complete": bool(
                    delisting_complete and delisting_evidence_verified
                ),
                "delisting_evidence_verified": bool(delisting_evidence_verified),
                "delisting_provenance_sha256": delisting_provenance_sha256 or "",
                "delisting_payoff_included": bool(payoff_accounting_verified),
                "delisting_payoff_ledger_verified": bool(payoff_digest_verified),
                "delisting_payoff_applied": False,
                "calendar_authoritative": bool(calendar_authoritative),
                "calendar_provenance_sha256": calendar_provenance_sha256 or "",
                "research_scope": scope,
                "scope_reasons": scope_reasons,
            }
            if exchange_sessions is None:
                row["return_status"] = "blocked_missing_exchange_calendar"
                rows.append(row)
                continue
            if require_clock_verified and (
                "clock_verified" not in event or not bool(event["clock_verified"])
            ):
                row["return_status"] = "blocked_clock_not_verified"
                rows.append(row)
                continue
            if require_eligible and ("eligible" not in event or not bool(event["eligible"])):
                row["return_status"] = "blocked_event_not_eligible"
                rows.append(row)
                continue
            if require_overlap_selected and (
                "selected_after_overlap" not in event
                or not bool(event["selected_after_overlap"])
            ):
                row["return_status"] = "blocked_overlap_not_selected"
                rows.append(row)
                continue
            entry_position = positions.get(entry_session)
            if entry_position is None:
                rows.append(row)
                continue
            first_after_position = int(
                calendar_opens.searchsorted(event[information_column], side="right")
            )
            if first_after_position >= len(calendar_opens):
                row["return_status"] = "insufficient_exchange_calendar_after_information"
                rows.append(row)
                continue
            if event[entry_column] != calendar_opens[first_after_position]:
                raise PointInTimeViolation(
                    "event entry_at is not the first independent exchange open "
                    "strictly after information_at"
                )
            exit_position = entry_position + horizon
            if exit_position >= len(calendar_dates):
                row["return_status"] = "insufficient_future_exchange_sessions"
                rows.append(row)
                continue
            exit_session = calendar_dates[exit_position]
            expected_entry_at = calendar_opens[entry_position]
            if event[entry_column] != expected_entry_at:
                raise PointInTimeViolation(
                    "event entry_at does not equal its independent exchange-session open"
                )
            if event[information_column] >= expected_entry_at:
                raise PointInTimeViolation(
                    "event entry is not strictly after information_at"
                )
            row["exit_session"] = exit_session
            row["exit_open_at"] = calendar_opens[exit_position]
            event_lookup = permanent_lookup if use_permanent_id else ticker_lookup
            event_entry = event_lookup.get((identity_value, entry_session))
            event_exit = event_lookup.get((identity_value, exit_session))
            benchmark_entry = ticker_lookup.get((benchmark, entry_session))
            benchmark_exit = ticker_lookup.get((benchmark, exit_session))
            payoff_return = (
                payoff_lookup.get((event[id_column], horizon))
                if payoff_accounting_verified
                else None
            )
            if event_entry is None:
                row["return_status"] = "missing_entry_open"
                _downgrade_scope(row, "missing_exact_entry_price")
            elif benchmark_entry is None:
                row["return_status"] = "missing_benchmark_entry_open"
                _downgrade_scope(row, "missing_exact_benchmark_entry_price")
            elif benchmark_exit is None:
                row["return_status"] = "missing_benchmark_exit_open"
                _downgrade_scope(row, "missing_exact_benchmark_exit_price")
            elif event_exit is None and payoff_return is None:
                row["return_status"] = "missing_exact_exit_open"
                _downgrade_scope(
                    row,
                    "missing_exact_exit_price",
                    "missing_verified_delisting_payoff_for_event_horizon",
                )
            else:
                payoff_applied = event_exit is None
                gross = (
                    float(payoff_return)
                    if payoff_applied
                    else event_exit / event_entry - 1.0
                )
                benchmark_return = benchmark_exit / benchmark_entry - 1.0
                net = gross - round_trip_cost
                row.update(
                    {
                        "entry_open": event_entry,
                        "exit_open": event_exit,
                        "benchmark_entry_open": benchmark_entry,
                        "benchmark_exit_open": benchmark_exit,
                        "event_gross_return": gross,
                        "event_net_return_after_event_cost_only": net,
                        "benchmark_return": benchmark_return,
                        "market_adjusted_return_gross": gross - benchmark_return,
                        "market_adjusted_return_after_event_cost_only": net
                        - benchmark_return,
                        "return_status": (
                            "ok_delisting_payoff" if payoff_applied else "ok"
                        ),
                        "delisting_payoff_applied": payoff_applied,
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def compute_pre_event_placebo(
    events: pd.DataFrame,
    daily_prices: pd.DataFrame,
    *,
    exchange_sessions: Sequence[str | pd.Timestamp] | pd.Series | pd.DatetimeIndex | None = None,
    calendar_authoritative: bool = False,
    horizons: Sequence[int] = (5, 20, 60),
    gap_sessions: int = 1,
    benchmark_ticker: str = "SPY",
    id_column: str = "event_id",
    ticker_column: str = "ticker",
    information_column: str = "information_at",
    entry_column: str = "entry_at",
    session_column: str = "date",
    open_column: str = "open",
    permanent_id_column: str | None = "security_id",
    issuer_column: str = "issuer_cik",
    require_clock_verified: bool = True,
    price_basis: str | None = None,
    corporate_action_complete: bool = False,
) -> pd.DataFrame:
    """Measure pre-event open-to-open returns ending before the true entry.

    For horizon ``h`` the placebo spans ``[-gap-h, -gap]`` exchange sessions
    relative to the true entry.  It is a pre-trend diagnostic, not a simulated
    trade triggered by information that did not yet exist.
    """

    normalized_horizons = sorted({int(value) for value in horizons})
    if not normalized_horizons or normalized_horizons[0] <= 0:
        raise ValueError("horizons must contain positive integers")
    if gap_sessions < 1:
        raise ValueError("gap_sessions must be at least 1")
    normalized_price_basis = _price_basis(price_basis)
    frame = _event_frame(
        events,
        id_column=id_column,
        ticker_column=ticker_column,
        information_column=information_column,
        entry_column=entry_column,
        permanent_id_column=permanent_id_column,
        issuer_column=issuer_column,
    )
    if "clock_verified" in frame:
        frame["clock_verified"] = _boolean(
            frame["clock_verified"], name="events.clock_verified"
        )
    prices = _prepare_open_prices(
        daily_prices,
        ticker_column=ticker_column,
        session_column=session_column,
        open_column=open_column,
        permanent_id_column=permanent_id_column,
    )
    ticker_lookup, permanent_lookup, price_permanent_ids = _open_lookups(
        prices, benchmark_ticker
    )
    if exchange_sessions is None:
        calendar_opens = pd.DatetimeIndex([])
        calendar_dates: list[date] = []
        positions: dict[date, int] = {}
    else:
        calendar_opens = _prepare_exchange_opens(exchange_sessions)
        calendar_dates = list(calendar_opens.tz_convert("America/New_York").date)
        positions = {session: position for position, session in enumerate(calendar_dates)}
    benchmark = benchmark_ticker.strip().upper()
    rows: list[dict[str, object]] = []
    for _, event in frame.iterrows():
        true_entry = event[entry_column].tz_convert("America/New_York").date()
        identity_value, identity_basis, reasons, use_permanent_id = _event_price_identity(
            event,
            permanent_id_column=permanent_id_column,
            price_permanent_ids=price_permanent_ids,
            prices_have_permanent_ids="_permanent_id" in prices,
            ticker_column=ticker_column,
        )
        reasons.append("placebo_analysis_only")
        if exchange_sessions is None:
            reasons.append("missing_authoritative_exchange_calendar")
        elif not calendar_authoritative:
            reasons.append("exchange_calendar_not_authoritative")
        if normalized_price_basis == "UNSPECIFIED":
            reasons.append("missing_explicit_price_basis")
        # This diagnostic API has no content-bound corporate-action artifact.
        reasons.append("corporate_action_history_not_proven_complete")
        scope, scope_reasons = _scope(event.get("research_scope"), reasons)
        for horizon in normalized_horizons:
            row: dict[str, object] = {
                id_column: event[id_column],
                ticker_column: event[ticker_column],
                "true_entry_session": true_entry,
                "horizon_sessions": horizon,
                "placebo_start_session": pd.NaT,
                "placebo_end_session": pd.NaT,
                "placebo_start_open_at": pd.NaT,
                "placebo_end_open_at": pd.NaT,
                "placebo_return": np.nan,
                "benchmark_placebo_return": np.nan,
                "placebo_active_return": np.nan,
                "placebo_status": "entry_not_exchange_session",
                "placebo_design": f"open_to_open_-{gap_sessions + horizon}_to_-{gap_sessions}",
                "analysis_role": "pre_event_placebo_only",
                "price_identity_basis": identity_basis,
                "price_basis": normalized_price_basis,
                "corporate_action_complete": False,
                "calendar_authoritative": bool(calendar_authoritative),
                "research_scope": scope,
                "scope_reasons": scope_reasons,
            }
            if exchange_sessions is None:
                row["placebo_status"] = "blocked_missing_exchange_calendar"
                rows.append(row)
                continue
            if require_clock_verified and (
                "clock_verified" not in event or not bool(event["clock_verified"])
            ):
                row["placebo_status"] = "blocked_clock_not_verified"
                rows.append(row)
                continue
            entry_position = positions.get(true_entry)
            if entry_position is None:
                rows.append(row)
                continue
            end_position = entry_position - gap_sessions
            start_position = end_position - horizon
            if start_position < 0:
                row["placebo_status"] = "insufficient_pre_event_exchange_sessions"
                rows.append(row)
                continue
            first_after_position = int(
                calendar_opens.searchsorted(event[information_column], side="right")
            )
            if first_after_position >= len(calendar_opens):
                row["placebo_status"] = "insufficient_exchange_calendar_after_information"
                rows.append(row)
                continue
            if event[entry_column] != calendar_opens[first_after_position]:
                raise PointInTimeViolation(
                    "placebo event entry_at is not the first independent exchange open "
                    "strictly after information_at"
                )
            start_session = calendar_dates[start_position]
            end_session = calendar_dates[end_position]
            start_open_at = calendar_opens[start_position]
            end_open_at = calendar_opens[end_position]
            row["placebo_start_session"] = start_session
            row["placebo_end_session"] = end_session
            row["placebo_start_open_at"] = start_open_at
            row["placebo_end_open_at"] = end_open_at
            if end_open_at >= event[information_column]:
                row["placebo_status"] = "placebo_end_not_before_information"
                rows.append(row)
                continue
            event_lookup = permanent_lookup if use_permanent_id else ticker_lookup
            event_start = event_lookup.get((identity_value, start_session))
            event_end = event_lookup.get((identity_value, end_session))
            benchmark_start = ticker_lookup.get((benchmark, start_session))
            benchmark_end = ticker_lookup.get((benchmark, end_session))
            if event_start is None:
                row["placebo_status"] = "missing_placebo_start_open"
            elif event_end is None:
                row["placebo_status"] = "missing_placebo_exact_end_open"
            elif benchmark_start is None:
                row["placebo_status"] = "missing_benchmark_placebo_start_open"
            elif benchmark_end is None:
                row["placebo_status"] = "missing_benchmark_placebo_end_open"
            else:
                event_return = event_end / event_start - 1.0
                benchmark_return = benchmark_end / benchmark_start - 1.0
                row.update(
                    {
                        "placebo_return": event_return,
                        "benchmark_placebo_return": benchmark_return,
                        "placebo_active_return": event_return - benchmark_return,
                        "placebo_status": "ok",
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def suppress_causal_overlaps(
    events: pd.DataFrame,
    *,
    exchange_sessions: Sequence[str | pd.Timestamp] | pd.Series | pd.DatetimeIndex | None = None,
    calendar_authoritative: bool = False,
    calendar_provenance_sha256: str | None = None,
    horizon_artifact: pd.DataFrame | None = None,
    horizon_artifact_sha256: str | None = None,
    overlap_horizon_sessions: int = 20,
    id_column: str = "event_id",
    information_column: str = "information_at",
    entry_column: str = "entry_at",
    end_column: str = "exit_at",
    permanent_id_column: str | None = "security_id",
    issuer_column: str = "issuer_cik",
    ticker_column: str = "ticker",
) -> pd.DataFrame:
    """Keep the first event and suppress later signals while its position is open.

    A suppressed event never extends the active interval.  An event at the
    exact prior exit open is allowed because the old position can exit before
    the new one enters at that same open.
    """

    _require(
        events,
        [id_column, information_column, entry_column, ticker_column],
        name="events",
    )
    if overlap_horizon_sessions <= 0:
        raise ValueError("overlap_horizon_sessions must be positive")
    frame = events.copy()
    validate_unique(frame, [id_column], "overlap events")
    frame[information_column] = _utc(
        frame[information_column], name=f"events.{information_column}"
    )
    frame[entry_column] = _utc(frame[entry_column], name=f"events.{entry_column}")
    frame[ticker_column] = _ticker(frame[ticker_column], name=f"events.{ticker_column}")
    if permanent_id_column and permanent_id_column in frame:
        frame[permanent_id_column] = _security_ids(
            frame[permanent_id_column], name=f"events.{permanent_id_column}"
        )
    if issuer_column in frame:
        frame[issuer_column] = _ciks(frame[issuer_column], name=f"events.{issuer_column}")
    invalid = ~frame[information_column].lt(frame[entry_column])
    if invalid.any():
        raise PointInTimeViolation("overlap events require information_at < entry_at")

    artifact_verified = False
    artifact_block_reason = "missing_authoritative_exact_horizon_artifact"
    if horizon_artifact is not None and horizon_artifact_sha256:
        _verify_digest(
            canonical_frame_sha256(horizon_artifact),
            horizon_artifact_sha256,
            name="overlap horizon artifact",
        )
        if exchange_sessions is None:
            artifact_block_reason = "missing_independent_exchange_calendar"
        elif not calendar_authoritative:
            artifact_block_reason = "exchange_calendar_not_authoritative"
        else:
            calendar_opens = _prepare_exchange_opens(exchange_sessions)
            actual_calendar_sha256 = exchange_calendar_sha256(calendar_opens)
            calendar_digest_verified = _verify_digest(
                actual_calendar_sha256,
                calendar_provenance_sha256,
                name="exchange calendar snapshot",
            )
            if not calendar_digest_verified:
                artifact_block_reason = "missing_calendar_provenance_hash"
    if (
        horizon_artifact is not None
        and horizon_artifact_sha256
        and exchange_sessions is not None
        and calendar_authoritative
        and calendar_digest_verified
    ):
        required_artifact_columns = [
            id_column,
            "horizon_sessions",
            "entry_at",
            "exit_open_at",
            "return_basis",
            "calendar_authoritative",
            "calendar_provenance_sha256",
        ]
        _require(
            horizon_artifact,
            required_artifact_columns,
            name="overlap horizon artifact",
        )
        artifact = horizon_artifact.copy()
        artifact["horizon_sessions"] = _number(
            artifact["horizon_sessions"],
            name="overlap horizon artifact.horizon_sessions",
        )
        artifact = artifact.loc[
            artifact["horizon_sessions"].eq(float(overlap_horizon_sessions))
        ].copy()
        validate_unique(artifact, [id_column], "overlap horizon artifact")
        missing_events = set(frame[id_column]).difference(artifact[id_column])
        if missing_events:
            raise Form4EventStudyError(
                "overlap horizon artifact missing events: "
                f"{sorted(str(value) for value in missing_events)[:5]}"
            )
        artifact["entry_at"] = _utc(
            artifact["entry_at"], name="overlap horizon artifact.entry_at"
        )
        artifact["exit_open_at"] = _utc(
            artifact["exit_open_at"], name="overlap horizon artifact.exit_open_at"
        )
        artifact["calendar_authoritative"] = _boolean(
            artifact["calendar_authoritative"],
            name="overlap horizon artifact.calendar_authoritative",
        )
        valid_basis = artifact["return_basis"].astype("string").eq(
            "exact_independent_exchange_session_open_to_open"
        )
        valid_calendar_lineage = artifact["calendar_authoritative"].eq(True) & artifact[
            "calendar_provenance_sha256"
        ].astype("string").str.lower().eq(actual_calendar_sha256)
        if not valid_basis.all() or not valid_calendar_lineage.all():
            raise PointInTimeViolation(
                "overlap horizon artifact lacks exact authoritative calendar lineage"
            )
        artifact = artifact.set_index(id_column)
        expected_entry = frame[id_column].map(artifact["entry_at"])
        if (~frame[entry_column].eq(expected_entry)).any():
            raise PointInTimeViolation(
                "overlap horizon artifact entry_at does not match event entry_at"
            )
        entry_positions = calendar_opens.get_indexer(
            pd.DatetimeIndex(artifact["entry_at"])
        )
        if (entry_positions < 0).any():
            raise PointInTimeViolation(
                "overlap horizon artifact entry_at is not an exact exchange open"
            )
        expected_exit_positions = entry_positions + overlap_horizon_sessions
        if (expected_exit_positions >= len(calendar_opens)).any():
            raise PointInTimeViolation(
                "exchange calendar cannot prove the requested overlap horizon"
            )
        exact_expected_exits = pd.Series(
            calendar_opens.take(expected_exit_positions), index=artifact.index
        )
        if (~artifact["exit_open_at"].eq(exact_expected_exits)).any():
            raise PointInTimeViolation(
                "overlap horizon artifact exit_open_at is not exactly "
                "entry_at + horizon_sessions on the authoritative calendar"
            )
        first_after_positions = calendar_opens.searchsorted(
            frame[information_column], side="right"
        )
        if (first_after_positions >= len(calendar_opens)).any() or (
            frame[entry_column]
            != pd.Series(calendar_opens.take(first_after_positions), index=frame.index)
        ).any():
            raise PointInTimeViolation(
                "overlap event entry_at is not the first exchange open strictly "
                "after information_at"
            )
        frame[end_column] = frame[id_column].map(artifact["exit_open_at"])
        if (~frame[entry_column].lt(frame[end_column])).any():
            raise PointInTimeViolation(
                "overlap horizon artifact requires entry_at < exit_open_at"
            )
        artifact_verified = True
        artifact_block_reason = ""
    elif horizon_artifact is not None and not horizon_artifact_sha256:
        artifact_block_reason = "missing_horizon_artifact_provenance_hash"

    keys: list[str] = []
    bases: list[str] = []
    reasons_by_row: list[list[str]] = []
    for _, row in frame.iterrows():
        reasons: list[str] = []
        if issuer_column in frame and pd.notna(row.get(issuer_column)) and str(
            row.get(issuer_column)
        ).strip():
            key = f"issuer:{row[issuer_column]}"
            basis = "issuer_cik"
            if not _has_identifier(row, permanent_id_column):
                reasons.append("missing_permanent_security_id")
        elif _has_identifier(row, permanent_id_column):
            key = f"permanent:{row[permanent_id_column]}"
            basis = "permanent_security_id_fallback_missing_issuer"
            reasons.append("missing_issuer_cik")
        else:
            key = f"ticker:{row[ticker_column]}"
            basis = "ticker_fallback"
            reasons.extend(["missing_permanent_security_id", "missing_issuer_cik"])
        keys.append(key)
        bases.append(basis)
        reasons_by_row.append(reasons)
    frame["overlap_identity"] = keys
    frame["overlap_identity_basis"] = bases
    frame["_scope_reasons"] = reasons_by_row
    if not artifact_verified:
        scopes = [
            _scope(
                row.get("research_scope"),
                [*row["_scope_reasons"], artifact_block_reason],
            )
            for _, row in frame.iterrows()
        ]
        frame["overlap_status"] = "blocked_unverified_horizon"
        frame["selected_after_overlap"] = False
        frame["suppressed_by_event_id"] = pd.NA
        frame["active_position_end_at"] = pd.NaT
        frame["overlap_end_basis"] = "UNVERIFIED_NO_SELECTION"
        frame["horizon_artifact_sha256"] = horizon_artifact_sha256 or ""
        frame["research_scope"] = [item[0] for item in scopes]
        frame["scope_reasons"] = [item[1] for item in scopes]
        return frame.drop(columns=["_scope_reasons"]).reset_index(drop=True)
    frame["_input_order"] = np.arange(len(frame))
    frame = frame.sort_values(
        ["overlap_identity", entry_column, information_column, id_column],
        kind="mergesort",
    )

    additions: dict[object, dict[str, object]] = {}
    for _, group in frame.groupby("overlap_identity", sort=False):
        active_end: pd.Timestamp | None = None
        active_event: object | None = None
        for index, event in group.iterrows():
            if active_end is not None and event[entry_column] < active_end:
                status = "suppressed_active_position"
                selected = False
                suppressed_by = active_event
            else:
                status = "kept"
                selected = True
                suppressed_by = pd.NA
                active_end = event[end_column]
                active_event = event[id_column]
            scope, scope_reasons = _scope(
                event.get("research_scope"), event["_scope_reasons"]
            )
            additions[index] = {
                "overlap_status": status,
                "selected_after_overlap": selected,
                "suppressed_by_event_id": suppressed_by,
                "active_position_end_at": active_end,
                "overlap_end_basis": "authoritative_exact_horizon_artifact",
                "horizon_artifact_sha256": horizon_artifact_sha256,
                "research_scope": scope,
                "scope_reasons": scope_reasons,
            }
    addition_frame = pd.DataFrame.from_dict(additions, orient="index")
    for column in addition_frame:
        frame[column] = addition_frame[column]
    return frame.drop(columns=["_scope_reasons"]).sort_values("_input_order").drop(
        columns=["_input_order"]
    ).reset_index(drop=True)


def _unmatched_row(
    event: pd.Series,
    *,
    id_column: str,
    ticker_column: str,
    entry_column: str,
    status: str,
    candidate_count: int,
    controls_per_event: int,
    scope: str,
    scope_reasons: str,
    industry_status: str,
    identity_status: str,
) -> dict[str, object]:
    return {
        id_column: event[id_column],
        "event_ticker": event[ticker_column],
        "entry_at": event[entry_column],
        "control_rank": pd.NA,
        "control_ticker": pd.NA,
        "control_permanent_id": pd.NA,
        "control_issuer_cik": pd.NA,
        "match_distance": np.nan,
        "match_status": status,
        "eligible_candidate_count": int(candidate_count),
        "matched_control_count": 0,
        "required_control_count": controls_per_event,
        "industry_match_basis": industry_status,
        "identity_exclusion_basis": identity_status,
        "research_scope": scope,
        "scope_reasons": scope_reasons,
    }


def match_liquidity_risk_controls(
    events: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    exchange_sessions: Sequence[str | pd.Timestamp] | pd.Series | pd.DatetimeIndex | None = None,
    calendar_authoritative: bool = False,
    calendar_provenance_sha256: str | None = None,
    controls_per_event: int = 3,
    feature_columns: Sequence[str] = ("log_adv_dollars", "realized_volatility"),
    id_column: str = "event_id",
    ticker_column: str = "ticker",
    information_column: str = "information_at",
    entry_column: str = "entry_at",
    available_column: str = "feature_available_at",
    eligible_column: str = "eligible",
    clock_verified_column: str = "clock_verified",
    industry_column: str | None = "sic2",
    permanent_id_column: str | None = "security_id",
    issuer_column: str = "issuer_cik",
    known_event_column: str | None = "known_event_at",
    security_master_known_column: str = "security_master_known_at",
    security_master_effective_column: str = "security_master_effective_at",
    industry_known_column: str = "sic_known_at",
    industry_effective_column: str = "sic_effective_at",
    security_master_clock_complete: bool = False,
    industry_clock_complete: bool = False,
    known_event_clock_complete: bool = False,
    known_event_provenance_sha256: str | None = None,
) -> pd.DataFrame:
    """Select same-entry nearest controls with explicit unmatched records.

    Matching uses event-time candidate standard deviations and Euclidean
    distance over liquidity/risk features.  If SIC or permanent identifiers
    are absent, the function can still create a diagnostic cross-industry or
    ticker/issuer-based match, but every such row is labelled
    ``DIAGNOSTIC_ONLY`` with the reason recorded.
    """

    if controls_per_event <= 0:
        raise ValueError("controls_per_event must be positive")
    if not feature_columns:
        raise ValueError("feature_columns must be non-empty")
    event_required = [
        id_column,
        ticker_column,
        information_column,
        entry_column,
        available_column,
        eligible_column,
        *feature_columns,
    ]
    candidate_required = [
        ticker_column,
        entry_column,
        available_column,
        eligible_column,
        *feature_columns,
    ]
    _require(events, event_required, name="events")
    _require(candidates, candidate_required, name="control candidates")
    known_event_digest_verified = False
    if known_event_provenance_sha256:
        if not known_event_column or known_event_column not in candidates:
            raise Form4EventStudyError(
                "known-event digest supplied without a known-event column"
            )
        known_event_digest_verified = _verify_digest(
            canonical_frame_sha256(candidates),
            known_event_provenance_sha256,
            name="known-event candidate snapshot",
        )
    event_frame = events.copy()
    candidate_frame = candidates.copy()
    validate_unique(event_frame, [id_column], "control-match events")
    for frame, name in [(event_frame, "events"), (candidate_frame, "control candidates")]:
        frame[ticker_column] = _ticker(frame[ticker_column], name=f"{name}.{ticker_column}")
        if permanent_id_column and permanent_id_column in frame:
            frame[permanent_id_column] = _security_ids(
                frame[permanent_id_column], name=f"{name}.{permanent_id_column}"
            )
        if issuer_column in frame:
            frame[issuer_column] = _ciks(
                frame[issuer_column], name=f"{name}.{issuer_column}"
            )
        if industry_column and industry_column in frame:
            frame[industry_column] = frame[industry_column].astype("string").str.strip()
            frame.loc[frame[industry_column].eq(""), industry_column] = pd.NA
        frame[entry_column] = _utc(frame[entry_column], name=f"{name}.{entry_column}")
        frame[available_column] = _utc(
            frame[available_column],
            name=f"{name}.{available_column}",
            allow_missing=True,
        )
        frame[eligible_column] = _boolean(
            frame[eligible_column], name=f"{name}.{eligible_column}"
        )
        for column in feature_columns:
            frame[column] = _number(
                frame[column], name=f"{name}.{column}", allow_missing=True
            )
        for clock_column in [
            security_master_known_column,
            security_master_effective_column,
            industry_known_column,
            industry_effective_column,
        ]:
            if clock_column in frame:
                frame[clock_column] = _utc(
                    frame[clock_column],
                    name=f"{name}.{clock_column}",
                    allow_missing=True,
                )
    validate_unique(candidate_frame, [entry_column, ticker_column], "control candidates")
    if permanent_id_column and permanent_id_column in candidate_frame:
        identified_candidates = candidate_frame.loc[
            candidate_frame[permanent_id_column].notna()
        ]
        validate_unique(
            identified_candidates,
            [entry_column, permanent_id_column],
            "control candidates by permanent security id",
        )
    event_frame[information_column] = _utc(
        event_frame[information_column], name=f"events.{information_column}"
    )
    if clock_verified_column in event_frame:
        event_frame[clock_verified_column] = _boolean(
            event_frame[clock_verified_column],
            name=f"events.{clock_verified_column}",
        )
    if (~event_frame[information_column].lt(event_frame[entry_column])).any():
        raise PointInTimeViolation("control events require information_at < entry_at")
    event_frame = validate_event_entry_clocks(
        event_frame,
        exchange_sessions,
        calendar_authoritative=calendar_authoritative,
        calendar_provenance_sha256=calendar_provenance_sha256,
        id_column=id_column,
        information_column=information_column,
        entry_column=entry_column,
    )

    if known_event_column and known_event_column in candidate_frame:
        candidate_frame[known_event_column] = _utc(
            candidate_frame[known_event_column],
            name=f"control candidates.{known_event_column}",
            allow_missing=True,
        )

    rows: list[dict[str, object]] = []
    for _, event in event_frame.sort_values(
        [information_column, id_column], kind="mergesort"
    ).iterrows():
        reasons: list[str] = [
            token
            for token in str(event.get("scope_reasons", "")).split("|")
            if token and token != "nan"
        ]
        if (
            clock_verified_column not in event_frame
            or not bool(event[clock_verified_column])
        ):
            reasons.append("event_entry_clock_not_verified")
        permanent_usable = (
            _has_identifier(event, permanent_id_column)
            and permanent_id_column in candidate_frame
            and candidate_frame[permanent_id_column].notna().any()
        )
        issuer_usable = (
            issuer_column in event_frame
            and issuer_column in candidate_frame
            and _has_identifier(event, issuer_column)
        )
        if permanent_usable and issuer_usable:
            identity_status = "permanent_security_id_and_issuer_cik"
        elif permanent_usable:
            identity_status = "permanent_security_id_only"
            reasons.append("missing_issuer_cik")
        elif issuer_usable:
            identity_status = "issuer_cik_fallback"
            reasons.append("missing_permanent_security_id")
        else:
            identity_status = "ticker_fallback"
            reasons.extend(["missing_permanent_security_id", "missing_issuer_cik"])

        industry_usable = (
            bool(industry_column)
            and industry_column in event_frame
            and industry_column in candidate_frame
            and pd.notna(event.get(industry_column))
            and str(event.get(industry_column)).strip() != ""
        )
        if industry_usable:
            industry_status = "same_sic"
        else:
            industry_status = "cross_industry_diagnostic_fallback"
            reasons.append("missing_same_sic_control_dimension")

        security_clock_columns_present = all(
            column in event_frame and column in candidate_frame
            for column in [
                security_master_known_column,
                security_master_effective_column,
            ]
        )
        industry_clock_columns_present = all(
            column in event_frame and column in candidate_frame
            for column in [industry_known_column, industry_effective_column]
        )
        if not security_master_clock_complete or not security_clock_columns_present:
            reasons.append("security_master_clock_not_proven_complete")
        if not industry_clock_complete or not industry_clock_columns_present:
            reasons.append("sic_history_clock_not_proven_complete")
        if (
            not known_event_clock_complete
            or not known_event_digest_verified
            or not known_event_column
            or known_event_column not in candidate_frame
        ):
            reasons.append("known_event_clock_not_proven_complete")

        pre_status = ""
        if security_clock_columns_present:
            if pd.isna(event[security_master_known_column]) or pd.isna(
                event[security_master_effective_column]
            ):
                reasons.append("missing_event_security_master_clock")
                if security_master_clock_complete:
                    pre_status = "unmatched_missing_event_security_master_clock"
            elif event[security_master_known_column] > event[information_column]:
                pre_status = "unmatched_event_security_master_known_after_information"
            elif event[security_master_effective_column] > event[entry_column]:
                pre_status = "unmatched_event_security_master_effective_after_entry"
        if industry_clock_columns_present:
            if pd.isna(event[industry_known_column]) or pd.isna(
                event[industry_effective_column]
            ):
                reasons.append("missing_event_sic_clock")
                if industry_clock_complete:
                    pre_status = "unmatched_missing_event_sic_clock"
            elif event[industry_known_column] > event[information_column]:
                pre_status = "unmatched_event_sic_known_after_information"
            elif event[industry_effective_column] > event[information_column]:
                pre_status = "unmatched_event_sic_effective_after_information"
        scope, scope_reasons = _scope(event.get("research_scope"), reasons)

        if pre_status:
            rows.append(
                _unmatched_row(
                    event,
                    id_column=id_column,
                    ticker_column=ticker_column,
                    entry_column=entry_column,
                    status=pre_status,
                    candidate_count=0,
                    controls_per_event=controls_per_event,
                    scope=scope,
                    scope_reasons=scope_reasons,
                    industry_status=industry_status,
                    identity_status=identity_status,
                )
            )
            continue
        if not bool(event[eligible_column]):
            rows.append(
                _unmatched_row(
                    event,
                    id_column=id_column,
                    ticker_column=ticker_column,
                    entry_column=entry_column,
                    status="unmatched_event_ineligible",
                    candidate_count=0,
                    controls_per_event=controls_per_event,
                    scope=scope,
                    scope_reasons=scope_reasons,
                    industry_status=industry_status,
                    identity_status=identity_status,
                )
            )
            continue
        if event[list(feature_columns)].isna().any():
            status = "unmatched_missing_event_features"
            candidate_count = 0
        elif pd.isna(event[available_column]):
            status = "unmatched_missing_event_feature_availability"
            candidate_count = 0
        elif event[available_column] > event[information_column]:
            status = "unmatched_event_features_after_information"
            candidate_count = 0
        else:
            pool = candidate_frame.loc[
                candidate_frame[entry_column].eq(event[entry_column])
                & candidate_frame[eligible_column].eq(True)
                & candidate_frame[available_column].le(event[information_column])
            ].copy()
            status = ""
            candidate_count = len(pool)
            if security_clock_columns_present:
                before_clock_count = len(pool)
                known_valid = candidate_frame[security_master_known_column].le(
                    event[information_column]
                )
                effective_valid = candidate_frame[security_master_effective_column].le(
                    event[entry_column]
                )
                if not security_master_clock_complete:
                    known_valid |= candidate_frame[security_master_known_column].isna()
                    effective_valid |= candidate_frame[
                        security_master_effective_column
                    ].isna()
                security_clock_valid = known_valid & effective_valid
                pool = pool.loc[security_clock_valid.loc[pool.index]]
                candidate_count = len(pool)
                if before_clock_count and pool.empty:
                    status = "unmatched_no_pit_security_master_candidates"
            if not status and industry_clock_columns_present:
                before_clock_count = len(pool)
                known_valid = candidate_frame[industry_known_column].le(
                    event[information_column]
                )
                effective_valid = candidate_frame[industry_effective_column].le(
                    event[information_column]
                )
                if not industry_clock_complete:
                    known_valid |= candidate_frame[industry_known_column].isna()
                    effective_valid |= candidate_frame[industry_effective_column].isna()
                industry_clock_valid = known_valid & effective_valid
                pool = pool.loc[industry_clock_valid.loc[pool.index]]
                candidate_count = len(pool)
                if before_clock_count and pool.empty:
                    status = "unmatched_no_pit_sic_candidates"
            if not status and known_event_column and known_event_column in pool:
                before_event_filter_count = len(pool)
                pool = pool.loc[
                    pool[known_event_column].isna()
                    | pool[known_event_column].gt(event[information_column])
                ]
                candidate_count = len(pool)
                if before_event_filter_count and pool.empty:
                    status = "unmatched_all_candidates_have_known_events"
            before_identity_count = len(pool)
            if not status and permanent_usable:
                pool = pool.loc[
                    pool[permanent_id_column].astype("string").ne(
                        str(event[permanent_id_column])
                    )
                ]
            if not status and issuer_usable:
                pool = pool.loc[
                    pool[issuer_column].astype("string").ne(str(event[issuer_column]))
                ]
            if not status:
                pool = pool.loc[pool[ticker_column].ne(event[ticker_column])]
                candidate_count = len(pool)
                if before_identity_count and pool.empty:
                    status = "unmatched_only_same_security_or_issuer"
            if not status and industry_usable:
                same_industry = pool.loc[
                    pool[industry_column].astype("string").eq(str(event[industry_column]))
                ]
                if same_industry.empty:
                    status = "unmatched_no_same_sic_candidates"
                    candidate_count = 0
                    pool = same_industry
                else:
                    pool = same_industry
                    status = ""
                    candidate_count = len(pool)
            elif not status:
                status = ""
                candidate_count = len(pool)
            if not status:
                pool = pool.dropna(subset=list(feature_columns))
                candidate_count = len(pool)
                if candidate_count < controls_per_event:
                    status = "unmatched_insufficient_complete_candidates"

        if status:
            rows.append(
                _unmatched_row(
                    event,
                    id_column=id_column,
                    ticker_column=ticker_column,
                    entry_column=entry_column,
                    status=status,
                    candidate_count=candidate_count,
                    controls_per_event=controls_per_event,
                    scope=scope,
                    scope_reasons=scope_reasons,
                    industry_status=industry_status,
                    identity_status=identity_status,
                )
            )
            continue

        squared_distance = np.zeros(len(pool), dtype=float)
        for column in feature_columns:
            scale = float(pool[column].std(ddof=0))
            if not np.isfinite(scale) or scale == 0:
                scale = 1.0
            squared_distance += (
                (pool[column].to_numpy(dtype=float) - float(event[column])) / scale
            ) ** 2
        pool["_match_distance"] = np.sqrt(squared_distance)
        selected = pool.sort_values(
            ["_match_distance", ticker_column], kind="mergesort"
        ).head(controls_per_event)
        for rank, (_, control) in enumerate(selected.iterrows(), start=1):
            selected_reasons = [
                token for token in scope_reasons.split("|") if token
            ]
            control_scope = control.get("research_scope", DIAGNOSTIC_ONLY)
            control_scope_reasons = str(control.get("scope_reasons", ""))
            if control_scope != RESEARCH_CANDIDATE:
                selected_reasons.append("selected_control_diagnostic_only")
                if "research_scope" not in control:
                    selected_reasons.append("selected_control_scope_not_established")
                selected_reasons.extend(
                    token
                    for token in control_scope_reasons.split("|")
                    if token and token != "nan"
                )
            if not _has_identifier(control, permanent_id_column):
                selected_reasons.append("selected_control_missing_permanent_security_id")
            if not _has_identifier(control, issuer_column):
                selected_reasons.append("selected_control_missing_issuer_cik")
            if security_clock_columns_present and (
                pd.isna(control[security_master_known_column])
                or pd.isna(control[security_master_effective_column])
            ):
                selected_reasons.append("selected_control_missing_security_master_clock")
            if industry_clock_columns_present and (
                pd.isna(control[industry_known_column])
                or pd.isna(control[industry_effective_column])
            ):
                selected_reasons.append("selected_control_missing_sic_clock")
            selected_scope, selected_scope_reasons = _scope(
                event.get("research_scope"), selected_reasons
            )
            rows.append(
                {
                    id_column: event[id_column],
                    "event_ticker": event[ticker_column],
                    "entry_at": event[entry_column],
                    "control_rank": rank,
                    "control_ticker": control[ticker_column],
                    "control_permanent_id": (
                        control.get(permanent_id_column)
                        if permanent_id_column and permanent_id_column in control
                        else pd.NA
                    ),
                    "control_issuer_cik": (
                        control.get(issuer_column) if issuer_column in control else pd.NA
                    ),
                    "match_distance": float(control["_match_distance"]),
                    "match_status": "matched",
                    "eligible_candidate_count": int(candidate_count),
                    "matched_control_count": controls_per_event,
                    "required_control_count": controls_per_event,
                    "industry_match_basis": industry_status,
                    "identity_exclusion_basis": identity_status,
                    "selected_control_research_scope": control_scope,
                    "selected_control_scope_reasons": control_scope_reasons,
                    "research_scope": selected_scope,
                    "scope_reasons": selected_scope_reasons,
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "DIAGNOSTIC_ONLY",
    "PRICE_BASES",
    "RESEARCH_CANDIDATE",
    "Form4EventStudyError",
    "build_prior_close_eligibility_snapshot",
    "canonical_frame_sha256",
    "compute_open_to_open_returns",
    "compute_pre_event_placebo",
    "exchange_calendar_sha256",
    "match_liquidity_risk_controls",
    "suppress_causal_overlaps",
    "validate_event_entry_clocks",
]
