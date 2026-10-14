"""Point-in-time SEC Form 4 code-P purchase event research primitives.

SEC transaction code P covers purchases on an exchange *or* from another
person.  The functions intentionally call the signal ``code_p_purchase`` and
do not claim that every observation is a confirmed open-market transaction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from alpha_factor_engine.contracts import PointInTimeViolation, validate_unique


class Form4ResearchError(ValueError):
    """Raised when an event-research input cannot be used without repair."""


@dataclass(frozen=True)
class Form4PurchaseFilter:
    minimum_purchase_usd: float = 25_000.0
    qualifying_role_pattern: str = r"officer|director"
    include_security_pattern: str = r"common|ordinary"
    exclude_security_pattern: str = (
        r"preferred|warrant|option|phantom|restricted|unit|depositary|convertible"
    )
    require_explicit_non_10b5_1: bool = True


FALSE_VALUES = {"0", "false", "f", "n", "no"}
TRUE_VALUES = {"1", "true", "t", "y", "yes"}


def _require(frame: pd.DataFrame, columns: Iterable[str], *, name: str) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise Form4ResearchError(f"{name} missing columns: {sorted(missing)}")


def _strict_boolean(values: pd.Series, *, name: str) -> pd.Series:
    text = values.astype("string").str.strip().str.lower()
    result = pd.Series(pd.NA, index=values.index, dtype="boolean")
    result.loc[text.isin(FALSE_VALUES)] = False
    result.loc[text.isin(TRUE_VALUES)] = True
    missing = values.isna() | text.eq("")
    invalid = ~missing & result.isna()
    if invalid.any():
        raise Form4ResearchError(
            f"{name} has invalid boolean values: {values.loc[invalid].head(5).tolist()}"
        )
    return result


def _strict_number(values: pd.Series, *, name: str) -> pd.Series:
    parsed = pd.to_numeric(values, errors="coerce")
    present = values.notna() & values.astype("string").str.strip().ne("")
    invalid = present & (parsed.isna() | ~np.isfinite(parsed))
    if invalid.any():
        raise Form4ResearchError(
            f"{name} has non-numeric values: {values.loc[invalid].head(5).tolist()}"
        )
    return parsed


def _pipe_unique(values: Iterable[object]) -> str:
    cleaned = sorted(
        {
            str(value).strip()
            for value in values
            if pd.notna(value) and str(value).strip()
        }
    )
    return "|".join(cleaned)


def _pipe_union(values: Iterable[object]) -> str:
    tokens: set[str] = set()
    for value in values:
        if pd.isna(value):
            continue
        tokens.update(token.strip() for token in str(value).split("|") if token.strip())
    return "|".join(sorted(tokens))


def filter_code_p_purchases(
    submissions: pd.DataFrame,
    reporting_owners: pd.DataFrame,
    nonderivative_transactions: pd.DataFrame,
    *,
    config: Form4PurchaseFilter | None = None,
) -> pd.DataFrame:
    """Create one conservative code-P purchase event per original accession.

    Amendments are retained in the raw layer but excluded from the primary
    signal.  Missing swap or Rule 10b5-1 indicators do not silently become
    ``False``.
    """

    cfg = config or Form4PurchaseFilter()
    if cfg.minimum_purchase_usd < 0:
        raise ValueError("minimum_purchase_usd must be non-negative")
    submission_columns = [
        "ACCESSION_NUMBER",
        "FILING_DATE",
        "DOCUMENT_TYPE",
        "ISSUERCIK",
        "ISSUERNAME",
        "ISSUERTRADINGSYMBOL",
    ]
    if cfg.require_explicit_non_10b5_1:
        submission_columns.append("AFF10B5ONE")
    _require(submissions, submission_columns, name="SUBMISSION")
    _require(
        reporting_owners,
        [
            "ACCESSION_NUMBER",
            "RPTOWNERCIK",
            "RPTOWNERNAME",
            "RPTOWNER_RELATIONSHIP",
            "RPTOWNER_TITLE",
        ],
        name="REPORTINGOWNER",
    )
    _require(
        nonderivative_transactions,
        [
            "ACCESSION_NUMBER",
            "NONDERIV_TRANS_SK",
            "SECURITY_TITLE",
            "TRANS_DATE",
            "TRANS_FORM_TYPE",
            "TRANS_CODE",
            "EQUITY_SWAP_INVOLVED",
            "TRANS_SHARES",
            "TRANS_PRICEPERSHARE",
            "TRANS_ACQUIRED_DISP_CD",
            "DIRECT_INDIRECT_OWNERSHIP",
            "TRANS_TIMELINESS",
        ],
        name="NONDERIV_TRANS",
    )
    validate_unique(submissions, ["ACCESSION_NUMBER"], "Form 4 submissions")
    validate_unique(
        nonderivative_transactions,
        ["ACCESSION_NUMBER", "NONDERIV_TRANS_SK"],
        "Form 4 non-derivative transactions",
    )
    validate_unique(
        reporting_owners,
        ["ACCESSION_NUMBER", "RPTOWNERCIK"],
        "Form 4 reporting owners",
    )

    submission = submissions.copy()
    if "AFF10B5ONE" in submission:
        submission["_10b5"] = _strict_boolean(
            submission["AFF10B5ONE"], name="AFF10B5ONE"
        )
    else:
        # Older official quarterly archives predate this field. They may be
        # used to construct an owner-history table, but callers must opt out
        # of the explicit non-10b5-1 primary-signal requirement.
        submission["_10b5"] = pd.Series(
            pd.NA, index=submission.index, dtype="boolean"
        )
    submission["FILING_DATE"] = pd.to_datetime(submission["FILING_DATE"], errors="coerce")
    invalid_filing_date = submission["FILING_DATE"].isna()
    if invalid_filing_date.any():
        raise Form4ResearchError(
            "SUBMISSION has invalid FILING_DATE values: "
            f"{submissions.loc[invalid_filing_date, 'FILING_DATE'].head(5).tolist()}"
        )
    submission = submission.loc[submission["DOCUMENT_TYPE"].astype("string").eq("4")].copy()
    if cfg.require_explicit_non_10b5_1:
        submission = submission.loc[submission["_10b5"].eq(False).fillna(False)].copy()

    owner = reporting_owners.copy()
    owner["_qualifying_role"] = owner["RPTOWNER_RELATIONSHIP"].astype("string").str.contains(
        cfg.qualifying_role_pattern, case=False, na=False, regex=True
    )
    for source, target in [
        ("RPTOWNERCIK", "_qualifying_owner_cik"),
        ("RPTOWNERNAME", "_qualifying_owner_name"),
        ("RPTOWNER_RELATIONSHIP", "_qualifying_owner_role"),
        ("RPTOWNER_TITLE", "_qualifying_owner_title"),
    ]:
        owner[target] = owner[source].where(owner["_qualifying_role"])
    owner_summary = (
        owner.groupby("ACCESSION_NUMBER", sort=True)
        .agg(
            reporting_owner_ciks=("RPTOWNERCIK", _pipe_unique),
            reporting_owner_count=("RPTOWNERCIK", "nunique"),
            owner_ciks=("_qualifying_owner_cik", _pipe_unique),
            owner_names=("_qualifying_owner_name", _pipe_unique),
            owner_roles=("_qualifying_owner_role", _pipe_unique),
            owner_titles=("_qualifying_owner_title", _pipe_unique),
            owner_count=("_qualifying_owner_cik", "nunique"),
            qualifying_owner_count=("_qualifying_role", "sum"),
        )
        .reset_index()
    )
    owner_summary = owner_summary.loc[owner_summary["qualifying_owner_count"].gt(0)].copy()

    transaction = nonderivative_transactions.copy()
    transaction["_swap"] = _strict_boolean(
        transaction["EQUITY_SWAP_INVOLVED"], name="EQUITY_SWAP_INVOLVED"
    )
    transaction["TRANS_SHARES"] = _strict_number(
        transaction["TRANS_SHARES"], name="TRANS_SHARES"
    )
    transaction["TRANS_PRICEPERSHARE"] = _strict_number(
        transaction["TRANS_PRICEPERSHARE"], name="TRANS_PRICEPERSHARE"
    )
    raw_transaction_date = transaction["TRANS_DATE"]
    transaction["TRANS_DATE"] = pd.to_datetime(raw_transaction_date, errors="coerce")
    invalid_transaction_date = (
        raw_transaction_date.notna()
        & raw_transaction_date.astype("string").str.strip().ne("")
        & transaction["TRANS_DATE"].isna()
    )
    if invalid_transaction_date.any():
        raise Form4ResearchError(
            "TRANS_DATE has invalid values: "
            f"{raw_transaction_date.loc[invalid_transaction_date].head(5).tolist()}"
        )
    title = transaction["SECURITY_TITLE"].astype("string")
    ownership = transaction["DIRECT_INDIRECT_OWNERSHIP"].astype("string").str.upper()
    eligible = (
        transaction["TRANS_FORM_TYPE"].astype("string").eq("4")
        & transaction["TRANS_CODE"].astype("string").str.upper().eq("P")
        & transaction["TRANS_ACQUIRED_DISP_CD"].astype("string").str.upper().eq("A")
        & transaction["_swap"].eq(False).fillna(False)
        & transaction["TRANS_SHARES"].gt(0)
        & transaction["TRANS_PRICEPERSHARE"].gt(0)
        & transaction["TRANS_DATE"].notna()
        & ownership.isin(["D", "I"])
        & title.str.contains(cfg.include_security_pattern, case=False, na=False, regex=True)
        & ~title.str.contains(cfg.exclude_security_pattern, case=False, na=False, regex=True)
    )
    transaction = transaction.loc[eligible].copy()
    transaction["purchase_usd"] = (
        transaction["TRANS_SHARES"] * transaction["TRANS_PRICEPERSHARE"]
    )
    transaction["direct_owner_flag"] = transaction["DIRECT_INDIRECT_OWNERSHIP"].astype(
        "string"
    ).str.upper().eq("D")
    transaction["late_flag"] = transaction["TRANS_TIMELINESS"].astype("string").str.upper().eq(
        "L"
    )

    joined = transaction.merge(
        submission,
        on="ACCESSION_NUMBER",
        how="inner",
        validate="many_to_one",
    ).merge(
        owner_summary,
        on="ACCESSION_NUMBER",
        how="inner",
        validate="many_to_one",
    )
    if joined.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "accession",
                "filing_date",
                "document_type",
                "issuer_cik",
                "issuer_name",
                "ticker",
                "purchase_usd",
                "transaction_date_min",
                "transaction_date_max",
                "qualifying_transaction_rows",
                "owner_ciks",
                "owner_names",
                "owner_roles",
                "owner_titles",
                "owner_count",
                "qualifying_owner_count",
                "reporting_owner_ciks",
                "reporting_owner_count",
                "any_direct_ownership",
                "any_late_transaction",
                "signal_type",
            ]
        )
    invalid_order = joined["TRANS_DATE"].dt.date > joined["FILING_DATE"].dt.date
    if invalid_order.any():
        raise PointInTimeViolation(
            "Form 4 transaction date occurs after filing date: "
            f"{joined.loc[invalid_order, ['ACCESSION_NUMBER', 'TRANS_DATE', 'FILING_DATE']].head(5).to_dict('records')}"
        )

    grouped = (
        joined.groupby("ACCESSION_NUMBER", sort=True)
        .agg(
            filing_date=("FILING_DATE", "first"),
            document_type=("DOCUMENT_TYPE", "first"),
            issuer_cik=("ISSUERCIK", "first"),
            issuer_name=("ISSUERNAME", "first"),
            ticker=("ISSUERTRADINGSYMBOL", "first"),
            purchase_usd=("purchase_usd", "sum"),
            transaction_date_min=("TRANS_DATE", "min"),
            transaction_date_max=("TRANS_DATE", "max"),
            qualifying_transaction_rows=("NONDERIV_TRANS_SK", "size"),
            owner_ciks=("owner_ciks", "first"),
            owner_names=("owner_names", "first"),
            owner_roles=("owner_roles", "first"),
            owner_titles=("owner_titles", "first"),
            owner_count=("owner_count", "first"),
            qualifying_owner_count=("qualifying_owner_count", "first"),
            reporting_owner_ciks=("reporting_owner_ciks", "first"),
            reporting_owner_count=("reporting_owner_count", "first"),
            any_direct_ownership=("direct_owner_flag", "any"),
            any_late_transaction=("late_flag", "any"),
        )
        .reset_index()
        .rename(columns={"ACCESSION_NUMBER": "accession"})
    )
    grouped = grouped.loc[grouped["purchase_usd"].ge(cfg.minimum_purchase_usd)].copy()
    grouped["event_id"] = grouped["accession"]
    grouped["signal_type"] = "form4_code_p_purchase"
    grouped["issuer_cik"] = grouped["issuer_cik"].astype("string")
    grouped["ticker"] = grouped["ticker"].astype("string").str.upper()
    ordered = ["event_id", "accession"] + [
        column for column in grouped.columns if column not in {"event_id", "accession"}
    ]
    return grouped[ordered].sort_values(["filing_date", "accession"]).reset_index(drop=True)


def explode_event_owners(events: pd.DataFrame) -> pd.DataFrame:
    """Return one owner-event row from the pipe-delimited event representation."""

    _require(events, ["event_id", "owner_ciks", "transaction_date_max"], name="events")
    output = events[["event_id", "owner_ciks", "transaction_date_max"]].copy()
    output["owner_cik"] = output["owner_ciks"].astype("string").str.split("|")
    output = output.explode("owner_cik", ignore_index=True)
    output = output.loc[output["owner_cik"].notna() & output["owner_cik"].str.strip().ne("")]
    return output.rename(columns={"transaction_date_max": "transaction_date"})[
        ["event_id", "owner_cik", "transaction_date"]
    ]


def classify_routine_opportunistic(
    owner_trades: pd.DataFrame,
    classification_dates: Sequence[str | pd.Timestamp],
    *,
    owner_column: str = "owner_cik",
    trade_date_column: str = "transaction_date",
    information_column: str = "information_at",
    lookback_years: int = 3,
) -> pd.DataFrame:
    """Classify owners using only complete calendar years before each as-of date.

    Routine owners traded in at least one common calendar month in every prior
    lookback year. Only trades public before the classification cutoff enter
    the history. Owners active in every year but without a common month are
    opportunistic; owners with incomplete history remain unclassified.
    """

    if lookback_years < 2:
        raise ValueError("lookback_years must be at least 2")
    _require(
        owner_trades,
        [owner_column, trade_date_column, information_column],
        name="owner trades",
    )
    trades = owner_trades[
        [owner_column, trade_date_column, information_column]
    ].copy()
    if trades[owner_column].isna().any():
        raise Form4ResearchError(f"{owner_column} contains missing identifiers")
    trades[owner_column] = trades[owner_column].astype("string")
    trades[trade_date_column] = pd.to_datetime(trades[trade_date_column], utc=True, errors="coerce")
    if trades[trade_date_column].isna().any():
        raise Form4ResearchError(f"{trade_date_column} contains invalid dates")
    trades[information_column] = pd.to_datetime(
        trades[information_column], utc=True, errors="coerce"
    )
    if trades[information_column].isna().any():
        raise PointInTimeViolation(
            f"{information_column} contains missing or invalid availability times"
        )
    impossible_clock = trades[information_column].lt(trades[trade_date_column])
    if impossible_clock.any():
        raise PointInTimeViolation(
            f"{information_column} precedes {trade_date_column}"
        )
    as_of_values = pd.to_datetime(pd.Series(list(classification_dates)), utc=True, errors="coerce")
    if as_of_values.isna().any() or as_of_values.empty:
        raise ValueError("classification_dates must contain valid dates")

    owners = sorted(trades[owner_column].unique())
    rows: list[dict[str, object]] = []
    for as_of in sorted(as_of_values.unique()):
        as_of = pd.Timestamp(as_of)
        if not (as_of.month == 1 and as_of.day == 1):
            raise ValueError("classification dates must be January 1 to avoid partial-year lookback")
        years = list(range(as_of.year - lookback_years, as_of.year))
        start = pd.Timestamp(year=years[0], month=1, day=1, tz="UTC")
        date_eligible = trades[trade_date_column].ge(start) & trades[
            trade_date_column
        ].lt(as_of)
        known_by_cutoff = trades[information_column].lt(as_of)
        history = trades.loc[date_eligible & known_by_cutoff].copy()
        history["year"] = history[trade_date_column].dt.year
        history["month"] = history[trade_date_column].dt.month
        for owner in owners:
            sample = history.loc[history[owner_column].eq(owner)]
            active_years = sorted(sample["year"].unique())
            if active_years != years:
                classification = "unclassified"
                routine_months: set[int] = set()
            else:
                month_sets = [set(sample.loc[sample["year"].eq(year), "month"]) for year in years]
                routine_months = set.intersection(*month_sets)
                classification = "routine" if routine_months else "opportunistic"
            rows.append(
                {
                    owner_column: owner,
                    "classification_at": as_of,
                    "classification": classification,
                    "routine_months": "|".join(str(value) for value in sorted(routine_months)),
                    "lookback_start": start,
                    "lookback_end": as_of - pd.Timedelta(nanoseconds=1),
                    "active_years": len(active_years),
                    "known_trade_count": int(len(sample)),
                    "classification_source_cutoff": as_of,
                }
            )
    return pd.DataFrame(rows).sort_values(["classification_at", owner_column]).reset_index(drop=True)


def build_filing_date_quarantine_schedule(
    events: pd.DataFrame,
    session_opens: Sequence[str | pd.Timestamp] | pd.Series,
    *,
    filing_date_column: str = "filing_date",
    quarantine_full_sessions: int = 1,
    holding_sessions: int = 20,
) -> pd.DataFrame:
    """Create a conservative schedule when only the filing date is trusted.

    The event is quarantined through ``quarantine_full_sessions`` complete
    exchange sessions after the stated filing date. Entry occurs at the next
    open. With the default of one full session, a Monday filing can first be
    traded at Wednesday's open. This deliberately sacrifices immediacy so an
    after-hours filing or delayed EDGAR dissemination cannot be treated as
    public before it was observable.
    """

    if quarantine_full_sessions < 0:
        raise ValueError("quarantine_full_sessions must be non-negative")
    if holding_sessions <= 0:
        raise ValueError("holding_sessions must be positive")
    _require(events, ["event_id", filing_date_column], name="events")
    result = events.copy()
    raw_dates = pd.to_datetime(result[filing_date_column], errors="coerce")
    if raw_dates.isna().any():
        raise PointInTimeViolation(f"{filing_date_column} contains invalid values")
    filing_dates = pd.Index(raw_dates.dt.date)

    opens = pd.DatetimeIndex(
        pd.to_datetime(pd.Series(list(session_opens)), utc=True, errors="coerce")
    )
    if opens.isna().any() or len(opens) == 0:
        raise Form4ResearchError("session_opens must contain valid timestamps")
    opens = opens.sort_values()
    if opens.has_duplicates:
        raise Form4ResearchError("session_opens contains duplicates")
    local_session_dates = pd.Index(opens.tz_convert("America/New_York").date)
    if local_session_dates.has_duplicates:
        raise Form4ResearchError("session_opens contains multiple opens per local session")

    first_after_positions = local_session_dates.searchsorted(filing_dates, side="right")
    quarantine_positions = first_after_positions + quarantine_full_sessions - 1
    entry_positions = first_after_positions + quarantine_full_sessions
    exit_positions = entry_positions + holding_sessions
    invalid = (
        (first_after_positions >= len(opens))
        | (quarantine_positions >= len(opens))
        | (entry_positions >= len(opens))
        | (exit_positions >= len(opens))
    )
    if invalid.any():
        raise PointInTimeViolation(
            "insufficient future sessions for filing-date quarantine schedule: "
            f"{result.loc[invalid, 'event_id'].head(5).tolist()}"
        )

    if quarantine_full_sessions == 0:
        local_information = pd.DatetimeIndex(
            pd.to_datetime(raw_dates.dt.date)
        ).tz_localize("America/New_York") + pd.Timedelta(days=1) - pd.Timedelta(
            nanoseconds=1
        )
    else:
        quarantine_dates = local_session_dates.take(quarantine_positions)
        local_information = pd.DatetimeIndex(quarantine_dates).tz_localize(
            "America/New_York"
        ) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    result["information_at"] = local_information.tz_convert("UTC")
    result["entry_at"] = opens.take(entry_positions)
    result["exit_at"] = opens.take(exit_positions)
    result["entry_session"] = result["entry_at"].dt.tz_convert(
        "America/New_York"
    ).dt.date
    result["exit_session"] = result["exit_at"].dt.tz_convert(
        "America/New_York"
    ).dt.date
    result["holding_sessions"] = holding_sessions
    result["clock_basis"] = (
        f"filing_date_plus_{quarantine_full_sessions}_full_exchange_session_quarantine"
    )
    invalid_order = ~(
        result["information_at"].lt(result["entry_at"])
        & result["entry_at"].lt(result["exit_at"])
    )
    if invalid_order.any():
        raise PointInTimeViolation("quarantine schedule violates information < entry < exit")
    return result


def build_event_schedule(
    events: pd.DataFrame,
    session_opens: Sequence[str | pd.Timestamp] | pd.Series,
    *,
    information_column: str = "information_at",
    holding_sessions: int = 20,
) -> pd.DataFrame:
    """Map each event to the first tradable open and a fixed open-to-open exit."""

    if holding_sessions <= 0:
        raise ValueError("holding_sessions must be positive")
    _require(events, ["event_id", information_column], name="events")
    result = events.copy()
    result[information_column] = pd.to_datetime(result[information_column], utc=True, errors="coerce")
    if result[information_column].isna().any():
        raise PointInTimeViolation(f"{information_column} contains invalid values")
    opens = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(session_opens)), utc=True, errors="coerce"))
    if opens.isna().any() or len(opens) == 0:
        raise Form4ResearchError("session_opens must contain valid timestamps")
    opens = opens.sort_values()
    if opens.has_duplicates:
        raise Form4ResearchError("session_opens contains duplicates")
    entry_positions = opens.searchsorted(result[information_column], side="right")
    exit_positions = entry_positions + holding_sessions
    invalid = (entry_positions >= len(opens)) | (exit_positions >= len(opens))
    if invalid.any():
        raise PointInTimeViolation(
            "insufficient future sessions for event schedule: "
            f"{result.loc[invalid, 'event_id'].head(5).tolist()}"
        )
    result["entry_at"] = opens.take(entry_positions)
    result["exit_at"] = opens.take(exit_positions)
    result["entry_session"] = result["entry_at"].dt.tz_convert("America/New_York").dt.date
    result["exit_session"] = result["exit_at"].dt.tz_convert("America/New_York").dt.date
    result["holding_sessions"] = holding_sessions
    invalid_order = ~(
        result[information_column].lt(result["entry_at"])
        & result["entry_at"].lt(result["exit_at"])
    )
    if invalid_order.any():
        raise PointInTimeViolation("event schedule violates information < entry < exit")
    return result


def build_trailing_purchase_clusters(
    events: pd.DataFrame,
    *,
    window_days: int = 5,
    time_column: str = "information_at",
) -> pd.DataFrame:
    """Attach causal trailing cluster features; no future filing enters a row."""

    if window_days < 0:
        raise ValueError("window_days must be non-negative")
    _require(
        events,
        ["event_id", "accession", "issuer_cik", time_column, "purchase_usd", "owner_ciks"],
        name="events",
    )
    result = events.copy()
    result[time_column] = pd.to_datetime(result[time_column], utc=True, errors="coerce")
    if result[time_column].isna().any():
        raise PointInTimeViolation(f"{time_column} contains invalid values")
    result["purchase_usd"] = _strict_number(
        result["purchase_usd"], name="events.purchase_usd"
    )
    invalid_purchase = result["purchase_usd"].isna() | result["purchase_usd"].le(0)
    if invalid_purchase.any():
        raise Form4ResearchError("cluster events require positive purchase_usd")
    invalid_owner = result["owner_ciks"].isna() | result["owner_ciks"].astype(
        "string"
    ).str.strip().eq("")
    if invalid_owner.any():
        raise Form4ResearchError("cluster events require qualifying owner_ciks")
    validate_unique(result, ["event_id"], "Form 4 events")
    result = result.sort_values(["issuer_cik", time_column, "accession"], kind="mergesort")
    additions: dict[object, dict[str, object]] = {}
    delta = pd.Timedelta(days=window_days)
    for _, issuer_events in result.groupby("issuer_cik", sort=True):
        for index, current in issuer_events.iterrows():
            start = current[time_column] - delta
            history = issuer_events.loc[
                issuer_events[time_column].between(start, current[time_column], inclusive="both")
            ]
            additions[index] = {
                "cluster_window_start_at": start,
                "cluster_purchase_usd": float(history["purchase_usd"].sum()),
                "cluster_accessions": _pipe_unique(history["accession"]),
                "cluster_accession_count": int(history["accession"].nunique()),
                "cluster_owner_ciks": _pipe_union(history["owner_ciks"]),
                "cluster_owner_count": len(
                    [value for value in _pipe_union(history["owner_ciks"]).split("|") if value]
                ),
            }
    addition_frame = pd.DataFrame.from_dict(additions, orient="index")
    for column in addition_frame:
        result[column] = addition_frame[column]
    return result.sort_values([time_column, "issuer_cik", "accession"]).reset_index(drop=True)


def collapse_entry_session_events(events: pd.DataFrame) -> pd.DataFrame:
    """Collapse events known before the same issuer/session entry into one order."""

    _require(
        events,
        [
            "event_id",
            "accession",
            "issuer_cik",
            "ticker",
            "information_at",
            "entry_at",
            "exit_at",
            "purchase_usd",
            "owner_ciks",
        ],
        name="events",
    )
    frame = events.copy()
    for column in ["information_at", "entry_at", "exit_at"]:
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    invalid = frame[["information_at", "entry_at", "exit_at"]].isna().any(axis=1) | ~(
        frame["information_at"].lt(frame["entry_at"])
        & frame["entry_at"].lt(frame["exit_at"])
    )
    if invalid.any():
        raise PointInTimeViolation("entry-session collapse contains invalid clocks")
    frame["purchase_usd"] = _strict_number(
        frame["purchase_usd"], name="events.purchase_usd"
    )
    if frame["purchase_usd"].isna().any() or frame["purchase_usd"].le(0).any():
        raise Form4ResearchError("entry-session events require positive purchase_usd")
    validate_unique(frame, ["event_id"], "Form 4 events")
    validate_unique(frame, ["accession"], "Form 4 event accessions")
    session_groups = frame.groupby(["issuer_cik", "entry_at"], dropna=False)
    conflict = session_groups.agg(
        ticker_values=("ticker", "nunique"),
        exit_values=("exit_at", "nunique"),
    )
    conflict = conflict[conflict["ticker_values"].ne(1) | conflict["exit_values"].ne(1)]
    if not conflict.empty:
        raise Form4ResearchError(
            "same issuer/entry session has conflicting ticker or exit: "
            f"{list(conflict.index[:5])}"
        )
    grouped = (
        frame.groupby(["issuer_cik", "ticker", "entry_at"], sort=True)
        .agg(
            event_id=("event_id", _pipe_unique),
            accessions=("accession", _pipe_unique),
            information_at=("information_at", "max"),
            exit_at=("exit_at", "first"),
            purchase_usd=("purchase_usd", "sum"),
            owner_ciks=("owner_ciks", _pipe_union),
            accession_count=("accession", "nunique"),
        )
        .reset_index()
    )
    grouped["owner_count"] = grouped["owner_ciks"].map(
        lambda value: len([token for token in str(value).split("|") if token])
    )
    return grouped


def match_event_controls(
    events: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    controls_per_event: int = 3,
    industry_column: str = "sic2",
    availability_column: str = "available_at",
) -> pd.DataFrame:
    """Nearest-neighbour controls using only the event-time eligible cross-section."""

    if controls_per_event <= 0:
        raise ValueError("controls_per_event must be positive")
    if not feature_columns:
        raise ValueError("feature_columns must be non-empty")
    event_required = {
        "event_id",
        "issuer_cik",
        "ticker",
        "information_at",
        "entry_at",
        industry_column,
        *feature_columns,
    }
    candidate_required = {
        "issuer_cik",
        "ticker",
        "entry_at",
        availability_column,
        "eligible",
        industry_column,
        *feature_columns,
    }
    _require(events, event_required, name="events")
    _require(candidates, candidate_required, name="control candidates")
    event_frame = events.copy()
    candidate_frame = candidates.copy()
    for frame, columns in [
        (event_frame, ["information_at", "entry_at"]),
        (candidate_frame, ["entry_at", availability_column]),
    ]:
        for column in columns:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
        if frame[columns].isna().any(axis=None):
            raise PointInTimeViolation(f"invalid control-matching timestamps: {columns}")
    invalid_event_order = ~event_frame["information_at"].lt(event_frame["entry_at"])
    if invalid_event_order.any():
        raise PointInTimeViolation("control matching requires information_at < entry_at")
    for name, frame in [("events", event_frame), ("control candidates", candidate_frame)]:
        invalid_identifier = (
            frame["issuer_cik"].isna()
            | frame["ticker"].isna()
            | frame["ticker"].astype("string").str.strip().eq("")
            | frame[industry_column].isna()
            | frame[industry_column].astype("string").str.strip().eq("")
        )
        if invalid_identifier.any():
            raise Form4ResearchError(f"{name} contains missing identifiers or industry")
        for column in feature_columns:
            frame[column] = _strict_number(frame[column], name=f"{name}.{column}")
    candidate_frame["_eligible"] = _strict_boolean(
        candidate_frame["eligible"], name="control candidates.eligible"
    ).eq(True).fillna(False)
    validate_unique(event_frame, ["event_id"], "events")
    validate_unique(candidate_frame, ["entry_at", "ticker"], "control candidates")

    if "feature_available_at" in event_frame:
        event_frame["feature_available_at"] = pd.to_datetime(
            event_frame["feature_available_at"], utc=True, errors="coerce"
        )
        future_event_features = event_frame["feature_available_at"].isna() | event_frame[
            "feature_available_at"
        ].gt(event_frame["information_at"])
        if future_event_features.any():
            raise PointInTimeViolation("event matching features were unavailable at information_at")

    if "known_event_at" in candidate_frame:
        raw_known_event = candidate_frame["known_event_at"]
        candidate_frame["known_event_at"] = pd.to_datetime(
            raw_known_event, utc=True, errors="coerce"
        )
        invalid_known_event = raw_known_event.notna() & raw_known_event.astype(
            "string"
        ).str.strip().ne("") & candidate_frame["known_event_at"].isna()
        if invalid_known_event.any():
            raise PointInTimeViolation("control candidates contain invalid known_event_at")

    rows: list[dict[str, object]] = []
    for event in event_frame.sort_values(["information_at", "event_id"]).itertuples(index=False):
        event_data = event._asdict()
        eligible_pool = (
            candidate_frame["entry_at"].eq(event_data["entry_at"])
            & candidate_frame["_eligible"]
            & candidate_frame[availability_column].le(event_data["information_at"])
            & candidate_frame[industry_column].astype("string").eq(
                str(event_data[industry_column])
            )
            & ~candidate_frame["issuer_cik"].astype("string").eq(
                str(event_data["issuer_cik"])
            )
        )
        if "known_event_at" in candidate_frame:
            eligible_pool &= ~candidate_frame["known_event_at"].le(
                event_data["information_at"]
            ).fillna(False)
        pool = candidate_frame.loc[eligible_pool].copy()
        pool = pool.dropna(subset=list(feature_columns))
        event_values = pd.Series(
            {column: event_data[column] for column in feature_columns},
            dtype="float64",
        )
        if event_values.isna().any():
            raise Form4ResearchError(
                f"event {event_data['event_id']} has missing matching features"
            )
        if len(pool) < controls_per_event:
            raise Form4ResearchError(
                f"event {event_data['event_id']} has {len(pool)} eligible same-industry controls; "
                f"requires {controls_per_event}"
            )
        squared = np.zeros(len(pool), dtype=float)
        for column in feature_columns:
            values = pool[column].to_numpy(dtype=float)
            scale = float(np.nanstd(values, ddof=0))
            if not np.isfinite(scale) or scale == 0:
                scale = 1.0
            squared += ((values - float(event_values[column])) / scale) ** 2
        pool["match_distance"] = np.sqrt(squared)
        selected = pool.sort_values(["match_distance", "ticker"], kind="mergesort").head(
            controls_per_event
        )
        for rank, control in enumerate(selected.itertuples(index=False), start=1):
            control_data = control._asdict()
            rows.append(
                {
                    "event_id": event_data["event_id"],
                    "event_ticker": event_data["ticker"],
                    "entry_at": event_data["entry_at"],
                    "control_rank": rank,
                    "control_ticker": control_data["ticker"],
                    "control_issuer_cik": control_data["issuer_cik"],
                    "control_available_at": control_data[availability_column],
                    "match_distance": float(control_data["match_distance"]),
                    "industry": event_data[industry_column],
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "Form4PurchaseFilter",
    "Form4ResearchError",
    "build_event_schedule",
    "build_filing_date_quarantine_schedule",
    "build_trailing_purchase_clusters",
    "classify_routine_opportunistic",
    "collapse_entry_session_events",
    "explode_event_owners",
    "filter_code_p_purchases",
    "match_event_controls",
]
