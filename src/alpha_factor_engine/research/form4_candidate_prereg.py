"""Deterministic, return-blind Form 4 candidate construction.

The SEC transaction code ``P`` describes a purchase on an exchange *or from
another person*.  This module therefore emits ``code_p_purchase_candidate``
records and never upgrades them to confirmed open-market trades.  It stops
before prices or returns and keeps exact-clock and permanent-security gates
closed until separately acquired evidence is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from alpha_factor_engine.contracts import PointInTimeViolation, validate_unique
from alpha_factor_engine.data.sec_insider import daily_master_url
from alpha_factor_engine.research.form4 import classify_routine_opportunistic


class Form4CandidateError(ValueError):
    """Raised when a candidate census cannot be constructed unambiguously."""


@dataclass(frozen=True)
class Form4CandidateSpec:
    minimum_direct_purchase_usd: float = 25_000.0
    qualifying_role_pattern: str = r"officer|director"
    include_security_pattern: str = r"common|ordinary"
    exclude_security_pattern: str = (
        r"preferred|warrant|option|phantom|restricted|unit|depositary|"
        r"convertible|right"
    )
    aff10b5_effective_date: str = "2023-04-01"
    maximum_filing_lag_calendar_days: int = 5
    development_start: str = "2023-04-01"
    holdout_start: str = "2025-01-01"
    holdout_end: str = "2025-12-31"
    routine_lookback_years: int = 3


FALSE_VALUES = {"0", "false", "f", "n", "no"}
TRUE_VALUES = {"1", "true", "t", "y", "yes"}
ACCESSION_PATTERN = re.compile(r"^\d{10}-\d{2}-\d{6}$")
QUARTER_PATTERN = re.compile(r"^20\d{2}q[1-4]$")


def _require(frame: pd.DataFrame, columns: Iterable[str], *, name: str) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise Form4CandidateError(f"{name} missing columns: {sorted(missing)}")


def _strict_bool(values: pd.Series, *, name: str) -> pd.Series:
    text = values.astype("string").str.strip().str.lower()
    result = pd.Series(pd.NA, index=values.index, dtype="boolean")
    result.loc[text.isin(FALSE_VALUES)] = False
    result.loc[text.isin(TRUE_VALUES)] = True
    missing = values.isna() | text.eq("")
    invalid = ~missing & result.isna()
    if invalid.any():
        raise Form4CandidateError(
            f"{name} contains invalid booleans: {values.loc[invalid].head(5).tolist()}"
        )
    return result


def _strict_number(values: pd.Series, *, name: str) -> pd.Series:
    parsed = pd.to_numeric(values, errors="coerce")
    present = values.notna() & values.astype("string").str.strip().ne("")
    invalid = present & (parsed.isna() | ~np.isfinite(parsed))
    if invalid.any():
        raise Form4CandidateError(
            f"{name} contains invalid numbers: {values.loc[invalid].head(5).tolist()}"
        )
    return parsed


def _pipe_unique(values: Iterable[object]) -> str:
    return "|".join(
        sorted(
            {
                str(value).strip()
                for value in values
                if pd.notna(value) and str(value).strip()
            }
        )
    )


def _normalize_date(values: pd.Series, *, name: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        raise Form4CandidateError(
            f"{name} contains missing or invalid dates: "
            f"{values.loc[parsed.isna()].head(5).tolist()}"
        )
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_convert("America/New_York").dt.tz_localize(None)
    return parsed.dt.normalize()


def _validate_accessions(values: pd.Series, *, name: str) -> None:
    text = values.astype("string")
    invalid = values.isna() | ~text.str.fullmatch(ACCESSION_PATTERN)
    if invalid.any():
        raise Form4CandidateError(
            f"{name} contains invalid accessions: {values.loc[invalid].head(5).tolist()}"
        )


def extract_quarter_candidates(
    submissions: pd.DataFrame,
    reporting_owners: pd.DataFrame,
    nonderivative_transactions: pd.DataFrame,
    *,
    quarter: str,
    spec: Form4CandidateSpec | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Mapping[str, int | str | bool]]:
    """Return accession candidates, owner-history rows, and a filter audit.

    Primary candidate value uses only direct ownership (``D``).  Indirect rows
    are retained as a diagnostic amount but never cause an accession to pass.
    Original Form 4 accessions remain as-filed; Form 4/A never creates a second
    event and is not allowed to retroactively select or remove an original.
    """

    cfg = spec or Form4CandidateSpec()
    normalized_quarter = str(quarter).strip().lower()
    if not QUARTER_PATTERN.fullmatch(normalized_quarter):
        raise ValueError(f"invalid quarter: {quarter!r}")
    if cfg.minimum_direct_purchase_usd < 0:
        raise ValueError("minimum_direct_purchase_usd must be non-negative")
    if cfg.maximum_filing_lag_calendar_days < 0:
        raise ValueError("maximum_filing_lag_calendar_days must be non-negative")

    submission_required = [
        "ACCESSION_NUMBER",
        "FILING_DATE",
        "DOCUMENT_TYPE",
        "ISSUERCIK",
        "ISSUERNAME",
        "ISSUERTRADINGSYMBOL",
        "DATE_OF_ORIG_SUB",
    ]
    owner_required = [
        "ACCESSION_NUMBER",
        "RPTOWNERCIK",
        "RPTOWNERNAME",
        "RPTOWNER_RELATIONSHIP",
        "RPTOWNER_TITLE",
    ]
    transaction_required = [
        "ACCESSION_NUMBER",
        "NONDERIV_TRANS_SK",
        "SECURITY_TITLE",
        "TRANS_DATE",
        "TRANS_FORM_TYPE",
        "TRANS_CODE",
        "EQUITY_SWAP_INVOLVED",
        "TRANS_TIMELINESS",
        "TRANS_SHARES",
        "TRANS_PRICEPERSHARE",
        "TRANS_ACQUIRED_DISP_CD",
        "DIRECT_INDIRECT_OWNERSHIP",
    ]
    _require(submissions, submission_required, name="SUBMISSION")
    _require(reporting_owners, owner_required, name="REPORTINGOWNER")
    _require(
        nonderivative_transactions,
        transaction_required,
        name="NONDERIV_TRANS",
    )
    validate_unique(submissions, ["ACCESSION_NUMBER"], f"{quarter} submissions")
    validate_unique(
        reporting_owners,
        ["ACCESSION_NUMBER", "RPTOWNERCIK"],
        f"{quarter} reporting owners",
    )
    validate_unique(
        nonderivative_transactions,
        ["ACCESSION_NUMBER", "NONDERIV_TRANS_SK"],
        f"{quarter} non-derivative transactions",
    )
    _validate_accessions(submissions["ACCESSION_NUMBER"], name="SUBMISSION")

    submission = submissions.copy()
    submission["FILING_DATE"] = _normalize_date(
        submission["FILING_DATE"], name="FILING_DATE"
    )
    if "AFF10B5ONE" in submission:
        submission["_aff10b5"] = _strict_bool(
            submission["AFF10B5ONE"], name="AFF10B5ONE"
        )
        aff_column_present = True
    else:
        submission["_aff10b5"] = pd.Series(
            pd.NA, index=submission.index, dtype="boolean"
        )
        aff_column_present = False
    quarter_year = int(normalized_quarter[:4])
    if quarter_year >= 2023 and not aff_column_present:
        raise Form4CandidateError(
            f"{normalized_quarter} is missing the required AFF10B5ONE schema field"
        )
    original = submission.loc[
        submission["DOCUMENT_TYPE"].astype("string").str.strip().eq("4")
    ].copy()
    amendments = submission.loc[
        submission["DOCUMENT_TYPE"].astype("string").str.strip().eq("4/A")
    ].copy()

    owner = reporting_owners.copy()
    missing_owner = owner["RPTOWNERCIK"].isna() | owner["RPTOWNERCIK"].astype(
        "string"
    ).str.strip().eq("")
    if missing_owner.any():
        raise Form4CandidateError("REPORTINGOWNER contains missing owner CIK")
    owner["_qualifying"] = owner["RPTOWNER_RELATIONSHIP"].astype(
        "string"
    ).str.contains(cfg.qualifying_role_pattern, case=False, regex=True, na=False)
    for source, target in [
        ("RPTOWNERCIK", "_q_owner_cik"),
        ("RPTOWNERNAME", "_q_owner_name"),
        ("RPTOWNER_RELATIONSHIP", "_q_owner_role"),
        ("RPTOWNER_TITLE", "_q_owner_title"),
    ]:
        owner[target] = owner[source].where(owner["_qualifying"])
    owner_summary = (
        owner.groupby("ACCESSION_NUMBER", sort=True)
        .agg(
            reporting_owner_count=("RPTOWNERCIK", "nunique"),
            reporting_owner_ciks=("RPTOWNERCIK", _pipe_unique),
            qualifying_owner_count=("_q_owner_cik", "nunique"),
            qualifying_owner_ciks=("_q_owner_cik", _pipe_unique),
            qualifying_owner_names=("_q_owner_name", _pipe_unique),
            qualifying_owner_roles=("_q_owner_role", _pipe_unique),
            qualifying_owner_titles=("_q_owner_title", _pipe_unique),
        )
        .reset_index()
    )

    transaction = nonderivative_transactions.copy()
    transaction["_swap"] = _strict_bool(
        transaction["EQUITY_SWAP_INVOLVED"], name="EQUITY_SWAP_INVOLVED"
    )
    transaction["_shares"] = _strict_number(
        transaction["TRANS_SHARES"], name="TRANS_SHARES"
    )
    transaction["_price"] = _strict_number(
        transaction["TRANS_PRICEPERSHARE"], name="TRANS_PRICEPERSHARE"
    )
    raw_trans_date = transaction["TRANS_DATE"]
    transaction["_trans_date"] = pd.to_datetime(raw_trans_date, errors="coerce")
    invalid_date = (
        raw_trans_date.notna()
        & raw_trans_date.astype("string").str.strip().ne("")
        & transaction["_trans_date"].isna()
    )
    if invalid_date.any():
        raise Form4CandidateError(
            "TRANS_DATE contains invalid non-empty values: "
            f"{raw_trans_date.loc[invalid_date].head(5).tolist()}"
        )
    transaction["_trans_date"] = transaction["_trans_date"].dt.normalize()
    title = transaction["SECURITY_TITLE"].astype("string")
    ownership = transaction["DIRECT_INDIRECT_OWNERSHIP"].astype(
        "string"
    ).str.strip().str.upper()
    base = (
        transaction["TRANS_FORM_TYPE"].astype("string").str.strip().eq("4")
        & transaction["TRANS_CODE"].astype("string").str.strip().str.upper().eq("P")
        & transaction["TRANS_ACQUIRED_DISP_CD"]
        .astype("string")
        .str.strip()
        .str.upper()
        .eq("A")
        & transaction["_swap"].eq(False).fillna(False)
        & transaction["_shares"].gt(0)
        & transaction["_price"].gt(0)
        & transaction["_trans_date"].notna()
        & title.str.contains(
            cfg.include_security_pattern, case=False, regex=True, na=False
        )
        & ~title.str.contains(
            cfg.exclude_security_pattern, case=False, regex=True, na=False
        )
        & ownership.isin(["D", "I"])
    )
    base_tx = transaction.loc[base].copy()
    base_tx["_ownership"] = ownership.loc[base]
    base_tx["_purchase_usd"] = base_tx["_shares"] * base_tx["_price"]
    base_tx["_direct_usd"] = base_tx["_purchase_usd"].where(
        base_tx["_ownership"].eq("D"), 0.0
    )
    base_tx["_indirect_usd"] = base_tx["_purchase_usd"].where(
        base_tx["_ownership"].eq("I"), 0.0
    )
    base_tx["_direct_row"] = base_tx["_ownership"].eq("D").astype(int)
    base_tx["_indirect_row"] = base_tx["_ownership"].eq("I").astype(int)
    base_tx["_late"] = base_tx["TRANS_TIMELINESS"].astype(
        "string"
    ).str.strip().str.upper().eq("L")
    tx_summary = (
        base_tx.groupby("ACCESSION_NUMBER", sort=True)
        .agg(
            direct_purchase_usd=("_direct_usd", "sum"),
            indirect_purchase_usd=("_indirect_usd", "sum"),
            direct_transaction_rows=("_direct_row", "sum"),
            indirect_transaction_rows=("_indirect_row", "sum"),
            transaction_date_min=("_trans_date", "min"),
            transaction_date_max=("_trans_date", "max"),
            any_late_transaction=("_late", "any"),
            security_titles=("SECURITY_TITLE", _pipe_unique),
        )
        .reset_index()
    )

    joined = original.merge(
        owner_summary, on="ACCESSION_NUMBER", how="left", validate="one_to_one"
    ).merge(tx_summary, on="ACCESSION_NUMBER", how="left", validate="one_to_one")
    numeric_zero = [
        "reporting_owner_count",
        "qualifying_owner_count",
        "direct_purchase_usd",
        "indirect_purchase_usd",
        "direct_transaction_rows",
        "indirect_transaction_rows",
    ]
    joined[numeric_zero] = joined[numeric_zero].fillna(0)
    for column in [
        "reporting_owner_ciks",
        "qualifying_owner_ciks",
        "qualifying_owner_names",
        "qualifying_owner_roles",
        "qualifying_owner_titles",
        "security_titles",
    ]:
        joined[column] = joined[column].fillna("").astype("string")
    joined["any_late_transaction"] = joined["any_late_transaction"].fillna(
        False
    ).astype(bool)

    has_tx = joined["transaction_date_min"].notna()
    impossible = has_tx & joined["transaction_date_max"].gt(joined["FILING_DATE"])
    # Real SEC flattened data contain a small number of chronologically
    # impossible as-filed rows.  They are observable source defects, so the
    # accession is quarantined and counted; its date is never repaired.
    joined["chronology_eligible"] = ~impossible
    joined["maximum_filing_lag_calendar_days"] = (
        joined["FILING_DATE"] - joined["transaction_date_min"]
    ).dt.days
    joined["structural_candidate"] = (
        joined["qualifying_owner_count"].gt(0)
        & joined["direct_transaction_rows"].gt(0)
        & joined["direct_purchase_usd"].ge(cfg.minimum_direct_purchase_usd)
        & joined["chronology_eligible"]
    )
    joined["owner_history_event"] = (
        joined["qualifying_owner_count"].gt(0)
        & joined["direct_transaction_rows"].gt(0)
        & joined["direct_purchase_usd"].gt(0)
        & joined["chronology_eligible"]
    )
    joined["timeliness_eligible"] = (
        ~joined["any_late_transaction"]
        & joined["maximum_filing_lag_calendar_days"].le(
            cfg.maximum_filing_lag_calendar_days
        )
    ).fillna(False)
    aff_effective = pd.Timestamp(cfg.aff10b5_effective_date)
    joined["aff10b5_required"] = joined["FILING_DATE"].ge(aff_effective)
    joined["aff10b5_explicit_false"] = joined["_aff10b5"].eq(False).fillna(False)
    joined["aff10b5_eligible"] = (
        ~joined["aff10b5_required"] | joined["aff10b5_explicit_false"]
    )
    holdout_start = pd.Timestamp(cfg.holdout_start)
    holdout_end = pd.Timestamp(cfg.holdout_end)
    development_start = pd.Timestamp(cfg.development_start)
    joined["evaluation_phase"] = np.select(
        [
            joined["FILING_DATE"].between(holdout_start, holdout_end),
            joined["FILING_DATE"].ge(development_start)
            & joined["FILING_DATE"].lt(holdout_start),
        ],
        ["holdout_2025", "development_2023_2024"],
        default="history_only",
    )
    joined["preclock_evaluation_candidate"] = (
        joined["structural_candidate"]
        & joined["timeliness_eligible"]
        & joined["aff10b5_eligible"]
        & joined["evaluation_phase"].isin(
            ["development_2023_2024", "holdout_2025"]
        )
    )
    joined["source_quarter"] = normalized_quarter
    joined["signal_type"] = "form4_code_p_purchase_candidate"
    joined["clock_complete"] = False
    joined["security_mapping_complete"] = False
    joined["research_eligible"] = False

    def reasons(row: pd.Series) -> str:
        values: list[str] = []
        if not bool(row["structural_candidate"]):
            values.append("fails_structural_direct_candidate")
        if row["evaluation_phase"] == "history_only":
            values.append("history_only")
        if not bool(row["timeliness_eligible"]):
            values.append("late_or_stale_filing")
        if not bool(row["aff10b5_eligible"]):
            values.append("10b5_true_or_missing")
        values.extend(["exact_public_clock_missing", "permanent_security_mapping_missing"])
        return "|".join(values)

    candidates = joined.loc[joined["structural_candidate"]].copy()
    candidates["blocking_reasons"] = candidates.apply(reasons, axis=1)
    candidates = candidates.rename(
        columns={
            "ACCESSION_NUMBER": "accession",
            "FILING_DATE": "filing_date",
            "ISSUERCIK": "issuer_cik",
            "ISSUERNAME": "issuer_name",
            "ISSUERTRADINGSYMBOL": "ticker",
            "DOCUMENT_TYPE": "document_type",
        }
    )
    candidates["event_id"] = candidates["accession"].astype("string")
    candidates["ticker"] = candidates["ticker"].astype("string").str.upper()
    candidate_columns = [
        "event_id",
        "accession",
        "source_quarter",
        "filing_date",
        "document_type",
        "issuer_cik",
        "issuer_name",
        "ticker",
        "signal_type",
        "direct_purchase_usd",
        "indirect_purchase_usd",
        "direct_transaction_rows",
        "indirect_transaction_rows",
        "security_titles",
        "transaction_date_min",
        "transaction_date_max",
        "maximum_filing_lag_calendar_days",
        "any_late_transaction",
        "reporting_owner_count",
        "reporting_owner_ciks",
        "qualifying_owner_count",
        "qualifying_owner_ciks",
        "qualifying_owner_names",
        "qualifying_owner_roles",
        "qualifying_owner_titles",
        "aff10b5_required",
        "aff10b5_explicit_false",
        "timeliness_eligible",
        "aff10b5_eligible",
        "evaluation_phase",
        "preclock_evaluation_candidate",
        "clock_complete",
        "security_mapping_complete",
        "research_eligible",
        "blocking_reasons",
    ]
    candidates = candidates[candidate_columns].sort_values(
        ["filing_date", "accession"], kind="mergesort"
    )

    history_accessions = joined.loc[
        joined["owner_history_event"],
        [
            "ACCESSION_NUMBER",
            "FILING_DATE",
            "ISSUERCIK",
            "ISSUERTRADINGSYMBOL",
            "transaction_date_max",
            "qualifying_owner_ciks",
        ],
    ].copy()
    history_accessions["owner_cik"] = history_accessions[
        "qualifying_owner_ciks"
    ].astype("string").str.split("|")
    history = history_accessions.explode("owner_cik", ignore_index=True)
    history = history.loc[
        history["owner_cik"].notna()
        & history["owner_cik"].astype("string").str.strip().ne("")
    ].copy()
    filing_eod_local = history["FILING_DATE"].dt.tz_localize(
        "America/New_York"
    ) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    history["provisional_information_at"] = filing_eod_local.dt.tz_convert("UTC")
    history = history.rename(
        columns={
            "ACCESSION_NUMBER": "accession",
            "FILING_DATE": "filing_date",
            "ISSUERCIK": "issuer_cik",
            "ISSUERTRADINGSYMBOL": "ticker",
            "transaction_date_max": "transaction_date",
        }
    )
    history["source_quarter"] = normalized_quarter
    history["clock_status"] = "filing_date_eod_proxy_not_research_eligible"
    history = history[
        [
            "accession",
            "source_quarter",
            "filing_date",
            "provisional_information_at",
            "issuer_cik",
            "ticker",
            "owner_cik",
            "transaction_date",
            "clock_status",
        ]
    ].sort_values(["filing_date", "accession", "owner_cik"], kind="mergesort")
    validate_unique(history, ["accession", "owner_cik"], f"{quarter} owner history")

    audit: dict[str, int | str | bool] = {
        "quarter": normalized_quarter,
        "aff10b5_column_present": aff_column_present,
        "submission_rows": int(len(submission)),
        "original_form4_rows": int(len(original)),
        "amendment_form4a_rows": int(len(amendments)),
        "nonderivative_rows": int(len(transaction)),
        "base_code_p_common_direct_or_indirect_rows": int(len(base_tx)),
        "base_code_p_common_direct_rows": int(base_tx["_ownership"].eq("D").sum()),
        "base_code_p_common_indirect_rows": int(base_tx["_ownership"].eq("I").sum()),
        "owner_history_accessions": int(joined["owner_history_event"].sum()),
        "structural_candidate_accessions": int(len(candidates)),
        "preclock_evaluation_candidate_accessions": int(
            candidates["preclock_evaluation_candidate"].sum()
        ),
        "research_eligible_accessions": 0,
        "chronology_invalid_accessions": int(impossible.sum()),
        "chronology_invalid_examples_json": json.dumps(
            joined.loc[
                impossible,
                ["ACCESSION_NUMBER", "transaction_date_max", "FILING_DATE"],
            ]
            .head(5)
            .astype("string")
            .to_dict("records"),
            sort_keys=True,
        ),
    }
    return (
        candidates.reset_index(drop=True),
        history.reset_index(drop=True),
        audit,
    )


def classify_candidate_events(
    candidates: pd.DataFrame,
    owner_history: pd.DataFrame,
    *,
    classification_years: Sequence[int] = (2023, 2024, 2025),
    lookback_years: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Attach provisional three-year owner and event classifications.

    The result is explicitly provisional because the bulk files provide only a
    filing date.  Final classification must be recomputed from exact public
    clocks before any return is opened.
    """

    _require(
        candidates,
        ["accession", "filing_date", "qualifying_owner_ciks"],
        name="candidates",
    )
    _require(
        owner_history,
        [
            "accession",
            "owner_cik",
            "transaction_date",
            "provisional_information_at",
        ],
        name="owner_history",
    )
    validate_unique(candidates, ["accession"], "candidate accessions")
    validate_unique(owner_history, ["accession", "owner_cik"], "owner history")
    years = sorted({int(year) for year in classification_years})
    if not years:
        raise ValueError("classification_years must be non-empty")
    labels = classify_routine_opportunistic(
        owner_history.rename(
            columns={"provisional_information_at": "information_at"}
        )[["owner_cik", "transaction_date", "information_at"]],
        [f"{year}-01-01" for year in years],
        lookback_years=lookback_years,
    )
    labels["classification_year"] = labels["classification_at"].dt.year
    labels["clock_status"] = "provisional_filing_date_eod_proxy"
    labels["research_eligible"] = False

    event_owner = candidates[
        ["accession", "filing_date", "qualifying_owner_ciks"]
    ].copy()
    event_owner["owner_cik"] = event_owner["qualifying_owner_ciks"].astype(
        "string"
    ).str.split("|")
    event_owner = event_owner.explode("owner_cik", ignore_index=True)
    event_owner["classification_year"] = pd.to_datetime(
        event_owner["filing_date"]
    ).dt.year
    event_owner = event_owner.merge(
        labels[
            [
                "owner_cik",
                "classification_year",
                "classification",
                "routine_months",
                "active_years",
                "known_trade_count",
            ]
        ],
        on=["owner_cik", "classification_year"],
        how="left",
        validate="many_to_one",
    )
    event_owner["classification"] = event_owner["classification"].fillna(
        "not_classified"
    )

    def classify_event(group: pd.DataFrame) -> str:
        values = set(group["classification"].astype(str))
        if values == {"opportunistic"}:
            return "opportunistic"
        if values == {"routine"}:
            return "routine"
        if "unclassified" in values or "not_classified" in values:
            return "unclassified"
        return "mixed"

    event_label = (
        event_owner.groupby("accession", sort=True)
        .apply(classify_event, include_groups=False)
        .rename("provisional_event_classification")
        .reset_index()
    )
    event_counts = (
        event_owner.groupby("accession", sort=True)["classification"]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
    )
    for column in ["routine", "opportunistic", "unclassified", "not_classified"]:
        if column not in event_counts:
            event_counts[column] = 0
    event_counts = event_counts.rename(
        columns={
            "routine": "routine_owner_count",
            "opportunistic": "opportunistic_owner_count",
            "unclassified": "unclassified_owner_count",
            "not_classified": "not_classified_owner_count",
        }
    )
    result = candidates.merge(event_label, on="accession", how="left", validate="one_to_one")
    result = result.merge(event_counts, on="accession", how="left", validate="one_to_one")
    result["preliminary_primary_candidate"] = (
        result["preclock_evaluation_candidate"]
        & result["provisional_event_classification"].eq("opportunistic")
    )
    result["classification_clock_complete"] = False
    result["research_eligible"] = False
    result["blocking_reasons"] = result["blocking_reasons"].astype("string")
    result.loc[
        ~result["provisional_event_classification"].eq("opportunistic"),
        "blocking_reasons",
    ] = (
        result.loc[
            ~result["provisional_event_classification"].eq("opportunistic"),
            "blocking_reasons",
        ].str.cat(pd.Series("not_provisionally_opportunistic", index=result.index), sep="|")
    )
    return (
        result.sort_values(["filing_date", "accession"], kind="mergesort").reset_index(drop=True),
        labels.sort_values(["classification_at", "owner_cik"], kind="mergesort").reset_index(drop=True),
        event_owner.sort_values(["filing_date", "accession", "owner_cik"], kind="mergesort").reset_index(drop=True),
    )


def validate_global_lineage(
    candidates: pd.DataFrame, owner_history: pd.DataFrame
) -> None:
    """Reject cross-quarter duplicate accessions or owner-event keys."""

    validate_unique(candidates, ["accession"], "all-quarter candidates")
    validate_unique(
        owner_history,
        ["accession", "owner_cik"],
        "all-quarter owner-history events",
    )
    candidate_quarters = candidates.groupby("accession")["source_quarter"].nunique()
    history_quarters = owner_history.groupby("accession")["source_quarter"].nunique()
    if candidate_quarters.gt(1).any() or history_quarters.gt(1).any():
        raise Form4CandidateError("an accession appears in more than one source quarter")


def _complete_submission_url(issuer_cik: object, accession: str) -> str:
    cik = str(issuer_cik).strip()
    if not re.fullmatch(r"\d{1,10}", cik):
        raise Form4CandidateError(f"invalid issuer CIK: {issuer_cik!r}")
    compact = accession.replace("-", "")
    return (
        "https://www.sec.gov/Archives/edgar/data/"
        f"{int(cik)}/{compact}/{accession}.txt"
    )


def build_clock_acquisition_plans(
    candidates: pd.DataFrame,
    owner_history: pd.DataFrame,
    *,
    daily_search_business_days_after: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build exact accession and cross-quarter daily-index acquisition plans."""

    if daily_search_business_days_after < 1:
        raise ValueError("daily_search_business_days_after must be positive")
    _require(
        candidates,
        [
            "accession",
            "issuer_cik",
            "ticker",
            "filing_date",
            "preclock_evaluation_candidate",
            "qualifying_owner_ciks",
        ],
        name="candidates",
    )
    _require(
        owner_history,
        ["accession", "issuer_cik", "ticker", "filing_date", "owner_cik"],
        name="owner_history",
    )
    evaluation = candidates.loc[candidates["preclock_evaluation_candidate"]].copy()
    evaluation_owners: set[str] = set()
    for value in evaluation["qualifying_owner_ciks"]:
        evaluation_owners.update(
            token.strip() for token in str(value).split("|") if token.strip()
        )
    history = owner_history.loc[
        owner_history["owner_cik"].astype("string").isin(evaluation_owners)
        & pd.to_datetime(owner_history["filing_date"]).lt(pd.Timestamp("2025-01-01"))
    ].copy()
    roles: dict[str, set[str]] = {}
    for accession in evaluation["accession"].astype(str):
        roles.setdefault(accession, set()).add("evaluation_candidate")
    for accession in history["accession"].astype(str):
        roles.setdefault(accession, set()).add("owner_classification_history")
    source = pd.concat(
        [
            evaluation[["accession", "issuer_cik", "ticker", "filing_date"]],
            history[["accession", "issuer_cik", "ticker", "filing_date"]],
        ],
        ignore_index=True,
    )
    if source.empty:
        raw = pd.DataFrame(
            columns=[
                "accession",
                "issuer_cik",
                "ticker",
                "filing_date",
                "required_for",
                "source_url",
                "cache_role",
                "status",
            ]
        )
        return raw, pd.DataFrame(), pd.DataFrame()
    conflict = source.groupby("accession").agg(
        issuer_count=("issuer_cik", "nunique"),
        filing_date_count=("filing_date", "nunique"),
    )
    if conflict["issuer_count"].gt(1).any() or conflict["filing_date_count"].gt(1).any():
        raise Form4CandidateError("clock plan has conflicting accession metadata")
    source = source.sort_values(["filing_date", "accession"], kind="mergesort").drop_duplicates(
        "accession", keep="first"
    )
    raw = source.copy()
    raw["required_for"] = raw["accession"].map(
        lambda value: "|".join(sorted(roles[str(value)]))
    )
    raw["source_url"] = [
        _complete_submission_url(cik, accession)
        for cik, accession in zip(raw["issuer_cik"], raw["accession"], strict=True)
    ]
    raw["cache_role"] = "raw_submission:" + raw["accession"].astype(str)
    raw["status"] = "required_not_acquired"
    raw["expected_clock_field"] = "<ACCEPTANCE-DATETIME>"

    window = raw[["accession", "filing_date"]].copy()
    window["filing_date"] = pd.to_datetime(window["filing_date"]).dt.normalize()
    window["search_start"] = window["filing_date"] - pd.offsets.BDay(1)
    window["search_end"] = window["filing_date"] + pd.offsets.BDay(
        daily_search_business_days_after
    )
    window["cross_quarter_required"] = True
    window["failure_policy"] = "extend_until_found_or_fail_closed"

    start = window["search_start"].min()
    end = window["search_end"].max()
    dates = pd.date_range(start, end, freq="B")
    daily = pd.DataFrame({"index_date": dates})
    daily["source_url"] = daily["index_date"].map(daily_master_url)
    daily["cache_role"] = "daily_master:" + daily["index_date"].dt.strftime(
        "%Y-%m-%d"
    )
    daily["status"] = "required_or_documented_not_published"
    daily["required_evidence"] = "bytes_sha256|http_last_modified|etag|retrieved_at"
    daily["note"] = (
        "weekday panel includes market/SEC holidays; a missing file must be "
        "recorded, never substituted"
    )
    return (
        raw.reset_index(drop=True),
        daily.reset_index(drop=True),
        window.sort_values(["filing_date", "accession"], kind="mergesort").reset_index(drop=True),
    )


def sha256_file(path: str | Path) -> str:
    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_closed_manifest(path: str | Path) -> bool:
    """Re-hash every record and verify output-directory closure."""

    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "form4_candidate_prereg_artifact_manifest_v1":
        raise Form4CandidateError("unsupported artifact manifest schema")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise Form4CandidateError("artifact manifest has no records")
    seen: set[Path] = set()
    for record in records:
        target = Path(str(record["path"])).resolve()
        if target in seen:
            raise Form4CandidateError(f"duplicate manifest path: {target}")
        seen.add(target)
        if not target.is_file() or target.is_symlink():
            raise Form4CandidateError(f"manifest target is not a regular file: {target}")
        if target.stat().st_size != int(record["bytes"]):
            raise Form4CandidateError(f"manifest byte mismatch: {target}")
        if sha256_file(target) != str(record["sha256"]):
            raise Form4CandidateError(f"manifest hash mismatch: {target}")
    expected_outputs = {
        item.resolve()
        for item in manifest_path.parent.iterdir()
        if item.is_file() and item.name != manifest_path.name
    }
    recorded_outputs = {
        Path(str(record["path"])).resolve()
        for record in records
        if record.get("role") == "output"
    }
    if expected_outputs != recorded_outputs:
        missing = sorted(str(path) for path in expected_outputs - recorded_outputs)
        extra = sorted(str(path) for path in recorded_outputs - expected_outputs)
        raise Form4CandidateError(
            f"artifact manifest output closure failed; missing={missing}, extra={extra}"
        )
    return True


__all__ = [
    "Form4CandidateError",
    "Form4CandidateSpec",
    "build_clock_acquisition_plans",
    "classify_candidate_events",
    "extract_quarter_candidates",
    "sha256_file",
    "validate_global_lineage",
    "verify_closed_manifest",
]
