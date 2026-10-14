"""Deterministic SEC 10-K/10-Q document-change features.

The implementation follows the reproducible parts of Cohen, Malloy, and
Nguyen's *Lazy Prices*: compare a filing with the same fiscal period one year
earlier, remove numeric-heavy tables, and compute raw term-frequency cosine
and binary-set Jaccard similarity.  It intentionally has no learned text
representation and no outcome-dependent parameters.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from html.parser import HTMLParser
import math
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?")
SUPPORTED_FORMS = frozenset({"10-K", "10-Q"})
SUPPORTED_FISCAL_PERIODS = frozenset({"FY", "Q1", "Q2", "Q3"})
NUMERIC_TABLE_THRESHOLD = 0.15


class FilingChangeError(ValueError):
    """Raised when filing text or metadata violates the frozen contract."""


@dataclass(frozen=True)
class CleanFilingText:
    text: str
    tokens: tuple[str, ...]
    input_bytes: int
    visible_characters: int
    token_count: int
    unique_token_count: int
    tables_seen: int
    numeric_tables_removed: int
    nonnumeric_tables_retained: int


class _VisibleFilingParser(HTMLParser):
    """Extract visible text while buffering tables before deciding to retain."""

    _SUPPRESSED = frozenset(
        {
            "script",
            "style",
            "noscript",
            "object",
            "svg",
            "canvas",
            "math",
            "ix:hidden",
        }
    )
    _BOUNDARIES = frozenset(
        {
            "address",
            "article",
            "br",
            "div",
            "footer",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "header",
            "li",
            "p",
            "section",
            "td",
            "th",
            "tr",
        }
    )
    _VOID = frozenset(
        {
            "area",
            "base",
            "br",
            "col",
            "embed",
            "hr",
            "img",
            "input",
            "link",
            "meta",
            "param",
            "source",
            "track",
            "wbr",
        }
    )

    def __init__(self, *, numeric_table_threshold: float) -> None:
        super().__init__(convert_charrefs=True)
        self.numeric_table_threshold = numeric_table_threshold
        self.output: list[str] = []
        self._suppressed_depth = 0
        self._table_depth = 0
        self._table_buffer: list[str] = []
        self.tables_seen = 0
        self.numeric_tables_removed = 0
        self.nonnumeric_tables_retained = 0

    @staticmethod
    def _local_name(tag: str) -> str:
        return tag.strip().lower()

    def _append(self, value: str) -> None:
        if self._table_depth:
            self._table_buffer.append(value)
        else:
            self.output.append(value)

    @staticmethod
    def _hidden_attributes(attrs: list[tuple[str, str | None]]) -> bool:
        normalized = {str(key).lower(): value for key, value in attrs}
        if "hidden" in normalized:
            return True
        if str(normalized.get("aria-hidden", "")).strip().lower() == "true":
            return True
        style = re.sub(r"\s+", "", str(normalized.get("style", "")).lower())
        return "display:none" in style or "visibility:hidden" in style

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        tag = self._local_name(tag)
        if self._suppressed_depth:
            if tag not in self._VOID:
                self._suppressed_depth += 1
            return
        if tag in self._SUPPRESSED or self._hidden_attributes(attrs):
            self._suppressed_depth = 1
            return
        if tag == "table":
            if self._table_depth == 0:
                self.tables_seen += 1
                self._table_buffer = []
            self._table_depth += 1
            return
        if tag in self._BOUNDARIES:
            self._append(" ")

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        tag = self._local_name(tag)
        if self._suppressed_depth or tag in self._SUPPRESSED or self._hidden_attributes(attrs):
            return
        if tag == "table":
            self.tables_seen += 1
            self.nonnumeric_tables_retained += 1
            return
        if tag in self._BOUNDARIES:
            self._append(" ")

    def handle_endtag(self, tag: str) -> None:
        tag = self._local_name(tag)
        if self._suppressed_depth:
            self._suppressed_depth -= 1
            return
        if tag == "table" and self._table_depth:
            self._table_depth -= 1
            if self._table_depth == 0:
                table_text = "".join(self._table_buffer)
                nonspace = [character for character in table_text if not character.isspace()]
                ratio = (
                    sum(character.isdigit() for character in nonspace) / len(nonspace)
                    if nonspace
                    else 0.0
                )
                if ratio > self.numeric_table_threshold:
                    self.numeric_tables_removed += 1
                else:
                    self.output.extend((" ", table_text, " "))
                    self.nonnumeric_tables_retained += 1
                self._table_buffer = []
            return
        if tag in self._BOUNDARIES:
            self._append(" ")

    def close(self) -> None:
        super().close()
        if self._suppressed_depth:
            raise FilingChangeError("unclosed suppressed HTML subtree")
        if self._table_depth:
            raise FilingChangeError("unclosed HTML table")

    def handle_data(self, data: str) -> None:
        if self._suppressed_depth:
            return
        self._append(data)


class _FiscalIdentityParser(HTMLParser):
    """Read DEI fiscal identity values from inline-XBRL primary documents."""

    _FIELDS = {
        "documentfiscalyearfocus": "fiscal_year",
        "documentfiscalperiodfocus": "fiscal_period",
        "documentperiodenddate": "period_end",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._capture: tuple[str, str] | None = None
        self._depth = 0
        self._buffer: list[str] = []
        self.values: dict[str, set[str]] = {value: set() for value in self._FIELDS.values()}

    @classmethod
    def _field_for_tag(
        cls, tag: str, attrs: list[tuple[str, str | None]]
    ) -> str | None:
        tag_lower = tag.lower()
        local = tag_lower.split(":")[-1]
        if local in cls._FIELDS:
            return cls._FIELDS[local]
        if tag_lower in {"ix:nonnumeric", "ix:nonfraction"}:
            name = dict(attrs).get("name") or ""
            local_name = name.lower().split(":")[-1]
            return cls._FIELDS.get(local_name)
        return None

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if self._capture is not None:
            if tag.lower() not in _VisibleFilingParser._VOID:
                self._depth += 1
            return
        field = self._field_for_tag(tag, attrs)
        if field is not None:
            self._capture = (tag.lower(), field)
            self._depth = 1
            self._buffer = []

    def handle_endtag(self, tag: str) -> None:
        if self._capture is None:
            return
        self._depth -= 1
        if self._depth == 0:
            _, field = self._capture
            value = " ".join("".join(self._buffer).split())
            if value:
                self.values[field].add(value)
            self._capture = None
            self._buffer = []

    def handle_data(self, data: str) -> None:
        if self._capture is not None:
            self._buffer.append(data)

    def close(self) -> None:
        super().close()
        if self._capture is not None or self._depth:
            raise FilingChangeError("unclosed DEI fiscal identity element")


def _decode_html(content: bytes | str) -> tuple[str, int]:
    if isinstance(content, str):
        return content, len(content.encode("utf-8"))
    if not isinstance(content, bytes) or not content:
        raise FilingChangeError("primary document must be nonempty bytes or text")
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            return content.decode(encoding), len(content)
        except UnicodeDecodeError:
            continue
    raise FilingChangeError("primary document cannot be decoded")  # pragma: no cover


def extract_fiscal_identity(content: bytes | str) -> dict[str, Any]:
    """Extract a unique DEI fiscal year, period, and period end from inline XBRL."""

    html, _ = _decode_html(content)
    parser = _FiscalIdentityParser()
    parser.feed(html)
    parser.close()
    conflicts = {key: sorted(values) for key, values in parser.values.items() if len(values) > 1}
    if conflicts:
        raise FilingChangeError(f"conflicting DEI fiscal identity: {conflicts}")
    raw = {key: next(iter(values), None) for key, values in parser.values.items()}
    year_match = re.search(r"(?:19|20)\d{2}", raw["fiscal_year"] or "")
    fiscal_year = int(year_match.group()) if year_match else None
    period = (raw["fiscal_period"] or "").strip().upper() or None
    if period not in SUPPORTED_FISCAL_PERIODS:
        period = None
    period_end = pd.to_datetime(raw["period_end"], errors="coerce")
    return {
        "fiscal_year": fiscal_year,
        "fiscal_period": period,
        "period_end": None if pd.isna(period_end) else period_end.normalize(),
        "fiscal_identity_complete": bool(fiscal_year is not None and period is not None),
    }


def clean_primary_filing(
    content: bytes | str,
    *,
    numeric_table_threshold: float = NUMERIC_TABLE_THRESHOLD,
) -> CleanFilingText:
    """Strip a primary filing with a frozen, auditable HTML procedure."""

    if not 0.0 <= numeric_table_threshold <= 1.0:
        raise FilingChangeError("numeric_table_threshold must be between zero and one")
    html, input_bytes = _decode_html(content)
    parser = _VisibleFilingParser(numeric_table_threshold=numeric_table_threshold)
    parser.feed(html)
    parser.close()
    normalized = unicodedata.normalize("NFKC", " ".join("".join(parser.output).split())).lower()
    tokens = tuple(TOKEN_PATTERN.findall(normalized))
    return CleanFilingText(
        text=normalized,
        tokens=tokens,
        input_bytes=input_bytes,
        visible_characters=len(normalized),
        token_count=len(tokens),
        unique_token_count=len(set(tokens)),
        tables_seen=parser.tables_seen,
        numeric_tables_removed=parser.numeric_tables_removed,
        nonnumeric_tables_retained=parser.nonnumeric_tables_retained,
    )


def raw_tf_cosine(left: Sequence[str], right: Sequence[str]) -> float:
    """Cosine similarity of raw term-frequency vectors over the pair union."""

    if not left or not right:
        raise FilingChangeError("cosine requires two nonempty token sequences")
    left_counts = Counter(left)
    right_counts = Counter(right)
    dot = sum(count * right_counts.get(term, 0) for term, count in left_counts.items())
    left_norm = math.sqrt(sum(count * count for count in left_counts.values()))
    right_norm = math.sqrt(sum(count * count for count in right_counts.values()))
    result = dot / (left_norm * right_norm)
    return float(min(1.0, max(0.0, result)))


def raw_tf_cosine_counts(
    left_counts: Mapping[str, int], right_counts: Mapping[str, int]
) -> float:
    """Cosine similarity from persisted raw term-count mappings."""

    if not left_counts or not right_counts:
        raise FilingChangeError("cosine requires two nonempty count mappings")
    if any(not isinstance(value, int) or value <= 0 for value in left_counts.values()):
        raise FilingChangeError("left term counts must be positive integers")
    if any(not isinstance(value, int) or value <= 0 for value in right_counts.values()):
        raise FilingChangeError("right term counts must be positive integers")
    dot = sum(count * right_counts.get(term, 0) for term, count in left_counts.items())
    left_norm = math.sqrt(sum(count * count for count in left_counts.values()))
    right_norm = math.sqrt(sum(count * count for count in right_counts.values()))
    return float(min(1.0, max(0.0, dot / (left_norm * right_norm))))


def binary_jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    """Jaccard similarity of the unique terms in two documents."""

    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    if not union:
        raise FilingChangeError("Jaccard requires at least one token")
    return float(len(left_set & right_set) / len(union))


def binary_jaccard_counts(
    left_counts: Mapping[str, int], right_counts: Mapping[str, int]
) -> float:
    """Binary Jaccard similarity from persisted term-count mappings."""

    if any(not isinstance(value, int) or value <= 0 for value in left_counts.values()):
        raise FilingChangeError("left term counts must be positive integers")
    if any(not isinstance(value, int) or value <= 0 for value in right_counts.values()):
        raise FilingChangeError("right term counts must be positive integers")
    union = set(left_counts) | set(right_counts)
    if not union:
        raise FilingChangeError("Jaccard requires at least one term")
    return float(len(set(left_counts) & set(right_counts)) / len(union))


def similarity_record(
    current: CleanFilingText,
    prior: CleanFilingText,
    *,
    minimum_tokens: int = 1_000,
) -> dict[str, Any]:
    """Return the two frozen similarity candidates and eligibility diagnostics."""

    if minimum_tokens <= 0:
        raise FilingChangeError("minimum_tokens must be positive")
    eligible = current.token_count >= minimum_tokens and prior.token_count >= minimum_tokens
    record: dict[str, Any] = {
        "current_token_count": current.token_count,
        "prior_token_count": prior.token_count,
        "minimum_tokens": minimum_tokens,
        "text_eligible": eligible,
        "cosine_similarity": np.nan,
        "jaccard_similarity": np.nan,
    }
    if eligible:
        record["cosine_similarity"] = raw_tf_cosine(current.tokens, prior.tokens)
        record["jaccard_similarity"] = binary_jaccard(current.tokens, prior.tokens)
    return record


def build_strict_fiscal_pairs(
    filings: pd.DataFrame,
    *,
    minimum_report_gap_days: int = 300,
    maximum_report_gap_days: int = 430,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair unique original filings to the prior fiscal year's same period.

    Fiscal year and period must have been extracted from filing content.  The
    function never guesses fiscal identity from a calendar month.
    """

    required = {
        "cik",
        "accession",
        "form",
        "accepted_at",
        "report_date",
        "fiscal_year",
        "fiscal_period",
    }
    missing = required.difference(filings.columns)
    if missing:
        raise FilingChangeError(f"filings missing columns: {sorted(missing)}")
    frame = (
        filings.loc[:, sorted(required | ({"ticker"} & set(filings.columns)))]
        .copy()
        .reset_index(drop=True)
    )
    frame["form"] = frame["form"].astype("string").str.strip().str.upper()
    frame["accession"] = frame["accession"].astype("string").str.strip()
    frame["accepted_at"] = pd.to_datetime(frame["accepted_at"], utc=True, errors="coerce")
    frame["report_date"] = pd.to_datetime(frame["report_date"], errors="coerce").dt.normalize()
    frame["fiscal_year"] = pd.to_numeric(frame["fiscal_year"], errors="coerce").astype("Int64")
    frame["fiscal_period"] = frame["fiscal_period"].astype("string").str.strip().str.upper()
    frame["cik"] = pd.to_numeric(frame["cik"], errors="coerce").astype("Int64")

    reason = pd.Series(pd.NA, index=frame.index, dtype="string")
    reason.loc[~frame["form"].isin(SUPPORTED_FORMS)] = "unsupported_or_amended_form"
    reason.loc[frame[["cik", "accession", "accepted_at", "report_date"]].isna().any(axis=1)] = (
        "missing_filing_identity_or_clock"
    )
    valid_period = frame["fiscal_period"].isin(SUPPORTED_FISCAL_PERIODS)
    valid_form_period = (
        (frame["form"].eq("10-K") & frame["fiscal_period"].eq("FY"))
        | (frame["form"].eq("10-Q") & frame["fiscal_period"].isin({"Q1", "Q2", "Q3"}))
    )
    reason.loc[reason.isna() & (~valid_period | frame["fiscal_year"].isna())] = (
        "missing_fiscal_identity"
    )
    reason.loc[reason.isna() & ~valid_form_period] = "form_fiscal_period_conflict"

    duplicate_key = ["cik", "form", "fiscal_year", "fiscal_period"]
    duplicates = frame.loc[reason.isna()].duplicated(duplicate_key, keep=False)
    duplicate_indexes = frame.loc[reason.isna()].index[duplicates]
    reason.loc[duplicate_indexes] = "duplicate_original_fiscal_filing"

    eligible = frame.loc[reason.isna()].copy()
    lookup: dict[tuple[int, str, int, str], pd.Series] = {}
    for _, row in eligible.iterrows():
        lookup[(int(row.cik), str(row.form), int(row.fiscal_year), str(row.fiscal_period))] = row

    pairs: list[dict[str, Any]] = []
    for index, current in eligible.iterrows():
        prior_key = (
            int(current.cik),
            str(current.form),
            int(current.fiscal_year) - 1,
            str(current.fiscal_period),
        )
        prior = lookup.get(prior_key)
        if prior is None:
            reason.loc[index] = "prior_fiscal_year_comparator_missing"
            continue
        gap = int((current.report_date - prior.report_date).days)
        if not minimum_report_gap_days <= gap <= maximum_report_gap_days:
            reason.loc[index] = "report_date_gap_outside_contract"
            continue
        if not prior.accepted_at < current.accepted_at:
            reason.loc[index] = "prior_acceptance_not_strictly_earlier"
            continue
        record: dict[str, Any] = {
            "cik": int(current.cik),
            "form": str(current.form),
            "fiscal_year": int(current.fiscal_year),
            "fiscal_period": str(current.fiscal_period),
            "current_accession": str(current.accession),
            "prior_accession": str(prior.accession),
            "current_accepted_at": current.accepted_at,
            "prior_accepted_at": prior.accepted_at,
            "current_report_date": current.report_date,
            "prior_report_date": prior.report_date,
            "report_gap_days": gap,
        }
        if "ticker" in frame.columns:
            record["ticker"] = current.ticker
        pairs.append(record)

    exclusions = frame.loc[reason.notna()].copy()
    exclusions["reason"] = reason.loc[exclusions.index]
    pair_columns = [
        "cik",
        "ticker",
        "form",
        "fiscal_year",
        "fiscal_period",
        "current_accession",
        "prior_accession",
        "current_accepted_at",
        "prior_accepted_at",
        "current_report_date",
        "prior_report_date",
        "report_gap_days",
    ]
    if "ticker" not in frame.columns:
        pair_columns.remove("ticker")
    return (
        pd.DataFrame(pairs, columns=pair_columns).sort_values(
            ["current_accepted_at", "cik", "current_accession"]
        ).reset_index(drop=True),
        exclusions.reset_index(drop=True),
    )


def assign_release_month_quintiles(
    scores: pd.DataFrame,
    *,
    score_column: str,
    minimum_monthly_observations: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign paper-style quintiles using the completed release-month cohort.

    A filing accepted in month ``m`` is ranked only after month ``m`` closes and
    receives a formation month of ``m+1``.  Breakpoints and counts are retained
    in an explicit ledger.
    """

    required = {"current_accession", "current_accepted_at", score_column}
    missing = required.difference(scores.columns)
    if missing:
        raise FilingChangeError(f"scores missing columns: {sorted(missing)}")
    if minimum_monthly_observations < 5:
        raise FilingChangeError("minimum_monthly_observations must be at least five")
    frame = scores.copy()
    frame["current_accepted_at"] = pd.to_datetime(
        frame["current_accepted_at"], utc=True, errors="coerce"
    )
    frame[score_column] = pd.to_numeric(frame[score_column], errors="coerce")
    finite = np.isfinite(frame[score_column]) & frame["current_accepted_at"].notna()
    frame = frame.loc[finite].copy()
    local = frame["current_accepted_at"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    frame["release_month"] = local.dt.to_period("M")
    frame["formation_month"] = frame["release_month"] + 1
    frame["similarity_quintile"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    frame["sort_eligible"] = False
    ledger: list[dict[str, Any]] = []
    for release_month, group in frame.groupby("release_month", sort=True):
        values = group[score_column]
        count = len(values)
        record: dict[str, Any] = {
            "score": score_column,
            "release_month": str(release_month),
            "formation_month": str(release_month + 1),
            "observations": count,
            "minimum_observations": minimum_monthly_observations,
            "eligible": count >= minimum_monthly_observations,
        }
        if count < minimum_monthly_observations:
            record.update({f"q{q:02d}": np.nan for q in (20, 40, 60, 80)})
            ledger.append(record)
            continue
        cutpoints = values.quantile([0.2, 0.4, 0.6, 0.8], interpolation="linear").to_numpy()
        if not np.all(np.isfinite(cutpoints)) or np.any(np.diff(cutpoints) <= 0):
            record["eligible"] = False
            record["reason"] = "non_strict_monthly_breakpoints"
            record.update({f"q{q:02d}": float(value) for q, value in zip((20, 40, 60, 80), cutpoints)})
            ledger.append(record)
            continue
        quintiles = np.searchsorted(cutpoints, values.to_numpy(), side="left") + 1
        frame.loc[group.index, "similarity_quintile"] = quintiles.astype("int8")
        frame.loc[group.index, "sort_eligible"] = True
        record.update({f"q{q:02d}": float(value) for q, value in zip((20, 40, 60, 80), cutpoints)})
        ledger.append(record)
    return frame.reset_index(drop=True), pd.DataFrame(ledger)
