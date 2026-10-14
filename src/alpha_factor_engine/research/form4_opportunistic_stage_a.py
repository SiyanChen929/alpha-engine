"""Fail-closed Stage A diagnostic for provisional Form 4 purchase classes.

The experiment is intentionally narrow.  It compares provisionally
opportunistic Form 4 purchase candidates with a routine-purchase negative
control during 2023-04-01 through 2024-12-31.  Candidate classification uses a
filing-date proxy, and prices come from retrospective current-ticker Yahoo
history.  Consequently every historical alpha verdict emitted here is
``INVALID`` regardless of the observed return.

The file's 2025 event metadata is read, but that cohort is never joined to
prices and no 2025 event return is computed. A separate freeze receipt
must hash the final configuration, this module, the outcome runner, candidates,
the acquired price panel, and its acquisition manifest before any candidate to
issuer-price or return join is allowed. Benchmark-calendar mapping is permitted
and audited before the freeze so incomplete SPY clocks fail closed.
"""

from __future__ import annotations

from datetime import UTC, datetime
import gzip
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_distribution
from scipy.stats import norm

CONFIG_SCHEMA = "form4_opportunistic_stage_a_config_v1"
PRICE_MANIFEST_SCHEMA = "form4_opportunistic_stage_a_price_manifest_v1"
DESIGN_RECEIPT_SCHEMA = "form4_opportunistic_stage_a_design_receipt_v1"
FREEZE_RECEIPT_SCHEMA = "form4_opportunistic_stage_a_freeze_receipt_v1"
SUMMARY_SCHEMA = "form4_opportunistic_stage_a_summary_v1"
ARTIFACT_MANIFEST_SCHEMA = "form4_opportunistic_stage_a_artifact_manifest_v1"

DEVELOPMENT_START = pd.Timestamp("2023-04-01")
DEVELOPMENT_END_EXCLUSIVE = pd.Timestamp("2025-01-01")
HOLDOUT_START = pd.Timestamp("2025-01-01")
HOLDOUT_END_EXCLUSIVE = pd.Timestamp("2026-01-01")
ACQUISITION_START = "2021-01-01"
ACQUISITION_END = "2025-02-28"
BENCHMARK = "SPY"
CANONICAL_HUMAN_PROTOCOL = "configs/form4_opportunistic_stage_a_v1.json"
CANONICAL_RUNTIME_CONFIG = "configs/form4_opportunistic_stage_a_v1_runtime.json"
CANONICAL_CORE_CODE = (
    "src/alpha_factor_engine/research/form4_opportunistic_stage_a.py"
)
CANONICAL_OUTCOME_RUNNER = "scripts/run_form4_opportunistic_stage_a.py"
CANONICAL_ACQUISITION_SCRIPT = (
    "scripts/acquire_form4_opportunistic_stage_a_prices.py"
)
CANONICAL_DESIGN_RECEIPT = (
    "artifacts/form4_opportunistic_stage_a_v1/design_receipt.json"
)
CANONICAL_ACQUISITION_DIRECTORY = (
    "artifacts/form4_opportunistic_stage_a_v1/acquisition"
)
CANONICAL_FREEZE_RECEIPT = (
    "artifacts/form4_opportunistic_stage_a_v1/freeze_receipt.json"
)
CANONICAL_RUN_DIRECTORY = "artifacts/form4_opportunistic_stage_a_v1/run"
CANONICAL_PRICE_PANEL = f"{CANONICAL_ACQUISITION_DIRECTORY}/prices.parquet"
CANONICAL_PRICE_MANIFEST = (
    f"{CANONICAL_ACQUISITION_DIRECTORY}/price_manifest.json"
)
CANONICAL_COVERAGE_LEDGER = (
    f"{CANONICAL_RUN_DIRECTORY}/scheduled_event_coverage.csv"
)
EVENT_CLASSES = ("opportunistic", "routine")
PRIMARY_CLASS = "opportunistic"
NEGATIVE_CONTROL_CLASS = "routine"

# ASCII exchange-style common-stock symbols, optionally with one class suffix.
# Slashes, spaces, dashes, multiple dots, and punctuation are excluded because
# converting them to a Yahoo symbol would require issuer-specific judgment.
UNAMBIGUOUS_TICKER_PATTERN = re.compile(
    r"^[A-Z][A-Z0-9]{0,9}(?:\.[A-Z][A-Z0-9]{0,2})?$"
)
EXCLUDED_TICKERS = frozenset({"", "NONE", "N/A", "NA"})

EXPECTED_PROTOCOL: dict[str, Any] = {
    "development_start": "2023-04-01",
    "development_end_exclusive": "2025-01-01",
    "holdout_start": "2025-01-01",
    "holdout_end_exclusive": "2026-01-01",
    "primary_class": PRIMARY_CLASS,
    "negative_control_class": NEGATIVE_CONTROL_CLASS,
    "candidate_gate": "preclock_evaluation_candidate_true",
    "classification_field": "provisional_event_classification",
    "required_signal_type": "form4_code_p_purchase_candidate",
    "minimum_direct_purchase_usd": 25_000.0,
    "require_timeliness_eligible": True,
    "require_aff10b5_eligible": True,
    "ticker_rule": UNAMBIGUOUS_TICKER_PATTERN.pattern,
    "yahoo_alias_rule": "replace_period_with_hyphen",
    "benchmark": BENCHMARK,
    "clock": {
        "source": "filing_calendar_date_proxy",
        "quarantined_full_spy_sessions": 1,
        "entry": "second_spy_open_strictly_after_filing_calendar_date",
        "exit_sessions_after_entry": 20,
        "benchmark_open_timestamp": "09:30_America/New_York",
    },
    "event_consolidation": {
        "same_ticker_entry_class": "collapse",
        "collapsed_filing_date": "latest_in_group",
        "same_ticker_same_class_overlap": "ignore_entry_on_or_before_prior_exit",
        "cross_class_same_ticker_entry": "exclude_both_classes",
    },
    "eligibility": {
        "prior_median_dollar_volume_sessions": 63,
        "minimum_prior_median_dollar_volume_usd": 20_000_000.0,
        "minimum_prior_observed_sessions": 252,
        "nominal_price_filter": None,
    },
    "beta": {
        "maximum_matched_adjusted_open_returns": 252,
        "minimum_matched_adjusted_open_returns": 200,
        "end": "placebo_start_open_inclusive",
        "intercept": True,
    },
    "return": {
        "field": "yahoo_vendor_adjusted_open",
        "stock_weight": "1/(1+abs(beta))",
        "benchmark_weight": "-beta/(1+abs(beta))",
        "gross": "stock_weight*stock_return+benchmark_weight*benchmark_return",
        "placebo": "20_spy_sessions_ending_last_open_strictly_before_filing_date",
    },
    "costs": {
        "primary_one_way_rate": 0.003,
        "stress_one_way_rate": 0.005,
        "formula": (
            "c*(abs(ws)+abs(wm)+abs(ws*(1+r_stock))+"
            "abs(wm*(1+r_spy)))"
        ),
    },
    "aggregation": "equal_weight_by_event_within_entry_date",
    "inference": {
        "hac_lags": 20,
        "bootstrap": "centered_circular_moving_blocks",
        "bootstrap_draws": 10_000,
        "bootstrap_block_length": 20,
        "bootstrap_seed": 929,
        "alternative": "mean_greater_than_zero",
        "index": "ordered_nonempty_entry_dates",
        "hac_lags_unit": "SPY_session_distance",
        "bootstrap_block_unit": "entry_date_observations",
        "minimum_common_entry_dates_for_mechanism": 30,
        "formal_tests": [
            "opportunistic_net_30bps_mean_20_sessions",
            "opportunistic_minus_routine_net_30bps_mean_20_sessions",
            "opportunistic_signed_pre_event_placebo_20_sessions",
            "opportunistic_net_50bps_mean_20_sessions",
        ],
        "formal_test_count": 4,
        "disclosed_trial_floor_screening_multiplier": 124,
        "family_wise_error_control_established": False,
    },
    "tuning_permitted": False,
    "alternate_horizons_permitted": False,
    "alternate_weights_permitted": False,
    "historical_alpha_verdict": "INVALID",
    "deployment_authorized": False,
}

REQUIRED_RECEIPT_ROLES = frozenset(
    {
        "config",
        "human_protocol",
        "core_code",
        "outcome_runner",
        "candidate_accessions",
        "candidate_upstream_manifest",
        "yahoo_adapter",
        "price_panel",
        "price_manifest",
        "design_receipt",
    }
)

REQUIRED_DESIGN_RECEIPT_ROLES = frozenset(
    {
        "human_protocol",
        "core_code",
        "outcome_runner",
        "acquisition_script",
        "candidate_accessions",
        "candidate_upstream_manifest",
        "yahoo_adapter",
    }
)


class Form4StageAError(ValueError):
    """Raised when Stage A inputs or execution violate the fixed contract."""


class FreezeReceiptError(Form4StageAError):
    """Raised before outcomes when the immutable-input receipt is incomplete."""


def validate_unique(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    """Local frozen uniqueness check; avoids an unfrozen runtime dependency."""

    keys = list(columns)
    duplicated = frame.duplicated(keys, keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, keys].head(5).to_dict("records")
        raise Form4StageAError(f"{name} duplicate keys {keys}: {examples}")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_stable(path: str | Path) -> bytes:
    target = Path(path)
    if target.is_symlink():
        raise FreezeReceiptError(f"symlink inputs are not allowed: {target}")
    try:
        before = target.stat()
        content = target.read_bytes()
        after = target.stat()
    except OSError as exc:
        raise FreezeReceiptError(f"cannot read required input: {target}") from exc
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise FreezeReceiptError(f"input changed while being read: {target}")
    return content


def _strict_json(content: bytes, *, source: str) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise FreezeReceiptError(
                    f"{source} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise FreezeReceiptError(f"{source} contains non-finite JSON {value}")

    try:
        decoded = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except FreezeReceiptError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise FreezeReceiptError(f"{source} is not strict JSON") from exc
    if not isinstance(decoded, dict):
        raise FreezeReceiptError(f"{source} must contain a JSON object")
    return decoded


def _exact_keys(payload: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    actual = set(payload)
    if actual != expected:
        raise Form4StageAError(
            f"{name} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _resolve_under_root(root: str | Path, relative: str) -> Path:
    repository = Path(root).resolve()
    token = PurePosixPath(str(relative))
    if token.is_absolute() or ".." in token.parts or "\\" in str(relative):
        raise Form4StageAError(f"unsafe repository-relative path: {relative!r}")
    target = (repository / token.as_posix()).resolve()
    try:
        target.relative_to(repository)
    except ValueError as exc:
        raise Form4StageAError(f"path escapes repository root: {relative!r}") from exc
    return target


def _relative_to_root(path: str | Path, root: str | Path) -> str:
    target = Path(path).resolve()
    repository = Path(root).resolve()
    try:
        return target.relative_to(repository).as_posix()
    except ValueError as exc:
        raise Form4StageAError(f"path is outside repository root: {target}") from exc


def _require_canonical_path(
    path: str | Path,
    *,
    root: str | Path,
    canonical_relative: str,
    role: str,
) -> Path:
    """Resolve one experiment-wide path and reject alternate trial locations."""

    target = Path(path).resolve()
    expected = (Path(root).resolve() / canonical_relative).resolve()
    if target != expected:
        raise Form4StageAError(
            f"{role} must use the single canonical path {canonical_relative}; "
            f"received {_relative_to_root(target, root)!r}"
        )
    return target


def yahoo_alias(ticker: object) -> str | None:
    """Return the sole permitted Yahoo alias or ``None`` when ambiguous."""

    if pd.isna(ticker):
        return None
    symbol = str(ticker).strip().upper()
    if (
        symbol in EXCLUDED_TICKERS
        or any(token in symbol for token in (",", ";", "/", " ", "-"))
        or not UNAMBIGUOUS_TICKER_PATTERN.fullmatch(symbol)
    ):
        return None
    return symbol.replace(".", "-")


def _require(frame: pd.DataFrame, columns: Iterable[str], *, name: str) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise Form4StageAError(f"{name} missing columns: {sorted(missing)}")


def select_candidate_events(
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Select only the development event cohort without consulting returns."""

    required = {
        "accession",
        "filing_date",
        "ticker",
        "preclock_evaluation_candidate",
        "provisional_event_classification",
        "signal_type",
        "direct_purchase_usd",
        "timeliness_eligible",
        "aff10b5_eligible",
    }
    _require(candidates, required, name="candidate_accessions")
    validate_unique(candidates, ["accession"], "candidate_accessions")

    frame = candidates[list(required)].copy()
    parsed_dates = pd.to_datetime(frame["filing_date"], errors="coerce")
    if parsed_dates.isna().any():
        raise Form4StageAError("candidate_accessions contains invalid filing_date")
    if getattr(parsed_dates.dt, "tz", None) is not None:
        parsed_dates = parsed_dates.dt.tz_convert("America/New_York").dt.tz_localize(None)
    frame["filing_date"] = parsed_dates.dt.normalize()
    if not pd.api.types.is_bool_dtype(frame["preclock_evaluation_candidate"]):
        raise Form4StageAError(
            "preclock_evaluation_candidate must be a non-null boolean column"
        )
    if frame["preclock_evaluation_candidate"].isna().any():
        raise Form4StageAError("preclock_evaluation_candidate contains null")
    frame["event_class"] = (
        frame["provisional_event_classification"]
        .astype("string")
        .str.strip()
        .str.lower()
    )
    frame["source_ticker"] = frame["ticker"].astype("string").str.strip().str.upper()
    frame["direct_purchase_usd"] = pd.to_numeric(
        frame["direct_purchase_usd"], errors="coerce"
    )
    if (~np.isfinite(frame["direct_purchase_usd"])).any():
        raise Form4StageAError("direct_purchase_usd contains non-finite values")
    for column in ("timeliness_eligible", "aff10b5_eligible"):
        if not pd.api.types.is_bool_dtype(frame[column]) or frame[column].isna().any():
            raise Form4StageAError(f"{column} must be a non-null boolean column")

    holdout_mask = frame["filing_date"].ge(HOLDOUT_START) & frame[
        "filing_date"
    ].lt(HOLDOUT_END_EXCLUSIVE)
    development_mask = frame["filing_date"].ge(DEVELOPMENT_START) & frame[
        "filing_date"
    ].lt(DEVELOPMENT_END_EXCLUSIVE)
    base = frame.loc[
        development_mask
        & frame["preclock_evaluation_candidate"]
        & frame["event_class"].isin(EVENT_CLASSES)
        & frame["signal_type"].astype("string").eq("form4_code_p_purchase_candidate")
        & frame["direct_purchase_usd"].ge(25_000.0)
        & frame["timeliness_eligible"]
        & frame["aff10b5_eligible"]
    ].copy()
    base["yahoo_ticker"] = base["source_ticker"].map(yahoo_alias)

    benchmark_conflict = base["source_ticker"].eq(BENCHMARK).fillna(False)
    invalid = (base["yahoo_ticker"].isna() | benchmark_conflict).fillna(True)
    exclusions = base.loc[
        invalid, ["accession", "filing_date", "source_ticker", "event_class"]
    ].copy()
    exclusions["reason"] = np.where(
        exclusions["source_ticker"].eq(BENCHMARK).fillna(False).to_numpy(dtype=bool),
        "issuer_ticker_conflicts_with_benchmark",
        "ambiguous_or_unsupported_ticker",
    )
    selected = base.loc[~invalid].copy()
    selected = selected[
        [
            "accession",
            "filing_date",
            "source_ticker",
            "yahoo_ticker",
            "event_class",
            "direct_purchase_usd",
        ]
    ].sort_values(
        ["filing_date", "event_class", "source_ticker", "accession"],
        kind="mergesort",
    )
    if selected.empty:
        raise Form4StageAError("no eligible 2023-04-01 through 2024-12-31 events")
    if selected["filing_date"].ge(HOLDOUT_START).any():
        raise Form4StageAError("2025 holdout event entered the development cohort")

    collisions = (
        selected[["source_ticker", "yahoo_ticker"]]
        .drop_duplicates()
        .groupby("yahoo_ticker")["source_ticker"]
        .nunique()
    )
    if collisions.gt(1).any():
        raise Form4StageAError("multiple source tickers map to one Yahoo alias")
    audit = {
        "candidate_rows": int(len(frame)),
        "preclock_development_rows_in_two_classes": int(len(base)),
        "selected_development_rows": int(len(selected)),
        "opportunistic_rows": int(selected["event_class"].eq(PRIMARY_CLASS).sum()),
        "routine_rows": int(
            selected["event_class"].eq(NEGATIVE_CONTROL_CLASS).sum()
        ),
        "invalid_ticker_rows": int(invalid.sum()),
        "holdout_metadata_rows_present_but_not_selected": int(holdout_mask.sum()),
        "holdout_event_outcomes_opened": False,
    }
    return selected.reset_index(drop=True), exclusions.reset_index(drop=True), audit


def build_acquisition_universe(
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build a label-only Yahoo request list; no price outcome is joined."""

    events, exclusions, audit = select_candidate_events(candidates)
    aliases = events[["source_ticker", "yahoo_ticker"]].drop_duplicates()
    benchmark = pd.DataFrame(
        {"source_ticker": [BENCHMARK], "yahoo_ticker": [BENCHMARK]}
    )
    aliases = pd.concat([benchmark, aliases], ignore_index=True).drop_duplicates()
    aliases["is_benchmark"] = aliases["source_ticker"].eq(BENCHMARK)
    aliases = aliases.sort_values(
        ["is_benchmark", "source_ticker"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    validate_unique(aliases, ["source_ticker"], "source-to-Yahoo aliases")
    validate_unique(aliases, ["yahoo_ticker"], "Yahoo-to-source aliases")
    audit = {
        **audit,
        "requested_source_tickers_including_spy": int(len(aliases)),
        "acquisition_start": ACQUISITION_START,
        "acquisition_end": ACQUISITION_END,
        "outcome_join_performed": False,
    }
    return aliases, exclusions, audit


def validate_stage_a_config(payload: Mapping[str, Any]) -> None:
    _exact_keys(payload, {"schema_version", "sources", "protocol"}, name="config")
    if payload["schema_version"] != CONFIG_SCHEMA:
        raise Form4StageAError(f"unsupported config schema: {payload['schema_version']!r}")
    sources = payload["sources"]
    if not isinstance(sources, Mapping):
        raise Form4StageAError("config sources must be an object")
    _exact_keys(
        sources,
        {
            "human_protocol",
            "candidate_accessions",
            "candidate_upstream_manifest",
            "yahoo_adapter",
            "design_receipt",
            "price_panel",
            "price_manifest",
        },
        name="config sources",
    )
    for role, relative in sources.items():
        token = PurePosixPath(str(relative))
        if token.is_absolute() or ".." in token.parts or "\\" in str(relative):
            raise Form4StageAError(f"unsafe config source path for {role}")
    canonical_sources = {
        "human_protocol": CANONICAL_HUMAN_PROTOCOL,
        "design_receipt": CANONICAL_DESIGN_RECEIPT,
        "price_panel": CANONICAL_PRICE_PANEL,
        "price_manifest": CANONICAL_PRICE_MANIFEST,
    }
    wrong_paths = sorted(
        role
        for role, canonical in canonical_sources.items()
        if str(sources.get(role)) != canonical
    )
    if wrong_paths:
        raise Form4StageAError(
            f"config sources do not use canonical experiment paths: {wrong_paths}"
        )
    if payload["protocol"] != EXPECTED_PROTOCOL:
        raise Form4StageAError(
            "config protocol differs from the fixed Stage A protocol; "
            "alternate tuning, horizons, weights, clocks, or costs are refused"
        )


def validate_human_protocol(payload: Mapping[str, Any]) -> None:
    """Check the authored research protocol against every runtime invariant."""

    if payload.get("schema_version") != "form4_opportunistic_stage_a_protocol_v1":
        raise Form4StageAError("unsupported human research protocol schema")
    try:
        experiment = payload["experiment"]
        sources = payload["sources"]
        selection = payload["candidate_selection"]
        clock = payload["clock_and_execution"]
        universe = payload["universe"]
        beta = payload["risk_adjustment"]
        estimands = payload["estimands"]
        inference = payload["inference"]
        stop_gates = payload["stop_gates"]
        outputs = payload["outputs"]
    except KeyError as exc:
        raise Form4StageAError(f"human protocol missing section {exc.args[0]}") from exc
    checks = {
        "preserve_2025_return_outcomes_uncomputed": experiment.get(
            "preserve_2025_return_outcomes_uncomputed"
        ) is True,
        "candidate_start": selection.get("filing_date_gte") == "2023-04-01",
        "candidate_end": selection.get("filing_date_lt") == "2025-01-01",
        "primary_class": selection.get("primary_class") == PRIMARY_CLASS,
        "negative_control": selection.get("negative_control_class") == NEGATIVE_CONTROL_CLASS,
        "signal_type": selection.get("required_signal_type") == "form4_code_p_purchase_candidate",
        "minimum_purchase": float(selection.get("minimum_direct_purchase_usd", -1)) == 25_000.0,
        "timeliness": selection.get("require_timeliness_eligible") is True,
        "10b5": selection.get("require_aff10b5_eligible") is True,
        "ticker_regex": selection.get("ticker_regex") == UNAMBIGUOUS_TICKER_PATTERN.pattern,
        "acquisition_start": sources["price_provider"].get("start") == ACQUISITION_START,
        "acquisition_end": sources["price_provider"].get("end") == ACQUISITION_END,
        "fresh_yahoo_after_design_freeze": sources["price_provider"].get(
            "fresh_network_retrieval_after_design_freeze"
        ) is True,
        "entry_clock": int(clock.get("holding_period_sessions", -1)) == 20,
        "benchmark_open_timestamp": clock.get("benchmark_open_timestamp_rule")
        == "09:30 America/New_York",
        "cross_class_conflict": selection.get("cross_class_same_ticker_entry_policy")
        == "exclude both classes before any outcome lookup",
        "no_nominal_price_filter": universe.get("minimum_entry_prior_price_usd") is None,
        "adv_window": int(universe.get("adv_window_sessions", -1)) == 63,
        "adv": float(universe.get("minimum_entry_prior_adv_usd", -1)) == 20_000_000.0,
        "history": int(universe.get("minimum_prior_stock_sessions", -1)) == 252,
        "beta_window": int(beta.get("beta_window_matched_sessions", -1)) == 252,
        "beta_minimum": int(beta.get("minimum_matched_beta_observations", -1)) == 200,
        "beta_intercept": beta.get("intercept") is True,
        "primary_cost": int(estimands.get("primary_cost_bps_one_way_per_traded_dollar", -1)) == 30,
        "stress_cost": int(estimands.get("stress_cost_bps_one_way_per_traded_dollar", -1)) == 50,
        "hac": int(inference.get("date_level_hac_lags", -1)) == 20,
        "draws": int(inference.get("circular_moving_block_draws", -1)) == 10_000,
        "block": int(
            inference.get("circular_moving_block_length_entry_dates", -1)
        )
        == 20,
        "inference_unit": inference.get("inference_unit")
        == (
            "HAC uses actual SPY session-index distance; circular MBB uses "
            "ordered nonempty entry-date observations as a non-decision sensitivity"
        ),
        "seed": int(inference.get("bootstrap_seed", -1)) == 929,
        "tests": int(inference.get("tests_added_count", -1)) == 4,
        "trial_floor": int(inference.get("resulting_disclosed_trial_floor", -1)) == 124,
        "multiple_testing_semantics": inference.get("multiple_testing_method")
        == "known_disclosed_trial_floor_screen_only_not_FWER",
        "paired_mechanism": inference.get("mechanism_control")
        == (
            "paired common-entry-date opportunistic net mean minus routine net "
            "mean greater than zero"
        ),
        "mechanism_minimum": int(
            inference.get("minimum_common_entry_dates_for_mechanism", -1)
        )
        == 30,
        "scheduled_coverage": float(
            stop_gates["sample"].get(
                "minimum_scheduled_opportunistic_event_price_path_coverage", -1
            )
        )
        == 0.8,
        "routine_scheduled_coverage": float(
            stop_gates["sample"].get(
                "minimum_scheduled_routine_event_price_path_coverage", -1
            )
        )
        == 0.8,
        "ticker_mapping_coverage": float(
            stop_gates["sample"].get("minimum_candidate_ticker_mapping_coverage", -1)
        ) == 0.95,
        "benchmark_clock_coverage": float(
            stop_gates["sample"].get("minimum_benchmark_clock_coverage", -1)
        ) == 1.0,
        "retained_fields": stop_gates["integrity"].get(
            "all_retained_outcome_fields_finite"
        ) is True,
        "concentration": float(
            stop_gates["concentration"].get(
                "maximum_single_ticker_gross_absolute_contribution_share", -1
            )
        )
        == 0.1,
        "leave_one_out": stop_gates["concentration"].get(
            "minimum_leave_one_ticker_out_net30_mean"
        ) == "greater_than_zero",
        "sample_mechanism_minimum": int(
            stop_gates["sample"].get(
                "minimum_common_entry_dates_for_mechanism", -1
            )
        )
        == 30,
        "canonical_outputs": outputs
        == {
            "runtime_config_path": CANONICAL_RUNTIME_CONFIG,
            "design_receipt_path": CANONICAL_DESIGN_RECEIPT,
            "acquisition_directory": CANONICAL_ACQUISITION_DIRECTORY,
            "run_directory": CANONICAL_RUN_DIRECTORY,
            "freeze_receipt_path": CANONICAL_FREEZE_RECEIPT,
            "scheduled_event_coverage_ledger": CANONICAL_COVERAGE_LEDGER,
        },
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise Form4StageAError(
            f"human protocol and runtime contract differ: {failed}"
        )


def validate_human_protocol_lineage(
    payload: Mapping[str, Any],
    *,
    repository_root: str | Path,
    expected_candidate: str | Path,
    expected_candidate_upstream_manifest: str | Path,
    expected_yahoo_adapter: str | Path,
) -> None:
    """Bind the human protocol's declared source hashes to runtime paths."""

    root = Path(repository_root).resolve()
    sources = payload.get("sources")
    if not isinstance(sources, Mapping):
        raise FreezeReceiptError("human protocol sources must be an object")
    candidate = sources.get("candidate_accessions")
    provider = sources.get("price_provider")
    if not isinstance(candidate, Mapping) or not isinstance(provider, Mapping):
        raise FreezeReceiptError("human protocol source declarations are incomplete")

    declarations = (
        (
            "candidate_accessions",
            candidate.get("path"),
            candidate.get("bytes"),
            candidate.get("sha256"),
            Path(expected_candidate).resolve(),
        ),
        (
            "candidate_upstream_manifest",
            candidate.get("upstream_manifest_path"),
            candidate.get("upstream_manifest_bytes"),
            candidate.get("upstream_manifest_sha256"),
            Path(expected_candidate_upstream_manifest).resolve(),
        ),
        (
            "yahoo_adapter",
            provider.get("adapter_path"),
            None,
            provider.get("adapter_sha256_at_protocol_authoring"),
            Path(expected_yahoo_adapter).resolve(),
        ),
    )
    for role, relative, expected_bytes, expected_sha, runtime_path in declarations:
        if not isinstance(relative, str) or not isinstance(expected_sha, str):
            raise FreezeReceiptError(f"human protocol omits {role} lineage")
        declared_path = _resolve_under_root(root, relative)
        if declared_path != runtime_path:
            raise FreezeReceiptError(f"human protocol path differs for {role}")
        content = _read_stable(declared_path)
        if expected_bytes is not None and len(content) != int(expected_bytes):
            raise FreezeReceiptError(f"human protocol byte count mismatch for {role}")
        if hashlib.sha256(content).hexdigest() != expected_sha:
            raise FreezeReceiptError(f"human protocol hash mismatch for {role}")

    upstream_content = _read_stable(Path(expected_candidate_upstream_manifest))
    upstream = _strict_json(
        upstream_content, source=str(expected_candidate_upstream_manifest)
    )
    records = upstream.get("records")
    if not isinstance(records, list):
        raise FreezeReceiptError("candidate upstream manifest has no records")
    candidate_path = Path(expected_candidate).resolve()
    candidate_matches = [
        record
        for record in records
        if isinstance(record, Mapping)
        and Path(str(record.get("path", ""))).resolve() == candidate_path
        and str(record.get("sha256", "")) == sha256_file(candidate_path)
        and int(record.get("bytes", -1)) == candidate_path.stat().st_size
    ]
    if len(candidate_matches) != 1:
        raise FreezeReceiptError(
            "candidate upstream manifest does not bind the exact candidate file"
        )


def stage_a_config_template(
    *,
    human_protocol: str,
    candidate_accessions: str,
    candidate_upstream_manifest: str,
    yahoo_adapter: str,
    design_receipt: str,
    price_panel: str,
    price_manifest: str,
) -> dict[str, Any]:
    """Return, but do not write or freeze, the exact future config payload."""

    payload = {
        "schema_version": CONFIG_SCHEMA,
        "sources": {
            "human_protocol": human_protocol,
            "candidate_accessions": candidate_accessions,
            "candidate_upstream_manifest": candidate_upstream_manifest,
            "yahoo_adapter": yahoo_adapter,
            "design_receipt": design_receipt,
            "price_panel": price_panel,
            "price_manifest": price_manifest,
        },
        "protocol": json.loads(json.dumps(EXPECTED_PROTOCOL)),
    }
    validate_stage_a_config(payload)
    return payload


def _receipt_record(role: str, path: Path, root: Path) -> dict[str, Any]:
    content = _read_stable(path)
    return {
        "role": role,
        "path": _relative_to_root(path, root),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def build_design_receipt(
    *,
    human_protocol_path: str | Path,
    core_code_path: str | Path,
    outcome_runner_path: str | Path,
    acquisition_script_path: str | Path,
    candidate_path: str | Path,
    candidate_upstream_manifest_path: str | Path,
    yahoo_adapter_path: str | Path,
    receipt_path: str | Path,
    repository_root: str | Path,
) -> Path:
    """Freeze the research design before outcome-bearing prices are acquired."""

    root = Path(repository_root).resolve()
    paths = {
        "human_protocol": _require_canonical_path(
            human_protocol_path,
            root=root,
            canonical_relative=CANONICAL_HUMAN_PROTOCOL,
            role="human protocol",
        ),
        "core_code": _require_canonical_path(
            core_code_path,
            root=root,
            canonical_relative=CANONICAL_CORE_CODE,
            role="core code",
        ),
        "outcome_runner": _require_canonical_path(
            outcome_runner_path,
            root=root,
            canonical_relative=CANONICAL_OUTCOME_RUNNER,
            role="outcome runner",
        ),
        "acquisition_script": _require_canonical_path(
            acquisition_script_path,
            root=root,
            canonical_relative=CANONICAL_ACQUISITION_SCRIPT,
            role="acquisition script",
        ),
        "candidate_accessions": Path(candidate_path).resolve(),
        "candidate_upstream_manifest": Path(
            candidate_upstream_manifest_path
        ).resolve(),
        "yahoo_adapter": Path(yahoo_adapter_path).resolve(),
    }
    if set(paths) != REQUIRED_DESIGN_RECEIPT_ROLES:
        raise FreezeReceiptError("design receipt roles differ from fixed contract")
    for path in paths.values():
        _relative_to_root(path, root)
    human_payload = _strict_json(
        _read_stable(paths["human_protocol"]), source=str(paths["human_protocol"])
    )
    validate_human_protocol(human_payload)
    validate_human_protocol_lineage(
        human_payload,
        repository_root=root,
        expected_candidate=paths["candidate_accessions"],
        expected_candidate_upstream_manifest=paths[
            "candidate_upstream_manifest"
        ],
        expected_yahoo_adapter=paths["yahoo_adapter"],
    )
    authored_at = pd.Timestamp(human_payload.get("protocol_authored_at"))
    if pd.isna(authored_at) or authored_at.tzinfo is None:
        raise FreezeReceiptError("protocol_authored_at must be timezone-aware")
    receipt = _require_canonical_path(
        receipt_path,
        root=root,
        canonical_relative=CANONICAL_DESIGN_RECEIPT,
        role="design receipt",
    )
    frozen_at = datetime.now(UTC)
    if authored_at.tz_convert("UTC") > pd.Timestamp(frozen_at):
        raise FreezeReceiptError("protocol_authored_at is later than design freeze")
    payload = {
        "schema_version": DESIGN_RECEIPT_SCHEMA,
        "frozen_at": frozen_at.isoformat(),
        "immutable": True,
        "purpose": "pre-price research-design freeze",
        "records": [
            _receipt_record(role, paths[role], root) for role in sorted(paths)
        ],
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    try:
        with receipt.open("x", encoding="utf-8") as handle:
            handle.write(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
                + "\n"
            )
    except FileExistsError as exc:
        raise FreezeReceiptError(
            f"design receipt already exists: {receipt}"
        ) from exc
    return receipt


def verify_design_receipt(
    receipt_path: str | Path,
    *,
    expected_paths: Mapping[str, str | Path],
    repository_root: str | Path,
) -> dict[str, Any]:
    """Verify the pre-price design receipt and every frozen design byte string."""

    root = Path(repository_root).resolve()
    receipt = _require_canonical_path(
        receipt_path,
        root=root,
        canonical_relative=CANONICAL_DESIGN_RECEIPT,
        role="design receipt",
    )
    payload = _strict_json(_read_stable(receipt), source=str(receipt))
    _exact_keys(
        payload,
        {"schema_version", "frozen_at", "immutable", "purpose", "records"},
        name="design receipt",
    )
    if payload["schema_version"] != DESIGN_RECEIPT_SCHEMA:
        raise FreezeReceiptError("unsupported design receipt schema")
    if payload["immutable"] is not True or payload["purpose"] != (
        "pre-price research-design freeze"
    ):
        raise FreezeReceiptError("design receipt does not establish the fixed purpose")
    if set(expected_paths) != REQUIRED_DESIGN_RECEIPT_ROLES:
        raise FreezeReceiptError("design receipt verifier paths differ")
    fixed_design_paths = {
        "human_protocol": CANONICAL_HUMAN_PROTOCOL,
        "core_code": CANONICAL_CORE_CODE,
        "outcome_runner": CANONICAL_OUTCOME_RUNNER,
        "acquisition_script": CANONICAL_ACQUISITION_SCRIPT,
    }
    for role, canonical in fixed_design_paths.items():
        _require_canonical_path(
            expected_paths[role],
            root=root,
            canonical_relative=canonical,
            role=role.replace("_", " "),
        )
    records = payload["records"]
    if not isinstance(records, list):
        raise FreezeReceiptError("design receipt records must be a list")
    roles = [str(record.get("role")) for record in records if isinstance(record, Mapping)]
    if set(roles) != REQUIRED_DESIGN_RECEIPT_ROLES or len(roles) != len(
        REQUIRED_DESIGN_RECEIPT_ROLES
    ):
        raise FreezeReceiptError("design receipt role coverage differs")
    for record in records:
        if not isinstance(record, Mapping):
            raise FreezeReceiptError("design receipt record is not an object")
        _exact_keys(record, {"role", "path", "bytes", "sha256"}, name="design record")
        role = str(record["role"])
        expected = Path(expected_paths[role]).resolve()
        if _resolve_under_root(root, str(record["path"])) != expected:
            raise FreezeReceiptError(f"design receipt path mismatch for {role}")
        content = _read_stable(expected)
        if len(content) != int(record["bytes"]):
            raise FreezeReceiptError(f"design receipt byte count mismatch for {role}")
        if hashlib.sha256(content).hexdigest() != str(record["sha256"]):
            raise FreezeReceiptError(f"design receipt hash mismatch for {role}")
    human_payload = _strict_json(
        _read_stable(expected_paths["human_protocol"]),
        source=str(expected_paths["human_protocol"]),
    )
    validate_human_protocol(human_payload)
    validate_human_protocol_lineage(
        human_payload,
        repository_root=root,
        expected_candidate=expected_paths["candidate_accessions"],
        expected_candidate_upstream_manifest=expected_paths[
            "candidate_upstream_manifest"
        ],
        expected_yahoo_adapter=expected_paths["yahoo_adapter"],
    )
    authored_at = pd.Timestamp(human_payload.get("protocol_authored_at"))
    frozen_at = pd.Timestamp(payload.get("frozen_at"))
    if (
        pd.isna(authored_at)
        or pd.isna(frozen_at)
        or authored_at.tzinfo is None
        or frozen_at.tzinfo is None
        or authored_at.tz_convert("UTC") > frozen_at.tz_convert("UTC")
    ):
        raise FreezeReceiptError("design receipt timestamp order is invalid")
    return payload


def _reconstruct_clean_price_panel_from_raw(
    raw_payloads: Mapping[str, bytes],
    *,
    alias_pairs: set[tuple[str, str]],
    expected_vendor_rows: Mapping[str, int],
) -> pd.DataFrame:
    """Independently rebuild the frozen outcome fields from raw Yahoo JSON."""

    yahoo_to_source = {yahoo: source for source, yahoo in alias_pairs}
    if len(yahoo_to_source) != len(alias_pairs):
        raise FreezeReceiptError("Yahoo aliases are not one-to-one")
    parts: list[pd.DataFrame] = []
    for yahoo_ticker, raw in sorted(raw_payloads.items()):
        try:
            payload = json.loads(raw)
        except Exception as exc:
            raise FreezeReceiptError(
                f"Yahoo raw payload is not JSON for {yahoo_ticker}"
            ) from exc
        chart = payload.get("chart") or {}
        error = chart.get("error")
        result = (chart.get("result") or [None])[0]
        if error or not isinstance(result, Mapping):
            raise FreezeReceiptError(
                f"Yahoo raw payload has no usable result for {yahoo_ticker}"
            )
        returned_symbol = (result.get("meta") or {}).get("symbol")
        if returned_symbol != yahoo_ticker:
            raise FreezeReceiptError(
                "Yahoo raw response symbol differs from requested ticker: "
                f"requested={yahoo_ticker!r}, returned={returned_symbol!r}"
            )
        timestamps = result.get("timestamp") or []
        indicators = result.get("indicators") or {}
        quotes = (indicators.get("quote") or [{}])[0]
        adjusted = (indicators.get("adjclose") or [{}])[0].get("adjclose")
        size = len(timestamps)

        def values(key: str) -> list[Any]:
            found = quotes.get(key)
            return found if found is not None else [None] * size

        parsed = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    timestamps, unit="s", utc=True, errors="coerce"
                ),
                "open": values("open"),
                "close": values("close"),
                "volume": values("volume"),
                "adj_close": adjusted if adjusted is not None else values("close"),
            }
        )
        for column in ("open", "close", "volume", "adj_close"):
            parsed[column] = pd.to_numeric(parsed[column], errors="coerce")
        parsed = parsed.dropna(subset=["close", "adj_close"]).copy()
        if len(parsed) != int(expected_vendor_rows.get(yahoo_ticker, -1)):
            raise FreezeReceiptError(
                f"Yahoo vendor row count differs from raw reconstruction for {yahoo_ticker}"
            )
        exchange_tz = str(
            (result.get("meta") or {}).get(
                "exchangeTimezoneName", "America/New_York"
            )
        )
        try:
            local_date = parsed["timestamp"].dt.tz_convert(exchange_tz).dt.date
        except Exception as exc:
            raise FreezeReceiptError(
                f"Yahoo exchange timezone is invalid for {yahoo_ticker}"
            ) from exc
        parsed["date"] = pd.to_datetime(local_date)
        parsed["adj_open"] = parsed["open"] * (
            parsed["adj_close"] / parsed["close"].replace(0, np.nan)
        )
        parsed["source_ticker"] = yahoo_to_source[yahoo_ticker]
        parsed["yahoo_ticker"] = yahoo_ticker
        usable = parsed["timestamp"].notna()
        for column in ("adj_open", "adj_close", "close", "volume"):
            usable &= np.isfinite(parsed[column])
        usable &= parsed["adj_open"].gt(0)
        usable &= parsed["adj_close"].gt(0)
        usable &= parsed["close"].gt(0)
        usable &= parsed["volume"].gt(0)
        parts.append(
            parsed.loc[
                usable,
                [
                    "source_ticker",
                    "yahoo_ticker",
                    "date",
                    "timestamp",
                    "adj_open",
                    "adj_close",
                    "close",
                    "volume",
                ],
            ]
        )
    if not parts:
        raise FreezeReceiptError("no successful Yahoo payload can be reconstructed")
    return pd.concat(parts, ignore_index=True).sort_values(
        ["source_ticker", "date"], kind="mergesort"
    ).reset_index(drop=True)


def _assert_price_panel_matches_reconstruction(
    frozen_price_panel: pd.DataFrame,
    reconstructed: pd.DataFrame,
    *,
    expected_source_tickers: set[str],
) -> None:
    """Verify raw-to-panel equality and SPY clock viability before freezing."""

    actual = _normalize_price_panel(
        frozen_price_panel, expected_source_tickers=expected_source_tickers
    )
    expected = _normalize_price_panel(
        reconstructed, expected_source_tickers=expected_source_tickers
    )
    columns = [
        "source_ticker",
        "yahoo_ticker",
        "date",
        "timestamp",
        "adj_open",
        "adj_close",
        "close",
        "volume",
    ]
    actual = actual[columns].sort_values(
        ["source_ticker", "date"], kind="mergesort"
    ).reset_index(drop=True)
    expected = expected[columns].sort_values(
        ["source_ticker", "date"], kind="mergesort"
    ).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            actual,
            expected,
            check_dtype=False,
            check_exact=True,
            check_like=False,
        )
    except AssertionError as exc:
        raise FreezeReceiptError(
            "frozen price panel differs from independent raw Yahoo reconstruction"
        ) from exc
    spy = actual.loc[actual["source_ticker"].eq(BENCHMARK)]
    if len(spy) < 275:
        raise FreezeReceiptError("clean reconstructed SPY history is too short")
    if spy["date"].min() < pd.Timestamp(ACQUISITION_START) or spy[
        "date"
    ].max() != pd.Timestamp(ACQUISITION_END):
        raise FreezeReceiptError("clean SPY dates fall outside the requested range")


def _validate_price_manifest_payload(
    payload: Mapping[str, Any],
    *,
    root: Path,
    expected_candidate: Path,
    expected_candidate_upstream_manifest: Path,
    expected_yahoo_adapter: Path,
    expected_design_receipt: Path,
    expected_price: Path,
) -> None:
    required = {
        "schema_version",
        "created_at",
        "provider",
        "source_class",
        "historical_alpha_verdict",
        "design_receipt",
        "candidate_file",
        "candidate_upstream_manifest",
        "yahoo_adapter",
        "price_file",
        "acquisition_script",
        "support_files",
        "acquisition",
        "ticker_aliases",
        "vendor_records",
        "limitations",
    }
    _exact_keys(payload, required, name="price manifest")
    if payload["schema_version"] != PRICE_MANIFEST_SCHEMA:
        raise FreezeReceiptError("unsupported price manifest schema")
    if payload["provider"] != "Yahoo Finance chart API":
        raise FreezeReceiptError("price manifest provider is not Yahoo Finance")
    if payload["source_class"] != "current_ticker_yahoo_retrospective":
        raise FreezeReceiptError("price source class is not the declared Yahoo proxy")
    if payload["historical_alpha_verdict"] != "INVALID":
        raise FreezeReceiptError("Yahoo proxy manifest must lock INVALID verdict")

    def verify_declared_file(
        declaration: object, *, name: str, expected_path: Path | None = None
    ) -> tuple[Path, bytes]:
        if not isinstance(declaration, Mapping):
            raise FreezeReceiptError(f"{name} must be an object")
        _exact_keys(declaration, {"path", "bytes", "sha256"}, name=name)
        actual_path = _resolve_under_root(root, str(declaration["path"]))
        if expected_path is not None and actual_path != expected_path.resolve():
            raise FreezeReceiptError(f"{name} path does not match final config")
        content = _read_stable(actual_path)
        if len(content) != int(declaration["bytes"]):
            raise FreezeReceiptError(f"{name} byte count mismatch")
        if hashlib.sha256(content).hexdigest() != declaration["sha256"]:
            raise FreezeReceiptError(f"{name} hash mismatch")
        return actual_path, content

    acquisition = payload["acquisition"]
    if not isinstance(acquisition, Mapping):
        raise FreezeReceiptError("price manifest acquisition must be an object")
    required_acquisition = {
        "request_start",
        "request_end",
        "outcome_join_performed",
        "holdout_return_outcomes_joined",
        "design_receipt_verified_before_download",
        "fresh_network_retrieval_after_design_freeze",
        "candidate_classes",
        "benchmark",
        "ticker_regex",
        "alias_rule",
    }
    _exact_keys(acquisition, required_acquisition, name="price acquisition")
    expected_acquisition = {
        "request_start": ACQUISITION_START,
        "request_end": ACQUISITION_END,
        "outcome_join_performed": False,
        "holdout_return_outcomes_joined": False,
        "design_receipt_verified_before_download": True,
        "fresh_network_retrieval_after_design_freeze": True,
        "candidate_classes": list(EVENT_CLASSES),
        "benchmark": BENCHMARK,
        "ticker_regex": UNAMBIGUOUS_TICKER_PATTERN.pattern,
        "alias_rule": "replace_period_with_hyphen",
    }
    if acquisition != expected_acquisition:
        raise FreezeReceiptError("price acquisition manifest differs from fixed design")
    declared_contents: dict[str, bytes] = {}
    for key, expected_path in (
        ("candidate_file", expected_candidate),
        ("candidate_upstream_manifest", expected_candidate_upstream_manifest),
        ("yahoo_adapter", expected_yahoo_adapter),
        ("design_receipt", expected_design_receipt),
        ("price_file", expected_price),
    ):
        _, content = verify_declared_file(
            payload[key], name=key, expected_path=expected_path
        )
        declared_contents[key] = content
    verify_declared_file(
        payload["acquisition_script"],
        name="acquisition_script",
        expected_path=root / "scripts/acquire_form4_opportunistic_stage_a_prices.py",
    )
    support_files = payload["support_files"]
    if not isinstance(support_files, Mapping):
        raise FreezeReceiptError("support_files must be an object")
    _exact_keys(
        support_files,
        {"download_audit", "ticker_aliases", "ticker_exclusions"},
        name="support_files",
    )
    support_contents: dict[str, bytes] = {}
    for role, declaration in support_files.items():
        _, content = verify_declared_file(declaration, name=f"support_files:{role}")
        support_contents[str(role)] = content

    aliases = payload["ticker_aliases"]
    if not isinstance(aliases, list) or not aliases:
        raise FreezeReceiptError("ticker_aliases must be a non-empty list")
    alias_pairs: set[tuple[str, str]] = set()
    for item in aliases:
        if not isinstance(item, Mapping):
            raise FreezeReceiptError("ticker alias row is not an object")
        source = str(item.get("source_ticker", ""))
        yahoo = str(item.get("yahoo_ticker", ""))
        if yahoo_alias(source) != yahoo:
            raise FreezeReceiptError(f"invalid frozen ticker alias: {source}->{yahoo}")
        if (source, yahoo) in alias_pairs:
            raise FreezeReceiptError("duplicate frozen ticker alias")
        alias_pairs.add((source, yahoo))
    if (BENCHMARK, BENCHMARK) not in alias_pairs:
        raise FreezeReceiptError("ticker aliases omit SPY benchmark")

    try:
        frozen_candidates = pd.read_parquet(
            io.BytesIO(declared_contents["candidate_file"])
        )
    except Exception as exc:
        raise FreezeReceiptError("frozen candidate parquet cannot be parsed") from exc
    expected_alias_frame, expected_ticker_exclusions, expected_candidate_audit = (
        build_acquisition_universe(frozen_candidates)
    )
    expected_alias_pairs = set(
        expected_alias_frame[["source_ticker", "yahoo_ticker"]]
        .astype(str)
        .itertuples(index=False, name=None)
    )
    if alias_pairs != expected_alias_pairs:
        raise FreezeReceiptError(
            "ticker aliases do not equal the universe recomputed from frozen candidates"
        )

    vendor_records = payload["vendor_records"]
    if not isinstance(vendor_records, list):
        raise FreezeReceiptError("vendor_records must be a list")
    seen_vendor_tickers: set[str] = set()
    seen_cache_paths: set[Path] = set()
    raw_payloads: dict[str, bytes] = {}
    vendor_rows: dict[str, int] = {}
    for record in vendor_records:
        if not isinstance(record, Mapping):
            raise FreezeReceiptError("vendor record is not an object")
        required_vendor_fields = {
            "yahoo_ticker",
            "raw_cache_path",
            "raw_cache_bytes",
            "raw_cache_sha256",
            "raw_payload_sha256",
        }
        if not required_vendor_fields.issubset(record):
            raise FreezeReceiptError("vendor record omits raw Yahoo cache evidence")
        yahoo_ticker = str(record["yahoo_ticker"])
        if yahoo_ticker in seen_vendor_tickers:
            raise FreezeReceiptError("duplicate Yahoo vendor record")
        seen_vendor_tickers.add(yahoo_ticker)
        cache_path = _resolve_under_root(root, str(record["raw_cache_path"]))
        safe_ticker = (
            yahoo_ticker.replace("/", "_")
            .replace(".", "_")
            .replace("^", "index_")
        )
        expected_cache_path = (
            expected_price.parent
            / "raw"
            / f"{safe_ticker}_{record['raw_payload_sha256']}.json.gz"
        ).resolve()
        if cache_path != expected_cache_path:
            raise FreezeReceiptError(
                f"Yahoo raw archive is not content-addressed for {yahoo_ticker}"
            )
        if cache_path in seen_cache_paths:
            raise FreezeReceiptError("duplicate Yahoo raw cache path")
        seen_cache_paths.add(cache_path)
        compressed = _read_stable(cache_path)
        if len(compressed) != int(record["raw_cache_bytes"]):
            raise FreezeReceiptError(f"Yahoo cache byte mismatch for {yahoo_ticker}")
        if hashlib.sha256(compressed).hexdigest() != str(record["raw_cache_sha256"]):
            raise FreezeReceiptError(f"Yahoo cache hash mismatch for {yahoo_ticker}")
        try:
            raw = gzip.decompress(compressed)
        except (OSError, EOFError) as exc:
            raise FreezeReceiptError(f"Yahoo cache is not valid gzip for {yahoo_ticker}") from exc
        if hashlib.sha256(raw).hexdigest() != str(record["raw_payload_sha256"]):
            raise FreezeReceiptError(f"Yahoo raw payload hash mismatch for {yahoo_ticker}")
        if str(record.get("sha256")) != str(record["raw_payload_sha256"]):
            raise FreezeReceiptError(
                f"Yahoo vendor and archived raw hashes differ for {yahoo_ticker}"
            )
        raw_payloads[yahoo_ticker] = raw
        if str(record.get("provider")) != "Yahoo Finance chart API":
            raise FreezeReceiptError(f"Yahoo provider mismatch for {yahoo_ticker}")
        if str(record.get("dataset")) != f"daily_prices:{yahoo_ticker}":
            raise FreezeReceiptError(f"Yahoo dataset name mismatch for {yahoo_ticker}")
        if str(record.get("finality")) != "confirmed":
            raise FreezeReceiptError(f"Yahoo history is not confirmed for {yahoo_ticker}")
        vendor_rows[yahoo_ticker] = int(record.get("rows", -1))
    try:
        download_audit = pd.read_csv(io.BytesIO(support_contents["download_audit"]))
    except Exception as exc:
        raise FreezeReceiptError("download audit cannot be parsed") from exc
    if set(download_audit.columns) != {"ticker", "status", "rows"}:
        raise FreezeReceiptError("download audit schema differs")
    if download_audit["ticker"].duplicated().any():
        raise FreezeReceiptError("download audit contains duplicate tickers")
    requested_yahoo = {pair[1] for pair in alias_pairs}
    if set(download_audit["ticker"].astype(str)) != requested_yahoo:
        raise FreezeReceiptError("download audit does not cover every requested ticker")
    successful_yahoo = set(
        download_audit.loc[download_audit["status"].eq("ok"), "ticker"].astype(str)
    )
    if BENCHMARK not in successful_yahoo:
        raise FreezeReceiptError("SPY acquisition did not succeed")
    if seen_vendor_tickers != successful_yahoo:
        raise FreezeReceiptError("vendor records differ from successful download audit")
    audit_row_map = {
        str(row.ticker): int(row.rows)
        for row in download_audit.loc[
            download_audit["status"].eq("ok"), ["ticker", "rows"]
        ].itertuples(index=False)
    }
    if audit_row_map != vendor_rows:
        raise FreezeReceiptError(
            "download audit row counts differ from Yahoo vendor records"
        )

    try:
        alias_csv = pd.read_csv(io.BytesIO(support_contents["ticker_aliases"]))
    except Exception as exc:
        raise FreezeReceiptError("ticker alias CSV cannot be parsed") from exc
    if set(alias_csv.columns) != {
        "source_ticker",
        "yahoo_ticker",
        "is_benchmark",
    }:
        raise FreezeReceiptError("ticker alias CSV schema differs")
    alias_csv_pairs = set(
        alias_csv[["source_ticker", "yahoo_ticker"]]
        .astype(str)
        .itertuples(index=False, name=None)
    )
    if alias_csv_pairs != alias_pairs:
        raise FreezeReceiptError("ticker alias CSV differs from manifest aliases")
    actual_alias_rows = sorted(
        (
            str(row.source_ticker),
            str(row.yahoo_ticker),
            bool(row.is_benchmark),
        )
        for row in alias_csv.itertuples(index=False)
    )
    expected_alias_rows = sorted(
        (
            str(row.source_ticker),
            str(row.yahoo_ticker),
            bool(row.is_benchmark),
        )
        for row in expected_alias_frame.itertuples(index=False)
    )
    manifest_alias_rows = sorted(
        (
            str(row.get("source_ticker")),
            str(row.get("yahoo_ticker")),
            bool(row.get("is_benchmark")),
        )
        for row in aliases
    )
    if actual_alias_rows != expected_alias_rows or manifest_alias_rows != expected_alias_rows:
        raise FreezeReceiptError(
            "ticker alias CSV or manifest rows differ from candidate recomputation"
        )

    try:
        exclusion_csv = pd.read_csv(
            io.BytesIO(support_contents["ticker_exclusions"]), dtype="string"
        ).fillna("")
    except Exception as exc:
        raise FreezeReceiptError("ticker exclusion CSV cannot be parsed") from exc
    expected_exclusion_csv = expected_ticker_exclusions.astype("string").fillna("")
    if list(exclusion_csv.columns) != list(expected_exclusion_csv.columns):
        raise FreezeReceiptError("ticker exclusion CSV schema differs")
    exclusion_records = sorted(
        tuple(row)
        for row in exclusion_csv.astype(str).itertuples(index=False, name=None)
    )
    expected_exclusion_records = sorted(
        tuple(row)
        for row in expected_exclusion_csv.astype(str).itertuples(
            index=False, name=None
        )
    )
    if exclusion_records != expected_exclusion_records:
        raise FreezeReceiptError(
            "ticker exclusion CSV differs from frozen-candidate recomputation"
        )

    reconstructed = _reconstruct_clean_price_panel_from_raw(
        raw_payloads,
        alias_pairs=alias_pairs,
        expected_vendor_rows=vendor_rows,
    )
    try:
        frozen_price_panel = pd.read_parquet(
            io.BytesIO(declared_contents["price_file"])
        )
    except Exception as exc:
        raise FreezeReceiptError("frozen price parquet cannot be parsed") from exc
    _assert_price_panel_matches_reconstruction(
        frozen_price_panel,
        reconstructed,
        expected_source_tickers={pair[0] for pair in alias_pairs},
    )
    benchmark_preflight = benchmark_clock_preflight(
        frozen_candidates,
        reconstructed,
    )
    limitations = payload["limitations"]
    if not isinstance(limitations, Mapping):
        raise FreezeReceiptError("price manifest limitations must be an object")
    for required_false in (
        "permanent_security_identifiers_complete",
        "delisting_returns_complete",
        "survivorship_free",
    ):
        if limitations.get(required_false) is not False:
            raise FreezeReceiptError(f"limitation {required_false} must remain false")
    if limitations.get("stage_a_candidate_audit") != expected_candidate_audit:
        raise FreezeReceiptError(
            "candidate audit summary differs from frozen-candidate recomputation"
        )
    if limitations.get("benchmark_clock_preflight") != benchmark_preflight:
        raise FreezeReceiptError(
            "benchmark clock preflight differs from frozen-input recomputation"
        )
    if int(limitations.get("retained_panel_rows", -1)) != len(reconstructed):
        raise FreezeReceiptError("retained panel row count differs from reconstruction")
    if int(limitations.get("downloaded_panel_rows", -1)) != sum(vendor_rows.values()):
        raise FreezeReceiptError("downloaded panel row count differs from vendor records")
    if int(limitations.get("removed_panel_rows", -1)) != (
        sum(vendor_rows.values()) - len(reconstructed)
    ):
        raise FreezeReceiptError("removed panel row count differs from reconstruction")
    design_payload = _strict_json(
        declared_contents["design_receipt"], source=str(expected_design_receipt)
    )
    design_frozen_at = pd.Timestamp(design_payload.get("frozen_at"))
    manifest_created_at = pd.Timestamp(payload.get("created_at"))
    if (
        pd.isna(design_frozen_at)
        or pd.isna(manifest_created_at)
        or design_frozen_at.tzinfo is None
        or manifest_created_at.tzinfo is None
        or design_frozen_at.tz_convert("UTC")
        > manifest_created_at.tz_convert("UTC")
    ):
        raise FreezeReceiptError(
            "price manifest was not created after the pre-price design freeze"
        )
    for record in vendor_records:
        retrieved_at = pd.Timestamp(record.get("retrieved_at"))
        if (
            pd.isna(retrieved_at)
            or retrieved_at.tzinfo is None
            or retrieved_at.tz_convert("UTC") < design_frozen_at.tz_convert("UTC")
            or retrieved_at.tz_convert("UTC")
            > manifest_created_at.tz_convert("UTC")
        ):
            raise FreezeReceiptError(
                "Yahoo retrieval timestamp is outside the design-freeze/manifest window"
            )


def build_freeze_receipt(
    *,
    config_path: str | Path,
    core_code_path: str | Path,
    outcome_runner_path: str | Path,
    receipt_path: str | Path,
    repository_root: str | Path,
) -> Path:
    """Freeze all outcome-bearing inputs; refuses to overwrite a receipt."""

    root = Path(repository_root).resolve()
    config_target = _require_canonical_path(
        config_path,
        root=root,
        canonical_relative=CANONICAL_RUNTIME_CONFIG,
        role="runtime config",
    )
    core_target = _require_canonical_path(
        core_code_path,
        root=root,
        canonical_relative=CANONICAL_CORE_CODE,
        role="core code",
    )
    runner_target = _require_canonical_path(
        outcome_runner_path,
        root=root,
        canonical_relative=CANONICAL_OUTCOME_RUNNER,
        role="outcome runner",
    )
    receipt = _require_canonical_path(
        receipt_path,
        root=root,
        canonical_relative=CANONICAL_FREEZE_RECEIPT,
        role="freeze receipt",
    )
    config_bytes = _read_stable(config_target)
    config = _strict_json(config_bytes, source=str(config_target))
    validate_stage_a_config(config)
    human_protocol = _resolve_under_root(root, config["sources"]["human_protocol"])
    human_payload = _strict_json(
        _read_stable(human_protocol), source=str(human_protocol)
    )
    validate_human_protocol(human_payload)
    candidate = _resolve_under_root(root, config["sources"]["candidate_accessions"])
    candidate_upstream_manifest = _resolve_under_root(
        root, config["sources"]["candidate_upstream_manifest"]
    )
    yahoo_adapter = _resolve_under_root(root, config["sources"]["yahoo_adapter"])
    design_receipt = _resolve_under_root(root, config["sources"]["design_receipt"])
    validate_human_protocol_lineage(
        human_payload,
        repository_root=root,
        expected_candidate=candidate,
        expected_candidate_upstream_manifest=candidate_upstream_manifest,
        expected_yahoo_adapter=yahoo_adapter,
    )
    prices = _resolve_under_root(root, config["sources"]["price_panel"])
    price_manifest = _resolve_under_root(root, config["sources"]["price_manifest"])
    verify_design_receipt(
        design_receipt,
        expected_paths={
            "human_protocol": human_protocol,
            "core_code": core_target,
            "outcome_runner": runner_target,
            "acquisition_script": root / CANONICAL_ACQUISITION_SCRIPT,
            "candidate_accessions": candidate,
            "candidate_upstream_manifest": candidate_upstream_manifest,
            "yahoo_adapter": yahoo_adapter,
        },
        repository_root=root,
    )
    manifest_payload = _strict_json(
        _read_stable(price_manifest), source=str(price_manifest)
    )
    _validate_price_manifest_payload(
        manifest_payload,
        root=root,
        expected_candidate=candidate,
        expected_candidate_upstream_manifest=candidate_upstream_manifest,
        expected_yahoo_adapter=yahoo_adapter,
        expected_design_receipt=design_receipt,
        expected_price=prices,
    )
    paths = {
        "config": config_target,
        "human_protocol": human_protocol,
        "core_code": core_target,
        "outcome_runner": runner_target,
        "candidate_accessions": candidate,
        "candidate_upstream_manifest": candidate_upstream_manifest,
        "yahoo_adapter": yahoo_adapter,
        "design_receipt": design_receipt,
        "price_panel": prices,
        "price_manifest": price_manifest,
    }
    records = [_receipt_record(role, paths[role], root) for role in sorted(paths)]
    receipt.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": FREEZE_RECEIPT_SCHEMA,
        "frozen_at": datetime.now(UTC).isoformat(),
        "immutable": True,
        "candidate_issuer_price_or_return_join_before_freeze": False,
        "records": records,
    }
    try:
        with receipt.open("x", encoding="utf-8") as handle:
            handle.write(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
                + "\n"
            )
    except FileExistsError as exc:
        raise FreezeReceiptError(f"freeze receipt already exists: {receipt}") from exc
    return receipt


def _verify_freeze_receipt_with_contents(
    receipt_path: str | Path,
    *,
    expected_paths: Mapping[str, str | Path],
    repository_root: str | Path,
) -> tuple[dict[str, Any], dict[str, bytes], bytes]:
    """Verify each stable byte string once and return those exact bytes."""

    root = Path(repository_root).resolve()
    receipt = _require_canonical_path(
        receipt_path,
        root=root,
        canonical_relative=CANONICAL_FREEZE_RECEIPT,
        role="freeze receipt",
    )
    fixed_receipt_paths = {
        "config": CANONICAL_RUNTIME_CONFIG,
        "human_protocol": CANONICAL_HUMAN_PROTOCOL,
        "core_code": CANONICAL_CORE_CODE,
        "outcome_runner": CANONICAL_OUTCOME_RUNNER,
        "design_receipt": CANONICAL_DESIGN_RECEIPT,
        "price_panel": CANONICAL_PRICE_PANEL,
        "price_manifest": CANONICAL_PRICE_MANIFEST,
    }
    for role, canonical in fixed_receipt_paths.items():
        if role not in expected_paths:
            raise FreezeReceiptError(f"runner omitted frozen role {role}")
        _require_canonical_path(
            expected_paths[role],
            root=root,
            canonical_relative=canonical,
            role=role.replace("_", " "),
        )
    receipt_content = _read_stable(receipt)
    payload = _strict_json(receipt_content, source=str(receipt))
    _exact_keys(
        payload,
        {
            "schema_version",
            "frozen_at",
            "immutable",
            "candidate_issuer_price_or_return_join_before_freeze",
            "records",
        },
        name="freeze receipt",
    )
    if payload["schema_version"] != FREEZE_RECEIPT_SCHEMA:
        raise FreezeReceiptError("unsupported freeze receipt schema")
    if (
        payload["immutable"] is not True
        or payload["candidate_issuer_price_or_return_join_before_freeze"]
        is not False
    ):
        raise FreezeReceiptError(
            "freeze receipt does not establish a pre-issuer-price/return-join freeze"
        )
    records = payload["records"]
    if not isinstance(records, list):
        raise FreezeReceiptError("freeze receipt records must be a list")
    roles = [str(record.get("role")) for record in records if isinstance(record, Mapping)]
    if set(roles) != REQUIRED_RECEIPT_ROLES or len(roles) != len(REQUIRED_RECEIPT_ROLES):
        raise FreezeReceiptError(
            f"freeze receipt roles differ: {sorted(set(roles))}"
        )
    if set(expected_paths) != REQUIRED_RECEIPT_ROLES:
        raise FreezeReceiptError("runner did not supply every required frozen role")
    verified_contents: dict[str, bytes] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise FreezeReceiptError("freeze receipt record is not an object")
        _exact_keys(record, {"role", "path", "bytes", "sha256"}, name="receipt record")
        role = str(record["role"])
        expected = Path(expected_paths[role]).resolve()
        if _resolve_under_root(root, str(record["path"])) != expected:
            raise FreezeReceiptError(f"frozen path mismatch for {role}")
        content = _read_stable(expected)
        if len(content) != int(record["bytes"]):
            raise FreezeReceiptError(f"frozen byte count mismatch for {role}")
        if hashlib.sha256(content).hexdigest() != str(record["sha256"]):
            raise FreezeReceiptError(f"frozen hash mismatch for {role}")
        verified_contents[role] = content
    return payload, verified_contents, receipt_content


def verify_freeze_receipt(
    receipt_path: str | Path,
    *,
    expected_paths: Mapping[str, str | Path],
    repository_root: str | Path,
) -> dict[str, Any]:
    """Verify exact role coverage, paths, bytes, and hashes before outcomes."""

    payload, _, _ = _verify_freeze_receipt_with_contents(
        receipt_path,
        expected_paths=expected_paths,
        repository_root=repository_root,
    )
    return payload


def _normalize_price_panel(
    prices: pd.DataFrame, *, expected_source_tickers: set[str]
) -> pd.DataFrame:
    required = {
        "source_ticker",
        "yahoo_ticker",
        "date",
        "timestamp",
        "adj_open",
        "adj_close",
        "close",
        "volume",
    }
    _require(prices, required, name="price panel")
    frame = prices[list(required)].copy()
    frame["source_ticker"] = frame["source_ticker"].astype("string").str.strip().str.upper()
    frame["yahoo_ticker"] = frame["yahoo_ticker"].astype("string").str.strip().str.upper()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any():
        raise Form4StageAError("price panel contains invalid date")
    if getattr(frame["date"].dt, "tz", None) is not None:
        frame["date"] = frame["date"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    frame["date"] = frame["date"].dt.normalize()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if frame["timestamp"].isna().any():
        raise Form4StageAError("price panel contains invalid open timestamp")
    for column in ("adj_open", "adj_close", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if (~np.isfinite(frame[column])).any():
            raise Form4StageAError(f"price panel contains non-finite {column}")
    if (
        frame["adj_open"].le(0).any()
        or frame["adj_close"].le(0).any()
        or frame["close"].le(0).any()
    ):
        raise Form4StageAError("price panel contains non-positive price")
    if frame["volume"].lt(0).any():
        raise Form4StageAError("price panel contains negative volume")
    expected_alias = frame["source_ticker"].map(yahoo_alias)
    benchmark = frame["source_ticker"].eq(BENCHMARK)
    if expected_alias.isna().any() or (
        ~frame["yahoo_ticker"].eq(expected_alias.where(~benchmark, BENCHMARK))
    ).any():
        raise Form4StageAError("price panel violates frozen Yahoo alias mapping")
    actual_tickers = set(frame["source_ticker"].astype(str).unique())
    if BENCHMARK not in actual_tickers or not actual_tickers.issubset(expected_source_tickers):
        raise Form4StageAError(
            "price panel has no SPY or contains unrequested tickers; "
            f"extra={sorted(actual_tickers - expected_source_tickers)}"
        )
    validate_unique(frame, ["source_ticker", "date"], "price panel")
    local_date = frame["timestamp"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    if ~local_date.eq(frame["date"]).all():
        raise Form4StageAError("price open timestamp does not match its New York date")
    spy_local = frame.loc[benchmark, "timestamp"].dt.tz_convert("America/New_York")
    regular_open = (
        spy_local.dt.hour.eq(9)
        & spy_local.dt.minute.eq(30)
        & spy_local.dt.second.eq(0)
    )
    if spy_local.empty or not bool(regular_open.all()):
        raise Form4StageAError("SPY timestamps must be exact 09:30 New York opens")
    frame["dollar_volume"] = frame["close"] * frame["volume"]
    return frame.sort_values(["source_ticker", "date"], kind="mergesort").reset_index(drop=True)


def benchmark_clock_preflight(
    candidates: pd.DataFrame,
    prices: pd.DataFrame,
) -> dict[str, Any]:
    """Require complete SPY clocks for every fixed development-period event.

    This uses event labels and the benchmark calendar only. It does not join an
    issuer price or compute any event return.
    """

    events, _, _ = select_candidate_events(candidates)
    if events.empty:
        raise Form4StageAError("benchmark preflight has no selected events")
    expected_tickers = set(events["source_ticker"].astype(str)) | {BENCHMARK}
    panel = _normalize_price_panel(
        prices, expected_source_tickers=expected_tickers
    )
    spy = panel.loc[panel["source_ticker"].eq(BENCHMARK)].sort_values(
        "date", kind="mergesort"
    ).reset_index(drop=True)
    if len(spy) < 275:
        raise Form4StageAError("clean SPY history is too short for Stage A")
    expected_last = pd.Timestamp(ACQUISITION_END)
    actual_last = pd.Timestamp(spy["date"].max())
    if actual_last != expected_last:
        raise Form4StageAError(
            "clean SPY history does not reach the fixed acquisition end; "
            f"expected {expected_last.date()}, observed {actual_last.date()}"
        )

    spy_dates = spy["date"].to_numpy(dtype="datetime64[ns]")
    horizon = int(EXPECTED_PROTOCOL["clock"]["exit_sessions_after_entry"])
    failures: list[dict[str, str]] = []
    for item in events[["accession", "event_class", "filing_date"]].to_dict(
        "records"
    ):
        filing = pd.Timestamp(item["filing_date"]).normalize()
        first_after = int(
            np.searchsorted(spy_dates, np.datetime64(filing), side="right")
        )
        entry_index = first_after + 1
        exit_index = entry_index + horizon
        placebo_end_index = int(
            np.searchsorted(spy_dates, np.datetime64(filing), side="left") - 1
        )
        placebo_start_index = placebo_end_index - horizon
        if (
            first_after >= len(spy_dates)
            or entry_index >= len(spy_dates)
            or exit_index >= len(spy_dates)
            or placebo_start_index < 1
        ):
            failures.append(
                {
                    "accession": str(item["accession"]),
                    "event_class": str(item["event_class"]),
                    "filing_date": filing.date().isoformat(),
                }
            )
    if failures:
        raise Form4StageAError(
            "SPY benchmark clock is incomplete before price freeze; "
            f"failed={len(failures)} examples={failures[:5]}"
        )
    class_counts = events["event_class"].value_counts().to_dict()
    return {
        "spy_session_rows": int(len(spy)),
        "spy_first_date": pd.Timestamp(spy["date"].min()).date().isoformat(),
        "spy_last_date": actual_last.date().isoformat(),
        "selected_event_rows": int(len(events)),
        "opportunistic_event_rows": int(class_counts.get(PRIMARY_CLASS, 0)),
        "routine_event_rows": int(class_counts.get(NEGATIVE_CONTROL_CLASS, 0)),
        "complete_event_clock_rows": int(len(events)),
        "all_selected_event_clocks_complete": True,
        "issuer_price_or_return_join_performed": False,
    }


def _exclusion(row: Mapping[str, Any], reason: str) -> dict[str, Any]:
    return {
        "accession_ids": str(row.get("accession_ids", row.get("accession", ""))),
        "filing_date": row.get("filing_date"),
        "source_ticker": row.get("source_ticker"),
        "event_class": row.get("event_class"),
        "entry_date": row.get("entry_date"),
        "schedule_id": row.get("schedule_id"),
        "reason": reason,
    }


def _daily_beta_sample(
    spy: pd.DataFrame,
    stock_by_date: pd.DataFrame,
    *,
    beta_end_date: pd.Timestamp,
) -> pd.DataFrame:
    aligned = spy[["date", "adj_open"]].rename(columns={"adj_open": "spy_adj_open"}).copy()
    aligned = aligned.merge(
        stock_by_date[["date", "adj_open"]].rename(columns={"adj_open": "stock_adj_open"}),
        on="date",
        how="left",
        validate="one_to_one",
    )
    aligned["stock_return"] = aligned["stock_adj_open"].pct_change(fill_method=None)
    aligned["spy_return"] = aligned["spy_adj_open"].pct_change(fill_method=None)
    sample = aligned.loc[
        aligned["date"].le(beta_end_date)
        & np.isfinite(aligned["stock_return"])
        & np.isfinite(aligned["spy_return"])
    ].tail(int(EXPECTED_PROTOCOL["beta"]["maximum_matched_adjusted_open_returns"]))
    return sample.reset_index(drop=True)


def _transaction_cost(
    *, ws: float, wm: float, stock_return: float, spy_return: float, one_way_rate: float
) -> float:
    return float(
        one_way_rate
        * (
            abs(ws)
            + abs(wm)
            + abs(ws * (1.0 + stock_return))
            + abs(wm * (1.0 + spy_return))
        )
    )


def attach_stage_a_outcomes(
    events: pd.DataFrame,
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the fixed clock, eligibility, beta, return, cost, and overlap rules."""

    required_events = {
        "accession",
        "filing_date",
        "source_ticker",
        "yahoo_ticker",
        "event_class",
        "direct_purchase_usd",
    }
    _require(events, required_events, name="selected events")
    if events["filing_date"].ge(HOLDOUT_START).any():
        raise Form4StageAError("outcome join attempted on the 2025 holdout cohort")
    expected_tickers = set(events["source_ticker"].astype(str)) | {BENCHMARK}
    panel = _normalize_price_panel(prices, expected_source_tickers=expected_tickers)
    spy = panel.loc[panel["source_ticker"].eq(BENCHMARK)].sort_values("date").reset_index(drop=True)
    if len(spy) < 275:
        raise Form4StageAError("SPY panel is too short for the fixed beta/event design")
    spy_dates = spy["date"].to_numpy(dtype="datetime64[ns]")

    clocked: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for item in events.sort_values(
        ["filing_date", "event_class", "source_ticker", "accession"], kind="mergesort"
    ).to_dict("records"):
        filing = pd.Timestamp(item["filing_date"]).normalize()
        strict_after = np.flatnonzero(spy_dates > np.datetime64(filing))
        if strict_after.size < 2:
            exclusions.append(_exclusion(item, "fewer_than_two_spy_opens_after_filing"))
            continue
        entry_index = int(strict_after[1])
        exit_index = entry_index + int(EXPECTED_PROTOCOL["clock"]["exit_sessions_after_entry"])
        placebo_end_index = int(np.searchsorted(spy_dates, np.datetime64(filing), side="left") - 1)
        placebo_start_index = placebo_end_index - int(
            EXPECTED_PROTOCOL["clock"]["exit_sessions_after_entry"]
        )
        if placebo_start_index < 1:
            exclusions.append(_exclusion(item, "placebo_or_beta_history_unavailable"))
            continue
        clocked.append(
            {
                **item,
                "entry_index": entry_index,
                "exit_index": exit_index,
                "placebo_start_index": placebo_start_index,
                "placebo_end_index": placebo_end_index,
                "quarantined_index": int(strict_after[0]),
                "quarantined_date": spy.loc[int(strict_after[0]), "date"],
                "entry_date": spy.loc[entry_index, "date"],
                "entry_at": spy.loc[entry_index, "timestamp"],
                "exit_date": spy.loc[exit_index, "date"] if exit_index < len(spy) else pd.NaT,
                "exit_at": spy.loc[exit_index, "timestamp"] if exit_index < len(spy) else pd.NaT,
                "placebo_start_date": spy.loc[placebo_start_index, "date"],
                "placebo_end_date": spy.loc[placebo_end_index, "date"],
            }
        )
    if not clocked:
        raise Form4StageAError("no events have complete benchmark clock coverage")
    clocked_frame = pd.DataFrame(clocked)

    # Same class, ticker, and executable entry becomes one event.  The latest
    # filing date preserves the most conservative information boundary.
    collapsed_rows: list[dict[str, Any]] = []
    group_keys = ["event_class", "source_ticker", "yahoo_ticker", "entry_date"]
    for _, group in clocked_frame.groupby(group_keys, sort=True, dropna=False):
        latest = group.sort_values(["filing_date", "accession"], kind="mergesort").iloc[-1]
        row = latest.to_dict()
        row["accession_ids"] = "|".join(sorted(group["accession"].astype(str).unique()))
        row["collapsed_accession_count"] = int(group["accession"].nunique())
        row["collapsed_direct_purchase_usd"] = float(group["direct_purchase_usd"].sum())
        collapsed_rows.append(row)
    collapsed = pd.DataFrame(collapsed_rows).sort_values(
        ["event_class", "source_ticker", "entry_date", "accession_ids"], kind="mergesort"
    )

    # Lock the event schedule before looking at terminal prices.  A missing
    # future exit can fail coverage, but can never allow a later overlapping
    # signal to replace the originally scheduled event.
    scheduled_rows: list[pd.Series] = []
    for (_, _), group in collapsed.groupby(["event_class", "source_ticker"], sort=True):
        last_exit_index: int | None = None
        for _, row in group.sort_values(["entry_index", "accession_ids"], kind="mergesort").iterrows():
            entry_index = int(row["entry_index"])
            if last_exit_index is not None and entry_index <= last_exit_index:
                exclusions.append(_exclusion(row, "overlap_same_class_until_prior_exit"))
                continue
            scheduled_rows.append(row)
            last_exit_index = int(row["exit_index"])
    collapsed = pd.DataFrame(scheduled_rows)

    # A ticker cannot serve as both the treatment and negative control on the
    # same executable entry date.  Remove both labels before reading any stock
    # return so the control comparison is not mechanically contaminated.
    cross_class_conflict = collapsed.groupby(
        ["source_ticker", "entry_date"], sort=False
    )["event_class"].transform("nunique").gt(1)
    for item in collapsed.loc[cross_class_conflict].to_dict("records"):
        exclusions.append(
            _exclusion(item, "cross_class_same_ticker_entry_conflict")
        )
    collapsed = collapsed.loc[~cross_class_conflict].copy()
    if collapsed.empty:
        raise Form4StageAError("event scheduling removed every candidate")
    collapsed["schedule_id"] = collapsed.apply(
        lambda row: (
            f"{row['event_class']}|{row['source_ticker']}|"
            f"{pd.Timestamp(row['entry_date']).date().isoformat()}"
        ),
        axis=1,
    )
    validate_unique(collapsed, ["schedule_id"], "frozen scheduled events")

    computed: list[dict[str, Any]] = []
    for item in collapsed.to_dict("records"):
        ticker = str(item["source_ticker"])
        stock = panel.loc[panel["source_ticker"].eq(ticker)].sort_values("date").reset_index(drop=True)
        if stock.empty:
            exclusions.append(_exclusion(item, "ticker_price_history_unavailable"))
            continue
        stock_lookup = stock.set_index("date")
        entry_index = int(item["entry_index"])
        exit_index = int(item["exit_index"])
        if exit_index >= len(spy):
            exclusions.append(_exclusion(item, "exit_spy_open_unavailable"))
            continue
        entry_date = pd.Timestamp(item["entry_date"])
        exit_date = pd.Timestamp(item["exit_date"])
        placebo_start = pd.Timestamp(item["placebo_start_date"])
        placebo_end = pd.Timestamp(item["placebo_end_date"])
        required_dates = [entry_date, exit_date, placebo_start, placebo_end]
        if any(date not in stock_lookup.index for date in required_dates):
            exclusions.append(_exclusion(item, "required_exact_stock_open_or_prior_close_missing"))
            continue
        prior = stock.loc[stock["date"].lt(entry_date)]
        prior_observed_sessions = int(len(prior))
        if prior_observed_sessions < int(
            EXPECTED_PROTOCOL["eligibility"]["minimum_prior_observed_sessions"]
        ):
            exclusions.append(_exclusion(item, "fewer_than_252_prior_stock_sessions"))
            continue
        adv_window = prior.tail(
            int(EXPECTED_PROTOCOL["eligibility"]["prior_median_dollar_volume_sessions"])
        )
        if len(adv_window) != int(
            EXPECTED_PROTOCOL["eligibility"]["prior_median_dollar_volume_sessions"]
        ):
            exclusions.append(_exclusion(item, "incomplete_63_session_adv_window"))
            continue
        prior_median_dollar_volume = float(adv_window["dollar_volume"].median())
        if prior_median_dollar_volume < float(
            EXPECTED_PROTOCOL["eligibility"]["minimum_prior_median_dollar_volume_usd"]
        ):
            exclusions.append(_exclusion(item, "prior_median_dollar_volume_below_20m"))
            continue

        beta_sample = _daily_beta_sample(
            spy,
            stock,
            beta_end_date=placebo_start,
        )
        if len(beta_sample) < int(
            EXPECTED_PROTOCOL["beta"]["minimum_matched_adjusted_open_returns"]
        ):
            exclusions.append(_exclusion(item, "fewer_than_200_matched_beta_returns"))
            continue
        market_variance = float(beta_sample["spy_return"].var(ddof=1))
        if not np.isfinite(market_variance) or market_variance <= 0:
            exclusions.append(_exclusion(item, "nonpositive_beta_market_variance"))
            continue
        beta = float(
            beta_sample[["stock_return", "spy_return"]].cov(ddof=1).iloc[0, 1]
            / market_variance
        )
        if not np.isfinite(beta):
            exclusions.append(_exclusion(item, "nonfinite_beta"))
            continue
        ws = float(1.0 / (1.0 + abs(beta)))
        wm = float(-beta / (1.0 + abs(beta)))
        stock_return = float(
            stock_lookup.loc[exit_date, "adj_open"]
            / stock_lookup.loc[entry_date, "adj_open"]
            - 1.0
        )
        spy_return = float(
            spy.loc[exit_index, "adj_open"] / spy.loc[entry_index, "adj_open"] - 1.0
        )
        placebo_stock_return = float(
            stock_lookup.loc[placebo_end, "adj_open"]
            / stock_lookup.loc[placebo_start, "adj_open"]
            - 1.0
        )
        placebo_spy_return = float(
            spy.loc[int(item["placebo_end_index"]), "adj_open"]
            / spy.loc[int(item["placebo_start_index"]), "adj_open"]
            - 1.0
        )
        gross = float(ws * stock_return + wm * spy_return)
        placebo_gross = float(ws * placebo_stock_return + wm * placebo_spy_return)
        cost30 = _transaction_cost(
            ws=ws,
            wm=wm,
            stock_return=stock_return,
            spy_return=spy_return,
            one_way_rate=float(EXPECTED_PROTOCOL["costs"]["primary_one_way_rate"]),
        )
        cost50 = _transaction_cost(
            ws=ws,
            wm=wm,
            stock_return=stock_return,
            spy_return=spy_return,
            one_way_rate=float(EXPECTED_PROTOCOL["costs"]["stress_one_way_rate"]),
        )
        computed.append(
            {
                **item,
                "prior_observed_sessions": prior_observed_sessions,
                "prior_history_end_date": prior["date"].max(),
                "adv_window_end_date": adv_window["date"].max(),
                "prior_63_session_median_dollar_volume": prior_median_dollar_volume,
                "beta": beta,
                "beta_observations": int(len(beta_sample)),
                "beta_sample_start_date": beta_sample["date"].min(),
                "beta_sample_end_date": beta_sample["date"].max(),
                "stock_weight": ws,
                "benchmark_weight": wm,
                "entry_gross_notional": abs(ws) + abs(wm),
                "stock_return_20": stock_return,
                "spy_return_20": spy_return,
                "hedged_gross_return_20": gross,
                "traded_notional_cost_30bps": cost30,
                "hedged_net_return_30bps": gross - cost30,
                "traded_notional_cost_50bps": cost50,
                "hedged_net_return_50bps": gross - cost50,
                "placebo_stock_return_20": placebo_stock_return,
                "placebo_spy_return_20": placebo_spy_return,
                "placebo_hedged_gross_return_20": placebo_gross,
            }
        )
    if not computed:
        raise Form4StageAError("no events pass the fixed price, liquidity, and beta gates")
    result = pd.DataFrame(computed).sort_values(
        ["event_class", "source_ticker", "entry_date", "accession_ids"], kind="mergesort"
    ).reset_index(drop=True)
    validate_unique(
        result,
        ["event_class", "source_ticker", "entry_date"],
        "collapsed non-overlapping Stage A events",
    )
    required_times = ["entry_at", "exit_at"]
    required_numeric = [
        "beta",
        "stock_weight",
        "benchmark_weight",
        "stock_return_20",
        "spy_return_20",
        "hedged_gross_return_20",
        "traded_notional_cost_30bps",
        "hedged_net_return_30bps",
        "traded_notional_cost_50bps",
        "hedged_net_return_50bps",
        "placebo_stock_return_20",
        "placebo_spy_return_20",
        "placebo_hedged_gross_return_20",
    ]
    if result[required_times].isna().any().any() or not np.isfinite(
        result[required_numeric].to_numpy(dtype=float)
    ).all():
        raise Form4StageAError(
            "retained outcomes contain missing or non-finite formal-test fields"
        )
    exclusion_frame = pd.DataFrame(
        exclusions,
        columns=[
            "accession_ids",
            "filing_date",
            "source_ticker",
            "event_class",
            "entry_date",
            "schedule_id",
            "reason",
        ],
    )
    return result, exclusion_frame


def aggregate_entry_dates(events: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight eligible events within each class and entry date."""

    metrics = [
        "hedged_gross_return_20",
        "hedged_net_return_30bps",
        "hedged_net_return_50bps",
        "placebo_hedged_gross_return_20",
    ]
    _require(
        events,
        {"event_class", "entry_date", "entry_index", *metrics},
        name="event outcomes",
    )
    grouped = events.groupby(["event_class", "entry_date"], sort=True)
    if grouped["entry_index"].nunique().gt(1).any():
        raise Form4StageAError("one entry date maps to multiple SPY session indices")
    result = grouped[metrics].mean().reset_index()
    counts = grouped.size().rename("event_count").reset_index()
    result = result.merge(counts, on=["event_class", "entry_date"], validate="one_to_one")
    indices = grouped["entry_index"].first().rename("entry_index").reset_index()
    result = result.merge(
        indices, on=["event_class", "entry_date"], validate="one_to_one"
    )
    validate_unique(result, ["event_class", "entry_date"], "entry-date returns")
    return result.sort_values(["event_class", "entry_date"], kind="mergesort").reset_index(drop=True)


def hac_mean_test(
    values: Iterable[float],
    *,
    session_indices: Iterable[int] | None = None,
    lags: int = 20,
) -> dict[str, Any]:
    x = np.asarray(list(values), dtype=float)
    if session_indices is None:
        indices = np.arange(len(x), dtype=int)
    else:
        indices = np.asarray(list(session_indices), dtype=int)
        if len(indices) != len(x):
            raise Form4StageAError("HAC values and session indices differ in length")
    finite = np.isfinite(x)
    x = x[finite]
    indices = indices[finite]
    if len(indices) and (
        np.any(np.diff(indices) <= 0) or len(np.unique(indices)) != len(indices)
    ):
        raise Form4StageAError("HAC session indices must be strictly increasing")
    n = int(x.size)
    if n < 3:
        return {
            "status": "INSUFFICIENT_ENTRY_DATES",
            "n": n,
            "mean": float(np.mean(x)) if n else None,
            "requested_lags": int(lags),
            "effective_lags": int(lags),
            "lag_unit": "SPY_session_distance",
            "standard_error": None,
            "t": None,
            "one_sided_p": None,
        }
    mean = float(x.mean())
    centered = x - mean
    sample_variance = float(np.var(centered, ddof=1))
    if not np.isfinite(sample_variance) or sample_variance <= 1e-20:
        return {
            "status": "DEGENERATE_VARIANCE",
            "n": n,
            "mean": mean,
            "requested_lags": int(lags),
            "effective_lags": int(lags),
            "lag_unit": "SPY_session_distance",
            "standard_error": None,
            "t": None,
            "one_sided_p": None,
        }
    effective_lags = int(lags)
    variance_sum = float(np.dot(centered, centered))
    for right in range(1, n):
        gaps = indices[right] - indices[:right]
        included = gaps <= int(lags)
        if included.any():
            weights = 1.0 - gaps[included] / (int(lags) + 1.0)
            variance_sum += float(
                2.0
                * np.dot(
                    weights,
                    centered[right] * centered[:right][included],
                )
            )
    variance_of_mean = variance_sum / (n * n)
    if not np.isfinite(variance_of_mean) or variance_of_mean <= 1e-24:
        return {
            "status": "DEGENERATE_VARIANCE",
            "n": n,
            "mean": mean,
            "requested_lags": int(lags),
            "effective_lags": effective_lags,
            "lag_unit": "SPY_session_distance",
            "standard_error": None,
            "t": None,
            "one_sided_p": None,
        }
    standard_error = float(math.sqrt(variance_of_mean))
    t_value = float(mean / standard_error)
    return {
        "status": "OK",
        "n": n,
        "mean": mean,
        "requested_lags": int(lags),
        "effective_lags": effective_lags,
        "lag_unit": "SPY_session_distance",
        "standard_error": standard_error,
        "t": t_value,
        "one_sided_p": float(norm.sf(t_value)),
    }


def centered_circular_mbb(
    values: Iterable[float],
    *,
    draws: int = 10_000,
    block_length: int = 20,
    seed: int = 929,
) -> dict[str, Any]:
    """One-sided null test with centered circular all-start moving blocks."""

    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n < 3:
        return {
            "status": "INSUFFICIENT_ENTRY_DATES",
            "n": n,
            "observed_mean": float(np.mean(x)) if n else None,
            "draws": int(draws),
            "block_length": int(block_length),
            "seed": int(seed),
            "null_mean": None,
            "null_std": None,
            "one_sided_empirical_p": None,
            "one_sided_binomial_95pct_upper": None,
        }
    if draws < 1 or block_length < 1:
        raise Form4StageAError("bootstrap draws and block length must be positive")
    observed = float(x.mean())
    centered = x - observed
    sample_variance = float(np.var(centered, ddof=1))
    if not np.isfinite(sample_variance) or sample_variance <= 1e-20:
        return {
            "status": "DEGENERATE_VARIANCE",
            "n": n,
            "observed_mean": observed,
            "draws": int(draws),
            "block_length": int(block_length),
            "seed": int(seed),
            "null_mean": None,
            "null_std": None,
            "one_sided_empirical_p": None,
            "one_sided_binomial_95pct_upper": None,
        }
    blocks_needed = int(math.ceil(n / int(block_length)))
    offsets = np.arange(int(block_length), dtype=int)
    rng = np.random.default_rng(int(seed))
    null_means = np.empty(int(draws), dtype=float)
    for draw in range(int(draws)):
        starts = rng.integers(0, n, size=blocks_needed)
        indices = ((starts[:, None] + offsets[None, :]) % n).reshape(-1)[:n]
        null_means[draw] = float(centered[indices].mean())
    exceedances = int(np.count_nonzero(null_means >= observed))
    binomial_upper = (
        1.0
        if exceedances == int(draws)
        else float(
            beta_distribution.ppf(
                0.95,
                exceedances + 1,
                int(draws) - exceedances,
            )
        )
    )
    return {
        "status": "OK",
        "variant": "centered_circular_all_start_moving_blocks",
        "n": n,
        "observed_mean": observed,
        "draws": int(draws),
        "block_length": int(block_length),
        "seed": int(seed),
        "null_mean": float(null_means.mean()),
        "null_std": float(null_means.std(ddof=1)),
        "exceedances": exceedances,
        "one_sided_empirical_p": float((exceedances + 1) / (int(draws) + 1)),
        "one_sided_binomial_95pct_upper": binomial_upper,
        "minimum_resolvable_p": float(1 / (int(draws) + 1)),
    }


def formal_stage_a_tests(entry_dates: pd.DataFrame) -> dict[str, Any]:
    """Run four authored hypotheses with a disclosed-trial-floor screen."""

    opp = entry_dates.loc[entry_dates["event_class"].eq(PRIMARY_CLASS)].copy()
    routine = entry_dates.loc[
        entry_dates["event_class"].eq(NEGATIVE_CONTROL_CLASS)
    ].copy()
    paired = opp[["entry_date", "entry_index", "hedged_net_return_30bps"]].merge(
        routine[["entry_date", "entry_index", "hedged_net_return_30bps"]],
        on="entry_date",
        how="inner",
        suffixes=("_opportunistic", "_routine"),
        validate="one_to_one",
    )
    paired["difference"] = (
        paired["hedged_net_return_30bps_opportunistic"]
        - paired["hedged_net_return_30bps_routine"]
    )
    if not paired["entry_index_opportunistic"].eq(
        paired["entry_index_routine"]
    ).all():
        raise Form4StageAError("paired classes disagree on SPY entry index")
    series = {
        "opportunistic_net_30bps_mean_20_sessions": (
            opp["hedged_net_return_30bps"].to_numpy(dtype=float),
            opp["entry_index"].to_numpy(dtype=int),
        ),
        "opportunistic_minus_routine_net_30bps_mean_20_sessions": (
            paired["difference"].to_numpy(dtype=float),
            paired["entry_index_opportunistic"].to_numpy(dtype=int),
        ),
        "opportunistic_signed_pre_event_placebo_20_sessions": (
            opp["placebo_hedged_gross_return_20"].to_numpy(dtype=float),
            opp["entry_index"].to_numpy(dtype=int),
        ),
        "opportunistic_net_50bps_mean_20_sessions": (
            opp["hedged_net_return_50bps"].to_numpy(dtype=float),
            opp["entry_index"].to_numpy(dtype=int),
        ),
    }
    expected_names = EXPECTED_PROTOCOL["inference"]["formal_tests"]
    if list(series) != expected_names:
        raise Form4StageAError("formal test registry differs from authored protocol")
    floor = int(
        EXPECTED_PROTOCOL["inference"][
            "disclosed_trial_floor_screening_multiplier"
        ]
    )
    critical = float(norm.ppf(1.0 - 0.05 / floor))
    output: dict[str, Any] = {}
    for name, (values, session_indices) in series.items():
        hac = hac_mean_test(values, session_indices=session_indices, lags=20)
        mbb = centered_circular_mbb(
            values, draws=10_000, block_length=20, seed=929
        )
        hac_p = hac["one_sided_p"]
        mbb_p = mbb["one_sided_empirical_p"]
        mbb_upper = mbb["one_sided_binomial_95pct_upper"]
        output[name] = {
            "series_unit": (
                "common_entry_date_pair"
                if name.startswith("opportunistic_minus_routine")
                else "opportunistic_entry_date"
            ),
            "disclosed_trial_floor_screening_multiplier": floor,
            "family_wise_error_control_established": False,
            "known_trial_floor_one_sided_normal_critical_value": critical,
            "hac20": {
                **hac,
                "known_trial_floor_adjusted_screening_p": (
                    min(1.0, float(hac_p) * floor) if hac_p is not None else None
                ),
            },
            "centered_circular_mbb": {
                **mbb,
                "decision_role": "NON_DECISION_SENSITIVITY",
                "known_trial_floor_adjusted_screening_p": (
                    min(1.0, float(mbb_p) * floor) if mbb_p is not None else None
                ),
                "known_trial_floor_adjusted_binomial_95pct_upper": (
                    min(1.0, float(mbb_upper) * floor)
                    if mbb_upper is not None
                    else None
                ),
            },
        }
    return output


def scheduled_event_coverage_ledger(
    events: pd.DataFrame,
    exclusions: pd.DataFrame,
) -> pd.DataFrame:
    """Materialize the frozen post-suppression schedule and path status."""

    _require(
        events,
        {"schedule_id", "event_class", "source_ticker", "entry_date"},
        name="event outcomes",
    )
    complete_filter_reasons = {
        "prior_median_dollar_volume_below_20m",
        "nonpositive_beta_market_variance",
        "nonfinite_beta",
    }
    successful = events[
        ["schedule_id", "event_class", "source_ticker", "entry_date"]
    ].copy()
    successful["terminal_status"] = "retained_outcome"
    successful["price_path_complete"] = True

    if "schedule_id" in exclusions.columns:
        failed = exclusions.loc[
            exclusions["schedule_id"].notna(),
            ["schedule_id", "event_class", "source_ticker", "entry_date", "reason"],
        ].copy()
    else:
        failed = pd.DataFrame(
            columns=[
                "schedule_id",
                "event_class",
                "source_ticker",
                "entry_date",
                "reason",
            ]
        )
    failed = failed.rename(columns={"reason": "terminal_status"})
    failed["price_path_complete"] = failed["terminal_status"].isin(
        complete_filter_reasons
    )
    ledger = (
        successful.reset_index(drop=True)
        if failed.empty
        else pd.concat([successful, failed], ignore_index=True, sort=False)
    )
    if ledger.empty:
        raise Form4StageAError("scheduled-event ledger is empty")
    validate_unique(ledger, ["schedule_id"], "scheduled-event coverage ledger")
    return ledger.sort_values(
        ["event_class", "entry_date", "source_ticker"], kind="mergesort"
    ).reset_index(drop=True)


def summarize_stage_a(
    events: pd.DataFrame,
    entry_dates: pd.DataFrame,
    exclusions: pd.DataFrame,
    candidate_audit: Mapping[str, Any],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for event_class in EVENT_CLASSES:
        sample = entry_dates.loc[entry_dates["event_class"].eq(event_class)]
        class_events = events.loc[events["event_class"].eq(event_class)]
        descriptive_means = {
            metric: (float(sample[metric].mean()) if not sample.empty else None)
            for metric in (
            "hedged_gross_return_20",
            "hedged_net_return_30bps",
            "hedged_net_return_50bps",
            "placebo_hedged_gross_return_20",
            )
        }
        results[event_class] = {
            "events": int(len(class_events)),
            "entry_dates": int(len(sample)),
            "tickers": int(class_events["source_ticker"].nunique()),
            "descriptive_means": descriptive_means,
        }
    formal_tests = formal_stage_a_tests(entry_dates)
    opp_events = events.loc[events["event_class"].eq(PRIMARY_CLASS)]
    clock_failure_reasons = {
        "fewer_than_two_spy_opens_after_filing",
        "placebo_or_beta_history_unavailable",
        "exit_spy_open_unavailable",
    }
    selected_primary_rows = int(candidate_audit["opportunistic_rows"])
    primary_clock_failures = int(
        (
            exclusions["event_class"].eq(PRIMARY_CLASS)
            & exclusions["reason"].isin(clock_failure_reasons)
        ).sum()
    )
    benchmark_clock_coverage = (
        float((selected_primary_rows - primary_clock_failures) / selected_primary_rows)
        if selected_primary_rows
        else 0.0
    )
    coverage_ledger = scheduled_event_coverage_ledger(events, exclusions)
    primary_coverage = coverage_ledger.loc[
        coverage_ledger["event_class"].eq(PRIMARY_CLASS)
    ]
    coverage_denominator = int(len(primary_coverage))
    coverage_numerator = int(primary_coverage["price_path_complete"].sum())
    scheduled_coverage = (
        float(coverage_numerator / coverage_denominator)
        if coverage_denominator
        else 0.0
    )
    routine_coverage = coverage_ledger.loc[
        coverage_ledger["event_class"].eq(NEGATIVE_CONTROL_CLASS)
    ]
    routine_coverage_denominator = int(len(routine_coverage))
    routine_coverage_numerator = int(routine_coverage["price_path_complete"].sum())
    routine_scheduled_coverage = (
        float(routine_coverage_numerator / routine_coverage_denominator)
        if routine_coverage_denominator
        else 0.0
    )
    required_event_fields = [
        "entry_at",
        "exit_at",
        "beta",
        "stock_weight",
        "benchmark_weight",
        "hedged_gross_return_20",
        "stock_return_20",
        "spy_return_20",
        "hedged_net_return_30bps",
        "hedged_net_return_50bps",
        "placebo_hedged_gross_return_20",
    ]
    retained_complete = bool(
        not events.empty
        and events[required_event_fields].notna().all().all()
        and np.isfinite(
            events[[
                "beta",
                "stock_weight",
                "benchmark_weight",
                "hedged_gross_return_20",
                "stock_return_20",
                "spy_return_20",
                "hedged_net_return_30bps",
                "hedged_net_return_50bps",
                "placebo_hedged_gross_return_20",
            ]].to_numpy(dtype=float)
        ).all()
    )
    mapping_denominator = int(
        candidate_audit["preclock_development_rows_in_two_classes"]
    )
    candidate_ticker_mapping_coverage = (
        float(candidate_audit["selected_development_rows"] / mapping_denominator)
        if mapping_denominator
        else 0.0
    )

    contribution_frame = opp_events[[
        "entry_date", "source_ticker", "hedged_net_return_30bps"
    ]].copy()
    if not contribution_frame.empty:
        contribution_frame["events_on_entry_date"] = contribution_frame.groupby(
            "entry_date"
        )["source_ticker"].transform("size")
        contribution_frame["portfolio_contribution"] = contribution_frame[
            "hedged_net_return_30bps"
        ] / (
            contribution_frame["events_on_entry_date"]
            * contribution_frame["entry_date"].nunique()
        )
        contribution_frame["gross_absolute_contribution"] = contribution_frame[
            "portfolio_contribution"
        ].abs()
        ticker_contribution = contribution_frame.groupby("source_ticker")[
            "gross_absolute_contribution"
        ].sum()
    else:
        ticker_contribution = pd.Series(dtype=float)
    concentration = (
        float(ticker_contribution.max() / ticker_contribution.sum())
        if not ticker_contribution.empty and ticker_contribution.sum() > 0
        else 1.0
    )
    leave_one_ticker_out_means: dict[str, float] = {}
    for ticker in sorted(opp_events["source_ticker"].astype(str).unique()):
        remaining = opp_events.loc[~opp_events["source_ticker"].eq(ticker)]
        if remaining.empty:
            continue
        leave_one_ticker_out_means[ticker] = float(
            remaining.groupby("entry_date", sort=True)[
                "hedged_net_return_30bps"
            ].mean().mean()
        )
    minimum_leave_one_out = (
        float(min(leave_one_ticker_out_means.values()))
        if leave_one_ticker_out_means
        else None
    )
    extreme_returns = int(
        (
            events["stock_return_20"].abs().gt(1.0)
            | events["spy_return_20"].abs().gt(1.0)
        ).sum()
    )
    ordered_events = events.sort_values(
        ["event_class", "source_ticker", "entry_index"], kind="mergesort"
    )
    previous_exit = ordered_events.groupby(
        ["event_class", "source_ticker"], sort=False
    )["exit_index"].shift(1)
    no_overlap = bool(
        (
            previous_exit.isna()
            | ordered_events["entry_index"].gt(previous_exit)
        ).all()
    )
    integrity = {
        "no_2025_event_outcomes": bool(events["filing_date"].lt(HOLDOUT_START).all()),
        "second_spy_open_after_filing": bool(
            events["filing_date"].lt(events["quarantined_date"]).all()
            and events["quarantined_date"].lt(events["entry_date"]).all()
            and events["entry_index"].sub(events["quarantined_index"]).eq(1).all()
        ),
        "exit_twenty_spy_sessions_after_entry": bool(
            events["exit_index"].sub(events["entry_index"]).eq(20).all()
        ),
        "all_beta_and_liquidity_observation_dates_precede_entry": bool(
            events["beta_sample_end_date"].le(events["placebo_start_date"]).all()
            and events["placebo_end_date"].lt(events["filing_date"]).all()
            and events["prior_history_end_date"].lt(events["entry_date"]).all()
            and events["adv_window_end_date"].lt(events["entry_date"]).all()
        ),
        "no_same_ticker_entry_cross_class": bool(
            events.groupby(["source_ticker", "entry_date"])["event_class"]
            .nunique()
            .le(1)
            .all()
        ),
        "no_same_class_ticker_overlap": no_overlap,
        "all_retained_outcome_fields_finite": retained_complete,
    }
    return {
        "schema_version": SUMMARY_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "historical_alpha_verdict": "INVALID",
        "model_verdict": "NOT VALIDATED",
        "deployment_verdict": "NOT AUTHORIZED",
        "candidate_classification": "PROVISIONAL_FILING_DATE_PROXY",
        "price_lineage": "CURRENT_TICKER_YAHOO_RETROSPECTIVE",
        "holdout_2025_return_outcomes": "NO_2025_EVENT_RETURN_JOIN",
        "protocol": EXPECTED_PROTOCOL,
        "candidate_audit": dict(candidate_audit),
        "usable_events": int(len(events)),
        "entry_date_rows": int(len(entry_dates)),
        "excluded_rows": int(len(exclusions)),
        "candidate_ticker_mapping_coverage": candidate_ticker_mapping_coverage,
        "opportunistic_benchmark_clock_coverage": benchmark_clock_coverage,
        "opportunistic_benchmark_clock_failures": primary_clock_failures,
        "scheduled_opportunistic_event_price_path_coverage": scheduled_coverage,
        "scheduled_opportunistic_path_coverage_numerator": coverage_numerator,
        "scheduled_opportunistic_path_coverage_denominator": coverage_denominator,
        "scheduled_routine_event_price_path_coverage": routine_scheduled_coverage,
        "scheduled_routine_path_coverage_numerator": routine_coverage_numerator,
        "scheduled_routine_path_coverage_denominator": routine_coverage_denominator,
        "maximum_single_ticker_gross_absolute_contribution_share": concentration,
        "minimum_leave_one_ticker_out_net30_mean": minimum_leave_one_out,
        "leave_one_ticker_out_net30_means": leave_one_ticker_out_means,
        "absolute_stock_or_spy_horizon_returns_over_100pct": extreme_returns,
        "integrity_checks": integrity,
        "results": results,
        "formal_test_count": len(formal_tests),
        "formal_tests": formal_tests,
    }


def build_stage_a_gates(summary: Mapping[str, Any]) -> tuple[pd.DataFrame, str]:
    """Evaluate the authored kill-test gates without changing INVALID lineage."""

    opp = summary["results"][PRIMARY_CLASS]
    tests = summary["formal_tests"]
    primary = tests["opportunistic_net_30bps_mean_20_sessions"]
    mechanism = tests["opportunistic_minus_routine_net_30bps_mean_20_sessions"]
    placebo = tests["opportunistic_signed_pre_event_placebo_20_sessions"]
    stress = tests["opportunistic_net_50bps_mean_20_sessions"]

    def value(test: Mapping[str, Any], key: str) -> float | None:
        observed = test["hac20"].get(key)
        return float(observed) if observed is not None else None

    primary_mean = value(primary, "mean")
    primary_t = value(primary, "t")
    primary_p = value(primary, "known_trial_floor_adjusted_screening_p")
    mechanism_mean = value(mechanism, "mean")
    placebo_mean = value(placebo, "mean")
    placebo_raw_p = value(placebo, "one_sided_p")
    stress_mean = value(stress, "mean")
    critical = float(
        primary["known_trial_floor_one_sided_normal_critical_value"]
    )
    integrity = summary["integrity_checks"]
    placebo_falsification_passed = bool(
        placebo_mean is not None
        and placebo_raw_p is not None
        and not (placebo_mean > 0 and placebo_raw_p <= 0.05)
    )
    minimum_leave_one_out = summary["minimum_leave_one_ticker_out_net30_mean"]

    raw_rows = [
        ("formal_test_count_is_four", summary["formal_test_count"] == 4, True, summary["formal_test_count"]),
        (
            "disclosed_trial_floor_screening_multiplier_is_124",
            primary["disclosed_trial_floor_screening_multiplier"] == 124,
            True,
            primary["disclosed_trial_floor_screening_multiplier"],
        ),
        (
            "family_wise_error_control_not_claimed",
            primary["family_wise_error_control_established"] is False,
            True,
            primary["family_wise_error_control_established"],
        ),
        ("minimum_usable_opportunistic_events", opp["events"] >= 100, True, opp["events"]),
        ("minimum_usable_opportunistic_tickers", opp["tickers"] >= 50, True, opp["tickers"]),
        ("minimum_opportunistic_entry_sessions", opp["entry_dates"] >= 80, True, opp["entry_dates"]),
        ("minimum_common_entry_dates_for_mechanism", mechanism["hac20"]["n"] >= 30, True, mechanism["hac20"]["n"]),
        (
            "minimum_candidate_ticker_mapping_coverage",
            summary["candidate_ticker_mapping_coverage"] >= 0.95,
            True,
            summary["candidate_ticker_mapping_coverage"],
        ),
        (
            "complete_opportunistic_benchmark_clock_coverage",
            summary["opportunistic_benchmark_clock_coverage"] == 1.0,
            True,
            summary["opportunistic_benchmark_clock_coverage"],
        ),
        (
            "minimum_scheduled_opportunistic_event_price_path_coverage",
            summary["scheduled_opportunistic_event_price_path_coverage"] >= 0.8,
            True,
            summary["scheduled_opportunistic_event_price_path_coverage"],
        ),
        (
            "minimum_scheduled_routine_event_price_path_coverage",
            summary["scheduled_routine_event_price_path_coverage"] >= 0.8,
            True,
            summary["scheduled_routine_event_price_path_coverage"],
        ),
        ("primary_net_mean_positive", primary_mean is not None and primary_mean > 0, True, primary_mean),
        ("primary_hac_t_known_trial_floor_screen", primary_t is not None and primary_t >= critical, True, primary_t),
        ("primary_known_trial_floor_adjusted_screening_p", primary_p is not None and primary_p <= 0.05, True, primary_p),
        ("stress_50bps_net_mean_positive", stress_mean is not None and stress_mean > 0, True, stress_mean),
        ("opportunistic_minus_routine_positive", mechanism_mean is not None and mechanism_mean > 0, True, mechanism_mean),
        (
            "no_significant_positive_pre_event_placebo",
            placebo_falsification_passed,
            True,
            {"mean": placebo_mean, "raw_one_sided_p": placebo_raw_p},
        ),
        (
            "single_ticker_gross_absolute_contribution_share",
            summary["maximum_single_ticker_gross_absolute_contribution_share"] <= 0.1,
            True,
            summary["maximum_single_ticker_gross_absolute_contribution_share"],
        ),
        (
            "leave_one_ticker_out_net30_mean_positive",
            minimum_leave_one_out is not None and minimum_leave_one_out > 0,
            True,
            minimum_leave_one_out,
        ),
        ("no_absolute_horizon_return_over_100pct", summary["absolute_stock_or_spy_horizon_returns_over_100pct"] == 0, True, summary["absolute_stock_or_spy_horizon_returns_over_100pct"]),
        ("no_2025_event_outcome_join", integrity["no_2025_event_outcomes"], True, integrity["no_2025_event_outcomes"]),
        ("second_spy_open_after_filing", integrity["second_spy_open_after_filing"], True, integrity["second_spy_open_after_filing"]),
        ("exit_twenty_spy_sessions_after_entry", integrity["exit_twenty_spy_sessions_after_entry"], True, integrity["exit_twenty_spy_sessions_after_entry"]),
        (
            "all_beta_and_liquidity_observation_dates_precede_entry",
            integrity["all_beta_and_liquidity_observation_dates_precede_entry"],
            True,
            integrity["all_beta_and_liquidity_observation_dates_precede_entry"],
        ),
        ("no_same_ticker_entry_cross_class", integrity["no_same_ticker_entry_cross_class"], True, integrity["no_same_ticker_entry_cross_class"]),
        ("no_same_class_ticker_overlap", integrity["no_same_class_ticker_overlap"], True, integrity["no_same_class_ticker_overlap"]),
        ("all_retained_outcome_fields_finite", integrity["all_retained_outcome_fields_finite"], True, integrity["all_retained_outcome_fields_finite"]),
        (
            "circular_mbb_non_decision_sensitivity",
            True,
            False,
            primary["centered_circular_mbb"],
        ),
        ("historical_exact_first_public_clock", False, False, "filing-date proxy"),
        ("permanent_security_and_delisting_lineage", False, False, "current-ticker Yahoo proxy"),
    ]
    rows = pd.DataFrame(
        [
            {"gate": name, "passed": bool(passed), "stage_a_decision_required": required, "detail": detail}
            for name, passed, required, detail in raw_rows
        ]
    )
    advance = bool(rows.loc[rows["stage_a_decision_required"], "passed"].all())
    decision = (
        "ADVANCE_TO_STAGE_B_EXACT_CLOCK_AND_PERMANENT_SECURITY_ENGINEERING"
        if advance
        else "RETIRE_STAGE_A_PROXY"
    )
    return rows, decision


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=str)
        + "\n",
        encoding="utf-8",
    )


def _write_stage_a_charts(
    *,
    events: pd.DataFrame,
    entry_dates: pd.DataFrame,
    summary: Mapping[str, Any],
    point_estimates_path: Path,
    cumulative_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    metrics = [
        ("hedged_gross_return_20", "Gross"),
        ("hedged_net_return_30bps", "Net 30 bps"),
        ("hedged_net_return_50bps", "Net 50 bps"),
        ("placebo_hedged_gross_return_20", "Pre-event placebo"),
    ]
    x = np.arange(len(metrics), dtype=float)
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for offset, event_class, color in (
        (-width / 2, PRIMARY_CLASS, "#2563eb"),
        (width / 2, NEGATIVE_CONTROL_CLASS, "#94a3b8"),
    ):
        means = summary["results"][event_class]["descriptive_means"]
        values = [
            float(means[key]) * 10_000 if means[key] is not None else np.nan
            for key, _ in metrics
        ]
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=event_class.title(),
            color=color,
            alpha=0.9,
        )
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    ax.axhline(0, color="#0f172a", linewidth=1)
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylabel("Mean 20-session return (bps)")
    ax.set_title("Form 4 Stage A point estimates")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(point_estimates_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for event_class, color in (
        (PRIMARY_CLASS, "#2563eb"),
        (NEGATIVE_CONTROL_CLASS, "#94a3b8"),
    ):
        sample = entry_dates.loc[
            entry_dates["event_class"].eq(event_class)
        ].sort_values("entry_date")
        ax.plot(
            sample["entry_date"],
            sample["hedged_net_return_30bps"].cumsum() * 100,
            label=event_class.title(),
            color=color,
            linewidth=1.8,
        )
    ax.axhline(0, color="#0f172a", linewidth=1)
    ax.set_ylabel("Cumulative arithmetic diagnostic (%)")
    ax.set_title("Net 30 bps entry-date diagnostic (not portfolio P&L)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(cumulative_path, dpi=180)
    plt.close(fig)


def _artifact_manifest(
    output: Path,
    paths: Mapping[str, Path],
    *,
    frozen_sources: Mapping[str, tuple[Path, bytes]],
    root: Path,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for role, path in sorted(paths.items()):
        records.append(
            {
                "category": "output",
                "role": role,
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    for role, (path, content) in sorted(frozen_sources.items()):
        records.append(
            {
                "category": "frozen_input_or_code",
                "role": role,
                "path": _relative_to_root(path, root),
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    return {
        "schema_version": ARTIFACT_MANIFEST_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "output_directory": output.as_posix(),
        "records": records,
    }


def run_stage_a_diagnostic(
    *,
    config_path: str | Path,
    freeze_receipt_path: str | Path,
    outcome_runner_path: str | Path,
    repository_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Verify frozen prices, then mark before issuer-price/return joins."""

    root = Path(repository_root).resolve()
    config_target = _require_canonical_path(
        config_path,
        root=root,
        canonical_relative=CANONICAL_RUNTIME_CONFIG,
        role="runtime config",
    )
    _require_canonical_path(
        freeze_receipt_path,
        root=root,
        canonical_relative=CANONICAL_FREEZE_RECEIPT,
        role="freeze receipt",
    )
    core_target = _require_canonical_path(
        Path(__file__),
        root=root,
        canonical_relative=CANONICAL_CORE_CODE,
        role="core code",
    )
    runner_target = _require_canonical_path(
        outcome_runner_path,
        root=root,
        canonical_relative=CANONICAL_OUTCOME_RUNNER,
        role="outcome runner",
    )
    output = _require_canonical_path(
        output_dir,
        root=root,
        canonical_relative=CANONICAL_RUN_DIRECTORY,
        role="run directory",
    )
    if output.exists():
        raise Form4StageAError(f"output directory already exists: {output}")

    # Reading the config bytes reveals paths, not returns.  Its hash is checked
    # before candidate or price parquet bytes are opened.
    config_bytes = _read_stable(config_target)
    config = _strict_json(config_bytes, source=str(config_target))
    validate_stage_a_config(config)
    candidate_path = _resolve_under_root(root, config["sources"]["candidate_accessions"])
    candidate_upstream_manifest_path = _resolve_under_root(
        root, config["sources"]["candidate_upstream_manifest"]
    )
    yahoo_adapter_path = _resolve_under_root(root, config["sources"]["yahoo_adapter"])
    design_receipt_path = _resolve_under_root(
        root, config["sources"]["design_receipt"]
    )
    price_path = _resolve_under_root(root, config["sources"]["price_panel"])
    price_manifest_path = _resolve_under_root(root, config["sources"]["price_manifest"])
    human_protocol_path = _resolve_under_root(root, config["sources"]["human_protocol"])
    expected_paths = {
        "config": config_target,
        "human_protocol": human_protocol_path,
        "core_code": core_target,
        "outcome_runner": runner_target,
        "candidate_accessions": candidate_path,
        "candidate_upstream_manifest": candidate_upstream_manifest_path,
        "yahoo_adapter": yahoo_adapter_path,
        "design_receipt": design_receipt_path,
        "price_panel": price_path,
        "price_manifest": price_manifest_path,
    }
    receipt, frozen, receipt_content = _verify_freeze_receipt_with_contents(
        freeze_receipt_path,
        expected_paths=expected_paths,
        repository_root=root,
    )
    verified_config = _strict_json(frozen["config"], source=str(config_target))
    validate_stage_a_config(verified_config)
    if verified_config != config:
        raise FreezeReceiptError("config changed between path resolution and verification")
    human_payload = _strict_json(
        frozen["human_protocol"], source=str(human_protocol_path)
    )
    validate_human_protocol(human_payload)
    validate_human_protocol_lineage(
        human_payload,
        repository_root=root,
        expected_candidate=candidate_path,
        expected_candidate_upstream_manifest=candidate_upstream_manifest_path,
        expected_yahoo_adapter=yahoo_adapter_path,
    )
    verify_design_receipt(
        design_receipt_path,
        expected_paths={
            "human_protocol": human_protocol_path,
            "core_code": core_target,
            "outcome_runner": runner_target,
            "acquisition_script": root
            / "scripts/acquire_form4_opportunistic_stage_a_prices.py",
            "candidate_accessions": candidate_path,
            "candidate_upstream_manifest": candidate_upstream_manifest_path,
            "yahoo_adapter": yahoo_adapter_path,
        },
        repository_root=root,
    )
    price_manifest = _strict_json(
        frozen["price_manifest"], source=str(price_manifest_path)
    )
    _validate_price_manifest_payload(
        price_manifest,
        root=root,
        expected_candidate=candidate_path,
        expected_candidate_upstream_manifest=candidate_upstream_manifest_path,
        expected_yahoo_adapter=yahoo_adapter_path,
        expected_design_receipt=design_receipt_path,
        expected_price=price_path,
    )

    # Atomically reserve the one-shot directory and record all four opened
    # trials before any candidate-to-issuer-price join or event return computation.
    # Benchmark-calendar mapping was already verified. A later failure leaves an
    # auditable opened-run marker and cannot be silently rerun.
    output.mkdir(parents=True, exist_ok=False)
    run_opened_path = output / "run_opened_before_outcomes.json"
    opened_trial_ledger_path = output / "opened_trial_ledger.json"
    _write_json(
        run_opened_path,
        {
            "schema_version": "form4_opportunistic_stage_a_run_opened_v1",
            "opened_at": datetime.now(UTC).isoformat(),
            "freeze_receipt_sha256": hashlib.sha256(receipt_content).hexdigest(),
            "state_at_write": (
                "FROZEN_PRICE_PANEL_AND_BENCHMARK_CLOCK_VERIFIED_BUT_"
                "NO_CANDIDATE_TO_ISSUER_PRICE_OR_RETURN_JOIN"
            ),
            "purpose": "one-shot marker written immediately before event-return computation",
        },
    )
    _write_json(
        opened_trial_ledger_path,
        {
            "schema_version": "form4_opportunistic_stage_a_opened_trial_ledger_v1",
            "opened_at": datetime.now(UTC).isoformat(),
            "prior_disclosed_trial_floor": 120,
            "opened_trials": [
                {
                    "trial_id": name,
                    "outcome_opened": True,
                    "alternative": "mean_greater_than_zero",
                    "series_unit": (
                        "common_entry_date_pair"
                        if name.startswith("opportunistic_minus_routine")
                        else "opportunistic_entry_date"
                    ),
                }
                for name in EXPECTED_PROTOCOL["inference"]["formal_tests"]
            ],
            "opened_trial_count": 4,
            "resulting_disclosed_trial_floor": 124,
            "family_wise_error_control_established": False,
            "use_of_trial_floor": "known-trial-floor screening hurdle only",
            "alternate_variants_opened": False,
            "holdout_2025_event_outcomes_opened": False,
        },
    )

    # The parquet bytes consumed below are the exact byte strings already
    # verified against the receipt, closing the post-verification TOCTOU gap.
    candidates = pd.read_parquet(io.BytesIO(frozen["candidate_accessions"]))
    selected, ticker_exclusions, candidate_audit = select_candidate_events(candidates)
    prices = pd.read_parquet(io.BytesIO(frozen["price_panel"]))
    events, event_exclusions = attach_stage_a_outcomes(selected, prices)
    exclusions = pd.concat([ticker_exclusions, event_exclusions], ignore_index=True, sort=False)
    entry_dates = aggregate_entry_dates(events)
    coverage_ledger = scheduled_event_coverage_ledger(events, exclusions)
    summary = summarize_stage_a(events, entry_dates, exclusions, candidate_audit)
    gates, candidate_decision = build_stage_a_gates(summary)
    summary["candidate_decision"] = candidate_decision

    paths = {
        "run_opened_before_outcomes": run_opened_path,
        "event_outcomes": output / "event_outcomes.parquet",
        "entry_date_returns": output / "entry_date_returns.csv",
        "exclusions": output / "exclusions.csv",
        "scheduled_event_coverage": output / "scheduled_event_coverage.csv",
        "summary": output / "summary.json",
        "validation_gates": output / "validation_gates.csv",
        "run_manifest": output / "run_manifest.json",
        "opened_trial_ledger": opened_trial_ledger_path,
        "point_estimates_chart": output / "point_estimates_bps.png",
        "cumulative_diagnostic_chart": output / "cumulative_diagnostic_net30.png",
        "report": output / "report-source.md",
    }
    events.to_parquet(paths["event_outcomes"], index=False)
    entry_dates.to_csv(paths["entry_date_returns"], index=False)
    exclusions.to_csv(paths["exclusions"], index=False)
    coverage_ledger.to_csv(paths["scheduled_event_coverage"], index=False)
    _write_json(paths["summary"], summary)
    gates = pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "gate": "freeze_receipt_verified_before_outcome_join",
                        "passed": True,
                        "stage_a_decision_required": True,
                        "detail": hashlib.sha256(receipt_content).hexdigest(),
                    }
                ]
            ),
            gates,
        ],
        ignore_index=True,
    )
    gates.to_csv(paths["validation_gates"], index=False)
    _write_stage_a_charts(
        events=events,
        entry_dates=entry_dates,
        summary=summary,
        point_estimates_path=paths["point_estimates_chart"],
        cumulative_path=paths["cumulative_diagnostic_chart"],
    )
    run_manifest = {
        "schema_version": "form4_opportunistic_stage_a_run_manifest_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "historical_alpha_verdict": "INVALID",
        "holdout_2025_return_outcomes": "NO_2025_EVENT_RETURN_JOIN",
        "freeze_receipt_sha256": hashlib.sha256(receipt_content).hexdigest(),
        "freeze_receipt_frozen_at": receipt["frozen_at"],
        "config_sha256": hashlib.sha256(frozen["config"]).hexdigest(),
        "human_protocol_sha256": hashlib.sha256(
            frozen["human_protocol"]
        ).hexdigest(),
        "core_code_sha256": hashlib.sha256(frozen["core_code"]).hexdigest(),
        "outcome_runner_sha256": hashlib.sha256(
            frozen["outcome_runner"]
        ).hexdigest(),
        "candidate_accessions_sha256": hashlib.sha256(
            frozen["candidate_accessions"]
        ).hexdigest(),
        "candidate_upstream_manifest_sha256": hashlib.sha256(
            frozen["candidate_upstream_manifest"]
        ).hexdigest(),
        "yahoo_adapter_sha256": hashlib.sha256(
            frozen["yahoo_adapter"]
        ).hexdigest(),
        "design_receipt_sha256": hashlib.sha256(
            frozen["design_receipt"]
        ).hexdigest(),
        "price_panel_sha256": hashlib.sha256(frozen["price_panel"]).hexdigest(),
        "price_manifest_sha256": hashlib.sha256(
            frozen["price_manifest"]
        ).hexdigest(),
    }
    _write_json(paths["run_manifest"], run_manifest)
    opp = summary["results"][PRIMARY_CLASS]
    routine = summary["results"][NEGATIVE_CONTROL_CLASS]
    primary = summary["formal_tests"][
        "opportunistic_net_30bps_mean_20_sessions"
    ]
    mechanism = summary["formal_tests"][
        "opportunistic_minus_routine_net_30bps_mean_20_sessions"
    ]
    placebo = summary["formal_tests"][
        "opportunistic_signed_pre_event_placebo_20_sessions"
    ]
    stress = summary["formal_tests"][
        "opportunistic_net_50bps_mean_20_sessions"
    ]

    def bps(value: object) -> str:
        return "NA" if value is None else f"{float(value) * 10_000:.2f} bps"

    def number(value: object, digits: int = 4) -> str:
        return "NA" if value is None else f"{float(value):.{digits}f}"

    failed_required = gates.loc[
        gates["stage_a_decision_required"] & ~gates["passed"], "gate"
    ].astype(str).tolist()
    paths["report"].write_text(
        "# Form 4 opportunistic purchase Stage A diagnostic\n\n"
        "## Verdict\n\n"
        "Historical alpha: **INVALID**. The event clock is a filing-date proxy "
        "and price history is a current-ticker Yahoo reconstruction without "
        "permanent identifiers or complete delisting payoffs. The candidate "
        "file's 2025 event metadata was read and counted, while **no 2025 event "
        "was joined to prices and no 2025 event return was computed**.\n\n"
        "## Frozen result\n\n"
        f"- Stage A decision: **{candidate_decision}**.\n"
        f"- Opportunistic: {opp['events']} events across {opp['entry_dates']} entry dates.\n"
        f"- Routine negative control: {routine['events']} events across {routine['entry_dates']} entry dates.\n"
        f"- Opportunistic gross mean: {bps(opp['descriptive_means']['hedged_gross_return_20'])}.\n"
        f"- Opportunistic net 30 bps mean: {bps(primary['hac20']['mean'])}; "
        f"HAC t={number(primary['hac20']['t'])}; raw p={number(primary['hac20']['one_sided_p'], 6)}; "
        f"known-124-trial-floor screening p={number(primary['hac20']['known_trial_floor_adjusted_screening_p'], 6)}.\n"
        f"- Circular MBB raw p={number(primary['centered_circular_mbb']['one_sided_empirical_p'], 6)}; "
        f"known-floor screening p={number(primary['centered_circular_mbb']['known_trial_floor_adjusted_screening_p'], 6)}; "
        f"known-floor 95% Monte Carlo upper={number(primary['centered_circular_mbb']['known_trial_floor_adjusted_binomial_95pct_upper'], 6)}; "
        "non-decision sensitivity only.\n"
        f"- Net 50 bps stress mean: {bps(stress['hac20']['mean'])}.\n"
        f"- Paired opportunistic-minus-routine mean: {bps(mechanism['hac20']['mean'])} "
        f"across {mechanism['hac20']['n']} common entry dates.\n"
        f"- Pre-event placebo mean: {bps(placebo['hac20']['mean'])}.\n"
        f"- Candidate ticker mapping coverage: {summary['candidate_ticker_mapping_coverage']:.2%}.\n"
        f"- Benchmark clock coverage: {summary['opportunistic_benchmark_clock_coverage']:.2%}.\n"
        f"- Frozen scheduled-event price-path coverage: "
        f"{summary['scheduled_opportunistic_event_price_path_coverage']:.2%}.\n"
        f"- Routine-control scheduled-event price-path coverage: "
        f"{summary['scheduled_routine_event_price_path_coverage']:.2%}.\n"
        f"- Required gates failed: {', '.join(failed_required) if failed_required else 'none'}.\n"
        "- Primary estimand: 20-session beta-hedged adjusted-open return, net of "
        "exact entry and exit traded-notional cost at 30 bps one way.\n"
        "- Stress cost: 50 bps one way. Net30/net50 include entry and exit "
        "trading costs only; financing, SPY short borrow, stock loan, and "
        "participation impact are not modeled. Inference: SPY-session-gap HAC(20); "
        "10,000 centered circular observation-block draws, seed 929, are a "
        "non-decision sensitivity. Multiplication by the disclosed trial floor "
        "124 is a screening hurdle and does not establish global FWER.\n\n"
        "No threshold, horizon, weight, or cost variant is authorized. An "
        "ADVANCE decision only authorizes Stage B exact-clock and permanent-ID "
        "engineering; it does not validate alpha, a model, or deployment. The "
        "cumulative chart is a diagnostic sum of overlapping event-date returns, "
        "not portfolio P&L.\n\n"
        "## Research interpretation\n\n"
        "Cohen, Malloy, and Pomorski motivate the opportunistic-versus-routine "
        "classification. Modern evidence also warns that apparent filing signals "
        "can be too illiquid or too short-lived to scale. This Stage A result is "
        "therefore a kill test, not a historical alpha claim.\n\n"
        "- https://www.nber.org/papers/w16454\n"
        "- https://doi.org/10.1016/j.frl.2024.106514\n"
        "- https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets\n",
        encoding="utf-8",
    )
    artifact_manifest_path = output / "artifact_manifest.json"
    _write_json(
        artifact_manifest_path,
        _artifact_manifest(
            output,
            paths,
            frozen_sources={
                **{
                    role: (Path(expected_paths[role]).resolve(), frozen[role])
                    for role in expected_paths
                },
                "freeze_receipt": (
                    Path(freeze_receipt_path).resolve(),
                    receipt_content,
                ),
            },
            root=root,
        ),
    )
    paths["artifact_manifest"] = artifact_manifest_path
    return paths
