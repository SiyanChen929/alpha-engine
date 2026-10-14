"""Fail-closed SEC-filing seasonal earnings-change drift diagnostic.

The module implements one frozen, literature-motivated public-data falsification.
It deliberately emits ``INVALID`` for historical alpha because the available
price panel is a current-survivor ticker panel without permanent-security or
delisting lineage.  The calculations are still useful for cheaply rejecting a
weak candidate before licensed data are purchased.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import norm

from alpha_factor_engine.contracts import validate_unique


RESEARCH_VERDICT = "INVALID"
MODEL_VERDICT = "NOT VALIDATED"
DEPLOYMENT_VERDICT = "NOT AUTHORIZED"


class SueDriftError(ValueError):
    """Raised when the frozen diagnostic cannot be executed faithfully."""


@dataclass(frozen=True)
class SueThresholds:
    lower: float
    upper: float
    development_events: int
    development_months: int


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_stable_json(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path)
    before = source.stat()
    content = source.read_bytes()
    after = source.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise SueDriftError(f"configuration changed while being read: {source}")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise SueDriftError(f"invalid JSON configuration: {source}") from exc
    if not isinstance(payload, dict):
        raise SueDriftError("configuration must be a JSON object")
    return payload, hashlib.sha256(content).hexdigest()


def _resolve_under_root(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise SueDriftError(f"configured path escapes repository root: {relative}") from exc
    return candidate


def verify_frozen_inputs(config: Mapping[str, Any], root: str | Path) -> dict[str, Any]:
    repository = Path(root).resolve()
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    for source_name in ("quarterly_release_features", "daily_prices"):
        declaration = config["sources"][source_name]
        for role, path_key, hash_key, bytes_key in (
            ("data", "path", "sha256", "bytes"),
            ("upstream_manifest", "upstream_manifest_path", "upstream_manifest_sha256", None),
        ):
            path = _resolve_under_root(repository, str(declaration[path_key]))
            exists = path.is_file()
            actual_bytes = path.stat().st_size if exists else None
            actual_hash = sha256_file(path) if exists else None
            expected_hash = str(declaration[hash_key])
            expected_bytes = declaration.get(bytes_key) if bytes_key else None
            passed = exists and actual_hash == expected_hash
            if expected_bytes is not None:
                passed = passed and actual_bytes == int(expected_bytes)
            if not passed:
                failures.append(f"{source_name}:{role}")
            records.append(
                {
                    "source": source_name,
                    "role": role,
                    "path": path.relative_to(repository).as_posix(),
                    "exists": exists,
                    "expected_bytes": expected_bytes,
                    "actual_bytes": actual_bytes,
                    "expected_sha256": expected_hash,
                    "actual_sha256": actual_hash,
                    "passed": bool(passed),
                }
            )
    return {
        "schema_version": "sec_filing_sue_input_verification_v1",
        "passed": not failures,
        "failures": failures,
        "records": records,
    }


def prepare_release_signals(
    releases: pd.DataFrame, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, SueThresholds, pd.DataFrame]:
    """Build the predeclared signal using only lagged issuer observations."""

    eligibility = config["release_eligibility"]
    required = set(eligibility["required_non_null"]) | {
        "form",
        "release_kind",
        "clock_source",
        "header_inventory_verified",
        "supersession_only",
        "net_income_same_tag_comparator",
    }
    missing = required.difference(releases.columns)
    if missing:
        raise SueDriftError(f"quarterly releases missing columns: {sorted(missing)}")

    frame = releases.copy()
    frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True, errors="coerce")
    frame["fiscal_end"] = pd.to_datetime(frame["fiscal_end"], errors="coerce")
    frame["ticker"] = frame["ticker"].astype("string").str.strip().str.upper()
    frame["accession"] = frame["accession"].astype("string").str.strip()
    mask = (
        frame["form"].isin(eligibility["forms"])
        & frame["release_kind"].isin(eligibility["release_kinds"])
        & frame["clock_source"].isin(eligibility["clock_sources"])
        & frame["header_inventory_verified"].eq(bool(eligibility["header_inventory_verified"]))
        & frame["supersession_only"].eq(bool(eligibility["supersession_only"]))
        & frame["net_income_same_tag_comparator"].eq(
            bool(eligibility["net_income_same_tag_comparator"])
        )
    )
    for column in eligibility["required_non_null"]:
        mask &= frame[column].notna()
    eligible = frame.loc[mask].copy()
    validate_unique(eligible, eligibility["deduplication_key"], "eligible SEC releases")
    eligible = eligible.sort_values(["ticker", "available_at", "accession"]).reset_index(drop=True)
    local_available = eligible["available_at"].dt.tz_convert("America/New_York")
    eligible["eod_clamped_clock"] = (
        local_available.dt.hour.eq(23)
        & local_available.dt.minute.eq(59)
        & local_available.dt.second.eq(59)
    )

    signal_spec = config["signal"]
    base = str(signal_spec["base_value"])
    eligible[base] = pd.to_numeric(eligible[base], errors="coerce")
    window = int(signal_spec["history_window_events"])
    minimum = int(signal_spec["minimum_history_events"])
    if int(signal_spec["history_shift_events"]) != 1:
        raise SueDriftError("only the frozen one-event history shift is supported")
    prior_std = eligible.groupby("ticker", sort=False)[base].transform(
        lambda values: values.shift(1).rolling(window, min_periods=minimum).std(ddof=1)
    )
    eligible["prior_signal_std"] = prior_std
    eligible["sue_like"] = eligible[base] / eligible["prior_signal_std"]
    finite = np.isfinite(eligible["sue_like"]) & eligible["prior_signal_std"].gt(0)
    eligible["signal_complete"] = finite
    eligible.loc[~finite, "sue_like"] = np.nan

    development = signal_spec["development_window"]
    dev_start = pd.Timestamp(development["information_at_gte"])
    dev_end = pd.Timestamp(development["information_at_lt"])
    dev_mask = (
        eligible["signal_complete"]
        & eligible["available_at"].ge(dev_start)
        & eligible["available_at"].lt(dev_end)
    )
    dev_values = eligible.loc[dev_mask, "sue_like"]
    if dev_values.empty:
        raise SueDriftError("development window has no complete signal observations")
    lower_q = float(signal_spec["tail_calibration"]["lower_quantile"])
    upper_q = float(signal_spec["tail_calibration"]["upper_quantile"])
    lower = float(dev_values.quantile(lower_q))
    upper = float(dev_values.quantile(upper_q))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise SueDriftError(f"invalid development thresholds: {lower}, {upper}")
    development_months = int(
        eligible.loc[dev_mask, "available_at"]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .dt.to_period("M")
        .nunique()
    )
    thresholds = SueThresholds(
        lower=lower,
        upper=upper,
        development_events=len(dev_values),
        development_months=development_months,
    )

    clock = config["universe_and_clock"]
    eval_start = pd.Timestamp(clock["evaluation_information_at_gte"])
    eval_end = pd.Timestamp(clock["evaluation_information_at_lt"])
    eval_mask = (
        eligible["signal_complete"]
        & eligible["available_at"].ge(eval_start)
        & eligible["available_at"].lt(eval_end)
    )
    evaluation = eligible.loc[eval_mask].copy()
    evaluation["direction"] = np.select(
        [evaluation["sue_like"].le(lower), evaluation["sue_like"].ge(upper)],
        [-1, 1],
        default=0,
    ).astype("int8")

    exclusion_rows: list[dict[str, Any]] = []
    incomplete = eligible.loc[
        eligible["available_at"].ge(eval_start)
        & eligible["available_at"].lt(eval_end)
        & ~eligible["signal_complete"]
    ]
    for row in incomplete.itertuples(index=False):
        exclusion_rows.append(
            {
                "accession": row.accession,
                "ticker": row.ticker,
                "available_at": row.available_at,
                "reason": "insufficient_or_zero_prior_signal_variability",
            }
        )
    middle = evaluation.loc[evaluation["direction"].eq(0)]
    for row in middle.itertuples(index=False):
        exclusion_rows.append(
            {
                "accession": row.accession,
                "ticker": row.ticker,
                "available_at": row.available_at,
                "reason": "middle_60_percent_not_traded",
            }
        )
    selected = evaluation.loc[evaluation["direction"].ne(0)].copy()
    exclusions = pd.DataFrame(
        exclusion_rows,
        columns=["accession", "ticker", "available_at", "reason"],
    )
    return selected, thresholds, exclusions


def _prepare_price_panel(prices: pd.DataFrame, *, adv_window: int) -> pd.DataFrame:
    required = {
        "ticker",
        "date",
        "timestamp",
        "close",
        "adj_open",
        "dollar_volume",
        "available_at",
    }
    missing = required.difference(prices.columns)
    if missing:
        raise SueDriftError(f"daily prices missing columns: {sorted(missing)}")
    frame = prices.loc[:, sorted(required)].copy()
    frame["ticker"] = frame["ticker"].astype("string").str.strip().str.upper()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True, errors="coerce")
    for column in ("close", "adj_open", "dollar_volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["ticker", "date", "timestamp"]].isna().any().any():
        raise SueDriftError("daily prices contain null keys or timestamps")
    validate_unique(frame, ["ticker", "date"], "daily price panel")
    frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)
    frame["prior_close"] = frame.groupby("ticker", sort=False)["close"].shift(1)
    frame["prior_adv_usd"] = frame.groupby("ticker", sort=False)["dollar_volume"].transform(
        lambda values: values.shift(1).rolling(adv_window, min_periods=adv_window).mean()
    )
    frame["prior_feature_available_at"] = frame.groupby("ticker", sort=False)[
        "available_at"
    ].shift(1)
    return frame


def attach_event_outcomes(
    selected: pd.DataFrame,
    prices: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach first-strictly-after entry and fixed-horizon open returns."""

    clock = config["universe_and_clock"]
    horizon = int(config["estimands"]["primary_horizon_sessions"])
    adv_window = int(clock["adv_window_sessions"])
    panel = _prepare_price_panel(prices, adv_window=adv_window)
    spy = panel.loc[panel["ticker"].eq(str(clock["benchmark"]).upper())].copy()
    if spy.empty:
        raise SueDriftError("benchmark SPY is absent from the price panel")
    spy = spy.sort_values("timestamp").reset_index(drop=True)
    if not spy["timestamp"].is_monotonic_increasing:
        raise SueDriftError("benchmark sessions are not monotonic")
    if spy["adj_open"].isna().any() or spy["adj_open"].le(0).any():
        raise SueDriftError("benchmark has missing or nonpositive adjusted opens")

    selected = selected.copy()
    selected["available_at"] = pd.to_datetime(selected["available_at"], utc=True)
    session_ns = spy["timestamp"].astype("int64").to_numpy()
    info_ns = selected["available_at"].astype("int64").to_numpy()
    entry_indices = np.searchsorted(session_ns, info_ns, side="right")
    exit_indices = entry_indices + horizon
    pre_indices = entry_indices - horizon
    valid_index = (
        (entry_indices >= 0)
        & (pre_indices >= 0)
        & (exit_indices < len(spy))
    )
    selected["entry_index"] = entry_indices
    selected["exit_index"] = exit_indices
    selected["pre_index"] = pre_indices
    selected["calendar_index_complete"] = valid_index

    def take(column: str, indices: np.ndarray) -> np.ndarray:
        out = np.full(len(indices), np.nan, dtype="object")
        valid = (indices >= 0) & (indices < len(spy))
        values = spy[column].to_numpy()
        out[valid] = values[indices[valid]]
        return out

    for prefix, indices in (
        ("entry", entry_indices),
        ("exit", exit_indices),
        ("pre", pre_indices),
    ):
        selected[f"{prefix}_at"] = pd.to_datetime(take("timestamp", indices), utc=True)
        selected[f"{prefix}_date"] = pd.to_datetime(take("date", indices), errors="coerce")
        selected[f"spy_{prefix}_adj_open"] = pd.to_numeric(
            pd.Series(take("adj_open", indices)), errors="coerce"
        ).to_numpy()

    lookup_columns = [
        "ticker",
        "date",
        "adj_open",
        "prior_close",
        "prior_adv_usd",
        "prior_feature_available_at",
    ]
    lookup = panel[lookup_columns]
    for prefix in ("entry", "exit", "pre"):
        side = lookup.rename(
            columns={
                "date": f"{prefix}_date",
                "adj_open": f"stock_{prefix}_adj_open",
                "prior_close": f"{prefix}_prior_close",
                "prior_adv_usd": f"{prefix}_prior_adv_usd",
                "prior_feature_available_at": f"{prefix}_prior_feature_available_at",
            }
        )
        selected = selected.merge(side, on=["ticker", f"{prefix}_date"], how="left", validate="many_to_one")

    cutoff = pd.Timestamp(clock["required_exit_open_lte"])
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")
    price_fields = [
        "stock_entry_adj_open",
        "stock_exit_adj_open",
        "stock_pre_adj_open",
        "spy_entry_adj_open",
        "spy_exit_adj_open",
        "spy_pre_adj_open",
    ]
    price_complete = selected[price_fields].notna().all(axis=1)
    price_positive = selected[price_fields].gt(0).all(axis=1)
    feature_clock_ok = pd.to_datetime(
        selected["entry_prior_feature_available_at"], utc=True, errors="coerce"
    ).lt(selected["entry_at"])
    selected["outcome_complete"] = (
        selected["calendar_index_complete"]
        & price_complete
        & price_positive
        & selected["exit_at"].le(cutoff)
    )
    selected["liquidity_complete"] = (
        selected["entry_prior_close"].notna()
        & selected["entry_prior_adv_usd"].notna()
        & feature_clock_ok
    )
    selected["liquid"] = (
        selected["liquidity_complete"]
        & selected["entry_prior_close"].ge(float(clock["minimum_prior_close_usd"]))
        & selected["entry_prior_adv_usd"].ge(float(clock["minimum_prior_adv_usd"]))
    )

    exclusions: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        reason = None
        if not row.calendar_index_complete:
            reason = "benchmark_calendar_boundary_missing"
        elif pd.isna(row.exit_at) or row.exit_at > cutoff:
            reason = "exit_after_frozen_evaluation_cutoff"
        elif not row.outcome_complete:
            reason = "stock_or_benchmark_open_missing_or_nonpositive"
        elif not row.liquidity_complete:
            reason = "prior_liquidity_history_or_clock_incomplete"
        elif not row.liquid:
            reason = "price_or_adv_below_frozen_floor"
        if reason:
            exclusions.append(
                {
                    "accession": row.accession,
                    "ticker": row.ticker,
                    "available_at": row.available_at,
                    "reason": reason,
                }
            )
    usable = selected.loc[selected["outcome_complete"] & selected["liquid"]].copy()

    conflict_keys: set[tuple[str, pd.Timestamp]] = set()
    for key, group in usable.groupby(["ticker", "entry_date"], sort=False):
        if group["direction"].nunique() > 1:
            conflict_keys.add(key)
            for row in group.itertuples(index=False):
                exclusions.append(
                    {
                        "accession": row.accession,
                        "ticker": row.ticker,
                        "available_at": row.available_at,
                        "reason": "conflicting_tail_directions_same_ticker_entry_session",
                    }
                )
    if conflict_keys:
        conflict_index = pd.MultiIndex.from_tuples(conflict_keys, names=["ticker", "entry_date"])
        keys = pd.MultiIndex.from_frame(usable[["ticker", "entry_date"]])
        usable = usable.loc[~keys.isin(conflict_index)].copy()

    collapsed_rows: list[pd.Series] = []
    for _, group in usable.groupby(["ticker", "entry_date"], sort=True):
        row = group.sort_values("accession").iloc[0].copy()
        row["accessions"] = "|".join(sorted(group["accession"].astype(str).unique()))
        row["collapsed_accession_count"] = int(group["accession"].nunique())
        row["sue_like"] = float(group["sue_like"].mean())
        collapsed_rows.append(row)
    collapsed = pd.DataFrame(collapsed_rows).reset_index(drop=True)
    if collapsed.empty:
        raise SueDriftError("no eligible tail events remain after price and liquidity gates")
    validate_unique(collapsed, ["ticker", "entry_date"], "collapsed release events")

    collapsed["stock_return_20"] = (
        collapsed["stock_exit_adj_open"] / collapsed["stock_entry_adj_open"] - 1.0
    )
    collapsed["spy_return_20"] = (
        collapsed["spy_exit_adj_open"] / collapsed["spy_entry_adj_open"] - 1.0
    )
    collapsed["stock_abnormal_return_20"] = (
        collapsed["stock_return_20"] - collapsed["spy_return_20"]
    )
    collapsed["pair_gross_return_20"] = (
        0.5 * collapsed["direction"] * collapsed["stock_abnormal_return_20"]
    )
    collapsed["pair_net_30bps_return_20"] = (
        collapsed["pair_gross_return_20"]
        - float(config["estimands"]["primary_round_trip_cost_per_gross_dollar"])
    )
    collapsed["pair_net_50bps_return_20"] = (
        collapsed["pair_gross_return_20"]
        - float(config["estimands"]["stress_round_trip_cost_per_gross_dollar"])
    )
    pre_stock = collapsed["stock_entry_adj_open"] / collapsed["stock_pre_adj_open"] - 1.0
    pre_spy = collapsed["spy_entry_adj_open"] / collapsed["spy_pre_adj_open"] - 1.0
    collapsed["pair_pre_event_gross_return_20"] = (
        0.5 * collapsed["direction"] * (pre_stock - pre_spy)
    )
    collapsed["absolute_stock_return_over_100pct"] = collapsed["stock_return_20"].abs().gt(1.0)
    collapsed["research_status"] = RESEARCH_VERDICT
    collapsed["scope_reasons"] = (
        "current_survivor_ticker_panel|no_permanent_security_id|"
        "no_complete_delisting_payoff|retrospectively_adjusted_open"
    )
    exclusion_frame = pd.DataFrame(
        exclusions,
        columns=["accession", "ticker", "available_at", "reason"],
    )
    return collapsed, exclusion_frame


def aggregate_entry_sessions(event_returns: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "pair_gross_return_20",
        "pair_net_30bps_return_20",
        "pair_net_50bps_return_20",
        "pair_pre_event_gross_return_20",
    ]
    result = (
        event_returns.groupby("entry_date", sort=True)
        .agg(
            events=("accession", "size"),
            tickers=("ticker", "nunique"),
            positive_tail_events=("direction", lambda x: int((x > 0).sum())),
            negative_tail_events=("direction", lambda x: int((x < 0).sum())),
            **{column: (column, "mean") for column in columns},
        )
        .reset_index()
    )
    validate_unique(result, ["entry_date"], "entry-session returns")
    return result


def hac_mean_test(values: Iterable[float], *, lags: int) -> dict[str, float | int]:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return {"n": n, "mean": math.nan, "standard_error": math.nan, "t": math.nan, "one_sided_p": math.nan}
    mean = float(x.mean())
    residual = x - mean
    gamma0 = float(np.dot(residual, residual) / n)
    long_run_variance = gamma0
    max_lag = min(int(lags), n - 1)
    for lag in range(1, max_lag + 1):
        covariance = float(np.dot(residual[lag:], residual[:-lag]) / n)
        weight = 1.0 - lag / (max_lag + 1.0)
        long_run_variance += 2.0 * weight * covariance
    long_run_variance = max(long_run_variance, 0.0)
    standard_error = math.sqrt(long_run_variance / n)
    t_value = mean / standard_error if standard_error > 0 else math.nan
    p_value = float(norm.sf(t_value)) if np.isfinite(t_value) else math.nan
    return {
        "n": n,
        "mean": mean,
        "standard_error": standard_error,
        "t": float(t_value),
        "one_sided_p": p_value,
    }


def moving_block_null_p(
    values: Iterable[float], *, draws: int, block_length: int, seed: int
) -> dict[str, Any]:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        raise SueDriftError("moving-block inference requires at least three values")
    observed = float(x.mean())
    centered = x - observed
    n = len(centered)
    length = max(1, min(int(block_length), n))
    starts = np.arange(0, n - length + 1)
    if starts.size == 0:
        starts = np.array([0])
    blocks_needed = int(math.ceil(n / length))
    rng = np.random.default_rng(int(seed))
    null_means = np.empty(int(draws), dtype=float)
    for draw in range(int(draws)):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([centered[start : start + length] for start in chosen])[:n]
        null_means[draw] = float(sample.mean())
    exceedances = int(np.count_nonzero(null_means >= observed))
    return {
        "observed_mean": observed,
        "draws": int(draws),
        "block_length": length,
        "seed": int(seed),
        "exceedances": exceedances,
        "one_sided_empirical_p": float((exceedances + 1) / (int(draws) + 1)),
        "minimum_resolvable_p": float(1 / (int(draws) + 1)),
        "null_mean": float(null_means.mean()),
        "null_std": float(null_means.std(ddof=1)),
    }


def summarize_results(
    events: pd.DataFrame,
    sessions: pd.DataFrame,
    thresholds: SueThresholds,
    config: Mapping[str, Any],
    *,
    selected_before_price_gates: int,
    price_exclusions: pd.DataFrame,
) -> dict[str, Any]:
    inference = config["inference"]
    floor = int(config["trial_ledger"]["resulting_disclosed_global_floor"])
    primary = hac_mean_test(
        sessions["pair_net_30bps_return_20"],
        lags=int(inference["hac_lags_over_ordered_entry_sessions"]),
    )
    gross = hac_mean_test(
        sessions["pair_gross_return_20"],
        lags=int(inference["hac_lags_over_ordered_entry_sessions"]),
    )
    stress = hac_mean_test(
        sessions["pair_net_50bps_return_20"],
        lags=int(inference["hac_lags_over_ordered_entry_sessions"]),
    )
    placebo = hac_mean_test(
        sessions["pair_pre_event_gross_return_20"],
        lags=int(inference["hac_lags_over_ordered_entry_sessions"]),
    )
    block = moving_block_null_p(
        sessions["pair_net_30bps_return_20"],
        draws=int(inference["moving_block_bootstrap_draws"]),
        block_length=int(inference["moving_block_length_entry_sessions"]),
        seed=int(inference["bootstrap_seed"]),
    )
    primary_adjusted = min(1.0, float(primary["one_sided_p"]) * floor)
    block_adjusted = min(1.0, float(block["one_sided_empirical_p"]) * floor)
    positive = events.loc[events["direction"].gt(0), "pair_gross_return_20"]
    negative = events.loc[events["direction"].lt(0), "pair_gross_return_20"]
    by_ticker = events.groupby("ticker")["pair_net_30bps_return_20"].sum()
    absolute_total = float(by_ticker.abs().sum())
    top_ten_share = (
        float(by_ticker.abs().nlargest(10).sum() / absolute_total)
        if absolute_total > 0
        else math.nan
    )
    events_by_year = (
        events.assign(year=pd.to_datetime(events["entry_date"]).dt.year)
        .groupby("year")
        .agg(
            events=("accession", "size"),
            entry_sessions=("entry_date", "nunique"),
            gross_mean=("pair_gross_return_20", "mean"),
            net_30bps_mean=("pair_net_30bps_return_20", "mean"),
            placebo_mean=("pair_pre_event_gross_return_20", "mean"),
        )
        .reset_index()
        .to_dict("records")
    )
    quarters = int(pd.PeriodIndex(pd.to_datetime(events["entry_date"]), freq="Q").nunique())
    reason_counts = (
        price_exclusions["reason"].value_counts().sort_index().astype(int).to_dict()
        if not price_exclusions.empty
        else {}
    )
    return {
        "schema_version": "sec_filing_sue_drift_summary_v1",
        "verdicts": {
            "historical_alpha": RESEARCH_VERDICT,
            "model_alpha": MODEL_VERDICT,
            "deployment": DEPLOYMENT_VERDICT,
        },
        "thresholds": {
            "development_lower_q20": thresholds.lower,
            "development_upper_q80": thresholds.upper,
            "development_events": thresholds.development_events,
            "development_signal_complete_months": thresholds.development_months,
        },
        "sample": {
            "selected_tail_events_before_price_gates": int(selected_before_price_gates),
            "usable_collapsed_events": int(len(events)),
            "tickers": int(events["ticker"].nunique()),
            "entry_sessions": int(events["entry_date"].nunique()),
            "calendar_quarters": quarters,
            "positive_tail_events": int(events["direction"].gt(0).sum()),
            "negative_tail_events": int(events["direction"].lt(0).sum()),
            "price_or_liquidity_exclusions": int(len(price_exclusions)),
            "price_exclusion_reason_counts": reason_counts,
            "eod_clamped_clock_events": int(events["eod_clamped_clock"].sum()),
            "non_eod_clamped_acceptance_proxy_coverage": float(
                1.0 - events["eod_clamped_clock"].mean()
            ),
            "absolute_stock_returns_over_100pct": int(events["absolute_stock_return_over_100pct"].sum()),
            "top_10_ticker_absolute_contribution_share": top_ten_share,
        },
        "primary_net_30bps_entry_session_test": {
            **primary,
            "global_floor": floor,
            "bonferroni_adjusted_p": primary_adjusted,
            "moving_block": block,
            "moving_block_bonferroni_adjusted_p": block_adjusted,
        },
        "gross_entry_session_test": gross,
        "stress_net_50bps_entry_session_test": stress,
        "pre_event_placebo_entry_session_test": placebo,
        "tails": {
            "positive_tail_event_mean_gross_pair_return": float(positive.mean()),
            "negative_tail_signed_event_mean_gross_pair_return": float(negative.mean()),
        },
        "by_year": events_by_year,
        "interpretation": (
            "This is an opened, current-survivor, ticker-keyed SEC filing diagnostic. "
            "It is neither a causal event study nor a validated portfolio backtest."
        ),
    }


def build_validation_gates(
    summary: Mapping[str, Any],
    input_verification: Mapping[str, Any],
    events: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    sample = summary["sample"]
    primary = summary["primary_net_30bps_entry_session_test"]
    tails = summary["tails"]
    placebo = summary["pre_event_placebo_entry_session_test"]
    stress = summary["stress_net_50bps_entry_session_test"]
    required_p = 0.05 / int(config["trial_ledger"]["resulting_disclosed_global_floor"])
    family_critical = float(norm.isf(required_p))
    selected = int(sample["selected_tail_events_before_price_gates"])
    reason_counts = sample["price_exclusion_reason_counts"]
    missing_reasons = {
        "benchmark_calendar_boundary_missing",
        "stock_or_benchmark_open_missing_or_nonpositive",
        "prior_liquidity_history_or_clock_incomplete",
        "conflicting_tail_directions_same_ticker_entry_session",
    }
    missing_count = sum(int(reason_counts.get(reason, 0)) for reason in missing_reasons)
    price_completeness = 1.0 - missing_count / selected if selected else 0.0
    non_eod_coverage = float(sample["non_eod_clamped_acceptance_proxy_coverage"])
    rows = [
        ("input_hashes_match", bool(input_verification["passed"]), True, "data", str(input_verification["failures"])),
        ("conservative_clock_proxy_recorded", bool(events["clock_source"].eq("submissions_acceptance_plus_3m").all()), True, "data", "acceptance-derived proxy present"),
        ("non_eod_clamped_acceptance_proxy_coverage", math.isclose(non_eod_coverage, 1.0), True, "lineage", f"{non_eod_coverage:.6f}"),
        ("verified_first_public_retrieval_clock", False, True, "lineage", "historical receive logs unavailable"),
        ("event_price_data_completeness", math.isclose(price_completeness, 1.0), True, "data", f"missing={missing_count}/{selected}; coverage={price_completeness:.6f}"),
        ("permanent_security_identifier_coverage", False, True, "lineage", "0%; ticker only"),
        ("complete_delisting_payoff_coverage", False, True, "lineage", "not available in public proxy"),
        ("minimum_entry_sessions", int(sample["entry_sessions"]) >= int(config["inference"]["minimum_evaluation_entry_sessions"]), True, "sample", str(sample["entry_sessions"])),
        ("minimum_calendar_quarters", int(sample["calendar_quarters"]) >= int(config["inference"]["minimum_evaluation_calendar_quarters"]), True, "sample", str(sample["calendar_quarters"])),
        ("minimum_signal_complete_development_months", int(summary["thresholds"]["development_signal_complete_months"]) >= int(config["inference"]["minimum_signal_complete_development_months_for_this_kill_test"]), True, "sample", str(summary["thresholds"]["development_signal_complete_months"])),
        ("primary_net_mean_positive", float(primary["mean"]) > 0, True, "efficacy", f"{primary['mean']:.10f}"),
        ("primary_hac_t_family_gate", float(primary["t"]) >= max(3.0, family_critical), True, "efficacy", f"t={primary['t']:.6f}; critical={family_critical:.6f}"),
        ("primary_bonferroni_adjusted_p", float(primary["bonferroni_adjusted_p"]) <= 0.05, True, "efficacy", f"{primary['bonferroni_adjusted_p']:.6f}"),
        ("moving_block_global_p", float(primary["moving_block_bonferroni_adjusted_p"]) <= 0.05, True, "efficacy", f"{primary['moving_block_bonferroni_adjusted_p']:.6f}"),
        ("both_tail_directions_correct", float(tails["positive_tail_event_mean_gross_pair_return"]) > 0 and float(tails["negative_tail_signed_event_mean_gross_pair_return"]) > 0, True, "efficacy", json.dumps(tails, sort_keys=True)),
        ("pre_event_placebo_not_positive", float(placebo["mean"]) <= 0, True, "negative_control", f"{placebo['mean']:.10f}"),
        ("positive_after_50bps_stress", float(stress["mean"]) > 0, True, "execution", f"{stress['mean']:.10f}"),
        ("no_extreme_20_session_stock_returns", int(sample["absolute_stock_returns_over_100pct"]) == 0, True, "quality", str(sample["absolute_stock_returns_over_100pct"])),
    ]
    return pd.DataFrame(
        rows,
        columns=["gate", "passed", "required", "domain", "evidence"],
    )


def manual_reconstruction(events: pd.DataFrame) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    for direction in (1, -1):
        subset = events.loc[events["direction"].eq(direction)].sort_values(
            ["entry_date", "ticker"]
        )
        if subset.empty:
            continue
        row = subset.iloc[0]
        stock_return = float(row["stock_exit_adj_open"] / row["stock_entry_adj_open"] - 1)
        spy_return = float(row["spy_exit_adj_open"] / row["spy_entry_adj_open"] - 1)
        gross = 0.5 * direction * (stock_return - spy_return)
        net = gross - 0.006
        samples.append(
            {
                "accession": row["accession"],
                "ticker": row["ticker"],
                "direction": direction,
                "entry_at": row["entry_at"],
                "exit_at": row["exit_at"],
                "stock_entry_adj_open": float(row["stock_entry_adj_open"]),
                "stock_exit_adj_open": float(row["stock_exit_adj_open"]),
                "spy_entry_adj_open": float(row["spy_entry_adj_open"]),
                "spy_exit_adj_open": float(row["spy_exit_adj_open"]),
                "independent_stock_return": stock_return,
                "independent_spy_return": spy_return,
                "independent_pair_gross_return": gross,
                "recorded_pair_gross_return": float(row["pair_gross_return_20"]),
                "independent_pair_net_30bps_return": net,
                "recorded_pair_net_30bps_return": float(row["pair_net_30bps_return_20"]),
                "absolute_gross_error": abs(gross - float(row["pair_gross_return_20"])),
                "absolute_net_error": abs(net - float(row["pair_net_30bps_return_20"])),
            }
        )
    maximum_error = max(
        [max(item["absolute_gross_error"], item["absolute_net_error"]) for item in samples],
        default=math.nan,
    )
    return {
        "schema_version": "sec_filing_sue_manual_reconstruction_v1",
        "sample_count": len(samples),
        "tolerance": 1e-12,
        "maximum_absolute_error": maximum_error,
        "passed": bool(samples) and maximum_error <= 1e-12,
        "samples": samples,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value)!r}")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def run_diagnostic(
    *,
    config_path: str | Path,
    repository_root: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    root = Path(repository_root).resolve()
    config_file = Path(config_path).resolve()
    config, config_sha = load_stable_json(config_file)
    if config.get("schema_version") != "sec_filing_sue_drift_preregistration_v1":
        raise SueDriftError("unsupported preregistration schema")
    frozen_at = pd.Timestamp(config["frozen_at"])
    started_at = datetime.now(UTC)
    if frozen_at.tzinfo is None or frozen_at >= pd.Timestamp(started_at):
        raise SueDriftError("protocol frozen_at must be timezone-aware and precede the run")
    verification = verify_frozen_inputs(config, root)
    if not verification["passed"]:
        raise SueDriftError(f"frozen input verification failed: {verification['failures']}")

    destination = (
        Path(output_dir).resolve()
        if output_dir is not None
        else _resolve_under_root(root, config["outputs"]["directory"])
    )
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise SueDriftError(f"output directory must be empty: {destination}")

    release_path = _resolve_under_root(
        root, config["sources"]["quarterly_release_features"]["path"]
    )
    price_path = _resolve_under_root(root, config["sources"]["daily_prices"]["path"])
    releases = pd.read_parquet(release_path)
    selected, thresholds, signal_exclusions = prepare_release_signals(releases, config)
    selected_before_price = len(selected)
    selected_tickers = sorted(set(selected["ticker"].dropna().astype(str)) | {"SPY"})
    prices = pd.read_parquet(
        price_path,
        columns=[
            "ticker",
            "date",
            "timestamp",
            "close",
            "adj_open",
            "dollar_volume",
            "available_at",
        ],
        filters=[("ticker", "in", selected_tickers)],
    )
    events, price_exclusions = attach_event_outcomes(selected, prices, config)
    exclusions = pd.concat([signal_exclusions, price_exclusions], ignore_index=True)
    sessions = aggregate_entry_sessions(events)
    summary = summarize_results(
        events,
        sessions,
        thresholds,
        config,
        selected_before_price_gates=selected_before_price,
        price_exclusions=price_exclusions,
    )
    gates = build_validation_gates(summary, verification, events, config)
    reconstruction = manual_reconstruction(events)
    if not reconstruction["passed"]:
        raise SueDriftError("manual reconstruction failed")

    paths = {
        "input_verification": destination / "input_verification.json",
        "release_events": destination / "release_events.parquet",
        "event_exclusions": destination / "event_exclusions.csv",
        "event_returns": destination / "event_returns.csv",
        "entry_session_returns": destination / "entry_session_returns.csv",
        "summary_metrics": destination / "summary_metrics.json",
        "trial_ledger": destination / "trial_ledger.json",
        "validation_gates": destination / "validation_gates.csv",
        "manual_reconstruction": destination / "manual_reconstruction.json",
        "run_manifest": destination / "run_manifest.json",
        "report": destination / "report-source.md",
        "artifact_manifest": destination / "artifact_manifest.json",
    }
    write_json(paths["input_verification"], verification)
    events.to_parquet(paths["release_events"], index=False)
    exclusions.to_csv(paths["event_exclusions"], index=False)
    events.to_csv(paths["event_returns"], index=False)
    sessions.to_csv(paths["entry_session_returns"], index=False)
    write_json(paths["summary_metrics"], summary)
    trial_ledger = {
        **config["trial_ledger"],
        "classification": "opened_historical_diagnostic_only",
        "protocol_sha256": config_sha,
        "outcomes_first_computed_at": started_at.isoformat(),
        "all_results_retained": True,
    }
    write_json(paths["trial_ledger"], trial_ledger)
    gates.to_csv(paths["validation_gates"], index=False)
    write_json(paths["manual_reconstruction"], reconstruction)
    source_path = Path(__file__).resolve()
    run_manifest = {
        "schema_version": "sec_filing_sue_drift_run_manifest_v1",
        "run_started_at": started_at.isoformat(),
        "run_finished_at": datetime.now(UTC).isoformat(),
        "protocol_path": config_file.relative_to(root).as_posix(),
        "protocol_sha256": config_sha,
        "protocol_frozen_at": frozen_at.isoformat(),
        "protocol_precedes_outcome_computation": frozen_at < pd.Timestamp(started_at),
        "historical_outcome_environment_previously_opened": True,
        "source_code_path": source_path.relative_to(root).as_posix(),
        "source_code_sha256": sha256_file(source_path),
        "input_verification_passed": True,
        "resulting_disclosed_global_trial_floor": config["trial_ledger"][
            "resulting_disclosed_global_floor"
        ],
        "verdicts": summary["verdicts"],
    }
    write_json(paths["run_manifest"], run_manifest)

    primary = summary["primary_net_30bps_entry_session_test"]
    gross = summary["gross_entry_session_test"]
    stress = summary["stress_net_50bps_entry_session_test"]
    placebo = summary["pre_event_placebo_entry_session_test"]
    gate_failures = gates.loc[gates["required"] & ~gates["passed"], "gate"].tolist()
    candidate_decision = (
        "SURVIVES_STAGE_A_BUT_NOT_VALIDATED"
        if not any(
            gate in gate_failures
            for gate in (
                "primary_net_mean_positive",
                "primary_hac_t_family_gate",
                "primary_bonferroni_adjusted_p",
                "moving_block_global_p",
                "both_tail_directions_correct",
                "pre_event_placebo_not_positive",
                "positive_after_50bps_stress",
            )
        )
        else "RETIRED_AFTER_FROZEN_STAGE_A_FALSIFICATION"
    )
    report = f"""# SEC filing seasonal earnings-change drift v1

## Direct result

Candidate decision: **{candidate_decision}**. Historical alpha verdict: **{RESEARCH_VERDICT}**. Model verdict: **{MODEL_VERDICT}**. Deployment: **{DEPLOYMENT_VERDICT}**.

This was the only outcome-tested specification. A pre-outcome red-team amendment narrowed the clock claim before calculation. It uses an SEC 10-Q acceptance-derived conservative availability proxy, a signal standardized only by the prior eight eligible issuer observations, fixed 2018–2022 feature-only quintile thresholds, first-open-after-proxy execution, and a 2023–2025 opened evaluation whose exit must remain in 2025.

## Sample and result

- Development signal observations: {thresholds.development_events:,} across {thresholds.development_months} signal-complete months; q20={thresholds.lower:.6f}, q80={thresholds.upper:.6f}.
- Selected tail events before price gates: {summary['sample']['selected_tail_events_before_price_gates']:,}; usable collapsed events: {summary['sample']['usable_collapsed_events']:,} across {summary['sample']['entry_sessions']:,} entry sessions and {summary['sample']['tickers']:,} tickers.
- Gross 20-session pair return per gross dollar: {gross['mean']:.4%}; HAC t={gross['t']:.3f}.
- Primary net after 30 bps per one-way traded dollar: {primary['mean']:.4%}; HAC t={primary['t']:.3f}; raw one-sided p={primary['one_sided_p']:.6f}; global-floor-119 adjusted p={primary['bonferroni_adjusted_p']:.6f}.
- Moving-block empirical p={primary['moving_block']['one_sided_empirical_p']:.6f}; global adjusted={primary['moving_block_bonferroni_adjusted_p']:.6f}; draws={primary['moving_block']['draws']:,}.
- Net after 50 bps per one-way traded dollar: {stress['mean']:.4%}; HAC t={stress['t']:.3f}.
- Signed pre-event placebo: {placebo['mean']:.4%}; HAC t={placebo['t']:.3f}.
- Positive-tail / negative-tail signed gross means: {summary['tails']['positive_tail_event_mean_gross_pair_return']:.4%} / {summary['tails']['negative_tail_signed_event_mean_gross_pair_return']:.4%}.

## Gate interpretation

Required failures: `{json.dumps(gate_failures)}`.

The event calculation is auditable but cannot support historical P&L. The stock panel was selected from current registrants, is keyed by ticker, uses retrospectively adjusted prices, and lacks complete corporate-action/delisting payoff lineage. Historical receive logs are absent, and {summary['sample']['eod_clamped_clock_events']:,} usable events use a filing-day-EOD clamp. A 10-Q filing can also follow an earlier earnings press release, so this estimates post-EDGAR-filing association rather than pure announcement drift. No horizon, threshold, universe, or model repair is permitted after observing this result.

## Sources

- [SEC EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)
- [Bernard and Thomas, Post-Earnings-Announcement Drift](https://doi.org/10.2307/2491062)

## Scope

The protocol was frozen before this candidate's outcome computation, but related historical outcomes were already accessible. Strict confirmation requires a newly frozen permanent-security cohort whose information arrives on or after 2026-10-01 ET.
"""
    paths["report"].write_text(report, encoding="utf-8")

    required_names = set(config["outputs"]["required"])
    actual_names = {path.name for key, path in paths.items() if key != "artifact_manifest"}
    missing_outputs = required_names.difference(actual_names | {"artifact_manifest.json"})
    if missing_outputs:
        raise SueDriftError(f"required outputs not mapped: {sorted(missing_outputs)}")
    records: list[dict[str, Any]] = []
    for path in sorted(actual_names):
        artifact = destination / path
        records.append(
            {
                "path": artifact.relative_to(root).as_posix(),
                "role": "output",
                "bytes": artifact.stat().st_size,
                "sha256": sha256_file(artifact),
            }
        )
    for role, artifact in (
        ("protocol", config_file),
        ("code", source_path),
    ):
        records.append(
            {
                "path": artifact.relative_to(root).as_posix(),
                "role": role,
                "bytes": artifact.stat().st_size,
                "sha256": sha256_file(artifact),
            }
        )
    manifest = {
        "schema_version": "sec_filing_sue_drift_artifact_manifest_v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "record_count": len(records),
        "records": sorted(records, key=lambda row: (row["role"], row["path"])),
        "all_records_exist": True,
        "verdicts": summary["verdicts"],
        "candidate_decision": candidate_decision,
    }
    write_json(paths["artifact_manifest"], manifest)
    return paths
