"""Fail-closed cross-sectional feature-family diagnostics.

The module deliberately separates signal construction from labels.  Oriented
scores and composite scores are built from signal-time columns only.  Future
returns enter only after those scores have been frozen for a month.  The
result is an opened diagnostic, not a trading strategy or portfolio backtest.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from alpha_factor_engine.research.metrics import newey_west_tstat


FUNDAMENTAL_FEATURES: tuple[str, ...] = (
    "book_to_market",
    "earnings_yield",
    "roe",
    "roa",
    "gross_profitability",
    "debt_to_equity",
    "asset_growth",
    "sales_growth",
    "accruals",
    "free_cash_flow_yield",
)


@dataclass(frozen=True, slots=True)
class FeatureDefinition:
    """One oriented individual signal or an equal-rank composite."""

    feature_id: str
    family: str
    kind: str
    components: tuple[tuple[str, int], ...]
    minimum_components: int
    direction_status: str

    def __post_init__(self) -> None:
        if not self.feature_id or not self.family:
            raise ValueError("feature_id and family must be non-empty")
        if self.kind not in {"individual", "composite"}:
            raise ValueError("feature kind must be individual or composite")
        if not self.components:
            raise ValueError("feature definition needs at least one component")
        if len({name for name, _ in self.components}) != len(self.components):
            raise ValueError("feature components must be unique")
        if any(direction not in {-1, 1} for _, direction in self.components):
            raise ValueError("feature directions must be -1 or 1")
        if not 1 <= self.minimum_components <= len(self.components):
            raise ValueError("minimum_components is outside component count")
        if self.kind == "individual":
            if len(self.components) != 1 or self.minimum_components != 1:
                raise ValueError("individual features must contain exactly one component")
            if self.components[0][0] != self.feature_id:
                raise ValueError("individual feature id must equal its component")


def definitions_from_config(config: Mapping[str, Any]) -> tuple[FeatureDefinition, ...]:
    """Parse and validate the frozen YAML feature specification."""

    try:
        feature_config = config["features"]
        direction_status = str(config["direction_policy"]["status"])
    except KeyError as error:
        raise ValueError(f"feature diagnostic config missing {error.args[0]!r}") from error
    definitions: list[FeatureDefinition] = []
    for feature_id, values in feature_config.get("individual", {}).items():
        definitions.append(
            FeatureDefinition(
                feature_id=str(feature_id),
                family=str(values["family"]),
                kind="individual",
                components=((str(feature_id), int(values["direction"])),),
                minimum_components=1,
                direction_status=direction_status,
            )
        )
    for feature_id, values in feature_config.get("composites", {}).items():
        components = tuple(
            (str(name), int(direction))
            for name, direction in values["components"].items()
        )
        definitions.append(
            FeatureDefinition(
                feature_id=str(feature_id),
                family=str(values["family"]),
                kind="composite",
                components=components,
                minimum_components=int(values["minimum_components"]),
                direction_status=direction_status,
            )
        )
    if not definitions:
        raise ValueError("feature diagnostic config contains no feature definitions")
    identifiers = [item.feature_id for item in definitions]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("feature identifiers must be unique")
    return tuple(definitions)


def _as_utc(series: pd.Series, *, column: str) -> pd.Series:
    try:
        output = pd.to_datetime(series, utc=True, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{column} must contain valid timestamps") from error
    if output.isna().any():
        raise ValueError(f"{column} contains missing timestamps")
    return output


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def centered_rank(series: pd.Series) -> pd.Series:
    """Cross-sectional average rank in [-1, 1], preserving missing values."""

    numeric = _numeric(series)
    count = int(numeric.notna().sum())
    if count < 2:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return numeric.rank(method="average", pct=True) * 2.0 - 1.0


def validate_source_panel(
    panel: pd.DataFrame,
    definitions: Sequence[FeatureDefinition],
    *,
    expected_rows: int | None = None,
    expected_months: int | None = None,
    expected_securities: int | None = None,
    security_column: str = "ticker",
    date_column: str = "signal_date",
    target_column: str = "target_return",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate the exact diagnostic panel and return normalized timestamps.

    The routine rejects partial label months, duplicate decision/security keys,
    future availability, impossible execution order, non-finite populated
    features, and any silent row-level eligibility filtering.
    """

    components = {name for item in definitions for name, _ in item.components}
    required = {
        security_column,
        date_column,
        target_column,
        "eligible",
        "entry_at",
        "exit_at",
        "entry_price",
        "exit_price",
        "price_available_at",
        "latest_input_available_at",
        "fundamental_available_at",
        "fundamental_fiscal_end",
        "fundamental_age_days",
        "debt_definition",
        *components,
    }
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"feature diagnostic source missing columns: {missing}")
    frame = panel.copy()
    for column in (
        date_column,
        "entry_at",
        "exit_at",
        "price_available_at",
        "latest_input_available_at",
    ):
        frame[column] = _as_utc(frame[column], column=column)
    frame["fundamental_available_at"] = pd.to_datetime(
        frame["fundamental_available_at"], utc=True, errors="coerce"
    )
    frame["fundamental_fiscal_end"] = pd.to_datetime(
        frame["fundamental_fiscal_end"], utc=True, errors="coerce"
    )
    if frame[security_column].isna().any():
        raise ValueError("source contains missing security identifiers")
    frame[security_column] = frame[security_column].astype(str)
    if frame.duplicated([date_column, security_column]).any():
        raise ValueError("source contains duplicate decision/security rows")
    if not frame["eligible"].fillna(False).astype(bool).all():
        raise ValueError("evaluation panel contains rows outside frozen eligibility")
    if not frame[date_column].lt(frame["entry_at"]).all():
        raise ValueError("decision must precede entry")
    if not frame["entry_at"].lt(frame["exit_at"]).all():
        raise ValueError("entry must precede exit")
    if frame["price_available_at"].gt(frame[date_column]).any():
        raise ValueError("price feature availability exceeds decision time")
    if frame["latest_input_available_at"].gt(frame[date_column]).any():
        raise ValueError("residual-momentum availability exceeds decision time")

    target = _numeric(frame[target_column])
    if target.isna().any():
        raise ValueError("evaluation panel has incomplete or non-finite labels")
    if target.le(-1.0).any():
        raise ValueError("target return at or below -100% is impossible for this price path")
    entry_price = _numeric(frame["entry_price"])
    exit_price = _numeric(frame["exit_price"])
    if entry_price.isna().any() or exit_price.isna().any():
        raise ValueError("execution prices are incomplete or non-finite")
    if entry_price.le(0).any() or exit_price.le(0).any():
        raise ValueError("execution prices must be positive")
    reconstructed = exit_price / entry_price - 1.0
    if not np.allclose(target, reconstructed, atol=1e-12, rtol=1e-12):
        raise ValueError("target return does not reconcile to entry/exit prices")
    frame[target_column] = target

    infinite_feature_rows = 0
    for column in sorted(components):
        raw = pd.to_numeric(frame[column], errors="coerce")
        infinite_feature_rows += int(np.isinf(raw.to_numpy(float, na_value=np.nan)).sum())
    if infinite_feature_rows:
        raise ValueError("source contains infinite feature values")

    fundamental_clock_columns = set(FUNDAMENTAL_FEATURES) | {"log_market_cap"}
    present_fundamental_columns = sorted(fundamental_clock_columns.intersection(components))
    populated_fundamental = frame[present_fundamental_columns].notna().any(axis=1)
    invalid_fundamental_clock = populated_fundamental & (
        frame["fundamental_available_at"].isna()
        | frame["fundamental_available_at"].gt(frame[date_column])
    )
    if invalid_fundamental_clock.any():
        raise ValueError("populated fundamentals fail their availability clock")
    fundamental_age = _numeric(frame["fundamental_age_days"])
    invalid_age = populated_fundamental & (
        fundamental_age.isna()
        | fundamental_age.lt(0.0)
        | frame["fundamental_fiscal_end"].isna()
        | frame["fundamental_fiscal_end"].gt(frame["fundamental_available_at"])
    )
    if invalid_age.any():
        raise ValueError("populated fundamentals fail fiscal-end or age integrity")
    reconstructed_age = (frame[date_column] - frame["fundamental_available_at"]).dt.days
    if not reconstructed_age.loc[populated_fundamental].eq(
        fundamental_age.loc[populated_fundamental]
    ).all():
        raise ValueError("fundamental age does not reconcile to its availability clock")

    monthly_label_counts = frame.groupby(date_column, sort=True)[target_column].apply(
        lambda values: int(values.notna().sum())
    )
    monthly_rows = frame.groupby(date_column, sort=True).size()
    if not monthly_label_counts.eq(monthly_rows).all():
        raise ValueError("partial label month would condition on future outcome availability")
    observed = {
        "rows": int(len(frame)),
        "months": int(frame[date_column].nunique()),
        "securities": int(frame[security_column].nunique()),
    }
    expected = {
        "rows": expected_rows,
        "months": expected_months,
        "securities": expected_securities,
    }
    for name, expected_value in expected.items():
        if expected_value is not None and observed[name] != int(expected_value):
            raise ValueError(
                f"source {name} drifted: observed={observed[name]}, expected={expected_value}"
            )

    quality = {
        **observed,
        "first_signal_date": frame[date_column].min(),
        "last_signal_date": frame[date_column].max(),
        "monthly_names_minimum": int(monthly_rows.min()),
        "monthly_names_median": float(monthly_rows.median()),
        "monthly_names_maximum": int(monthly_rows.max()),
        "duplicate_signal_security_rows": 0,
        "incomplete_label_rows": 0,
        "partial_label_months": 0,
        "future_price_availability_rows": 0,
        "future_residual_availability_rows": 0,
        "future_or_missing_fundamental_availability_rows": 0,
        "populated_fundamental_rows": int(populated_fundamental.sum()),
        "fundamental_age_days_median": float(
            fundamental_age.loc[populated_fundamental].median()
        ),
        "fundamental_age_days_p95": float(
            fundamental_age.loc[populated_fundamental].quantile(0.95)
        ),
        "fundamental_age_days_maximum": float(
            fundamental_age.loc[populated_fundamental].max()
        ),
        "fundamental_age_above_365_days_rows": int(
            fundamental_age.loc[populated_fundamental].gt(365).sum()
        ),
        "fundamental_age_above_730_days_rows": int(
            fundamental_age.loc[populated_fundamental].gt(730).sum()
        ),
        "total_liabilities_debt_fallback_rows": int(
            frame.loc[populated_fundamental, "debt_definition"]
            .eq("total_liabilities_fallback")
            .sum()
        ),
        "target_return_minimum": float(target.min()),
        "target_return_maximum": float(target.max()),
        "target_return_above_100pct_rows": int(target.gt(1.0).sum()),
        "target_return_absolute_above_50pct_rows": int(target.abs().gt(0.5).sum()),
        "target_return_reconstruction_max_abs_error": float(
            np.max(np.abs(target - reconstructed))
        ),
        "membership_selected_before_label_completion": True,
        "survivorship_free": False,
        "delisting_complete": False,
        "permanent_security_identifier": False,
    }
    return frame.sort_values([date_column, security_column], kind="mergesort").reset_index(
        drop=True
    ), quality


def build_oriented_scores(
    signal_frame: pd.DataFrame,
    definitions: Sequence[FeatureDefinition],
    *,
    security_column: str = "ticker",
) -> pd.DataFrame:
    """Build oriented signals without accepting a target/label column."""

    if "target_return" in signal_frame.columns:
        raise ValueError("signal construction must not receive target_return")
    required = {security_column, *(name for item in definitions for name, _ in item.components)}
    missing = sorted(required.difference(signal_frame.columns))
    if missing:
        raise ValueError(f"signal frame missing columns: {missing}")
    if signal_frame[security_column].isna().any() or signal_frame[security_column].duplicated().any():
        raise ValueError("signal cross-section needs unique, non-missing security ids")
    rows: list[pd.DataFrame] = []
    for definition in definitions:
        if definition.kind == "individual":
            component, direction = definition.components[0]
            raw = _numeric(signal_frame[component])
            oriented = raw * float(direction)
            component_count = raw.notna().astype(int)
        else:
            ranked_components: list[pd.Series] = []
            for component, direction in definition.components:
                ranked_components.append(centered_rank(signal_frame[component]) * float(direction))
            component_matrix = pd.concat(ranked_components, axis=1)
            component_count = component_matrix.notna().sum(axis=1)
            oriented = component_matrix.mean(axis=1, skipna=True)
            oriented = oriented.where(component_count.ge(definition.minimum_components))
        rows.append(
            pd.DataFrame(
                {
                    security_column: signal_frame[security_column].astype(str).to_numpy(),
                    "feature_id": definition.feature_id,
                    "family": definition.family,
                    "definition_kind": definition.kind,
                    "direction_status": definition.direction_status,
                    "oriented_score": oriented.to_numpy(float),
                    "component_count": component_count.to_numpy(int),
                    "component_total": len(definition.components),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3:
        return np.nan
    left_std = float(np.std(left, ddof=1))
    right_std = float(np.std(right, ddof=1))
    if left_std <= 1e-14 or right_std <= 1e-14:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def _residualize(values: np.ndarray, control: np.ndarray) -> np.ndarray:
    if len(values) != len(control):
        raise ValueError("residualization arrays differ in length")
    design = np.column_stack([np.ones(len(values)), control])
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coefficients


def deterministic_deciles(
    scores: pd.Series,
    securities: pd.Series,
    *,
    bins: int = 10,
) -> pd.Series:
    """Assign equal-count deciles with a declared ticker tie breaker."""

    if bins < 2:
        raise ValueError("bins must be at least two")
    frame = pd.DataFrame(
        {"score": _numeric(scores), "security": securities.astype(str)}, index=scores.index
    )
    if frame["score"].isna().any():
        raise ValueError("decile score contains missing values")
    if frame["security"].duplicated().any():
        raise ValueError("decile securities must be unique")
    if len(frame) < bins:
        raise ValueError("cross-section is smaller than requested decile count")
    ordered = frame.sort_values(["score", "security"], kind="mergesort")
    labels = np.floor(np.arange(len(ordered)) * bins / len(ordered)).astype(int) + 1
    result = pd.Series(index=ordered.index, data=labels, dtype=int)
    return result.reindex(scores.index)


def compute_monthly_diagnostics(
    panel: pd.DataFrame,
    definitions: Sequence[FeatureDefinition],
    *,
    momentum_column: str = "exact_total_momentum_12_1",
    security_column: str = "ticker",
    date_column: str = "signal_date",
    target_column: str = "target_return",
    minimum_cross_section: int = 100,
    deciles: int = 10,
    target_winsor_lower: float = 0.01,
    target_winsor_upper: float = 0.99,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute monthly IC, momentum-orthogonal IC, coverage, and deciles."""

    if minimum_cross_section < deciles:
        raise ValueError("minimum_cross_section must be at least the decile count")
    if not 0 <= target_winsor_lower < target_winsor_upper <= 1:
        raise ValueError("winsor quantiles are invalid")
    monthly_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    decile_rows: list[dict[str, Any]] = []
    for signal_date, month in panel.groupby(date_column, sort=True):
        if month[target_column].isna().any():
            raise ValueError(f"partial label month at {signal_date}")
        signal_columns = {
            security_column,
            momentum_column,
            *(name for item in definitions for name, _ in item.components),
        }
        # Explicitly exclude the target from the score-construction API.
        signal_frame = month.loc[:, sorted(signal_columns)].copy()
        scores = build_oriented_scores(
            signal_frame, definitions, security_column=security_column
        )
        labels = month[[security_column, target_column, momentum_column]].copy()
        evaluated = scores.merge(
            labels,
            on=security_column,
            how="left",
            validate="many_to_one",
        )
        if evaluated[target_column].isna().any():
            raise ValueError("score/label merge lost complete realized labels")
        total_names = int(len(month))
        for definition in definitions:
            feature = evaluated.loc[evaluated["feature_id"].eq(definition.feature_id)].copy()
            valid_mask = (
                np.isfinite(feature["oriented_score"])
                & np.isfinite(feature[target_column])
                & np.isfinite(feature[momentum_column])
            )
            valid = feature.loc[valid_mask].copy()
            count = int(len(valid))
            coverage = count / total_names
            coverage_rows.append(
                {
                    "signal_date": signal_date,
                    "feature_id": definition.feature_id,
                    "family": definition.family,
                    "definition_kind": definition.kind,
                    "total_names": total_names,
                    "observed_names": count,
                    "coverage": coverage,
                    "mean_component_count": float(
                        feature.loc[valid_mask, "component_count"].mean()
                    )
                    if count
                    else np.nan,
                    "component_total": len(definition.components),
                }
            )
            base_row: dict[str, Any] = {
                "signal_date": signal_date,
                "feature_id": definition.feature_id,
                "family": definition.family,
                "definition_kind": definition.kind,
                "direction_status": definition.direction_status,
                "total_names": total_names,
                "observed_names": count,
                "coverage": coverage,
                "mean_component_count": float(valid["component_count"].mean())
                if count
                else np.nan,
                "component_total": len(definition.components),
                "status": "evaluable" if count >= minimum_cross_section else "insufficient_cross_section",
            }
            metric_names = (
                "oriented_rank_ic",
                "oriented_decile_spread",
                "oriented_winsorized_decile_spread",
                "oriented_decile_monotonicity",
                "score_momentum_rank_correlation",
                "momentum_orthogonal_rank_ic",
                "momentum_orthogonal_decile_spread",
                "momentum_orthogonal_winsorized_decile_spread",
                "momentum_orthogonal_decile_monotonicity",
            )
            if count < minimum_cross_section:
                base_row.update({name: np.nan for name in metric_names})
                monthly_rows.append(base_row)
                continue

            score_rank = centered_rank(valid["oriented_score"]).to_numpy(float)
            target_rank = centered_rank(valid[target_column]).to_numpy(float)
            momentum_rank = centered_rank(valid[momentum_column]).to_numpy(float)
            raw_ic = _correlation(score_rank, target_rank)
            score_momentum_correlation = _correlation(score_rank, momentum_rank)
            score_residual = _residualize(score_rank, momentum_rank)
            target_residual = _residualize(target_rank, momentum_rank)
            orthogonal_ic = _correlation(score_residual, target_residual)
            target_lower = float(valid[target_column].quantile(target_winsor_lower))
            target_upper = float(valid[target_column].quantile(target_winsor_upper))
            valid["winsorized_target_return"] = valid[target_column].clip(
                lower=target_lower, upper=target_upper
            )

            raw_decile = deterministic_deciles(
                valid["oriented_score"], valid[security_column], bins=deciles
            )
            raw_means = valid.assign(decile=raw_decile).groupby("decile", sort=True).agg(
                count=(target_column, "size"),
                mean_return=(target_column, "mean"),
                winsorized_mean_return=("winsorized_target_return", "mean"),
            )
            if len(raw_means) != deciles:
                raise RuntimeError("raw score did not produce every requested decile")
            raw_spread = float(raw_means["mean_return"].iloc[-1] - raw_means["mean_return"].iloc[0])
            raw_winsor_spread = float(
                raw_means["winsorized_mean_return"].iloc[-1]
                - raw_means["winsorized_mean_return"].iloc[0]
            )
            raw_monotonicity = _correlation(
                np.arange(1, deciles + 1, dtype=float),
                raw_means["mean_return"].to_numpy(float),
            )
            for decile, row in raw_means.iterrows():
                decile_rows.append(
                    {
                        "signal_date": signal_date,
                        "feature_id": definition.feature_id,
                        "family": definition.family,
                        "score_basis": "oriented_raw",
                        "decile": int(decile),
                        "names": int(row["count"]),
                        "mean_return": float(row["mean_return"]),
                        "winsorized_mean_return": float(row["winsorized_mean_return"]),
                    }
                )

            orthogonal_spread = np.nan
            orthogonal_winsor_spread = np.nan
            orthogonal_monotonicity = np.nan
            if np.isfinite(orthogonal_ic):
                orthogonal_score = pd.Series(score_residual, index=valid.index)
                orthogonal_decile = deterministic_deciles(
                    orthogonal_score, valid[security_column], bins=deciles
                )
                orthogonal_means = valid.assign(decile=orthogonal_decile).groupby(
                    "decile", sort=True
                ).agg(
                    count=(target_column, "size"),
                    mean_return=(target_column, "mean"),
                    winsorized_mean_return=("winsorized_target_return", "mean"),
                )
                if len(orthogonal_means) != deciles:
                    raise RuntimeError("orthogonal score did not produce every requested decile")
                orthogonal_spread = float(
                    orthogonal_means["mean_return"].iloc[-1]
                    - orthogonal_means["mean_return"].iloc[0]
                )
                orthogonal_winsor_spread = float(
                    orthogonal_means["winsorized_mean_return"].iloc[-1]
                    - orthogonal_means["winsorized_mean_return"].iloc[0]
                )
                orthogonal_monotonicity = _correlation(
                    np.arange(1, deciles + 1, dtype=float),
                    orthogonal_means["mean_return"].to_numpy(float),
                )
                for decile, row in orthogonal_means.iterrows():
                    decile_rows.append(
                        {
                            "signal_date": signal_date,
                            "feature_id": definition.feature_id,
                            "family": definition.family,
                            "score_basis": "momentum_orthogonal",
                            "decile": int(decile),
                            "names": int(row["count"]),
                            "mean_return": float(row["mean_return"]),
                            "winsorized_mean_return": float(row["winsorized_mean_return"]),
                        }
                    )

            base_row.update(
                {
                    "oriented_rank_ic": raw_ic,
                    "oriented_decile_spread": raw_spread,
                    "oriented_winsorized_decile_spread": raw_winsor_spread,
                    "oriented_decile_monotonicity": raw_monotonicity,
                    "score_momentum_rank_correlation": score_momentum_correlation,
                    "momentum_orthogonal_rank_ic": orthogonal_ic,
                    "momentum_orthogonal_decile_spread": orthogonal_spread,
                    "momentum_orthogonal_winsorized_decile_spread": orthogonal_winsor_spread,
                    "momentum_orthogonal_decile_monotonicity": orthogonal_monotonicity,
                }
            )
            monthly_rows.append(base_row)
    monthly = pd.DataFrame(monthly_rows).sort_values(
        ["feature_id", "signal_date"], kind="mergesort"
    )
    coverage = pd.DataFrame(coverage_rows).sort_values(
        ["feature_id", "signal_date"], kind="mergesort"
    )
    decile_frame = pd.DataFrame(decile_rows).sort_values(
        ["feature_id", "signal_date", "score_basis", "decile"], kind="mergesort"
    )
    return monthly.reset_index(drop=True), coverage.reset_index(drop=True), decile_frame.reset_index(drop=True)


def stable_seed(base_seed: int, *parts: str) -> int:
    payload = "|".join([str(base_seed), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def moving_block_bootstrap_mean(
    values: pd.Series | np.ndarray,
    *,
    block_length: int,
    simulations: int,
    seed: int,
) -> dict[str, float]:
    """Circular moving-block interval and centered one-sided null p-value."""

    numeric = np.asarray(pd.Series(values).dropna(), dtype=float)
    numeric = numeric[np.isfinite(numeric)]
    n = len(numeric)
    if block_length < 1 or simulations < 100:
        raise ValueError("invalid block bootstrap settings")
    if n < 2 * block_length:
        return {
            "block_mean_p05": np.nan,
            "block_mean_p50": np.nan,
            "block_mean_p95": np.nan,
            "block_one_sided_pvalue": np.nan,
        }
    rng = np.random.default_rng(seed)
    block_count = int(np.ceil(n / block_length))
    offsets = np.arange(block_length)
    draws = np.empty(simulations, dtype=float)
    null_draws = np.empty(simulations, dtype=float)
    centered = numeric - numeric.mean()
    for simulation in range(simulations):
        starts = rng.integers(0, n, size=block_count)
        indices = ((starts[:, None] + offsets[None, :]) % n).reshape(-1)[:n]
        draws[simulation] = numeric[indices].mean()
        null_draws[simulation] = centered[indices].mean()
    quantiles = np.quantile(draws, [0.05, 0.50, 0.95])
    observed = float(numeric.mean())
    pvalue = (1.0 + float(np.count_nonzero(null_draws >= observed))) / (
        simulations + 1.0
    )
    return {
        "block_mean_p05": float(quantiles[0]),
        "block_mean_p50": float(quantiles[1]),
        "block_mean_p95": float(quantiles[2]),
        "block_one_sided_pvalue": float(pvalue),
    }


def _metric_summary(
    series: pd.Series,
    *,
    prefix: str,
    hac_lags: int,
    block_length: int,
    simulations: int,
    seed: int,
) -> dict[str, float]:
    values = _numeric(series).dropna()
    tstat = float(newey_west_tstat(values, lags=hac_lags))
    block = moving_block_bootstrap_mean(
        values,
        block_length=block_length,
        simulations=simulations,
        seed=seed,
    )
    return {
        f"{prefix}_months": int(len(values)),
        f"mean_{prefix}": float(values.mean()) if len(values) else np.nan,
        f"{prefix}_hac_tstat": tstat,
        f"{prefix}_hac_one_sided_pvalue": float(norm.sf(tstat))
        if np.isfinite(tstat)
        else np.nan,
        f"{prefix}_positive_month_fraction": float(values.gt(0).mean())
        if len(values)
        else np.nan,
        **{f"{prefix}_{name}": value for name, value in block.items()},
    }


def period_feature_summary(
    monthly: pd.DataFrame,
    period_config: Mapping[str, Mapping[str, Any]],
    *,
    simulations: int,
    base_seed: int,
) -> pd.DataFrame:
    """Summarize each feature with HAC and month-block inference by period."""

    frame = monthly.copy()
    frame["signal_date"] = _as_utc(frame["signal_date"], column="signal_date")
    metrics = (
        "oriented_rank_ic",
        "oriented_decile_spread",
        "oriented_winsorized_decile_spread",
        "momentum_orthogonal_rank_ic",
        "momentum_orthogonal_decile_spread",
        "momentum_orthogonal_winsorized_decile_spread",
    )
    rows: list[dict[str, Any]] = []
    for period, specification in period_config.items():
        mask = pd.Series(True, index=frame.index)
        if specification.get("signal_at_or_after") is not None:
            start = pd.Timestamp(specification["signal_at_or_after"])
            start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
            mask &= frame["signal_date"].ge(start)
        if specification.get("signal_before") is not None:
            end = pd.Timestamp(specification["signal_before"])
            end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
            mask &= frame["signal_date"].lt(end)
        subset = frame.loc[mask].copy()
        if subset.empty:
            raise ValueError(f"period {period!r} contains no monthly feature rows")
        hac_lags = int(specification["hac_lags"])
        block_length = int(specification["month_block_length"])
        for feature_id, group in subset.groupby("feature_id", sort=False):
            row: dict[str, Any] = {
                "period": period,
                "feature_id": feature_id,
                "family": group["family"].iloc[0],
                "definition_kind": group["definition_kind"].iloc[0],
                "direction_status": group["direction_status"].iloc[0],
                "calendar_months": int(group["signal_date"].nunique()),
                "first_signal_date": group["signal_date"].min(),
                "last_signal_date": group["signal_date"].max(),
                "mean_observed_names": float(group["observed_names"].mean()),
                "mean_coverage": float(group["coverage"].mean()),
                "minimum_monthly_coverage": float(group["coverage"].min()),
                "maximum_monthly_coverage": float(group["coverage"].max()),
                "mean_score_momentum_rank_correlation": float(
                    group["score_momentum_rank_correlation"].mean()
                ),
                "hac_lags": hac_lags,
                "month_block_length": block_length,
                "month_block_simulations": simulations,
            }
            for metric in metrics:
                row.update(
                    _metric_summary(
                        group[metric],
                        prefix=metric,
                        hac_lags=hac_lags,
                        block_length=block_length,
                        simulations=simulations,
                        seed=stable_seed(base_seed, period, feature_id, metric),
                    )
                )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["period", "feature_id"], kind="mergesort").reset_index(
        drop=True
    )


def benjamini_hochberg(pvalues: pd.Series) -> pd.Series:
    values = _numeric(pvalues)
    output = pd.Series(np.nan, index=pvalues.index, dtype=float)
    valid = values.dropna().sort_values(kind="mergesort")
    if valid.empty:
        return output
    adjusted = valid.to_numpy(float) * len(valid) / np.arange(1, len(valid) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output.loc[valid.index] = np.minimum(adjusted, 1.0)
    return output


def apply_primary_screen(
    period_summary: pd.DataFrame,
    *,
    primary_period: str,
    momentum_feature: str,
    gates: Mapping[str, float],
    family_size_minimum: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply one frozen screen; every row remains NOT VALIDATED."""

    primary = period_summary.loc[period_summary["period"].eq(primary_period)].copy()
    if primary.empty:
        raise ValueError("primary period is absent from feature summary")
    is_momentum = primary["feature_id"].eq(momentum_feature)
    primary["screen_metric_basis"] = np.where(
        is_momentum, "oriented_raw_benchmark", "momentum_orthogonal_increment"
    )
    primary["screen_mean_ic"] = np.where(
        is_momentum,
        primary["mean_oriented_rank_ic"],
        primary["mean_momentum_orthogonal_rank_ic"],
    )
    primary["screen_hac_tstat"] = np.where(
        is_momentum,
        primary["oriented_rank_ic_hac_tstat"],
        primary["momentum_orthogonal_rank_ic_hac_tstat"],
    )
    primary["screen_hac_pvalue"] = np.where(
        is_momentum,
        primary["oriented_rank_ic_hac_one_sided_pvalue"],
        primary["momentum_orthogonal_rank_ic_hac_one_sided_pvalue"],
    )
    primary["screen_month_block_pvalue"] = np.where(
        is_momentum,
        primary["oriented_rank_ic_block_one_sided_pvalue"],
        primary["momentum_orthogonal_rank_ic_block_one_sided_pvalue"],
    )
    primary["screen_winsorized_decile_spread"] = np.where(
        is_momentum,
        primary["mean_oriented_winsorized_decile_spread"],
        primary["mean_momentum_orthogonal_winsorized_decile_spread"],
    )
    primary["screen_bh_qvalue_28"] = benjamini_hochberg(primary["screen_hac_pvalue"])
    primary["screen_family_bonferroni_pvalue_105"] = np.minimum(
        1.0, primary["screen_hac_pvalue"] * int(family_size_minimum)
    )

    conditions = {
        "minimum_monthly_coverage": primary["minimum_monthly_coverage"].ge(
            float(gates["minimum_monthly_coverage"])
        ),
        "positive_mean_oriented_ic": primary["screen_mean_ic"].gt(
            float(gates["minimum_mean_oriented_ic"])
        ),
        "hac_tstat": primary["screen_hac_tstat"].ge(
            float(gates["minimum_hac_tstat"])
        ),
        "month_block_pvalue": primary["screen_month_block_pvalue"].le(
            float(gates["maximum_month_block_pvalue"])
        ),
        "bh_qvalue": primary["screen_bh_qvalue_28"].le(
            float(gates["maximum_bh_qvalue"])
        ),
        "family_bonferroni_pvalue": primary[
            "screen_family_bonferroni_pvalue_105"
        ].le(float(gates["maximum_family_bonferroni_pvalue"])),
        "positive_winsorized_decile_spread": primary[
            "screen_winsorized_decile_spread"
        ].gt(float(gates["minimum_winsorized_decile_spread"])),
    }
    for name, values in conditions.items():
        primary[f"gate_{name}"] = values.fillna(False).astype(bool)
    gate_columns = [f"gate_{name}" for name in conditions]
    primary["statistical_screen_pass"] = primary[gate_columns].all(axis=1)
    primary["research_verdict"] = "NOT VALIDATED"
    primary["portfolio_authorized"] = False

    gate_rows: list[dict[str, Any]] = []
    for row in primary.itertuples(index=False):
        for name in conditions:
            gate_rows.append(
                {
                    "feature_id": row.feature_id,
                    "gate": name,
                    "passed": bool(getattr(row, f"gate_{name}")),
                    "research_verdict": "NOT VALIDATED",
                    "portfolio_authorized": False,
                }
            )
    return primary.reset_index(drop=True), pd.DataFrame(gate_rows)


def feature_redundancy_summary(
    panel: pd.DataFrame,
    definitions: Sequence[FeatureDefinition],
    *,
    minimum_pair_names: int = 100,
    security_column: str = "ticker",
    date_column: str = "signal_date",
) -> pd.DataFrame:
    """Average monthly rank correlations among individual feature signals."""

    individual = tuple(item for item in definitions if item.kind == "individual")
    correlations: dict[tuple[str, str], list[tuple[float, int]]] = {}
    for _, month in panel.groupby(date_column, sort=True):
        components = {security_column, *(item.feature_id for item in individual)}
        signal_frame = month.loc[:, sorted(components)].copy()
        scores = build_oriented_scores(
            signal_frame, individual, security_column=security_column
        )
        pivot = scores.pivot(index=security_column, columns="feature_id", values="oriented_score")
        ranked = pivot.rank(method="average", pct=True)
        identifiers = [item.feature_id for item in individual]
        for left_index, left in enumerate(identifiers):
            for right in identifiers[left_index + 1 :]:
                pair = ranked[[left, right]].dropna()
                if len(pair) < minimum_pair_names:
                    continue
                correlation = _correlation(
                    pair[left].to_numpy(float), pair[right].to_numpy(float)
                )
                if np.isfinite(correlation):
                    correlations.setdefault((left, right), []).append(
                        (correlation, int(len(pair)))
                    )
    rows: list[dict[str, Any]] = []
    for (left, right), values in correlations.items():
        array = np.asarray([value for value, _ in values], dtype=float)
        overlaps = np.asarray([overlap for _, overlap in values], dtype=int)
        rows.append(
            {
                "feature_left": left,
                "feature_right": right,
                "months": int(len(values)),
                "mean_pair_names": float(overlaps.mean()),
                "minimum_pair_names": int(overlaps.min()),
                "mean_monthly_rank_correlation": float(array.mean()),
                "mean_absolute_monthly_rank_correlation": float(np.abs(array).mean()),
                "maximum_absolute_monthly_rank_correlation": float(np.abs(array).max()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["mean_absolute_monthly_rank_correlation", "feature_left", "feature_right"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


__all__ = [
    "FUNDAMENTAL_FEATURES",
    "FeatureDefinition",
    "apply_primary_screen",
    "benjamini_hochberg",
    "build_oriented_scores",
    "centered_rank",
    "compute_monthly_diagnostics",
    "definitions_from_config",
    "deterministic_deciles",
    "feature_redundancy_summary",
    "moving_block_bootstrap_mean",
    "period_feature_summary",
    "stable_seed",
    "validate_source_panel",
]
