"""Strict fold-local validation for the liquid momentum ML challenger.

The existing generic held-out-security helper prepares cross-sectional target
ranks before securities are assigned to folds.  That is convenient for an
ordinary cross-validation diagnostic, but it lets a training label rank depend
on the realized return of a held-out security.  This module deliberately uses
the opposite order:

1. form the signal-time candidate set;
2. assign permanent security identifiers to stable folds;
3. select the non-held-out training rows whose labels have exited;
4. construct feature ranks and the incremental target inside that training
   subset only.

The functions do not mutate their inputs.  The scoring path never reads a
return label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingRegressor

from alpha_factor_engine.contracts import PointInTimeViolation
from alpha_factor_engine.research.metrics import newey_west_tstat
from alpha_factor_engine.research.universe_validation import assign_security_folds


DEFAULT_INCREMENTAL_CONTROLS: tuple[str, ...] = (
    "exact_total_momentum_12_1",
    "beta_12m",
    "log_market_cap",
    "log_dollar_volume_3m",
)

CONTROL_MISSING_POLICY = "fold_local_centered_rank_zero_plus_missing_indicator"


@dataclass(frozen=True, slots=True)
class MonotonicGBDTConfig:
    """Fixed model settings for the single nonlinear challenger."""

    max_iter: int = 250
    learning_rate: float = 0.04
    max_leaf_nodes: int = 15
    l2_regularization: float = 1.0
    min_samples_leaf: int = 20
    seed: int = 929

    def __post_init__(self) -> None:
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if self.max_leaf_nodes < 2:
            raise ValueError("max_leaf_nodes must be at least two")
        if not np.isfinite(self.l2_regularization) or self.l2_regularization < 0:
            raise ValueError("l2_regularization must be non-negative and finite")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be positive")


@dataclass(frozen=True, slots=True)
class FoldLocalSplit:
    """Prepared inputs and direct lineage evidence for one fold/year fit."""

    train: pd.DataFrame
    test: pd.DataFrame
    feature_columns: tuple[str, ...]
    cutoff: pd.Timestamp
    security_fold: int
    test_year: int
    audit: Mapping[str, object]


def _as_utc(series: pd.Series, *, column: str) -> pd.Series:
    try:
        output = pd.to_datetime(series, utc=True, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{column} must contain valid timestamps") from error
    if output.isna().any():
        raise ValueError(f"{column} contains missing timestamps")
    return output


def _centered_rank(series: pd.Series) -> pd.Series:
    count = int(series.notna().sum())
    if count < 2:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return series.rank(method="average", pct=True) * 2.0 - 1.0


def select_top_momentum_candidates(
    frame: pd.DataFrame,
    *,
    candidate_count: int | None = 150,
    date_column: str = "signal_date",
    security_column: str = "ticker",
    momentum_column: str = "exact_total_momentum_12_1",
    eligible_column: str = "eligible",
) -> pd.DataFrame:
    """Select a deterministic candidate set using signal-time fields only.

    ``candidate_count=None`` retains every eligible row in each month while
    preserving the same stable momentum/ticker ordering used by the fixed-size
    candidate path.  This supports a full-mother-universe diagnostic without
    changing the existing Top-150 behavior.
    """

    if candidate_count is not None and candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    required = {date_column, security_column, momentum_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"candidate panel missing columns: {missing}")
    output = frame.copy()
    output[date_column] = _as_utc(output[date_column], column=date_column)
    if output[security_column].isna().any():
        raise ValueError(f"{security_column} contains missing identifiers")
    output[security_column] = output[security_column].astype(str)
    if output.duplicated([date_column, security_column]).any():
        raise ValueError("candidate panel contains duplicate decision/security rows")
    eligible = (
        output[eligible_column].fillna(False).astype(bool)
        if eligible_column in output
        else pd.Series(True, index=output.index)
    )
    output = output.loc[eligible].copy()
    momentum = pd.to_numeric(output[momentum_column], errors="coerce")
    if momentum.isna().any() or not np.isfinite(momentum.to_numpy()).all():
        raise PointInTimeViolation("eligible candidate momentum must be complete and finite")
    output[momentum_column] = momentum

    parts: list[pd.DataFrame] = []
    for signal_date, group in output.groupby(date_column, sort=True):
        if candidate_count is not None and len(group) < candidate_count:
            raise PointInTimeViolation(
                f"{signal_date} has {len(group)} eligible names; "
                f"candidate_count={candidate_count}"
            )
        selected = group.sort_values(
            [momentum_column, security_column],
            ascending=[False, True],
            kind="mergesort",
        )
        if candidate_count is not None:
            selected = selected.head(candidate_count)
        selected = selected.copy()
        selected["momentum_candidate_rank"] = np.arange(
            1, len(selected) + 1, dtype=int
        )
        parts.append(selected)
    if not parts:
        raise PointInTimeViolation("candidate panel contains no eligible rows")
    return pd.concat(parts, ignore_index=True).sort_values(
        [date_column, "momentum_candidate_rank"], kind="mergesort"
    ).reset_index(drop=True)


def build_incremental_target(
    frame: pd.DataFrame,
    *,
    target_column: str = "target_return",
    control_columns: Sequence[str] = DEFAULT_INCREMENTAL_CONTROLS,
    date_column: str = "signal_date",
    output_column: str = "incremental_target",
    control_missing_policy: str = CONTROL_MISSING_POLICY,
) -> pd.DataFrame:
    """Build a return-rank residual inside the supplied cross-sections only.

    Callers are responsible for passing only labels that are available at the
    fit cutoff.  Missing labels fail closed, so the function cannot silently
    shrink a historical cross-section using future outcome availability.
    """

    controls = tuple(control_columns)
    if control_missing_policy != CONTROL_MISSING_POLICY:
        raise ValueError(
            "unsupported incremental-target control missing policy: "
            f"{control_missing_policy!r}"
        )
    if not controls:
        raise ValueError("at least one incremental-target control is required")
    if len(set(controls)) != len(controls):
        raise ValueError("incremental-target controls must be unique")
    required = {date_column, target_column, *controls}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"incremental-target panel missing columns: {missing}")
    output = frame.copy()
    output[date_column] = _as_utc(output[date_column], column=date_column)
    output[output_column] = np.nan
    output["raw_target_rank"] = np.nan

    for signal_date, group in output.groupby(date_column, sort=True):
        target = pd.to_numeric(group[target_column], errors="coerce")
        if target.isna().any() or not np.isfinite(target.to_numpy()).all():
            raise PointInTimeViolation(
                f"training target is incomplete or non-finite at {signal_date}"
            )
        ranked_controls: list[np.ndarray] = []
        for column in controls:
            values = pd.to_numeric(group[column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            missing_control = values.isna().astype(float).to_numpy()
            ranked_control = _centered_rank(values).fillna(0.0).to_numpy(float)
            # A missing value is signal-time information.  Zero is the centered
            # rank median, and the separate flag prevents median imputation from
            # silently treating missing fundamentals as observed medians.
            ranked_controls.extend([ranked_control, missing_control])
        target_rank = _centered_rank(target).to_numpy(float)
        design = np.column_stack([np.ones(len(group)), *ranked_controls])
        effective_rank = int(np.linalg.matrix_rank(design))
        if len(group) <= effective_rank + 2:
            raise ValueError(
                "too few rows relative to the effective control rank to "
                f"residualize the target at {signal_date}: "
                f"rows={len(group)}, rank={effective_rank}"
            )
        coefficients, *_ = np.linalg.lstsq(design, target_rank, rcond=None)
        residual = target_rank - design @ coefficients
        output.loc[group.index, "raw_target_rank"] = target_rank
        output.loc[group.index, output_column] = residual
    return output


def prepare_cross_sectional_features(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    date_column: str = "signal_date",
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Rank features within the supplied subset without touching any label."""

    features = tuple(feature_columns)
    if not features or len(set(features)) != len(features):
        raise ValueError("feature_columns must contain unique feature names")
    missing = sorted(set(features).difference(frame.columns))
    if missing:
        raise ValueError(f"feature panel missing columns: {missing}")
    if date_column not in frame:
        raise ValueError(f"feature panel missing date column: {date_column}")
    output = frame.copy()
    output[date_column] = _as_utc(output[date_column], column=date_column)
    prepared_columns: list[str] = []
    for column in features:
        numeric = pd.to_numeric(output[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        ranked_column = f"x_{column}"
        missing_column = f"x_missing_{column}"
        output[ranked_column] = numeric.groupby(output[date_column]).transform(
            _centered_rank
        )
        output[missing_column] = output[ranked_column].isna().astype(float)
        output[ranked_column] = output[ranked_column].fillna(0.0)
        prepared_columns.extend([ranked_column, missing_column])
    return output, tuple(prepared_columns)


def monotonic_constraint_vector(
    prepared_feature_columns: Sequence[str],
    directions: Mapping[str, int],
) -> tuple[int, ...]:
    """Map raw-feature economic directions to the prepared feature schema."""

    invalid = {name: value for name, value in directions.items() if value not in {-1, 0, 1}}
    if invalid:
        raise ValueError(f"monotonic directions must be -1, 0, or 1: {invalid}")
    constraints: list[int] = []
    for column in prepared_feature_columns:
        if column.startswith("x_missing_"):
            constraints.append(0)
        elif column.startswith("x_"):
            constraints.append(int(directions.get(column[2:], 0)))
        else:
            raise ValueError(f"unexpected prepared feature column: {column}")
    return tuple(constraints)


def prepare_strict_fold_local_split(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    security_fold: int,
    test_year: int,
    folds: int = 5,
    seed: int = 929,
    candidate_count: int | None = 150,
    security_column: str = "ticker",
    date_column: str = "signal_date",
    exit_column: str = "exit_at",
    momentum_column: str = "exact_total_momentum_12_1",
    target_column: str = "target_return",
    control_columns: Sequence[str] = DEFAULT_INCREMENTAL_CONTROLS,
    control_missing_policy: str = CONTROL_MISSING_POLICY,
) -> FoldLocalSplit:
    """Prepare one fit after fold assignment, with no cross-fold target ranks."""

    if not 0 <= security_fold < folds:
        raise ValueError("security_fold must be in [0, folds)")
    candidates = select_top_momentum_candidates(
        frame,
        candidate_count=candidate_count,
        date_column=date_column,
        security_column=security_column,
        momentum_column=momentum_column,
    )
    if exit_column not in candidates:
        raise ValueError(f"candidate panel missing exit column: {exit_column}")
    candidates[exit_column] = _as_utc(candidates[exit_column], column=exit_column)
    fold_map = assign_security_folds(
        candidates,
        security_column=security_column,
        folds=folds,
        seed=seed,
    )
    candidates = candidates.merge(
        fold_map, on=security_column, how="left", validate="many_to_one"
    )
    return _prepare_split_from_folded_candidates(
        candidates,
        feature_columns,
        security_fold=security_fold,
        test_year=test_year,
        security_column=security_column,
        date_column=date_column,
        exit_column=exit_column,
        target_column=target_column,
        control_columns=control_columns,
        control_missing_policy=control_missing_policy,
    )


def _prepare_split_from_folded_candidates(
    candidates: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    security_fold: int,
    test_year: int,
    security_column: str,
    date_column: str,
    exit_column: str,
    target_column: str,
    control_columns: Sequence[str],
    control_missing_policy: str,
) -> FoldLocalSplit:
    """Prepare one split from a candidate panel with stable folds attached."""

    test = candidates.loc[
        candidates["security_fold"].eq(security_fold)
        & candidates[date_column].dt.year.eq(int(test_year))
    ].copy()
    if test.empty:
        raise ValueError(f"fold {security_fold} has no test rows in {test_year}")
    cutoff = pd.Timestamp(test[date_column].min()).tz_convert("UTC")
    train_raw = candidates.loc[
        candidates["security_fold"].ne(security_fold)
        & candidates[exit_column].lt(cutoff)
    ].copy()
    if train_raw.empty:
        raise ValueError(f"fold {security_fold} has no training rows before {cutoff}")
    train_target = build_incremental_target(
        train_raw,
        target_column=target_column,
        control_columns=control_columns,
        date_column=date_column,
        control_missing_policy=control_missing_policy,
    )
    train, prepared_columns = prepare_cross_sectional_features(
        train_target, feature_columns, date_column=date_column
    )
    test, test_columns = prepare_cross_sectional_features(
        test, feature_columns, date_column=date_column
    )
    if prepared_columns != test_columns:
        raise RuntimeError("train and test feature schemas differ")
    train_ids = set(train[security_column])
    test_ids = set(test[security_column])
    overlap = len(train_ids.intersection(test_ids))
    if overlap:
        raise RuntimeError("strict held-out split contains overlapping securities")
    latest_exit = pd.Timestamp(train[exit_column].max())
    if not latest_exit < cutoff:
        raise PointInTimeViolation("strict held-out split includes an unexited training label")
    audit: dict[str, object] = {
        "security_fold": int(security_fold),
        "test_year": int(test_year),
        "cutoff": cutoff,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_security_count": int(train[security_column].nunique()),
        "test_security_count": int(test[security_column].nunique()),
        "security_overlap": overlap,
        "latest_label_exit_at": latest_exit,
        "train_feature_rank_scope": "nonheldout_fold_only",
        "train_target_scope": "nonheldout_fold_only",
        "target_control_missing_policy": control_missing_policy,
        "test_feature_rank_scope": "heldout_fold_only",
    }
    return FoldLocalSplit(
        train=train,
        test=test,
        feature_columns=prepared_columns,
        cutoff=cutoff,
        security_fold=int(security_fold),
        test_year=int(test_year),
        audit=audit,
    )


def walk_forward_strict_security_holdout_predict(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    first_prediction: str,
    folds: int = 5,
    seed: int = 929,
    candidate_count: int | None = 150,
    security_column: str = "ticker",
    target_column: str = "target_return",
    control_columns: Sequence[str] = DEFAULT_INCREMENTAL_CONTROLS,
    control_missing_policy: str = CONTROL_MISSING_POLICY,
    minimum_training_rows: int = 3_500,
    half_life_years: float = 5.0,
    monotonic_directions: Mapping[str, int] | None = None,
    model_config: MonotonicGBDTConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run annual expanding fits with strict security and target isolation."""

    if minimum_training_rows <= 0:
        raise ValueError("minimum_training_rows must be positive")
    if not np.isfinite(half_life_years) or half_life_years <= 0:
        raise ValueError("half_life_years must be positive and finite")
    first = pd.Timestamp(first_prediction)
    first = first.tz_localize("UTC") if first.tzinfo is None else first.tz_convert("UTC")
    candidates = select_top_momentum_candidates(
        frame,
        candidate_count=candidate_count,
        security_column=security_column,
    )
    if "exit_at" not in candidates:
        raise ValueError("candidate panel missing exit column: exit_at")
    candidates["exit_at"] = _as_utc(candidates["exit_at"], column="exit_at")
    fold_map = assign_security_folds(
        candidates,
        security_column=security_column,
        folds=folds,
        seed=seed,
    )
    candidates = candidates.merge(
        fold_map, on=security_column, how="left", validate="many_to_one"
    )
    dates = candidates["signal_date"]
    years = sorted(dates.loc[dates.ge(first)].dt.year.unique().astype(int).tolist())
    if not years:
        raise ValueError("first_prediction leaves no test years")
    resolved = model_config or MonotonicGBDTConfig(seed=seed)
    directions = dict(monotonic_directions or {})
    prediction_parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    for security_fold in range(folds):
        for test_year in years:
            split = _prepare_split_from_folded_candidates(
                candidates,
                feature_columns,
                security_fold=security_fold,
                test_year=test_year,
                security_column=security_column,
                date_column="signal_date",
                exit_column="exit_at",
                target_column=target_column,
                control_columns=control_columns,
                control_missing_policy=control_missing_policy,
            )
            audit = dict(split.audit)
            if len(split.train) < minimum_training_rows:
                audit["status"] = "insufficient_history"
                audit_rows.append(audit)
                continue
            constraints = monotonic_constraint_vector(split.feature_columns, directions)
            ages = (
                (split.cutoff - split.train["exit_at"])
                .dt.total_seconds()
                .div(86_400.0)
                .clip(lower=0.0)
                .to_numpy(float)
            )
            sample_weight = np.power(
                0.5, ages / (365.25 * float(half_life_years))
            )
            model = HistGradientBoostingRegressor(
                max_iter=resolved.max_iter,
                learning_rate=resolved.learning_rate,
                max_leaf_nodes=resolved.max_leaf_nodes,
                l2_regularization=resolved.l2_regularization,
                min_samples_leaf=resolved.min_samples_leaf,
                early_stopping=False,
                monotonic_cst=constraints,
                random_state=resolved.seed,
            )
            model.fit(
                split.train.loc[:, split.feature_columns],
                split.train["incremental_target"],
                sample_weight=sample_weight,
            )
            predicted = split.test.copy()
            predicted["strict_ml_score"] = model.predict(
                split.test.loc[:, split.feature_columns]
            )
            prediction_parts.append(predicted)
            audit["status"] = "ok"
            audit["monotonic_nonzero_constraints"] = int(
                np.count_nonzero(constraints)
            )
            audit["target_column"] = "incremental_target"
            audit["gbdt_inner_validation"] = "disabled_fixed_iterations"
            audit_rows.append(audit)
    if not prediction_parts:
        raise RuntimeError("strict held-out model produced no predictions")
    predictions = pd.concat(prediction_parts, ignore_index=True).sort_values(
        ["signal_date", "security_fold", security_column], kind="mergesort"
    )
    return predictions.reset_index(drop=True), pd.DataFrame(audit_rows)


def strict_heldout_information_coefficient(
    predictions: pd.DataFrame,
    *,
    score_column: str = "strict_ml_score",
    target_column: str = "target_return",
    control_columns: Sequence[str] = DEFAULT_INCREMENTAL_CONTROLS,
    control_missing_policy: str = CONTROL_MISSING_POLICY,
    date_column: str = "signal_date",
    fold_column: str = "security_fold",
    minimum_cross_section: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Evaluate IC within each held-out fold, then average folds by month."""

    required = {score_column, target_column, date_column, fold_column, *control_columns}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"strict held-out IC missing columns: {missing}")
    rows: list[dict[str, object]] = []
    for (signal_date, security_fold), group in predictions.groupby(
        [date_column, fold_column], sort=True
    ):
        if len(group) < minimum_cross_section:
            continue
        evaluated = build_incremental_target(
            group,
            target_column=target_column,
            control_columns=control_columns,
            date_column=date_column,
            output_column="heldout_incremental_target",
            control_missing_policy=control_missing_policy,
        )
        score = pd.to_numeric(evaluated[score_column], errors="coerce")
        if score.isna().any() or not np.isfinite(score.to_numpy()).all():
            raise ValueError("strict held-out score is incomplete or non-finite")
        raw_ic = float(
            np.corrcoef(rankdata(score), rankdata(evaluated[target_column]))[0, 1]
        )
        incremental_ic = float(
            np.corrcoef(
                rankdata(score), rankdata(evaluated["heldout_incremental_target"])
            )[0, 1]
        )
        rows.append(
            {
                "signal_date": signal_date,
                "security_fold": int(security_fold),
                "n": int(len(evaluated)),
                "raw_rank_ic": raw_ic,
                "incremental_rank_ic": incremental_ic,
            }
        )
    fold_month = pd.DataFrame(rows)
    if fold_month.empty:
        raise RuntimeError("strict held-out IC has no evaluable fold-months")
    monthly = (
        fold_month.groupby("signal_date", sort=True)
        .agg(
            folds=("security_fold", "nunique"),
            names=("n", "sum"),
            raw_rank_ic=("raw_rank_ic", "mean"),
            incremental_rank_ic=("incremental_rank_ic", "mean"),
        )
        .reset_index()
    )
    summary = {
        "months": float(len(monthly)),
        "fold_months": float(len(fold_month)),
        "mean_raw_rank_ic": float(monthly["raw_rank_ic"].mean()),
        "mean_incremental_rank_ic": float(monthly["incremental_rank_ic"].mean()),
        "raw_rank_ic_hac_tstat": float(newey_west_tstat(monthly["raw_rank_ic"])),
        "incremental_rank_ic_hac_tstat": float(
            newey_west_tstat(monthly["incremental_rank_ic"])
        ),
        "positive_incremental_month_fraction": float(
            monthly["incremental_rank_ic"].gt(0.0).mean()
        ),
    }
    return fold_month, monthly, summary


__all__ = [
    "CONTROL_MISSING_POLICY",
    "DEFAULT_INCREMENTAL_CONTROLS",
    "FoldLocalSplit",
    "MonotonicGBDTConfig",
    "build_incremental_target",
    "monotonic_constraint_vector",
    "prepare_cross_sectional_features",
    "prepare_strict_fold_local_split",
    "select_top_momentum_candidates",
    "strict_heldout_information_coefficient",
    "walk_forward_strict_security_holdout_predict",
]
