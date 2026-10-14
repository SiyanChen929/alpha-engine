"""Leakage-safe momentum-anchored residual-ML score overlays.

The overlay is deliberately a scoring-time utility.  It accepts only
contemporaneous scores and exposures, rejects return-label columns, and never
fits weights from realized outcomes.  The fixed blend is applied to centered
cross-sectional ranks so score units cannot change the allocation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from alpha_factor_engine.contracts import PointInTimeViolation


_KNOWN_RETURN_LABELS = frozenset(
    {
        "forward_return",
        "future_return",
        "label",
        "next_period_return",
        "next_return",
        "realized_return",
        "residual_target_return",
        "target",
        "target_rank",
        "target_return",
        "y",
    }
)
_ORTHOGONALIZATION_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class ResidualOverlayConfig:
    """Fixed, ex-ante overlay choices shared by every decision cross-section."""

    momentum_weight: float = 0.8
    orthogonalize_ml: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.momentum_weight, bool):
            raise ValueError("momentum_weight must be a finite number strictly between zero and one")
        try:
            weight = float(self.momentum_weight)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "momentum_weight must be a finite number strictly between zero and one"
            ) from error
        if not np.isfinite(weight) or not 0.0 < weight < 1.0:
            raise ValueError("momentum_weight must be a finite number strictly between zero and one")
        if not isinstance(self.orthogonalize_ml, bool):
            raise ValueError("orthogonalize_ml must be a boolean")
        object.__setattr__(self, "momentum_weight", weight)

    @property
    def residual_ml_weight(self) -> float:
        """The complementary fixed ML weight; 0.2 under the default configuration."""

        return float(round(1.0 - self.momentum_weight, 15))


def _normalized_column_name(column: Any) -> str:
    return str(column).strip().casefold().replace("-", "_").replace(" ", "_")


def _looks_like_return_label(column: Any) -> bool:
    name = _normalized_column_name(column)
    if name in _KNOWN_RETURN_LABELS:
        return True
    if name.startswith(("target_", "label_")) or name.endswith(("_target", "_label")):
        return True
    return "return" in name and name.startswith(
        ("forward_", "future_", "next_", "realized_", "subsequent_")
    )


def _reject_return_labels(frame: pd.DataFrame) -> None:
    forbidden = sorted(str(column) for column in frame.columns if _looks_like_return_label(column))
    if forbidden:
        raise PointInTimeViolation(
            "scoring frame contains return-label columns; provide a scoring-only frame: "
            f"{forbidden}"
        )


def _finite_numeric(series: pd.Series, *, column: str, kind: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    invalid = numeric.isna() | ~np.isfinite(numeric.to_numpy())
    if invalid.any():
        raise ValueError(
            f"{kind} column {column!r} contains {int(invalid.sum())} missing or non-finite values"
        )
    return numeric


def _centered_cross_sectional_rank(series: pd.Series) -> pd.Series:
    """Map average ranks to [-1, 1], preserving neutral score ties."""

    count = len(series)
    if count == 1:
        return pd.Series(0.0, index=series.index, dtype=float)
    average_rank = series.rank(method="average", ascending=True)
    return 2.0 * (average_rank - 1.0) / (count - 1.0) - 1.0


def _rank_correlation(left: pd.Series, right: pd.Series) -> float:
    left_values = left.to_numpy(float)
    right_values = right.to_numpy(float)
    if (
        len(left_values) < 2
        or np.ptp(left_values) <= _ORTHOGONALIZATION_TOLERANCE
        or np.ptp(right_values) <= _ORTHOGONALIZATION_TOLERANCE
    ):
        return float("nan")
    return float(np.corrcoef(left_values, right_values)[0, 1])


def _orthogonalize_one_cross_section(
    group: pd.DataFrame,
    *,
    ml_score_column: str,
    exposure_columns: tuple[str, str, str],
) -> tuple[np.ndarray, float, int]:
    """OLS-residualize ML scores using only this decision's exposures."""

    y = group[ml_score_column].to_numpy(float)
    design_parts = [np.ones(len(group), dtype=float)]
    active_exposures = 0
    for column in exposure_columns:
        values = group[column].to_numpy(float)
        centered = values - values.mean()
        scale = float(np.sqrt(np.mean(np.square(centered))))
        if scale > _ORTHOGONALIZATION_TOLERANCE:
            design_parts.append(centered / scale)
            active_exposures += 1
    design = np.column_stack(design_parts)
    design_rank = int(np.linalg.matrix_rank(design))
    if len(group) <= design_rank:
        raise ValueError(
            "orthogonalization requires at least one residual degree of freedom in every "
            "decision cross-section"
        )
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefficients
    residual = y - fitted
    scale = max(1.0, float(np.max(np.abs(y))))
    if float(np.max(np.abs(residual))) <= _ORTHOGONALIZATION_TOLERANCE * scale:
        residual = np.zeros_like(residual)

    centered_y = y - y.mean()
    total_sum_squares = float(centered_y @ centered_y)
    if total_sum_squares <= _ORTHOGONALIZATION_TOLERANCE:
        r_squared = 0.0
    else:
        residual_sum_squares = float(residual @ residual)
        r_squared = float(np.clip(1.0 - residual_sum_squares / total_sum_squares, 0.0, 1.0))
    return residual, r_squared, active_exposures


def build_momentum_residual_overlay(
    scoring_frame: pd.DataFrame,
    *,
    date_column: str = "signal_date",
    security_column: str = "ticker",
    momentum_score_column: str = "momentum_12_1",
    residual_ml_score_column: str = "residual_ml_score",
    beta_column: str = "beta_12m",
    size_column: str = "log_market_cap",
    momentum_weight: float = 0.8,
    orthogonalize_ml: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a fixed momentum/ML rank blend and per-decision diagnostics.

    The default score is ``0.8 * momentum_rank + 0.2 * residual_ml_rank``.
    When requested, the ML score is first projected on contemporaneous
    momentum, beta, and size within each decision cross-section.  Realized or
    forward-return labels are forbidden in ``scoring_frame``.

    Returns
    -------
    scores, diagnostics
        ``scores`` is deterministically ordered by decision and overlay rank.
        ``diagnostics`` reports rank correlation, orthogonalization strength,
        and the absolute contribution of each fixed sleeve for every decision.
    """

    if not isinstance(scoring_frame, pd.DataFrame):
        raise TypeError("scoring_frame must be a pandas DataFrame")
    if scoring_frame.empty:
        raise ValueError("scoring_frame cannot be empty")
    config = ResidualOverlayConfig(
        momentum_weight=momentum_weight,
        orthogonalize_ml=orthogonalize_ml,
    )
    _reject_return_labels(scoring_frame)

    required = [date_column, security_column, momentum_score_column, residual_ml_score_column]
    if config.orthogonalize_ml:
        required.extend([beta_column, size_column])
    if len(set(required)) != len(required):
        raise ValueError("scoring column names must be distinct")
    missing = [column for column in required if column not in scoring_frame.columns]
    if missing:
        raise ValueError(f"scoring frame missing columns: {missing}")

    working = scoring_frame.loc[:, required].copy()
    try:
        working[date_column] = pd.to_datetime(working[date_column], utc=True, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(f"date column {date_column!r} must contain valid timestamps") from error
    if working[date_column].isna().any():
        raise ValueError(f"date column {date_column!r} contains missing timestamps")
    if working[security_column].isna().any():
        raise ValueError(f"security column {security_column!r} contains missing identifiers")
    working[security_column] = working[security_column].astype(str).str.strip()
    if working[security_column].eq("").any():
        raise ValueError(f"security column {security_column!r} contains empty identifiers")
    duplicate = working.duplicated([date_column, security_column], keep=False)
    if duplicate.any():
        examples = working.loc[duplicate, [date_column, security_column]].head(5).to_dict("records")
        raise ValueError(f"duplicate decision/security rows are not allowed: {examples}")

    working[momentum_score_column] = _finite_numeric(
        working[momentum_score_column],
        column=momentum_score_column,
        kind="momentum score",
    )
    working[residual_ml_score_column] = _finite_numeric(
        working[residual_ml_score_column],
        column=residual_ml_score_column,
        kind="residual ML score",
    )
    if config.orthogonalize_ml:
        working[beta_column] = _finite_numeric(
            working[beta_column], column=beta_column, kind="orthogonalization exposure"
        )
        working[size_column] = _finite_numeric(
            working[size_column], column=size_column, kind="orthogonalization exposure"
        )

    scored_cross_sections: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for decision_at, group in working.groupby(date_column, sort=True):
        group = group.sort_values(security_column, kind="mergesort").copy()
        raw_ml_rank = _centered_cross_sectional_rank(group[residual_ml_score_column])
        if config.orthogonalize_ml:
            adjusted_ml, orthogonalization_r_squared, active_exposures = (
                _orthogonalize_one_cross_section(
                    group,
                    ml_score_column=residual_ml_score_column,
                    exposure_columns=(momentum_score_column, beta_column, size_column),
                )
            )
        else:
            adjusted_ml = group[residual_ml_score_column].to_numpy(float).copy()
            orthogonalization_r_squared = 0.0
            active_exposures = 0

        group["residual_ml_adjusted_score"] = adjusted_ml
        group["momentum_rank_score"] = _centered_cross_sectional_rank(
            group[momentum_score_column]
        )
        group["residual_ml_rank_score"] = _centered_cross_sectional_rank(
            group["residual_ml_adjusted_score"]
        )
        group["momentum_weight"] = config.momentum_weight
        group["residual_ml_weight"] = config.residual_ml_weight
        group["momentum_contribution"] = (
            config.momentum_weight * group["momentum_rank_score"]
        )
        group["residual_ml_contribution"] = (
            config.residual_ml_weight * group["residual_ml_rank_score"]
        )
        group["overlay_score"] = (
            group["momentum_contribution"] + group["residual_ml_contribution"]
        )

        raw_correlation = _rank_correlation(group["momentum_rank_score"], raw_ml_rank)
        adjusted_correlation = _rank_correlation(
            group["momentum_rank_score"], group["residual_ml_rank_score"]
        )
        momentum_absolute = float(group["momentum_contribution"].abs().mean())
        ml_absolute = float(group["residual_ml_contribution"].abs().mean())
        total_absolute = momentum_absolute + ml_absolute
        ml_share = ml_absolute / total_absolute if total_absolute > 0.0 else 0.0

        diagnostic_rows.append(
            {
                date_column: decision_at,
                "cross_section_size": len(group),
                "momentum_weight": config.momentum_weight,
                "residual_ml_weight": config.residual_ml_weight,
                "orthogonalized_ml": config.orthogonalize_ml,
                "active_orthogonalization_exposures": active_exposures,
                "orthogonalization_r_squared": orthogonalization_r_squared,
                "raw_momentum_ml_rank_correlation": raw_correlation,
                "momentum_ml_rank_correlation": adjusted_correlation,
                "mean_abs_momentum_contribution": momentum_absolute,
                "mean_abs_residual_ml_contribution": ml_absolute,
                "residual_ml_absolute_contribution_share": ml_share,
                "mean_abs_overlay_change_from_momentum": float(
                    (group["overlay_score"] - group["momentum_rank_score"]).abs().mean()
                ),
            }
        )

        group = group.sort_values(
            ["overlay_score", security_column],
            ascending=[False, True],
            kind="mergesort",
        )
        group["overlay_rank"] = np.arange(1, len(group) + 1, dtype=int)
        scored_cross_sections.append(group)

    scores = pd.concat(scored_cross_sections, ignore_index=True).sort_values(
        [date_column, "overlay_rank"], kind="mergesort"
    ).reset_index(drop=True)
    diagnostics = pd.DataFrame(diagnostic_rows).sort_values(date_column, kind="mergesort").reset_index(
        drop=True
    )
    return scores, diagnostics


__all__ = ["ResidualOverlayConfig", "build_momentum_residual_overlay"]
