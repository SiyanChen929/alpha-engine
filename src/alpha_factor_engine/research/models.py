"""Cross-sectional nonlinear ensemble with chronological, expanding refits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from alpha_factor_engine.features.fundamentals import FUNDAMENTAL_FEATURES
from alpha_factor_engine.features.price import PRICE_FEATURES


MACRO_FEATURES = [
    "fed_funds",
    "yield_curve",
    "vix",
    "baa_yield",
    "fed_funds_change_3m",
    "yield_curve_change_3m",
    "vix_change_3m",
    "baa_yield_change_3m",
]

SUPPLEMENTAL_FEATURES = [
    "log_market_cap",
    "fundamental_age_days",
    "gross_margin",
    "cash_flow_to_assets",
]

FEATURE_GROUPS: dict[str, list[str]] = {
    "price": PRICE_FEATURES,
    "fundamental": FUNDAMENTAL_FEATURES + SUPPLEMENTAL_FEATURES,
    "macro": MACRO_FEATURES,
}


def _cross_sectional_rank(series: pd.Series) -> pd.Series:
    count = series.notna().sum()
    if count < 2:
        return pd.Series(np.nan, index=series.index)
    return series.rank(pct=True, method="average") * 2 - 1


def prepare_model_panel(
    panel: pd.DataFrame,
    feature_columns: list[str],
    *,
    target_column: str = "target_return",
) -> tuple[pd.DataFrame, list[str]]:
    output = panel.copy()
    missing = [column for column in feature_columns if column not in output.columns]
    if missing:
        raise ValueError(f"model panel missing features: {missing}")
    if target_column not in output.columns:
        raise ValueError(f"model panel missing target column: {target_column}")
    ranked_columns: list[str] = []
    for column in feature_columns:
        ranked = f"x_{column}"
        if column in MACRO_FEATURES:
            # Macro states are identical across names on a date; a
            # cross-sectional rank would collapse them to a constant.  Freeze
            # a causal time-series z-score using only earlier dates.
            by_date = (
                output[["signal_date", column]]
                .drop_duplicates("signal_date")
                .sort_values("signal_date")
                .set_index("signal_date")[column]
            )
            history_mean = by_date.expanding(min_periods=12).mean().shift(1)
            history_std = by_date.expanding(min_periods=12).std(ddof=0).shift(1).replace(0, np.nan)
            causal_z = (by_date - history_mean) / history_std
            output[ranked] = output["signal_date"].map(causal_z)
        else:
            output[ranked] = output.groupby("signal_date", group_keys=False)[column].transform(
                _cross_sectional_rank
            )
        ranked_columns.append(ranked)
        # Declare missingness schema ex ante for every feature. Deciding which
        # indicators exist from full-sample coverage would let future data
        # distribution alter an earlier walk-forward model's feature space.
        missing_column = f"x_missing_{column}"
        output[missing_column] = output[ranked].isna().astype(float)
        ranked_columns.append(missing_column)
    output[ranked_columns] = output[ranked_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    output["target_rank"] = output.groupby("signal_date", group_keys=False)[target_column].transform(
        _cross_sectional_rank
    )
    output["model_target_column"] = target_column
    return output, ranked_columns


@dataclass
class ModelConfig:
    seed: int = 929
    ridge_alpha: float = 10.0
    extra_trees_n_estimators: int = 300
    extra_trees_min_samples_leaf: int = 20
    extra_trees_max_features: float = 0.65
    gbdt_max_iter: int = 250
    gbdt_learning_rate: float = 0.04
    gbdt_max_leaf_nodes: int = 15
    gbdt_l2_regularization: float = 1.0
    ridge_weight: float = 0.20
    extra_trees_weight: float = 0.45
    gbdt_weight: float = 0.35


class NonlinearEnsemble:
    """Additive baseline plus two nonlinear learners with fixed ex-ante weights."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self.ridge = Ridge(alpha=self.config.ridge_alpha)
        self.extra_trees = ExtraTreesRegressor(
            n_estimators=self.config.extra_trees_n_estimators,
            min_samples_leaf=self.config.extra_trees_min_samples_leaf,
            max_features=self.config.extra_trees_max_features,
            n_jobs=-1,
            random_state=self.config.seed,
        )
        self.gbdt = HistGradientBoostingRegressor(
            max_iter=self.config.gbdt_max_iter,
            learning_rate=self.config.gbdt_learning_rate,
            max_leaf_nodes=self.config.gbdt_max_leaf_nodes,
            l2_regularization=self.config.gbdt_l2_regularization,
            # A random internal split would mix later training dates into an
            # earlier validation fold.  Use the ex-ante fixed iteration count.
            early_stopping=False,
            random_state=self.config.seed,
        )

    def fit(self, x: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray) -> "NonlinearEnsemble":
        self.ridge.fit(x, y, sample_weight=sample_weight)
        self.extra_trees.fit(x, y, sample_weight=sample_weight)
        self.gbdt.fit(x, y, sample_weight=sample_weight)
        return self

    def predict_components(self, x: pd.DataFrame) -> pd.DataFrame:
        ridge = self.ridge.predict(x)
        trees = self.extra_trees.predict(x)
        gbdt = self.gbdt.predict(x)
        return pd.DataFrame(
            {
                "ridge_score": ridge,
                "extra_trees_score": trees,
                "histogram_gbdt_score": gbdt,
            },
            index=x.index,
        )

    def feature_importance(self, feature_names: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {"feature": feature_names, "importance": self.extra_trees.feature_importances_}
        ).sort_values("importance", ascending=False)


def model_config_from_dict(config: dict[str, Any]) -> ModelConfig:
    weights = config.get("ensemble_weights", {})
    trees = config.get("extra_trees", {})
    gbdt = config.get("histogram_gbdt", {})
    return ModelConfig(
        seed=int(config.get("seed", 929)),
        extra_trees_n_estimators=int(trees.get("n_estimators", 300)),
        extra_trees_min_samples_leaf=int(trees.get("min_samples_leaf", 20)),
        extra_trees_max_features=float(trees.get("max_features", 0.65)),
        gbdt_max_iter=int(gbdt.get("max_iter", 250)),
        gbdt_learning_rate=float(gbdt.get("learning_rate", 0.04)),
        gbdt_max_leaf_nodes=int(gbdt.get("max_leaf_nodes", 15)),
        gbdt_l2_regularization=float(gbdt.get("l2_regularization", 1.0)),
        ridge_weight=float(weights.get("ridge", 0.20)),
        extra_trees_weight=float(weights.get("extra_trees", 0.45)),
        gbdt_weight=float(weights.get("histogram_gbdt", 0.35)),
    )


def rank_and_combine_components(
    components: pd.DataFrame,
    signal_dates: pd.Series,
    config: ModelConfig,
) -> pd.DataFrame:
    """Normalize each learner within a decision cross-section before weighting."""

    output = components.copy()
    dates = pd.Series(signal_dates.to_numpy(), index=output.index)
    component_weights = {
        "ridge_score": config.ridge_weight,
        "extra_trees_score": config.extra_trees_weight,
        "histogram_gbdt_score": config.gbdt_weight,
    }
    for column in component_weights:
        output[column] = output[column].groupby(dates).transform(_cross_sectional_rank)
    output["ensemble_score"] = sum(
        component_weights[column] * output[column] for column in component_weights
    )
    return output


def walk_forward_predict(
    panel: pd.DataFrame,
    feature_columns: list[str],
    *,
    first_prediction: str,
    half_life_years: float = 5.0,
    minimum_training_rows: int = 500,
    target_column: str = "target_return",
    config: ModelConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prepared, x_columns = prepare_model_panel(
        panel,
        feature_columns,
        target_column=target_column,
    )
    prepared["signal_date"] = pd.to_datetime(prepared["signal_date"], utc=True)
    prepared["exit_at"] = pd.to_datetime(prepared["exit_at"], utc=True, errors="coerce")
    prediction_universe = prepared.loc[
        prepared["signal_date"].ge(pd.Timestamp(first_prediction, tz="UTC"))
        & prepared["eligible"].fillna(False)
    ].copy()
    predictions: list[pd.DataFrame] = []
    importance_rows: list[pd.DataFrame] = []
    fit_audit: list[dict[str, object]] = []
    for year, test in prediction_universe.groupby(prediction_universe["signal_date"].dt.year, sort=True):
        cutoff = test["signal_date"].min()
        train = prepared.loc[
            prepared["exit_at"].lt(cutoff)
            & prepared["target_rank"].notna()
            & prepared["eligible"].fillna(False)
        ].copy()
        if len(train) < minimum_training_rows:
            fit_audit.append({"year": year, "status": "insufficient_history", "train_rows": len(train)})
            continue
        ages = (cutoff - train["exit_at"]).dt.total_seconds().div(86_400).clip(lower=0).to_numpy()
        weights = np.power(0.5, ages / (365.25 * half_life_years))
        resolved_config = config or ModelConfig()
        model = NonlinearEnsemble(resolved_config)
        model.fit(train[x_columns], train["target_rank"], sample_weight=weights)
        raw_component = model.predict_components(test[x_columns])
        component = rank_and_combine_components(
            raw_component, test["signal_date"], resolved_config
        )
        predicted = test.copy()
        for column in component:
            predicted[column] = component[column]
        predictions.append(predicted)
        importance = model.feature_importance(x_columns)
        importance["fit_year"] = year
        importance_rows.append(importance)
        fit_audit.append(
            {
                "year": year,
                "status": "ok",
                "cutoff": cutoff,
                "train_rows": len(train),
                "train_start": train["signal_date"].min(),
                "latest_label_exit_at": train["exit_at"].max(),
                "predict_rows": len(test),
                "predict_months": test["signal_date"].nunique(),
                "target_column": target_column,
                "gbdt_inner_validation": "disabled_fixed_iterations",
            }
        )
    if not predictions:
        raise RuntimeError("walk-forward model produced no predictions")
    return (
        pd.concat(predictions, ignore_index=True).sort_values(["signal_date", "ticker"]),
        pd.concat(importance_rows, ignore_index=True),
        pd.DataFrame(fit_audit),
    )
