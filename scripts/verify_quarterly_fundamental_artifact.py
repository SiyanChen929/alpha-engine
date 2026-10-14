#!/usr/bin/env python3
"""Independently reconstruct the quarterly-fundamental diagnostic evidence.

This verifier intentionally does not import the producer or project metric
helpers.  It recomputes residual targets, rank ICs, deciles, HAC inference and
manifest integrity directly from the persisted artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


FEATURES = (
    "quarterly_fundamental_momentum_score",
    "earnings_change_on_assets",
    "sales_growth_yoy",
    "gross_margin_change_yoy",
)
CONTROLS = (
    "exact_total_momentum_12_1",
    "reversal_1m",
    "beta_12m",
    "realized_vol_3m",
    "illiquidity_3m",
    "log_market_cap",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _centered_rank(values: pd.Series) -> np.ndarray:
    return (values.rank(method="average", pct=True) * 2.0 - 1.0).to_numpy()


def _hac_tstat(values: np.ndarray, lags: int = 3) -> float:
    values = np.asarray(values, dtype=float)
    count = len(values)
    centered = values - values.mean()
    long_run_variance = float(centered @ centered / count)
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        long_run_variance += 2.0 * weight * float(
            centered[lag:] @ centered[:-lag] / count
        )
    standard_error = np.sqrt(max(long_run_variance, 0.0) / count)
    return float(values.mean() / standard_error)


def _reconstruct(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ic_rows: list[dict[str, object]] = []
    decile_rows: list[dict[str, object]] = []
    for signal_date, month in panel.groupby("signal_date", sort=True):
        for feature in FEATURES:
            sample = month.loc[month[feature].notna()].copy()
            if len(sample) < 50:
                continue
            if sample["target_return"].isna().any():
                raise RuntimeError("target coverage is incomplete")
            ranked_controls: list[np.ndarray] = []
            for control in CONTROLS:
                values = pd.to_numeric(sample[control], errors="coerce").replace(
                    [np.inf, -np.inf], np.nan
                )
                ranked_controls.extend(
                    [
                        np.nan_to_num(_centered_rank(values), nan=0.0),
                        values.isna().astype(float).to_numpy(),
                    ]
                )
            target = sample["target_return"].to_numpy(float)
            target_rank = _centered_rank(sample["target_return"])
            design = np.column_stack(
                [np.ones(len(sample), dtype=float), *ranked_controls]
            )
            coefficients = np.linalg.lstsq(design, target_rank, rcond=None)[0]
            residual = target_rank - design @ coefficients
            score = sample[feature].to_numpy(float)
            score_rank = rankdata(score)
            ic_rows.append(
                {
                    "signal_date": signal_date,
                    "feature": feature,
                    "names": len(sample),
                    "raw_rank_ic": float(
                        np.corrcoef(score_rank, rankdata(target))[0, 1]
                    ),
                    "incremental_rank_ic": float(
                        np.corrcoef(score_rank, rankdata(residual))[0, 1]
                    ),
                }
            )
            ordered_rank = pd.Series(score).rank(method="first")
            deciles = (
                np.floor((ordered_rank - 1) * 10 / len(sample)).astype(int) + 1
            ).clip(1, 10)
            for decile in range(1, 11):
                member = target[deciles.to_numpy() == decile]
                decile_rows.append(
                    {
                        "signal_date": signal_date,
                        "feature": feature,
                        "decile": decile,
                        "names": len(member),
                        "equal_weight_return": float(member.mean()),
                    }
                )
    return pd.DataFrame(ic_rows), pd.DataFrame(decile_rows)


def verify(artifact: Path) -> dict[str, object]:
    panel = pd.read_parquet(artifact / "monthly_quarterly_signal_panel.parquet")
    panel["signal_date"] = pd.to_datetime(panel["signal_date"], utc=True)
    reconstructed_ic, reconstructed_deciles = _reconstruct(panel)

    observed_ic = pd.read_csv(artifact / "monthly_feature_ic.csv")
    observed_ic["signal_date"] = pd.to_datetime(observed_ic["signal_date"], utc=True)
    ic_compare = observed_ic.merge(
        reconstructed_ic,
        on=["signal_date", "feature", "names"],
        suffixes=("_observed", "_reconstructed"),
        validate="one_to_one",
    )
    if len(ic_compare) != len(observed_ic) or len(ic_compare) != len(reconstructed_ic):
        raise RuntimeError("monthly IC keys do not reconcile")
    ic_max_error = max(
        float(
            np.abs(
                ic_compare["raw_rank_ic_observed"]
                - ic_compare["raw_rank_ic_reconstructed"]
            ).max()
        ),
        float(
            np.abs(
                ic_compare["incremental_rank_ic_observed"]
                - ic_compare["incremental_rank_ic_reconstructed"]
            ).max()
        ),
    )

    observed_deciles = pd.read_csv(artifact / "decile_monthly.csv")
    observed_deciles["signal_date"] = pd.to_datetime(
        observed_deciles["signal_date"], utc=True
    )
    decile_compare = observed_deciles.merge(
        reconstructed_deciles,
        on=["signal_date", "feature", "decile", "names"],
        suffixes=("_observed", "_reconstructed"),
        validate="one_to_one",
    )
    if len(decile_compare) != len(observed_deciles) or len(decile_compare) != len(
        reconstructed_deciles
    ):
        raise RuntimeError("monthly decile keys do not reconcile")
    decile_max_error = float(
        np.abs(
            decile_compare["equal_weight_return_observed"]
            - decile_compare["equal_weight_return_reconstructed"]
        ).max()
    )

    primary = reconstructed_ic.loc[
        reconstructed_ic["feature"].eq(FEATURES[0])
        & reconstructed_ic["signal_date"].lt(pd.Timestamp("2026-01-01", tz="UTC"))
    ].sort_values("signal_date")
    primary_mean = float(primary["incremental_rank_ic"].mean())
    primary_hac_t = _hac_tstat(primary["incremental_rank_ic"].to_numpy())

    manifest_path = artifact / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    bad_records: list[str] = []
    for record in manifest["records"]:
        path = Path(record["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or _sha256(path) != record["sha256"]
        ):
            bad_records.append(record["path"])
    expected_outputs = {
        path.resolve().as_posix()
        for path in artifact.rglob("*")
        if path.is_file()
        and path.name != "artifact_manifest.json"
        and "independent_audit" not in path.relative_to(artifact).parts
    }
    recorded_outputs = {
        record["path"]
        for record in manifest["records"]
        if record["role"] == "output"
    }
    return {
        "verification_passed": bool(
            ic_max_error < 1e-12
            and decile_max_error < 1e-12
            and not bad_records
            and expected_outputs == recorded_outputs
        ),
        "monthly_ic_rows": len(reconstructed_ic),
        "monthly_decile_rows": len(reconstructed_deciles),
        "maximum_ic_absolute_error": ic_max_error,
        "maximum_decile_return_absolute_error": decile_max_error,
        "primary_pre2026_months": len(primary),
        "primary_pre2026_mean_incremental_rank_ic": primary_mean,
        "primary_pre2026_hac_tstat": primary_hac_t,
        "manifest_records": len(manifest["records"]),
        "manifest_bad_records": len(bad_records),
        "manifest_output_omissions": len(expected_outputs - recorded_outputs),
        "manifest_extra_outputs": len(recorded_outputs - expected_outputs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("artifacts/quarterly_fundamental_momentum_diagnostic_v1"),
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.artifact.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
