"""End-to-end public research pipeline."""

from __future__ import annotations

import json
import os
import platform
import shutil
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from alpha_factor_engine.contracts import (
    DatasetRecord,
    RunManifest,
    frame_bounds,
    sha256_directory,
    sha256_file,
    stable_json_hash,
    utc_now,
)
from alpha_factor_engine.data import FredClient, MarketDataClient, SecCompanyFactsClient, YahooFinanceClient
from alpha_factor_engine.features import (
    build_fundamental_features,
    build_macro_features,
    build_monthly_price_panel,
)
from alpha_factor_engine.features.fundamentals import FUNDAMENTAL_FEATURES
from alpha_factor_engine.reporting import create_charts, write_report
from alpha_factor_engine.research.metrics import (
    information_coefficient,
    moving_block_bootstrap_sharpe,
    performance_summary,
)
from alpha_factor_engine.research.models import (
    FEATURE_GROUPS,
    model_config_from_dict,
    walk_forward_predict,
)
from alpha_factor_engine.research.portfolio import build_long_short_portfolio


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def _run_id(config: dict[str, Any]) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{stable_json_hash(config)[:8]}"


def _ensure_confirmed_end(end: str) -> None:
    requested = pd.Timestamp(end).date()
    today = pd.Timestamp.now(tz="America/New_York").date()
    if requested >= today:
        raise ValueError(
            f"public formal run end={requested} is not strictly before current date={today}; "
            "use a separate provisional run"
        )


def _portfolio_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    portfolio = config["portfolio"]
    return {
        "long_quantile": float(portfolio["long_quantile"]),
        "short_quantile": float(portfolio["short_quantile"]),
        "gross_exposure": float(portfolio["gross_exposure"]),
        "max_name_weight": float(portfolio["max_name_weight"]),
        "standardized_exposure_tolerance": float(
            portfolio.get("standardized_exposure_tolerance", 0.15)
        ),
        "trading_cost_bps_one_way": float(portfolio["trading_cost_bps_one_way"]),
        "annual_short_borrow_bps": float(portfolio["annual_short_borrow_bps"]),
    }


def _metrics_table(
    portfolio_returns: dict[str, pd.DataFrame], holdout_start: str, seed: int
) -> pd.DataFrame:
    holdout = pd.Timestamp(holdout_start, tz="UTC")
    rows: list[dict[str, object]] = []
    for model, returns in portfolio_returns.items():
        signal = pd.to_datetime(returns["signal_date"], utc=True)
        periods = {
            "all_oos": returns,
            "validation": returns.loc[signal.lt(holdout)],
            "holdout": returns.loc[signal.ge(holdout)],
        }
        for period, values in periods.items():
            summary = performance_summary(values)
            if period == "holdout" and not values.empty:
                summary.update(moving_block_bootstrap_sharpe(values["net_return"], seed=seed))
            rows.append({"model": model, "period": period, **summary})
    return pd.DataFrame(rows)


def _artifact_inventory(output_dir: Path) -> list[dict[str, Any]]:
    """Hash every delivered file except the self-referential manifest."""

    inventory: list[dict[str, Any]] = []
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        if path.name == "data_manifest.json":
            continue
        item: dict[str, Any] = {
            "path": path.relative_to(output_dir).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        if path.suffix == ".parquet":
            frame = pd.read_parquet(path)
            item.update({"rows": len(frame), "columns": list(frame.columns)})
        elif path.suffix == ".csv":
            frame = pd.read_csv(path)
            item.update({"rows": len(frame), "columns": list(frame.columns)})
        inventory.append(item)
    return inventory


def run_public_research(config_path: str | Path, *, refresh: bool = False) -> dict[str, Path]:
    config_path = Path(config_path)
    config = load_config(config_path)
    _ensure_confirmed_end(config["research"]["end"])
    run_id = _run_id(config)
    output_dir = Path(config["data"]["output_dir"])
    data_dir = output_dir / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest = RunManifest(run_id, sha256_file(config_path))
    project_root = config_path.resolve().parent.parent
    source_sha256, source_file_count = sha256_directory(project_root / "src")
    runner_path = project_root / "scripts" / "run_public_research.py"
    pyproject_path = project_root / "pyproject.toml"
    code_provenance = {
        "source_tree_sha256": source_sha256,
        "source_file_count": source_file_count,
        "runner_sha256": sha256_file(runner_path),
        "pyproject_sha256": sha256_file(pyproject_path),
        "version_control": "unavailable; workspace is not a Git checkout",
        "environment": {
            "python": platform.python_version(),
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
            "scikit_learn": package_version("scikit-learn"),
            "scipy": package_version("scipy"),
        },
    }

    tickers = list(dict.fromkeys(config["universe"]["tickers"]))
    benchmark = config["data"]["benchmark"]
    price_tickers = [*tickers, benchmark] if benchmark not in tickers else tickers
    yahoo = YahooFinanceClient(Path(config["data"]["cache_dir"]) / "yahoo")
    daily_prices, price_records, price_audit = yahoo.fetch_many(
        price_tickers,
        config["research"]["start"],
        config["research"]["end"],
        refresh=refresh,
    )
    for record in price_records:
        manifest.add(record)
    daily_prices.to_parquet(data_dir / "daily_prices.parquet", index=False)
    price_audit.to_csv(output_dir / "yahoo_download_audit.csv", index=False)

    sec = SecCompanyFactsClient(Path(config["data"]["cache_dir"]) / "sec")
    concepts_path = data_dir / "sec_annual_concepts.parquet"
    metadata_path = data_dir / "company_metadata.parquet"
    sec_audit_path = output_dir / "sec_download_audit.csv"
    reuse_normalized_sec = (
        not refresh and concepts_path.exists() and metadata_path.exists() and sec_audit_path.exists()
    )
    if reuse_normalized_sec:
        annual_concepts = pd.read_parquet(concepts_path)
        company_metadata = pd.read_parquet(metadata_path)
        sec_audit = pd.read_csv(sec_audit_path)
        reuse_normalized_sec = set(company_metadata["ticker"].astype(str)) == set(tickers)
    if reuse_normalized_sec:
        eb = frame_bounds(annual_concepts, "effective_at")
        ab = frame_bounds(annual_concepts, "available_at")
        sec_records = [
            DatasetRecord(
                provider="SEC EDGAR XBRL Company Facts",
                dataset="normalized_companyfacts_universe",
                retrieved_at=utc_now(),
                source_url="https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json",
                sha256=sha256_file(concepts_path),
                rows=len(annual_concepts),
                min_effective_at=eb[0],
                max_effective_at=eb[1],
                min_available_at=ab[0],
                max_available_at=ab[1],
                finality="filed",
                revision_policy="normalized cache reused; raw responses remain in data/cache/sec",
                notes={"tickers": len(tickers), "raw_payloads_retained": True},
            )
        ]
    else:
        annual_concepts, company_metadata, sec_records, sec_audit = sec.fetch_universe(
            tickers, refresh=refresh
        )
    for record in sec_records:
        manifest.add(record)
    annual_concepts.to_parquet(concepts_path, index=False)
    company_metadata.to_parquet(metadata_path, index=False)
    sec_audit.to_csv(sec_audit_path, index=False)
    sec_filing_events = (
        annual_concepts[
            ["ticker", "cik", "accession", "form", "fiscal_end", "effective_at", "filed", "available_at"]
        ]
        .drop_duplicates(["ticker", "accession", "available_at"])
        .rename(columns={"form": "event_type", "filed": "announced_at"})
    )
    sec_filing_events["event_id"] = (
        sec_filing_events["ticker"].astype(str) + ":" + sec_filing_events["accession"].astype(str)
    )
    sec_filing_events["revision_id"] = sec_filing_events["accession"]
    sec_filing_events.to_parquet(data_dir / "sec_filing_events.parquet", index=False)

    fred = FredClient(Path(config["data"]["cache_dir"]) / "fred")
    macro_series: dict[str, pd.DataFrame] = {}
    for name, series_id in config["data"]["fred_series"].items():
        observations, record = fred.fetch_latest_series(
            series_id,
            config["research"]["start"],
            config["research"]["end"],
            availability_lag_days=1,
            refresh=refresh,
        )
        requested_start = pd.Timestamp(config["research"]["start"])
        observed_start = pd.to_datetime(observations["observation_date"]).min()
        if observed_start > requested_start + pd.Timedelta(days=31):
            raise RuntimeError(
                f"FRED {series_id} begins {observed_start.date()}, more than 31 days after "
                f"requested start {requested_start.date()}"
            )
        macro_series[name] = observations
        manifest.add(record)
        observations.to_parquet(data_dir / f"fred_{series_id}.parquet", index=False)

    option_smoke_status: dict[str, Any] = {"status": "not_configured"}
    option_config = config["data"].get("option_chain_smoke")
    if option_config:
        try:
            option_frame, option_record = MarketDataClient(
                cache_dir=Path(config["data"]["cache_dir"]) / "marketdata"
            ).fetch_option_chain(
                option_config["underlying"],
                quote_date=option_config["quote_date"],
                expiration=option_config.get("expiration"),
                refresh=refresh,
            )
            option_frame.to_parquet(
                data_dir
                / f"marketdata_{option_config['underlying'].lower()}_options_{option_config['quote_date']}.parquet",
                index=False,
            )
            manifest.add(option_record)
            option_smoke_status = {
                "status": "ok",
                "underlying": option_config["underlying"],
                "quote_date": option_config["quote_date"],
                "expiration": option_config.get("expiration"),
                "rows": len(option_frame),
                "used_in_strategy_pnl": False,
                "execution_grade": False,
            }
        except Exception as exc:
            option_smoke_status = {"status": "unavailable", "error": str(exc)[:240]}

    price_panel = build_monthly_price_panel(
        daily_prices,
        benchmark=benchmark,
        minimum_price=float(config["universe"]["minimum_price"]),
        minimum_adv_usd=float(config["universe"]["minimum_adv_usd"]),
    )
    fundamental_features = build_fundamental_features(price_panel, annual_concepts, daily_prices)
    macro_features = build_macro_features(price_panel[["signal_date"]], macro_series)
    panel = price_panel.merge(
        fundamental_features, on=["ticker", "signal_date"], how="left", validate="one_to_one"
    ).merge(macro_features, on="signal_date", how="left", validate="many_to_one")
    if not company_metadata.empty:
        company_metadata["sic"] = pd.to_numeric(company_metadata["sic"], errors="coerce")
        # The SEC submissions endpoint supplies a current SIC, not historical
        # classifications.  Keep it descriptive and exclude it from historical
        # score residualization.
        company_metadata["sic_division_current"] = (company_metadata["sic"] // 1000).astype("Int64")
        panel = panel.merge(
            company_metadata[
                ["ticker", "cik", "entity_name", "sic", "sic_description", "sic_division_current"]
            ],
            on="ticker",
            how="left",
            validate="many_to_one",
        )
    panel.to_parquet(data_dir / "monthly_feature_panel.parquet", index=False)

    feature_columns = [column for group in FEATURE_GROUPS.values() for column in group if column in panel]
    model_dict = dict(config["model"])
    model_dict["seed"] = config["research"]["seed"]
    resolved_model_config = model_config_from_dict(model_dict)
    predictions, importance, fit_audit = walk_forward_predict(
        panel,
        feature_columns,
        first_prediction=config["research"]["first_prediction"],
        half_life_years=float(config["model"]["recency_half_life_years"]),
        config=resolved_model_config,
    )
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)
    fit_audit.to_csv(output_dir / "fit_audit.csv", index=False)

    score_columns = ["ridge_score", "extra_trees_score", "histogram_gbdt_score", "ensemble_score"]
    portfolio_returns: dict[str, pd.DataFrame] = {}
    holding_frames: list[pd.DataFrame] = []
    for score in score_columns:
        returns, holdings = build_long_short_portfolio(
            predictions, score_column=score, **_portfolio_kwargs(config)
        )
        portfolio_returns[score] = returns
        holdings["model"] = score
        holding_frames.append(holdings)
    monthly_returns = pd.concat(portfolio_returns.values(), ignore_index=True)
    monthly_returns.to_csv(output_dir / "monthly_portfolio_returns.csv", index=False)
    pd.concat(holding_frames, ignore_index=True).to_parquet(output_dir / "portfolio_holdings.parquet", index=False)

    metrics = _metrics_table(
        portfolio_returns,
        config["research"]["holdout_start"],
        int(config["research"]["seed"]),
    )
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    ic = information_coefficient(predictions, "ensemble_score", neutralize=True)
    raw_ic = information_coefficient(predictions, "ensemble_score", neutralize=False).rename(
        columns={"ic": "raw_ic", "n": "raw_n", "score_basis": "raw_score_basis"}
    )
    ic = ic.merge(raw_ic, on="signal_date", how="outer", validate="one_to_one")
    ic.to_csv(output_dir / "monthly_ic.csv", index=False)

    holdout = pd.Timestamp(config["research"]["holdout_start"], tz="UTC")
    cost_rows: list[dict[str, float]] = []
    for cost in config["portfolio"]["cost_sensitivity_bps"]:
        kwargs = _portfolio_kwargs(config)
        kwargs["trading_cost_bps_one_way"] = float(cost)
        returns, _ = build_long_short_portfolio(predictions, score_column="ensemble_score", **kwargs)
        values = returns.loc[pd.to_datetime(returns["signal_date"], utc=True).ge(holdout)]
        cost_rows.append({"cost_bps": float(cost), **performance_summary(values)})
    cost_sensitivity = pd.DataFrame(cost_rows)
    cost_sensitivity.to_csv(output_dir / "cost_sensitivity.csv", index=False)

    factor_coverage = (
        predictions.assign(period=np.where(predictions["signal_date"].ge(holdout), "holdout", "validation"))
        .groupby("period")[feature_columns]
        .agg(lambda x: x.notna().mean())
        .T.reset_index()
        .rename(columns={"index": "feature"})
    )
    factor_coverage.to_csv(output_dir / "factor_coverage.csv", index=False)

    charts = create_charts(
        output_dir,
        predictions,
        portfolio_returns,
        importance,
        ic,
        cost_sensitivity,
        metrics,
        feature_columns,
        config["research"]["holdout_start"],
    )
    holdout_mask = pd.to_datetime(ic["signal_date"], utc=True).ge(holdout)
    holdout_ic = ic.loc[holdout_mask, "ic"]
    holdout_raw_ic = ic.loc[holdout_mask, "raw_ic"]
    ic_summary = {
        "mean_ic": float(holdout_ic.mean()),
        "ic_ir": float(holdout_ic.mean() / holdout_ic.std(ddof=1) * np.sqrt(12)) if holdout_ic.std(ddof=1) > 0 else np.nan,
        "positive_ic_fraction": float(holdout_ic.gt(0).mean()),
        "raw_mean_ic": float(holdout_raw_ic.mean()),
        "raw_ic_ir": (
            float(holdout_raw_ic.mean() / holdout_raw_ic.std(ddof=1) * np.sqrt(12))
            if holdout_raw_ic.std(ddof=1) > 0
            else np.nan
        ),
    }
    realized_months = int(
        predictions.groupby("signal_date")["target_return"].apply(lambda values: values.notna().all()).sum()
    )
    coverage_summary = {
        "tickers_requested": len(tickers),
        "tickers_with_prices": int(price_audit["status"].eq("ok").sum() - int(benchmark not in tickers)),
        "tickers_with_sec": int(sec_audit["status"].eq("ok").sum()),
        "tickers_modeled": int(predictions["ticker"].nunique()),
        "prediction_months": realized_months,
        "forecast_months_total": int(predictions["signal_date"].nunique()),
        "last_evaluable_signal_date": str(
            predictions.loc[predictions["target_return"].notna(), "signal_date"].max()
        ),
        "panel_rows": len(panel),
        "median_fundamental_coverage": float(predictions[FUNDAMENTAL_FEATURES].notna().mean().median()),
        "minimum_fundamental_coverage": float(predictions[FUNDAMENTAL_FEATURES].notna().mean().min()),
    }
    fundamental_coverage_by_factor = {
        column: float(value)
        for column, value in predictions[FUNDAMENTAL_FEATURES].notna().mean().items()
    }
    data_quality = {
        **coverage_summary,
        "fundamental_coverage_by_factor": fundamental_coverage_by_factor,
        "yahoo_errors": price_audit.loc[~price_audit["status"].eq("ok")].to_dict("records"),
        "sec_errors": sec_audit.loc[~sec_audit["status"].eq("ok")].to_dict("records"),
        "survivorship_free": False,
        "delisting_returns": False,
        "historical_borrow": False,
        "historical_industry_classification": False,
        "alfred_vintages_used": False,
        "execution_data": "next-session 09:30 America/New_York adjusted Yahoo open proxy; no TAQ/NBBO",
        "marketdata_token_present": bool(os.getenv("MARKETDATA_TOKEN") or os.getenv("MARKETDATA_API_KEY")),
        "marketdata_option_chain_smoke": option_smoke_status,
    }
    (output_dir / "data_quality.json").write_text(json.dumps(data_quality, indent=2, default=str))
    ensemble_holdout = metrics.loc[
        metrics["model"].eq("ensemble_score") & metrics["period"].eq("holdout")
    ].iloc[0]
    gates = config.get("gates", {})
    research_checks = {
        "holdout_sharpe": {
            "value": float(ensemble_holdout["sharpe"]),
            "minimum": float(gates.get("holdout_sharpe_minimum", 0.5)),
        },
        "newey_west_tstat": {
            "value": float(ensemble_holdout["newey_west_tstat"]),
            "minimum": float(gates.get("newey_west_tstat_minimum", 2.0)),
        },
        "bootstrap_sharpe_p05": {
            "value": float(ensemble_holdout["bootstrap_sharpe_p05"]),
            "minimum": float(gates.get("bootstrap_sharpe_p05_minimum", 0.0)),
        },
        "max_drawdown": {
            "value": float(ensemble_holdout["max_drawdown"]),
            "minimum": float(gates.get("maximum_drawdown_floor", -0.2)),
        },
    }
    research_pass = all(check["value"] >= check["minimum"] for check in research_checks.values())
    production_data_checks = {
        "survivorship_free": bool(data_quality["survivorship_free"]),
        "delisting_returns": bool(data_quality["delisting_returns"]),
        "historical_borrow": bool(data_quality["historical_borrow"]),
        "historical_industry_classification": bool(
            data_quality["historical_industry_classification"]
        ),
        "alfred_vintages_used": bool(data_quality["alfred_vintages_used"]),
        "execution_grade_quotes": data_quality["execution_data"] == "TAQ/NBBO",
    }
    deployment_decision = {
        "strategy": "nonlinear_ensemble_long_short",
        "research_status": "PASS" if research_pass else "FAIL",
        "production_status": "PASS" if research_pass and all(production_data_checks.values()) else "FAIL",
        "action": "CONTINUE_TO_PAPER_TRADING" if research_pass and all(production_data_checks.values()) else "REJECT_CURRENT_STRATEGY",
        "research_checks": research_checks,
        "production_data_checks": production_data_checks,
        "holdout_opened": True,
        "note": "Any redesign after this run requires a new sealed holdout; do not relabel this holdout as untouched.",
    }
    (output_dir / "deployment_decision.json").write_text(
        json.dumps(deployment_decision, indent=2, sort_keys=True)
    )
    shutil.copyfile(config_path, output_dir / "config_snapshot.yml")
    report = write_report(
        output_dir,
        run_id=run_id,
        metrics=metrics,
        ic_summary=ic_summary,
        coverage_summary=coverage_summary,
        data_quality=data_quality,
        charts=charts,
        config=config,
    )
    manifest.write(
        output_dir / "data_manifest.json",
        extra={
            "data_quality": data_quality,
            "code_provenance": code_provenance,
            "resolved_model_config": asdict(resolved_model_config),
            "resolved_portfolio_config": _portfolio_kwargs(config),
            "research_timing": {
                "signal": "month-end after close",
                "entry": "next trading day 09:30 America/New_York adjusted-open proxy",
                "exit": "next rebalance 09:30 America/New_York adjusted-open proxy",
                "formal_data_end": config["research"]["end"],
            },
            "artifacts": _artifact_inventory(output_dir),
        },
    )
    return {"report": report, "output_dir": output_dir, "manifest": output_dir / "data_manifest.json"}
