"""Research charts and a compact, evidence-linked run report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


COLORS = {
    "ensemble_score": "#155EEF",
    "extra_trees_score": "#17B26A",
    "histogram_gbdt_score": "#7A5AF8",
    "ridge_score": "#667085",
}


def _style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def create_charts(
    output_dir: str | Path,
    predictions: pd.DataFrame,
    portfolio_returns: dict[str, pd.DataFrame],
    importance: pd.DataFrame,
    ic: pd.DataFrame,
    cost_sensitivity: pd.DataFrame,
    metrics: pd.DataFrame,
    feature_columns: list[str],
    holdout_start: str,
) -> list[Path]:
    _style()
    root = Path(output_dir) / "charts"
    root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    holdout = pd.Timestamp(holdout_start, tz="UTC")

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for score, returns in portfolio_returns.items():
        values = returns.loc[pd.to_datetime(returns["signal_date"], utc=True).ge(holdout)].copy()
        if values.empty:
            continue
        wealth = (1 + values["net_return"]).cumprod()
        label = score.replace("_score", "").replace("_", " ").title()
        ax.plot(pd.to_datetime(values["exit_at"]), wealth, label=label, color=COLORS.get(score), lw=2)
    ax.axhline(1.0, color="#98A2B3", lw=1)
    ax.set(
        title="Opened chronological test: net long-short wealth",
        ylabel="Growth of $1",
        xlabel="Execution date",
    )
    ax.legend(frameon=False, ncol=2)
    path = root / "holdout_cumulative_returns.png"
    _save(fig, path)
    paths.append(path)

    holdout_metrics = metrics.loc[metrics["period"].eq("holdout")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    order = [item for item in COLORS if item in set(holdout_metrics["model"])]
    plot = holdout_metrics.set_index("model").reindex(order).reset_index()
    labels = plot["model"].str.replace("_score", "", regex=False).str.replace("_", " ").str.title()
    axes[0].bar(labels, plot["annualized_return"] * 100, color=[COLORS.get(item) for item in plot["model"]])
    axes[0].axhline(0, color="#667085", lw=1)
    axes[0].set(title="Opened-test annualized net return", ylabel="Percent")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(labels, plot["sharpe"], color=[COLORS.get(item) for item in plot["model"]])
    axes[1].axhline(0, color="#667085", lw=1)
    axes[1].set(title="Opened-test Sharpe ratio", ylabel="Sharpe")
    axes[1].tick_params(axis="x", rotation=25)
    path = root / "holdout_model_comparison.png"
    _save(fig, path)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    if not ic.empty:
        values = ic.copy()
        values["signal_date"] = pd.to_datetime(values["signal_date"])
        values["rolling_12m_ic"] = values["ic"].rolling(12, min_periods=6).mean()
        ax.bar(values["signal_date"], values["ic"], width=20, color="#D0D5DD", alpha=0.65, label="Monthly IC")
        ax.plot(values["signal_date"], values["rolling_12m_ic"], color="#155EEF", lw=2.2, label="12m mean")
    ax.axhline(0, color="#667085", lw=1)
    ax.set(title="Ensemble rank information coefficient", ylabel="Spearman IC", xlabel="Signal month")
    ax.legend(frameon=False)
    path = root / "information_coefficient.png"
    _save(fig, path)
    paths.append(path)

    mean_importance = importance.groupby("feature", as_index=False)["importance"].mean()
    mean_importance["feature"] = mean_importance["feature"].str.removeprefix("x_")
    mean_importance = mean_importance.loc[~mean_importance["feature"].str.startswith("missing_")]
    mean_importance = mean_importance.nlargest(18, "importance").sort_values("importance")
    fig, ax = plt.subplots(figsize=(9.5, 6.6))
    ax.barh(mean_importance["feature"], mean_importance["importance"], color="#7A5AF8")
    ax.set(title="Extra Trees importance across walk-forward refits", xlabel="Mean impurity importance")
    path = root / "feature_importance.png"
    _save(fig, path)
    paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    axes[0].plot(cost_sensitivity["cost_bps"], cost_sensitivity["annualized_return"] * 100, marker="o", color="#155EEF")
    axes[0].axhline(0, color="#667085", lw=1)
    axes[0].set(title="Opened-test cost sensitivity", xlabel="One-way trading cost (bps)", ylabel="Annualized net return (%)")
    axes[1].plot(cost_sensitivity["cost_bps"], cost_sensitivity["sharpe"], marker="o", color="#17B26A")
    axes[1].axhline(0, color="#667085", lw=1)
    axes[1].set(title="Sharpe after costs", xlabel="One-way trading cost (bps)", ylabel="Sharpe")
    path = root / "cost_sensitivity.png"
    _save(fig, path)
    paths.append(path)

    holdout_predictions = predictions.loc[predictions["signal_date"].ge(holdout)].copy()
    coverage = holdout_predictions[feature_columns].notna().mean().sort_values()
    fig, ax = plt.subplots(figsize=(9.5, max(7.0, 0.27 * len(coverage))))
    ax.barh(coverage.index, coverage.values * 100, color="#0BA5EC")
    ax.set(title="Opened-test feature coverage", xlabel="Non-missing observations (%)", xlim=(0, 100))
    path = root / "feature_coverage.png"
    _save(fig, path)
    paths.append(path)

    interaction = holdout_predictions[
        ["signal_date", "momentum_12_1", "gross_profitability", "target_return"]
    ].dropna()
    if len(interaction) >= 100:
        interaction["momentum_bin"] = interaction.groupby("signal_date")["momentum_12_1"].transform(
            lambda values: pd.qcut(values, 5, labels=False, duplicates="drop") + 1
        )
        interaction["profitability_bin"] = interaction.groupby("signal_date")[
            "gross_profitability"
        ].transform(lambda values: pd.qcut(values, 5, labels=False, duplicates="drop") + 1)
        date_cells = (
            interaction.dropna(subset=["momentum_bin", "profitability_bin"])
            .groupby(["signal_date", "profitability_bin", "momentum_bin"], as_index=False)[
                "target_return"
            ]
            .mean()
        )
        surface = date_cells.pivot_table(
            index="profitability_bin",
            columns="momentum_bin",
            values="target_return",
            aggfunc="mean",
        ) * 10_000
        cell_months = date_cells.pivot_table(
            index="profitability_bin",
            columns="momentum_bin",
            values="signal_date",
            aggfunc="count",
        ).reindex_like(surface)
        annotations = surface.copy().astype(object)
        for row in surface.index:
            for column in surface.columns:
                mean_value = surface.loc[row, column]
                count_value = cell_months.loc[row, column]
                annotations.loc[row, column] = (
                    f"{mean_value:.0f}\nN={int(count_value)}"
                    if pd.notna(mean_value) and pd.notna(count_value)
                    else ""
                )
        fig, ax = plt.subplots(figsize=(7.3, 5.8))
        sns.heatmap(
            surface,
            annot=annotations,
            fmt="",
            center=0,
            cmap="RdYlGn",
            ax=ax,
            cbar_kws={"label": "Equal-weight monthly mean return (bps)"},
        )
        ax.set(
            title="Descriptive within-month momentum × profitability",
            xlabel="12-1 momentum quintile",
            ylabel="Gross profitability quintile",
        )
        path = root / "momentum_profitability_interaction.png"
        _save(fig, path)
        paths.append(path)

    ensemble = portfolio_returns["ensemble_score"].copy()
    ensemble["year"] = pd.to_datetime(ensemble["signal_date"], utc=True).dt.year
    annual = ensemble.groupby("year")["net_return"].apply(lambda x: (1 + x).prod() - 1)
    annual_months = ensemble.groupby("year").size().reindex(annual.index)
    labels = [
        f"{year}\n({int(annual_months.loc[year])}m)"
        for year in annual.index
    ]
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    positions = np.arange(len(annual))
    ax.bar(positions, annual.values * 100, color=np.where(annual.ge(0), "#17B26A", "#D92D20"))
    ax.axhline(0, color="#667085", lw=1)
    opened_year = pd.Timestamp(holdout_start).year
    opened_position = next(
        (index - 0.5 for index, year in enumerate(annual.index) if year >= opened_year),
        len(annual) - 0.5,
    )
    ax.axvline(opened_position, color="#155EEF", ls="--", lw=1.5, label="Opened test begins")
    ax.set_xticks(positions, labels)
    ax.set(title="Ensemble net return by signal year", ylabel="Percent", xlabel="Signal year (months)")
    ax.legend(frameon=False)
    path = root / "annual_returns.png"
    _save(fig, path)
    paths.append(path)
    return paths


def _pct(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value * 100:.2f}%"


def _number(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:.2f}"


def write_report(
    output_dir: str | Path,
    *,
    run_id: str,
    metrics: pd.DataFrame,
    ic_summary: dict[str, float],
    coverage_summary: dict[str, Any],
    data_quality: dict[str, Any],
    charts: list[Path],
    config: dict[str, Any],
) -> Path:
    root = Path(output_dir)
    report_path = root / "research_report.md"
    holdout = metrics.loc[(metrics["period"] == "holdout") & (metrics["model"] == "ensemble_score")]
    validation = metrics.loc[(metrics["period"] == "validation") & (metrics["model"] == "ensemble_score")]
    h = holdout.iloc[0].to_dict() if not holdout.empty else {}
    v = validation.iloc[0].to_dict() if not validation.empty else {}
    ridge_h = metrics.loc[(metrics["period"] == "holdout") & (metrics["model"] == "ridge_score")]
    ridge_v = metrics.loc[(metrics["period"] == "validation") & (metrics["model"] == "ridge_score")]
    gbdt_h = metrics.loc[
        (metrics["period"] == "holdout") & (metrics["model"] == "histogram_gbdt_score")
    ]
    gbdt_v = metrics.loc[
        (metrics["period"] == "validation") & (metrics["model"] == "histogram_gbdt_score")
    ]
    rh = ridge_h.iloc[0].to_dict() if not ridge_h.empty else {}
    rv = ridge_v.iloc[0].to_dict() if not ridge_v.empty else {}
    gh = gbdt_h.iloc[0].to_dict() if not gbdt_h.empty else {}
    gv = gbdt_v.iloc[0].to_dict() if not gbdt_v.empty else {}
    gates = config.get("gates", {})
    passed_research_bar = bool(
        h.get("sharpe", np.nan) >= float(gates.get("holdout_sharpe_minimum", 0.5))
        and h.get("newey_west_tstat", np.nan) >= float(gates.get("newey_west_tstat_minimum", 2.0))
        and h.get("bootstrap_sharpe_p05", np.nan)
        >= float(gates.get("bootstrap_sharpe_p05_minimum", 0.0))
        and h.get("max_drawdown", np.nan) >= float(gates.get("maximum_drawdown_floor", -0.2))
    )
    decision = "CONTINUE" if passed_research_bar else "REJECT CURRENT STRATEGY"
    chart_links = "\n".join(f"- [{path.name}](charts/{path.name})" for path in charts)
    lines = f"""# Alpha Factor Engine — public point-in-time research run

Run ID: `{run_id}`  
Data through: `{config['research']['end']}`  
Opened chronological test: `{config['research']['holdout_start']}` onward

## Result first

**Decision: {decision}.**

The nonlinear ensemble's opened-test annualized net return is **{_pct(h.get('annualized_return', np.nan))}**, Sharpe is **{_number(h.get('sharpe', np.nan))}**, maximum drawdown is **{_pct(h.get('max_drawdown', np.nan))}**, and HAC t-stat is **{_number(h.get('newey_west_tstat', np.nan))}**. Its earlier validation Sharpe is **{_number(v.get('sharpe', np.nan))}**. These are research diagnostics for a current-ticker public universe, after {config['portfolio']['trading_cost_bps_one_way']} bps one-way trading cost and {config['portfolio']['annual_short_borrow_bps']} bps annual generic short borrow. The 2022-onward period was initially designated as a holdout, but audit-driven code corrections required reusing it; it is no longer sealed evidence.

The configured ensemble {'passes' if passed_research_bar else 'fails'} the research gate. The additive Ridge control earns **{_pct(rh.get('annualized_return', np.nan))}** with opened-test Sharpe **{_number(rh.get('sharpe', np.nan))}**, versus earlier-validation Sharpe **{_number(rv.get('sharpe', np.nan))}**. Histogram boosting moves from validation Sharpe **{_number(gv.get('sharpe', np.nan))}** to opened-test Sharpe **{_number(gh.get('sharpe', np.nan))}**. These component comparisons are diagnostic and are not a general verdict on nonlinear methods.

This run therefore {'advances the research candidate' if passed_research_bar else 'rejects promotion of the current strategy'}. Production promotion still fails because the public universe is survivorship-biased and historical borrow, delistings, institutional fundamentals, and historical constituents are absent.

## What was actually tested

- Monthly US equity cross-section using Yahoo daily bars, SEC filings known by filing/acceptance time, and lagged FRED market-regime series.
- Ten requested fundamentals: book-to-market, earnings yield, ROE, ROA, gross profitability, debt-to-equity, asset growth, sales growth, accruals, and free-cash-flow yield.
- Price characteristics centered on 12-1/6-1 momentum, reversal, liquidity, volatility, beta, drawdown, skew, and dollar volume.
- Fixed ensemble: Ridge (additive control), Extra Trees, and histogram gradient boosting. The checked-in weights and hyperparameters are fixed, but the 2022-onward period is now opened and cannot support a fresh promotion decision.
- Signals formed after the month-end close; trades enter on the next trading day's adjusted open and exit at the next rebalance open.
- Expanding chronological refits. A label is admitted to training only after its exit date.
- Cross-sectional score residualization against beta and size, dollar-neutral 50/50 long-short construction, explicit turnover, trading cost, and generic borrow cost.

## Data and point-in-time controls

- Yahoo is a public demonstration source and not a delisting-complete security master. Historical symbols are based on today's universe.
- SEC Company Facts observations become usable at EDGAR acceptance time when available; otherwise the pipeline applies a conservative next-day filing lag. Amendments remain later vintages.
- Public FRED CSVs are marked `latest_vintage`. The chosen rate/market series have limited revision risk, but revised macroeconomic series must use the ALFRED adapter and a `FRED_API_KEY`.
- MarketData.app is implemented as an alternative price adapter with physically separate `confirmed/` and `provisional/` caches. No token was present in this run, so the public study used Yahoo.
- The run also performs a small historical AAPL MarketData option-chain smoke test and stores the chain snapshot. It is excluded from strategy P&L because an end-of-day chain is not historical OPRA execution evidence.
- CRSP, Compustat PIT, I/B/E/S, TAQ, OPRA/OCC, historical borrow, and event tapes are schema-gated adapters. They were not silently substituted or simulated.

Coverage: {coverage_summary.get('tickers_modeled', 'n/a')} modeled tickers; {coverage_summary.get('prediction_months', 'n/a')} out-of-sample months; SEC factor median coverage {coverage_summary.get('median_fundamental_coverage', np.nan):.1%}, minimum factor coverage {coverage_summary.get('minimum_fundamental_coverage', np.nan):.1%}; total panel rows {coverage_summary.get('panel_rows', 'n/a')}. The low-coverage tail is visible in `factor_coverage.csv` and the coverage chart; missingness indicators are modeled explicitly, not filled with future or cross-company accounting values.

## Statistical read

Mean monthly portfolio-neutralized rank IC: **{ic_summary.get('mean_ic', np.nan):.3f}**; neutralized IC information ratio: **{ic_summary.get('ic_ir', np.nan):.2f}**; raw-score mean IC: **{ic_summary.get('raw_mean_ic', np.nan):.3f}**; positive neutralized-IC months: **{ic_summary.get('positive_ic_fraction', np.nan):.1%}**. Moving-block-bootstrap opened-test Sharpe 5/50/95 percentiles: **{h.get('bootstrap_sharpe_p05', np.nan):.2f} / {h.get('bootstrap_sharpe_p50', np.nan):.2f} / {h.get('bootstrap_sharpe_p95', np.nan):.2f}**.

Read these together with the charted cost curve. A high point estimate with a weak lower bootstrap bound, unstable rolling IC, or collapse above 25 bps is not robust alpha.

## Charts

{chart_links}

## Research basis and CogAlpha decision

- [Literature deep dive](../../docs/literature_review.md) covers the academic and industry evidence, mechanisms, replication failures, and implementation choices.
- [Liu et al., ACL 2026 (CogAlpha)](https://aclanthology.org/2026.acl-long.538.pdf) motivates using LLM code evolution only as a quarantined hypothesis generator. Its published return table is not accepted as production evidence; see the literature review for the assessment.
- Gu, Kelly, and Xiu's [NBER paper](https://www.nber.org/papers/w25398) motivates trees/neural networks through nonlinear interactions among momentum, liquidity, and volatility.
- SEC's [official EDGAR API documentation](https://www.sec.gov/search-filings/edgar-application-programming-interfaces) supports the filing and XBRL ingestion design.
- The St. Louis Fed's [ALFRED documentation](https://fred.stlouisfed.org/docs/api/fred/alfred.html) explains real-time vintages and revisions.

## Production gates

1. Replace today's ticker list with CRSP PERMNO histories, delisting returns, exchange/share-code filters, and historical index membership.
2. Replace annual SEC-tag harmonization with Compustat PIT plus audited link histories; retain EDGAR as an independent source and text/event tape.
3. Add I/B/E/S estimate revisions with announcement timestamps, TAQ/OPRA NBBO execution, OCC adjustments, historical borrow inventory/fees, and event revisions.
4. Re-run nested chronological model selection, embargoed holdout, multiple-testing controls, capacity/impact simulation, and live paper trading.
5. Permit LLM/CogAlpha proposals only in a sandboxed research queue. Require deterministic code, unit tests, economic rationale, PIT audit, and a fresh holdout before promotion.

## Machine-readable evidence

- `metrics.csv`, `monthly_portfolio_returns.csv`, `predictions.parquet`, `factor_coverage.csv`
- `data_manifest.json`, `fit_audit.csv`, `data_quality.json`, `config_snapshot.yml`
- `cost_sensitivity.csv`, `feature_importance.csv`, `monthly_ic.csv`, `deployment_decision.json`

Data-quality summary:

```json
{json.dumps(data_quality, indent=2, sort_keys=True, default=str)}
```
"""
    report_path.write_text(lines)
    return report_path
