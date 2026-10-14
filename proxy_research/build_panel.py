"""Build a common-calendar monthly price and SEC filing-time factor panel."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alpha_factor_engine.features.fundamentals import _safe_divide
from alpha_factor_engine.features.macro import build_macro_features
from alpha_factor_engine.features.price import build_monthly_price_panel
from proxy_research.acquire_prices import load_config


FACTOR_COLUMNS = [
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
]


def _statement_snapshot(state: dict[tuple[pd.Timestamp, str], float]) -> dict[str, object] | None:
    ends = sorted({end for end, concept in state if concept == "assets"})
    if not ends:
        return None
    current_end = ends[-1]
    previous_ends = [end for end in ends if end < current_end - pd.Timedelta(days=250)]
    previous_end = previous_ends[-1] if previous_ends else None

    def value(concept: str, end: pd.Timestamp | None = current_end) -> float:
        if end is None:
            return np.nan
        return float(state.get((end, concept), np.nan))

    assets = value("assets")
    equity = value("equity")
    revenue = value("revenue")
    net_income = value("net_income")
    gross_profit = value("gross_profit")
    cfo = value("operating_cash_flow")
    capex = value("capital_expenditure")
    shares = value("diluted_shares")
    debt = value("debt")
    debt_definition = "reported_long_term_debt"
    if not np.isfinite(debt):
        debt = value("liabilities")
        debt_definition = "total_liabilities_fallback"
    prior_assets = value("assets", previous_end)
    prior_equity = value("equity", previous_end)
    prior_revenue = value("revenue", previous_end)
    average_assets = np.nanmean([assets, prior_assets]) if np.isfinite(prior_assets) else assets
    average_equity = np.nanmean([equity, prior_equity]) if np.isfinite(prior_equity) else equity
    return {
        "fundamental_fiscal_end": current_end,
        "assets": assets,
        "equity": equity,
        "revenue": revenue,
        "net_income": net_income,
        "gross_profit": gross_profit,
        "operating_cash_flow": cfo,
        "capital_expenditure": capex,
        "diluted_shares": shares,
        "debt": debt,
        "debt_definition": debt_definition,
        "average_assets": average_assets,
        "average_equity": average_equity,
        "prior_assets": prior_assets,
        "prior_revenue": prior_revenue,
    }


def build_fast_fundamental_features(
    price_panel: pd.DataFrame,
    concepts: pd.DataFrame,
    daily_prices: pd.DataFrame,
) -> pd.DataFrame:
    concept_frame = concepts.copy()
    concept_frame["available_at"] = pd.to_datetime(concept_frame["available_at"], utc=True)
    concept_frame["fiscal_end"] = pd.to_datetime(concept_frame["fiscal_end"])
    price_groups = {ticker: group for ticker, group in daily_prices.groupby("ticker", sort=False)}
    outputs: list[pd.DataFrame] = []
    for ticker, decisions in price_panel.groupby("ticker", sort=True):
        company = concept_frame.loc[concept_frame["ticker"].eq(ticker)].sort_values("available_at")
        snapshots: list[dict[str, object]] = []
        state: dict[tuple[pd.Timestamp, str], float] = {}
        for available_at, updates in company.groupby("available_at", sort=True):
            for row in updates.itertuples(index=False):
                state[(pd.Timestamp(row.fiscal_end), str(row.concept))] = float(row.value)
            snapshot = _statement_snapshot(state)
            if snapshot is not None:
                snapshots.append(
                    {"fundamental_available_at": available_at, **snapshot}
                )
        base = decisions[["ticker", "signal_date", "date", "pit_close"]].sort_values("signal_date")
        if snapshots:
            snapshot_frame = pd.DataFrame(snapshots).sort_values("fundamental_available_at")
            joined = pd.merge_asof(
                base,
                snapshot_frame,
                left_on="signal_date",
                right_on="fundamental_available_at",
                direction="backward",
            )
        else:
            joined = base.copy()
            joined["fundamental_available_at"] = pd.NaT
            joined["fundamental_fiscal_end"] = pd.NaT
            for column in [
                "assets", "equity", "revenue", "net_income", "gross_profit",
                "operating_cash_flow", "capital_expenditure", "diluted_shares",
                "debt", "average_assets", "average_equity", "prior_assets",
                "prior_revenue",
            ]:
                joined[column] = np.nan
            joined["debt_definition"] = None

        split_events = price_groups[ticker].loc[
            price_groups[ticker]["split_ratio"].ne(1.0), ["date", "split_ratio"]
        ].copy() if ticker in price_groups else pd.DataFrame(columns=["date", "split_ratio"])
        split_events["date"] = pd.to_datetime(split_events["date"])
        market_caps = []
        for row in joined.itertuples(index=False):
            if pd.isna(row.fundamental_fiscal_end) or not np.isfinite(row.diluted_shares):
                market_caps.append(np.nan)
                continue
            split_factor = split_events.loc[
                split_events["date"].gt(row.fundamental_fiscal_end)
                & split_events["date"].le(pd.Timestamp(row.date)),
                "split_ratio",
            ].prod()
            split_factor = float(split_factor) if pd.notna(split_factor) else 1.0
            market_caps.append(float(row.pit_close) * float(row.diluted_shares) * split_factor)
        joined["market_cap"] = market_caps
        joined["log_market_cap"] = np.log(joined["market_cap"].where(joined["market_cap"].gt(0)))
        joined["fundamental_age_days"] = (
            pd.to_datetime(joined["signal_date"], utc=True)
            - pd.to_datetime(joined["fundamental_available_at"], utc=True)
        ).dt.days
        joined["book_to_market"] = joined.apply(
            lambda row: _safe_divide(row["equity"], row["market_cap"]), axis=1
        )
        joined["earnings_yield"] = joined.apply(
            lambda row: _safe_divide(row["net_income"], row["market_cap"]), axis=1
        )
        joined["roe"] = joined.apply(
            lambda row: _safe_divide(row["net_income"], row["average_equity"]), axis=1
        )
        joined["roa"] = joined.apply(
            lambda row: _safe_divide(row["net_income"], row["average_assets"]), axis=1
        )
        joined["gross_profitability"] = joined.apply(
            lambda row: _safe_divide(row["gross_profit"], row["average_assets"]), axis=1
        )
        joined["debt_to_equity"] = joined.apply(
            lambda row: _safe_divide(row["debt"], row["equity"]), axis=1
        )
        joined["asset_growth"] = np.where(
            joined["prior_assets"].gt(0), joined["assets"] / joined["prior_assets"] - 1, np.nan
        )
        joined["sales_growth"] = np.where(
            joined["prior_revenue"].gt(0), joined["revenue"] / joined["prior_revenue"] - 1, np.nan
        )
        joined["accruals"] = (
            joined["net_income"] - joined["operating_cash_flow"]
        ) / joined["average_assets"].replace(0, np.nan)
        joined["free_cash_flow_yield"] = (
            joined["operating_cash_flow"] - joined["capital_expenditure"]
        ) / joined["market_cap"].replace(0, np.nan)
        outputs.append(
            joined[
                [
                    "ticker", "signal_date", "fundamental_available_at",
                    "fundamental_fiscal_end", "fundamental_age_days", "log_market_cap",
                    "debt_definition", *FACTOR_COLUMNS,
                ]
            ]
        )
    result = pd.concat(outputs, ignore_index=True)
    populated = result[FACTOR_COLUMNS].notna().any(axis=1)
    future = pd.to_datetime(result["fundamental_available_at"], utc=True).gt(
        pd.to_datetime(result["signal_date"], utc=True)
    )
    if (populated & future).any():
        raise RuntimeError("future SEC fundamental leaked into proxy panel")
    return result


def align_to_common_month_end(
    panel: pd.DataFrame,
    daily_prices: pd.DataFrame,
    *,
    benchmark: str,
) -> pd.DataFrame:
    output = panel.copy()
    output["signal_month"] = pd.to_datetime(output["date"]).dt.to_period("M")
    benchmark_dates = daily_prices.loc[daily_prices["ticker"].eq(benchmark), ["date"]].copy()
    benchmark_dates["signal_month"] = pd.to_datetime(benchmark_dates["date"]).dt.to_period("M")
    calendar = benchmark_dates.groupby("signal_month")["date"].max().rename("common_signal_date")
    output = output.merge(calendar, on="signal_month", how="left", validate="many_to_one")
    output["stale_calendar_days"] = (
        pd.to_datetime(output["common_signal_date"]) - pd.to_datetime(output["date"])
    ).dt.days
    output["eligible"] = output["eligible"].fillna(False) & output["stale_calendar_days"].eq(0)
    local = pd.to_datetime(output["common_signal_date"]).dt.tz_localize(
        "America/New_York", nonexistent="raise", ambiguous="raise"
    )
    output["signal_date"] = (local + pd.Timedelta(hours=20)).dt.tz_convert("UTC")
    output["signal_at"] = output["signal_date"]
    return output.sort_values(["signal_date", "ticker"]).reset_index(drop=True)


def build_proxy_panel(config_path: str | Path) -> Path:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    root = config_path.parent.parent
    data_dir = root / config["data"]["output_dir"] / "data"
    daily = pd.read_parquet(data_dir / "daily_prices.parquet")
    price_panel_path = data_dir / "monthly_price_panel.parquet"
    if price_panel_path.is_file():
        price_panel = pd.read_parquet(price_panel_path)
    else:
        price_panel = build_monthly_price_panel(
            daily,
            benchmark=config["data"]["benchmark"],
            minimum_price=float(config["universe"]["minimum_price"]),
            minimum_adv_usd=float(config["universe"]["minimum_adv_usd"]),
        )
    price_panel = align_to_common_month_end(
        price_panel, daily, benchmark=config["data"]["benchmark"]
    )
    concepts = pd.read_parquet(data_dir / "sec_annual_concepts.parquet")
    fundamentals = build_fast_fundamental_features(price_panel, concepts, daily)
    panel = price_panel.merge(
        fundamentals, on=["ticker", "signal_date"], how="left", validate="one_to_one"
    )
    macro_sources = {
        "fed_funds": root / "artifacts/public_research/data/fred_DFF.parquet",
        "yield_curve": root / "artifacts/public_research/data/fred_T10Y2Y.parquet",
        "vix": root / "artifacts/public_research/data/fred_VIXCLS.parquet",
        "baa_yield": root / "artifacts/public_research/data/fred_DBAA.parquet",
    }
    if all(path.is_file() for path in macro_sources.values()):
        macro = build_macro_features(
            panel[["signal_date"]],
            {name: pd.read_parquet(path) for name, path in macro_sources.items()},
        )
        panel = panel.merge(macro, on="signal_date", how="left", validate="many_to_one")
    output_path = data_dir / "monthly_feature_panel.parquet"
    panel.to_parquet(output_path, index=False)
    return output_path


__all__ = [
    "FACTOR_COLUMNS",
    "align_to_common_month_end",
    "build_fast_fundamental_features",
    "build_proxy_panel",
]

