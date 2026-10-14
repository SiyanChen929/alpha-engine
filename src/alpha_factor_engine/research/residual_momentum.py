"""Leakage-safe market-residual momentum signals from daily price history.

The implementation is an explicitly labelled public-data proxy for the
three-factor residual momentum construction in Blitz, Huij, and Martens
(2011).  It estimates one market beta at each decision from trailing data,
excludes the regression intercept from the signal, sums residual returns over
the exact 12-minus-1 formation interval, and scales by residual volatility.
No realized portfolio return is accepted by this module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_DAILY_COLUMNS = {"ticker", "date", "adj_close", "available_at"}
REQUIRED_DECISION_COLUMNS = {"formation_date", "signal_at"}


def _normalize_inputs(
    daily_prices: pd.DataFrame,
    decisions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing_daily = sorted(REQUIRED_DAILY_COLUMNS.difference(daily_prices.columns))
    missing_decisions = sorted(REQUIRED_DECISION_COLUMNS.difference(decisions.columns))
    if missing_daily:
        raise ValueError(f"daily_prices missing required columns: {missing_daily}")
    if missing_decisions:
        raise ValueError(f"decisions missing required columns: {missing_decisions}")

    daily = daily_prices.loc[:, sorted(REQUIRED_DAILY_COLUMNS)].copy()
    daily["ticker"] = daily["ticker"].astype(str)
    daily["date"] = pd.to_datetime(daily["date"], errors="raise").dt.normalize()
    daily["adj_close"] = pd.to_numeric(daily["adj_close"], errors="coerce")
    daily["available_at"] = pd.to_datetime(daily["available_at"], utc=True, errors="raise")
    if daily.duplicated(["ticker", "date"]).any():
        raise ValueError("daily_prices contains duplicate ticker/date rows")
    if (~np.isfinite(daily["adj_close"]) | daily["adj_close"].le(0)).any():
        raise ValueError("daily_prices contains non-positive or non-finite adjusted closes")

    decision = decisions.loc[:, sorted(REQUIRED_DECISION_COLUMNS)].copy()
    decision["formation_date"] = pd.to_datetime(
        decision["formation_date"], errors="raise"
    ).dt.normalize()
    decision["signal_at"] = pd.to_datetime(decision["signal_at"], utc=True, errors="raise")
    if decision.duplicated("formation_date").any():
        raise ValueError("decisions contains duplicate formation dates")
    if not decision["formation_date"].is_monotonic_increasing:
        decision = decision.sort_values("formation_date").reset_index(drop=True)
    return daily, decision


def _rolling_market_beta(
    stock_returns: np.ndarray,
    market_returns: np.ndarray,
    *,
    minimum_observations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return column-wise OLS slopes with intercept and paired counts."""

    market = market_returns.reshape(-1, 1)
    valid = np.isfinite(stock_returns) & np.isfinite(market)
    count = valid.sum(axis=0).astype(int)
    stock = np.where(valid, stock_returns, 0.0)
    factor = np.where(valid, market, 0.0)
    count_safe = np.where(count > 0, count, 1)
    stock_sum = stock.sum(axis=0)
    factor_sum = factor.sum(axis=0)
    cross_sum = (stock * factor).sum(axis=0)
    factor_square_sum = np.square(factor).sum(axis=0)
    covariance_numerator = cross_sum - stock_sum * factor_sum / count_safe
    variance_numerator = factor_square_sum - np.square(factor_sum) / count_safe
    beta = np.divide(
        covariance_numerator,
        variance_numerator,
        out=np.full(stock_returns.shape[1], np.nan, dtype=float),
        where=(count >= minimum_observations) & (variance_numerator > 1e-18),
    )
    return beta, count


def compute_market_residual_momentum(
    daily_prices: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    benchmark_ticker: str = "SPY",
    beta_window_sessions: int = 756,
    minimum_beta_observations: int = 504,
    formation_sessions: int = 253,
    skip_sessions: int = 21,
    minimum_formation_observations: int = 220,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute an auditable market-residual 12-minus-1 momentum proxy.

    Prices are aligned to the benchmark trading calendar.  At each signal, the
    market beta uses at most ``beta_window_sessions`` ending at the one-month
    endpoint
    close.  The score uses returns strictly after the 12-month endpoint and up
    to the one-month endpoint.  A constant signal-time beta is applied to that
    formation interval.  The OLS intercept is deliberately excluded, matching
    the paper's signal convention.
    """

    if beta_window_sessions < 2:
        raise ValueError("beta_window_sessions must be at least two")
    if not 2 <= minimum_beta_observations <= beta_window_sessions:
        raise ValueError("minimum_beta_observations must be in [2, beta_window_sessions]")
    if formation_sessions <= skip_sessions or skip_sessions <= 0:
        raise ValueError("formation_sessions must exceed a positive skip_sessions")
    formation_return_count = formation_sessions - skip_sessions
    if not 2 <= minimum_formation_observations <= formation_return_count:
        raise ValueError(
            "minimum_formation_observations must be in [2, formation_sessions-skip_sessions]"
        )

    daily, decision = _normalize_inputs(daily_prices, decisions)
    benchmark = daily.loc[daily["ticker"].eq(benchmark_ticker)].sort_values("date")
    if benchmark.empty:
        raise ValueError(f"benchmark {benchmark_ticker!r} is absent from daily_prices")
    calendar = pd.DatetimeIndex(benchmark["date"].drop_duplicates())
    if len(calendar) != len(benchmark):
        raise ValueError("benchmark has duplicate calendar dates")
    calendar_positions = {value: index for index, value in enumerate(calendar)}

    absent_decisions = sorted(
        value for value in decision["formation_date"] if value not in calendar_positions
    )
    if absent_decisions:
        raise ValueError(
            f"formation dates absent from benchmark calendar: {absent_decisions[:5]}"
        )

    # Every price used by a month-end decision must have been observable by
    # that decision.  The window-level audit below catches delayed corrections
    # as well as ordinary future rows.
    availability_by_date = daily.groupby("date", sort=True)["available_at"].max()
    close_matrix = (
        daily.pivot(index="date", columns="ticker", values="adj_close")
        .reindex(calendar)
        .sort_index(axis=1)
    )
    if benchmark_ticker not in close_matrix:
        raise ValueError(f"benchmark {benchmark_ticker!r} disappeared during pivot")
    return_matrix = close_matrix.pct_change(fill_method=None)
    market_returns = return_matrix[benchmark_ticker].to_numpy(float)
    stock_tickers = [ticker for ticker in close_matrix.columns if ticker != benchmark_ticker]
    stock_returns = return_matrix[stock_tickers].to_numpy(float)
    stock_closes = close_matrix[stock_tickers].to_numpy(float)

    score_parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    for row in decision.itertuples(index=False):
        formation_date = pd.Timestamp(row.formation_date)
        signal_at = pd.Timestamp(row.signal_at)
        position = calendar_positions[formation_date]
        start_position = position - formation_sessions
        end_position = position - skip_sessions
        beta_start_position = max(0, end_position - beta_window_sessions + 1)
        if start_position < 0:
            audit_rows.append(
                {
                    "formation_date": formation_date,
                    "signal_at": signal_at,
                    "status": "insufficient_calendar_history",
                    "scored_names": 0,
                }
            )
            continue

        used_start = calendar[min(start_position + 1, beta_start_position)]
        used_end = calendar[end_position]
        used_availability = availability_by_date.loc[used_start:used_end]
        latest_input_available_at = used_availability.max()
        if pd.isna(latest_input_available_at) or latest_input_available_at > signal_at:
            raise ValueError(
                "price availability exceeds decision clock at "
                f"{formation_date.date()}: latest={latest_input_available_at}, signal={signal_at}"
            )

        beta_slice = stock_returns[beta_start_position : end_position + 1, :]
        market_beta_slice = market_returns[beta_start_position : end_position + 1]
        beta, beta_count = _rolling_market_beta(
            beta_slice,
            market_beta_slice,
            minimum_observations=minimum_beta_observations,
        )

        # P[end] / P[start] contains returns (start, end], exactly matching the
        # existing 253/21-session total-momentum endpoints.
        formation_slice = stock_returns[start_position + 1 : end_position + 1, :]
        market_formation = market_returns[start_position + 1 : end_position + 1]
        residual = formation_slice - market_formation.reshape(-1, 1) * beta.reshape(1, -1)
        residual_count = np.isfinite(residual).sum(axis=0).astype(int)
        residual_sum = np.nansum(residual, axis=0)
        residual_volatility = np.nanstd(residual, axis=0, ddof=1)
        exact_start = stock_closes[start_position, :]
        exact_end = stock_closes[end_position, :]
        exact_total_momentum = exact_end / exact_start - 1.0
        score = np.divide(
            residual_sum,
            residual_volatility,
            out=np.full(len(stock_tickers), np.nan, dtype=float),
            where=(
                (beta_count >= minimum_beta_observations)
                & (residual_count >= minimum_formation_observations)
                & np.isfinite(exact_total_momentum)
                & (residual_volatility > 1e-12)
            ),
        )
        valid = np.isfinite(score)
        score_parts.append(
            pd.DataFrame(
                {
                    "formation_date": formation_date,
                    "signal_at": signal_at,
                    "ticker": np.asarray(stock_tickers, dtype=object)[valid],
                    "market_beta_signal": beta[valid],
                    "beta_observations": beta_count[valid],
                    "formation_observations": residual_count[valid],
                    "residual_return_sum_12_1": residual_sum[valid],
                    "residual_volatility_daily_12_1": residual_volatility[valid],
                    "residual_momentum_score": score[valid],
                    "exact_total_momentum_12_1": exact_total_momentum[valid],
                    "latest_input_available_at": latest_input_available_at,
                }
            )
        )
        audit_rows.append(
            {
                "formation_date": formation_date,
                "signal_at": signal_at,
                "status": "scored",
                "calendar_position": position,
                "beta_window_start": calendar[beta_start_position],
                "formation_start": calendar[start_position],
                "formation_end": calendar[end_position],
                "latest_input_available_at": latest_input_available_at,
                "candidate_names": len(stock_tickers),
                "scored_names": int(valid.sum()),
                "minimum_beta_observations": int(beta_count[valid].min()) if valid.any() else 0,
                "minimum_formation_observations": (
                    int(residual_count[valid].min()) if valid.any() else 0
                ),
            }
        )

    if not score_parts:
        raise RuntimeError("residual momentum produced no scored decision dates")
    scores = pd.concat(score_parts, ignore_index=True).sort_values(
        ["signal_at", "ticker"]
    ).reset_index(drop=True)
    if scores.duplicated(["formation_date", "ticker"]).any():
        raise RuntimeError("residual momentum produced duplicate formation-date/ticker rows")
    return scores, pd.DataFrame(audit_rows)


def neutralize_cross_sectional_score(
    frame: pd.DataFrame,
    *,
    score_column: str,
    exposure_columns: list[str],
    group_column: str = "signal_date",
    output_column: str = "neutralized_residual_momentum_score",
) -> pd.DataFrame:
    """Residualize a score against contemporaneously observable exposures."""

    required = {group_column, score_column, *exposure_columns}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"neutralization frame missing columns: {missing}")
    output = frame.copy()
    output[output_column] = np.nan
    for _, group in output.groupby(group_column, sort=True):
        columns = [score_column, *exposure_columns]
        valid = group[columns].apply(pd.to_numeric, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        if len(valid) < len(exposure_columns) + 10:
            continue
        ranked = valid.rank(method="average", pct=True)
        design = np.column_stack(
            [np.ones(len(ranked)), ranked[exposure_columns].to_numpy(float)]
        )
        response = ranked[score_column].to_numpy(float)
        coefficients = np.linalg.lstsq(design, response, rcond=None)[0]
        residual = response - design @ coefficients
        output.loc[valid.index, output_column] = residual
    return output


__all__ = ["compute_market_residual_momentum", "neutralize_cross_sectional_score"]
