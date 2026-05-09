from __future__ import annotations

import numpy as np
import pandas as pd


def _clean_numeric_series(series: pd.Series | list[float] | tuple[float, ...]) -> pd.Series:
    """Return finite numeric observations only.

    The cockpit treats unavailable metrics as ``np.nan``. This keeps the UI and
    tests explicit when a metric is not economically meaningful, for example a
    Sharpe ratio with zero volatility.
    """
    cleaned = pd.Series(series, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    return cleaned


def daily_returns(series: pd.Series | list[float] | tuple[float, ...]) -> pd.Series:
    """Compute simple daily returns from a value or price series."""
    values = _clean_numeric_series(series)
    if values.size < 2:
        return pd.Series(dtype="float64")
    returns = values.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    return returns.astype("float64")


def cumulative_return(series: pd.Series | list[float] | tuple[float, ...]) -> float:
    """Return total return from first valid observation to last valid observation."""
    values = _clean_numeric_series(series)
    if values.size < 2 or values.iloc[0] == 0:
        return np.nan
    return float(values.iloc[-1] / values.iloc[0] - 1)


def annualized_return(returns: pd.Series | list[float] | tuple[float, ...], periods: int = 252) -> float:
    """Annualize a stream of periodic returns using geometric compounding."""
    cleaned = _clean_numeric_series(returns)
    if cleaned.empty:
        return np.nan
    compounded = float((1 + cleaned).prod())
    if compounded <= 0:
        return np.nan
    return float(compounded ** (periods / cleaned.size) - 1)


def annualized_volatility(returns: pd.Series | list[float] | tuple[float, ...], periods: int = 252) -> float:
    """Annualize sample volatility of periodic returns."""
    cleaned = _clean_numeric_series(returns)
    if cleaned.size < 2:
        return np.nan
    volatility = float(cleaned.std(ddof=1))
    if volatility == 0:
        return 0.0
    return float(volatility * np.sqrt(periods))


def sharpe_ratio(
    returns: pd.Series | list[float] | tuple[float, ...],
    risk_free_rate: float = 0,
    periods: int = 252,
) -> float:
    """Compute annualized Sharpe ratio.

    ``risk_free_rate`` is an annual rate. For classroom simplicity it defaults
    to zero and is converted to a per-period rate before excess returns are
    measured.
    """
    cleaned = _clean_numeric_series(returns)
    if cleaned.size < 2:
        return np.nan
    periodic_rf = risk_free_rate / periods
    excess = cleaned - periodic_rf
    std = float(excess.std(ddof=1))
    if std == 0 or np.isnan(std):
        return np.nan
    return float((excess.mean() / std) * np.sqrt(periods))


def sortino_ratio(
    returns: pd.Series | list[float] | tuple[float, ...],
    risk_free_rate: float = 0,
    periods: int = 252,
) -> float:
    """Compute annualized Sortino ratio using downside deviation below RF."""
    cleaned = _clean_numeric_series(returns)
    if cleaned.size < 2:
        return np.nan
    periodic_rf = risk_free_rate / periods
    excess = cleaned - periodic_rf
    downside = excess[excess < 0]
    if downside.empty:
        return np.nan
    downside_deviation = float(np.sqrt((downside**2).mean()))
    if downside_deviation == 0 or np.isnan(downside_deviation):
        return np.nan
    return float((excess.mean() / downside_deviation) * np.sqrt(periods))


def max_drawdown(value_series: pd.Series | list[float] | tuple[float, ...]) -> float:
    """Compute maximum drawdown as the worst peak-to-trough percentage loss."""
    values = _clean_numeric_series(value_series)
    if values.size < 2:
        return np.nan
    running_max = values.cummax()
    drawdowns = values / running_max - 1
    return float(drawdowns.min())


def concentration_metrics(weights: pd.Series | list[float] | tuple[float, ...]) -> dict[str, float]:
    """Compute simple portfolio concentration indicators from position weights."""
    cleaned = _clean_numeric_series(weights).abs()
    if cleaned.empty:
        return {
            "largest_weight": np.nan,
            "top_5_weight": np.nan,
            "herfindahl_index": np.nan,
            "effective_number_positions": np.nan,
        }
    sorted_weights = cleaned.sort_values(ascending=False)
    hhi = float((cleaned**2).sum())
    return {
        "largest_weight": float(sorted_weights.iloc[0]),
        "top_5_weight": float(sorted_weights.head(5).sum()),
        "herfindahl_index": hhi,
        "effective_number_positions": float(1 / hhi) if hhi > 0 else np.nan,
    }


def _aligned_returns(
    portfolio_returns: pd.Series | list[float] | tuple[float, ...],
    benchmark_returns: pd.Series | list[float] | tuple[float, ...],
) -> tuple[pd.Series, pd.Series]:
    """Align portfolio and benchmark return observations by index and remove missing values."""
    aligned = pd.concat(
        [
            pd.Series(portfolio_returns, dtype="float64").rename("portfolio"),
            pd.Series(benchmark_returns, dtype="float64").rename("benchmark"),
        ],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if aligned.empty:
        return pd.Series(dtype="float64"), pd.Series(dtype="float64")
    return aligned["portfolio"], aligned["benchmark"]


def beta_vs_benchmark(
    portfolio_returns: pd.Series | list[float] | tuple[float, ...],
    benchmark_returns: pd.Series | list[float] | tuple[float, ...],
) -> float:
    """Compute beta as covariance with benchmark divided by benchmark variance."""
    portfolio, benchmark = _aligned_returns(portfolio_returns, benchmark_returns)
    if portfolio.size < 2:
        return np.nan
    benchmark_variance = float(benchmark.var(ddof=1))
    if benchmark_variance == 0 or np.isnan(benchmark_variance):
        return np.nan
    covariance = float(portfolio.cov(benchmark))
    return float(covariance / benchmark_variance)


def tracking_error(
    portfolio_returns: pd.Series | list[float] | tuple[float, ...],
    benchmark_returns: pd.Series | list[float] | tuple[float, ...],
    periods: int = 252,
) -> float:
    """Annualized volatility of active returns versus the benchmark."""
    portfolio, benchmark = _aligned_returns(portfolio_returns, benchmark_returns)
    if portfolio.size < 2:
        return np.nan
    active = portfolio - benchmark
    active_volatility = float(active.std(ddof=1))
    if np.isnan(active_volatility):
        return np.nan
    return float(active_volatility * np.sqrt(periods))


def information_ratio(
    portfolio_returns: pd.Series | list[float] | tuple[float, ...],
    benchmark_returns: pd.Series | list[float] | tuple[float, ...],
    periods: int = 252,
) -> float:
    """Annualized active return divided by annualized tracking error."""
    portfolio, benchmark = _aligned_returns(portfolio_returns, benchmark_returns)
    if portfolio.size < 2:
        return np.nan
    active = portfolio - benchmark
    te = tracking_error(portfolio, benchmark, periods=periods)
    if te == 0 or np.isnan(te):
        return np.nan
    annualized_active_return = float(active.mean() * periods)
    return float(annualized_active_return / te)


def turnover_proxy(trades: pd.DataFrame, average_portfolio_value: float | None) -> float:
    """Estimate simple turnover as gross traded value divided by average portfolio value."""
    if trades is None or trades.empty or average_portfolio_value is None or average_portfolio_value == 0:
        return np.nan
    amounts = pd.to_numeric(trades.get("amount"), errors="coerce").abs().dropna()
    if amounts.empty:
        return np.nan
    return float(amounts.sum() / average_portfolio_value)
