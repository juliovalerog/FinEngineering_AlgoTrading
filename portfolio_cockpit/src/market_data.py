from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from . import portfolio_engine, storage


YAHOO_SOURCE = "Yahoo Finance"
SP500_BENCHMARK = "S&P 500"
SP500_YAHOO_TICKER = "SPY"


def _to_date(value: Any) -> date | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def get_open_position_symbols(positions: pd.DataFrame) -> list[str]:
    """Return only tickers that still have positive open quantity."""
    if positions is None or positions.empty or "symbol" not in positions.columns:
        return []
    data = positions.copy()
    data["quantity"] = pd.to_numeric(data.get("quantity"), errors="coerce").fillna(0)
    symbols = data.loc[data["quantity"] > 0, "symbol"].dropna().astype(str).str.upper().str.strip()
    return sorted(symbol for symbol in symbols.unique().tolist() if symbol)


def get_last_price_date(prices: pd.DataFrame, symbol: str) -> date | None:
    if prices is None or prices.empty or "symbol" not in prices.columns or "date" not in prices.columns:
        return None
    data = prices.copy()
    data["symbol"] = data["symbol"].astype(str).str.upper()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    filtered = data.loc[data["symbol"] == symbol.upper()].dropna(subset=["date"])
    if filtered.empty:
        return None
    return filtered["date"].max().date()


def get_last_benchmark_date(benchmark_prices: pd.DataFrame, benchmark: str = SP500_BENCHMARK) -> date | None:
    if benchmark_prices is None or benchmark_prices.empty or "benchmark" not in benchmark_prices.columns or "date" not in benchmark_prices.columns:
        return None
    data = benchmark_prices.copy()
    data["benchmark"] = data["benchmark"].astype(str)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    filtered = data.loc[data["benchmark"] == benchmark].dropna(subset=["date"])
    if filtered.empty:
        return None
    return filtered["date"].max().date()


def normalize_yahoo_prices(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance output into date, symbol, price, source rows."""
    if raw_data is None or raw_data.empty:
        return pd.DataFrame(columns=["date", "symbol", "price", "source"])

    data = raw_data.copy()
    data.index = pd.to_datetime(data.index, errors="coerce")
    data = data.loc[data.index.notna()]
    rows: list[pd.DataFrame] = []

    if isinstance(data.columns, pd.MultiIndex):
        close_level = "Close" if "Close" in data.columns.get_level_values(0) else "Adj Close"
        if close_level not in data.columns.get_level_values(0):
            return pd.DataFrame(columns=["date", "symbol", "price", "source"])
        close_data = data[close_level]
        if isinstance(close_data, pd.Series):
            symbol = str(close_data.name or raw_data.attrs.get("symbol", "")).upper()
            close_data = close_data.to_frame(symbol)
        for symbol in close_data.columns:
            series = pd.to_numeric(close_data[symbol], errors="coerce").dropna()
            if series.empty:
                continue
            rows.append(
                pd.DataFrame(
                    {
                        "date": series.index.date.astype(str),
                        "symbol": str(symbol).upper(),
                        "price": series.values,
                        "source": YAHOO_SOURCE,
                    }
                )
            )
    else:
        price_column = "Close" if "Close" in data.columns else "Adj Close" if "Adj Close" in data.columns else None
        if price_column is None:
            return pd.DataFrame(columns=["date", "symbol", "price", "source"])
        series = pd.to_numeric(data[price_column], errors="coerce").dropna()
        if series.empty:
            return pd.DataFrame(columns=["date", "symbol", "price", "source"])
        symbol = str(raw_data.attrs.get("symbol", "")).upper().strip() or "UNKNOWN"
        rows.append(
            pd.DataFrame(
                {
                    "date": series.index.date.astype(str),
                    "symbol": symbol,
                    "price": series.values,
                    "source": YAHOO_SOURCE,
                }
            )
        )

    if not rows:
        return pd.DataFrame(columns=["date", "symbol", "price", "source"])
    normalized = pd.concat(rows, ignore_index=True)
    return normalized.drop_duplicates(subset=["date", "symbol", "source"], keep="last")


def download_daily_prices_from_yahoo(symbols, start_date, end_date=None) -> pd.DataFrame:
    """Download daily Yahoo closes for explicit symbols only."""
    symbol_list = [str(symbol).upper().strip() for symbol in symbols if str(symbol).strip()]
    if not symbol_list or start_date is None:
        return pd.DataFrame(columns=["date", "symbol", "price", "source"])
    start = _to_date(start_date)
    end = _to_date(end_date) if end_date is not None else None
    if start is None:
        return pd.DataFrame(columns=["date", "symbol", "price", "source"])

    frames = []
    for symbol in symbol_list:
        raw = yf.download(
            symbol,
            start=start.isoformat(),
            end=end.isoformat() if end else None,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        raw.attrs["symbol"] = symbol
        normalized = normalize_yahoo_prices(raw)
        if not normalized.empty:
            normalized["symbol"] = symbol
            frames.append(normalized)
    if not frames:
        return pd.DataFrame(columns=["date", "symbol", "price", "source"])
    return pd.concat(frames, ignore_index=True)


def download_sp500_reference_from_yahoo(start_date, end_date=None) -> pd.DataFrame:
    raw = yf.download(
        SP500_YAHOO_TICKER,
        start=_to_date(start_date).isoformat() if _to_date(start_date) else None,
        end=_to_date(end_date).isoformat() if end_date is not None and _to_date(end_date) else None,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    raw.attrs["symbol"] = SP500_YAHOO_TICKER
    normalized = normalize_yahoo_prices(raw)
    if normalized.empty:
        return pd.DataFrame(columns=["date", "benchmark", "price", "source"])
    return normalized.assign(benchmark=SP500_BENCHMARK)[["date", "benchmark", "price", "source"]]


def _earliest_trade_date(trades: pd.DataFrame, symbol: str | None = None) -> date | None:
    if trades is None or trades.empty or "trade_date" not in trades.columns:
        return None
    data = trades.copy()
    if symbol and "symbol" in data.columns:
        data = data[data["symbol"].astype(str).str.upper() == symbol.upper()]
    dates = pd.to_datetime(data["trade_date"], errors="coerce").dropna()
    if dates.empty:
        return None
    return dates.min().date()


def _latest_price_coverage(prices: pd.DataFrame, symbols: list[str]) -> dict[str, str | None]:
    return {symbol: (last.isoformat() if (last := get_last_price_date(prices, symbol)) else None) for symbol in symbols}


def _latest_available_market_date(prices: pd.DataFrame, benchmark_prices: pd.DataFrame, symbols: list[str]) -> date | None:
    dates: list[pd.Timestamp] = []
    effective_prices = portfolio_engine.get_effective_daily_prices(prices)
    if not effective_prices.empty and symbols:
        price_dates = effective_prices[effective_prices["symbol"].isin(symbols)]["date"]
        parsed_prices = pd.to_datetime(price_dates, errors="coerce").dropna()
        if not parsed_prices.empty:
            dates.append(parsed_prices.max())
    if benchmark_prices is not None and not benchmark_prices.empty:
        parsed_benchmark = pd.to_datetime(benchmark_prices.get("date"), errors="coerce").dropna()
        if not parsed_benchmark.empty:
            dates.append(parsed_benchmark.max())
    if not dates:
        return None
    return max(dates).date()


def refresh_open_position_prices(db_path: Path | str | None = None) -> dict:
    """Refresh Yahoo daily data for open positions and the S&P 500 reference only."""
    trades = storage.get_trades(db_path)
    prices = storage.get_prices(db_path)
    benchmark_prices = storage.get_benchmark_prices(db_path)
    positions = storage.get_positions(db_path)
    snapshots = storage.get_portfolio_snapshots(db_path)
    symbols = get_open_position_symbols(positions)
    today = date.today()
    snapshot_dates = pd.to_datetime(snapshots.get("date"), errors="coerce").dropna() if snapshots is not None and not snapshots.empty else pd.Series(dtype="datetime64[ns]")
    latest_snapshot_date = snapshot_dates.max().date().isoformat() if not snapshot_dates.empty else None
    result = {
        "status": "not_started",
        "open_symbols": symbols,
        "latest_local_dates": _latest_price_coverage(prices, symbols),
        "benchmark": SP500_BENCHMARK,
        "benchmark_ticker": SP500_YAHOO_TICKER,
        "latest_benchmark_date": None,
        "rows_by_symbol": {},
        "benchmark_rows": 0,
        "failed_symbols": [],
        "messages": [],
        "prices_upserted": 0,
        "benchmark_upserted": 0,
        "previous_latest_snapshot_date": latest_snapshot_date,
        "new_latest_snapshot_date": latest_snapshot_date,
        "latest_snapshot_date_before_refresh": latest_snapshot_date,
        "latest_snapshot_date_after_refresh": latest_snapshot_date,
        "snapshots_extended": False,
        "snapshot_rows_before": 0 if snapshots is None else int(len(snapshots)),
        "snapshot_rows_after": 0 if snapshots is None else int(len(snapshots)),
    }

    downloaded_prices: list[pd.DataFrame] = []
    for symbol in symbols:
        last_date = get_last_price_date(prices, symbol)
        start = last_date + timedelta(days=1) if last_date else _earliest_trade_date(trades, symbol)
        if start is None:
            result["messages"].append(f"{symbol}: no local price or trade date available to choose a refresh start date.")
            result["rows_by_symbol"][symbol] = 0
            continue
        if start > today:
            result["messages"].append(f"{symbol}: local prices are already current through {last_date}.")
            result["rows_by_symbol"][symbol] = 0
            continue
        try:
            frame = download_daily_prices_from_yahoo([symbol], start)
        except Exception as exc:  # Yahoo/network failures should never break the app.
            result["failed_symbols"].append({"symbol": symbol, "error": str(exc)})
            result["rows_by_symbol"][symbol] = 0
            continue
        result["rows_by_symbol"][symbol] = 0 if frame.empty else int(len(frame))
        if not frame.empty:
            downloaded_prices.append(frame)

    latest_benchmark = get_last_benchmark_date(benchmark_prices, SP500_BENCHMARK)
    result["latest_benchmark_date"] = latest_benchmark.isoformat() if latest_benchmark else None
    benchmark_start = latest_benchmark + timedelta(days=1) if latest_benchmark else _earliest_trade_date(trades)
    benchmark_frame = pd.DataFrame(columns=["date", "benchmark", "price", "source"])
    if benchmark_start and benchmark_start <= today:
        try:
            benchmark_frame = download_sp500_reference_from_yahoo(benchmark_start)
        except Exception as exc:
            result["failed_symbols"].append({"symbol": SP500_YAHOO_TICKER, "error": str(exc)})
    elif benchmark_start:
        result["messages"].append(f"{SP500_BENCHMARK}: local benchmark prices are already current through {latest_benchmark}.")
    else:
        result["messages"].append(f"{SP500_BENCHMARK}: no local benchmark or trade date available to choose a refresh start date.")

    combined_prices = pd.concat(downloaded_prices, ignore_index=True) if downloaded_prices else pd.DataFrame(columns=["date", "symbol", "price", "source"])
    price_count = storage.upsert_prices(combined_prices, db_path)
    benchmark_count = storage.upsert_benchmark_prices(benchmark_frame, db_path)
    result["prices_upserted"] = price_count
    result["benchmark_upserted"] = benchmark_count
    result["benchmark_rows"] = 0 if benchmark_frame.empty else int(len(benchmark_frame))

    latest_market_date = _latest_available_market_date(storage.get_prices(db_path), storage.get_benchmark_prices(db_path), symbols)
    snapshot_needs_recompute = bool(
        latest_market_date
        and (
            not latest_snapshot_date
            or pd.to_datetime(latest_market_date) > pd.to_datetime(latest_snapshot_date)
        )
    )
    if price_count or benchmark_count or snapshot_needs_recompute:
        storage.write_audit_log(
            "MARKET_DATA_REFRESH",
            f"Yahoo Finance refresh upserted {price_count} price rows and {benchmark_count} S&P 500 rows.",
            db_path,
        )
        recomputed = storage.recompute_all_after_market_data_refresh(db_path)
        recompute_summary = recomputed.get("summary", {})
        result.update(recompute_summary)
        snapshots = recomputed.get("portfolio_snapshots")
        if snapshots is not None and not snapshots.empty:
            latest_snapshot = pd.to_datetime(snapshots["date"], errors="coerce").dropna()
            if not latest_snapshot.empty:
                result["latest_snapshot_date_after_refresh"] = latest_snapshot.max().date().isoformat()
                result["new_latest_snapshot_date"] = result["latest_snapshot_date_after_refresh"]
        result["status"] = "refreshed"
        if price_count or benchmark_count:
            result["messages"].append("Market data refresh completed and daily mark-to-market snapshots were recomputed.")
        else:
            result["messages"].append("No new Yahoo rows were inserted, but snapshots were recomputed from existing market data.")
    else:
        result["status"] = "no_update"
        if not result["messages"]:
            result["messages"].append("No Yahoo Finance updates were available for open positions or the S&P 500 reference.")
    return result
