from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import portfolio_engine


def _effective_prices_for_validation(prices: pd.DataFrame) -> pd.DataFrame:
    if hasattr(portfolio_engine, "get_effective_daily_prices"):
        return portfolio_engine.get_effective_daily_prices(prices)
    columns = ["date", "symbol", "price", "source"]
    data = prices.copy()
    for column in columns:
        if column not in data.columns:
            data[column] = None
    data = data[columns]
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.date.astype(str)
    data["symbol"] = data["symbol"].astype(str).str.upper().str.strip()
    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    return data.dropna(subset=["date", "symbol", "price"]).drop_duplicates(subset=["date", "symbol"], keep="last")


def _issue(
    severity: str,
    check_name: str,
    message: str,
    affected_rows: list[Any] | None = None,
    recommendation: str = "",
    affected_symbols: list[Any] | None = None,
) -> dict[str, Any]:
    return {
        "severity": severity,
        "check_name": check_name,
        "message": message,
        "affected_rows": affected_rows or [],
        "affected_symbols": affected_symbols or [],
        "recommendation": recommendation,
    }


def _row_ids(frame: pd.DataFrame) -> list[Any]:
    if frame.empty:
        return []
    if "source_row" in frame.columns:
        return frame["source_row"].dropna().astype(int).tolist()
    if "trade_id" in frame.columns:
        return frame["trade_id"].dropna().tolist()
    return frame.index.tolist()


def validate_manual_trade(trade: dict[str, Any], current_positions: pd.DataFrame) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for field in ["symbol", "side", "trade_date", "quantity", "price"]:
        if trade.get(field) in (None, ""):
            issues.append(
                _issue(
                    "Error",
                    "manual_trade_missing_field",
                    f"Manual trade is missing {field}.",
                    recommendation="Complete the trade ticket before inserting it into SQLite.",
                )
            )
    try:
        quantity = float(trade.get("quantity", 0))
        price = float(trade.get("price", 0))
    except (TypeError, ValueError):
        quantity = 0
        price = 0
    if quantity <= 0 or price <= 0:
        issues.append(
            _issue(
                "Error",
                "manual_trade_invalid_size",
                "Manual trade quantity and price must be positive.",
                recommendation="Correct the operational trade ticket before recalculating the portfolio.",
            )
        )

    if str(trade.get("side", "")).upper() == "SELL" and current_positions is not None and not current_positions.empty:
        symbol = str(trade.get("symbol", "")).upper()
        available = current_positions.loc[current_positions["symbol"].astype(str).str.upper() == symbol, "quantity"]
        available_quantity = float(available.sum()) if not available.empty else 0.0
        if quantity > available_quantity + 1e-9:
            issues.append(
                _issue(
                    "Error",
                    "manual_trade_negative_position",
                    f"Sell quantity {quantity:,.2f} exceeds current available position {available_quantity:,.2f} for {symbol}.",
                    recommendation="Check whether the trade is a data error or should be explicitly modeled as a short position.",
                )
            )
    return issues


def run_data_quality_checks(
    trades: pd.DataFrame,
    positions: pd.DataFrame,
    portfolio_snapshots: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    trades = trades.copy() if trades is not None else pd.DataFrame()

    if trades.empty:
        return [
            _issue(
                "Error",
                "empty_trade_ledger",
                "No trades are available in SQLite.",
                recommendation="Reset the demo database from Excel and check that the TRACK sheet is present.",
            )
        ]

    for column, severity, label in [
        ("symbol", "Error", "missing_symbols"),
        ("trade_date", "Error", "missing_dates"),
        ("price", "Error", "missing_prices"),
        ("quantity", "Error", "missing_quantities"),
    ]:
        missing = trades[trades[column].isna() | (trades[column].astype(str).str.strip() == "")]
        if not missing.empty:
            issues.append(
                _issue(
                    severity,
                    label,
                    f"{len(missing)} trade rows are missing {column}.",
                    _row_ids(missing),
                    "Repair the raw Excel row or add a controlled correction in SQLite.",
                    missing.get("symbol", pd.Series(dtype=str)).dropna().astype(str).unique().tolist(),
                )
            )

    duplicate_subset = ["symbol", "side", "trade_date", "quantity", "price"]
    duplicates = trades[trades.duplicated(subset=duplicate_subset, keep=False)]
    if not duplicates.empty:
        issues.append(
                _issue(
                    "Warning",
                    "duplicate_like_trades",
                    f"{len(duplicates)} trades look duplicated based on symbol, side, date, quantity and price.",
                    _row_ids(duplicates),
                    "Confirm whether these are legitimate split executions or accidental duplicate imports.",
                    duplicates.get("symbol", pd.Series(dtype=str)).dropna().astype(str).unique().tolist(),
                )
            )

    missing_sectors = trades[
        trades["sector"].isna()
        | (trades["sector"].astype(str).str.strip() == "")
    ]
    if not missing_sectors.empty:
        issues.append(
                _issue(
                    "Warning",
                    "missing_sectors",
                    f"{len(missing_sectors)} trades do not have a sector classification.",
                    _row_ids(missing_sectors),
                    "Add sector mapping before using sector allocation for investment decisions.",
                    missing_sectors.get("symbol", pd.Series(dtype=str)).dropna().astype(str).unique().tolist(),
                )
            )

    numeric = trades.copy()
    numeric["quantity"] = pd.to_numeric(numeric["quantity"], errors="coerce")
    numeric["price"] = pd.to_numeric(numeric["price"], errors="coerce")
    numeric["amount"] = pd.to_numeric(numeric["amount"], errors="coerce")
    amount_check = numeric.dropna(subset=["quantity", "price", "amount"]).copy()
    amount_check["expected_amount"] = amount_check["quantity"] * amount_check["price"]
    inconsistent = amount_check[
        (amount_check["amount"] - amount_check["expected_amount"]).abs()
        > np.maximum(1.0, amount_check["expected_amount"].abs() * 0.01)
    ]
    if not inconsistent.empty:
        issues.append(
                _issue(
                    "Warning",
                    "inconsistent_amount",
                    f"{len(inconsistent)} trades have amount materially different from quantity times price.",
                    _row_ids(inconsistent),
                    "Reconcile execution value, fees and manual Excel formulas.",
                    inconsistent.get("symbol", pd.Series(dtype=str)).dropna().astype(str).unique().tolist(),
                )
            )

    negative_sell_rows: list[Any] = []
    running: dict[str, float] = {}
    ordered = trades.copy()
    ordered["trade_date"] = pd.to_datetime(ordered["trade_date"], errors="coerce")
    ordered["quantity"] = pd.to_numeric(ordered["quantity"], errors="coerce")
    ordered = ordered.sort_values(["trade_date", "side"])
    for _, row in ordered.iterrows():
        symbol = str(row.get("symbol", "")).upper()
        quantity = float(row.get("quantity") or 0)
        if row.get("side") == "BUY":
            running[symbol] = running.get(symbol, 0.0) + quantity
        elif row.get("side") == "SELL":
            if quantity > running.get(symbol, 0.0) + 1e-9:
                negative_sell_rows.append(row.get("source_row", row.get("trade_id")))
            running[symbol] = max(0.0, running.get(symbol, 0.0) - quantity)
    if negative_sell_rows:
        issues.append(
            _issue(
                "Error",
                "sell_without_enough_position",
                f"{len(negative_sell_rows)} sell operations exceed the available position in the normalized ledger.",
                negative_sell_rows,
                "Check symbol mapping and whether any short sale should be modeled explicitly.",
            )
        )

    if positions is not None and not positions.empty:
        negative_positions = positions[pd.to_numeric(positions["quantity"], errors="coerce") < -1e-9]
        if not negative_positions.empty:
            issues.append(
                _issue(
                    "Error",
                "negative_positions",
                f"{len(negative_positions)} positions are negative.",
                negative_positions["symbol"].tolist(),
                "Prevent silent short positions unless the class case explicitly allows them.",
                negative_positions["symbol"].tolist(),
            )
        )

    if prices is None or prices.empty:
        issues.append(
            _issue(
                "Warning",
                "missing_prices",
                "No local price history was loaded from the Excel workbook.",
                recommendation="Use cached Excel prices or connect a controlled market-data provider.",
            )
        )
    elif positions is not None and not positions.empty:
        effective_prices = _effective_prices_for_validation(prices)
        open_positions = positions.copy()
        open_positions["quantity"] = pd.to_numeric(open_positions["quantity"], errors="coerce").fillna(0)
        open_symbols = sorted(open_positions.loc[open_positions["quantity"] > 0, "symbol"].dropna().astype(str).str.upper().unique().tolist())
        priced_symbols = set(effective_prices["symbol"].dropna().astype(str).str.upper()) if not effective_prices.empty else set()
        missing_price_symbols = [symbol for symbol in open_symbols if symbol not in priced_symbols]
        if missing_price_symbols:
            issues.append(
                _issue(
                    "Warning",
                    "open_position_without_market_price",
                    f"{len(missing_price_symbols)} open positions do not have any market price observation.",
                    recommendation="Load an Excel cached price or refresh market data before relying on valuation.",
                    affected_symbols=missing_price_symbols,
                )
            )

        if portfolio_snapshots is not None and not portfolio_snapshots.empty and not effective_prices.empty:
            snapshot_dates = pd.to_datetime(portfolio_snapshots["date"], errors="coerce").dropna()
            latest_snapshot = snapshot_dates.max() if not snapshot_dates.empty else None
            stale_symbols: list[str] = []
            if latest_snapshot is not None:
                effective_prices["date"] = pd.to_datetime(effective_prices["date"], errors="coerce")
                for symbol in open_symbols:
                    symbol_dates = effective_prices.loc[effective_prices["symbol"] == symbol, "date"].dropna()
                    if not symbol_dates.empty and symbol_dates.max() < latest_snapshot:
                        stale_symbols.append(symbol)
                if stale_symbols:
                    issues.append(
                        _issue(
                            "Warning",
                            "open_position_price_older_than_snapshot",
                            f"{len(stale_symbols)} open positions have latest prices older than the latest portfolio snapshot.",
                            recommendation="Refresh market data or confirm whether the latest snapshot should use stale prices.",
                            affected_symbols=stale_symbols,
                        )
                    )

        source_counts = prices.copy()
        if {"date", "symbol", "source"}.issubset(source_counts.columns):
            source_counts["date"] = pd.to_datetime(source_counts["date"], errors="coerce")
            source_counts["symbol"] = source_counts["symbol"].astype(str).str.upper()
            source_counts = source_counts.dropna(subset=["date", "symbol", "source"])
            source_pairs = source_counts.groupby(["date", "symbol"])["source"].nunique().reset_index(name="source_count")
            duplicated_sources = source_pairs[source_pairs["source_count"] > 1]
            if not duplicated_sources.empty:
                issues.append(
                    _issue(
                        "Warning",
                        "duplicate_prices_resolved_by_source_priority",
                        f"{len(duplicated_sources)} date/symbol price observations exist in multiple sources and were resolved by source priority.",
                        recommendation="Review source coverage; Yahoo Finance overrides the Excel Precios sheet for identical date/symbol observations.",
                        affected_symbols=duplicated_sources["symbol"].dropna().astype(str).unique().tolist(),
                    )
                )

    if benchmark_prices is None or benchmark_prices.empty:
        issues.append(
            _issue(
                "Warning",
                "missing_benchmark_data",
                "No S&P 500 benchmark prices are available locally.",
                recommendation="Load benchmark data from Excel or a vetted local cache before discussing relative performance.",
            )
        )
    elif portfolio_snapshots is not None and not portfolio_snapshots.empty:
        snapshot_dates = pd.to_datetime(portfolio_snapshots["date"], errors="coerce").dropna()
        benchmark_data = benchmark_prices.copy()
        benchmark_data["date"] = pd.to_datetime(benchmark_data["date"], errors="coerce")
        benchmark_data["price"] = pd.to_numeric(benchmark_data["price"], errors="coerce")
        benchmark_data = benchmark_data.dropna(subset=["date", "price"]).sort_values("date")
        benchmark_dates = benchmark_data["date"].dropna()
        if not snapshot_dates.empty and not benchmark_dates.empty and benchmark_dates.max() < snapshot_dates.max():
            issues.append(
                _issue(
                    "Warning",
                    "benchmark_older_than_portfolio_snapshot",
                    "The latest S&P 500 benchmark observation is older than the latest portfolio snapshot.",
                    recommendation="Refresh benchmark data or explain that relative performance is based on stale benchmark data.",
                )
            )
        if len(benchmark_data) >= 2:
            ratios = benchmark_data["price"] / benchmark_data["price"].shift(1)
            scale_jumps = benchmark_data[(ratios > 3) | (ratios < 1 / 3)]
            if not scale_jumps.empty:
                issues.append(
                    _issue(
                        "Warning",
                        "benchmark_scale_may_be_inconsistent",
                        "Benchmark scale inconsistency detected: possible mix of SPY-like and ^GSPC-like levels.",
                        recommendation="Check whether ETF-level prices and raw index levels were mixed. The Yahoo refresh uses SPY because the Excel S&P500 series is SPY-like in scale.",
                    )
                )

    incomplete = trades[
        trades["status"].astype(str).str.contains("INCOMPLETE", case=False, na=False)
        | trades["notes"].fillna("").astype(str).str.contains("Missing", case=False, na=False)
    ]
    if not incomplete.empty:
        issues.append(
            _issue(
                "Warning",
                "rows_not_fully_interpreted",
                f"{len(incomplete)} ledger events were kept but could not be fully interpreted.",
                _row_ids(incomplete),
                "Review the affected source rows; the MVP keeps them visible instead of crashing.",
                incomplete.get("symbol", pd.Series(dtype=str)).dropna().astype(str).unique().tolist(),
            )
        )

    if portfolio_snapshots is None or portfolio_snapshots.empty:
        issues.append(
            _issue(
                "Error",
                "missing_portfolio_history",
                "No portfolio snapshot history is available for performance analysis.",
                recommendation="Check the Portfolio sheet or rebuild snapshots from the ledger and price cache.",
            )
        )

    if not issues:
        issues.append(
            _issue(
                "Info",
                "core_checks_passed",
                "Core ledger, position and benchmark checks did not find blocking issues.",
                recommendation="Continue to performance and risk analysis, while retaining the audit trail.",
            )
        )
    return issues


def issues_to_frame(issues: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(issues)
    if frame.empty:
        return pd.DataFrame(columns=["severity", "check_name", "message", "affected_rows", "affected_symbols", "recommendation"])
    frame["affected_rows"] = frame["affected_rows"].map(lambda rows: ", ".join(map(str, rows[:20])) if rows else "")
    if "affected_symbols" not in frame.columns:
        frame["affected_symbols"] = ""
    else:
        frame["affected_symbols"] = frame["affected_symbols"].map(lambda rows: ", ".join(map(str, rows[:20])) if rows else "")
    severity_order = {"Error": 0, "Warning": 1, "Info": 2}
    frame["_order"] = frame["severity"].map(severity_order).fillna(3)
    return frame.sort_values(["_order", "check_name"]).drop(columns=["_order"]).reset_index(drop=True)


def lineage_frame(issues: list[dict[str, Any]], trades: pd.DataFrame) -> pd.DataFrame:
    """Expand issue rows into source-level lineage for analyst review."""
    rows: list[dict[str, Any]] = []
    trade_lookup = pd.DataFrame() if trades is None else trades.copy()
    if not trade_lookup.empty and "source_row" in trade_lookup.columns:
        trade_lookup["source_row"] = pd.to_numeric(trade_lookup["source_row"], errors="coerce")

    for issue in issues:
        affected_rows = issue.get("affected_rows") or [None]
        for source_row in affected_rows:
            matched = pd.DataFrame()
            numeric_row = pd.to_numeric(pd.Series([source_row]), errors="coerce").iloc[0]
            if not trade_lookup.empty and pd.notna(numeric_row):
                matched = trade_lookup[trade_lookup["source_row"] == numeric_row]
            if matched.empty:
                rows.append(
                    {
                        "severity": issue.get("severity"),
                        "issue": issue.get("check_name"),
                        "source_sheet": None,
                        "source_row": source_row,
                        "symbol": ", ".join(map(str, issue.get("affected_symbols") or [])) or None,
                        "recommendation": issue.get("recommendation"),
                    }
                )
            else:
                for _, trade in matched.iterrows():
                    rows.append(
                        {
                            "severity": issue.get("severity"),
                            "issue": issue.get("check_name"),
                            "source_sheet": trade.get("source_sheet"),
                            "source_row": trade.get("source_row"),
                            "symbol": trade.get("symbol"),
                            "recommendation": issue.get("recommendation"),
                        }
                    )
    frame = pd.DataFrame(rows, columns=["severity", "issue", "source_sheet", "source_row", "symbol", "recommendation"])
    severity_order = {"Error": 0, "Warning": 1, "Info": 2}
    if not frame.empty:
        frame["_order"] = frame["severity"].map(severity_order).fillna(3)
        frame = frame.sort_values(["_order", "issue", "source_sheet", "source_row"]).drop(columns=["_order"])
    return frame.reset_index(drop=True)
