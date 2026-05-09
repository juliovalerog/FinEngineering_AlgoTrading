from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import data_loader


POSITION_COLUMNS = [
    "as_of_date",
    "symbol",
    "sector",
    "quantity",
    "average_cost",
    "latest_price",
    "market_value",
    "unrealized_pnl",
    "weight",
]


def _empty_positions() -> pd.DataFrame:
    return pd.DataFrame(columns=POSITION_COLUMNS)


def build_trade_ledger_from_excel(excel_path: Path) -> pd.DataFrame:
    """Public entry point used by storage reset and tests."""
    return data_loader.parse_track_trades(Path(excel_path))


def _prepare_trades(trades: pd.DataFrame, as_of_date: str | None = None) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame()
    prepared = trades.copy()
    prepared["trade_date"] = pd.to_datetime(prepared["trade_date"], errors="coerce")
    prepared["quantity"] = pd.to_numeric(prepared["quantity"], errors="coerce")
    prepared["price"] = pd.to_numeric(prepared["price"], errors="coerce")
    prepared["amount"] = pd.to_numeric(prepared.get("amount"), errors="coerce")
    prepared["side"] = prepared["side"].astype(str).str.upper()
    prepared["symbol"] = prepared["symbol"].astype(str).str.upper()
    prepared = prepared.dropna(subset=["trade_date", "symbol", "side", "quantity", "price"])
    prepared = prepared[prepared["quantity"] > 0]
    if as_of_date:
        prepared = prepared[prepared["trade_date"] <= pd.to_datetime(as_of_date)]
    prepared["_side_order"] = prepared["side"].map({"BUY": 0, "SELL": 1}).fillna(2)
    return prepared.sort_values(["trade_date", "_side_order", "source_row"], na_position="last")


def _latest_prices(prices: pd.DataFrame | None, trades: pd.DataFrame, as_of_date: str | None) -> dict[str, float]:
    latest: dict[str, float] = {}
    if prices is not None and not prices.empty:
        price_data = prices.copy()
        price_data["date"] = pd.to_datetime(price_data["date"], errors="coerce")
        price_data["price"] = pd.to_numeric(price_data["price"], errors="coerce")
        price_data["symbol"] = price_data["symbol"].astype(str).str.upper()
        price_data = price_data.dropna(subset=["date", "symbol", "price"])
        if as_of_date:
            price_data = price_data[price_data["date"] <= pd.to_datetime(as_of_date)]
        if not price_data.empty:
            latest_prices = price_data.sort_values("date").groupby("symbol").tail(1)
            latest.update(dict(zip(latest_prices["symbol"], latest_prices["price"])))

    trade_prices = _prepare_trades(trades, as_of_date)
    if not trade_prices.empty:
        for _, row in trade_prices.groupby("symbol").tail(1).iterrows():
            latest.setdefault(str(row["symbol"]), float(row["price"]))
    return latest


def compute_positions(
    trades: pd.DataFrame,
    prices: pd.DataFrame | None = None,
    as_of_date: str | None = None,
) -> pd.DataFrame:
    """Compute current open positions from the normalized trade ledger.

    Buys create lots. Sells consume lots FIFO. If a sell would take the ledger
    negative, the engine consumes the available lots and leaves the exception for
    the data-quality layer instead of manufacturing a short position silently.
    """
    prepared = _prepare_trades(trades, as_of_date)
    if prepared.empty:
        return _empty_positions()

    if as_of_date is None:
        as_of = prepared["trade_date"].max()
        if prices is not None and not prices.empty:
            price_dates = pd.to_datetime(prices.get("date"), errors="coerce").dropna()
            if not price_dates.empty:
                as_of = max(as_of, price_dates.max())
    else:
        as_of = pd.to_datetime(as_of_date)

    lots: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sector_by_symbol: dict[str, str | None] = {}

    for _, trade in prepared.iterrows():
        symbol = str(trade["symbol"])
        sector = trade.get("sector")
        if isinstance(sector, str) and sector.strip():
            sector_by_symbol[symbol] = sector.strip()

        quantity = float(trade["quantity"])
        price = float(trade["price"])
        if trade["side"] == "BUY":
            lots[symbol].append({"quantity": quantity, "cost": price})
            continue

        if trade["side"] == "SELL":
            remaining = quantity
            while remaining > 1e-9 and lots[symbol]:
                lot = lots[symbol][0]
                consumed = min(remaining, lot["quantity"])
                lot["quantity"] -= consumed
                remaining -= consumed
                if lot["quantity"] <= 1e-9:
                    lots[symbol].pop(0)

    latest = _latest_prices(prices, trades, as_of.date().isoformat())
    position_rows: list[dict[str, Any]] = []
    for symbol, symbol_lots in lots.items():
        quantity = sum(lot["quantity"] for lot in symbol_lots)
        if quantity <= 1e-9:
            continue
        cost_basis = sum(lot["quantity"] * lot["cost"] for lot in symbol_lots)
        average_cost = cost_basis / quantity
        latest_price = latest.get(symbol, average_cost)
        market_value = quantity * latest_price
        position_rows.append(
            {
                "as_of_date": as_of.date().isoformat(),
                "symbol": symbol,
                "sector": sector_by_symbol.get(symbol),
                "quantity": quantity,
                "average_cost": average_cost,
                "latest_price": latest_price,
                "market_value": market_value,
                "unrealized_pnl": market_value - cost_basis,
                "weight": 0.0,
            }
        )

    positions = pd.DataFrame(position_rows, columns=POSITION_COLUMNS)
    if positions.empty:
        return positions
    invested_value = positions["market_value"].sum()
    positions["weight"] = np.where(invested_value != 0, positions["market_value"] / invested_value, np.nan)
    return positions.sort_values("market_value", ascending=False).reset_index(drop=True)


def compute_cash(trades: pd.DataFrame, initial_cash: float = 1_000_000, as_of_date: str | None = None) -> float:
    """Reconstruct cash by applying cash movements from the trade ledger."""
    prepared = _prepare_trades(trades, as_of_date)
    cash = float(initial_cash)
    for _, trade in prepared.iterrows():
        amount = trade["amount"]
        if pd.isna(amount):
            amount = float(trade["quantity"]) * float(trade["price"])
        if trade["side"] == "BUY":
            cash -= float(amount)
        elif trade["side"] == "SELL":
            cash += float(amount)
    return float(cash)


def compute_realized_unrealized_pnl(
    trades: pd.DataFrame,
    prices: pd.DataFrame | None = None,
    as_of_date: str | None = None,
) -> dict[str, float]:
    """Compute realized P&L from closed lots and unrealized P&L from open lots."""
    prepared = _prepare_trades(trades, as_of_date)
    if prepared.empty:
        return {"realized_pnl": 0.0, "unrealized_pnl": 0.0, "total_pnl": 0.0}

    lots: dict[str, list[dict[str, float]]] = defaultdict(list)
    realized = 0.0
    for _, trade in prepared.iterrows():
        symbol = str(trade["symbol"])
        quantity = float(trade["quantity"])
        price = float(trade["price"])
        if trade["side"] == "BUY":
            lots[symbol].append({"quantity": quantity, "cost": price})
            continue
        if trade["side"] == "SELL":
            remaining = quantity
            while remaining > 1e-9 and lots[symbol]:
                lot = lots[symbol][0]
                consumed = min(remaining, lot["quantity"])
                realized += consumed * (price - lot["cost"])
                lot["quantity"] -= consumed
                remaining -= consumed
                if lot["quantity"] <= 1e-9:
                    lots[symbol].pop(0)

    positions = compute_positions(trades, prices, as_of_date)
    unrealized = 0.0 if positions.empty else float(positions["unrealized_pnl"].sum())
    return {"realized_pnl": float(realized), "unrealized_pnl": unrealized, "total_pnl": float(realized + unrealized)}


def compute_portfolio_snapshots(
    trades: pd.DataFrame,
    prices: pd.DataFrame | None = None,
    benchmark_prices: pd.DataFrame | None = None,
    initial_cash: float = 1_000_000,
    existing_snapshots: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build or extend portfolio snapshots.

    On initial load the MVP trusts the professor workbook's cached Portfolio
    sheet. When trades are added later, the SQLite ledger becomes the system of
    record and the engine appends recalculated snapshots for new trade dates.
    """
    base = pd.DataFrame()
    if existing_snapshots is not None and not existing_snapshots.empty:
        base = existing_snapshots.copy()
        base["date"] = pd.to_datetime(base["date"], errors="coerce")
        base = base.dropna(subset=["date"]).sort_values("date")

    prepared = _prepare_trades(trades)
    if prepared.empty:
        return base.assign(date=base["date"].dt.date.astype(str)) if not base.empty else pd.DataFrame()

    if not base.empty:
        last_snapshot_date = base["date"].max()
        snapshot_dates = sorted(prepared.loc[prepared["trade_date"] > last_snapshot_date, "trade_date"].dt.date.unique())
        first_value = float(base["total_portfolio_value"].dropna().iloc[0])
    else:
        snapshot_dates = sorted(prepared["trade_date"].dt.date.unique())
        first_value = initial_cash

    rows: list[dict[str, Any]] = []
    benchmark_lookup = pd.DataFrame()
    if benchmark_prices is not None and not benchmark_prices.empty:
        benchmark_lookup = benchmark_prices.copy()
        benchmark_lookup["date"] = pd.to_datetime(benchmark_lookup["date"], errors="coerce")
        benchmark_lookup["price"] = pd.to_numeric(benchmark_lookup["price"], errors="coerce")
        benchmark_lookup = benchmark_lookup.dropna(subset=["date", "price"]).sort_values("date")
        first_benchmark = float(benchmark_lookup["price"].iloc[0]) if not benchmark_lookup.empty else np.nan
    else:
        first_benchmark = np.nan

    for snapshot_date in snapshot_dates:
        as_of = snapshot_date.isoformat()
        positions = compute_positions(trades, prices, as_of)
        invested_value = 0.0 if positions.empty else float(positions["market_value"].sum())
        cash = compute_cash(trades, initial_cash=initial_cash, as_of_date=as_of)
        total_value = invested_value + cash
        benchmark_return = np.nan
        if not benchmark_lookup.empty and not np.isnan(first_benchmark) and first_benchmark != 0:
            available = benchmark_lookup[benchmark_lookup["date"] <= pd.to_datetime(as_of)]
            if not available.empty:
                benchmark_return = float(available.iloc[-1]["price"] / first_benchmark - 1)
        rows.append(
            {
                "date": as_of,
                "invested_value": invested_value,
                "cash": cash,
                "total_portfolio_value": total_value,
                "portfolio_return": total_value / first_value - 1 if first_value else np.nan,
                "benchmark_return": benchmark_return,
            }
        )

    appended = pd.DataFrame(rows)
    if base.empty:
        return appended
    base["date"] = base["date"].dt.date.astype(str)
    if appended.empty:
        return base
    combined = pd.concat([base, appended], ignore_index=True)
    return combined.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)


def recompute_all_after_trade(
    trades: pd.DataFrame,
    prices: pd.DataFrame | None = None,
    benchmark_prices: pd.DataFrame | None = None,
    existing_snapshots: pd.DataFrame | None = None,
    initial_cash: float = 1_000_000,
) -> dict[str, pd.DataFrame]:
    """Return recalculated derived tables after a ledger change.

    This function is intentionally storage-agnostic. SQLite reads and writes
    stay in ``storage.py`` so the classroom architecture has one database
    boundary.
    """
    positions = compute_positions(trades, prices)
    snapshots = compute_portfolio_snapshots(
        trades,
        prices=prices,
        benchmark_prices=benchmark_prices,
        initial_cash=initial_cash,
        existing_snapshots=existing_snapshots,
    )
    return {"positions": positions, "portfolio_snapshots": snapshots}
