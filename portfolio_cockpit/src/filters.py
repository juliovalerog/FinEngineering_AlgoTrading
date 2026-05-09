from __future__ import annotations

from typing import Iterable

import pandas as pd


def _selected(values: Iterable[str] | None) -> set[str]:
    if values is None:
        return set()
    return {str(value).upper() for value in values if str(value).strip()}


def filter_trades(
    trades: pd.DataFrame,
    date_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    symbols: Iterable[str] | None = None,
    sectors: Iterable[str] | None = None,
    sides: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Apply display filters to trades. Empty selections mean no filter."""
    if trades is None or trades.empty:
        return pd.DataFrame() if trades is None else trades.copy()
    data = trades.copy()
    if date_range is not None and "trade_date" in data.columns:
        start, end = date_range
        dates = pd.to_datetime(data["trade_date"], errors="coerce")
        data = data[(dates >= pd.to_datetime(start)) & (dates <= pd.to_datetime(end))]
    selected_symbols = _selected(symbols)
    if selected_symbols and "symbol" in data.columns:
        data = data[data["symbol"].astype(str).str.upper().isin(selected_symbols)]
    selected_sectors = _selected(sectors)
    if selected_sectors and "sector" in data.columns:
        data = data[data["sector"].fillna("Unclassified").astype(str).str.upper().isin(selected_sectors)]
    selected_sides = _selected(sides)
    if selected_sides and "side" in data.columns:
        data = data[data["side"].astype(str).str.upper().isin(selected_sides)]
    return data.reset_index(drop=True)


def filter_positions(
    positions: pd.DataFrame,
    symbols: Iterable[str] | None = None,
    sectors: Iterable[str] | None = None,
    open_only: bool = True,
) -> pd.DataFrame:
    """Apply display filters to current positions."""
    if positions is None or positions.empty:
        return pd.DataFrame() if positions is None else positions.copy()
    data = positions.copy()
    if open_only and "quantity" in data.columns:
        data = data[pd.to_numeric(data["quantity"], errors="coerce") > 0]
    selected_symbols = _selected(symbols)
    if selected_symbols and "symbol" in data.columns:
        data = data[data["symbol"].astype(str).str.upper().isin(selected_symbols)]
    selected_sectors = _selected(sectors)
    if selected_sectors and "sector" in data.columns:
        data = data[data["sector"].fillna("Unclassified").astype(str).str.upper().isin(selected_sectors)]
    return data.reset_index(drop=True)


def filter_snapshots(
    snapshots: pd.DataFrame,
    date_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Apply display date filters to portfolio time series."""
    if snapshots is None or snapshots.empty:
        return pd.DataFrame() if snapshots is None else snapshots.copy()
    data = snapshots.copy()
    if date_range is not None and "date" in data.columns:
        start, end = date_range
        dates = pd.to_datetime(data["date"], errors="coerce")
        data = data[(dates >= pd.to_datetime(start)) & (dates <= pd.to_datetime(end))]
    return data.reset_index(drop=True)
