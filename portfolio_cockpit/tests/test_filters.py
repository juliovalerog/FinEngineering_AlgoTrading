from __future__ import annotations

import pandas as pd

from src import filters


def test_filters_do_not_crash_with_empty_selections() -> None:
    trades = pd.DataFrame(
        [
            {"trade_date": "2026-01-01", "symbol": "ABC", "sector": "Tech", "side": "BUY"},
            {"trade_date": "2026-01-02", "symbol": "XYZ", "sector": "Health", "side": "SELL"},
        ]
    )
    positions = pd.DataFrame(
        [
            {"symbol": "ABC", "sector": "Tech", "quantity": 10},
            {"symbol": "XYZ", "sector": "Health", "quantity": 0},
        ]
    )
    snapshots = pd.DataFrame(
        [
            {"date": "2026-01-01", "total_portfolio_value": 100},
            {"date": "2026-01-02", "total_portfolio_value": 101},
        ]
    )

    assert len(filters.filter_trades(trades, symbols=[], sectors=[], sides=[])) == 2
    assert len(filters.filter_positions(positions, symbols=[], sectors=[], open_only=True)) == 1
    assert len(filters.filter_snapshots(snapshots, None)) == 2
