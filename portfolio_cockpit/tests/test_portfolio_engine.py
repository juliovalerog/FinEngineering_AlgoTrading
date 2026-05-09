from __future__ import annotations

import numpy as np
import pandas as pd

from src import portfolio_engine


def _trade(symbol: str, side: str, trade_date: str, quantity: float, price: float) -> dict:
    return {
        "trade_id": f"{symbol}-{side}-{trade_date}-{quantity}-{price}",
        "source_sheet": "TEST",
        "source_row": 1,
        "symbol": symbol,
        "side": side,
        "trade_date": trade_date,
        "quantity": quantity,
        "price": price,
        "amount": quantity * price,
        "sector": "Technology",
        "status": "TEST",
        "notes": None,
        "created_at": "2026-01-01T00:00:00",
    }


def test_position_calculation_after_buy() -> None:
    trades = pd.DataFrame([_trade("ABC", "BUY", "2026-01-01", 10, 100)])
    prices = pd.DataFrame([{"date": "2026-01-02", "symbol": "ABC", "price": 110, "source": "TEST"}])
    positions = portfolio_engine.compute_positions(trades, prices)
    assert len(positions) == 1
    assert positions.iloc[0]["quantity"] == 10
    assert positions.iloc[0]["average_cost"] == 100
    assert positions.iloc[0]["latest_price"] == 110
    assert positions.iloc[0]["market_value"] == 1100


def test_position_calculation_after_buy_sell() -> None:
    trades = pd.DataFrame(
        [
            _trade("ABC", "BUY", "2026-01-01", 10, 100),
            _trade("ABC", "SELL", "2026-01-03", 4, 120),
        ]
    )
    positions = portfolio_engine.compute_positions(trades)
    pnl = portfolio_engine.compute_realized_unrealized_pnl(trades)
    assert len(positions) == 1
    assert positions.iloc[0]["quantity"] == 6
    assert positions.iloc[0]["average_cost"] == 100
    assert np.isclose(pnl["realized_pnl"], 80)


def test_trade_simulation_does_not_mutate_original_ledger() -> None:
    trades = pd.DataFrame([_trade("ABC", "BUY", "2026-01-01", 10, 100)])
    original_count = len(trades)
    simulated = portfolio_engine.simulate_trade_impact(
        trades,
        _trade("ABC", "BUY", "2026-01-02", 5, 110),
    )
    assert len(trades) == original_count
    assert len(simulated["trades"]) == original_count + 1
    assert simulated["positions"].iloc[0]["quantity"] == 15
