from __future__ import annotations

import pandas as pd

from src import validation


def test_open_position_without_market_price_warns_without_crashing() -> None:
    trades = pd.DataFrame(
        [
            {
                "trade_id": "abc-buy",
                "source_sheet": "TEST",
                "source_row": 1,
                "symbol": "ABC",
                "side": "BUY",
                "trade_date": "2026-01-01",
                "quantity": 10,
                "price": 100,
                "amount": 1000,
                "sector": "Technology",
                "status": "TEST",
                "notes": None,
            }
        ]
    )
    positions = pd.DataFrame(
        [
            {"as_of_date": "2026-01-02", "symbol": "ABC", "sector": "Technology", "quantity": 10, "average_cost": 100, "latest_price": 100, "market_value": 1000, "unrealized_pnl": 0, "weight": 1},
        ]
    )
    snapshots = pd.DataFrame(
        [
            {"date": "2026-01-02", "invested_value": 1000, "cash": 999000, "total_portfolio_value": 1000000, "portfolio_return": 0, "benchmark_return": 0},
        ]
    )
    prices = pd.DataFrame(
        [
            {"date": "2026-01-02", "symbol": "XYZ", "price": 20, "source": "Precios sheet"},
        ]
    )
    benchmark_prices = pd.DataFrame(
        [
            {"date": "2026-01-02", "benchmark": "S&P 500", "price": 100, "source": "Portfolio sheet"},
        ]
    )

    issues = validation.run_data_quality_checks(trades, positions, snapshots, prices, benchmark_prices)

    assert any(issue["check_name"] == "open_position_without_market_price" and issue["severity"] == "Warning" for issue in issues)


def test_benchmark_scale_jump_warns() -> None:
    trades = pd.DataFrame(
        [
            {
                "trade_id": "abc-buy",
                "source_sheet": "TEST",
                "source_row": 1,
                "symbol": "ABC",
                "side": "BUY",
                "trade_date": "2026-01-01",
                "quantity": 10,
                "price": 100,
                "amount": 1000,
                "sector": "Technology",
                "status": "TEST",
                "notes": None,
            }
        ]
    )
    positions = pd.DataFrame(
        [
            {"as_of_date": "2026-01-02", "symbol": "ABC", "sector": "Technology", "quantity": 10, "average_cost": 100, "latest_price": 100, "market_value": 1000, "unrealized_pnl": 0, "weight": 1},
        ]
    )
    snapshots = pd.DataFrame(
        [
            {"date": "2026-01-02", "invested_value": 1000, "cash": 999000, "total_portfolio_value": 1000000, "portfolio_return": 0, "benchmark_return": 0},
        ]
    )
    prices = pd.DataFrame(
        [
            {"date": "2026-01-02", "symbol": "ABC", "price": 100, "source": "Precios sheet"},
        ]
    )
    benchmark_prices = pd.DataFrame(
        [
            {"date": "2026-01-01", "benchmark": "S&P 500", "price": 620, "source": "Portfolio sheet"},
            {"date": "2026-01-02", "benchmark": "S&P 500", "price": 7000, "source": "Yahoo Finance"},
        ]
    )

    issues = validation.run_data_quality_checks(trades, positions, snapshots, prices, benchmark_prices)

    assert any(issue["check_name"] == "benchmark_scale_may_be_inconsistent" and issue["severity"] == "Warning" for issue in issues)
