from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import market_data, storage


def _trade(trade_id: str, symbol: str, side: str, trade_date: str, quantity: float = 10, price: float = 100) -> dict:
    return {
        "trade_id": trade_id,
        "source_sheet": "TEST",
        "source_row": None,
        "symbol": symbol,
        "side": side,
        "trade_date": trade_date,
        "quantity": quantity,
        "price": price,
        "amount": quantity * price,
        "sector": "Test",
        "status": "TEST",
        "notes": None,
    }


def test_get_open_position_symbols() -> None:
    positions = pd.DataFrame(
        [
            {"symbol": "MSFT", "quantity": 5},
            {"symbol": "AAPL", "quantity": 0},
            {"symbol": "NVDA", "quantity": 2},
        ]
    )

    assert market_data.get_open_position_symbols(positions) == ["MSFT", "NVDA"]


def test_get_last_price_date() -> None:
    prices = pd.DataFrame(
        [
            {"date": "2026-01-01", "symbol": "ABC", "price": 10, "source": "Precios sheet"},
            {"date": "2026-01-03", "symbol": "ABC", "price": 11, "source": "Yahoo Finance"},
            {"date": "2026-01-02", "symbol": "XYZ", "price": 20, "source": "Precios sheet"},
        ]
    )

    assert market_data.get_last_price_date(prices, "abc").isoformat() == "2026-01-03"
    assert market_data.get_last_price_date(prices, "MISSING") is None


def test_refresh_does_not_request_closed_positions(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "portfolio.sqlite"
    storage.init_database(db_path)
    storage.insert_trade(_trade("buy-open", "OPEN", "BUY", "2026-01-01"), db_path)
    storage.insert_trade(_trade("buy-closed", "CLOSED", "BUY", "2026-01-01"), db_path)
    storage.insert_trade(_trade("sell-closed", "CLOSED", "SELL", "2026-01-02"), db_path)
    storage.write_positions(
        pd.DataFrame(
            [
                {"as_of_date": "2026-01-02", "symbol": "OPEN", "sector": "Test", "quantity": 10, "average_cost": 100, "latest_price": 100, "market_value": 1000, "unrealized_pnl": 0, "weight": 1},
            ]
        ),
        db_path,
    )

    requested: list[str] = []

    def fake_download(symbols, start_date, end_date=None):
        requested.extend(symbols)
        return pd.DataFrame(columns=["date", "symbol", "price", "source"])

    monkeypatch.setattr(market_data, "download_daily_prices_from_yahoo", fake_download)
    monkeypatch.setattr(market_data, "download_sp500_reference_from_yahoo", lambda start_date, end_date=None: pd.DataFrame(columns=["date", "benchmark", "price", "source"]))

    result = market_data.refresh_open_position_prices(db_path)

    assert requested == ["OPEN"]
    assert "CLOSED" not in requested
    assert result["status"] == "no_update"


def test_empty_yahoo_response_handled_gracefully(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "portfolio.sqlite"
    storage.init_database(db_path)
    storage.insert_trade(_trade("buy-open", "OPEN", "BUY", "2026-01-01"), db_path)
    storage.write_positions(
        pd.DataFrame(
            [
                {"as_of_date": "2026-01-01", "symbol": "OPEN", "sector": "Test", "quantity": 10, "average_cost": 100, "latest_price": 100, "market_value": 1000, "unrealized_pnl": 0, "weight": 1},
            ]
        ),
        db_path,
    )
    monkeypatch.setattr(market_data, "download_daily_prices_from_yahoo", lambda symbols, start_date, end_date=None: pd.DataFrame(columns=["date", "symbol", "price", "source"]))
    monkeypatch.setattr(market_data, "download_sp500_reference_from_yahoo", lambda start_date, end_date=None: pd.DataFrame(columns=["date", "benchmark", "price", "source"]))

    result = market_data.refresh_open_position_prices(db_path)

    assert result["status"] == "no_update"
    assert result["prices_upserted"] == 0
    assert result["benchmark_upserted"] == 0
