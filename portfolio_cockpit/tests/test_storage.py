from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import storage


def _create_minimal_excel(path: Path) -> None:
    track = pd.DataFrame(
        [
            [None] * 14,
            [None, "Stock", "Buy", "Sell", "Shares0", "Shares", "PriceAcq", "Amount", "% Fund0", "% fund1", "MtM", "%perform", "Net P&L", "Sector"],
            [None, "ABC", pd.Timestamp("2026-01-01"), None, None, 10, 100, 1000, None, None, 110, None, 100, "Technology"],
        ]
    )
    portfolio = pd.DataFrame(
        [
            [None] * 7,
            ["Date", "Invested", "Cash", "Portoflio", "Performance", None, "S&P500"],
            [pd.Timestamp("2026-01-01"), 1000, 999000, 1_000_000, 0.0, None, 100],
            [pd.Timestamp("2026-01-02"), 1100, 999000, 1_000_100, 0.0001, None, 101],
        ]
    )
    precios = pd.DataFrame(
        [
            ["BUY", pd.Timestamp("2026-01-01")],
            ["SELL", None],
            ["Shares", 10],
            [None, "ABC"],
            [pd.Timestamp("2026-01-01"), 100],
            [pd.Timestamp("2026-01-02"), 110],
        ]
    )
    new_trades = pd.DataFrame(
        [
            [None] * 9,
            [None] * 9,
            [None, "Stock", "Buy", "Sell", "Shares0", "Shares", "PriceAcq", "Amount", "Sector"],
            [None, "ABC", None, pd.Timestamp("2026-01-03"), None, 1, 111, 111, "Technology"],
        ]
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        track.to_excel(writer, sheet_name="TRACK", header=False, index=False)
        portfolio.to_excel(writer, sheet_name="Portfolio", header=False, index=False)
        precios.to_excel(writer, sheet_name="Precios", header=False, index=False)
        pd.DataFrame().to_excel(writer, sheet_name="Cost", header=False, index=False)
        pd.DataFrame().to_excel(writer, sheet_name="Value", header=False, index=False)
        new_trades.to_excel(writer, sheet_name="NEW TRADES", header=False, index=False)


def test_new_trade_persistence(tmp_path: Path) -> None:
    db_path = tmp_path / "portfolio.sqlite"
    storage.init_database(db_path)
    trade = {
        "trade_id": "manual-1",
        "source_sheet": "MANUAL",
        "source_row": None,
        "symbol": "ABC",
        "side": "BUY",
        "trade_date": "2026-01-01",
        "quantity": 1,
        "price": 10,
        "amount": 10,
        "sector": "Technology",
        "status": "TEST",
        "notes": None,
    }
    assert storage.insert_trade(trade, db_path) is True
    assert storage.insert_trade(trade, db_path) is False
    assert len(storage.get_trades(db_path)) == 1


def test_upsert_prices_does_not_duplicate_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "portfolio.sqlite"
    price_frame = pd.DataFrame(
        [
            {"date": "2026-01-01", "symbol": "ABC", "price": 10, "source": "Yahoo Finance"},
            {"date": "2026-01-01", "symbol": "ABC", "price": 10.5, "source": "Yahoo Finance"},
        ]
    )

    storage.upsert_prices(price_frame, db_path)
    storage.upsert_prices(price_frame, db_path)

    stored = storage.get_prices(db_path)
    assert len(stored) == 1
    assert stored.iloc[0]["price"] == 10.5


def test_reset_database_behavior(tmp_path: Path) -> None:
    excel_path = tmp_path / "portfolio.xlsx"
    db_path = tmp_path / "portfolio.sqlite"
    _create_minimal_excel(excel_path)

    storage.reset_database_from_excel(excel_path, db_path)
    initial_count = len(storage.get_trades(db_path))
    assert initial_count == 1

    storage.insert_trade(
        {
            "trade_id": "manual-1",
            "source_sheet": "MANUAL",
            "source_row": None,
            "symbol": "ABC",
            "side": "BUY",
            "trade_date": "2026-01-02",
            "quantity": 1,
            "price": 10,
            "amount": 10,
            "sector": "Technology",
            "status": "TEST",
            "notes": None,
        },
        db_path,
    )
    assert len(storage.get_trades(db_path)) == initial_count + 1

    storage.reset_database_from_excel(excel_path, db_path)
    assert len(storage.get_trades(db_path)) == initial_count
