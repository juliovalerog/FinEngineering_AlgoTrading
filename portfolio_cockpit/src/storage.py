from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from . import data_loader, portfolio_engine


APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXCEL_PATH = APP_ROOT / "data" / "input" / "Portfolio Example JULIO.xlsx"
DEFAULT_DB_PATH = APP_ROOT / "data" / "store" / "portfolio_mvp.sqlite"


def _resolve_db_path(db_path: Path | str | None = None) -> Path:
    return Path(db_path) if db_path is not None else DEFAULT_DB_PATH


def _connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    resolved = _resolve_db_path(db_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(resolved)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


@contextmanager
def _connection(db_path: Path | str | None = None):
    connection = _connect(db_path)
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def database_exists(db_path: Path | str | None = None) -> bool:
    return _resolve_db_path(db_path).exists()


def init_database(db_path: Path | str | None = None) -> None:
    """Create the local SQLite schema if it does not already exist."""
    with _connection(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                source_sheet TEXT,
                source_row INTEGER,
                symbol TEXT,
                side TEXT,
                trade_date TEXT,
                quantity REAL,
                price REAL,
                amount REAL,
                sector TEXT,
                status TEXT,
                notes TEXT,
                created_at TEXT
            );

            CREATE TABLE IF NOT EXISTS prices (
                date TEXT,
                symbol TEXT,
                price REAL,
                source TEXT,
                PRIMARY KEY (date, symbol, source)
            );

            CREATE TABLE IF NOT EXISTS benchmark_prices (
                date TEXT,
                benchmark TEXT,
                price REAL,
                source TEXT,
                PRIMARY KEY (date, benchmark, source)
            );

            CREATE TABLE IF NOT EXISTS positions (
                as_of_date TEXT,
                symbol TEXT,
                sector TEXT,
                quantity REAL,
                average_cost REAL,
                latest_price REAL,
                market_value REAL,
                unrealized_pnl REAL,
                weight REAL
            );

            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                date TEXT PRIMARY KEY,
                invested_value REAL,
                cash REAL,
                total_portfolio_value REAL,
                portfolio_return REAL,
                benchmark_return REAL
            );

            CREATE TABLE IF NOT EXISTS audit_log (
                event_time TEXT,
                event_type TEXT,
                description TEXT
            );

            CREATE TABLE IF NOT EXISTS configuration (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            """
        )


def _write_table(table_name: str, frame: pd.DataFrame, db_path: Path | str | None = None) -> None:
    init_database(db_path)
    with _connection(db_path) as conn:
        conn.execute(f"DELETE FROM {table_name}")
        if frame is not None and not frame.empty:
            frame.to_sql(table_name, conn, if_exists="append", index=False)


def _read_table(table_name: str, db_path: Path | str | None = None) -> pd.DataFrame:
    if not database_exists(db_path):
        return pd.DataFrame()
    with _connection(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)


def write_audit_log(event_type: str, description: str, db_path: Path | str | None = None) -> None:
    init_database(db_path)
    with _connection(db_path) as conn:
        conn.execute(
            "INSERT INTO audit_log (event_time, event_type, description) VALUES (?, ?, ?)",
            (_utc_now(), event_type, description),
        )


def get_audit_log(db_path: Path | str | None = None, limit: int | None = None) -> pd.DataFrame:
    if not database_exists(db_path):
        return pd.DataFrame(columns=["event_time", "event_type", "description"])
    query = "SELECT event_time, event_type, description FROM audit_log ORDER BY event_time DESC"
    if limit:
        query += f" LIMIT {int(limit)}"
    with _connection(db_path) as conn:
        return pd.read_sql_query(query, conn)


def set_configuration(key: str, value: Any, db_path: Path | str | None = None) -> None:
    init_database(db_path)
    stored_value = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
    with _connection(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO configuration (key, value) VALUES (?, ?)",
            (key, stored_value),
        )


def get_configuration(db_path: Path | str | None = None) -> dict[str, str]:
    data = _read_table("configuration", db_path)
    if data.empty:
        return {}
    return dict(zip(data["key"], data["value"]))


def get_initial_cash(db_path: Path | str | None = None) -> float:
    config = get_configuration(db_path)
    try:
        return float(config.get("initial_cash", "1000000"))
    except ValueError:
        return 1_000_000.0


def insert_trade(trade: dict[str, Any], db_path: Path | str | None = None) -> bool:
    """Insert one trade. Returns False when a deterministic import ID already exists."""
    init_database(db_path)
    record = dict(trade)
    record.setdefault("created_at", _utc_now())
    columns = [
        "trade_id",
        "source_sheet",
        "source_row",
        "symbol",
        "side",
        "trade_date",
        "quantity",
        "price",
        "amount",
        "sector",
        "status",
        "notes",
        "created_at",
    ]
    values = [record.get(column) for column in columns]
    with _connection(db_path) as conn:
        cursor = conn.execute(
            f"INSERT OR IGNORE INTO trades ({', '.join(columns)}) VALUES ({', '.join(['?'] * len(columns))})",
            values,
        )
        inserted = cursor.rowcount > 0
    return inserted


def insert_trades(trades: pd.DataFrame, db_path: Path | str | None = None) -> int:
    inserted = 0
    if trades is None or trades.empty:
        return inserted
    for trade in trades.to_dict("records"):
        inserted += int(insert_trade(trade, db_path))
    return inserted


def get_trades(db_path: Path | str | None = None) -> pd.DataFrame:
    return _read_table("trades", db_path)


def get_prices(db_path: Path | str | None = None) -> pd.DataFrame:
    return _read_table("prices", db_path)


def get_benchmark_prices(db_path: Path | str | None = None) -> pd.DataFrame:
    return _read_table("benchmark_prices", db_path)


def get_positions(db_path: Path | str | None = None) -> pd.DataFrame:
    return _read_table("positions", db_path)


def get_portfolio_snapshots(db_path: Path | str | None = None) -> pd.DataFrame:
    return _read_table("portfolio_snapshots", db_path)


def write_positions(positions: pd.DataFrame, db_path: Path | str | None = None) -> None:
    _write_table("positions", positions, db_path)


def write_portfolio_snapshots(snapshots: pd.DataFrame, db_path: Path | str | None = None) -> None:
    _write_table("portfolio_snapshots", snapshots, db_path)


def upsert_prices(price_frame: pd.DataFrame, db_path: Path | str | None = None) -> int:
    """Insert or replace normalized daily prices without duplicating observations."""
    init_database(db_path)
    if price_frame is None or price_frame.empty:
        return 0
    data = price_frame.copy()
    required = ["date", "symbol", "price", "source"]
    for column in required:
        if column not in data.columns:
            data[column] = None
    data = data[required]
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.date.astype("string")
    data["symbol"] = data["symbol"].astype(str).str.upper().str.strip()
    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    data["source"] = data["source"].fillna("Unknown").astype(str)
    data = data.dropna(subset=["date", "symbol", "price", "source"])
    if data.empty:
        return 0
    rows = list(data.itertuples(index=False, name=None))
    with _connection(db_path) as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO prices (date, symbol, price, source)
            VALUES (?, ?, ?, ?)
            """,
            rows,
        )
    return len(rows)


def upsert_benchmark_prices(benchmark_frame: pd.DataFrame, db_path: Path | str | None = None) -> int:
    """Insert or replace normalized daily benchmark prices."""
    init_database(db_path)
    if benchmark_frame is None or benchmark_frame.empty:
        return 0
    data = benchmark_frame.copy()
    required = ["date", "benchmark", "price", "source"]
    for column in required:
        if column not in data.columns:
            data[column] = None
    data = data[required]
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.date.astype("string")
    data["benchmark"] = data["benchmark"].fillna("S&P 500").astype(str)
    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    data["source"] = data["source"].fillna("Unknown").astype(str)
    data = data.dropna(subset=["date", "benchmark", "price", "source"])
    if data.empty:
        return 0
    rows = list(data.itertuples(index=False, name=None))
    with _connection(db_path) as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO benchmark_prices (date, benchmark, price, source)
            VALUES (?, ?, ?, ?)
            """,
            rows,
        )
    return len(rows)


def delete_benchmark_prices(
    benchmark: str = "S&P 500",
    source: str | None = None,
    db_path: Path | str | None = None,
) -> int:
    """Delete benchmark observations by benchmark name and optional source."""
    init_database(db_path)
    with _connection(db_path) as conn:
        if source is None:
            cursor = conn.execute("DELETE FROM benchmark_prices WHERE benchmark = ?", (benchmark,))
        else:
            cursor = conn.execute(
                "DELETE FROM benchmark_prices WHERE benchmark = ? AND source = ?",
                (benchmark, source),
            )
        return int(cursor.rowcount)


def recompute_all_after_trade(db_path: Path | str | None = None) -> dict[str, pd.DataFrame]:
    """Read the SQLite ledger, recompute derived tables, and write them back."""
    trades = get_trades(db_path)
    prices = get_prices(db_path)
    benchmark_prices = get_benchmark_prices(db_path)
    existing_snapshots = get_portfolio_snapshots(db_path)
    recalculated = portfolio_engine.recompute_all_after_trade(
        trades,
        prices=prices,
        benchmark_prices=benchmark_prices,
        existing_snapshots=existing_snapshots,
        initial_cash=get_initial_cash(db_path),
    )
    write_positions(recalculated["positions"], db_path)
    write_portfolio_snapshots(recalculated["portfolio_snapshots"], db_path)
    write_audit_log("RECOMPUTE", "Recomputed positions and portfolio snapshots from SQLite ledger.", db_path)
    return recalculated


def recompute_all_after_market_data_refresh(db_path: Path | str | None = None) -> dict[str, Any]:
    """Recompute positions and extend snapshots after market-data updates."""
    trades = get_trades(db_path)
    prices = get_prices(db_path)
    benchmark_prices = get_benchmark_prices(db_path)
    existing_snapshots = get_portfolio_snapshots(db_path)
    previous_dates = pd.to_datetime(existing_snapshots.get("date"), errors="coerce").dropna() if existing_snapshots is not None and not existing_snapshots.empty else pd.Series(dtype="datetime64[ns]")
    previous_latest = previous_dates.max().date().isoformat() if not previous_dates.empty else None
    rows_before = 0 if existing_snapshots is None else int(len(existing_snapshots))
    positions = portfolio_engine.compute_positions(trades, prices)
    snapshots = portfolio_engine.compute_daily_mark_to_market_snapshots(
        trades,
        prices=prices,
        benchmark_prices=benchmark_prices,
        existing_snapshots=existing_snapshots,
        initial_cash=get_initial_cash(db_path),
    )
    write_positions(positions, db_path)
    write_portfolio_snapshots(snapshots, db_path)
    new_dates = pd.to_datetime(snapshots.get("date"), errors="coerce").dropna() if snapshots is not None and not snapshots.empty else pd.Series(dtype="datetime64[ns]")
    new_latest = new_dates.max().date().isoformat() if not new_dates.empty else None
    summary = {
        "previous_latest_snapshot_date": previous_latest,
        "new_latest_snapshot_date": new_latest,
        "snapshot_rows_before": rows_before,
        "snapshot_rows_after": 0 if snapshots is None else int(len(snapshots)),
        "snapshots_extended": bool(previous_latest and new_latest and pd.to_datetime(new_latest) > pd.to_datetime(previous_latest)),
    }
    write_audit_log(
        "MARKET_DATA_RECOMPUTE",
        (
            "Market-data refresh triggered daily mark-to-market recomputation from effective prices. "
            f"Latest snapshot moved from {previous_latest or 'n/a'} to {new_latest or 'n/a'}."
        ),
        db_path,
    )
    return {"positions": positions, "portfolio_snapshots": snapshots, "summary": summary}


def reset_database_from_excel(
    excel_path: Path | str | None = None,
    db_path: Path | str | None = None,
) -> dict[str, pd.DataFrame]:
    """Delete and rebuild the demo SQLite database from the immutable Excel input."""
    resolved_excel = Path(excel_path) if excel_path is not None else DEFAULT_EXCEL_PATH
    resolved_db = _resolve_db_path(db_path)
    if resolved_db.exists():
        resolved_db.unlink()

    init_database(resolved_db)
    model = data_loader.load_excel_model(resolved_excel)
    trades = model["trades"]
    prices = model["prices"]
    benchmark_prices = model["benchmark_prices"]
    snapshots = model["portfolio_snapshots"]

    _write_table("trades", trades.assign(created_at=_utc_now()) if not trades.empty else trades, resolved_db)
    _write_table("prices", prices, resolved_db)
    _write_table("benchmark_prices", benchmark_prices, resolved_db)
    _write_table("portfolio_snapshots", snapshots, resolved_db)

    positions = portfolio_engine.compute_positions(trades, prices)
    _write_table("positions", positions, resolved_db)

    metadata = model["sheet_metadata"].to_dict("records")
    set_configuration("excel_path", str(resolved_excel), resolved_db)
    set_configuration("initial_cash", "1000000", resolved_db)
    set_configuration("loaded_sheets", metadata, resolved_db)
    set_configuration("database_status", "initialized from Excel", resolved_db)
    write_audit_log("INITIALIZE", f"SQLite database initialized from {resolved_excel}.", resolved_db)
    return model | {"positions": positions}
