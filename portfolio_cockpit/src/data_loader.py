from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPECTED_SHEETS = ["TRACK", "Portfolio", "Precios", "Cost", "Value", "NEW TRADES"]


def _normalise_label(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def _normalise_symbol(value: Any) -> str | None:
    text = _normalise_label(value)
    if not text:
        return None
    return text.upper()


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return None
    return float(number)


def _to_date(value: Any) -> str | None:
    date_value = pd.to_datetime(value, errors="coerce")
    if pd.isna(date_value):
        return None
    return date_value.date().isoformat()


def _stable_trade_id(*parts: Any) -> str:
    raw = "|".join("" if part is None else str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]


def _read_raw_sheet(excel_path: Path, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(excel_path, sheet_name=sheet_name, header=None)


def available_sheets(excel_path: Path) -> list[str]:
    if not Path(excel_path).exists():
        return []
    return pd.ExcelFile(excel_path).sheet_names


def sheet_metadata(excel_path: Path) -> pd.DataFrame:
    """Return present sheet names and simple row counts for the cockpit landing tab."""
    rows: list[dict[str, Any]] = []
    present = available_sheets(excel_path)
    for sheet in EXPECTED_SHEETS:
        if sheet not in present:
            rows.append({"sheet_name": sheet, "present": False, "row_count": 0, "column_count": 0})
            continue
        raw = _read_raw_sheet(excel_path, sheet)
        rows.append(
            {
                "sheet_name": sheet,
                "present": True,
                "row_count": int(raw.dropna(how="all").shape[0]),
                "column_count": int(raw.dropna(how="all", axis=1).shape[1]),
            }
        )
    return pd.DataFrame(rows)


def _find_header_row(raw: pd.DataFrame, required_labels: set[str]) -> int | None:
    required = {label.lower() for label in required_labels}
    for index, row in raw.iterrows():
        values = {_normalise_label(value).lower() for value in row.tolist()}
        if required.issubset(values):
            return int(index)
    return None


def _table_from_header(excel_path: Path, sheet_name: str, required_labels: set[str]) -> pd.DataFrame:
    raw = _read_raw_sheet(excel_path, sheet_name)
    header_row = _find_header_row(raw, required_labels)
    if header_row is None:
        return pd.DataFrame()

    headers = [_normalise_label(value) or f"unnamed_{column}" for column, value in enumerate(raw.iloc[header_row])]
    table = raw.iloc[header_row + 1 :].copy()
    table.columns = headers
    table = table.dropna(how="all")
    table["_source_row"] = table.index + 1
    return table


def _amount_from_fields(quantity: float | None, price: float | None, amount: float | None) -> float | None:
    if amount is not None:
        return amount
    if quantity is not None and price is not None:
        return quantity * price
    return None


def _trade_event(
    *,
    source_sheet: str,
    source_row: int,
    symbol: str | None,
    side: str,
    trade_date: str | None,
    quantity: float | None,
    price: float | None,
    amount: float | None,
    sector: str | None,
    status: str,
    notes: str | None = None,
) -> dict[str, Any]:
    missing = []
    if symbol is None:
        missing.append("symbol")
    if trade_date is None:
        missing.append("trade_date")
    if quantity is None:
        missing.append("quantity")
    if price is None:
        missing.append("price")
    status_value = status if not missing else f"{status}; INCOMPLETE"
    note_text = notes or ""
    if missing:
        note_text = f"{note_text} Missing {', '.join(missing)}.".strip()
    trade_id = _stable_trade_id(source_sheet, source_row, symbol, side, trade_date, quantity, price)
    return {
        "trade_id": trade_id,
        "source_sheet": source_sheet,
        "source_row": int(source_row),
        "symbol": symbol,
        "side": side,
        "trade_date": trade_date,
        "quantity": quantity,
        "price": price,
        "amount": _amount_from_fields(quantity, price, amount),
        "sector": sector,
        "status": status_value,
        "notes": note_text or None,
    }


def parse_track_trades(excel_path: Path) -> pd.DataFrame:
    """Normalize the TRACK sheet into buy and sell events.

    The professor workbook stores lots in business form: buy date, optional sell
    date, quantity, acquisition price and mark-to-market or sale price. The MVP
    converts those lots into a ledger because every downstream calculation is
    easier to audit when buys and sells are explicit events.
    """
    if "TRACK" not in available_sheets(excel_path):
        return pd.DataFrame()

    table = _table_from_header(excel_path, "TRACK", {"Stock", "Buy", "Sell", "Shares", "PriceAcq"})
    if table.empty:
        return pd.DataFrame()

    events: list[dict[str, Any]] = []
    summary_labels = {"CASH", "INVESTED", "PORTFOLIO", "S&P500", "SP500"}
    for _, row in table.iterrows():
        symbol = _normalise_symbol(row.get("Stock"))
        if not symbol or symbol in summary_labels:
            continue

        source_row = int(row["_source_row"])
        buy_date = _to_date(row.get("Buy"))
        sell_date = _to_date(row.get("Sell"))
        quantity = _to_float(row.get("Shares")) or _to_float(row.get("Shares0"))
        acquisition_price = _to_float(row.get("PriceAcq"))
        amount = _to_float(row.get("Amount"))
        sell_or_mtm_price = _to_float(row.get("MtM"))
        sector = _normalise_label(row.get("Sector")) or None

        if buy_date:
            events.append(
                _trade_event(
                    source_sheet="TRACK",
                    source_row=source_row,
                    symbol=symbol,
                    side="BUY",
                    trade_date=buy_date,
                    quantity=quantity,
                    price=acquisition_price,
                    amount=amount,
                    sector=sector,
                    status="OPEN_LOT" if not sell_date else "CLOSED_LOT_BUY",
                )
            )

        if sell_date:
            notes = None if sell_or_mtm_price is not None else "Sell price missing; acquisition price used as fallback."
            sell_price = sell_or_mtm_price if sell_or_mtm_price is not None else acquisition_price
            events.append(
                _trade_event(
                    source_sheet="TRACK",
                    source_row=source_row,
                    symbol=symbol,
                    side="SELL",
                    trade_date=sell_date,
                    quantity=quantity,
                    price=sell_price,
                    amount=_amount_from_fields(quantity, sell_price, None),
                    sector=sector,
                    status="CLOSED_LOT_SELL" if sell_or_mtm_price is not None else "INCOMPLETE_SELL_PRICE",
                    notes=notes,
                )
            )

    return pd.DataFrame(events)


def parse_new_trades(excel_path: Path) -> pd.DataFrame:
    """Normalize the NEW TRADES sheet without writing back to Excel."""
    if "NEW TRADES" not in available_sheets(excel_path):
        return pd.DataFrame()

    table = _table_from_header(excel_path, "NEW TRADES", {"Stock", "Buy", "Sell", "Shares", "PriceAcq"})
    if table.empty:
        return pd.DataFrame()

    events: list[dict[str, Any]] = []
    for _, row in table.iterrows():
        symbol = _normalise_symbol(row.get("Stock"))
        source_row = int(row["_source_row"])
        buy_date = _to_date(row.get("Buy"))
        sell_date = _to_date(row.get("Sell"))
        quantity = _to_float(row.get("Shares")) or _to_float(row.get("Shares0"))
        price = _to_float(row.get("PriceAcq"))
        amount = _to_float(row.get("Amount"))
        sector = _normalise_label(row.get("Sector")) or None

        if buy_date:
            events.append(
                _trade_event(
                    source_sheet="NEW TRADES",
                    source_row=source_row,
                    symbol=symbol,
                    side="BUY",
                    trade_date=buy_date,
                    quantity=quantity,
                    price=price,
                    amount=amount,
                    sector=sector,
                    status="PENDING_EXCEL_IMPORT",
                )
            )
        if sell_date:
            events.append(
                _trade_event(
                    source_sheet="NEW TRADES",
                    source_row=source_row,
                    symbol=symbol,
                    side="SELL",
                    trade_date=sell_date,
                    quantity=quantity,
                    price=price,
                    amount=amount,
                    sector=sector,
                    status="PENDING_EXCEL_IMPORT",
                )
            )

    return pd.DataFrame(events)


def parse_portfolio_history(excel_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read local portfolio and benchmark history from the Excel cache."""
    if "Portfolio" not in available_sheets(excel_path):
        return pd.DataFrame(), pd.DataFrame()

    table = _table_from_header(excel_path, "Portfolio", {"Date", "Invested", "Cash"})
    if table.empty:
        return pd.DataFrame(), pd.DataFrame()

    total_column = "Portfolio" if "Portfolio" in table.columns else "Portoflio"
    snapshots = pd.DataFrame(
        {
            "date": table["Date"].map(_to_date),
            "invested_value": table.get("Invested").map(_to_float),
            "cash": table.get("Cash").map(_to_float),
            "total_portfolio_value": table.get(total_column).map(_to_float),
            "portfolio_return": table.get("Performance").map(_to_float),
        }
    ).dropna(subset=["date"])

    benchmark_prices = pd.DataFrame()
    if "S&P500" in table.columns:
        benchmark_prices = pd.DataFrame(
            {
                "date": table["Date"].map(_to_date),
                "benchmark": "S&P 500",
                "price": table["S&P500"].map(_to_float),
                "source": "Portfolio sheet",
            }
        ).dropna(subset=["date", "price"])

    if not benchmark_prices.empty:
        first_price = benchmark_prices["price"].iloc[0]
        snapshots["benchmark_return"] = np.where(
            first_price != 0,
            benchmark_prices.set_index("date").reindex(snapshots["date"])["price"].to_numpy() / first_price - 1,
            np.nan,
        )
    else:
        snapshots["benchmark_return"] = np.nan

    return snapshots, benchmark_prices


def parse_prices(excel_path: Path) -> pd.DataFrame:
    """Read cached local price observations from the wide Precios sheet."""
    if "Precios" not in available_sheets(excel_path):
        return pd.DataFrame(columns=["date", "symbol", "price", "source"])

    raw = _read_raw_sheet(excel_path, "Precios")
    if raw.shape[0] < 5 or raw.shape[1] < 2:
        return pd.DataFrame(columns=["date", "symbol", "price", "source"])

    rows: list[dict[str, Any]] = []
    for column in range(1, raw.shape[1]):
        symbol = _normalise_symbol(raw.iat[3, column])
        if not symbol:
            continue
        for row in range(4, raw.shape[0]):
            date_value = _to_date(raw.iat[row, 0])
            price = _to_float(raw.iat[row, column])
            if date_value and price is not None:
                rows.append({"date": date_value, "symbol": symbol, "price": price, "source": "Precios sheet"})

    prices = pd.DataFrame(rows, columns=["date", "symbol", "price", "source"])
    if prices.empty:
        return prices
    return prices.drop_duplicates(subset=["date", "symbol", "source"], keep="last")


def load_excel_model(excel_path: Path) -> dict[str, pd.DataFrame]:
    """Load every Excel-derived input needed to initialize the SQLite MVP."""
    snapshots, benchmark_prices = parse_portfolio_history(excel_path)
    return {
        "sheet_metadata": sheet_metadata(excel_path),
        "trades": parse_track_trades(excel_path),
        "prices": parse_prices(excel_path),
        "benchmark_prices": benchmark_prices,
        "portfolio_snapshots": snapshots,
        "new_trades_preview": parse_new_trades(excel_path),
    }

