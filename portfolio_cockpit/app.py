from __future__ import annotations

import json
import os
import uuid
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src import charts, data_loader, filters as display_filters, llm_report, market_data, metrics, portfolio_engine, reporting, storage, validation


APP_ROOT = Path(__file__).resolve().parent
EXCEL_PATH = APP_ROOT / "data" / "input" / "Portfolio Example JULIO.xlsx"
DB_PATH = APP_ROOT / "data" / "store" / "portfolio_mvp.sqlite"


st.set_page_config(page_title="Portfolio Management Cockpit", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.6rem; }
    .pipeline {
        border: 1px solid #d8dee4;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        background: #f6f8fa;
        font-weight: 600;
        text-align: center;
    }
    .section-note {
        border-left: 4px solid #667085;
        padding: 0.65rem 0.9rem;
        background: #f8fafc;
        color: #344054;
        margin-bottom: 1rem;
    }
    .small-muted { color: #667085; font-size: 0.9rem; }
    .kpi-card {
        border: 1px solid #d8dee4;
        border-radius: 8px;
        padding: 0.8rem 0.9rem;
        background: #ffffff;
        min-height: 96px;
    }
    .kpi-label { color: #667085; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0; }
    .kpi-value { color: #111827; font-size: 1.45rem; font-weight: 650; margin-top: 0.15rem; }
    .kpi-caption { color: #667085; font-size: 0.82rem; margin-top: 0.25rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _fmt_money(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"${float(value):,.0f}"


def _fmt_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.2%}"


def _fmt_num(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):,.2f}"


def is_public_demo_mode() -> bool:
    """Detect Streamlit Community Cloud and allow explicit local override for testing."""
    explicit = os.getenv("PORTFOLIO_COCKPIT_PUBLIC_DEMO", "").lower()
    if explicit in {"1", "true", "yes"}:
        return True
    if explicit in {"0", "false", "no"}:
        return False
    return Path("/mount/src").exists() or os.getenv("HOME") == "/home/adminuser" or bool(os.getenv("STREAMLIT_CLOUD"))


def kpi_card(label: str, value: str, caption: str = "") -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def teaching_note(text: str, enabled: bool) -> None:
    if enabled:
        st.markdown(f'<div class="section-note">{text}</div>', unsafe_allow_html=True)


def ensure_database() -> str:
    if not storage.database_exists(DB_PATH):
        storage.reset_database_from_excel(EXCEL_PATH, DB_PATH)
        return "initialized from Excel"
    return "loaded from existing SQLite"


def _session_trades() -> pd.DataFrame:
    records = st.session_state.get("session_trades", [])
    return pd.DataFrame(records) if records else pd.DataFrame()


def load_context(public_demo_mode: bool = False) -> dict[str, Any]:
    trades = storage.get_trades(DB_PATH)
    prices = storage.get_prices(DB_PATH)
    benchmark_prices = storage.get_benchmark_prices(DB_PATH)
    positions = storage.get_positions(DB_PATH)
    snapshots = storage.get_portfolio_snapshots(DB_PATH)
    session_trades = _session_trades() if public_demo_mode else pd.DataFrame()
    if public_demo_mode and not session_trades.empty:
        trades = pd.concat([trades, session_trades], ignore_index=True)
        recalculated = portfolio_engine.recompute_all_after_trade(
            trades,
            prices=prices,
            benchmark_prices=benchmark_prices,
            existing_snapshots=snapshots,
            initial_cash=storage.get_initial_cash(DB_PATH),
        )
        positions = recalculated["positions"]
        snapshots = recalculated["portfolio_snapshots"]
    issues = validation.run_data_quality_checks(trades, positions, snapshots, prices, benchmark_prices)
    risk = compute_risk_metrics(snapshots, positions)
    benchmark = compute_benchmark_metrics(snapshots, benchmark_prices, risk)
    risk.update(
        {
            "beta": benchmark.get("beta"),
            "tracking_error": benchmark.get("tracking_error"),
            "information_ratio": benchmark.get("information_ratio"),
            "turnover_proxy": metrics.turnover_proxy(trades, risk.get("average_portfolio_value")),
        }
    )
    summary = compute_portfolio_summary(trades, positions, snapshots, prices, risk)
    recent_trades = trades.sort_values("trade_date", ascending=False).head(12) if not trades.empty else pd.DataFrame()
    return {
        "trades": trades,
        "prices": prices,
        "benchmark_prices": benchmark_prices,
        "positions": positions,
        "snapshots": snapshots,
        "issues": issues,
        "risk": risk,
        "benchmark": benchmark,
        "summary": summary,
        "recent_trades": recent_trades,
        "session_trades": session_trades,
    }


def compute_risk_metrics(snapshots: pd.DataFrame, positions: pd.DataFrame, risk_free_rate: float = 0.0) -> dict[str, Any]:
    if snapshots is None or snapshots.empty:
        returns = pd.Series(dtype="float64")
        values = pd.Series(dtype="float64")
    else:
        ordered = snapshots.copy()
        ordered["date"] = pd.to_datetime(ordered["date"], errors="coerce")
        ordered = ordered.dropna(subset=["date"]).sort_values("date")
        values = pd.to_numeric(ordered["total_portfolio_value"], errors="coerce")
        returns = metrics.daily_returns(values)

    concentration = metrics.concentration_metrics(
        pd.to_numeric(positions["weight"], errors="coerce") if positions is not None and not positions.empty else []
    )
    average_value = float(values.mean()) if not values.empty else np.nan
    return {
        "cumulative_return": metrics.cumulative_return(values),
        "annualized_return": metrics.annualized_return(returns),
        "annualized_volatility": metrics.annualized_volatility(returns),
        "sharpe_ratio": metrics.sharpe_ratio(returns, risk_free_rate=risk_free_rate),
        "sortino_ratio": metrics.sortino_ratio(returns, risk_free_rate=risk_free_rate),
        "max_drawdown": metrics.max_drawdown(values),
        "best_day": float(returns.max()) if not returns.empty else np.nan,
        "worst_day": float(returns.min()) if not returns.empty else np.nan,
        "hit_ratio": float((returns > 0).mean()) if not returns.empty else np.nan,
        "average_portfolio_value": average_value,
        **concentration,
    }


def compute_benchmark_metrics(
    snapshots: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
    risk: dict[str, Any],
) -> dict[str, Any]:
    benchmark_return = np.nan
    if benchmark_prices is not None and not benchmark_prices.empty:
        benchmark_return = metrics.cumulative_return(pd.to_numeric(benchmark_prices["price"], errors="coerce"))
    elif snapshots is not None and not snapshots.empty and "benchmark_return" in snapshots:
        valid = pd.to_numeric(snapshots["benchmark_return"], errors="coerce").dropna()
        benchmark_return = float(valid.iloc[-1]) if not valid.empty else np.nan
    portfolio_return = risk.get("cumulative_return", np.nan)
    excess = portfolio_return - benchmark_return if not pd.isna(portfolio_return) and not pd.isna(benchmark_return) else np.nan
    portfolio_daily = pd.Series(dtype="float64")
    benchmark_daily = pd.Series(dtype="float64")
    if snapshots is not None and not snapshots.empty:
        ordered = snapshots.copy()
        ordered["date"] = pd.to_datetime(ordered["date"], errors="coerce")
        ordered = ordered.dropna(subset=["date"]).sort_values("date")
        portfolio_values = pd.to_numeric(ordered["total_portfolio_value"], errors="coerce")
        portfolio_daily = portfolio_values.pct_change().dropna()
        if "benchmark_return" in ordered.columns and ordered["benchmark_return"].notna().any():
            benchmark_index = 1 + pd.to_numeric(ordered["benchmark_return"], errors="coerce")
            benchmark_daily = benchmark_index.pct_change().dropna()
    if benchmark_daily.empty and benchmark_prices is not None and not benchmark_prices.empty:
        benchmark_daily = metrics.daily_returns(pd.to_numeric(benchmark_prices["price"], errors="coerce"))
    return {
        "cumulative_benchmark_return": benchmark_return,
        "excess_return": excess,
        "beta": metrics.beta_vs_benchmark(portfolio_daily.reset_index(drop=True), benchmark_daily.reset_index(drop=True)),
        "tracking_error": metrics.tracking_error(portfolio_daily.reset_index(drop=True), benchmark_daily.reset_index(drop=True)),
        "information_ratio": metrics.information_ratio(portfolio_daily.reset_index(drop=True), benchmark_daily.reset_index(drop=True)),
    }


def compute_portfolio_summary(
    trades: pd.DataFrame,
    positions: pd.DataFrame,
    snapshots: pd.DataFrame,
    prices: pd.DataFrame,
    risk: dict[str, Any],
) -> dict[str, Any]:
    latest_snapshot = None
    if snapshots is not None and not snapshots.empty:
        ordered = snapshots.copy()
        ordered["date"] = pd.to_datetime(ordered["date"], errors="coerce")
        ordered = ordered.dropna(subset=["date"]).sort_values("date")
        latest_snapshot = ordered.iloc[-1].to_dict()

    pnl = portfolio_engine.compute_realized_unrealized_pnl(trades, prices) if trades is not None and not trades.empty else {}
    invested = float(positions["market_value"].sum()) if positions is not None and not positions.empty else 0.0
    cash = float(latest_snapshot.get("cash")) if latest_snapshot else storage.get_initial_cash(DB_PATH)
    total = float(latest_snapshot.get("total_portfolio_value")) if latest_snapshot else invested + cash
    concentration = metrics.concentration_metrics(
        pd.to_numeric(positions["weight"], errors="coerce") if positions is not None and not positions.empty else []
    )
    return {
        "as_of_date": latest_snapshot.get("date").date().isoformat() if latest_snapshot and hasattr(latest_snapshot.get("date"), "date") else None,
        "total_portfolio_value": total,
        "cash": cash,
        "invested_value": float(latest_snapshot.get("invested_value")) if latest_snapshot else invested,
        "total_pnl": total - storage.get_initial_cash(DB_PATH),
        "realized_pnl": pnl.get("realized_pnl", np.nan),
        "unrealized_pnl": pnl.get("unrealized_pnl", np.nan),
        "cumulative_return": risk.get("cumulative_return"),
        "open_positions": 0 if positions is None or positions.empty else int(len(positions)),
        "top_5_concentration": concentration.get("top_5_weight"),
        "largest_single_name_exposure": concentration.get("largest_weight"),
    }


def impact_row(summary: dict[str, Any], risk: dict[str, Any]) -> dict[str, Any]:
    exposure = summary["invested_value"] / summary["total_portfolio_value"] if summary.get("total_portfolio_value") else np.nan
    return {
        "cash": summary.get("cash"),
        "invested_value": summary.get("invested_value"),
        "total_portfolio_value": summary.get("total_portfolio_value"),
        "exposure": exposure,
        "top_5_concentration": summary.get("top_5_concentration"),
        "sharpe_ratio": risk.get("sharpe_ratio"),
        "sortino_ratio": risk.get("sortino_ratio"),
        "max_drawdown": risk.get("max_drawdown"),
    }


def add_position_contributions(positions: pd.DataFrame) -> pd.DataFrame:
    if positions is None or positions.empty:
        return pd.DataFrame() if positions is None else positions.copy()
    data = positions.copy()
    market_value_total = pd.to_numeric(data["market_value"], errors="coerce").sum()
    absolute_pnl_total = pd.to_numeric(data["unrealized_pnl"], errors="coerce").abs().sum()
    data["market_value_contribution"] = np.where(market_value_total != 0, data["market_value"] / market_value_total, np.nan)
    data["absolute_unrealized_pnl_contribution"] = np.where(
        absolute_pnl_total != 0,
        data["unrealized_pnl"].abs() / absolute_pnl_total,
        np.nan,
    )
    return data


def recent_trade_impact_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame(columns=["side", "trade_count", "gross_amount", "symbols"])
    data = trades.copy()
    data["amount"] = pd.to_numeric(data["amount"], errors="coerce").fillna(0)
    grouped = data.groupby("side", as_index=False).agg(
        trade_count=("trade_id", "count"),
        gross_amount=("amount", "sum"),
        symbols=("symbol", lambda values: ", ".join(sorted(set(map(str, values)))[:8])),
    )
    return grouped


def price_source_coverage(prices: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "symbol",
        "latest_price_date",
        "latest_price",
        "latest_source",
        "latest_source_type",
        "yahoo_rows",
        "excel_rows",
        "other_rows",
        "observation_count",
    ]
    if prices is None or prices.empty:
        return pd.DataFrame(columns=columns)
    data = prices.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    data["symbol"] = data["symbol"].astype(str).str.upper().str.strip()
    data = data.dropna(subset=["date", "symbol", "price"])
    if data.empty:
        return pd.DataFrame(columns=columns)
    effective = portfolio_engine.get_effective_daily_prices(data)
    effective["date"] = pd.to_datetime(effective["date"], errors="coerce")
    latest = effective.sort_values(["date", "source"]).groupby("symbol", as_index=False).tail(1)
    source_counts = data.assign(
        is_yahoo=data["source"].astype(str).str.lower().eq("yahoo finance"),
        is_excel=data["source"].astype(str).str.lower().eq("precios sheet"),
    )
    counts = source_counts.groupby("symbol", as_index=False).agg(
        observation_count=("price", "size"),
        yahoo_rows=("is_yahoo", "sum"),
        excel_rows=("is_excel", "sum"),
    )
    counts["other_rows"] = counts["observation_count"] - counts["yahoo_rows"] - counts["excel_rows"]
    coverage = latest.merge(counts, on="symbol", how="left")
    coverage = coverage.rename(columns={"date": "latest_price_date", "price": "latest_price", "source": "latest_source"})
    coverage["latest_price_date"] = coverage["latest_price_date"].dt.date.astype(str)
    coverage["latest_source_type"] = np.select(
        [
            coverage["latest_source"].astype(str).str.lower().eq("yahoo finance"),
            coverage["latest_source"].astype(str).str.lower().eq("precios sheet"),
        ],
        ["Yahoo Finance", "Precios sheet"],
        default="Other",
    )
    open_symbols = market_data.get_open_position_symbols(positions)
    if open_symbols:
        coverage = coverage[coverage["symbol"].isin(open_symbols)]
    return coverage[columns].sort_values("symbol").reset_index(drop=True)


def make_trade_record(symbol: str, side: str, trade_date: date, quantity: float, price: float, sector: str, notes: str) -> dict[str, Any]:
    return {
        "trade_id": uuid.uuid4().hex,
        "source_sheet": "MANUAL",
        "source_row": None,
        "symbol": symbol.upper().strip(),
        "side": side.upper(),
        "trade_date": trade_date.isoformat(),
        "quantity": quantity,
        "price": price,
        "amount": quantity * price,
        "sector": sector.strip() or None,
        "status": "MANUAL_APPROVED",
        "notes": notes or None,
    }


def simulated_context(base_context: dict[str, Any], trade: dict[str, Any]) -> dict[str, Any]:
    simulated = portfolio_engine.simulate_trade_impact(
        base_context["trades"],
        trade,
        prices=base_context["prices"],
        benchmark_prices=base_context["benchmark_prices"],
        existing_snapshots=base_context["snapshots"],
        initial_cash=storage.get_initial_cash(DB_PATH),
    )
    sim_risk = compute_risk_metrics(simulated["portfolio_snapshots"], simulated["positions"])
    sim_benchmark = compute_benchmark_metrics(simulated["portfolio_snapshots"], base_context["benchmark_prices"], sim_risk)
    sim_risk.update(
        {
            "beta": sim_benchmark.get("beta"),
            "tracking_error": sim_benchmark.get("tracking_error"),
            "information_ratio": sim_benchmark.get("information_ratio"),
            "turnover_proxy": metrics.turnover_proxy(simulated["trades"], sim_risk.get("average_portfolio_value")),
        }
    )
    sim_summary = compute_portfolio_summary(simulated["trades"], simulated["positions"], simulated["portfolio_snapshots"], base_context["prices"], sim_risk)
    return {"summary": sim_summary, "risk": sim_risk, "benchmark": sim_benchmark, **simulated}


def recent_trades_for_prompt(recent_trades: pd.DataFrame) -> list[dict[str, Any]]:
    if recent_trades is None or recent_trades.empty:
        return []
    data = recent_trades.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data["amount"] = pd.to_numeric(data["amount"], errors="coerce")
    data["side"] = data["side"].astype(str).str.upper()
    data["symbol"] = data["symbol"].astype(str).str.upper()
    return [
        {
            "period_start": data["trade_date"].min().date().isoformat() if data["trade_date"].notna().any() else None,
            "period_end": data["trade_date"].max().date().isoformat() if data["trade_date"].notna().any() else None,
            "trade_count": int(len(data)),
            "buy_count": int((data["side"] == "BUY").sum()),
            "sell_count": int((data["side"] == "SELL").sum()),
            "gross_buy_amount": float(data.loc[data["side"] == "BUY", "amount"].fillna(0).sum()),
            "gross_sell_amount": float(data.loc[data["side"] == "SELL", "amount"].fillna(0).sum()),
            "symbols_touched": sorted(data["symbol"].dropna().unique().tolist())[:10],
            "sectors_touched": sorted(data.get("sector", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())[:10],
        }
    ]


def quality_warnings_for_prompt(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "severity": issue["severity"],
            "check_name": issue["check_name"],
            "message": issue["message"],
            "recommendation": issue["recommendation"],
        }
        for issue in issues[:15]
    ]


public_demo_mode = is_public_demo_mode()
db_status = ensure_database()
context = load_context(public_demo_mode)

st.title("Portfolio Management Cockpit")
st.caption("Financial Engineering course MVP: Excel as operational input, SQLite as system of record, Python as calculation engine, Streamlit as decision interface.")

if public_demo_mode:
    st.info("Public demo mode: this Streamlit Community Cloud review app may not retain runtime changes. Trade commits are treated as session-level demo actions; deterministic reporting remains available.")

with st.sidebar:
    st.header("Cockpit Status")
    teaching_mode = st.toggle("Teaching Mode", value=True, help="Switch off for a cleaner analyst cockpit.")
    st.write(f"Database: {db_status}")
    st.write(f"Excel input: `{EXCEL_PATH.name}`")
    st.write(f"SQLite: `{DB_PATH.name}`")
    st.write(f"Mode: `{'Public demo' if public_demo_mode else 'Local SQLite'}`")
    st.divider()
    st.subheader("Global Display Filters")
    st.caption("Filters affect displayed charts and tables. Headline portfolio KPIs remain full-portfolio unless a section states otherwise.")
    if not context["snapshots"].empty:
        snapshot_dates = pd.to_datetime(context["snapshots"]["date"], errors="coerce").dropna()
        min_date = snapshot_dates.min().date()
        max_date = snapshot_dates.max().date()
        date_selection = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        if isinstance(date_selection, tuple) and len(date_selection) == 2:
            filter_date_range = (pd.Timestamp(date_selection[0]), pd.Timestamp(date_selection[1]))
        else:
            filter_date_range = (pd.Timestamp(min_date), pd.Timestamp(max_date))
    else:
        filter_date_range = None
    symbols = sorted(set(context["trades"].get("symbol", pd.Series(dtype=str)).dropna().astype(str).str.upper()) | set(context["positions"].get("symbol", pd.Series(dtype=str)).dropna().astype(str).str.upper()))
    sectors = sorted(set(context["trades"].get("sector", pd.Series(dtype=str)).fillna("Unclassified").astype(str)) | set(context["positions"].get("sector", pd.Series(dtype=str)).fillna("Unclassified").astype(str)))
    selected_symbols = st.multiselect("Symbol", symbols)
    selected_sectors = st.multiselect("Sector", sectors)
    selected_sides = st.multiselect("Trade side", ["BUY", "SELL"], default=["BUY", "SELL"])
    open_positions_only = st.checkbox("Show open positions only", value=True)
    st.divider()
    if teaching_mode:
        st.write("Navigation is organized as the same workflow students follow in class.")

filtered_trades = display_filters.filter_trades(context["trades"], filter_date_range, selected_symbols, selected_sectors, selected_sides)
filtered_positions = add_position_contributions(display_filters.filter_positions(context["positions"], selected_symbols, selected_sectors, open_positions_only))
filtered_snapshots = display_filters.filter_snapshots(context["snapshots"], filter_date_range)
full_positions_with_contributions = add_position_contributions(context["positions"])

summary = context["summary"]
benchmark = context["benchmark"]
risk = context["risk"]
issue_frame = validation.issues_to_frame(context["issues"])
error_count = int((issue_frame["severity"] == "Error").sum()) if not issue_frame.empty else 0
warning_count = int((issue_frame["severity"] == "Warning").sum()) if not issue_frame.empty else 0
data_quality_status = "Blocked" if error_count else ("Review" if warning_count else "Clean")

st.markdown("### Executive Overview")
overview_cols = st.columns(5)
with overview_cols[0]:
    kpi_card("Total value", _fmt_money(summary["total_portfolio_value"]), "Full portfolio")
with overview_cols[1]:
    kpi_card("Cash", _fmt_money(summary["cash"]), "Liquidity buffer")
with overview_cols[2]:
    kpi_card("Invested", _fmt_money(summary["invested_value"]), "Market exposure")
with overview_cols[3]:
    kpi_card("Portfolio return", _fmt_pct(risk["cumulative_return"]), "Since first snapshot")
with overview_cols[4]:
    kpi_card("Excess return", _fmt_pct(benchmark["excess_return"]), "Versus S&P 500")

overview_cols = st.columns(5)
with overview_cols[0]:
    kpi_card("S&P 500 return", _fmt_pct(benchmark["cumulative_benchmark_return"]), "Local benchmark")
with overview_cols[1]:
    kpi_card("Sharpe", _fmt_num(risk["sharpe_ratio"]), "Risk-adjusted")
with overview_cols[2]:
    kpi_card("Max drawdown", _fmt_pct(risk["max_drawdown"]), "Peak to trough")
with overview_cols[3]:
    kpi_card("Positions", str(summary["open_positions"]), "Open holdings")
with overview_cols[4]:
    kpi_card("Top 5 concentration", _fmt_pct(summary["top_5_concentration"]), f"Data quality: {data_quality_status}")

attention_items = []
if error_count:
    attention_items.append(f"{error_count} blocking data-quality checks require review before relying on performance metrics.")
if warning_count:
    attention_items.append(f"{warning_count} warning-level data-quality checks should be reconciled.")
if not pd.isna(summary.get("top_5_concentration")) and summary["top_5_concentration"] > 0.5:
    attention_items.append("Top five concentration is elevated; review single-name and sector exposure.")
if not pd.isna(risk.get("max_drawdown")) and risk["max_drawdown"] < -0.1:
    attention_items.append("Drawdown profile deserves portfolio committee attention.")
if not attention_items:
    attention_items.append("No blocking control issue is visible in the summary view; continue with lineage and attribution checks.")
st.markdown("**What requires attention?**")
for item in attention_items:
    st.write(f"- {item}")

tabs = st.tabs(
    [
        "Executive Snapshot",
        "1. Excel to Portfolio System",
        "2. Data Quality",
        "3. Current Portfolio",
        "4. Performance & Risk",
        "5. Performance Attribution",
        "6. Add New Trade",
        "7. Executive Report",
        "8. From MVP to Production",
    ]
)

with tabs[0]:
    st.subheader("Portfolio Committee Snapshot")
    st.plotly_chart(charts.portfolio_value_chart(filtered_snapshots), width="stretch", key="executive_portfolio_value_chart")
    st.subheader("Market Data Refresh")
    st.warning(
        "Yahoo Finance refresh is optional and intended for educational/demo use. The Excel remains the initial source; Yahoo prices are used only as a market-data update layer."
    )
    open_symbols_for_refresh = market_data.get_open_position_symbols(context["positions"])
    coverage = price_source_coverage(context["prices"], context["positions"])
    benchmark_latest_date = market_data.get_last_benchmark_date(context["benchmark_prices"])
    snapshot_dates = pd.to_datetime(context["snapshots"].get("date"), errors="coerce").dropna() if not context["snapshots"].empty else pd.Series(dtype="datetime64[ns]")
    latest_snapshot_date = snapshot_dates.max().date().isoformat() if not snapshot_dates.empty else "n/a"
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Open tickers eligible", len(open_symbols_for_refresh))
    col_b.metric("S&P 500 latest date", benchmark_latest_date.isoformat() if benchmark_latest_date else "n/a")
    col_c.metric("Benchmark source", "^GSPC / S&P 500")
    col_d.metric("Latest portfolio snapshot", latest_snapshot_date)
    st.caption("Only tickers with open positions are refreshed. Closed historical trade symbols are not requested from Yahoo Finance.")
    st.write("Open position tickers:", ", ".join(open_symbols_for_refresh) if open_symbols_for_refresh else "No open tickers available.")
    st.subheader("Price source coverage")
    st.dataframe(coverage, hide_index=True, width="stretch")
    if public_demo_mode:
        st.info("Public demo mode: a refresh may update the temporary Streamlit runtime database, but it is not guaranteed to persist after the app restarts.")
    if st.button("Refresh daily market prices from Yahoo Finance"):
        with st.spinner("Refreshing Yahoo Finance daily prices for open positions and S&P 500..."):
            st.session_state["last_market_refresh_result"] = market_data.refresh_open_position_prices(DB_PATH)
        st.rerun()
    if "last_market_refresh_result" in st.session_state:
        refresh_result = st.session_state["last_market_refresh_result"]
        st.write("Latest refresh status:", refresh_result.get("status", "unknown"))
        before_snapshot = refresh_result.get("previous_latest_snapshot_date") or refresh_result.get("latest_snapshot_date_before_refresh")
        after_snapshot = refresh_result.get("new_latest_snapshot_date") or refresh_result.get("latest_snapshot_date_after_refresh")
        snapshots_extended = bool(refresh_result.get("snapshots_extended")) or bool(before_snapshot and after_snapshot and pd.to_datetime(after_snapshot) > pd.to_datetime(before_snapshot))
        refresh_cols = st.columns(3)
        refresh_cols[0].metric("Previous snapshot date", before_snapshot or "n/a")
        refresh_cols[1].metric("New snapshot date", after_snapshot or "n/a")
        refresh_cols[2].metric("Snapshot rows", refresh_result.get("snapshot_rows_after", "n/a"))
        st.write(f"Snapshots extended after refresh: {'Yes' if snapshots_extended else 'No'}")
        if (
            refresh_result.get("status") == "refreshed"
            and not snapshots_extended
            and (refresh_result.get("prices_upserted", 0) or refresh_result.get("benchmark_upserted", 0))
        ):
            st.warning(
                "New market rows were stored, but snapshots were not extended. This usually means the new rows did not fall after the current latest portfolio snapshot or did not provide a valuation-compatible date for open positions."
            )
        if refresh_result.get("messages"):
            for message in refresh_result["messages"]:
                st.write(f"- {message}")
        rows_by_symbol = refresh_result.get("rows_by_symbol", {})
        if rows_by_symbol:
            st.dataframe(
                pd.DataFrame([{"symbol": symbol, "rows_added_or_updated": rows} for symbol, rows in rows_by_symbol.items()]),
                hide_index=True,
                width="stretch",
            )
        st.metric("S&P 500 rows added/updated", refresh_result.get("benchmark_upserted", 0))
        if refresh_result.get("failed_symbols"):
            st.warning("Some Yahoo Finance requests failed. The app kept running and used local data where available.")
            st.dataframe(pd.DataFrame(refresh_result["failed_symbols"]), hide_index=True, width="stretch")

    st.dataframe(
        full_positions_with_contributions[["symbol", "sector", "market_value", "unrealized_pnl", "weight", "market_value_contribution", "absolute_unrealized_pnl_contribution"]]
        if not full_positions_with_contributions.empty
        else full_positions_with_contributions,
        hide_index=True,
        width="stretch",
    )

with tabs[1]:
    teaching_note(
        "The Excel is the operational input. The professional workflow starts when we separate raw data, validation, calculations, storage and reporting.",
        teaching_mode,
    )
    if teaching_mode:
        st.markdown(
            '<div class="pipeline">Excel -> SQLite ledger -> positions -> valuation -> performance -> risk -> report</div>',
            unsafe_allow_html=True,
        )
    st.write("")

    config = storage.get_configuration(DB_PATH)
    loaded_sheets = config.get("loaded_sheets")
    if loaded_sheets:
        sheet_frame = pd.DataFrame(json.loads(loaded_sheets))
    else:
        sheet_frame = data_loader.sheet_metadata(EXCEL_PATH)
    st.subheader("Loaded workbook structure")
    st.dataframe(sheet_frame, hide_index=True, width="stretch")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Database status", db_status)
    col_b.metric("Ledger events", len(context["trades"]))
    col_c.metric("Snapshots", len(context["snapshots"]))

    st.caption("Reset rebuilds the local SQLite database from the original Excel input. Market refresh updates SQLite; it does not modify the Excel.")
    if st.button("Reset to original Excel state"):
        if public_demo_mode:
            st.session_state["session_trades"] = []
            st.info("Public demo mode reset: session-level demo trades were cleared. The bundled Excel source remains unchanged.")
        else:
            storage.reset_database_from_excel(EXCEL_PATH, DB_PATH)
            storage.write_audit_log("RESET", "Demo database reset from immutable Excel input.", DB_PATH)
            st.success("Local SQLite database rebuilt from the original Excel input. Yahoo-updated prices and extended snapshots were removed from the live demo database.")
        st.rerun()

    st.subheader("Initialization and reset audit log")
    audit = storage.get_audit_log(DB_PATH)
    if not audit.empty:
        init_audit = audit[audit["event_type"].isin(["INITIALIZE", "RESET"])]
        st.dataframe(init_audit, hide_index=True, width="stretch")
    else:
        st.info("No audit events have been recorded yet.")

with tabs[2]:
    teaching_note(
        "Before analysing performance, a professional analyst checks whether the data can be trusted.",
        teaching_mode,
    )
    severity_counts = issue_frame["severity"].value_counts() if not issue_frame.empty else pd.Series(dtype=int)
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Errors", int(severity_counts.get("Error", 0)))
    col_b.metric("Warnings", int(severity_counts.get("Warning", 0)))
    col_c.metric("Info notes", int(severity_counts.get("Info", 0)))
    st.subheader("Control summary")
    st.dataframe(issue_frame, hide_index=True, width="stretch")
    st.subheader("Data lineage")
    lineage = validation.lineage_frame(context["issues"], context["trades"])
    st.dataframe(lineage, hide_index=True, width="stretch")

with tabs[3]:
    teaching_note(
        "This is the portfolio manager's current decision base: what we own, where the exposure is, and where risk is concentrated.",
        teaching_mode,
    )
    cols = st.columns(4)
    cols[0].metric("Total portfolio value", _fmt_money(summary["total_portfolio_value"]))
    cols[1].metric("Cash", _fmt_money(summary["cash"]))
    cols[2].metric("Invested value", _fmt_money(summary["invested_value"]))
    cols[3].metric("Total P&L", _fmt_money(summary["total_pnl"]))

    cols = st.columns(4)
    cols[0].metric("Realized P&L", _fmt_money(summary["realized_pnl"]))
    cols[1].metric("Unrealized P&L", _fmt_money(summary["unrealized_pnl"]))
    cols[2].metric("Cumulative return", _fmt_pct(summary["cumulative_return"]))
    cols[3].metric("Open positions", summary["open_positions"])

    cols = st.columns(2)
    cols[0].metric("Top 5 concentration", _fmt_pct(summary["top_5_concentration"]))
    cols[1].metric("Largest single-name exposure", _fmt_pct(summary["largest_single_name_exposure"]))

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.plotly_chart(charts.allocation_by_asset_chart(filtered_positions), width="stretch", key="current_allocation_asset_chart")
        st.plotly_chart(charts.cash_vs_invested_chart(context["snapshots"].sort_values("date").iloc[-1] if not context["snapshots"].empty else {}), width="stretch", key="current_cash_invested_chart")
    with chart_cols[1]:
        st.plotly_chart(charts.allocation_by_sector_chart(filtered_positions), width="stretch", key="current_allocation_sector_chart")
        st.plotly_chart(charts.top_positions_chart(filtered_positions, "market_value", "Top contributors to market value"), width="stretch", key="current_top_market_value_chart")

    st.plotly_chart(charts.top_positions_chart(filtered_positions, "unrealized_pnl", "Top contributors to unrealized P&L"), width="stretch", key="current_top_unrealized_pnl_chart")
    st.subheader("Current positions drill-down")
    position_columns = ["symbol", "sector", "quantity", "average_cost", "latest_price", "market_value", "unrealized_pnl", "weight", "market_value_contribution", "absolute_unrealized_pnl_contribution"]
    st.dataframe(filtered_positions[position_columns] if not filtered_positions.empty else filtered_positions, hide_index=True, width="stretch")

with tabs[4]:
    teaching_note(
        "A portfolio can make money and still be a poor portfolio if the risk-adjusted performance, drawdown or concentration profile is weak.",
        teaching_mode,
    )
    risk_free_rate = st.number_input("Annual risk-free rate", min_value=0.0, max_value=0.20, value=0.0, step=0.005, format="%.3f")
    display_risk = compute_risk_metrics(filtered_snapshots, filtered_positions, risk_free_rate=risk_free_rate)
    display_benchmark = compute_benchmark_metrics(filtered_snapshots, context["benchmark_prices"], display_risk)
    display_risk.update(
        {
            "beta": display_benchmark.get("beta"),
            "tracking_error": display_benchmark.get("tracking_error"),
            "information_ratio": display_benchmark.get("information_ratio"),
            "turnover_proxy": metrics.turnover_proxy(filtered_trades, display_risk.get("average_portfolio_value")),
        }
    )

    cols = st.columns(4)
    cols[0].metric("Cumulative portfolio return", _fmt_pct(display_risk["cumulative_return"]))
    cols[1].metric("Cumulative S&P 500 return", _fmt_pct(display_benchmark["cumulative_benchmark_return"]))
    cols[2].metric("Excess return", _fmt_pct(display_benchmark["excess_return"]))
    cols[3].metric("Annualized return", _fmt_pct(display_risk["annualized_return"]))

    cols = st.columns(4)
    cols[0].metric("Annualized volatility", _fmt_pct(display_risk["annualized_volatility"]))
    cols[1].metric("Sharpe ratio", _fmt_num(display_risk["sharpe_ratio"]))
    cols[2].metric("Sortino ratio", _fmt_num(display_risk["sortino_ratio"]))
    cols[3].metric("Maximum drawdown", _fmt_pct(display_risk["max_drawdown"]))

    cols = st.columns(4)
    cols[0].metric("Beta vs S&P 500", _fmt_num(display_risk["beta"]))
    cols[1].metric("Tracking error", _fmt_pct(display_risk["tracking_error"]))
    cols[2].metric("Information ratio", _fmt_num(display_risk["information_ratio"]))
    cols[3].metric("Turnover proxy", _fmt_pct(display_risk["turnover_proxy"]))

    cols = st.columns(4)
    cols[0].metric("Best day", _fmt_pct(display_risk["best_day"]))
    cols[1].metric("Worst day", _fmt_pct(display_risk["worst_day"]))
    cols[2].metric("Hit ratio", _fmt_pct(display_risk["hit_ratio"]))
    cols[3].metric("Effective positions", _fmt_num(display_risk["effective_number_positions"]))

    if context["benchmark_prices"].empty:
        st.warning("S&P 500 benchmark data is unavailable locally. The app continues without internet access.")

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.plotly_chart(charts.portfolio_value_chart(filtered_snapshots), width="stretch", key="risk_portfolio_value_chart")
        st.plotly_chart(charts.drawdown_chart(filtered_snapshots), width="stretch", key="risk_drawdown_chart")
    with chart_cols[1]:
        st.plotly_chart(charts.cumulative_return_vs_benchmark_chart(filtered_snapshots), width="stretch", key="risk_cumulative_vs_benchmark_chart")
        st.plotly_chart(charts.daily_returns_time_series_chart(filtered_snapshots), width="stretch", key="risk_daily_returns_chart")

with tabs[5]:
    st.subheader("Performance Attribution")
    teaching_note(
        "Attribution links the current exposure base to where unrealized P&L is being created or lost.",
        teaching_mode,
    )
    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.plotly_chart(charts.top_positions_chart(filtered_positions, "unrealized_pnl", "Top contributors to unrealized P&L"), width="stretch", key="attribution_top_contributors_chart")
    with chart_cols[1]:
        detractors = filtered_positions.copy()
        if not detractors.empty:
            detractors["pnl_detractor"] = -pd.to_numeric(detractors["unrealized_pnl"], errors="coerce")
        st.plotly_chart(charts.top_positions_chart(detractors, "pnl_detractor", "Top detractors by unrealized P&L"), width="stretch", key="attribution_top_detractors_chart")
    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.plotly_chart(charts.sector_pnl_chart(filtered_positions), width="stretch", key="attribution_sector_pnl_chart")
    with chart_cols[1]:
        st.plotly_chart(charts.allocation_vs_pnl_chart(filtered_positions), width="stretch", key="attribution_allocation_vs_pnl_chart")
    st.subheader("Recent trades impact summary")
    st.dataframe(recent_trade_impact_summary(filtered_trades), hide_index=True, width="stretch")

with tabs[6]:
    teaching_note(
        "This is the moment where the MVP stops being a static dashboard and becomes an operating tool: new trades enter the system and the portfolio is recalculated.",
        teaching_mode,
    )
    st.subheader("Manual trade ticket")
    with st.form("manual_trade_form"):
        col_a, col_b, col_c = st.columns(3)
        symbol = col_a.text_input("Symbol").upper().strip()
        side = col_b.selectbox("Side", ["BUY", "SELL"])
        trade_date = col_c.date_input("Date", value=date.today())
        col_d, col_e, col_f = st.columns(3)
        quantity = col_d.number_input("Quantity", min_value=0.0, value=0.0, step=1.0)
        price = col_e.number_input("Price", min_value=0.0, value=0.0, step=0.01)
        sector = col_f.text_input("Sector").strip()
        notes = st.text_area("Notes", value="", height=80)
        col_sim, col_commit = st.columns(2)
        simulate_clicked = col_sim.form_submit_button("Simulate trade impact")
        commit_clicked = col_commit.form_submit_button("Commit trade")

    if simulate_clicked or commit_clicked:
        before = load_context(public_demo_mode)
        trade = make_trade_record(symbol, side, trade_date, quantity, price, sector, notes)
        manual_issues = validation.validate_manual_trade(trade, before["positions"])
        errors = [issue for issue in manual_issues if issue["severity"] == "Error"]
        if errors:
            for issue in errors:
                st.error(issue["message"])
        else:
            simulation = simulated_context(before, trade)
            st.session_state["last_impact"] = pd.DataFrame(
                [impact_row(before["summary"], before["risk"]), impact_row(simulation["summary"], simulation["risk"])],
                index=["Before", "After"],
            )
            st.session_state["pending_trade"] = trade
            if simulate_clicked:
                st.info("Simulation complete. No database write was performed.")
            if commit_clicked:
                if public_demo_mode:
                    session_records = st.session_state.get("session_trades", [])
                    trade["status"] = "PUBLIC_SESSION_DEMO"
                    trade["notes"] = f"{trade.get('notes') or ''} Public demo session action; not durable.".strip()
                    session_records.append(trade)
                    st.session_state["session_trades"] = session_records
                    st.success("Public demo commit recorded for this session only. It may disappear when the app restarts.")
                else:
                    storage.insert_trade(trade, DB_PATH)
                    storage.write_audit_log("TRADE_INSERT", f"Manual {side} trade inserted for {symbol}.", DB_PATH)
                    storage.recompute_all_after_trade(DB_PATH)
                    st.success("Trade stored in SQLite and portfolio recalculated.")
                st.rerun()

    st.subheader("Import operational NEW TRADES sheet")
    if st.button("Import trades from NEW TRADES sheet"):
        before = load_context(public_demo_mode)
        imported_trades = data_loader.parse_new_trades(EXCEL_PATH)
        if public_demo_mode:
            existing_ids = set(before["trades"].get("trade_id", pd.Series(dtype=str)).astype(str))
            new_records = [record for record in imported_trades.to_dict("records") if str(record.get("trade_id")) not in existing_ids]
            for record in new_records:
                record["status"] = "PUBLIC_SESSION_DEMO"
                record["notes"] = f"{record.get('notes') or ''} Imported in public demo session; not durable.".strip()
            st.session_state["session_trades"] = st.session_state.get("session_trades", []) + new_records
            inserted = len(new_records)
            after = load_context(public_demo_mode)
            st.info(f"Imported {inserted} NEW TRADES events into this public demo session. These actions are non-durable.")
        else:
            inserted = storage.insert_trades(imported_trades, DB_PATH)
            storage.write_audit_log("NEW_TRADES_IMPORT", f"Imported {inserted} new trade events from NEW TRADES sheet.", DB_PATH)
            if inserted:
                storage.recompute_all_after_trade(DB_PATH)
            after = load_context(public_demo_mode)
            st.info(f"Inserted {inserted} new trade events. Existing deterministic import IDs were skipped.")
        st.session_state["last_impact"] = pd.DataFrame(
            [impact_row(before["summary"], before["risk"]), impact_row(after["summary"], after["risk"])],
            index=["Before", "After"],
        )

    if "last_impact" in st.session_state:
        st.subheader("Before and after impact")
        st.dataframe(st.session_state["last_impact"], width="stretch")

with tabs[7]:
    st.markdown(
        """
        **Educational and privacy disclaimer**

        Educational demo report generated from portfolio summary data. Do not use Gemini Free Tier with confidential real portfolio data.
        """
    )
    deterministic_report = reporting.generate_deterministic_report(
        context["summary"],
        context["risk"],
        context["benchmark"],
        context["issues"],
        context["recent_trades"],
    )
    st.subheader("Deterministic report")
    st.markdown(deterministic_report)
    report_html = reporting.deterministic_report_to_html(deterministic_report)
    col_md, col_html = st.columns(2)
    col_md.download_button(
        "Download deterministic report (Markdown)",
        deterministic_report,
        file_name="portfolio_cockpit_report.md",
        mime="text/markdown",
    )
    col_html.download_button(
        "Download deterministic report (HTML)",
        report_html,
        file_name="portfolio_cockpit_report.html",
        mime="text/html",
    )

    st.subheader("Gemini professional report")
    st.info("Gemini is optional. In the public review deployment it is intentionally not configured; the deterministic report remains available.")
    st.write(f"Gemini status: {'configured' if llm_report.gemini_available() else 'not configured'}")
    st.write(f"Model: `{llm_report.DEFAULT_GEMINI_MODEL}`")

    if st.button("Generate Gemini professional report"):
        storage.write_audit_log("GEMINI_ATTEMPT", "User requested Gemini professional portfolio report.", DB_PATH)
        with st.spinner("Generating Gemini report from summarized portfolio data..."):
            response = llm_report.generate_gemini_portfolio_report(
                context["summary"],
                context["risk"],
                context["benchmark"],
                quality_warnings_for_prompt(context["issues"]),
                recent_trades_for_prompt(context["recent_trades"]),
            )
        storage.write_audit_log(
            "GEMINI_SUCCESS" if response["success"] else "GEMINI_FALLBACK",
            response["message"][:400],
            DB_PATH,
        )
        st.write(f"Generated at: `{response['generated_at']}`")
        if response["success"]:
            st.markdown(response["message"])
        else:
            st.warning(response["message"])
            st.info("Falling back to deterministic report above.")

with tabs[8]:
    st.markdown(
        """
        ### What this MVP proves

        This MVP proves the professional workflow: Excel trades become a normalized ledger, the ledger drives positions, positions drive valuation, valuation drives returns, returns drive risk metrics, and only then does reporting begin.

        The LLM does not calculate the portfolio. Python and SQLite calculate and control the portfolio. Gemini only helps draft a professional narrative from validated summary data.

        ### What remains fragile

        The demo still relies on cached Excel prices, simplified cash reconstruction, classroom-level validation rules and a local SQLite file without automated backup rotation. It is designed for teaching the operating model, not for live fiduciary use.

        ### What must be hardened before real use

        A production path would require immutable raw Excel files, a versioned input folder, SQLite backups, a complete audit trail, stricter validation rules, broker and custodian reconciliation, market-data provider integration, benchmark refresh controls, secrets management, Gemini API key management, explicit Free Tier privacy limits, avoidance of confidential data in LLM prompts, error logging, automated tests, local packaging as a desktop-like app, deployment options, access control, a recovery plan and a documented operating procedure.
        """
    )
