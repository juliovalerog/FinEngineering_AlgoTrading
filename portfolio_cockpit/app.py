from __future__ import annotations

import json
import uuid
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src import charts, data_loader, llm_report, metrics, portfolio_engine, reporting, storage, validation


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


def ensure_database() -> str:
    if not storage.database_exists(DB_PATH):
        storage.reset_database_from_excel(EXCEL_PATH, DB_PATH)
        return "initialized from Excel"
    return "loaded from existing SQLite"


def load_context() -> dict[str, Any]:
    trades = storage.get_trades(DB_PATH)
    prices = storage.get_prices(DB_PATH)
    benchmark_prices = storage.get_benchmark_prices(DB_PATH)
    positions = storage.get_positions(DB_PATH)
    snapshots = storage.get_portfolio_snapshots(DB_PATH)
    issues = validation.run_data_quality_checks(trades, positions, snapshots, prices, benchmark_prices)
    risk = compute_risk_metrics(snapshots, positions)
    benchmark = compute_benchmark_metrics(snapshots, benchmark_prices, risk)
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
    return {
        "cumulative_benchmark_return": benchmark_return,
        "excess_return": excess,
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


db_status = ensure_database()
context = load_context()

st.title("Portfolio Management Cockpit")
st.caption("Financial Engineering course MVP: Excel as operational input, SQLite as system of record, Python as calculation engine, Streamlit as decision interface.")

with st.sidebar:
    st.header("Cockpit Status")
    st.write(f"Database: {db_status}")
    st.write(f"Excel input: `{EXCEL_PATH.name}`")
    st.write(f"SQLite: `{DB_PATH.name}`")
    st.divider()
    st.write("Navigation is organized as the same workflow students follow in class.")

tabs = st.tabs(
    [
        "1. Excel to Portfolio System",
        "2. Data Quality",
        "3. Current Portfolio",
        "4. Performance & Risk",
        "5. Add New Trade",
        "6. Executive Report",
        "7. From MVP to Production",
    ]
)

with tabs[0]:
    st.markdown(
        '<div class="section-note">The Excel is the operational input. The professional workflow starts when we separate raw data, validation, calculations, storage and reporting.</div>',
        unsafe_allow_html=True,
    )
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

    if st.button("Reset demo database from Excel"):
        storage.reset_database_from_excel(EXCEL_PATH, DB_PATH)
        storage.write_audit_log("RESET", "Demo database reset from immutable Excel input.", DB_PATH)
        st.success("Demo database rebuilt from Excel.")
        st.rerun()

    st.subheader("Initialization and reset audit log")
    audit = storage.get_audit_log(DB_PATH)
    if not audit.empty:
        init_audit = audit[audit["event_type"].isin(["INITIALIZE", "RESET"])]
        st.dataframe(init_audit, hide_index=True, width="stretch")
    else:
        st.info("No audit events have been recorded yet.")

with tabs[1]:
    st.markdown(
        '<div class="section-note">Before analysing performance, a professional analyst checks whether the data can be trusted.</div>',
        unsafe_allow_html=True,
    )
    issue_frame = validation.issues_to_frame(context["issues"])
    severity_counts = issue_frame["severity"].value_counts() if not issue_frame.empty else pd.Series(dtype=int)
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Errors", int(severity_counts.get("Error", 0)))
    col_b.metric("Warnings", int(severity_counts.get("Warning", 0)))
    col_c.metric("Info notes", int(severity_counts.get("Info", 0)))
    st.dataframe(issue_frame, hide_index=True, width="stretch")

with tabs[2]:
    st.markdown(
        '<div class="section-note">This is the portfolio manager’s current decision base: what we own, where the exposure is, and where risk is concentrated.</div>',
        unsafe_allow_html=True,
    )
    summary = context["summary"]
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
        st.plotly_chart(charts.allocation_by_asset_chart(context["positions"]), width="stretch")
        st.plotly_chart(charts.cash_vs_invested_chart(context["snapshots"].sort_values("date").iloc[-1] if not context["snapshots"].empty else {}), width="stretch")
    with chart_cols[1]:
        st.plotly_chart(charts.allocation_by_sector_chart(context["positions"]), width="stretch")
        st.plotly_chart(charts.top_positions_chart(context["positions"], "market_value", "Top contributors to market value"), width="stretch")

    st.plotly_chart(charts.top_positions_chart(context["positions"], "unrealized_pnl", "Top contributors to unrealized P&L"), width="stretch")
    st.subheader("Current positions")
    position_columns = ["symbol", "sector", "quantity", "average_cost", "latest_price", "market_value", "unrealized_pnl", "weight"]
    st.dataframe(context["positions"][position_columns] if not context["positions"].empty else context["positions"], hide_index=True, width="stretch")

with tabs[3]:
    st.markdown(
        '<div class="section-note">A portfolio can make money and still be a poor portfolio if the risk-adjusted performance, drawdown or concentration profile is weak.</div>',
        unsafe_allow_html=True,
    )
    risk_free_rate = st.number_input("Annual risk-free rate", min_value=0.0, max_value=0.20, value=0.0, step=0.005, format="%.3f")
    risk = compute_risk_metrics(context["snapshots"], context["positions"], risk_free_rate=risk_free_rate)
    benchmark = compute_benchmark_metrics(context["snapshots"], context["benchmark_prices"], risk)

    cols = st.columns(4)
    cols[0].metric("Cumulative portfolio return", _fmt_pct(risk["cumulative_return"]))
    cols[1].metric("Cumulative S&P 500 return", _fmt_pct(benchmark["cumulative_benchmark_return"]))
    cols[2].metric("Excess return", _fmt_pct(benchmark["excess_return"]))
    cols[3].metric("Annualized return", _fmt_pct(risk["annualized_return"]))

    cols = st.columns(4)
    cols[0].metric("Annualized volatility", _fmt_pct(risk["annualized_volatility"]))
    cols[1].metric("Sharpe ratio", _fmt_num(risk["sharpe_ratio"]))
    cols[2].metric("Sortino ratio", _fmt_num(risk["sortino_ratio"]))
    cols[3].metric("Maximum drawdown", _fmt_pct(risk["max_drawdown"]))

    cols = st.columns(4)
    cols[0].metric("Best day", _fmt_pct(risk["best_day"]))
    cols[1].metric("Worst day", _fmt_pct(risk["worst_day"]))
    cols[2].metric("Hit ratio", _fmt_pct(risk["hit_ratio"]))
    cols[3].metric("Effective positions", _fmt_num(risk["effective_number_positions"]))

    if context["benchmark_prices"].empty:
        st.warning("S&P 500 benchmark data is unavailable locally. The app continues without internet access.")

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.plotly_chart(charts.portfolio_value_chart(context["snapshots"]), width="stretch")
        st.plotly_chart(charts.drawdown_chart(context["snapshots"]), width="stretch")
    with chart_cols[1]:
        st.plotly_chart(charts.cumulative_return_vs_benchmark_chart(context["snapshots"]), width="stretch")
        st.plotly_chart(charts.daily_returns_distribution_chart(context["snapshots"]), width="stretch")

with tabs[4]:
    st.markdown(
        '<div class="section-note">This is the moment where the MVP stops being a static dashboard and becomes an operating tool: new trades enter the system and the portfolio is recalculated.</div>',
        unsafe_allow_html=True,
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
        submitted = st.form_submit_button("Add trade to SQLite")

    if submitted:
        before = load_context()
        trade = {
            "trade_id": uuid.uuid4().hex,
            "source_sheet": "MANUAL",
            "source_row": None,
            "symbol": symbol,
            "side": side,
            "trade_date": trade_date.isoformat(),
            "quantity": quantity,
            "price": price,
            "amount": quantity * price,
            "sector": sector or None,
            "status": "MANUAL_APPROVED",
            "notes": notes or None,
        }
        manual_issues = validation.validate_manual_trade(trade, before["positions"])
        errors = [issue for issue in manual_issues if issue["severity"] == "Error"]
        if errors:
            for issue in errors:
                st.error(issue["message"])
        else:
            storage.insert_trade(trade, DB_PATH)
            storage.write_audit_log("TRADE_INSERT", f"Manual {side} trade inserted for {symbol}.", DB_PATH)
            storage.recompute_all_after_trade(DB_PATH)
            after = load_context()
            st.session_state["last_impact"] = pd.DataFrame(
                [impact_row(before["summary"], before["risk"]), impact_row(after["summary"], after["risk"])],
                index=["Before", "After"],
            )
            st.success("Trade stored in SQLite and portfolio recalculated.")

    st.subheader("Import operational NEW TRADES sheet")
    if st.button("Import trades from NEW TRADES sheet"):
        before = load_context()
        imported_trades = data_loader.parse_new_trades(EXCEL_PATH)
        inserted = storage.insert_trades(imported_trades, DB_PATH)
        storage.write_audit_log("NEW_TRADES_IMPORT", f"Imported {inserted} new trade events from NEW TRADES sheet.", DB_PATH)
        if inserted:
            storage.recompute_all_after_trade(DB_PATH)
        after = load_context()
        st.session_state["last_impact"] = pd.DataFrame(
            [impact_row(before["summary"], before["risk"]), impact_row(after["summary"], after["risk"])],
            index=["Before", "After"],
        )
        st.info(f"Inserted {inserted} new trade events. Existing deterministic import IDs were skipped.")

    if "last_impact" in st.session_state:
        st.subheader("Before and after impact")
        st.dataframe(st.session_state["last_impact"], width="stretch")

with tabs[5]:
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

    st.subheader("Gemini professional report")
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

with tabs[6]:
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
