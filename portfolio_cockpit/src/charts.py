from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .metrics import daily_returns


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False)
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=35, b=20))
    return fig


def _add_time_controls(fig: go.Figure) -> go.Figure:
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(step="all", label="All"),
            ]
        ),
    )
    return fig


def allocation_by_asset_chart(positions: pd.DataFrame) -> go.Figure:
    if positions is None or positions.empty:
        return _empty_figure("No open positions")
    fig = px.pie(positions, names="symbol", values="market_value", hole=0.45)
    fig.update_layout(title="Allocation by asset", legend_title_text="", margin=dict(l=20, r=20, t=45, b=20))
    return fig


def allocation_by_sector_chart(positions: pd.DataFrame) -> go.Figure:
    if positions is None or positions.empty or "sector" not in positions.columns:
        return _empty_figure("No sector allocation")
    data = positions.copy()
    data["sector"] = data["sector"].fillna("Unclassified")
    grouped = data.groupby("sector", as_index=False)["market_value"].sum().sort_values("market_value", ascending=False)
    fig = px.bar(grouped, x="sector", y="market_value", title="Allocation by sector")
    fig.update_layout(xaxis_title="", yaxis_title="Market value", margin=dict(l=20, r=20, t=45, b=20))
    return fig


def cash_vs_invested_chart(snapshot: pd.Series | dict) -> go.Figure:
    if snapshot is None or len(snapshot) == 0:
        return _empty_figure("No snapshot available")
    data = pd.DataFrame(
        {
            "bucket": ["Cash", "Invested"],
            "value": [float(snapshot.get("cash", 0) or 0), float(snapshot.get("invested_value", 0) or 0)],
        }
    )
    fig = px.bar(data, x="bucket", y="value", title="Cash vs invested")
    fig.update_layout(xaxis_title="", yaxis_title="Value", margin=dict(l=20, r=20, t=45, b=20))
    return fig


def top_positions_chart(positions: pd.DataFrame, metric: str = "market_value", title: str | None = None) -> go.Figure:
    if positions is None or positions.empty or metric not in positions.columns:
        return _empty_figure("No positions available")
    data = positions.sort_values(metric, ascending=False).head(10)
    fig = px.bar(data, x=metric, y="symbol", orientation="h", title=title or f"Top positions by {metric}")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_title="", yaxis_title="", margin=dict(l=20, r=20, t=45, b=20))
    return fig


def sector_pnl_chart(positions: pd.DataFrame) -> go.Figure:
    if positions is None or positions.empty or "unrealized_pnl" not in positions.columns:
        return _empty_figure("No sector P&L available")
    data = positions.copy()
    data["sector"] = data["sector"].fillna("Unclassified")
    grouped = data.groupby("sector", as_index=False).agg(
        market_value=("market_value", "sum"),
        unrealized_pnl=("unrealized_pnl", "sum"),
    )
    fig = px.bar(grouped.sort_values("unrealized_pnl"), x="unrealized_pnl", y="sector", orientation="h", title="Sector-level unrealized P&L")
    fig.update_layout(xaxis_title="Unrealized P&L", yaxis_title="", margin=dict(l=20, r=20, t=45, b=20))
    return fig


def allocation_vs_pnl_chart(positions: pd.DataFrame) -> go.Figure:
    if positions is None or positions.empty or "unrealized_pnl" not in positions.columns:
        return _empty_figure("No allocation vs P&L data")
    data = positions.copy()
    data["sector"] = data["sector"].fillna("Unclassified")
    grouped = data.groupby("sector", as_index=False).agg(
        market_value=("market_value", "sum"),
        unrealized_pnl=("unrealized_pnl", "sum"),
    )
    total_market_value = grouped["market_value"].sum()
    total_abs_pnl = grouped["unrealized_pnl"].abs().sum()
    grouped["allocation_weight"] = grouped["market_value"] / total_market_value if total_market_value else np.nan
    grouped["absolute_pnl_share"] = grouped["unrealized_pnl"].abs() / total_abs_pnl if total_abs_pnl else np.nan
    long = grouped.melt("sector", value_vars=["allocation_weight", "absolute_pnl_share"], var_name="metric", value_name="value")
    long["metric"] = long["metric"].map({"allocation_weight": "Allocation", "absolute_pnl_share": "Absolute P&L share"})
    fig = px.bar(long, x="sector", y="value", color="metric", barmode="group", title="Allocation vs P&L contribution")
    fig.update_layout(xaxis_title="", yaxis_title="Share", margin=dict(l=20, r=20, t=45, b=20))
    return fig


def portfolio_value_chart(snapshots: pd.DataFrame) -> go.Figure:
    if snapshots is None or snapshots.empty:
        return _empty_figure("No portfolio history")
    data = snapshots.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    fig = px.line(data, x="date", y="total_portfolio_value", title="Portfolio value through time")
    fig.update_layout(xaxis_title="", yaxis_title="Portfolio value", margin=dict(l=20, r=20, t=45, b=20))
    return _add_time_controls(fig)


def cumulative_return_vs_benchmark_chart(snapshots: pd.DataFrame) -> go.Figure:
    if snapshots is None or snapshots.empty:
        return _empty_figure("No performance history")
    data = snapshots.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["date"], y=data["portfolio_return"], mode="lines", name="Portfolio"))
    if "benchmark_return" in data.columns and data["benchmark_return"].notna().any():
        fig.add_trace(go.Scatter(x=data["date"], y=data["benchmark_return"], mode="lines", name="S&P 500"))
    fig.update_layout(title="Cumulative return vs S&P 500", xaxis_title="", yaxis_title="Return", margin=dict(l=20, r=20, t=45, b=20))
    return _add_time_controls(fig)


def drawdown_chart(snapshots: pd.DataFrame) -> go.Figure:
    if snapshots is None or snapshots.empty:
        return _empty_figure("No drawdown history")
    data = snapshots.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    values = pd.to_numeric(data["total_portfolio_value"], errors="coerce")
    running_max = values.cummax()
    data["drawdown"] = values / running_max - 1
    fig = px.area(data, x="date", y="drawdown", title="Drawdown")
    fig.update_layout(xaxis_title="", yaxis_title="Drawdown", margin=dict(l=20, r=20, t=45, b=20))
    return _add_time_controls(fig)


def daily_returns_distribution_chart(snapshots: pd.DataFrame) -> go.Figure:
    if snapshots is None or snapshots.empty:
        return _empty_figure("No returns history")
    returns = daily_returns(pd.to_numeric(snapshots["total_portfolio_value"], errors="coerce"))
    if returns.empty:
        return _empty_figure("Not enough observations")
    fig = px.histogram(x=returns, nbins=30, title="Daily returns distribution")
    fig.update_layout(xaxis_title="Daily return", yaxis_title="Observations", margin=dict(l=20, r=20, t=45, b=20))
    return fig


def daily_returns_time_series_chart(snapshots: pd.DataFrame) -> go.Figure:
    if snapshots is None or snapshots.empty:
        return _empty_figure("No returns history")
    data = snapshots.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"]).sort_values("date")
    values = pd.to_numeric(data["total_portfolio_value"], errors="coerce")
    data["daily_return"] = values.pct_change()
    data = data.dropna(subset=["daily_return"])
    if data.empty:
        return _empty_figure("Not enough observations")
    fig = px.bar(data, x="date", y="daily_return", title="Daily returns through time")
    fig.update_layout(xaxis_title="", yaxis_title="Daily return", margin=dict(l=20, r=20, t=45, b=20))
    return _add_time_controls(fig)
