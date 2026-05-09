from __future__ import annotations

from typing import Any
from html import escape

import numpy as np
import pandas as pd


def _fmt_money(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"${float(value):,.0f}"


def _fmt_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.2%}"


def _fmt_ratio(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.2f}"


def generate_deterministic_report(
    portfolio_summary: dict[str, Any],
    risk_metrics: dict[str, Any],
    benchmark_metrics: dict[str, Any],
    data_quality_warnings: list[dict[str, Any]] | pd.DataFrame,
    recent_trades: pd.DataFrame | list[dict[str, Any]],
) -> str:
    """Generate a template-based report from computed facts only."""
    if isinstance(data_quality_warnings, pd.DataFrame):
        dq_count = len(data_quality_warnings)
        blocking = int((data_quality_warnings.get("severity") == "Error").sum()) if "severity" in data_quality_warnings else 0
    else:
        dq_count = len(data_quality_warnings)
        blocking = sum(1 for item in data_quality_warnings if item.get("severity") == "Error")

    if isinstance(recent_trades, list):
        recent_count = len(recent_trades)
    else:
        recent_count = 0 if recent_trades is None else len(recent_trades)

    return f"""
## Executive summary

The portfolio has a current total value of {_fmt_money(portfolio_summary.get("total_portfolio_value"))}, with {_fmt_money(portfolio_summary.get("invested_value"))} invested and {_fmt_money(portfolio_summary.get("cash"))} in cash. Cumulative portfolio return is {_fmt_pct(risk_metrics.get("cumulative_return"))}.

## Current positioning

There are {portfolio_summary.get("open_positions", 0)} open positions. The largest single-name exposure is {_fmt_pct(portfolio_summary.get("largest_single_name_exposure"))}, and the top five positions represent {_fmt_pct(portfolio_summary.get("top_5_concentration"))} of invested capital.

## Benchmark comparison

The local benchmark series shows cumulative S&P 500 return of {_fmt_pct(benchmark_metrics.get("cumulative_benchmark_return"))}. The resulting excess return is {_fmt_pct(benchmark_metrics.get("excess_return"))}. If benchmark data is unavailable or incomplete, this comparison should be treated as a data-quality caveat.

## Risk profile

Annualized return is {_fmt_pct(risk_metrics.get("annualized_return"))}, annualized volatility is {_fmt_pct(risk_metrics.get("annualized_volatility"))}, Sharpe ratio is {_fmt_ratio(risk_metrics.get("sharpe_ratio"))}, Sortino ratio is {_fmt_ratio(risk_metrics.get("sortino_ratio"))}, and maximum drawdown is {_fmt_pct(risk_metrics.get("max_drawdown"))}. Beta versus S&P 500 is {_fmt_ratio(risk_metrics.get("beta"))}, tracking error is {_fmt_pct(risk_metrics.get("tracking_error"))}, and information ratio is {_fmt_ratio(risk_metrics.get("information_ratio"))}.

## Recent trading activity

The SQLite ledger contains {recent_count} recent trade events in the current view. New trades are stored in SQLite, not written back to the original Excel file.

## Data-quality caveats

Automated checks currently report {dq_count} issue or note rows, including {blocking} blocking error-level items. The data-quality tab is the control point before relying on any performance conclusion.

## Recommended next checks

Reconcile open positions against broker or custodian records, refresh benchmark and market prices from an approved provider, and review concentration before using the report for an investment decision.
""".strip()


def deterministic_report_to_html(markdown_text: str) -> str:
    """Convert the deterministic Markdown report into simple standalone HTML."""
    html_lines = [
        "<!doctype html>",
        "<html>",
        "<head>",
        "<meta charset=\"utf-8\">",
        "<title>Portfolio Management Cockpit Report</title>",
        "<style>body{font-family:Arial,sans-serif;max-width:900px;margin:32px auto;line-height:1.5;color:#111827;}h1,h2{color:#111827;}p{margin:0 0 14px;}</style>",
        "</head>",
        "<body>",
        "<h1>Portfolio Management Cockpit Report</h1>",
    ]
    for line in markdown_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("## "):
            html_lines.append(f"<h2>{escape(stripped[3:])}</h2>")
        elif stripped.startswith("- "):
            html_lines.append(f"<p>&bull; {escape(stripped[2:])}</p>")
        else:
            html_lines.append(f"<p>{escape(stripped)}</p>")
    html_lines.extend(["</body>", "</html>"])
    return "\n".join(html_lines)
