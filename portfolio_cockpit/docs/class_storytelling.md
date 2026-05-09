# Class Storytelling Script

## 0-10 min: Excel as operational reality

Open the original workbook and explain that Excel is not the enemy. It is the operational reality in many finance teams.

Key phrase:

> We are not replacing Excel; we are professionalizing the workflow around it.

## 10-25 min: From Excel to analytical model

Show the app's first tab and the pipeline:

Excel -> SQLite ledger -> positions -> valuation -> performance -> risk -> report

Key phrase:

> Excel is the input, SQLite is the system of record, Python is the engine, Streamlit is the interface, Gemini is the reporting assistant.

## 25-45 min: Data quality and normalized ledger

Use the Data Quality tab to show missing sectors, missing fields, duplicate-like rows and sell operations that need validation.

Key phrase:

> A professional analyst does not start with charts; they start with data quality and traceability.

## 45-65 min: Current portfolio cockpit

Move to Current Portfolio. Discuss current holdings, cash, invested value, unrealized P&L, sector exposure and concentration.

## 65-85 min: Performance and risk vs benchmark

Open Performance & Risk. Discuss total return, S&P 500 comparison, annualized volatility, Sharpe ratio, Sortino ratio, maximum drawdown, hit ratio and concentration.

## 85-105 min: Adding new trades and recalculating

Use the Add New Trade tab. Add a manual trade or import the NEW TRADES sheet. Show that the trade enters SQLite, not Excel, and the portfolio recalculates.

## 105-115 min: Deterministic vs Gemini report

Open Executive Report. First show the deterministic report. Then optionally generate the Gemini report if credentials are configured.

Key phrase:

> The LLM does not calculate the portfolio. It writes a controlled narrative based on validated metrics.

## 115-120 min: From MVP to production

Use the final tab to explain what would need to be hardened before using the tool in a real finance workflow.

