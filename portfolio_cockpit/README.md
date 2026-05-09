# Portfolio Management Cockpit

Streamlit MVP for the third Python session of the Financial Engineering course.

The teaching narrative is:

> Excel is the input, SQLite is the system of record, Python is the engine, Streamlit is the interface, Gemini is the reporting assistant.

The app shows how a professional analyst turns an operational Excel portfolio file into a controlled monitoring, validation, risk and reporting cockpit. The class is not about Python syntax. It is about the workflow a financial data analyst or data scientist would build around an Excel process.

The architecture is deliberately simple for class: SQLite is the only storage backend. There is no Docker setup and no MariaDB, PostgreSQL or external database server.

## Install And Run On Windows

Use the local Python interpreter inside `.venv` and run Streamlit as a Python module. This avoids calling `streamlit.exe` directly, which may be blocked by Windows Application Control policies.

```powershell
cd C:/Users/julio/Documents/DemoCode/FinEngineering_AlgoTrading/portfolio_cockpit

py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Do not use `streamlit run app.py` directly if Windows blocks `streamlit.exe`. Use `python.exe -m streamlit` because it runs Streamlit as a Python module through the approved Python interpreter.

Activation with `Activate.ps1` is optional and not needed for the commands above.

## Run Options

Recommended:

```powershell
.\run_app_windows.bat
```

PowerShell:

```powershell
.\run_app_windows.ps1
```

Manual:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The expected Excel input is:

```text
data/input/Portfolio Example JULIO.xlsx
```

The app does not modify this Excel file.

## First Load And SQLite Storage

On first run, if `data/store/portfolio_mvp.sqlite` does not exist, the app reads the Excel workbook and creates a local SQLite database.

The initial load detects these sheets when present:

- `TRACK`
- `Portfolio`
- `Precios`
- `Cost`
- `Value`
- `NEW TRADES`

The `TRACK` sheet is converted into a normalized trade ledger. Current positions, portfolio snapshots, benchmark prices and audit events are stored in SQLite. Subsequent runs load from SQLite instead of rebuilding from Excel.

## App Workflow

The app is organized into five classroom sections:

- `Home / Portfolio Cockpit`: executive KPIs, portfolio value, benchmark comparison and attention points.
- `Data & Controls`: Excel/SQLite status, reset, Yahoo refresh, price coverage, data quality, lineage and audit log.
- `Portfolio Analysis`: current positions, allocation, contributors, detractors and concentration.
- `Performance & Risk`: return, benchmark, volatility, Sharpe, Sortino, beta, tracking error, information ratio and drawdown.
- `Actions & Reporting`: trade simulation, SQLite/session commits, NEW TRADES import, deterministic report, optional Gemini report and production roadmap.

## Reset Demo Database

Use the button **Reset to original Excel state** in the first tab. This deletes and rebuilds only the demo SQLite database from the original Excel input. It removes Yahoo-updated prices and Yahoo-extended snapshot dates from the live local database. It does not overwrite the Excel file.

## Manual Market-Data Refresh

The Yahoo Finance refresh is manual and optional. It updates only open-position tickers plus the S&P 500 reference in SQLite; it never modifies the Excel workbook.

Developer verification flow:

1. Reset database from Excel.
2. Observe the latest portfolio snapshot date from the original Excel history.
3. Refresh Yahoo prices in the Market Data Refresh section.
4. Confirm that Yahoo rows were added to `prices`, S&P 500 rows were added to `benchmark_prices`, `portfolio_snapshots` extends beyond the Excel date when market data is available, and the charts plus sidebar date range update to the new latest snapshot date.

## Add New Trades

New manual trades are inserted into the SQLite `trades` table and the portfolio is recalculated. The original Excel is not changed.

The button **Import trades from NEW TRADES sheet** imports pending Excel rows into SQLite using deterministic trade IDs, so repeated imports do not duplicate the same rows.

## Reporting

The deterministic report is always available and uses only computed metrics plus data-quality warnings.

Gemini reporting is optional. It drafts a professional narrative from summarized portfolio data. It does not calculate, validate or override portfolio metrics.

Configure Gemini locally with an environment variable:

```powershell
$env:GEMINI_API_KEY="your_key_here"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The app also accepts `GOOGLE_API_KEY`, matching the existing LBO demo credential pattern.

Gemini is optional. If credentials are missing or the API call fails, the app falls back to the deterministic report.

Do not use Gemini Free Tier with confidential real portfolio data. The LLM drafts the narrative; it does not calculate or validate the portfolio.

## Deploy The Portfolio Management Cockpit On Streamlit Community Cloud

Use these Streamlit Community Cloud settings:

- Repository: `juliovalerog/FinEngineering_AlgoTrading`
- Branch: `main`
- Main file path: `portfolio_cockpit/app.py`

For this public review deployment, publish without adding `GEMINI_API_KEY` or `GOOGLE_API_KEY` to Streamlit Cloud secrets. Leave Streamlit Cloud secrets empty. The deterministic report remains available by default, and Gemini will show as not configured. This is intentional.

The bundled Excel data is public educational demo data and must remain committed because the deployed app initializes SQLite from it.
