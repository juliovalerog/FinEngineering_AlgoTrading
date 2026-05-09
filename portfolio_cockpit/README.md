# Portfolio Management Cockpit

Streamlit MVP for the third Python session of the Financial Engineering course.

The teaching narrative is:

> Excel is the input, SQLite is the system of record, Python is the engine, Streamlit is the interface, Gemini is the reporting assistant.

The app shows how a professional analyst turns an operational Excel portfolio file into a controlled monitoring, validation, risk and reporting cockpit. The class is not about Python syntax. It is about the workflow a financial data analyst or data scientist would build around an Excel process.

The architecture is deliberately simple for class: SQLite is the only storage backend. There is no Docker setup and no MariaDB, PostgreSQL or external database server.

## Install

From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
streamlit run app.py
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

## Reset Demo Database

Use the button **Reset demo database from Excel** in the first tab. This deletes and rebuilds only the demo SQLite database from the original Excel input. It does not overwrite the Excel file.

## Add New Trades

New manual trades are inserted into the SQLite `trades` table and the portfolio is recalculated. The original Excel is not changed.

The button **Import trades from NEW TRADES sheet** imports pending Excel rows into SQLite using deterministic trade IDs, so repeated imports do not duplicate the same rows.

## Reporting

The deterministic report is always available and uses only computed metrics plus data-quality warnings.

Gemini reporting is optional. It drafts a professional narrative from summarized portfolio data. It does not calculate, validate or override portfolio metrics.

Configure Gemini locally with an environment variable:

```powershell
$env:GEMINI_API_KEY="your_key_here"
streamlit run app.py
```

The app also accepts `GOOGLE_API_KEY`, matching the existing LBO demo credential pattern.

Gemini is optional. If credentials are missing or the API call fails, the app falls back to the deterministic report.

Do not use Gemini Free Tier with confidential real portfolio data. The LLM drafts the narrative; it does not calculate or validate the portfolio.
