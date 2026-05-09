# Financial Engineering with Python

This repository is a teaching resource for a Financial Engineering course. It shows how Python supports professional workflows across market-data analysis, leveraged buyout modeling, and portfolio management.

The materials are classroom-first. Students focus on workflow design, financial logic, data quality, and interpretation of results rather than Python syntax for its own sake.

## Repository Structure

```text
FinEngineering_AlgoTrading/
|- README.md
|- requirements.txt
|- .gitignore
|- .streamlit/
|  |- config.toml
|- app/
|  |- streamlit_lbo_demo.py
|- data/
|  |- README.md
|  |- market_data_foundations_prices.csv
|  |- us_tech_watchlist_prices.csv
|- notebooks/
|  |- 01_financial_data_analysis.ipynb
|  |- 02_lbo_model_python.ipynb
|- portfolio_cockpit/
|  |- app.py
|  |- requirements.txt
|  |- run_app_windows.bat
|  |- run_app_windows.ps1
|  |- data/
|  |  |- input/
|  |  |  |- Portfolio Example JULIO.xlsx
|  |  |- store/
|  |- docs/
|  |- src/
|  |- tests/
|- utils/
|  |- README.md
|  |- lbo_engine.py
|  |- gemini_reporting.py
```

## Learning Path

1. **Session 1 - Market Data And Financial Data Analysis**
   Uses `notebooks/01_financial_data_analysis.ipynb` to frame a realistic investment-screening workflow: data intake, quality control, price-table preparation, return/risk comparison, shortlist construction, and executive watchlist.

2. **Session 2 - LBO Modeling And Productization**
   Uses `notebooks/02_lbo_model_python.ipynb` to rebuild analyst logic for an LBO transaction: inputs, valuation, Sources & Uses, operating projection, cash sweep, deleveraging, returns, and sensitivities. The same analytical engine is also packaged in `app/streamlit_lbo_demo.py` as a lightweight Streamlit demo with optional Gemini commentary.

3. **Session 3 - Portfolio Management Cockpit**
   Uses `portfolio_cockpit/app.py` as the main professional portfolio-management demo. The class shows how an operational Excel file becomes a monitored portfolio system: Excel raw input, normalized SQLite ledger, Python calculations, Streamlit decision interface, deterministic reporting, and optional Gemini narrative.

Algorithmic trading and backtesting material may be used as optional/additional material when available, but the third classroom session now centers on the Portfolio Management Cockpit.

## Technologies Used

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- plotly
- yfinance
- numpy-financial
- vectorbt
- streamlit
- google-genai
- openpyxl
- SQLite
- Jupyter Notebook

## Install For The Course Repository

```bash
git clone https://github.com/juliovalerog/FinEngineering_AlgoTrading.git
cd FinEngineering_AlgoTrading

python -m venv .venv
python -m pip install -r requirements.txt
jupyter notebook
```

Then open the notebooks in `notebooks/` and run each notebook from top to bottom.

## Run The LBO Streamlit Demo

```bash
python -m streamlit run app/streamlit_lbo_demo.py
```

The LBO app mirrors the same case logic as `notebooks/02_lbo_model_python.ipynb` and is intended as a short productization demo rather than the main third-session teaching asset.

## Run The Portfolio Management Cockpit Locally

```bash
cd portfolio_cockpit
python -m streamlit run app.py
```

On Windows, if Application Control blocks `streamlit.exe`, use the helper launcher:

```powershell
.\run_app_windows.bat
```

The cockpit initializes a local SQLite database at `portfolio_cockpit/data/store/portfolio_mvp.sqlite` from the bundled Excel file in `portfolio_cockpit/data/input/Portfolio Example JULIO.xlsx`. The original Excel is raw input and is not modified. SQLite becomes the local system of record for the demo.

Gemini reporting is optional and local-only unless explicitly configured with `GEMINI_API_KEY` or `GOOGLE_API_KEY`.

## Deploy The Portfolio Management Cockpit On Streamlit Community Cloud

Use these Streamlit Community Cloud settings:

- Repository: `juliovalerog/FinEngineering_AlgoTrading`
- Branch: `main`
- Main file path: `portfolio_cockpit/app.py`

For this public review deployment, publish without adding `GEMINI_API_KEY` or `GOOGLE_API_KEY` to Streamlit Cloud secrets. Leave Streamlit Cloud secrets empty. The public app should use the deterministic report by default, and Gemini will show as not configured. This is intentional.

The bundled Excel data is public educational demo data and is required by the deployed app.

## Optional Gemini Reporting Layer

The LBO demo and Portfolio Management Cockpit can generate optional Gemini commentary on top of deterministic calculations.

For local use only, set an API key before running the Gemini section:

```powershell
$env:GEMINI_API_KEY="your_api_key_here"
# or
$env:GOOGLE_API_KEY="your_api_key_here"
```

Do not commit API keys, `.streamlit/secrets.toml`, `.env`, or any credential file. If no key is present, the apps fall back gracefully to deterministic outputs.

## Notes For Students

These materials are templates for structured quantitative reasoning in finance. Pay attention to how the financial question is framed, why each transformation exists, how assumptions are made explicit, and how outputs are interpreted before conclusions are drawn.

In class, we prioritize clarity, reproducibility, data quality, traceability, and economic interpretation over unnecessary technical complexity.
