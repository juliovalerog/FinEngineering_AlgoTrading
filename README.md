# Financial Engineering with Python: Financial Analysis, LBO Modeling, and Algorithmic Trading

This repository is a compact teaching resource for a Financial Engineering course. It contains three classroom-ready Jupyter notebooks that demonstrate how Python supports professional workflows across:

- financial data analysis,
- leveraged buyout (LBO) modeling, and
- algorithmic trading with backtesting.

The notebooks are designed for guided review in class: students focus on workflow design, financial logic, and interpretation of results.

The repository is intentionally teaching-first. When there is a trade-off between technical sophistication and classroom clarity, the materials prioritize clarity, explicit analytical logic, and heavily commented code that students can revisit after class.

The LBO section now also includes a lightweight Streamlit demo that reuses the same analytical engine and an optional Gemini-based reporting layer for short investment commentary.

## Repository structure

```text
FinEngineering_AlgoTrading/
|- README.md
|- requirements.txt
|- .gitignore
|- data/
|  |- README.md
|- app/
|  |- streamlit_lbo_demo.py
|- notebooks/
|  |- 01_financial_data_analysis.ipynb
|  |- 02_lbo_model_python.ipynb
|  |- 03_algorithmic_trading_backtest.ipynb
|- utils/
|  |- README.md
|  |- lbo_engine.py
|  |- gemini_reporting.py
```

## Learning path

1. **Notebook 1 - Market Data Foundations for Financial Engineering**
   Builds a professional market-data workflow: universe selection, programmatic collection, reliability checks, canonical price-table construction, finance-ready metrics, and analyst-style summary outputs.
   The notebook is designed as a guided walkthrough of how raw market data becomes a trusted analytical base for later monitoring, modeling, and trading work.

2. **Notebook 2 - Leveraged Buyout Modeling from Entry to Exit**
   Rebuilds the analyst logic of an LBO transaction: inputs, valuation, Sources & Uses, operating projection, cash sweep, deleveraging, returns, sensitivity, and productization.
   The notebook is written as a guided walkthrough, with detailed markdown and code comments that explain the financial meaning of each modeling block.

3. **Notebook 3 - Algorithmic Trading, Backtesting, and Performance Evaluation**
   Applies a full quantitative loop from indicators to performance diagnostics, first manually and then with `vectorbt`.

The progression is intentional: **data discipline -> model discipline -> strategy discipline**.

## Technologies used

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- yfinance
- numpy-financial
- vectorbt
- streamlit
- google-genai
- Jupyter Notebook

## How to run

```bash
git clone https://github.com/juliovalerog/FinEngineering_AlgoTrading.git
cd FinEngineering_AlgoTrading

python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
jupyter notebook
```

Then open the notebooks in `notebooks/` and run each notebook from top to bottom.

## Notebook 1 data workflow

Notebook 1 downloads market data programmatically with `yfinance` and also includes a local cache in `data/market_data_foundations_prices.csv` for classroom robustness.

This keeps the workflow professional and reproducible while still allowing the notebook to run if live download is temporarily unavailable.

## Run the LBO Streamlit demo

```bash
streamlit run app/streamlit_lbo_demo.py
```

The app mirrors the same case logic as `notebooks/02_lbo_model_python.ipynb` and is intended as a short end-of-class productization demo rather than the main teaching asset.
Its purpose is to show how the same analytical engine can be packaged into a lightweight decision-support prototype without changing the underlying finance logic.

## Optional Gemini reporting layer

The notebook and the Streamlit app can generate a short, finance-oriented commentary on top of the quantitative outputs.

Set the API key before running the Gemini section:

```bash
# Windows PowerShell
$env:GEMINI_API_KEY="your_api_key_here"

# macOS / Linux
# export GEMINI_API_KEY="your_api_key_here"
```

If no key is present, the notebook and app fail gracefully and simply skip the AI-generated commentary.

## Notes for students

These notebooks are not only coding examples. They are templates for **structured quantitative reasoning** in finance.

When reviewing each notebook, pay attention to:

- how the financial question is framed,
- why each library and transformation is used,
- how assumptions are made explicit,
- how outputs are interpreted before conclusions are drawn.

In class, we prioritize clarity, reproducibility, and economic interpretation over unnecessary technical complexity.
