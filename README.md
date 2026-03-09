# Financial Engineering with Python: Financial Analysis, LBO Modeling, and Algorithmic Trading

This repository is a compact teaching resource for a Financial Engineering course. It contains three classroom-ready Jupyter notebooks that demonstrate how Python supports professional workflows across:

- financial data analysis,
- leveraged buyout (LBO) modeling, and
- algorithmic trading with backtesting.

The notebooks are designed for guided review in class: students focus on workflow design, financial logic, and interpretation of results.

## Repository structure

```text
FinEngineering_AlgoTrading/
|- README.md
|- requirements.txt
|- .gitignore
|- data/
|  |- README.md
|- notebooks/
|  |- 01_financial_data_analysis.ipynb
|  |- 02_lbo_model_python.ipynb
|  |- 03_algorithmic_trading_backtest.ipynb
|- utils/
|  |- README.md
```

## Learning path

1. **Notebook 1 - Financial Data Analysis with Python**
   Builds a clean market-data workflow: data acquisition, cleaning, feature engineering, and visual interpretation.

2. **Notebook 2 - Constructing an LBO Model in Python**
   Transfers the same structured thinking to corporate finance modeling: assumptions, projections, debt mechanics, returns, and sensitivity.

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

## Notes for students

These notebooks are not only coding examples. They are templates for **structured quantitative reasoning** in finance.

When reviewing each notebook, pay attention to:

- how the financial question is framed,
- why each library and transformation is used,
- how assumptions are made explicit,
- how outputs are interpreted before conclusions are drawn.

In class, we prioritize clarity, reproducibility, and economic interpretation over unnecessary technical complexity.
