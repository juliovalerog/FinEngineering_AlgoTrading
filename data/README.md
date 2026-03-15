# Data folder

This repository keeps static data minimal by design.

All market data used in the teaching notebooks is downloaded programmatically with `yfinance` to preserve a reproducible workflow and to reflect realistic analyst practice.

The file `market_data_foundations_prices.csv` is a small classroom cache used by Notebook 1.

Its purpose is practical rather than architectural:

- keep the market-data workflow reproducible,
- allow the notebook to run when live Yahoo Finance access is unavailable,
- preserve a fixed teaching case for discussion in class.

You can optionally store other exported intermediate datasets here if needed for class activities.
