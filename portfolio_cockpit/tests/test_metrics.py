from __future__ import annotations

import numpy as np
import pandas as pd

from src import metrics


def test_sharpe_ratio_matches_formula() -> None:
    returns = pd.Series([0.01, 0.02, -0.005, 0.015])
    expected = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
    assert metrics.sharpe_ratio(returns) == expected


def test_sortino_ratio_uses_downside_deviation() -> None:
    returns = pd.Series([0.01, -0.02, 0.015, -0.01])
    downside = returns[returns < 0]
    expected = returns.mean() / np.sqrt((downside**2).mean()) * np.sqrt(252)
    assert metrics.sortino_ratio(returns) == expected


def test_max_drawdown() -> None:
    values = pd.Series([100, 120, 90, 130])
    assert np.isclose(metrics.max_drawdown(values), -0.25)


def test_concentration_metrics() -> None:
    result = metrics.concentration_metrics(pd.Series([0.5, 0.3, 0.2]))
    assert result["largest_weight"] == 0.5
    assert result["top_5_weight"] == 1.0
    assert np.isclose(result["herfindahl_index"], 0.38)
    assert np.isclose(result["effective_number_positions"], 1 / 0.38)

