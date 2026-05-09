from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import pandas as pd

from src import portfolio_engine


APP_ROOT = Path(__file__).resolve().parents[1]


def _load_app_prefix() -> dict:
    source = (APP_ROOT / "app.py").read_text(encoding="utf-8")
    prefix = source.split("\npublic_demo_mode =")[0]
    namespace: dict = {"__file__": str(APP_ROOT / "app.py"), "__name__": "app_prefix_for_test"}
    fake_streamlit = SimpleNamespace(
        set_page_config=lambda *args, **kwargs: None,
        markdown=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )
    previous_streamlit = sys.modules.get("streamlit")
    sys.modules["streamlit"] = fake_streamlit
    try:
        exec(compile(prefix, str(APP_ROOT / "app.py"), "exec"), namespace)
    finally:
        if previous_streamlit is None:
            sys.modules.pop("streamlit", None)
        else:
            sys.modules["streamlit"] = previous_streamlit
    return namespace


def test_local_portfolio_engine_import_path() -> None:
    assert hasattr(portfolio_engine, "get_effective_daily_prices")
    assert Path(portfolio_engine.__file__).resolve() == APP_ROOT / "src" / "portfolio_engine.py"


def test_app_prefix_inserts_portfolio_cockpit_on_sys_path() -> None:
    namespace = _load_app_prefix()

    assert str(APP_ROOT) in namespace["sys"].path
    diagnostics = namespace["module_diagnostics"]()
    assert diagnostics["portfolio_engine_file"].endswith("portfolio_engine.py")
    assert diagnostics["has_get_effective_daily_prices"] is True


def test_price_source_coverage_fallback_does_not_crash() -> None:
    namespace = _load_app_prefix()
    namespace["portfolio_engine"] = SimpleNamespace()
    prices = pd.DataFrame(
        [
            {"date": "2026-01-01", "symbol": "ABC", "price": 10, "source": "Precios sheet"},
            {"date": "2026-01-01", "symbol": "ABC", "price": 11, "source": "Other"},
        ]
    )
    positions = pd.DataFrame([{"symbol": "ABC", "quantity": 1}])

    coverage = namespace["price_source_coverage"](prices, positions)

    assert len(coverage) == 1
    assert coverage.iloc[0]["symbol"] == "ABC"
