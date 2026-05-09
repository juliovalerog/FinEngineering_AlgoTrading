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
        session_state={},
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


def test_public_reset_rebuilds_sqlite_from_excel() -> None:
    namespace = _load_app_prefix()
    calls: list[tuple] = []

    namespace["storage"] = SimpleNamespace(
        reset_database_from_excel=lambda excel_path, db_path: calls.append(("reset", excel_path, db_path)),
        write_audit_log=lambda event_type, description, db_path: calls.append(("audit", event_type, description, db_path)),
    )
    namespace["st"].session_state["session_trades"] = [{"trade_id": "demo"}]

    message = namespace["reset_to_original_excel_state"](public_demo=True)

    assert calls[0][0] == "reset"
    assert calls[1][1] == "RESET_FROM_EXCEL"
    assert namespace["st"].session_state["session_trades"] == []
    assert "SQLite rebuilt from the original Excel input" in message


def test_compute_benchmark_metrics_prefers_snapshot_returns() -> None:
    namespace = _load_app_prefix()
    snapshots = pd.DataFrame(
        [
            {"date": "2026-01-01", "total_portfolio_value": 100, "benchmark_return": 0.0},
            {"date": "2026-01-02", "total_portfolio_value": 101, "benchmark_return": 0.02},
        ]
    )
    benchmark_prices = pd.DataFrame(
        [
            {"date": "2026-01-01", "benchmark": "S&P 500", "price": 620, "source": "Portfolio sheet"},
            {"date": "2026-01-02", "benchmark": "S&P 500", "price": 7200, "source": "Yahoo Finance"},
        ]
    )

    result = namespace["compute_benchmark_metrics"](snapshots, benchmark_prices, {"cumulative_return": 0.03})

    assert result["cumulative_benchmark_return"] == 0.02


def test_deal_delta_table_has_before_after_delta_metrics() -> None:
    namespace = _load_app_prefix()
    before_summary = {
        "cash": 1000,
        "invested_value": 500,
        "total_portfolio_value": 1500,
        "open_positions": 1,
        "top_5_concentration": 0.4,
        "largest_single_name_exposure": 0.4,
        "unrealized_pnl": 20,
        "total_pnl": 30,
        "cumulative_return": 0.03,
    }
    after_summary = before_summary | {"cash": 900, "invested_value": 620, "total_portfolio_value": 1520, "open_positions": 2}
    before_risk = {"max_drawdown": -0.02, "sharpe_ratio": 1.1, "sortino_ratio": 1.4}
    after_risk = {"max_drawdown": -0.025, "sharpe_ratio": 1.2, "sortino_ratio": 1.5}

    frame = namespace["build_deal_delta_table"](before_summary, before_risk, after_summary, after_risk)

    assert frame.columns.tolist() == ["Metric", "Before", "After", "Delta"]
    assert "Cash" in frame["Metric"].tolist()
    assert "Sharpe ratio" in frame["Metric"].tolist()


def test_new_deal_simulation_does_not_persist_before_accept() -> None:
    namespace = _load_app_prefix()
    persisted: list[dict] = []
    namespace["storage"] = SimpleNamespace(
        insert_trade=lambda trade, db_path: persisted.append(trade),
        get_initial_cash=lambda db_path: 1_000_000,
    )
    base_context = {
        "trades": pd.DataFrame(
            [
                {
                    "trade_id": "abc-buy",
                    "source_sheet": "TEST",
                    "source_row": 1,
                    "symbol": "ABC",
                    "side": "BUY",
                    "trade_date": "2026-01-01",
                    "quantity": 10,
                    "price": 100,
                    "amount": 1000,
                    "sector": "Technology",
                    "status": "TEST",
                    "notes": None,
                }
            ]
        ),
        "prices": pd.DataFrame([{"date": "2026-01-02", "symbol": "ABC", "price": 105, "source": "Precios sheet"}]),
        "benchmark_prices": pd.DataFrame(),
        "snapshots": pd.DataFrame(),
    }
    trade = {
        "trade_id": "abc-buy-2",
        "source_sheet": "MANUAL",
        "source_row": None,
        "symbol": "ABC",
        "side": "BUY",
        "trade_date": "2026-01-02",
        "quantity": 1,
        "price": 105,
        "amount": 105,
        "sector": "Technology",
        "status": "MANUAL_APPROVED",
        "notes": None,
    }

    simulation = namespace["simulated_context"](base_context, trade)

    assert persisted == []
    assert len(simulation["trades"]) == 2


def test_accepting_simulated_deal_persists_locally() -> None:
    namespace = _load_app_prefix()
    calls: list[tuple] = []
    namespace["storage"] = SimpleNamespace(
        insert_trade=lambda trade, db_path: calls.append(("insert", trade, db_path)),
        write_audit_log=lambda event_type, description, db_path: calls.append(("audit", event_type, description, db_path)),
        recompute_all_after_trade=lambda db_path: calls.append(("recompute", db_path)),
    )
    trade = {"trade_id": "manual-1", "symbol": "ABC", "side": "BUY"}

    message = namespace["accept_deal"](trade, public_demo=False, db_path=APP_ROOT / "tmp.sqlite")

    assert [call[0] for call in calls] == ["insert", "audit", "recompute"]
    assert "SQLite" in message


def test_public_demo_accept_is_session_level() -> None:
    namespace = _load_app_prefix()
    namespace["st"].session_state.clear()
    trade = {"trade_id": "manual-1", "symbol": "ABC", "side": "BUY", "notes": None}

    message = namespace["accept_deal"](trade, public_demo=True, db_path=APP_ROOT / "tmp.sqlite")

    assert len(namespace["st"].session_state["session_trades"]) == 1
    assert namespace["st"].session_state["session_trades"][0]["status"] == "PUBLIC_SESSION_DEMO"
    assert "session only" in message


def test_manual_benchmark_repair_button_is_not_present() -> None:
    source = (APP_ROOT / "app.py").read_text(encoding="utf-8")

    assert "Repair S&P 500 benchmark scale" not in source
    assert "Fix S&P scale" not in source
