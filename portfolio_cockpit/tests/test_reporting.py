from __future__ import annotations

from src import llm_report, reporting


def test_deterministic_report_generation_without_gemini() -> None:
    text = reporting.generate_deterministic_report(
        {"total_portfolio_value": 1_100_000, "invested_value": 600_000, "cash": 500_000, "open_positions": 3},
        {"cumulative_return": 0.1, "annualized_return": 0.2, "annualized_volatility": 0.15},
        {"cumulative_benchmark_return": 0.05, "excess_return": 0.05},
        [],
        [],
    )
    assert "Executive summary" in text
    assert "Gemini" not in text


def test_deterministic_report_html_export_text() -> None:
    markdown = reporting.generate_deterministic_report(
        {"total_portfolio_value": 1_100_000, "invested_value": 600_000, "cash": 500_000, "open_positions": 3},
        {"cumulative_return": 0.1, "annualized_return": 0.2, "annualized_volatility": 0.15, "beta": 1.1},
        {"cumulative_benchmark_return": 0.05, "excess_return": 0.05},
        [],
        [],
    )
    html = reporting.deterministic_report_to_html(markdown)
    assert "<html>" in html
    assert "Executive summary" in html
    assert "Risk profile" in html


def test_gemini_fallback_when_credentials_missing(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    assert llm_report.gemini_available() is False
    response = llm_report.generate_gemini_portfolio_report({}, {}, {}, [], [])
    assert response["success"] is False
    assert "Gemini not configured" in response["message"]
