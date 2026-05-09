from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

try:
    from google import genai
except ImportError:  # pragma: no cover - optional dependency
    genai = None


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _resolve_api_key() -> str | None:
    """Match the LBO app's environment-variable credential pattern."""
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def gemini_available() -> bool:
    return genai is not None and bool(_resolve_api_key())


def _format_exception(exc: Exception) -> str:
    error_text = str(exc)
    lowered = error_text.lower()
    if "api_key_invalid" in lowered or "api key not valid" in lowered:
        return "Gemini not configured: the configured API key was rejected."
    if "permission_denied" in lowered or "permission denied" in lowered:
        return "Gemini call failed: the API key does not have permission to use the selected model."
    return f"Gemini call failed: {error_text}"


def _build_prompt(
    portfolio_summary: dict[str, Any],
    risk_metrics: dict[str, Any],
    benchmark_metrics: dict[str, Any],
    data_quality_warnings: list[dict[str, Any]],
    recent_trades: list[dict[str, Any]],
) -> str:
    payload = {
        "portfolio_summary": portfolio_summary,
        "risk_metrics": risk_metrics,
        "benchmark_metrics": benchmark_metrics,
        "data_quality_warnings": data_quality_warnings[:20],
        "recent_trades_summary": recent_trades[:15],
    }
    return f"""
You are drafting a professional portfolio management report for a Financial Engineering class.

Rules:
- Deterministic Python and SQLite metrics are the source of truth.
- Use only the summarized portfolio data below.
- Do not invent figures, holdings, transactions, causes or recommendations that are not supported by the data.
- Do not add placeholder dates, reporting periods or missing facts.
- Do not request or expose confidential transaction-level data.
- Separate facts, interpretations, risks and recommended checks.
- Keep the tone professional, concise and suitable for a portfolio manager.

Required sections:
1. Executive summary
2. Portfolio evolution
3. Current positioning
4. Benchmark comparison vs S&P 500
5. Risk profile: volatility, Sharpe, Sortino, max drawdown, concentration
6. Recent trades and likely impact
7. Data-quality caveats
8. Questions for the portfolio manager
9. Recommended next analytical checks

Summarized data:
{json.dumps(payload, indent=2, default=str)}
""".strip()


def generate_gemini_portfolio_report(
    portfolio_summary: dict[str, Any],
    risk_metrics: dict[str, Any],
    benchmark_metrics: dict[str, Any],
    data_quality_warnings: list[dict[str, Any]],
    recent_trades: list[dict[str, Any]],
    model: str = DEFAULT_GEMINI_MODEL,
) -> dict[str, Any]:
    if genai is None:
        return {
            "success": False,
            "message": "Gemini not configured: install the optional dependency `google-genai`.",
            "model": model,
            "generated_at": _utc_now(),
        }

    api_key = _resolve_api_key()
    if not api_key:
        return {
            "success": False,
            "message": "Gemini not configured",
            "model": model,
            "generated_at": _utc_now(),
        }

    prompt = _build_prompt(portfolio_summary, risk_metrics, benchmark_metrics, data_quality_warnings, recent_trades)
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=model, contents=prompt)
        text = (response.text or "").strip()
    except Exception as exc:  # pragma: no cover - network / credential dependent
        return {
            "success": False,
            "message": _format_exception(exc),
            "model": model,
            "generated_at": _utc_now(),
        }

    if not text:
        return {
            "success": False,
            "message": "Gemini call failed: empty response.",
            "model": model,
            "generated_at": _utc_now(),
        }

    return {
        "success": True,
        "message": text,
        "model": model,
        "generated_at": _utc_now(),
    }
