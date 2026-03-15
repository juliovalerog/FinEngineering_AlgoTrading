from __future__ import annotations

import json
import os
from typing import Any

try:
    from google import genai
except ImportError:  # pragma: no cover - optional dependency
    genai = None


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def _resolve_api_key(explicit_api_key: str | None = None) -> str | None:
    """Accept either project-specific or Google-standard environment variable names."""
    return explicit_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def _format_gemini_exception(exc: Exception) -> str:
    """Convert SDK errors into short, actionable messages for classroom demos."""
    error_text = str(exc)
    lowered_error_text = error_text.lower()

    if "api_key_invalid" in lowered_error_text or "api key not valid" in lowered_error_text:
        return (
            "Gemini commentary skipped: the configured API key was rejected. "
            "Set a valid Gemini API key in GEMINI_API_KEY or GOOGLE_API_KEY."
        )

    if "permission_denied" in lowered_error_text or "permission denied" in lowered_error_text:
        return (
            "Gemini commentary skipped: the API key does not have permission to use the selected model."
        )

    return f"Gemini commentary skipped: {error_text}"


def build_commentary_payload(results: dict[str, Any]) -> dict[str, float | int]:
    """Extract only the model outputs the reporting layer is allowed to discuss."""
    returns_summary = results["returns_summary"].set_index("Metric")["Value"]
    credit_metrics = results["credit_metrics"]
    debt_schedule = results["debt_schedule"]
    operating_projection = results["operating_projection"]
    entry_valuation = results["entry_valuation"].set_index("Metric")["Value"]
    inputs = results["inputs"]

    exit_label = f"Year {inputs['exit']['exit_year']}"
    opening_total_debt = float(
        debt_schedule.iloc[0]["Beginning senior debt"] + debt_schedule.iloc[0]["Beginning subordinated debt"]
    )
    exit_total_debt = float(
        debt_schedule.loc[exit_label, "Ending senior debt"] + debt_schedule.loc[exit_label, "Ending subordinated debt"]
    )
    debt_paydown = opening_total_debt - exit_total_debt
    historical_ebitda = float(operating_projection.loc["Historical", "EBITDA"])
    exit_ebitda = float(returns_summary.loc["Exit EBITDA"])

    return {
        "share_price": float(inputs["deal"]["share_price"]),
        "implied_offer_price_per_share": float(entry_valuation.loc["Implied offer price per share"]),
        "takeover_premium": float(entry_valuation.loc["Takeover premium"]),
        "entry_multiple": float(inputs["valuation"]["entry_multiple"]),
        "exit_multiple": float(inputs["exit"]["exit_multiple"]),
        "exit_year": int(inputs["exit"]["exit_year"]),
        "historical_ebitda": historical_ebitda,
        "exit_ebitda": exit_ebitda,
        "ebitda_growth_pct": (exit_ebitda / historical_ebitda) - 1,
        "entry_enterprise_value": float(entry_valuation.loc["Entry Enterprise Value"]),
        "sponsor_equity_invested": float(returns_summary.loc["Sponsor equity invested"]),
        "exit_equity_value": float(returns_summary.loc["Exit Equity Value"]),
        "irr": float(returns_summary.loc["IRR"]),
        "moic": float(returns_summary.loc["MOIC"]),
        "exit_net_debt": float(returns_summary.loc["Exit net debt"]),
        "debt_paydown": debt_paydown,
        "year_1_cash_flow_for_debt_repayment": float(
            debt_schedule.loc["Year 1", "Cash flow available for debt repayment"]
        ),
        "exit_year_cash_flow_for_debt_repayment": float(
            debt_schedule.loc[exit_label, "Cash flow available for debt repayment"]
        ),
        "exit_total_debt_to_ebitda": float(credit_metrics.loc[exit_label, "Total Debt / EBITDA"]),
        "exit_interest_coverage": float(credit_metrics.loc[exit_label, "EBITDA / Interest expense"]),
        "total_debt_repaid_pct": float(credit_metrics.loc[exit_label, "% of total debt repaid"]),
    }


def _build_prompt(payload: dict[str, float | int]) -> str:
    """Keep the prompt narrow so the narrative stays grounded in the computed case."""
    return f"""
You are assisting with a classroom LBO case. Write a tight investment note in English for finance students.

Rules:
- Use only the numbers provided below.
- Do not invent any figures, assumptions, or strategic claims.
- Keep the tone disciplined, finance-oriented, and professional.
- Avoid generic private-equity language and avoid marketing phrasing.
- Interpret the economics directly: operating performance, deleveraging, and exit valuation.
- Mention the actual IRR, MOIC, debt paydown, and leverage level explicitly.
- Keep the full answer below 180 words.
- Use exactly these four headings:
  1. Executive summary
  2. Value creation drivers
  3. Key risks
  4. Overall assessment
- Under each heading, write 1 to 2 compact bullet points.

Case metrics:
{json.dumps(payload, indent=2)}
""".strip()


def generate_investment_commentary(
    results: dict[str, Any],
    api_key: str | None = None,
    model: str = DEFAULT_GEMINI_MODEL,
) -> dict[str, Any]:
    """Generate a short finance-oriented commentary with graceful fallback."""
    resolved_api_key = _resolve_api_key(api_key)

    if not resolved_api_key:
        return {
            "success": False,
            "message": "Gemini commentary skipped: set GEMINI_API_KEY or GOOGLE_API_KEY to enable the optional reporting layer.",
        }

    if genai is None:
        return {
            "success": False,
            "message": "Gemini commentary skipped: install the optional dependency `google-genai` first.",
        }

    payload = build_commentary_payload(results)
    prompt = _build_prompt(payload)

    try:
        # The model is used only after the quantitative outputs already exist.
        # It is a reporting layer, not a source of valuation logic.
        client = genai.Client(api_key=resolved_api_key)
        response = client.models.generate_content(model=model, contents=prompt)
        text = (response.text or "").strip()
    except Exception as exc:  # pragma: no cover - network / credential dependent
        return {
            "success": False,
            "message": _format_gemini_exception(exc),
        }

    if not text:
        return {
            "success": False,
            "message": "Gemini commentary skipped: the API response did not contain text.",
        }

    return {
        "success": True,
        "message": text,
        "payload": payload,
        "model": model,
    }
