"""Teaching-oriented LBO engine.

The goal of this module is not to be a production valuation library.
It is intentionally small and explicit so students can trace the full
LBO logic from entry valuation to sponsor returns.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

try:
    import numpy_financial as npf
except ImportError:  # pragma: no cover - fallback used only in lean environments
    npf = None


BASE_CASE_INPUTS: dict[str, dict[str, Any]] = {
    # These are the classroom case inputs taken from the Excel model.
    # Keeping them grouped this way mirrors the way an analyst thinks about the deal.
    "deal": {
        "share_price": 2.5,
        "diluted_shares": 369.0,
        "net_debt": 135.2,
    },
    "valuation": {
        "entry_multiple": 7.0,
    },
    "financing": {
        "senior_debt_pct": 0.35,
        "subordinated_debt_pct": 0.35,
        "senior_interest_rate": 0.06,
        "subordinated_interest_rate": 0.065,
    },
    "operating": {
        "ebit": 163.9,
        "d_and_a": 76.9,
        "owc": -86.9,
        "capex": 127.7,
    },
    "tax_and_fees": {
        "fees_pct_of_ev": 0.005,
        "tax_rate": 0.35,
    },
    "exit": {
        "exit_multiple": 7.0,
        "exit_year": 3,
    },
    "projection": {
        "projection_years": 3,
        "ebit_growth_shift": 0.0,
        "d_and_a_growth_shift": 0.0,
        "owc_growth_shift": 0.0,
        "capex_growth_shift": 0.0,
    },
}

_BASE_OPERATING_CASE = pd.DataFrame(
    {
        # Year 0 is the historical starting point.
        # Years 1-3 reproduce the classroom projection so the notebook stays aligned with Excel.
        "Year": [0, 1, 2, 3],
        "EBIT": [163.9, 177.2925, 186.157125, 195.46498125],
        "D&A": [76.9, 81.8978, 84.7074766, 87.7720210226],
        "OWC": [-86.9, -92.1921, -96.801705, -101.64179025],
        "Capex": [127.7, 113.4672, 119.14056, 125.097588],
    }
).set_index("Year")


def _series_growth_rates(series: pd.Series) -> list[float]:
    """Translate a reference series into year-over-year growth rates."""
    values = series.to_numpy(dtype=float)
    return [(values[idx] / values[idx - 1]) - 1 for idx in range(1, len(values))]


_BASE_GROWTH_RATES = {
    "EBIT": _series_growth_rates(_BASE_OPERATING_CASE["EBIT"]),
    "D&A": _series_growth_rates(_BASE_OPERATING_CASE["D&A"]),
    "OWC": _series_growth_rates(_BASE_OPERATING_CASE["OWC"].abs()),
    "Capex": _series_growth_rates(_BASE_OPERATING_CASE["Capex"]),
}


def _compute_irr(cash_flows: list[float]) -> float:
    """Compute IRR with a transparent fallback if numpy-financial is unavailable.

    The fallback uses a simple bisection approach because it is easy to explain:
    we look for the discount rate that sets the NPV of sponsor cash flows to zero.
    """
    if npf is not None:
        return float(npf.irr(cash_flows))

    def npv(rate: float) -> float:
        return sum(cash_flow / ((1 + rate) ** period) for period, cash_flow in enumerate(cash_flows))

    lower_bound = -0.95
    upper_bound = 5.0
    lower_value = npv(lower_bound)
    upper_value = npv(upper_bound)

    while lower_value * upper_value > 0 and upper_bound < 100:
        upper_bound *= 2
        upper_value = npv(upper_bound)

    if lower_value * upper_value > 0:
        raise ValueError("IRR could not be bracketed for the provided cash flows.")

    for _ in range(200):
        midpoint = (lower_bound + upper_bound) / 2
        midpoint_value = npv(midpoint)
        if abs(midpoint_value) < 1e-10:
            return midpoint
        if lower_value * midpoint_value <= 0:
            upper_bound = midpoint
            upper_value = midpoint_value
        else:
            lower_bound = midpoint
            lower_value = midpoint_value

    return (lower_bound + upper_bound) / 2


def get_base_case_inputs() -> dict[str, dict[str, Any]]:
    """Return a fresh copy so notebook or app scenarios never mutate the base case."""
    return deepcopy(BASE_CASE_INPUTS)


def _deep_update(target: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Apply nested scenario overrides while keeping the original structure intact."""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def _validate_inputs(inputs: dict[str, dict[str, Any]]) -> None:
    """Keep the classroom model within a simple and teachable range."""
    financing = inputs["financing"]
    projection = inputs["projection"]
    exit_assumptions = inputs["exit"]

    total_debt_pct = financing["senior_debt_pct"] + financing["subordinated_debt_pct"]
    if total_debt_pct >= 1:
        raise ValueError("Senior debt % plus subordinated debt % must stay below 100%.")

    if projection["projection_years"] > 3:
        raise ValueError("The classroom operating projection is intentionally capped at 3 years.")

    if exit_assumptions["exit_year"] > projection["projection_years"]:
        raise ValueError("Exit year cannot exceed the projection horizon.")


def _project_series(
    historical_value: float,
    growth_rates: list[float],
    shift: float = 0.0,
    negative_balance: bool = False,
) -> list[float]:
    """Project one operating line from a historical value.

    We reuse the Excel-case growth pattern and then optionally shift it up or down
    for scenarios. This keeps the sensitivity logic close to the original case.
    """
    values = [historical_value]
    current_value = abs(historical_value) if negative_balance else historical_value

    for rate in growth_rates:
        current_value = current_value * (1 + rate + shift)
        values.append(-current_value if negative_balance else current_value)

    return values


def build_inputs_table(inputs: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Convert raw assumptions into a classroom-friendly table."""
    financing = inputs["financing"]
    tax_and_fees = inputs["tax_and_fees"]
    exit_assumptions = inputs["exit"]

    rows = [
        ("Deal inputs", "Share price", inputs["deal"]["share_price"], "EUR/share"),
        ("Deal inputs", "Diluted shares", inputs["deal"]["diluted_shares"], "m shares"),
        ("Deal inputs", "Net debt", inputs["deal"]["net_debt"], "EURm"),
        ("Valuation assumptions", "Entry multiple", inputs["valuation"]["entry_multiple"], "x"),
        ("Financing assumptions", "Senior debt", financing["senior_debt_pct"], "%"),
        (
            "Financing assumptions",
            "Subordinated debt",
            financing["subordinated_debt_pct"],
            "%",
        ),
        (
            "Financing assumptions",
            "Sponsor equity",
            1 - financing["senior_debt_pct"] - financing["subordinated_debt_pct"],
            "%",
        ),
        (
            "Financing assumptions",
            "Senior interest rate",
            financing["senior_interest_rate"],
            "%",
        ),
        (
            "Financing assumptions",
            "Subordinated interest rate",
            financing["subordinated_interest_rate"],
            "%",
        ),
        ("Operating starting point", "EBIT", inputs["operating"]["ebit"], "EURm"),
        ("Operating starting point", "D&A", inputs["operating"]["d_and_a"], "EURm"),
        (
            "Operating starting point",
            "EBITDA",
            inputs["operating"]["ebit"] + inputs["operating"]["d_and_a"],
            "EURm",
        ),
        ("Operating starting point", "Operating working capital", inputs["operating"]["owc"], "EURm"),
        ("Operating starting point", "Capital expenditure", inputs["operating"]["capex"], "EURm"),
        ("Tax / fees", "Fees as % of entry EV", tax_and_fees["fees_pct_of_ev"], "%"),
        ("Tax / fees", "Tax rate", tax_and_fees["tax_rate"], "%"),
        ("Exit assumptions", "Exit multiple", exit_assumptions["exit_multiple"], "x"),
        ("Exit assumptions", "Exit year", exit_assumptions["exit_year"], "year"),
    ]

    table = pd.DataFrame(rows, columns=["Group", "Assumption", "Value", "Unit"])
    return table


def build_operating_projection(inputs: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Build the historical point plus the 3-year operating projection.

    If the user is still on the exact classroom case, we keep the Excel values.
    If the user changes assumptions in Streamlit or in a scenario, we preserve
    the same shape of the case but shift growth rates transparently.
    """
    projection_years = inputs["projection"]["projection_years"]
    year_index = list(range(projection_years + 1))

    use_exact_classroom_case = (
        inputs["operating"] == BASE_CASE_INPUTS["operating"]
        and all(abs(inputs["projection"][f"{metric}_growth_shift"]) < 1e-12 for metric in ["ebit", "d_and_a", "owc", "capex"])
    )

    if use_exact_classroom_case:
        # This branch keeps the Python base case numerically aligned with the Excel file.
        projection = _BASE_OPERATING_CASE.loc[year_index].copy()
    else:
        projection = pd.DataFrame(index=year_index)
        # Each line is projected separately so students can see that EBITDA is built
        # from EBIT plus D&A, not entered as a disconnected assumption.
        projection["EBIT"] = _project_series(
            historical_value=inputs["operating"]["ebit"],
            growth_rates=_BASE_GROWTH_RATES["EBIT"][:projection_years],
            shift=inputs["projection"]["ebit_growth_shift"],
        )
        projection["D&A"] = _project_series(
            historical_value=inputs["operating"]["d_and_a"],
            growth_rates=_BASE_GROWTH_RATES["D&A"][:projection_years],
            shift=inputs["projection"]["d_and_a_growth_shift"],
        )
        projection["OWC"] = _project_series(
            historical_value=inputs["operating"]["owc"],
            growth_rates=_BASE_GROWTH_RATES["OWC"][:projection_years],
            shift=inputs["projection"]["owc_growth_shift"],
            negative_balance=True,
        )
        projection["Capex"] = _project_series(
            historical_value=inputs["operating"]["capex"],
            growth_rates=_BASE_GROWTH_RATES["Capex"][:projection_years],
            shift=inputs["projection"]["capex_growth_shift"],
        )

    # EBITDA is the operating earnings measure used for valuation and leverage ratios.
    projection["EBITDA"] = projection["EBIT"] + projection["D&A"]
    # A more negative working-capital balance is a source of cash in this classroom case,
    # so the change is computed as prior balance minus current balance.
    projection["Change in OWC"] = projection["OWC"].shift(1) - projection["OWC"]
    projection["Change in OWC"] = projection["Change in OWC"].fillna(0.0)
    projection.index = [f"Historical" if year == 0 else f"Year {year}" for year in projection.index]

    return projection


def build_entry_valuation(inputs: dict[str, dict[str, Any]], operating_projection: pd.DataFrame) -> pd.DataFrame:
    """Move from historical EBITDA to entry EV and implied equity value."""
    historical_ebitda = operating_projection.loc["Historical", "EBITDA"]
    enterprise_value = historical_ebitda * inputs["valuation"]["entry_multiple"]
    equity_value = enterprise_value - inputs["deal"]["net_debt"]
    implied_offer_price = equity_value / inputs["deal"]["diluted_shares"]
    takeover_premium = implied_offer_price / inputs["deal"]["share_price"] - 1

    rows = [
        ("Historical EBITDA", historical_ebitda),
        ("Entry Enterprise Value", enterprise_value),
        ("Less: Net debt", -inputs["deal"]["net_debt"]),
        ("Implied Equity Value", equity_value),
        ("Implied offer price per share", implied_offer_price),
        ("Takeover premium", takeover_premium),
    ]

    return pd.DataFrame(rows, columns=["Metric", "Value"])


def build_sources_and_uses(
    inputs: dict[str, dict[str, Any]],
    entry_valuation: pd.DataFrame,
) -> pd.DataFrame:
    """Build the transaction funding schedule.

    This is where valuation becomes deal mechanics: uses are what must be paid,
    and sources explain how the sponsor funds those uses.
    """
    financing = inputs["financing"]
    tax_and_fees = inputs["tax_and_fees"]

    equity_value = entry_valuation.loc[entry_valuation["Metric"] == "Implied Equity Value", "Value"].iloc[0]
    entry_enterprise_value = entry_valuation.loc[
        entry_valuation["Metric"] == "Entry Enterprise Value",
        "Value",
    ].iloc[0]
    fees = entry_enterprise_value * tax_and_fees["fees_pct_of_ev"]
    total_uses = equity_value + inputs["deal"]["net_debt"] + fees

    senior_debt = total_uses * financing["senior_debt_pct"]
    subordinated_debt = total_uses * financing["subordinated_debt_pct"]
    sponsor_equity = total_uses - senior_debt - subordinated_debt

    sources_and_uses = pd.DataFrame(
        {
            "Uses": {
                "Acquisition of equity": equity_value,
                "Refinanced debt": inputs["deal"]["net_debt"],
                "Fees": fees,
                "Total Uses": total_uses,
            },
            "Sources": {
                "Senior debt": senior_debt,
                "Subordinated debt": subordinated_debt,
                "Sponsor equity": sponsor_equity,
                "Total Sources": total_uses,
            },
        }
    )

    return sources_and_uses


def _solve_cash_sweep(
    ebit: float,
    depreciation: float,
    change_in_owc: float,
    capex: float,
    beginning_senior_debt: float,
    beginning_subordinated_debt: float,
    beginning_cash_balance: float,
    senior_interest_rate: float,
    subordinated_interest_rate: float,
    tax_rate: float,
) -> dict[str, float]:
    """Solve one year of the debt sweep.

    The repayment amount affects average debt, which affects interest expense,
    which affects taxes and therefore cash available for repayment. To keep the
    classroom logic explicit, we solve that circularity with a very transparent
    iteration instead of hiding it inside a more advanced numerical routine.
    """
    senior_repayment = 0.0
    subordinated_repayment = 0.0

    for _ in range(250):
        # Interest is charged on the average balance, just like in the Excel case.
        senior_interest = senior_interest_rate * max(beginning_senior_debt - 0.5 * senior_repayment, 0.0)
        subordinated_interest = subordinated_interest_rate * max(
            beginning_subordinated_debt - 0.5 * subordinated_repayment,
            0.0,
        )
        total_interest = senior_interest + subordinated_interest
        profit_before_tax = ebit - total_interest
        taxes = profit_before_tax * tax_rate
        # This is the central LBO cash-flow identity:
        # EBITDA
        # - interest
        # - taxes
        # +/- working capital release
        # - capex
        # = cash available to reduce debt
        cash_available_for_debt_repayment = (
            ebit
            + depreciation
            - total_interest
            - taxes
            + change_in_owc
            - capex
        )
        cash_available_for_sweep = max(beginning_cash_balance + cash_available_for_debt_repayment, 0.0)

        # Senior debt is repaid first. Only leftover cash can reach subordinated debt.
        updated_senior_repayment = min(beginning_senior_debt, cash_available_for_sweep)
        updated_subordinated_repayment = min(
            beginning_subordinated_debt,
            max(cash_available_for_sweep - updated_senior_repayment, 0.0),
        )

        if (
            abs(updated_senior_repayment - senior_repayment) < 1e-10
            and abs(updated_subordinated_repayment - subordinated_repayment) < 1e-10
        ):
            senior_repayment = updated_senior_repayment
            subordinated_repayment = updated_subordinated_repayment
            break

        senior_repayment = updated_senior_repayment
        subordinated_repayment = updated_subordinated_repayment

    ending_senior_debt = max(beginning_senior_debt - senior_repayment, 0.0)
    ending_subordinated_debt = max(beginning_subordinated_debt - subordinated_repayment, 0.0)
    ending_cash_balance = (
        beginning_cash_balance
        + cash_available_for_debt_repayment
        - senior_repayment
        - subordinated_repayment
    )

    return {
        "Interest expense on senior debt": senior_interest,
        "Interest expense on subordinated debt": subordinated_interest,
        "Interest expense": total_interest,
        "Profit before tax": profit_before_tax,
        "Tax expense": taxes,
        "Cash flow available for debt repayment": cash_available_for_debt_repayment,
        "Repayment of senior debt": senior_repayment,
        "Repayment of subordinated debt": subordinated_repayment,
        "Ending senior debt": ending_senior_debt,
        "Ending subordinated debt": ending_subordinated_debt,
        "Ending cash balance": ending_cash_balance,
    }


def build_debt_schedule(
    inputs: dict[str, dict[str, Any]],
    operating_projection: pd.DataFrame,
    sources_and_uses: pd.DataFrame,
) -> pd.DataFrame:
    """Create the year-by-year debt schedule used in the notebook and app."""
    financing = inputs["financing"]

    beginning_senior_debt = float(sources_and_uses.loc["Senior debt", "Sources"])
    beginning_subordinated_debt = float(sources_and_uses.loc["Subordinated debt", "Sources"])
    beginning_cash_balance = 0.0

    debt_rows: list[dict[str, float | str]] = []
    projected_years = [label for label in operating_projection.index if label != "Historical"]

    for year_label in projected_years:
        # Each projected year reuses the same cash-sweep logic.
        operating_row = operating_projection.loc[year_label]
        solved_year = _solve_cash_sweep(
            ebit=float(operating_row["EBIT"]),
            depreciation=float(operating_row["D&A"]),
            change_in_owc=float(operating_row["Change in OWC"]),
            capex=float(operating_row["Capex"]),
            beginning_senior_debt=beginning_senior_debt,
            beginning_subordinated_debt=beginning_subordinated_debt,
            beginning_cash_balance=beginning_cash_balance,
            senior_interest_rate=financing["senior_interest_rate"],
            subordinated_interest_rate=financing["subordinated_interest_rate"],
            tax_rate=inputs["tax_and_fees"]["tax_rate"],
        )

        debt_rows.append(
            {
                "Year": year_label,
                "Beginning senior debt": beginning_senior_debt,
                "Beginning subordinated debt": beginning_subordinated_debt,
                **solved_year,
            }
        )

        # Ending balances of one year become beginning balances of the next year.
        beginning_senior_debt = solved_year["Ending senior debt"]
        beginning_subordinated_debt = solved_year["Ending subordinated debt"]
        beginning_cash_balance = solved_year["Ending cash balance"]

    return pd.DataFrame(debt_rows).set_index("Year")


def build_credit_metrics(
    operating_projection: pd.DataFrame,
    debt_schedule: pd.DataFrame,
    sources_and_uses: pd.DataFrame,
) -> pd.DataFrame:
    """Build lender-style monitoring ratios from the operating and debt schedules."""
    projected_operating = operating_projection.drop(index="Historical")
    opening_total_debt = float(
        sources_and_uses.loc["Senior debt", "Sources"] + sources_and_uses.loc["Subordinated debt", "Sources"]
    )

    metrics = pd.DataFrame(index=debt_schedule.index)
    # Coverage asks whether operating earnings are comfortably servicing interest.
    metrics["EBITDA / Interest expense"] = (
        projected_operating["EBITDA"].to_numpy(dtype=float) / debt_schedule["Interest expense"].to_numpy(dtype=float)
    )
    # Leverage asks how much debt remains relative to the earnings base.
    metrics["Total Debt / EBITDA"] = (
        debt_schedule["Ending senior debt"].to_numpy(dtype=float)
        + debt_schedule["Ending subordinated debt"].to_numpy(dtype=float)
    ) / projected_operating["EBITDA"].to_numpy(dtype=float)
    metrics["% of total debt repaid"] = 1 - (
        debt_schedule["Ending senior debt"].to_numpy(dtype=float)
        + debt_schedule["Ending subordinated debt"].to_numpy(dtype=float)
    ) / opening_total_debt
    metrics["% of senior debt repaid"] = 1 - (
        debt_schedule["Ending senior debt"].to_numpy(dtype=float)
        / debt_schedule["Beginning senior debt"].to_numpy(dtype=float)
    )

    return metrics


def build_exit_returns(
    inputs: dict[str, dict[str, Any]],
    operating_projection: pd.DataFrame,
    debt_schedule: pd.DataFrame,
    sources_and_uses: pd.DataFrame,
    entry_valuation: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Translate the operating case and final debt balance into sponsor returns."""
    exit_year = inputs["exit"]["exit_year"]
    exit_label = f"Year {exit_year}"
    sponsor_equity = float(sources_and_uses.loc["Sponsor equity", "Sources"])
    entry_enterprise_value = float(
        entry_valuation.loc[entry_valuation["Metric"] == "Entry Enterprise Value", "Value"].iloc[0]
    )
    fees = float(sources_and_uses.loc["Fees", "Uses"])
    opening_total_debt = float(
        sources_and_uses.loc["Senior debt", "Sources"] + sources_and_uses.loc["Subordinated debt", "Sources"]
    )

    exit_ebitda = float(operating_projection.loc[exit_label, "EBITDA"])
    exit_enterprise_value = exit_ebitda * inputs["exit"]["exit_multiple"]
    exit_net_debt = float(
        debt_schedule.loc[exit_label, "Ending senior debt"]
        + debt_schedule.loc[exit_label, "Ending subordinated debt"]
        - debt_schedule.loc[exit_label, "Ending cash balance"]
    )
    exit_equity_value = exit_enterprise_value - exit_net_debt

    # Sponsor cash flows are intentionally simple for the class:
    # equity outflow at entry, no interim dividends, full equity value realized at exit.
    sponsor_cash_flows = [-sponsor_equity] + [0.0] * (exit_year - 1) + [exit_equity_value]
    irr = _compute_irr(sponsor_cash_flows)
    moic = exit_equity_value / sponsor_equity

    returns_summary = pd.DataFrame(
        {
            "Metric": [
                "Exit EBITDA",
                "Exit Enterprise Value",
                "Exit net debt",
                "Exit Equity Value",
                "Sponsor equity invested",
                "IRR",
                "MOIC",
            ],
            "Value": [
                exit_ebitda,
                exit_enterprise_value,
                exit_net_debt,
                exit_equity_value,
                sponsor_equity,
                irr,
                moic,
            ],
        }
    )

    # This bridge keeps the sources of value creation visible for students.
    operating_improvement = (exit_ebitda - operating_projection.loc["Historical", "EBITDA"]) * inputs["valuation"][
        "entry_multiple"
    ]
    multiple_effect = exit_ebitda * (inputs["exit"]["exit_multiple"] - inputs["valuation"]["entry_multiple"])
    debt_paydown = opening_total_debt - exit_net_debt

    value_creation_bridge = pd.DataFrame(
        {
            "Driver": [
                "Sponsor equity invested",
                "Operating improvement",
                "Multiple effect",
                "Debt paydown",
                "Entry fees",
                "Exit equity value",
            ],
            "Value": [
                sponsor_equity,
                operating_improvement,
                multiple_effect,
                debt_paydown,
                -fees,
                exit_equity_value,
            ],
        }
    )

    return returns_summary, value_creation_bridge


def _run_model_core(model_inputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Run the core LBO steps once and return all intermediate tables."""
    inputs_table = build_inputs_table(model_inputs)
    operating_projection = build_operating_projection(model_inputs)
    entry_valuation = build_entry_valuation(model_inputs, operating_projection)
    sources_and_uses = build_sources_and_uses(model_inputs, entry_valuation)
    debt_schedule = build_debt_schedule(model_inputs, operating_projection, sources_and_uses)
    credit_metrics = build_credit_metrics(operating_projection, debt_schedule, sources_and_uses)
    returns_summary, value_creation_bridge = build_exit_returns(
        model_inputs,
        operating_projection,
        debt_schedule,
        sources_and_uses,
        entry_valuation,
    )

    return {
        "inputs": model_inputs,
        "inputs_table": inputs_table,
        "operating_projection": operating_projection,
        "entry_valuation": entry_valuation,
        "sources_and_uses": sources_and_uses,
        "debt_schedule": debt_schedule,
        "credit_metrics": credit_metrics,
        "returns_summary": returns_summary,
        "value_creation_bridge": value_creation_bridge,
    }


def _run_scenario(
    base_inputs: dict[str, dict[str, Any]],
    overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create one scenario from the base case without overwriting the original inputs."""
    scenario_inputs = deepcopy(base_inputs)
    if overrides:
        scenario_inputs = _deep_update(scenario_inputs, deepcopy(overrides))
    _validate_inputs(scenario_inputs)
    return _run_model_core(scenario_inputs)


def build_sensitivity_tables(
    base_inputs: dict[str, dict[str, Any]],
) -> dict[str, pd.DataFrame]:
    """Build the two classroom sensitivity tables.

    The goal is not exhaustive scenario design. The goal is to show students
    that returns move materially when valuation or operating assumptions change.
    """
    entry_grid = np.arange(6.0, 8.5, 0.5)
    exit_grid = np.arange(6.0, 8.5, 0.5)
    growth_shift_grid = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])

    entry_exit_irr = pd.DataFrame(index=entry_grid, columns=exit_grid, dtype=float)
    exit_growth_irr = pd.DataFrame(index=growth_shift_grid, columns=exit_grid, dtype=float)
    exit_growth_moic = pd.DataFrame(index=growth_shift_grid, columns=exit_grid, dtype=float)

    for entry_multiple in entry_grid:
        for exit_multiple in exit_grid:
            # This table isolates valuation discipline at entry and exit.
            scenario = _run_scenario(
                base_inputs,
                {
                    "valuation": {"entry_multiple": float(entry_multiple)},
                    "exit": {
                        "exit_multiple": float(exit_multiple),
                        "exit_year": base_inputs["exit"]["exit_year"],
                    },
                }
            )
            irr = float(
                scenario["returns_summary"].loc[
                    scenario["returns_summary"]["Metric"] == "IRR",
                    "Value",
                ].iloc[0]
            )
            entry_exit_irr.loc[entry_multiple, exit_multiple] = irr

    for growth_shift in growth_shift_grid:
        for exit_multiple in exit_grid:
            # This table combines an exit valuation assumption with a change
            # in operating momentum, which is one of the clearest classroom stress tests.
            scenario = _run_scenario(
                base_inputs,
                {
                    "projection": {
                        "ebit_growth_shift": float(growth_shift),
                        "d_and_a_growth_shift": float(growth_shift * 0.5),
                    },
                    "exit": {
                        "exit_multiple": float(exit_multiple),
                        "exit_year": base_inputs["exit"]["exit_year"],
                    },
                }
            )
            returns_summary = scenario["returns_summary"].set_index("Metric")
            exit_growth_irr.loc[growth_shift, exit_multiple] = float(returns_summary.loc["IRR", "Value"])
            exit_growth_moic.loc[growth_shift, exit_multiple] = float(returns_summary.loc["MOIC", "Value"])

    entry_exit_irr.index.name = "Entry Multiple"
    entry_exit_irr.columns.name = "Exit Multiple"
    exit_growth_irr.index.name = "EBIT growth shift"
    exit_growth_irr.columns.name = "Exit Multiple"
    exit_growth_moic.index.name = "EBIT growth shift"
    exit_growth_moic.columns.name = "Exit Multiple"

    return {
        "entry_exit_irr": entry_exit_irr,
        "exit_growth_irr": exit_growth_irr,
        "exit_growth_moic": exit_growth_moic,
    }


def run_lbo_model(
    overrides: dict[str, dict[str, Any]] | None = None,
    base_inputs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Public entry point used by the notebook and Streamlit app.

    ``base_inputs`` lets the teaching notebook keep its assumptions visible
    while the Streamlit app can keep using the shared default case.
    """
    model_inputs = deepcopy(base_inputs) if base_inputs is not None else get_base_case_inputs()
    if overrides:
        model_inputs = _deep_update(model_inputs, deepcopy(overrides))

    _validate_inputs(model_inputs)
    results = _run_model_core(model_inputs)
    results["sensitivities"] = build_sensitivity_tables(model_inputs)
    return results
