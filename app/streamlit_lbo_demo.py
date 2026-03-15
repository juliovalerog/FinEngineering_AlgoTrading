from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.ticker import FuncFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.gemini_reporting import generate_investment_commentary
from utils.lbo_engine import get_base_case_inputs, run_lbo_model


COLORS = {
    "navy": "#14324A",
    "teal": "#1F6F78",
    "sand": "#D7C3A3",
    "slate": "#5A6C7D",
    "green": "#2D7D46",
    "red": "#A6473C",
    "light": "#F6F3EE",
}


def eur_millions(value: float, _position: int | None = None) -> str:
    # Keeping units explicit helps students read every chart in finance terms.
    return f"EUR {value:,.0f}m"


def percent(value: float, decimals: int = 1) -> str:
    # Percent formatting is reused in KPI cards and labels.
    return f"{value * 100:.{decimals}f}%"


def style_page() -> None:
    # The page style is intentionally restrained: the app should feel like a
    # professional prototype that supports the notebook, not a flashy dashboard.
    st.set_page_config(page_title="LBO Teaching Demo", layout="wide")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(180deg, #fbfaf6 0%, #f1ece3 100%);
        }}
        h1, h2, h3 {{
            color: {COLORS["navy"]};
            letter-spacing: -0.02em;
        }}
        [data-testid="stMetricValue"] {{
            color: {COLORS["navy"]};
        }}
        .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1250px;
        }}
        .intro-card {{
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(20, 50, 74, 0.10);
            padding: 1.2rem 1.4rem;
            border-radius: 18px;
            margin-bottom: 1rem;
        }}
        .section-note {{
            color: {COLORS["slate"]};
            font-size: 0.97rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def plot_entry_bridge(entry_valuation: pd.DataFrame) -> plt.Figure:
    # This chart answers the first question in the deal:
    # how do we move from enterprise value to equity value?
    figure, axis = plt.subplots(figsize=(7, 4))
    bridge = pd.DataFrame(
        {
            "Label": ["Enterprise Value", "Net Debt", "Equity Value"],
            "Value": [
                entry_valuation.loc[entry_valuation["Metric"] == "Entry Enterprise Value", "Value"].iloc[0],
                entry_valuation.loc[entry_valuation["Metric"] == "Less: Net debt", "Value"].iloc[0],
                entry_valuation.loc[entry_valuation["Metric"] == "Implied Equity Value", "Value"].iloc[0],
            ],
            "Color": [COLORS["navy"], COLORS["red"], COLORS["teal"]],
        }
    )
    axis.bar(bridge["Label"], bridge["Value"], color=bridge["Color"])
    for idx, row in bridge.iterrows():
        axis.text(idx, row["Value"] + 35, f"{row['Value']:,.0f}", ha="center", color=COLORS["navy"], fontsize=10)
    axis.yaxis.set_major_formatter(FuncFormatter(eur_millions))
    axis.set_title("Entry valuation bridge")
    axis.set_ylabel("EUR millions")
    axis.spines[["top", "right"]].set_visible(False)
    return figure


def plot_sources_and_uses(sources_and_uses: pd.DataFrame) -> plt.Figure:
    # Sources & Uses is easier to explain when students can compare both sides visually.
    uses = sources_and_uses["Uses"].dropna().drop("Total Uses")
    sources = sources_and_uses["Sources"].dropna().drop("Total Sources")

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    axes[0].barh(uses.index, uses.values, color=[COLORS["sand"], COLORS["slate"], COLORS["red"]])
    axes[0].set_title("Uses of funds")
    axes[1].barh(sources.index, sources.values, color=[COLORS["navy"], COLORS["teal"], COLORS["green"]])
    axes[1].set_title("Sources of funds")

    for axis in axes:
        axis.xaxis.set_major_formatter(FuncFormatter(eur_millions))
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlabel("EUR millions")

    axes[1].text(
        0.98,
        0.08,
        f"Equity check: {sources['Sponsor equity'] / sources.sum():.0%} of total sources",
        transform=axes[1].transAxes,
        ha="right",
        color=COLORS["navy"],
        fontsize=10,
    )
    return figure


def plot_operating_projection(operating_projection: pd.DataFrame, debt_schedule: pd.DataFrame) -> plt.Figure:
    # The two panels tell a single story:
    # operating performance improves, and part of that improvement becomes debt-paying cash.
    projected = operating_projection.drop(index="Historical")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(projected.index, projected["EBITDA"], marker="o", linewidth=2.5, color=COLORS["navy"])
    axes[0].bar(projected.index, projected["EBIT"], alpha=0.3, color=COLORS["teal"])
    axes[0].set_title("Operating performance")
    axes[0].set_ylabel("EUR millions")
    axes[0].yaxis.set_major_formatter(FuncFormatter(eur_millions))
    axes[0].legend(["EBITDA", "EBIT"], frameon=False)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].annotate(
        f"Exit-year EBITDA: {projected['EBITDA'].iloc[-1]:.1f}",
        xy=(len(projected.index) - 1, projected["EBITDA"].iloc[-1]),
        xytext=(1.5, projected["EBITDA"].iloc[-1] + 12),
        textcoords="data",
        arrowprops={"arrowstyle": "->", "color": COLORS["navy"]},
        color=COLORS["navy"],
    )

    axes[1].bar(
        debt_schedule.index,
        debt_schedule["Cash flow available for debt repayment"],
        color=COLORS["green"],
    )
    axes[1].set_title("Cash flow available for debt repayment")
    axes[1].set_ylabel("EUR millions")
    axes[1].yaxis.set_major_formatter(FuncFormatter(eur_millions))
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].plot(
        debt_schedule.index,
        debt_schedule["Cash flow available for debt repayment"],
        color=COLORS["navy"],
        marker="o",
        linewidth=1.5,
    )

    return figure


def plot_deleveraging(debt_schedule: pd.DataFrame) -> plt.Figure:
    # Stacked bars show capital structure composition, while the line highlights total debt reduction.
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(
        debt_schedule.index,
        debt_schedule["Ending senior debt"],
        color=COLORS["navy"],
        label="Senior debt",
    )
    axis.bar(
        debt_schedule.index,
        debt_schedule["Ending subordinated debt"],
        bottom=debt_schedule["Ending senior debt"],
        color=COLORS["sand"],
        label="Subordinated debt",
    )
    axis.set_title("Debt evolution and composition")
    axis.set_ylabel("EUR millions")
    axis.yaxis.set_major_formatter(FuncFormatter(eur_millions))
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    total_debt = debt_schedule["Ending senior debt"] + debt_schedule["Ending subordinated debt"]
    axis.plot(debt_schedule.index, total_debt, color=COLORS["red"], linewidth=2, marker="o", label="Total debt")
    axis.legend(frameon=False, ncol=3, loc="upper right")
    return figure


def plot_credit_metrics(credit_metrics: pd.DataFrame) -> plt.Figure:
    # These are lender-facing metrics, so they sit immediately after the debt schedule.
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(
        credit_metrics.index,
        credit_metrics["Total Debt / EBITDA"],
        marker="o",
        linewidth=2.5,
        color=COLORS["red"],
    )
    axes[0].set_title("Leverage trend")
    axes[0].set_ylabel("x")
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].annotate(
        f"{credit_metrics['Total Debt / EBITDA'].iloc[-1]:.2f}x",
        xy=(len(credit_metrics.index) - 1, credit_metrics["Total Debt / EBITDA"].iloc[-1]),
        xytext=(1.6, credit_metrics["Total Debt / EBITDA"].iloc[-1] + 0.25),
        textcoords="data",
        arrowprops={"arrowstyle": "->", "color": COLORS["red"]},
        color=COLORS["red"],
    )

    axes[1].plot(
        credit_metrics.index,
        credit_metrics["EBITDA / Interest expense"],
        marker="o",
        linewidth=2.5,
        color=COLORS["green"],
    )
    axes[1].set_title("Interest coverage trend")
    axes[1].set_ylabel("x")
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].annotate(
        f"{credit_metrics['EBITDA / Interest expense'].iloc[-1]:.2f}x",
        xy=(len(credit_metrics.index) - 1, credit_metrics["EBITDA / Interest expense"].iloc[-1]),
        xytext=(1.5, credit_metrics["EBITDA / Interest expense"].iloc[-1] + 0.25),
        textcoords="data",
        arrowprops={"arrowstyle": "->", "color": COLORS["green"]},
        color=COLORS["green"],
    )

    return figure


def plot_sensitivity_heatmap(table: pd.DataFrame, title: str, value_format: str) -> plt.Figure:
    # The heatmap is intentionally simple because the teaching point is pattern recognition:
    # students should immediately see where returns improve or deteriorate.
    figure, axis = plt.subplots(figsize=(7, 5))
    image = axis.imshow(table.to_numpy(dtype=float), cmap="YlGnBu", aspect="auto")
    axis.set_xticks(np.arange(len(table.columns)))
    axis.set_xticklabels(table.columns)
    axis.set_yticks(np.arange(len(table.index)))
    axis.set_yticklabels(table.index)
    axis.set_title(title)
    axis.set_xlabel(table.columns.name)
    axis.set_ylabel(table.index.name)

    for row_idx, row_value in enumerate(table.index):
        for col_idx, col_value in enumerate(table.columns):
            display_value = table.loc[row_value, col_value]
            axis.text(
                col_idx,
                row_idx,
                format(display_value, value_format),
                ha="center",
                va="center",
                color="white" if display_value > table.to_numpy(dtype=float).mean() else COLORS["navy"],
                fontsize=9,
            )

    figure.colorbar(image, ax=axis, shrink=0.85)
    return figure


def plot_return_bridge(returns_summary: pd.DataFrame, value_creation_bridge: pd.DataFrame) -> plt.Figure:
    # Returns deserve two views:
    # 1) what equity went in versus what came out,
    # 2) which levers actually created that value.
    lookup = returns_summary.set_index("Metric")["Value"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(
        ["Equity invested", "Equity realized"],
        [lookup["Sponsor equity invested"], lookup["Exit Equity Value"]],
        color=[COLORS["navy"], COLORS["green"]],
    )
    axes[0].set_title("Sponsor equity invested vs realized")
    axes[0].yaxis.set_major_formatter(FuncFormatter(eur_millions))
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].bar(
        value_creation_bridge["Driver"],
        value_creation_bridge["Value"],
        color=[COLORS["navy"], COLORS["teal"], COLORS["sand"], COLORS["green"], COLORS["red"], COLORS["navy"]],
    )
    axes[1].set_title("Value creation bridge")
    axes[1].yaxis.set_major_formatter(FuncFormatter(eur_millions))
    axes[1].spines[["top", "right"]].set_visible(False)
    plt.setp(axes[1].get_xticklabels(), rotation=18, ha="right")
    return figure


def build_sidebar_inputs() -> dict[str, dict[str, float | int]]:
    # The sidebar controls only the key teaching levers.
    # This keeps the app aligned with the notebook instead of turning it into a complex platform.
    base_case = get_base_case_inputs()
    st.sidebar.header("Scenario controls")

    entry_multiple = st.sidebar.slider("Entry multiple (x)", 5.5, 8.5, 7.0, 0.5)
    exit_multiple = st.sidebar.slider("Exit multiple (x)", 5.5, 8.5, 7.0, 0.5)
    exit_year = st.sidebar.slider("Exit year", 1, 3, 3, 1)
    senior_debt_pct = st.sidebar.slider("Senior debt (% of sources)", 0.20, 0.50, 0.35, 0.01)
    subordinated_debt_pct = st.sidebar.slider("Subordinated debt (% of sources)", 0.20, 0.50, 0.35, 0.01)
    senior_interest_rate = st.sidebar.slider("Senior interest rate", 0.04, 0.09, 0.06, 0.0025)
    subordinated_interest_rate = st.sidebar.slider("Subordinated interest rate", 0.05, 0.10, 0.065, 0.0025)
    ebit_growth_shift = st.sidebar.slider("EBIT growth shift vs base case", -0.02, 0.02, 0.0, 0.005)

    if senior_debt_pct + subordinated_debt_pct >= 0.95:
        st.sidebar.error("Debt percentages are too high for a residual equity slice.")

    # Start from the base case and then overwrite only the assumptions the user changed.
    overrides = deepcopy_case(base_case)
    overrides["valuation"]["entry_multiple"] = entry_multiple
    overrides["financing"]["senior_debt_pct"] = senior_debt_pct
    overrides["financing"]["subordinated_debt_pct"] = subordinated_debt_pct
    overrides["financing"]["senior_interest_rate"] = senior_interest_rate
    overrides["financing"]["subordinated_interest_rate"] = subordinated_interest_rate
    overrides["exit"]["exit_multiple"] = exit_multiple
    overrides["exit"]["exit_year"] = exit_year
    overrides["projection"]["ebit_growth_shift"] = ebit_growth_shift
    overrides["projection"]["d_and_a_growth_shift"] = ebit_growth_shift * 0.5

    return overrides


def deepcopy_case(case: dict[str, dict[str, float | int]]) -> dict[str, dict[str, float | int]]:
    # Streamlit reruns the script often, so copying avoids accidental mutation of the base case.
    return {
        section: values.copy()
        for section, values in case.items()
    }


def main() -> None:
    style_page()

    st.title("Leveraged Buyout Modeling in Python")
    st.markdown(
        """
        <div class="intro-card">
            <strong>Purpose.</strong> This app is a lightweight decision-support prototype built on the same analytical engine as the classroom notebook.
            It follows the same analyst sequence: define assumptions, value the deal, fund the acquisition, project cash generation, observe deleveraging,
            and test sponsor returns under different scenarios.
        </div>
        """,
        unsafe_allow_html=True,
    )

    overrides = build_sidebar_inputs()

    try:
        # The app uses the exact same model engine as the notebook,
        # which keeps classroom logic and demo logic fully consistent.
        results = run_lbo_model(overrides)
    except ValueError as exc:
        st.error(str(exc))
        return

    # The top KPI row summarizes the sponsor outcome before the user scrolls into detail.
    returns_summary = results["returns_summary"].set_index("Metric")["Value"]
    exit_label = f"Year {results['inputs']['exit']['exit_year']}"
    sponsor_equity_pct = 1 - results["inputs"]["financing"]["senior_debt_pct"] - results["inputs"]["financing"]["subordinated_debt_pct"]

    metric_columns = st.columns(5)
    metric_columns[0].metric("IRR", percent(float(returns_summary["IRR"])))
    metric_columns[1].metric("MOIC", f"{float(returns_summary['MOIC']):.2f}x")
    metric_columns[2].metric("Exit equity value", eur_millions(float(returns_summary["Exit Equity Value"])))
    metric_columns[3].metric(
        "Exit leverage",
        f"{results['credit_metrics'].loc[exit_label, 'Total Debt / EBITDA']:.2f}x",
    )
    metric_columns[4].metric("Sponsor equity", percent(sponsor_equity_pct))

    st.markdown("## 1. Inputs and deal framing")
    st.markdown(
        '<div class="section-note">Start with the transaction perimeter: what we pay, how we finance it, and what operating base we are underwriting.</div>',
        unsafe_allow_html=True,
    )
    left_column, right_column = st.columns([1.1, 0.9])
    with left_column:
        # Showing the table and the bridge together helps students connect assumptions to valuation.
        st.dataframe(results["inputs_table"], hide_index=True, use_container_width=True)
    with right_column:
        st.pyplot(plot_entry_bridge(results["entry_valuation"]), use_container_width=True)

    st.markdown("## 2. Sources & Uses")
    st.markdown(
        '<div class="section-note">This is the funding blueprint of the deal: every euro spent on the acquisition must be covered by debt or equity.</div>',
        unsafe_allow_html=True,
    )
    left_column, right_column = st.columns([0.9, 1.1])
    with left_column:
        st.dataframe(results["sources_and_uses"], use_container_width=True)
    with right_column:
        st.pyplot(plot_sources_and_uses(results["sources_and_uses"]), use_container_width=True)

    st.markdown("## 3. Operating projection")
    st.markdown(
        '<div class="section-note">The key question here is not accounting profit in isolation, but how much cash the business can release to repay debt.</div>',
        unsafe_allow_html=True,
    )
    st.pyplot(
        plot_operating_projection(results["operating_projection"], results["debt_schedule"]),
        use_container_width=True,
    )
    # The full table stays available below the chart so the visual story never hides the numbers.
    st.dataframe(results["operating_projection"], use_container_width=True)

    st.markdown("## 4. Deleveraging and credit monitoring")
    st.markdown(
        '<div class="section-note">As cash is swept into debt repayment, leverage should fall and interest coverage should improve. That is the credit story of the LBO.</div>',
        unsafe_allow_html=True,
    )
    left_column, right_column = st.columns(2)
    with left_column:
        st.pyplot(plot_deleveraging(results["debt_schedule"]), use_container_width=True)
    with right_column:
        st.pyplot(plot_credit_metrics(results["credit_metrics"]), use_container_width=True)
    st.dataframe(results["debt_schedule"], use_container_width=True)
    st.dataframe(results["credit_metrics"], use_container_width=True)

    st.markdown("## 5. Exit valuation and sponsor returns")
    st.markdown(
        '<div class="section-note">The sponsor return is the combined result of operating performance, debt paydown, and the exit valuation applied to the business.</div>',
        unsafe_allow_html=True,
    )
    left_column, right_column = st.columns([0.9, 1.1])
    with left_column:
        st.dataframe(results["returns_summary"], hide_index=True, use_container_width=True)
    with right_column:
        st.pyplot(
            plot_return_bridge(results["returns_summary"], results["value_creation_bridge"]),
            use_container_width=True,
        )
    # The bridge table remains visible because students often want to reconcile the chart exactly.
    st.dataframe(results["value_creation_bridge"], hide_index=True, use_container_width=True)

    st.markdown("## 6. Sensitivity analysis")
    st.markdown(
        '<div class="section-note">No analyst should rely on a single base case. Sensitivities show how quickly returns move when valuation or operating assumptions change.</div>',
        unsafe_allow_html=True,
    )
    sensitivity_columns = st.columns(2)
    with sensitivity_columns[0]:
        st.pyplot(
            plot_sensitivity_heatmap(
                results["sensitivities"]["entry_exit_irr"] * 100,
                "IRR sensitivity: entry multiple vs exit multiple",
                ".1f",
            ),
            use_container_width=True,
        )
    with sensitivity_columns[1]:
        st.pyplot(
            plot_sensitivity_heatmap(
                results["sensitivities"]["exit_growth_irr"] * 100,
                "IRR sensitivity: exit multiple vs EBIT growth shift",
                ".1f",
            ),
            use_container_width=True,
        )

    st.markdown("## 7. AI-generated investment commentary")
    st.write(
        "This optional layer sits on top of the model and turns the computed outputs into a short executive note. "
        "It does not replace the quantitative analysis and fails gracefully when Gemini is not configured."
    )

    if st.button("Generate investment commentary"):
        with st.spinner("Preparing commentary..."):
            # Gemini is deliberately optional. The quantitative model still stands on its own.
            response = generate_investment_commentary(results)
        if response["success"]:
            st.markdown(response["message"])
        else:
            st.info(response["message"])

    st.markdown(
        """
        **Run locally**

        ```bash
        streamlit run app/streamlit_lbo_demo.py
        ```
        """
    )


if __name__ == "__main__":
    main()
