# app.py
"""
Streamlit UI for MMM budget optimization.

This app:
- Loads config and data
- Fits MMM models once (cached)
- Lets the user choose a total budget
- Supports two optimisation modes:
    - Revenue
    - Profit (using brand-level profit margins)
- Optimizes cross-brand allocation for that budget
- Displays brand-level spend, revenue, and profit comparisons
- Regenerates PNG charts and allocation_revenue_summary.csv
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import yaml
import os

from data_prep import load_and_prep_data, group_sparse_channels
from model import OptimizedMMM
from optimizer import optimize_budget
from reporting import (
    plot_budget_pies,
    plot_spend_comparison,
    plot_saturation_curves,
    build_and_plot_revenue_decomposition,
    save_allocation_summary,
)


BRAND_DISPLAY_NAMES = {
    "bb": "Bobbi Brown",
    "mac": "MAC",
    "tf": "Too Faced",
}


def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def format_currency(amount: float) -> str:
    """Format a numeric amount as a short currency string in £k / £M."""
    if amount is None or np.isnan(amount):
        return "£0"
    if amount >= 1e6:
        return f"£{amount / 1e6:,.1f}M"
    if amount >= 1e3:
        return f"£{amount / 1e3:,.0f}k"
    return f"£{amount:,.0f}"


def format_delta_currency(delta: float) -> str:
    """Format a delta as +/- £X with k/M suffix."""
    if delta == 0 or np.isnan(delta):
        return "£0"
    sign = "-" if delta < 0 else "+"
    return f"{sign}{format_currency(abs(delta))}"


@st.cache_resource(show_spinner=True)
def build_models_and_data(
    config_path: str = "config.yaml",
) -> Tuple[dict, pd.DataFrame, Dict[str, OptimizedMMM]]:
    """
    Load config & data, and fit MMM models for all brands (cached).
    """
    cfg = load_config(config_path)

    input_file: str = cfg["input_file"]
    brands: List[str] = cfg["brands"]
    offline_prefixes: List[str] = cfg["offline_prefixes"]
    exclude_channels: List[str] = cfg["exclude_channels"]
    iterations: int = int(cfg["iterations"])
    param_ranges = {
        "decay": tuple(cfg["param_ranges"]["decay"]),
        "S": tuple(cfg["param_ranges"]["S"]),
    }
    ridge_alpha: float = float(cfg["ridge_alpha"])
    control_alpha_ratio: float = float(cfg.get("control_alpha_ratio", 0.1))
    search_penalty_factor: float = float(cfg.get("search_penalty_factor", 5.0))

    df = load_and_prep_data(input_file, fourier_order=2)

    models: Dict[str, OptimizedMMM] = {}
    for brand in brands:
        channels = group_sparse_channels(
            df,
            brand=brand,
            offline_prefixes=offline_prefixes,
            exclude_channels=exclude_channels,
        )
        m = OptimizedMMM(
            brand=brand,
            param_ranges=param_ranges,
            iterations=iterations,
            ridge_alpha=ridge_alpha,
            control_alpha_ratio=control_alpha_ratio,
            search_penalty_factor=search_penalty_factor,
        )
        m.tune_and_fit(df, channels, kpi_col=f"m0_kpi_{brand}")
        models[brand] = m

    return cfg, df, models


def compute_brand_level_metrics(
    df: pd.DataFrame,
    brands: List[str],
    models: Dict[str, OptimizedMMM],
    res_df: pd.DataFrame,
    x_opt: np.ndarray,
    channel_map: Dict[int, Tuple[str, str]],
    profit_margins: Dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Compute brand-level historical and optimized spend, revenue, and profit.
    """
    spend_brand = (
        res_df.groupby("Brand")[["Historical", "Optimized"]]
        .sum()
        .rename(columns={"Historical": "Hist_Spend", "Optimized": "Opt_Spend"})
    )

    opt_weekly_map: Dict[str, Dict[str, float]] = {}
    for i, val in enumerate(x_opt):
        brand, ch = channel_map[i]
        opt_weekly_map.setdefault(brand, {})
        opt_weekly_map[brand][ch] = float(val)

    records: List[dict] = []

    for brand in brands:
        model = models[brand]

        hist_decomp = model.decompose_historical(df)
        hist_revenue = float(sum(hist_decomp.values()))

        opt_decomp = model.decompose_optimized(opt_weekly_map.get(brand, {}))
        opt_revenue = float(sum(opt_decomp.values()))

        margin = float(profit_margins.get(brand, 1.0)) if profit_margins else 1.0
        hist_profit = hist_revenue * margin
        opt_profit = opt_revenue * margin

        records.append(
            {
                "Brand": brand,
                "Brand_Display": BRAND_DISPLAY_NAMES.get(brand, brand.upper()),
                "Hist_Spend": spend_brand.loc[brand, "Hist_Spend"]
                if brand in spend_brand.index
                else 0.0,
                "Opt_Spend": spend_brand.loc[brand, "Opt_Spend"]
                if brand in spend_brand.index
                else 0.0,
                "Hist_Revenue": hist_revenue,
                "Opt_Revenue": opt_revenue,
                "Hist_Profit": hist_profit,
                "Opt_Profit": opt_profit,
            }
        )

    return pd.DataFrame(records)


def make_spend_chart(brand_summary: pd.DataFrame) -> alt.Chart:
    df_melt = brand_summary.melt(
        id_vars=["Brand", "Brand_Display"],
        value_vars=["Hist_Spend", "Opt_Spend"],
        var_name="Scenario",
        value_name="Spend",
    )
    df_melt["Scenario"] = df_melt["Scenario"].replace(
        {"Hist_Spend": "Historical", "Opt_Spend": "Optimized"}
    )

    chart = (
        alt.Chart(df_melt, background="white")
        .mark_bar()
        .encode(
            x=alt.X("Brand_Display:N", title="Brand"),
            xOffset="Scenario:N",
            y=alt.Y("Spend:Q", title="Annual Spend"),
            color=alt.Color(
                "Scenario:N",
                title="Scenario",
                scale=alt.Scale(scheme="tableau10"),
            ),
            tooltip=[
                alt.Tooltip("Brand_Display:N", title="Brand"),
                alt.Tooltip("Scenario:N"),
                alt.Tooltip("Spend:Q", format=",.0f", title="Spend (£)"),
            ],
        )
        .properties(height=320)
        .configure_view(stroke=None)
        .configure_axis(
            labelColor="#4b5563",
            titleColor="#111827",
            gridColor="#e5e7eb",
            domainColor="#9ca3af",
        )
        .configure_legend(labelColor="#111827", titleColor="#111827")
    )
    return chart


def make_revenue_chart(brand_summary: pd.DataFrame) -> alt.Chart:
    df_melt = brand_summary.melt(
        id_vars=["Brand", "Brand_Display"],
        value_vars=["Hist_Revenue", "Opt_Revenue"],
        var_name="Scenario",
        value_name="Revenue",
    )
    df_melt["Scenario"] = df_melt["Scenario"].replace(
        {"Hist_Revenue": "Historical", "Opt_Revenue": "Optimized"}
    )

    chart = (
        alt.Chart(df_melt, background="white")
        .mark_bar()
        .encode(
            x=alt.X("Brand_Display:N", title="Brand"),
            xOffset="Scenario:N",
            y=alt.Y("Revenue:Q", title="Annual Revenue"),
            color=alt.Color(
                "Scenario:N",
                title="Scenario",
                scale=alt.Scale(scheme="tableau10"),
            ),
            tooltip=[
                alt.Tooltip("Brand_Display:N", title="Brand"),
                alt.Tooltip("Scenario:N"),
                alt.Tooltip("Revenue:Q", format=",.0f", title="Revenue (£)"),
            ],
        )
        .properties(height=320)
        .configure_view(stroke=None)
        .configure_axis(
            labelColor="#4b5563",
            titleColor="#111827",
            gridColor="#e5e7eb",
            domainColor="#9ca3af",
        )
        .configure_legend(labelColor="#111827", titleColor="#111827")
    )
    return chart


def main() -> None:
    st.set_page_config(
        page_title="MMM Budget Planner",
        page_icon="📊",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        :root { color-scheme: light; }
        .stApp { background-color: #ffffff !important; color: #111827 !important; }
        html, body, p, span, div, h1, h2, h3, h4, h5, h6 { color: #111827; }
        .main-title { font-size: 2.0rem; font-weight: 600; margin-bottom: 0.25rem; }
        .subtitle { color: #6b7280; font-size: 0.9rem; }
        [data-testid="stMetricLabel"] { color: #374151 !important; font-weight: 600 !important; opacity: 1 !important; }
        [data-testid="stMetricValue"] { color: #111827 !important; font-weight: 700 !important; font-size: 1.7rem !important; opacity: 1 !important; }
        [data-testid="stMetricDelta"] { font-weight: 600 !important; opacity: 1 !important; }
        section[data-testid="stSidebar"] { background-color: #f3f4f6 !important; }
        section[data-testid="stSidebar"] * { color: #111827 !important; font-weight: 500 !important; }
        input[type="number"], input[type="text"], textarea {
            background-color: #ffffff !important;
            color: #111827 !important;
            border: 1px solid #d1d5db !important;
            border-radius: 6px !important;
        }
        .stButton>button,
        button[kind="primary"],
        button[kind="secondary"],
        [data-testid="baseButton-primary"],
        [data-testid="baseButton-secondary"] {
            background-color: #2563eb !important;
            color: #f9fafb !important;
            border-radius: 999px !important;
            border: none !important;
            padding: 0.35rem 1.2rem !important;
        }
        .stButton>button:hover,
        button[kind="primary"]:hover,
        button[kind="secondary"]:hover,
        [data-testid="baseButton-primary"]:hover,
        [data-testid="baseButton-secondary"]:hover {
            background-color: #1d4ed8 !important;
            color: #f9fafb !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="main-title">MMM Budget Planner</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle">Explore optimized cross-brand budget allocations, revenue, and profit.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    with st.spinner("Loading data and fitting models... (cached after first run)"):
        cfg, df, models = build_models_and_data("config.yaml")

    brands: List[str] = cfg["brands"]
    default_budget: float = float(cfg["total_budget"])
    output_dir: str = cfg.get("output_dir", ".")
    default_mode: str = cfg.get("optimization_mode", "revenue").lower()
    cfg_profit_margins: Dict[str, float] = {
        b: float(v) for b, v in cfg.get("profit_margin", {}).items()
    }

    # --------- SIDEBAR ----------
    st.sidebar.header("Controls")

    total_budget = st.sidebar.number_input(
        "Total annual budget (£)",
        min_value=0.0,
        value=float(default_budget),
        step=100000.0,
        format="%.0f",
    )
    st.sidebar.markdown(f"**Default from config:** {format_currency(default_budget)}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Optimisation Mode")
    mode_label = st.sidebar.radio(
        "Optimise for",
        options=["Revenue", "Profit"],
        index=0 if default_mode == "revenue" else 1,
    )
    opt_mode = "profit" if mode_label == "Profit" else "revenue"

    profit_margins: Dict[str, float] = cfg_profit_margins.copy()
    if opt_mode == "profit":
        st.sidebar.markdown("**Brand Profit Margins (%):**")
        for b in brands:
            display = BRAND_DISPLAY_NAMES.get(b, b.upper())
            default_margin = profit_margins.get(b, 0.3) * 100
            val = st.sidebar.number_input(
                f"{display} margin (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(default_margin),
                step=1.0,
            )
            profit_margins[b] = val / 100.0
    else:
        profit_margins = {}


    st.sidebar.markdown("---")
    st.sidebar.subheader("Brand Budget Guardrails")

    brand_min_mult: Dict[str, float] = {}
    brand_max_mult: Dict[str, float] = {}

    for b in brands:
        display = BRAND_DISPLAY_NAMES.get(b, b.upper())

        min_m, max_m = st.sidebar.slider(
            f"{display} budget range (% of historical)",
            min_value=0.2,
            max_value=2.5,
            value=(0.8, 1.2),
            step=0.05,
        )

        brand_min_mult[b] = min_m
        brand_max_mult[b] = max_m


    st.sidebar.markdown("---")
    run_button = st.sidebar.button("Run Optimization")

    if not run_button and "last_result" in st.session_state:
        result = st.session_state["last_result"]
        opt_mode = st.session_state.get("last_mode", opt_mode)
        profit_margins = st.session_state.get("last_margins", profit_margins)
    elif total_budget <= 0:
        st.info("Set a positive total budget and click **Run Optimization**.")
        return
    else:
        with st.spinner("Running budget optimization and regenerating reports..."):
            try:
                res_df, x_opt, channel_map = optimize_budget(
                    models=models,
                    df=df,
                    brands=brands,
                    total_budget=total_budget,
                    mode=opt_mode,
                    profit_margins=profit_margins if opt_mode == "profit" else None,
                    brand_min_mult=brand_min_mult,
                    brand_max_mult=brand_max_mult,
                )
            except ValueError as e:
                st.error(str(e))
                return

            brand_summary = compute_brand_level_metrics(
                df=df,
                brands=brands,
                models=models,
                res_df=res_df,
                x_opt=x_opt,
                channel_map=channel_map,
                profit_margins=profit_margins if opt_mode == "profit" else None,
            )

            plot_budget_pies(res_df, brands, output_dir)
            plot_spend_comparison(res_df, output_dir)
            plot_saturation_curves(df, models, brands, x_opt, channel_map, output_dir)
            full_decomp_df = build_and_plot_revenue_decomposition(
                df, models, brands, x_opt, channel_map, output_dir
            )
            save_allocation_summary(
                res_df,
                full_decomp_df,
                output_dir,
                profit_margins=profit_margins if opt_mode == "profit" else None,
            )

            result = {
                "res_df": res_df,
                "x_opt": x_opt,
                "channel_map": channel_map,
                "brand_summary": brand_summary,
            }
            st.session_state["last_result"] = result
            st.session_state["last_mode"] = opt_mode
            st.session_state["last_margins"] = profit_margins

    res_df = result["res_df"]
    brand_summary = result["brand_summary"]

    # --------- TOP METRICS ----------
    st.subheader("Brand-level Budget, Revenue & Profit Summary")

    total_hist_spend = float(brand_summary["Hist_Spend"].sum())
    total_opt_spend = float(brand_summary["Opt_Spend"].sum())
    total_hist_rev = float(brand_summary["Hist_Revenue"].sum())
    total_opt_rev = float(brand_summary["Opt_Revenue"].sum())
    total_hist_profit = float(brand_summary["Hist_Profit"].sum())
    total_opt_profit = float(brand_summary["Opt_Profit"].sum())

    delta_budget = total_opt_spend - total_hist_spend
    delta_revenue = total_opt_rev - total_hist_rev
    delta_profit = total_opt_profit - total_hist_profit

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Budget (Optimized)",
            format_currency(total_opt_spend),
            delta=format_delta_currency(delta_budget),
            delta_color="normal",
        )

    if opt_mode == "profit":
        with col2:
            st.metric(
                "Total Profit (Optimized)",
                format_currency(total_opt_profit),
                delta=format_delta_currency(delta_profit),
                delta_color="normal",
            )
        with col3:
            st.metric(
                "Total Revenue (Optimized)",
                format_currency(total_opt_rev),
                delta=format_delta_currency(delta_revenue),
                delta_color="normal",
            )
    else:
        with col2:
            st.metric(
                "Total Revenue (Optimized)",
                format_currency(total_opt_rev),
                delta=format_delta_currency(delta_revenue),
                delta_color="normal",
            )
        with col3:
            st.metric(
                "Total Profit (Optimized)",
                format_currency(total_opt_profit),
                delta=format_delta_currency(delta_profit),
                delta_color="normal",
            )

    st.markdown("")

    # --------- BRAND BREAKDOWN ----------
    st.markdown("#### Brand Breakdown")
    brand_cols = st.columns(len(brands))

    for i, brand in enumerate(brands):
        row = brand_summary[brand_summary["Brand"] == brand].iloc[0]
        display_name = row["Brand_Display"]

        with brand_cols[i]:
            st.markdown(f"**{display_name}**")
            delta_spend = float(row["Opt_Spend"] - row["Hist_Spend"])
            delta_rev_b = float(row["Opt_Revenue"] - row["Hist_Revenue"])
            delta_profit_b = float(row["Opt_Profit"] - row["Hist_Profit"])

            st.metric(
                "Budget (Optimized)",
                format_currency(row["Opt_Spend"]),
                delta=format_delta_currency(delta_spend),
                delta_color="normal",
            )
            st.metric(
                "Revenue (Optimized)",
                format_currency(row["Opt_Revenue"]),
                delta=format_delta_currency(delta_rev_b),
                delta_color="normal",
            )
            if opt_mode == "profit":
                st.metric(
                    "Profit (Optimized)",
                    format_currency(row["Opt_Profit"]),
                    delta=format_delta_currency(delta_profit_b),
                    delta_color="normal",
                )

    st.markdown("---")

    # --------- CHARTS ----------
    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("#### Spend: Historical vs Optimized (by Brand)")
        spend_chart = make_spend_chart(brand_summary)
        st.altair_chart(spend_chart, use_container_width=True)

    with right_col:
        st.markdown("#### Revenue: Historical vs Optimized (by Brand)")
        revenue_chart = make_revenue_chart(brand_summary)
        st.altair_chart(revenue_chart, use_container_width=True)


    # # --------- TABLE GUARDRAILS ----------
    def build_guardrail_table(
        brand_summary: pd.DataFrame,
        brand_min_mult: Dict[str, float],
        brand_max_mult: Dict[str, float],
    ) -> pd.DataFrame:

        rows = []

        for _, row in brand_summary.iterrows():
            brand = row["Brand"]
            hist_spend = row["Hist_Spend"]

            min_mult = brand_min_mult.get(brand, np.nan)
            max_mult = brand_max_mult.get(brand, np.nan)

            rows.append(
                {
                    "Brand": row["Brand_Display"],
                    "Historical Spend (£)": hist_spend,
                    "Min Multiplier": min_mult,
                    "Max Multiplier": max_mult,
                    "Min Allowed Budget (£)": hist_spend * min_mult,
                    "Max Allowed Budget (£)": hist_spend * max_mult,
                }
            )

        return pd.DataFrame(rows)
    
    # st.markdown("---")
    # st.subheader("Brand Budget Guardrails Summary")

    # guardrail_df = build_guardrail_table(
    #     brand_summary,
    #     brand_min_mult,
    #     brand_max_mult,
    # )

    # display_df = guardrail_df.copy()
    # for col in [
    #     "Historical Spend (£)",
    #     "Min Allowed Budget (£)",
    #     "Max Allowed Budget (£)",
    # ]:
    #     display_df[col] = display_df[col].apply(format_currency)
    
    # styled_df = (
    #     display_df.style
    #     .set_properties(**{"text-align": "left"})
    #     .set_table_styles(
    #         [
    #             {"selector": "th", "props": [("text-align", "left")]},
    #             {"selector": "td", "props": [("text-align", "left")]},
    #         ]
    #     )
    # )


    # st.dataframe(display_df, use_container_width=True)

    # --------- TABLE GUARDRAILS ----------
    with st.expander("Show brand budget guardrails"):
        # st.subheader("Brand Budget Guardrails Summary")

        guardrail_df = build_guardrail_table(
            brand_summary,
            brand_min_mult,
            brand_max_mult,
        )

        display_df = guardrail_df.copy()

        # Format currency columns
        for col in ["Historical Spend (£)", "Min Allowed Budget (£)", "Max Allowed Budget (£)"]:
            display_df[col] = display_df[col].apply(format_currency)

        # Round multipliers nicely
        display_df["Min Multiplier"] = display_df["Min Multiplier"].round(2)
        display_df["Max Multiplier"] = display_df["Max Multiplier"].round(2)

        # Set column order
        display_df = display_df[
            [
                "Brand",
                "Historical Spend (£)",
                "Min Multiplier",
                "Max Multiplier",
                "Min Allowed Budget (£)",
                "Max Allowed Budget (£)",
            ]
        ]

        # Display table
        st.dataframe(display_df, use_container_width=True)

    # --------- DATA TABLE ----------
    with st.expander("Show brand-level table"):
        table_df = brand_summary.copy()
        table_df["Historical Spend"] = table_df["Hist_Spend"].apply(format_currency)
        table_df["Historical Revenue"] = table_df["Hist_Revenue"].apply(
            format_currency
        )
        table_df["Historical Profit"] = table_df["Hist_Profit"].apply(format_currency)
        table_df["Optimized Spend"] = table_df["Opt_Spend"].apply(format_currency)
        table_df["Optimized Revenue"] = table_df["Opt_Revenue"].apply(
            format_currency
        )
        table_df["Optimized Profit"] = table_df["Opt_Profit"].apply(format_currency)

        table_df = table_df[
            [
                "Brand_Display",
                "Historical Spend",
                "Historical Revenue",
                "Historical Profit",
                "Optimized Spend",
                "Optimized Revenue",
                "Optimized Profit",
            ]
        ].rename(columns={"Brand_Display": "Brand"})

        st.dataframe(table_df, use_container_width=True)

    # ----------- Saturation curves ---------
    st.markdown("---")
    st.subheader("Saturation Curves (Response vs Spend)")

    st.caption(
        "These curves illustrate diminishing returns for each channel and brand. "
        "Flatter sections indicate limited incremental revenue at higher spend levels."
    )

    img_path = os.path.join(output_dir, "saturation_curves.png")

    if os.path.exists(img_path):
        st.image(img_path,width=1000)
    else:
        st.info("Saturation curves image not available.")


if __name__ == "__main__":
    main()
