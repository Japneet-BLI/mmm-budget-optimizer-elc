# reporting.py
"""
Reporting and visualization utilities for MMM.

Generates:
- optimized_budget_split.png
- hist_vs_opt_bar.png
- saturation_curves.png
- revenue_decomposition_comparison.png
- optimized_revenue_breakdown.png
- allocation_revenue_summary.csv
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

from model import hill_function, geometric_adstock, OptimizedMMM


def _format_currency_func(x, pos):
    if x >= 1e6:
        return f"£{x * 1e-6:.1f}M"
    elif x >= 1e3:
        return f"£{x * 1e-3:.0f}k"
    else:
        return f"£{x:.0f}"


currency_fmt = mtick.FuncFormatter(_format_currency_func)


def _ensure_dir(output_dir: str) -> None:
    """Create output directory if it does not exist."""
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)


# ----------------------------
# Budget split & spend comparison
# ----------------------------

def plot_budget_pies(
    res_df: pd.DataFrame,
    brands: List[str],
    output_dir: str,
) -> None:
    """Plot optimized budget split by channel for each brand (pie charts)."""
    if res_df.empty:
        return

    _ensure_dir(output_dir)

    n_brands = len(brands)
    fig, axes = plt.subplots(1, n_brands, figsize=(6 * n_brands, 6))
    if n_brands == 1:
        axes = [axes]

    for ax, brand in zip(axes, brands):
        b_df = res_df[res_df["Brand"] == brand]
        if not b_df.empty and b_df["Optimized"].sum() > 0:
            ax.pie(
                b_df["Optimized"],
                labels=b_df["Channel"],
                autopct="%1.1f%%",
            )
            ax.set_title(f"{brand.upper()} Optimized Split")
        else:
            ax.axis("off")

    plt.tight_layout()
    path = os.path.join(output_dir, "optimized_budget_split.png")
    plt.savefig(path)
    plt.close(fig)


def plot_spend_comparison(
    res_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """Plot historical vs optimized spend by brand (bar chart)."""
    if res_df.empty:
        return

    _ensure_dir(output_dir)

    comp = (
        res_df.groupby("Brand")[["Historical", "Optimized"]]
        .sum()
        .reset_index()
        .melt(
            id_vars="Brand",
            value_vars=["Historical", "Optimized"],
            var_name="Scenario",
            value_name="Spend",
        )
    )

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=comp, x="Brand", y="Spend", hue="Scenario")
    ax.set_title("Historical vs Optimized Spend (by Brand)")
    ax.yaxis.set_major_formatter(currency_fmt)
    plt.tight_layout()
    path = os.path.join(output_dir, "hist_vs_opt_bar.png")
    plt.savefig(path)
    plt.close()


# ----------------------------
# Saturation curves
# ----------------------------

def plot_saturation_curves(
    df: pd.DataFrame,
    models: Dict[str, OptimizedMMM],
    brands: List[str],
    x_opt: np.ndarray,
    channel_map: Dict[int, Tuple[str, str]],
    output_dir: str,
) -> None:
    """Plot saturation curves for each channel with historical & optimized points."""
    all_channels = [channel_map[i][1] for i in range(len(channel_map))]
    if not all_channels:
        return

    _ensure_dir(output_dir)

    n = len(all_channels)
    cols = 3
    rows = (n // cols) + (1 if n % cols else 0)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for i, ch in enumerate(all_channels):
        brand = channel_map[i][0]
        model = models[brand]
        p = model.best_params[ch]

        x_hist = df[ch].values
        max_x = x_hist.max() * 1.5 if x_hist.max() > 0 else 1.0
        x = np.linspace(0, max_x, 100)

        ss_x = x / (1.0 - p["decay"])
        y_curve = p["beta"] * hill_function(ss_x, p["K"], p["S"])

        ax = axes[i]
        ax.plot(x, y_curve, color="gray", alpha=0.7, label="Response curve")

        opt_weekly = x_opt[i]
        opt_y = p["beta"] * hill_function(
            np.array([opt_weekly / (1.0 - p["decay"])]), p["K"], p["S"]
        )[0]
        ax.scatter([opt_weekly], [opt_y], c="red", s=60, zorder=5, label="Optimized")

        hist_avg = df[ch].mean()
        hist_y = p["beta"] * hill_function(
            np.array([hist_avg / (1.0 - p["decay"])]), p["K"], p["S"]
        )[0]
        ax.scatter([hist_avg], [hist_y], c="blue", s=60, zorder=5, label="Historical")

        ax.set_title(f"{brand.upper()} - {ch}")
        ax.xaxis.set_major_formatter(currency_fmt)
        ax.yaxis.set_major_formatter(currency_fmt)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    plt.tight_layout()
    path = os.path.join(output_dir, "saturation_curves.png")
    plt.savefig(path)
    plt.close(fig)


# ----------------------------
# Revenue decomposition & CSV
# ----------------------------

def build_and_plot_revenue_decomposition(
    df: pd.DataFrame,
    models: Dict[str, OptimizedMMM],
    brands: List[str],
    x_opt: np.ndarray,
    channel_map: Dict[int, Tuple[str, str]],
    output_dir: str,
) -> pd.DataFrame:
    """
    Build historical and optimized revenue decomposition and plot:
    - revenue_decomposition_comparison.png
    - optimized_revenue_breakdown.png

    Returns
    -------
    full_decomp_df : pd.DataFrame
        Columns: ['Brand', 'Scenario', 'Driver', 'Revenue', 'Percentage']
    """
    hist_data: List[dict] = []
    opt_data: List[dict] = []

    _ensure_dir(output_dir)

    # Map weekly optimized spends per brand/channel
    opt_weekly_map: Dict[str, Dict[str, float]] = {}
    for i, val in enumerate(x_opt):
        brand, ch = channel_map[i]
        opt_weekly_map.setdefault(brand, {})
        opt_weekly_map[brand][ch] = float(val)

    for brand in brands:
        model = models[brand]

        # Historical
        hist_decomp = model.decompose_historical(df)
        for driver, rev in hist_decomp.items():
            hist_data.append(
                {
                    "Brand": brand,
                    "Scenario": "Historical",
                    "Driver": driver,
                    "Revenue": float(rev),
                }
            )

        # Optimized
        opt_decomp = model.decompose_optimized(opt_weekly_map.get(brand, {}))
        for driver, rev in opt_decomp.items():
            opt_data.append(
                {
                    "Brand": brand,
                    "Scenario": "Optimized",
                    "Driver": driver,
                    "Revenue": float(rev),
                }
            )

    full_decomp_df = pd.DataFrame(hist_data + opt_data)
    if full_decomp_df.empty:
        return full_decomp_df

    total_rev = full_decomp_df.groupby(["Brand", "Scenario"])["Revenue"].transform(
        "sum"
    )
    full_decomp_df["Percentage"] = np.where(
        total_rev > 0, full_decomp_df["Revenue"] / total_rev, 0.0
    )

    brands_unique = full_decomp_df["Brand"].unique()
    drivers_unique = full_decomp_df["Driver"].unique()
    sorted_drivers = ["Baseline"] + [d for d in drivers_unique if d != "Baseline"]

    fig, axes = plt.subplots(
        1, len(brands_unique), figsize=(6 * len(brands_unique), 6), sharey=True
    )
    if len(brands_unique) == 1:
        axes = [axes]

    color_map = dict(
        zip(sorted_drivers, plt.cm.tab20(np.linspace(0, 1, len(sorted_drivers))))
    )

    for ax, brand in zip(axes, brands_unique):
        brand_df = full_decomp_df[full_decomp_df["Brand"] == brand]
        pivot = (
            brand_df.pivot(index="Scenario", columns="Driver", values="Revenue")
            .fillna(0.0)
        )
        cols = [d for d in sorted_drivers if d in pivot.columns]

        pivot[cols].plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color=[color_map[c] for c in cols],
            width=0.6,
        )
        ax.set_title(f"{brand.upper()} Revenue")
        ax.set_xlabel("")
        ax.yaxis.set_major_formatter(currency_fmt)
        ax.tick_params(axis="x", rotation=0)

    axes[-1].legend(
        title="Driver", bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "revenue_decomposition_comparison.png")
    plt.savefig(path)
    plt.close(fig)

    pivot_opt = (
        full_decomp_df[full_decomp_df["Scenario"] == "Optimized"]
        .pivot(index="Brand", columns="Driver", values="Revenue")
        .fillna(0.0)
    )
    cols = ["Baseline"] + [c for c in pivot_opt.columns if c != "Baseline"]

    ax = pivot_opt[cols].plot(
        kind="bar",
        stacked=True,
        colormap="tab20",
        figsize=(8, 6),
        width=0.7,
    )
    ax.set_title("Future Revenue Breakdown (Optimized)")
    ax.yaxis.set_major_formatter(currency_fmt)
    ax.set_xlabel("Brand")
    plt.tight_layout()
    path = os.path.join(output_dir, "optimized_revenue_breakdown.png")
    plt.savefig(path)
    plt.close()

    return full_decomp_df


def save_allocation_summary(
    res_df: pd.DataFrame,
    full_decomp_df: pd.DataFrame,
    output_dir: str,
    profit_margins: Dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Build and save allocation_revenue_summary.csv by joining spend and revenue.

    The summary is at Brand + Channel level, with:

    - Historical / Optimized Spend
    - Historical / Optimized Revenue
    - Historical / Optimized Profit (if margins provided)
    - Historical / Optimized ROI (Revenue-based)
    - Historical / Optimized Profit ROI
    - Share of Brand Revenue (Historical / Optimized)

    Returns
    -------
    summary_df : pd.DataFrame
        The final summary table.
    """
    if res_df.empty or full_decomp_df.empty:
        return pd.DataFrame()

    _ensure_dir(output_dir)

    # Filter out Baseline; we want channel rows only
    hist_rev = (
        full_decomp_df[
            (full_decomp_df["Scenario"] == "Historical")
            & (full_decomp_df["Driver"] != "Baseline")
        ]
        .rename(columns={"Driver": "Channel", "Revenue": "Historical_Revenue"})
        [["Brand", "Channel", "Historical_Revenue"]]
    )

    opt_rev = (
        full_decomp_df[
            (full_decomp_df["Scenario"] == "Optimized")
            & (full_decomp_df["Driver"] != "Baseline")
        ]
        .rename(columns={"Driver": "Channel", "Revenue": "Optimized_Revenue"})
        [["Brand", "Channel", "Optimized_Revenue"]]
    )

    summary = (
        res_df.copy()
        .merge(hist_rev, on=["Brand", "Channel"], how="left")
        .merge(opt_rev, on=["Brand", "Channel"], how="left")
    )

    summary[["Historical_Revenue", "Optimized_Revenue"]] = summary[
        ["Historical_Revenue", "Optimized_Revenue"]
    ].fillna(0.0)

    # Revenue ROI
    hist_spend = summary["Historical_Spend"] = summary["Historical"]
    opt_spend = summary["Optimized_Spend"] = summary["Optimized"]

    hist_spend_nonzero = hist_spend.replace(0, np.nan)
    opt_spend_nonzero = opt_spend.replace(0, np.nan)

    summary["Hist_ROI_Revenue"] = np.where(
        hist_spend_nonzero.notna(),
        summary["Historical_Revenue"] / hist_spend_nonzero,
        np.nan,
    )
    summary["Opt_ROI_Revenue"] = np.where(
        opt_spend_nonzero.notna(),
        summary["Optimized_Revenue"] / opt_spend_nonzero,
        np.nan,
    )

    # Profit metrics (if margins provided)
    if profit_margins is not None:
        brand_margin = summary["Brand"].map(
            lambda b: float(profit_margins.get(b, 1.0))
        )

        summary["Historical_Profit"] = summary["Historical_Revenue"] * brand_margin
        summary["Optimized_Profit"] = summary["Optimized_Revenue"] * brand_margin

        hist_spend_nonzero = hist_spend_nonzero.replace(0, np.nan)
        opt_spend_nonzero = opt_spend_nonzero.replace(0, np.nan)

        summary["Hist_ROI_Profit"] = np.where(
            hist_spend_nonzero.notna(),
            summary["Historical_Profit"] / hist_spend_nonzero,
            np.nan,
        )
        summary["Opt_ROI_Profit"] = np.where(
            opt_spend_nonzero.notna(),
            summary["Optimized_Profit"] / opt_spend_nonzero,
            np.nan,
        )
    else:
        summary["Historical_Profit"] = np.nan
        summary["Optimized_Profit"] = np.nan
        summary["Hist_ROI_Profit"] = np.nan
        summary["Opt_ROI_Profit"] = np.nan

    # Within-brand revenue shares (channels only, excluding Baseline)
    hist_brand_tot = (
        summary.groupby("Brand")["Historical_Revenue"].transform("sum").replace(0, np.nan)
    )
    opt_brand_tot = (
        summary.groupby("Brand")["Optimized_Revenue"].transform("sum").replace(0, np.nan)
    )

    summary["Hist_Revenue_Share"] = np.where(
        hist_brand_tot.notna(),
        summary["Historical_Revenue"] / hist_brand_tot,
        np.nan,
    )
    summary["Opt_Revenue_Share"] = np.where(
        opt_brand_tot.notna(),
        summary["Optimized_Revenue"] / opt_brand_tot,
        np.nan,
    )

    summary_out = summary[
        [
            "Brand",
            "Channel",
            "Historical_Spend",
            "Optimized_Spend",
            "Historical_Revenue",
            "Optimized_Revenue",
            "Historical_Profit",
            "Optimized_Profit",
            "Hist_ROI_Revenue",
            "Opt_ROI_Revenue",
            "Hist_ROI_Profit",
            "Opt_ROI_Profit",
            "Hist_Revenue_Share",
            "Opt_Revenue_Share",
        ]
    ]

    path = os.path.join(output_dir, "allocation_revenue_summary.csv")
    summary_out.to_csv(path, index=False)

    return summary_out
