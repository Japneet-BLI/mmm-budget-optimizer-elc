# eda.py
from __future__ import annotations

import logging
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from scipy.stats import pearsonr
import pandas as pd

from reporting import currency_formatter  # reuse formatter


logger = logging.getLogger(__name__)


def create_advanced_eda(
    df: pd.DataFrame,
    brands: List[str],
    exclude_channels: List[str],
    output_dir: str = ".",
) -> None:
    """
    Generate basic EDA plots: scatter and time series by channel for each brand.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared weekly dataset.
    brands : list of str
        List of brands to process.
    exclude_channels : list of str
        Channels to exclude from time series plots.
    output_dir : str, optional
        Folder to save generated figures. Default is current directory.

    Returns
    -------
    None
    """
    logger.info("Generating advanced EDA plots")
    currency_fmt = FuncFormatter(currency_formatter)

    for brand in brands:
        logger.info("  EDA for brand %s", brand)
        kpi_col = f"m0_kpi_{brand}"
        spend_cols = [c for c in df.columns if brand in c and "_sp" in c]

        # Scatter: total spend vs sales
        total_spend = df[spend_cols].sum(axis=1)
        if total_spend.sum() == 0:
            logger.warning("  Brand %s has zero total spend, skipping scatter", brand)
            continue

        corr, _ = pearsonr(total_spend, df[kpi_col])

        plt.figure(figsize=(8, 6))
        ax = sns.regplot(
            x=total_spend,
            y=df[kpi_col],
            scatter_kws={"alpha": 0.6},
            line_kws={"color": "red"},
        )
        plt.title(f"{brand.upper()}: Total Spend vs Sales (r={corr:.2f})")
        plt.xlabel("Total Weekly Spend")
        plt.ylabel("Weekly Sales")
        ax.xaxis.set_major_formatter(currency_fmt)
        ax.yaxis.set_major_formatter(currency_fmt)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{brand}_scatter_spend_vs_kpi.png")
        plt.close()

        # Time series by channel
        raw_cols = [c for c in spend_cols if c not in exclude_channels]
        if not raw_cols:
            logger.warning("  Brand %s has no time series channels, skipping", brand)
            continue

        num_plots = len(raw_cols)
        rows = (num_plots // 2) + (1 if num_plots % 2 > 0 else 0)
        fig, axes = plt.subplots(rows, 2, figsize=(16, 4 * rows))
        axes = axes.flatten()

        for i, channel in enumerate(raw_cols):
            if df[channel].std() == 0:
                r = 0.0
            else:
                r = pearsonr(df[channel], df[kpi_col])[0]

            ax1 = axes[i]
            ax1.bar(df["wc_sun"], df[channel], alpha=0.5, label="Spend")
            ax2 = ax1.twinx()
            ax2.plot(df["wc_sun"], df[kpi_col], linewidth=2, label="Sales")

            ax1.set_title(f"{brand.upper()} - {channel}\nCorrelation (r): {r:.2f}")
            ax1.grid(True, alpha=0.3)
            ax1.yaxis.set_major_formatter(currency_fmt)
            ax2.yaxis.set_major_formatter(currency_fmt)

        # Turn off any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{brand}_time_series_corr.png")
        plt.close()
