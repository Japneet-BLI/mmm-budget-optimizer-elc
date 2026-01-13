# main.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error

from data_prep import load_and_prep_data, group_sparse_channels
from eda import create_advanced_eda
from model import OptimizedMMM
from optimizer import optimize_budget
from reporting import (
    plot_model_performance,
    plot_budget_pies,
    plot_saturation_curves,
    plot_spend_comparison,
    build_and_plot_revenue_decomposition,
    save_tables,
)


def setup_logging() -> None:
    """Configure root logger with a simple console handler."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def load_config(config_path: str) -> dict:
    """
    Load YAML configuration file.

    Parameters
    ----------
    config_path : str
        Path to YAML config.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main(config_path: str = "config.yaml") -> None:
    """
    Main entry point for MMM training, optimization and reporting.

    Parameters
    ----------
    config_path : str, optional
        Path to YAML configuration file. Default is 'config.yaml'.
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    cfg = load_config(config_path)

    input_file: str = cfg["input_file"]
    output_dir: str = cfg.get("output_dir", ".")
    total_budget: float = float(cfg["total_budget"])
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

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Data prep
    df = load_and_prep_data(input_file, fourier_order=2)

    # 2. EDA
    create_advanced_eda(
        df, brands=brands, exclude_channels=exclude_channels, output_dir=output_dir
    )

    # 3. Modeling
    models: Dict[str, OptimizedMMM] = {}
    metrics_list: List[dict] = []

    for brand in brands:
        logger.info("Fitting model for brand %s", brand)

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

        y_true = df[f"m0_kpi_{brand}"]
        y_pred = m.y_pred
        mape = mean_absolute_percentage_error(y_true, y_pred)
        metrics_list.append(
            {"Brand": brand, "R2": m.best_r2, "MAPE": float(mape)}
        )

    # 4. Optimization
    res_df, x_opt, channel_map = optimize_budget(
        models=models,
        df=df,
        brands=brands,
        total_budget=total_budget,
    )

    # 5. Reporting and plots
    plot_model_performance(df, models, brands, output_dir)
    plot_budget_pies(res_df, brands, output_dir)
    plot_spend_comparison(res_df, output_dir)
    plot_saturation_curves(df, models, brands, x_opt, channel_map, output_dir)

    full_decomp_df = build_and_plot_revenue_decomposition(
        df, models, brands, x_opt, channel_map, output_dir
    )

    save_tables(res_df, full_decomp_df, metrics_list, output_dir)

    logger.info("Process complete. All files generated in %s", output_dir)


if __name__ == "__main__":
    main()
