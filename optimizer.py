# optimizer.py
"""
Two-stage budget optimization for MMM with optional profit-aware mode.

Stage 1: Brand-level optimization
Stage 2: Intra-brand channel optimization

Public API:
    res_df, x_opt, channel_map = optimize_budget(...)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# --------------------------------------------------
# Constants
# --------------------------------------------------
WEEKS_PER_YEAR = 52
AVG_CONTROLS = np.array([26.5, 0.0, 0.0, 0.0, 0.0], dtype=float)


# --------------------------------------------------
# Historical shares
# --------------------------------------------------
def _compute_hist_shares(
    df: pd.DataFrame,
    models: Dict[str, object],
    brands: List[str],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    Compute historical annual spend shares per channel for each brand.
    """
    channel_shares: Dict[str, Dict[str, float]] = {}
    brand_hist_totals: Dict[str, float] = {}

    for brand in brands:
        model = models[brand]
        channels = list(model.channels)

        if not channels:
            channel_shares[brand] = {}
            brand_hist_totals[brand] = 0.0
            continue

        hist_spends = np.array([df[ch].sum() for ch in channels], dtype=float)
        total = float(hist_spends.sum())
        brand_hist_totals[brand] = total

        if total > 0:
            shares = hist_spends / total
        else:
            shares = np.ones(len(channels), dtype=float) / len(channels)

        channel_shares[brand] = dict(zip(channels, shares))

    return channel_shares, brand_hist_totals


# --------------------------------------------------
# Stage 1: Brand-level optimization
# --------------------------------------------------
def _optimize_brand_budgets(
    models: Dict[str, object],
    brands: List[str],
    total_budget: float,
    channel_shares: Dict[str, Dict[str, float]],
    brand_hist_totals: Dict[str, float],
    mode: str = "revenue",
    profit_margins: Dict[str, float] | None = None,
    brand_min_mult: Dict[str, float] | None = None,
    brand_max_mult: Dict[str, float] | None = None,
) -> Dict[str, float]:

    if mode == "profit" and not profit_margins:
        raise ValueError("profit_margins must be provided when mode='profit'")

    n_brands = len(brands)

    # ---- Initial guess ----
    hist_totals = np.array([brand_hist_totals.get(b, 0.0) for b in brands])
    if hist_totals.sum() > 0:
        x0 = total_budget * hist_totals / hist_totals.sum()
    else:
        x0 = np.ones(n_brands) * total_budget / n_brands

    # ---- Bounds ----
    bounds = []
    for brand in brands:
        hist = brand_hist_totals.get(brand, 0.0)

        lb_mult = brand_min_mult.get(brand, 0.0) if brand_min_mult else 0.0
        ub_mult = brand_max_mult.get(brand, np.inf) if brand_max_mult else np.inf

        lb = hist * lb_mult if hist > 0 else 0.0
        ub = hist * ub_mult if hist > 0 else total_budget

        bounds.append((lb, ub))

    # ---- Feasibility check ----
    min_required = sum(lb for lb, _ in bounds)
    if min_required > total_budget:
        raise ValueError(
            f"Infeasible brand guardrails: minimum required "
            f"{min_required:.2f} > total budget {total_budget:.2f}"
        )

    # ---- Objective ----
    def objective(B: np.ndarray) -> float:
        B = np.clip(B, 0.0, None)
        total_obj = 0.0

        for i, brand in enumerate(brands):
            weekly_budget = B[i] / WEEKS_PER_YEAR
            shares = channel_shares.get(brand, {})
            if not shares:
                continue

            spends = {ch: weekly_budget * s for ch, s in shares.items()}
            sales = float(
                models[brand].predict_sales(spends, future_controls=AVG_CONTROLS)
            )

            if mode == "profit":
                sales *= profit_margins.get(brand, 1.0)

            total_obj += sales

        return -total_obj

    constraints = [{
        "type": "eq",
        "fun": lambda B: total_budget - np.sum(B)
    }]

    res = minimize(
        objective,
        x0,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
    )

    if not res.success:
        B_opt = x0
    else:
        B_opt = np.clip(res.x, 0.0, None)

    return dict(zip(brands, B_opt))


# --------------------------------------------------
# Stage 2: Intra-brand optimization
# --------------------------------------------------
def _optimize_within_brand(
    model: object,
    brand_budget_annual: float,
    channel_shares: Dict[str, float],
    alpha_min: float,
    alpha_max: float,
) -> Dict[str, float]:

    channels = list(model.channels)
    if not channels or brand_budget_annual <= 0:
        return {ch: 0.0 for ch in channels}

    weekly_budget = brand_budget_annual / WEEKS_PER_YEAR
    base = np.array(
        [weekly_budget * channel_shares.get(ch, 1 / len(channels)) for ch in channels]
    )

    lb = alpha_min * base
    ub = alpha_max * base
    bounds = list(zip(lb, ub))

    def objective(x: np.ndarray) -> float:
        spends = dict(zip(channels, x))
        sales = model.predict_sales(spends, future_controls=AVG_CONTROLS)
        return -float(sales)

    constraints = [{
        "type": "eq",
        "fun": lambda x: weekly_budget - np.sum(x)
    }]

    res = minimize(
        objective,
        base,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
    )

    x_opt = base if not res.success else np.clip(res.x, 0.0, None)
    return dict(zip(channels, x_opt))


# --------------------------------------------------
# Public API
# --------------------------------------------------
def optimize_budget(
    models: Dict[str, object],
    df: pd.DataFrame,
    brands: List[str],
    total_budget: float,
    mode: str = "revenue",
    profit_margins: Dict[str, float] | None = None,
    alpha_min: float = 0.2,
    alpha_max: float = 2.5,
    brand_min_mult: Dict[str, float] | None = None,
    brand_max_mult: Dict[str, float] | None = None,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[int, Tuple[str, str]]]:

    channel_shares, brand_hist_totals = _compute_hist_shares(df, models, brands)

    brand_budgets = _optimize_brand_budgets(
        models=models,
        brands=brands,
        total_budget=total_budget,
        channel_shares=channel_shares,
        brand_hist_totals=brand_hist_totals,
        mode=mode,
        profit_margins=profit_margins,
        brand_min_mult=brand_min_mult,
        brand_max_mult=brand_max_mult,
    )

    rows = []
    x_opt = []
    channel_map = {}
    idx = 0

    for brand in brands:
        weekly_spends = _optimize_within_brand(
            model=models[brand],
            brand_budget_annual=brand_budgets[brand],
            channel_shares=channel_shares[brand],
            alpha_min=alpha_min,
            alpha_max=alpha_max,
        )

        for ch, w in weekly_spends.items():
            rows.append({
                "Brand": brand,
                "Channel": ch,
                "Historical": float(df[ch].sum()),
                "Optimized": w * WEEKS_PER_YEAR,
            })

            x_opt.append(w)
            channel_map[idx] = (brand, ch)
            idx += 1

    return pd.DataFrame(rows), np.array(x_opt), channel_map
