# optimizer.py
"""
Two-stage budget optimization for MMM with optional profit-aware mode.

Stage 1: Brand-level optimization
    - Decision variables: annual brand budgets B_b
    - Constraint: sum_b B_b = TOTAL_BUDGET, B_b >= 0
    - Within a brand, budget is split across channels in proportion
      to historical channel spend (approximation).
    - Objective:
        - "revenue" mode: maximize total predicted sales across brands
        - "profit" mode: maximize sum_b margin_b * sales_b

Stage 2: Intra-brand channel optimization
    - For each brand, given B_b* from Stage 1:
        - Optimize weekly channel spends x_{b,c}
        - Constraint: sum_c x_{b,c} * 52 = B_b*
        - Bounds: [alpha_min * base_week, alpha_max * base_week]
          where base_week is the scaled baseline weekly spend for that brand
          under B_b* (using historical channel shares).
    - Objective: maximize predicted sales for that brand.

Returned API is compatible with previous code:
    res_df, x_opt, channel_map = optimize_budget(...)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _compute_hist_shares(
    df: pd.DataFrame,
    models: Dict[str, object],
    brands: List[str],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    Compute historical annual spend shares per channel for each brand.

    Parameters
    ----------
    df : pd.DataFrame
        Weekly dataset with spend columns.
    models : dict
        brand -> OptimizedMMM instance (must have .channels list).
    brands : list of str
        Brand codes.

    Returns
    -------
    channel_shares : dict
        channel_shares[brand][channel] = share in (0,1),
        sums to 1 for each brand (unless brand has no channels).
    brand_hist_totals : dict
        brand_hist_totals[brand] = total historical annual spend for that brand.
    """
    channel_shares: Dict[str, Dict[str, float]] = {}
    brand_hist_totals: Dict[str, float] = {}

    for brand in brands:
        model = models[brand]
        channels = model.channels

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
            # If no historical spend, default to equal shares across channels
            shares = np.ones_like(hist_spends) / len(hist_spends)

        channel_shares[brand] = {ch: float(s) for ch, s in zip(channels, shares)}

    return channel_shares, brand_hist_totals


def _optimize_brand_budgets(
    models: Dict[str, object],
    df: pd.DataFrame,
    brands: List[str],
    total_budget: float,
    channel_shares: Dict[str, Dict[str, float]],
    brand_hist_totals: Dict[str, float],
    mode: str = "revenue",
    profit_margins: Dict[str, float] | None = None,
    brand_min_mult: Dict[str, float] | None = None,
    brand_max_mult: Dict[str, float] | None = None,
) -> Dict[str, float]:
    """
    Stage 1: brand-level optimization of annual budgets B_b.

    Parameters
    ----------
    models : dict
        brand -> OptimizedMMM instance.
    df : pd.DataFrame
        Weekly dataset (not used directly, kept for flexibility).
    brands : list of str
        Brand codes.
    total_budget : float
        Total annual budget to allocate.
    channel_shares : dict
        channel_shares[brand][channel] = historical share within brand.
    brand_hist_totals : dict
        brand_hist_totals[brand] = historical total annual spend for brand.
    mode : {"revenue", "profit"}
        Objective variant:
            "revenue" -> maximize total revenue
            "profit"  -> maximize sum_b margin_b * revenue_b
    profit_margins : dict or None
        profit_margins[brand] = margin as decimal (e.g. 0.35 for 35%).
        Used only if mode == "profit".

    Returns
    -------
    brand_budgets : dict
        Optimized annual budgets per brand, summing approximately to total_budget.
    """
    n_brands = len(brands)
    avg_controls = np.array([26.5, 0.0, 0.0, 0.0, 0.0], dtype=float)

    # Initial guess: proportional to historical brand totals (or equal if no history)
    hist_totals = np.array(
        [brand_hist_totals.get(b, 0.0) for b in brands], dtype=float
    )
    hist_sum = float(hist_totals.sum())
    if hist_sum > 0:
        init_shares = hist_totals / hist_sum
    else:
        init_shares = np.ones(n_brands, dtype=float) / n_brands

    x0 = init_shares * total_budget
    bounds = []


    for b in brands:
        hist = brand_hist_totals.get(b, 0.0)

        if hist > 0:
            lb = hist * (brand_min_mult.get(b, 0.0) if brand_min_mult else 0.0)
            ub = hist * (brand_max_mult.get(b, np.inf) if brand_max_mult else np.inf)
        else:
            lb, ub = 0.0, np.inf

        bounds.append((lb, ub))

    # ---- Safety check ----
    min_required = sum(lb for lb, _ in bounds)

    if min_required > total_budget:
        raise ValueError(
            f"Infeasible constraints: min required budget "
            f"{min_required:.2f} exceeds total budget {total_budget:.2f}"
        )

    # bounds = [(0.0, None)] * n_brands

    def objective(B: np.ndarray) -> float:
        """
        Negative total objective across brands for given brand budgets B.
        Objective is revenue or profit depending on `mode`.
        """
        B = np.clip(B, 0.0, None)
        total_obj = 0.0

        for idx, brand in enumerate(brands):
            model = models[brand]
            brand_budget_annual = float(B[idx])
            weekly_budget = brand_budget_annual / 52.0

            shares = channel_shares[brand]
            if not shares:
                continue

            # Split weekly budget across channels using historical shares
            channel_spends: Dict[str, float] = {
                ch: weekly_budget * share for ch, share in shares.items()
            }

            weekly_sales = float(
                model.predict_sales(channel_spends, future_controls=avg_controls)
            )
            # weekly vs annual scale doesn't matter for argmax, it's a positive constant
            if mode == "profit" and profit_margins is not None:
                margin = float(profit_margins.get(brand, 1.0))
                total_obj += margin * weekly_sales
            else:
                total_obj += weekly_sales

        # We minimize, so return negative
        return -total_obj

    # Equality constraint: sum of brand budgets == total_budget
    def eq_constraint(B: np.ndarray) -> float:
        return total_budget - float(np.sum(B))

    constraints = [{"type": "eq", "fun": eq_constraint}]

    res = minimize(
        objective,
        x0,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
    )

    B_opt = np.clip(res.x, 0.0, None)
    brand_budgets = {brand: float(B_opt[i]) for i, brand in enumerate(brands)}
    return brand_budgets


def _optimize_within_brand(
    brand: str,
    model: object,
    brand_budget_annual: float,
    channel_shares: Dict[str, float],
    alpha_min: float = 0.2,
    alpha_max: float = 2.5,
) -> Dict[str, float]:
    """
    Stage 2 for a single brand: optimize weekly spends per channel.

    Within a brand, margin is a constant multiplier on revenue, so maximizing
    revenue is equivalent to maximizing profit under a fixed brand budget.

    Parameters
    ----------
    brand : str
        Brand code.
    model : OptimizedMMM
        MMM model for this brand (must expose .channels and .predict_sales()).
    brand_budget_annual : float
        Optimized annual budget for this brand from Stage 1.
    channel_shares : dict
        channel_shares[channel] = historical share for this brand (sums to 1).
    alpha_min : float, optional
        Lower bound multiplier for scaled baseline weekly spend.
    alpha_max : float, optional
        Upper bound multiplier for scaled baseline weekly spend.

    Returns
    -------
    weekly_spends : dict
        weekly_spends[channel] = optimized weekly spend.
    """
    avg_controls = np.array([26.5, 0.0, 0.0, 0.0, 0.0], dtype=float)
    channels = list(model.channels)

    if not channels or brand_budget_annual <= 0:
        return {ch: 0.0 for ch in channels}

    n_ch = len(channels)
    weekly_budget = brand_budget_annual / 52.0

    # Scaled baseline weekly spends using shares
    base_weekly = np.zeros(n_ch, dtype=float)
    for i, ch in enumerate(channels):
        share = channel_shares.get(ch, 1.0 / n_ch)
        base_weekly[i] = weekly_budget * share

    # Bounds as band around baseline
    lb = alpha_min * base_weekly
    ub = alpha_max * base_weekly

    x0 = base_weekly.copy()
    bounds = [(float(l), float(u)) for l, u in zip(lb, ub)]

    def objective(x: np.ndarray) -> float:
        x = np.clip(x, 0.0, None)
        spends = {ch: float(x[i]) for i, ch in enumerate(channels)}
        sales = float(model.predict_sales(spends, future_controls=avg_controls))
        return -sales

    def eq_constraint(x: np.ndarray) -> float:
        # Weekly budget should sum to weekly_budget
        return weekly_budget - float(np.sum(x))

    constraints = [{"type": "eq", "fun": eq_constraint}]

    res = minimize(
        objective,
        x0,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
    )

    x_opt = np.clip(res.x, 0.0, None)
    weekly_spends = {ch: float(x_opt[i]) for i, ch in enumerate(channels)}
    return weekly_spends



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
    """
    Two-stage budget optimization entry point.

    Parameters
    ----------
    models : dict
        brand -> OptimizedMMM instance.
    df : pd.DataFrame
        Weekly dataset with spend columns.
    brands : list of str
        Brand codes in the order to be optimized.
    total_budget : float
        Total annual budget (in £) to allocate across brands and channels.
    mode : {"revenue", "profit"}
        Optimisation mode:
            "revenue" -> pure revenue-based objective
            "profit"  -> weight revenue by brand profit margins in Stage 1
    profit_margins : dict or None
        profit_margins[brand] = margin as decimal (0.3 = 30%).
        Required/used only when mode == "profit".
    alpha_min : float, optional
        Lower bound multiplier around the scaled baseline weekly spends
        for intra-brand optimization.
    alpha_max : float, optional
        Upper bound multiplier around the scaled baseline weekly spends
        for intra-brand optimization.

    Returns
    -------
    res_df : pd.DataFrame
        Channel-level allocation with columns:
            ['Brand', 'Channel', 'Historical', 'Optimized']
        where Historical/Optimized are ANNUAL spends.
    x_opt : np.ndarray
        Flattened vector of optimized weekly spends, in the same order as channel_map.
    channel_map : dict
        channel_map[i] = (brand, channel) mapping indices of x_opt to channels.
    """
    # -----------------------------------
    # Shared historical information
    # -----------------------------------
    channel_shares, brand_hist_totals = _compute_hist_shares(df, models, brands)

    # -----------------------------------
    # Stage 1: brand-level optimization (revenue vs profit)
    # -----------------------------------
    brand_budgets = _optimize_brand_budgets(
        models=models,
        df=df,
        brands=brands,
        total_budget=total_budget,
        channel_shares=channel_shares,
        brand_hist_totals=brand_hist_totals,
        mode=mode,
        profit_margins=profit_margins,
        brand_min_mult=brand_min_mult,
        brand_max_mult=brand_max_mult,
    )

    # -----------------------------------
    # Stage 2: intra-brand channel optimization
    # -----------------------------------
    all_rows: List[Dict[str, float]] = []
    x_opt_list: List[float] = []
    channel_map: Dict[int, Tuple[str, str]] = {}
    idx = 0

    for brand in brands:
        model = models[brand]
        B_annual = brand_budgets.get(brand, 0.0)

        weekly_spends = _optimize_within_brand(
            brand=brand,
            model=model,
            brand_budget_annual=B_annual,
            channel_shares=channel_shares[brand],
            alpha_min=alpha_min,
            alpha_max=alpha_max,
        )

        for ch, weekly_val in weekly_spends.items():
            annual_opt = weekly_val * 52.0
            hist_annual = float(df[ch].sum())

            all_rows.append(
                {
                    "Brand": brand,
                    "Channel": ch,
                    "Historical": hist_annual,
                    "Optimized": annual_opt,
                }
            )

            x_opt_list.append(weekly_val)
            channel_map[idx] = (brand, ch)
            idx += 1

    res_df = pd.DataFrame(all_rows)
    x_opt = np.array(x_opt_list, dtype=float)

    return res_df, x_opt, channel_map
