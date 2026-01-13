# model.py
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from scipy.optimize import lsq_linear


logger = logging.getLogger(__name__)


def geometric_adstock(x: np.ndarray, decay: float) -> np.ndarray:
    """
    Apply geometric adstock transformation to a spend series.

    Parameters
    ----------
    x : np.ndarray
        Raw weekly spend vector.
    decay : float
        Adstock decay parameter in (0, 1).

    Returns
    -------
    np.ndarray
        Adstocked spend vector.
    """
    x_adstocked = np.zeros_like(x, dtype=float)
    if len(x) == 0:
        return x_adstocked

    x_adstocked[0] = x[0]
    for t in range(1, len(x)):
        x_adstocked[t] = x[t] + decay * x_adstocked[t - 1]
    return x_adstocked


def hill_function(x: np.ndarray, K: float, S: float) -> np.ndarray:
    """
    Apply Hill saturation function to a non negative input vector.

    Parameters
    ----------
    x : np.ndarray
        Input (e.g. adstocked spend).
    K : float
        Half-saturation point.
    S : float
        Shape parameter.

    Returns
    -------
    np.ndarray
        Saturated response for each element in x.
    """
    x = np.clip(x, 0, None)
    denom = (K ** S + x ** S + 1e-9)
    return (x ** S) / denom


class OptimizedMMM:
    """
    Media mix model with geometric adstock, Hill saturation and ridge regularization.

    This class encapsulates all brand level modeling:
    - random search over adstock and Hill parameters
    - ridge regularized linear regression with positivity constraints on media
    - functions for prediction and revenue decomposition
    """

    def __init__(
        self,
        brand: str,
        param_ranges: Dict[str, Tuple[float, float]],
        iterations: int,
        ridge_alpha: float,
        control_alpha_ratio: float = 0.1,
        search_penalty_factor: float = 5.0,
    ) -> None:
        """
        Parameters
        ----------
        brand : str
            Brand short code.
        param_ranges : dict
            Ranges for hyperparameters, keys: 'decay', 'S'.
            Example: {'decay': (0.1, 0.8), 'S': (0.8, 2.5)}.
        iterations : int
            Number of random hyperparameter samples per brand.
        ridge_alpha : float
            Base ridge penalty strength applied to media coefficients.
        control_alpha_ratio : float, optional
            Controls ridge penalty as a fraction of ridge_alpha.
        search_penalty_factor : float, optional
            Extra multiplier for Search channels (name contains 'search').
        """
        self.brand = brand
        self.param_ranges = param_ranges
        self.iterations = iterations
        self.ridge_alpha = ridge_alpha
        self.control_alpha_ratio = control_alpha_ratio
        self.search_penalty_factor = search_penalty_factor

        self.best_params: Dict[str, Dict[str, float]] = {}
        self.best_r2: float = float("-inf")
        self.channels: List[str] = []

        self.intercept: float = 0.0
        self.control_betas: np.ndarray = np.array([])
        self.media_betas: np.ndarray = np.array([])
        self.y_pred: np.ndarray = np.array([])

    def tune_and_fit(
        self,
        df: pd.DataFrame,
        channels: List[str],
        kpi_col: str,
    ) -> None:
        """
        Tune adstock and Hill parameters via random search and fit ridge MMM.

        Parameters
        ----------
        df : pd.DataFrame
            Prepared dataset including KPI and media columns.
        channels : list of str
            List of channel column names for this brand.
        kpi_col : str
            Name of the KPI column for this brand (e.g. 'm0_kpi_bb').

        Returns
        -------
        None
        """
        logger.info("[%s] Tuning %d channels", self.brand.upper(), len(channels))

        self.channels = channels
        y_raw = df[kpi_col].values.astype(float)

        control_cols = ["trend", "sin_1", "cos_1", "sin_2", "cos_2"]
        X_controls_raw = df[control_cols].values.astype(float)

        # Standardise y and controls once per brand
        y_mean = float(np.mean(y_raw))
        y_std = float(np.std(y_raw)) or 1.0
        y_scaled = (y_raw - y_mean) / y_std

        X_controls_mean = np.mean(X_controls_raw, axis=0)
        X_controls_std = np.std(X_controls_raw, axis=0)
        X_controls_std[X_controls_std == 0] = 1.0
        X_controls_scaled = (X_controls_raw - X_controls_mean) / X_controls_std

        n_media = len(channels)
        n_controls = X_controls_raw.shape[1]
        n_intercept = 1

        # --- Ridge penalties ---
        # Media: base ridge_alpha, with extra weight for Search.
        media_penalties: List[float] = []
        for col in channels:
            w = 1.0
            if "search" in col.lower():
                w *= self.search_penalty_factor
            media_penalties.append(np.sqrt(self.ridge_alpha * w))

        # Controls: small ridge penalty
        control_alpha = self.ridge_alpha * self.control_alpha_ratio
        control_penalties = np.full(n_controls, np.sqrt(control_alpha))

        # Intercept: no penalty
        intercept_penalties = np.zeros(n_intercept)

        diag_penalty = np.concatenate(
            [np.array(media_penalties), control_penalties, intercept_penalties]
        )

        X_aug_extra = np.diag(diag_penalty)
        y_aug_extra = np.zeros(len(diag_penalty))
        y_final = np.concatenate([y_scaled, y_aug_extra])

        for _ in range(self.iterations):
            current_params: Dict[str, Dict[str, float]] = {}
            X_media_list: List[np.ndarray] = []

            # Build transformed media features for the current hyperparameters
            for col in channels:
                decay = float(
                    np.random.uniform(*self.param_ranges["decay"])
                )
                S = float(
                    np.random.uniform(*self.param_ranges["S"])
                )

                x_raw = df[col].values.astype(float)
                x_ads = geometric_adstock(x_raw, decay)
                # Use mean positive adstock value as heuristic K
                K = float(np.mean(x_ads[x_ads > 0])) if np.sum(x_ads) > 0 else 1.0

                X_media_list.append(hill_function(x_ads, K, S))

                current_params[col] = {"decay": decay, "K": K, "S": S}

            # Standardise media features per iteration
            X_media_raw = np.column_stack(X_media_list)
            X_media_mean = np.mean(X_media_raw, axis=0)
            X_media_std = np.std(X_media_raw, axis=0)
            X_media_std[X_media_std == 0] = 1.0
            X_media_scaled = (X_media_raw - X_media_mean) / X_media_std

            intercept_col = np.ones((len(y_scaled), 1))
            X_mat_scaled = np.hstack(
                [X_media_scaled, X_controls_scaled, intercept_col]
            )

            X_final = np.vstack([X_mat_scaled, X_aug_extra])

            # Constraints on scaled coefficients
            # - Media >= 0
            # - Controls free
            # - Intercept >= 0  (reintroduced lower bound)
            lb = [0.0] * n_media + [-np.inf] * n_controls + [0.0]
            ub = [np.inf] * n_media + [np.inf] * n_controls + [np.inf]

            res = lsq_linear(
                X_final,
                y_final,
                bounds=(lb, ub),
                lsq_solver="lsmr",
            )

            betas_scaled = res.x

            # Unscale coefficients back to original space
            b_media_scaled = betas_scaled[:n_media]
            b_controls_scaled = betas_scaled[n_media : n_media + n_controls]
            b_intercept_scaled = betas_scaled[-1]

            b_media_raw = b_media_scaled * (y_std / X_media_std)
            b_controls_raw = b_controls_scaled * (y_std / X_controls_std)

            shift_media = float(np.sum(b_media_raw * X_media_mean))
            shift_controls = float(np.sum(b_controls_raw * X_controls_mean))
            b_intercept_raw = (
                y_mean - shift_media - shift_controls + (b_intercept_scaled * y_std)
            )

            # Compute in sample predictions
            X_mat_raw = np.hstack(
                [X_media_raw, X_controls_raw, np.ones((len(y_raw), 1))]
            )
            all_betas_raw = np.concatenate(
                [b_media_raw, b_controls_raw, [b_intercept_raw]]
            )
            preds = X_mat_raw @ all_betas_raw
            r2 = r2_score(y_raw, preds)

            if r2 > self.best_r2:
                self.best_r2 = r2
                self.best_params = current_params
                self.y_pred = preds
                self.media_betas = b_media_raw
                self.control_betas = b_controls_raw
                self.intercept = b_intercept_raw

        # Attach beta per channel for downstream use
        for i, col in enumerate(self.channels):
            self.best_params[col]["beta"] = float(self.media_betas[i])

        logger.info("[%s] Best R2: %.4f", self.brand.upper(), self.best_r2)

    def predict_sales(
        self,
        spend_dict: Dict[str, float],
        future_controls: np.ndarray,
    ) -> float:
        """
        Predict sales for a brand given weekly spends and control values.

        Parameters
        ----------
        spend_dict : dict
            Mapping of channel name to weekly spend level.
        future_controls : np.ndarray
            Vector of control values for prediction horizon
            (trend, sin_1, cos_1, sin_2, cos_2).

        Returns
        -------
        float
            Predicted weekly sales.
        """
        base_sales = float(self.intercept + np.dot(future_controls, self.control_betas))
        media_sales = 0.0

        for col in self.channels:
            if col in spend_dict:
                val = float(spend_dict[col])
                p = self.best_params[col]
                ss = val / (1 - p["decay"])
                media_sales += p["beta"] * hill_function(
                    np.array([ss]), p["K"], p["S"]
                )[0]

        return base_sales + media_sales

    def decompose_historical(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Decompose historical sales into baseline and media driver contributions.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset containing historical period.

        Returns
        -------
        dict
            Mapping driver name to summed revenue contribution over history.
        """
        decomp: Dict[str, float] = {}

        control_cols = ["trend", "sin_1", "cos_1", "sin_2", "cos_2"]
        X_controls = df[control_cols].values.astype(float)

        baseline_contrib = self.intercept + (X_controls @ self.control_betas)
        baseline_sum = float(np.sum(baseline_contrib))

        # Diagnostic: log raw baseline before any clamp
        logger.info(
            "[%s] Baseline_raw (sum of intercept + controls) = %.2f",
            self.brand.upper(),
            baseline_sum,
        )

        # Clamp baseline at zero for reporting (unchanged behaviour for downstream)
        decomp["Baseline"] = max(0.0, baseline_sum)

        for col in self.channels:
            p = self.best_params[col]
            x_raw = df[col].values.astype(float)
            x_ads = geometric_adstock(x_raw, p["decay"])
            contrib = np.sum(
                p["beta"] * hill_function(x_ads, p["K"], p["S"])
            )
            decomp[col] = float(contrib)

        return decomp

    def decompose_optimized(
        self,
        weekly_spend_dict: Dict[str, float],
        num_weeks: int = 52,
    ) -> Dict[str, float]:
        """
        Decompose optimized future revenue into baseline and media drivers.

        Parameters
        ----------
        weekly_spend_dict : dict
            Mapping channel -> weekly spend level under optimized plan.
        num_weeks : int, optional
            Number of weeks in the planning horizon. Default is 52.

        Returns
        -------
        dict
            Mapping driver name to total revenue over the horizon.
        """
        # Fixed average controls, same as original script
        avg_controls = np.array([26.5, 0.0, 0.0, 0.0, 0.0])
        baseline_weekly = float(
            self.intercept + np.dot(avg_controls, self.control_betas)
        )

        decomp: Dict[str, float] = {}
        decomp["Baseline"] = max(0.0, baseline_weekly * num_weeks)

        for col in self.channels:
            val = float(weekly_spend_dict.get(col, 0.0))
            p = self.best_params[col]
            ss = val / (1 - p["decay"])
            weekly_contrib = p["beta"] * hill_function(
                np.array([ss]), p["K"], p["S"]
            )[0]
            decomp[col] = float(weekly_contrib * num_weeks)

        return decomp
