# data_prep.py
from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def create_fourier_terms(df: pd.DataFrame, k: int = 2) -> pd.DataFrame:
    """
    Add Fourier seasonal terms to a weekly dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe sorted by time index.
    k : int, optional
        Number of sine/cosine orders to create. Default is 2.

    Returns
    -------
    pd.DataFrame
        Original dataframe with added columns: sin_i, cos_i for i in 1..k.
    """
    t = df.index.values + 1
    n = 52  # weekly seasonality

    for order in range(1, k + 1):
        df[f"sin_{order}"] = np.sin(2 * np.pi * order * t / n)
        df[f"cos_{order}"] = np.cos(2 * np.pi * order * t / n)

    return df


def load_and_prep_data(filepath: str, fourier_order: int = 2) -> pd.DataFrame:
    """
    Load raw KPI and media data and apply basic feature engineering.

    Parameters
    ----------
    filepath : str
        Path to the input CSV file.
    fourier_order : int, optional
        Number of Fourier terms to add. Default is 2.

    Returns
    -------
    pd.DataFrame
        Prepared weekly dataset with trend, Fourier terms and peak flags.
    """
    logger.info("Loading data from %s", filepath)
    df = pd.read_csv(filepath)

    df["wc_sun"] = pd.to_datetime(df["wc_sun"])
    df = df.sort_values("wc_sun").reset_index(drop=True)

    # Simple linear trend
    df["trend"] = df.index + 1

    # Fourier seasonality
    df = create_fourier_terms(df, k=fourier_order)

    # Peak months (Nov, Dec)
    df["month"] = df["wc_sun"].dt.month
    df["is_peak"] = df["month"].isin([11, 12]).astype(int)

    logger.info("Data loaded and preprocessed. Shape: %s", df.shape)
    return df


def group_sparse_channels(
    df: pd.DataFrame,
    brand: str,
    offline_prefixes: List[str],
    exclude_channels: List[str],
) -> List[str]:
    """
    Collapse sparse offline channels and filter active media channels for a brand.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset including all brands and channels.
    brand : str
        Brand short code, e.g. 'bb', 'mac', 'tf'.
    offline_prefixes : list of str
        List of prefixes used to identify offline channels.
    exclude_channels : list of str
        Channels to exclude entirely from modeling.

    Returns
    -------
    list of str
        Active channel names for the given brand, after grouping and filtering.
    """
    logger.debug("Grouping channels for brand %s", brand)

    # Offline channels grouped into a single bucket
    brand_offline_cols = [
        c for c in df.columns
        if brand in c and any(p in c for p in offline_prefixes)
    ]
    if brand_offline_cols:
        col_name = f"m3_offline_{brand}_sp"
        df[col_name] = df[brand_offline_cols].sum(axis=1)

    # Digital media: standard m1_*_sp naming
    brand_digital_cols = [
        c for c in df.columns if brand in c and "m1_" in c and "_sp" in c
    ]

    final_channels = brand_digital_cols + (
        [f"m3_offline_{brand}_sp"] if brand_offline_cols else []
    )

    # Remove excluded channels
    active = [c for c in final_channels if c not in exclude_channels]

    # Drop channels with <= 2 non-zero weeks
    active = [c for c in active if (df[c] > 0).sum() > 2]

    logger.info("Brand %s has %d active channels", brand, len(active))
    return active
