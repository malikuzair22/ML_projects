"""
Feature engineering for Credit Card Fraud Detection.
"""

import numpy as np
import pandas as pd
from loguru import logger


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour_of_day"] = (df["Time"] // 3600) % 24
    df["is_night"] = df["hour_of_day"].apply(lambda h: 1 if (h >= 22 or h <= 6) else 0)
    return df


def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_amount"] = np.log1p(df["Amount"])
    df["amount_bin"] = pd.cut(
        df["Amount"],
        bins=[0, 10, 50, 200, 1000, np.inf],
        labels=["micro", "small", "medium", "large", "xlarge"]
    ).astype(str)
    return df


def add_v_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    top_v = ["V1", "V3", "V4", "V10", "V11", "V12", "V14", "V17"]
    pairs = [("V1", "V3"), ("V4", "V11"), ("V12", "V14"), ("V10", "V17")]
    for a, b in pairs:
        if a in df.columns and b in df.columns:
            df[f"{a}_x_{b}"] = df[a] * df[b]
    available = [v for v in top_v if v in df.columns]
    df["v_top_magnitude"] = np.sqrt((df[available] ** 2).sum(axis=1))
    return df


def add_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    v_cols = [c for c in df.columns if c.startswith("V")]
    df["v_mean"] = df[v_cols].mean(axis=1)
    df["v_std"] = df[v_cols].std(axis=1)
    df["v_skew"] = df[v_cols].skew(axis=1)
    df["v_kurtosis"] = df[v_cols].kurtosis(axis=1)
    return df


def engineer_features(df: pd.DataFrame, drop_original=True) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    logger.info("Starting feature engineering...")
    df = add_time_features(df)
    df = add_amount_features(df)
    df = add_v_interaction_features(df)
    df = add_statistical_features(df)
    df = pd.get_dummies(df, columns=["amount_bin"], prefix="amt")
    if drop_original:
        df.drop([c for c in ["Time", "Amount"] if c in df.columns], axis=1, inplace=True)
    logger.success(f"Feature engineering complete. Shape: {df.shape}")
    return df
