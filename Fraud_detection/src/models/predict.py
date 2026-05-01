"""
Prediction module — loads model and makes predictions on new data.
"""

import pandas as pd
from loguru import logger
from src.models.train import load_model


def predict_single(transaction: dict, model_name: str = None) -> dict:
    """Predict fraud for a single transaction."""
    artifact = load_model(name=model_name)
    model = artifact["model"]
    threshold = artifact["threshold"]
    feature_names = artifact["feature_names"]

    df = pd.DataFrame([transaction]).reindex(columns=feature_names, fill_value=0)
    proba = model.predict_proba(df)[:, 1][0]
    prediction = int(proba >= threshold)

    risk_level = "LOW"
    if proba >= 0.8:
        risk_level = "CRITICAL"
    elif proba >= 0.5:
        risk_level = "HIGH"
    elif proba >= threshold:
        risk_level = "MEDIUM"

    return {
        "is_fraud": bool(prediction),
        "fraud_probability": round(float(proba), 4),
        "risk_level": risk_level,
        "threshold_used": threshold,
        "model_used": artifact["model_name"],
    }


def predict_batch(df: pd.DataFrame, model_name: str = None) -> pd.DataFrame:
    """Predict fraud for a batch of transactions."""
    artifact = load_model(name=model_name)
    model = artifact["model"]
    threshold = artifact["threshold"]
    feature_names = artifact["feature_names"]

    X = df.reindex(columns=feature_names, fill_value=0)
    probas = model.predict_proba(X)[:, 1]
    predictions = (probas >= threshold).astype(int)

    result_df = df.copy()
    result_df["fraud_probability"] = probas.round(4)
    result_df["is_fraud"] = predictions
    result_df["risk_level"] = pd.cut(
        probas,
        bins=[-0.001, threshold * 0.5, threshold, 0.8, 1.001],
        labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    )
    logger.info(f"Batch: {predictions.sum()} fraud out of {len(df)} transactions")
    return result_df
