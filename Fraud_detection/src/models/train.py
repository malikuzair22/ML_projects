"""
Model training module — trains multiple models, tunes thresholds, saves best model.
"""

import joblib
import numpy as np
import mlflow
import mlflow.sklearn
from pathlib import Path
from loguru import logger
from typing import Dict, Tuple, Any

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_curve, f1_score
)

MODELS_DIR = Path(__file__).resolve().parents[2] / "saved_models"
MODELS_DIR.mkdir(exist_ok=True)


def get_models() -> Dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=42),
        "xgboost": XGBClassifier(n_estimators=200, scale_pos_weight=577, learning_rate=0.05, max_depth=6, random_state=42, verbosity=0),
        "lightgbm": LGBMClassifier(n_estimators=200, class_weight="balanced", learning_rate=0.05, random_state=42, verbose=-1),
    }


def find_optimal_threshold(y_true, y_proba) -> float:
    """Find threshold that maximizes F1 score."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    logger.info(f"Optimal threshold: {best_threshold:.4f} (F1: {f1_scores[best_idx]:.4f})")
    return float(best_threshold)


def train_single_model(name, model, X_train, y_train, X_val, y_val):
    logger.info(f"Training {name}...")
    with mlflow.start_run(run_name=name, nested=True):
        model.fit(X_train, y_train)
        y_val_proba = model.predict_proba(X_val)[:, 1]
        threshold = find_optimal_threshold(y_val, y_val_proba)
        y_val_pred = (y_val_proba >= threshold).astype(int)
        metrics = {
            "roc_auc": round(roc_auc_score(y_val, y_val_proba), 4),
            "avg_precision": round(average_precision_score(y_val, y_val_proba), 4),
            "f1": round(f1_score(y_val, y_val_pred), 4),
            "threshold": round(threshold, 4),
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path=name)
        logger.success(f"{name} → ROC-AUC: {metrics['roc_auc']} | AP: {metrics['avg_precision']} | F1: {metrics['f1']}")
    return model, threshold, metrics


def train_all_models(X_train, y_train, X_val, y_val, experiment_name="credit_card_fraud") -> dict:
    mlflow.set_experiment(experiment_name)
    models = get_models()
    results = {}
    with mlflow.start_run(run_name="all_models"):
        for name, model in models.items():
            trained_model, threshold, metrics = train_single_model(name, model, X_train, y_train, X_val, y_val)
            results[name] = {"model": trained_model, "threshold": threshold, "metrics": metrics}
    return results


def select_best_model(results: dict, metric="avg_precision"):
    best_name = max(results, key=lambda k: results[k]["metrics"][metric])
    best = results[best_name]
    logger.success(f"Best model: {best_name} ({metric}: {best['metrics'][metric]})")
    return best_name, best["model"], best["threshold"]


def save_model(model, name: str, threshold: float, feature_names: list) -> Path:
    artifact = {"model": model, "threshold": threshold, "feature_names": feature_names, "model_name": name}
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(artifact, path)
    logger.success(f"Model saved to: {path}")
    return path


def load_model(name: str = None, path: str = None) -> dict:
    if path:
        load_path = Path(path)
    else:
        files = list(MODELS_DIR.glob("*.joblib"))
        if not files:
            raise FileNotFoundError("No saved models found. Train first.")
        load_path = files[0] if not name else MODELS_DIR / f"{name}.joblib"
    artifact = joblib.load(load_path)
    logger.info(f"Loaded model: {artifact['model_name']}")
    return artifact
