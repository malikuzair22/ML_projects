"""
Run this script ONCE from inside your 'files' folder.
It will create all folders and write every project file automatically.

Usage:
    cd C:\\Users\\Hp\\Desktop\\ML_projects\\files
    python setup_project.py
"""

import os

# ── Helper ────────────────────────────────────────────────────────────────────

def write(path, content):
    parent = os.path.dirname(path)
    if parent:
      os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  created: {path}")

def touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("")

# ── Create __init__.py files ──────────────────────────────────────────────────

for p in [
    "src/__init__.py",
    "src/data/__init__.py",
    "src/features/__init__.py",
    "src/models/__init__.py",
    "src/evaluation/__init__.py",
    "src/api/__init__.py",
]:
    touch(p)

print("\n✅ Created __init__.py files\n")

# ── requirements.txt ─────────────────────────────────────────────────────────

write("requirements.txt", """\
# Data
pandas==2.1.0
numpy==1.24.3
kaggle==1.6.6

# ML
scikit-learn==1.3.0
xgboost==2.0.0
imbalanced-learn==0.11.0
lightgbm==4.1.0

# Evaluation & Visualization
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.17.0
shap==0.43.0

# API
fastapi==0.103.1
uvicorn==0.23.2
pydantic==2.3.0
python-multipart==0.0.6

# Experiment Tracking
mlflow==2.7.1

# Utilities
joblib==1.3.2
python-dotenv==1.0.0
loguru==0.7.2
""")

# ── src/data/load_data.py ─────────────────────────────────────────────────────

write("src/data/load_data.py", '''\
"""
Data loading module for Credit Card Fraud Detection.
Downloads dataset from Kaggle or loads from local path.
"""

import os
import pandas as pd
from pathlib import Path
from loguru import logger


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATASET_NAME = "mlg-ulb/creditcardfraud"
CSV_FILENAME = "creditcard.csv"


def download_from_kaggle() -> Path:
    """Download dataset from Kaggle using kaggle API."""
    try:
        import kaggle
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading dataset from Kaggle...")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            DATASET_NAME, path=str(DATA_DIR), unzip=True
        )
        logger.success("Dataset downloaded successfully.")
    except Exception as e:
        logger.error(f"Kaggle download failed: {e}")
        logger.info("Make sure ~/.kaggle/kaggle.json is configured.")
        raise
    return DATA_DIR / CSV_FILENAME


def load_data(filepath: str = None) -> pd.DataFrame:
    """Load credit card fraud dataset."""
    if filepath:
        path = Path(filepath)
    else:
        path = DATA_DIR / CSV_FILENAME
        if not path.exists():
            path = download_from_kaggle()

    logger.info(f"Loading data from: {path}")
    df = pd.read_csv(path)
    logger.success(f"Loaded {len(df):,} rows and {df.shape[1]} columns.")
    return df


def get_basic_info(df: pd.DataFrame) -> dict:
    """Return basic dataset statistics."""
    fraud = df["Class"].sum()
    total = len(df)
    return {
        "total_transactions": total,
        "fraud_cases": int(fraud),
        "legitimate_cases": int(total - fraud),
        "fraud_percentage": round((fraud / total) * 100, 4),
        "missing_values": int(df.isnull().sum().sum()),
        "shape": df.shape,
    }


if __name__ == "__main__":
    df = load_data()
    info = get_basic_info(df)
    for k, v in info.items():
        print(f"{k}: {v}")
''')

# ── src/data/preprocess.py ────────────────────────────────────────────────────

write("src/data/preprocess.py", '''\
"""
Preprocessing module: scaling, train/test split, handling class imbalance.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from loguru import logger
from typing import Tuple


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Scale Amount and Time using RobustScaler."""
    df = df.copy()
    scaler = RobustScaler()
    df["scaled_amount"] = scaler.fit_transform(df[["Amount"]])
    df["scaled_time"] = scaler.fit_transform(df[["Time"]])
    df.drop(["Amount", "Time"], axis=1, inplace=True)
    logger.info("Scaled Amount and Time using RobustScaler.")
    return df


def split_data(df: pd.DataFrame, test_size=0.2, val_size=0.1, random_state=42):
    """Split into train, validation, and test sets (stratified)."""
    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=val_size / (1 - test_size),
        random_state=random_state, stratify=y_train
    )
    logger.info(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def handle_imbalance(X_train, y_train, strategy="smote", random_state=42):
    """Handle class imbalance: smote | undersample | combined."""
    logger.info(f"Before resampling — Fraud: {y_train.sum():,} | Legit: {(y_train==0).sum():,}")

    if strategy == "smote":
        sampler = SMOTE(random_state=random_state)
    elif strategy == "undersample":
        sampler = RandomUnderSampler(random_state=random_state)
    elif strategy == "combined":
        sampler = ImbPipeline([
            ("smote", SMOTE(sampling_strategy=0.1, random_state=random_state)),
            ("under", RandomUnderSampler(sampling_strategy=0.5, random_state=random_state)),
        ])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    X_res, y_res = sampler.fit_resample(X_train, y_train)
    logger.success(f"After resampling — Fraud: {y_res.sum():,} | Legit: {(y_res==0).sum():,}")
    return X_res, y_res


def preprocess_pipeline(df: pd.DataFrame, imbalance_strategy="smote") -> dict:
    """Full preprocessing pipeline."""
    df = scale_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    X_train_res, y_train_res = handle_imbalance(X_train, y_train, strategy=imbalance_strategy)
    return {
        "X_train": X_train_res, "X_val": X_val, "X_test": X_test,
        "y_train": y_train_res, "y_val": y_val, "y_test": y_test,
        "X_train_orig": X_train, "y_train_orig": y_train,
    }
''')

# ── src/features/feature_engineering.py ──────────────────────────────────────

write("src/features/feature_engineering.py", '''\
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
''')

# ── src/models/train.py ───────────────────────────────────────────────────────

write("src/models/train.py", '''\
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
        logger.success(f"{name} → ROC-AUC: {metrics[\'roc_auc\']} | AP: {metrics[\'avg_precision\']} | F1: {metrics[\'f1\']}")
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
    logger.success(f"Best model: {best_name} ({metric}: {best[\'metrics\'][metric]})")
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
    logger.info(f"Loaded model: {artifact[\'model_name\']}")
    return artifact
''')

# ── src/models/predict.py ─────────────────────────────────────────────────────

write("src/models/predict.py", '''\
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
''')

# ── src/evaluation/evaluate.py ────────────────────────────────────────────────

write("src/evaluation/evaluate.py", '''\
"""
Evaluation module — metrics and visualizations for fraud detection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from loguru import logger
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score
)

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
plt.style.use("seaborn-v0_8-darkgrid")


def compute_all_metrics(y_true, y_proba, threshold=0.5) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
        "avg_precision": round(average_precision_score(y_true, y_proba), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "fraud_caught": int(y_pred[y_true == 1].sum()),
        "total_fraud": int(y_true.sum()),
        "false_alarms": int(y_pred[y_true == 0].sum()),
    }


def plot_confusion_matrix(y_true, y_pred, model_name="Model", save=True):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, title in zip(axes, [cm, cm_norm], ["Counts", "Normalized"]):
        sns.heatmap(data, annot=True, fmt=".2f" if title == "Normalized" else "d",
                    cmap="Blues", ax=ax,
                    xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
        ax.set_title(f"{model_name} — Confusion Matrix ({title})")
        ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    plt.tight_layout()
    if save:
        plt.savefig(REPORTS_DIR / f"{model_name.lower().replace(\' \', \'_\')}_cm.png", dpi=150)
    plt.show()


def plot_roc_pr_curves(models_results: dict, y_true, save=True):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for name, result in models_results.items():
        y_proba = result["proba"]
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        axes[0].plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc_score(y_true, y_proba):.4f})")
        prec, rec, _ = precision_recall_curve(y_true, y_proba)
        axes[1].plot(rec, prec, lw=2, label=f"{name} (AP={average_precision_score(y_true, y_proba):.4f})")
    axes[0].plot([0,1],[0,1],"k--"); axes[0].set_title("ROC Curve"); axes[0].legend()
    axes[1].set_title("Precision-Recall Curve"); axes[1].legend()
    for ax in axes: ax.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig(REPORTS_DIR / "roc_pr_curves.png", dpi=150)
    plt.show()


def plot_shap_values(model, X_sample, model_name="Model", save=True):
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        plt.sca(axes[0]); shap.summary_plot(sv, X_sample, plot_type="bar", show=False)
        plt.sca(axes[1]); shap.summary_plot(sv, X_sample, show=False)
        plt.tight_layout()
        if save:
            plt.savefig(REPORTS_DIR / f"{model_name.lower()}_shap.png", dpi=150)
        plt.show()
    except Exception as e:
        logger.warning(f"SHAP failed: {e}")


def generate_full_report(y_true, y_proba, threshold, model_name) -> dict:
    metrics = compute_all_metrics(y_true, y_proba, threshold)
    print(f"\\n{\'=\'*55}")
    print(f"  REPORT: {model_name.upper()}")
    print(f"{\'=\'*55}")
    for k, v in metrics.items():
        print(f"  {k:<20}: {v}")
    print(f"{\'=\'*55}\\n")
    y_pred = (y_proba >= threshold).astype(int)
    print(classification_report(y_true, y_pred, target_names=["Legitimate", "Fraud"]))
    return metrics
''')

# ── src/api/schemas.py ────────────────────────────────────────────────────────

write("src/api/schemas.py", '''\
"""
Pydantic schemas for FastAPI request/response validation.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List
from enum import Enum


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class TransactionInput(BaseModel):
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float = Field(..., ge=0)
    Time: float = Field(..., ge=0)

    class Config:
        json_schema_extra = {"example": {
            "V1": -1.36, "V2": -0.07, "V3": 2.53, "V4": 1.38,
            "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
            "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
            "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
            "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
            "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
            "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02,
            "Amount": 149.62, "Time": 0.0
        }}


class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float = Field(..., ge=0, le=1)
    risk_level: RiskLevel
    threshold_used: float
    model_used: str
    message: str


class BatchTransactionInput(BaseModel):
    transactions: List[TransactionInput]

    @validator("transactions")
    def check_size(cls, v):
        if len(v) > 1000:
            raise ValueError("Batch size cannot exceed 1000.")
        return v


class BatchPredictionResponse(BaseModel):
    total: int
    fraud_detected: int
    fraud_rate: float
    predictions: List[PredictionResponse]


class ModelInfoResponse(BaseModel):
    model_name: str
    threshold: float
    feature_count: int
    status: str
''')

# ── src/api/main.py ───────────────────────────────────────────────────────────

write("src/api/main.py", '''\
"""
FastAPI application for Credit Card Fraud Detection.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import time
from loguru import logger
from contextlib import asynccontextmanager

from src.api.schemas import (
    TransactionInput, PredictionResponse,
    BatchTransactionInput, BatchPredictionResponse,
    ModelInfoResponse, RiskLevel
)
from src.models.predict import predict_single, predict_batch
from src.models.train import load_model

model_artifact = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading fraud detection model...")
    try:
        model_artifact.update(load_model())
        logger.success(f"Model loaded: {model_artifact[\'model_name\']}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
    yield

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="AI-powered fraud detection — single, batch, and CSV predictions.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def build_response(result: dict) -> PredictionResponse:
    messages = {
        "LOW": "Transaction appears legitimate.",
        "MEDIUM": "Transaction requires monitoring.",
        "HIGH": "Potential fraud detected. Review recommended.",
        "CRITICAL": "High confidence fraud. Block immediately.",
    }
    return PredictionResponse(**result, message=messages.get(result["risk_level"], ""))


@app.get("/", tags=["Health"])
async def root():
    return {"status": "running", "version": "1.0.0"}


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy" if model_artifact else "degraded", "model_loaded": bool(model_artifact)}


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    if not model_artifact:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return ModelInfoResponse(
        model_name=model_artifact["model_name"],
        threshold=model_artifact["threshold"],
        feature_count=len(model_artifact["feature_names"]),
        status="loaded",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(transaction: TransactionInput):
    try:
        result = predict_single(transaction.model_dump())
        return build_response(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch_endpoint(batch: BatchTransactionInput):
    try:
        df = pd.DataFrame([t.model_dump() for t in batch.transactions])
        results_df = predict_batch(df)
        predictions = [
            PredictionResponse(
                is_fraud=bool(row["is_fraud"]),
                fraud_probability=row["fraud_probability"],
                risk_level=RiskLevel(row["risk_level"]),
                threshold_used=model_artifact.get("threshold", 0.5),
                model_used=model_artifact.get("model_name", "unknown"),
                message="Batch prediction"
            ) for _, row in results_df.iterrows()
        ]
        fraud_count = results_df["is_fraud"].sum()
        return BatchPredictionResponse(
            total=len(predictions), fraud_detected=int(fraud_count),
            fraud_rate=round(float(fraud_count / len(predictions)), 4),
            predictions=predictions,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/csv", tags=["Prediction"])
async def predict_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files accepted.")
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        if len(df) > 5000:
            raise HTTPException(status_code=400, detail="CSV cannot exceed 5000 rows.")
        results_df = predict_batch(df)
        fraud_count = int(results_df["is_fraud"].sum())
        return {
            "filename": file.filename, "total_rows": len(df),
            "fraud_detected": fraud_count,
            "fraud_rate": round(fraud_count / len(df), 4),
            "results": results_df[["fraud_probability", "is_fraud", "risk_level"]].to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
''')

# ── Dockerfile ────────────────────────────────────────────────────────────────

write("Dockerfile", """\
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p data saved_models reports/figures
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
""")

# ── docker-compose.yml ────────────────────────────────────────────────────────

write("docker-compose.yml", """\
version: "3.9"
services:
  fraud-api:
    build: .
    container_name: fraud-detection-api
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./saved_models:/app/saved_models
      - ./reports:/app/reports
    restart: unless-stopped

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    container_name: mlflow-server
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlflow/mlruns
    command: mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri /mlflow/mlruns
    restart: unless-stopped
""")

# ── .gitignore ────────────────────────────────────────────────────────────────

write(".gitignore", """\
data/
saved_models/
mlruns/
__pycache__/
*.pyc
.env
*.joblib
reports/figures/*.png
.ipynb_checkpoints/
.venv/
""")

# ── Final summary ─────────────────────────────────────────────────────────────

print("\n" + "="*55)
print("  ✅ PROJECT SETUP COMPLETE!")
print("="*55)
print("\nFolder structure:")
for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in [".venv", "__pycache__", ".git", "mlruns"]]
    level = root.replace(".", "").count(os.sep)
    indent = "  " * level
    print(f"{indent}{os.path.basename(root)}/")
    for f in files:
        if not f.endswith(".pyc"):
            print(f"{indent}  {f}")

print("\n📌 Next steps:")
print("  1. Add Kaggle credentials: https://www.kaggle.com/settings -> API -> Create Token")
print("     Save kaggle.json to: C:\\Users\\Hp\\.kaggle\\kaggle.json")
print("  2. Download dataset:")
print("     python -c \"from src.data.load_data import download_from_kaggle; download_from_kaggle()\"")
print("  3. Open notebook:")
print("     jupyter notebook notebooks/fraud_detection.ipynb")
print("  4. Run API:")
print("     uvicorn src.api.main:app --reload")
