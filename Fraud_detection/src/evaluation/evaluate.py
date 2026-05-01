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
        plt.savefig(REPORTS_DIR / f"{model_name.lower().replace(' ', '_')}_cm.png", dpi=150)
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
    print(f"\n{'='*55}")
    print(f"  REPORT: {model_name.upper()}")
    print(f"{'='*55}")
    for k, v in metrics.items():
        print(f"  {k:<20}: {v}")
    print(f"{'='*55}\n")
    y_pred = (y_proba >= threshold).astype(int)
    print(classification_report(y_true, y_pred, target_names=["Legitimate", "Fraud"]))
    return metrics
