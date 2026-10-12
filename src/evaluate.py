"""
Evaluation metrics and scientific report utilities.
Ensures correct ground-truth / prediction ordering and robust error handling.
"""

import os
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)


def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Compute full evaluation metrics with explicit positive-class definition (1 = sarcastic).
    
    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        y_prob: Optional predicted probabilities for positive class (float in [0, 1]).
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    
    acc = float(accuracy_score(y_true, y_pred))
    
    # Binary metrics for class 1 (sarcastic)
    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    
    # Averaged metrics
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    
    # Confusion matrix: rows = true labels, cols = predicted labels
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    
    report = classification_report(y_true, y_pred, target_names=["Non-sarcastic", "Sarcastic"], output_dict=True, zero_division=0)
    
    metrics = {
        "accuracy": acc,
        "precision_sarcastic": prec,
        "recall_sarcastic": rec,
        "f1_sarcastic": f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm,
        "detailed_report": report,
        "total_samples": len(y_true),
        "true_class_counts": {
            "non_sarcastic": int(np.sum(y_true == 0)),
            "sarcastic": int(np.sum(y_true == 1))
        }
    }
    
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            metrics["average_precision"] = float(average_precision_score(y_true, y_prob))
        except Exception as e:
            metrics["roc_auc_error"] = str(e)
            
    return metrics


def save_metrics_to_json(metrics_dict, filepath):
    """Save evaluation metrics as machine-readable JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)
