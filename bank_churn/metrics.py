"""Probability and threshold-based classification metrics."""
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def calculate_metrics(y_true, probability, threshold=0.5):
    probability = np.asarray(probability, dtype=float)
    y_true = np.asarray(y_true)
    if probability.ndim != 1 or len(probability) != len(y_true):
        raise ValueError("Probabilities and labels must be aligned one-dimensional arrays.")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Probabilities must be finite and between 0 and 1.")
    predicted = (probability >= threshold).astype(int)
    return {"accuracy": float(accuracy_score(y_true, predicted)),
            "precision": float(precision_score(y_true, predicted, zero_division=0)),
            "recall": float(recall_score(y_true, predicted, zero_division=0)),
            "f1": float(f1_score(y_true, predicted, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, probability))}
