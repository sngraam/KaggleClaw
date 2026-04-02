"""
metrics.py — Competition evaluation functions.
Fill in the evaluate() function to match your competition's metric.
The model will import and call evaluate() to self-assess its predictions.
"""

import numpy as np
from typing import Any


# ── Fill this in for your competition ──────────────────────────────────────────

METRIC_NAME = "accuracy"   # human-readable name shown in logs
HIGHER_IS_BETTER = True    # True for accuracy/AUC/F1, False for RMSE/log-loss


def evaluate(y_true: Any, y_pred: Any) -> float:
    """
    Compute the competition metric.

    Args:
        y_true: ground-truth labels / values (array-like)
        y_pred: model predictions (array-like)

    Returns:
        float — the metric score

    Examples (swap in your own logic):
        # Accuracy (classification)
        return float(np.mean(np.array(y_true) == np.array(y_pred)))

        # RMSE (regression)
        return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))

        # Log-loss (binary)
        from sklearn.metrics import log_loss
        return log_loss(y_true, y_pred)

        # AUC
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y_true, y_pred)
    """
    # ── Default: accuracy ──────────────────────────────────────────────────────
    return float(np.mean(np.array(y_true) == np.array(y_pred)))


# ── Helpers (model may use these) ──────────────────────────────────────────────

def score_submission(submission_path: str, ground_truth_path: str, target_col: str) -> float:
    """
    Load a CSV submission + ground truth and compute metric.
    Only usable if ground truth is available (e.g. on train split).
    """
    import pandas as pd

    sub = pd.read_csv(submission_path)
    gt = pd.read_csv(ground_truth_path)

    # align on index
    merged = gt.merge(sub, on=sub.columns[0], suffixes=("_true", "_pred"))
    y_true = merged[f"{target_col}_true"].values
    y_pred = merged[f"{target_col}_pred"].values

    score = evaluate(y_true, y_pred)
    direction = "↑ higher better" if HIGHER_IS_BETTER else "↓ lower better"
    print(f"[metrics] {METRIC_NAME} = {score:.6f}  ({direction})")
    return score
