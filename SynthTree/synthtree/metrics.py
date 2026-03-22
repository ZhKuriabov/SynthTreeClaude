from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def score_predictions(y_true, scores, classification: bool) -> float:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    if classification:
        if np.unique(y_true).size < 2:
            return 0.5
        return float(roc_auc_score(y_true, scores))
    return rmse(y_true, scores)


def validation_objective(y_true, scores, classification: bool) -> float:
    metric = score_predictions(y_true, scores, classification)
    return metric if classification else -metric


def mutual_prediction_disparity(model_a, model_b, X_eval: np.ndarray) -> float:
    pred_a = np.asarray(model_a.predict_score(X_eval), dtype=float)
    pred_b = np.asarray(model_b.predict_score(X_eval), dtype=float)
    return float(np.mean((pred_a - pred_b) ** 2))


def inverse_f1_disparity(model_a, model_b, X_eval: np.ndarray) -> float:
    pred_a = np.asarray(model_a.predict_label(X_eval), dtype=int)
    pred_b = np.asarray(model_b.predict_label(X_eval), dtype=int)
    tp = int(np.sum((pred_a == 1) & (pred_b == 1)))
    fp = int(np.sum((pred_a == 1) & (pred_b == 0)))
    fn = int(np.sum((pred_a == 0) & (pred_b == 1)))
    if tp == 0:
        return float(fp + fn + 1.0)
    return float((fp + fn) / (2.0 * tp))


def interpretability_score(leaf_info: pd.DataFrame, alpha: float = 0.5) -> float:
    if leaf_info.empty or leaf_info["num_observations"].sum() == 0:
        return float("nan")

    total = leaf_info["num_observations"].sum()
    weighted_depth = (
        leaf_info["depth"] * leaf_info["num_observations"]
    ).sum() / total
    weighted_complexity = (
        leaf_info["num_observations"] * (1 + leaf_info["k_l"])
    ).sum() / total
    return float(alpha * weighted_depth + (1 - alpha) * weighted_complexity)
