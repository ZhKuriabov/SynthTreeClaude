from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class _LeafSummary:
    leaf_id: int
    depth: int
    weight: float
    path_gains: np.ndarray
    coef_abs: np.ndarray
    leaf_score: np.ndarray


def _extract_coef_abs(model, n_features: int) -> np.ndarray:
    coef_abs = np.zeros(n_features, dtype=float)
    if model is None:
        return coef_abs
    inner = getattr(model, "model_", None)
    if inner is None or not hasattr(inner, "params"):
        return coef_abs
    params = np.asarray(inner.params, dtype=float)
    if params.size <= 1:
        return coef_abs
    col_mask = getattr(model, "_col_mask", None)
    if col_mask is not None:
        # Map reduced params back to full feature space.
        # col_mask[0] is intercept; col_mask[1:] are features.
        feature_mask = col_mask[1:]  # skip intercept position
        reduced_coef = np.abs(params[1:])  # skip intercept value
        j = 0
        for i in range(min(len(feature_mask), n_features)):
            if feature_mask[i] and j < len(reduced_coef):
                coef_abs[i] = reduced_coef[j]
                j += 1
    else:
        coef = np.abs(params[1 : 1 + n_features])
        coef_abs[: coef.size] = coef
    return coef_abs


def _enumerate_leaves(node, n_features: int, path_gains: np.ndarray | None = None, leaves=None):
    if leaves is None:
        leaves = []
    if path_gains is None:
        path_gains = np.zeros(n_features, dtype=float)
    if node.is_leaf:
        leaves.append(
            {
                "leaf_id": len(leaves),
                "node": node,
                "depth": int(node.depth),
                "path_gains": path_gains.copy(),
            }
        )
        return leaves

    next_gains = path_gains.copy()
    if node.feature_index is not None and node.gain is not None:
        next_gains[int(node.feature_index)] += max(float(node.gain), 0.0)
    _enumerate_leaves(node.left, n_features, next_gains, leaves)
    _enumerate_leaves(node.right, n_features, next_gains, leaves)
    return leaves


def _leaf_assignments(node, X: np.ndarray, leaf_lookup: dict[int, int]) -> np.ndarray:
    assignments = np.empty(X.shape[0], dtype=int)

    def visit(current, row_idx: np.ndarray):
        if row_idx.size == 0:
            return
        if current.is_leaf:
            assignments[row_idx] = leaf_lookup[id(current)]
            return
        left_mask = X[row_idx, current.feature_index] <= current.threshold
        visit(current.left, row_idx[left_mask])
        visit(current.right, row_idx[~left_mask])

    visit(node, np.arange(X.shape[0], dtype=int))
    return assignments


def synthtree_feature_importance(
    tree,
    X_original: np.ndarray,
    feature_names: list[str],
    split_weight: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_features = len(feature_names)
    leaves = _enumerate_leaves(tree.tree, n_features=n_features)
    if not leaves:
        empty = pd.DataFrame(columns=["feature", "importance", "importance_norm"])
        return empty, pd.DataFrame()

    leaf_lookup = {id(leaf["node"]): int(leaf["leaf_id"]) for leaf in leaves}
    assignments = _leaf_assignments(tree.tree, np.asarray(X_original, dtype=float), leaf_lookup)
    counts = np.bincount(assignments, minlength=len(leaves)).astype(float)
    total = float(max(np.sum(counts), 1.0))

    leaf_rows = []
    global_importance = np.zeros(n_features, dtype=float)
    for leaf in leaves:
        leaf_id = int(leaf["leaf_id"])
        weight = counts[leaf_id] / total
        path_gains = np.asarray(leaf["path_gains"], dtype=float)
        coef_abs = _extract_coef_abs(leaf["node"].model, n_features)
        gain_norm = path_gains / path_gains.sum() if path_gains.sum() > 0 else np.zeros(n_features, dtype=float)
        coef_norm = coef_abs / coef_abs.sum() if coef_abs.sum() > 0 else np.zeros(n_features, dtype=float)
        leaf_score = split_weight * gain_norm + (1.0 - split_weight) * coef_norm
        global_importance += weight * leaf_score
        leaf_rows.append(
            _LeafSummary(
                leaf_id=leaf_id,
                depth=int(leaf["depth"]),
                weight=float(weight),
                path_gains=path_gains,
                coef_abs=coef_abs,
                leaf_score=leaf_score,
            )
        )

    importance_norm = global_importance / global_importance.sum() if global_importance.sum() > 0 else global_importance
    global_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": global_importance,
            "importance_norm": importance_norm,
        }
    ).sort_values("importance_norm", ascending=False, ignore_index=True)

    leaf_df_rows = []
    for leaf in leaf_rows:
        for feature_idx, feature_name in enumerate(feature_names):
            leaf_df_rows.append(
                {
                    "leaf_id": leaf.leaf_id,
                    "depth": leaf.depth,
                    "leaf_weight": leaf.weight,
                    "feature": feature_name,
                    "path_gain": float(leaf.path_gains[feature_idx]),
                    "coef_abs": float(leaf.coef_abs[feature_idx]),
                    "leaf_score": float(leaf.leaf_score[feature_idx]),
                    "weighted_contribution": float(leaf.weight * leaf.leaf_score[feature_idx]),
                }
            )
    leaf_df = pd.DataFrame(leaf_df_rows)
    return global_df, leaf_df


def _binary_scores(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probs = np.asarray(model.predict_proba(X), dtype=float)
        if probs.ndim == 2:
            if probs.shape[1] == 1:
                return probs[:, 0]
            return probs[:, -1]
        return probs.reshape(-1)
    if hasattr(model, "predict_score"):
        return np.asarray(model.predict_score(X), dtype=float).reshape(-1)
    return np.asarray(model.predict(X), dtype=float).reshape(-1)


def feature_set_auc_drop(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_indices: np.ndarray | list[int],
    n_repeats: int = 30,
    random_state: int = 0,
    baseline_auc: float | None = None,
) -> tuple[float, np.ndarray]:
    from sklearn.metrics import roc_auc_score as _roc_auc

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    feature_indices = np.asarray(feature_indices, dtype=int)
    rng = np.random.default_rng(random_state)
    if baseline_auc is None:
        baseline_auc = float(_roc_auc(y, _binary_scores(model, X)))

    drops = np.zeros(int(n_repeats), dtype=float)
    for repeat_idx in range(int(n_repeats)):
        X_perm = X.copy()
        for feature_idx in feature_indices:
            X_perm[:, feature_idx] = X[rng.permutation(X.shape[0]), feature_idx]
        perm_auc = float(_roc_auc(y, _binary_scores(model, X_perm)))
        drops[repeat_idx] = baseline_auc - perm_auc
    return float(np.mean(drops)), drops


def feature_set_permutation_test(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_indices: np.ndarray | list[int],
    n_repeats: int = 30,
    n_null_draws: int = 500,
    random_state: int = 0,
) -> tuple[dict[str, float], pd.DataFrame]:
    from sklearn.metrics import roc_auc_score as _roc_auc

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    feature_indices = np.asarray(feature_indices, dtype=int)
    rng = np.random.default_rng(random_state)
    baseline_auc = float(_roc_auc(y, _binary_scores(model, X)))

    observed_mean_drop, observed_drops = feature_set_auc_drop(
        model, X, y, feature_indices,
        n_repeats=n_repeats,
        random_state=int(rng.integers(0, 2**32 - 1)),
        baseline_auc=baseline_auc,
    )

    null_rows = []
    null_stats = np.zeros(int(n_null_draws), dtype=float)
    set_size = int(len(feature_indices))
    for draw_idx in range(int(n_null_draws)):
        sampled = np.sort(rng.choice(X.shape[1], size=set_size, replace=False)).astype(int)
        null_mean_drop, _ = feature_set_auc_drop(
            model, X, y, sampled,
            n_repeats=n_repeats,
            random_state=int(rng.integers(0, 2**32 - 1)),
            baseline_auc=baseline_auc,
        )
        null_stats[draw_idx] = null_mean_drop
        null_rows.append({
            "draw": draw_idx,
            "mean_auc_drop": float(null_mean_drop),
            "feature_indices": ",".join(str(int(idx)) for idx in sampled),
        })

    null_mean = float(np.mean(null_stats))
    null_sd = float(np.std(null_stats, ddof=1)) if len(null_stats) > 1 else 0.0
    p_value = float((1.0 + np.sum(null_stats >= observed_mean_drop)) / (1.0 + len(null_stats)))

    summary = {
        "baseline_auc": baseline_auc,
        "observed_mean_auc_drop": float(observed_mean_drop),
        "null_mean_auc_drop": null_mean,
        "null_sd_auc_drop": null_sd,
        "empirical_p_value": p_value,
    }
    return summary, pd.DataFrame(null_rows)
