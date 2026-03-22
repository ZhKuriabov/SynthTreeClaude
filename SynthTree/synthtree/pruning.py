from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from .metrics import score_predictions


def _splitter(classification: bool, cv: int, random_state: int):
    if classification:
        return StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    return KFold(n_splits=cv, shuffle=True, random_state=random_state)


def _pruning_validation_error(y_true, scores, classification: bool) -> float:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    if classification:
        labels = (scores >= 0.5).astype(int)
        return float(np.sum(labels != y_true.astype(int)))
    residuals = y_true.astype(float) - scores
    return float(np.sum(residuals ** 2))


def _select_subtree_for_alpha(path, alpha: float, min_leaves: int = 1):
    selected = None
    for entry in path:
        if entry["alpha"] > alpha + 1e-12:
            break
        if entry["leaves"] >= min_leaves:
            selected = entry["tree"].copy()
    if selected is None:
        selected = path[0]["tree"].copy()
    return selected


def select_tree_size(
    X,
    y,
    classification: bool,
    max_depth: int,
    method: str,
    fit_depth_fn=None,
    fit_full_tree_fn=None,
    cv: int = 10,
    random_state: int = 0,
    alpha_grid=None,
    min_leaves: int = 1,
):
    splitter = _splitter(classification, cv=cv, random_state=random_state)
    depth_grid = list(range(1, max_depth + 1))
    if method == "l_trim":
        if fit_depth_fn is None:
            raise ValueError("l_trim requires fit_depth_fn")
        fold_rows = []
        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            for depth in depth_grid:
                model = fit_depth_fn(depth, X_train, y_train)
                leaves = model.tree_.count_leaves()
                if leaves < min_leaves:
                    continue
                scores = model.predict_scores(X_val)
                metric = score_predictions(y_val, scores, classification)
                fold_rows.append(
                    {
                        "fold": fold_idx,
                        "depth": depth,
                        "metric": metric,
                        "leaves": leaves,
                    }
                )
        if not fold_rows:
            for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]
                for depth in depth_grid:
                    model = fit_depth_fn(depth, X_train, y_train)
                    scores = model.predict_scores(X_val)
                    metric = score_predictions(y_val, scores, classification)
                    fold_rows.append(
                        {
                            "fold": fold_idx,
                            "depth": depth,
                            "metric": metric,
                            "leaves": model.tree_.count_leaves(),
                        }
                    )
        best_depth = None
        best_score = None
        for depth in depth_grid:
            metrics = [row["metric"] for row in fold_rows if row["depth"] == depth]
            if not metrics:
                continue
            mean_metric = float(np.mean(metrics))
            if best_depth is None:
                best_depth = depth
                best_score = mean_metric
                continue
            if classification and mean_metric > best_score:
                best_depth = depth
                best_score = mean_metric
            if (not classification) and mean_metric < best_score:
                best_depth = depth
                best_score = mean_metric
        return {"method": "l_trim", "depth": best_depth, "alpha": None, "fold_rows": fold_rows}

    if fit_full_tree_fn is None:
        raise ValueError("cc_prune requires fit_full_tree_fn")

    fold_models = []
    alpha_candidates = {0.0}
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        model = fit_full_tree_fn(X_train, y_train)
        path = model.tree_.weakest_link_pruning_path(X_train, y_train)
        for entry in path:
            alpha_candidates.add(float(entry["alpha"]))
        fold_models.append(
            {
                "fold": fold_idx,
                "X_val": X_val,
                "y_val": y_val,
                "path": path,
            }
        )

    if alpha_grid is None:
        alpha_grid = sorted(alpha_candidates)
    else:
        alpha_grid = sorted(set(float(alpha) for alpha in alpha_grid).union(alpha_candidates))

    cc_rows = []
    for fold_info in fold_models:
        for alpha in alpha_grid:
            subtree = _select_subtree_for_alpha(fold_info["path"], alpha, min_leaves=min_leaves)
            scores = subtree.predict_scores(fold_info["X_val"])
            error = _pruning_validation_error(fold_info["y_val"], scores, classification)
            cc_rows.append(
                {
                    "fold": fold_info["fold"],
                    "alpha": float(alpha),
                    "error": error,
                    "depth": subtree.final_depth,
                    "leaves": subtree.count_leaves(),
                }
            )

    best = None
    for alpha in alpha_grid:
        rows = [row for row in cc_rows if abs(row["alpha"] - alpha) <= 1e-12]
        if not rows:
            continue
        mean_error = float(np.mean([row["error"] for row in rows]))
        mean_leaves = float(np.mean([row["leaves"] for row in rows]))
        mean_depth = float(np.mean([row["depth"] for row in rows]))
        candidate = {
            "method": "cc_prune",
            "depth": int(round(mean_depth)),
            "alpha": float(alpha),
            "objective": -mean_error,
            "fold_rows": cc_rows,
            "mean_error": mean_error,
            "mean_leaves": mean_leaves,
        }
        if best is None:
            best = candidate
            continue
        if mean_error < best["mean_error"] - 1e-12:
            best = candidate
            continue
        if abs(mean_error - best["mean_error"]) <= 1e-12 and mean_leaves < best["mean_leaves"] - 1e-12:
            best = candidate

    if best is None:
        raise ValueError("No valid tree-size candidate found.")

    return best
