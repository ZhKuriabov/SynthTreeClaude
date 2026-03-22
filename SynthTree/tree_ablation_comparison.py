from __future__ import annotations

import argparse
import copy
import warnings
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from example_preprocessing import prep_data
from synthtree.co_supervision import build_co_supervision, default_teacher, select_num_cells
from synthtree.metrics import (
    inverse_f1_disparity,
    mutual_prediction_disparity,
    score_predictions,
)
from synthtree.models import make_statsmodels_model
from synthtree.pruning import select_tree_size, _select_subtree_for_alpha
from synthtree.tree import DistanceDecisionTree

warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


CLASS_DATASETS = {"SKCM", "Road Safety", "Compas", "Upselling"}
REG_DATASETS = {"Cal Housing", "Bike Sharing", "Abalone", "Servo"}
DEFAULT_DATASETS = ["SKCM", "Compas", "Bike Sharing", "Abalone"]
DEFAULT_N_CELLS_GRID = (5, 10, 15, 20)
MAX_DEPTH = 4
PRUNING_CV = 10

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")


def task_is_classification(dataset_name: str) -> bool:
    return dataset_name in CLASS_DATASETS


def _splitter(classification: bool, cv: int, random_state: int):
    if classification:
        return StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    return KFold(n_splits=cv, shuffle=True, random_state=random_state)


def _teacher_spec(classification: bool):
    return {"RF": default_teacher(classification)}


def _prediction_distance_matrix(cell_models, classification: bool) -> np.ndarray:
    n_cells = len(cell_models)
    distance_matrix = np.zeros((n_cells, n_cells), dtype=float)
    for i in range(n_cells):
        for j in range(i + 1, n_cells):
            X_eval = np.vstack([cell_models[i].X_augmented, cell_models[j].X_augmented])
            if classification:
                dist = inverse_f1_disparity(cell_models[i].model, cell_models[j].model, X_eval)
            else:
                dist = mutual_prediction_disparity(cell_models[i].model, cell_models[j].model, X_eval)
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    return distance_matrix


def _evaluate_model_loss(model, X: np.ndarray, y: np.ndarray, classification: bool) -> float:
    if X.shape[0] == 0:
        return 0.0
    scores = np.asarray(model.predict_score(X), dtype=float)
    if classification:
        probs = np.clip(scores, 1e-6, 1 - 1e-6)
        return float(
            -np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs))
        )
    return float(np.sum((np.asarray(y, dtype=float) - scores) ** 2))


def _combine_cells(cell_models, cell_indices: np.ndarray):
    X = np.vstack([cell_models[int(i)].X_augmented for i in cell_indices])
    y = np.concatenate([cell_models[int(i)].y_augmented for i in cell_indices])
    return X, y


def _resubstitution_error(model, X: np.ndarray, y: np.ndarray, classification: bool) -> float:
    if X.shape[0] == 0:
        return 0.0
    if classification:
        predictions = np.asarray(model.predict_label(X), dtype=int)
        return float(np.sum(predictions != np.asarray(y, dtype=int)))
    residuals = np.asarray(y, dtype=float) - np.asarray(model.predict_score(X), dtype=float)
    return float(np.sum(residuals ** 2))


@dataclass
class RSSNode:
    cell_indices: np.ndarray
    depth: int
    feature_index: int | None = None
    threshold: float | None = None
    left: "RSSNode | None" = None
    right: "RSSNode | None" = None
    model: object | None = None
    node_error: float = 0.0
    subtree_error: float = 0.0
    subtree_leaves: int = 1
    prune_alpha: float | None = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class RSSDecisionTree:
    def __init__(self, max_depth: int | None, classification: bool, model_factory):
        self.max_depth = max_depth
        self.classification = classification
        self.model_factory = model_factory
        self.tree = None

    def fit(self, centroids: np.ndarray, cell_models):
        cell_indices = np.arange(centroids.shape[0])
        self.tree = self._build(centroids, cell_models, cell_indices, depth=0)
        return self

    def _split_gain(self, cell_models, parent_indices, left_indices, right_indices) -> float:
        X_parent, y_parent = _combine_cells(cell_models, parent_indices)
        X_left, y_left = _combine_cells(cell_models, left_indices)
        X_right, y_right = _combine_cells(cell_models, right_indices)

        parent_model = self.model_factory()
        parent_model.fit(X_parent, y_parent)
        left_model = self.model_factory()
        left_model.fit(X_left, y_left)
        right_model = self.model_factory()
        right_model.fit(X_right, y_right)

        parent_loss = _evaluate_model_loss(parent_model, X_parent, y_parent, self.classification)
        child_loss = _evaluate_model_loss(left_model, X_left, y_left, self.classification)
        child_loss += _evaluate_model_loss(right_model, X_right, y_right, self.classification)
        return parent_loss - child_loss

    def _build(self, centroids: np.ndarray, cell_models, cell_indices: np.ndarray, depth: int):
        node = RSSNode(cell_indices=np.asarray(cell_indices, dtype=int), depth=depth)
        if self.max_depth is not None and depth >= self.max_depth:
            return node
        if len(cell_indices) <= 1:
            return node

        best = None
        best_gain = 0.0
        centroid_subset = centroids[cell_indices]

        for feature_index in range(centroid_subset.shape[1]):
            for threshold in np.unique(centroid_subset[:, feature_index]):
                left_mask = centroid_subset[:, feature_index] <= threshold
                right_mask = ~left_mask
                if not np.any(left_mask) or not np.any(right_mask):
                    continue
                left_indices = cell_indices[left_mask]
                right_indices = cell_indices[right_mask]
                gain = self._split_gain(cell_models, cell_indices, left_indices, right_indices)
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best = (feature_index, float(threshold), left_indices, right_indices)

        if best is None:
            return node

        node.feature_index = best[0]
        node.threshold = best[1]
        node.left = self._build(centroids, cell_models, best[2], depth + 1)
        node.right = self._build(centroids, cell_models, best[3], depth + 1)
        return node

    def fit_leaf_models(self, X: np.ndarray, y: np.ndarray):
        self._fit_leaf_models(self.tree, X, y)
        return self

    def fit_node_models(self, X: np.ndarray, y: np.ndarray):
        self._fit_node_models(self.tree, X, y)
        return self

    def _fit_leaf_models(self, node: RSSNode, X: np.ndarray, y: np.ndarray):
        if node.is_leaf:
            node.model = self.model_factory()
            node.model.fit(X, y)
            return
        left_mask = X[:, node.feature_index] <= node.threshold
        right_mask = ~left_mask
        self._fit_leaf_models(node.left, X[left_mask], y[left_mask])
        self._fit_leaf_models(node.right, X[right_mask], y[right_mask])

    def _fit_node_models(self, node: RSSNode, X: np.ndarray, y: np.ndarray):
        node.model = self.model_factory()
        node.model.fit(X, y)
        if node.is_leaf:
            return
        left_mask = X[:, node.feature_index] <= node.threshold
        right_mask = ~left_mask
        self._fit_node_models(node.left, X[left_mask], y[left_mask])
        self._fit_node_models(node.right, X[right_mask], y[right_mask])

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        return np.asarray([self._predict_one(x, self.tree) for x in X], dtype=float)

    def _predict_one(self, x: np.ndarray, node: RSSNode) -> float:
        if node.is_leaf:
            return float(node.model.predict_score(x.reshape(1, -1))[0])
        if x[node.feature_index] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def leaf_info(self, X: np.ndarray) -> pd.DataFrame:
        rows = []

        def visit(node: RSSNode, X_node: np.ndarray):
            if node.is_leaf:
                rows.append(
                    {
                        "depth": node.depth,
                        "num_observations": int(len(X_node)),
                        "k_l": int(node.model.complexity() if node.model is not None else 0),
                    }
                )
                return
            left_mask = X_node[:, node.feature_index] <= node.threshold
            right_mask = ~left_mask
            visit(node.left, X_node[left_mask])
            visit(node.right, X_node[right_mask])

        visit(self.tree, X)
        return pd.DataFrame(rows)

    def count_leaves(self) -> int:
        def count(node: RSSNode) -> int:
            if node.is_leaf:
                return 1
            return count(node.left) + count(node.right)

        return count(self.tree)

    def copy(self):
        copied = RSSDecisionTree(self.max_depth, self.classification, self.model_factory)
        copied.tree = copy.deepcopy(self.tree)
        return copied

    def weakest_link_pruning_path(self, X: np.ndarray, y: np.ndarray):
        working_tree = self.copy()
        path = [
            {
                "alpha": 0.0,
                "tree": working_tree.copy(),
                "leaves": working_tree.count_leaves(),
                "depth": working_tree.max_depth_of_tree(),
            }
        ]
        previous_alpha = 0.0
        while not working_tree.tree.is_leaf:
            candidates = []
            working_tree._annotate_pruning_stats(working_tree.tree, np.asarray(X), np.asarray(y), candidates)
            if not candidates:
                break
            prune_scores = np.asarray([candidate.prune_alpha for candidate in candidates], dtype=float)
            eligible = prune_scores[prune_scores >= previous_alpha - 1e-12]
            current_alpha = float(np.min(eligible if eligible.size else prune_scores))
            leaves_before = working_tree.count_leaves()
            working_tree._prune_by_alpha(working_tree.tree, current_alpha)
            leaves_after = working_tree.count_leaves()
            if leaves_after >= leaves_before:
                break
            previous_alpha = current_alpha
            path.append(
                {
                    "alpha": current_alpha,
                    "tree": working_tree.copy(),
                    "leaves": leaves_after,
                    "depth": working_tree.max_depth_of_tree(),
                }
            )
        return path

    def _annotate_pruning_stats(self, node: RSSNode, X: np.ndarray, y: np.ndarray, candidates: list[RSSNode]):
        node.node_error = _resubstitution_error(node.model, X, y, self.classification)
        if node.is_leaf:
            node.subtree_error = node.node_error
            node.subtree_leaves = 1
            node.prune_alpha = None
            return node.subtree_error, node.subtree_leaves
        left_mask = X[:, node.feature_index] <= node.threshold
        right_mask = ~left_mask
        left_error, left_leaves = self._annotate_pruning_stats(node.left, X[left_mask], y[left_mask], candidates)
        right_error, right_leaves = self._annotate_pruning_stats(node.right, X[right_mask], y[right_mask], candidates)
        node.subtree_error = left_error + right_error
        node.subtree_leaves = left_leaves + right_leaves
        denominator = max(node.subtree_leaves - 1, 1)
        node.prune_alpha = float((node.node_error - node.subtree_error) / denominator)
        candidates.append(node)
        return node.subtree_error, node.subtree_leaves

    def _prune_by_alpha(self, node: RSSNode, alpha: float):
        if node.is_leaf:
            return
        self._prune_by_alpha(node.left, alpha)
        self._prune_by_alpha(node.right, alpha)
        if node.prune_alpha is not None and node.prune_alpha <= alpha + 1e-12:
            node.left = None
            node.right = None
            node.feature_index = None
            node.threshold = None

    def max_depth_of_tree(self) -> int:
        def visit(node: RSSNode) -> int:
            if node.is_leaf:
                return node.depth
            return max(visit(node.left), visit(node.right))

        return visit(self.tree)


class StatsmodelsCartLeafRefitModel:
    def __init__(self, classification: bool, alpha: float, max_iter: int, ccp_alpha: float = 0.0):
        self.classification = classification
        self.alpha = alpha
        self.max_iter = max_iter
        self.ccp_alpha = ccp_alpha
        self.structure = None
        self.leaf_models = {}
        self.leaf_depths = {}

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.classification:
            self.structure = DecisionTreeClassifier(random_state=0, ccp_alpha=self.ccp_alpha)
        else:
            self.structure = DecisionTreeRegressor(random_state=0, ccp_alpha=self.ccp_alpha)
        self.structure.fit(X, y)
        leaf_ids = self.structure.apply(X)
        for leaf_id in np.unique(leaf_ids):
            idx = leaf_ids == leaf_id
            model = make_statsmodels_model(
                classification=self.classification,
                alpha=self.alpha,
                max_iter=self.max_iter,
            )
            model.fit(X[idx], y[idx])
            self.leaf_models[int(leaf_id)] = model
        self.leaf_depths = self._compute_leaf_depths()
        return self

    def _compute_leaf_depths(self):
        tree = self.structure.tree_
        depths = {}

        def recurse(node_id, depth):
            left = tree.children_left[node_id]
            right = tree.children_right[node_id]
            if left == -1 and right == -1:
                depths[node_id] = depth
                return
            recurse(left, depth + 1)
            recurse(right, depth + 1)

        recurse(0, 0)
        return depths

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        leaf_ids = self.structure.apply(X)
        scores = np.zeros(X.shape[0], dtype=float)
        for idx, leaf_id in enumerate(leaf_ids):
            model = self.leaf_models[int(leaf_id)]
            scores[idx] = float(model.predict_score(X[idx : idx + 1])[0])
        return scores

    def leaf_info(self, X: np.ndarray) -> pd.DataFrame:
        leaf_ids = self.structure.apply(X)
        rows = []
        for leaf_id, model in self.leaf_models.items():
            rows.append(
                {
                    "depth": self.leaf_depths.get(int(leaf_id), 0),
                    "num_observations": int(np.sum(leaf_ids == leaf_id)),
                    "k_l": int(model.complexity()),
                }
            )
        return pd.DataFrame(rows)

    def count_leaves(self) -> int:
        return int(len(self.leaf_models))


def interpretability_from_leaf_info(leaf_info: pd.DataFrame) -> float:
    if leaf_info.empty or leaf_info["num_observations"].sum() == 0:
        return float("nan")
    total = float(leaf_info["num_observations"].sum())
    weighted_depth = float(np.sum(leaf_info["depth"] * leaf_info["num_observations"]) / total)
    weighted_complexity = float(np.sum((1 + leaf_info["k_l"]) * leaf_info["num_observations"]) / total)
    return 0.5 * weighted_depth + 0.5 * weighted_complexity


def _validation_objective(y_true, scores, classification: bool) -> float:
    metric = score_predictions(y_true, scores, classification)
    return metric if classification else -metric


@dataclass
class DraftCellTreeModel:
    criterion: str
    classification: bool
    n_cells_grid: tuple[int, ...]
    num_aug: int
    local_model_alpha: float
    leaf_model_alpha: float
    local_model_max_iter: int
    leaf_model_max_iter: int
    random_state: int
    max_depth: int
    tree_sizer: str

    def fit(self, X: np.ndarray, y: np.ndarray):
        selected_num_cells, selection_info = select_num_cells(
            X=X,
            y=y,
            teachers=_teacher_spec(self.classification),
            n_cells_grid=self.n_cells_grid,
            classification=self.classification,
            selection="validation_mlm",
            num_aug=self.num_aug,
            covariance_type="diag",
            local_model_alpha=self.local_model_alpha,
            local_model_max_iter=self.local_model_max_iter,
            random_state=self.random_state,
        )
        self.selected_num_cells_ = int(selected_num_cells)
        self.n_cells_selection_info_ = selection_info

        fit_depth_fn = lambda depth, X_sub, y_sub: self._fit_depth(depth, X_sub, y_sub)
        size_info = select_tree_size(
            X=X,
            y=y,
            classification=self.classification,
            max_depth=self.max_depth,
            method=self.tree_sizer,
            fit_depth_fn=fit_depth_fn,
            fit_full_tree_fn=lambda X_sub, y_sub: self._fit_full_tree(X_sub, y_sub),
            cv=PRUNING_CV,
            random_state=self.random_state,
        )
        self.selected_alpha_ = size_info["alpha"]
        self.tree_size_info_ = size_info
        if self.tree_sizer == "cc_prune":
            fitted = self._fit_full_tree(X, y)
            self.tree_ = _select_subtree_for_alpha(
                fitted.tree_.weakest_link_pruning_path(fitted.X_fit_, fitted.y_fit_),
                float(self.selected_alpha_),
            )
            self.selected_depth_ = int(
                self.tree_.final_depth if hasattr(self.tree_, "final_depth") else self.tree_.max_depth_of_tree()
            )
            self.leaf_info_ = self.tree_.leaf_info(X)
            self.interpretability_ = interpretability_from_leaf_info(self.leaf_info_)
            return self

        self.selected_depth_ = int(size_info["depth"])
        fitted = self._fit_full_depth(self.selected_depth_, X, y, self.selected_num_cells_)
        self.tree_ = fitted.tree_
        self.leaf_info_ = fitted.leaf_info_
        self.interpretability_ = fitted.interpretability_
        return self

    def _fit_depth(self, depth: int, X: np.ndarray, y: np.ndarray):
        selected_num_cells, _ = select_num_cells(
            X=X,
            y=y,
            teachers=_teacher_spec(self.classification),
            n_cells_grid=self.n_cells_grid,
            classification=self.classification,
            selection="validation_mlm",
            num_aug=self.num_aug,
            covariance_type="diag",
            local_model_alpha=self.local_model_alpha,
            local_model_max_iter=self.local_model_max_iter,
            random_state=self.random_state,
        )
        return self._fit_full_depth(depth, X, y, int(selected_num_cells))

    def _fit_full_tree(self, X: np.ndarray, y: np.ndarray):
        selected_num_cells, _ = select_num_cells(
            X=X,
            y=y,
            teachers=_teacher_spec(self.classification),
            n_cells_grid=self.n_cells_grid,
            classification=self.classification,
            selection="validation_mlm",
            num_aug=self.num_aug,
            covariance_type="diag",
            local_model_alpha=self.local_model_alpha,
            local_model_max_iter=self.local_model_max_iter,
            random_state=self.random_state,
        )
        fitted = self._fit_full_depth(self.max_depth, X, y, int(selected_num_cells), node_models=True)
        fitted.X_fit_ = np.asarray(fitted.X_fit_)
        fitted.y_fit_ = np.asarray(fitted.y_fit_)
        return fitted

    def _fit_full_depth(self, depth: int, X: np.ndarray, y: np.ndarray, selected_num_cells: int, node_models: bool = False):
        cosup = build_co_supervision(
            X=X,
            y=y,
            teachers=_teacher_spec(self.classification),
            n_cells=selected_num_cells,
            classification=self.classification,
            num_aug=self.num_aug,
            covariance_type="diag",
            local_model_alpha=self.local_model_alpha,
            local_model_max_iter=self.local_model_max_iter,
            random_state=self.random_state,
        )
        X_augmented, y_augmented = cosup.augmented_training_data
        model_factory = lambda: make_statsmodels_model(
            classification=self.classification,
            alpha=self.leaf_model_alpha,
            max_iter=self.leaf_model_max_iter,
        )

        if self.criterion == "prediction_disparity":
            tree = DistanceDecisionTree(max_depth=depth, classification=self.classification)
            tree.fit(cosup.centroids, _prediction_distance_matrix(cosup.cell_models, self.classification))
            if node_models:
                tree.fit_node_models(X_augmented, y_augmented, model_factory)
            else:
                tree.fit_leaf_models(X_augmented, y_augmented, model_factory)
            leaf_info = tree.leaf_info(X)
        elif self.criterion == "rss":
            tree = RSSDecisionTree(
                max_depth=depth,
                classification=self.classification,
                model_factory=model_factory,
            )
            tree.fit(cosup.centroids, cosup.cell_models)
            if node_models:
                tree.fit_node_models(X_augmented, y_augmented)
            else:
                tree.fit_leaf_models(X_augmented, y_augmented)
            leaf_info = tree.leaf_info(X)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")

        fitted = _TreeWrapper(tree)
        fitted.leaf_info_ = leaf_info
        fitted.interpretability_ = interpretability_from_leaf_info(leaf_info)
        fitted.X_fit_ = X_augmented
        fitted.y_fit_ = y_augmented
        return fitted

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        return self.tree_.predict_scores(X)


class _TreeWrapper:
    def __init__(self, tree):
        self.tree_ = tree
        self.leaf_info_ = None
        self.interpretability_ = np.nan

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        return self.tree_.predict_scores(X)


def _cart_alpha_candidates(X: np.ndarray, y: np.ndarray, classification: bool) -> np.ndarray:
    if classification:
        base_tree = DecisionTreeClassifier(random_state=0)
    else:
        base_tree = DecisionTreeRegressor(random_state=0)
    path = base_tree.cost_complexity_pruning_path(X, y)
    alphas = np.unique(path.ccp_alphas)
    if alphas.size == 0:
        return np.array([0.0])
    if alphas.size > 25:
        keep_idx = np.linspace(0, alphas.size - 1, 25, dtype=int)
        alphas = alphas[keep_idx]
    return np.unique(np.append(alphas, 0.0))


def select_cart_alpha(
    X: np.ndarray,
    y: np.ndarray,
    classification: bool,
    leaf_model_alpha: float,
    leaf_model_max_iter: int,
    random_state: int,
    cv: int = 10,
):
    alpha_grid = _cart_alpha_candidates(X, y, classification)
    splitter = _splitter(classification, cv=cv, random_state=random_state)
    rows = []

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        for alpha in alpha_grid:
            model = StatsmodelsCartLeafRefitModel(
                classification=classification,
                alpha=leaf_model_alpha,
                max_iter=leaf_model_max_iter,
                ccp_alpha=float(alpha),
            )
            model.fit(X_train, y_train)
            scores = model.predict_scores(X_val)
            rows.append(
                {
                    "fold": fold_idx,
                    "alpha": float(alpha),
                    "metric": score_predictions(y_val, scores, classification),
                    "leaves": model.count_leaves(),
                }
            )

    summary = pd.DataFrame(rows).groupby("alpha", as_index=False).agg(
        metric_mean=("metric", "mean"),
        metric_std=("metric", "std"),
        leaves_mean=("leaves", "mean"),
    )
    summary["metric_std"] = summary["metric_std"].fillna(0.0)
    summary["metric_se"] = summary["metric_std"] / np.sqrt(cv)

    if classification:
        best_idx = int(summary["metric_mean"].idxmax())
        best_mean = float(summary.loc[best_idx, "metric_mean"])
        best_se = float(summary.loc[best_idx, "metric_se"])
        eligible = summary[summary["metric_mean"] >= best_mean - best_se].copy()
        eligible = eligible.sort_values(["leaves_mean", "alpha"], ascending=[True, False])
        best_alpha = float(summary.loc[best_idx, "alpha"])
        one_se_alpha = float(eligible.iloc[0]["alpha"])
    else:
        best_idx = int(summary["metric_mean"].idxmin())
        best_mean = float(summary.loc[best_idx, "metric_mean"])
        best_se = float(summary.loc[best_idx, "metric_se"])
        eligible = summary[summary["metric_mean"] <= best_mean + best_se].copy()
        eligible = eligible.sort_values(["leaves_mean", "alpha"], ascending=[True, False])
        best_alpha = float(summary.loc[best_idx, "alpha"])
        one_se_alpha = float(eligible.iloc[0]["alpha"])

    return {
        "best": best_alpha,
        "one_se": one_se_alpha,
        "summary": summary,
    }


def fit_cart_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    classification: bool,
    leaf_model_alpha: float,
    leaf_model_max_iter: int,
    ccp_alpha: float,
):
    model = StatsmodelsCartLeafRefitModel(
        classification=classification,
        alpha=leaf_model_alpha,
        max_iter=leaf_model_max_iter,
        ccp_alpha=ccp_alpha,
    )
    model.fit(X_train, y_train)
    return model


def run_one(
    dataset_name: str,
    seed: int,
    n_cells_grid: tuple[int, ...],
    num_aug: int,
    local_model_alpha: float,
    leaf_model_alpha: float,
    local_model_max_iter: int,
    leaf_model_max_iter: int,
):
    classification = task_is_classification(dataset_name)
    X_train, y_train, X_test, y_test = prep_data(dataset_name, random_state=seed)
    if classification:
        classes = np.sort(np.unique(y_train))
        if classes.size != 2:
            raise ValueError("Classification ablation currently supports binary outcomes only.")
        y_train = (np.asarray(y_train) == classes[1]).astype(int)
        y_test = (np.asarray(y_test) == classes[1]).astype(int)
    tree_sizer = "cc_prune" if classification else "l_trim"

    synthtree = DraftCellTreeModel(
        criterion="prediction_disparity",
        classification=classification,
        n_cells_grid=n_cells_grid,
        num_aug=num_aug,
        local_model_alpha=local_model_alpha,
        leaf_model_alpha=leaf_model_alpha,
        local_model_max_iter=local_model_max_iter,
        leaf_model_max_iter=leaf_model_max_iter,
        random_state=seed,
        max_depth=MAX_DEPTH,
        tree_sizer=tree_sizer,
    )
    synthtree.fit(X_train, y_train)

    synthtree_rss = DraftCellTreeModel(
        criterion="rss",
        classification=classification,
        n_cells_grid=n_cells_grid,
        num_aug=num_aug,
        local_model_alpha=local_model_alpha,
        leaf_model_alpha=leaf_model_alpha,
        local_model_max_iter=local_model_max_iter,
        leaf_model_max_iter=leaf_model_max_iter,
        random_state=seed,
        max_depth=MAX_DEPTH,
        tree_sizer=tree_sizer,
    )
    synthtree_rss.fit(X_train, y_train)

    models = {
        "SynthTree": synthtree,
        "SynthTree-RSS": synthtree_rss,
    }

    rows = []
    for model_name, model in models.items():
        scores = model.predict_scores(X_test)
        metric = score_predictions(y_test, scores, classification)
        if hasattr(model, "leaf_info_"):
            leaf_info = model.leaf_info_
        else:
            leaf_info = model.leaf_info(X_train)
        row = {
            "dataset": dataset_name,
            "task": "classification" if classification else "regression",
            "teacher": "RF",
            "model": model_name,
            "seed": seed,
            "score": metric,
            "interpretability": interpretability_from_leaf_info(leaf_info),
            "num_leaves": int(len(leaf_info)),
            "mean_leaf_depth": float(
                np.average(leaf_info["depth"], weights=leaf_info["num_observations"])
            ) if leaf_info["num_observations"].sum() > 0 else np.nan,
        }
        row["selected_num_cells"] = int(model.selected_num_cells_)
        row["selected_depth"] = int(model.selected_depth_)
        row["selected_alpha"] = model.selected_alpha_
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Focused reviewer ablation for SynthTree vs classical comparators")
    parser.add_argument("--runs", "-r", type=int, default=3, help="Number of successful repeated splits per dataset")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=DEFAULT_DATASETS,
        choices=sorted(CLASS_DATASETS | REG_DATASETS),
        help="Datasets to run",
    )
    parser.add_argument(
        "--n-cells-grid",
        nargs="*",
        type=int,
        default=list(DEFAULT_N_CELLS_GRID),
        help="Candidate values of J for validation-based selection",
    )
    parser.add_argument("--num-aug", type=int, default=100, help="Synthetic points per cell (default=100)")
    parser.add_argument("--local-model-alpha", type=float, default=0.1, help="L1 regularization for cell-wise models")
    parser.add_argument("--leaf-model-alpha", type=float, default=0.1, help="L1 regularization for final leaf models")
    parser.add_argument("--local-model-max-iter", type=int, default=1000, help="Max iterations for cell-wise statsmodels fits")
    parser.add_argument("--leaf-model-max-iter", type=int, default=1000, help="Max iterations for final leaf statsmodels fits")
    args = parser.parse_args()

    n_cells_grid = tuple(sorted(set(int(v) for v in args.n_cells_grid if int(v) > 1)))
    if not n_cells_grid:
        raise ValueError("n_cells_grid must contain at least one integer greater than 1.")

    all_rows = []
    dataset_iterator = args.datasets
    if tqdm is not None:
        dataset_iterator = tqdm(args.datasets, desc="Ablation datasets", unit="dataset")

    for dataset_name in dataset_iterator:
        successes = 0
        trial = 0
        run_bar = None
        if tqdm is not None:
            run_bar = tqdm(total=args.runs, desc=f"{dataset_name} runs", unit="run", leave=False)

        while successes < args.runs and trial < args.runs * 10:
            seed = successes * 100 + trial
            try:
                rows = run_one(
                    dataset_name=dataset_name,
                    seed=seed,
                    n_cells_grid=n_cells_grid,
                    num_aug=args.num_aug,
                    local_model_alpha=args.local_model_alpha,
                    leaf_model_alpha=args.leaf_model_alpha,
                    local_model_max_iter=args.local_model_max_iter,
                    leaf_model_max_iter=args.leaf_model_max_iter,
                )
                all_rows.extend(rows)
                successes += 1
                if run_bar is not None:
                    run_bar.update(1)
                    run_bar.set_postfix(seed=seed)
                else:
                    print(f"{dataset_name}: completed run {successes}/{args.runs} with seed={seed}")
            except Exception as exc:
                if run_bar is not None:
                    run_bar.write(f"{dataset_name}: failed seed={seed} with error: {exc}")
                else:
                    print(f"{dataset_name}: failed seed={seed} with error: {exc}")
            trial += 1
        if run_bar is not None:
            run_bar.close()
        if successes < args.runs:
            raise RuntimeError(f"Unable to obtain {args.runs} successful runs for {dataset_name}")

    results = pd.DataFrame(all_rows)
    results.to_csv("tree_ablation_results.csv", index=False)

    summary = (
        results.groupby(["dataset", "task", "teacher", "model"])
        .agg(
            score_mean=("score", "mean"),
            score_std=("score", "std"),
            interpretability_mean=("interpretability", "mean"),
            interpretability_std=("interpretability", "std"),
            selected_num_cells_mode=("selected_num_cells", lambda s: int(pd.Series.mode(s.dropna()).iloc[0]) if s.dropna().size else np.nan),
            selected_depth_mode=("selected_depth", lambda s: int(pd.Series.mode(s).iloc[0])),
            selected_alpha_mode=("selected_alpha", lambda s: float(pd.Series.mode(s.dropna()).iloc[0]) if s.dropna().size else np.nan),
            num_leaves_mean=("num_leaves", "mean"),
            mean_leaf_depth_mean=("mean_leaf_depth", "mean"),
        )
        .reset_index()
    )
    summary.to_csv("tree_ablation_summary.csv", index=False)
    print(summary)


if __name__ == "__main__":
    main()
