from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import interpretability_score


@dataclass
class TreeNode:
    cell_indices: np.ndarray
    depth: int
    impurity: float = 0.0
    gain: float = 0.0
    feature_index: int | None = None
    threshold: float | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None
    model: object | None = None
    node_error: float = 0.0
    subtree_error: float = 0.0
    subtree_leaves: int = 1
    prune_alpha: float | None = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class DistanceDecisionTree:
    def __init__(self, max_depth: int | None = None, classification: bool = False):
        self.max_depth = max_depth
        self.classification = classification
        self.tree = None
        self.final_depth = 0

    def copy(self):
        new_instance = DistanceDecisionTree(self.max_depth, self.classification)
        new_instance.tree = copy.deepcopy(self.tree)
        new_instance.final_depth = self.final_depth
        return new_instance

    def fit(self, location_matrix: np.ndarray, distance_matrix: np.ndarray, depth: int = 0, parent=None):
        cell_indices = np.arange(location_matrix.shape[0])
        self.tree, self.final_depth = self._build_tree(
            location_matrix=location_matrix,
            distance_matrix=distance_matrix,
            cell_indices=cell_indices,
            depth=depth,
        )
        return self

    def _node_impurity(self, distance_matrix: np.ndarray, cell_indices: np.ndarray) -> float:
        if len(cell_indices) <= 1:
            return 0.0
        sub = distance_matrix[np.ix_(cell_indices, cell_indices)]
        return float(np.sum(sub) / (2.0 * len(cell_indices)))

    def _build_tree(
        self,
        location_matrix: np.ndarray,
        distance_matrix: np.ndarray,
        cell_indices: np.ndarray,
        depth: int,
    ):
        node = TreeNode(
            cell_indices=np.asarray(cell_indices, dtype=int),
            depth=depth,
            impurity=self._node_impurity(distance_matrix, cell_indices),
        )
        if self.max_depth is not None and depth >= self.max_depth:
            return node, depth
        if len(cell_indices) <= 1:
            return node, depth

        best_split = self._find_best_split(location_matrix, distance_matrix, cell_indices)
        if best_split is None or best_split["gain"] <= 0:
            return node, depth

        node.feature_index = best_split["feature_index"]
        node.threshold = best_split["threshold"]
        node.gain = best_split["gain"]
        left_child, left_depth = self._build_tree(
            location_matrix,
            distance_matrix,
            best_split["left_indices"],
            depth + 1,
        )
        right_child, right_depth = self._build_tree(
            location_matrix,
            distance_matrix,
            best_split["right_indices"],
            depth + 1,
        )
        node.left = left_child
        node.right = right_child
        return node, max(left_depth, right_depth)

    def _find_best_split(self, location_matrix, distance_matrix, cell_indices):
        best = None
        best_gain = 0.0
        node_impurity = self._node_impurity(distance_matrix, cell_indices)
        location_subset = location_matrix[cell_indices]
        for feature_index in range(location_subset.shape[1]):
            for threshold in np.unique(location_subset[:, feature_index]):
                left_mask = location_subset[:, feature_index] <= threshold
                right_mask = ~left_mask
                if not np.any(left_mask) or not np.any(right_mask):
                    continue
                left_indices = cell_indices[left_mask]
                right_indices = cell_indices[right_mask]
                left_impurity = self._node_impurity(distance_matrix, left_indices)
                right_impurity = self._node_impurity(distance_matrix, right_indices)
                weighted_impurity = (
                    len(left_indices) * left_impurity + len(right_indices) * right_impurity
                ) / len(cell_indices)
                gain = node_impurity - weighted_impurity
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best = {
                        "feature_index": feature_index,
                        "threshold": float(threshold),
                        "left_indices": left_indices,
                        "right_indices": right_indices,
                        "gain": gain,
                    }
        return best

    def fit_leaf_models(self, X: np.ndarray, y: np.ndarray, model_factory):
        self._fit_leaf_models(self.tree, X, y, model_factory)
        return self

    def fit_node_models(self, X: np.ndarray, y: np.ndarray, model_factory):
        self._fit_node_models(self.tree, X, y, model_factory)
        return self

    def _fit_leaf_models(self, node: TreeNode, X: np.ndarray, y: np.ndarray, model_factory):
        if node.is_leaf:
            node.model = model_factory()
            node.model.fit(X, y)
            return
        left_mask = X[:, node.feature_index] <= node.threshold
        right_mask = ~left_mask
        self._fit_leaf_models(node.left, X[left_mask], y[left_mask], model_factory)
        self._fit_leaf_models(node.right, X[right_mask], y[right_mask], model_factory)

    def _fit_node_models(self, node: TreeNode, X: np.ndarray, y: np.ndarray, model_factory):
        node.model = model_factory()
        node.model.fit(X, y)
        if node.is_leaf:
            return
        left_mask = X[:, node.feature_index] <= node.threshold
        right_mask = ~left_mask
        self._fit_node_models(node.left, X[left_mask], y[left_mask], model_factory)
        self._fit_node_models(node.right, X[right_mask], y[right_mask], model_factory)

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        return np.asarray([self._predict_one(x, self.tree) for x in X], dtype=float)

    def _predict_one(self, x: np.ndarray, node: TreeNode) -> float:
        if node.is_leaf:
            return float(node.model.predict_score(x.reshape(1, -1))[0])
        if x[node.feature_index] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def leaf_info(self, X: np.ndarray) -> pd.DataFrame:
        rows = []

        def visit(node: TreeNode, X_node: np.ndarray):
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

    def interpretability(self, X: np.ndarray, alpha: float = 0.5):
        leaf_df = self.leaf_info(X)
        return leaf_df, interpretability_score(leaf_df, alpha=alpha)

    def count_leaves(self) -> int:
        def count(node: TreeNode) -> int:
            if node.is_leaf:
                return 1
            return count(node.left) + count(node.right)

        return count(self.tree)

    def weakest_link_pruning_path(self, X: np.ndarray, y: np.ndarray):
        working_tree = self.copy()
        path = [
            {
                "alpha": 0.0,
                "tree": working_tree.copy(),
                "leaves": working_tree.count_leaves(),
                "depth": working_tree.final_depth,
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
            if eligible.size == 0:
                current_alpha = float(np.min(prune_scores))
            else:
                current_alpha = float(np.min(eligible))
            leaves_before = working_tree.count_leaves()
            working_tree._prune_by_alpha(working_tree.tree, current_alpha)
            working_tree.final_depth = working_tree._max_depth(working_tree.tree)
            leaves_after = working_tree.count_leaves()
            if leaves_after >= leaves_before:
                break
            previous_alpha = current_alpha
            path.append(
                {
                    "alpha": current_alpha,
                    "tree": working_tree.copy(),
                    "leaves": leaves_after,
                    "depth": working_tree.final_depth,
                }
            )
        return path

    def _annotate_pruning_stats(self, node: TreeNode, X: np.ndarray, y: np.ndarray, candidates: list[TreeNode]):
        node.node_error = self._resubstitution_error(node.model, X, y)
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

    def _prune_by_alpha(self, node: TreeNode, alpha: float):
        if node.is_leaf:
            return
        self._prune_by_alpha(node.left, alpha)
        self._prune_by_alpha(node.right, alpha)
        if node.prune_alpha is not None and node.prune_alpha <= alpha + 1e-12:
            node.left = None
            node.right = None
            node.feature_index = None
            node.threshold = None
            node.gain = 0.0

    def _resubstitution_error(self, model, X: np.ndarray, y: np.ndarray) -> float:
        if X.shape[0] == 0:
            return 0.0
        if self.classification:
            predictions = np.asarray(model.predict_label(X), dtype=int)
            return float(np.sum(predictions != np.asarray(y, dtype=int)))
        scores = np.asarray(model.predict_score(X), dtype=float)
        residuals = np.asarray(y, dtype=float) - scores
        return float(np.sum(residuals ** 2))

    def _max_depth(self, node: TreeNode | None) -> int:
        if node is None:
            return 0
        if node.is_leaf:
            return node.depth
        return max(self._max_depth(node.left), self._max_depth(node.right))
