from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from .co_supervision import build_co_supervision, default_teacher, select_num_cells
from .metrics import inverse_f1_disparity, mutual_prediction_disparity
from .models import make_statsmodels_model
from .pruning import select_tree_size, _select_subtree_for_alpha
from .tree import DistanceDecisionTree


@dataclass
class _PreparedData:
    X_augmented: np.ndarray
    y_augmented: np.ndarray
    centroids: np.ndarray
    distance_matrix: np.ndarray


class _BaseSynthTree(BaseEstimator):
    classification = False

    def __init__(
        self,
        teachers=None,
        n_cells_grid=(3, 5, 8, 10),
        n_cells_selection="validation_mlm",
        num_aug=100,
        augmentation_covariance="diag",
        tree_sizer="auto",
        max_depth=4,
        local_model_alpha=0.1,
        leaf_model_alpha=0.1,
        local_model_max_iter=1000,
        leaf_model_max_iter=1000,
        interpretability_alpha=0.5,
        pruning_cv=10,
        min_leaves=1,
        random_state=0,
        verbose=False,
    ):
        self.teachers = teachers
        self.n_cells_grid = n_cells_grid
        self.n_cells_selection = n_cells_selection
        self.num_aug = num_aug
        self.augmentation_covariance = augmentation_covariance
        self.tree_sizer = tree_sizer
        self.max_depth = max_depth
        self.local_model_alpha = local_model_alpha
        self.leaf_model_alpha = leaf_model_alpha
        self.local_model_max_iter = local_model_max_iter
        self.leaf_model_max_iter = leaf_model_max_iter
        self.interpretability_alpha = interpretability_alpha
        self.pruning_cv = pruning_cv
        self.min_leaves = min_leaves
        self.random_state = random_state
        self.verbose = verbose

    def _effective_tree_sizer(self):
        if self.tree_sizer != "auto":
            return self.tree_sizer
        return "cc_prune" if self.classification else "l_trim"

    def _default_teachers(self):
        return {"RF": default_teacher(self.classification)}

    def _prepare(self, X, y, n_cells):
        cosup = build_co_supervision(
            X=X,
            y=y,
            teachers=self.teachers if self.teachers is not None else self._default_teachers(),
            n_cells=n_cells,
            classification=self.classification,
            num_aug=self.num_aug,
            covariance_type=self.augmentation_covariance,
            local_model_alpha=self.local_model_alpha,
            local_model_max_iter=self.local_model_max_iter,
            random_state=self.random_state,
        )
        X_augmented, y_augmented = cosup.augmented_training_data
        distance_matrix = self._distance_matrix(cosup.cell_models)
        return cosup, _PreparedData(
            X_augmented=X_augmented,
            y_augmented=y_augmented,
            centroids=cosup.centroids,
            distance_matrix=distance_matrix,
        )

    def _distance_matrix(self, cell_models):
        n_cells = len(cell_models)
        distance_matrix = np.zeros((n_cells, n_cells), dtype=float)
        for i in range(n_cells):
            for j in range(i + 1, n_cells):
                X_eval = np.vstack([cell_models[i].X_augmented, cell_models[j].X_augmented])
                if self.classification:
                    dist = inverse_f1_disparity(cell_models[i].model, cell_models[j].model, X_eval)
                else:
                    dist = mutual_prediction_disparity(cell_models[i].model, cell_models[j].model, X_eval)
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        return distance_matrix

    def _model_factory(self):
        return make_statsmodels_model(
            classification=self.classification,
            alpha=self.leaf_model_alpha,
            max_iter=self.leaf_model_max_iter,
        )

    def _fit_tree_for_depth(self, depth, X, y):
        _, prepared = self._prepare(X, y, self.selected_num_cells_)
        tree = DistanceDecisionTree(max_depth=depth, classification=self.classification)
        tree.fit(prepared.centroids, prepared.distance_matrix)
        tree.fit_leaf_models(prepared.X_augmented, prepared.y_augmented, self._model_factory)
        wrapped = _FittedTreeWrapper(tree)
        return wrapped

    def _fit_full_tree(self, X, y):
        _, prepared = self._prepare(X, y, self.selected_num_cells_)
        tree = DistanceDecisionTree(max_depth=self.max_depth, classification=self.classification)
        tree.fit(prepared.centroids, prepared.distance_matrix)
        tree.fit_node_models(prepared.X_augmented, prepared.y_augmented, self._model_factory)
        wrapped = _FittedTreeWrapper(tree)
        return wrapped

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if self.classification:
            self.classes_ = np.sort(np.unique(y))
            if self.classes_.size != 2:
                raise ValueError("SynthTreeClassifier currently supports binary classification only.")
            y_internal = (y == self.classes_[1]).astype(int)
        else:
            y_internal = y
        selected_num_cells, selection_info = select_num_cells(
            X=X,
            y=y_internal,
            teachers=self.teachers if self.teachers is not None else self._default_teachers(),
            n_cells_grid=self.n_cells_grid,
            classification=self.classification,
            selection=self.n_cells_selection,
            num_aug=self.num_aug,
            covariance_type=self.augmentation_covariance,
            local_model_alpha=self.local_model_alpha,
            local_model_max_iter=self.local_model_max_iter,
            random_state=self.random_state,
        )
        self.selected_num_cells_ = int(selected_num_cells)
        self.n_cells_selection_info_ = selection_info

        tree_sizer = self._effective_tree_sizer()
        tree_size_info = select_tree_size(
            X=X,
            y=y_internal,
            classification=self.classification,
            max_depth=self.max_depth,
            method=tree_sizer,
            fit_depth_fn=self._fit_tree_for_depth,
            fit_full_tree_fn=self._fit_full_tree,
            cv=self.pruning_cv,
            random_state=self.random_state,
            min_leaves=self.min_leaves,
        )
        self.selected_alpha_ = tree_size_info["alpha"]
        self.tree_sizer_ = tree_sizer
        self.tree_size_info_ = tree_size_info

        cosup, prepared = self._prepare(X, y_internal, self.selected_num_cells_)
        if tree_sizer == "cc_prune":
            full_tree = DistanceDecisionTree(max_depth=self.max_depth, classification=self.classification)
            full_tree.fit(prepared.centroids, prepared.distance_matrix)
            full_tree.fit_node_models(prepared.X_augmented, prepared.y_augmented, self._model_factory)
            path = full_tree.weakest_link_pruning_path(prepared.X_augmented, prepared.y_augmented)
            tree = _select_subtree_for_alpha(path, float(self.selected_alpha_), min_leaves=self.min_leaves)
            self.selected_depth_ = int(tree.final_depth)
        else:
            self.selected_depth_ = int(tree_size_info["depth"])
            tree = DistanceDecisionTree(max_depth=self.selected_depth_, classification=self.classification)
            tree.fit(prepared.centroids, prepared.distance_matrix)
            tree.fit_leaf_models(prepared.X_augmented, prepared.y_augmented, self._model_factory)

        self.tree_ = tree
        self.cell_models_ = cosup.cell_models
        self.cell_centroids_ = prepared.centroids
        self.selected_teachers_ = cosup.selected_teacher_names
        self.leaf_info_, self.interpretability_ = tree.interpretability(
            prepared.X_augmented,
            alpha=self.interpretability_alpha,
        )
        self.leaf_models_ = self._collect_leaf_models(tree.tree)
        self.X_augmented_ = prepared.X_augmented
        self.y_augmented_ = prepared.y_augmented
        return self

    def _collect_leaf_models(self, node):
        if node.is_leaf:
            return [node.model]
        return self._collect_leaf_models(node.left) + self._collect_leaf_models(node.right)

    def predict_scores(self, X):
        X = np.asarray(X, dtype=float)
        return self.tree_.predict_scores(X)

    def predict(self, X):
        scores = self.predict_scores(X)
        if self.classification:
            labels = (scores >= 0.5).astype(int)
            return self.classes_[labels]
        return scores


class _FittedTreeWrapper:
    def __init__(self, tree):
        self.tree_ = tree

    def predict_scores(self, X):
        return self.tree_.predict_scores(X)


class SynthTreeClassifier(_BaseSynthTree, ClassifierMixin):
    classification = True

    def predict_proba(self, X):
        scores = np.clip(self.predict_scores(X), 1e-6, 1 - 1e-6)
        return np.column_stack([1 - scores, scores])


class SynthTreeRegressor(_BaseSynthTree, RegressorMixin):
    classification = False
