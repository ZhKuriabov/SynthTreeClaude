from __future__ import annotations

import unittest

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from synthtree import SynthTreeClassifier, SynthTreeRegressor
from synthtree.analysis import synthtree_feature_importance
from synthtree.co_supervision import select_num_cells
from synthtree.metrics import inverse_f1_disparity, mutual_prediction_disparity
from synthtree.pruning import _select_subtree_for_alpha
from synthtree.tree import DistanceDecisionTree


class _StubModel:
    def __init__(self, scores, labels=None):
        self.scores = np.asarray(scores, dtype=float)
        self.labels = np.asarray(labels if labels is not None else (self.scores >= 0.5).astype(int))

    def predict_score(self, X):
        return self.scores[: len(X)]

    def predict_label(self, X):
        return self.labels[: len(X)]


class _MeanRegModel:
    def __init__(self):
        self.value = 0.0

    def fit(self, X, y):
        self.value = float(np.mean(y)) if len(y) else 0.0
        return self

    def predict_score(self, X):
        return np.repeat(self.value, len(X))

    def predict_label(self, X):
        return self.predict_score(X)

    def complexity(self):
        return 0


class SynthTreeTests(unittest.TestCase):
    def test_regressor_fit_predict(self):
        X, y = make_regression(n_samples=80, n_features=6, noise=0.2, random_state=0)
        model = SynthTreeRegressor(
            teachers={"RF": RandomForestRegressor(n_estimators=20, random_state=0)},
            n_cells_grid=(2, 3),
            num_aug=5,
            max_depth=2,
            pruning_cv=2,
            random_state=0,
        )
        model.fit(X, y)
        preds = model.predict(X[:10])
        self.assertEqual(preds.shape, (10,))
        self.assertTrue(hasattr(model, "tree_"))
        self.assertTrue(hasattr(model, "selected_num_cells_"))
        self.assertTrue(hasattr(model, "interpretability_"))

    def test_classifier_fit_predict_proba(self):
        X, y = make_classification(
            n_samples=100,
            n_features=8,
            n_informative=5,
            n_redundant=0,
            random_state=0,
        )
        model = SynthTreeClassifier(
            teachers={"RF": RandomForestClassifier(n_estimators=20, random_state=0)},
            n_cells_grid=(2, 3),
            num_aug=5,
            max_depth=2,
            pruning_cv=2,
            random_state=0,
        )
        model.fit(X, y)
        probs = model.predict_proba(X[:10])
        preds = model.predict(X[:10])
        self.assertEqual(probs.shape, (10, 2))
        self.assertEqual(preds.shape, (10,))
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0))

    def test_select_num_cells_validation_and_silhouette(self):
        X, y = make_classification(
            n_samples=60,
            n_features=6,
            n_informative=4,
            n_redundant=0,
            random_state=0,
        )
        teacher = {"RF": RandomForestClassifier(n_estimators=10, random_state=0)}
        selected_validation, info_validation = select_num_cells(
            X,
            y,
            teachers=teacher,
            n_cells_grid=(2, 3),
            classification=True,
            selection="validation_mlm",
            num_aug=3,
            covariance_type="diag",
            local_model_alpha=0.1,
            local_model_max_iter=100,
            random_state=0,
        )
        selected_silhouette, info_silhouette = select_num_cells(
            X,
            y,
            teachers=teacher,
            n_cells_grid=(2, 3),
            classification=True,
            selection="silhouette",
            num_aug=3,
            covariance_type="diag",
            local_model_alpha=0.1,
            local_model_max_iter=100,
            random_state=0,
        )
        self.assertIn(selected_validation, (2, 3))
        self.assertIn(selected_silhouette, (2, 3))
        self.assertEqual(info_validation["selection"], "validation_mlm")
        self.assertEqual(info_silhouette["selection"], "silhouette")

    def test_disparity_functions(self):
        X = np.zeros((4, 2))
        reg_a = _StubModel(scores=[0.0, 0.0, 1.0, 1.0])
        reg_b = _StubModel(scores=[0.0, 1.0, 1.0, 2.0])
        cls_a = _StubModel(scores=[0.9, 0.9, 0.1, 0.1], labels=[1, 1, 0, 0])
        cls_b = _StubModel(scores=[0.9, 0.1, 0.1, 0.9], labels=[1, 0, 0, 1])
        self.assertGreaterEqual(mutual_prediction_disparity(reg_a, reg_b, X), 0.0)
        self.assertGreaterEqual(inverse_f1_disparity(cls_a, cls_b, X), 0.0)

    def test_tree_stops_without_positive_gain(self):
        location_matrix = np.array([[0.0], [1.0], [2.0]])
        distance_matrix = np.zeros((3, 3))
        tree = DistanceDecisionTree(max_depth=3, classification=False)
        tree.fit(location_matrix, distance_matrix)
        self.assertTrue(tree.tree.is_leaf)

    def test_weakest_link_pruning_path_is_nested(self):
        X = np.array([[0.0], [0.5], [1.5], [2.0], [3.0], [3.5]])
        y = np.array([0.0, 0.2, 1.0, 1.2, 3.0, 3.2])
        location_matrix = np.array([[0.0], [1.0], [2.0], [3.0]])
        distance_matrix = np.array(
            [
                [0.0, 0.1, 2.0, 2.5],
                [0.1, 0.0, 1.8, 2.2],
                [2.0, 1.8, 0.0, 0.2],
                [2.5, 2.2, 0.2, 0.0],
            ]
        )
        tree = DistanceDecisionTree(max_depth=3, classification=False)
        tree.fit(location_matrix, distance_matrix)
        tree.fit_node_models(X, y, _MeanRegModel)
        path = tree.weakest_link_pruning_path(X, y)
        alphas = [entry["alpha"] for entry in path]
        leaves = [entry["leaves"] for entry in path]
        self.assertTrue(all(a2 >= a1 - 1e-12 for a1, a2 in zip(alphas, alphas[1:])))
        self.assertTrue(all(l2 <= l1 for l1, l2 in zip(leaves, leaves[1:])))
        self.assertEqual(leaves[-1], 1)

    def test_select_subtree_for_alpha_respects_min_leaves(self):
        X = np.array([[0.0], [0.5], [1.5], [2.0], [3.0], [3.5]])
        y = np.array([0.0, 0.2, 1.0, 1.2, 3.0, 3.2])
        location_matrix = np.array([[0.0], [1.0], [2.0], [3.0]])
        distance_matrix = np.array(
            [
                [0.0, 0.1, 2.0, 2.5],
                [0.1, 0.0, 1.8, 2.2],
                [2.0, 1.8, 0.0, 0.2],
                [2.5, 2.2, 0.2, 0.0],
            ]
        )
        tree = DistanceDecisionTree(max_depth=3, classification=False)
        tree.fit(location_matrix, distance_matrix)
        tree.fit_node_models(X, y, _MeanRegModel)
        path = tree.weakest_link_pruning_path(X, y)
        unconstrained = _select_subtree_for_alpha(path, alpha=path[-1]["alpha"], min_leaves=1)
        constrained = _select_subtree_for_alpha(path, alpha=path[-1]["alpha"], min_leaves=2)
        self.assertEqual(unconstrained.count_leaves(), 1)
        self.assertGreaterEqual(constrained.count_leaves(), 2)

    def test_classifier_cc_prune_runs(self):
        X, y = make_classification(
            n_samples=80,
            n_features=6,
            n_informative=4,
            n_redundant=0,
            random_state=1,
        )
        model = SynthTreeClassifier(
            teachers={"RF": RandomForestClassifier(n_estimators=10, random_state=1)},
            n_cells_grid=(2, 3),
            num_aug=3,
            max_depth=3,
            pruning_cv=2,
            tree_sizer="cc_prune",
            random_state=1,
        )
        model.fit(X, y)
        self.assertGreaterEqual(model.tree_.count_leaves(), 1)
        self.assertIsNotNone(model.selected_alpha_)

    def test_classifier_cc_prune_min_leaves_runs(self):
        X, y = make_classification(
            n_samples=80,
            n_features=6,
            n_informative=4,
            n_redundant=0,
            random_state=3,
        )
        model = SynthTreeClassifier(
            teachers={"RF": RandomForestClassifier(n_estimators=10, random_state=3)},
            n_cells_grid=(2, 3),
            num_aug=3,
            max_depth=3,
            pruning_cv=2,
            tree_sizer="cc_prune",
            min_leaves=2,
            random_state=3,
        )
        model.fit(X, y)
        self.assertGreaterEqual(model.tree_.count_leaves(), 2)

    def test_synthtree_feature_importance_is_normalized(self):
        X, y = make_classification(
            n_samples=80,
            n_features=6,
            n_informative=4,
            n_redundant=0,
            random_state=2,
        )
        model = SynthTreeClassifier(
            teachers={"RF": RandomForestClassifier(n_estimators=10, random_state=2)},
            n_cells_grid=(2, 3),
            num_aug=3,
            max_depth=2,
            pruning_cv=2,
            tree_sizer="l_trim",
            random_state=2,
        )
        model.fit(X, y)
        feature_names = [f"x{i}" for i in range(X.shape[1])]
        global_df, leaf_df = synthtree_feature_importance(model.tree_, X, feature_names)
        self.assertEqual(len(global_df), X.shape[1])
        self.assertAlmostEqual(float(global_df["importance_norm"].sum()), 1.0, places=6)
        leaf_weights = leaf_df[["leaf_id", "leaf_weight"]].drop_duplicates()["leaf_weight"]
        self.assertAlmostEqual(float(leaf_weights.sum()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
