from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from example_preprocessing import prep_data
from synthtree import SynthTreeClassifier, SynthTreeRegressor
from synthtree.metrics import rmse

warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

try:
    from rpy2 import robjects as ro
    from rpy2.robjects import default_converter, pandas2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr

    ro.r('library(devtools)')
    ro.r('devtools::load_all("../Rforestry")')
    forestry = importr("Rforestry")
    HAVE_RPY2 = True
except Exception:
    HAVE_RPY2 = False


CLASS_DATASETS = {"SKCM", "Road Safety", "Compas", "Upselling"}
REG_DATASETS = {"Cal Housing", "Bike Sharing", "Abalone", "Servo"}
ALL_DATASETS = sorted(CLASS_DATASETS | REG_DATASETS)


class _ForestryMixin(BaseEstimator):
    _ntree: int = 1

    def __init__(self):
        self.model_ = None

    def fit(self, X, y):
        if not HAVE_RPY2:
            raise RuntimeError("rpy2 / Rforestry is not available in this environment")
        with localconverter(default_converter + pandas2ri.converter):
            ro.globalenv["X"] = pd.DataFrame(X)
            ro.globalenv["y"] = pd.Series(y)
        ro.r(f"model <- forestry(x = X, y = y, ntree = {self._ntree})")
        self.model_ = ro.globalenv["model"]
        return self

    def _predict_r(self, newX):
        with localconverter(default_converter + pandas2ri.converter):
            ro.globalenv["X_new"] = pd.DataFrame(newX)
        preds = ro.r("predict(model, newdata = X_new, feature.new = X_new)")
        return np.asarray(preds, dtype=float)


class LRTClassifier(_ForestryMixin, ClassifierMixin):
    _ntree = 1

    def predict_proba(self, X):
        p = self._predict_r(X)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self._predict_r(X) > 0.5).astype(int)


class LRTRegressor(_ForestryMixin, RegressorMixin):
    _ntree = 1

    def predict(self, X):
        return self._predict_r(X)


class LRFClassifier(LRTClassifier):
    _ntree = 100


class LRFRegressor(LRTRegressor):
    _ntree = 100


class _OOBTunedForestryMixin(BaseEstimator):
    """OOB-based grid search over ntree and mtry for Rforestry."""

    _ntree_grid = [100, 300, 500]
    _mtry_options = ["half", "all"]

    def __init__(self):
        self.model_ = None
        self.best_ntree_ = None
        self.best_mtry_ = None

    def fit(self, X, y):
        if not HAVE_RPY2:
            raise RuntimeError("rpy2 / Rforestry is not available in this environment")
        p = X.shape[1]
        mtry_values = {"half": max(1, p // 2), "all": p}

        best_oob = float("inf")
        best_params = (self._ntree_grid[0], p)

        with localconverter(default_converter + pandas2ri.converter):
            ro.globalenv["X"] = pd.DataFrame(X)
            ro.globalenv["y"] = pd.Series(y)

        for ntree in self._ntree_grid:
            for mtry_label, mtry_val in mtry_values.items():
                try:
                    ro.r(f"candidate <- forestry(x = X, y = y, ntree = {ntree}, mtry = {mtry_val})")
                    oob = float(ro.r("getOOB(candidate)")[0])
                    if oob < best_oob:
                        best_oob = oob
                        best_params = (ntree, mtry_val)
                except Exception:
                    continue

        self.best_ntree_, self.best_mtry_ = best_params
        ro.r(f"model <- forestry(x = X, y = y, ntree = {self.best_ntree_}, mtry = {self.best_mtry_})")
        self.model_ = ro.globalenv["model"]
        return self

    def _predict_r(self, newX):
        with localconverter(default_converter + pandas2ri.converter):
            ro.globalenv["X_new"] = pd.DataFrame(newX)
        preds = ro.r("predict(model, newdata = X_new, feature.new = X_new)")
        result = np.asarray(preds, dtype=float)
        result = np.nan_to_num(result, nan=0.5)
        return result


class AutoLRFClassifier(_OOBTunedForestryMixin, ClassifierMixin):
    def predict_proba(self, X):
        p = np.clip(self._predict_r(X), 0.0, 1.0)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self._predict_r(X) > 0.5).astype(int)


class AutoLRFRegressor(_OOBTunedForestryMixin, RegressorMixin):
    def predict(self, X):
        return self._predict_r(X)


class _CVForestryLRT(BaseEstimator):
    """3-fold CV over maxDepth for a single linear regression tree."""

    def __init__(self, cv: int = 3, max_depth_grid=None):
        self.cv = cv
        self.max_depth_grid = max_depth_grid or [3, 5, 10, 20]
        self.model_ = None
        self.best_depth_ = None

    def fit(self, X, y):
        if not HAVE_RPY2:
            raise RuntimeError("rpy2 / Rforestry is not available in this environment")
        from sklearn.model_selection import KFold, StratifiedKFold
        from sklearn.metrics import roc_auc_score, mean_squared_error

        classification = self._is_classification()
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)

        if classification:
            splitter = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=0)
        else:
            splitter = KFold(n_splits=self.cv, shuffle=True, random_state=0)

        best_score = None
        best_depth = self.max_depth_grid[0]

        for depth in self.max_depth_grid:
            scores = []
            for train_idx, val_idx in splitter.split(X_arr, y_arr):
                X_tr, y_tr = X_arr[train_idx], y_arr[train_idx]
                X_val, y_val = X_arr[val_idx], y_arr[val_idx]
                with localconverter(default_converter + pandas2ri.converter):
                    ro.globalenv["X_tr"] = pd.DataFrame(X_tr)
                    ro.globalenv["y_tr"] = pd.Series(y_tr)
                ro.r(f"cv_model <- forestry(x = X_tr, y = y_tr, ntree = 1, maxDepth = {depth})")
                with localconverter(default_converter + pandas2ri.converter):
                    ro.globalenv["X_val"] = pd.DataFrame(X_val)
                preds = np.asarray(ro.r("predict(cv_model, newdata = X_val, feature.new = X_val)"), dtype=float)
                preds = np.nan_to_num(preds, nan=0.5 if classification else float(np.mean(y_tr)))
                if classification:
                    preds_clipped = np.clip(preds, 0.0, 1.0)
                    scores.append(roc_auc_score(y_val, preds_clipped))
                else:
                    scores.append(-np.sqrt(mean_squared_error(y_val, preds)))
            mean_score = np.mean(scores)
            if best_score is None or mean_score > best_score:
                best_score = mean_score
                best_depth = depth

        self.best_depth_ = best_depth
        with localconverter(default_converter + pandas2ri.converter):
            ro.globalenv["X"] = pd.DataFrame(X_arr)
            ro.globalenv["y"] = pd.Series(y_arr)
        ro.r(f"model <- forestry(x = X, y = y, ntree = 1, maxDepth = {best_depth})")
        self.model_ = ro.globalenv["model"]
        return self

    def _predict_r(self, newX):
        with localconverter(default_converter + pandas2ri.converter):
            ro.globalenv["X_new"] = pd.DataFrame(newX)
        preds = ro.r("predict(model, newdata = X_new, feature.new = X_new)")
        result = np.asarray(preds, dtype=float)
        result = np.nan_to_num(result, nan=0.5)
        return result

    def _is_classification(self):
        raise NotImplementedError


class CVLRTClassifier(_CVForestryLRT, ClassifierMixin):
    def _is_classification(self):
        return True

    def predict_proba(self, X):
        p = np.clip(self._predict_r(X), 0.0, 1.0)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self._predict_r(X) > 0.5).astype(int)


class CVLRTRegressor(_CVForestryLRT, RegressorMixin):
    def _is_classification(self):
        return False

    def predict(self, X):
        return self._predict_r(X)


TEACHERS = {
    "RF": (RandomForestClassifier(n_estimators=100, random_state=0), RandomForestRegressor(n_estimators=100, random_state=0)),
    "GB": (GradientBoostingClassifier(random_state=0), GradientBoostingRegressor(random_state=0)),
    "MLP": (MLPClassifier(max_iter=1000, random_state=0), MLPRegressor(max_iter=1000, random_state=0)),
    "LRF": (LRFClassifier(), LRFRegressor()),
}


def task_is_classification(dataset_name: str) -> bool:
    return dataset_name in CLASS_DATASETS


def build_students(dataset_name: str, teacher_name: str):
    classification = task_is_classification(dataset_name)
    teacher_cls, teacher_reg = TEACHERS[teacher_name]
    teacher = teacher_cls if classification else teacher_reg
    synthtree = (
        SynthTreeClassifier(teachers={teacher_name: teacher}, tree_sizer="auto", n_cells_selection="validation_mlm")
        if classification
        else SynthTreeRegressor(teachers={teacher_name: teacher}, tree_sizer="auto", n_cells_selection="validation_mlm")
    )
    students = {
        "SynthTree": synthtree,
        "CART": DecisionTreeClassifier(random_state=0) if classification else DecisionTreeRegressor(random_state=0),
    }
    if HAVE_RPY2:
        students["LRT"] = LRTClassifier() if classification else LRTRegressor()
    return students


def score_model(model_name: str, model, X_test, y_test, classification: bool):
    if classification:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
        else:
            probs = model.predict(X_test)
        return roc_auc_score(y_test, probs)
    preds = model.predict(X_test)
    return rmse(y_test, preds)


def interpretability_value(model_name: str, model):
    if model_name == "SynthTree":
        return float(model.interpretability_)
    if model_name == "CART":
        tree = model.tree_
        is_leaf = tree.children_left == -1
        leaf_ids = np.where(is_leaf)[0]
        depths = np.zeros(tree.node_count, dtype=int)

        def traverse(node_id=0, depth=0):
            depths[node_id] = depth
            if tree.children_left[node_id] != -1:
                traverse(tree.children_left[node_id], depth + 1)
                traverse(tree.children_right[node_id], depth + 1)

        traverse()
        weights = tree.n_node_samples[leaf_ids]
        return float(np.sum(weights * depths[leaf_ids]) / np.sum(weights))
    return np.nan


def run_one(dataset_name: str, teacher_name: str, seed: int):
    classification = task_is_classification(dataset_name)
    X_train, y_train, X_test, y_test = prep_data(dataset_name, random_state=seed)
    rows = []
    for model_name, model in build_students(dataset_name, teacher_name).items():
        model.fit(X_train, y_train)
        rows.append(
            {
                "dataset": dataset_name,
                "teacher": teacher_name,
                "model": model_name,
                "seed": seed,
                "score": score_model(model_name, model, X_test, y_test, classification),
                "interpretability": interpretability_value(model_name, model),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Run cleaned SynthTree experiments with repeated splits")
    parser.add_argument("--runs", "-r", type=int, default=3, help="Number of repeated train/test splits")
    parser.add_argument("dataset", nargs="?", choices=ALL_DATASETS, help="Optional single dataset")
    parser.add_argument("teacher", nargs="?", choices=list(TEACHERS), help="Optional single teacher")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else ALL_DATASETS
    teachers = [args.teacher] if args.teacher else list(TEACHERS)
    rows = []
    jobs = [(run_idx, dataset_name, teacher_name) for run_idx in range(args.runs) for dataset_name in datasets for teacher_name in teachers]
    iterator = jobs
    if tqdm is not None:
        iterator = tqdm(jobs, desc="SynthTree experiments", unit="job")

    for run_idx, dataset_name, teacher_name in iterator:
        if tqdm is None:
            print(f"[{run_idx + 1}/{args.runs}] {dataset_name} | teacher={teacher_name}")
        rows.extend(run_one(dataset_name, teacher_name, seed=run_idx))

    results = pd.DataFrame(rows)
    results.to_csv("cleaned_synthtree_results.csv", index=False)
    summary = (
        results.groupby(["dataset", "teacher", "model"])
        .agg(score_mean=("score", "mean"), score_std=("score", "std"), interp_mean=("interpretability", "mean"))
        .reset_index()
    )
    summary.to_csv("cleaned_synthtree_summary.csv", index=False)
    print(summary)


if __name__ == "__main__":
    main()
