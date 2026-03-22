from __future__ import annotations

import argparse
import os
import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from example_preprocessing import prep_data
from synthtree.co_supervision import build_co_supervision, select_num_cells
from synthtree.metrics import mutual_prediction_disparity, inverse_f1_disparity, rmse
from synthtree.models import make_statsmodels_model
from synthtree.pruning import select_tree_size, _select_subtree_for_alpha
from synthtree.tree import DistanceDecisionTree

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Maximum Likelihood optimization failed to converge.*")
warnings.filterwarnings("ignore", message=".*QC check did not pass.*")
warnings.filterwarnings("ignore", message=".*Could not trim params automatically.*")

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


CLASS_DATASETS = {"SKCM", "Road Safety", "Compas", "Upselling"}
REG_DATASETS = {"Cal Housing", "Bike Sharing", "Abalone", "Servo"}
DEFAULT_DATASETS = ["SKCM", "Upselling", "Abalone", "Bike Sharing"]
DEFAULT_METHODS = ["SynthTree", "CART", "LRT", "MLM-EPIC"]


def task_is_classification(dataset_name: str) -> bool:
    return dataset_name in CLASS_DATASETS


def make_rf_teacher(classification: bool):
    if classification:
        return RandomForestClassifier(n_estimators=100, random_state=0)
    return RandomForestRegressor(n_estimators=100, random_state=0)


def score_predictions(y_true, scores, classification: bool) -> float:
    if classification:
        return float(roc_auc_score(y_true, scores))
    return float(rmse(y_true, scores))


def normalize_binary_labels(y_train, y_test):
    classes = pd.Series(y_train).sort_values().unique()
    positive_class = classes[-1]
    return (y_train == positive_class).astype(int), (y_test == positive_class).astype(int)


@dataclass
class PreparedSynthTree:
    X_augmented: np.ndarray
    y_augmented: np.ndarray
    centroids: np.ndarray
    distance_matrix: np.ndarray
    selected_teachers: list[str]


def compute_distance_matrix(cell_models, classification: bool) -> np.ndarray:
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


def prepare_synthtree(
    X,
    y,
    classification: bool,
    teachers,
    n_cells: int,
    num_aug: int,
    augmentation_covariance: str,
    local_model_alpha: float,
    local_model_max_iter: int,
    random_state: int,
) -> PreparedSynthTree:
    cosup = build_co_supervision(
        X=X,
        y=y,
        teachers=teachers,
        n_cells=n_cells,
        classification=classification,
        num_aug=num_aug,
        covariance_type=augmentation_covariance,
        local_model_alpha=local_model_alpha,
        local_model_max_iter=local_model_max_iter,
        random_state=random_state,
    )
    X_augmented, y_augmented = cosup.augmented_training_data
    distance_matrix = compute_distance_matrix(cosup.cell_models, classification)
    return PreparedSynthTree(
        X_augmented=X_augmented,
        y_augmented=y_augmented,
        centroids=cosup.centroids,
        distance_matrix=distance_matrix,
        selected_teachers=cosup.selected_teacher_names,
    )


def fit_tree_for_depth(
    depth: int,
    X,
    y,
    classification: bool,
    teachers,
    selected_num_cells: int,
    num_aug: int,
    augmentation_covariance: str,
    local_model_alpha: float,
    local_model_max_iter: int,
    leaf_model_alpha: float,
    leaf_model_max_iter: int,
    random_state: int,
):
    prepared = prepare_synthtree(
        X=X,
        y=y,
        classification=classification,
        teachers=teachers,
        n_cells=selected_num_cells,
        num_aug=num_aug,
        augmentation_covariance=augmentation_covariance,
        local_model_alpha=local_model_alpha,
        local_model_max_iter=local_model_max_iter,
        random_state=random_state,
    )
    tree = DistanceDecisionTree(max_depth=depth, classification=classification)
    tree.fit(prepared.centroids, prepared.distance_matrix)
    tree.fit_leaf_models(
        prepared.X_augmented,
        prepared.y_augmented,
        lambda: make_statsmodels_model(
            classification=classification,
            alpha=leaf_model_alpha,
            max_iter=leaf_model_max_iter,
        ),
    )
    return tree


def fit_tree_full(
    X,
    y,
    classification: bool,
    teachers,
    selected_num_cells: int,
    num_aug: int,
    augmentation_covariance: str,
    local_model_alpha: float,
    local_model_max_iter: int,
    leaf_model_alpha: float,
    leaf_model_max_iter: int,
    random_state: int,
    max_depth: int,
):
    prepared = prepare_synthtree(
        X=X,
        y=y,
        classification=classification,
        teachers=teachers,
        n_cells=selected_num_cells,
        num_aug=num_aug,
        augmentation_covariance=augmentation_covariance,
        local_model_alpha=local_model_alpha,
        local_model_max_iter=local_model_max_iter,
        random_state=random_state,
    )
    tree = DistanceDecisionTree(max_depth=max_depth, classification=classification)
    tree.fit(prepared.centroids, prepared.distance_matrix)
    tree.fit_node_models(
        prepared.X_augmented,
        prepared.y_augmented,
        lambda: make_statsmodels_model(
            classification=classification,
            alpha=leaf_model_alpha,
            max_iter=leaf_model_max_iter,
        ),
    )
    return tree


def fit_synthtree_timed(X_train, y_train, X_test, y_test, classification: bool, seed: int):
    teachers = {"RF": make_rf_teacher(classification)}
    params = {
        "n_cells_grid": (5, 10, 15, 20, 25, 30),
        "num_aug": 100,
        "augmentation_covariance": "diag",
        "local_model_alpha": 0.1,
        "local_model_max_iter": 200,
        "leaf_model_alpha": 0.1,
        "leaf_model_max_iter": 1000,
        "max_depth": 4,
        "pruning_cv": 10,
        "tree_sizer": "cc_prune" if classification else "l_trim",
    }

    t0 = time.perf_counter()
    selected_num_cells, selection_info = select_num_cells(
        X=X_train,
        y=y_train,
        teachers=teachers,
        n_cells_grid=params["n_cells_grid"],
        classification=classification,
        selection="validation_mlm",
        num_aug=params["num_aug"],
        covariance_type=params["augmentation_covariance"],
        local_model_alpha=params["local_model_alpha"],
        local_model_max_iter=params["local_model_max_iter"],
        random_state=seed,
    )
    j_selection_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    tree_size_info = select_tree_size(
        X=X_train,
        y=y_train,
        classification=classification,
        max_depth=params["max_depth"],
        method=params["tree_sizer"],
        fit_depth_fn=lambda depth, X_fold, y_fold: _TreeWrapper(
            fit_tree_for_depth(
                depth=depth,
                X=X_fold,
                y=y_fold,
                classification=classification,
                teachers=teachers,
                selected_num_cells=int(selected_num_cells),
                num_aug=params["num_aug"],
                augmentation_covariance=params["augmentation_covariance"],
                local_model_alpha=params["local_model_alpha"],
                local_model_max_iter=params["local_model_max_iter"],
                leaf_model_alpha=params["leaf_model_alpha"],
                leaf_model_max_iter=params["leaf_model_max_iter"],
                random_state=seed,
            )
        ),
        fit_full_tree_fn=lambda X_fold, y_fold: _TreeWrapper(
            fit_tree_full(
                X=X_fold,
                y=y_fold,
                classification=classification,
                teachers=teachers,
                selected_num_cells=int(selected_num_cells),
                num_aug=params["num_aug"],
                augmentation_covariance=params["augmentation_covariance"],
                local_model_alpha=params["local_model_alpha"],
                local_model_max_iter=params["local_model_max_iter"],
                leaf_model_alpha=params["leaf_model_alpha"],
                leaf_model_max_iter=params["leaf_model_max_iter"],
                random_state=seed,
                max_depth=params["max_depth"],
            )
        ),
        cv=params["pruning_cv"],
        random_state=seed,
    )
    tree_size_selection_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    prepared = prepare_synthtree(
        X=X_train,
        y=y_train,
        classification=classification,
        teachers=teachers,
        n_cells=int(selected_num_cells),
        num_aug=params["num_aug"],
        augmentation_covariance=params["augmentation_covariance"],
        local_model_alpha=params["local_model_alpha"],
        local_model_max_iter=params["local_model_max_iter"],
        random_state=seed,
    )
    final_cosup_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    if params["tree_sizer"] == "cc_prune":
        full_tree = DistanceDecisionTree(max_depth=params["max_depth"], classification=classification)
        full_tree.fit(prepared.centroids, prepared.distance_matrix)
        full_tree.fit_node_models(
            prepared.X_augmented,
            prepared.y_augmented,
            lambda: make_statsmodels_model(
                classification=classification,
                alpha=params["leaf_model_alpha"],
                max_iter=params["leaf_model_max_iter"],
            ),
        )
        path = full_tree.weakest_link_pruning_path(prepared.X_augmented, prepared.y_augmented)
        tree = _select_subtree_for_alpha(path, float(tree_size_info["alpha"]))
        selected_depth = int(tree.final_depth)
    else:
        tree = DistanceDecisionTree(max_depth=int(tree_size_info["depth"]), classification=classification)
        tree.fit(prepared.centroids, prepared.distance_matrix)
        tree.fit_leaf_models(
            prepared.X_augmented,
            prepared.y_augmented,
            lambda: make_statsmodels_model(
                classification=classification,
                alpha=params["leaf_model_alpha"],
                max_iter=params["leaf_model_max_iter"],
            ),
        )
        selected_depth = int(tree_size_info["depth"])
    final_tree_fit_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    test_scores = tree.predict_scores(X_test)
    predict_time_sec = time.perf_counter() - t0

    total_fit_sec = j_selection_sec + tree_size_selection_sec + final_cosup_sec + final_tree_fit_sec
    return {
        "score": score_predictions(y_test, test_scores, classification),
        "fit_time_sec": total_fit_sec,
        "predict_time_sec": predict_time_sec,
        "selected_num_cells": int(selected_num_cells),
        "selected_depth": selected_depth,
        "selected_alpha": tree_size_info["alpha"],
        "breakdown": {
            "j_selection_sec": j_selection_sec,
            "tree_size_selection_sec": tree_size_selection_sec,
            "final_cosup_sec": final_cosup_sec,
            "final_tree_fit_sec": final_tree_fit_sec,
            "total_fit_sec": total_fit_sec,
        },
    }


class _TreeWrapper:
    def __init__(self, tree):
        self.tree_ = tree

    def predict_scores(self, X):
        return self.tree_.predict_scores(X)


def fit_cart_timed(X_train, y_train, X_test, y_test, classification: bool):
    model = DecisionTreeClassifier(random_state=0) if classification else DecisionTreeRegressor(random_state=0)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    if classification:
        scores = model.predict_proba(X_test)[:, 1]
    else:
        scores = model.predict(X_test)
    predict_time_sec = time.perf_counter() - t0

    return {
        "score": score_predictions(y_test, scores, classification),
        "fit_time_sec": fit_time_sec,
        "predict_time_sec": predict_time_sec,
    }


def fit_lrt_timed(X_train, y_train, X_test, y_test, classification: bool):
    from co_supervision_test import HAVE_RPY2, LRTClassifier, LRTRegressor

    if not HAVE_RPY2:
        raise RuntimeError("rpy2 / Rforestry is not available in this environment")
    model = LRTClassifier() if classification else LRTRegressor()
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    if classification:
        scores = model.predict_proba(X_test)[:, 1]
    else:
        scores = model.predict(X_test)
    predict_time_sec = time.perf_counter() - t0

    return {
        "score": score_predictions(y_test, scores, classification),
        "fit_time_sec": fit_time_sec,
        "predict_time_sec": predict_time_sec,
    }


def fit_mlm_epic_core(
    X_train,
    y_train,
    classification: bool,
    seed: int,
    initial_cells: int,
    num_aug: int,
    max_iter: int,
):
    from generalized_mlm import MixtureLinearModel

    teacher = make_rf_teacher(classification)
    teacher.fit(X_train, y_train)
    mlm = MixtureLinearModel(base_model=[(teacher, "RF")], verbose=False)
    mlm.compute_kmeans_CELL(X_train, K=min(initial_cells, X_train.shape[0] - 1), verbose=False, random_seed=seed)
    mlm.fit_LocalModels(
        X_train,
        y_train,
        eps=0.001,
        num_noise_samp=num_aug,
        classification=classification,
        alpha=0.1,
        max_iter=max_iter,
        verbose=False,
        random_seed=seed,
        statsmodels=True,
    )
    mlm.compute_LocalModelsDist()
    return mlm


def fit_mlm_epic_timed(X_train, y_train, X_test, y_test, classification: bool, seed: int):
    initial_cells = 10
    num_aug = 100
    max_iter = 200
    epic_grid = (2, 3, 4, 5, 6)

    X_sub, X_val, y_sub, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=seed,
        stratify=y_train if classification else None,
    )

    t0 = time.perf_counter()
    mlm_sub = fit_mlm_epic_core(
        X_train=X_sub,
        y_train=y_sub,
        classification=classification,
        seed=seed,
        initial_cells=initial_cells,
        num_aug=num_aug,
        max_iter=max_iter,
    )
    best_jtilde = None
    best_metric = None
    for jtilde in epic_grid:
        if jtilde > mlm_sub.Ktilde:
            continue
        mlm_sub.fit_MergedLocalModels(
            Jtilde=jtilde,
            dist_mat_avg=mlm_sub.dist_mat_avg,
            classification=classification,
            alpha=0.1,
            max_iter=max_iter,
            verbose=False,
            random_seed=seed,
            statsmodels=True,
        )
        val_scores = np.asarray(mlm_sub.predict(X_val, merged=True), dtype=float)
        metric = score_predictions(y_val, val_scores, classification)
        if best_jtilde is None:
            best_jtilde = jtilde
            best_metric = metric
            continue
        if classification and metric > best_metric:
            best_jtilde = jtilde
            best_metric = metric
        if (not classification) and metric < best_metric:
            best_jtilde = jtilde
            best_metric = metric
    epic_selection_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    mlm = fit_mlm_epic_core(
        X_train=X_train,
        y_train=y_train,
        classification=classification,
        seed=seed,
        initial_cells=initial_cells,
        num_aug=num_aug,
        max_iter=max_iter,
    )
    mlm.fit_MergedLocalModels(
        Jtilde=int(best_jtilde),
        dist_mat_avg=mlm.dist_mat_avg,
        classification=classification,
        alpha=0.1,
        max_iter=max_iter,
        verbose=False,
        random_seed=seed,
        statsmodels=True,
    )
    final_fit_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    test_scores = np.asarray(mlm.predict(X_test, merged=True), dtype=float)
    predict_time_sec = time.perf_counter() - t0

    total_fit_sec = epic_selection_sec + final_fit_sec
    return {
        "score": score_predictions(y_test, test_scores, classification),
        "fit_time_sec": total_fit_sec,
        "predict_time_sec": predict_time_sec,
        "selected_num_epics": int(best_jtilde),
        "breakdown": {
            "epic_selection_sec": epic_selection_sec,
            "final_fit_sec": final_fit_sec,
            "total_fit_sec": total_fit_sec,
        },
    }


def run_method(method_name: str, X_train, y_train, X_test, y_test, classification: bool, seed: int):
    if method_name == "SynthTree":
        return fit_synthtree_timed(X_train, y_train, X_test, y_test, classification, seed)
    if method_name == "CART":
        return fit_cart_timed(X_train, y_train, X_test, y_test, classification)
    if method_name == "LRT":
        return fit_lrt_timed(X_train, y_train, X_test, y_test, classification)
    if method_name == "MLM-EPIC":
        return fit_mlm_epic_timed(X_train, y_train, X_test, y_test, classification, seed)
    raise ValueError(f"Unknown method: {method_name}")


def summarize_runtime(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["dataset", "task", "method"])
        .agg(
            fit_time_mean=("fit_time_sec", "mean"),
            fit_time_std=("fit_time_sec", "std"),
            predict_time_mean=("predict_time_sec", "mean"),
            predict_time_std=("predict_time_sec", "std"),
            score_mean=("score", "mean"),
            score_std=("score", "std"),
        )
        .reset_index()
    )


def summarize_breakdown(breakdown_rows: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [col for col in breakdown_rows.columns if col.endswith("_sec")]
    grouped = breakdown_rows.groupby(["dataset", "task", "method"])
    summary = grouped[metric_cols].agg(["mean", "std"])
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.to_flat_index()]
    return summary.reset_index()


def main():
    parser = argparse.ArgumentParser(description="Runtime benchmark for SynthTree and comparator methods")
    parser.add_argument("--runs", "-r", type=int, default=3, help="Number of repeated train/test splits")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        choices=sorted(CLASS_DATASETS | REG_DATASETS),
        help="Datasets to evaluate",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        choices=DEFAULT_METHODS,
        help="Methods to benchmark",
    )
    args = parser.parse_args()

    methods = list(args.methods)
    if "LRT" in methods:
        try:
            from co_supervision_test import HAVE_RPY2
        except Exception:
            HAVE_RPY2 = False
        if not HAVE_RPY2:
            methods = [method for method in methods if method != "LRT"]
            print("Skipping LRT because rpy2 / Rforestry is unavailable.")

    jobs = [(dataset_name, seed, method_name) for dataset_name in args.datasets for seed in range(args.runs) for method_name in methods]
    rows = []
    breakdown_rows = []

    iterator = jobs
    if tqdm is not None:
        iterator = tqdm(jobs, desc="Runtime benchmark", unit="job")

    for dataset_name, seed, method_name in iterator:
        classification = task_is_classification(dataset_name)
        X_train, y_train, X_test, y_test = prep_data(dataset_name, random_state=seed)
        if classification:
            y_train, y_test = normalize_binary_labels(y_train, y_test)

        result = run_method(
            method_name=method_name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            classification=classification,
            seed=seed,
        )
        rows.append(
            {
                "dataset": dataset_name,
                "task": "classification" if classification else "regression",
                "method": method_name,
                "seed": seed,
                "fit_time_sec": result["fit_time_sec"],
                "predict_time_sec": result["predict_time_sec"],
                "score": result["score"],
            }
        )
        if "breakdown" in result:
            breakdown_row = {
                "dataset": dataset_name,
                "task": "classification" if classification else "regression",
                "method": method_name,
                "seed": seed,
            }
            breakdown_row.update(result["breakdown"])
            breakdown_rows.append(breakdown_row)

        message = (
            f"{dataset_name} | seed={seed} | method={method_name} "
            f"| fit={result['fit_time_sec']:.2f}s | score={result['score']:.4f}"
        )
        if tqdm is not None:
            tqdm.write(message)
        else:
            print(message)

    results = pd.DataFrame(rows)
    results.to_csv("runtime_results.csv", index=False)
    summary = summarize_runtime(results)
    summary.to_csv("runtime_summary.csv", index=False)

    if breakdown_rows:
        breakdown = pd.DataFrame(breakdown_rows)
        breakdown.to_csv("runtime_breakdown_results.csv", index=False)
        breakdown_summary = summarize_breakdown(breakdown)
        breakdown_summary.to_csv("runtime_breakdown_summary.csv", index=False)
        print(breakdown_summary)

    print(summary)


if __name__ == "__main__":
    main()
