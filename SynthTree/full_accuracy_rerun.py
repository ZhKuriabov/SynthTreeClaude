from __future__ import annotations

import argparse
import warnings
from dataclasses import asdict, dataclass
import os

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from example_preprocessing import prep_data
from synthtree import SynthTreeClassifier, SynthTreeRegressor
from synthtree.co_supervision import build_co_supervision, select_num_cells
from synthtree.metrics import rmse

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    from co_supervision_test import (
        HAVE_RPY2,
        LRFClassifier,
        LRFRegressor,
        LRTClassifier,
        LRTRegressor,
        AutoLRFClassifier,
        AutoLRFRegressor,
        CVLRTClassifier,
        CVLRTRegressor,
    )
except Exception:
    HAVE_RPY2 = False
    LRFClassifier = None
    LRFRegressor = None
    LRTClassifier = None
    LRTRegressor = None
    AutoLRFClassifier = None
    AutoLRFRegressor = None
    CVLRTClassifier = None
    CVLRTRegressor = None


CLASS_DATASETS = ["SKCM", "Road Safety", "Compas", "Upselling"]
REG_DATASETS = ["Cal Housing", "Bike Sharing", "Abalone", "Servo"]
ALL_DATASETS = CLASS_DATASETS + REG_DATASETS
SINGLE_TEACHERS = ["MLP", "RF", "GB", "LRF"]
TEACHER_SETTINGS = ["INT"] + SINGLE_TEACHERS

# Dataset-adaptive J grids: chosen so that n_train/J >= ~20-30 original
# points per cell, ranging from tens to hundreds as stated in the manuscript.
DATASET_J_GRIDS = {
    "Servo":        (5, 10),
    "SKCM":         (5, 10, 15),
    "Abalone":      (10, 20, 30, 50),
    "Upselling":    (10, 20, 30, 50),
    "Compas":       (20, 50, 80, 100),
    "Bike Sharing": (20, 50, 80, 100),
    "Cal Housing":  (20, 50, 80, 100),
    "Road Safety":  (50, 100, 150, 200),
}
DEFAULT_J_GRID = (5, 10, 15, 20)


@dataclass
class FairnessConfig:
    tune_black_box_teachers: bool = True
    tune_cart: bool = True
    tune_lr: bool = True
    synthtree_n_cells_selection: str = "validation_mlm"
    cosup_baseline_n_cells_selection: str = "validation_mlm"
    n_cells_grid: tuple[int, ...] = (5, 10, 15, 20)
    num_aug: int = 100
    cv_folds: int = 10
    mlm_epic_grid: tuple[int, ...] = (2, 3, 4, 5, 6)
    local_model_max_iter: int = 1000
    random_state: int = 0


def task_is_classification(dataset_name: str) -> bool:
    return dataset_name in CLASS_DATASETS


def normalize_binary_labels(y_train, y_test):
    classes = pd.Series(y_train).sort_values().unique()
    positive_class = classes[-1]
    return (y_train == positive_class).astype(int), (y_test == positive_class).astype(int)


def score_predictions(y_true, scores, classification: bool) -> float:
    if classification:
        return float(roc_auc_score(y_true, scores))
    return float(rmse(y_true, scores))


def splitter(classification: bool, cv_folds: int, random_state: int):
    if classification:
        return StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    return KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)


def teacher_prototype(name: str, classification: bool, random_state: int):
    if name == "RF":
        return (
            RandomForestClassifier(random_state=random_state)
            if classification
            else RandomForestRegressor(random_state=random_state)
        )
    if name == "GB":
        return (
            GradientBoostingClassifier(random_state=random_state)
            if classification
            else GradientBoostingRegressor(random_state=random_state)
        )
    if name == "MLP":
        return (
            MLPClassifier(max_iter=1000, random_state=random_state)
            if classification
            else MLPRegressor(max_iter=1000, random_state=random_state)
        )
    if name == "LRF":
        if not HAVE_RPY2:
            raise RuntimeError("LRF requires rpy2 / Rforestry.")
        return AutoLRFClassifier() if classification else AutoLRFRegressor()
    raise ValueError(f"Unknown teacher: {name}")


def tune_teacher(name: str, X, y, classification: bool, cfg: FairnessConfig, random_state: int):
    estimator = teacher_prototype(name, classification, random_state)
    if (not cfg.tune_black_box_teachers) or name == "LRF":
        estimator.fit(X, y)
        return estimator

    scoring = "roc_auc" if classification else "neg_root_mean_squared_error"
    cv = splitter(classification, cfg.cv_folds, random_state)

    if name == "RF":
        grid = {
            "n_estimators": [100, 300],
            "max_depth": [None, 10],
        }
    elif name == "GB":
        grid = {
            "n_estimators": [100, 300],
            "learning_rate": [0.05, 0.1],
        }
    elif name == "MLP":
        grid = {
            "hidden_layer_sizes": [(50,), (100,)],
            "alpha": [1e-4, 1e-3],
        }
    else:
        estimator.fit(X, y)
        return estimator

    search = GridSearchCV(estimator, grid, scoring=scoring, cv=cv, n_jobs=1)
    search.fit(X, y)
    return search.best_estimator_


def fit_lr_baseline(X, y, classification: bool, cfg: FairnessConfig, random_state: int):
    if classification:
        if cfg.tune_lr:
            model = LogisticRegressionCV(
                Cs=[0.1, 1.0, 10.0],
                cv=cfg.cv_folds,
                scoring="roc_auc",
                solver="lbfgs",
                max_iter=1000,
                random_state=random_state,
            )
        else:
            model = LogisticRegression(max_iter=1000, random_state=random_state)
    else:
        if cfg.tune_lr:
            model = RidgeCV(alphas=[0.1, 1.0, 10.0])
        else:
            model = LinearRegression()
    model.fit(X, y)
    return model


def cart_alpha_candidates(X, y, classification: bool):
    if classification:
        base = DecisionTreeClassifier(random_state=0)
    else:
        base = DecisionTreeRegressor(random_state=0)
    path = base.cost_complexity_pruning_path(X, y)
    alphas = np.unique(path.ccp_alphas)
    if alphas.size == 0:
        return np.array([0.0])
    if alphas.size > 25:
        keep = np.linspace(0, alphas.size - 1, 25, dtype=int)
        alphas = alphas[keep]
    return np.unique(np.append(alphas, 0.0))


def fit_cart(X, y, classification: bool, cfg: FairnessConfig, random_state: int):
    if not cfg.tune_cart:
        model = DecisionTreeClassifier(random_state=random_state) if classification else DecisionTreeRegressor(random_state=random_state)
        model.fit(X, y)
        return model

    cv = splitter(classification, cfg.cv_folds, random_state)
    alpha_grid = cart_alpha_candidates(X, y, classification)
    best_alpha = 0.0
    best_score = -np.inf

    for alpha in alpha_grid:
        fold_scores = []
        for train_idx, val_idx in cv.split(X, y):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            model = DecisionTreeClassifier(random_state=random_state, ccp_alpha=float(alpha)) if classification else DecisionTreeRegressor(random_state=random_state, ccp_alpha=float(alpha))
            model.fit(X_train, y_train)
            if classification:
                scores = model.predict_proba(X_val)[:, 1]
            else:
                scores = model.predict(X_val)
            metric = score_predictions(y_val, scores, classification)
            fold_scores.append(metric if classification else -metric)
        mean_score = float(np.mean(fold_scores))
        if mean_score > best_score:
            best_score = mean_score
            best_alpha = float(alpha)

    model = DecisionTreeClassifier(random_state=random_state, ccp_alpha=best_alpha) if classification else DecisionTreeRegressor(random_state=random_state, ccp_alpha=best_alpha)
    model.fit(X, y)
    return model


def model_scores(model, X, classification: bool):
    if classification:
        if hasattr(model, "predict_proba"):
            return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
        return np.asarray(model.predict(X), dtype=float)
    return np.asarray(model.predict(X), dtype=float)


def cart_interpretability(model) -> float:
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


def make_teacher_pool(X_train, y_train, classification: bool, cfg: FairnessConfig, seed: int):
    teachers = {}
    for teacher_name in SINGLE_TEACHERS:
        if teacher_name == "LRF" and not HAVE_RPY2:
            continue
        teachers[teacher_name] = tune_teacher(teacher_name, X_train, y_train, classification, cfg, seed)
    return teachers


def fit_mlm_epic(
    teacher_pool: dict[str, object],
    X_train,
    y_train,
    X_test,
    classification: bool,
    cfg: FairnessConfig,
    seed: int,
):
    from generalized_mlm import MixtureLinearModel

    teacher_list = [(teacher, name) for name, teacher in teacher_pool.items()]
    mlm = MixtureLinearModel(base_model=teacher_list, verbose=False)
    initial_cells = min(max(2, max(cfg.n_cells_grid)), X_train.shape[0] - 1)
    mlm.compute_kmeans_CELL(X_train, K=initial_cells, verbose=False, random_seed=seed)
    mlm.fit_LocalModels(
        X_train,
        y_train,
        eps=0.001,
        num_noise_samp=cfg.num_aug,
        classification=classification,
        alpha=0.1,
        max_iter=cfg.local_model_max_iter,
        verbose=False,
        random_seed=seed,
        statsmodels=True,
    )
    mlm.compute_LocalModelsDist()

    X_sub, X_val, y_sub, y_val = (
        (X_train, X_train, y_train, y_train)
        if X_train.shape[0] < 10
        else __import__("sklearn.model_selection", fromlist=["train_test_split"]).train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=seed,
            stratify=y_train if classification else None,
        )
    )

    mlm_sub = MixtureLinearModel(base_model=teacher_list, verbose=False)
    mlm_sub.compute_kmeans_CELL(X_sub, K=min(initial_cells, X_sub.shape[0] - 1), verbose=False, random_seed=seed)
    mlm_sub.fit_LocalModels(
        X_sub,
        y_sub,
        eps=0.001,
        num_noise_samp=cfg.num_aug,
        classification=classification,
        alpha=0.1,
        max_iter=cfg.local_model_max_iter,
        verbose=False,
        random_seed=seed,
        statsmodels=True,
    )
    mlm_sub.compute_LocalModelsDist()

    best_jtilde = None
    best_metric = None
    for jtilde in cfg.mlm_epic_grid:
        if jtilde > mlm_sub.Ktilde:
            continue
        mlm_sub.fit_MergedLocalModels(
            Jtilde=jtilde,
            dist_mat_avg=mlm_sub.dist_mat_avg,
            classification=classification,
            alpha=0.1,
            max_iter=cfg.local_model_max_iter,
            verbose=False,
            random_seed=seed,
            statsmodels=True,
        )
        val_scores = np.asarray(mlm_sub.predict(X_val, merged=True), dtype=float)
        metric = score_predictions(y_val, val_scores, classification)
        objective = metric if classification else -metric
        if best_metric is None or objective > best_metric:
            best_metric = objective
            best_jtilde = jtilde

    mlm.fit_MergedLocalModels(
        Jtilde=int(best_jtilde),
        dist_mat_avg=mlm.dist_mat_avg,
        classification=classification,
        alpha=0.1,
        max_iter=cfg.local_model_max_iter,
        verbose=False,
        random_seed=seed,
        statsmodels=True,
    )
    test_scores = np.asarray(mlm.predict(X_test, merged=True), dtype=float)
    return mlm, test_scores, int(best_jtilde)


def select_cosup_n_cells(X_train, y_train, teachers, classification: bool, cfg: FairnessConfig, seed: int):
    if cfg.cosup_baseline_n_cells_selection == "fixed":
        return min(max(2, 10), X_train.shape[0] - 1)
    selected_j, _ = select_num_cells(
        X=X_train,
        y=y_train,
        teachers=teachers,
        n_cells_grid=cfg.n_cells_grid,
        classification=classification,
        selection=cfg.cosup_baseline_n_cells_selection,
        num_aug=cfg.num_aug,
        covariance_type="diag",
        local_model_alpha=0.1,
        local_model_max_iter=cfg.local_model_max_iter,
        random_state=seed,
    )
    return int(selected_j)


def build_augmented_dataset(X_train, y_train, teachers, classification: bool, cfg: FairnessConfig, seed: int):
    selected_j = select_cosup_n_cells(X_train, y_train, teachers, classification, cfg, seed)
    cosup = build_co_supervision(
        X=X_train,
        y=y_train,
        teachers=teachers,
        n_cells=selected_j,
        classification=classification,
        num_aug=cfg.num_aug,
        covariance_type="diag",
        local_model_alpha=0.1,
        local_model_max_iter=cfg.local_model_max_iter,
        random_state=seed,
    )
    X_aug, y_aug = cosup.augmented_training_data
    return X_aug, y_aug, selected_j


def run_single_split(dataset_name: str, seed: int, cfg: FairnessConfig):
    classification = task_is_classification(dataset_name)
    X_train, y_train, X_test, y_test = prep_data(dataset_name, random_state=seed)
    if classification:
        y_train, y_test = normalize_binary_labels(y_train, y_test)

    rows = []
    teachers = make_teacher_pool(X_train, y_train, classification, cfg, seed)

    for teacher_name, teacher in teachers.items():
        score = score_predictions(y_test, model_scores(teacher, X_test, classification), classification)
        rows.append(
            {
                "dataset": dataset_name,
                "task": "classification" if classification else "regression",
                "table_group": "baseline",
                "method_family": "black_box",
                "teacher_setting": "",
                "model": teacher_name,
                "seed": seed,
                "score": score,
                "interpretability": np.nan,
                "selected_num_cells": np.nan,
                "selected_num_epics": np.nan,
            }
        )

    lr_model = fit_lr_baseline(X_train, y_train, classification, cfg, seed)
    cart_model = fit_cart(X_train, y_train, classification, cfg, seed)
    baseline_models = {
        "LR": lr_model,
        "CART": cart_model,
    }
    if HAVE_RPY2:
        baseline_models["LRT"] = CVLRTClassifier(cv=3) if classification else CVLRTRegressor(cv=3)
        baseline_models["LRT"].fit(X_train, y_train)

    for model_name, model in baseline_models.items():
        score = score_predictions(y_test, model_scores(model, X_test, classification), classification)
        interp = cart_interpretability(model) if model_name == "CART" else np.nan
        rows.append(
            {
                "dataset": dataset_name,
                "task": "classification" if classification else "regression",
                "table_group": "baseline",
                "method_family": "explainable",
                "teacher_setting": "",
                "model": model_name,
                "seed": seed,
                "score": score,
                "interpretability": interp,
                "selected_num_cells": np.nan,
                "selected_num_epics": np.nan,
            }
        )

    synth_cls = SynthTreeClassifier if classification else SynthTreeRegressor
    for teacher_setting in TEACHER_SETTINGS:
        if teacher_setting != "INT" and teacher_setting not in teachers:
            continue
        teacher_pool = teachers if teacher_setting == "INT" else {teacher_setting: teachers[teacher_setting]}

        synth_model = synth_cls(
            teachers={name: clone(model) for name, model in teacher_pool.items()},
            tree_sizer="auto",
            n_cells_selection=cfg.synthtree_n_cells_selection,
            n_cells_grid=cfg.n_cells_grid,
            num_aug=cfg.num_aug,
            local_model_max_iter=cfg.local_model_max_iter,
            leaf_model_max_iter=cfg.local_model_max_iter,
        )
        synth_model.fit(X_train, y_train)
        synth_score = score_predictions(y_test, synth_model.predict_proba(X_test)[:, 1] if classification else synth_model.predict(X_test), classification)
        rows.append(
            {
                "dataset": dataset_name,
                "task": "classification" if classification else "regression",
                "table_group": "co_supervision",
                "method_family": "explainable",
                "teacher_setting": teacher_setting,
                "model": f"SynthTree-{teacher_setting}",
                "seed": seed,
                "score": synth_score,
                "interpretability": float(synth_model.interpretability_),
                "selected_num_cells": int(synth_model.selected_num_cells_),
                "selected_num_epics": np.nan,
            }
        )

        X_aug, y_aug, selected_j = build_augmented_dataset(X_train, y_train, teacher_pool, classification, cfg, seed)

        cart_cosup = fit_cart(X_aug, y_aug, classification, cfg, seed)
        cart_score = score_predictions(y_test, model_scores(cart_cosup, X_test, classification), classification)
        rows.append(
            {
                "dataset": dataset_name,
                "task": "classification" if classification else "regression",
                "table_group": "co_supervision",
                "method_family": "explainable",
                "teacher_setting": teacher_setting,
                "model": f"CART-{teacher_setting}",
                "seed": seed,
                "score": cart_score,
                "interpretability": cart_interpretability(cart_cosup),
                "selected_num_cells": selected_j,
                "selected_num_epics": np.nan,
            }
        )

        if HAVE_RPY2:
            lrt_cosup = CVLRTClassifier(cv=3) if classification else CVLRTRegressor(cv=3)
            lrt_cosup.fit(X_aug, y_aug)
            lrt_score = score_predictions(y_test, model_scores(lrt_cosup, X_test, classification), classification)
            rows.append(
                {
                    "dataset": dataset_name,
                    "task": "classification" if classification else "regression",
                    "table_group": "co_supervision",
                    "method_family": "explainable",
                    "teacher_setting": teacher_setting,
                    "model": f"LRT-{teacher_setting}",
                    "seed": seed,
                    "score": lrt_score,
                    "interpretability": np.nan,
                    "selected_num_cells": selected_j,
                    "selected_num_epics": np.nan,
                }
            )

        mlm, mlm_scores, selected_epics = fit_mlm_epic(teacher_pool, X_train, y_train, X_test, classification, cfg, seed)
        del mlm
        rows.append(
            {
                "dataset": dataset_name,
                "task": "classification" if classification else "regression",
                "table_group": "co_supervision",
                "method_family": "explainable",
                "teacher_setting": teacher_setting,
                "model": f"MLM-EPIC-{teacher_setting}",
                "seed": seed,
                "score": score_predictions(y_test, mlm_scores, classification),
                "interpretability": np.nan,
                "selected_num_cells": selected_j,
                "selected_num_epics": selected_epics,
            }
        )

    return rows


def summarize_results(results: pd.DataFrame):
    baseline_summary = (
        results[results["table_group"] == "baseline"]
        .groupby(["dataset", "task", "method_family", "model"])
        .agg(score_mean=("score", "mean"), score_sd=("score", "std"))
        .reset_index()
    )

    cosup_summary = (
        results[results["table_group"] == "co_supervision"]
        .groupby(["dataset", "task", "model"])
        .agg(
            score_mean=("score", "mean"),
            score_sd=("score", "std"),
            interp_mean=("interpretability", "mean"),
            selected_num_cells_mode=("selected_num_cells", lambda s: int(pd.Series.mode(s.dropna()).iloc[0]) if s.dropna().size else np.nan),
            selected_num_epics_mode=("selected_num_epics", lambda s: int(pd.Series.mode(s.dropna()).iloc[0]) if s.dropna().size else np.nan),
        )
        .reset_index()
    )
    return baseline_summary, cosup_summary


def save_table_views(results: pd.DataFrame):
    baseline = results[results["table_group"] == "baseline"].copy()
    cosup = results[results["table_group"] == "co_supervision"].copy()

    baseline_class = baseline[baseline["task"] == "classification"].copy()
    baseline_reg = baseline[baseline["task"] == "regression"].copy()
    cosup_class = cosup[cosup["task"] == "classification"].copy()
    cosup_reg = cosup[cosup["task"] == "regression"].copy()

    baseline_class.groupby(["dataset", "model"]).agg(mean=("score", "mean"), sd=("score", "std")).reset_index().to_csv(
        "table1_baseline_classification_long.csv", index=False
    )
    baseline_reg.groupby(["dataset", "model"]).agg(mean=("score", "mean"), sd=("score", "std")).reset_index().to_csv(
        "table2_baseline_regression_long.csv", index=False
    )
    cosup_class.groupby(["dataset", "model"]).agg(mean=("score", "mean"), sd=("score", "std"), interp=("interpretability", "mean")).reset_index().to_csv(
        "table1_new_cosup_classification_long.csv", index=False
    )
    cosup_reg.groupby(["dataset", "model"]).agg(mean=("score", "mean"), sd=("score", "std"), interp=("interpretability", "mean")).reset_index().to_csv(
        "table2_new_cosup_regression_long.csv", index=False
    )


def main():
    parser = argparse.ArgumentParser(description="Full rerun for main accuracy tables with optional CV-based fairness.")
    parser.add_argument("--runs", "-r", type=int, default=5, help="Number of repeated train/test splits")
    parser.add_argument("--datasets", nargs="*", default=ALL_DATASETS, choices=ALL_DATASETS, help="Datasets to evaluate")
    parser.add_argument("--tune-black-box-teachers", action=argparse.BooleanOptionalAction, default=True, help="Tune RF/GB/MLP teachers by 10-fold CV (default: on)")
    parser.add_argument("--tune-cart", action=argparse.BooleanOptionalAction, default=True, help="Tune CART by cost-complexity 10-fold CV (default: on)")
    parser.add_argument("--tune-lr", action=argparse.BooleanOptionalAction, default=True, help="Use 10-fold CV-selected regularized linear/logistic baseline (default: on)")
    parser.add_argument("--synthtree-n-cells-selection", default="validation_mlm", choices=["validation_mlm", "silhouette"], help="J selection rule for SynthTree")
    parser.add_argument("--cosup-baseline-n-cells-selection", default="validation_mlm", choices=["validation_mlm", "silhouette", "fixed"], help="J selection rule for co-supervised baseline augmentation")
    parser.add_argument("--n-cells-grid", nargs="*", type=int, default=None, help="Candidate J values (default: dataset-adaptive)")
    parser.add_argument("--num-aug", type=int, default=100, help="Synthetic points per cell")
    parser.add_argument("--cv-folds", type=int, default=10, help="CV folds for tuning")
    parser.add_argument("--local-model-max-iter", type=int, default=1000, help="Max iterations for co-supervised statsmodels fits")
    args = parser.parse_args()

    explicit_grid = None
    if args.n_cells_grid is not None:
        explicit_grid = tuple(sorted(set(v for v in args.n_cells_grid if v > 1)))

    base_cfg = FairnessConfig(
        tune_black_box_teachers=args.tune_black_box_teachers,
        tune_cart=args.tune_cart,
        tune_lr=args.tune_lr,
        synthtree_n_cells_selection=args.synthtree_n_cells_selection,
        cosup_baseline_n_cells_selection=args.cosup_baseline_n_cells_selection,
        n_cells_grid=explicit_grid or DEFAULT_J_GRID,
        num_aug=args.num_aug,
        cv_folds=args.cv_folds,
        local_model_max_iter=args.local_model_max_iter,
    )

    jobs = [(dataset_name, seed) for dataset_name in args.datasets for seed in range(args.runs)]
    iterator = jobs
    if tqdm is not None:
        iterator = tqdm(jobs, desc="Main accuracy rerun", unit="job")

    all_rows = []
    for dataset_name, seed in iterator:
        from dataclasses import replace as _dc_replace
        if explicit_grid is not None:
            cfg = base_cfg
        else:
            cfg = _dc_replace(base_cfg, n_cells_grid=DATASET_J_GRIDS.get(dataset_name, DEFAULT_J_GRID))
        rows = run_single_split(dataset_name, seed, cfg)
        all_rows.extend(rows)
        if tqdm is None:
            print(f"{dataset_name} | seed={seed} | rows={len(rows)}")

    results = pd.DataFrame(all_rows)

    # Use dataset-specific filenames when running a single dataset
    if len(args.datasets) == 1:
        tag = args.datasets[0].replace(" ", "_")
        results_file = f"main_accuracy_results_{tag}.csv"
        baseline_file = f"main_accuracy_baseline_{tag}.csv"
        cosup_file = f"main_accuracy_cosup_{tag}.csv"
        config_file = f"main_accuracy_config_{tag}.csv"
    else:
        results_file = "main_accuracy_results.csv"
        baseline_file = "main_accuracy_baseline_summary.csv"
        cosup_file = "main_accuracy_cosup_summary.csv"
        config_file = "main_accuracy_config.csv"

    results.to_csv(results_file, index=False)
    baseline_summary, cosup_summary = summarize_results(results)
    baseline_summary.to_csv(baseline_file, index=False)
    cosup_summary.to_csv(cosup_file, index=False)
    save_table_views(results)

    pd.DataFrame([asdict(base_cfg)]).to_csv(config_file, index=False)
    print(baseline_summary)
    print(cosup_summary)


if __name__ == "__main__":
    main()
