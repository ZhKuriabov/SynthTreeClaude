from __future__ import annotations

import argparse
import warnings

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from example_preprocessing import prep_data
from synthtree.co_supervision import build_co_supervision, select_num_cells
from synthtree.metrics import score_predictions

warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


CLASS_DATASETS = {"SKCM", "Road Safety", "Compas", "Upselling"}
REG_DATASETS = {"Cal Housing", "Bike Sharing", "Abalone", "Servo"}
DEFAULT_DATASETS = ["SKCM", "Upselling", "Abalone", "Servo"]
DEFAULT_GRID = (5, 10, 20)
FIXED_J = 10


def task_is_classification(dataset_name: str) -> bool:
    return dataset_name in CLASS_DATASETS


def make_teacher(classification: bool):
    if classification:
        return RandomForestClassifier(n_estimators=100, random_state=0)
    return RandomForestRegressor(n_estimators=100, random_state=0)


def evaluate_strategy(
    dataset_name: str,
    strategy: str,
    seed: int,
    n_cells_grid: tuple[int, ...],
    num_aug: int,
    local_model_max_iter: int,
):
    classification = task_is_classification(dataset_name)
    X_train, y_train, X_test, y_test = prep_data(dataset_name, random_state=seed)
    if classification:
        classes = pd.Series(y_train).sort_values().unique()
        positive_class = classes[-1]
        y_train_internal = (y_train == positive_class).astype(int)
        y_test_internal = (y_test == positive_class).astype(int)
    else:
        y_train_internal = y_train
        y_test_internal = y_test
    teacher = make_teacher(classification)
    teachers = {"RF": clone(teacher)}

    if strategy == "fixed_j":
        selected_num_cells = FIXED_J
        selection_info = {"selection": "fixed_j", "results": [{"n_cells": FIXED_J, "metric": None, "objective": None}]}
    else:
        selected_num_cells, selection_info = select_num_cells(
            X=X_train,
            y=y_train_internal,
            teachers=teachers,
            n_cells_grid=n_cells_grid,
            classification=classification,
            selection=strategy,
            num_aug=num_aug,
            covariance_type="diag",
            local_model_alpha=0.1,
            local_model_max_iter=local_model_max_iter,
            random_state=seed,
        )

    cosup = build_co_supervision(
        X=X_train,
        y=y_train_internal,
        teachers=teachers,
        n_cells=selected_num_cells,
        classification=classification,
        num_aug=num_aug,
        covariance_type="diag",
        local_model_alpha=0.1,
        local_model_max_iter=local_model_max_iter,
        random_state=seed,
    )
    test_scores = cosup.predict_initial_mlm(X_test)
    test_metric = score_predictions(y_test_internal, test_scores, classification)
    return {
        "dataset": dataset_name,
        "task": "classification" if classification else "regression",
        "teacher": "RF",
        "strategy": strategy,
        "seed": seed,
        "selected_num_cells": int(selected_num_cells),
        "test_metric": float(test_metric),
        "selection_info": repr(selection_info),
        "teacher_assignments": repr(cosup.selected_teacher_names),
    }


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["dataset", "task", "teacher", "strategy"])
        .agg(
            selected_num_cells_mean=("selected_num_cells", "mean"),
            selected_num_cells_mode=("selected_num_cells", lambda s: int(pd.Series.mode(s).iloc[0])),
            test_metric_mean=("test_metric", "mean"),
            test_metric_std=("test_metric", "std"),
        )
        .reset_index()
    )


def main():
    parser = argparse.ArgumentParser(description="Run focused sensitivity experiments for the initial clustering step of SynthTree")
    parser.add_argument("--runs", "-r", type=int, default=3, help="Number of repeated train/test splits")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        choices=sorted(CLASS_DATASETS | REG_DATASETS),
        help="Datasets to evaluate",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["validation_mlm", "silhouette", "fixed_j"],
        choices=["validation_mlm", "silhouette", "fixed_j"],
        help="J-selection strategies to compare",
    )
    parser.add_argument(
        "--grid",
        nargs="+",
        type=int,
        default=list(DEFAULT_GRID),
        help="Candidate initial numbers of cells for the selection-based strategies",
    )
    parser.add_argument("--num-aug", type=int, default=100, help="Number of synthetic samples generated per cell")
    parser.add_argument("--local-model-max-iter", type=int, default=200, help="Maximum optimizer iterations for local cell models")
    args = parser.parse_args()

    n_cells_grid = tuple(sorted(set(args.grid)))
    jobs = [
        (dataset_name, strategy, seed)
        for dataset_name in args.datasets
        for strategy in args.strategies
        for seed in range(args.runs)
    ]

    rows = []
    iterator = jobs
    if tqdm is not None:
        iterator = tqdm(jobs, desc="J-sensitivity experiments", unit="job")

    for dataset_name, strategy, seed in iterator:
        row = evaluate_strategy(
            dataset_name=dataset_name,
            strategy=strategy,
            seed=seed,
            n_cells_grid=n_cells_grid,
            num_aug=args.num_aug,
            local_model_max_iter=args.local_model_max_iter,
        )
        rows.append(row)
        message = (
            f"{dataset_name} | strategy={strategy} | seed={seed} "
            f"| selected_J={row['selected_num_cells']} | test_metric={row['test_metric']:.4f}"
        )
        if tqdm is not None:
            tqdm.write(message)
        else:
            print(message)

    results = pd.DataFrame(rows)
    summary = summarize(results)
    results.to_csv("j_selection_sensitivity_results.csv", index=False)
    summary.to_csv("j_selection_sensitivity_summary.csv", index=False)
    print(summary)


if __name__ == "__main__":
    main()
