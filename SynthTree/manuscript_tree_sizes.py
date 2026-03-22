from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import co_supervision_test as legacy_rtrees
from example_preprocessing import prep_data
from synthtree import SynthTreeClassifier, SynthTreeRegressor


CLASS_DATASETS = ["SKCM", "Road Safety", "Compas", "Upselling"]
REG_DATASETS = ["Cal Housing", "Bike Sharing", "Abalone", "Servo"]
ALL_DATASETS = CLASS_DATASETS + REG_DATASETS


def task_is_classification(dataset: str) -> bool:
    return dataset in CLASS_DATASETS


def normalize_binary_labels(y):
    classes = np.sort(pd.Series(y).unique())
    return (y == classes[-1]).astype(int)


def lrt_leaf_count(model) -> int:
    if not legacy_rtrees.HAVE_RPY2:
        return np.nan
    legacy_rtrees.ro.globalenv["model"] = model.model_
    result = legacy_rtrees.ro.r("get_tree_leaf_count_and_depth(model)")
    return int(np.asarray(result, dtype=float)[0])


def make_teacher_pool(classification: bool):
    teachers = {
        "RF": RandomForestClassifier(n_estimators=100, random_state=0)
        if classification
        else RandomForestRegressor(n_estimators=100, random_state=0),
        "GB": GradientBoostingClassifier(random_state=0)
        if classification
        else GradientBoostingRegressor(random_state=0),
        "MLP": MLPClassifier(max_iter=1000, random_state=0)
        if classification
        else MLPRegressor(max_iter=1000, random_state=0),
    }
    if legacy_rtrees.HAVE_RPY2:
        teachers["LRF"] = legacy_rtrees.LRFClassifier() if classification else legacy_rtrees.LRFRegressor()
    return teachers


def run_one(dataset: str, seed: int):
    classification = task_is_classification(dataset)
    X_train, y_train, _, _ = prep_data(dataset, random_state=seed)
    if classification:
        y_train = normalize_binary_labels(y_train)

    rows = []

    cart = DecisionTreeClassifier(random_state=seed) if classification else DecisionTreeRegressor(random_state=seed)
    cart.fit(X_train, y_train)
    rows.append(
        {
            "dataset": dataset,
            "seed": seed,
            "method": "CART",
            "leaf_count": int(cart.tree_.n_leaves),
        }
    )

    if legacy_rtrees.HAVE_RPY2:
        lrt = legacy_rtrees.LRTClassifier() if classification else legacy_rtrees.LRTRegressor()
        lrt.fit(X_train, y_train)
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "method": "LRT",
                "leaf_count": lrt_leaf_count(lrt),
            }
        )

    teachers = make_teacher_pool(classification)
    synthtree = (
        SynthTreeClassifier(
            teachers=teachers,
            tree_sizer="auto",
            n_cells_selection="validation_mlm",
            random_state=seed,
        )
        if classification
        else SynthTreeRegressor(
            teachers=teachers,
            tree_sizer="auto",
            n_cells_selection="validation_mlm",
            random_state=seed,
        )
    )
    synthtree.fit(X_train, y_train)
    rows.append(
        {
            "dataset": dataset,
            "seed": seed,
            "method": "SynthTree-INT",
            "leaf_count": int(synthtree.tree_.count_leaves()),
        }
    )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate maintained tree-size table source for the manuscript.")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--datasets", nargs="+", choices=ALL_DATASETS, default=ALL_DATASETS)
    parser.add_argument("--output-dir", type=str, default="manuscript_outputs")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in range(args.runs):
        for dataset in args.datasets:
            rows.extend(run_one(dataset, seed))

    results = pd.DataFrame(rows)
    summary = (
        results.groupby(["dataset", "method"], dropna=False)
        .agg(leaf_count_mean=("leaf_count", "mean"), leaf_count_sd=("leaf_count", "std"))
        .reset_index()
    )
    results.to_csv(out_dir / "manuscript_tree_sizes_results.csv", index=False)
    summary.to_csv(out_dir / "manuscript_tree_sizes_summary.csv", index=False)


if __name__ == "__main__":
    main()
