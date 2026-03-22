from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.patches import FancyBboxPatch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score

from example_preprocessing import prep_data_with_feature_names
from synthtree import SynthTreeClassifier, SynthTreeRegressor
from synthtree.metrics import rmse


@dataclass
class LeafMeta:
    leaf_id: int
    depth: int
    node: object
    path: list[tuple[int, float, str]]


def dataset_is_classification(name: str) -> bool:
    return name == "SKCM"


def _enumerate_leaves(node, path=None, leaves=None):
    if path is None:
        path = []
    if leaves is None:
        leaves = []
    if node.is_leaf:
        leaves.append(LeafMeta(len(leaves), int(node.depth), node, list(path)))
        return leaves
    left_rule = (int(node.feature_index), float(node.threshold), "<=")
    right_rule = (int(node.feature_index), float(node.threshold), ">")
    _enumerate_leaves(node.left, path + [left_rule], leaves)
    _enumerate_leaves(node.right, path + [right_rule], leaves)
    return leaves


def _assign_rows_to_leaf_ids(tree, X: np.ndarray, leaves: list[LeafMeta]) -> np.ndarray:
    lookup = {id(leaf.node): leaf.leaf_id for leaf in leaves}
    assignments = np.empty(X.shape[0], dtype=int)

    def visit(node, row_idx):
        if row_idx.size == 0:
            return
        if node.is_leaf:
            assignments[row_idx] = lookup[id(node)]
            return
        left_mask = X[row_idx, node.feature_index] <= node.threshold
        visit(node.left, row_idx[left_mask])
        visit(node.right, row_idx[~left_mask])

    visit(tree.tree, np.arange(X.shape[0], dtype=int))
    return assignments


def _extract_coefficients(model, feature_names: list[str]) -> pd.Series:
    params = getattr(getattr(model, "model_", None), "params", None)
    if params is None:
        return pd.Series(np.zeros(len(feature_names)), index=feature_names, dtype=float)
    values = np.asarray(params, dtype=float)
    coef = values[1 : 1 + len(feature_names)] if values.size > 1 else np.zeros(len(feature_names))
    if coef.size < len(feature_names):
        coef = np.pad(coef, (0, len(feature_names) - coef.size))
    return pd.Series(coef[: len(feature_names)], index=feature_names, dtype=float)


def _safe_confint(X_leaf: np.ndarray, y_leaf: np.ndarray, feature_names: list[str], classification: bool):
    if X_leaf.shape[0] < max(8, X_leaf.shape[1] + 2):
        return None
    try:
        exog = sm.add_constant(X_leaf, has_constant="add")
        if classification:
            if np.unique(y_leaf).size < 2:
                return None
            fit = sm.Logit(y_leaf.astype(int), exog).fit(disp=0, maxiter=200)
        else:
            fit = sm.OLS(y_leaf.astype(float), exog).fit()
        ci = fit.conf_int()
        arr = np.asarray(ci, dtype=float)
        if arr.shape[0] < len(feature_names) + 1:
            return None
        return pd.DataFrame(arr[1 : len(feature_names) + 1], index=feature_names, columns=["lower", "upper"])
    except Exception:
        return None


def _leaf_support(assignments: np.ndarray, num_leaves: int) -> np.ndarray:
    counts = np.bincount(assignments, minlength=num_leaves).astype(float)
    return counts


def _fit_case_study(dataset: str, seed: int):
    X_train, y_train, X_test, y_test, feature_names = prep_data_with_feature_names(dataset, random_state=seed)
    classification = dataset_is_classification(dataset)
    if classification:
        model = SynthTreeClassifier(
            teachers={"RF": RandomForestClassifier(n_estimators=100, random_state=0)},
            n_cells_grid=(2, 3),
            n_cells_selection="validation_mlm",
            num_aug=100,
            tree_sizer="cc_prune",
            max_depth=5,
            min_leaves=2,
            pruning_cv=10,
            random_state=seed,
        )
    else:
        model = SynthTreeRegressor(
            teachers={"RF": RandomForestRegressor(n_estimators=100, random_state=0)},
            n_cells_grid=(5, 10, 15, 20),
            n_cells_selection="validation_mlm",
            num_aug=100,
            tree_sizer="l_trim",
            max_depth=5,
            pruning_cv=10,
            random_state=seed,
        )
    model.fit(X_train, y_train)
    leaves = _enumerate_leaves(model.tree_.tree)
    train_assignments = _assign_rows_to_leaf_ids(model.tree_, X_train, leaves)
    test_assignments = _assign_rows_to_leaf_ids(model.tree_, X_test, leaves)
    train_support = _leaf_support(train_assignments, len(leaves))

    if classification:
        y_test_internal = (y_test == model.classes_[1]).astype(int)
        scores = model.predict_proba(X_test)[:, 1]
        preds = (scores >= 0.5).astype(int)
        leaf_metric = {}
        for leaf_id in range(len(leaves)):
            mask = test_assignments == leaf_id
            leaf_metric[leaf_id] = float(accuracy_score(y_test_internal[mask], preds[mask])) if np.any(mask) else np.nan
    else:
        preds = model.predict(X_test)
        leaf_metric = {}
        for leaf_id in range(len(leaves)):
            mask = test_assignments == leaf_id
            leaf_metric[leaf_id] = float(rmse(y_test[mask], preds[mask])) if np.any(mask) else np.nan

    return {
        "model": model,
        "feature_names": feature_names,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "leaves": leaves,
        "train_assignments": train_assignments,
        "test_assignments": test_assignments,
        "train_support": train_support,
        "leaf_metric": leaf_metric,
        "classification": classification,
    }


def _node_positions(node, depth=0, positions=None, next_x=None, max_depth=None):
    if positions is None:
        positions = {}
    if next_x is None:
        next_x = [0]
    if node.is_leaf or (max_depth is not None and depth >= max_depth):
        x = next_x[0]
        next_x[0] += 1
        positions[id(node)] = (x, -depth)
        return positions
    _node_positions(node.left, depth + 1, positions, next_x, max_depth=max_depth)
    _node_positions(node.right, depth + 1, positions, next_x, max_depth=max_depth)
    left_x, _ = positions[id(node.left)]
    right_x, _ = positions[id(node.right)]
    positions[id(node)] = ((left_x + right_x) / 2.0, -depth)
    return positions


def _plot_tree(tree_bundle, dataset: str, output_base: Path, max_display_depth: int | None = None):
    tree = tree_bundle["model"].tree_
    feature_names = tree_bundle["feature_names"]
    positions = _node_positions(tree.tree, max_depth=max_display_depth)
    leaves = tree_bundle["leaves"]
    leaf_lookup = {id(leaf.node): leaf for leaf in leaves}
    fig, ax = plt.subplots(figsize=(13, 6 if dataset == "SKCM" else 10))

    def draw(node):
        x, y = positions[id(node)]
        if node.is_leaf or (max_display_depth is not None and node.depth >= max_display_depth):
            leaf = leaf_lookup.get(id(node))
            if leaf is not None:
                support = int(tree_bundle["train_support"][leaf.leaf_id])
                metric = tree_bundle["leaf_metric"].get(leaf.leaf_id, np.nan)
                if tree_bundle["classification"]:
                    label = f"Leaf {leaf.leaf_id}\ntrain n={support}\nacc={metric:.2f}" if not math.isnan(metric) else f"Leaf {leaf.leaf_id}\ntrain n={support}"
                else:
                    label = f"Leaf {leaf.leaf_id}\ntrain n={support}\nrmse={metric:.2f}" if not math.isnan(metric) else f"Leaf {leaf.leaf_id}\ntrain n={support}"
            else:
                label = "..."
            box = FancyBboxPatch((x - 0.44, y - 0.22), 0.88, 0.44, boxstyle="round,pad=0.03", linewidth=1.0, edgecolor="#374151", facecolor="#F3F4F6")
            ax.add_patch(box)
            ax.text(x, y, label, ha="center", va="center", fontsize=8.5)
            return

        label = f"{feature_names[int(node.feature_index)]}\n<= {float(node.threshold):.3f}"
        box = FancyBboxPatch((x - 0.55, y - 0.26), 1.10, 0.52, boxstyle="round,pad=0.03", linewidth=1.1, edgecolor="#7C2D12", facecolor="#FEE2E2")
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", fontsize=8.5)
        for child in [node.left, node.right]:
            cx, cy = positions[id(child)]
            ax.plot([x, cx], [y - 0.26, cy + 0.22], color="#6B7280", linewidth=1.0)
            draw(child)

    draw(tree.tree)
    ax.set_title(f"SynthTree for {dataset}", fontsize=16)
    ax.axis("off")
    xs = [pos[0] for pos in positions.values()]
    ys = [pos[1] for pos in positions.values()]
    ax.set_xlim(min(xs) - 1, max(xs) + 1)
    ax.set_ylim(min(ys) - 1, 1)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmap(tree_bundle, dataset: str, output_path: Path):
    feature_names = tree_bundle["feature_names"]
    leaves = tree_bundle["leaves"]
    coeffs = []
    labels = []
    order = np.argsort(-tree_bundle["train_support"])
    if dataset == "Bike Sharing":
        order = order[:5]
    for idx in order:
        leaf = leaves[int(idx)]
        coeffs.append(_extract_coefficients(leaf.node.model, feature_names).to_numpy(dtype=float))
        labels.append(f"Leaf {leaf.leaf_id}")
    coeff_mat = np.vstack(coeffs)
    vmax = np.nanmax(np.abs(coeff_mat)) if coeff_mat.size else 1.0
    fig, ax = plt.subplots(figsize=(15, 4 if dataset == "SKCM" else 3.5))
    im = ax.imshow(coeff_mat, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=90, fontsize=8)
    ax.set_title(f"{dataset}: leaf-wise coefficients")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _preferred_features(dataset: str, feature_names: list[str]) -> list[str]:
    if dataset == "SKCM":
        candidates = [
            "AGE",
            "SEX",
            "RADIATION_TREATMENT_ADJUVANT",
            "HISTORY_OTHER_MALIGNANCY",
        ]
    else:
        candidates = ["hr", "holiday", "workingday", "weekday"]
    chosen = [name for name in candidates if name in feature_names]
    if len(chosen) < 4:
        extras = [name for name in feature_names if name not in chosen][: 4 - len(chosen)]
        chosen.extend(extras)
    return chosen[:4]


def _plot_selected_coefficients(tree_bundle, dataset: str, output_path: Path):
    feature_names = tree_bundle["feature_names"]
    selected = _preferred_features(dataset, feature_names)
    leaves = tree_bundle["leaves"]
    x_axis = np.arange(len(leaves))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()
    assignments = _assign_rows_to_leaf_ids(tree_bundle["model"].tree_, tree_bundle["model"].X_augmented_, leaves)
    X_aug = tree_bundle["model"].X_augmented_
    y_aug = tree_bundle["model"].y_augmented_

    for ax, feature in zip(axes, selected):
        feature_idx = feature_names.index(feature)
        coeffs = []
        lowers = []
        uppers = []
        for leaf in leaves:
            coeff_series = _extract_coefficients(leaf.node.model, feature_names)
            coeff = float(coeff_series[feature])
            coeffs.append(coeff)
            mask = assignments == leaf.leaf_id
            ci = _safe_confint(
                X_aug[mask][:, [feature_idx]],
                y_aug[mask],
                [feature],
                classification=tree_bundle["classification"],
            )
            if ci is None:
                lowers.append(np.nan)
                uppers.append(np.nan)
            else:
                lowers.append(float(ci.loc[feature, "lower"]))
                uppers.append(float(ci.loc[feature, "upper"]))
        coeffs = np.asarray(coeffs, dtype=float)
        lowers = np.asarray(lowers, dtype=float)
        uppers = np.asarray(uppers, dtype=float)
        yerr = np.vstack([
            np.where(np.isnan(lowers), 0.0, coeffs - lowers),
            np.where(np.isnan(uppers), 0.0, uppers - coeffs),
        ])
        ax.errorbar(x_axis, coeffs, yerr=yerr, fmt="o", color="#B33C26", ecolor="#1F2937", capsize=3, linewidth=1.2)
        ax.axhline(0.0, color="#9CA3AF", linewidth=1.0, linestyle="--")
        ax.set_title(feature, fontsize=11)
        ax.set_xticks(x_axis)
        ax.set_xticklabels([f"L{leaf.leaf_id}" for leaf in leaves], rotation=0, fontsize=9)
    for ax in axes[len(selected):]:
        ax.axis("off")
    fig.suptitle(f"{dataset}: leaf-wise coefficients with approximate confidence intervals", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_leaf_count_summary(skcm_bundle, bike_bundle, output_path: Path):
    rows = []
    for dataset, bundle in [("SKCM", skcm_bundle), ("Bike Sharing", bike_bundle)]:
        rows.append(
            {
                "dataset": dataset,
                "num_leaves": bundle["model"].tree_.count_leaves(),
                "selected_depth": bundle["model"].selected_depth_,
                "selected_num_cells": bundle["model"].selected_num_cells_,
                "tree_sizer": bundle["model"].tree_sizer_,
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Generate maintained SKCM/Bike case-study figures for the manuscript.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="../../../draft/Images")
    parser.add_argument("--bike-display-depth", type=int, default=3)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["SKCM", "Bike Sharing"],
        default=["SKCM", "Bike Sharing"],
        help="Subset of manuscript case studies to generate.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lime_dir = out_dir / "LIME experiment"
    lime_dir.mkdir(parents=True, exist_ok=True)

    bundles = {}
    for dataset in args.datasets:
        bundles[dataset] = _fit_case_study(dataset, seed=args.seed)

    if "SKCM" in bundles:
        _plot_tree(bundles["SKCM"], "SKCM", out_dir / "SKCM_tree", max_display_depth=None)
        _plot_heatmap(bundles["SKCM"], "SKCM", out_dir / "SKCM_heatmap_2.png")
        _plot_selected_coefficients(bundles["SKCM"], "SKCM", out_dir / "SKCMRegression.png")
    if "Bike Sharing" in bundles:
        _plot_tree(
            bundles["Bike Sharing"],
            "Bike Sharing",
            out_dir / "BikeDataTree",
            max_display_depth=args.bike_display_depth,
        )
        _plot_heatmap(bundles["Bike Sharing"], "Bike Sharing", out_dir / "Bike_Sharing_heatmap_2.png")
        _plot_selected_coefficients(bundles["Bike Sharing"], "Bike Sharing", out_dir / "BikeRegression.png")

    if {"SKCM", "Bike Sharing"}.issubset(bundles):
        _save_leaf_count_summary(bundles["SKCM"], bundles["Bike Sharing"], out_dir / "case_study_leaf_counts.csv")


if __name__ == "__main__":
    main()
