from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

from example_preprocessing import prep_data_with_feature_names
from generalized_mlm import MixtureLinearModel
from synthtree import SynthTreeClassifier


def _enumerate_leaves(node, path=None, leaves=None):
    if path is None:
        path = []
    if leaves is None:
        leaves = []
    if node.is_leaf:
        leaves.append({"leaf_id": len(leaves), "node": node, "path": list(path)})
        return leaves
    _enumerate_leaves(node.left, path + [(int(node.feature_index), float(node.threshold), "<=")], leaves)
    _enumerate_leaves(node.right, path + [(int(node.feature_index), float(node.threshold), ">")], leaves)
    return leaves


def _assign_rows_to_leaf_ids(tree, X: np.ndarray, leaves) -> np.ndarray:
    lookup = {id(leaf["node"]): leaf["leaf_id"] for leaf in leaves}
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


def _fit_models(seed: int):
    X_train, y_train_raw, X_test, y_test_raw, feature_names = prep_data_with_feature_names("SKCM", random_state=seed)
    classes = pd.Series(y_train_raw).sort_values().unique()
    positive_class = classes[-1]
    y_train = (y_train_raw == positive_class).astype(int)
    y_test = (y_test_raw == positive_class).astype(int)

    rf = RandomForestClassifier(n_estimators=100, random_state=0)
    rf.fit(X_train, y_train)

    synthtree = SynthTreeClassifier(
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
    synthtree.fit(X_train, y_train_raw)

    mlm, selected_num_epics = _fit_mlm_epic({"RF": rf}, X_train, y_train, seed)
    return X_train, y_train, X_test, y_test, feature_names, rf, synthtree, mlm, selected_num_epics


def _fit_mlm_epic(teacher_pool: dict[str, object], X_train, y_train, seed: int):
    teacher_list = [(teacher, name) for name, teacher in teacher_pool.items()]
    mlm = MixtureLinearModel(base_model=teacher_list, verbose=False)
    initial_cells = min(3, X_train.shape[0] - 1)
    mlm.compute_kmeans_CELL(X_train, K=initial_cells, verbose=False, random_seed=seed)
    mlm.fit_LocalModels(
        X_train,
        y_train,
        eps=0.001,
        num_noise_samp=100,
        classification=True,
        alpha=0.1,
        max_iter=1000,
        verbose=False,
        random_seed=seed,
        statsmodels=True,
    )
    mlm.compute_LocalModelsDist()

    if X_train.shape[0] < 10:
        X_sub, X_val, y_sub, y_val = X_train, X_train, y_train, y_train
    else:
        X_sub, X_val, y_sub, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=seed,
            stratify=y_train,
        )

    mlm_sub = MixtureLinearModel(base_model=teacher_list, verbose=False)
    mlm_sub.compute_kmeans_CELL(
        X_sub,
        K=min(initial_cells, X_sub.shape[0] - 1),
        verbose=False,
        random_seed=seed,
    )
    mlm_sub.fit_LocalModels(
        X_sub,
        y_sub,
        eps=0.001,
        num_noise_samp=100,
        classification=True,
        alpha=0.1,
        max_iter=1000,
        verbose=False,
        random_seed=seed,
        statsmodels=True,
    )
    mlm_sub.compute_LocalModelsDist()

    best_num_epics = None
    best_auc = None
    for num_epics in (2, 3, 4, 5):
        if num_epics > mlm_sub.Ktilde:
            continue
        mlm_sub.fit_MergedLocalModels(
            Jtilde=num_epics,
            dist_mat_avg=mlm_sub.dist_mat_avg,
            classification=True,
            alpha=0.1,
            max_iter=1000,
            verbose=False,
            random_seed=seed,
            statsmodels=True,
        )
        scores = np.asarray(mlm_sub.predict(X_val, merged=True), dtype=float)
        auc = float(roc_auc_score(y_val, scores))
        if best_auc is None or auc > best_auc:
            best_auc = auc
            best_num_epics = num_epics

    mlm.fit_MergedLocalModels(
        Jtilde=int(best_num_epics),
        dist_mat_avg=mlm.dist_mat_avg,
        classification=True,
        alpha=0.1,
        max_iter=1000,
        verbose=False,
        random_seed=seed,
        statsmodels=True,
    )
    return mlm, int(best_num_epics)


def _compute_lime_matrix(X_train, X_test, feature_names, y_train, rf):
    explainer = LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=["0", "1"],
        mode="classification",
        discretize_continuous=False,
        random_state=0,
    )
    coef_matrix = np.zeros((X_test.shape[0], X_test.shape[1]), dtype=float)
    detail_rows = []
    for i in range(X_test.shape[0]):
        exp = explainer.explain_instance(
            X_test[i],
            rf.predict_proba,
            num_features=X_test.shape[1],
            model_regressor=Lasso(alpha=0.001),
        )
        local_map = dict(exp.local_exp[1])
        for feat_idx, weight in local_map.items():
            coef_matrix[i, int(feat_idx)] = float(weight)
            detail_rows.append({"sample_index": i, "feature_index": int(feat_idx), "feature": feature_names[int(feat_idx)], "weight": float(weight)})
    return coef_matrix, pd.DataFrame(detail_rows)


def _plot_boxplots(top_features, lime_matrix, synth_coeffs, output_path: Path, title: str, xlabel: str):
    n_features = len(top_features)
    n_cols = 5
    n_rows = int(np.ceil(n_features / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8))
    axes = np.ravel(axes)
    for idx, feature in enumerate(top_features):
        ax = axes[idx]
        values = lime_matrix[:, idx]
        ax.boxplot(values, widths=0.55, patch_artist=True, boxprops=dict(facecolor="#DBEAFE", edgecolor="#1D4ED8"))
        overlay = synth_coeffs[feature].to_numpy(dtype=float)
        ax.scatter(np.repeat(1.0, len(overlay)), overlay, color="#B91C1C", s=34, zorder=3)
        ax.axhline(0.0, color="#9CA3AF", linewidth=0.9, linestyle="--")
        ax.set_title(feature, fontsize=10)
        ax.set_xticks([1])
        ax.set_xticklabels([xlabel], fontsize=9)
    for ax in axes[n_features:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _bic_curve(X: np.ndarray, max_components: int = 6):
    comps = np.arange(1, max_components + 1)
    bics = []
    for n in comps:
        model = GaussianMixture(n_components=n, covariance_type="full", random_state=0)
        model.fit(X)
        bics.append(model.bic(X))
    return comps, np.asarray(bics, dtype=float)


def _plot_bic_separate(top_features, lime_df, output_path: Path):
    fig, ax = plt.subplots(figsize=(10, 8))
    for feature in top_features:
        vals = lime_df[feature].to_numpy(dtype=float).reshape(-1, 1)
        comps, bics = _bic_curve(vals)
        ax.plot(comps, bics, marker="o", linestyle="--", label=feature)
    ax.set_xlabel("Number of components")
    ax.set_ylabel("BIC")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_bic_joint(lime_df, output_path: Path):
    comps, bics = _bic_curve(lime_df.to_numpy(dtype=float))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(comps, bics, marker="o", linestyle="--", color="#1D4ED8")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("BIC")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _top_features_from_permutation(rf, X_test, y_test, feature_names, k=10):
    result = permutation_importance(rf, X_test, y_test, n_repeats=30, random_state=0, scoring="roc_auc")
    ranking = pd.Series(result.importances_mean, index=feature_names).sort_values(ascending=False)
    return ranking.head(k).index.tolist(), ranking


def _plot_same_leaf_explanation(feature_order, synth_vals, lime_left, lime_right, left_idx, right_idx, output_path: Path):
    y = np.arange(len(feature_order))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=True)

    def colors(values):
        return ["forestgreen" if v > 0 else "firebrick" if v < 0 else "#D1D5DB" for v in values]

    for ax, lime_vals, sample_idx in zip(axes, [lime_left, lime_right], [left_idx, right_idx]):
        ax.barh(y - width / 2, synth_vals, height=width, color=colors(synth_vals), label="SynthTree", alpha=0.9)
        ax.barh(y + width / 2, lime_vals, height=width, color=colors(lime_vals), label="LIME", alpha=0.6, hatch="//", edgecolor="black")
        ax.axvline(0.0, color="#111827", linewidth=1.0)
        ax.set_title(f"Sample {sample_idx}", fontsize=13)
        ax.set_yticks(y)
        ax.set_yticklabels(feature_order, fontsize=9)
        ax.grid(axis="x", linestyle="--", alpha=0.25)
    axes[0].invert_yaxis()
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("SynthTree vs LIME for two test points in the same leaf", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate maintained SKCM LIME/MLM manuscript figures.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="../../../draft/Images/LIME experiment")
    parser.add_argument("--lime-limit", type=int, default=None, help="Optional limit on the number of test points for a quicker smoke run.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test, feature_names, rf, synthtree, mlm, selected_num_epics = _fit_models(args.seed)
    top_features, perm_ranking = _top_features_from_permutation(rf, X_test, y_test, feature_names, k=10)
    top_idx = [feature_names.index(name) for name in top_features]

    X_test_lime = X_test if args.lime_limit is None else X_test[: args.lime_limit]
    lime_matrix, lime_long = _compute_lime_matrix(X_train, X_test_lime, feature_names, y_train, rf)
    lime_top = pd.DataFrame(lime_matrix[:, top_idx], columns=top_features)

    leaves = _enumerate_leaves(synthtree.tree_.tree)
    synth_coeff_df = pd.DataFrame(
        {
            feature: [_extract_coefficients(leaf["node"].model, feature_names)[feature] for leaf in leaves]
            for feature in top_features
        }
    )
    _plot_boxplots(
        top_features,
        lime_top.to_numpy(dtype=float),
        synth_coeff_df,
        out_dir / "ALL-Boxplots-ST-LIME (3).png",
        title="LIME local coefficients vs SynthTree leaf coefficients",
        xlabel="LIME",
    )

    mlm_coeff_df = pd.DataFrame(mlm.coef_EPIC[:, top_idx], columns=top_features)
    _plot_boxplots(
        top_features,
        mlm_coeff_df.to_numpy(dtype=float),
        synth_coeff_df,
        out_dir / "ALL-Boxplots-ST-MLM (2).png",
        title="MLM-EPIC coefficients vs SynthTree leaf coefficients",
        xlabel="MLM-EPIC",
    )

    _plot_bic_separate(top_features, lime_top, out_dir / "ALL-BIC-sep-new (4).png")
    _plot_bic_joint(lime_top, out_dir / "BIC-All-new (4).png")

    assignments = _assign_rows_to_leaf_ids(synthtree.tree_, X_test_lime, leaves)
    selected_leaf = None
    pair = None
    for leaf_id in np.unique(assignments):
        idx = np.where(assignments == leaf_id)[0]
        if idx.size >= 2:
            selected_leaf = int(leaf_id)
            pair = idx[:2]
            break
    if pair is not None:
        sample_left, sample_right = int(pair[0]), int(pair[1])
        synth_coeffs = _extract_coefficients(leaves[selected_leaf]["node"].model, feature_names)
        lime_left = pd.Series(lime_matrix[sample_left], index=feature_names)
        lime_right = pd.Series(lime_matrix[sample_right], index=feature_names)
        union_features = (
            pd.concat(
                [
                    synth_coeffs.abs(),
                    lime_left.abs(),
                    lime_right.abs(),
                ],
                axis=1,
            )
            .max(axis=1)
            .sort_values(ascending=False)
            .head(15)
            .index.tolist()
        )
        _plot_same_leaf_explanation(
            union_features,
            synth_coeffs[union_features].to_numpy(dtype=float),
            lime_left[union_features].to_numpy(dtype=float),
            lime_right[union_features].to_numpy(dtype=float),
            sample_left,
            sample_right,
            out_dir.parent / "same_leaf_explanation_colored.png",
        )
        path_rows = []
        for feature_idx, threshold, direction in leaves[selected_leaf]["path"]:
            path_rows.append(
                {
                    "leaf_id": selected_leaf,
                    "feature": feature_names[feature_idx],
                    "direction": direction,
                    "threshold": threshold,
                }
            )
        pd.DataFrame(path_rows).to_csv(out_dir / "same_leaf_path.csv", index=False)

    perm_ranking.rename("permutation_importance").to_csv(out_dir / "top_rf_features.csv")
    lime_long.to_csv(out_dir / "lime_coefficients_long.csv", index=False)
    pd.DataFrame(
        {
            "feature": top_features,
            "selected_num_epics": selected_num_epics,
        }
    ).to_csv(out_dir / "lime_analysis_metadata.csv", index=False)


if __name__ == "__main__":
    main()
