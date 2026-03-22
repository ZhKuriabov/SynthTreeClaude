from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from example_preprocessing import prep_data_with_feature_names
from synthtree import SynthTreeClassifier
from synthtree.analysis import (
    synthtree_feature_importance,
    feature_set_permutation_test,
)


def _normalize_binary_labels(y_train, y_test):
    classes = pd.Series(y_train).sort_values().unique()
    positive_class = classes[-1]
    return (y_train == positive_class).astype(int), (y_test == positive_class).astype(int)


def _mean_abs_shap(model, X: np.ndarray) -> np.ndarray:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        values = shap_values[-1]
    else:
        values = np.asarray(shap_values)
        if values.ndim == 3:
            values = values[:, :, -1]
    values = np.asarray(values, dtype=float)
    return np.mean(np.abs(values), axis=0)


def _normalize_nonnegative(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), 0.0, None)
    total = clipped.sum()
    return clipped / total if total > 0 else clipped


def _top_feature_set(values: pd.Series, k: int) -> set[str]:
    return set(values.sort_values(ascending=False).head(k).index.tolist())


def _top_feature_indices(values: pd.Series, feature_names: list[str], k: int) -> np.ndarray:
    top_names = values.sort_values(ascending=False).head(k).index.tolist()
    lookup = {name: idx for idx, name in enumerate(feature_names)}
    return np.asarray([lookup[name] for name in top_names if name in lookup], dtype=int)


def run_one_seed(
    seed: int,
    n_cells_grid: tuple[int, ...],
    num_aug: int,
    tree_sizer: str,
    max_depth: int,
    min_leaves: int,
    pruning_cv: int,
    top_k: int,
    perm_test_repeats: int,
    perm_test_null_draws: int,
):
    """Run the full STFI / perm-importance / SHAP analysis for one train/test split."""
    X_train, y_train_raw, X_test, y_test_raw, feature_names = prep_data_with_feature_names(
        "SKCM", random_state=seed
    )
    y_train, y_test = _normalize_binary_labels(y_train_raw, y_test_raw)

    print(f"  [seed={seed}] Fitting RF...")
    rf = RandomForestClassifier(n_estimators=100, random_state=0)
    rf.fit(X_train, y_train)
    rf_scores = rf.predict_proba(X_test)[:, 1]

    print(f"  [seed={seed}] Fitting SynthTree-RF...")
    synthtree = SynthTreeClassifier(
        teachers={"RF": RandomForestClassifier(n_estimators=100, random_state=0)},
        n_cells_grid=n_cells_grid,
        n_cells_selection="validation_mlm",
        num_aug=num_aug,
        tree_sizer=tree_sizer,
        max_depth=max_depth,
        min_leaves=min_leaves,
        pruning_cv=pruning_cv,
        random_state=seed,
    )
    synthtree.fit(X_train, y_train_raw)
    synthtree_scores = synthtree.predict_proba(X_test)[:, 1]

    print(f"  [seed={seed}] Computing STFI, perm importance, SHAP...")
    stfi_df, leaf_df = synthtree_feature_importance(synthtree.tree_, X_train, feature_names)

    perm_result = permutation_importance(
        rf, X_test, y_test, n_repeats=30, random_state=seed, scoring="roc_auc",
    )
    perm_norm = _normalize_nonnegative(perm_result.importances_mean)
    shap_raw = _mean_abs_shap(rf, X_test)
    shap_norm = _normalize_nonnegative(shap_raw)

    # Build per-feature comparison table
    comparison = pd.DataFrame({"feature": feature_names})
    comparison = comparison.merge(
        stfi_df.rename(columns={"importance": "stfi_raw", "importance_norm": "stfi_norm"}),
        on="feature", how="left",
    )
    comparison["stfi_raw"] = comparison["stfi_raw"].fillna(0.0)
    comparison["stfi_norm"] = comparison["stfi_norm"].fillna(0.0)
    comparison["perm_raw"] = perm_result.importances_mean
    comparison["perm_norm"] = perm_norm
    comparison["shap_raw"] = shap_raw
    comparison["shap_norm"] = shap_norm
    comparison["stfi_rank"] = comparison["stfi_norm"].rank(ascending=False, method="min").astype(int)
    comparison["perm_rank"] = comparison["perm_norm"].rank(ascending=False, method="min").astype(int)
    comparison["shap_rank"] = comparison["shap_norm"].rank(ascending=False, method="min").astype(int)
    comparison["seed"] = seed

    # Permutation test
    stfi_series = comparison.set_index("feature")["stfi_norm"]
    perm_series = comparison.set_index("feature")["perm_norm"]
    shap_series = comparison.set_index("feature")["shap_norm"]

    ranking_specs = [
        ("STFI", stfi_series, synthtree, "SynthTree"),
        ("Perm. Importance (RF)", perm_series, rf, "RF"),
        ("SHAP (RF)", shap_series, rf, "RF"),
    ]
    evaluation_models = [
        ("RF", rf),
        ("SynthTree", synthtree),
    ]

    perm_rows = []
    for ranking_label, ranking_series, _, source_label in ranking_specs:
        feature_indices = _top_feature_indices(ranking_series, feature_names, top_k)
        feature_list = [feature_names[int(idx)] for idx in feature_indices]
        for eval_label, eval_model in evaluation_models:
            summary_stats, _ = feature_set_permutation_test(
                eval_model, X_test, y_test, feature_indices,
                n_repeats=perm_test_repeats,
                n_null_draws=perm_test_null_draws,
                random_state=seed * 1000 + hash(ranking_label) % 10000,
            )
            perm_rows.append({
                "seed": seed,
                "ranking_method": ranking_label,
                "evaluation_model": eval_label,
                "top_k": top_k,
                "delta_auc": summary_stats["observed_mean_auc_drop"],
                "p_value": summary_stats["empirical_p_value"],
                "features": ", ".join(feature_list),
            })

    seed_summary = {
        "seed": seed,
        "rf_auc": float(roc_auc_score(y_test, rf_scores)),
        "synthtree_auc": float(roc_auc_score(y_test, synthtree_scores)),
        "synthtree_num_leaves": int(synthtree.tree_.count_leaves()),
    }

    return comparison, pd.DataFrame(perm_rows), seed_summary, leaf_df


def main():
    parser = argparse.ArgumentParser(
        description="Multi-seed STFI vs RF permutation importance vs RF SHAP (R2.C13)."
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                        help="Train/test split seeds (default: 0 1 2 3 4)")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--pruning-cv", type=int, default=10)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--min-leaves", type=int, default=1)
    parser.add_argument("--num-aug", type=int, default=50)
    parser.add_argument("--n-cells-grid", nargs="+", type=int, default=[5, 10])
    parser.add_argument("--tree-sizer", type=str, choices=["auto", "l_trim", "cc_prune"], default="l_trim")
    parser.add_argument("--perm-test-repeats", type=int, default=30)
    parser.add_argument("--perm-test-null-draws", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default="r2c13_outputs")
    parser.add_argument("--figure-path", type=str,
                        default="../../draft/Images/LIME experiment/STFI_RF_SHAP_comparison.png")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = Path(args.figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    all_comparisons = []
    all_perm_tests = []
    all_summaries = []
    all_leaves = []

    for seed in args.seeds:
        print(f"=== Seed {seed} ===")
        comparison, perm_df, seed_summary, leaf_df = run_one_seed(
            seed=seed,
            n_cells_grid=tuple(args.n_cells_grid),
            num_aug=args.num_aug,
            tree_sizer=args.tree_sizer,
            max_depth=args.max_depth,
            min_leaves=args.min_leaves,
            pruning_cv=args.pruning_cv,
            top_k=args.top_k,
            perm_test_repeats=args.perm_test_repeats,
            perm_test_null_draws=args.perm_test_null_draws,
        )
        all_comparisons.append(comparison)
        all_perm_tests.append(perm_df)
        all_summaries.append(seed_summary)
        leaf_df["seed"] = seed
        all_leaves.append(leaf_df)

    # Save per-seed raw results
    full_df = pd.concat(all_comparisons, ignore_index=True)
    full_df.to_csv(output_dir / "r2c13_feature_importance_all_seeds.csv", index=False)

    perm_test_df = pd.concat(all_perm_tests, ignore_index=True)
    perm_test_df.to_csv(output_dir / "r2c13_feature_permutation_test_all_seeds.csv", index=False)

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(output_dir / "r2c13_seed_summaries.csv", index=False)

    pd.concat(all_leaves, ignore_index=True).to_csv(
        output_dir / "r2c13_feature_importance_leaves_all_seeds.csv", index=False
    )

    # Aggregate permutation test across seeds: mean and SD of delta_auc and p_value
    perm_agg = (
        perm_test_df.groupby(["ranking_method", "evaluation_model"])
        .agg(
            delta_auc_mean=("delta_auc", "mean"),
            delta_auc_sd=("delta_auc", "std"),
            p_value_mean=("p_value", "mean"),
            p_value_sd=("p_value", "std"),
            n_seeds=("seed", "count"),
        )
        .reset_index()
    )
    perm_agg.to_csv(output_dir / "r2c13_permutation_test_summary.csv", index=False)

    # Aggregate AUC summaries
    print("\n=== Per-seed AUC summaries ===")
    print(summary_df.to_string(index=False))
    print(f"\nRF  AUC: {summary_df['rf_auc'].mean():.3f} (±{summary_df['rf_auc'].std():.3f})")
    print(f"ST  AUC: {summary_df['synthtree_auc'].mean():.3f} (±{summary_df['synthtree_auc'].std():.3f})")
    print(f"Leaves:  {summary_df['synthtree_num_leaves'].mean():.1f} (±{summary_df['synthtree_num_leaves'].std():.1f})")

    print("\n=== Permutation test summary (mean over seeds) ===")
    print(perm_agg.to_string(index=False))

    # Plot using seed 0 (or first available seed) for the figure
    seed0_comparison = full_df[full_df["seed"] == args.seeds[0]]
    perm_top = seed0_comparison.sort_values("perm_norm", ascending=False).head(args.top_k).copy()
    _plot_top_features(perm_top, figure_path)
    print(f"\nSaved figure (seed={args.seeds[0]}) to {figure_path}")
    print(f"Saved all outputs to {output_dir}/")


def _plot_top_features(display_df: pd.DataFrame, output_path: Path):
    ordered = display_df.copy().reset_index(drop=True)
    method_specs = [
        ("stfi_norm", "SynthTree STFI", "#B33C26", -0.2, "o"),
        ("perm_norm", "RF permutation", "#2D6A4F", 0.0, "s"),
        ("shap_norm", "RF SHAP", "#4361EE", 0.2, "D"),
    ]

    top_row = ordered.iloc[[0]].copy()
    rest = ordered.iloc[1:].copy()

    fig = plt.figure(figsize=(10.2, 7.2))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.1, 5.2], hspace=0.06)
    ax_top = fig.add_subplot(grid[0, 0])
    ax_bottom = fig.add_subplot(grid[1, 0], sharex=None)

    def draw_panel(ax, panel_df, show_labels):
        y_base = np.arange(len(panel_df))
        for column, label, color, offset, marker in method_specs:
            values = panel_df[column].to_numpy(dtype=float)
            y = y_base + offset
            ax.hlines(y, 0.0, values, color=color, linewidth=2.2, alpha=0.35)
            ax.scatter(
                values, y, s=58, color=color, marker=marker,
                edgecolors="white", linewidths=0.9, label=label, zorder=3,
            )
        ax.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.35)
        ax.set_axisbelow(True)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.tick_params(axis="both", labelsize=11)
        ax.set_yticks(y_base)
        if show_labels:
            ax.set_yticklabels(panel_df["feature"])
        else:
            ax.set_yticklabels([])
        return y_base

    top_y = draw_panel(ax_top, top_row, show_labels=True)
    bottom_y = draw_panel(ax_bottom, rest, show_labels=True)

    ax_top.set_yticks(top_y)
    ax_top.set_yticklabels(top_row["feature"], fontsize=11)
    ax_bottom.set_yticks(bottom_y)
    ax_bottom.set_yticklabels(rest["feature"], fontsize=11)
    ax_top.invert_yaxis()
    ax_bottom.invert_yaxis()

    top_max = max(float(top_row[col].max()) for col, *_ in method_specs)
    bottom_max = max(float(rest[col].max()) for col, *_ in method_specs)
    ax_top.set_xlim(0.0, top_max * 1.08)
    ax_bottom.set_xlim(0.0, bottom_max * 1.18)
    ax_top.set_xticks(np.linspace(0.0, round(top_max * 1.05, 1), 4))
    ax_bottom.set_xticks(np.linspace(0.0, round(bottom_max * 1.15, 2), 5))

    ax_top.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_bottom.set_xlabel("Normalized importance", fontsize=12)
    ax_top.set_title("SKCM RF case study: global feature summaries", fontsize=17, pad=10)

    legend_handles = []
    for _, label, color, _, marker in method_specs:
        handle = plt.Line2D(
            [0], [0], color=color, linewidth=2.2, marker=marker,
            markersize=7.5, markerfacecolor=color, markeredgecolor="white",
            markeredgewidth=0.9, label=label,
        )
        legend_handles.append(handle)
    ax_top.legend(
        handles=legend_handles, frameon=False, ncol=3,
        loc="upper center", bbox_to_anchor=(0.5, 1.18),
        fontsize=11, columnspacing=1.4, handletextpad=0.6,
    )

    d = 0.009
    kwargs = dict(transform=ax_top.transAxes, color="#444444", clip_on=False, linewidth=1.0)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs = dict(transform=ax_bottom.transAxes, color="#444444", clip_on=False, linewidth=1.0)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    fig.subplots_adjust(left=0.28, right=0.98, top=0.92, bottom=0.08, hspace=0.06)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
