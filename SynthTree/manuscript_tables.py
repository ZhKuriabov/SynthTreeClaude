from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CLASS_DATASETS = ["SKCM", "Road Safety", "Compas", "Upselling"]
REG_DATASETS = ["Cal Housing", "Bike Sharing", "Abalone", "Servo"]
BASELINE_MODELS = ["RF", "GB", "MLP", "LRF", "CART", "LR", "LRT"]
TEACHER_SUFFIXES = ["INT", "MLP", "RF", "GB", "LRF"]
COSUP_FAMILIES = ["SynthTree", "CART", "LRT", "MLM-EPIC"]
DATASET_INFO = [
    ("SKCM", 388, 34, 73),
    ("Bike Sharing", 17379, 12, 16),
    ("Compas", 16644, 17, 17),
    ("Abalone", 4177, 10, 10),
    ("Road Safety", 111762, 32, 32),
    ("Upselling", 5032, 45, 45),
    ("Servo", 167, 4, 19),
    ("Cal Housing", 20640, 8, 8),
]


def fmt_score(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def fmt_mean_sd(mean: float, sd: float, digits: int = 3) -> str:
    if pd.isna(mean):
        return "-"
    if pd.isna(sd):
        return fmt_score(mean, digits)
    return f"{float(mean):.{digits}f} ($\\pm${float(sd):.{digits}f})"


def write_text(path: Path, text: str):
    path.write_text(text + "\n", encoding="utf-8")


def latex_dataset_info() -> str:
    lines = [
        "\\begin{tabular}{c|c|c|c}",
        "\\toprule",
        "\\multirow{2}{*}{Data} & No. & No. Orig. & No. Trans. \\\\",
        " & Samples & Variables & Variables\\\\",
        "\\hline",
    ]
    for dataset, n, orig, trans in DATASET_INFO:
        lines.append(f"{dataset} & {n} & {orig} & {trans} \\\\")
        lines.append("\\hline")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def load_csv(base_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(base_dir / name)


def latex_baseline_table(df: pd.DataFrame, datasets: list[str], metric_label: str) -> str:
    lines = [
        "\\begin{tabular}{l|" + "|".join(["cc"] * len(datasets)) + "}",
        "\\toprule",
        "\\multirow{2}{*}{Model} & "
        + " & ".join([f"\\multicolumn{{2}}{{c{'|' if i < len(datasets)-1 else ''}}}{{{dataset}}}" for i, dataset in enumerate(datasets)])
        + " \\\\",
        "\\cmidrule(){2-" + str(1 + 2 * len(datasets)) + "}",
        "& " + " & ".join([f"Mean & SD" for _ in datasets]) + " \\\\",
        "\\midrule",
    ]
    for model in BASELINE_MODELS:
        row = [model]
        for dataset in datasets:
            match = df[(df["dataset"] == dataset) & (df["model"] == model)]
            if match.empty:
                row.extend(["-", "-"])
            else:
                row.extend([fmt_score(match.iloc[0]["score_mean"]), fmt_score(match.iloc[0]["score_sd"])])
        lines.append(" & ".join(row) + " \\\\")
        if model == "LRF":
            lines.append("\\hline")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def cosup_row_order() -> list[str]:
    order = []
    for family in COSUP_FAMILIES:
        for suffix in TEACHER_SUFFIXES:
            if family == "MLM-EPIC" and suffix == "LRF":
                continue
            order.append(f"{family}-{suffix}")
    return order


def latex_cosup_table(df: pd.DataFrame, datasets: list[str], classification: bool) -> str:
    metric_label = "AUC" if classification else "RMSE"
    lines = [
        "\\begin{tabular}{l|" + "|".join(["cc"] * len(datasets)) + "}",
        "\\toprule",
        "\\multirow{2}{*}{Model} & "
        + " & ".join([f"\\multicolumn{{2}}{{c{'|' if i < len(datasets)-1 else ''}}}{{{dataset}}}" for i, dataset in enumerate(datasets)])
        + " \\\\",
        "\\cmidrule(){2-" + str(1 + 2 * len(datasets)) + "}",
        "& " + " & ".join([f"${metric_label}$ & $S_{{\\mathrm{{Interp}}}}$" for _ in datasets]) + " \\\\",
        "\\midrule",
    ]

    best_lookup: dict[tuple[str, str], float] = {}
    for dataset in datasets:
        for family in COSUP_FAMILIES:
            family_rows = df[(df["dataset"] == dataset) & (df["model"].str.startswith(family + "-"))]
            if family_rows.empty:
                continue
            if classification:
                best_lookup[(dataset, family)] = float(family_rows["score_mean"].max())
            else:
                best_lookup[(dataset, family)] = float(family_rows["score_mean"].min())

    for model in cosup_row_order():
        family = model.split("-", 1)[0]
        row = [model]
        for dataset in datasets:
            match = df[(df["dataset"] == dataset) & (df["model"] == model)]
            if match.empty:
                row.extend(["-", "-"])
                continue
            score = float(match.iloc[0]["score_mean"])
            interp = match.iloc[0]["interp_mean"]
            is_best = np.isclose(score, best_lookup.get((dataset, family), np.nan), equal_nan=False)
            score_text = fmt_score(score)
            if is_best:
                score_text = f"\\textbf{{{score_text}}}"
            row.extend([score_text, fmt_score(interp)])
        lines.append(" & ".join(row) + " \\\\")
        if model.endswith("LRF"):
            lines.append("\\hline")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def latex_j_sensitivity(df: pd.DataFrame) -> str:
    strategy_labels = {
        "validation_mlm": "Validation-MLM",
        "silhouette": "Silhouette",
        "fixed_j": "Fixed $J=10$",
    }
    dataset_order = ["SKCM", "Upselling", "Abalone", "Servo"]
    metric_map = {"classification": "AUC", "regression": "RMSE"}
    lines = [
        "\\begin{tabular}{l|c|cc|cc|cc}",
        "\\toprule",
        "\\multirow{2}{*}{Dataset} & \\multirow{2}{*}{Metric} & \\multicolumn{2}{c|}{Validation-MLM} & \\multicolumn{2}{c|}{Silhouette} & \\multicolumn{2}{c}{Fixed $J=10$} \\\\",
        "\\cmidrule(){3-8}",
        "& & Mean $J$ & Score & Mean $J$ & Score & Mean $J$ & Score \\\\",
        "\\midrule",
    ]
    for dataset in dataset_order:
        block = df[df["dataset"] == dataset]
        if block.empty:
            continue
        task = block.iloc[0]["task"]
        row = [dataset, f"${metric_map[task]}$"]
        for strategy in ["validation_mlm", "silhouette", "fixed_j"]:
            match = block[block["strategy"] == strategy].iloc[0]
            row.extend(
                [
                    fmt_score(match["selected_num_cells_mean"], digits=1),
                    fmt_mean_sd(match["test_metric_mean"], match["test_metric_std"]),
                ]
            )
        lines.append(" & ".join(row) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def latex_ablation(df: pd.DataFrame) -> str:
    dataset_order = ["SKCM", "Compas", "Abalone", "Bike Sharing"]
    model_order = ["SynthTree", "SynthTree-RSS", "CART-GLM", "CART-LM"]
    lines = [
        "\\begin{tabular}{l|l|c|c}",
        "\\toprule",
        "Dataset & Method & Score & $S_{\\mathrm{Interp}}$ \\\\",
        "\\midrule",
    ]
    for dataset in dataset_order:
        block = df[df["dataset"] == dataset]
        for model in model_order:
            match = block[block["model"] == model]
            if match.empty:
                continue
            row = match.iloc[0]
            lines.append(
                f"{dataset} & {model} & {fmt_mean_sd(row['score_mean'], row['score_std'])} & {fmt_score(row['interpretability_mean'], digits=2)} \\\\"
            )
        lines.append("\\midrule")
    if lines[-1] == "\\midrule":
        lines.pop()
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def latex_runtime(df: pd.DataFrame) -> str:
    datasets = ["SKCM", "Compas", "Abalone", "Bike Sharing"]
    methods = ["SynthTree", "CART", "LRT", "MLM-EPIC"]
    lines = [
        "\\begin{tabular}{l|cccc}",
        "\\toprule",
        "Dataset & SynthTree & CART & LRT & MLM-EPIC \\\\",
        "\\midrule",
    ]
    for dataset in datasets:
        row = [dataset]
        for method in methods:
            match = df[(df["dataset"] == dataset) & (df["method"] == method)]
            if match.empty:
                row.append("-")
            else:
                row.append(fmt_mean_sd(match.iloc[0]["fit_time_mean"], match.iloc[0]["fit_time_std"], digits=2))
        lines.append(" & ".join(row) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def latex_tree_sizes(df: pd.DataFrame) -> str:
    datasets = ["SKCM", "Bike Sharing", "Compas", "Abalone", "Road Safety", "Upselling", "Servo", "Cal Housing"]
    methods = ["CART", "LRT", "SynthTree-INT"]
    lines = [
        "\\begin{tabular}{c|c|c|c}",
        "\\toprule",
        "Data & CART & LRT & SynthTree-INT \\\\",
        "\\hline",
    ]
    for dataset in datasets:
        row = [dataset]
        for method in methods:
            match = df[(df["dataset"] == dataset) & (df["method"] == method)]
            row.append("-" if match.empty else str(int(round(float(match.iloc[0]["leaf_count_mean"])))))
        lines.append(" & ".join(row) + " \\\\")
        lines.append("\\hline")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Export maintained manuscript tables from reproducible CSV summaries.")
    parser.add_argument("--input-dir", type=str, default=".")
    parser.add_argument("--output-dir", type=str, default="manuscript_outputs/tables")
    parser.add_argument("--tree-size-csv", type=str, default="manuscript_outputs/manuscript_tree_sizes_summary.csv")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = load_csv(input_dir, "main_accuracy_baseline_summary.csv")
    cosup = load_csv(input_dir, "main_accuracy_cosup_summary.csv")
    j_sensitivity = load_csv(input_dir, "j_selection_sensitivity_summary.csv")
    ablation = load_csv(input_dir, "tree_ablation_summary.csv")
    runtime = load_csv(input_dir, "runtime_summary.csv")
    tree_size_path = Path(args.tree_size_csv)

    write_text(output_dir / "dataset_info_table.tex", latex_dataset_info())
    write_text(
        output_dir / "classification_baseline_table.tex",
        latex_baseline_table(baseline[baseline["task"] == "classification"], CLASS_DATASETS, "AUC"),
    )
    write_text(
        output_dir / "classification_cosup_table.tex",
        latex_cosup_table(cosup[cosup["task"] == "classification"], CLASS_DATASETS, classification=True),
    )
    write_text(
        output_dir / "regression_baseline_table.tex",
        latex_baseline_table(baseline[baseline["task"] == "regression"], REG_DATASETS, "RMSE"),
    )
    write_text(
        output_dir / "regression_cosup_table.tex",
        latex_cosup_table(cosup[cosup["task"] == "regression"], REG_DATASETS, classification=False),
    )
    write_text(output_dir / "j_sensitivity_table.tex", latex_j_sensitivity(j_sensitivity))
    write_text(output_dir / "ablation_table.tex", latex_ablation(ablation))
    write_text(output_dir / "runtime_table.tex", latex_runtime(runtime))
    if tree_size_path.exists():
        tree_sizes = pd.read_csv(tree_size_path)
        write_text(output_dir / "tree_size_table.tex", latex_tree_sizes(tree_sizes))


if __name__ == "__main__":
    main()
