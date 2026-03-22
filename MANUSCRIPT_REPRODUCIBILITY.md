# Manuscript Reproducibility Manifest

This file maps the figures and tables in [draft.tex](/Users/evgeniikuriabov/Documents/SynthTree_Revision/draft/draft.tex) to the code paths available in the clean bundle.

Status labels:
- `Clean`: reproducible directly from the cleaned code in `CleanCode/SynthTree`
- `Manual`: conceptual figure or hand-maintained LaTeX content

## Figures

| Draft label / asset | Status | Source |
|---|---|---|
| `Images/Scheme_new.png` | `Manual` | Conceptual overview figure; no generator script in the codebase |
| `Images/Tree3.pdf` | `Manual` | Conceptual schematic plot; no generator script in the codebase |
| `Images/SKCM_tree.pdf` | `Clean` | [manuscript_case_studies.py](/Users/evgeniikuriabov/Documents/SynthTree_Revision/CleanCode/SynthTree/manuscript_case_studies.py) |
| `Images/BikeDataTree.pdf` | `Clean` | [manuscript_case_studies.py](/Users/evgeniikuriabov/Documents/SynthTree_Revision/CleanCode/SynthTree/manuscript_case_studies.py) |
| `Images/SKCM_heatmap_2.png` | `Clean` | [manuscript_case_studies.py](/Users/evgeniikuriabov/Documents/SynthTree_Revision/CleanCode/SynthTree/manuscript_case_studies.py) |
| `Images/Bike_Sharing_heatmap_2.png` | `Clean` | [manuscript_case_studies.py](/Users/evgeniikuriabov/Documents/SynthTree_Revision/CleanCode/SynthTree/manuscript_case_studies.py) |
| `Images/SKCMRegression.png` | `Clean` | [manuscript_case_studies.py](/Users/evgeniikuriabov/Documents/SynthTree_Revision/CleanCode/SynthTree/manuscript_case_studies.py) |
| `Images/BikeRegression.png` | `Clean` | [manuscript_case_studies.py](/Users/evgeniikuriabov/Documents/SynthTree_Revision/CleanCode/SynthTree/manuscript_case_studies.py) |
| `Images/LIME experiment/STFI_RF_SHAP_comparison.png` | `Clean` | [r2c13_feature_importance.py](/Users/evgeniikuriabov/Documents/SynthTree_Revision/CleanCode/SynthTree/r2c13_feature_importance.py) |
| `Images/LIME experiment/ALL-Boxplots-ST-LIME (3).png` | `Clean` | [manuscript_lime_analysis.py](/Users/evgeniikuriabov/Documents/SynthTree_Revision/CleanCode/SynthTree/manuscript_lime_analysis.py) |
| `Images/LIME experiment/ALL-Boxplots-ST-MLM (2).png` | `Clean` | [manuscript_lime_analysis.py](/Users/evgeniikuriabov/Documents/SynthTree_Revision/CleanCode/SynthTree/manuscript_lime_analysis.py) |
| `Images/same_leaf_explanation_colored.png` | `Clean` | [manuscript_lime_analysis.py](/Users/evgeniikuriabov/Documents/SynthTree_Revision/CleanCode/SynthTree/manuscript_lime_analysis.py) |
| `Images/LIME experiment/ALL-BIC-sep-new (4).png` | `Clean` | [manuscript_lime_analysis.py](/Users/evgeniikuriabov/Documents/SynthTree_Revision/CleanCode/SynthTree/manuscript_lime_analysis.py) |
| `Images/LIME experiment/BIC-All-new (4).png` | `Clean` | [manuscript_lime_analysis.py](/Users/evgeniikuriabov/Documents/SynthTree_Revision/CleanCode/SynthTree/manuscript_lime_analysis.py) |

## Tables

| Draft table | Status | Source |
|---|---|---|
| Dataset summary table | `Clean` | [manuscript_tables.py](/Users/evgeniikuriabov/Documents/SynthTree_Revision/CleanCode/SynthTree/manuscript_tables.py) |
| Baseline classification table | `Clean` | [full_accuracy_rerun.py](/Users/evgeniikuriabov/Documents/SynthTree_Revision/CleanCode/SynthTree/full_accuracy_rerun.py) + [manuscript_tables.py](/Users/evgeniikuriabov/Documents/SynthTree_Revision/CleanCode/SynthTree/manuscript_tables.py) |
| Co-supervised classification table | `Clean` | [full_accuracy_rerun.py](/Users/evgeniikuriabov/Documents/SynthTree_Revision/CleanCode/SynthTree/full_accuracy_rerun.py) + [manuscript_tables.py](/Users/evgeniikuriabov/Documents/SynthTree_Revision/CleanCode/SynthTree/manuscript_tables.py) |
| Baseline regression table | `Clean` | [full_accuracy_rerun.py](/Users/evgeniikuriabov/Documents/SynthTree_Revision/CleanCode/SynthTree/full_accuracy_rerun.py) + [manuscript_tables.py](/Users/evgeniikuriabov/Documents/SynthTree_Revision/CleanCode/SynthTree/manuscript_tables.py) |
| Co-supervised regression table | `Clean` | [full_accuracy_rerun.py](/Users/evgeniikuriabov/Documents/SynthTree_Revision/CleanCode/SynthTree/full_accuracy_rerun.py) + [manuscript_tables.py](/Users/evgeniikuriabov/Documents/SynthTree_Revision/CleanCode/SynthTree/manuscript_tables.py) |
| `J`-selection sensitivity table | `Clean` | [j_selection_sensitivity.py](/Users/evgeniikuriabov/Documents/SynthTree_Revision/CleanCode/SynthTree/j_selection_sensitivity.py) + [manuscript_tables.py](/Users/evgeniikuriabov/Documents/SynthTree_Revision/CleanCode/SynthTree/manuscript_tables.py) |
| Tree-growing ablation table (`R2.C8`) | `Clean` | [tree_ablation_comparison.py](/Users/evgeniikuriabov/Documents/SynthTree_Revision/CleanCode/SynthTree/tree_ablation_comparison.py) + [manuscript_tables.py](/Users/evgeniikuriabov/Documents/SynthTree_Revision/CleanCode/SynthTree/manuscript_tables.py) |
| Runtime table (`R2.C10`) | `Clean` | [runtime_benchmark.py](/Users/evgeniikuriabov/Documents/SynthTree_Revision/CleanCode/SynthTree/runtime_benchmark.py) + [manuscript_tables.py](/Users/evgeniikuriabov/Documents/SynthTree_Revision/CleanCode/SynthTree/manuscript_tables.py) |
| Leaf-count comparison table | `Clean` | [manuscript_tree_sizes.py](/Users/evgeniikuriabov/Documents/SynthTree_Revision/CleanCode/SynthTree/manuscript_tree_sizes.py) + [manuscript_tables.py](/Users/evgeniikuriabov/Documents/SynthTree_Revision/CleanCode/SynthTree/manuscript_tables.py) |

## Commands for clean-reproducible manuscript assets

Run from `CleanCode/SynthTree`.

Main accuracy tables:

```bash
../.venv/bin/python full_accuracy_rerun.py --runs 5 --datasets SKCM "Road Safety" Compas Upselling "Cal Housing" "Bike Sharing" Abalone Servo --tune-black-box-teachers --tune-cart --tune-lr --synthtree-n-cells-selection validation_mlm --cosup-baseline-n-cells-selection validation_mlm --n-cells-grid 5 10 15 20 --num-aug 100 --cv-folds 3
```

`J` sensitivity:

```bash
../.venv/bin/python j_selection_sensitivity.py --runs 5 --datasets SKCM Upselling Abalone Servo --grid 5 10 15 20 25 30 --num-aug 100 --local-model-max-iter 200
```

Tree-growing ablation:

```bash
../.venv/bin/python tree_ablation_comparison.py --runs 3 --datasets SKCM Compas "Bike Sharing" Abalone --n-cells-grid 5 10 15 20 --num-aug 100
```

Runtime:

```bash
../.venv/bin/python runtime_benchmark.py --runs 5 --datasets SKCM Upselling Abalone "Bike Sharing" --methods SynthTree CART LRT MLM-EPIC
```

SKCM / Bike case-study figures:

```bash
../.venv/bin/python manuscript_case_studies.py --seed 0 --output-dir ../../draft/Images
```

LIME / MLM-EPIC manuscript figures:

```bash
../.venv/bin/python manuscript_lime_analysis.py --seed 0 --output-dir ../../draft/Images/LIME\ experiment
```

`R2.C13` figure:

```bash
../.venv/bin/python -u r2c13_feature_importance.py --seed 0 --tree-sizer cc_prune --pruning-cv 10 --max-depth 5 --min-leaves 2 --num-aug 100 --n-cells-grid 2 3 --output-dir r2c13_outputs_ccprune_numaug100_depth5_minleaves2 --figure-path ../../draft/Images/LIME\ experiment/STFI_RF_SHAP_comparison.png
```

Leaf-count table source:

```bash
../.venv/bin/python manuscript_tree_sizes.py --runs 1 --output-dir manuscript_outputs
```

LaTeX-ready manuscript tables:

```bash
../.venv/bin/python manuscript_tables.py --input-dir . --output-dir manuscript_outputs/tables --tree-size-csv manuscript_outputs/manuscript_tree_sizes_summary.csv
```

## Bottom line

The clean bundle can now reproduce:
- the maintained `synthtree` library behavior,
- the main experiment CSV outputs,
- the manuscript tables from those CSVs,
- the reviewer ablation and runtime outputs,
- the maintained SKCM / Bike / LIME case-study figures,
- the `R2.C13` STFI figure.

The only manuscript assets still outside the maintained path are the two manual conceptual figures:
- `Images/Scheme_new.png`
- `Images/Tree3.pdf`
