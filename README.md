# Clean Reproducibility Bundle

This folder is a stripped-down copy of the maintained code needed to rerun the
current SynthTree experiments in a fresh environment.

It intentionally excludes:
- notebooks,
- legacy compatibility shims,
- old CSV outputs,
- temporary result folders,
- git metadata,
- compiled R artifacts.

For manuscript-level reproducibility status of every figure and table, see
[MANUSCRIPT_REPRODUCIBILITY.md](/Users/evgeniikuriabov/Documents/SynthTree_Revision/CleanCode/MANUSCRIPT_REPRODUCIBILITY.md).

## Layout

```text
CleanCode/
├─ requirements.txt
├─ install_r_dependencies.R
├─ SynthTree/
│  ├─ synthtree/
│  ├─ tests/
│  ├─ data/
│  ├─ co_supervision_test.py
│  ├─ full_accuracy_rerun.py
│  ├─ j_selection_sensitivity.py
│  ├─ manuscript_case_studies.py
│  ├─ manuscript_lime_analysis.py
│  ├─ manuscript_tables.py
│  ├─ manuscript_tree_sizes.py
│  ├─ tree_ablation_comparison.py
│  ├─ runtime_benchmark.py
│  ├─ r2c13_feature_importance.py
│  ├─ generalized_mlm.py
│  └─ example_preprocessing.py
└─ Rforestry/
   ├─ DESCRIPTION
   ├─ NAMESPACE
   ├─ R/
   └─ src/
```

## Python setup

From `CleanCode/`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If matplotlib cache warnings appear in a locked-down environment, set:

```bash
export MPLCONFIGDIR=/tmp/matplotlib-cache
mkdir -p "$MPLCONFIGDIR"
```

## R setup

The Python scripts that use LRT/LRF expect `Rforestry` to live as a sibling of
`SynthTree`, which is already true in this clean bundle.

Install the required R-side dependencies from `CleanCode/`:

```bash
Rscript install_r_dependencies.R
```

This installs `devtools` and the R packages required by `Rforestry`.

## Sanity check

Run the unit tests from `CleanCode/SynthTree`:

```bash
cd SynthTree
../.venv/bin/python -m unittest tests.test_synthtree
```

## Main experiment entry points

All commands below should be run from `CleanCode/SynthTree`.

Main benchmark refresh:

```bash
../.venv/bin/python full_accuracy_rerun.py --runs 5 --datasets SKCM "Road Safety" Compas Upselling "Cal Housing" "Bike Sharing" Abalone Servo --tune-black-box-teachers --tune-cart --tune-lr --synthtree-n-cells-selection validation_mlm --cosup-baseline-n-cells-selection validation_mlm --n-cells-grid 5 10 15 20 --num-aug 100 --cv-folds 3
```

Clustering sensitivity:

```bash
../.venv/bin/python j_selection_sensitivity.py --runs 5 --datasets SKCM Upselling Abalone Servo --grid 5 10 15 20 25 30 --num-aug 100 --local-model-max-iter 200
```

Tree-growing ablation:

```bash
../.venv/bin/python tree_ablation_comparison.py --runs 3 --datasets SKCM Compas "Bike Sharing" Abalone --n-cells-grid 5 10 15 20 --num-aug 100
```

Runtime benchmark:

```bash
../.venv/bin/python runtime_benchmark.py --runs 5 --datasets SKCM Upselling Abalone "Bike Sharing" --methods SynthTree CART LRT MLM-EPIC
```

Manuscript case-study tree / heatmap / coefficient figures:

```bash
../.venv/bin/python manuscript_case_studies.py --seed 0 --output-dir ../../draft/Images
```

Maintained LIME / MLM-EPIC case-study figures:

```bash
../.venv/bin/python manuscript_lime_analysis.py --seed 0 --output-dir ../../draft/Images/LIME\ experiment
```

Focused `R2.C13` interpretation case study:

```bash
../.venv/bin/python -u r2c13_feature_importance.py --seed 0 --tree-sizer cc_prune --pruning-cv 10 --max-depth 5 --min-leaves 2 --num-aug 100 --n-cells-grid 2 3 --output-dir r2c13_outputs_ccprune_numaug100_depth5_minleaves2 --figure-path ../../draft/Images/LIME\ experiment/STFI_RF_SHAP_comparison.png
```

Tree-size table source:

```bash
../.venv/bin/python manuscript_tree_sizes.py --runs 1 --output-dir manuscript_outputs
```

LaTeX-ready manuscript tables from the maintained CSV summaries:

```bash
../.venv/bin/python manuscript_tables.py --input-dir . --output-dir manuscript_outputs/tables --tree-size-csv manuscript_outputs/manuscript_tree_sizes_summary.csv
```

## Notes

- `Cal Housing` is fetched via `sklearn.datasets.fetch_california_housing()`.
  If the new environment is offline, skip that dataset or pre-cache it.
- New CSV outputs are generated in `CleanCode/SynthTree/` when you run the
  scripts.
- The maintained library is the `synthtree/` package inside `CleanCode/SynthTree`.
- The manuscript figure scripts regenerate maintained replacements for the old
  SKCM/Bike/LIME notebook outputs using the cleaned library.
