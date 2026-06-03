# PRISM Comparator Benchmarks

Minimal wrappers and a unified evaluation pipeline for the six comparator
methods reported in the PRISM manuscript.

| Method        | Language | Source                                       |
|---------------|----------|----------------------------------------------|
| RCTD          | R        | `run_rctd.R`                                 |
| SpatialDWLS   | R        | `run_dwls.R`                                 |
| Cell2location | Python   | `run_cell2location.py`                       |
| Tangram       | Python   | `run_tangram.py`                             |
| Spatial-ID    | Python   | `templates/spatial_id_template.py` (template)|
| DSCT          | Python   | `templates/dsct_template.py` (template)      |

Spatial-ID and DSCT were run using the authors' official repositories;
the files under `templates/` are minimal command templates provided for
reference and require small edits before use (see disclaimers at the
top of each template).

The evaluation logic (Accuracy, Macro-F1, weighted-F1, coverage,
per-cell-type F1, confusion matrix) is in `evaluate.py`. Dataset paths
and label mappings are centralised in `datasets.py`.

---

## Layout

```
benchmarks/
  README.md
  datasets.py              # DATASETS dict + load_sc_st_raw(name)
  evaluate.py              # Unified metrics from a prediction CSV + dataset
  prepare_for_R.py         # Export AnnData to CSV/RDS for the R methods
  run_rctd.R               # RCTD wrapper (spacexr)
  run_dwls.R               # SpatialDWLS wrapper (Giotto)
  run_cell2location.py     # Cell2location wrapper
  run_tangram.py           # Tangram wrapper (cells mode)
  envs/
    tangram.yml
    cell2location.yml
    rctd_dwls.yml          # R env (Seurat, spacexr, Giotto)
  templates/
    spatial_id_template.py
    dsct_template.py
    spatial_id_template.yml
    dsct_template.yml
```

## Supported datasets (`--dataset` values)

| Name           | Tissue / platform                              |
|----------------|------------------------------------------------|
| `HIP_merfish`  | Mouse hippocampus, MERFISH                     |
| `OB_merfish`   | Mouse olfactory bulb, MERFISH                  |
| `CTX_mouse`    | Mouse cortex, MERFISH (full)                   |
| `CTX_mouse_L`  | Mouse cortex, MERFISH (excitatory subset)      |
| `CTX_mouse_I`  | Mouse cortex, MERFISH (inhibitory subset)      |
| `CTX_mouse_O`  | Mouse cortex, MERFISH (non-neuronal subset)    |
| `Liver_merfish`| Mouse liver, MERFISH                           |
| `HCC_cosmx`    | Human HCC, CosMx                               |

Set the `PRISM_DATA_ROOT` environment variable to point at your local
data directory, or edit the per-dataset path constants at the top of
`datasets.py` directly.

---

## Unified output schema

Each runner writes a prediction CSV with columns:

| cell_id | predicted_label |
|---------|-----------------|

where `cell_id == st_adata.obs_names` and `predicted_label` is the
discrete cell-type assignment in the dataset's `anno_final` vocabulary.

Deconvolution-based methods (RCTD, SpatialDWLS, Cell2location) report
the dominant inferred type (argmax over the proportion vector). Cells
that the method did not return a prediction for are stored as missing
(NA) and counted as misclassified by `evaluate.py`, matching the
coverage policy described in the paper.

---

## Running

### Python methods

```bash
# Tangram
conda activate tangram
python run_tangram.py --dataset <NAME> --out_dir results/

# Cell2location
conda activate cell2location
python run_cell2location.py --dataset <NAME> --out_dir results/
```

### R methods

R cannot easily share AnnData; first export to CSV:

```bash
python prepare_for_R.py --dataset <NAME> --out_dir results/rds/
```

Then run the R wrapper:

```bash
conda activate rctd_dwls
Rscript run_rctd.R --dataset <NAME> --rds_dir results/rds/ --out_dir results/
Rscript run_dwls.R --dataset <NAME> --rds_dir results/rds/ --out_dir results/
```

Replace `<NAME>` with any value from the **Supported datasets** table
above.

---

## Evaluation

After running any method, compute metrics:

```bash
python evaluate.py \
    --dataset  <NAME> \
    --pred_csv <prediction_csv> \
    --method   <METHOD> \
    --out_dir  results/metrics/
```

Writes, for each prediction file:

- `*_metrics.csv` (Accuracy, Macro-F1, weighted-F1, coverage)
- `*_per_type_f1.csv` (per-cell-type F1)
- `*_confmat.csv` (row-normalised confusion matrix)

Coverage policy: cells that the method failed to predict (e.g. RCTD's
quality-filtered cells, DWLS's deconvolution failures) are read as NA
from the prediction CSV and counted as misclassified.

---

## Environments

Each baseline has its own conda env to avoid dependency conflicts.
Reproduce with:

```bash
conda env create -f envs/tangram.yml
conda env create -f envs/cell2location.yml
conda env create -f envs/rctd_dwls.yml
```

These environments are intended as a reference for the manuscript runs
and may need adjustment for newer hardware or upstream package versions.
