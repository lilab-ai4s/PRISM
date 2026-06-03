#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified evaluation for all comparator methods.

Reads a prediction CSV (cell_id, predicted_label) + ground-truth from the
dataset registry, computes:
    - Accuracy        (cells without prediction count as misclassified)
    - Macro-F1
    - Weighted-F1
    - Coverage        (fraction of ST cells that received a prediction)
    - Per-cell-type F1 (one row per ground-truth class)
    - Confusion matrix (row-normalised; rows = ground truth, cols = predicted)

Usage:
    python evaluate.py \\
        --dataset  <NAME> \\
        --pred_csv <prediction_csv> \\
        --method   <METHOD> \\
        --out_dir  results/metrics/

Outputs to {out_dir}, one of each per evaluated prediction:
    {dataset}_{method}_seed{seed}_metrics.csv
    {dataset}_{method}_seed{seed}_per_type_f1.csv
    {dataset}_{method}_seed{seed}_confmat.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
from datasets import DATASETS, get_ground_truth  # noqa: E402


# =====================================================================
# CTX-specific SC fine -> ST coarse remap (used only when dataset is
# CTX_mouse[_*] and the comparator was trained on SC fine labels).
# =====================================================================
CTX_SC_TO_ST = {
    # excitatory
    "L2_3_IT": "L2_3_IT", "L4_5_IT": "L4_5_IT", "L5_IT": "L5_IT",
    "L5_PT":   "L5_PT",   "L6_CT":   "L6_CT",   "L6_IT":  "L6_IT",
    "L6b":     "L6b",     "NP":      "NP",
    # inhibitory (fine -> coarse)
    "Pvalb_A": "iPVALB", "Pvalb_B": "iPVALB", "Pvalb_C": "iPVALB", "Pvalb_D": "iPVALB",
    "Sst_A": "iSST",   "Sst_B": "iSST",   "Sst_C": "iSST",   "Sst_D": "iSST", "Sst_E": "iSST",
    "Lamp5": "iLAMP5",
    "Vip_A": "iVIP",   "Vip_B": "iVIP",   "Vip_C": "iVIP",
    # non-neuronal
    "Endo": "lENDO", "VLMC_A": "lVLMC", "VLMC_B": "lVLMC",
    "OPC_A": "lOPC", "OPC_B": "lOPC",
    "OD_A":  "lOLG", "OD_B":  "lOLG", "OD_C":  "lOLG",
    "Astro_A": "lASC", "Astro_B": "lASC",
    "Micro":   "lMGC",
}


def _maybe_remap_ctx(dataset, y_pred):
    """Apply the SC fine -> ST coarse remap; preserve missing values as-is."""
    if not dataset.startswith("CTX_mouse"):
        return y_pred
    out = []
    for p in y_pred:
        if p is None or (isinstance(p, float) and np.isnan(p)) or p is pd.NA:
            out.append(pd.NA)
        else:
            out.append(CTX_SC_TO_ST.get(str(p), str(p)))
    return np.array(out, dtype=object)


def evaluate_one(dataset, pred_csv, method, seed, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n=== Evaluating {method} on {dataset} (seed={seed}) ===")

    # 1) Ground truth
    gt = get_ground_truth(dataset)
    print(f"    ground truth: {len(gt)} cells, {gt.nunique()} classes")

    # 2) Predictions -- keep NA / NaN as missing; do NOT cast to str yet, otherwise
    #    real NaN cells would become literal "nan" and be treated as a valid label.
    pred = pd.read_csv(pred_csv, na_values=["", "NA", "NaN", "nan", "None"],
                       keep_default_na=True)
    if "cell_id" not in pred.columns:
        raise ValueError(f"{pred_csv} missing 'cell_id' column")
    pred_col = [c for c in pred.columns if c != "cell_id"][0]
    pred = pred.set_index("cell_id")[pred_col]
    pred.index = pred.index.astype(str)
    n_predicted = int(pred.notna().sum())
    print(f"    predictions:  {len(pred)} rows "
          f"({n_predicted} with a label, {len(pred) - n_predicted} missing)")

    # 3) Optional CTX SC->ST remap (apply only to non-null predictions)
    remapped = _maybe_remap_ctx(
        dataset, pred.where(pred.notna(), other=pd.NA).astype(object).values)
    pred = pd.Series(remapped, index=pred.index, name="predicted_label")

    # 4) Align -- cells in GT but missing from pred (NaN/NA) become "__missing__"
    #    and count as wrong, matching the coverage policy described in the paper.
    aligned = pd.DataFrame({"y_true": gt, "y_pred": pred})
    aligned["y_pred"] = aligned["y_pred"].where(
        aligned["y_pred"].notna(), other="__missing__")
    y_true = aligned["y_true"].astype(str).values
    y_pred = aligned["y_pred"].astype(str).values
    coverage = float((aligned["y_pred"] != "__missing__").mean())

    # 5) Metrics
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    wf1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    metrics = pd.DataFrame([{
        "dataset":     dataset,
        "method":      method,
        "seed":        seed,
        "n_cells_st":  len(y_true),
        "n_classes":   int(pd.Series(y_true).nunique()),
        "coverage":    coverage,
        "accuracy":    acc,
        "macro_f1":    mf1,
        "weighted_f1": wf1,
    }])
    print(f"    Acc={acc:.4f}  MF1={mf1:.4f}  WF1={wf1:.4f}  Cov={coverage:.4f}")

    # 6) Per-cell-type F1
    classes = sorted(pd.Series(y_true).unique())
    _, _, f1_per, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0)
    per_type = pd.DataFrame({
        "cell_type": classes,
        "support":   support,
        "f1":        f1_per,
    })

    # 7) Confusion matrix (row-normalised; rows = GT, cols = pred including __missing__)
    pred_classes = sorted(pd.Series(y_pred).unique())
    cm = confusion_matrix(y_true, y_pred, labels=classes)  # row order = classes
    # extend columns with any extra predicted labels not in classes (e.g. SC-only labels)
    extra_pred = [c for c in pred_classes if c not in classes]
    if extra_pred:
        cm_extra = confusion_matrix(
            y_true, y_pred,
            labels=list(classes) + extra_pred)[:len(classes),
                                               len(classes):]
        cm_full = np.concatenate([cm, cm_extra], axis=1)
        cols = list(classes) + extra_pred
    else:
        cm_full = cm
        cols = list(classes)

    row_sums = cm_full.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_full, row_sums, out=np.zeros_like(cm_full, dtype=float),
                        where=row_sums > 0)
    cm_df = pd.DataFrame(cm_norm, index=classes, columns=cols)
    cm_df.index.name = "true_label"

    # 8) Save
    prefix = f"{dataset}_{method}_seed{seed}"
    metrics.to_csv(os.path.join(out_dir, f"{prefix}_metrics.csv"), index=False)
    per_type.to_csv(os.path.join(out_dir, f"{prefix}_per_type_f1.csv"), index=False)
    cm_df.to_csv(os.path.join(out_dir, f"{prefix}_confmat.csv"))

    print(f"    wrote metrics + per_type_f1 + confmat under {out_dir}")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",  required=True, choices=list(DATASETS))
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--method",   required=True,
                    help="Method name used in output filenames "
                         "(e.g. RCTD, SpatialDWLS, Cell2location, Tangram, "
                         "Spatial-ID, DSCT, PRISM).")
    ap.add_argument("--seed",     type=int, default=0)
    ap.add_argument("--out_dir",  required=True)
    args = ap.parse_args()
    evaluate_one(args.dataset, args.pred_csv, args.method, args.seed, args.out_dir)


if __name__ == "__main__":
    main()
