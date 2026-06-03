#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Export SC + ST AnnData to RDS files for the R-based comparator methods
(RCTD via spacexr, SpatialDWLS via Giotto).

Usage:
    python prepare_for_R.py --dataset <NAME> --out_dir results/rds/

Writes:
    {out_dir}/{dataset}_sc_counts.csv          (genes x cells, integer counts)
    {out_dir}/{dataset}_sc_labels.csv          (cell_id, cell_type)
    {out_dir}/{dataset}_st_counts.csv          (genes x cells, integer counts)
    {out_dir}/{dataset}_st_coords.csv          (cell_id, x, y)

CSVs are used by run_rctd.R / run_dwls.R via read.csv. CSV is intentionally
chosen over RDS to avoid Python<->R version mismatches.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
from datasets import DATASETS, load_sc_st_raw  # noqa: E402


def _to_dense_df(adata):
    """Return genes (rows) x cells (cols) DataFrame of integer counts."""
    X = adata.X.toarray() if sp.issparse(adata.X) else adata.X
    X = np.asarray(X, dtype=np.int32)
    return pd.DataFrame(X.T, index=adata.var_names.astype(str),
                        columns=adata.obs_names.astype(str))


def run(dataset, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cfg = DATASETS[dataset]
    anno = cfg["anno_final"]

    sc_obj, st_obj = load_sc_st_raw(dataset)
    shared = sorted(set(sc_obj.var_names) & set(st_obj.var_names))
    sc_obj = sc_obj[:, shared].copy()
    st_obj = st_obj[:, shared].copy()
    print(f"    SC: {sc_obj.shape}  ST: {st_obj.shape}  shared genes: {len(shared)}")

    _to_dense_df(sc_obj).to_csv(
        os.path.join(out_dir, f"{dataset}_sc_counts.csv"))
    pd.DataFrame({
        "cell_id":   sc_obj.obs_names.astype(str),
        "cell_type": sc_obj.obs[anno].astype(str).values,
    }).to_csv(os.path.join(out_dir, f"{dataset}_sc_labels.csv"), index=False)

    _to_dense_df(st_obj).to_csv(
        os.path.join(out_dir, f"{dataset}_st_counts.csv"))
    pd.DataFrame({
        "cell_id": st_obj.obs_names.astype(str),
        "x":       st_obj.obs["x"].astype(float).values
                   if "x" in st_obj.obs.columns else np.arange(st_obj.n_obs),
        "y":       st_obj.obs["y"].astype(float).values
                   if "y" in st_obj.obs.columns else np.arange(st_obj.n_obs),
    }).to_csv(os.path.join(out_dir, f"{dataset}_st_coords.csv"), index=False)

    print(f"    wrote 4 CSVs under {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    run(args.dataset, args.out_dir)


if __name__ == "__main__":
    main()
