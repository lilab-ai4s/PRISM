#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cell2location benchmark wrapper (minimal).

Usage:
    python run_cell2location.py --dataset <NAME> --out_dir results/

Writes:
    {out_dir}/{dataset}_Cell2location_seed{seed}_prediction.csv

Notes:
    Cell2location estimates posterior cell-type abundance vectors; for
    categorical comparison we report the dominant inferred type (argmax).
    The defaults below come from the Cell2location tutorial and were used
    in the manuscript runs; tune them via --max_epochs_ref / --max_epochs_loc
    if you wish to trade accuracy for runtime.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import scanpy as sc
import torch

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
from datasets import DATASETS, load_sc_st_raw  # noqa: E402

import cell2location
from cell2location.models import RegressionModel, Cell2location


def run(dataset, out_dir, seed, max_epochs_ref, max_epochs_loc, cuda_device):
    os.makedirs(out_dir, exist_ok=True)
    cfg = DATASETS[dataset]
    anno = cfg["anno_final"]
    np.random.seed(seed); torch.manual_seed(seed)

    sc_obj, st_obj = load_sc_st_raw(dataset)

    # Shared-gene restriction (Cell2location supports gene mismatch but we keep it consistent)
    shared = sorted(set(sc_obj.var_names) & set(st_obj.var_names))
    sc_obj = sc_obj[:, shared].copy()
    st_obj = st_obj[:, shared].copy()

    # ---- 1) Reference model on SC ----
    RegressionModel.setup_anndata(sc_obj, labels_key=anno)
    ref = RegressionModel(sc_obj)
    ref.train(max_epochs=max_epochs_ref,
              batch_size=2500, lr=0.002, use_gpu=cuda_device)
    sc_obj = ref.export_posterior(sc_obj, sample_kwargs={"num_samples": 1000,
                                                         "batch_size": 2500,
                                                         "use_gpu": cuda_device})
    inf_aver = sc_obj.varm["means_per_cluster_mu_fg"]
    inf_aver.columns = [c.replace("means_per_cluster_mu_fg_", "") for c in inf_aver.columns]

    # ---- 2) Cell2location on ST ----
    Cell2location.setup_anndata(st_obj)
    mod = Cell2location(
        st_obj, cell_state_df=inf_aver,
        N_cells_per_location=1, detection_alpha=20,
    )
    mod.train(max_epochs=max_epochs_loc, batch_size=None,
              train_size=1, use_gpu=cuda_device)
    st_obj = mod.export_posterior(st_obj, sample_kwargs={"num_samples": 1000,
                                                         "batch_size": mod.adata.n_obs,
                                                         "use_gpu": cuda_device})

    # Abundance -> categorical (argmax)
    abundance = st_obj.obsm["q05_cell_abundance_w_sf"]
    abundance.columns = [c.replace("q05cell_abundance_w_sf_", "") for c in abundance.columns]
    pred = abundance.idxmax(axis=1).values

    out_csv = os.path.join(
        out_dir, f"{dataset}_Cell2location_seed{seed}_prediction.csv")
    pd.DataFrame({
        "cell_id":         st_obj.obs_names.astype(str),
        "predicted_label": pred,
    }).to_csv(out_csv, index=False)
    print(f"    wrote {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",         required=True, choices=list(DATASETS))
    ap.add_argument("--out_dir",         required=True)
    ap.add_argument("--seed",            type=int, default=0)
    ap.add_argument("--max_epochs_ref",  type=int, default=250)
    ap.add_argument("--max_epochs_loc",  type=int, default=30000)
    ap.add_argument("--cuda_device",     type=int, default=0)
    args = ap.parse_args()
    run(args.dataset, args.out_dir, args.seed,
        args.max_epochs_ref, args.max_epochs_loc, args.cuda_device)


if __name__ == "__main__":
    main()
