#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Spatial-ID benchmark template (NOT a runnable wrapper out of the box).

In the PRISM manuscript, Spatial-ID was run using the authors' official
repository (https://github.com/genome-bioinformatics/Spatial-ID) following
their default settings. This file is provided only as a minimal reference
showing how a Spatial-ID run was integrated with the unified evaluation
pipeline in this directory: it loads SC + ST from `datasets.py` and writes
predictions in the unified `(cell_id, predicted_label)` schema expected by
`evaluate.py`. The `spatial_id.*` imports below are placeholders -- adjust
them to match the upstream module layout before use.
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

# Spatial-ID is imported from the user's local checkout
from spatial_id.train import train_spatial_id  # adjust import to match local layout
from spatial_id.predict import predict_spatial_id


def run(dataset, out_dir, seed, n_epochs, cuda_device):
    os.makedirs(out_dir, exist_ok=True)
    cfg = DATASETS[dataset]
    anno = cfg["anno_final"]
    device = f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu"

    np.random.seed(seed); torch.manual_seed(seed)
    sc_obj, st_obj = load_sc_st_raw(dataset)

    shared = sorted(set(sc_obj.var_names) & set(st_obj.var_names))
    sc_obj = sc_obj[:, shared].copy()
    st_obj = st_obj[:, shared].copy()
    sc.pp.normalize_total(sc_obj); sc.pp.log1p(sc_obj)
    sc.pp.normalize_total(st_obj); sc.pp.log1p(st_obj)

    # Spatial coords (required for the GCN branch)
    coords = st_obj.obs[["x", "y"]].values.astype(float)

    model = train_spatial_id(
        sc_X=sc_obj.X, sc_y=sc_obj.obs[anno].astype(str).values,
        st_X=st_obj.X, st_coords=coords,
        n_epochs=n_epochs, device=device, seed=seed,
    )
    pred = predict_spatial_id(model, st_obj.X, coords, device=device)

    out_csv = os.path.join(
        out_dir, f"{dataset}_Spatial-ID_seed{seed}_prediction.csv")
    pd.DataFrame({
        "cell_id":         st_obj.obs_names.astype(str),
        "predicted_label": pred,
    }).to_csv(out_csv, index=False)
    print(f"    wrote {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",     required=True, choices=list(DATASETS))
    ap.add_argument("--out_dir",     required=True)
    ap.add_argument("--seed",        type=int, default=0)
    ap.add_argument("--n_epochs",    type=int, default=200)
    ap.add_argument("--cuda_device", type=int, default=0)
    args = ap.parse_args()
    run(args.dataset, args.out_dir, args.seed, args.n_epochs, args.cuda_device)


if __name__ == "__main__":
    main()
