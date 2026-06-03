#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tangram benchmark wrapper (cells mode).

Usage:
    python run_tangram.py --dataset <NAME> --out_dir results/

Writes:
    {out_dir}/{dataset}_Tangram_seed{seed}_prediction.csv  (cell_id, predicted_label)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import tangram as tg

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
from datasets import DATASETS, load_sc_st_raw  # noqa: E402


def run(dataset, out_dir, seed, cuda_device):
    os.makedirs(out_dir, exist_ok=True)
    cfg = DATASETS[dataset]
    anno = cfg["anno_final"]
    device = f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu"

    np.random.seed(seed); torch.manual_seed(seed)

    sc_obj, st_obj = load_sc_st_raw(dataset)
    print(f"    SC: {sc_obj.shape}  ST: {st_obj.shape}")

    # Tangram: shared genes only
    shared = sorted(set(sc_obj.var_names) & set(st_obj.var_names))
    print(f"    shared genes: {len(shared)}")
    sc_obj = sc_obj[:, shared].copy()
    st_obj = st_obj[:, shared].copy()

    # Tangram preprocessing
    sc.pp.normalize_total(sc_obj); sc.pp.log1p(sc_obj)
    sc.pp.normalize_total(st_obj); sc.pp.log1p(st_obj)

    tg.pp_adatas(sc_obj, st_obj, genes=shared)

    ad_map = tg.map_cells_to_space(
        sc_obj, st_obj, mode="cells", device=device,
        num_epochs=1000, random_state=seed)
    # Project cell-type labels
    tg.project_cell_annotations(ad_map, st_obj, annotation=anno)
    prob = st_obj.obsm["tangram_ct_pred"]
    if isinstance(prob, pd.DataFrame):
        pred = prob.idxmax(axis=1).values
    else:
        cols = sc_obj.obs[anno].cat.categories
        pred = np.array(cols)[np.argmax(prob, axis=1)]

    out_csv = os.path.join(
        out_dir, f"{dataset}_Tangram_seed{seed}_prediction.csv")
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
    ap.add_argument("--cuda_device", type=int, default=0)
    args = ap.parse_args()
    run(args.dataset, args.out_dir, args.seed, args.cuda_device)


if __name__ == "__main__":
    main()
