
import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from typing import Optional

import scanpy as sc
import anndata as ad

import diopy
from scipy import sparse
from scipy.sparse import issparse

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed=42

def ssim(part1, part2):
    # Replace NaN values with 0

    mu1 = np.mean(part1)
    mu2 = np.mean(part2)
    
    C1 = 0.01**2
    C2 = 0.03**2
    
    numerator = (2 * mu1 * mu2 + C1) * (2 * np.cov(part1, part2)[0, 1] + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (np.var(part1) + np.var(part2) + C2)
    
    ssim_result = numerator / denominator
    return ssim_result

def KL(part1, part2):
    ee=0.001
    ratio = (part1+ee) / (part2+ee)
    result = part1 * np.log(ratio)
    
    # Avoiding extreme values
    result[result > 1000] = 0
    result[result < (-1000)] = 0
    
    return np.sum(result)

def JS(part1, part2):
    M = (part1 + part2) / 2
    return 0.5 * KL(part1, M) + 0.5 * KL(part2, M)

def cosi(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def pear(vec1, vec2):
    return np.corrcoef(vec1, vec2)[0, 1]

def mse(vec1, vec2):
    return np.mean((vec1 - vec2) ** 2)




def csg_fast(
    adata,
    groupby: str,
    layer: Optional[str] = None,
    mu: float = 0.25,
    expressed_pct: float = 0.1,
    remove_lowly_expressed: bool = True,
    min_cells: int = 10,
    n_genes_user: int = 300,
    key_added: str = "csg",
):
    X = adata.layers[layer] if layer else adata.X          # (N,G)
    if sparse.issparse(X):
        X = X.tocsr(copy=False)                            

    if remove_lowly_expressed:
        keep_gene = np.asarray((X > 0).sum(0)).ravel() >= min_cells
        X = X[:, keep_gene]
        gene_names = adata.var_names[keep_gene]
    else:
        gene_names = adata.var_names

    N, G = X.shape


    groups = adata.obs[groupby].astype(str).values
    cell_types, inverse = np.unique(groups, return_inverse=True)    # len=K
    K = cell_types.size
    one_hot = sparse.csr_matrix(
        (np.ones(N, dtype=np.float32), inverse, np.arange(N + 1)),
        shape=(N, K)
    )                                                               # (N,K)


    #       dot = X.T @ one_hot      (G,K)
    dot = (X.T @ one_hot).toarray() if sparse.issparse(X) else X.T @ one_hot


    gene_norm = np.sqrt((X.multiply(X)).sum(0)).A1 if sparse.issparse(X) else np.linalg.norm(X, axis=0)
    y_norm = np.sqrt(np.asarray(one_hot.sum(0)).ravel())            # (K,)

    cos = dot / (gene_norm[:, None] * y_norm[None, :] + 1e-9)       # (G,K)


    expr_sum = dot                                 # (= step 3)
    cell_count = np.asarray(one_hot.sum(0)).ravel()   
    mean_in = expr_sum / (cell_count + 1e-9)          # (G,K)


    global_sum = X.sum(0).A1 if sparse.issparse(X) else X.sum(0)
    global_mean = global_sum / N                     # (G,)

    mean_out = (global_sum[:, None] - expr_sum) / (N - cell_count + 1e-9)
    pct_in = ( (X > 0).astype(np.float32).T @ one_hot ).toarray() / (cell_count + 1e-9)

    filter_mask = (pct_in >= expressed_pct) & ((mean_in - mean_out) >= mu)


    cos2 = cos ** 2
    sum_cos2 = cos2.sum(1, keepdims=True)
    score = cos2 / ((1 - mu) * cos2 + mu * sum_cos2)
    score[~filter_mask] = -np.inf


    gene_arr = gene_names.to_numpy()          

    top_idx = np.argpartition(-score, n_genes_user - 1, axis=0)[:n_genes_user, :]
    order   = np.take_along_axis(
                top_idx,
                np.argsort(-np.take_along_axis(score, top_idx, 0), axis=0),
                axis=0
              )

    marker_table = pd.DataFrame(
        gene_arr[order],                      
        index=np.arange(n_genes_user),
        columns=cell_types
    )


    adata.uns[key_added] = dict(
        names=marker_table,
        scores=score,
        params=dict(mu=mu, expressed_pct=expressed_pct,
                    min_cells=min_cells, n_genes_user=n_genes_user,
                    layer=layer)
    )
    return marker_table


def inverse_csg(
    adata,
    groupby: str,
    layer: Optional[str] = None,      
    mu: float = 0.25,
    pct_in_max: float = 0.05,          
    pct_out_min: float = 0.2,          
    remove_lowly_expressed: bool = True,
    min_cells: int = 10,
    n_genes_user: int = 150,
    key_added: str = "csg_inv"
):

    X = adata.layers[layer] if layer else adata.X
    if issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)

    if remove_lowly_expressed:
        keep_gene = (X > 0).sum(0) >= min_cells
        X = X[:, keep_gene]
        gene_names = adata.var_names[keep_gene]
    else:
        gene_names = adata.var_names


    gene_norm = np.linalg.norm(X, axis=0) + 1e-9


    groups = adata.obs[groupby].astype(str).values
    cell_types, inverse = np.unique(groups, return_inverse=True)
    n_gene, n_type = X.shape[1], len(cell_types)


    dot_bar = np.zeros((n_gene, n_type), dtype=np.float32)
    for k in range(n_type):
        idx_notk = inverse != k
        dot_bar[:, k] = X[idx_notk].sum(0)


    n_notk = X.shape[0] - np.bincount(inverse)  # (K,)
    ybar_norm = np.sqrt(n_notk) + 1e-9

    cos_bar = dot_bar / (gene_norm[:, None] * ybar_norm[None, :])


    pct_in  = np.zeros_like(cos_bar)
    pct_out = np.zeros_like(cos_bar)
    mean_in = np.zeros_like(cos_bar)
    mean_out= np.zeros_like(cos_bar)
    for k in range(n_type):
        idx_k   = inverse == k
        idx_not = ~idx_k
        pct_in[:, k]   = (X[idx_k]   > 0).mean(0)
        pct_out[:, k]  = (X[idx_not] > 0).mean(0)
        mean_in[:, k]  = X[idx_k].mean(0)
        mean_out[:, k] = X[idx_not].mean(0)

    filter_mask = (pct_in <= pct_in_max) & (pct_out >= pct_out_min) & \
                  ((mean_out - mean_in) >= mu)


    cos2 = cos_bar**2
    sum_cos2 = cos2.sum(1, keepdims=True)
    score = cos2 / ((1 - mu) * cos2 + mu * sum_cos2)
    score[~filter_mask] = -np.inf


    marker_table = pd.DataFrame(index=np.arange(n_genes_user),
                                columns=cell_types)
    for k, ct in enumerate(cell_types):
        order = np.argsort(-score[:, k], kind="mergesort")
        top_idx = order[:n_genes_user]
        marker_table[ct] = gene_names[top_idx].values


    adata.uns[key_added] = dict(
        names  = marker_table,
        scores = score,
        params = dict(mu=mu, pct_in_max=pct_in_max,
                      pct_out_min=pct_out_min,
                      min_cells=min_cells,
                      n_genes_user=n_genes_user,
                      layer=layer)
    )
    return marker_table

def fmap_load(sc_data,st_data,anno,gene_number):
    seed=42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    sc_data.obs['total_counts'] = sc_data.X.sum(axis=1)
    st_data.obs['total_counts'] = st_data.X.sum(axis=1)

    sc.pp.filter_cells(sc_data, min_counts=0)
    sc.pp.filter_cells(st_data, min_counts=0)
 
    sc.pp.filter_genes(sc_data, min_cells=20)
    sc.pp.filter_genes(st_data, min_cells=20)
    corrected_sc_data = sc_data 
    corrected_st_data = st_data 

    sc_genes = sc_data.var_names
    st_genes = st_data.var_names

    common_genes = set(sc_genes).intersection(set(st_genes))
    sc_data = sc_data[:, list(common_genes)]
    st_data = st_data[:, list(common_genes)]
    sorted_genes = sorted(st_data.var_names)
    print('csg_first')
    print(len(sorted_genes))
    st_data = st_data[:, sorted_genes]
    sc_data = sc_data[:, sorted_genes]


    groupby=anno
    print('start')
    start = time.time()   
    csg_fast(
    sc_data,
    groupby=anno,
    layer=None,           
    n_genes_user=gene_number,
    expressed_pct=0.1,
    mu=0.25,
    min_cells=5
)
 
    print('end')
    elapsed = time.time() - start  
    print(f"Total time: {elapsed:.3f} s")

    sc.pp.highly_variable_genes(st_data, n_top_genes=10000)
    high_var_genes = st_data.var[st_data.var['highly_variable']].index
    gene_names_series = pd.Series(high_var_genes)
    st_data = corrected_st_data
    markers_df = pd.DataFrame(sc_data.uns["csg"]["names"]).iloc[0:500, :]
    marker_genes = set(markers_df.values.flatten().tolist())
    intersect_genes = high_var_genes.intersection(marker_genes)
    sc_data = sc_data[:, list(intersect_genes)]
    st_data = st_data[:, list(intersect_genes)]
    
    sorted_genes = sorted(st_data.var_names)
    
    print(len(sorted_genes))
    st_data = st_data[:, sorted_genes]
    sc_data = sc_data[:, sorted_genes]

    
    return sc_data,st_data
    