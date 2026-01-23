import glob
import re
from pathlib import Path
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





def evaluate_and_rank_predictions(root_dir, dataset_name, gene_number, pattern):


    root_dir = Path(root_dir)
    metric_cols = ["SSIM", "KL", "COSINE", "PEARSON"]
    higher_is_better = dict(SSIM=True, COSINE=True, PEARSON=True, KL=False)

    for seed_idx in range(10):
        in_file = root_dir / f"{dataset_name}_{gene_number}{seed_idx}_all_values{gene_number}.csv"
        type_mean_file = root_dir / f"{dataset_name}_{seed_idx}_type_means_{gene_number}.csv"
        overall_file = root_dir / f"{dataset_name}_{seed_idx}_average_values_{gene_number}.csv"

        df = pd.read_csv(in_file)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

        type_means = (
            df
            .groupby("Type", as_index=True)[metric_cols]
            .mean()
        )
        overall_means = type_means.mean(axis=0)

        type_means.to_csv(type_mean_file)
        overall_means.to_frame("Average Value").to_csv(overall_file)



    rows = []
    for fp in glob.glob(pattern):
        m = re.search(rf"(.+?)_(\d+)_average_values_{gene_number}\.csv$", os.path.basename(fp))
        if not m:
            continue
        method, rnd = m.group(1), int(m.group(2))

        df = pd.read_csv(fp, index_col=0)["Average Value"]
        df.name = "value"
        df = df.reset_index().rename(columns={"index": "Metric"})
        df["Round"] = rnd
        df["Method"] = method
        rows.append(df)

    long_df = pd.concat(rows, ignore_index=True)

    wide_df = (
        long_df
        .pivot_table(index=["Method", "Round"], columns="Metric", values="value")
        .reset_index()
    )
    wide_df.to_csv(root_dir / f"round_average_scores{gene_number}.csv", index=False)

    rank_across = wide_df.copy()
    for m in metric_cols:
        ascending = not higher_is_better[m]
        rank_across[m + "_rank"] = (
            rank_across
            .groupby("Method")[m]
            .rank(method="min", ascending=ascending)
            .astype(int)
        )


    metric_rank_cols = [m + "_rank" for m in metric_cols]
    rank_across["RankSum"] = rank_across[metric_rank_cols].sum(axis=1)
    rank_across["RankMean"] = rank_across[metric_rank_cols].mean(axis=1)
    rank_across_out = rank_across[["Method", "Round"] + metric_rank_cols + ["RankSum", "RankMean"]]
    rank_across_out = rank_across_out.sort_values("RankSum")
    rank_across_out.to_csv(root_dir / f"metric_round_rank{gene_number}.csv", index=False)



    top3_rounds_vec = rank_across_out.head(3)["Round"].to_numpy()
    return top3_rounds_vec
import pandas as pd
import numpy as np

def evaluate_prediction_vs_reference(
    result_csv_path,
    df_grouped_means,
    st_data,
    output_path,
    anno_col="annotation"
):
  

    result = pd.read_csv(result_csv_path)
    result = result.iloc[:, 1:]  

    max_cell_type = result.idxmax(axis=1)
    max_cell_type = (
        max_cell_type
        .str.replace("meanscell_abundance_w_sf_", "")
        .str.replace("-", "")
        .str.replace("_", "")
    )

    st_data.obs[anno_col] = max_cell_type.values


    df_grouped_means.index = df_grouped_means.index.str.replace("_", "")
    df_grouped_means.index = df_grouped_means.index.str.replace("-", "")
    df_grouped_means.index = df_grouped_means.index.str.replace("/", " ")

    results_df = pd.DataFrame(columns=["Cell", "Type", "SSIM", "KL", "JS", "COSINE", "PEARSON", "MSE"])

    for cell in st_data.obs_names:
        cell_type = st_data.obs.loc[cell, anno_col]
        cell_type = cell_type.replace("-", "").replace("/", " ")

        if cell_type not in df_grouped_means.index:
            continue  

        real_expression = st_data[cell, :].X.toarray().squeeze().astype('float')
        avg_expression  = df_grouped_means.loc[cell_type].values.astype('float')

        ssim_val  = ssim(real_expression, avg_expression)
        kl_val    = KL(real_expression, avg_expression)
        js_val    = JS(real_expression, avg_expression)
        cosi_val  = cosi(real_expression, avg_expression)
        pear_val  = pear(real_expression, avg_expression)
        mse_val   = mse(real_expression, avg_expression)

        new_row = {
            "Cell": cell, 
            "Type": cell_type, 
            "SSIM": ssim_val,
            "KL": kl_val,
            "JS": js_val,
            "COSINE": cosi_val,
            "PEARSON": pear_val,
            "MSE": mse_val
        }
        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

    results_df.to_csv(output_path, index=False)




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
