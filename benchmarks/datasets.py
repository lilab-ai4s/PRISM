#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Centralised dataset registry for the PRISM benchmarks.

Each runner imports `DATASETS` and `load_sc_st_raw` from this module so the
data paths, annotation columns, label remappings, and spatial coordinate
columns are defined in exactly one place.

Loaders return AnnData objects with INTEGER counts in `.X` (required by
RCTD / Cell2location / DWLS).  Tangram / Spatial-ID / DSCT normalise
internally as needed.

Public API
----------
DATASETS                                 dict of dataset configs
load_sc_st_raw(name)                     -> (sc_adata, st_adata)
get_ground_truth(name)                   -> pd.Series indexed by cell_id
"""
import os
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp


# =====================================================================
# Path constants (override via PRISM_DATA_ROOT environment variable, or edit
# the per-dataset subdirectories below to match your local layout).
# =====================================================================
DATA_ROOT       = os.environ.get("PRISM_DATA_ROOT", "./data")
HPF_DIR         = os.path.join(DATA_ROOT, "HPF")
OB_DIR          = os.path.join(DATA_ROOT, "OB")
CTX_MOUSE_DIR   = os.path.join(DATA_ROOT, "CTX_mouse")
CTX_MERFISH_DIR = os.path.join(DATA_ROOT, "CTX_merfish")
LIVER_DIR       = os.path.join(DATA_ROOT, "Liver")
HCC_DATA_DIR    = os.path.join(DATA_ROOT, "HCC")


# =====================================================================
# Liver SC/ST label maps (consolidate SC and ST vocabularies)
# =====================================================================
LIVER_SC_MAP = {
    "hepatocyte":                           "hepatocyte",
    "Kupffer cell":                         "Kupffer_cell",
    "endothelial cell of hepatic sinusoid": "endothelial_cell",
    "natural killer cell":                  "immune_cell",
    "B cell":                               "immune_cell",
    "myeloid leukocyte":                    "immune_cell",
    "plasmacytoid dendritic cell":          "immune_cell",
    "hepatic stellate cell":                "hepatic_stellate_cell",
    "duct epithelial cell":                 None,
}
LIVER_ST_MAP = {
    "periportal hepatocyte":                "hepatocyte",
    "pericentral hepatocyte":               "hepatocyte",
    "Kupffer cell":                         "Kupffer_cell",
    "periportal endothelial cell":          "endothelial_cell",
    "pericentral endothelial cell":         "endothelial_cell",
    "immune cell":                          "immune_cell",
    "hepatic stellate cell":                "hepatic_stellate_cell",
    "bile duct epithelial cell":            None,
}


# =====================================================================
# CTX subset whitelists (excitatory L / inhibitory I / non-neuronal O)
# =====================================================================
CTX_L_TYPES = ["L2_3_IT", "L4_5_IT", "L5_IT", "L5_PT", "L6_CT", "L6_IT", "L6b", "NP"]
CTX_I_TYPES = ["iLAMP5", "iPVALB", "iSST", "iVIP"]
CTX_O_TYPES = ["lASC", "lMGC", "lOLG", "lOPC", "lENDO", "lVLMC"]


# =====================================================================
# DATASETS registry
# =====================================================================
DATASETS = {
    # ---------- brain MERFISH (3 labeled benchmarks) ----------
    "HIP_merfish": {
        "sc_path":   f"{HPF_DIR}/sc.h5ad",
        "st_path":   f"{HPF_DIR}/st.h5ad",
        "sc_loader": "scanpy", "st_loader": "scanpy",
        "sc_gene_col": "gene_symbol", "st_gene_col": "gene_symbol",
        "lowercase_genes": False,
        "sc_anno_in": "subclass", "st_anno_in": "subclass",
        "anno_final": "subclass",
        "sc_label_map": None, "st_label_map": None,
        "st_use_counts": False,
        "x_col": "x", "y_col": "y",
    },
    "OB_merfish": {
        "sc_path":   f"{OB_DIR}/sc.h5ad",
        "st_path":   f"{OB_DIR}/st.h5ad",
        "sc_loader": "scanpy", "st_loader": "scanpy",
        "sc_gene_col": "gene_symbol", "st_gene_col": "gene_symbol",
        "lowercase_genes": False,
        "sc_anno_in": "subclass", "st_anno_in": "subclass",
        "anno_final": "subclass",
        "sc_label_map": None, "st_label_map": None,
        "st_use_counts": False,
        "x_col": "x", "y_col": "y",
    },
    "CTX_mouse": {
        "sc_path":   f"{CTX_MOUSE_DIR}/sc.h5",
        "st_path":   f"{CTX_MOUSE_DIR}/st.h5",
        "sc_loader": "diopy", "st_loader": "diopy",
        "sc_gene_col": None, "st_gene_col": None,
        "lowercase_genes": False,
        "sc_anno_in": "Type", "st_anno_in": "predict_modified",
        "anno_final": "Type",
        "sc_label_map": None, "st_label_map": None,
        "st_use_counts": False,
        "x_col": "imagerow", "y_col": "imagecol",
        # SC has fine labels (Pvalb_A, ...) while ST uses coarse (iPVALB, lASC, ...).
        # evaluate.py applies the standard SC->ST remap before comparison.
    },
    # CTX MERFISH L / I / O subsets -- labelled cortex slices with curated
    # cell-type sets. Place the per-subset ST file under CTX_merfish/.
    "CTX_mouse_L": {
        "sc_path":   f"{CTX_MOUSE_DIR}/sc.h5",
        "st_path":   f"{CTX_MERFISH_DIR}/st_L.h5",
        "sc_loader": "diopy", "st_loader": "diopy",
        "sc_gene_col": None, "st_gene_col": None,
        "lowercase_genes": False,
        "sc_anno_in": "Type", "st_anno_in": "predict_modified",
        "anno_final": "Type",
        "sc_label_map": None, "st_label_map": None,
        "st_use_counts": False,
        "x_col": "imagerow", "y_col": "imagecol",
        "ctx_subset": "L",
    },
    "CTX_mouse_I": {
        "sc_path":   f"{CTX_MOUSE_DIR}/sc.h5",
        "st_path":   f"{CTX_MERFISH_DIR}/st_I.h5",
        "sc_loader": "diopy", "st_loader": "diopy",
        "sc_gene_col": None, "st_gene_col": None,
        "lowercase_genes": False,
        "sc_anno_in": "Type", "st_anno_in": "predict_modified",
        "anno_final": "Type",
        "sc_label_map": None, "st_label_map": None,
        "st_use_counts": False,
        "x_col": "imagerow", "y_col": "imagecol",
        "ctx_subset": "I",
    },
    "CTX_mouse_O": {
        "sc_path":   f"{CTX_MOUSE_DIR}/sc.h5",
        "st_path":   f"{CTX_MERFISH_DIR}/st_O.h5",
        "sc_loader": "diopy", "st_loader": "diopy",
        "sc_gene_col": None, "st_gene_col": None,
        "lowercase_genes": False,
        "sc_anno_in": "Type", "st_anno_in": "predict_modified",
        "anno_final": "Type",
        "sc_label_map": None, "st_label_map": None,
        "st_use_counts": False,
        "x_col": "imagerow", "y_col": "imagecol",
        "ctx_subset": "O",
    },

    # ---------- non-brain ----------
    "Liver_merfish": {
        "sc_path":   f"{LIVER_DIR}/sc.h5ad",
        "st_path":   f"{LIVER_DIR}/st.h5ad",
        "sc_loader": "scanpy", "st_loader": "scanpy",
        "sc_gene_col": "feature_name", "st_gene_col": None,
        "lowercase_genes": True,
        "sc_anno_in": "cell_type", "st_anno_in": "free_annotation",
        "anno_final": "unified_type",
        "sc_label_map": LIVER_SC_MAP, "st_label_map": LIVER_ST_MAP,
        "st_use_counts": True,
        "x_col": "center_x", "y_col": "center_y",
    },
    "HCC_cosmx": {
        "sc_path":   f"{HCC_DATA_DIR}/sc.h5ad",
        "st_path":   f"{HCC_DATA_DIR}/st.h5ad",
        "sc_loader": "scanpy", "st_loader": "scanpy",
        "sc_gene_col": None, "st_gene_col": None,
        "lowercase_genes": False,
        "sc_anno_in": "major_annotation", "st_anno_in": "annotation",
        "anno_final": "unified_type",
        "sc_label_map": None, "st_label_map": None,
        "st_use_counts": False,
        "x_col": "x", "y_col": "y",
    },
}


# =====================================================================
# Loader helpers
# =====================================================================
def _load_adata_raw(path, loader):
    if loader == "diopy":
        import diopy
        return diopy.input.read_h5(path)
    import scanpy as sc
    return sc.read_h5ad(path)


def _to_integer_counts(adata, tag=""):
    """Return adata copy with integer-count `.X` (RCTD / Cell2location / DWLS need this)."""
    X = adata.X
    src = "X"
    if "counts" in adata.layers:
        X = adata.layers["counts"]; src = "layers['counts']"
    elif getattr(adata, "raw", None) is not None:
        try:
            X = adata.raw.X; src = "raw.X"
        except AttributeError:
            pass

    sample = X[:50, :50]
    if sp.issparse(sample):
        sample = sample.toarray()
    sample = np.asarray(sample)
    is_int = np.all(sample == np.round(sample))
    if not is_int:
        print(f"    [WARN] {tag}: {src} non-integer; applying expm1+round")
        if sp.issparse(X):
            X = X.copy()
            X.data = np.round(np.expm1(X.data)).astype(np.float32)
            X.eliminate_zeros()
        else:
            X = np.round(np.expm1(X)).astype(np.float32)
        src += " + expm1+round"

    out = adata.copy()
    out.X = X
    print(f"    {tag}: counts from {src}")
    return out


def _apply_gene_symbol(adata, gene_col, lowercase):
    if gene_col and gene_col in adata.var.columns:
        adata.var_names = adata.var[gene_col].astype(str)
    if lowercase:
        adata.var_names = adata.var_names.str.lower()
    adata.var_names_make_unique()
    return adata


def _apply_label_map(adata, anno_in, anno_final, label_map):
    if label_map is not None:
        adata.obs[anno_final] = adata.obs[anno_in].astype(str).map(label_map)
        adata = adata[adata.obs[anno_final].notna()].copy()
    else:
        adata.obs[anno_final] = adata.obs[anno_in].astype(str)
    return adata


def _apply_ctx_subset(sc_obj, st_obj, subset_key, anno_col):
    """For CTX_mouse_L/I/O: filter SC to the curated whitelist for that subset.
    ST is already pre-split by subset (one h5 per subset)."""
    whitelist = {"L": CTX_L_TYPES, "I": CTX_I_TYPES, "O": CTX_O_TYPES}[subset_key]
    # SC: fine labels -- match by str.split('_')[0] or direct whitelist
    keep = sc_obj.obs[anno_col].astype(str).isin(whitelist) | \
           sc_obj.obs[anno_col].astype(str).str.split("_").str[0].isin(
               [w.split("_")[0] for w in whitelist])
    return sc_obj[keep].copy(), st_obj


def load_sc_st_raw(name):
    """Load SC + ST AnnData for the given dataset name. Returns (sc_obj, st_obj)
    with integer counts in `.X` and `obs[anno_final]` populated."""
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset {name!r}. Available: {list(DATASETS)}")
    cfg = DATASETS[name]
    anno = cfg["anno_final"]

    # ---- SC ----
    sc_obj = _load_adata_raw(cfg["sc_path"], cfg.get("sc_loader", "scanpy"))
    sc_obj = _apply_gene_symbol(sc_obj, cfg.get("sc_gene_col"),
                                cfg.get("lowercase_genes", False))
    sc_obj = _apply_label_map(sc_obj, cfg["sc_anno_in"], anno,
                              cfg.get("sc_label_map"))
    counts = sc_obj.obs[anno].value_counts()
    sc_obj = sc_obj[sc_obj.obs[anno].isin(counts[counts > 5].index)].copy()
    sc_obj.obs[anno] = sc_obj.obs[anno].astype("category")
    sc_obj = _to_integer_counts(sc_obj, f"{name}/SC")

    # ---- ST ----
    st_obj = _load_adata_raw(cfg["st_path"], cfg.get("st_loader", "scanpy"))
    if cfg.get("st_use_counts") and "counts" in st_obj.layers:
        st_obj.X = st_obj.layers["counts"].copy()
    st_obj = _apply_gene_symbol(st_obj, cfg.get("st_gene_col"),
                                cfg.get("lowercase_genes", False))
    st_obj = _apply_label_map(st_obj, cfg["st_anno_in"], anno,
                              cfg.get("st_label_map"))
    x_col, y_col = cfg.get("x_col", "x"), cfg.get("y_col", "y")
    if x_col != "x" and x_col in st_obj.obs.columns:
        st_obj.obs["x"] = st_obj.obs[x_col].astype(float)
    if y_col != "y" and y_col in st_obj.obs.columns:
        st_obj.obs["y"] = st_obj.obs[y_col].astype(float)
    st_obj = _to_integer_counts(st_obj, f"{name}/ST")

    # ---- CTX subset filter (L / I / O) ----
    if cfg.get("ctx_subset"):
        sc_obj, st_obj = _apply_ctx_subset(
            sc_obj, st_obj, cfg["ctx_subset"], anno)

    return sc_obj, st_obj


def get_ground_truth(name):
    """Return ST ground-truth labels indexed by st_adata.obs_names."""
    _, st_obj = load_sc_st_raw(name)
    return pd.Series(
        st_obj.obs[DATASETS[name]["anno_final"]].astype(str).values,
        index=st_obj.obs_names.astype(str),
        name="ground_truth",
    )
