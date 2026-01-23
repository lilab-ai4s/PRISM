import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
import scanpy as sc

def concat_self_neighbor_expression(
    adata,
    x_key="x",
    y_key="y",
    k=15,
    include_self=False,
    layer=None,
    layer_key_for_raw="raw_counts"
):
    """

    """
    adata0 = adata.copy()
    N, G = adata0.n_obs, adata0.n_vars

    coords = adata0.obs[[x_key, y_key]].to_numpy(float)
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nbrs.kneighbors(coords)
    nbr_idx = idx[:, 1:] if not include_self else idx[:, :k]

  
    X_source = adata0.layers[layer] if layer else adata0.X
    X_dense = X_source.toarray() if sp.issparse(X_source) else X_source

 
    X_nb = X_dense[nbr_idx].mean(axis=1)
    X_concat = np.hstack([X_dense, X_nb])  # [N, 2G]


    var_nb = adata0.var.copy()
    var_nb.index = var_nb.index + "_nb"
    var_concat = pd.concat([adata0.var, var_nb])


    adata_new = sc.AnnData(
        X=sp.csr_matrix(X_concat),
        obs=adata0.obs.copy(),
        var=var_concat,
        obsm=adata0.obsm.copy(),
    )

  
    X_raw = X_source
    zeros = sp.csr_matrix((N, G))
    raw_pad = sp.hstack([X_raw, zeros], format='csr')
    adata_new.layers[layer_key_for_raw] = raw_pad

    return adata_new
