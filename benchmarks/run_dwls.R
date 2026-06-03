#!/usr/bin/env Rscript
# =====================================================================
# SpatialDWLS benchmark wrapper (Giotto).
#
# Usage:
#   Rscript run_dwls.R --dataset <NAME> --rds_dir results/rds/ --out_dir results/
#
# Writes:
#   {out_dir}/{dataset}_SpatialDWLS_seed{seed}_prediction.csv
#
# We report the argmax over the cell-type proportion vector as the
# categorical prediction. Cells where DWLS fails to converge are written
# with NA in predicted_label (counted as misclassified by evaluate.py).
# =====================================================================
suppressPackageStartupMessages({
    library(optparse); library(Matrix); library(Giotto)
})

opt <- parse_args(OptionParser(option_list = list(
    make_option("--dataset", type = "character"),
    make_option("--rds_dir", type = "character"),
    make_option("--out_dir", type = "character"),
    make_option("--seed",    type = "integer", default = 0L),
    make_option("--n_cores", type = "integer", default = 8L)
)))
dir.create(opt$out_dir, showWarnings = FALSE, recursive = TRUE)
set.seed(opt$seed)

# ---- Load SC + ST ----
sc_counts  <- read.csv(file.path(opt$rds_dir, paste0(opt$dataset, "_sc_counts.csv")),
                       row.names = 1, check.names = FALSE)
sc_labels  <- read.csv(file.path(opt$rds_dir, paste0(opt$dataset, "_sc_labels.csv")))
st_counts  <- read.csv(file.path(opt$rds_dir, paste0(opt$dataset, "_st_counts.csv")),
                       row.names = 1, check.names = FALSE)
st_coords  <- read.csv(file.path(opt$rds_dir, paste0(opt$dataset, "_st_coords.csv")))

sc_counts_m <- as(as.matrix(sc_counts), "dgCMatrix")
st_counts_m <- as(as.matrix(st_counts), "dgCMatrix")
coords <- data.frame(sdimx = st_coords$x, sdimy = st_coords$y,
                     row.names = st_coords$cell_id)

# ---- Build Giotto object for ST ----
gobj <- createGiottoObject(raw_exprs = st_counts_m, spatial_locs = coords)
gobj <- normalizeGiotto(gobj)
gobj <- addStatistics(gobj)
gobj <- calculateHVG(gobj)
gobj <- runPCA(gobj, genes_to_use = NULL, scale_unit = FALSE)

# ---- Build signature matrix from SC ----
labels <- setNames(as.character(sc_labels$cell_type), sc_labels$cell_id)
sig_mat <- makeSignMatrixDWLSfromMatrix(
    matrix = sc_counts_m,
    cell_type_vector = labels[colnames(sc_counts_m)],
    sign_gene = rownames(sc_counts_m)
)

# ---- Run SpatialDWLS ----
gobj <- runDWLSDeconv(gobj, sign_matrix = sig_mat,
                      n_cell = 1, cluster_column = NULL)
dwls_props <- gobj@spatial_enrichment$DWLS  # cell_id x cell_type

pred <- apply(as.matrix(dwls_props[, -1, drop = FALSE]), 1, function(v) {
    if (all(is.na(v)) || sum(v, na.rm = TRUE) == 0) NA_character_
    else colnames(dwls_props)[-1][which.max(v)]
})
pred_df <- data.frame(cell_id = dwls_props$cell_ID,
                      predicted_label = unname(pred),
                      stringsAsFactors = FALSE)

# Re-attach missing cells
all_cells <- colnames(st_counts)
missing   <- setdiff(all_cells, pred_df$cell_id)
if (length(missing) > 0) {
    pred_df <- rbind(pred_df,
        data.frame(cell_id = missing, predicted_label = NA_character_,
                   stringsAsFactors = FALSE))
    cat(sprintf("DWLS dropped %d / %d cells (kept as NA)\n",
                length(missing), length(all_cells)))
}

out_csv <- file.path(opt$out_dir,
    sprintf("%s_SpatialDWLS_seed%d_prediction.csv", opt$dataset, opt$seed))
write.csv(pred_df, out_csv, row.names = FALSE)
cat("wrote", out_csv, "\n")
