#!/usr/bin/env Rscript
# =====================================================================
# RCTD benchmark wrapper (spacexr).
#
# Usage:
#   Rscript run_rctd.R --dataset <NAME> --rds_dir results/rds/ --out_dir results/
#
# Writes:
#   {out_dir}/{dataset}_RCTD_seed{seed}_prediction.csv  (cell_id, predicted_label)
#
# RCTD's quality filter drops some ST cells; these are written with NA in
# predicted_label and evaluate.py counts them as misclassified.
# =====================================================================
suppressPackageStartupMessages({
    library(optparse); library(Matrix); library(spacexr)
})

opt <- parse_args(OptionParser(option_list = list(
    make_option("--dataset", type = "character", help = "Dataset key from datasets.py"),
    make_option("--rds_dir", type = "character", help = "Directory with the 4 CSVs from prepare_for_R.py"),
    make_option("--out_dir", type = "character", help = "Where to write the prediction CSV"),
    make_option("--seed",    type = "integer",   default = 0L),
    make_option("--n_cores", type = "integer",   default = 8L)
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

cat(sprintf("SC: %d x %d  ST: %d x %d  shared genes: %d\n",
            nrow(sc_counts), ncol(sc_counts),
            nrow(st_counts), ncol(st_counts), nrow(sc_counts)))

# ---- Build Reference + SpatialRNA ----
sc_counts_m <- as(as.matrix(sc_counts), "dgCMatrix")
cell_types  <- as.factor(sc_labels$cell_type); names(cell_types) <- sc_labels$cell_id
ref <- Reference(sc_counts_m, cell_types)

st_counts_m <- as(as.matrix(st_counts), "dgCMatrix")
coords <- data.frame(x = st_coords$x, y = st_coords$y, row.names = st_coords$cell_id)
puck <- SpatialRNA(coords, st_counts_m)

# ---- Run RCTD (doublet mode for per-cell assignment) ----
rctd <- create.RCTD(puck, ref, max_cores = opt$n_cores, UMI_min = 0)
rctd <- run.RCTD(rctd, doublet_mode = "doublet")

results <- rctd@results$results_df
# Dominant cell type per location
pred_df <- data.frame(
    cell_id         = rownames(results),
    predicted_label = as.character(results$first_type),
    stringsAsFactors = FALSE
)

# Re-attach cells that RCTD filtered out (will be NA in predicted_label)
all_cells <- colnames(st_counts)
missing   <- setdiff(all_cells, pred_df$cell_id)
if (length(missing) > 0) {
    pred_df <- rbind(pred_df,
        data.frame(cell_id = missing, predicted_label = NA_character_,
                   stringsAsFactors = FALSE))
    cat(sprintf("RCTD quality-filtered %d / %d cells (kept as NA)\n",
                length(missing), length(all_cells)))
}

out_csv <- file.path(opt$out_dir,
    sprintf("%s_RCTD_seed%d_prediction.csv", opt$dataset, opt$seed))
write.csv(pred_df, out_csv, row.names = FALSE)
cat("wrote", out_csv, "\n")
