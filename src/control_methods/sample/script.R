#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(reticulate))
conda_env <- Sys.getenv("CONDA_PREFIX")
if (nzchar(conda_env)) use_python(file.path(conda_env, "bin/python"), required = TRUE)

suppressPackageStartupMessages({
  library(anndata)
  library(argparse)
})


main <- function() {
  parser <- ArgumentParser(description = "Sample baseline — randomly samples training values as predictions.")
  parser$add_argument("--de_train", type = "character", required = TRUE, help = "Path to de_train.h5ad")
  parser$add_argument("--id_map",   type = "character", required = TRUE, help = "Path to id_map.csv")
  parser$add_argument("--layer",    type = "character", default = "clipped_sign_log10_pval", help = "AnnData layer to use")
  parser$add_argument("--output",   type = "character", required = TRUE, help = "Path to output .h5ad file")

  par <- parser$parse_args()

  cat("Read data\n")
  de_train   <- read_h5ad(par$de_train)
  id_map     <- read.csv(par$id_map)
  gene_names <- de_train$var_names
  input_layer <- de_train$layers[[par$layer]]

  prediction <- sapply(gene_names, function(gene_name) {
    sample(input_layer[, gene_name], size = nrow(id_map), replace = TRUE)
  })
  rownames(prediction) <- id_map$id

  output <- AnnData(
    layers = list(prediction = prediction),
    var    = de_train$var[, character(0)],
    shape  = c(nrow(id_map), length(gene_names)),
    uns    = list(
      dataset_id = de_train$uns[["dataset_id"]],
      method_id  = "sample"
    )
  )

  cat("Write output\n")
  output_dir <- dirname(par$output)
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  output$write_h5ad(par$output, compression = "gzip")
}

main()
