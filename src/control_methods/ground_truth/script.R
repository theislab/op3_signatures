#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(reticulate))
conda_env <- Sys.getenv("CONDA_PREFIX")
if (nzchar(conda_env)) use_python(file.path(conda_env, "bin/python"), required = TRUE)

suppressPackageStartupMessages({
  library(anndata)
  library(argparse)
})


main <- function() {
  parser <- ArgumentParser(description = "Ground-truth control — returns the test labels as predictions.")
  parser$add_argument("--de_test", type = "character", required = TRUE, help = "Path to de_test.h5ad")
  parser$add_argument("--layer",   type = "character", default = "clipped_sign_log10_pval", help = "AnnData layer to use")
  parser$add_argument("--output",  type = "character", required = TRUE, help = "Path to output .h5ad file")

  par <- parser$parse_args()

  cat("Read de_test\n")
  de_test <- read_h5ad(par$de_test)

  output <- AnnData(
    layers = list(prediction = de_test$layers[[par$layer]]),
    obs    = de_test$obs[, character(0)],
    var    = de_test$var[, character(0)],
    uns    = list(
      dataset_id = de_test$uns[["dataset_id"]],
      method_id  = "ground_truth"
    )
  )

  cat("Write output\n")
  output_dir <- dirname(par$output)
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  output$write_h5ad(par$output, compression = "gzip")
}

main()
