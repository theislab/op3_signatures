#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(reticulate))
conda_env <- Sys.getenv("CONDA_PREFIX")
if (nzchar(conda_env)) use_python(file.path(conda_env, "bin/python"), required = TRUE)

suppressPackageStartupMessages({
  library(anndata)
  library(argparse)
})


main <- function() {
  parser <- ArgumentParser(description = "Compute mean rowwise RMSE and MAE between predictions and ground truth.")
  parser$add_argument("--de_test",          type = "character", required = TRUE,  help = "Path to de_test.h5ad")
  parser$add_argument("--de_test_layer",    type = "character", default = "clipped_sign_log10_pval", help = "Layer to use from de_test")
  parser$add_argument("--prediction",       type = "character", required = TRUE,  help = "Path to predictions.h5ad")
  parser$add_argument("--prediction_layer", type = "character", default = "prediction", help = "Layer to use from predictions")
  parser$add_argument("--resolve_genes",    type = "character", default = "de_test", choices = c("de_test", "intersection"), help = "How to resolve gene sets")
  parser$add_argument("--output",           type = "character", required = TRUE,  help = "Path to write output .h5ad")

  par <- parser$parse_args()

  cat("Load data\n")
  de_test    <- read_h5ad(par$de_test)
  cat("de_test: "); print(de_test)
  prediction <- read_h5ad(par$prediction)
  cat("prediction: "); print(prediction)

  cat("Resolve genes\n")
  genes <- if (par$resolve_genes == "de_test") {
    de_test$var_names
  } else {
    intersect(de_test$var_names, prediction$var_names)
  }
  de_test    <- de_test[, genes]
  prediction <- prediction[, genes]

  de_test_X    <- de_test$layers[[par$de_test_layer]]
  prediction_X <- prediction$layers[[par$prediction_layer]]

  if (any(is.na(de_test_X))) stop("NA values in de_test_X")
  if (any(is.na(prediction_X))) {
    warning("NA values in prediction_X")
    prediction_X[is.na(prediction_X)] <- 0
  }

  cat("Calculate mean rowwise RMSE\n")
  mean_rowwise_rmse <- mean(sqrt(rowMeans((de_test_X - prediction_X)^2)))

  cat("Calculate mean rowwise MAE\n")
  mean_rowwise_mae <- mean(rowMeans(abs(de_test_X - prediction_X)))

  cat("Create output\n")
  output <- AnnData(
    shape = c(0L, 0L),
    uns = list(
      dataset_id    = de_test$uns[["dataset_id"]],
      method_id     = prediction$uns[["method_id"]],
      metric_ids    = c("mean_rowwise_rmse", "mean_rowwise_mae"),
      metric_values = zapsmall(c(mean_rowwise_rmse, mean_rowwise_mae), 10)
    )
  )

  cat("mean_rowwise_rmse:", mean_rowwise_rmse, "\n")
  cat("mean_rowwise_mae: ", mean_rowwise_mae,  "\n")

  cat("Write output\n")
  output_dir <- dirname(par$output)
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  output$write_h5ad(par$output, compression = "gzip")
}

main()
