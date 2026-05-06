#!/bin/bash

set -e

# ============================================================================
# Run metrics for all nn_retraining_with_pseudolabels_mol_emb_subsample*
# methods across all stability* subfolders that contain prediction files.
#
# Discovers combinations automatically from the results directory — no need
# to update this script when new methods or stability subfolders are added.
# ============================================================================

RESULTS_DIR="./data/benchmark/results/methods"
METHOD_PATTERN="nn_retraining_with_pseudolabels_mol_emb_subsample"

for method_dir in "${RESULTS_DIR}/${METHOD_PATTERN}"*/; do
    [ -d "$method_dir" ] || continue
    method_name=$(basename "$method_dir")

    for stability_dir in "${method_dir}"stability*/; do
        [ -d "$stability_dir" ] || continue

        # Skip if no prediction files are present
        shopt -s nullglob
        pred_files=("${stability_dir}"predictions_seed_*.h5ad)
        shopt -u nullglob
        [ ${#pred_files[@]} -eq 0 ] && continue

        subfolder=$(basename "$stability_dir")

        echo "> method=${method_name}  subfolder=${subfolder}  (${#pred_files[@]} prediction file(s))"
        ./run_pipelines/run_metrics_stability.sh \
            -m "${method_name}" \
            -b "${subfolder}"
    done
done
