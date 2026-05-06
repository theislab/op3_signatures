#!/bin/bash

set -e

# ============================================================================
# Stability sweep across multiple LPM/FP embedding versions
# (learning_missed flavour: fp/lpm columns may contain None values; missing
# compounds get a learnable fallback embedding instead of a zero vector).
#
# Embeddings live under
#   data/benchmark/resources/datasets/neurips-2023-data-subsample/op3_emb_<X>.pkl
# where <X> is e.g. an epoch number (op3_emb_5.pkl). The inner runner
# auto-derives the tag from the filename, so each <X> writes to its own
# results folder:
#   data/benchmark/results/methods/<pipeline>_<X>/stability/predictions_seed_*.h5ad
#
# Usage:
#   ./run_stability_pipeline.sh                      # sweep default EPOCHS
#   ./run_stability_pipeline.sh 5 10 15              # sweep just these
# ============================================================================

EPOCHS=("$@")
if [ "${#EPOCHS[@]}" -eq 0 ]; then
    EPOCHS=(20)
fi

RUNNER=./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh

# File holding the epoch-independent embeddings (ECFP:2 column). ECFP doesn't
# change with LPM training epoch, so we use a dedicated epoch-agnostic file.
# Note: the inner runner auto-derives the tag from the filename
# (op3_emb_<X>.pkl -> "<X>"), so the fp results land in folders like
# ..._fp_concat_not_dense_fp/.
FP_FILE="op3_emb_fp.pkl"

# Run the embedding-free baseline once (no need to repeat per epoch).
"$RUNNER" -e none -D subsample

# Fingerprint (fp / ECFP:2) embeddings are deterministic from molecular
# structure -- they don't change with LPM training epoch -- so run them once.
#echo ""
#echo "============================================================"
#echo "  Running fp (epoch-independent) configs once  (file: ${FP_FILE})"
#echo "============================================================"
#"$RUNNER" -e fp -l concat    -D subsample      -f "$FP_FILE"
#"$RUNNER" -e fp -l concat -d -D subsample      -f "$FP_FILE"
"$RUNNER" -e fp -l fixed     -D subsample      -f "$FP_FILE"
"$RUNNER" -e fp -l trainable -D subsample      -f "$FP_FILE"

# lpm embeddings DO change with epoch, so sweep them.
for E in "${EPOCHS[@]}"; do
    EMB_FILE="op3_emb_all_${E}.pkl"
    echo ""
    echo "============================================================"
    echo "  Sweeping epoch ${E}  (embedding file: ${EMB_FILE})"
    echo "============================================================"
    #"$RUNNER" -e lpm -l concat    -D subsample      -f "$EMB_FILE"
    #"$RUNNER" -e lpm -l concat -d -D subsample      -f "$EMB_FILE"
    #"$RUNNER" -e lpm -l fixed     -D subsample      -f "$EMB_FILE"
    "$RUNNER" -e lpm -l trainable -D subsample      -f "$EMB_FILE"
done




# lpm embeddings DO change with epoch, so sweep them.
for E in "${EPOCHS[@]}"; do
    EMB_FILE="op3_emb_l1000_${E}.pkl"
    echo ""
    echo "============================================================"
    echo "  Sweeping epoch ${E}  (embedding file: ${EMB_FILE})"
    echo "============================================================"
    #"$RUNNER" -e lpm -l concat    -D subsample      -f "$EMB_FILE"
    #"$RUNNER" -e lpm -l concat -d -D subsample      -f "$EMB_FILE"
    #"$RUNNER" -e lpm -l fixed     -D subsample      -f "$EMB_FILE"
    "$RUNNER" -e lpm -l trainable -D subsample      -f "$EMB_FILE"
done
