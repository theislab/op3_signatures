#!/bin/bash

set -e

# ============================================================================
# Stability sweep, coverage-matched (134 OP3 compounds) counterpart of
# run_stability_pipeline_full_coverage.sh
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
# Also runs the LPM ablation and step-matched checkpoints with explicit -T
# tags (see ABL_CONFIGS below).
#
# The none baseline and the l1000 and narrowed (114-compound) runs are
# identical in both coverage setups, so they are commented out here and
# produced by the full-coverage script.
#
# Usage:
#   ./run_stability_pipeline_coverage_match.sh            # sweep default EPOCHS
#   ./run_stability_pipeline_coverage_match.sh 5 10 15    # sweep just these
# ============================================================================

EPOCHS=("$@")
if [ "${#EPOCHS[@]}" -eq 0 ]; then
    EPOCHS=(20)
fi

# Run folders: ./data_<RUN_TAG>/benchmark (inputs + results) and ./logs_<RUN_TAG>.
# Empty -> ./data and ./logs.
export RUN_TAG="coverage_match"

RUNNER=./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh
DATA_BASE="./data${RUN_TAG:+_${RUN_TAG}}/benchmark/resources/datasets/neurips-2023-data-subsample"

# Ablation / step-matched checkpoints: "<embedding file>:<results folder tag>"
ABL_CONFIGS=(
    "op3_emb_all_20_data_dataset_coverage_match.pkl:all_20_abl_dataset_coverage_match"
    "op3_emb_all_20_data_dataset_dose_time_coverage_match.pkl:all_20_abl_dataset_dose_time_coverage_match"
    "op3_emb_all_20_data_dose_time_coverage_match.pkl:all_20_abl_dose_time_coverage_match"
    "op3_emb_all_last_stepmatched_coverage_match.pkl:all_last_stepmatched_coverage_match"
    #"op3_emb_l1000_20_dataset.pkl:l1000_20_abl_dataset"
    #"op3_emb_l1000_20_dataset_dose_time.pkl:l1000_20_abl_dataset_dose_time"
    #"op3_emb_l1000_20_dose_time.pkl:l1000_20_abl_dose_time"
    #"op3_emb_l1000_last_stepmatched.pkl:l1000_last_stepmatched"
)

# Fail before submitting anything if an ablation embedding is missing.
MISSING=0
for CFG in "${ABL_CONFIGS[@]}"; do
    EMB_FILE="${CFG%%:*}"
    if [ ! -f "${DATA_BASE}/${EMB_FILE}" ]; then
        echo "Error: missing embedding ${DATA_BASE}/${EMB_FILE}" >&2
        MISSING=1
    fi
done
[ "$MISSING" -eq 0 ] || exit 1

# File holding the epoch-independent embeddings (ECFP:2 column). ECFP doesn't
# change with LPM training epoch, so we use a dedicated epoch-agnostic file.
# Note: the inner runner auto-derives the tag from the filename
# (op3_emb_<X>.pkl -> "<X>"), so the fp results land in folders like
# ..._fp_concat_not_dense_fp/.
FP_FILE="op3_emb_fp.pkl"

# Run the embedding-free baseline once (no need to repeat per epoch).
#"$RUNNER" -e none -D subsample

# Fingerprint (fp / ECFP:2) embeddings are deterministic from molecular
# structure -- they don't change with LPM training epoch -- so run them once.
echo ""
echo "============================================================"
echo "  Running fp (epoch-independent) configs once  (file: ${FP_FILE})"
echo "============================================================"
#"$RUNNER" -e fp -l concat    -D subsample      -f "$FP_FILE"
#"$RUNNER" -e fp -l concat -d -D subsample      -f "$FP_FILE"
"$RUNNER" -e fp -l fixed     -D subsample      -f "$FP_FILE"
"$RUNNER" -e fp -l trainable -D subsample      -f "$FP_FILE"

# ChemBERTa embeddings (epoch-independent), fixed + trainable × 10 seeds.
CHEMB1_FILE="op3_emb_chemb1.pkl"
CHEMB2_FILE="op3_emb_chemb2.pkl"
echo ""
echo "============================================================"
echo "  Running chemb1/chemb2 (ChemBERTa) configs  (files: ${CHEMB1_FILE}, ${CHEMB2_FILE})"
echo "============================================================"
"$RUNNER" -e chemb1 -l fixed     -D subsample      -f "$CHEMB1_FILE"
"$RUNNER" -e chemb1 -l trainable -D subsample      -f "$CHEMB1_FILE"
"$RUNNER" -e chemb2 -l fixed     -D subsample      -f "$CHEMB2_FILE"
"$RUNNER" -e chemb2 -l trainable -D subsample      -f "$CHEMB2_FILE"

# MoLFormer-XL embeddings (epoch-independent), fixed + trainable × 10 seeds.
MOLFORMER_FILE="op3_emb_molformer.pkl"
echo ""
echo "============================================================"
echo "  Running molformer (MoLFormer-XL) configs  (file: ${MOLFORMER_FILE})"
echo "============================================================"
"$RUNNER" -e molformer -l fixed     -D subsample      -f "$MOLFORMER_FILE"
"$RUNNER" -e molformer -l trainable -D subsample      -f "$MOLFORMER_FILE"

# lpm embeddings DO change with epoch, so sweep them.
for E in "${EPOCHS[@]}"; do
    EMB_FILE="op3_emb_all_${E}.pkl"
    echo ""
    echo "============================================================"
    echo "  Sweeping epoch ${E}  (embedding file: ${EMB_FILE})"
    echo "============================================================"
    #"$RUNNER" -e lpm -l concat    -D subsample      -f "$EMB_FILE"
    #"$RUNNER" -e lpm -l concat -d -D subsample      -f "$EMB_FILE"
    "$RUNNER" -e lpm -l fixed     -D subsample      -f "$EMB_FILE"
    "$RUNNER" -e lpm -l trainable -D subsample      -f "$EMB_FILE"
done

# all_20 restricted to the 114 l1000 compounds (fixed epoch, not swept).
#NARROWED_FILE="op3_emb_all_20_narrowed.pkl"
#echo ""
#echo "============================================================"
#echo "  Running narrowed lpm configs  (file: ${NARROWED_FILE})"
#echo "============================================================"
#"$RUNNER" -e lpm -l fixed     -D subsample      -f "$NARROWED_FILE"
#"$RUNNER" -e lpm -l trainable -D subsample      -f "$NARROWED_FILE"

# lpm embeddings DO change with epoch, so sweep them.
#for E in "${EPOCHS[@]}"; do
#    EMB_FILE="op3_emb_l1000_${E}.pkl"
#    echo ""
#    echo "============================================================"
#    echo "  Sweeping epoch ${E}  (embedding file: ${EMB_FILE})"
#    echo "============================================================"
#    #"$RUNNER" -e lpm -l concat    -D subsample      -f "$EMB_FILE"
#    #"$RUNNER" -e lpm -l concat -d -D subsample      -f "$EMB_FILE"
#    "$RUNNER" -e lpm -l fixed     -D subsample      -f "$EMB_FILE"
#    "$RUNNER" -e lpm -l trainable -D subsample      -f "$EMB_FILE"
#done

# Ablation / step-matched checkpoints (fixed epochs, not swept).
for CFG in "${ABL_CONFIGS[@]}"; do
    EMB_FILE="${CFG%%:*}"
    EMB_TAG="${CFG##*:}"

    echo ""
    echo "============================================================"
    echo "  ${EMB_TAG}  (embedding file: ${EMB_FILE})"
    echo "============================================================"
    "$RUNNER" -e lpm -l fixed     -D subsample -f "$EMB_FILE" -T "$EMB_TAG"
    "$RUNNER" -e lpm -l trainable -D subsample -f "$EMB_FILE" -T "$EMB_TAG"
done
