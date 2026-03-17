#!/bin/bash

set -e

# ============================================================================
# NN retraining with pseudolabels (subsample) — configurable embedding pipeline
# ============================================================================
#
# Runs a neural network with pseudolabel retraining with optional external
# embeddings concatenated to the input.
# Expects to be called from the project root (directly or via run_pipelines/).
#
# Options:
#   -e  Embedding type: lpm, fp, or none  (default: none)
#   -l  Embedding layer: concat, fixed, or trainable  (default: concat)
#   -D  Dataset: subsample or original  (default: subsample)
#   -d  Use a Dense(256) projection before concatenation  (flag, default: off)
#   -h  Show this help message
#
# Examples:
#   ./pipelines/.../nn_retraining_...sh -e lpm -l concat -d -D subsample
#   ./pipelines/.../nn_retraining_...sh -e fp -l fixed -D original
# ============================================================================

# ============================================================================
# Parse options
# ============================================================================

EMBEDDING_TYPE="none"
EMBEDDING_LAYER="concat"
DATASET="subsample"
USE_FP_DENSE=false

while getopts ":e:l:D:dh" opt; do
    case $opt in
        e) EMBEDDING_TYPE=$OPTARG ;;
        l) EMBEDDING_LAYER=$OPTARG ;;
        D) DATASET=$OPTARG ;;
        d) USE_FP_DENSE=true ;;
        h)
            echo "Usage: $0 [-e embedding_type] [-l embedding_layer] [-D dataset] [-d]"
            echo ""
            echo "Optional:"
            echo "  -e  Embedding type: lpm, fp, or none (default: none)"
            echo "  -l  Embedding layer: concat, fixed, or trainable (default: concat; only when -e is lpm or fp)"
            echo "  -D  Dataset: subsample or original (default: subsample)"
            echo "  -d  Project embeddings through Dense(256) before concatenation (only for -l concat)"
            echo "  -h  Show this help message"
            exit 0
            ;;
        \?) echo "Error: Invalid option -$OPTARG" >&2; exit 1 ;;
        :)  echo "Error: Option -$OPTARG requires an argument" >&2; exit 1 ;;
    esac
done

if [[ ! "$EMBEDDING_TYPE" =~ ^(lpm|fp|none)$ ]]; then
    echo "Error: embedding_type must be 'lpm', 'fp', or 'none', got '${EMBEDDING_TYPE}'" >&2; exit 1
fi

if [[ "$EMBEDDING_TYPE" != "none" ]] && [[ ! "$EMBEDDING_LAYER" =~ ^(concat|fixed|trainable)$ ]]; then
    echo "Error: embedding_layer must be 'concat', 'fixed', or 'trainable', got '${EMBEDDING_LAYER}'" >&2; exit 1
fi

if [[ ! "$DATASET" =~ ^(subsample|original)$ ]]; then
    echo "Error: dataset must be 'subsample' or 'original', got '${DATASET}'" >&2; exit 1
fi

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ "$DATASET" = "subsample" ]; then
    DATA_DIR="neurips-2023-data-subsample"
else
    DATA_DIR="neurips-2023-data"
fi

if [ "$EMBEDDING_TYPE" = "none" ]; then
    PIPELINE_NAME="nn_retraining_with_pseudolabels_mol_emb_${DATASET}_none"
elif [ "$EMBEDDING_LAYER" = "concat" ]; then
    DENSE_SUFFIX=$([ "$USE_FP_DENSE" = true ] && echo "dense" || echo "not_dense")
    PIPELINE_NAME="nn_retraining_with_pseudolabels_mol_emb_${DATASET}_${EMBEDDING_TYPE}_concat_${DENSE_SUFFIX}"
else
    PIPELINE_NAME="nn_retraining_with_pseudolabels_mol_emb_${DATASET}_${EMBEDDING_TYPE}_${EMBEDDING_LAYER}"
fi

ENV_DIR=./venvs/venvs/nn_retraining_with_pseudolabels
LOGS_DIR=./logs
QOS=gpu_normal
PARTITION=gpu_p

BASE="./data/benchmark"
DATA_BASE="${BASE}/resources/datasets/${DATA_DIR}"

DE_TRAIN="${DATA_BASE}/de_train.h5ad"
ID_MAP="${DATA_BASE}/id_map.csv"
LAYER="clipped_sign_log10_pval"
REPS=10
OUTPUT="${BASE}/results/methods/${PIPELINE_NAME}/predictions.h5ad"

if [ "$EMBEDDING_TYPE" = "none" ]; then
    EMBEDDINGS_FLAG=""
    EMBEDDING_LAYER_FLAG=""
    DENSE_FLAG=""
else
    EMBEDDINGS_FLAG="--embeddings ${DATA_BASE}/op3_emb.pkl"
    EMBEDDING_LAYER_FLAG="--embedding_layer ${EMBEDDING_LAYER}"
    DENSE_FLAG=$([ "$USE_FP_DENSE" = true ] && echo "--use_fp_dense" || echo "")
fi

# ============================================================================
# Setup
# ============================================================================

mkdir -p "$(dirname "${OUTPUT}")" "${LOGS_DIR}/methods/${PIPELINE_NAME}"

SBATCH_PREAMBLE="export PATH=\${HOME}/miniforge3/bin:\${PATH} && \
    export TMPDIR=\${HOME}/tmp && mkdir -p \${TMPDIR} && \
    cd ${PROJECT_ROOT} && \
    eval \"\$(mamba shell hook --shell bash)\" && \
    mamba activate ${ENV_DIR}"

# ============================================================================
# Submit job
# ============================================================================

echo "> Submitting ${PIPELINE_NAME} job"
echo "  dataset:        ${DATASET} (${DATA_DIR})"
echo "  de_train:       ${DE_TRAIN}"
echo "  id_map:         ${ID_MAP}"
echo "  embeddings:     ${EMBEDDINGS_FLAG}"
echo "  embedding_type:  ${EMBEDDING_TYPE}"
echo "  embedding_layer: ${EMBEDDING_LAYER}"
echo "  use_fp_dense:    ${USE_FP_DENSE}"
echo "  layer:          ${LAYER}"
echo "  reps:           ${REPS}"
echo "  output:         ${OUTPUT}"

sbatch -W \
    -J ${PIPELINE_NAME} \
    --partition=${PARTITION} \
    --qos=${QOS} \
    --mem=64G \
    --time=8:00:00 \
    --cpus-per-task=4 \
    --gpus=1 \
    --output="${LOGS_DIR}/methods/${PIPELINE_NAME}/${PIPELINE_NAME}.%j.out" \
    --error="${LOGS_DIR}/methods/${PIPELINE_NAME}/${PIPELINE_NAME}.%j.err" \
    --wrap="${SBATCH_PREAMBLE} && \
        export TF_USE_LEGACY_KERAS=1 && \
        python3 -m src.methods.nn_retraining_with_pseudolabels_mol_emb.script \
            --de_train ${DE_TRAIN} \
            --id_map ${ID_MAP} \
            ${EMBEDDINGS_FLAG} \
            --embedding_type ${EMBEDDING_TYPE} \
            ${EMBEDDING_LAYER_FLAG} \
            ${DENSE_FLAG} \
            --layer ${LAYER} \
            --reps ${REPS} \
            --output ${OUTPUT}"

echo "> ${PIPELINE_NAME} completed"
echo "> Output saved to: ${OUTPUT}"
