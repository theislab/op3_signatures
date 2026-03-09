#!/bin/bash

set -e

# ============================================================================
# NN retraining with pseudolabels prediction pipeline
# ============================================================================
#
# Runs a neural network with pseudolabel retraining to predict perturbation signatures.
# Expects to be called from the project root via run_pipelines/run_methods.sh.
#
# To change parameters, edit the Configuration section below.
# ============================================================================

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PIPELINE_NAME="nn_retraining_with_pseudolabels_embeddings"
ENV_DIR=./venvs/venvs/nn_retraining_with_pseudolabels
LOGS_DIR=./logs
QOS=gpu_normal
PARTITION=gpu_p

BASE="./data/benchmark"

DE_TRAIN="${BASE}/resources/datasets/neurips-2023-data/de_train.h5ad"
ID_MAP="${BASE}/resources/datasets/neurips-2023-data/id_map.csv"
LAYER="clipped_sign_log10_pval"
REPS=10
OUTPUT="${BASE}/results/methods/nn_retraining_with_pseudolabels_embeddings_dense/predictions.h5ad"

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

echo "> Submitting NN retraining with pseudolabels prediction job"
echo "  de_train: ${DE_TRAIN}"
echo "  id_map:   ${ID_MAP}"
echo "  layer:    ${LAYER}"
echo "  reps:     ${REPS}"
echo "  output:   ${OUTPUT}"

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
        python3 -m src.methods.nn_retraining_with_pseudolabels_embeddings.script \
            --de_train ${DE_TRAIN} \
            --id_map ${ID_MAP} \
            --layer ${LAYER} \
            --reps ${REPS} \
            --use_fp_dense \
            --output ${OUTPUT}"

echo "> NN retraining with pseudolabels prediction completed"
echo "> Output saved to: ${OUTPUT}"
