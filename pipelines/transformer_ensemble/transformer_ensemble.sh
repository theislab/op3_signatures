#!/bin/bash

set -e

# ============================================================================
# Transformer ensemble prediction pipeline
# ============================================================================
#
# Runs a transformer ensemble model to predict perturbation signatures.
# Expects to be called from the project root via run_pipelines/run_methods.sh.
#
# To change parameters, edit the Configuration section below.
# ============================================================================

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PIPELINE_NAME="transformer_ensemble"
ENV_DIR=./venvs/venvs/transformer_ensemble
LOGS_DIR=./logs
QOS=gpu_normal
PARTITION=gpu_p

BASE="./data/benchmark"

DE_TRAIN="${BASE}/resources/datasets/neurips-2023-data/de_train.h5ad"
ID_MAP="${BASE}/resources/datasets/neurips-2023-data/id_map.csv"
LAYER="clipped_sign_log10_pval"
NUM_TRAIN_EPOCHS=20000
EARLY_STOPPING=5000
BATCH_SIZE=32
D_MODEL=128
OUTPUT="${BASE}/results/methods/transformer_ensemble/predictions.h5ad"

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

echo "> Submitting transformer ensemble prediction job"
echo "  de_train:          ${DE_TRAIN}"
echo "  id_map:            ${ID_MAP}"
echo "  layer:             ${LAYER}"
echo "  num_train_epochs:  ${NUM_TRAIN_EPOCHS}"
echo "  early_stopping:    ${EARLY_STOPPING}"
echo "  batch_size:        ${BATCH_SIZE}"
echo "  d_model:           ${D_MODEL}"
echo "  output:            ${OUTPUT}"

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
        python3 -m src.methods.transformer_ensemble.script \
            --de_train ${DE_TRAIN} \
            --id_map ${ID_MAP} \
            --layer ${LAYER} \
            --num_train_epochs ${NUM_TRAIN_EPOCHS} \
            --early_stopping ${EARLY_STOPPING} \
            --batch_size ${BATCH_SIZE} \
            --d_model ${D_MODEL} \
            --output ${OUTPUT}"

echo "> Transformer ensemble prediction completed"
echo "> Output saved to: ${OUTPUT}"
