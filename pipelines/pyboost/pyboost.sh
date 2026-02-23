#!/bin/bash

set -e

# ============================================================================
# Py-boost prediction pipeline
# ============================================================================
#
# Runs the py-boost model to predict t-scores from de_train.h5ad.
# Expects to be called from the project root via run_pipelines/run_methods.sh.
#
# To change parameters, edit the Configuration section below.
# ============================================================================

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PIPELINE_NAME="pyboost"
ENV_DIR=./venvs/venvs/pyboost
LOGS_DIR=./logs
QOS=gpu_normal
PARTITION=gpu_p

BASE="./data/benchmark"

DE_TRAIN="${BASE}/resources/datasets/neurips-2023-data/de_train.h5ad"
ID_MAP="${BASE}/resources/datasets/neurips-2023-data/id_map.csv"
LAYER="clipped_sign_log10_pval"
PREDICTOR_NAMES="py_boost"
OUTPUT="${BASE}/results/pyboost/predictions.h5ad"

# ============================================================================
# Setup
# ============================================================================

mkdir -p "$(dirname "${BASE}")" "$(dirname "${OUTPUT}")" "${LOGS_DIR}/${PIPELINE_NAME}"

SBATCH_PREAMBLE="export PATH=\${HOME}/miniforge3/bin:\${PATH} && \
    export TMPDIR=\${HOME}/tmp && mkdir -p \${TMPDIR} && \
    cd ${PROJECT_ROOT} && \
    eval \"\$(mamba shell hook --shell bash)\" && \
    mamba activate ${ENV_DIR}"

# ============================================================================
# Submit job
# ============================================================================

echo "> Submitting py-boost prediction job"
echo "  de_train:        ${DE_TRAIN}"
echo "  id_map:          ${ID_MAP}"
echo "  layer:           ${LAYER}"
echo "  predictor_names: ${PREDICTOR_NAMES}"
echo "  output:          ${OUTPUT}"

sbatch -W \
    -J ${PIPELINE_NAME} \
    --partition=${PARTITION} \
    --qos=${QOS} \
    --mem=64G \
    --time=4:00:00 \
    --cpus-per-task=4 \
    --gpus=1 \
    --output="${LOGS_DIR}/${PIPELINE_NAME}/${PIPELINE_NAME}.%j.out" \
    --error="${LOGS_DIR}/${PIPELINE_NAME}/${PIPELINE_NAME}.%j.err" \
    --wrap="${SBATCH_PREAMBLE} && \
        python3 -m src.methods.pyboost.script \
            --de_train ${DE_TRAIN} \
            --id_map ${ID_MAP} \
            --layer ${LAYER} \
            --predictor_names ${PREDICTOR_NAMES} \
            --output ${OUTPUT}"

echo "> Py-boost prediction completed"
echo "> Output saved to: ${OUTPUT}"
