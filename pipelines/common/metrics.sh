#!/bin/bash

set -e

# ============================================================================
# Metrics pipeline
# ============================================================================
#
# Computes mean rowwise error and correlation metrics on method predictions.
# Expects to be called from the project root via run_pipelines/run_metrics.sh.
#
# To change parameters, edit the Configuration section below.
# ============================================================================

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PIPELINE_NAME="metrics"
ENV_DIR=./venvs/venvs/metrics
LOGS_DIR=./logs
QOS=cpu_normal
PARTITION=cpu_p

BASE="./data/benchmark"
DE_TEST="${BASE}/resources/datasets/neurips-2023-data/de_test.h5ad"
DE_TEST_LAYER="clipped_sign_log10_pval"
PREDICTION="${BASE}/results/pyboost/predictions.h5ad"
PREDICTION_LAYER="prediction"
OUTPUT_DIR="${BASE}/results/metrics"

SCRIPT_ERROR="${PROJECT_ROOT}/src/metrics/mean_rowwise_error/script.R"
SCRIPT_CORR="${PROJECT_ROOT}/src/metrics/mean_rowwise_correlation/script.R"

# ============================================================================
# Setup
# ============================================================================

mkdir -p "${OUTPUT_DIR}" "${LOGS_DIR}/${PIPELINE_NAME}"

SBATCH_PREAMBLE="export PATH=\${HOME}/miniforge3/bin:\${PATH} && \
    export TMPDIR=\${HOME}/tmp && mkdir -p \${TMPDIR} && \
    cd ${PROJECT_ROOT} && \
    eval \"\$(mamba shell hook --shell bash)\" && \
    mamba activate ${ENV_DIR}"

# ============================================================================
# Submit job
# ============================================================================

echo "> Submitting metrics job"
echo "  de_test:    ${DE_TEST}"
echo "  prediction: ${PREDICTION}"
echo "  output_dir: ${OUTPUT_DIR}"

sbatch -W \
    -J ${PIPELINE_NAME} \
    --partition=${PARTITION} \
    --qos=${QOS} \
    --mem=16G \
    --time=1:00:00 \
    --cpus-per-task=2 \
    --output="${LOGS_DIR}/${PIPELINE_NAME}/${PIPELINE_NAME}.%j.out" \
    --error="${LOGS_DIR}/${PIPELINE_NAME}/${PIPELINE_NAME}.%j.err" \
    --wrap="${SBATCH_PREAMBLE} && \
        Rscript ${SCRIPT_ERROR} \
            --de_test ${DE_TEST} \
            --de_test_layer ${DE_TEST_LAYER} \
            --prediction ${PREDICTION} \
            --prediction_layer ${PREDICTION_LAYER} \
            --output ${OUTPUT_DIR}/mean_rowwise_error.h5ad && \
        Rscript ${SCRIPT_CORR} \
            --de_test ${DE_TEST} \
            --de_test_layer ${DE_TEST_LAYER} \
            --prediction ${PREDICTION} \
            --prediction_layer ${PREDICTION_LAYER} \
            --output ${OUTPUT_DIR}/mean_rowwise_correlation.h5ad"

echo "> Metrics computation completed"
echo "> Results saved to: ${OUTPUT_DIR}"
