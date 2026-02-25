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
RESULTS_DIR="${BASE}/results"
DE_TEST="${BASE}/resources/datasets/neurips-2023-data/de_test.h5ad"
DE_TEST_LAYER="clipped_sign_log10_pval"
PREDICTION_LAYER="prediction"
OUTPUT_DIR="${BASE}/results/metrics"

SCRIPT_ERROR="${PROJECT_ROOT}/src/metrics/mean_rowwise_error/script.R"
SCRIPT_CORR="${PROJECT_ROOT}/src/metrics/mean_rowwise_correlation/script.R"

# ============================================================================
# Setup
# ============================================================================

mkdir -p "${OUTPUT_DIR}" "${LOGS_DIR}/${PIPELINE_NAME}"

# Discover all methods and control_methods that have predictions
# (results/methods/<name>/predictions.h5ad and results/control_methods/<name>/predictions.h5ad)
PRED_LIST=()
for category in methods control_methods; do
    cat_dir="${RESULTS_DIR}/${category}"
    if [ ! -d "${cat_dir}" ]; then
        continue
    fi
    for d in "${cat_dir}"/*/; do
        [ -d "$d" ] || continue
        name=$(basename "$d")
        if [ -f "${d}/predictions.h5ad" ]; then
            PRED_LIST+=("${category}/${name}")
        fi
    done
done

if [ ${#PRED_LIST[@]} -eq 0 ]; then
    echo "> No prediction files found under ${RESULTS_DIR}/methods/ or ${RESULTS_DIR}/control_methods/. Nothing to do."
    exit 0
fi

SBATCH_PREAMBLE="export PATH=\${HOME}/miniforge3/bin:\${PATH} && \
    export TMPDIR=\${HOME}/tmp && mkdir -p \${TMPDIR} && \
    cd ${PROJECT_ROOT} && \
    eval \"\$(mamba shell hook --shell bash)\" && \
    mamba activate ${ENV_DIR}"

# ============================================================================
# Submit job: run metrics for each method
# ============================================================================

echo "> Submitting metrics job for ${#PRED_LIST[@]} result(s): ${PRED_LIST[*]}"
echo "  de_test:    ${DE_TEST}"
echo "  output_dir: ${OUTPUT_DIR}"

RUN_METRICS=""
for entry in "${PRED_LIST[@]}"; do
    pred="${RESULTS_DIR}/${entry}/predictions.h5ad"
    out_dir="${OUTPUT_DIR}/${entry}"
    RUN_METRICS="${RUN_METRICS} mkdir -p ${out_dir} && \
        Rscript ${SCRIPT_ERROR} \
            --de_test ${DE_TEST} \
            --de_test_layer ${DE_TEST_LAYER} \
            --prediction ${pred} \
            --prediction_layer ${PREDICTION_LAYER} \
            --output ${out_dir}/mean_rowwise_error.h5ad && \
        Rscript ${SCRIPT_CORR} \
            --de_test ${DE_TEST} \
            --de_test_layer ${DE_TEST_LAYER} \
            --prediction ${pred} \
            --prediction_layer ${PREDICTION_LAYER} \
            --output ${out_dir}/mean_rowwise_correlation.h5ad && "
done
# Remove trailing " && "
RUN_METRICS="${RUN_METRICS% && }"

sbatch -W \
    -J ${PIPELINE_NAME} \
    --partition=${PARTITION} \
    --qos=${QOS} \
    --mem=16G \
    --time=2:00:00 \
    --cpus-per-task=2 \
    --output="${LOGS_DIR}/${PIPELINE_NAME}/${PIPELINE_NAME}.%j.out" \
    --error="${LOGS_DIR}/${PIPELINE_NAME}/${PIPELINE_NAME}.%j.err" \
    --wrap="${SBATCH_PREAMBLE} && ${RUN_METRICS}"

echo "> Metrics computation completed"
echo "> Results saved under: ${OUTPUT_DIR}/methods/<name>/ and ${OUTPUT_DIR}/control_methods/<name>/"
