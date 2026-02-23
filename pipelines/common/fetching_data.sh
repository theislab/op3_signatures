#!/bin/bash

set -e

# ============================================================================
# Data fetching pipeline
# ============================================================================
#
# Downloads NeurIPS-2023 benchmark datasets from S3.
# Expects to be called from the project root via run_pipelines/run_fetching_data.sh.
#
# To change parameters, edit the Configuration section below.
# ============================================================================

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PIPELINE_NAME="fetching_data"
ENV_DIR=./venvs/venvs/fetching_data
LOGS_DIR=./logs
QOS=cpu_normal
PARTITION=cpu_p

BASE="./data/benchmark"
DATA_ROOT="${BASE}/resources/datasets/neurips-2023-data"

# ============================================================================
# Setup
# ============================================================================

mkdir -p "${DATA_ROOT}" "${LOGS_DIR}/${PIPELINE_NAME}"

SBATCH_PREAMBLE="export PATH=\${HOME}/miniforge3/bin:\${PATH} && \
    export TMPDIR=\${HOME}/tmp && mkdir -p \${TMPDIR} && \
    cd ${PROJECT_ROOT} && \
    eval \"\$(mamba shell hook --shell bash)\" && \
    mamba activate ${ENV_DIR}"

# ============================================================================
# Submit job
# ============================================================================

echo "> Submitting data fetching job"
echo "  data_root: ${DATA_ROOT}"

sbatch -W \
    -J ${PIPELINE_NAME} \
    --partition=${PARTITION} \
    --qos=${QOS} \
    --mem=16G \
    --time=2:00:00 \
    --cpus-per-task=2 \
    --output="${LOGS_DIR}/${PIPELINE_NAME}/${PIPELINE_NAME}.%j.out" \
    --error="${LOGS_DIR}/${PIPELINE_NAME}/${PIPELINE_NAME}.%j.err" \
    --wrap="${SBATCH_PREAMBLE} && \
        python3 -m src.data.run_downloading \
            --data_root ${DATA_ROOT}"

echo "> Data fetching completed"
echo "> Data saved to: ${DATA_ROOT}"
