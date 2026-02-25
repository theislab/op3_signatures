#!/bin/bash

set -e

# ============================================================================
# LGC Ensemble prediction pipeline (direct / single-job)
# ============================================================================
#
# Runs prepare → train → predict in a single job.
# Expects to be called from the project root via run_pipelines/run_methods.sh.
#
# To change parameters, edit the Configuration section below.
# ============================================================================

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PIPELINE_NAME="lgc_ensemble_direct"
ENV_DIR=./venvs/venvs/lgc_ensemble
LOGS_DIR=./logs
QOS=gpu_normal
PARTITION=gpu_p

BASE="./data/benchmark"

DE_TRAIN="${BASE}/resources/datasets/neurips-2023-data/de_train.h5ad"
ID_MAP="${BASE}/resources/datasets/neurips-2023-data/id_map.csv"
LAYER="clipped_sign_log10_pval"
SCHEMES="initial light heavy"
MODELS="LSTM GRU Conv"
EPOCHS=250
KF_N_SPLITS=5
OUTPUT="${BASE}/results/methods/lgc_ensemble_direct/predictions.h5ad"
OUTPUT_MODEL="${BASE}/results/models/lgc_ensemble_direct"

# ============================================================================
# Setup
# ============================================================================

mkdir -p "$(dirname "${OUTPUT}")" "${OUTPUT_MODEL}" "${LOGS_DIR}/methods/${PIPELINE_NAME}"

SBATCH_PREAMBLE="export PATH=\${HOME}/miniforge3/bin:\${PATH} && \
    export TMPDIR=\${HOME}/tmp && mkdir -p \${TMPDIR} && \
    cd ${PROJECT_ROOT} && \
    eval \"\$(mamba shell hook --shell bash)\" && \
    mamba activate ${ENV_DIR}"

# ============================================================================
# Submit job
# ============================================================================

echo "> Submitting LGC ensemble (direct) prediction job"
echo "  de_train:    ${DE_TRAIN}"
echo "  id_map:      ${ID_MAP}"
echo "  layer:       ${LAYER}"
echo "  schemes:     ${SCHEMES}"
echo "  models:      ${MODELS}"
echo "  epochs:       ${EPOCHS}"
echo "  kf_n_splits:  ${KF_N_SPLITS}"
echo "  output:       ${OUTPUT}"
echo "  output_model: ${OUTPUT_MODEL}"

sbatch -W \
    -J ${PIPELINE_NAME} \
    --partition=${PARTITION} \
    --qos=${QOS} \
    --mem=64G \
    --time=24:00:00 \
    --cpus-per-task=8 \
    --gpus=1 \
    --output="${LOGS_DIR}/methods/${PIPELINE_NAME}/${PIPELINE_NAME}.%j.out" \
    --error="${LOGS_DIR}/methods/${PIPELINE_NAME}/${PIPELINE_NAME}.%j.err" \
    --wrap="${SBATCH_PREAMBLE} && \
        python3 -m src.methods.lgc_ensemble_direct.script \
            --de_train ${DE_TRAIN} \
            --id_map ${ID_MAP} \
            --layer ${LAYER} \
            --schemes ${SCHEMES} \
            --models ${MODELS} \
            --epochs ${EPOCHS} \
            --kf_n_splits ${KF_N_SPLITS} \
            --output ${OUTPUT} \
            --output_model ${OUTPUT_MODEL}"

echo "> LGC ensemble (direct) prediction completed"
echo "> Output saved to: ${OUTPUT}"
echo "> Models saved to: ${OUTPUT_MODEL}"
