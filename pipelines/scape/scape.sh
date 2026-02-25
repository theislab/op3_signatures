#!/bin/bash

set -e

# ============================================================================
# SCAPE prediction pipeline
# ============================================================================
#
# Runs the SCAPE model to predict perturbation signatures.
# Expects to be called from the project root via run_pipelines/run_methods.sh.
#
# To change parameters, edit the Configuration section below.
# ============================================================================

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PIPELINE_NAME="scape"
ENV_DIR=./venvs/venvs/scape
LOGS_DIR=./logs
QOS=gpu_normal
PARTITION=gpu_p

BASE="./data/benchmark"

DE_TRAIN="${BASE}/resources/datasets/neurips-2023-data/de_train.h5ad"
ID_MAP="${BASE}/resources/datasets/neurips-2023-data/id_map.csv"
LAYER="clipped_sign_log10_pval"
CELL="NK cells"
EPOCHS=300
EPOCHS_ENHANCED=800
N_GENES=64
N_GENES_ENHANCED=256
MIN_N_TOP_DRUGS=50
OUTPUT="${BASE}/results/methods/scape/predictions.h5ad"

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

echo "> Submitting SCAPE prediction job"
echo "  de_train:         ${DE_TRAIN}"
echo "  id_map:           ${ID_MAP}"
echo "  layer:            ${LAYER}"
echo "  cell:             ${CELL}"
echo "  epochs:           ${EPOCHS}"
echo "  epochs_enhanced:  ${EPOCHS_ENHANCED}"
echo "  n_genes:          ${N_GENES}"
echo "  n_genes_enhanced: ${N_GENES_ENHANCED}"
echo "  min_n_top_drugs:  ${MIN_N_TOP_DRUGS}"
echo "  output:           ${OUTPUT}"

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
        python3 -m src.methods.scape.script \
            --de_train ${DE_TRAIN} \
            --id_map ${ID_MAP} \
            --layer ${LAYER} \
            --cell \"${CELL}\" \
            --epochs ${EPOCHS} \
            --epochs_enhanced ${EPOCHS_ENHANCED} \
            --n_genes ${N_GENES} \
            --n_genes_enhanced ${N_GENES_ENHANCED} \
            --min_n_top_drugs ${MIN_N_TOP_DRUGS} \
            --output ${OUTPUT}"

echo "> SCAPE prediction completed"
echo "> Output saved to: ${OUTPUT}"
