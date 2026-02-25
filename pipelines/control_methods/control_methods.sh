#!/bin/bash

set -e

# ============================================================================
# Control methods pipeline
# ============================================================================
#
# Runs a single control / baseline method.
# Expects to be called from the project root via run_pipelines/run_control_methods.sh.
#
# Usage:  ./pipelines/control_methods/control_methods.sh -m METHOD
#
# Python methods (CPU): zeros, mean_across_celltypes, mean_across_compounds, mean_outcome
# R methods      (CPU): ground_truth, sample
# ============================================================================

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

METHOD=""
VALID_METHODS=(
    "zeros"
    "mean_across_celltypes"
    "mean_across_compounds"
    "mean_outcome"
    "ground_truth"
    "sample"
)
R_METHODS=("ground_truth" "sample")

while getopts ":m:h" opt; do
    case $opt in
        h)
            echo "Usage: $0 -m METHOD"
            echo "Methods: ${VALID_METHODS[*]}"
            exit 0 ;;
        m) METHOD=$OPTARG ;;
        \?) echo "Error: Invalid option -$OPTARG" >&2; exit 1 ;;
        :)  echo "Error: Option -$OPTARG requires an argument" >&2; exit 1 ;;
    esac
done

if [ -z "$METHOD" ]; then
    echo "Error: -m METHOD is required. Choose one of: ${VALID_METHODS[*]}" >&2; exit 1
fi
if [[ ! " ${VALID_METHODS[*]} " =~ " ${METHOD} " ]]; then
    echo "Error: Unknown method '${METHOD}'. Choose one of: ${VALID_METHODS[*]}" >&2; exit 1
fi

ENV_DIR=./venvs/venvs/control_methods
LOGS_DIR=./logs
PARTITION=cpu_p
QOS=cpu_normal

BASE="./data/benchmark"
DE_TRAIN="${BASE}/resources/datasets/neurips-2023-data/de_train.h5ad"
DE_TEST="${BASE}/resources/datasets/neurips-2023-data/de_test.h5ad"
ID_MAP="${BASE}/resources/datasets/neurips-2023-data/id_map.csv"
LAYER="clipped_sign_log10_pval"
OUTPUT="${BASE}/results/control_methods/${METHOD}/predictions.h5ad"

# ============================================================================
# Setup
# ============================================================================

mkdir -p "$(dirname "${OUTPUT}")" "${LOGS_DIR}/control_methods/${METHOD}"

SBATCH_PREAMBLE="export PATH=\${HOME}/miniforge3/bin:\${PATH} && \
    export TMPDIR=\${HOME}/tmp && mkdir -p \${TMPDIR} && \
    cd ${PROJECT_ROOT} && \
    eval \"\$(mamba shell hook --shell bash)\" && \
    mamba activate ${ENV_DIR}"

# ============================================================================
# Build command
# ============================================================================

IS_R=false
for r_method in "${R_METHODS[@]}"; do
    if [ "${METHOD}" = "${r_method}" ]; then IS_R=true; break; fi
done

if $IS_R; then
    SCRIPT="src/control_methods/${METHOD}/script.R"
    if [ "${METHOD}" = "ground_truth" ]; then
        CMD="export RETICULATE_PYTHON=\$(which python) && \
            Rscript ${SCRIPT} \
                --de_test ${DE_TEST} \
                --layer ${LAYER} \
                --output ${OUTPUT}"
    else
        CMD="export RETICULATE_PYTHON=\$(which python) && \
            Rscript ${SCRIPT} \
                --de_train ${DE_TRAIN} \
                --id_map ${ID_MAP} \
                --layer ${LAYER} \
                --output ${OUTPUT}"
    fi
else
    if [ "${METHOD}" = "zeros" ]; then
        CMD="python3 -m src.control_methods.${METHOD}.script \
                --de_train ${DE_TRAIN} \
                --id_map ${ID_MAP} \
                --output ${OUTPUT}"
    else
        CMD="python3 -m src.control_methods.${METHOD}.script \
                --de_train ${DE_TRAIN} \
                --id_map ${ID_MAP} \
                --layer ${LAYER} \
                --output ${OUTPUT}"
    fi
fi

# ============================================================================
# Submit job
# ============================================================================

echo "> Submitting control method: ${METHOD}"
echo "  output: ${OUTPUT}"

sbatch -W \
    -J "${METHOD}" \
    --partition=${PARTITION} \
    --qos=${QOS} \
    --mem=32G \
    --time=2:00:00 \
    --cpus-per-task=4 \
    --output="${LOGS_DIR}/control_methods/${METHOD}/${METHOD}.%j.out" \
    --error="${LOGS_DIR}/control_methods/${METHOD}/${METHOD}.%j.err" \
    --wrap="${SBATCH_PREAMBLE} && ${CMD}"

echo "> ${METHOD} completed"
echo "> Output saved to: ${OUTPUT}"
