#!/bin/bash

set -e

# ============================================================================
# Metrics pipeline
# ============================================================================
#
# Computes mean rowwise error and correlation metrics on method predictions.
# Expects to be called from the project root via run_pipelines/run_metrics.sh.
#
# Options:
#   -m  Run metrics for a single method only (e.g. methods/my_method).
#       Omit to run for all discovered methods.
#   -h  Show this help message
# ============================================================================

# ============================================================================
# Parse options
# ============================================================================

METHOD_FILTER=""

while getopts ":m:h" opt; do
    case $opt in
        m) METHOD_FILTER=$OPTARG ;;
        h)
            echo "Usage: $0 [-m method_name]"
            echo ""
            echo "Optional:"
            echo "  -m  Run metrics for one method only."
            echo "      Provide the path relative to results/, e.g.:"
            echo "        methods/nn_retraining_with_pseudolabels_mol_emb_subsample_lpm_concat_dense"
            echo "      Omit to run for all discovered methods."
            echo "  -h  Show this help message"
            exit 0
            ;;
        \?) echo "Error: Invalid option -$OPTARG" >&2; exit 1 ;;
        :)  echo "Error: Option -$OPTARG requires an argument" >&2; exit 1 ;;
    esac
done

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
DE_TEST_SUBSAMPLE="${BASE}/resources/datasets/neurips-2023-data-subsample/de_test.h5ad"
DE_TEST_LAYER="clipped_sign_log10_pval"
PREDICTION_LAYER="prediction"
OUTPUT_DIR="${BASE}/results/metrics"

SCRIPT_ERROR="${PROJECT_ROOT}/src/metrics/mean_rowwise_error/script.R"
SCRIPT_CORR="${PROJECT_ROOT}/src/metrics/mean_rowwise_correlation/script.R"

# ============================================================================
# Setup
# ============================================================================

mkdir -p "${OUTPUT_DIR}" "${LOGS_DIR}/${PIPELINE_NAME}"

# Discover all methods that have predictions
# (results/methods/<name>/predictions.h5ad)
PRED_LIST=()
cat_dir="${RESULTS_DIR}/methods"
if [ -d "${cat_dir}" ]; then
    for d in "${cat_dir}"/*/; do
        [ -d "$d" ] || continue
        name=$(basename "$d")
        if [ -f "${d}/predictions.h5ad" ]; then
            PRED_LIST+=("methods/${name}")
        fi
    done
fi

if [ ${#PRED_LIST[@]} -eq 0 ]; then
    echo "> No prediction files found under ${RESULTS_DIR}/methods/. Nothing to do."
    exit 0
fi

# Apply method filter if -m was provided
if [ -n "${METHOD_FILTER}" ]; then
    FILTERED=()
    for entry in "${PRED_LIST[@]}"; do
        # Match on the last path component (method name) or the full category/name
        name=$(basename "$entry")
        if [[ "${entry}" == "${METHOD_FILTER}" || "${name}" == "${METHOD_FILTER}" ]]; then
            FILTERED+=("${entry}")
        fi
    done
    if [ ${#FILTERED[@]} -eq 0 ]; then
        echo "> Error: no predictions found matching '${METHOD_FILTER}'." >&2
        echo "  Available entries:" >&2
        for e in "${PRED_LIST[@]}"; do echo "    ${e}" >&2; done
        exit 1
    fi
    PRED_LIST=("${FILTERED[@]}")
    echo "> Filtered to method: ${METHOD_FILTER}"
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
echo "  de_test (full):      ${DE_TEST}"
echo "  de_test (subsample): ${DE_TEST_SUBSAMPLE}"
echo "  output_dir: ${OUTPUT_DIR}"

RUN_METRICS=""
for entry in "${PRED_LIST[@]}"; do
    pred="${RESULTS_DIR}/${entry}/predictions.h5ad"
    out_dir="${OUTPUT_DIR}/${entry}"
    # subsample methods must be evaluated against the subsampled test set
    if [[ "${entry}" == *subsample* ]]; then
        ENTRY_DE_TEST="${DE_TEST_SUBSAMPLE}"
    else
        ENTRY_DE_TEST="${DE_TEST}"
    fi
    RUN_METRICS="${RUN_METRICS} mkdir -p ${out_dir} && \
        Rscript ${SCRIPT_ERROR} \
            --de_test ${ENTRY_DE_TEST} \
            --de_test_layer ${DE_TEST_LAYER} \
            --prediction ${pred} \
            --prediction_layer ${PREDICTION_LAYER} \
            --output ${out_dir}/mean_rowwise_error.h5ad && \
        Rscript ${SCRIPT_CORR} \
            --de_test ${ENTRY_DE_TEST} \
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
echo "> Results saved under: ${OUTPUT_DIR}/methods/<name>/"
