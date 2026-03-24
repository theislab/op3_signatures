#!/bin/bash

set -e

# ============================================================================
# Stability metrics pipeline
# ============================================================================
#
# Computes mean rowwise error and correlation metrics on all seed-based
# prediction files produced by the stability runner.
#
# Predictions are expected at:
#   results/methods/<method>/<stability_subfolder>/predictions_seed_<n>.h5ad
#
# Metrics are saved to:
#   results/metrics/methods/<method>/<stability_subfolder>/seed_<n>/
#
# Options:
#   -m  Restrict to one method name (matches against the method dir name).
#       Omit to process all methods that have the given stability subfolder.
#   -b  Stability subfolder name to read predictions from and write metrics to
#       (default: stability)
#   -h  Show this help message
# ============================================================================

# ============================================================================
# Parse options
# ============================================================================

METHOD_FILTER=""
STABILITY_SUBFOLDER="stability"

while getopts ":m:b:h" opt; do
    case $opt in
        m) METHOD_FILTER=$OPTARG ;;
        b) STABILITY_SUBFOLDER=$OPTARG ;;
        h)
            echo "Usage: $0 [-m method_name] [-b stability_subfolder]"
            echo ""
            echo "Optional:"
            echo "  -m  Restrict to one method by name, e.g.:"
            echo "        nn_retraining_with_pseudolabels_mol_emb_subsample_none"
            echo "      Omit to process all methods with stability predictions."
            echo "  -b  Stability subfolder name to read predictions from and write metrics to"
            echo "      (default: stability)"
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

PIPELINE_NAME="metrics_stability"
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
# Discover stability prediction files
# ============================================================================
# Each entry in PRED_LIST is "method_name::seed_file_path"

PRED_LIST=()
methods_dir="${RESULTS_DIR}/methods"

if [ ! -d "${methods_dir}" ]; then
    echo "> No methods directory found at ${methods_dir}. Nothing to do." >&2
    exit 0
fi

for method_dir in "${methods_dir}"/*/; do
    [ -d "$method_dir" ] || continue
    method_name=$(basename "$method_dir")

    # Apply method filter if provided
    if [ -n "${METHOD_FILTER}" ] && [[ "${method_name}" != "${METHOD_FILTER}" ]]; then
        continue
    fi

    stability_dir="${method_dir}${STABILITY_SUBFOLDER}"
    [ -d "${stability_dir}" ] || continue

    for pred_file in "${stability_dir}"/predictions_seed_*.h5ad; do
        [ -f "${pred_file}" ] || continue
        PRED_LIST+=("${method_name}::${pred_file}")
    done
done

if [ ${#PRED_LIST[@]} -eq 0 ]; then
    echo "> No stability prediction files found under ${methods_dir}/*/${STABILITY_SUBFOLDER}/."
    if [ -n "${METHOD_FILTER}" ]; then
        echo "  (filter was: '${METHOD_FILTER}')"
    fi
    exit 0
fi

mkdir -p "${LOGS_DIR}/${PIPELINE_NAME}"

SBATCH_PREAMBLE="export PATH=\${HOME}/miniforge3/bin:\${PATH} && \
    export TMPDIR=\${HOME}/tmp && mkdir -p \${TMPDIR} && \
    cd ${PROJECT_ROOT} && \
    eval \"\$(mamba shell hook --shell bash)\" && \
    mamba activate ${ENV_DIR}"

# ============================================================================
# Submit job: run metrics for each seed file
# ============================================================================

echo "> Submitting stability metrics job for ${#PRED_LIST[@]} prediction file(s)"
echo "  de_test (full):      ${DE_TEST}"
echo "  de_test (subsample): ${DE_TEST_SUBSAMPLE}"
echo "  de_test (pca):       ${BASE}/resources/datasets/neurips-2023-data-subsample-pca/<version>/de_test.h5ad"
echo "  stability subfolder: ${STABILITY_SUBFOLDER}"
echo "  output_dir: ${OUTPUT_DIR}/methods/<method>/${STABILITY_SUBFOLDER}/seed_<n>/"

RUN_METRICS=""
for entry in "${PRED_LIST[@]}"; do
    method_name="${entry%%::*}"
    pred_file="${entry##*::}"

    # Extract seed number from filename (predictions_seed_<n>.h5ad)
    seed_tag=$(basename "${pred_file}" .h5ad)   # predictions_seed_<n>
    seed_tag="${seed_tag#predictions_}"          # seed_<n>

    out_dir="${OUTPUT_DIR}/methods/${method_name}/${STABILITY_SUBFOLDER}/${seed_tag}"

    # Choose the correct de_test:
    #   stability_pca_<version>  → neurips-2023-data-subsample-pca/<version>/de_test.h5ad
    #   *subsample*              → neurips-2023-data-subsample/de_test.h5ad
    #   otherwise                → neurips-2023-data/de_test.h5ad
    pred_subfolder=$(basename "$(dirname "${pred_file}")")
    if [[ "${pred_subfolder}" == stability_pca_* ]]; then
        pca_version="${pred_subfolder#stability_pca_}"
        ENTRY_DE_TEST="${BASE}/resources/datasets/neurips-2023-data-subsample-pca/${pca_version}/de_test.h5ad"
    elif [[ "${method_name}" == *subsample* ]]; then
        ENTRY_DE_TEST="${DE_TEST_SUBSAMPLE}"
    else
        ENTRY_DE_TEST="${DE_TEST}"
    fi

    RUN_METRICS="${RUN_METRICS} mkdir -p ${out_dir} && \
        Rscript ${SCRIPT_ERROR} \
            --de_test ${ENTRY_DE_TEST} \
            --de_test_layer ${DE_TEST_LAYER} \
            --prediction ${pred_file} \
            --prediction_layer ${PREDICTION_LAYER} \
            --output ${out_dir}/mean_rowwise_error.h5ad && \
        Rscript ${SCRIPT_CORR} \
            --de_test ${ENTRY_DE_TEST} \
            --de_test_layer ${DE_TEST_LAYER} \
            --prediction ${pred_file} \
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

echo "> Stability metrics computation completed"
echo "> Results saved under: ${OUTPUT_DIR}/methods/<method>/${STABILITY_SUBFOLDER}/seed_<n>/"
