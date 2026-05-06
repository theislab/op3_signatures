#!/bin/bash

set -e

# ============================================================================
# NN retraining with pseudolabels (subsample) — configurable embedding pipeline
# ============================================================================
#
# Runs a neural network with pseudolabel retraining with optional external
# embeddings concatenated to the input.
# Expects to be called from the project root (directly or via run_pipelines/).
#
# Options:
#   -e  Embedding type: lpm, fp, pca_logfc, pca_t, or none  (default: none)
#   -l  Embedding layer: concat, fixed, or trainable  (default: concat)
#   -D  Dataset: subsample or original  (default: subsample)
#   -P  PCA version subdirectory (e.g. v2_128); forces dataset=subsample and
#       DATA_DIR=neurips-2023-data-subsample-pca/<version>
#   -d  Use a Dense(256) projection before concatenation  (flag, default: off)
#   -f  Embedding filename within the dataset directory, or absolute/relative
#       path containing a "/"  (default: op3_emb.pkl)
#   -T  Embedding tag appended to PIPELINE_NAME so multiple embedding versions
#       get their own results folder. If omitted and -f matches op3_emb_<X>.pkl,
#       the tag is auto-derived as "<X>" (e.g. op3_emb_5.pkl -> tag "5").
#   -b  Stability subfolder name  (default: stability)
#   -s  Random seed  (default: 42)
#   -h  Show this help message
#
# Examples:
#   ./pipelines/.../nn_retraining_...sh -e lpm -l concat -d -D subsample
#   ./pipelines/.../nn_retraining_...sh -e fp -l fixed -D original
# ============================================================================

# ============================================================================
# Parse options
# ============================================================================

EMBEDDING_TYPE="none"
EMBEDDING_LAYER="concat"
DATASET="subsample"
PCA_VERSION=""
USE_FP_DENSE=false
EMBEDDING_FILE="op3_emb.pkl"
EMBEDDING_TAG=""
STABILITY_SUBFOLDER=""   # auto: stability_pca_<version> when -P is set, else stability
SEED=42

while getopts ":e:l:D:P:f:T:b:s:dh" opt; do
    case $opt in
        e) EMBEDDING_TYPE=$OPTARG ;;
        l) EMBEDDING_LAYER=$OPTARG ;;
        D) DATASET=$OPTARG ;;
        P) PCA_VERSION=$OPTARG ;;
        f) EMBEDDING_FILE=$OPTARG ;;
        T) EMBEDDING_TAG=$OPTARG ;;
        b) STABILITY_SUBFOLDER=$OPTARG ;;
        s) SEED=$OPTARG ;;
        d) USE_FP_DENSE=true ;;
        h)
            echo "Usage: $0 [-e embedding_type] [-l embedding_layer] [-D dataset] [-P pca_version] [-f embedding_file] [-T embedding_tag] [-b stability_subfolder] [-s seed] [-d]"
            echo ""
            echo "Optional:"
            echo "  -e  Embedding type: lpm, fp, pca_logfc, pca_t, or none (default: none)"
            echo "  -l  Embedding layer: concat, fixed, or trainable (default: concat; only when -e is not none)"
            echo "  -D  Dataset: subsample or original (default: subsample)"
            echo "  -P  PCA version subdirectory (e.g. v2_128); forces dataset=subsample and"
            echo "      DATA_DIR=neurips-2023-data-subsample-pca/<version>"
            echo "  -f  Embedding filename within the dataset directory, or path containing a '/'"
            echo "      (default: op3_emb.pkl)"
            echo "  -T  Embedding tag appended to PIPELINE_NAME so each embedding version writes"
            echo "      to its own results folder. Auto-derived from filenames matching"
            echo "      op3_emb_<X>.pkl (e.g. op3_emb_5.pkl -> tag '5') when omitted."
            echo "  -b  Stability subfolder name (default: stability_pca_<version> when -P is set, else stability)"
            echo "  -s  Random seed (default: 42)"
            echo "  -d  Project embeddings through Dense(256) before concatenation (only for -l concat)"
            echo "  -h  Show this help message"
            exit 0
            ;;
        \?) echo "Error: Invalid option -$OPTARG" >&2; exit 1 ;;
        :)  echo "Error: Option -$OPTARG requires an argument" >&2; exit 1 ;;
    esac
done

if [[ ! "$EMBEDDING_TYPE" =~ ^(lpm|fp|pca_logfc|pca_t|none)$ ]]; then
    echo "Error: embedding_type must be 'lpm', 'fp', 'pca_logfc', 'pca_t', or 'none', got '${EMBEDDING_TYPE}'" >&2; exit 1
fi

if [[ "$EMBEDDING_TYPE" != "none" ]] && [[ ! "$EMBEDDING_LAYER" =~ ^(concat|fixed|trainable)$ ]]; then
    echo "Error: embedding_layer must be 'concat', 'fixed', or 'trainable', got '${EMBEDDING_LAYER}'" >&2; exit 1
fi

if [[ ! "$DATASET" =~ ^(subsample|original)$ ]]; then
    echo "Error: dataset must be 'subsample' or 'original', got '${DATASET}'" >&2; exit 1
fi

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -n "${PCA_VERSION}" ]; then
    DATA_DIR="neurips-2023-data-subsample-pca/${PCA_VERSION}"
    DATASET_LABEL="subsample"
elif [ "$DATASET" = "subsample" ]; then
    DATA_DIR="neurips-2023-data-subsample"
    DATASET_LABEL="subsample"
else
    DATA_DIR="neurips-2023-data"
    DATASET_LABEL="original"
fi

# Resolve stability subfolder: explicit -b wins; otherwise auto-derive from PCA_VERSION
if [ -z "${STABILITY_SUBFOLDER}" ]; then
    if [ -n "${PCA_VERSION}" ]; then
        STABILITY_SUBFOLDER="stability_pca_${PCA_VERSION}"
    else
        STABILITY_SUBFOLDER="stability"
    fi
fi

PIPELINE_PREFIX="nn_retraining_with_pseudolabels_mol_emb_learning_missed"
if [ "$EMBEDDING_TYPE" = "none" ]; then
    PIPELINE_NAME="${PIPELINE_PREFIX}_${DATASET_LABEL}_none"
elif [ "$EMBEDDING_LAYER" = "concat" ]; then
    DENSE_SUFFIX=$([ "$USE_FP_DENSE" = true ] && echo "dense" || echo "not_dense")
    PIPELINE_NAME="${PIPELINE_PREFIX}_${DATASET_LABEL}_${EMBEDDING_TYPE}_concat_${DENSE_SUFFIX}"
else
    PIPELINE_NAME="${PIPELINE_PREFIX}_${DATASET_LABEL}_${EMBEDDING_TYPE}_${EMBEDDING_LAYER}"
fi

# Auto-derive tag from embedding filename if -T not given.
# Strips dataset directory (if any) and matches op3_emb_<X>.pkl.
# Doesn't fire for embedding_type=none (no embedding to version) or for the
# canonical op3_emb.pkl (back-compat: keeps the un-tagged folder name).
if [ -z "${EMBEDDING_TAG}" ] && [ "$EMBEDDING_TYPE" != "none" ]; then
    _emb_basename="$(basename "${EMBEDDING_FILE}")"
    if [[ "${_emb_basename}" =~ ^op3_emb_(.+)\.pkl$ ]]; then
        EMBEDDING_TAG="${BASH_REMATCH[1]}"
    fi
fi
if [ -n "${EMBEDDING_TAG}" ] && [ "$EMBEDDING_TYPE" != "none" ]; then
    PIPELINE_NAME="${PIPELINE_NAME}_${EMBEDDING_TAG}"
fi

ENV_DIR=./venvs/venvs/nn_retraining_with_pseudolabels
LOGS_DIR=./logs

BASE="./data/benchmark"
DATA_BASE="${BASE}/resources/datasets/${DATA_DIR}"

source "${PROJECT_ROOT}/pipelines/common/sbatch_env.sh"
build_sbatch_args

DE_TRAIN="${DATA_BASE}/de_train.h5ad"
ID_MAP="${DATA_BASE}/id_map.csv"
LAYER="clipped_sign_log10_pval"
REPS=10
if [ "${SEED}" -eq 42 ]; then
    OUTPUT="${BASE}/results/methods/${PIPELINE_NAME}/predictions.h5ad"
else
    OUTPUT="${BASE}/results/methods/${PIPELINE_NAME}/${STABILITY_SUBFOLDER}/predictions_seed_${SEED}.h5ad"
fi

if [ "$EMBEDDING_TYPE" = "none" ]; then
    EMBEDDINGS_FLAG=""
    EMBEDDING_LAYER_FLAG=""
    DENSE_FLAG=""
else
    # If -f contains a "/", treat it as an absolute or relative path; otherwise
    # assume it's a filename inside the dataset directory.
    if [[ "${EMBEDDING_FILE}" == */* ]]; then
        EMBEDDING_PATH="${EMBEDDING_FILE}"
    else
        EMBEDDING_PATH="${DATA_BASE}/${EMBEDDING_FILE}"
    fi
    EMBEDDINGS_FLAG="--embeddings ${EMBEDDING_PATH}"
    EMBEDDING_LAYER_FLAG="--embedding_layer ${EMBEDDING_LAYER}"
    DENSE_FLAG=$([ "$USE_FP_DENSE" = true ] && echo "--use_fp_dense" || echo "")
fi

if [[ -f "${OUTPUT}" ]]; then
    echo "File exists, skipping the pipeline"
    exit 0
fi
# ============================================================================
# Setup
# ============================================================================

mkdir -p "$(dirname "${OUTPUT}")" "${LOGS_DIR}/methods/${PIPELINE_NAME}"

build_sbatch_preamble "${ENV_DIR}"

# ============================================================================
# Submit job
# ============================================================================

echo "> Submitting ${PIPELINE_NAME} job"
echo "  dataset:            ${DATASET_LABEL} (${DATA_DIR})"
echo "  de_train:           ${DE_TRAIN}"
echo "  id_map:             ${ID_MAP}"
echo "  embeddings:         ${EMBEDDINGS_FLAG}"
echo "  embedding_type:     ${EMBEDDING_TYPE}"
echo "  embedding_layer:    ${EMBEDDING_LAYER}"
echo "  embedding_tag:      ${EMBEDDING_TAG:-<none>}"
echo "  use_fp_dense:       ${USE_FP_DENSE}"
echo "  stability_subfolder: ${STABILITY_SUBFOLDER}"
echo "  seed:               ${SEED}"
echo "  layer:              ${LAYER}"
echo "  reps:               ${REPS}"
echo "  output:             ${OUTPUT}"

sbatch -W \
    -J "${PIPELINE_NAME}" \
    "${SBATCH_ARGS[@]}" \
    --mem=64G \
    --time=8:00:00 \
    --cpus-per-task=4 \
    --gpus=1 \
    --output="${LOGS_DIR}/methods/${PIPELINE_NAME}/${PIPELINE_NAME}.%j.out" \
    --error="${LOGS_DIR}/methods/${PIPELINE_NAME}/${PIPELINE_NAME}.%j.err" \
    --wrap="${SBATCH_PREAMBLE} && \
        export TF_USE_LEGACY_KERAS=1 && \
        python3 -m src.methods.nn_retraining_with_pseudolabels_mol_emb_learning_missed.script \
            --de_train ${DE_TRAIN} \
            --id_map ${ID_MAP} \
            ${EMBEDDINGS_FLAG} \
            --embedding_type ${EMBEDDING_TYPE} \
            ${EMBEDDING_LAYER_FLAG} \
            ${DENSE_FLAG} \
            --layer ${LAYER} \
            --reps ${REPS} \
            --seed ${SEED} \
            --output ${OUTPUT}"

echo "> ${PIPELINE_NAME} completed"
echo "> Output saved to: ${OUTPUT}"
