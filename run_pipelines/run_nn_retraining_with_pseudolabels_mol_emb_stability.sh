#!/bin/bash

set -e

# ============================================================================
# Stability runner: NN retraining with pseudolabels — multiple seeds
# ============================================================================
#
# Submits one sbatch job per seed. Predictions are saved under:
#   data/benchmark/results/methods/<pipeline_name>/stability/predictions_seed_<n>.h5ad
#
# Options:
#   -e  Embedding type: lpm, fp, pca_logfc, pca_t, or none  (default: none)
#   -l  Embedding layer: concat, fixed, or trainable  (default: concat)
#   -D  Dataset: subsample or original  (default: subsample)
#   -P  PCA version subdirectory (e.g. v2_128); forces dataset=subsample and
#       DATA_DIR=neurips-2023-data-subsample-pca/<version>
#   -d  Use Dense(256) projection before concatenation  (flag, default: off)
#   -f  Embedding filename within the dataset directory  (default: op3_emb.pkl)
#   -b  Stability subfolder name  (default: stability)
#   -S  Comma-separated list of seeds to run  (default: 0,1,2,3,4)
#   -h  Show this help message
#
# Examples:
#   ./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh \
#       -e none -D subsample -S 0,1,2,3,4
#   ./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb_stability.sh \
#       -e fp -l concat -d -D original -S 0,1,2
# ============================================================================

PIPELINE="nn_retraining_with_pseudolabels_mol_emb"
PIPELINE_SCRIPT="./pipelines/${PIPELINE}/${PIPELINE}.sh"

EMBEDDING_TYPE="none"
EMBEDDING_LAYER="concat"
DATASET="subsample"
PCA_VERSION=""
PCA_FLAG=""
USE_FP_DENSE=false
DENSE_FLAG=""
EMBEDDING_FILE="op3_emb.pkl"
STABILITY_SUBFOLDER=""   # auto: stability_pca_<version> when -P is set, else stability
SEEDS="0,1,2,3,4,10,11,12,13,14"

LOGS_DIR=./logs

while getopts ":e:l:D:P:f:b:S:dh" opt; do
    case $opt in
        e) EMBEDDING_TYPE=$OPTARG ;;
        l) EMBEDDING_LAYER=$OPTARG ;;
        D) DATASET=$OPTARG ;;
        P) PCA_VERSION=$OPTARG; PCA_FLAG="-P ${OPTARG}" ;;
        f) EMBEDDING_FILE=$OPTARG ;;
        b) STABILITY_SUBFOLDER=$OPTARG ;;
        S) SEEDS=$OPTARG ;;
        d) USE_FP_DENSE=true; DENSE_FLAG="-d" ;;
        h)
            echo "Usage: $0 [-e embedding_type] [-l embedding_layer] [-D dataset] [-P pca_version] [-f embedding_file] [-b stability_subfolder] [-S seeds] [-d]"
            echo ""
            echo "Optional:"
            echo "  -e  Embedding type: lpm, fp, pca_logfc, pca_t, or none (default: none)"
            echo "  -l  Embedding layer: concat, fixed, or trainable (default: concat)"
            echo "  -D  Dataset: subsample or original (default: subsample)"
            echo "  -P  PCA version subdirectory (e.g. v2_128); forces dataset=subsample and"
            echo "      DATA_DIR=neurips-2023-data-subsample-pca/<version>"
            echo "  -f  Embedding filename within the dataset directory (default: op3_emb.pkl)"
            echo "  -b  Stability subfolder name (default: stability_pca_<version> when -P is set, else stability)"
            echo "  -S  Comma-separated seeds to run (default: 0,1,2,3,4)"
            echo "  -d  Project embeddings through Dense(256) before concatenation"
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

# Build the base pipeline name (mirrors logic in the pipeline script)
DATASET_LABEL="${DATASET}"   # PCA version goes into stability subfolder, not pipeline name

# Resolve stability subfolder: explicit -b wins; otherwise auto-derive from PCA_VERSION
if [ -z "${STABILITY_SUBFOLDER}" ]; then
    if [ -n "${PCA_VERSION}" ]; then
        STABILITY_SUBFOLDER="stability_pca_${PCA_VERSION}"
    else
        STABILITY_SUBFOLDER="stability"
    fi
fi

if [ "$EMBEDDING_TYPE" = "none" ]; then
    BASE_LABEL="${PIPELINE}_${DATASET_LABEL}_none"
elif [ "$EMBEDDING_LAYER" = "concat" ]; then
    DENSE_SUFFIX=$([ "$USE_FP_DENSE" = true ] && echo "dense" || echo "not_dense")
    BASE_LABEL="${PIPELINE}_${DATASET_LABEL}_${EMBEDDING_TYPE}_concat_${DENSE_SUFFIX}"
else
    BASE_LABEL="${PIPELINE}_${DATASET_LABEL}_${EMBEDDING_TYPE}_${EMBEDDING_LAYER}"
fi

LAYER_FLAG=$([ "$EMBEDDING_TYPE" != "none" ] && echo "-l ${EMBEDDING_LAYER}" || echo "")

echo "> Stability run for: ${BASE_LABEL}"
echo "  Seeds:               ${SEEDS}"
echo "  Embedding file:      ${EMBEDDING_FILE}"
echo "  Stability subfolder: ${STABILITY_SUBFOLDER}"

IFS=',' read -ra SEED_LIST <<< "$SEEDS"
for SEED in "${SEED_LIST[@]}"; do
    SEED=$(echo "$SEED" | tr -d ' ')
    RUN_LABEL="${BASE_LABEL}_seed_${SEED}"
    mkdir -p "${LOGS_DIR}/methods/${BASE_LABEL}"

    echo "> Launching seed=${SEED} → ${RUN_LABEL}"
    bash "${PIPELINE_SCRIPT}" \
        -e "${EMBEDDING_TYPE}" \
        ${LAYER_FLAG} \
        ${DENSE_FLAG} \
        -D "${DATASET}" \
        ${PCA_FLAG} \
        -f "${EMBEDDING_FILE}" \
        -b "${STABILITY_SUBFOLDER}" \
        -s "${SEED}" \
        > "${LOGS_DIR}/methods/${BASE_LABEL}/run_${RUN_LABEL}.PID$$.out" \
        2> "${LOGS_DIR}/methods/${BASE_LABEL}/run_${RUN_LABEL}.PID$$.err" &

    echo "  PID $! | logs: ${LOGS_DIR}/methods/${BASE_LABEL}/run_${RUN_LABEL}.PID$$.{out,err}"
done

echo ""
echo "> All seeds submitted. Predictions will be saved to:"
echo "  data/benchmark/results/methods/${BASE_LABEL}/${STABILITY_SUBFOLDER}/predictions_seed_<n>.h5ad"
