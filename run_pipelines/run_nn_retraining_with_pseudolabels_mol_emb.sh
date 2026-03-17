#!/bin/bash

set -e

# ============================================================================
# Runner: NN retraining with pseudolabels (subsample)
# ============================================================================
#
# Convenience entry-point called from the project root.
# Validates options and forwards them to the unified pipeline script.
#
# Options:
#   -e  Embedding type: lpm, fp, or none  (default: none)
#   -l  Embedding layer: concat, fixed, or trainable  (default: concat)
#   -D  Dataset: subsample or original  (default: subsample)
#   -d  Use a Dense(256) projection before concatenation  (flag, default: off)
#   -h  Show this help message
#
# Examples:
#   ./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb.sh -e lpm -l concat -d -D subsample
#   ./run_pipelines/run_nn_retraining_with_pseudolabels_mol_emb.sh -e fp -l fixed -D original
# ============================================================================

PIPELINE="nn_retraining_with_pseudolabels_mol_emb"
PIPELINE_SCRIPT="./pipelines/${PIPELINE}/${PIPELINE}.sh"

EMBEDDING_TYPE="none"
EMBEDDING_LAYER="concat"
DATASET="subsample"
USE_FP_DENSE=false
DENSE_FLAG=""

LOGS_DIR=./logs

while getopts ":e:l:D:dh" opt; do
    case $opt in
        e) EMBEDDING_TYPE=$OPTARG ;;
        l) EMBEDDING_LAYER=$OPTARG ;;
        D) DATASET=$OPTARG ;;
        d) USE_FP_DENSE=true; DENSE_FLAG="-d" ;;
        h)
            echo "Usage: $0 [-e embedding_type] [-l embedding_layer] [-D dataset] [-d]"
            echo ""
            echo "Optional:"
            echo "  -e  Embedding type: lpm, fp, or none (default: none)"
            echo "  -l  Embedding layer: concat, fixed, or trainable (default: concat; only when -e is lpm or fp)"
            echo "  -D  Dataset: subsample or original (default: subsample)"
            echo "  -d  Project embeddings through Dense(256) before concatenation (only for -l concat)"
            echo "  -h  Show this help message"
            exit 0
            ;;
        \?) echo "Error: Invalid option -$OPTARG" >&2; exit 1 ;;
        :)  echo "Error: Option -$OPTARG requires an argument" >&2; exit 1 ;;
    esac
done

if [[ ! "$EMBEDDING_TYPE" =~ ^(lpm|fp|none)$ ]]; then
    echo "Error: embedding_type must be 'lpm', 'fp', or 'none', got '${EMBEDDING_TYPE}'" >&2; exit 1
fi

if [[ "$EMBEDDING_TYPE" != "none" ]] && [[ ! "$EMBEDDING_LAYER" =~ ^(concat|fixed|trainable)$ ]]; then
    echo "Error: embedding_layer must be 'concat', 'fixed', or 'trainable', got '${EMBEDDING_LAYER}'" >&2; exit 1
fi

if [[ ! "$DATASET" =~ ^(subsample|original)$ ]]; then
    echo "Error: dataset must be 'subsample' or 'original', got '${DATASET}'" >&2; exit 1
fi

if [ "$EMBEDDING_TYPE" = "none" ]; then
    RUN_LABEL="nn_retraining_with_pseudolabels_mol_emb_${DATASET}_none"
elif [ "$EMBEDDING_LAYER" = "concat" ]; then
    DENSE_SUFFIX=$([ "$USE_FP_DENSE" = true ] && echo "dense" || echo "not_dense")
    RUN_LABEL="nn_retraining_with_pseudolabels_mol_emb_${DATASET}_${EMBEDDING_TYPE}_concat_${DENSE_SUFFIX}"
else
    RUN_LABEL="nn_retraining_with_pseudolabels_mol_emb_${DATASET}_${EMBEDDING_TYPE}_${EMBEDDING_LAYER}"
fi

mkdir -p "${LOGS_DIR}/methods/${RUN_LABEL}"

echo "> Starting: ${RUN_LABEL}"

LAYER_FLAG=$([ "$EMBEDDING_TYPE" != "none" ] && echo "-l ${EMBEDDING_LAYER}" || echo "")

bash "${PIPELINE_SCRIPT}" -e "${EMBEDDING_TYPE}" ${LAYER_FLAG} ${DENSE_FLAG} -D "${DATASET}" \
    > "${LOGS_DIR}/methods/${RUN_LABEL}/run_${RUN_LABEL}.PID$$.out" \
    2> "${LOGS_DIR}/methods/${RUN_LABEL}/run_${RUN_LABEL}.PID$$.err" &

echo "> Launched in background (PID $!)"
echo "> Logs: ${LOGS_DIR}/methods/${RUN_LABEL}/"
