#!/bin/bash

set -e

LOGS_DIR=./logs_learning_missed/metrics_stability

EXTRA_ARGS=()
while getopts ":m:b:h" opt; do
    case $opt in
        m) EXTRA_ARGS+=("-m" "$OPTARG") ;;
        b) EXTRA_ARGS+=("-b" "$OPTARG") ;;
        h) ./pipelines/common/metrics_stability_learning_missed.sh -h; exit 0 ;;
        \?) echo "Error: Invalid option -$OPTARG" >&2; exit 1 ;;
        :)  echo "Error: Option -$OPTARG requires an argument" >&2; exit 1 ;;
    esac
done

mkdir -p "${LOGS_DIR}"

echo "> Running stability metrics pipeline (learning_missed)"

./pipelines/common/metrics_stability_learning_missed.sh "${EXTRA_ARGS[@]}" \
    > "${LOGS_DIR}/run_metrics_stability_learning_missed.PID$$.out" \
    2> "${LOGS_DIR}/run_metrics_stability_learning_missed.PID$$.err" &
