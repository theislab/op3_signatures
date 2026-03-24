#!/bin/bash

set -e

LOGS_DIR=./logs/metrics_stability

EXTRA_ARGS=()
while getopts ":m:b:h" opt; do
    case $opt in
        m) EXTRA_ARGS+=("-m" "$OPTARG") ;;
        b) EXTRA_ARGS+=("-b" "$OPTARG") ;;
        h) ./pipelines/common/metrics_stability.sh -h; exit 0 ;;
        \?) echo "Error: Invalid option -$OPTARG" >&2; exit 1 ;;
        :)  echo "Error: Option -$OPTARG requires an argument" >&2; exit 1 ;;
    esac
done

mkdir -p "${LOGS_DIR}"

echo "> Running stability metrics pipeline"

./pipelines/common/metrics_stability.sh "${EXTRA_ARGS[@]}" \
    > "${LOGS_DIR}/run_metrics_stability.PID$$.out" \
    2> "${LOGS_DIR}/run_metrics_stability.PID$$.err" &
