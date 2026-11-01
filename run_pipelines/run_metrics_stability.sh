#!/bin/bash

set -e

RUN_TAG="${RUN_TAG:-}"
EXTRA_ARGS=()
while getopts ":m:b:r:h" opt; do
    case $opt in
        m) EXTRA_ARGS+=("-m" "$OPTARG") ;;
        b) EXTRA_ARGS+=("-b" "$OPTARG") ;;
        r) RUN_TAG=$OPTARG ;;
        h) ./pipelines/common/metrics_stability.sh -h; exit 0 ;;
        \?) echo "Error: Invalid option -$OPTARG" >&2; exit 1 ;;
        :)  echo "Error: Option -$OPTARG requires an argument" >&2; exit 1 ;;
    esac
done
if [ -n "${RUN_TAG}" ]; then
    EXTRA_ARGS+=("-r" "${RUN_TAG}")
fi

WRAPPER_LOGS_DIR="./logs${RUN_TAG:+_${RUN_TAG}}/metrics_stability"

mkdir -p "${WRAPPER_LOGS_DIR}"

echo "> Running stability metrics pipeline${RUN_TAG:+ (run tag: ${RUN_TAG})}"

./pipelines/common/metrics_stability.sh "${EXTRA_ARGS[@]}" \
    > "${WRAPPER_LOGS_DIR}/run_metrics_stability.PID$$.out" \
    2> "${WRAPPER_LOGS_DIR}/run_metrics_stability.PID$$.err" &
