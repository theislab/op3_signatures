#!/bin/bash

set -e

LOGS_DIR=./logs/metrics

# Forward any arguments to the metrics pipeline (e.g. -m method_name)
EXTRA_ARGS=()
while getopts ":m:h" opt; do
    case $opt in
        m) EXTRA_ARGS+=("-m" "$OPTARG") ;;
        h) ./pipelines/common/metrics.sh -h; exit 0 ;;
        \?) echo "Error: Invalid option -$OPTARG" >&2; exit 1 ;;
        :)  echo "Error: Option -$OPTARG requires an argument" >&2; exit 1 ;;
    esac
done

mkdir -p "${LOGS_DIR}"

echo "> Running metrics pipeline"

./pipelines/common/metrics.sh "${EXTRA_ARGS[@]}" \
    > "${LOGS_DIR}/run_metrics.PID$$.out" \
    2> "${LOGS_DIR}/run_metrics.PID$$.err" &
