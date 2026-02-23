#!/bin/bash

set -e

LOGS_DIR=./logs/metrics

mkdir -p "${LOGS_DIR}"

echo "> Running metrics pipeline"

./pipelines/common/metrics.sh \
    > "${LOGS_DIR}/run_metrics.PID$$.out" \
    2> "${LOGS_DIR}/run_metrics.PID$$.err" &
