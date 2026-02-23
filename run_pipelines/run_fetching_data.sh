#!/bin/bash

set -e

LOGS_DIR=./logs/fetching_data

mkdir -p "${LOGS_DIR}"

echo "> Running data fetching pipeline"

./pipelines/common/fetching_data.sh \
    > "${LOGS_DIR}/run_fetching_data.PID$$.out" \
    2> "${LOGS_DIR}/run_fetching_data.PID$$.err" &
