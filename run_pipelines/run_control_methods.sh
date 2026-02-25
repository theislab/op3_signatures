#!/bin/bash

set -e

LOGS_DIR=./logs
METHOD=""

VALID_METHODS=(
    "zeros"
    "mean_across_celltypes"
    "mean_across_compounds"
    "mean_outcome"
    "ground_truth"
    "sample"
)

RUN_ALL=false
while getopts ":m:ah" opt; do
    case $opt in
        h)
            echo "Run: $0 [-m METHOD] | -a"
            echo ""
            echo "Options:"
            echo "  -m  Run a single control method. One of: ${VALID_METHODS[*]}"
            echo "  -a  Run all control methods (submits one job per method)"
            echo "  -h  Show this help message"
            echo ""
            echo "If neither -m nor -a is given, all methods are run."
            exit 0
            ;;
        m) METHOD=$OPTARG ;;
        a) RUN_ALL=true ;;
        \?)
            echo "Error: Invalid option -$OPTARG" >&2; exit 1 ;;
        :)
            echo "Error: Option -$OPTARG requires an argument" >&2; exit 1 ;;
    esac
done

if [ -z "$METHOD" ] && [ "$RUN_ALL" != "true" ]; then
    RUN_ALL=true
fi

if [ -n "$METHOD" ]; then
    if [[ ! " ${VALID_METHODS[*]} " =~ " ${METHOD} " ]]; then
        echo "Error: Unknown method '${METHOD}'. Choose one of: ${VALID_METHODS[*]}" >&2; exit 1
    fi
fi

run_one() {
    local m="$1"
    mkdir -p "${LOGS_DIR}/control_methods/${m}"
    echo "> Running control method: ${m}"
    ./pipelines/control_methods/control_methods.sh -m "${m}" \
        >> "${LOGS_DIR}/control_methods/${m}/run_control_methods.PID$$.out" \
        2>> "${LOGS_DIR}/control_methods/${m}/run_control_methods.PID$$.err"
}

if [ "$RUN_ALL" = true ]; then
    echo "> Running all control methods: ${VALID_METHODS[*]}"
    for m in "${VALID_METHODS[@]}"; do
        run_one "$m"
    done
    echo "> Done submitting/running all control methods."
else
    run_one "$METHOD"
fi
