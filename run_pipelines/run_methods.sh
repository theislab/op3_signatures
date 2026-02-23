#!/bin/bash

set -e

LOGS_DIR=./logs
METHOD=""

VALID_METHODS=(
    "pyboost"
)

while getopts ":m:h" opt; do
    case $opt in
        h)
            echo "Run: $0 -m METHOD"
            echo ""
            echo "Required:"
            echo "  -m  Method to run. One of: ${VALID_METHODS[*]}"
            echo ""
            echo "  -h  Show this help message"
            exit 0
            ;;
        m) METHOD=$OPTARG ;;
        \?)
            echo "Error: Invalid option -$OPTARG" >&2; exit 1 ;;
        :)
            echo "Error: Option -$OPTARG requires an argument" >&2; exit 1 ;;
    esac
done

if [ -z "$METHOD" ]; then
    echo "Error: -m METHOD is required. Choose one of: ${VALID_METHODS[*]}" >&2; exit 1
fi

if [[ ! " ${VALID_METHODS[*]} " =~ " ${METHOD} " ]]; then
    echo "Error: Unknown method '${METHOD}'. Choose one of: ${VALID_METHODS[*]}" >&2; exit 1
fi

mkdir -p "${LOGS_DIR}/${METHOD}"

echo "> Running method: ${METHOD}"

./pipelines/${METHOD}/${METHOD}.sh \
    > "${LOGS_DIR}/${METHOD}/run_${METHOD}.PID$$.out" \
    2> "${LOGS_DIR}/${METHOD}/run_${METHOD}.PID$$.err" &
