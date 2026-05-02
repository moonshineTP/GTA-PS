#!/usr/bin/env bash
set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
ARTIFACTS_DIR="$SCRIPT_DIR/artifacts"
CODE_ROOT="$SCRIPT_DIR/.."
export PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT/threads:$CODE_ROOT/misc"

if [ ! -d "$ARTIFACTS_DIR" ]; then
    echo "ERROR: artifacts directory not found at $ARTIFACTS_DIR"
    exit 1
fi

echo "======================================================"
echo "GTA-PS Compiled Sanity Suite (Benchmarks Only)"
echo "======================================================"
echo "Using artifacts from: $ARTIFACTS_DIR"
echo ""

# Add default publication flags if not provided
ARGS=("$@")
if [[ ! " ${ARGS[*]} " =~ " --seeds " ]]; then
    ARGS+=("--seeds" "1000")
fi
if [[ ! " ${ARGS[*]} " =~ " --workers " ]]; then
    ARGS+=("--workers" "10")
fi

# Parse --start-from argument
START_FROM=1
NEW_ARGS=()
while [[ ${#ARGS[@]} -gt 0 ]]; do
    case "${ARGS[0]}" in
        --start-from)
            shift
            START_FROM="${ARGS[1]}"
            shift
            ;;
        *)
            NEW_ARGS+=("${ARGS[0]}")
            shift
            ;;
    esac
    ARGS=("${ARGS[@]:1}")
done
set -- "${NEW_ARGS[@]}"
INDEX=0

# 1. Time Benchmark
INDEX=$((INDEX+1))
if [ "$INDEX" -lt "$START_FROM" ]; then
    echo "[$INDEX/2] Time Benchmark - skipped (start-from=$START_FROM)"
else
    echo "[$INDEX/2] Time Benchmark (Compiled)"
    "$ARTIFACTS_DIR/time_benchmark.dist/time_benchmark" "$@" || (echo "ERROR: time_benchmark not found in $ARTIFACTS_DIR/time_benchmark.dist" && exit 1)
fi
echo ""

# 2. Convergence Benchmark
INDEX=$((INDEX+1))
if [ "$INDEX" -lt "$START_FROM" ]; then
    echo "[$INDEX/2] Convergence Benchmark - skipped (start-from=$START_FROM)"
else
    echo "[$INDEX/2] Convergence Benchmark (Compiled)"
    "$ARTIFACTS_DIR/convergence_benchmark.dist/convergence_benchmark" "$@" || (echo "ERROR: convergence_benchmark not found in $ARTIFACTS_DIR/convergence_benchmark.dist" && exit 1)
fi
echo ""

echo "Done."
