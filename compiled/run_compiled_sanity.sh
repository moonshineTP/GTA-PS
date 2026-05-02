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

# Parse --start-from argument
START_FROM=1
NEW_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --start-from)
            shift
            START_FROM="$1"
            shift
            ;;
        *)
            NEW_ARGS+=("$1")
            shift
            ;;
    esac
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
    echo "[$INDEX/4] Convergence Benchmark - skipped (start-from=$START_FROM)"
else
    echo "[$INDEX/4] Convergence Benchmark (Compiled)"
    "$ARTIFACTS_DIR/convergence_benchmark.dist/convergence_benchmark" "$@" || (echo "ERROR: convergence_benchmark not found in $ARTIFACTS_DIR/convergence_benchmark.dist" && exit 1)
fi
echo ""

# 3. Showcase (ablation mode)
INDEX=$((INDEX+1))
if [ "$INDEX" -lt "$START_FROM" ]; then
    echo "[$INDEX/4] Showcase (ablation) - skipped (start-from=$START_FROM)"
else
    echo "[$INDEX/4] Showcase (ablation) (Compiled)"
    "$ARTIFACTS_DIR/showcase.dist/showcase" "$@" --mode ablation || (echo "ERROR: showcase not found" && exit 1)
fi
echo ""

# 4. Showcase (showcase mode)
INDEX=$((INDEX+1))
if [ "$INDEX" -lt "$START_FROM" ]; then
    echo "[$INDEX/4] Showcase (showcase) - skipped (start-from=$START_FROM)"
else
    echo "[$INDEX/4] Showcase (showcase) (Compiled)"
    "$ARTIFACTS_DIR/showcase.dist/showcase" "$@" --mode showcase || (echo "ERROR: showcase not found" && exit 1)
fi
echo ""

echo "Done."
