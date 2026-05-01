#!/usr/bin/env bash
set -e

# Path to the artifacts directory relative to this script
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
ARTIFACTS_DIR="$SCRIPT_DIR/artifacts"

echo "======================================================"
echo "GTA-PS Compiled Sanity Suite (Benchmarks Only)"
echo "======================================================"

# 1. Time Benchmark
echo "[1/2] Time Benchmark (Compiled)"
if [ -f "$ARTIFACTS_DIR/time_benchmark.dist/time_benchmark" ]; then
    "$ARTIFACTS_DIR/time_benchmark.dist/time_benchmark" "$@"
elif [ -f "$ARTIFACTS_DIR/misc__time__benchmark.dist/misc__time__benchmark" ]; then
    "$ARTIFACTS_DIR/misc__time__benchmark.dist/misc__time__benchmark" "$@"
else
    echo "WARNING: Compiled time_benchmark not found."
fi
echo ""

# 2. Convergence Benchmark
echo "[2/2] Convergence Benchmark (Compiled)"
if [ -f "$ARTIFACTS_DIR/convergence_benchmark.dist/convergence_benchmark" ]; then
    "$ARTIFACTS_DIR/convergence_benchmark.dist/convergence_benchmark" "$@"
elif [ -f "$ARTIFACTS_DIR/misc__convergence__benchmark.dist/misc__convergence__benchmark" ]; then
    "$ARTIFACTS_DIR/misc__convergence__benchmark.dist/misc__convergence__benchmark" "$@"
else
    echo "WARNING: Compiled convergence_benchmark not found."
fi
echo ""

echo "Done."
