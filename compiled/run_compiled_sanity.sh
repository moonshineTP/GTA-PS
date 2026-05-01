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
"$ARTIFACTS_DIR/time_benchmark.dist/time_benchmark" "$@"
echo ""

# 2. Convergence Benchmark
echo "[2/2] Convergence Benchmark (Compiled)"
"$ARTIFACTS_DIR/convergence_benchmark.dist/convergence_benchmark" "$@"
echo ""

echo "Done."
