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

# 1. Time Benchmark
echo "[1/2] Time Benchmark (Compiled)"
"$ARTIFACTS_DIR/time_benchmark.dist/time_benchmark" "$@"
echo ""

# 2. Convergence Benchmark
echo "[2/2] Convergence Benchmark (Compiled)"
"$ARTIFACTS_DIR/convergence_benchmark.dist/convergence_benchmark" "$@"
echo ""

echo "Done."
