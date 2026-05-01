#!/usr/bin/env bash
set -e

# Path to the artifacts directory relative to this script
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# Find artifacts directory
ARTIFACTS_DIR=""

# Strategy 1: Check if ARTIFACTS_DIR_ENV is set
if [ -n "$ARTIFACTS_DIR_ENV" ] && [ -d "$ARTIFACTS_DIR_ENV" ]; then
    ARTIFACTS_DIR="$ARTIFACTS_DIR_ENV"
fi

# Strategy 2: Check relative to script (code/compiled/artifacts)
if [ -z "$ARTIFACTS_DIR" ] && [ -d "$SCRIPT_DIR/artifacts" ]; then
    ARTIFACTS_DIR="$SCRIPT_DIR/artifacts"
fi

# Strategy 3: Check in code directory (code/artifacts)
if [ -z "$ARTIFACTS_DIR" ] && [ -d "$SCRIPT_DIR/../artifacts" ]; then
    ARTIFACTS_DIR="$SCRIPT_DIR/../artifacts"
fi

# Strategy 4: Check in parent workspace (parent/artifacts)
if [ -z "$ARTIFACTS_DIR" ] && [ -d "$SCRIPT_DIR/../../artifacts" ]; then
    ARTIFACTS_DIR="$SCRIPT_DIR/../../artifacts"
fi

# Strategy 5: Check further up
if [ -z "$ARTIFACTS_DIR" ] && [ -d "$SCRIPT_DIR/../../../artifacts" ]; then
    ARTIFACTS_DIR="$SCRIPT_DIR/../../../artifacts"
fi

if [ -z "$ARTIFACTS_DIR" ]; then
    echo "ERROR: Cannot find artifacts directory"
    echo ""
    echo "Searched in:"
    echo "  - $SCRIPT_DIR/artifacts"
    echo "  - $SCRIPT_DIR/../artifacts"
    echo "  - $SCRIPT_DIR/../../artifacts"
    echo "  - $SCRIPT_DIR/../../../artifacts"
    echo ""
    echo "Solution: Set ARTIFACTS_DIR_ENV environment variable or ensure artifacts folder exists"
    echo "  Example: export ARTIFACTS_DIR_ENV=/path/to/artifacts"
    exit 1
fi

export PYTHONPATH="$SCRIPT_DIR/../src:$SCRIPT_DIR/../threads:$SCRIPT_DIR/../misc"

echo "======================================================"
echo "GTA-PS Compiled Sanity Suite (Benchmarks Only)"
echo "======================================================"
echo "Using artifacts from: $ARTIFACTS_DIR"
echo ""

# 1. Time Benchmark
echo "[1/2] Time Benchmark (Compiled)"
"$ARTIFACTS_DIR/time_benchmark.dist/time_benchmark" "$@" || (echo "ERROR: time_benchmark not found in $ARTIFACTS_DIR/time_benchmark.dist" && exit 1)
echo ""

# 2. Convergence Benchmark
echo "[2/2] Convergence Benchmark (Compiled)"
"$ARTIFACTS_DIR/convergence_benchmark.dist/convergence_benchmark" "$@" || (echo "ERROR: convergence_benchmark not found in $ARTIFACTS_DIR/convergence_benchmark.dist" && exit 1)
echo ""

echo "Done."
