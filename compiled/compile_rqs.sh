#!/usr/bin/env bash
set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
CODE_ROOT="$SCRIPT_DIR/.."
ARTIFACTS_DIR="$SCRIPT_DIR/artifacts"
PYTHON="$CODE_ROOT/.venv/bin/python3"

# Fallback to system python if venv not found
if [ ! -f "$PYTHON" ]; then
    PYTHON="python3"
fi

export PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT/threads:$CODE_ROOT/misc"

echo "Starting RQs Compilation (Linux)..."
echo ""

RQS=(
    "experiments/rq1_proactiveness.py"
    "experiments/rq1_dp_timeline.py"
    "experiments/rq1_dyna_timeline.py"
    "experiments/rq2_allocation.py"
    "experiments/rq2_beta.py"
    "experiments/rq3_tradeoff.py"
    "experiments/rq4_asymmetry.py"
    "experiments/rq5_bottlenecks.py"
    "experiments/rq6_curl_divergence.py"
)

# Use Zig compiler pipeline if setup exists
if [ -f "$SCRIPT_DIR/../compiler_setup/gcc" ]; then
    export CC="$SCRIPT_DIR/../compiler_setup/gcc"
    export CXX="$SCRIPT_DIR/../compiler_setup/g++"
    export CPATH="$SCRIPT_DIR/../compiler_setup/usr/include/python3.12:$SCRIPT_DIR/../compiler_setup/usr/include/x86_64-linux-gnu/python3.12"
    export LIBRARY_PATH="$SCRIPT_DIR/../compiler_setup/usr/lib/x86_64-linux-gnu"
    export PATH="$SCRIPT_DIR/../compiler_setup:$PATH"
fi

for script in "${RQS[@]}"; do
    echo "--- Compiling $script ---"
    "$PYTHON" -m nuitka --standalone --output-dir="$ARTIFACTS_DIR" --assume-yes-for-downloads --jobs=5 --include-package=gta_ps --static-libpython=no "$CODE_ROOT/$script"
done

echo ""
echo "All RQs compiled to $ARTIFACTS_DIR"
