#!/usr/bin/env bash
set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
CODE_ROOT="$SCRIPT_DIR/.."
ARTIFACTS_DIR="$SCRIPT_DIR/artifacts"
PYTHON="$CODE_ROOT/.venv/bin/python3"

if [ ! -f "$PYTHON" ]; then
    PYTHON="python3"
fi

export PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT/threads:$CODE_ROOT/misc"

echo "Starting Sanity Compilation (Linux)..."
echo ""

SANITY=(
    "misc/time_benchmark.py"
    "misc/convergence_benchmark.py"
    "misc/showcase.py"
)

# Use Zig compiler pipeline if setup exists
if [ -f "$SCRIPT_DIR/../compiler_setup/gcc" ]; then
    export CC="$SCRIPT_DIR/../compiler_setup/gcc"
    export CXX="$SCRIPT_DIR/../compiler_setup/g++"
    export CPATH="$SCRIPT_DIR/../compiler_setup/usr/include/python3.12:$SCRIPT_DIR/../compiler_setup/usr/include/x86_64-linux-gnu/python3.12"
    export LIBRARY_PATH="$SCRIPT_DIR/../compiler_setup/usr/lib/x86_64-linux-gnu"
    export PATH="$SCRIPT_DIR/../compiler_setup:$PATH"
fi

for script in "${SANITY[@]}"; do
    echo "--- Compiling $script ---"
    "$PYTHON" -m nuitka --standalone --output-dir="$ARTIFACTS_DIR" --assume-yes-for-downloads --jobs=5 --include-package=gta_ps --static-libpython=no "$CODE_ROOT/$script"
done

echo ""
echo "Sanity benchmarks compiled to $ARTIFACTS_DIR"
