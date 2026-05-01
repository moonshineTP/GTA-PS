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
echo "GTA-PS Compiled Ablations Suite"
echo "======================================================"
echo "Using artifacts from: $ARTIFACTS_DIR"
echo ""

# Smoothness Ablation
echo "[1/11] Smoothness Ablation (Compiled)"
"$ARTIFACTS_DIR/smoothness_ablation.dist/smoothness_ablation" "$@" || (echo "ERROR: smoothness_ablation not found in $ARTIFACTS_DIR/smoothness_ablation.dist" && exit 1)
echo ""

# Mixing Threshold Ablation
echo "[2/11] Mixing Threshold Ablation (Compiled)"
"$ARTIFACTS_DIR/mixing_threshold_ablation.dist/mixing_threshold_ablation" "$@" || (echo "ERROR: mixing_threshold_ablation not found" && exit 1)
echo ""

# Shift-rate Ablation
echo "[3/11] Shift-rate Ablation (Compiled)"
"$ARTIFACTS_DIR/shift_rate_ablation.dist/shift_rate_ablation" "$@" || (echo "ERROR: shift_rate_ablation not found" && exit 1)
echo ""

# Boltzmann-b Ablation
echo "[4/11] Boltzmann-b Ablation (Compiled)"
"$ARTIFACTS_DIR/boltzmann_b_ablation.dist/boltzmann_b_ablation" "$@" || (echo "ERROR: boltzmann_b_ablation not found" && exit 1)
echo ""

# Scheduler Ablation
echo "[5/11] Scheduler Ablation (Compiled)"
"$ARTIFACTS_DIR/scheduler_ablation.dist/scheduler_ablation" "$@" || (echo "ERROR: scheduler_ablation not found" && exit 1)
echo ""

# Gamma Ablation
echo "[6/11] Gamma Ablation (Compiled)"
"$ARTIFACTS_DIR/gamma_ablation.dist/gamma_ablation" "$@" || (echo "ERROR: gamma_ablation not found" && exit 1)
echo ""

# Mixing Ablation
echo "[7/11] Mixing Ablation (Compiled)"
"$ARTIFACTS_DIR/mixing_ablation.dist/mixing_ablation" "$@" || (echo "ERROR: mixing_ablation not found" && exit 1)
echo ""

# Periodicity Ablation
echo "[8/11] Periodicity Ablation (Compiled)"
"$ARTIFACTS_DIR/periodicity_ablation.dist/periodicity_ablation" "$@" || (echo "ERROR: periodicity_ablation not found" && exit 1)
echo ""

# Reward-support Ablation
echo "[9/11] Reward-support Ablation (Compiled)"
"$ARTIFACTS_DIR/reward_support_ablation.dist/reward_support_ablation" "$@" || (echo "ERROR: reward_support_ablation not found" && exit 1)
echo ""

# Cumulative Ablation
echo "[10/11] Cumulative Ablation (Compiled)"
"$ARTIFACTS_DIR/cumulative_ablation.dist/cumulative_ablation" "$@" || (echo "ERROR: cumulative_ablation not found" && exit 1)
echo ""

# Showcase
echo "[11/11] Showcase (Compiled)"
"$ARTIFACTS_DIR/showcase.dist/showcase" "$@" || (echo "ERROR: showcase not found" && exit 1)
echo ""

echo "Done."
