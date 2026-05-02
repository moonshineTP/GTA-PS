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
echo "GTA-PS Compiled Ablations Suite"
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

# Smoothness Ablation
INDEX=$((INDEX+1))
if [ "$INDEX" -lt "$START_FROM" ]; then
    echo "[$INDEX/11] Smoothness Ablation - skipped (start-from=$START_FROM)"
else
    echo "[$INDEX/11] Smoothness Ablation (Compiled)"
    "$ARTIFACTS_DIR/smoothness_ablation.dist/smoothness_ablation" "$@" || (echo "ERROR: smoothness_ablation not found in $ARTIFACTS_DIR/smoothness_ablation.dist" && exit 1)
fi
echo ""

# Mixing Threshold Ablation
INDEX=$((INDEX+1))
if [ "$INDEX" -lt "$START_FROM" ]; then
    echo "[$INDEX/11] Mixing Threshold Ablation - skipped (start-from=$START_FROM)"
else
    echo "[$INDEX/11] Mixing Threshold Ablation (Compiled)"
    "$ARTIFACTS_DIR/mixing_threshold_ablation.dist/mixing_threshold_ablation" "$@" || (echo "ERROR: mixing_threshold_ablation not found" && exit 1)
fi
echo ""

# Shift-rate Ablation
INDEX=$((INDEX+1))
if [ "$INDEX" -lt "$START_FROM" ]; then
    echo "[$INDEX/11] Shift-rate Ablation - skipped (start-from=$START_FROM)"
else
    echo "[$INDEX/11] Shift-rate Ablation (Compiled)"
    "$ARTIFACTS_DIR/shift_rate_ablation.dist/shift_rate_ablation" "$@" || (echo "ERROR: shift_rate_ablation not found" && exit 1)
fi
echo ""

# Boltzmann-b Ablation
INDEX=$((INDEX+1))
if [ "$INDEX" -lt "$START_FROM" ]; then
    echo "[$INDEX/11] Boltzmann-b Ablation - skipped (start-from=$START_FROM)"
else
    echo "[$INDEX/11] Boltzmann-b Ablation (Compiled)"
    "$ARTIFACTS_DIR/boltzmann_b_ablation.dist/boltzmann_b_ablation" "$@" || (echo "ERROR: boltzmann_b_ablation not found" && exit 1)
fi
echo ""

# Scheduler Ablation
INDEX=$((INDEX+1))
if [ "$INDEX" -lt "$START_FROM" ]; then
    echo "[$INDEX/11] Scheduler Ablation - skipped (start-from=$START_FROM)"
else
    echo "[$INDEX/11] Scheduler Ablation (Compiled)"
    "$ARTIFACTS_DIR/scheduler_ablation.dist/scheduler_ablation" "$@" || (echo "ERROR: scheduler_ablation not found" && exit 1)
fi
echo ""

# Gamma Ablation
INDEX=$((INDEX+1))
if [ "$INDEX" -lt "$START_FROM" ]; then
    echo "[$INDEX/11] Gamma Ablation - skipped (start-from=$START_FROM)"
else
    echo "[$INDEX/11] Gamma Ablation (Compiled)"
    "$ARTIFACTS_DIR/gamma_ablation.dist/gamma_ablation" "$@" || (echo "ERROR: gamma_ablation not found" && exit 1)
fi
echo ""

# Mixing Ablation
INDEX=$((INDEX+1))
if [ "$INDEX" -lt "$START_FROM" ]; then
    echo "[$INDEX/11] Mixing Ablation - skipped (start-from=$START_FROM)"
else
    echo "[$INDEX/11] Mixing Ablation (Compiled)"
    "$ARTIFACTS_DIR/mixing_ablation.dist/mixing_ablation" "$@" || (echo "ERROR: mixing_ablation not found" && exit 1)
fi
echo ""

# Periodicity Ablation
INDEX=$((INDEX+1))
if [ "$INDEX" -lt "$START_FROM" ]; then
    echo "[$INDEX/11] Periodicity Ablation - skipped (start-from=$START_FROM)"
else
    echo "[$INDEX/11] Periodicity Ablation (Compiled)"
    "$ARTIFACTS_DIR/periodicity_ablation.dist/periodicity_ablation" "$@" || (echo "ERROR: periodicity_ablation not found" && exit 1)
fi
echo ""

# Reward-support Ablation
INDEX=$((INDEX+1))
if [ "$INDEX" -lt "$START_FROM" ]; then
    echo "[$INDEX/11] Reward-support Ablation - skipped (start-from=$START_FROM)"
else
    echo "[$INDEX/11] Reward-support Ablation (Compiled)"
    "$ARTIFACTS_DIR/reward_support_ablation.dist/reward_support_ablation" "$@" || (echo "ERROR: reward_support_ablation not found" && exit 1)
fi
echo ""

# Cumulative Ablation
INDEX=$((INDEX+1))
if [ "$INDEX" -lt "$START_FROM" ]; then
    echo "[$INDEX/11] Cumulative Ablation - skipped (start-from=$START_FROM)"
else
    echo "[$INDEX/11] Cumulative Ablation (Compiled)"
    "$ARTIFACTS_DIR/cumulative_ablation.dist/cumulative_ablation" "$@" || (echo "ERROR: cumulative_ablation not found" && exit 1)
fi
echo ""

# Showcase
INDEX=$((INDEX+1))
if [ "$INDEX" -lt "$START_FROM" ]; then
    echo "[$INDEX/11] Showcase - skipped (start-from=$START_FROM)"
else
    echo "[$INDEX/11] Showcase (Compiled)"
    "$ARTIFACTS_DIR/showcase.dist/showcase" "$@" || (echo "ERROR: showcase not found" && exit 1)
fi
echo ""

echo "Done."
