#!/usr/bin/env bash
set -e

# Path to the artifacts directory relative to this script
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
ARTIFACTS_DIR="$SCRIPT_DIR/artifacts"

echo "======================================================"
echo "GTA-PS Compiled Ablations Suite"
echo "======================================================"

# Smoothness Ablation
echo "[1/11] Smoothness Ablation (Compiled)"
"$ARTIFACTS_DIR/smoothness_ablation.dist/smoothness_ablation" "$@"
echo ""

# Mixing Threshold Ablation
echo "[2/11] Mixing Threshold Ablation (Compiled)"
"$ARTIFACTS_DIR/mixing_threshold_ablation.dist/mixing_threshold_ablation" "$@"
echo ""

# Shift-rate Ablation
echo "[3/11] Shift-rate Ablation (Compiled)"
"$ARTIFACTS_DIR/shift_rate_ablation.dist/shift_rate_ablation" "$@"
echo ""

# Boltzmann-b Ablation
echo "[4/11] Boltzmann-b Ablation (Compiled)"
"$ARTIFACTS_DIR/boltzmann_b_ablation.dist/boltzmann_b_ablation" "$@"
echo ""

# Scheduler Ablation
echo "[5/11] Scheduler Ablation (Compiled)"
"$ARTIFACTS_DIR/scheduler_ablation.dist/scheduler_ablation" "$@"
echo ""

# Gamma Ablation
echo "[6/11] Gamma Ablation (Compiled)"
"$ARTIFACTS_DIR/gamma_ablation.dist/gamma_ablation" "$@"
echo ""

# Mixing Ablation
echo "[7/11] Mixing Ablation (Compiled)"
"$ARTIFACTS_DIR/mixing_ablation.dist/mixing_ablation" "$@"
echo ""

# Periodicity Ablation
echo "[8/11] Periodicity Ablation (Compiled)"
"$ARTIFACTS_DIR/periodicity_ablation.dist/periodicity_ablation" "$@"
echo ""

# Reward-support Ablation
echo "[9/11] Reward-support Ablation (Compiled)"
"$ARTIFACTS_DIR/reward_support_ablation.dist/reward_support_ablation" "$@"
echo ""

# Cumulative Ablation
echo "[10/11] Cumulative Ablation (Compiled)"
"$ARTIFACTS_DIR/cumulative_ablation.dist/cumulative_ablation" "$@"
echo ""

# Showcase
echo "[11/11] Showcase (Compiled)"
"$ARTIFACTS_DIR/showcase.dist/showcase" "$@"
echo ""

echo "Done."
