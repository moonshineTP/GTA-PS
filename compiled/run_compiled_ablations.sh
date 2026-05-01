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
if [ -f "$ARTIFACTS_DIR/misc__smoothness__ablation.dist/misc__smoothness__ablation" ]; then
    "$ARTIFACTS_DIR/misc__smoothness__ablation.dist/misc__smoothness__ablation" "$@"
fi
echo ""

# Mixing Threshold Ablation
echo "[2/11] Mixing Threshold Ablation (Compiled)"
if [ -f "$ARTIFACTS_DIR/misc__mixing__threshold__ablation.dist/misc__mixing__threshold__ablation" ]; then
    "$ARTIFACTS_DIR/misc__mixing__threshold__ablation.dist/misc__mixing__threshold__ablation" "$@"
fi
echo ""

# Shift-rate Ablation
echo "[3/11] Shift-rate Ablation (Compiled)"
if [ -f "$ARTIFACTS_DIR/misc__shift__rate__ablation.dist/misc__shift__rate__ablation" ]; then
    "$ARTIFACTS_DIR/misc__shift__rate__ablation.dist/misc__shift__rate__ablation" "$@"
fi
echo ""

# Boltzmann-b Ablation
echo "[4/11] Boltzmann-b Ablation (Compiled)"
if [ -f "$ARTIFACTS_DIR/misc__boltzmann__b__ablation.dist/misc__boltzmann__b__ablation" ]; then
    "$ARTIFACTS_DIR/misc__boltzmann__b__ablation.dist/misc__boltzmann__b__ablation" "$@"
fi
echo ""

# Scheduler Ablation
echo "[5/11] Scheduler Ablation (Compiled)"
if [ -f "$ARTIFACTS_DIR/misc__scheduler__ablation.dist/misc__scheduler__ablation" ]; then
    "$ARTIFACTS_DIR/misc__scheduler__ablation.dist/misc__scheduler__ablation" "$@"
fi
echo ""

# Gamma Ablation
echo "[6/11] Gamma Ablation (Compiled)"
if [ -f "$ARTIFACTS_DIR/misc__gamma__ablation.dist/misc__gamma__ablation" ]; then
    "$ARTIFACTS_DIR/misc__gamma__ablation.dist/misc__gamma__ablation" "$@"
fi
echo ""

# Mixing Ablation
echo "[7/11] Mixing Ablation (Compiled)"
if [ -f "$ARTIFACTS_DIR/misc__mixing__ablation.dist/misc__mixing__ablation" ]; then
    "$ARTIFACTS_DIR/misc__mixing__ablation.dist/misc__mixing__ablation" "$@"
fi
echo ""

# Periodicity Ablation
echo "[8/11] Periodicity Ablation (Compiled)"
if [ -f "$ARTIFACTS_DIR/misc__periodicity__ablation.dist/misc__periodicity__ablation" ]; then
    "$ARTIFACTS_DIR/misc__periodicity__ablation.dist/misc__periodicity__ablation" "$@"
fi
echo ""

# Reward-support Ablation
echo "[9/11] Reward-support Ablation (Compiled)"
if [ -f "$ARTIFACTS_DIR/misc__reward__support__ablation.dist/misc__reward__support__ablation" ]; then
    "$ARTIFACTS_DIR/misc__reward__support__ablation.dist/misc__reward__support__ablation" "$@"
fi
echo ""

# Cumulative Ablation
echo "[10/11] Cumulative Ablation (Compiled)"
if [ -f "$ARTIFACTS_DIR/misc__cumulative__ablation.dist/misc__cumulative__ablation" ]; then
    "$ARTIFACTS_DIR/misc__cumulative__ablation.dist/misc__cumulative__ablation" "$@"
fi
echo ""

# Showcase
echo "[11/11] Showcase (Compiled)"
if [ -f "$ARTIFACTS_DIR/misc__showcase.dist/misc__showcase" ]; then
    "$ARTIFACTS_DIR/misc__showcase.dist/misc__showcase" "$@"
fi
echo ""

echo "Done."
