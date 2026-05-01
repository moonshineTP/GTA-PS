#!/usr/bin/env bash
set -e

# Path to the artifacts directory relative to this script
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
ARTIFACTS_DIR="$SCRIPT_DIR/artifacts"

echo "======================================================"
echo "GTA-PS Compiled RQs Suite"
echo "======================================================"

# RQ1 Proactiveness
echo "[1/9] RQ1 Proactiveness (Compiled)"
if [ -f "$ARTIFACTS_DIR/experiments__rq1__proactiveness.dist/experiments__rq1__proactiveness" ]; then
    "$ARTIFACTS_DIR/experiments__rq1__proactiveness.dist/experiments__rq1__proactiveness" "$@"
fi
echo ""

# RQ1 DP Timeline
echo "[2/9] RQ1 DP Timeline (Compiled)"
if [ -f "$ARTIFACTS_DIR/experiments__rq1__dp__timeline.dist/experiments__rq1__dp__timeline" ]; then
    "$ARTIFACTS_DIR/experiments__rq1__dp__timeline.dist/experiments__rq1__dp__timeline" "$@"
fi
echo ""

# RQ1 Dyna Timeline
echo "[3/9] RQ1 Dyna Timeline (Compiled)"
if [ -f "$ARTIFACTS_DIR/experiments__rq1__dyna__timeline.dist/experiments__rq1__dyna__timeline" ]; then
    "$ARTIFACTS_DIR/experiments__rq1__dyna__timeline.dist/experiments__rq1__dyna__timeline" "$@"
fi
echo ""

# RQ2 Allocation
echo "[4/9] RQ2 Allocation (Compiled)"
if [ -f "$ARTIFACTS_DIR/experiments__rq2__allocation.dist/experiments__rq2__allocation" ]; then
    "$ARTIFACTS_DIR/experiments__rq2__allocation.dist/experiments__rq2__allocation" "$@"
fi
echo ""

# RQ2 Beta
echo "[5/9] RQ2 Beta (Compiled)"
if [ -f "$ARTIFACTS_DIR/experiments__rq2__beta.dist/experiments__rq2__beta" ]; then
    "$ARTIFACTS_DIR/experiments__rq2__beta.dist/experiments__rq2__beta" "$@"
fi
echo ""

# RQ3 Tradeoff
echo "[6/9] RQ3 Tradeoff (Compiled)"
if [ -f "$ARTIFACTS_DIR/experiments__rq3__tradeoff.dist/experiments__rq3__tradeoff" ]; then
    "$ARTIFACTS_DIR/experiments__rq3__tradeoff.dist/experiments__rq3__tradeoff" "$@"
fi
echo ""

# RQ4 Asymmetry
echo "[7/9] RQ4 Asymmetry (Compiled)"
if [ -f "$ARTIFACTS_DIR/experiments__rq4__asymmetry.dist/experiments__rq4__asymmetry" ]; then
    "$ARTIFACTS_DIR/experiments__rq4__asymmetry.dist/experiments__rq4__asymmetry" "$@"
fi
echo ""

# RQ5 Bottlenecks
echo "[8/9] RQ5 Bottlenecks (Compiled)"
if [ -f "$ARTIFACTS_DIR/experiments__rq5__bottlenecks.dist/experiments__rq5__bottlenecks" ]; then
    "$ARTIFACTS_DIR/experiments__rq5__bottlenecks.dist/experiments__rq5__bottlenecks" "$@"
fi
echo ""

# RQ6 Curl Divergence
echo "[9/9] RQ6 Curl Divergence (Compiled)"
if [ -f "$ARTIFACTS_DIR/experiments__rq6__curl__divergence.dist/experiments__rq6__curl__divergence" ]; then
    "$ARTIFACTS_DIR/experiments__rq6__curl__divergence.dist/experiments__rq6__curl__divergence" "$@"
fi
echo ""

echo "Done."
