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
"$ARTIFACTS_DIR/rq1_proactiveness.dist/rq1_proactiveness" "$@"
echo ""

# RQ1 DP Timeline
echo "[2/9] RQ1 DP Timeline (Compiled)"
"$ARTIFACTS_DIR/rq1_dp_timeline.dist/rq1_dp_timeline" "$@"
echo ""

# RQ1 Dyna Timeline
echo "[3/9] RQ1 Dyna Timeline (Compiled)"
"$ARTIFACTS_DIR/rq1_dyna_timeline.dist/rq1_dyna_timeline" "$@"
echo ""

# RQ2 Allocation
echo "[4/9] RQ2 Allocation (Compiled)"
"$ARTIFACTS_DIR/rq2_allocation.dist/rq2_allocation" "$@"
echo ""

# RQ2 Beta
echo "[5/9] RQ2 Beta (Compiled)"
"$ARTIFACTS_DIR/rq2_beta.dist/rq2_beta" "$@"
echo ""

# RQ3 Tradeoff
echo "[6/9] RQ3 Tradeoff (Compiled)"
"$ARTIFACTS_DIR/rq3_tradeoff.dist/rq3_tradeoff" "$@"
echo ""

# RQ4 Asymmetry
echo "[7/9] RQ4 Asymmetry (Compiled)"
"$ARTIFACTS_DIR/rq4_asymmetry.dist/rq4_asymmetry" "$@"
echo ""

# RQ5 Bottlenecks
echo "[8/9] RQ5 Bottlenecks (Compiled)"
"$ARTIFACTS_DIR/rq5_bottlenecks.dist/rq5_bottlenecks" "$@"
echo ""

# RQ6 Curl Divergence
echo "[9/9] RQ6 Curl Divergence (Compiled)"
"$ARTIFACTS_DIR/rq6_curl_divergence.dist/rq6_curl_divergence" "$@"
echo ""

echo "Done."
