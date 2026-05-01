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
echo "GTA-PS Compiled RQs Suite"
echo "======================================================"
echo "Using artifacts from: $ARTIFACTS_DIR"
echo ""

# RQ1 Proactiveness
echo "[1/9] RQ1 Proactiveness (Compiled)"
"$ARTIFACTS_DIR/rq1_proactiveness.dist/rq1_proactiveness" "$@" || (echo "ERROR: rq1_proactiveness not found" && exit 1)
echo ""

# RQ1 DP Timeline
echo "[2/9] RQ1 DP Timeline (Compiled)"
"$ARTIFACTS_DIR/rq1_dp_timeline.dist/rq1_dp_timeline" "$@" || (echo "ERROR: rq1_dp_timeline not found" && exit 1)
echo ""

# RQ1 Dyna Timeline
echo "[3/9] RQ1 Dyna Timeline (Compiled)"
"$ARTIFACTS_DIR/rq1_dyna_timeline.dist/rq1_dyna_timeline" "$@" || (echo "ERROR: rq1_dyna_timeline not found" && exit 1)
echo ""

# RQ2 Allocation
echo "[4/9] RQ2 Allocation (Compiled)"
"$ARTIFACTS_DIR/rq2_allocation.dist/rq2_allocation" "$@" || (echo "ERROR: rq2_allocation not found" && exit 1)
echo ""

# RQ2 Beta
echo "[5/9] RQ2 Beta (Compiled)"
"$ARTIFACTS_DIR/rq2_beta.dist/rq2_beta" "$@" || (echo "ERROR: rq2_beta not found" && exit 1)
echo ""

# RQ3 Tradeoff
echo "[6/9] RQ3 Tradeoff (Compiled)"
"$ARTIFACTS_DIR/rq3_tradeoff.dist/rq3_tradeoff" "$@" || (echo "ERROR: rq3_tradeoff not found" && exit 1)
echo ""

# RQ4 Asymmetry
echo "[7/9] RQ4 Asymmetry (Compiled)"
"$ARTIFACTS_DIR/rq4_asymmetry.dist/rq4_asymmetry" "$@" || (echo "ERROR: rq4_asymmetry not found" && exit 1)
echo ""

# RQ5 Bottlenecks
echo "[8/9] RQ5 Bottlenecks (Compiled)"
"$ARTIFACTS_DIR/rq5_bottlenecks.dist/rq5_bottlenecks" "$@" || (echo "ERROR: rq5_bottlenecks not found" && exit 1)
echo ""

# RQ6 Curl Divergence
echo "[9/9] RQ6 Curl Divergence (Compiled)"
"$ARTIFACTS_DIR/rq6_curl_divergence.dist/rq6_curl_divergence" "$@" || (echo "ERROR: rq6_curl_divergence not found" && exit 1)
echo ""

echo "Done."
