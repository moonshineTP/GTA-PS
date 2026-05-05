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
echo "GTA-PS Compiled RQs Suite"
echo "======================================================"
echo "Using artifacts from: $ARTIFACTS_DIR"
echo ""

# Add default publication flags if not provided
ARGS=("$@")
if [[ ! " ${ARGS[*]} " =~ " --seeds " ]]; then
    ARGS+=("--seeds" "1000")
fi
if [[ ! " ${ARGS[*]} " =~ " --workers " ]]; then
    ARGS+=("--workers" "10")
fi

# Parse --start-from argument
START_FROM=1
NEW_ARGS=()
while [[ ${#ARGS[@]} -gt 0 ]]; do
    case "${ARGS[0]}" in
        --start-from)
            shift
            START_FROM="${ARGS[1]}"
            shift
            ;;
        *)
            NEW_ARGS+=("${ARGS[0]}")
            shift
            ;;
    esac
    ARGS=("${ARGS[@]:1}")
done
set -- "${NEW_ARGS[@]}"
INDEX=0

# RQ1 Convergence
INDEX=$((INDEX+1))
if [ "$INDEX" -lt "$START_FROM" ]; then
    echo "[$INDEX/9] RQ1 Convergence - skipped (start-from=$START_FROM)"
else
    echo "[$INDEX/9] RQ1 Convergence (Compiled)"
    "$ARTIFACTS_DIR/rq1_convergence.dist/rq1_convergence" "$@" || (echo "ERROR: rq1_convergence not found" && exit 1)
fi
echo ""

# RQ1 DP Timeline
INDEX=$((INDEX+1))
if [ "$INDEX" -lt "$START_FROM" ]; then
    echo "[$INDEX/9] RQ1 DP Timeline - skipped (start-from=$START_FROM)"
else
    echo "[$INDEX/9] RQ1 DP Timeline (Compiled)"
    "$ARTIFACTS_DIR/rq1_dp_timeline.dist/rq1_dp_timeline" "$@" || (echo "ERROR: rq1_dp_timeline not found" && exit 1)
fi
echo ""

# RQ1 Dyna Timeline
INDEX=$((INDEX+1))
if [ "$INDEX" -lt "$START_FROM" ]; then
    echo "[$INDEX/9] RQ1 Dyna Timeline - skipped (start-from=$START_FROM)"
else
    echo "[$INDEX/9] RQ1 Dyna Timeline (Compiled)"
    "$ARTIFACTS_DIR/rq1_dyna_timeline.dist/rq1_dyna_timeline" "$@" || (echo "ERROR: rq1_dyna_timeline not found" && exit 1)
fi
echo ""

# RQ2 Allocation
INDEX=$((INDEX+1))
if [ "$INDEX" -lt "$START_FROM" ]; then
    echo "[$INDEX/9] RQ2 Allocation - skipped (start-from=$START_FROM)"
else
    echo "[$INDEX/9] RQ2 Allocation (Compiled)"
    "$ARTIFACTS_DIR/rq2_allocation.dist/rq2_allocation" "$@" || (echo "ERROR: rq2_allocation not found" && exit 1)
fi
echo ""

# RQ2 Beta
INDEX=$((INDEX+1))
if [ "$INDEX" -lt "$START_FROM" ]; then
    echo "[$INDEX/9] RQ2 Beta - skipped (start-from=$START_FROM)"
else
    echo "[$INDEX/9] RQ2 Beta (Compiled)"
    "$ARTIFACTS_DIR/rq2_beta.dist/rq2_beta" "$@" || (echo "ERROR: rq2_beta not found" && exit 1)
fi
echo ""

# RQ3 Tradeoff
INDEX=$((INDEX+1))
if [ "$INDEX" -lt "$START_FROM" ]; then
    echo "[$INDEX/9] RQ3 Tradeoff - skipped (start-from=$START_FROM)"
else
    echo "[$INDEX/9] RQ3 Tradeoff (Compiled)"
    "$ARTIFACTS_DIR/rq3_tradeoff.dist/rq3_tradeoff" "$@" || (echo "ERROR: rq3_tradeoff not found" && exit 1)
fi
echo ""

# RQ4 Asymmetry
INDEX=$((INDEX+1))
if [ "$INDEX" -lt "$START_FROM" ]; then
    echo "[$INDEX/9] RQ4 Asymmetry - skipped (start-from=$START_FROM)"
else
    echo "[$INDEX/9] RQ4 Asymmetry (Compiled)"
    "$ARTIFACTS_DIR/rq4_asymmetry.dist/rq4_asymmetry" "$@" || (echo "ERROR: rq4_asymmetry not found" && exit 1)
fi
echo ""

# RQ5 Bottlenecks
INDEX=$((INDEX+1))
if [ "$INDEX" -lt "$START_FROM" ]; then
    echo "[$INDEX/9] RQ5 Bottlenecks - skipped (start-from=$START_FROM)"
else
    echo "[$INDEX/9] RQ5 Bottlenecks (Compiled)"
    "$ARTIFACTS_DIR/rq5_bottlenecks.dist/rq5_bottlenecks" "$@" || (echo "ERROR: rq5_bottlenecks not found" && exit 1)
fi
echo ""

# RQ6 Curl Divergence
INDEX=$((INDEX+1))
if [ "$INDEX" -lt "$START_FROM" ]; then
    echo "[$INDEX/9] RQ6 Curl Divergence - skipped (start-from=$START_FROM)"
else
    echo "[$INDEX/9] RQ6 Curl Divergence (Compiled)"
    "$ARTIFACTS_DIR/rq6_curl_divergence.dist/rq6_curl_divergence" "$@" || (echo "ERROR: rq6_curl_divergence not found" && exit 1)
fi
echo ""

echo "Done."
