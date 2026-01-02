#!/bin/bash
# Simple script to run the full comparison
# This avoids all the import issues by running everything properly

echo "================================================================================"
echo "FlowSeek vs Lucas-Kanade Comparison Pipeline"
echo "================================================================================"

# Set environment with ALL necessary paths
export PYTHONPATH=/proj/ciptmp/we03cyna/flowseek_cloned:/proj/ciptmp/we03cyna/flowseek_cloned/core:/proj/ciptmp/we03cyna/flowseek_cloned/core/depth_anything_v2:$PYTHONPATH
export MPLCONFIGDIR=/tmp/matplotlib-cache

# Paths
SINTEL_ROOT="/proj/ciptmp/we03cyna/my_ml_data/MPI-Sintel-complete"
FLOWSEEK_ROOT="/proj/ciptmp/we03cyna/flowseek_cloned"
SCRIPT_DIR="/proj/ciptmp/we03cyna/fickdichwindows"
OUTPUT_DIR="$SCRIPT_DIR/results_flowseek_vs_lk"

echo ""
echo "Configuration:"
echo "  Sintel:   $SINTEL_ROOT"
echo "  FlowSeek: $FLOWSEEK_ROOT"
echo "  Output:   $OUTPUT_DIR"
echo ""

# IMPORTANT: Change to FlowSeek directory so relative paths work
cd "$FLOWSEEK_ROOT"

echo "Running from: $(pwd)"
echo ""

# Run comparison (use absolute path to script)
python "$SCRIPT_DIR/optical_flow_comparison.py" \
    --sintel-root "$SINTEL_ROOT" \
    --flowseek-root "$FLOWSEEK_ROOT" \
    --use-flowseek \
    --sequences alley_1 bamboo_1 cave_2 market_5 temple_2 \
    --visualize-count 5 \
    --texture-percentile 25 \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "================================================================================"
echo "Complete! Check results in: $OUTPUT_DIR"
echo "================================================================================"