#!/bin/bash
#
# Auto-analyze P8-2 when experiment completes
# Usage: bash scripts/p8_2_auto_analyze_when_complete.sh
#

set -e

CHECKPOINT="experiments/logs/checkpoints/p8_2_cf_factors_beauty_checkpoint.json"
TOTAL_USERS=502
CHECK_INTERVAL=300  # Check every 5 minutes

echo "================================"
echo "P8-2 Auto-Analyzer"
echo "================================"
echo "Monitoring: $CHECKPOINT"
echo "Target: $TOTAL_USERS users"
echo "Check interval: $CHECK_INTERVAL seconds"
echo ""

while true; do
    # Check if checkpoint exists
    if [ ! -f "$CHECKPOINT" ]; then
        echo "[$(date '+%H:%M:%S')] Checkpoint not found, waiting..."
        sleep $CHECK_INTERVAL
        continue
    fi

    # Get completion status
    COMPLETED=$(python3 -c "
import json
with open('$CHECKPOINT') as f:
    data = json.load(f)
print(len(data.get('completed_indices', [])))
")

    PERCENT=$((COMPLETED * 100 / TOTAL_USERS))

    echo "[$(date '+%H:%M:%S')] Progress: $COMPLETED/$TOTAL_USERS ($PERCENT%)"

    # Check if complete
    if [ "$COMPLETED" -ge "$TOTAL_USERS" ]; then
        echo ""
        echo "================================"
        echo "✅ EXPERIMENT COMPLETE!"
        echo "================================"
        echo ""

        # Run full analysis
        echo "[1/3] Running full P8-2 analysis..."
        python scripts/analyze_p8_2_intermediate.py > experiments/logs/p8_2_final_analysis.txt
        echo "✓ Analysis saved to: experiments/logs/p8_2_final_analysis.txt"

        # Compute Bootstrap CI
        echo ""
        echo "[2/3] Computing Bootstrap confidence intervals..."
        python scripts/compute_bootstrap_ci_p8_2.py
        echo "✓ Bootstrap CI computed"

        # Generate paper update suggestions
        echo ""
        echo "[3/3] Generating paper update suggestions..."
        echo "✓ Review P8_2_ANALYSIS_DRAFT.md for paper updates"

        echo ""
        echo "================================"
        echo "All post-processing complete!"
        echo "================================"
        echo ""
        echo "Next steps:"
        echo "1. Review: experiments/logs/p8_2_final_analysis.txt"
        echo "2. Review: experiments/P8_2_ANALYSIS_DRAFT.md"
        echo "3. Update paper Section 6.3 with findings"
        echo "4. Compute statistical significance tests"
        echo ""

        break
    else
        REMAINING=$((TOTAL_USERS - COMPLETED))
        sleep $CHECK_INTERVAL
    fi
done
