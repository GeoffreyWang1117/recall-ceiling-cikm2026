#!/bin/bash
# Monitor P8-2 Beauty experiment progress

CHECKPOINT="/home/coder-gw/Projects/GraphLLMRec/experiments/logs/checkpoints/p8_2_cf_factors_beauty_checkpoint.json"
LOG_FILE="/home/coder-gw/Projects/GraphLLMRec/experiments/logs/p8_2_beauty_execution.log"

echo "================================"
echo "P8-2 Beauty Experiment Monitor"
echo "================================"
echo ""

# Check if experiment is running
if ps aux | grep -v grep | grep "p8_2_cf_factors_beauty.py" > /dev/null; then
    echo "✅ Experiment is RUNNING"
else
    echo "⚠️  Experiment is NOT running"
fi

echo ""

# Check checkpoint progress
if [ -f "$CHECKPOINT" ]; then
    echo "📊 Progress from checkpoint:"
    python3 - <<EOF
import json
try:
    with open("$CHECKPOINT") as f:
        data = json.load(f)
    completed = len([k for k, v in data.get('per_item_results', {}).items() if v is not None])
    total = data.get('metadata', {}).get('total_items', 502)
    pct = (completed / total * 100) if total > 0 else 0
    print(f"  Completed: {completed}/{total} users ({pct:.1f}%)")

    if completed > 0 and 'metadata' in data:
        elapsed = data['metadata'].get('execution_time', 0)
        avg_time = elapsed / completed if completed > 0 else 0
        remaining = (total - completed) * avg_time
        print(f"  Avg time/user: {avg_time:.1f}s")
        print(f"  Estimated remaining: {remaining/60:.0f} minutes")
except Exception as e:
    print(f"  Error reading checkpoint: {e}")
EOF
else
    echo "📊 Checkpoint not yet created"
fi

echo ""

# Show last 15 lines of log
if [ -f "$LOG_FILE" ]; then
    echo "📝 Last log output:"
    tail -15 "$LOG_FILE"
else
    echo "📝 Log file not yet created"
fi

echo ""
echo "================================"
echo "Commands:"
echo "  watch -n 30 bash scripts/monitor_p8_2.sh  # Auto-refresh every 30s"
echo "  tail -f $LOG_FILE  # Follow log"
echo "================================"
