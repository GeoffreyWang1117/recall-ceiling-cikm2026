#!/bin/bash
# Monitor P0-2 CoT Prompt experiment progress

LOG_FILE="/tmp/claude/-home-coder-gw-Projects-GraphLLMRec/tasks/b838cbd.output"

while true; do
    clear
    echo "=== P0-2 CoT Prompt优化实验监控 ($(date)) ==="
    echo ""

    # Check current strategy being tested
    CURRENT_STRATEGY=$(tail -100 "$LOG_FILE" | grep "测试策略:" | tail -1)
    echo "当前测试: $CURRENT_STRATEGY"
    echo ""

    # Show progress
    tail -20 "$LOG_FILE" | grep -E "(测试:|NDCG@10:|Recall@10:)" | tail -10
    echo ""

    # Check if finished
    if grep -q "实验成功!" "$LOG_FILE" || grep -q "CoT未能提升性能" "$LOG_FILE" || grep -q "部分成功" "$LOG_FILE"; then
        echo "✅ 实验完成!"
        break
    fi

    sleep 30
done

echo ""
echo "显示完整结果..."
tail -60 "$LOG_FILE"
