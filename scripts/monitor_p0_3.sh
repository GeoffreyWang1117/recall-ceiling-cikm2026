#!/bin/bash
# Monitor P0-3 experiment progress

LOG_FILE="/tmp/claude/-home-coder-gw-Projects-GraphLLMRec/tasks/bfc186b.output"

while true; do
    clear
    echo "=== P0-3 实验监控 ($(date)) ==="
    echo ""

    # Check current K value being tested
    CURRENT_K=$(tail -100 "$LOG_FILE" | grep "测试候选集大小: K =" | tail -1)
    echo "当前测试: $CURRENT_K"
    echo ""

    # Show progress bar
    tail -10 "$LOG_FILE" | grep "K=" | tail -1
    echo ""

    # Show completed results
    echo "已完成的测试结果:"
    grep "NDCG@10:" "$LOG_FILE" | tail -5
    echo ""

    # Check if finished
    if grep -q "验证成功!" "$LOG_FILE" || grep -q "结果与理论预期不符" "$LOG_FILE"; then
        echo "✅ 实验完成!"
        break
    fi

    sleep 30
done

echo ""
echo "显示完整结果..."
tail -50 "$LOG_FILE"
