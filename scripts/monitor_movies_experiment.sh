#!/bin/bash
# Monitor Movies CF experiment and report when complete

OUTPUT_FILE="/tmp/claude/-home-coder-gw-Projects-GraphLLMRec/tasks/bd068fb.output"

echo "Monitoring Movies CF Recall experiment..."
echo ""

while true; do
    if grep -q "✓ Experiment Complete!" "$OUTPUT_FILE" 2>/dev/null; then
        echo "EXPERIMENT COMPLETED!"
        echo ""
        tail -30 "$OUTPUT_FILE"
        break
    fi
    
    CURRENT_LINE=$(tail -1 "$OUTPUT_FILE" 2>/dev/null | grep "Processing users")
    if [ -n "$CURRENT_LINE" ]; then
        echo -ne "\r$CURRENT_LINE                    "
    fi
    
    sleep 30
done
