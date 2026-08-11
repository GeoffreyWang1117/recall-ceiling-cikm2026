#!/bin/bash

# Main experiment runner script for GraphLLMRec
# Runs complete experimental pipeline for paper

set -e  # Exit on error

echo "=========================================="
echo "GraphLLMRec Experiment Suite"
echo "=========================================="

# Configuration
DATASET="amazon"
DOMAIN="All_Beauty"
DATA_DIR="data/processed/amazon/${DOMAIN}"
NUM_GPUS=2

# Check if data exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Processed data not found at $DATA_DIR"
    echo "Please run data preparation first:"
    echo "  python scripts/prepare_data.py --dataset amazon --domain $DOMAIN ..."
    exit 1
fi

# Export API key (make sure it's set!)
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY not set. OpenAI features will be disabled."
fi

echo ""
echo "=========================================="
echo "Phase 1: Baseline Experiments"
echo "=========================================="

echo ""
echo "Running LightGCN baseline..."
python scripts/train.py \
    --config experiments/configs/amazon_lightgcn.yaml \
    --dataset amazon \
    --exp_name amazon_beauty_lightgcn_baseline

echo ""
echo "Running LLM-only baseline..."
python scripts/train.py \
    --config experiments/configs/amazon_llm_only.yaml \
    --dataset amazon \
    --exp_name amazon_beauty_llm_only

echo ""
echo "=========================================="
echo "Phase 2: Main Method - Graph-as-Expert"
echo "=========================================="

echo ""
echo "Running Graph-as-Expert MoE (Learned Routing)..."
torchrun --nproc_per_node=$NUM_GPUS scripts/train.py \
    --config experiments/configs/amazon_graph_as_expert.yaml \
    --dataset amazon \
    --exp_name amazon_beauty_graph_as_expert_learned

echo ""
echo "Running Graph-as-Expert MoE (Rule-based Routing)..."
# Modify config for rule-based routing
python scripts/train.py \
    --config experiments/configs/amazon_graph_as_expert.yaml \
    --dataset amazon \
    --exp_name amazon_beauty_graph_as_expert_rulebased \
    --routing_strategy rule_based

echo ""
echo "=========================================="
echo "Phase 3: Ablation Studies"
echo "=========================================="

echo ""
echo "Ablation 1: Graph-as-Context only..."
# TODO: Add ablation config

echo ""
echo "Ablation 2: Graph-as-Embedding only..."
# TODO: Add ablation config

echo ""
echo "Ablation 3: Different k-hop depths..."
for k in 1 2 3; do
    echo "Testing k-hop=$k..."
    # TODO: Modify config and run
done

echo ""
echo "=========================================="
echo "Phase 4: Extended Training for Grokking"
echo "=========================================="

echo ""
echo "Running extended training (200 epochs) for grokking observation..."
python scripts/train.py \
    --config experiments/configs/amazon_graph_as_expert.yaml \
    --dataset amazon \
    --exp_name amazon_beauty_grokking_longrun \
    --max_epochs 200

echo ""
echo "=========================================="
echo "Phase 5: Cross-domain Transfer"
echo "=========================================="

echo ""
echo "Training on All_Beauty, testing on Electronics..."
# TODO: Implement cross-domain evaluation

echo ""
echo "=========================================="
echo "All experiments complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - Logs: experiments/logs/"
echo "  - Checkpoints: experiments/checkpoints/"
echo "  - Wandb: https://wandb.ai/your-entity/GraphLLMRec"
echo ""
echo "Next steps:"
echo "  1. Review grokking curves: experiments/checkpoints/*/grokking_curves.png"
echo "  2. Review emergence curves: experiments/checkpoints/*/emergence_curves.png"
echo "  3. Read summary reports: experiments/checkpoints/*/*.txt"
echo "  4. Analyze on wandb dashboard"
echo ""
