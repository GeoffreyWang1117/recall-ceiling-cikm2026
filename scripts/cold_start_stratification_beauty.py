"""
Cold-Start Stratification Experiment - Beauty Dataset

Analyze CF performance across different user activity levels:
- Tier 1 (Cold-start): ≤5 historical interactions
- Tier 2 (Medium): 6-20 historical interactions
- Tier 3 (Active): >20 historical interactions

Sample 50 users from each tier (150 total)
Supports checkpoint for interruption/resume
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'experiments' / 'realistic_recall'))

import time
import json
import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, List

from data.amazon_beauty_loader import AmazonBeautyDataset
from models.llm.llm_recommender import LLMRecommender
from utils.metrics import calculate_metrics
from utils.checkpoint_utils import ExperimentCheckpoint, get_pending_items

from cf_recall import CFRecall


# ==================== Configuration ====================
CHECKPOINT_PATH = "experiments/logs/checkpoints/cold_start_beauty_checkpoint.json"
RESULTS_PATH = "experiments/logs/cold_start_stratification_beauty.json"

USERS_PER_TIER = 50
CANDIDATE_SIZE = 100
TOP_K = 10
N_FACTORS = 128

# Tier definitions
TIER_DEFS = {
    'cold_start': {'min': 0, 'max': 5, 'label': 'Cold-start (≤5)'},
    'medium': {'min': 6, 'max': 20, 'label': 'Medium (6-20)'},
    'active': {'min': 21, 'max': float('inf'), 'label': 'Active (>20)'}
}


def build_3step_cot_prompt(
    user_history: List[int],
    candidates: List[int],
    item_texts: Dict[int, str],
    top_k: int = 10
) -> str:
    """Build 3-step CoT prompt"""
    prompt = "Task: Rerank items based on relevance to user preferences using 3-step reasoning.\n\n"

    prompt += "User's interaction history:\n"
    for idx, item_id in enumerate(user_history[-10:], 1):
        text = item_texts.get(item_id, f"Item {item_id}")
        prompt += f"  {idx}. {text}\n"

    prompt += f"\nCandidate items:\n"
    for idx, item_id in enumerate(candidates, 1):
        text = item_texts.get(item_id, f"Item {item_id}")
        prompt += f"  {idx}. Item {item_id}: {text}\n"

    prompt += "\nReasoning Steps:\n"
    prompt += "Step 1 - Understand User Preferences: Analyze the user's history to identify key patterns.\n"
    prompt += "Step 2 - Match Candidates: Compare each candidate against user preferences.\n"
    prompt += "Step 3 - Rank Output: Select and rank the most relevant items.\n\n"

    prompt += f"Now complete the 3 steps and output the top {top_k} items:\n"
    prompt += "Format: One item per line as 'Item <ID>'\n\n"
    prompt += f"Output:\n"

    return prompt


def llm_rerank(
    llm_model: LLMRecommender,
    prompt: str,
    candidates: List[int],
    top_k: int = 10
) -> List[int]:
    """Use LLM to rerank candidates"""
    output = llm_model.recommend(
        prompt=prompt,
        top_k=top_k,
        return_explanation=False,
        temperature=0.1,
        max_new_tokens=200
    )

    # Extract valid item IDs
    reranked_ids = []
    candidates_set = set(candidates)
    for item_id in output['item_ids']:
        if item_id in candidates_set:
            reranked_ids.append(item_id)
        if len(reranked_ids) >= top_k:
            break

    # Fallback
    if len(reranked_ids) < top_k:
        for item_id in candidates:
            if item_id not in reranked_ids:
                reranked_ids.append(item_id)
            if len(reranked_ids) >= top_k:
                break

    return reranked_ids[:top_k]


def process_single_user(sample, cf_recall, llm, dataset):
    """Process a single user with CF+LLM"""
    user_id = sample['user_id']
    history = sample.get('history', [])
    ground_truth = sample.get('ground_truth', [])[:TOP_K]

    if not history or not ground_truth:
        return None

    # CF recall
    cf_candidates, _ = cf_recall.recall(
        user_id=user_id,
        K=CANDIDATE_SIZE,
        exclude_ids=history
    )

    # Recall quality
    recall_quality = len(set(cf_candidates) & set(ground_truth)) / len(ground_truth)

    # LLM reranking
    prompt = build_3step_cot_prompt(history, cf_candidates, dataset.item_texts, TOP_K)
    ranking = llm_rerank(llm, prompt, cf_candidates, TOP_K)

    # Metrics
    metrics = calculate_metrics([ranking], [ground_truth], k_values=[TOP_K])

    return {
        'user_id': user_id,
        'history_len': len(history),
        'recall_quality': recall_quality,
        'ndcg': metrics['ndcg@10'],
        'recall': metrics['recall@10']
    }


def stratify_users(test_samples):
    """Stratify users by history length"""
    stratified = {tier: [] for tier in TIER_DEFS.keys()}

    for sample in test_samples:
        history_len = len(sample.get('history', []))

        for tier, defn in TIER_DEFS.items():
            if defn['min'] <= history_len <= defn['max']:
                stratified[tier].append(sample)
                break

    return stratified


def aggregate_tier_results(checkpoint, tier_name, all_samples):
    """Aggregate results for a specific tier"""
    per_item = checkpoint.get_per_item_results()

    tier_results = []
    for idx_str, result in per_item.items():
        idx = int(idx_str)
        if result and idx < len(all_samples):
            # Check if this user belongs to the tier
            sample = all_samples[idx]
            history_len = len(sample.get('history', []))
            tier_def = TIER_DEFS[tier_name]

            if tier_def['min'] <= history_len <= tier_def['max']:
                tier_results.append(result)

    if not tier_results:
        return {
            'recall_quality': 0,
            'ndcg@10': 0,
            'recall@10': 0,
            'n_users': 0
        }

    return {
        'recall_quality': np.mean([r['recall_quality'] for r in tier_results]),
        'ndcg@10': np.mean([r['ndcg'] for r in tier_results]),
        'recall@10': np.mean([r['recall'] for r in tier_results]),
        'n_users': len(tier_results),
        'recall_quality_scores': [r['recall_quality'] for r in tier_results],
        'ndcg_scores': [r['ndcg'] for r in tier_results],
        'recall_scores': [r['recall'] for r in tier_results]
    }


def main():
    print("=" * 80)
    print("Cold-Start Stratification Experiment - Beauty Dataset")
    print("=" * 80)

    # ==================== Initialize Checkpoint ====================
    checkpoint = ExperimentCheckpoint(CHECKPOINT_PATH, auto_save_interval=5)

    completed_count = len(checkpoint.get_completed_indices())
    if completed_count > 0:
        print(f"\n⏭️  Resuming from checkpoint: {completed_count} users already completed")
    else:
        print(f"\n🆕 Starting fresh experiment")

    print(f"\nConfiguration:")
    print(f"  Dataset: Beauty")
    print(f"  Users per tier: {USERS_PER_TIER}")
    print(f"  Total users: {USERS_PER_TIER * 3}")
    print(f"  Candidate size: K={CANDIDATE_SIZE}")
    print(f"  Checkpoint: {CHECKPOINT_PATH}")

    # ==================== Load Dataset ====================
    print(f"\n[1/5] Loading Beauty dataset...")
    dataset = AmazonBeautyDataset(
        data_path=str(project_root / "data" / "processed" / "amazon_beauty_sampled")
    )

    # ==================== Stratify Users ====================
    print(f"\n[2/5] Stratifying test users...")
    _, _, test_samples = dataset.split()

    stratified = stratify_users(test_samples)

    print(f"\nStratification:")
    for tier, defn in TIER_DEFS.items():
        print(f"  {defn['label']}: {len(stratified[tier])} users available")

    # Sample users from each tier
    np.random.seed(42)
    sampled_users = []

    for tier in ['cold_start', 'medium', 'active']:
        available = stratified[tier]
        n_sample = min(USERS_PER_TIER, len(available))
        sampled = np.random.choice(len(available), size=n_sample, replace=False)
        sampled_users.extend([available[i] for i in sampled])
        print(f"  Sampled {n_sample} from {tier}")

    print(f"\nTotal sampled: {len(sampled_users)} users")

    # Check pending
    pending = get_pending_items(sampled_users, checkpoint)
    print(f"  Pending: {len(pending)} users")
    print(f"  Already completed: {len(checkpoint.get_completed_indices())} users")

    if len(pending) == 0:
        print("\n✓ All users already processed!")
        print("  Computing final results...")
    else:
        # ==================== Build CF Recall ====================
        print(f"\n[3/5] Building CF recall (128 factors)...")
        cf_recall = CFRecall(n_factors=N_FACTORS)
        cf_recall.fit(dataset.train_df)

        print("\n✓ CF recall built")

        # ==================== Load LLM ====================
        print(f"\n[4/5] Loading LLM (Qwen2.5-7B-Instruct)...")

        llm_config = {
            'name': "Qwen/Qwen2.5-7B-Instruct",
            'quantization': '4bit',
            'max_length': 2048,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        llm = LLMRecommender(llm_config)

        print(f"✓ LLM loaded on {llm_config['device']}")

        # ==================== Process Users ====================
        print(f"\n[5/5] Processing users (resumable with Ctrl+C)...")
        print(f"  Progress: {checkpoint.get_progress_str(len(sampled_users))}")

        start_time = time.time()

        try:
            for idx, sample in tqdm(pending, desc="Processing users"):
                result = process_single_user(sample, cf_recall, llm, dataset)
                checkpoint.mark_completed(idx, result)

        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user (Ctrl+C)")
            print(f"  Progress saved: {checkpoint.get_progress_str(len(sampled_users))}")
            print(f"  To resume, simply run the script again")
            return

        elapsed_time = time.time() - start_time
        checkpoint.update_metadata({
            'execution_time': elapsed_time,
            'avg_time_per_user': elapsed_time / len(pending) if len(pending) > 0 else 0
        })

    # ==================== Aggregate Results ====================
    print(f"\nComputing tier-specific results...")

    final_results = {}
    for tier in ['cold_start', 'medium', 'active']:
        tier_result = aggregate_tier_results(checkpoint, tier, sampled_users)
        final_results[tier] = tier_result
        print(f"  {TIER_DEFS[tier]['label']}: {tier_result['n_users']} users")

    # ==================== Display Results ====================
    print("\n" + "=" * 80)
    print("Cold-Start Stratification Results")
    print("=" * 80)
    print()
    print("| Tier             | N Users | Recall@100 | NDCG@10 | Recall@10 |")
    print("|------------------|---------|------------|---------|-----------|")

    for tier in ['cold_start', 'medium', 'active']:
        result = final_results[tier]
        label = TIER_DEFS[tier]['label']
        print(f"| {label:16} | {result['n_users']:7} | "
              f"{result['recall_quality']:10.4f} | "
              f"{result['ndcg@10']:7.4f} | "
              f"{result['recall@10']:9.4f} |")

    metadata = checkpoint.get_metadata()
    if 'execution_time' in metadata:
        print(f"\nExecution time: {metadata['execution_time']/60:.1f} minutes")
        print(f"Avg time per user: {metadata['avg_time_per_user']:.1f} seconds")

    # ==================== Save Results ====================
    output = {
        'config': {
            'dataset': 'Beauty',
            'users_per_tier': USERS_PER_TIER,
            'total_users': len(sampled_users),
            'candidate_size': CANDIDATE_SIZE,
            'top_k': TOP_K,
            'tier_definitions': TIER_DEFS
        },
        'results': final_results,
        'execution_time': metadata.get('execution_time', 0)
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n✓ Results saved to: {RESULTS_PATH}")

    # Finalize checkpoint
    checkpoint.finalize(final_results)

    print("\n" + "=" * 80)
    print("✓ Experiment Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
