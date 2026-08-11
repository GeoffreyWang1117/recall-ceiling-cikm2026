"""
Realistic vs Oracle Setting Experiment - Electronics Dataset
Using CF Recall (NO TEXT REQUIRED)

This version uses Collaborative Filtering for recall instead of content-based,
solving the issue of poor item text quality in the Electronics dataset.
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
import pandas as pd

from data.amazon_beauty_loader import AmazonBeautyDataset
from models.llm.llm_recommender import LLMRecommender
from utils.metrics import calculate_metrics

from cf_recall import CFRecall


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
        text = item_texts.get(item_id)
        if text is None:
            text = f"Item {item_id}"
        prompt += f"  {idx}. {text}\n"

    prompt += f"\nCandidate items:\n"
    for idx, item_id in enumerate(candidates, 1):
        text = item_texts.get(item_id)
        if text is None:
            text = f"Item {item_id}"
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


def main():
    print("="*80)
    print("Realistic vs Oracle Experiment - Electronics Dataset (CF Recall)")
    print("="*80)

    # Configuration
    DATASET = 'Electronics'
    N_TEST_USERS = 50
    CANDIDATE_SIZE = 100
    TOP_K = 10
    N_FACTORS = 128  # CF latent factors (increased from 64)

    print(f"\nConfiguration:")
    print(f"  Dataset: {DATASET}")
    print(f"  Test users: {N_TEST_USERS}")
    print(f"  Candidate size: K={CANDIDATE_SIZE}")
    print(f"  Recall method: Collaborative Filtering (CF)")
    print(f"  CF factors: {N_FACTORS} (optimized)")

    # ==================== Load Dataset ====================
    print(f"\n[1/6] Loading {DATASET} dataset...")

    dataset = AmazonBeautyDataset(
        data_path=str(project_root / "data" / "processed" / "amazon_electronics_sampled")
    )

    train_samples, val_samples, test_samples = dataset.split()

    print(f"✓ Dataset loaded:")
    print(f"  Train: {len(train_samples)}")
    print(f"  Val: {len(val_samples)}")
    print(f"  Test: {len(test_samples)}")
    print(f"  Items with text: {len(dataset.item_texts)}")

    # ==================== Data Quality Checks ====================
    print(f"\n[2/6] Data quality checks...")

    # Filter valid test samples (with history)
    valid_test_samples = [s for s in test_samples if len(s.get('history', [])) > 0]
    valid_user_ids = sorted(set(s['user_id'] for s in valid_test_samples))

    print(f"  Test user stats:")
    print(f"    Total test users: {len(set(s['user_id'] for s in test_samples))}")
    print(f"    Valid users (with history): {len(valid_user_ids)}")
    print(f"    Cold-start ratio: {1 - len(valid_user_ids)/len(test_samples):.1%}")

    # Sample test users
    print(f"\n  Sampling {N_TEST_USERS} test users...")
    np.random.seed(42)

    if len(valid_user_ids) < N_TEST_USERS:
        print(f"  ⚠️  WARNING: Only {len(valid_user_ids)} valid users available (requested {N_TEST_USERS})")
        N_TEST_USERS = len(valid_user_ids)

    sampled_user_ids = np.random.choice(
        valid_user_ids,
        size=N_TEST_USERS,
        replace=False
    )

    # Build sampled test set
    sampled_test = []
    for user_id in sampled_user_ids:
        user_samples = [s for s in valid_test_samples if s['user_id'] == user_id]
        if user_samples:
            sampled_test.append(user_samples[0])

    print(f"  Sampled: {len(sampled_test)} test users")

    # Verify sample quality
    sample = sampled_test[0]
    print(f"\n  Sample verification (User {sample['user_id']}):")
    print(f"    History: {len(sample['history'])} items")
    print(f"    Ground truth: {len(sample['ground_truth'])} items")

    if len(sample['history']) == 0:
        raise ValueError("ERROR: Sample user has no history! Data loading failed.")

    print(f"\n✓ Data quality checks passed")

    # ==================== Build CF Recall ====================
    print(f"\n[3/6] Building CF recall...")

    cf_recall = CFRecall(n_factors=N_FACTORS)
    cf_recall.fit(dataset.train_df)

    # Test recall quality on first sample
    print(f"\n  Testing recall quality on sample user...")
    test_candidates, test_scores = cf_recall.recall(
        user_id=sample['user_id'],
        K=20,
        exclude_ids=sample['history']
    )

    test_recall_hits = len(set(test_candidates) & set(sample['ground_truth']))
    test_recall_quality = test_recall_hits / len(sample['ground_truth']) if sample['ground_truth'] else 0

    print(f"    Recall@20 quality: {test_recall_hits}/{len(sample['ground_truth'])} = {test_recall_quality:.1%}")

    if test_recall_quality == 0:
        print(f"    ⚠️  WARNING: Zero recall quality detected on sample!")
        print(f"    This suggests CF may not work well on this user")
        print(f"    Continuing anyway...")

    print(f"\n✓ CF recall built")

    # ==================== Load LLM ====================
    print(f"\n[4/6] Loading LLM (Qwen2.5-7B-Instruct)...")

    llm_config = {
        'name': "Qwen/Qwen2.5-7B-Instruct",
        'quantization': '4bit',
        'max_length': 2048,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    llm = LLMRecommender(llm_config)

    print(f"✓ LLM loaded on {llm_config['device']}")

    # ==================== Run Experiments ====================
    print(f"\n[5/6] Running experiments...")

    results = {
        'oracle': {'ndcg_scores': [], 'recall_scores': []},
        'realistic': {'ndcg_scores': [], 'recall_scores': [], 'recall_quality': []}
    }

    start_time = time.time()

    for sample in tqdm(sampled_test, desc="Processing users"):
        user_id = sample['user_id']
        history = sample.get('history', [])
        ground_truth = sample.get('ground_truth', [])

        if not history or not ground_truth:
            continue

        # Limit ground truth to TOP_K for fair evaluation
        ground_truth = ground_truth[:TOP_K]

        # -------------------- Oracle Setting --------------------
        # Get candidates from CF recall (K - |GT|), then inject GT
        cf_candidates, _ = cf_recall.recall(
            user_id=user_id,
            K=CANDIDATE_SIZE - len(ground_truth),
            exclude_ids=history
        )

        oracle_candidates = list(set(cf_candidates + ground_truth))[:CANDIDATE_SIZE]

        # Build prompt
        prompt_oracle = build_3step_cot_prompt(history, oracle_candidates, dataset.item_texts, TOP_K)

        # LLM rerank
        oracle_ranking = llm_rerank(llm, prompt_oracle, oracle_candidates, TOP_K)

        # Compute metrics
        oracle_metrics = calculate_metrics([oracle_ranking], [ground_truth], k_values=[TOP_K])
        results['oracle']['ndcg_scores'].append(oracle_metrics['ndcg@10'])
        results['oracle']['recall_scores'].append(oracle_metrics['recall@10'])

        # -------------------- Realistic Setting --------------------
        # Get candidates from CF recall only (no GT injection)
        realistic_candidates, _ = cf_recall.recall(
            user_id=user_id,
            K=CANDIDATE_SIZE,
            exclude_ids=history
        )

        # Check recall quality
        recall_quality = len(set(realistic_candidates) & set(ground_truth)) / len(ground_truth)
        results['realistic']['recall_quality'].append(recall_quality)

        # Build prompt
        prompt_realistic = build_3step_cot_prompt(history, realistic_candidates, dataset.item_texts, TOP_K)

        # LLM rerank
        realistic_ranking = llm_rerank(llm, prompt_realistic, realistic_candidates, TOP_K)

        # Compute metrics
        realistic_metrics = calculate_metrics([realistic_ranking], [ground_truth], k_values=[TOP_K])
        results['realistic']['ndcg_scores'].append(realistic_metrics['ndcg@10'])
        results['realistic']['recall_scores'].append(realistic_metrics['recall@10'])

    elapsed_time = time.time() - start_time

    # ==================== Summary ====================
    print(f"\n[6/6] Computing summary...")

    oracle_ndcg = np.mean(results['oracle']['ndcg_scores'])
    oracle_recall = np.mean(results['oracle']['recall_scores'])

    realistic_ndcg = np.mean(results['realistic']['ndcg_scores'])
    realistic_recall = np.mean(results['realistic']['recall_scores'])
    realistic_recall_quality = np.mean(results['realistic']['recall_quality'])

    print(f"\n{'='*80}")
    print("Results Summary")
    print(f"{'='*80}\n")

    print(f"| Setting | Recall@{CANDIDATE_SIZE} | NDCG@{TOP_K} | Recall@{TOP_K} |")
    print(f"|---------|-------------|----------|------------|")
    print(f"| Oracle (GT injected) | 1.0000 | {oracle_ndcg:.4f} | {oracle_recall:.4f} |")
    print(f"| Realistic (CF)       | {realistic_recall_quality:.4f} | {realistic_ndcg:.4f} | {realistic_recall:.4f} |")

    gap = (oracle_ndcg - realistic_ndcg) / oracle_ndcg * 100 if oracle_ndcg > 0 else 0
    print(f"\nPerformance Gap: {gap:.1f}% NDCG drop from oracle to realistic")

    print(f"\nExecution time: {elapsed_time/60:.1f} minutes")
    print(f"Avg time per user: {elapsed_time/len(sampled_test):.1f} seconds")

    # Save results
    output_path = f'/home/coder-gw/Projects/GraphLLMRec/experiments/logs/realistic_vs_oracle_electronics_cf_128f.json'
    with open(output_path, 'w') as f:
        json.dump({
            'config': {
                'dataset': DATASET,
                'n_test_users': N_TEST_USERS,
                'candidate_size': CANDIDATE_SIZE,
                'top_k': TOP_K,
                'recall_method': 'CF',
                'cf_n_factors': N_FACTORS
            },
            'results': {
                'oracle': {
                    'ndcg@10': oracle_ndcg,
                    'recall@10': oracle_recall,
                    'ndcg_scores': results['oracle']['ndcg_scores'],
                    'recall_scores': results['oracle']['recall_scores']
                },
                'realistic': {
                    'ndcg@10': realistic_ndcg,
                    'recall@10': realistic_recall,
                    'recall_quality': realistic_recall_quality,
                    'ndcg_scores': results['realistic']['ndcg_scores'],
                    'recall_scores': results['realistic']['recall_scores'],
                    'recall_quality_per_user': results['realistic']['recall_quality']
                }
            },
            'execution_time': elapsed_time
        }, f, indent=2)

    print(f"\n✓ Results saved to: {output_path}")
    print(f"\n{'='*80}")
    print("✓ Experiment Complete!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
