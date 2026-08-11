"""
Large-Scale Experiment - Movies Dataset (200 Users)

Scale up to 200 users with multiple random seeds for robust evaluation.
Uses checkpoint for safe interruption/resumption.
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
CHECKPOINT_PATH = "experiments/logs/checkpoints/large_scale_movies_checkpoint.json"
RESULTS_PATH = "experiments/logs/large_scale_movies_200users.json"

N_TEST_USERS = 200
CANDIDATE_SIZE = 100
TOP_K = 10
N_FACTORS = 128
RANDOM_SEED = 2025  # Different from previous experiments (42)


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
    """Process a single user (Realistic + Oracle)"""
    user_id = sample['user_id']
    history = sample.get('history', [])
    ground_truth = sample.get('ground_truth', [])[:TOP_K]

    if not history or not ground_truth:
        return None

    result = {}

    # -------------------- Realistic (CF Recall) --------------------
    cf_candidates, _ = cf_recall.recall(
        user_id=user_id,
        K=CANDIDATE_SIZE,
        exclude_ids=history
    )

    recall_quality = len(set(cf_candidates) & set(ground_truth)) / len(ground_truth)

    prompt_cf = build_3step_cot_prompt(history, cf_candidates, dataset.item_texts, TOP_K)
    cf_ranking = llm_rerank(llm, prompt_cf, cf_candidates, TOP_K)

    cf_metrics = calculate_metrics([cf_ranking], [ground_truth], k_values=[TOP_K])

    result['realistic'] = {
        'recall_quality': recall_quality,
        'ndcg': cf_metrics['ndcg@10'],
        'recall': cf_metrics['recall@10']
    }

    # -------------------- Oracle (GT Injected) --------------------
    oracle_candidates = list(ground_truth) + [
        c for c in cf_candidates if c not in ground_truth
    ][:CANDIDATE_SIZE]

    prompt_oracle = build_3step_cot_prompt(history, oracle_candidates, dataset.item_texts, TOP_K)
    oracle_ranking = llm_rerank(llm, prompt_oracle, oracle_candidates, TOP_K)

    oracle_metrics = calculate_metrics([oracle_ranking], [ground_truth], k_values=[TOP_K])

    result['oracle'] = {
        'ndcg': oracle_metrics['ndcg@10'],
        'recall': oracle_metrics['recall@10']
    }

    return result


def aggregate_results(checkpoint, all_samples):
    """Aggregate results from all completed users"""
    per_item = checkpoint.get_per_item_results()

    aggregated = {
        'realistic': {'recall_quality': [], 'ndcg_scores': [], 'recall_scores': []},
        'oracle': {'ndcg_scores': [], 'recall_scores': []}
    }

    for idx_str, result in per_item.items():
        if result:
            aggregated['realistic']['recall_quality'].append(result['realistic']['recall_quality'])
            aggregated['realistic']['ndcg_scores'].append(result['realistic']['ndcg'])
            aggregated['realistic']['recall_scores'].append(result['realistic']['recall'])

            aggregated['oracle']['ndcg_scores'].append(result['oracle']['ndcg'])
            aggregated['oracle']['recall_scores'].append(result['oracle']['recall'])

    # Compute means
    final_results = {
        'oracle': {
            'recall@100': 1.0,  # By definition
            'ndcg@10': np.mean(aggregated['oracle']['ndcg_scores']) if aggregated['oracle']['ndcg_scores'] else 0,
            'recall@10': np.mean(aggregated['oracle']['recall_scores']) if aggregated['oracle']['recall_scores'] else 0,
            'ndcg_scores': aggregated['oracle']['ndcg_scores'],
            'recall_scores': aggregated['oracle']['recall_scores']
        },
        'realistic': {
            'recall@100': np.mean(aggregated['realistic']['recall_quality']) if aggregated['realistic']['recall_quality'] else 0,
            'ndcg@10': np.mean(aggregated['realistic']['ndcg_scores']) if aggregated['realistic']['ndcg_scores'] else 0,
            'recall@10': np.mean(aggregated['realistic']['recall_scores']) if aggregated['realistic']['recall_scores'] else 0,
            'recall_quality_per_user': aggregated['realistic']['recall_quality'],
            'ndcg_scores': aggregated['realistic']['ndcg_scores'],
            'recall_scores': aggregated['realistic']['recall_scores']
        }
    }

    return final_results


def main():
    print("=" * 80)
    print("Large-Scale Experiment - Movies Dataset (200 Users)")
    print("=" * 80)

    # ==================== Initialize Checkpoint ====================
    checkpoint = ExperimentCheckpoint(CHECKPOINT_PATH, auto_save_interval=10)

    completed_count = len(checkpoint.get_completed_indices())
    if completed_count > 0:
        print(f"\n⏭️  Resuming from checkpoint: {completed_count} users already completed")
    else:
        print(f"\n🆕 Starting fresh experiment")

    print(f"\nConfiguration:")
    print(f"  Dataset: Movies")
    print(f"  Test users: {N_TEST_USERS}")
    print(f"  Random seed: {RANDOM_SEED}")
    print(f"  Candidate size: K={CANDIDATE_SIZE}")
    print(f"  Checkpoint: {CHECKPOINT_PATH}")

    # ==================== Load Dataset ====================
    print(f"\n[1/5] Loading Movies dataset...")
    dataset = AmazonBeautyDataset(
        data_path=str(project_root / "data" / "processed" / "amazon_movies_sampled")
    )

    # ==================== Sample Test Users ====================
    print(f"\n[2/5] Sampling test users...")
    _, _, test_samples = dataset.split()

    valid_test_samples = [s for s in test_samples if len(s.get('history', [])) > 0]
    valid_user_ids = sorted(set(s['user_id'] for s in valid_test_samples))

    np.random.seed(RANDOM_SEED)
    sampled_user_ids = np.random.choice(
        valid_user_ids,
        size=min(N_TEST_USERS, len(valid_user_ids)),
        replace=False
    )

    sampled_test = []
    for user_id in sampled_user_ids:
        user_samples = [s for s in valid_test_samples if s['user_id'] == user_id]
        if user_samples:
            sampled_test.append(user_samples[0])

    print(f"  Total: {len(sampled_test)} test users")

    # Get pending
    pending = get_pending_items(sampled_test, checkpoint)
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
        print(f"  Progress: {checkpoint.get_progress_str(len(sampled_test))}")

        start_time = time.time()

        try:
            for idx, sample in tqdm(pending, desc="Processing users"):
                result = process_single_user(sample, cf_recall, llm, dataset)
                checkpoint.mark_completed(idx, result)

        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user (Ctrl+C)")
            print(f"  Progress saved: {checkpoint.get_progress_str(len(sampled_test))}")
            print(f"  To resume, simply run the script again")
            return

        elapsed_time = time.time() - start_time
        checkpoint.update_metadata({
            'execution_time': elapsed_time,
            'avg_time_per_user': elapsed_time / len(pending) if len(pending) > 0 else 0
        })

    # ==================== Aggregate Results ====================
    print(f"\nComputing summary...")
    final_results = aggregate_results(checkpoint, sampled_test)

    # Compute performance gap
    oracle_ndcg = final_results['oracle']['ndcg@10']
    realistic_ndcg = final_results['realistic']['ndcg@10']
    gap = 100 * (oracle_ndcg - realistic_ndcg) / oracle_ndcg if oracle_ndcg > 0 else 0

    # ==================== Display Results ====================
    print("\n" + "=" * 80)
    print("Results Summary")
    print("=" * 80)
    print()
    print("| Setting | Recall@100 | NDCG@10 | Recall@10 |")
    print("|---------|-------------|----------|------------|")
    print(f"| Oracle (GT injected) | {final_results['oracle']['recall@100']:.4f} | "
          f"{oracle_ndcg:.4f} | {final_results['oracle']['recall@10']:.4f} |")
    print(f"| Realistic (CF)       | {final_results['realistic']['recall@100']:.4f} | "
          f"{realistic_ndcg:.4f} | {final_results['realistic']['recall@10']:.4f} |")

    print(f"\nPerformance Gap: {gap:.1f}% NDCG drop from oracle to realistic")

    metadata = checkpoint.get_metadata()
    if 'execution_time' in metadata:
        print(f"\nExecution time: {metadata['execution_time']/60:.1f} minutes")
        print(f"Avg time per user: {metadata['avg_time_per_user']:.1f} seconds")

    # ==================== Save Results ====================
    output = {
        'config': {
            'dataset': 'Movies',
            'n_test_users': N_TEST_USERS,
            'random_seed': RANDOM_SEED,
            'candidate_size': CANDIDATE_SIZE,
            'top_k': TOP_K,
            'cf_n_factors': N_FACTORS
        },
        'results': final_results,
        'performance_gap_pct': gap,
        'execution_time': metadata.get('execution_time', 0)
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Results saved to: {RESULTS_PATH}")

    # Finalize checkpoint
    checkpoint.finalize(final_results)

    print("\n" + "=" * 80)
    print("✓ Experiment Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
