"""
Prompt Strategy Comparison - Beauty Dataset

Compare different prompt designs:
- P1 (Current): 3-step CoT with detailed instructions
- P2 (Simplified): Direct ranking without CoT
- P3 (Enhanced): 3-step CoT + explicit preference signals

Sample: 50 users from Beauty dataset
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
CHECKPOINT_PATH = "experiments/logs/checkpoints/prompt_strategy_beauty_checkpoint.json"
RESULTS_PATH = "experiments/logs/prompt_strategy_comparison_beauty.json"

N_TEST_USERS = 50
CANDIDATE_SIZE = 100
TOP_K = 10
N_FACTORS = 128


def build_prompt_p1_cot(
    user_history: List[int],
    candidates: List[int],
    item_texts: Dict[int, str],
    top_k: int = 10
) -> str:
    """
    P1: Current 3-step CoT prompt (Baseline)
    """
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


def build_prompt_p2_direct(
    user_history: List[int],
    candidates: List[int],
    item_texts: Dict[int, str],
    top_k: int = 10
) -> str:
    """
    P2: Simplified direct ranking (No CoT)
    """
    prompt = "Task: Rank items by relevance to the user's preferences.\n\n"

    prompt += "User's interaction history:\n"
    for idx, item_id in enumerate(user_history[-10:], 1):
        text = item_texts.get(item_id, f"Item {item_id}")
        prompt += f"  {idx}. {text}\n"

    prompt += f"\nCandidate items to rank:\n"
    for idx, item_id in enumerate(candidates, 1):
        text = item_texts.get(item_id, f"Item {item_id}")
        prompt += f"  {idx}. Item {item_id}: {text}\n"

    prompt += f"\nOutput the top {top_k} most relevant items:\n"
    prompt += "Format: One item per line as 'Item <ID>'\n\n"
    prompt += f"Output:\n"

    return prompt


def build_prompt_p3_enhanced(
    user_history: List[int],
    candidates: List[int],
    item_texts: Dict[int, str],
    top_k: int = 10
) -> str:
    """
    P3: Enhanced CoT with explicit preference signals
    """
    prompt = "Task: Recommend items based on user preferences using detailed 3-step reasoning.\n\n"

    prompt += "User's interaction history (most recent items):\n"
    for idx, item_id in enumerate(user_history[-10:], 1):
        text = item_texts.get(item_id, f"Item {item_id}")
        prompt += f"  {idx}. {text}\n"

    prompt += f"\nCandidate items:\n"
    for idx, item_id in enumerate(candidates, 1):
        text = item_texts.get(item_id, f"Item {item_id}")
        prompt += f"  {idx}. Item {item_id}: {text}\n"

    prompt += "\nReasoning Steps:\n"
    prompt += "Step 1 - Extract User Preferences:\n"
    prompt += "  - Identify product categories, brands, and features the user likes\n"
    prompt += "  - Note any patterns in purchase behavior\n"
    prompt += "\nStep 2 - Evaluate Candidates:\n"
    prompt += "  - For each candidate, assess similarity to user's preferred items\n"
    prompt += "  - Consider product attributes, categories, and relevance\n"
    prompt += "\nStep 3 - Generate Ranking:\n"
    prompt += "  - Rank candidates by relevance score (highest to lowest)\n"
    prompt += f"  - Select top {top_k} items for recommendation\n\n"

    prompt += f"Now complete all 3 steps and output the top {top_k} items:\n"
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
        max_new_tokens=300  # Increased for P3
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
    """Process a single user with all 3 prompt strategies"""
    user_id = sample['user_id']
    history = sample.get('history', [])
    ground_truth = sample.get('ground_truth', [])[:TOP_K]

    if not history or not ground_truth:
        return None

    # Get CF candidates (same for all prompts)
    cf_candidates, _ = cf_recall.recall(
        user_id=user_id,
        K=CANDIDATE_SIZE,
        exclude_ids=history
    )

    recall_quality = len(set(cf_candidates) & set(ground_truth)) / len(ground_truth)

    result = {'recall_quality': recall_quality}

    # -------------------- P1: Current 3-step CoT --------------------
    prompt_p1 = build_prompt_p1_cot(history, cf_candidates, dataset.item_texts, TOP_K)
    ranking_p1 = llm_rerank(llm, prompt_p1, cf_candidates, TOP_K)
    metrics_p1 = calculate_metrics([ranking_p1], [ground_truth], k_values=[TOP_K])

    result['p1_cot'] = {
        'ndcg': metrics_p1['ndcg@10'],
        'recall': metrics_p1['recall@10']
    }

    # -------------------- P2: Simplified Direct --------------------
    prompt_p2 = build_prompt_p2_direct(history, cf_candidates, dataset.item_texts, TOP_K)
    ranking_p2 = llm_rerank(llm, prompt_p2, cf_candidates, TOP_K)
    metrics_p2 = calculate_metrics([ranking_p2], [ground_truth], k_values=[TOP_K])

    result['p2_direct'] = {
        'ndcg': metrics_p2['ndcg@10'],
        'recall': metrics_p2['recall@10']
    }

    # -------------------- P3: Enhanced CoT --------------------
    prompt_p3 = build_prompt_p3_enhanced(history, cf_candidates, dataset.item_texts, TOP_K)
    ranking_p3 = llm_rerank(llm, prompt_p3, cf_candidates, TOP_K)
    metrics_p3 = calculate_metrics([ranking_p3], [ground_truth], k_values=[TOP_K])

    result['p3_enhanced'] = {
        'ndcg': metrics_p3['ndcg@10'],
        'recall': metrics_p3['recall@10']
    }

    return result


def aggregate_results(checkpoint, all_samples):
    """Aggregate results from all completed users"""
    per_item = checkpoint.get_per_item_results()

    aggregated = {
        'recall_quality': [],
        'p1_cot': {'ndcg_scores': [], 'recall_scores': []},
        'p2_direct': {'ndcg_scores': [], 'recall_scores': []},
        'p3_enhanced': {'ndcg_scores': [], 'recall_scores': []}
    }

    for idx_str, result in per_item.items():
        if result:
            aggregated['recall_quality'].append(result['recall_quality'])

            for strategy in ['p1_cot', 'p2_direct', 'p3_enhanced']:
                aggregated[strategy]['ndcg_scores'].append(result[strategy]['ndcg'])
                aggregated[strategy]['recall_scores'].append(result[strategy]['recall'])

    # Compute means
    final_results = {}
    for strategy in ['p1_cot', 'p2_direct', 'p3_enhanced']:
        final_results[strategy] = {
            'recall@100': np.mean(aggregated['recall_quality']) if aggregated['recall_quality'] else 0,
            'ndcg@10': np.mean(aggregated[strategy]['ndcg_scores']) if aggregated[strategy]['ndcg_scores'] else 0,
            'recall@10': np.mean(aggregated[strategy]['recall_scores']) if aggregated[strategy]['recall_scores'] else 0,
            'ndcg_scores': aggregated[strategy]['ndcg_scores'],
            'recall_scores': aggregated[strategy]['recall_scores']
        }

    return final_results


def main():
    print("=" * 80)
    print("Prompt Strategy Comparison - Beauty Dataset")
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
    print(f"  Test users: {N_TEST_USERS}")
    print(f"  Prompt strategies: 3 (P1-CoT, P2-Direct, P3-Enhanced)")
    print(f"  Checkpoint: {CHECKPOINT_PATH}")

    # ==================== Load Dataset ====================
    print(f"\n[1/5] Loading Beauty dataset...")
    dataset = AmazonBeautyDataset(
        data_path=str(project_root / "data" / "processed" / "amazon_beauty_sampled")
    )

    # ==================== Sample Test Users ====================
    print(f"\n[2/5] Sampling test users...")
    _, _, test_samples = dataset.split()

    valid_test_samples = [s for s in test_samples if len(s.get('history', [])) > 0]
    valid_user_ids = sorted(set(s['user_id'] for s in valid_test_samples))

    np.random.seed(42)
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

    # ==================== Display Results ====================
    print("\n" + "=" * 80)
    print("Prompt Strategy Comparison Results")
    print("=" * 80)
    print()
    print("| Strategy          | Recall@100 | NDCG@10 | Recall@10 |")
    print("|-------------------|------------|---------|-----------|")

    strategies = {
        'p1_cot': 'P1 (3-step CoT)',
        'p2_direct': 'P2 (Direct)',
        'p3_enhanced': 'P3 (Enhanced CoT)'
    }

    for key, label in strategies.items():
        result = final_results[key]
        print(f"| {label:17} | {result['recall@100']:10.4f} | "
              f"{result['ndcg@10']:7.4f} | {result['recall@10']:9.4f} |")

    metadata = checkpoint.get_metadata()
    if 'execution_time' in metadata:
        print(f"\nExecution time: {metadata['execution_time']/60:.1f} minutes")
        print(f"Avg time per user: {metadata['avg_time_per_user']:.1f} seconds")

    # ==================== Save Results ====================
    output = {
        'config': {
            'dataset': 'Beauty',
            'n_test_users': N_TEST_USERS,
            'candidate_size': CANDIDATE_SIZE,
            'top_k': TOP_K
        },
        'results': final_results,
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
