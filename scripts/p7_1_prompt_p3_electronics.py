"""
Phase 7-1: Cross-dataset Prompt Validation - Electronics Dataset

Validate P3 Enhanced CoT effectiveness on Electronics dataset.
Compare P1 (baseline CoT) vs P3 (Enhanced CoT) to verify:
- P3 provides +200-400% NDCG@10 improvement (like Beauty)
- Enhanced preference signals help even with low recall (~2%)
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

from data.amazon_beauty_loader import AmazonBeautyDataset  # Reuse for all Amazon datasets
from models.llm.llm_recommender import LLMRecommender
from utils.metrics import calculate_metrics
from utils.checkpoint_utils import ExperimentCheckpoint, get_pending_items

from cf_recall import CFRecall


# ==================== Configuration ====================
CHECKPOINT_PATH = "experiments/logs/checkpoints/p7_1_prompt_electronics_checkpoint.json"
RESULTS_PATH = "experiments/logs/p7_1_prompt_p3_electronics.json"

N_TEST_USERS = 50
CANDIDATE_SIZE = 100
TOP_K = 10
N_FACTORS = 128
RANDOM_SEED = 42


def build_prompt_p1_cot(
    user_history: List[int],
    candidates: List[int],
    item_texts: Dict[int, str],
    top_k: int = 10
) -> str:
    """P1: Current 3-step CoT prompt (Baseline)"""
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


def build_prompt_p3_enhanced(
    user_history: List[int],
    candidates: List[int],
    item_texts: Dict[int, str],
    top_k: int = 10
) -> str:
    """P3: Enhanced CoT with explicit preference signals"""
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
    prompt += "  - Consider product specifications, categories, and relevance\n"
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
        max_new_tokens=300
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
    """Process a single user with P1 and P3 prompts"""
    user_id = sample['user_id']
    history = sample.get('history', [])
    ground_truth = sample.get('ground_truth', [])[:TOP_K]

    if not history or not ground_truth:
        return None

    # Get CF candidates (same for both prompts)
    cf_candidates, _ = cf_recall.recall(
        user_id=user_id,
        K=CANDIDATE_SIZE,
        exclude_ids=history
    )

    recall_quality = len(set(cf_candidates) & set(ground_truth)) / len(ground_truth)

    result = {'recall_quality': recall_quality}

    # -------------------- P1: Baseline 3-step CoT --------------------
    prompt_p1 = build_prompt_p1_cot(history, cf_candidates, dataset.item_texts, TOP_K)
    ranking_p1 = llm_rerank(llm, prompt_p1, cf_candidates, TOP_K)
    metrics_p1 = calculate_metrics([ranking_p1], [ground_truth], k_values=[TOP_K])

    result['p1_baseline_cot'] = {
        'ndcg': metrics_p1['ndcg@10'],
        'recall': metrics_p1['recall@10']
    }

    # -------------------- P3: Enhanced CoT --------------------
    prompt_p3 = build_prompt_p3_enhanced(history, cf_candidates, dataset.item_texts, TOP_K)
    ranking_p3 = llm_rerank(llm, prompt_p3, cf_candidates, TOP_K)
    metrics_p3 = calculate_metrics([ranking_p3], [ground_truth], k_values=[TOP_K])

    result['p3_enhanced_cot'] = {
        'ndcg': metrics_p3['ndcg@10'],
        'recall': metrics_p3['recall@10']
    }

    return result


def aggregate_results(checkpoint, all_samples):
    """Aggregate results from all completed users"""
    per_item = checkpoint.get_per_item_results()

    aggregated = {
        'recall_quality': [],
        'p1_baseline_cot': {'ndcg_scores': [], 'recall_scores': []},
        'p3_enhanced_cot': {'ndcg_scores': [], 'recall_scores': []}
    }

    for idx_str, result in per_item.items():
        if result:
            aggregated['recall_quality'].append(result['recall_quality'])

            for strategy in ['p1_baseline_cot', 'p3_enhanced_cot']:
                aggregated[strategy]['ndcg_scores'].append(result[strategy]['ndcg'])
                aggregated[strategy]['recall_scores'].append(result[strategy]['recall'])

    # Compute means
    final_results = {}
    for strategy in ['p1_baseline_cot', 'p3_enhanced_cot']:
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
    print("Phase 7-1: Cross-dataset Prompt Validation - Electronics Dataset")
    print("=" * 80)

    # ==================== Initialize Checkpoint ====================
    checkpoint = ExperimentCheckpoint(CHECKPOINT_PATH, auto_save_interval=5)

    completed_count = len(checkpoint.get_completed_indices())
    if completed_count > 0:
        print(f"\n⏭️  Resuming from checkpoint: {completed_count} users already completed")
    else:
        print(f"\n🆕 Starting fresh experiment")

    print(f"\nConfiguration:")
    print(f"  Dataset: Electronics")
    print(f"  Test users: {N_TEST_USERS}")
    print(f"  Prompts: P1 (Baseline CoT) vs P3 (Enhanced CoT)")
    print(f"  Checkpoint: {CHECKPOINT_PATH}")

    # ==================== Load Dataset ====================
    print(f"\n[1/5] Loading Electronics dataset...")
    dataset = AmazonBeautyDataset(
        data_path=str(project_root / "data" / "processed" / "amazon_electronics_sampled")
    )

    print(f"✓ Dataset loaded:")
    print(f"  Users: {dataset.num_users}")
    print(f"  Items: {dataset.num_items}")
    print(f"  Items with text: {len(dataset.item_texts)}")

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

    # ==================== Display Results ====================
    print("\n" + "=" * 80)
    print("Phase 7-1 Results: Electronics Dataset")
    print("=" * 80)
    print()
    print("| Prompt Strategy     | Recall@100 | NDCG@10 | Recall@10 | vs P1 |")
    print("|---------------------|------------|---------|-----------|-------|")

    p1_ndcg = final_results['p1_baseline_cot']['ndcg@10']
    p3_ndcg = final_results['p3_enhanced_cot']['ndcg@10']
    improvement = ((p3_ndcg / p1_ndcg - 1) * 100) if p1_ndcg > 0 else 0

    print(f"| P1 (Baseline CoT)   | {final_results['p1_baseline_cot']['recall@100']:10.4f} | "
          f"{p1_ndcg:7.4f} | {final_results['p1_baseline_cot']['recall@10']:9.4f} | -     |")
    print(f"| P3 (Enhanced CoT)   | {final_results['p3_enhanced_cot']['recall@100']:10.4f} | "
          f"{p3_ndcg:7.4f} | {final_results['p3_enhanced_cot']['recall@10']:9.4f} | +{improvement:4.0f}% |")

    metadata = checkpoint.get_metadata()
    if 'execution_time' in metadata:
        print(f"\nExecution time: {metadata['execution_time']/60:.1f} minutes")
        print(f"Avg time per user: {metadata['avg_time_per_user']:.1f} seconds")

    # ==================== Analysis ====================
    print(f"\n📊 Analysis:")
    print(f"  - CF Recall Quality: {final_results['p1_baseline_cot']['recall@100']*100:.1f}%")
    print(f"  - P3 NDCG@10 Improvement: +{improvement:.0f}% over P1")

    if improvement >= 200:
        print(f"  ✅ SUCCESS: P3 achieves >200% improvement (as expected)")
    elif improvement >= 100:
        print(f"  ⚠️  PARTIAL: P3 improves but below 200% target")
    else:
        print(f"  ❌ CONCERN: P3 improvement <100%, investigate prompt suitability")

    # ==================== Save Results ====================
    output = {
        'config': {
            'dataset': 'Electronics',
            'n_test_users': N_TEST_USERS,
            'candidate_size': CANDIDATE_SIZE,
            'top_k': TOP_K,
            'random_seed': RANDOM_SEED
        },
        'results': final_results,
        'execution_time': metadata.get('execution_time', 0),
        'improvement_percentage': improvement
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Results saved to: {RESULTS_PATH}")

    # Finalize checkpoint
    checkpoint.finalize(final_results)

    print("\n" + "=" * 80)
    print("✓ Phase 7-1 (Electronics) Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
