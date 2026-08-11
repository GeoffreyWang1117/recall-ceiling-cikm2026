"""
Phase 8-2: CF Factor Optimization - Beauty Dataset

Test different CF factor sizes on successful dataset:
- 128 factors (baseline from Phase 7-1)
- 256 factors
- 512 factors

Goal: Validate if CF optimization works on small-scale dataset (Beauty 1.4K items)
Expected: Unlike Movies failure, Beauty might benefit from more factors
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
CHECKPOINT_PATH = "experiments/logs/checkpoints/p8_2_cf_factors_beauty_checkpoint.json"
RESULTS_PATH = "experiments/logs/p8_2_cf_factors_beauty.json"

N_TEST_USERS = 142  # All Beauty test users
CANDIDATE_SIZE = 100
TOP_K = 10
FACTORS_TO_TEST = [128, 256, 512]  # Test different factor sizes
PROMPTS_TO_TEST = ["P1", "P3"]  # Test baseline and enhanced prompts
RANDOM_SEED = 42


def build_prompt_p1_baseline(
    user_history: List[int],
    candidates: List[int],
    item_texts: Dict[int, str],
    top_k: int = 10
) -> str:
    """P1: 3-step CoT (Baseline)"""
    prompt = "Task: Recommend products based on user preferences using 3-step reasoning.\n\n"

    prompt += "User's purchase history:\n"
    for idx, item_id in enumerate(user_history[-10:], 1):
        text = item_texts.get(item_id, f"Item {item_id}")
        prompt += f"  {idx}. {text}\n"

    prompt += f"\nCandidate products:\n"
    for idx, item_id in enumerate(candidates, 1):
        text = item_texts.get(item_id, f"Item {item_id}")
        prompt += f"  {idx}. Item {item_id}: {text}\n"

    prompt += "\nStep 1: Understand user's preferences.\n"
    prompt += "Step 2: Match candidates to preferences.\n"
    prompt += "Step 3: Rank candidates by relevance.\n\n"

    prompt += f"Output the top {top_k} items:\n"
    prompt += "Format: One item per line as 'Item <ID>'\n\n"
    prompt += f"Output:\n"

    return prompt


def build_prompt_p3_enhanced(
    user_history: List[int],
    candidates: List[int],
    item_texts: Dict[int, str],
    top_k: int = 10
) -> str:
    """P3: Enhanced CoT with explicit preference signals (Beauty version)"""
    prompt = "Task: Recommend beauty products based on user preferences using detailed 3-step analysis.\n\n"

    prompt += "User's purchase history (most recent items):\n"
    for idx, item_id in enumerate(user_history[-10:], 1):
        text = item_texts.get(item_id, f"Item {item_id}")
        prompt += f"  {idx}. {text}\n"

    prompt += f"\nCandidate products:\n"
    for idx, item_id in enumerate(candidates, 1):
        text = item_texts.get(item_id, f"Item {item_id}")
        prompt += f"  {idx}. Item {item_id}: {text}\n"

    prompt += "\nReasoning Steps:\n"
    prompt += "Step 1 - Preference Analysis:\n"
    prompt += "  - Identify user's preferred: (a) categories (e.g., makeup, skincare), "
    prompt += "(b) brands (e.g., Maybelline, NYX), (c) features/attributes (e.g., matte, hydrating)\n"
    prompt += "\nStep 2 - Candidate Evaluation:\n"
    prompt += "  - For each candidate, assess: (a) category match, (b) brand affinity, (c) feature similarity\n"
    prompt += "  - Rate relevance based on alignment with user preferences\n"
    prompt += "\nStep 3 - Ranking Decision:\n"
    prompt += "  - Rank candidates by overall relevance to user preferences\n"
    prompt += f"  - Select top {top_k} products for recommendation\n\n"

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


def process_single_user(sample, cf_recalls, llm, dataset):
    """Process a single user with different CF factor sizes and prompts"""
    user_id = sample['user_id']
    history = sample.get('history', [])
    ground_truth = sample.get('ground_truth', [])[:TOP_K]

    if not history or not ground_truth:
        return None

    result = {}

    # Test each CF configuration
    for n_factors in FACTORS_TO_TEST:
        cf_recall = cf_recalls[n_factors]

        # Get CF candidates
        cf_candidates, _ = cf_recall.recall(
            user_id=user_id,
            K=CANDIDATE_SIZE,
            exclude_ids=history
        )

        recall_quality = len(set(cf_candidates) & set(ground_truth)) / len(ground_truth)

        # Test both P1 and P3 prompts
        for prompt_name in PROMPTS_TO_TEST:
            if prompt_name == "P1":
                prompt = build_prompt_p1_baseline(history, cf_candidates, dataset.item_texts, TOP_K)
            else:  # P3
                prompt = build_prompt_p3_enhanced(history, cf_candidates, dataset.item_texts, TOP_K)

            ranking = llm_rerank(llm, prompt, cf_candidates, TOP_K)
            metrics = calculate_metrics([ranking], [ground_truth], k_values=[TOP_K])

            key = f'cf_{n_factors}_{prompt_name}'
            result[key] = {
                'recall@100': recall_quality,
                'ndcg': metrics['ndcg@10'],
                'recall@10': metrics['recall@10']
            }

    return result


def aggregate_results(checkpoint, all_samples):
    """Aggregate results from all completed users"""
    per_item = checkpoint.get_per_item_results()

    aggregated = {}
    for n_factors in FACTORS_TO_TEST:
        for prompt_name in PROMPTS_TO_TEST:
            key = f'cf_{n_factors}_{prompt_name}'
            aggregated[key] = {
                'recall@100_scores': [],
                'ndcg_scores': [],
                'recall@10_scores': []
            }

    for idx_str, result in per_item.items():
        if result:
            for n_factors in FACTORS_TO_TEST:
                for prompt_name in PROMPTS_TO_TEST:
                    key = f'cf_{n_factors}_{prompt_name}'
                    aggregated[key]['recall@100_scores'].append(result[key]['recall@100'])
                    aggregated[key]['ndcg_scores'].append(result[key]['ndcg'])
                    aggregated[key]['recall@10_scores'].append(result[key]['recall@10'])

    # Compute means
    final_results = {}
    for n_factors in FACTORS_TO_TEST:
        for prompt_name in PROMPTS_TO_TEST:
            key = f'cf_{n_factors}_{prompt_name}'
            final_results[key] = {
                'recall@100': np.mean(aggregated[key]['recall@100_scores']) if aggregated[key]['recall@100_scores'] else 0,
                'ndcg@10': np.mean(aggregated[key]['ndcg_scores']) if aggregated[key]['ndcg_scores'] else 0,
                'recall@10': np.mean(aggregated[key]['recall@10_scores']) if aggregated[key]['recall@10_scores'] else 0,
                'recall@100_scores': aggregated[key]['recall@100_scores'],
                'ndcg_scores': aggregated[key]['ndcg_scores'],
                'recall@10_scores': aggregated[key]['recall@10_scores']
            }

    return final_results


def main():
    print("=" * 80)
    print("Phase 8-2: CF Factor Optimization - Beauty Dataset")
    print("=" * 80)

    # ==================== Initialize Checkpoint ====================
    checkpoint = ExperimentCheckpoint(CHECKPOINT_PATH, auto_save_interval=10)

    completed_count = len(checkpoint.get_completed_indices())
    if completed_count > 0:
        print(f"\n⏭️  Resuming from checkpoint: {completed_count} users already completed")
    else:
        print(f"\n🆕 Starting fresh experiment")

    print(f"\nConfiguration:")
    print(f"  Dataset: Beauty (Success Case)")
    print(f"  Test users: {N_TEST_USERS}")
    print(f"  CF Factors: {FACTORS_TO_TEST}")
    print(f"  Prompts: {PROMPTS_TO_TEST}")
    print(f"  Checkpoint: {CHECKPOINT_PATH}")

    # ==================== Load Dataset ====================
    print(f"\n[1/5] Loading Beauty dataset...")
    dataset = AmazonBeautyDataset(
        data_path=str(project_root / "data" / "processed" / "amazon_beauty_sampled")
    )

    print(f"✓ Dataset loaded:")
    print(f"  Users: {dataset.num_users}")
    print(f"  Items: {dataset.num_items}")
    print(f"  Items with text: {len(dataset.item_texts)}")

    # ==================== Sample Test Users ====================
    print(f"\n[2/5] Loading all test users...")
    _, _, test_samples = dataset.split()

    valid_test_samples = [s for s in test_samples if len(s.get('history', [])) > 0]
    print(f"  Total valid test users: {len(valid_test_samples)}")

    # Use all test users (no sampling)
    sampled_test = valid_test_samples

    # Get pending
    pending = get_pending_items(sampled_test, checkpoint)
    print(f"  Pending: {len(pending)} users")
    print(f"  Already completed: {len(checkpoint.get_completed_indices())} users")

    if len(pending) == 0:
        print("\n✓ All users already processed!")
        print("  Computing final results...")
    else:
        # ==================== Build CF Recalls ====================
        print(f"\n[3/5] Building CF recalls with different factors...")
        cf_recalls = {}

        for n_factors in FACTORS_TO_TEST:
            print(f"  Building CF with {n_factors} factors...")
            cf_recall = CFRecall(n_factors=n_factors)
            cf_recall.fit(dataset.train_df)
            cf_recalls[n_factors] = cf_recall

        print("\n✓ All CF models built")

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
        print(f"  NOTE: Each user tests {len(FACTORS_TO_TEST)} CF × {len(PROMPTS_TO_TEST)} prompts = {len(FACTORS_TO_TEST)*len(PROMPTS_TO_TEST)} configs")

        start_time = time.time()

        try:
            for idx, sample in tqdm(pending, desc="Processing users"):
                result = process_single_user(sample, cf_recalls, llm, dataset)
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
    print("Phase 8-2 Results: CF Factor Optimization (Beauty)")
    print("=" * 80)

    for prompt_name in PROMPTS_TO_TEST:
        print(f"\n### Prompt: {prompt_name}")
        print()
        print("| CF Factors | Recall@100 | NDCG@10 | Recall@10 | vs 128f |")
        print("|------------|------------|---------|-----------|---------|")

        baseline_key = f'cf_128_{prompt_name}'
        baseline_ndcg = final_results[baseline_key]['ndcg@10']
        baseline_recall = final_results[baseline_key]['recall@100']

        for n_factors in FACTORS_TO_TEST:
            key = f'cf_{n_factors}_{prompt_name}'
            ndcg = final_results[key]['ndcg@10']
            recall100 = final_results[key]['recall@100']
            recall10 = final_results[key]['recall@10']

            if n_factors == 128:
                improvement = "-"
            else:
                recall_improvement = ((recall100 / baseline_recall - 1) * 100) if baseline_recall > 0 else 0
                ndcg_improvement = ((ndcg / baseline_ndcg - 1) * 100) if baseline_ndcg > 0 else 0
                improvement = f"+{recall_improvement:.0f}% R / {ndcg_improvement:+.0f}% N"

            print(f"| {n_factors:10} | {recall100:10.4f} | {ndcg:7.4f} | {recall10:9.4f} | {improvement:7} |")

    metadata = checkpoint.get_metadata()
    if 'execution_time' in metadata:
        print(f"\nExecution time: {metadata['execution_time']/60:.1f} minutes")
        print(f"Avg time per user: {metadata['avg_time_per_user']:.1f} seconds")

    # ==================== Analysis ====================
    print(f"\n📊 Critical Analysis: Beauty vs Movies Comparison")
    print(f"\n--- CF Factor Optimization Results ---")

    # Compare P3 results (best prompt)
    baseline_p3 = final_results['cf_128_P3']
    factors_256_p3 = final_results['cf_256_P3']
    factors_512_p3 = final_results['cf_512_P3']

    recall_improvement_256 = ((factors_256_p3['recall@100'] / baseline_p3['recall@100'] - 1) * 100) if baseline_p3['recall@100'] > 0 else 0
    ndcg_improvement_256 = ((factors_256_p3['ndcg@10'] / baseline_p3['ndcg@10'] - 1) * 100) if baseline_p3['ndcg@10'] > 0 else 0

    print(f"\n🔍 Beauty (P3 Enhanced):")
    print(f"  - 128f: Recall@100={baseline_p3['recall@100']*100:.1f}%, NDCG@10={baseline_p3['ndcg@10']:.4f}")
    print(f"  - 256f: Recall@100={factors_256_p3['recall@100']*100:.1f}% ({recall_improvement_256:+.0f}%), "
          f"NDCG@10={factors_256_p3['ndcg@10']:.4f} ({ndcg_improvement_256:+.0f}%)")

    # Compare to Movies Phase 8-1 results
    print(f"\n🔍 Movies (P3 Enhanced, from Phase 8-1):")
    print(f"  - 128f: Recall@100=2.8%, NDCG@10=0.0084")
    print(f"  - 256f: Recall@100=3.4% (+21%), NDCG@10=0.0056 (-33%)")

    # Conclusion
    print(f"\n💡 Key Findings:")
    if ndcg_improvement_256 > 0 and recall_improvement_256 > 10:
        print(f"  ✅ SUCCESS: Beauty benefits from 256 factors (unlike Movies)")
        print(f"     → Small item pool (1.4K) can utilize finer-grained factorization")
        print(f"     → Quality-quantity tradeoff does NOT occur on small datasets")
    elif ndcg_improvement_256 < 0:
        print(f"  ⚠️  TRADEOFF: Beauty shows same quality-quantity tradeoff as Movies")
        print(f"     → More factors improve recall but hurt NDCG")
        print(f"     → This is a universal phenomenon, not dataset-specific")
    else:
        print(f"  ➡️  NEUTRAL: Marginal improvements or no clear trend")

    # ==================== Save Results ====================
    output = {
        'config': {
            'dataset': 'Beauty',
            'n_test_users': len(sampled_test),
            'candidate_size': CANDIDATE_SIZE,
            'top_k': TOP_K,
            'factors_tested': FACTORS_TO_TEST,
            'prompts_tested': PROMPTS_TO_TEST,
            'random_seed': RANDOM_SEED
        },
        'results': final_results,
        'execution_time': metadata.get('execution_time', 0),
        'analysis': {
            'p3_256_vs_128': {
                'recall@100_improvement': recall_improvement_256,
                'ndcg@10_improvement': ndcg_improvement_256
            },
            'comparison_to_movies': {
                'movies_256_recall_improvement': 21.4,  # From Phase 8-1
                'movies_256_ndcg_improvement': -33.3,   # From Phase 8-1
                'beauty_256_recall_improvement': recall_improvement_256,
                'beauty_256_ndcg_improvement': ndcg_improvement_256
            }
        }
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Results saved to: {RESULTS_PATH}")

    # Finalize checkpoint
    checkpoint.finalize(final_results)

    print("\n" + "=" * 80)
    print("✓ Phase 8-2 (Beauty) Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
