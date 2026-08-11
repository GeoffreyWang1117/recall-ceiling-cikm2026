"""
Phase 7-1 Fix: Movies Domain-Optimized Prompt

Test P3-Movies-Optimized variant that focuses on:
- Themes and emotional tones (not just genres)
- Storytelling styles (not just directors)
- Narrative preferences (not just actors)

Compare:
- P1 (Baseline CoT)
- P3 (Original Enhanced CoT) - already tested, failed (-10%)
- P3-Movies (Domain-Optimized) - NEW
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
CHECKPOINT_PATH = "experiments/logs/checkpoints/p7_1_fix_movies_checkpoint.json"
RESULTS_PATH = "experiments/logs/p7_1_fix_movies_optimized.json"

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
    """P1: Baseline 3-step CoT prompt"""
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


def build_prompt_p3_movies_optimized(
    user_history: List[int],
    candidates: List[int],
    item_texts: Dict[int, str],
    top_k: int = 10
) -> str:
    """
    P3-Movies-Optimized: Domain-specific Enhanced CoT

    Focus on:
    - Themes and emotional tones (primary)
    - Storytelling styles (secondary)
    - Genre as supporting signal (tertiary)
    """
    prompt = "Task: Recommend movies based on user preferences using detailed 3-step reasoning.\n\n"

    prompt += "User's viewing history (most recent items):\n"
    for idx, item_id in enumerate(user_history[-10:], 1):
        text = item_texts.get(item_id, f"Item {item_id}")
        prompt += f"  {idx}. {text}\n"

    prompt += f"\nCandidate movies:\n"
    for idx, item_id in enumerate(candidates, 1):
        text = item_texts.get(item_id, f"Item {item_id}")
        prompt += f"  {idx}. Item {item_id}: {text}\n"

    prompt += "\nReasoning Steps:\n"
    prompt += "Step 1 - Extract User Preferences:\n"
    prompt += "  - Identify the THEMES and EMOTIONAL TONES the user enjoys (e.g., adventure, romance, suspense, humor)\n"
    prompt += "  - Note the STORYTELLING STYLES (e.g., fast-paced, character-driven, plot-driven)\n"
    prompt += "  - Consider any recurring GENRES as supporting signals\n"
    prompt += "\nStep 2 - Evaluate Candidates:\n"
    prompt += "  - For each candidate, assess THEMATIC and EMOTIONAL similarity to user's preferred movies\n"
    prompt += "  - Consider narrative style and tone alignment\n"
    prompt += "  - Evaluate overall relevance to user's viewing patterns\n"
    prompt += "\nStep 3 - Generate Ranking:\n"
    prompt += "  - Rank candidates by overall relevance (emphasizing thematic/emotional fit)\n"
    prompt += f"  - Select top {top_k} movies for recommendation\n\n"

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
    """Process a single user with P1 and P3-Movies-Optimized"""
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

    # -------------------- P1: Baseline CoT --------------------
    prompt_p1 = build_prompt_p1_cot(history, cf_candidates, dataset.item_texts, TOP_K)
    ranking_p1 = llm_rerank(llm, prompt_p1, cf_candidates, TOP_K)
    metrics_p1 = calculate_metrics([ranking_p1], [ground_truth], k_values=[TOP_K])

    result['p1_baseline_cot'] = {
        'ndcg': metrics_p1['ndcg@10'],
        'recall': metrics_p1['recall@10']
    }

    # -------------------- P3-Movies-Optimized --------------------
    prompt_p3_movies = build_prompt_p3_movies_optimized(history, cf_candidates, dataset.item_texts, TOP_K)
    ranking_p3_movies = llm_rerank(llm, prompt_p3_movies, cf_candidates, TOP_K)
    metrics_p3_movies = calculate_metrics([ranking_p3_movies], [ground_truth], k_values=[TOP_K])

    result['p3_movies_optimized'] = {
        'ndcg': metrics_p3_movies['ndcg@10'],
        'recall': metrics_p3_movies['recall@10']
    }

    return result


def aggregate_results(checkpoint, all_samples):
    """Aggregate results from all completed users"""
    per_item = checkpoint.get_per_item_results()

    aggregated = {
        'recall_quality': [],
        'p1_baseline_cot': {'ndcg_scores': [], 'recall_scores': []},
        'p3_movies_optimized': {'ndcg_scores': [], 'recall_scores': []}
    }

    for idx_str, result in per_item.items():
        if result:
            aggregated['recall_quality'].append(result['recall_quality'])

            for strategy in ['p1_baseline_cot', 'p3_movies_optimized']:
                aggregated[strategy]['ndcg_scores'].append(result[strategy]['ndcg'])
                aggregated[strategy]['recall_scores'].append(result[strategy]['recall'])

    # Compute means
    final_results = {}
    for strategy in ['p1_baseline_cot', 'p3_movies_optimized']:
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
    print("Phase 7-1 Fix: Movies Domain-Optimized Prompt")
    print("=" * 80)

    # ==================== Initialize Checkpoint ====================
    checkpoint = ExperimentCheckpoint(CHECKPOINT_PATH, auto_save_interval=5)

    completed_count = len(checkpoint.get_completed_indices())
    if completed_count > 0:
        print(f"\n⏭️  Resuming from checkpoint: {completed_count} users already completed")
    else:
        print(f"\n🆕 Starting fresh experiment")

    print(f"\nConfiguration:")
    print(f"  Dataset: Movies")
    print(f"  Test users: {N_TEST_USERS}")
    print(f"  Prompts: P1 (Baseline) vs P3-Movies-Optimized (theme/mood focused)")
    print(f"  Checkpoint: {CHECKPOINT_PATH}")

    # ==================== Load Dataset ====================
    print(f"\n[1/5] Loading Movies dataset...")
    dataset = AmazonBeautyDataset(
        data_path=str(project_root / "data" / "processed" / "amazon_movies_sampled")
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
    print("Phase 7-1 Fix Results: Movies Domain-Optimized Prompt")
    print("=" * 80)
    print()
    print("| Prompt Strategy          | Recall@100 | NDCG@10 | Recall@10 | vs P1 |")
    print("|--------------------------|------------|---------|-----------|-------|")

    p1_ndcg = final_results['p1_baseline_cot']['ndcg@10']
    p3_movies_ndcg = final_results['p3_movies_optimized']['ndcg@10']
    improvement = ((p3_movies_ndcg / p1_ndcg - 1) * 100) if p1_ndcg > 0 else 0

    print(f"| P1 (Baseline CoT)        | {final_results['p1_baseline_cot']['recall@100']:10.4f} | "
          f"{p1_ndcg:7.4f} | {final_results['p1_baseline_cot']['recall@10']:9.4f} | -     |")
    print(f"| P3-Movies (Optimized)    | {final_results['p3_movies_optimized']['recall@100']:10.4f} | "
          f"{p3_movies_ndcg:7.4f} | {final_results['p3_movies_optimized']['recall@10']:9.4f} | +{improvement:4.0f}% |")

    metadata = checkpoint.get_metadata()
    if 'execution_time' in metadata:
        print(f"\nExecution time: {metadata['execution_time']/60:.1f} minutes")
        print(f"Avg time per user: {metadata['avg_time_per_user']:.1f} seconds")

    # ==================== Analysis ====================
    print(f"\n📊 Analysis:")
    print(f"  - CF Recall Quality: {final_results['p1_baseline_cot']['recall@100']*100:.1f}%")
    print(f"  - P3-Movies NDCG@10 Improvement: +{improvement:.0f}% over P1")

    # Compare with original P3 result
    print(f"\n🔍 Comparison with Original P3:")
    print(f"  - P3 (Original, genres/directors/actors): NDCG@10=0.0051 (-10% vs P1)")
    print(f"  - P3-Movies (Optimized, themes/moods): NDCG@10={p3_movies_ndcg:.4f} ({improvement:+.0f}% vs P1)")

    if improvement >= 100:
        print(f"  ✅ SUCCESS: P3-Movies achieves >100% improvement!")
        print(f"  → Domain-specific optimization validated")
    elif improvement >= 0:
        print(f"  ⚠️  PARTIAL: P3-Movies improves but below 100% target")
        print(f"  → Better than original P3 (-10%), but still room for improvement")
    else:
        print(f"  ❌ CONCERN: P3-Movies still negative, needs further refinement")

    # ==================== Save Results ====================
    output = {
        'config': {
            'dataset': 'Movies',
            'n_test_users': N_TEST_USERS,
            'candidate_size': CANDIDATE_SIZE,
            'top_k': TOP_K,
            'random_seed': RANDOM_SEED,
            'prompt_variant': 'P3-Movies-Optimized (theme/mood focused)'
        },
        'results': final_results,
        'execution_time': metadata.get('execution_time', 0),
        'improvement_percentage': improvement,
        'comparison': {
            'p3_original': {
                'ndcg@10': 0.0051,
                'improvement_vs_p1': -10.2
            },
            'p3_movies_optimized': {
                'ndcg@10': p3_movies_ndcg,
                'improvement_vs_p1': improvement
            }
        }
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Results saved to: {RESULTS_PATH}")

    # Finalize checkpoint
    checkpoint.finalize(final_results)

    print("\n" + "=" * 80)
    print("✓ Phase 7-1 Fix Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
