"""
Baseline Comparison Experiment - Movies Dataset

Compare different recall methods:
1. Pop (Popularity-based)
2. Random
3. CF (our method)

All use same LLM reranking for fair comparison.
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


class PopularityRecall:
    """Popularity-based recall"""

    def __init__(self):
        self.item_popularity = {}

    def fit(self, train_df: pd.DataFrame):
        """Count item popularity from training data"""
        self.item_popularity = train_df['item_id'].value_counts().to_dict()
        print(f"[PopRecall] Learned popularity for {len(self.item_popularity)} items")

    def recall(self, user_id: int, K: int = 100, exclude_ids: List[int] = None) -> tuple:
        """Return top-K popular items"""
        exclude_set = set(exclude_ids) if exclude_ids else set()

        # Sort items by popularity
        sorted_items = sorted(
            self.item_popularity.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Get top-K excluding user history
        candidates = []
        scores = []
        for item_id, count in sorted_items:
            if item_id not in exclude_set:
                candidates.append(item_id)
                scores.append(float(count))
            if len(candidates) >= K:
                break

        return candidates, scores


class RandomRecall:
    """Random recall baseline"""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.all_items = []

    def fit(self, train_df: pd.DataFrame):
        """Get all items from training data"""
        self.all_items = list(train_df['item_id'].unique())
        print(f"[RandomRecall] Total items: {len(self.all_items)}")

    def recall(self, user_id: int, K: int = 100, exclude_ids: List[int] = None) -> tuple:
        """Return random K items"""
        np.random.seed(self.random_state + user_id)  # Deterministic per user

        exclude_set = set(exclude_ids) if exclude_ids else set()
        available_items = [item for item in self.all_items if item not in exclude_set]

        # Random sample
        K_actual = min(K, len(available_items))
        candidates = np.random.choice(available_items, size=K_actual, replace=False).tolist()
        scores = [1.0] * len(candidates)

        return candidates, scores


def main():
    print("="*80)
    print("Baseline Comparison Experiment - Movies Dataset")
    print("="*80)

    # Configuration
    DATASET = 'Movies'
    N_TEST_USERS = 50
    CANDIDATE_SIZE = 100
    TOP_K = 10
    N_FACTORS = 128

    print(f"\nConfiguration:")
    print(f"  Dataset: {DATASET}")
    print(f"  Test users: {N_TEST_USERS}")
    print(f"  Candidate size: K={CANDIDATE_SIZE}")
    print(f"  Methods: Pop, Random, CF (128 factors)")

    # ==================== Load Dataset ====================
    print(f"\n[1/6] Loading {DATASET} dataset...")

    dataset = AmazonBeautyDataset(
        data_path=str(project_root / "data" / "processed" / "amazon_movies_sampled")
    )

    train_samples, val_samples, test_samples = dataset.split()

    print(f"✓ Dataset loaded:")
    print(f"  Train: {len(train_samples)}")
    print(f"  Val: {len(val_samples)}")
    print(f"  Test: {len(test_samples)}")

    # ==================== Sample Test Users ====================
    print(f"\n[2/6] Sampling test users...")

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

    print(f"  Sampled: {len(sampled_test)} test users")

    # ==================== Build Recall Methods ====================
    print(f"\n[3/6] Building recall methods...")

    # Pop
    print("\n  [3.1] Popularity-based recall...")
    pop_recall = PopularityRecall()
    pop_recall.fit(dataset.train_df)

    # Random
    print("\n  [3.2] Random recall...")
    random_recall = RandomRecall()
    random_recall.fit(dataset.train_df)

    # CF
    print("\n  [3.3] CF recall (128 factors)...")
    cf_recall = CFRecall(n_factors=N_FACTORS)
    cf_recall.fit(dataset.train_df)

    print("\n✓ All recall methods built")

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
    print(f"\n[5/6] Running baseline comparison...")

    results = {
        'pop': {'ndcg_scores': [], 'recall_scores': [], 'recall_quality': []},
        'random': {'ndcg_scores': [], 'recall_scores': [], 'recall_quality': []},
        'cf': {'ndcg_scores': [], 'recall_scores': [], 'recall_quality': []}
    }

    start_time = time.time()

    for sample in tqdm(sampled_test, desc="Processing users"):
        user_id = sample['user_id']
        history = sample.get('history', [])
        ground_truth = sample.get('ground_truth', [])[:TOP_K]

        if not history or not ground_truth:
            continue

        # -------------------- Pop Baseline --------------------
        pop_candidates, _ = pop_recall.recall(
            user_id=user_id,
            K=CANDIDATE_SIZE,
            exclude_ids=history
        )

        pop_recall_quality = len(set(pop_candidates) & set(ground_truth)) / len(ground_truth)
        results['pop']['recall_quality'].append(pop_recall_quality)

        prompt_pop = build_3step_cot_prompt(history, pop_candidates, dataset.item_texts, TOP_K)
        pop_ranking = llm_rerank(llm, prompt_pop, pop_candidates, TOP_K)

        pop_metrics = calculate_metrics([pop_ranking], [ground_truth], k_values=[TOP_K])
        results['pop']['ndcg_scores'].append(pop_metrics['ndcg@10'])
        results['pop']['recall_scores'].append(pop_metrics['recall@10'])

        # -------------------- Random Baseline --------------------
        random_candidates, _ = random_recall.recall(
            user_id=user_id,
            K=CANDIDATE_SIZE,
            exclude_ids=history
        )

        random_recall_quality = len(set(random_candidates) & set(ground_truth)) / len(ground_truth)
        results['random']['recall_quality'].append(random_recall_quality)

        prompt_random = build_3step_cot_prompt(history, random_candidates, dataset.item_texts, TOP_K)
        random_ranking = llm_rerank(llm, prompt_random, random_candidates, TOP_K)

        random_metrics = calculate_metrics([random_ranking], [ground_truth], k_values=[TOP_K])
        results['random']['ndcg_scores'].append(random_metrics['ndcg@10'])
        results['random']['recall_scores'].append(random_metrics['recall@10'])

        # -------------------- CF (Our Method) --------------------
        cf_candidates, _ = cf_recall.recall(
            user_id=user_id,
            K=CANDIDATE_SIZE,
            exclude_ids=history
        )

        cf_recall_quality = len(set(cf_candidates) & set(ground_truth)) / len(ground_truth)
        results['cf']['recall_quality'].append(cf_recall_quality)

        prompt_cf = build_3step_cot_prompt(history, cf_candidates, dataset.item_texts, TOP_K)
        cf_ranking = llm_rerank(llm, prompt_cf, cf_candidates, TOP_K)

        cf_metrics = calculate_metrics([cf_ranking], [ground_truth], k_values=[TOP_K])
        results['cf']['ndcg_scores'].append(cf_metrics['ndcg@10'])
        results['cf']['recall_scores'].append(cf_metrics['recall@10'])

    elapsed_time = time.time() - start_time

    # ==================== Summary ====================
    print(f"\n[6/6] Computing summary...")

    summary = {}
    for method in ['pop', 'random', 'cf']:
        summary[method] = {
            'recall_quality': np.mean(results[method]['recall_quality']),
            'ndcg@10': np.mean(results[method]['ndcg_scores']),
            'recall@10': np.mean(results[method]['recall_scores'])
        }

    print(f"\n{'='*80}")
    print("Baseline Comparison Results")
    print(f"{'='*80}\n")

    print(f"| Method | Recall@{CANDIDATE_SIZE} | NDCG@{TOP_K} | Recall@{TOP_K} |")
    print(f"|--------|-------------|----------|------------|")
    print(f"| Pop (popularity) | {summary['pop']['recall_quality']:.4f} | {summary['pop']['ndcg@10']:.4f} | {summary['pop']['recall@10']:.4f} |")
    print(f"| Random           | {summary['random']['recall_quality']:.4f} | {summary['random']['ndcg@10']:.4f} | {summary['random']['recall@10']:.4f} |")
    print(f"| CF (ours)        | {summary['cf']['recall_quality']:.4f} | {summary['cf']['ndcg@10']:.4f} | {summary['cf']['recall@10']:.4f} |")

    print(f"\nExecution time: {elapsed_time/60:.1f} minutes")
    print(f"Avg time per user: {elapsed_time/len(sampled_test):.1f} seconds")

    # Save results
    output_path = f'/home/coder-gw/Projects/GraphLLMRec/experiments/logs/baseline_comparison_movies.json'
    with open(output_path, 'w') as f:
        json.dump({
            'config': {
                'dataset': DATASET,
                'n_test_users': len(sampled_test),
                'candidate_size': CANDIDATE_SIZE,
                'top_k': TOP_K
            },
            'results': {
                'pop': {
                    'recall_quality': summary['pop']['recall_quality'],
                    'ndcg@10': summary['pop']['ndcg@10'],
                    'recall@10': summary['pop']['recall@10'],
                    'ndcg_scores': results['pop']['ndcg_scores'],
                    'recall_scores': results['pop']['recall_scores'],
                    'recall_quality_per_user': results['pop']['recall_quality']
                },
                'random': {
                    'recall_quality': summary['random']['recall_quality'],
                    'ndcg@10': summary['random']['ndcg@10'],
                    'recall@10': summary['random']['recall@10'],
                    'ndcg_scores': results['random']['ndcg_scores'],
                    'recall_scores': results['random']['recall_scores'],
                    'recall_quality_per_user': results['random']['recall_quality']
                },
                'cf': {
                    'recall_quality': summary['cf']['recall_quality'],
                    'ndcg@10': summary['cf']['ndcg@10'],
                    'recall@10': summary['cf']['recall@10'],
                    'ndcg_scores': results['cf']['ndcg_scores'],
                    'recall_scores': results['cf']['recall_scores'],
                    'recall_quality_per_user': results['cf']['recall_quality']
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
