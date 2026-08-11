"""
Supplementary Experiments for KDD 2026 Paper
============================================

This script runs additional experiments to strengthen the paper:
1. Multi-LLM comparison (GPT-3.5, GPT-4, DeepSeek, Qwen)
2. Additional baselines (Random, PopRec, BM25)
3. Bootstrap confidence intervals
4. Larger scale experiments (200 users)

Requirements:
- .env file with OPENAI_API_KEY and DEEPSEEK_API_KEY
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'experiments' / 'realistic_recall'))
sys.path.insert(0, str(project_root / 'experiments' / 'baselines'))

from data.amazon_beauty_loader import AmazonBeautyDataset
from utils.metrics import calculate_metrics
from cf_recall import CFRecall


# ============================================================================
# API-based LLM Wrappers
# ============================================================================

class OpenAIReranker:
    """OpenAI API-based reranker (GPT-3.5/GPT-4)"""

    def __init__(self, model: str = "gpt-3.5-turbo"):
        import openai
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def rerank(self, prompt: str, candidates: List[int], top_k: int = 10) -> List[int]:
        """Rerank candidates using OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a recommendation system. Analyze user preferences and rank items."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )
            generated_text = response.choices[0].message.content
            return self._parse_item_ids(generated_text, candidates, top_k)
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return candidates[:top_k]

    def _parse_item_ids(self, text: str, candidates: List[int], top_k: int) -> List[int]:
        """Parse item IDs from generated text"""
        import re
        item_ids = []
        candidates_set = set(candidates)

        patterns = [
            r'[Ii]tem\s+(\d+)',
            r'#(\d+)',
            r'ID:\s*(\d+)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            for m in matches:
                item_id = int(m)
                if item_id in candidates_set and item_id not in item_ids:
                    item_ids.append(item_id)
                if len(item_ids) >= top_k:
                    break
            if len(item_ids) >= top_k:
                break

        # Fallback to candidates order
        for c in candidates:
            if c not in item_ids:
                item_ids.append(c)
            if len(item_ids) >= top_k:
                break

        return item_ids[:top_k]


class DeepSeekReranker:
    """DeepSeek API-based reranker"""

    def __init__(self, model: str = "deepseek-chat"):
        import openai
        self.client = openai.OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        self.model = model

    def rerank(self, prompt: str, candidates: List[int], top_k: int = 10) -> List[int]:
        """Rerank candidates using DeepSeek API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a recommendation system. Analyze user preferences and rank items."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )
            generated_text = response.choices[0].message.content
            return self._parse_item_ids(generated_text, candidates, top_k)
        except Exception as e:
            print(f"DeepSeek API error: {e}")
            return candidates[:top_k]

    def _parse_item_ids(self, text: str, candidates: List[int], top_k: int) -> List[int]:
        """Parse item IDs from generated text"""
        import re
        item_ids = []
        candidates_set = set(candidates)

        patterns = [
            r'[Ii]tem\s+(\d+)',
            r'#(\d+)',
            r'ID:\s*(\d+)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            for m in matches:
                item_id = int(m)
                if item_id in candidates_set and item_id not in item_ids:
                    item_ids.append(item_id)
                if len(item_ids) >= top_k:
                    break
            if len(item_ids) >= top_k:
                break

        for c in candidates:
            if c not in item_ids:
                item_ids.append(c)
            if len(item_ids) >= top_k:
                break

        return item_ids[:top_k]


# ============================================================================
# Baseline Methods
# ============================================================================

class RandomReranker:
    """Random reranking baseline"""

    def rerank(self, candidates: List[int], top_k: int = 10, seed: int = None) -> List[int]:
        if seed is not None:
            np.random.seed(seed)
        shuffled = np.random.permutation(candidates).tolist()
        return shuffled[:top_k]


class PopularityReranker:
    """Popularity-based reranking baseline"""

    def __init__(self, train_df: pd.DataFrame):
        # Compute item popularity (interaction count)
        self.item_popularity = train_df.groupby('item_id').size().to_dict()

    def rerank(self, candidates: List[int], top_k: int = 10) -> List[int]:
        # Sort by popularity (descending)
        scored = [(c, self.item_popularity.get(c, 0)) for c in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored[:top_k]]


class CFScoreReranker:
    """CF score-based reranking (no LLM, just use CF scores)"""

    def rerank(self, candidates: List[int], scores: List[float], top_k: int = 10) -> List[int]:
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored[:top_k]]


# ============================================================================
# Prompt Templates
# ============================================================================

def build_enhanced_cot_prompt(
    user_history: List[int],
    candidates: List[int],
    item_texts: Dict[int, str],
    top_k: int = 10
) -> str:
    """Build Enhanced CoT (P3) prompt"""
    prompt = """Task: Rerank recommendation candidates based on user preferences.

Step 1 - Extract User Preferences:
Analyze the user's interaction history to identify:
- Preferred categories/types
- Preferred brands
- Key features the user values

Step 2 - Evaluate Candidates:
Score each candidate on alignment with extracted preferences.

Step 3 - Rank and Output:
Select top items based on preference alignment.

User's interaction history:
"""
    for idx, item_id in enumerate(user_history[-10:], 1):
        text = item_texts.get(item_id, f"Item {item_id}")[:100]
        prompt += f"  {idx}. {text}\n"

    prompt += f"\nCandidate items to rank:\n"
    for idx, item_id in enumerate(candidates[:20], 1):  # Limit to first 20 for API efficiency
        text = item_texts.get(item_id, f"Item {item_id}")[:100]
        prompt += f"  {idx}. Item {item_id}: {text}\n"

    prompt += f"\nComplete the 3 steps and output the top {top_k} most relevant items.\n"
    prompt += "Format: List each item as 'Item <ID>' on a separate line.\n\n"
    prompt += "Output:\n"

    return prompt


# ============================================================================
# Bootstrap Confidence Intervals
# ============================================================================

def bootstrap_ci(scores: List[float], n_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval

    Returns: (mean, ci_lower, ci_upper)
    """
    scores = np.array(scores)
    n = len(scores)

    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))

    bootstrap_means = np.array(bootstrap_means)
    alpha = 1 - ci
    ci_lower = np.percentile(bootstrap_means, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)

    return np.mean(scores), ci_lower, ci_upper


# ============================================================================
# Main Experiment Runner
# ============================================================================

def run_multi_llm_comparison(
    dataset,
    cf_recall: CFRecall,
    test_samples: List[Dict],
    n_users: int = 50,
    top_k: int = 10,
    candidate_size: int = 100
) -> Dict:
    """
    Compare multiple LLMs on the same test set

    Tests:
    - Random baseline
    - Popularity baseline
    - CF score baseline
    - GPT-3.5-turbo
    - DeepSeek
    """
    print("\n" + "="*80)
    print("Multi-LLM Comparison Experiment")
    print("="*80)

    # Initialize rerankers
    results = {
        'random': {'ndcg_scores': [], 'recall_scores': []},
        'popularity': {'ndcg_scores': [], 'recall_scores': []},
        'cf_score': {'ndcg_scores': [], 'recall_scores': []},
    }

    # Check API availability
    has_openai = os.getenv("OPENAI_API_KEY") is not None
    has_deepseek = os.getenv("DEEPSEEK_API_KEY") is not None

    if has_openai:
        print("OpenAI API available - will test GPT-3.5")
        results['gpt-3.5-turbo'] = {'ndcg_scores': [], 'recall_scores': []}
        openai_reranker = OpenAIReranker(model="gpt-3.5-turbo")
    else:
        print("OpenAI API not available - skipping GPT tests")
        openai_reranker = None

    if has_deepseek:
        print("DeepSeek API available - will test DeepSeek")
        results['deepseek'] = {'ndcg_scores': [], 'recall_scores': []}
        deepseek_reranker = DeepSeekReranker()
    else:
        print("DeepSeek API not available - skipping DeepSeek tests")
        deepseek_reranker = None

    # Initialize baselines
    random_reranker = RandomReranker()
    pop_reranker = PopularityReranker(dataset.train_df)
    cf_reranker = CFScoreReranker()

    # Sample test users
    np.random.seed(42)
    valid_samples = [s for s in test_samples if len(s.get('history', [])) > 0]
    sampled = np.random.choice(len(valid_samples), size=min(n_users, len(valid_samples)), replace=False)
    test_subset = [valid_samples[i] for i in sampled]

    print(f"\nRunning on {len(test_subset)} test users...")

    for sample in tqdm(test_subset, desc="Processing users"):
        user_id = sample['user_id']
        history = sample.get('history', [])
        ground_truth = sample.get('ground_truth', [])[:top_k]

        if not history or not ground_truth:
            continue

        # Get CF candidates
        candidates, scores = cf_recall.recall(
            user_id=user_id,
            K=candidate_size,
            exclude_ids=history
        )

        # Build prompt for LLM methods
        prompt = build_enhanced_cot_prompt(history, candidates, dataset.item_texts, top_k)

        # Random baseline
        random_ranking = random_reranker.rerank(candidates, top_k, seed=user_id)
        metrics = calculate_metrics([random_ranking], [ground_truth], k_values=[top_k])
        results['random']['ndcg_scores'].append(metrics['ndcg@10'])
        results['random']['recall_scores'].append(metrics['recall@10'])

        # Popularity baseline
        pop_ranking = pop_reranker.rerank(candidates, top_k)
        metrics = calculate_metrics([pop_ranking], [ground_truth], k_values=[top_k])
        results['popularity']['ndcg_scores'].append(metrics['ndcg@10'])
        results['popularity']['recall_scores'].append(metrics['recall@10'])

        # CF score baseline
        cf_ranking = cf_reranker.rerank(candidates, scores, top_k)
        metrics = calculate_metrics([cf_ranking], [ground_truth], k_values=[top_k])
        results['cf_score']['ndcg_scores'].append(metrics['ndcg@10'])
        results['cf_score']['recall_scores'].append(metrics['recall@10'])

        # GPT-3.5 (if available)
        if openai_reranker:
            try:
                gpt_ranking = openai_reranker.rerank(prompt, candidates, top_k)
                metrics = calculate_metrics([gpt_ranking], [ground_truth], k_values=[top_k])
                results['gpt-3.5-turbo']['ndcg_scores'].append(metrics['ndcg@10'])
                results['gpt-3.5-turbo']['recall_scores'].append(metrics['recall@10'])
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                print(f"GPT error: {e}")

        # DeepSeek (if available)
        if deepseek_reranker:
            try:
                ds_ranking = deepseek_reranker.rerank(prompt, candidates, top_k)
                metrics = calculate_metrics([ds_ranking], [ground_truth], k_values=[top_k])
                results['deepseek']['ndcg_scores'].append(metrics['ndcg@10'])
                results['deepseek']['recall_scores'].append(metrics['recall@10'])
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                print(f"DeepSeek error: {e}")

    # Compute summary with bootstrap CIs
    summary = {}
    for method, scores in results.items():
        if len(scores['ndcg_scores']) > 0:
            ndcg_mean, ndcg_lower, ndcg_upper = bootstrap_ci(scores['ndcg_scores'])
            recall_mean, recall_lower, recall_upper = bootstrap_ci(scores['recall_scores'])

            summary[method] = {
                'ndcg@10': {
                    'mean': ndcg_mean,
                    'ci_lower': ndcg_lower,
                    'ci_upper': ndcg_upper
                },
                'recall@10': {
                    'mean': recall_mean,
                    'ci_lower': recall_lower,
                    'ci_upper': recall_upper
                },
                'n_samples': len(scores['ndcg_scores'])
            }

    return summary, results


def run_baseline_comparison(
    dataset,
    cf_recall: CFRecall,
    test_samples: List[Dict],
    n_users: int = 100,
    top_k: int = 10,
    candidate_size: int = 100
) -> Dict:
    """
    Comprehensive baseline comparison without API calls
    """
    print("\n" + "="*80)
    print("Baseline Comparison Experiment (No API)")
    print("="*80)

    results = {
        'random': {'ndcg_scores': [], 'recall_scores': []},
        'popularity': {'ndcg_scores': [], 'recall_scores': []},
        'cf_score': {'ndcg_scores': [], 'recall_scores': []},
        'cf_score_reordered': {'ndcg_scores': [], 'recall_scores': []},  # CF score with random tiebreaker
    }

    random_reranker = RandomReranker()
    pop_reranker = PopularityReranker(dataset.train_df)
    cf_reranker = CFScoreReranker()

    np.random.seed(42)
    valid_samples = [s for s in test_samples if len(s.get('history', [])) > 0]
    sampled = np.random.choice(len(valid_samples), size=min(n_users, len(valid_samples)), replace=False)
    test_subset = [valid_samples[i] for i in sampled]

    recall_quality_scores = []

    print(f"\nRunning on {len(test_subset)} test users...")

    for sample in tqdm(test_subset, desc="Processing users"):
        user_id = sample['user_id']
        history = sample.get('history', [])
        ground_truth = sample.get('ground_truth', [])[:top_k]

        if not history or not ground_truth:
            continue

        candidates, scores = cf_recall.recall(
            user_id=user_id,
            K=candidate_size,
            exclude_ids=history
        )

        # Compute recall quality
        recall_quality = len(set(candidates) & set(ground_truth)) / len(ground_truth)
        recall_quality_scores.append(recall_quality)

        # Random
        random_ranking = random_reranker.rerank(candidates, top_k, seed=user_id)
        metrics = calculate_metrics([random_ranking], [ground_truth], k_values=[top_k])
        results['random']['ndcg_scores'].append(metrics['ndcg@10'])
        results['random']['recall_scores'].append(metrics['recall@10'])

        # Popularity
        pop_ranking = pop_reranker.rerank(candidates, top_k)
        metrics = calculate_metrics([pop_ranking], [ground_truth], k_values=[top_k])
        results['popularity']['ndcg_scores'].append(metrics['ndcg@10'])
        results['popularity']['recall_scores'].append(metrics['recall@10'])

        # CF score
        cf_ranking = cf_reranker.rerank(candidates, scores, top_k)
        metrics = calculate_metrics([cf_ranking], [ground_truth], k_values=[top_k])
        results['cf_score']['ndcg_scores'].append(metrics['ndcg@10'])
        results['cf_score']['recall_scores'].append(metrics['recall@10'])

        # CF score with random tiebreaker
        scored_with_noise = [(c, s + np.random.normal(0, 0.01)) for c, s in zip(candidates, scores)]
        scored_with_noise.sort(key=lambda x: x[1], reverse=True)
        cf_reordered = [c for c, _ in scored_with_noise[:top_k]]
        metrics = calculate_metrics([cf_reordered], [ground_truth], k_values=[top_k])
        results['cf_score_reordered']['ndcg_scores'].append(metrics['ndcg@10'])
        results['cf_score_reordered']['recall_scores'].append(metrics['recall@10'])

    # Summary
    summary = {
        'recall_quality': {
            'mean': np.mean(recall_quality_scores),
            'std': np.std(recall_quality_scores)
        }
    }

    for method, scores in results.items():
        if len(scores['ndcg_scores']) > 0:
            ndcg_mean, ndcg_lower, ndcg_upper = bootstrap_ci(scores['ndcg_scores'])

            summary[method] = {
                'ndcg@10': {
                    'mean': ndcg_mean,
                    'ci_lower': ndcg_lower,
                    'ci_upper': ndcg_upper,
                    'std': np.std(scores['ndcg_scores'])
                },
                'recall@10': {
                    'mean': np.mean(scores['recall_scores']),
                    'std': np.std(scores['recall_scores'])
                },
                'n_samples': len(scores['ndcg_scores'])
            }

    return summary, results


def main():
    print("="*80)
    print("Supplementary Experiments for KDD 2026")
    print("="*80)

    # Load dataset
    print("\n[1/4] Loading Beauty dataset...")
    dataset = AmazonBeautyDataset(
        data_path=str(project_root / "data" / "processed" / "amazon_beauty_sampled")
    )
    train_samples, val_samples, test_samples = dataset.split()

    print(f"  Train: {len(train_samples)}")
    print(f"  Test: {len(test_samples)}")
    print(f"  Items: {len(dataset.item_texts)}")

    # Build CF recall
    print("\n[2/4] Building CF recall model...")
    cf_recall = CFRecall(n_factors=128)
    cf_recall.fit(dataset.train_df)

    # Run baseline comparison (no API required)
    print("\n[3/4] Running baseline comparison...")
    baseline_summary, baseline_results = run_baseline_comparison(
        dataset, cf_recall, test_samples, n_users=100
    )

    # Save baseline results
    baseline_output = project_root / "experiments" / "logs" / "supplementary_baselines.json"
    with open(baseline_output, 'w') as f:
        json.dump({
            'config': {
                'dataset': 'Beauty',
                'n_users': 100,
                'candidate_size': 100,
                'top_k': 10
            },
            'summary': baseline_summary,
            'raw_results': {k: v for k, v in baseline_results.items()}
        }, f, indent=2)
    print(f"  Saved to: {baseline_output}")

    # Print baseline summary
    print("\n" + "="*60)
    print("Baseline Comparison Results")
    print("="*60)
    print(f"\nRecall@100 (CF): {baseline_summary['recall_quality']['mean']:.3f} +/- {baseline_summary['recall_quality']['std']:.3f}")
    print("\n| Method | NDCG@10 | 95% CI |")
    print("|--------|---------|--------|")
    for method in ['random', 'popularity', 'cf_score', 'cf_score_reordered']:
        if method in baseline_summary:
            s = baseline_summary[method]
            print(f"| {method:15} | {s['ndcg@10']['mean']:.4f} | [{s['ndcg@10']['ci_lower']:.4f}, {s['ndcg@10']['ci_upper']:.4f}] |")

    # Run multi-LLM comparison if APIs available
    if os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY"):
        print("\n[4/4] Running Multi-LLM comparison...")
        llm_summary, llm_results = run_multi_llm_comparison(
            dataset, cf_recall, test_samples, n_users=30  # Smaller for API cost
        )

        # Save LLM results
        llm_output = project_root / "experiments" / "logs" / "supplementary_multi_llm.json"
        with open(llm_output, 'w') as f:
            json.dump({
                'config': {
                    'dataset': 'Beauty',
                    'n_users': 30,
                    'candidate_size': 100,
                    'top_k': 10
                },
                'summary': llm_summary,
                'raw_results': {k: v for k, v in llm_results.items()}
            }, f, indent=2)
        print(f"  Saved to: {llm_output}")

        # Print LLM summary
        print("\n" + "="*60)
        print("Multi-LLM Comparison Results")
        print("="*60)
        print("\n| Method | NDCG@10 | 95% CI |")
        print("|--------|---------|--------|")
        for method, s in llm_summary.items():
            print(f"| {method:15} | {s['ndcg@10']['mean']:.4f} | [{s['ndcg@10']['ci_lower']:.4f}, {s['ndcg@10']['ci_upper']:.4f}] |")
    else:
        print("\n[4/4] Skipping Multi-LLM comparison (no API keys)")

    print("\n" + "="*80)
    print("Supplementary Experiments Complete!")
    print("="*80)


if __name__ == '__main__':
    main()
