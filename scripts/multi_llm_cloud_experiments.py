"""
Multi-LLM Cloud Experiments for KDD 2026 Paper
==============================================

This script runs experiments using Ollama Cloud API with large-scale models:
- DeepSeek-V3.1 (671B) - State-of-the-art reasoning model
- Qwen3 (80B) - Efficient large model
- Llama 3.3 (70B) - Meta's latest model

Goal: Compare multiple LLMs on recommendation reranking to reach "strong accept"
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
import requests

# Load environment variables
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'experiments' / 'realistic_recall'))

from data.amazon_beauty_loader import AmazonBeautyDataset
from utils.metrics import calculate_metrics
from cf_recall import CFRecall


# ============================================================================
# Ollama Cloud API Wrapper
# ============================================================================

class OllamaCloudReranker:
    """Ollama Cloud API-based reranker for large models"""

    def __init__(self, model: str = "deepseek-v3.1"):
        self.api_key = os.getenv("OLLAMA_API_KEY")
        if not self.api_key:
            raise ValueError("OLLAMA_API_KEY not set in environment")

        self.model = model
        # Try official Ollama Cloud endpoint
        self.base_url = "https://ollama.com/api"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def rerank(self, prompt: str, candidates: List[int], top_k: int = 10) -> List[int]:
        """Rerank candidates using Ollama Cloud API"""
        try:
            # Ollama uses /chat endpoint with different format
            system_msg = "You are a recommendation system expert. Analyze user preferences from their history and rank candidate items by relevance. Output only item IDs in order of relevance."

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 300
                }
            }

            # Try chat endpoint
            response = requests.post(
                f"{self.base_url}/chat",
                headers=self.headers,
                json=payload,
                timeout=120
            )

            if response.status_code == 200:
                result = response.json()
                # Ollama returns "message" with "content"
                if "message" in result:
                    generated_text = result["message"]["content"]
                elif "choices" in result:
                    generated_text = result["choices"][0]["message"]["content"]
                else:
                    generated_text = str(result)
                return self._parse_item_ids(generated_text, candidates, top_k)
            else:
                print(f"\nOllama API Error {response.status_code}: {response.text[:300]}")
                return candidates[:top_k]

        except Exception as e:
            print(f"\nOllama Cloud error ({self.model}): {e}")
            return candidates[:top_k]

    def _parse_item_ids(self, text: str, candidates: List[int], top_k: int) -> List[int]:
        """Parse item IDs from generated text"""
        import re
        item_ids = []
        candidates_set = set(candidates)

        # Try multiple parsing patterns
        patterns = [
            r'[Ii]tem\s+(\d+)',
            r'#(\d+)',
            r'ID[:\s]+(\d+)',
            r'\b(\d+)\b',  # Any number as fallback
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

        # Fallback to candidates order if parsing fails
        for c in candidates:
            if c not in item_ids:
                item_ids.append(c)
            if len(item_ids) >= top_k:
                break

        return item_ids[:top_k]


# ============================================================================
# OpenAI API Wrapper (for comparison)
# ============================================================================

class OpenAIReranker:
    """OpenAI API-based reranker"""

    def __init__(self, model: str = "gpt-4o-mini"):
        import openai
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def rerank(self, prompt: str, candidates: List[int], top_k: int = 10) -> List[int]:
        """Rerank candidates using OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a recommendation system. Analyze user preferences and rank items. Output format: List item IDs as 'Item X' on separate lines, most relevant first."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )
            generated_text = response.choices[0].message.content
            return self._parse_item_ids(generated_text, candidates, top_k)
        except Exception as e:
            print(f"OpenAI error: {e}")
            return candidates[:top_k]

    def _parse_item_ids(self, text: str, candidates: List[int], top_k: int) -> List[int]:
        """Parse item IDs from generated text"""
        import re
        item_ids = []
        candidates_set = set(candidates)

        patterns = [
            r'[Ii]tem\s+(\d+)',
            r'#(\d+)',
            r'ID[:\s]+(\d+)',
            r'\b(\d+)\b',
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
        return np.random.permutation(candidates).tolist()[:top_k]


class PopularityReranker:
    """Popularity-based reranking baseline"""
    def __init__(self, train_df: pd.DataFrame):
        self.item_popularity = train_df.groupby('item_id').size().to_dict()

    def rerank(self, candidates: List[int], top_k: int = 10) -> List[int]:
        scored = [(c, self.item_popularity.get(c, 0)) for c in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored[:top_k]]


class CFScoreReranker:
    """CF score-based reranking"""
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
        text = item_texts.get(item_id, f"Item {item_id}")[:80]
        prompt += f"  {idx}. {text}\n"

    prompt += f"\nCandidate items to rank:\n"
    for idx, item_id in enumerate(candidates[:30], 1):  # Limit for token efficiency
        text = item_texts.get(item_id, f"Item {item_id}")[:80]
        prompt += f"  {idx}. Item {item_id}: {text}\n"

    prompt += f"\nOutput the top {top_k} most relevant items.\n"
    prompt += "Format: List each as 'Item <ID>' on a separate line, most relevant first.\n"

    return prompt


# ============================================================================
# Bootstrap Confidence Intervals
# ============================================================================

def bootstrap_ci(scores: List[float], n_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval"""
    scores = np.array(scores)
    if len(scores) == 0:
        return 0.0, 0.0, 0.0

    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        bootstrap_means.append(np.mean(sample))

    bootstrap_means = np.array(bootstrap_means)
    alpha = 1 - ci
    ci_lower = np.percentile(bootstrap_means, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)

    return np.mean(scores), ci_lower, ci_upper


# ============================================================================
# Main Experiment Runner
# ============================================================================

def run_multi_llm_experiment(
    dataset,
    cf_recall: CFRecall,
    test_samples: List[Dict],
    n_users: int = 50,
    top_k: int = 10,
    candidate_size: int = 100,
    models_to_test: List[str] = None
) -> Dict:
    """
    Compare multiple LLMs on recommendation reranking

    Models tested:
    - Baselines: Random, PopRec, CF Score
    - Cloud LLMs: DeepSeek-V3.1, Qwen3, Llama3.3
    - OpenAI: GPT-4o-mini
    """
    print("\n" + "="*80)
    print("Multi-LLM Cloud Experiment")
    print("="*80)

    # Default models to test (correct Ollama Cloud model names)
    if models_to_test is None:
        models_to_test = [
            "qwen3-next:80b",      # Qwen3 80B - efficient large model
            "deepseek-v3.2",       # DeepSeek V3.2 - latest version
            "gemma3:27b",          # Gemma3 27B - Google's model
        ]

    # Initialize results
    results = {
        'random': {'ndcg_scores': [], 'recall_scores': [], 'times': []},
        'popularity': {'ndcg_scores': [], 'recall_scores': [], 'times': []},
        'cf_score': {'ndcg_scores': [], 'recall_scores': [], 'times': []},
    }

    # Initialize rerankers
    rerankers = {}

    # Baselines
    random_reranker = RandomReranker()
    pop_reranker = PopularityReranker(dataset.train_df)
    cf_reranker = CFScoreReranker()

    # Ollama Cloud models
    ollama_api_key = os.getenv("OLLAMA_API_KEY")
    if ollama_api_key:
        for model in models_to_test:
            try:
                rerankers[model] = OllamaCloudReranker(model=model)
                results[model] = {'ndcg_scores': [], 'recall_scores': [], 'times': []}
                print(f"  ✓ Initialized {model}")
            except Exception as e:
                print(f"  ✗ Failed to init {model}: {e}")
    else:
        print("  ⚠ OLLAMA_API_KEY not set, skipping cloud models")

    # OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            rerankers['gpt-4o-mini'] = OpenAIReranker(model="gpt-4o-mini")
            results['gpt-4o-mini'] = {'ndcg_scores': [], 'recall_scores': [], 'times': []}
            print(f"  ✓ Initialized GPT-4o-mini")
        except Exception as e:
            print(f"  ✗ Failed to init GPT-4o-mini: {e}")

    # Sample test users
    np.random.seed(42)
    valid_samples = [s for s in test_samples if len(s.get('history', [])) > 0]
    n_available = min(n_users, len(valid_samples))
    sampled_indices = np.random.choice(len(valid_samples), size=n_available, replace=False)
    test_subset = [valid_samples[i] for i in sampled_indices]

    print(f"\nRunning on {len(test_subset)} test users...")
    print(f"Models: {list(results.keys())}")

    recall_quality_scores = []

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

        # Compute recall quality
        recall_quality = len(set(candidates) & set(ground_truth)) / len(ground_truth)
        recall_quality_scores.append(recall_quality)

        # Build prompt
        prompt = build_enhanced_cot_prompt(history, candidates, dataset.item_texts, top_k)

        # ===== Baselines =====

        # Random
        t0 = time.time()
        random_ranking = random_reranker.rerank(candidates, top_k, seed=user_id)
        results['random']['times'].append(time.time() - t0)
        metrics = calculate_metrics([random_ranking], [ground_truth], k_values=[top_k])
        results['random']['ndcg_scores'].append(metrics['ndcg@10'])
        results['random']['recall_scores'].append(metrics['recall@10'])

        # Popularity
        t0 = time.time()
        pop_ranking = pop_reranker.rerank(candidates, top_k)
        results['popularity']['times'].append(time.time() - t0)
        metrics = calculate_metrics([pop_ranking], [ground_truth], k_values=[top_k])
        results['popularity']['ndcg_scores'].append(metrics['ndcg@10'])
        results['popularity']['recall_scores'].append(metrics['recall@10'])

        # CF Score
        t0 = time.time()
        cf_ranking = cf_reranker.rerank(candidates, scores, top_k)
        results['cf_score']['times'].append(time.time() - t0)
        metrics = calculate_metrics([cf_ranking], [ground_truth], k_values=[top_k])
        results['cf_score']['ndcg_scores'].append(metrics['ndcg@10'])
        results['cf_score']['recall_scores'].append(metrics['recall@10'])

        # ===== LLM Rerankers =====
        for model_name, reranker in rerankers.items():
            try:
                t0 = time.time()
                ranking = reranker.rerank(prompt, candidates, top_k)
                elapsed = time.time() - t0
                results[model_name]['times'].append(elapsed)

                metrics = calculate_metrics([ranking], [ground_truth], k_values=[top_k])
                results[model_name]['ndcg_scores'].append(metrics['ndcg@10'])
                results[model_name]['recall_scores'].append(metrics['recall@10'])

                # Rate limiting
                time.sleep(0.3)
            except Exception as e:
                print(f"\n  Error with {model_name}: {e}")
                results[model_name]['ndcg_scores'].append(0.0)
                results[model_name]['recall_scores'].append(0.0)
                results[model_name]['times'].append(0.0)

    # Compute summary with bootstrap CIs
    summary = {
        'recall_quality': {
            'mean': np.mean(recall_quality_scores),
            'std': np.std(recall_quality_scores)
        }
    }

    for method, scores in results.items():
        if len(scores['ndcg_scores']) > 0:
            ndcg_mean, ndcg_lower, ndcg_upper = bootstrap_ci(scores['ndcg_scores'])
            recall_mean, recall_lower, recall_upper = bootstrap_ci(scores['recall_scores'])
            avg_time = np.mean(scores['times']) if scores['times'] else 0

            summary[method] = {
                'ndcg@10': {
                    'mean': ndcg_mean,
                    'ci_lower': ndcg_lower,
                    'ci_upper': ndcg_upper,
                    'std': np.std(scores['ndcg_scores'])
                },
                'recall@10': {
                    'mean': recall_mean,
                    'ci_lower': recall_lower,
                    'ci_upper': recall_upper
                },
                'avg_time_s': avg_time,
                'n_samples': len(scores['ndcg_scores'])
            }

    return summary, results


def main():
    print("="*80)
    print("Multi-LLM Cloud Experiments for KDD 2026")
    print("="*80)

    # Check API keys
    print("\nChecking API keys...")
    ollama_key = os.getenv("OLLAMA_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if ollama_key:
        print(f"  ✓ OLLAMA_API_KEY: {ollama_key[:20]}...")
    else:
        print("  ✗ OLLAMA_API_KEY not found")

    if openai_key:
        print(f"  ✓ OPENAI_API_KEY: {openai_key[:20]}...")
    else:
        print("  ✗ OPENAI_API_KEY not found")

    # Load dataset
    print("\n[1/3] Loading Beauty dataset...")
    dataset = AmazonBeautyDataset(
        data_path=str(project_root / "data" / "processed" / "amazon_beauty_sampled")
    )
    train_samples, val_samples, test_samples = dataset.split()

    print(f"  Train: {len(train_samples)}")
    print(f"  Test: {len(test_samples)}")
    print(f"  Items: {len(dataset.item_texts)}")

    # Build CF recall
    print("\n[2/3] Building CF recall model...")
    cf_recall = CFRecall(n_factors=128)
    cf_recall.fit(dataset.train_df)

    # Run multi-LLM experiment
    print("\n[3/3] Running Multi-LLM experiment...")

    # Test with smaller subset first, then scale up
    summary, raw_results = run_multi_llm_experiment(
        dataset=dataset,
        cf_recall=cf_recall,
        test_samples=test_samples,
        n_users=50,  # Start with 50 users
        top_k=10,
        candidate_size=100,
        models_to_test=["qwen3-next:80b", "deepseek-v3.2", "gemma3:27b"]
    )

    # Save results
    output_path = project_root / "experiments" / "logs" / "multi_llm_cloud_comparison.json"
    with open(output_path, 'w') as f:
        json.dump({
            'config': {
                'dataset': 'Beauty',
                'n_users': 50,
                'candidate_size': 100,
                'top_k': 10,
                'models': list(raw_results.keys())
            },
            'summary': summary,
            'raw_results': {k: {
                'ndcg_scores': v['ndcg_scores'],
                'recall_scores': v['recall_scores'],
                'times': v['times']
            } for k, v in raw_results.items()}
        }, f, indent=2)

    print(f"\n✓ Results saved to: {output_path}")

    # Print summary table
    print("\n" + "="*80)
    print("Multi-LLM Comparison Results")
    print("="*80)

    print(f"\nRecall@100 (CF): {summary['recall_quality']['mean']:.3f} ± {summary['recall_quality']['std']:.3f}")

    print("\n| Method | NDCG@10 | 95% CI | Avg Time |")
    print("|--------|---------|--------|----------|")

    # Sort by NDCG
    methods = [(m, s) for m, s in summary.items() if m != 'recall_quality']
    methods.sort(key=lambda x: x[1].get('ndcg@10', {}).get('mean', 0), reverse=True)

    for method, stats in methods:
        if 'ndcg@10' in stats:
            ndcg = stats['ndcg@10']
            avg_time = stats.get('avg_time_s', 0)
            print(f"| {method:18} | {ndcg['mean']:.4f} | [{ndcg['ci_lower']:.4f}, {ndcg['ci_upper']:.4f}] | {avg_time:.2f}s |")

    # Compute improvements over CF baseline
    cf_ndcg = summary.get('cf_score', {}).get('ndcg@10', {}).get('mean', 0)
    if cf_ndcg > 0:
        print("\n| Method | vs CF Score |")
        print("|--------|-------------|")
        for method, stats in methods:
            if method != 'cf_score' and 'ndcg@10' in stats:
                ndcg = stats['ndcg@10']['mean']
                improvement = (ndcg - cf_ndcg) / cf_ndcg * 100 if cf_ndcg > 0 else 0
                sign = "+" if improvement >= 0 else ""
                print(f"| {method:18} | {sign}{improvement:.1f}% |")

    print("\n" + "="*80)
    print("Experiment Complete!")
    print("="*80)


if __name__ == '__main__':
    main()
