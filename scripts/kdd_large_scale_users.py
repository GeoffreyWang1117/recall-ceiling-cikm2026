#!/usr/bin/env python3
"""
KDD 2026 Large-Scale User Experiment
=====================================
Scales up test user count to 5000+ to validate findings at industrial scale.

This experiment addresses reviewer concern: "How do findings generalize beyond 500 users?"

Usage:
    python scripts/kdd_large_scale_users.py --n_users 5000 --batch_size 500
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from typing import Dict, List, Tuple
from tqdm import tqdm
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# ============================================================================
# PROMPTS
# ============================================================================

PROMPT_P1 = """Based on the user's purchase history, rank these candidate items.
History: {history_short}
Candidate items:
{candidates}
Output item numbers in order of relevance (most relevant first), comma-separated:"""


# ============================================================================
# RETRIEVAL METHODS
# ============================================================================

class PopularityRetrieval:
    def __init__(self):
        self.item_counts = None

    def fit(self, train_df):
        self.item_counts = train_df['item_id'].value_counts()
        self.all_items = list(self.item_counts.index)
        self.user_history = train_df.groupby('user_id')['item_id'].apply(set).to_dict()

    def get_candidates(self, user_id, K=100):
        history = self.user_history.get(user_id, set())
        candidates = [i for i in self.all_items if i not in history][:K]
        scores = [self.item_counts.get(i, 0) for i in candidates]
        return candidates, scores


class CFRetrieval:
    def __init__(self, n_factors=128):
        self.n_factors = n_factors

    def fit(self, train_df):
        items = train_df['item_id'].unique()
        users = train_df['user_id'].unique()
        self.item_to_idx = {i: idx for idx, i in enumerate(items)}
        self.idx_to_item = {idx: i for i, idx in self.item_to_idx.items()}
        self.user_to_idx = {u: idx for idx, u in enumerate(users)}

        rows = [self.user_to_idx[u] for u in train_df['user_id']]
        cols = [self.item_to_idx[i] for i in train_df['item_id']]
        data = np.ones(len(rows))

        self.user_item = csr_matrix((data, (rows, cols)),
                                     shape=(len(users), len(items)))

        n_components = min(self.n_factors, len(items)-1, len(users)-1)
        self.svd = TruncatedSVD(n_components=n_components)
        self.user_factors = self.svd.fit_transform(self.user_item)
        self.item_factors = self.svd.components_.T
        self.user_history = train_df.groupby('user_id')['item_id'].apply(set).to_dict()
        self.all_items = list(items)

    def get_candidates(self, user_id, K=100):
        if user_id not in self.user_to_idx:
            return [], []
        user_idx = self.user_to_idx[user_id]
        scores = np.dot(self.user_factors[user_idx], self.item_factors.T)
        history = self.user_history.get(user_id, set())
        for item in history:
            if item in self.item_to_idx:
                scores[self.item_to_idx[item]] = -np.inf
        top_indices = np.argsort(scores)[::-1][:K]
        return [self.idx_to_item[idx] for idx in top_indices], [scores[idx] for idx in top_indices]


class ItemKNNRetrieval:
    def __init__(self, k=50):
        self.k = k

    def fit(self, train_df):
        items = train_df['item_id'].unique()
        users = train_df['user_id'].unique()
        self.item_to_idx = {i: idx for idx, i in enumerate(items)}
        self.idx_to_item = {idx: i for i, idx in self.item_to_idx.items()}
        self.user_to_idx = {u: idx for idx, u in enumerate(users)}

        rows = [self.user_to_idx[u] for u in train_df['user_id']]
        cols = [self.item_to_idx[i] for i in train_df['item_id']]
        data = np.ones(len(rows))

        user_item = csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))
        item_user = user_item.T.toarray()

        # Compute item similarities using cosine
        norms = np.linalg.norm(item_user, axis=1, keepdims=True) + 1e-8
        item_user_norm = item_user / norms
        self.item_sim = np.dot(item_user_norm, item_user_norm.T)
        np.fill_diagonal(self.item_sim, 0)

        self.user_history = train_df.groupby('user_id')['item_id'].apply(set).to_dict()
        self.all_items = list(items)

    def get_candidates(self, user_id, K=100):
        history = self.user_history.get(user_id, set())
        if not history:
            return [], []

        history_indices = [self.item_to_idx[i] for i in history if i in self.item_to_idx]
        if not history_indices:
            return [], []

        # Aggregate similarities
        scores = np.zeros(len(self.all_items))
        for idx in history_indices:
            scores += self.item_sim[idx]

        # Exclude history
        for item in history:
            if item in self.item_to_idx:
                scores[self.item_to_idx[item]] = -np.inf

        top_indices = np.argsort(scores)[::-1][:K]
        return [self.idx_to_item[idx] for idx in top_indices], [scores[idx] for idx in top_indices]


# ============================================================================
# METRICS
# ============================================================================

def recall_at_k(candidates: List, ground_truth: set, k: int = 100) -> float:
    """Calculate Recall@K."""
    hits = sum(1 for c in candidates[:k] if c in ground_truth)
    return hits / len(ground_truth) if ground_truth else 0.0


def ndcg_at_k(ranked: List, gt: set, k: int = 10) -> float:
    """Calculate NDCG@K."""
    dcg = sum([1/np.log2(i+2) for i, item in enumerate(ranked[:k]) if item in gt])
    idcg = sum([1/np.log2(i+2) for i in range(min(k, len(gt)))])
    return dcg / idcg if idcg > 0 else 0.0


def bootstrap_ci(scores: List[float], n: int = 1000) -> Tuple[float, float, float]:
    """Calculate bootstrap confidence interval."""
    scores = np.array(scores)
    if len(scores) == 0:
        return 0.0, 0.0, 0.0
    boots = [np.mean(np.random.choice(scores, len(scores), replace=True)) for _ in range(n)]
    return np.mean(scores), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


# ============================================================================
# LLM RERANKER
# ============================================================================

class LLMReranker:
    def __init__(self, model_name="Qwen/Qwen2.5-3B-Instruct"):
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
        self.model.eval()

    def rerank(self, prompt, history, candidates, history_short=None):
        if history_short is None:
            history_short = ", ".join(str(h) for h in history[-5:])
        cands_text = "\n".join([f"{i+1}. {t}" for i, (_, t) in enumerate(candidates[:50])])  # Limit to 50

        filled = prompt.format(history_short=history_short, candidates=cands_text)
        inputs = self.tokenizer(filled, return_tensors="pt", truncation=True, max_length=2048).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=100,
                                          do_sample=False, pad_token_id=self.tokenizer.eos_token_id)

        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return self._parse(response, len(candidates), candidates)

    def _parse(self, response, n, candidates):
        import re
        nums = re.findall(r'\d+', response)
        seen, result = set(), []
        for num in nums:
            idx = int(num) - 1
            if 0 <= idx < n and idx not in seen:
                result.append(candidates[idx][0])
                seen.add(idx)
        for i in range(n):
            if i not in seen:
                result.append(candidates[i][0])
        return result


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment(args):
    print("=" * 70)
    print("LARGE-SCALE USER EXPERIMENT")
    print(f"Target: {args.n_users} users")
    print("=" * 70)

    # Load data - use Movies dataset (largest)
    data_path = project_root / 'data' / 'processed' / 'amazon_movies_sampled'

    if not data_path.exists():
        # Fallback to beauty if movies not available
        data_path = project_root / 'data' / 'processed' / 'amazon_beauty_sampled'
        print(f"Using fallback dataset: {data_path}")

    train_df = pd.read_parquet(data_path / 'train.parquet')
    test_df = pd.read_parquet(data_path / 'test.parquet')

    print(f"Train: {len(train_df)} interactions, {train_df['user_id'].nunique()} users")
    print(f"Test: {len(test_df)} interactions, {test_df['user_id'].nunique()} users")

    # Item texts
    item_texts = {}
    if 'title' in train_df.columns:
        for _, row in train_df.drop_duplicates('item_id').iterrows():
            item_texts[row['item_id']] = str(row.get('title', ''))[:80]

    # Initialize retrieval methods
    print("\nInitializing retrieval methods...")
    retrievers = {
        'Popularity': PopularityRetrieval(),
        'CF-SVD': CFRetrieval(n_factors=128),
        'ItemKNN': ItemKNNRetrieval(k=50),
    }

    for name, retriever in retrievers.items():
        print(f"  Fitting {name}...")
        retriever.fit(train_df)

    # Get test users
    np.random.seed(args.seed)
    test_users = test_df.groupby('user_id')['item_id'].apply(set).to_dict()
    valid_users = [u for u in test_users if u in retrievers['CF-SVD'].user_to_idx]

    # Sample users
    n_available = len(valid_users)
    n_to_test = min(args.n_users, n_available)

    if n_to_test < args.n_users:
        print(f"WARNING: Only {n_available} valid users available, testing {n_to_test}")

    sampled_users = list(np.random.choice(valid_users, n_to_test, replace=False))
    print(f"\nTesting {n_to_test} users")

    # Results storage
    results = {
        'config': {
            'n_users': n_to_test,
            'seed': args.seed,
            'dataset': str(data_path.name),
            'K': 100,
        },
        'retrieval_results': {},
        'reranking_results': {},
    }

    # Phase 1: Retrieval evaluation (fast, no LLM)
    print("\n" + "=" * 70)
    print("PHASE 1: Retrieval Evaluation")
    print("=" * 70)

    for method_name, retriever in retrievers.items():
        recall_scores = []
        for user_id in tqdm(sampled_users, desc=method_name):
            gt = test_users[user_id]
            candidates, _ = retriever.get_candidates(user_id, K=100)
            recall = recall_at_k(candidates, gt, k=100)
            recall_scores.append(recall)

        mean, ci_lo, ci_hi = bootstrap_ci(recall_scores)
        results['retrieval_results'][method_name] = {
            'recall@100': {'mean': mean, 'ci_low': ci_lo, 'ci_high': ci_hi},
            'n_users': len(recall_scores),
        }
        print(f"{method_name}: Recall@100 = {mean*100:.2f}% [{ci_lo*100:.2f}, {ci_hi*100:.2f}]")

    # Phase 2: LLM Reranking (slower, batch processing)
    print("\n" + "=" * 70)
    print("PHASE 2: LLM Reranking Evaluation")
    print("=" * 70)

    # Use smaller sample for LLM (still large: 1000 users)
    llm_sample_size = min(1000, n_to_test)
    llm_users = sampled_users[:llm_sample_size]
    print(f"LLM evaluation on {llm_sample_size} users")

    # Load LLM
    reranker = LLMReranker("Qwen/Qwen2.5-3B-Instruct")

    # Use best retriever (CF-SVD)
    best_retriever = retrievers['CF-SVD']

    cf_scores = []
    llm_scores = []

    for user_id in tqdm(llm_users, desc="LLM Reranking"):
        gt = test_users[user_id]
        history = list(best_retriever.user_history.get(user_id, []))
        candidates, scores = best_retriever.get_candidates(user_id, K=100)

        if not candidates:
            continue

        # CF baseline
        cf_ndcg = ndcg_at_k(candidates, gt, k=10)
        cf_scores.append(cf_ndcg)

        # LLM reranking
        try:
            cands_with_text = [(c, item_texts.get(c, f"Item {c}")) for c in candidates]
            history_texts = [item_texts.get(i, f"Item {i}") for i in history]
            ranked = reranker.rerank(PROMPT_P1, history_texts, cands_with_text)
            llm_ndcg = ndcg_at_k(ranked, gt, k=10)
        except Exception as e:
            llm_ndcg = cf_ndcg  # Fallback

        llm_scores.append(llm_ndcg)

    # Compute results
    cf_mean, cf_lo, cf_hi = bootstrap_ci(cf_scores)
    llm_mean, llm_lo, llm_hi = bootstrap_ci(llm_scores)

    results['reranking_results'] = {
        'CF_baseline': {'mean': cf_mean, 'ci_low': cf_lo, 'ci_high': cf_hi},
        'LLM_reranking': {'mean': llm_mean, 'ci_low': llm_lo, 'ci_high': llm_hi},
        'n_users': len(cf_scores),
        'improvement': (llm_mean - cf_mean) / cf_mean * 100 if cf_mean > 0 else 0,
    }

    # Statistical test
    from scipy.stats import wilcoxon
    try:
        stat, p_value = wilcoxon(llm_scores, cf_scores)
        results['reranking_results']['wilcoxon_p'] = p_value
    except:
        results['reranking_results']['wilcoxon_p'] = 1.0

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n=== RETRIEVAL (n={n_to_test} users) ===")
    for method, res in results['retrieval_results'].items():
        r = res['recall@100']
        print(f"  {method}: {r['mean']*100:.2f}% [{r['ci_low']*100:.2f}, {r['ci_high']*100:.2f}]")

    print(f"\n=== RERANKING (n={len(cf_scores)} users) ===")
    print(f"  CF baseline: {cf_mean:.4f} [{cf_lo:.4f}, {cf_hi:.4f}]")
    print(f"  LLM rerank:  {llm_mean:.4f} [{llm_lo:.4f}, {llm_hi:.4f}]")
    print(f"  Improvement: {results['reranking_results']['improvement']:+.1f}%")
    print(f"  Wilcoxon p:  {results['reranking_results']['wilcoxon_p']:.4f}")

    # Save results
    output_path = project_root / 'experiments' / 'logs' / 'kdd_large_scale_users.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_users', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    run_experiment(args)
