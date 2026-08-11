#!/usr/bin/env python3
"""
LLM Reranking on New Datasets
================================
Runs actual LLM (Qwen2.5-7B via Ollama) reranking on new datasets
to validate the ceiling claim beyond the original 3 Amazon datasets.

Uses a single prompt strategy (P2 Direct) as representative.

Usage:
    python scripts/llm_reranking_new_datasets.py --datasets toys,mind --n_users 100
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import time
import numpy as np
import subprocess
from datetime import datetime
from typing import List
from tqdm import tqdm
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from data_interface import load_dataset, Dataset


class CFRecall:
    def __init__(self, n_factors=128):
        self.n_factors = n_factors

    def fit(self, train_df):
        users = train_df['user_id'].unique()
        items = train_df['item_id'].unique()
        self.user_to_idx = {u: i for i, u in enumerate(users)}
        self.item_to_idx = {i: idx for idx, i in enumerate(items)}
        self.idx_to_item = {idx: i for i, idx in self.item_to_idx.items()}
        rows = [self.user_to_idx[u] for u in train_df['user_id']]
        cols = [self.item_to_idx[i] for i in train_df['item_id']]
        data = np.ones(len(rows))
        user_item = csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))
        n_components = min(self.n_factors, min(user_item.shape) - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_factors = svd.fit_transform(user_item)
        self.item_factors = svd.components_.T

    def recall(self, user_id, K, exclude_ids):
        if user_id not in self.user_to_idx:
            return [], []
        u_idx = self.user_to_idx[user_id]
        scores = np.dot(self.item_factors, self.user_factors[u_idx])
        for item in exclude_ids:
            if item in self.item_to_idx:
                scores[self.item_to_idx[item]] = -np.inf
        top_indices = np.argsort(scores)[::-1][:K]
        candidates = [self.idx_to_item[idx] for idx in top_indices if scores[idx] > -np.inf]
        item_scores = [float(scores[self.item_to_idx[c]]) for c in candidates]
        return candidates[:K], item_scores[:K]


def compute_ndcg(ranked_list, ground_truth, k=10):
    gt_set = set(ground_truth)
    dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(ranked_list[:k]) if item in gt_set)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(gt_set), k)))
    return dcg / ideal if ideal > 0 else 0


def call_ollama(prompt: str, model: str = "qwen2.5:7b", timeout: int = 30) -> str:
    """Call Ollama API for LLM reranking."""
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True, text=True, timeout=timeout
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return ""
    except Exception:
        return ""


def build_prompt(history_texts: List[str], candidate_texts: List[str]) -> str:
    """P2 Direct ranking prompt."""
    history_str = "\n".join(f"- {t}" for t in history_texts[-10:])
    candidates_str = "\n".join(f"{i+1}. {t}" for i, t in enumerate(candidate_texts[:20]))

    return f"""A user has interacted with these items:
{history_str}

Rank the following candidates from most to least likely to interest this user.
Return ONLY a comma-separated list of numbers (e.g., 3,1,5,2,4).

Candidates:
{candidates_str}

Ranking:"""


def parse_ranking(response: str, n_candidates: int) -> List[int]:
    """Parse LLM ranking response into indices."""
    try:
        nums = []
        for token in response.replace('\n', ',').split(','):
            token = token.strip().rstrip('.')
            if token.isdigit():
                idx = int(token) - 1  # 1-indexed to 0-indexed
                if 0 <= idx < n_candidates and idx not in nums:
                    nums.append(idx)
        # Fill missing positions
        for i in range(n_candidates):
            if i not in nums:
                nums.append(i)
        return nums[:n_candidates]
    except Exception:
        return list(range(n_candidates))


def run_dataset(ds: Dataset, n_users: int = 100, model: str = "qwen2.5:7b",
                seed: int = 42, K: int = 50):
    """Run LLM reranking on a dataset."""
    print(f"\n{'='*60}")
    print(f"LLM RERANKING: {ds.name} (model={model}, n={n_users}, K={K})")
    print(f"{'='*60}")

    np.random.seed(seed)
    test_samples = ds.get_test_samples(n_users=n_users, seed=seed)
    print(f"  Test users: {len(test_samples)}")

    cf = CFRecall(n_factors=min(128, ds.n_items - 1))
    cf.fit(ds.train_df)

    # Evaluate CF baseline and Oracle
    cf_ndcgs = []
    oracle_ndcgs = []
    llm_ndcgs = []
    n_llm_success = 0

    for i, sample in enumerate(tqdm(test_samples, desc="LLM reranking")):
        candidates, scores = cf.recall(sample['user_id'], K, sample['history'])
        gt = sample['ground_truth']

        if not candidates:
            cf_ndcgs.append(0); oracle_ndcgs.append(0); llm_ndcgs.append(0)
            continue

        # CF baseline
        cf_ndcgs.append(compute_ndcg(candidates, gt))

        # Oracle
        gt_set = set(gt)
        oracle_ranked = [c for c in candidates if c in gt_set] + [c for c in candidates if c not in gt_set]
        oracle_ndcgs.append(compute_ndcg(oracle_ranked, gt))

        # LLM reranking (on first 20 candidates to keep prompt short)
        n_rerank = min(20, len(candidates))
        history_texts = [(ds.get_item_text(h) or f'item {h}')[:80] for h in sample['history'][-10:]]
        candidate_texts = [(ds.get_item_text(c) or f'item {c}')[:80] for c in candidates[:n_rerank]]

        prompt = build_prompt(history_texts, candidate_texts)
        response = call_ollama(prompt, model=model, timeout=30)

        if response:
            ranking = parse_ranking(response, n_rerank)
            reranked = [candidates[idx] for idx in ranking] + candidates[n_rerank:]
            llm_ndcgs.append(compute_ndcg(reranked, gt))
            n_llm_success += 1
        else:
            llm_ndcgs.append(cf_ndcgs[-1])  # fallback to CF

    # Results
    cf_mean = float(np.mean(cf_ndcgs))
    oracle_mean = float(np.mean(oracle_ndcgs))
    llm_mean = float(np.mean(llm_ndcgs))
    recall = float(np.mean([1 if compute_ndcg(
        cf.recall(s['user_id'], K, s['history'])[0], s['ground_truth'], k=K) > 0
        else 0 for s in test_samples[:50]]))

    print(f"\n  Results:")
    print(f"    CF-Score:  NDCG@10 = {cf_mean:.4f}")
    print(f"    LLM ({model}): NDCG@10 = {llm_mean:.4f} ({n_llm_success}/{len(test_samples)} success)")
    print(f"    Oracle:    NDCG@10 = {oracle_mean:.4f}")
    print(f"    LLM vs CF: {(llm_mean - cf_mean):.4f} ({'better' if llm_mean > cf_mean else 'worse/same'})")
    print(f"    Ceiling utilization (LLM): {llm_mean/oracle_mean*100:.1f}%" if oracle_mean > 0 else "    N/A")

    return {
        'dataset': ds.key,
        'name': ds.name,
        'model': model,
        'n_users': len(test_samples),
        'K': K,
        'n_llm_success': n_llm_success,
        'cf_ndcg': cf_mean,
        'llm_ndcg': llm_mean,
        'oracle_ndcg': oracle_mean,
        'llm_vs_cf': float(llm_mean - cf_mean),
        'ceiling_utilization_llm': float(llm_mean / oracle_mean * 100) if oracle_mean > 0 else 0,
        'ceiling_utilization_cf': float(cf_mean / oracle_mean * 100) if oracle_mean > 0 else 0,
        'timestamp': datetime.now().isoformat()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='toys,mind')
    parser.add_argument('--n_users', type=int, default=100)
    parser.add_argument('--model', type=str, default='qwen2.5:7b')
    parser.add_argument('--K', type=int, default=50,
                        help='Candidate set size (smaller than 100 for LLM prompt length)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    all_results = {}
    for ds_name in args.datasets.split(','):
        ds = load_dataset(ds_name)
        result = run_dataset(ds, args.n_users, args.model, args.seed, args.K)
        all_results[ds_name] = result

    # Summary
    print("\n" + "=" * 60)
    print("LLM RERANKING SUMMARY")
    print("=" * 60)
    for ds, r in all_results.items():
        print(f"  {ds:12s}: CF={r['cf_ndcg']:.4f}, LLM={r['llm_ndcg']:.4f}, "
              f"Oracle={r['oracle_ndcg']:.4f}, LLM ceil={r['ceiling_utilization_llm']:.0f}%")

    output_path = project_root / 'experiments' / 'logs' / 'llm_reranking_new_datasets.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    main()
