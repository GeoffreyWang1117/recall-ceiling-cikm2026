#!/usr/bin/env python3
"""
Prompt Complexity × Recall Interaction Analysis
=================================================
Tests whether more sophisticated prompting strategies help under different
recall levels. We simulate different recall conditions by controlling the
candidate set quality, then compare prompt strategies at each level.

Since we can't run actual LLM calls at every combination, we use a
controlled simulation: at each recall level, we measure the theoretical
ceiling and the actual reranking performance of CF-score vs shuffled orderings
as a proxy for "prompt quality" (perfect ordering = ideal prompt, random = bad prompt).

This validates the claim: "prompt engineering provides zero value under low recall."

Usage:
    python scripts/prompt_complexity_analysis.py --datasets beauty,toys --n_users 200
    python scripts/prompt_complexity_analysis.py --all --n_users 200
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List
from tqdm import tqdm
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from data_interface import load_dataset, load_all, Dataset


# ============================================================================
# RETRIEVAL
# ============================================================================

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
        self.all_items = list(self.item_to_idx.keys())

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


# ============================================================================
# NDCG
# ============================================================================

def compute_ndcg(ranked_list, ground_truth, k=10):
    gt_set = set(ground_truth)
    dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(ranked_list[:k]) if item in gt_set)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(gt_set), k)))
    return dcg / ideal if ideal > 0 else 0


# ============================================================================
# CONTROLLED RECALL EXPERIMENT
# ============================================================================

def controlled_recall_experiment(cf, test_samples, recall_levels, K=100):
    """
    For each recall level, create candidate sets with that approximate recall,
    then compare reranking strategies.

    recall_levels: target recall percentages (e.g., [0.05, 0.10, 0.20, 0.50, 1.0])

    Strategy to control recall:
    - Start with CF candidates
    - Inject ground truth items to achieve target recall
    - Or remove them to achieve lower recall
    """
    results = []

    for target_recall in recall_levels:
        ndcg_cf_order = []
        ndcg_random = []
        ndcg_oracle = []
        ndcg_reverse = []
        actual_recalls = []

        for sample in test_samples:
            candidates, scores = cf.recall(sample['user_id'], K, sample['history'])
            gt = set(sample['ground_truth'])

            if not candidates:
                continue

            # Current recall
            hits_in_candidates = [c for c in candidates if c in gt]
            non_hits = [c for c in candidates if c not in gt]
            current_recall = len(hits_in_candidates) / len(gt) if gt else 0

            # Adjust candidates to approximate target recall
            n_target_hits = max(0, int(target_recall * len(gt)))

            if n_target_hits <= len(hits_in_candidates):
                # Need to REMOVE some hits (reduce recall)
                keep_hits = hits_in_candidates[:n_target_hits]
                # Replace removed hits with random non-interacted items
                all_items = list(cf.item_to_idx.keys())
                exclude = set(sample['history']) | gt | set(candidates)
                fillers = [i for i in all_items if i not in exclude]
                n_fill = len(hits_in_candidates) - n_target_hits
                if len(fillers) >= n_fill:
                    fill_items = list(np.random.choice(fillers, size=n_fill, replace=False))
                else:
                    fill_items = fillers
                adjusted_candidates = keep_hits + non_hits + fill_items
                adjusted_candidates = adjusted_candidates[:K]
            else:
                # Need to ADD more hits (increase recall)
                # Inject ground truth items not yet in candidates
                missing_gt = [g for g in gt if g not in set(candidates)]
                n_inject = min(n_target_hits - len(hits_in_candidates), len(missing_gt))
                injected = missing_gt[:n_inject]
                # Remove some non-hits to make room
                adjusted_candidates = hits_in_candidates + injected + non_hits
                adjusted_candidates = adjusted_candidates[:K]

            actual_hits = len([c for c in adjusted_candidates if c in gt])
            actual_recall = actual_hits / len(gt) if gt else 0
            actual_recalls.append(actual_recall)

            if not adjusted_candidates:
                continue

            gt_list = list(gt)

            # Strategy 1: CF order (as-is, proxy for "basic prompt")
            ndcg_cf_order.append(compute_ndcg(adjusted_candidates, gt_list))

            # Strategy 2: Random order (proxy for "bad prompt")
            perm = np.random.permutation(len(adjusted_candidates))
            random_ranked = [adjusted_candidates[i] for i in perm]
            ndcg_random.append(compute_ndcg(random_ranked, gt_list))

            # Strategy 3: Oracle (proxy for "perfect prompt")
            gt_items = [c for c in adjusted_candidates if c in gt]
            non_gt = [c for c in adjusted_candidates if c not in gt]
            oracle_ranked = gt_items + non_gt
            ndcg_oracle.append(compute_ndcg(oracle_ranked, gt_list))

            # Strategy 4: Reverse CF (proxy for "adversarial prompt")
            ndcg_reverse.append(compute_ndcg(list(reversed(adjusted_candidates)), gt_list))

        if not actual_recalls:
            continue

        mean_actual_recall = np.mean(actual_recalls)

        results.append({
            'target_recall': target_recall,
            'actual_recall': float(mean_actual_recall),
            'n_users': len(actual_recalls),
            'strategies': {
                'oracle': {
                    'ndcg_mean': float(np.mean(ndcg_oracle)),
                    'ndcg_std': float(np.std(ndcg_oracle))
                },
                'cf_order': {
                    'ndcg_mean': float(np.mean(ndcg_cf_order)),
                    'ndcg_std': float(np.std(ndcg_cf_order))
                },
                'random': {
                    'ndcg_mean': float(np.mean(ndcg_random)),
                    'ndcg_std': float(np.std(ndcg_random))
                },
                'reverse': {
                    'ndcg_mean': float(np.mean(ndcg_reverse)),
                    'ndcg_std': float(np.std(ndcg_reverse))
                }
            },
            # Key metric: does prompt quality matter?
            'oracle_vs_cf_gap': float(np.mean(ndcg_oracle)) - float(np.mean(ndcg_cf_order)),
            'cf_vs_random_gap': float(np.mean(ndcg_cf_order)) - float(np.mean(ndcg_random)),
        })

    return results


def run_dataset(ds: Dataset, n_users: int, seed: int):
    """Run prompt complexity analysis on a single dataset."""
    print(f"\n{'='*60}")
    print(f"PROMPT × RECALL: {ds.name} (density={ds.density*100:.3f}%)")
    print(f"{'='*60}")

    np.random.seed(seed)
    test_samples = ds.get_test_samples(n_users=n_users, seed=seed)
    print(f"  Test users: {len(test_samples)}")

    cf = CFRecall(n_factors=min(128, ds.n_items - 1))
    cf.fit(ds.train_df)

    # Test at different recall levels
    recall_levels = [0.0, 0.05, 0.10, 0.20, 0.50, 1.0]
    results = controlled_recall_experiment(cf, test_samples, recall_levels, K=100)

    # Print summary
    print(f"\n  {'Recall':>8s} {'Oracle':>8s} {'CF-order':>8s} {'Random':>8s} {'Gap(O-CF)':>9s} {'Gap(CF-R)':>9s}")
    print("  " + "-" * 55)
    for r in results:
        s = r['strategies']
        print(f"  {r['actual_recall']*100:>7.1f}% "
              f"{s['oracle']['ndcg_mean']:.4f}   "
              f"{s['cf_order']['ndcg_mean']:.4f}   "
              f"{s['random']['ndcg_mean']:.4f}   "
              f"{r['oracle_vs_cf_gap']:.4f}    "
              f"{r['cf_vs_random_gap']:.4f}")

    # Key finding
    realistic_result = next((r for r in results if r['actual_recall'] < 0.15), None)
    high_recall_result = next((r for r in results if r['actual_recall'] > 0.40), None)

    if realistic_result and high_recall_result:
        low_gap = realistic_result['cf_vs_random_gap']
        high_gap = high_recall_result['cf_vs_random_gap']
        print(f"\n  Key finding:")
        print(f"    Low recall ({realistic_result['actual_recall']*100:.0f}%): "
              f"CF vs Random gap = {low_gap:.4f}")
        print(f"    High recall ({high_recall_result['actual_recall']*100:.0f}%): "
              f"CF vs Random gap = {high_gap:.4f}")
        if high_gap > 0 and low_gap >= 0:
            ratio = low_gap / high_gap if high_gap > 0 else 0
            print(f"    Prompt quality matters {ratio:.0%} as much under low recall")

    return {
        'dataset': ds.key,
        'name': ds.name,
        'density': ds.density,
        'catalog_size': ds.catalog_size,
        'n_users': len(test_samples),
        'recall_levels': results
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default=None)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--n_users', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.all:
        datasets = load_all()
    elif args.datasets:
        datasets = {name: load_dataset(name) for name in args.datasets.split(',')}
    else:
        datasets = {name: load_dataset(name) for name in ['beauty', 'toys', 'sports', 'mind']}

    print("=" * 60)
    print("PROMPT COMPLEXITY × RECALL INTERACTION")
    print("=" * 60)

    all_results = {}
    for key, ds in datasets.items():
        result = run_dataset(ds, args.n_users, args.seed)
        all_results[key] = result

    # Cross-dataset summary
    print("\n" + "=" * 60)
    print("CROSS-DATASET: DOES PROMPT QUALITY MATTER?")
    print("=" * 60)
    print(f"{'Dataset':15s} {'Low-R Gap':>10s} {'High-R Gap':>10s} {'Ratio':>7s}")
    print("-" * 45)

    for key, res in all_results.items():
        levels = res['recall_levels']
        low = next((l for l in levels if l['actual_recall'] < 0.15), None)
        high = next((l for l in levels if l['actual_recall'] > 0.40), None)
        if low and high:
            lg = low['cf_vs_random_gap']
            hg = high['cf_vs_random_gap']
            ratio = lg / hg if hg > 0 else 0
            print(f"{key:15s} {lg:>10.4f} {hg:>10.4f} {ratio:>6.0%}")

    # Save
    output_path = project_root / 'experiments' / 'logs' / 'prompt_complexity_analysis.json'
    with open(output_path, 'w') as f:
        json.dump({
            'datasets': all_results,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
