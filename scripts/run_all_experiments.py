#!/usr/bin/env python3
"""
Unified Experiment Runner
===========================
Runs the full recall ceiling pipeline on any dataset via the data_interface.
Combines retrieval, reranking, RAEP, and density analysis into a single script.

Usage:
    # Run on a single dataset
    python scripts/run_all_experiments.py --dataset beauty --n_users 500

    # Run on all datasets
    python scripts/run_all_experiments.py --all --n_users 500

    # Run on a specific domain
    python scripts/run_all_experiments.py --domain product --n_users 300

    # Quick smoke test
    python scripts/run_all_experiments.py --dataset beauty --n_users 50 --quick
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from typing import Dict, List
from tqdm import tqdm
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import wilcoxon

from data_interface import load_dataset, load_all, list_datasets, Dataset


# ============================================================================
# RETRIEVAL METHODS
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
        self.popularity = train_df['item_id'].value_counts().to_dict()

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
        if item_scores:
            mn, mx = min(item_scores), max(item_scores)
            if mx > mn:
                item_scores = [(s - mn) / (mx - mn) for s in item_scores]
        return candidates[:K], item_scores[:K]


class ItemKNNRecall:
    def __init__(self, n_neighbors=50):
        self.n_neighbors = n_neighbors

    def fit(self, train_df):
        users = train_df['user_id'].unique()
        items = train_df['item_id'].unique()
        self.user_to_idx = {u: i for i, u in enumerate(users)}
        self.item_to_idx = {i: idx for idx, i in enumerate(items)}
        self.idx_to_item = {idx: i for i, idx in self.item_to_idx.items()}
        rows = [self.user_to_idx[u] for u in train_df['user_id']]
        cols = [self.item_to_idx[i] for i in train_df['item_id']]
        data = np.ones(len(rows))
        self.user_item = csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))
        self.item_sim = cosine_similarity(self.user_item.T)
        np.fill_diagonal(self.item_sim, 0)

    def recall(self, user_id, K, exclude_ids):
        if user_id not in self.user_to_idx:
            return [], []
        u_idx = self.user_to_idx[user_id]
        user_items = self.user_item[u_idx].toarray().flatten()
        interacted = np.where(user_items > 0)[0]
        if len(interacted) == 0:
            return [], []
        scores = self.item_sim[interacted].sum(axis=0)
        for item in exclude_ids:
            if item in self.item_to_idx:
                scores[self.item_to_idx[item]] = -np.inf
        for idx in interacted:
            scores[idx] = -np.inf
        top_indices = np.argsort(scores)[::-1][:K]
        candidates = [self.idx_to_item[idx] for idx in top_indices if scores[idx] > -np.inf]
        item_scores = [float(scores[self.item_to_idx[c]]) for c in candidates]
        if item_scores:
            mn, mx = min(item_scores), max(item_scores)
            if mx > mn:
                item_scores = [(s - mn) / (mx - mn) for s in item_scores]
        return candidates[:K], item_scores[:K]


class PopularityRecall:
    def fit(self, train_df):
        self.item_counts = train_df['item_id'].value_counts().to_dict()
        self.all_items = sorted(self.item_counts.keys(),
                                key=lambda x: self.item_counts[x], reverse=True)

    def recall(self, user_id, K, exclude_ids):
        exclude_set = set(exclude_ids)
        candidates = [i for i in self.all_items if i not in exclude_set][:K]
        mx = max(self.item_counts.values())
        scores = [self.item_counts.get(c, 0) / mx for c in candidates]
        return candidates, scores


# ============================================================================
# EVALUATION
# ============================================================================

def compute_ndcg(ranked_list, ground_truth, k=10):
    gt_set = set(ground_truth)
    dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(ranked_list[:k]) if item in gt_set)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(gt_set), k)))
    return dcg / ideal if ideal > 0 else 0


def evaluate_recall(method, test_samples, K):
    recalls = []
    for sample in test_samples:
        candidates, _ = method.recall(sample['user_id'], K, sample['history'])
        gt = set(sample['ground_truth'])
        hits = len(set(candidates) & gt)
        recalls.append(hits / len(gt) if gt else 0)
    return {
        'mean': float(np.mean(recalls)),
        'ci_low': float(np.percentile(recalls, 2.5)),
        'ci_high': float(np.percentile(recalls, 97.5)),
        'scores': recalls
    }


# ============================================================================
# FULL PIPELINE
# ============================================================================

def run_full_pipeline(ds: Dataset, n_users: int = 500, seed: int = 42,
                      quick: bool = False):
    """Run the complete recall ceiling pipeline on a dataset."""
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"FULL PIPELINE: {ds.name}")
    print(f"  {ds.n_users} users, {ds.n_items} items, density={ds.density*100:.4f}%")
    print(f"{'='*60}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    test_samples = ds.get_test_samples(n_users=n_users, seed=seed)
    print(f"  Test samples: {len(test_samples)}")

    # ---- 1. Build retrieval methods ----
    print("\n--- Retrieval Methods ---")

    methods = {}

    print("  CF-SVD...")
    cf = CFRecall(n_factors=min(128, ds.n_items - 1))
    cf.fit(ds.train_df)
    methods['CF-SVD'] = cf

    if not quick:
        print("  ItemKNN...")
        knn = ItemKNNRecall(n_neighbors=50)
        knn.fit(ds.train_df)
        methods['ItemKNN'] = knn

    print("  Popularity...")
    pop = PopularityRecall()
    pop.fit(ds.train_df)
    methods['Popularity'] = pop

    # ---- 2. Evaluate Recall@K ----
    print("\n--- Recall@K ---")
    recall_results = {}
    K_values = [100] if quick else [50, 100, 200, 500]

    for K in K_values:
        recall_results[f'K={K}'] = {}
        for name, method in methods.items():
            res = evaluate_recall(method, test_samples, K)
            recall_results[f'K={K}'][name] = {
                'recall_mean': res['mean'],
                'recall_ci': [res['ci_low'], res['ci_high']]
            }
            print(f"  {name:15s} @{K}: {res['mean']*100:.2f}%")

    # ---- 3. Reranker evaluation ----
    print("\n--- Reranker Strategies ---")
    reranker_results = {}
    for strategy_name, strategy_fn in [
        ('cf_order', lambda c, s: c),
        ('random', lambda c, s: [c[i] for i in np.random.permutation(len(c))]),
        ('reverse', lambda c, s: list(reversed(c))),
        ('oracle', lambda c, s: sorted(c, key=lambda x: x in s, reverse=True)),
    ]:
        ndcgs = []
        for sample in test_samples:
            candidates, scores = cf.recall(sample['user_id'], 100, sample['history'])
            if not candidates:
                ndcgs.append(0)
                continue
            gt_set = set(sample['ground_truth'])
            ranked = strategy_fn(candidates, gt_set)
            ndcgs.append(compute_ndcg(ranked, sample['ground_truth']))
        reranker_results[strategy_name] = {
            'ndcg_mean': float(np.mean(ndcgs)),
            'ndcg_std': float(np.std(ndcgs))
        }
        print(f"  {strategy_name:15s}: NDCG@10 = {np.mean(ndcgs):.4f}")

    # ---- 4. RAEP ----
    print("\n--- RAEP ---")
    recalls = []
    ceiling_ndcgs = []
    actual_ndcgs = []

    for sample in test_samples:
        candidates, scores = cf.recall(sample['user_id'], 100, sample['history'])
        gt = set(sample['ground_truth'])
        if not candidates:
            recalls.append(0); ceiling_ndcgs.append(0); actual_ndcgs.append(0)
            continue
        hits = len(set(candidates) & gt)
        recalls.append(hits / len(gt) if gt else 0)
        # Oracle NDCG
        hit_items = [c for c in candidates if c in gt]
        non_hits = [c for c in candidates if c not in gt]
        ceiling_ndcgs.append(compute_ndcg(hit_items + non_hits, list(gt)))
        actual_ndcgs.append(compute_ndcg(candidates, list(gt)))

    mean_recall = np.mean(recalls)
    zero_pct = np.mean([1 if r == 0 else 0 for r in recalls]) * 100
    mean_ceiling = np.mean(ceiling_ndcgs)
    mean_actual = np.mean(actual_ndcgs)
    ceil_util = (mean_actual / mean_ceiling * 100) if mean_ceiling > 0 else 0

    raep = {
        'mean_recall': float(mean_recall),
        'zero_recall_pct': float(zero_pct),
        'mean_ceiling_ndcg': float(mean_ceiling),
        'mean_actual_ndcg': float(mean_actual),
        'ceiling_utilization_pct': float(ceil_util),
        'diagnosis': ('CRITICAL' if mean_recall < 0.05 else
                      'WARNING' if mean_recall < 0.15 else
                      'MODERATE' if mean_recall < 0.30 else 'GOOD')
    }
    print(f"  Recall@100: {mean_recall*100:.2f}%, Zero-recall: {zero_pct:.1f}%")
    print(f"  Ceiling util: {ceil_util:.1f}%, Diagnosis: {raep['diagnosis']}")

    elapsed = time.time() - t0

    # ---- Best recall ----
    best_method = max(recall_results['K=100'].items(),
                      key=lambda x: x[1]['recall_mean'])

    result = {
        'dataset': ds.key,
        'name': ds.name,
        'domain': ds.domain,
        'stats': ds.stats(),
        'recall': recall_results,
        'rerankers': reranker_results,
        'raep': raep,
        'best_recall_100': {
            'method': best_method[0],
            'recall': best_method[1]['recall_mean']
        },
        'elapsed_seconds': round(elapsed, 1),
        'timestamp': datetime.now().isoformat()
    }

    print(f"\n  Done in {elapsed:.1f}s")
    return result


def main():
    parser = argparse.ArgumentParser(description='Unified experiment runner')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--domain', type=str, default=None,
                        help='Filter by domain (product/news/movie)')
    parser.add_argument('--n_users', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: fewer methods, K=100 only')
    args = parser.parse_args()

    if args.all:
        datasets = load_all(domains=[args.domain] if args.domain else None)
    elif args.dataset:
        names = args.dataset.split(',')
        datasets = {n: load_dataset(n) for n in names}
    else:
        print("Usage: --dataset <name> or --all [--domain <domain>]")
        print("\nAvailable datasets:")
        for key, info in list_datasets().items():
            if info.exists:
                print(f"  {key:15s} {info.name:20s} ({info.domain})")
        return

    print("=" * 60)
    print("UNIFIED RECALL CEILING EXPERIMENT")
    print(f"Datasets: {list(datasets.keys())}")
    print(f"n_users: {args.n_users}, quick: {args.quick}")
    print("=" * 60)

    all_results = {}
    for key, ds in datasets.items():
        result = run_full_pipeline(ds, args.n_users, args.seed, args.quick)
        all_results[key] = result

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Dataset':15s} {'Domain':8s} {'Catalog':>7s} {'Dens%':>7s} "
          f"{'R@100':>7s} {'Zero%':>6s} {'CeilU':>6s} {'Diag':>8s}")
    print("-" * 70)
    for key, res in all_results.items():
        s = res['stats']
        r = res['raep']
        b = res['best_recall_100']
        print(f"{key:15s} {s['domain']:8s} {s['n_items']:>7d} "
              f"{s['density_pct']:>6.3f}% {b['recall']*100:>6.2f}% "
              f"{r['zero_recall_pct']:>5.1f}% {r['ceiling_utilization_pct']:>5.1f}% "
              f"{r['diagnosis']:>8s}")

    # Save
    output_path = project_root / 'experiments' / 'logs' / 'unified_experiment_results.json'

    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(i) for i in obj]
        return obj

    with open(output_path, 'w') as f:
        json.dump(convert(all_results), f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
