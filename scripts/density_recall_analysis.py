#!/usr/bin/env python3
"""
Density-Aware Recall Analysis
================================
Isolates the effect of data density on the recall ceiling by subsampling
interactions at different rates and re-evaluating retrieval recall.

For each dataset, we randomly drop 25%/50%/75% of interactions, retrain
the retrieval model, and measure Recall@K. This separates density from
catalog size as factors affecting the ceiling.

Usage:
    python scripts/density_recall_analysis.py --datasets beauty,toys,mind --n_users 300
    python scripts/density_recall_analysis.py --all --n_users 200
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import numpy as np
import pandas as pd
import time
from datetime import datetime
from typing import Dict, List
from tqdm import tqdm
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from data_interface import load_dataset, load_all, Dataset


# ============================================================================
# RETRIEVAL (CF-SVD)
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
        return candidates[:K], []


def evaluate_recall(cf, test_samples, K=100):
    recalls = []
    for sample in test_samples:
        candidates, _ = cf.recall(sample['user_id'], K, sample['history'])
        gt = set(sample['ground_truth'])
        hits = len(set(candidates) & gt)
        recalls.append(hits / len(gt) if gt else 0)
    return float(np.mean(recalls))


# ============================================================================
# MAIN
# ============================================================================

def run_density_analysis(ds: Dataset, drop_rates: List[float],
                         n_users: int = 200, seed: int = 42):
    """Run density subsampling analysis on a single dataset."""
    print(f"\n{'='*60}")
    print(f"DENSITY ANALYSIS: {ds.name}")
    print(f"  Original: {ds.n_interactions} interactions, density={ds.density*100:.4f}%")
    print(f"{'='*60}")

    # Get test samples from original dataset
    test_samples = ds.get_test_samples(n_users=n_users, seed=seed)
    print(f"  Test users: {len(test_samples)}")

    results = []

    # Full density (baseline)
    cf = CFRecall(n_factors=min(128, ds.n_items - 1))
    cf.fit(ds.train_df)
    for K in [50, 100, 200]:
        r = evaluate_recall(cf, test_samples, K)
        results.append({
            'keep_ratio': 1.0,
            'density': ds.density,
            'n_interactions': ds.n_interactions,
            'K': K,
            'recall': r
        })
        print(f"  keep=100%: density={ds.density*100:.4f}%, Recall@{K}={r*100:.2f}%")

    # Subsampled versions
    for drop_rate in drop_rates:
        keep_ratio = 1.0 - drop_rate
        sub_ds = ds.subsample_interactions(keep_ratio, seed=seed)

        # Filter test samples: only users still in subsampled data
        sub_history = sub_ds.user_history
        valid_samples = [s for s in test_samples
                         if s['user_id'] in sub_history and len(sub_history[s['user_id']]) >= 3]

        if len(valid_samples) < 30:
            print(f"  keep={keep_ratio*100:.0f}%: too few valid users ({len(valid_samples)}), skip")
            continue

        # Update histories for valid samples
        sub_samples = [{'user_id': s['user_id'],
                        'history': sub_history[s['user_id']],
                        'ground_truth': s['ground_truth']}
                       for s in valid_samples]

        cf_sub = CFRecall(n_factors=min(128, sub_ds.n_items - 1))
        cf_sub.fit(sub_ds.train_df)

        for K in [50, 100, 200]:
            r = evaluate_recall(cf_sub, sub_samples, K)
            results.append({
                'keep_ratio': keep_ratio,
                'density': sub_ds.density,
                'n_interactions': sub_ds.n_interactions,
                'n_valid_users': len(sub_samples),
                'K': K,
                'recall': r
            })
            if K == 100:
                print(f"  keep={keep_ratio*100:.0f}%: density={sub_ds.density*100:.4f}%, "
                      f"Recall@100={r*100:.2f}% ({len(sub_samples)} users)")

    return {
        'dataset': ds.key,
        'name': ds.name,
        'original_density': ds.density,
        'original_n_interactions': ds.n_interactions,
        'catalog_size': ds.catalog_size,
        'results': results
    }


def main():
    parser = argparse.ArgumentParser(description='Density-aware recall analysis')
    parser.add_argument('--datasets', type=str, default=None,
                        help='Comma-separated dataset names')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--n_users', type=int, default=200)
    parser.add_argument('--drop_rates', type=str, default='0.25,0.50,0.75',
                        help='Drop rates (comma-separated)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    drop_rates = [float(x) for x in args.drop_rates.split(',')]

    if args.all:
        datasets = load_all()
    elif args.datasets:
        datasets = {name: load_dataset(name) for name in args.datasets.split(',')}
    else:
        datasets = {name: load_dataset(name) for name in ['beauty', 'toys', 'mind', 'movielens']}

    print("=" * 60)
    print("DENSITY-AWARE RECALL ANALYSIS")
    print(f"Drop rates: {drop_rates}")
    print(f"Datasets: {list(datasets.keys())}")
    print("=" * 60)

    all_results = {}
    for key, ds in datasets.items():
        result = run_density_analysis(ds, drop_rates, args.n_users, args.seed)
        all_results[key] = result

    # Cross-dataset analysis
    print("\n" + "=" * 60)
    print("CROSS-DATASET DENSITY vs RECALL (K=100)")
    print("=" * 60)
    print(f"{'Dataset':15s} {'Keep%':>6s} {'Density':>9s} {'Recall@100':>10s}")
    print("-" * 45)

    all_points = []
    for key, res in all_results.items():
        for r in res['results']:
            if r['K'] == 100:
                print(f"{key:15s} {r['keep_ratio']*100:>5.0f}% {r['density']*100:>8.4f}% {r['recall']*100:>9.2f}%")
                all_points.append((r['density'], r['recall'], key, r['keep_ratio']))

    # Compute correlation across all density-recall points
    if len(all_points) >= 4:
        densities = np.array([p[0] for p in all_points])
        recalls = np.array([p[1] for p in all_points])
        log_densities = np.log10(densities + 1e-10)
        corr = np.corrcoef(log_densities, recalls)[0, 1]
        print(f"\nOverall correlation (log_density, recall@100): r = {corr:.3f}")

        # Within-dataset correlations
        print("\nWithin-dataset density sensitivity:")
        for key, res in all_results.items():
            points = [(r['density'], r['recall']) for r in res['results'] if r['K'] == 100]
            if len(points) >= 3:
                d = np.array([p[0] for p in points])
                r = np.array([p[1] for p in points])
                slope = (r[-1] - r[0]) / (d[-1] - d[0]) if d[-1] != d[0] else 0
                pct_drop = (1 - r[-1] / r[0]) * 100 if r[0] > 0 else 0
                print(f"  {key:15s}: 75% density drop → {pct_drop:.1f}% recall drop")

    # Save
    output_path = project_root / 'experiments' / 'logs' / 'density_recall_analysis.json'
    with open(output_path, 'w') as f:
        json.dump({
            'datasets': all_results,
            'drop_rates': drop_rates,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
