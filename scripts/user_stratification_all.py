#!/usr/bin/env python3
"""
User Stratification by History Length (All 8 Datasets)
=======================================================
Tests whether the recall ceiling is less severe for warm users.

Strata: cold (3-10), medium (10-50), warm (50+) interactions.

Usage:
    python scripts/user_stratification_all.py --n_users 500
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import numpy as np
from datetime import datetime
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from data_interface import load_all, Dataset


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


def compute_ndcg(ranked_list, ground_truth, k=10):
    gt_set = set(ground_truth)
    dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(ranked_list[:k]) if item in gt_set)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(gt_set), k)))
    return dcg / ideal if ideal > 0 else 0


STRATA = [
    ('cold', 3, 10),
    ('medium', 10, 50),
    ('warm', 50, 999999),
]


def run_dataset(ds, n_users=500, seed=42):
    np.random.seed(seed)
    test_samples = ds.get_test_samples(n_users=n_users, seed=seed)

    cf = CFRecall(n_factors=min(128, ds.n_items - 1))
    cf.fit(ds.train_df)

    # Evaluate per stratum
    results = {}
    for stratum_name, lo, hi in STRATA:
        stratum_samples = [s for s in test_samples if lo <= len(s['history']) < hi]
        if len(stratum_samples) < 10:
            results[stratum_name] = {'n': len(stratum_samples), 'skip': True}
            continue

        recalls = []
        ndcgs = []
        oracle_ndcgs = []
        for s in stratum_samples:
            candidates, _ = cf.recall(s['user_id'], 100, s['history'])
            gt = set(s['ground_truth'])
            hits = len(set(candidates) & gt)
            recalls.append(hits / len(gt) if gt else 0)
            ndcgs.append(compute_ndcg(candidates, list(gt)))
            # Oracle
            oracle_ranked = [c for c in candidates if c in gt] + [c for c in candidates if c not in gt]
            oracle_ndcgs.append(compute_ndcg(oracle_ranked, list(gt)))

        results[stratum_name] = {
            'n': len(stratum_samples),
            'avg_history': float(np.mean([len(s['history']) for s in stratum_samples])),
            'recall_mean': float(np.mean(recalls)),
            'zero_recall_pct': float(np.mean([1 if r == 0 else 0 for r in recalls]) * 100),
            'ndcg_mean': float(np.mean(ndcgs)),
            'oracle_ndcg': float(np.mean(oracle_ndcgs)),
            'ceiling_util': float(np.mean(ndcgs) / np.mean(oracle_ndcgs) * 100) if np.mean(oracle_ndcgs) > 0 else 0,
        }

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_users', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    datasets = load_all()
    all_results = {}

    print("=" * 70)
    print("USER STRATIFICATION BY HISTORY LENGTH")
    print("=" * 70)

    for key, ds in datasets.items():
        results = run_dataset(ds, args.n_users, args.seed)
        all_results[key] = results

        print(f"\n  {ds.name}:")
        for stratum, r in results.items():
            if r.get('skip'):
                print(f"    {stratum:8s}: n={r['n']} (skip)")
            else:
                print(f"    {stratum:8s}: n={r['n']:>4d}, hist={r['avg_history']:.0f}, "
                      f"recall={r['recall_mean']*100:.1f}%, zero={r['zero_recall_pct']:.0f}%, "
                      f"ceil_util={r['ceiling_util']:.0f}%")

    # Cross-dataset: does warm help?
    print("\n" + "=" * 70)
    print("KEY QUESTION: Is the ceiling less severe for warm users?")
    print("=" * 70)
    for key in sorted(all_results):
        cold = all_results[key].get('cold', {})
        warm = all_results[key].get('warm', {})
        if cold.get('skip') or warm.get('skip'):
            continue
        cold_r = cold.get('recall_mean', 0) * 100
        warm_r = warm.get('recall_mean', 0) * 100
        diff = warm_r - cold_r
        print(f"  {key:12s}: cold={cold_r:.1f}%, warm={warm_r:.1f}%, diff={diff:+.1f}%")

    output_path = project_root / 'experiments' / 'logs' / 'user_stratification_all.json'
    with open(output_path, 'w') as f:
        json.dump({'results': all_results, 'strata': {s[0]: {'lo': s[1], 'hi': s[2]} for s in STRATA},
                   'timestamp': datetime.now().isoformat()}, f, indent=2, default=str)
    print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    main()
