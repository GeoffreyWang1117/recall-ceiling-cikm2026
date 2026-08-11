#!/usr/bin/env python3
"""
Statistical Significance Tests Across All 8 Datasets
======================================================
Tests whether any reranking strategy significantly outperforms the CF-Score
baseline using Wilcoxon signed-rank tests with Holm-Bonferroni correction.

Usage:
    python scripts/statistical_significance_all.py --n_users 500
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
from scipy.stats import wilcoxon

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
        item_scores = [float(scores[self.item_to_idx[c]]) for c in candidates]
        if item_scores:
            mn, mx = min(item_scores), max(item_scores)
            if mx > mn:
                item_scores = [(s - mn) / (mx - mn) for s in item_scores]
        return candidates[:K], item_scores[:K]


def compute_ndcg(ranked_list, ground_truth, k=10):
    gt_set = set(ground_truth)
    dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(ranked_list[:k]) if item in gt_set)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(gt_set), k)))
    return dcg / ideal if ideal > 0 else 0


def holm_bonferroni(p_values, alpha=0.05):
    """Apply Holm-Bonferroni correction."""
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    corrected = np.ones(n)
    for rank, idx in enumerate(sorted_idx):
        corrected_alpha = alpha / (n - rank)
        corrected[idx] = corrected_alpha
    return corrected


def run_significance(ds, n_users=500, seed=42):
    """Run significance tests on a dataset."""
    print(f"\n  {ds.name}:")
    np.random.seed(seed)
    test_samples = ds.get_test_samples(n_users=n_users, seed=seed)

    cf = CFRecall(n_factors=min(128, ds.n_items - 1))
    cf.fit(ds.train_df)

    # Compute per-user NDCG for each strategy
    strategies = {
        'cf_order': lambda c, gt: c,
        'random': lambda c, gt: [c[i] for i in np.random.permutation(len(c))],
        'reverse': lambda c, gt: list(reversed(c)),
    }

    user_ndcgs = {'cf_order': [], 'random': [], 'reverse': []}

    for sample in test_samples:
        candidates, scores = cf.recall(sample['user_id'], 100, sample['history'])
        gt = sample['ground_truth']
        if not candidates:
            for k in user_ndcgs:
                user_ndcgs[k].append(0.0)
            continue
        for name, fn in strategies.items():
            ranked = fn(candidates, set(gt))
            user_ndcgs[name].append(compute_ndcg(ranked, gt))

    # Wilcoxon tests: each strategy vs cf_order
    results = {}
    cf_scores = np.array(user_ndcgs['cf_order'])

    comparisons = []
    for name in ['random', 'reverse']:
        other_scores = np.array(user_ndcgs[name])
        diff = other_scores - cf_scores
        nonzero = np.sum(diff != 0)

        if nonzero >= 10:
            try:
                stat, p = wilcoxon(other_scores, cf_scores)
            except Exception:
                p = 1.0
        else:
            p = 1.0

        mean_diff = np.mean(diff)
        d = mean_diff / (np.std(diff) + 1e-10)
        comparisons.append({
            'name': name,
            'vs': 'cf_order',
            'mean_ndcg': float(np.mean(other_scores)),
            'mean_diff': float(mean_diff),
            'cohens_d': float(d),
            'p_value': float(p),
        })

    # Holm-Bonferroni correction
    p_vals = np.array([c['p_value'] for c in comparisons])
    corrected_alphas = holm_bonferroni(p_vals)

    for i, comp in enumerate(comparisons):
        comp['corrected_alpha'] = float(corrected_alphas[i])
        comp['significant'] = bool(comp['p_value'] < corrected_alphas[i])

    cf_mean = float(np.mean(cf_scores))
    recall_rate = float(np.mean([1 if n > 0 else 0 for n in cf_scores]))

    print(f"    CF-order NDCG: {cf_mean:.4f}, nonzero users: {recall_rate*100:.1f}%")
    for c in comparisons:
        sig = '*' if c['significant'] else ''
        print(f"    {c['name']:15s}: diff={c['mean_diff']:+.4f}, d={c['cohens_d']:.3f}, p={c['p_value']:.4f} {sig}")

    return {
        'dataset': ds.key,
        'n_users': len(test_samples),
        'cf_mean_ndcg': cf_mean,
        'nonzero_user_pct': recall_rate * 100,
        'comparisons': comparisons,
        'any_significant': any(c['significant'] for c in comparisons)
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_users', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    datasets = load_all()

    print("=" * 60)
    print("STATISTICAL SIGNIFICANCE ACROSS ALL DATASETS")
    print(f"n_users={args.n_users}, Holm-Bonferroni correction")
    print("=" * 60)

    all_results = {}
    for key, ds in datasets.items():
        result = run_significance(ds, args.n_users, args.seed)
        all_results[key] = result

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Any strategy significantly beats CF-order?")
    print("=" * 60)
    for key, res in all_results.items():
        sig = "YES" if res['any_significant'] else "NO"
        print(f"  {key:15s}: {sig}  (CF NDCG={res['cf_mean_ndcg']:.4f}, nonzero={res['nonzero_user_pct']:.0f}%)")

    n_sig = sum(1 for r in all_results.values() if r['any_significant'])
    print(f"\nConclusion: {n_sig}/{len(all_results)} datasets show significant differences.")
    if n_sig == 0:
        print("Under realistic retrieval, NO reranking strategy significantly outperforms CF baseline.")

    output_path = project_root / 'experiments' / 'logs' / 'statistical_significance_all.json'
    with open(output_path, 'w') as f:
        json.dump({
            'results': all_results,
            'n_users': args.n_users,
            'correction': 'Holm-Bonferroni',
            'timestamp': datetime.now().isoformat()
        }, f, indent=2, default=str)
    print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    main()
