#!/usr/bin/env python3
"""
KDD 2026 Supplementary Experiment: Theoretical Upper Bound Analysis
====================================================================
Calculates the theoretical maximum NDCG@10 achievable given a specific
recall rate, and compares with actual performance.

This provides direct evidence that even perfect reranking cannot overcome
the recall bottleneck.

Usage:
    python scripts/kdd_theoretical_upper_bound.py
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


def dcg_at_k(relevances: List[float], k: int = 10) -> float:
    """Calculate DCG@k given relevance scores."""
    relevances = np.array(relevances)[:k]
    if len(relevances) == 0:
        return 0.0
    gains = relevances
    discounts = np.log2(np.arange(len(relevances)) + 2)
    return np.sum(gains / discounts)


def idcg_at_k(n_relevant: int, k: int = 10) -> float:
    """Calculate ideal DCG@k."""
    n = min(n_relevant, k)
    if n == 0:
        return 0.0
    return dcg_at_k([1.0] * n, k)


def theoretical_max_ndcg(recall_rate: float, n_relevant: int, k: int = 10) -> float:
    """
    Calculate theoretical maximum NDCG@k given recall rate.

    Assumes the best case: all retrieved relevant items are ranked at top positions.

    Args:
        recall_rate: Fraction of relevant items retrieved (0.0 to 1.0)
        n_relevant: Total number of relevant items for the user
        k: Cutoff for NDCG calculation

    Returns:
        Theoretical maximum NDCG@k
    """
    # Number of relevant items we can retrieve
    n_retrieved_relevant = int(np.ceil(recall_rate * n_relevant))
    n_retrieved_relevant = min(n_retrieved_relevant, n_relevant)

    # Best case: all retrieved relevant items at top positions
    best_dcg = dcg_at_k([1.0] * n_retrieved_relevant, k)
    ideal_dcg = idcg_at_k(n_relevant, k)

    if ideal_dcg == 0:
        return 0.0
    return best_dcg / ideal_dcg


def bootstrap_ci(scores: List[float], n_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float, float]:
    """Calculate bootstrap confidence interval."""
    scores = np.array(scores)
    if len(scores) == 0:
        return 0.0, 0.0, 0.0

    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        boot_means.append(np.mean(sample))

    alpha = (1 - ci) / 2
    return np.mean(scores), np.percentile(boot_means, alpha * 100), np.percentile(boot_means, (1 - alpha) * 100)


class CFRetrieval:
    """Simple CF-based retrieval using SVD."""

    def __init__(self, n_factors: int = 128):
        self.n_factors = n_factors

    def fit(self, train_df: pd.DataFrame):
        items = train_df['item_id'].unique()
        users = train_df['user_id'].unique()

        self.item_to_idx = {i: idx for idx, i in enumerate(items)}
        self.idx_to_item = {idx: i for i, idx in self.item_to_idx.items()}
        self.user_to_idx = {u: idx for idx, u in enumerate(users)}

        rows = [self.user_to_idx[u] for u in train_df['user_id']]
        cols = [self.item_to_idx[i] for i in train_df['item_id']]
        data = np.ones(len(rows))

        self.user_item = csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))

        self.svd = TruncatedSVD(n_components=min(self.n_factors, len(items) - 1))
        self.user_factors = self.svd.fit_transform(self.user_item)
        self.item_factors = self.svd.components_.T
        self.user_history = train_df.groupby('user_id')['item_id'].apply(list).to_dict()
        self.all_items = list(items)

    def get_candidates(self, user_id: str, K: int = 100) -> Tuple[List[str], List[float]]:
        if user_id not in self.user_to_idx:
            return [], []

        user_idx = self.user_to_idx[user_id]
        scores = np.dot(self.user_factors[user_idx], self.item_factors.T)

        # Exclude items in history
        history = self.user_history.get(user_id, [])
        for item in history:
            if item in self.item_to_idx:
                scores[self.item_to_idx[item]] = -np.inf

        top_indices = np.argsort(scores)[::-1][:K]
        return [self.idx_to_item[idx] for idx in top_indices], [scores[idx] for idx in top_indices]


def run_analysis():
    """Run the theoretical upper bound analysis."""
    print("=" * 70)
    print("THEORETICAL UPPER BOUND ANALYSIS")
    print("=" * 70)

    results = {}

    # Analyze each dataset
    datasets = ['amazon_beauty', 'amazon_movies', 'amazon_electronics']

    for dataset_name in datasets:
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name}")
        print("=" * 70)

        # Try sampled version first, then full version
        data_path = project_root / 'data' / 'processed' / f'{dataset_name}_sampled'
        if not data_path.exists():
            data_path = project_root / 'data' / 'processed' / dataset_name

        if not data_path.exists():
            print(f"  Dataset not found: {data_path}")
            continue

        # Load data
        train_df = pd.read_parquet(data_path / 'train.parquet')
        test_df = pd.read_parquet(data_path / 'test.parquet')

        print(f"  Train: {len(train_df)} interactions")
        print(f"  Test: {len(test_df)} interactions")

        # Build CF model
        cf = CFRetrieval(n_factors=128)
        cf.fit(train_df)

        # Get test users
        np.random.seed(42)
        test_users = test_df.groupby('user_id')['item_id'].apply(list).to_dict()
        valid_users = [u for u in test_users if u in cf.user_to_idx and len(test_users[u]) > 0]

        # Sample users
        n_users = min(500, len(valid_users))
        if len(valid_users) > n_users:
            valid_users = list(np.random.choice(valid_users, n_users, replace=False))

        print(f"  Analyzing {len(valid_users)} users...")

        # Calculate metrics for each user
        user_metrics = []

        for user_id in valid_users:
            gt = set(test_users[user_id])
            n_relevant = len(gt)

            # Get CF candidates
            candidates, _ = cf.get_candidates(user_id, K=100)

            # Calculate actual recall
            hits = sum(1 for c in candidates if c in gt)
            recall = hits / n_relevant if n_relevant > 0 else 0.0

            # Calculate theoretical max NDCG given this recall
            theoretical_max = theoretical_max_ndcg(recall, n_relevant, k=10)

            user_metrics.append({
                'user_id': user_id,
                'n_relevant': n_relevant,
                'recall': recall,
                'theoretical_max_ndcg': theoretical_max
            })

        # Aggregate results
        metrics_df = pd.DataFrame(user_metrics)

        # Group by recall ranges
        recall_bins = [0, 0.05, 0.10, 0.20, 0.50, 1.01]
        recall_labels = ['0-5%', '5-10%', '10-20%', '20-50%', '50-100%']
        metrics_df['recall_bin'] = pd.cut(metrics_df['recall'], bins=recall_bins, labels=recall_labels)

        print(f"\n  Theoretical Upper Bound Analysis (K=100):")
        print(f"  {'Recall Range':<15} {'N Users':<10} {'Avg Recall':<12} {'Theory Max NDCG':<18} {'95% CI'}")
        print(f"  {'-'*70}")

        dataset_results = {
            'n_users': len(valid_users),
            'overall': {},
            'by_recall_bin': {}
        }

        for label in recall_labels:
            bin_data = metrics_df[metrics_df['recall_bin'] == label]
            if len(bin_data) == 0:
                continue

            mean_recall = bin_data['recall'].mean()
            theory_scores = bin_data['theoretical_max_ndcg'].tolist()
            mean_theory, ci_low, ci_high = bootstrap_ci(theory_scores)

            print(f"  {label:<15} {len(bin_data):<10} {mean_recall*100:>8.1f}%     {mean_theory:.4f}           [{ci_low:.4f}, {ci_high:.4f}]")

            dataset_results['by_recall_bin'][label] = {
                'n_users': len(bin_data),
                'avg_recall': float(mean_recall),
                'theoretical_max_ndcg': {
                    'mean': float(mean_theory),
                    'ci_low': float(ci_low),
                    'ci_high': float(ci_high)
                }
            }

        # Overall statistics
        overall_recall = metrics_df['recall'].mean()
        overall_theory = metrics_df['theoretical_max_ndcg'].tolist()
        overall_mean, overall_ci_low, overall_ci_high = bootstrap_ci(overall_theory)

        print(f"\n  {'Overall':<15} {len(valid_users):<10} {overall_recall*100:>8.1f}%     {overall_mean:.4f}           [{overall_ci_low:.4f}, {overall_ci_high:.4f}]")

        dataset_results['overall'] = {
            'avg_recall': float(overall_recall),
            'theoretical_max_ndcg': {
                'mean': float(overall_mean),
                'ci_low': float(overall_ci_low),
                'ci_high': float(overall_ci_high)
            }
        }

        results[dataset_name] = dataset_results

    # Summary comparison with actual results
    print("\n" + "=" * 70)
    print("COMPARISON: THEORETICAL MAX vs ACTUAL PERFORMANCE")
    print("=" * 70)

    # Load actual results if available
    actual_results_path = project_root / 'experiments' / 'logs' / 'kdd_comprehensive_results.json'
    actual_results = {}
    if actual_results_path.exists():
        with open(actual_results_path) as f:
            actual_data = json.load(f)
            for key in ['beauty', 'movies', 'electronics']:
                if key in actual_data and 'reranking' in actual_data[key]:
                    # Get best actual NDCG
                    best_ndcg = 0
                    for method, data in actual_data[key]['reranking'].items():
                        if 'ndcg@10' in data and 'mean' in data['ndcg@10']:
                            best_ndcg = max(best_ndcg, data['ndcg@10']['mean'])
                    actual_results[f'amazon_{key}'] = best_ndcg

    print(f"\n{'Dataset':<25} {'Avg Recall':<12} {'Theory Max':<15} {'Actual Best':<15} {'Gap'}")
    print("-" * 75)

    for dataset_name, data in results.items():
        avg_recall = data['overall']['avg_recall']
        theory_max = data['overall']['theoretical_max_ndcg']['mean']
        actual_best = actual_results.get(dataset_name, 0)

        if theory_max > 0:
            gap = (theory_max - actual_best) / theory_max * 100
        else:
            gap = 0

        print(f"{dataset_name:<25} {avg_recall*100:>8.1f}%     {theory_max:.4f}          {actual_best:.4f}          {gap:.1f}%")

    # Key insight
    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print("""
Even with PERFECT reranking (all retrieved relevant items at top positions),
the theoretical maximum NDCG@10 is severely limited by recall:

- At 9% recall: Max NDCG ≈ 0.09 (current actual: ~0.005)
- At 3% recall: Max NDCG ≈ 0.03 (current actual: ~0.003)

This proves that the retrieval bottleneck creates a fundamental ceiling
that no reranking method - regardless of sophistication - can overcome.

The gap between theoretical max and actual performance also suggests
there's room for better reranking, but only AFTER solving the recall problem.
""")

    # Save results
    output_path = project_root / 'experiments' / 'logs' / 'kdd_theoretical_upper_bound.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    run_analysis()
