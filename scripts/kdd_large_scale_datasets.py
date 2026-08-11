#!/usr/bin/env python3
"""
KDD 2026 Large-Scale Industrial Dataset Experiments
====================================================
Processes and evaluates on large-scale Amazon datasets to validate
the recall bottleneck at industrial scale.

Datasets:
- Home_and_Kitchen: 67M reviews, 23M users, 3.7M items
- Sports_and_Outdoors: 19.6M reviews, 10M users, 1.6M items
- Toys_and_Games: 16M reviews, 8M users, 890K items

Usage:
    python scripts/kdd_large_scale_datasets.py --dataset home_kitchen --n_users 500
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
from typing import Dict, List, Tuple
from tqdm import tqdm
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================================
# DATA PROCESSING
# ============================================================================

def load_and_process_dataset(dataset_name: str, min_user_interactions: int = 10,
                             min_item_interactions: int = 5, sample_users: int = 5000):
    """Load and process a large-scale Amazon dataset."""

    dataset_map = {
        'home_kitchen': 'Home_and_Kitchen',
        'sports': 'Sports_and_Outdoors',
        'toys': 'Toys_and_Games',
        'office': 'Office_Products',
        'automotive': 'Automotive',
    }

    if dataset_name not in dataset_map:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    raw_name = dataset_map[dataset_name]
    raw_path = project_root / 'data' / 'raw' / 'amazon'
    reviews_file = raw_path / f'{raw_name}_reviews.parquet'

    print(f"Loading {raw_name}...")
    df = pd.read_parquet(reviews_file)
    print(f"  Raw: {len(df):,} reviews")

    # Rename columns for consistency
    df = df.rename(columns={'parent_asin': 'item_id'})

    # Filter users and items with minimum interactions
    print(f"  Filtering (min_user={min_user_interactions}, min_item={min_item_interactions})...")

    for _ in range(3):  # Iterate to ensure both conditions
        user_counts = df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_user_interactions].index
        df = df[df['user_id'].isin(valid_users)]

        item_counts = df['item_id'].value_counts()
        valid_items = item_counts[item_counts >= min_item_interactions].index
        df = df[df['item_id'].isin(valid_items)]

    print(f"  After filtering: {len(df):,} reviews, {df['user_id'].nunique():,} users, {df['item_id'].nunique():,} items")

    # Sample users if too many
    unique_users = df['user_id'].unique()
    if len(unique_users) > sample_users:
        print(f"  Sampling {sample_users} users from {len(unique_users):,}...")
        np.random.seed(42)
        sampled_users = np.random.choice(unique_users, sample_users, replace=False)
        df = df[df['user_id'].isin(sampled_users)]

    print(f"  Final: {len(df):,} reviews, {df['user_id'].nunique():,} users, {df['item_id'].nunique():,} items")

    # Split train/test (last interaction per user as test)
    if 'timestamp' in df.columns:
        df = df.sort_values(['user_id', 'timestamp'])

    test_indices = df.groupby('user_id').tail(1).index
    train_df = df[~df.index.isin(test_indices)].copy()
    test_df = df[df.index.isin(test_indices)].copy()

    print(f"  Train: {len(train_df):,} | Test: {len(test_df):,}")

    return train_df, test_df


# ============================================================================
# RETRIEVAL METHODS
# ============================================================================

class PopularityRecall:
    def __init__(self):
        self.sorted_items = None
        self.popularity = None

    def fit(self, train_df):
        self.popularity = train_df['item_id'].value_counts().to_dict()
        self.sorted_items = sorted(self.popularity.keys(),
                                   key=lambda x: self.popularity[x], reverse=True)

    def recall(self, user_id, K, exclude_ids):
        exclude_set = set(exclude_ids)
        candidates = [i for i in self.sorted_items if i not in exclude_set][:K]
        scores = [self.popularity.get(c, 0) for c in candidates]
        if scores:
            max_s = max(scores)
            scores = [s/max_s for s in scores]
        return candidates, scores


class ItemKNNRecall:
    def __init__(self, n_neighbors=50):
        self.n_neighbors = n_neighbors

    def fit(self, train_df):
        items = train_df['item_id'].unique()
        users = train_df['user_id'].unique()
        self.item_to_idx = {i: idx for idx, i in enumerate(items)}
        self.idx_to_item = {idx: i for i, idx in self.item_to_idx.items()}
        user_to_idx = {u: idx for idx, u in enumerate(users)}

        rows = [user_to_idx[u] for u in train_df['user_id']]
        cols = [self.item_to_idx[i] for i in train_df['item_id']]
        data = np.ones(len(rows))
        user_item = csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))

        # Compute item similarity (sample if too large)
        item_matrix = user_item.T
        n_items = item_matrix.shape[0]

        if n_items > 50000:
            # For very large item sets, use approximate similarity
            print(f"    Large item set ({n_items:,}), using approximate similarity...")
            self.item_sim = None
            self.user_items = train_df.groupby('user_id')['item_id'].apply(list).to_dict()
            self.item_vectors = item_matrix.toarray()
            self.item_norms = np.linalg.norm(self.item_vectors, axis=1, keepdims=True)
            self.item_norms[self.item_norms == 0] = 1
            self.item_vectors = self.item_vectors / self.item_norms
        else:
            item_matrix = item_matrix.toarray()
            self.item_sim = cosine_similarity(item_matrix)
            np.fill_diagonal(self.item_sim, 0)
            self.user_items = train_df.groupby('user_id')['item_id'].apply(list).to_dict()

    def recall(self, user_id, K, exclude_ids):
        history = self.user_items.get(user_id, [])
        if not history:
            return [], []

        item_scores = np.zeros(len(self.item_to_idx))

        if self.item_sim is not None:
            for hist_item in history:
                if hist_item in self.item_to_idx:
                    item_scores += self.item_sim[self.item_to_idx[hist_item]]
        else:
            # Approximate: average history vectors and compute similarity
            hist_vectors = []
            for hist_item in history:
                if hist_item in self.item_to_idx:
                    hist_vectors.append(self.item_vectors[self.item_to_idx[hist_item]])
            if hist_vectors:
                user_vec = np.mean(hist_vectors, axis=0)
                user_vec = user_vec / (np.linalg.norm(user_vec) + 1e-8)
                item_scores = np.dot(self.item_vectors, user_vec)

        for item in exclude_ids:
            if item in self.item_to_idx:
                item_scores[self.item_to_idx[item]] = -np.inf

        top_indices = np.argsort(item_scores)[::-1][:K]
        candidates = [self.idx_to_item[idx] for idx in top_indices if item_scores[idx] > -np.inf]
        scores = [item_scores[self.item_to_idx[c]] for c in candidates]

        if scores:
            max_s = max(scores) if max(scores) > 0 else 1
            scores = [s/max_s for s in scores]
        return candidates[:K], scores[:K]


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
            sorted_items = sorted(self.popularity.keys(), key=lambda x: self.popularity[x], reverse=True)
            candidates = [i for i in sorted_items if i not in set(exclude_ids)][:K]
            scores = [self.popularity.get(c, 0) / max(self.popularity.values()) for c in candidates]
            return candidates, scores

        u_idx = self.user_to_idx[user_id]
        scores = np.dot(self.item_factors, self.user_factors[u_idx])

        for item in exclude_ids:
            if item in self.item_to_idx:
                scores[self.item_to_idx[item]] = -np.inf

        top_indices = np.argsort(scores)[::-1][:K]
        candidates = [self.idx_to_item[idx] for idx in top_indices if scores[idx] > -np.inf]
        item_scores = [scores[self.item_to_idx[c]] for c in candidates]

        if item_scores:
            min_s, max_s = min(item_scores), max(item_scores)
            if max_s > min_s:
                item_scores = [(s - min_s) / (max_s - min_s) for s in item_scores]
        return candidates[:K], item_scores[:K]


# ============================================================================
# EVALUATION
# ============================================================================

def bootstrap_ci(scores, n_bootstrap=1000, ci=0.95):
    scores = np.array(scores)
    if len(scores) == 0:
        return 0.0, 0.0, 0.0
    boot_means = [np.mean(np.random.choice(scores, len(scores), replace=True)) for _ in range(n_bootstrap)]
    alpha = (1 - ci) / 2
    return np.mean(scores), np.percentile(boot_means, alpha * 100), np.percentile(boot_means, (1 - alpha) * 100)


def evaluate_retrieval(methods: Dict, test_samples: List, K: int = 100) -> Dict:
    """Evaluate retrieval methods."""
    results = {}

    for method_name, method in methods.items():
        print(f"\n  {method_name}:")
        recalls = []

        for sample in tqdm(test_samples, desc=f"    Evaluating"):
            user_id = sample['user_id']
            history = set(sample['history'])
            ground_truth = set(sample['ground_truth'])

            candidates, _ = method.recall(user_id, K, history)

            if len(ground_truth) > 0:
                hits = len(set(candidates) & ground_truth)
                recall = hits / len(ground_truth)
                recalls.append(recall)

        mean, ci_low, ci_high = bootstrap_ci(recalls)
        results[method_name] = {
            'recall@100': {
                'mean': float(mean),
                'ci_low': float(ci_low),
                'ci_high': float(ci_high)
            },
            'n_users': len(recalls)
        }
        print(f"    Recall@{K}: {mean*100:.2f}% [{ci_low*100:.2f}, {ci_high*100:.2f}]")

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='home_kitchen',
                        choices=['home_kitchen', 'sports', 'toys', 'office', 'automotive'])
    parser.add_argument('--n_users', type=int, default=500)
    parser.add_argument('--K', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("=" * 70)
    print("LARGE-SCALE INDUSTRIAL DATASET EXPERIMENT")
    print(f"Dataset: {args.dataset.upper()}")
    print("=" * 70)

    # Load and process data
    train_df, test_df = load_and_process_dataset(
        args.dataset,
        min_user_interactions=10,
        min_item_interactions=5,
        sample_users=args.n_users * 2  # Sample more to ensure enough valid users
    )

    # Prepare test samples
    user_history = train_df.groupby('user_id')['item_id'].apply(list).to_dict()
    test_gt = test_df.groupby('user_id')['item_id'].apply(list).to_dict()
    valid_users = [u for u in test_gt.keys() if u in user_history and len(user_history[u]) >= 5]

    n_users = min(args.n_users, len(valid_users))
    sampled_users = np.random.choice(valid_users, size=n_users, replace=False)

    test_samples = [{'user_id': u, 'history': user_history[u], 'ground_truth': test_gt[u]}
                    for u in sampled_users]

    print(f"\nTest samples: {len(test_samples)} users")
    print(f"Catalog size: {train_df['item_id'].nunique():,} items")

    # Build retrieval methods
    print("\n" + "=" * 70)
    print("BUILDING RETRIEVAL METHODS")
    print("=" * 70)

    methods = {}

    print("\n  Popularity...")
    pop = PopularityRecall()
    pop.fit(train_df)
    methods['Popularity'] = pop

    print("\n  CF-SVD...")
    cf = CFRecall()
    cf.fit(train_df)
    methods['CF-SVD'] = cf

    print("\n  ItemKNN...")
    knn = ItemKNNRecall()
    knn.fit(train_df)
    methods['ItemKNN'] = knn

    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUATING RETRIEVAL")
    print("=" * 70)

    results = evaluate_retrieval(methods, test_samples, args.K)

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'dataset': args.dataset,
            'n_users': len(test_samples),
            'n_items': int(train_df['item_id'].nunique()),
            'n_interactions': int(len(train_df)),
            'K': args.K,
            'seed': args.seed
        },
        'retrieval_results': results,
        'key_finding': f'Large-scale {args.dataset} dataset recall bottleneck validation'
    }

    output_path = project_root / 'experiments' / 'logs' / f'kdd_large_scale_{args.dataset}.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    best_method = max(results.keys(), key=lambda x: results[x]['recall@100']['mean'])
    best_recall = results[best_method]['recall@100']['mean']

    print(f"\nDataset: {args.dataset.upper()}")
    print(f"Catalog size: {train_df['item_id'].nunique():,} items")
    print(f"Best method: {best_method} with Recall@{args.K} = {best_recall*100:.2f}%")

    if best_recall < 0.05:
        print(f"\n⚠️  Recall bottleneck confirmed at industrial scale!")
        print(f"   Only {best_recall*100:.1f}% of relevant items can be retrieved.")

    return output


if __name__ == '__main__':
    main()
