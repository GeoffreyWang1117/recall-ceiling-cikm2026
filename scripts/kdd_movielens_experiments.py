#!/usr/bin/env python3
"""
KDD 2026 MovieLens + Multi-Item Evaluation
=============================================
Addresses:
1. "Only Amazon datasets" -> adds MovieLens-25M (high-density standard benchmark)
2. "Leave-one-out is too simple" -> multi-item test set (last-5 split)

Usage:
    python scripts/kdd_movielens_experiments.py --download   # Download ML-25M
    python scripts/kdd_movielens_experiments.py --n_users 500 --eval_mode both
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import os
import zipfile
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from typing import Dict, List, Tuple
from tqdm import tqdm
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import wilcoxon


# ============================================================================
# DATASET PREPARATION
# ============================================================================

def download_movielens(data_dir):
    """Download and extract MovieLens-25M."""
    ml_dir = data_dir / 'movielens-25m'
    if (ml_dir / 'ratings.csv').exists():
        print("MovieLens-25M already downloaded.")
        return ml_dir

    print("Downloading MovieLens-25M...")
    import urllib.request
    url = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
    zip_path = data_dir / 'ml-25m.zip'

    if not zip_path.exists():
        urllib.request.urlretrieve(url, zip_path)
        print(f"Downloaded to {zip_path}")

    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(data_dir)

    # Rename
    extracted = data_dir / 'ml-25m'
    if extracted.exists() and not ml_dir.exists():
        extracted.rename(ml_dir)

    print(f"Extracted to {ml_dir}")
    return ml_dir


def prepare_movielens(data_dir, n_core=5, max_users=None):
    """Process MovieLens-25M with k-core filtering.

    Returns train/val/test DataFrames with leave-one-out AND last-5 splits.
    """
    ml_dir = data_dir / 'movielens-25m'
    processed_dir = data_dir / 'processed' / 'movielens_25m'
    processed_dir.mkdir(parents=True, exist_ok=True)

    if (processed_dir / 'train.parquet').exists():
        print("MovieLens-25M already processed.")
        return processed_dir

    print("Processing MovieLens-25M...")
    ratings = pd.read_csv(ml_dir / 'ratings.csv')
    movies = pd.read_csv(ml_dir / 'movies.csv')

    print(f"  Raw: {len(ratings)} ratings, {ratings['userId'].nunique()} users, {ratings['movieId'].nunique()} movies")

    # Convert to implicit (rating >= 4 = positive)
    ratings = ratings[ratings['rating'] >= 4.0].copy()
    print(f"  After threshold (>=4): {len(ratings)} ratings")

    # K-core filtering
    for _ in range(10):
        user_counts = ratings['userId'].value_counts()
        valid_users = user_counts[user_counts >= n_core].index
        ratings = ratings[ratings['userId'].isin(valid_users)]

        item_counts = ratings['movieId'].value_counts()
        valid_items = item_counts[item_counts >= n_core].index
        ratings = ratings[ratings['movieId'].isin(valid_items)]

    print(f"  After {n_core}-core: {len(ratings)} ratings, {ratings['userId'].nunique()} users, {ratings['movieId'].nunique()} movies")

    # Subsample users if needed
    if max_users and ratings['userId'].nunique() > max_users:
        # Sample active users
        user_counts = ratings.groupby('userId').size()
        sampled_users = user_counts.nlargest(max_users).index
        ratings = ratings[ratings['userId'].isin(sampled_users)]
        print(f"  After user sampling: {len(ratings)} ratings, {ratings['userId'].nunique()} users")

    # Create user/item mappings
    user_ids = sorted(ratings['userId'].unique())
    item_ids = sorted(ratings['movieId'].unique())
    user_map = {uid: i for i, uid in enumerate(user_ids)}
    item_map = {iid: i for i, iid in enumerate(item_ids)}

    ratings['user_id'] = ratings['userId'].map(user_map)
    ratings['item_id'] = ratings['movieId'].map(item_map)

    # Sort by timestamp
    ratings = ratings.sort_values(['user_id', 'timestamp'])

    # Split: leave-last-5 for test, leave-last-1 before that for val
    train_rows = []
    val_rows = []
    test_rows = []

    for uid, group in ratings.groupby('user_id'):
        group = group.sort_values('timestamp')
        items = group[['user_id', 'item_id', 'rating', 'timestamp']].values

        if len(items) < 7:  # Need at least 7: 1 val + 5 test + some train
            train_rows.extend(items.tolist())
            continue

        # Last 5 for test
        test_items = items[-5:]
        # Item before that for val
        val_item = items[-6:-5]
        # Rest for train
        train_items = items[:-6]

        train_rows.extend(train_items.tolist())
        val_rows.extend(val_item.tolist())
        test_rows.extend(test_items.tolist())

    cols = ['user_id', 'item_id', 'rating', 'timestamp']
    train_df = pd.DataFrame(train_rows, columns=cols)
    val_df = pd.DataFrame(val_rows, columns=cols)
    test_df = pd.DataFrame(test_rows, columns=cols)

    # Also create a leave-one-out test (last item only, for comparability)
    loo_test_rows = []
    for uid, group in test_df.groupby('user_id'):
        last = group.sort_values('timestamp').iloc[-1:]
        loo_test_rows.append(last)
    loo_test_df = pd.concat(loo_test_rows, ignore_index=True) if loo_test_rows else pd.DataFrame(columns=cols)

    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test (last-5): {len(test_df)}, Test (LOO): {len(loo_test_df)}")
    print(f"  Users: {train_df['user_id'].nunique()}, Items: {train_df['item_id'].nunique()}")
    print(f"  Density: {len(train_df) / (train_df['user_id'].nunique() * train_df['item_id'].nunique()) * 100:.4f}%")

    # Save
    train_df.to_parquet(processed_dir / 'train.parquet')
    val_df.to_parquet(processed_dir / 'val.parquet')
    test_df.to_parquet(processed_dir / 'test.parquet')
    loo_test_df.to_parquet(processed_dir / 'test_loo.parquet')

    # Movie metadata
    movie_meta = {}
    for _, row in movies.iterrows():
        mid = item_map.get(row['movieId'])
        if mid is not None:
            movie_meta[str(mid)] = {
                'title': row['title'],
                'genres': row['genres']
            }
    with open(processed_dir / 'item_metadata.json', 'w') as f:
        json.dump(movie_meta, f)

    return processed_dir


# ============================================================================
# RETRIEVAL BASELINES (same as other scripts)
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
        item_matrix = user_item.T.toarray()
        self.item_sim = cosine_similarity(item_matrix)
        np.fill_diagonal(self.item_sim, 0)
        self.user_items = train_df.groupby('user_id')['item_id'].apply(list).to_dict()

    def recall(self, user_id, K, exclude_ids):
        history = self.user_items.get(user_id, [])
        if not history:
            return [], []
        item_scores = np.zeros(len(self.item_to_idx))
        for h in history:
            if h in self.item_to_idx:
                item_scores += self.item_sim[self.item_to_idx[h]]
        for item in exclude_ids:
            if item in self.item_to_idx:
                item_scores[self.item_to_idx[item]] = -np.inf
        top_indices = np.argsort(item_scores)[::-1][:K]
        candidates = [self.idx_to_item[idx] for idx in top_indices if item_scores[idx] > -np.inf]
        scores = [item_scores[self.item_to_idx[c]] for c in candidates]
        if scores:
            max_s = max(scores) if max(scores) > 0 else 1
            scores = [s / max_s for s in scores]
        return candidates[:K], scores[:K]


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
            scores = [s / max_s for s in scores]
        return candidates, scores


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def ndcg_at_k(ranked_list, ground_truth, k=10):
    gt_set = set(ground_truth) if isinstance(ground_truth, list) else {ground_truth}
    dcg = 0.0
    for i, item in enumerate(ranked_list[:k]):
        if item in gt_set:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gt_set), k)))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_list, ground_truth, k=10):
    gt_set = set(ground_truth) if isinstance(ground_truth, list) else {ground_truth}
    hits = len(set(ranked_list[:k]) & gt_set)
    return hits / len(gt_set) if gt_set else 0


def hit_at_k(ranked_list, ground_truth, k=10):
    gt_set = set(ground_truth) if isinstance(ground_truth, list) else {ground_truth}
    return 1.0 if len(set(ranked_list[:k]) & gt_set) > 0 else 0.0


def map_at_k(ranked_list, ground_truth, k=10):
    gt_set = set(ground_truth) if isinstance(ground_truth, list) else {ground_truth}
    score = 0.0
    hits = 0
    for i, item in enumerate(ranked_list[:k]):
        if item in gt_set:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(gt_set), k) if gt_set else 0


def bootstrap_ci(scores, n_bootstrap=1000, ci=0.95):
    scores = np.array(scores)
    if len(scores) == 0:
        return 0.0, 0.0, 0.0
    boot_means = [np.mean(np.random.choice(scores, len(scores), replace=True))
                  for _ in range(n_bootstrap)]
    alpha = (1 - ci) / 2
    return np.mean(scores), np.percentile(boot_means, alpha * 100), \
           np.percentile(boot_means, (1 - alpha) * 100)


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment(args):
    data_dir = project_root / 'data'

    # Prepare MovieLens
    if args.download:
        download_movielens(data_dir)

    processed_dir = prepare_movielens(data_dir, n_core=5, max_users=args.max_users)

    # Load data
    train_df = pd.read_parquet(processed_dir / 'train.parquet')
    test_df = pd.read_parquet(processed_dir / 'test.parquet')
    loo_test_df = pd.read_parquet(processed_dir / 'test_loo.parquet')

    n_users = train_df['user_id'].nunique()
    n_items = train_df['item_id'].nunique()
    density = len(train_df) / (n_users * n_items) * 100

    print(f"\n{'='*70}")
    print(f"MOVIELENS-25M EXPERIMENT")
    print(f"{'='*70}")
    print(f"Train: {len(train_df)} interactions")
    print(f"Users: {n_users}, Items: {n_items}")
    print(f"Density: {density:.4f}% (vs Amazon Beauty ~0.47%)")

    np.random.seed(args.seed)

    # Prepare test samples
    user_history = train_df.groupby('user_id')['item_id'].apply(list).to_dict()

    # Last-5 test ground truth
    test_gt_5 = test_df.groupby('user_id')['item_id'].apply(list).to_dict()
    # LOO test ground truth
    test_gt_1 = loo_test_df.groupby('user_id')['item_id'].apply(list).to_dict()

    valid_users = [u for u in test_gt_5 if u in user_history and len(user_history[u]) >= 5]
    n_sample = min(args.n_users, len(valid_users))
    sampled_users = np.random.choice(valid_users, size=n_sample, replace=False)

    print(f"Sampled {n_sample} users for evaluation")

    # Build retrieval methods
    print("\nBuilding retrieval methods...")
    methods = {}

    print("  CF-SVD...")
    cf = CFRecall(n_factors=128)
    cf.fit(train_df)
    methods['CF-SVD'] = cf

    print("  ItemKNN...")
    knn = ItemKNNRecall(n_neighbors=50)
    knn.fit(train_df)
    methods['ItemKNN'] = knn

    print("  Popularity...")
    pop = PopularityRecall()
    pop.fit(train_df)
    methods['Popularity'] = pop

    # ========================================================================
    # EVALUATE UNDER BOTH PROTOCOLS
    # ========================================================================

    results = {}

    for eval_mode in ['loo', 'last5']:
        if eval_mode == 'loo' and args.eval_mode not in ['loo', 'both']:
            continue
        if eval_mode == 'last5' and args.eval_mode not in ['last5', 'both']:
            continue

        test_gt = test_gt_1 if eval_mode == 'loo' else test_gt_5
        n_test_items = 1 if eval_mode == 'loo' else 5

        print(f"\n{'='*70}")
        print(f"EVALUATION MODE: {eval_mode.upper()} (test items per user: {n_test_items})")
        print(f"{'='*70}")

        mode_results = {}

        for K in [100, 500]:
            print(f"\n  K = {K}")
            mode_results[f'K={K}'] = {}

            for method_name, method in methods.items():
                ndcg_scores = []
                recall_scores = []
                hit_scores = []
                map_scores = []
                retrieval_recall_scores = []

                for uid in tqdm(sampled_users, desc=f"    {method_name}", leave=False):
                    gt = test_gt.get(uid, [])
                    if not gt:
                        continue
                    history = user_history.get(uid, [])
                    cands, _ = method.recall(uid, K, history)

                    # Retrieval recall
                    gt_set = set(gt)
                    hits = len(set(cands) & gt_set)
                    retrieval_recall_scores.append(hits / len(gt_set) if gt_set else 0)

                    # Reranking metrics (using CF order = no reranking)
                    ndcg_scores.append(ndcg_at_k(cands, gt, k=10))
                    recall_scores.append(recall_at_k(cands, gt, k=10))
                    hit_scores.append(hit_at_k(cands, gt, k=10))
                    map_scores.append(map_at_k(cands, gt, k=10))

                r_mean, r_lo, r_hi = bootstrap_ci(retrieval_recall_scores)
                n_mean, n_lo, n_hi = bootstrap_ci(ndcg_scores)
                h_mean, h_lo, h_hi = bootstrap_ci(hit_scores)
                m_mean, m_lo, m_hi = bootstrap_ci(map_scores)

                mode_results[f'K={K}'][method_name] = {
                    'retrieval_recall': {'mean': float(r_mean), 'ci': [float(r_lo), float(r_hi)]},
                    'ndcg@10': {'mean': float(n_mean), 'ci': [float(n_lo), float(n_hi)]},
                    'hit@10': {'mean': float(h_mean), 'ci': [float(h_lo), float(h_hi)]},
                    'map@10': {'mean': float(m_mean), 'ci': [float(m_lo), float(m_hi)]}
                }

                print(f"      {method_name:15s}: Recall={r_mean*100:.2f}%, NDCG@10={n_mean:.4f}, Hit@10={h_mean:.4f}")

        results[eval_mode] = mode_results

    # ========================================================================
    # COMPARE LOO vs LAST-5
    # ========================================================================
    if 'loo' in results and 'last5' in results:
        print(f"\n{'='*70}")
        print(f"COMPARISON: LEAVE-ONE-OUT vs LAST-5")
        print(f"{'='*70}")

        for method_name in methods:
            loo_recall = results['loo']['K=100'][method_name]['retrieval_recall']['mean']
            l5_recall = results['last5']['K=100'][method_name]['retrieval_recall']['mean']
            loo_ndcg = results['loo']['K=100'][method_name]['ndcg@10']['mean']
            l5_ndcg = results['last5']['K=100'][method_name]['ndcg@10']['mean']

            print(f"  {method_name:15s}: LOO Recall={loo_recall*100:.2f}% / Last5 Recall={l5_recall*100:.2f}%")
            print(f"  {'':15s}  LOO NDCG={loo_ndcg:.4f} / Last5 NDCG={l5_ndcg:.4f}")

    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'dataset': 'MovieLens-25M',
        'n_users_total': int(n_users),
        'n_items': int(n_items),
        'n_interactions': int(len(train_df)),
        'density': float(density),
        'n_eval_users': int(n_sample),
        'results': results,
        'comparison_with_amazon': {
            'note': 'MovieLens-25M has much higher density than Amazon datasets',
            'ml25m_density': float(density),
            'amazon_beauty_density': 0.47,
            'amazon_movies_density': 0.27,
            'amazon_electronics_density': 0.25
        }
    }

    output_path = project_root / 'experiments' / 'logs' / 'kdd_movielens_experiments.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    # Key finding
    best_recall_loo = 0
    best_recall_l5 = 0
    if 'loo' in results:
        best_recall_loo = max(r['retrieval_recall']['mean']
                              for r in results['loo']['K=100'].values())
    if 'last5' in results:
        best_recall_l5 = max(r['retrieval_recall']['mean']
                             for r in results['last5']['K=100'].values())

    print(f"\nKEY FINDINGS:")
    print(f"  MovieLens-25M density: {density:.4f}% (vs Amazon Beauty: 0.47%)")
    if best_recall_loo:
        print(f"  Best Recall@100 (LOO): {best_recall_loo*100:.2f}%")
    if best_recall_l5:
        print(f"  Best Recall@100 (Last-5): {best_recall_l5*100:.2f}%")
    if best_recall_loo < 0.15:
        print(f"  Even on a HIGH-DENSITY dataset, recall ceiling persists.")
    else:
        print(f"  Higher density does improve recall - ceiling is density-dependent.")
        print(f"  This supports our theory: recall = f(density, catalog_size)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_users', type=int, default=500)
    parser.add_argument('--max_users', type=int, default=10000,
                        help='Max users to keep from ML-25M (for tractability)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--download', action='store_true',
                        help='Download MovieLens-25M')
    parser.add_argument('--eval_mode', type=str, default='both',
                        choices=['loo', 'last5', 'both'])
    args = parser.parse_args()

    run_experiment(args)


if __name__ == '__main__':
    main()
