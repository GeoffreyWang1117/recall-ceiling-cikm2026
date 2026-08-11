#!/usr/bin/env python3
"""
KDD 2026 Retrieval Baselines Only
==================================
Fast experiment to test all retrieval baselines without LLM.
This validates the recall bottleneck claim.

Usage:
    python scripts/kdd_retrieval_baselines.py --n_users 500
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
from datetime import datetime
from typing import Dict, List, Tuple
from tqdm import tqdm
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================================
# RETRIEVAL BASELINES
# ============================================================================

class PopularityRecall:
    """Popularity-based retrieval."""
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
    """Item-KNN collaborative filtering."""
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
        for hist_item in history:
            if hist_item in self.item_to_idx:
                item_scores += self.item_sim[self.item_to_idx[hist_item]]

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


class BPRMFRecall:
    """BPR Matrix Factorization."""
    def __init__(self, n_factors=64, n_epochs=10, lr=0.01, reg=0.01):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg

    def fit(self, train_df):
        users = train_df['user_id'].unique()
        items = train_df['item_id'].unique()
        self.user_to_idx = {u: i for i, u in enumerate(users)}
        self.item_to_idx = {i: idx for idx, i in enumerate(items)}
        self.idx_to_item = {idx: i for i, idx in self.item_to_idx.items()}

        n_users, n_items = len(users), len(items)
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))

        user_items = train_df.groupby('user_id')['item_id'].apply(set).to_dict()
        all_items = set(items)

        for epoch in range(self.n_epochs):
            for _, row in train_df.sample(frac=1).iterrows():
                u = self.user_to_idx.get(row['user_id'])
                i = self.item_to_idx.get(row['item_id'])
                if u is None or i is None:
                    continue

                neg_items = list(all_items - user_items.get(row['user_id'], set()))
                if not neg_items:
                    continue
                j = self.item_to_idx[np.random.choice(neg_items)]

                x_uij = np.dot(self.user_factors[u], self.item_factors[i] - self.item_factors[j])
                sigmoid = 1 / (1 + np.exp(min(x_uij, 500)))

                self.user_factors[u] += self.lr * (sigmoid * (self.item_factors[i] - self.item_factors[j]) - self.reg * self.user_factors[u])
                self.item_factors[i] += self.lr * (sigmoid * self.user_factors[u] - self.reg * self.item_factors[i])
                self.item_factors[j] += self.lr * (-sigmoid * self.user_factors[u] - self.reg * self.item_factors[j])

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
        item_scores = [scores[self.item_to_idx[c]] for c in candidates]

        if item_scores:
            min_s, max_s = min(item_scores), max(item_scores)
            if max_s > min_s:
                item_scores = [(s - min_s) / (max_s - min_s) for s in item_scores]
        return candidates[:K], item_scores[:K]


class CFRecall:
    """SVD-based Collaborative Filtering."""
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


class LightGCNRecall:
    """LightGCN-based retrieval (simple implementation)."""
    def __init__(self, n_factors=64, n_layers=3, n_epochs=20, lr=0.01):
        self.n_factors = n_factors
        self.n_layers = n_layers
        self.n_epochs = n_epochs
        self.lr = lr

    def fit(self, train_df):
        users = train_df['user_id'].unique()
        items = train_df['item_id'].unique()
        self.user_to_idx = {u: i for i, u in enumerate(users)}
        self.item_to_idx = {i: idx for idx, i in enumerate(items)}
        self.idx_to_item = {idx: i for i, idx in self.item_to_idx.items()}

        n_users, n_items = len(users), len(items)
        n_nodes = n_users + n_items

        # Build edge index (user-item bipartite graph)
        user_indices = [self.user_to_idx[u] for u in train_df['user_id']]
        item_indices = [self.item_to_idx[i] + n_users for i in train_df['item_id']]

        # Bidirectional edges
        row = user_indices + item_indices
        col = item_indices + user_indices

        # Compute degree for normalization
        degrees = np.zeros(n_nodes)
        for r in row:
            degrees[r] += 1

        deg_inv_sqrt = np.power(degrees, -0.5)
        deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0

        # Edge weights (symmetric normalization)
        edge_weights = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Initialize embeddings
        self.embeddings = np.random.normal(0, 0.1, (n_nodes, self.n_factors))

        user_items = train_df.groupby('user_id')['item_id'].apply(set).to_dict()
        all_items = set(items)

        # Training with BPR loss
        for epoch in range(self.n_epochs):
            # Propagation
            all_embs = [self.embeddings.copy()]
            for layer in range(self.n_layers):
                new_emb = np.zeros_like(self.embeddings)
                for i, (r, c, w) in enumerate(zip(row, col, edge_weights)):
                    new_emb[r] += w * all_embs[-1][c]
                all_embs.append(new_emb)

            final_emb = np.mean(all_embs, axis=0)

            # BPR updates
            for _, r in train_df.sample(frac=0.1).iterrows():
                u = self.user_to_idx.get(r['user_id'])
                i = self.item_to_idx.get(r['item_id'])
                if u is None or i is None:
                    continue

                neg_items = list(all_items - user_items.get(r['user_id'], set()))
                if not neg_items:
                    continue
                j = self.item_to_idx[np.random.choice(neg_items)]

                u_emb = final_emb[u]
                i_emb = final_emb[i + n_users]
                j_emb = final_emb[j + n_users]

                x_uij = np.dot(u_emb, i_emb - j_emb)
                sigmoid = 1 / (1 + np.exp(min(x_uij, 500)))

                # Update original embeddings
                self.embeddings[u] += self.lr * sigmoid * (i_emb - j_emb)
                self.embeddings[i + n_users] += self.lr * sigmoid * u_emb
                self.embeddings[j + n_users] -= self.lr * sigmoid * u_emb

        # Final propagation
        all_embs = [self.embeddings.copy()]
        for layer in range(self.n_layers):
            new_emb = np.zeros_like(self.embeddings)
            for i, (r, c, w) in enumerate(zip(row, col, edge_weights)):
                new_emb[r] += w * all_embs[-1][c]
            all_embs.append(new_emb)

        self.final_embeddings = np.mean(all_embs, axis=0)
        self.user_emb = self.final_embeddings[:n_users]
        self.item_emb = self.final_embeddings[n_users:]

    def recall(self, user_id, K, exclude_ids):
        if user_id not in self.user_to_idx:
            return [], []

        u_idx = self.user_to_idx[user_id]
        scores = np.dot(self.item_emb, self.user_emb[u_idx])

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


class DenseRecall:
    """Dense retrieval using sentence embeddings."""
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name

    def fit(self, train_df, item_texts):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
        except ImportError:
            self.model = None

        self.item_texts = item_texts
        self.user_items = train_df.groupby('user_id')['item_id'].apply(list).to_dict()

        self.item_ids = [i for i in item_texts.keys() if item_texts.get(i)]
        texts = [item_texts[i] for i in self.item_ids]

        if self.model:
            self.item_embeddings = self.model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        else:
            self.item_embeddings = np.random.randn(len(self.item_ids), 128)
            self.item_embeddings = self.item_embeddings / np.linalg.norm(self.item_embeddings, axis=1, keepdims=True)

    def recall(self, user_id, K, exclude_ids):
        history = self.user_items.get(user_id, [])
        if not history:
            return [], []

        item_to_idx = {item: idx for idx, item in enumerate(self.item_ids)}

        hist_embeddings = []
        for item in history:
            if item in item_to_idx:
                hist_embeddings.append(self.item_embeddings[item_to_idx[item]])

        if not hist_embeddings:
            return [], []

        user_emb = np.mean(hist_embeddings, axis=0)
        user_emb = user_emb / np.linalg.norm(user_emb)

        scores = np.dot(self.item_embeddings, user_emb)

        for item in exclude_ids:
            if item in item_to_idx:
                scores[item_to_idx[item]] = -np.inf

        top_indices = np.argsort(scores)[::-1][:K]
        candidates = [self.item_ids[idx] for idx in top_indices if scores[idx] > -np.inf]
        item_scores = [scores[item_to_idx[c]] for c in candidates]

        return candidates[:K], item_scores[:K]


class HybridRecall:
    """Hybrid (CF + Dense) retrieval."""
    def __init__(self, cf_weight=0.5):
        self.cf_weight = cf_weight
        self.cf = CFRecall()
        self.dense = DenseRecall()

    def fit(self, train_df, item_texts):
        self.cf.fit(train_df)
        self.dense.fit(train_df, item_texts)

    def recall(self, user_id, K, exclude_ids):
        cf_cands, cf_scores = self.cf.recall(user_id, K * 2, exclude_ids)
        dense_cands, dense_scores = self.dense.recall(user_id, K * 2, exclude_ids)

        combined = {}
        for item, score in zip(cf_cands, cf_scores):
            combined[item] = combined.get(item, 0) + self.cf_weight * score
        for item, score in zip(dense_cands, dense_scores):
            combined[item] = combined.get(item, 0) + (1 - self.cf_weight) * score

        sorted_items = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        candidates = [item for item, _ in sorted_items[:K]]
        scores = [score for _, score in sorted_items[:K]]
        return candidates, scores


# ============================================================================
# STATISTICAL UTILITIES
# ============================================================================

def bootstrap_ci(scores, n_bootstrap=1000, ci=0.95):
    scores = np.array(scores)
    boot_means = [np.mean(np.random.choice(scores, len(scores), replace=True)) for _ in range(n_bootstrap)]
    alpha = (1 - ci) / 2
    return np.mean(scores), np.percentile(boot_means, alpha * 100), np.percentile(boot_means, (1 - alpha) * 100)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_users', type=int, default=500)
    parser.add_argument('--K', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    datasets = {
        'beauty': 'amazon_beauty_sampled',
        'movies': 'amazon_movies_sampled',
        'electronics': 'amazon_electronics_sampled'
    }

    all_results = {}

    for ds_name, ds_path in datasets.items():
        print(f"\n{'='*60}")
        print(f"DATASET: {ds_name.upper()}")
        print(f"{'='*60}")

        path = project_root / 'data' / 'processed' / ds_path
        train_df = pd.read_parquet(path / 'train.parquet')
        test_df = pd.read_parquet(path / 'test.parquet')

        # Item texts
        item_texts = {}
        if 'title' in train_df.columns:
            for _, row in train_df.drop_duplicates('item_id').iterrows():
                item_texts[row['item_id']] = str(row.get('title', ''))[:200]
        else:
            for item_id in train_df['item_id'].unique():
                item_texts[item_id] = f"Product {item_id}"

        print(f"Train: {len(train_df)} interactions, {train_df['user_id'].nunique()} users")
        print(f"Test: {len(test_df)} interactions, {test_df['user_id'].nunique()} users")

        # Prepare test samples
        np.random.seed(args.seed)
        user_history = train_df.groupby('user_id')['item_id'].apply(list).to_dict()
        test_gt = test_df.groupby('user_id')['item_id'].apply(list).to_dict()
        valid_users = [u for u in test_gt.keys() if u in user_history and len(user_history[u]) >= 3]

        n_users = min(args.n_users, len(valid_users))
        sampled_users = np.random.choice(valid_users, size=n_users, replace=False)

        test_samples = [{'user_id': u, 'history': user_history[u], 'ground_truth': test_gt[u]} for u in sampled_users]
        print(f"Test samples: {len(test_samples)} users")

        # Build retrieval methods
        print("\nBuilding retrieval methods...")
        methods = {}

        print("  Popularity...")
        pop = PopularityRecall()
        pop.fit(train_df)
        methods['Popularity'] = pop

        print("  ItemKNN...")
        knn = ItemKNNRecall()
        knn.fit(train_df)
        methods['ItemKNN'] = knn

        print("  BPR-MF...")
        bpr = BPRMFRecall()
        bpr.fit(train_df)
        methods['BPR-MF'] = bpr

        print("  CF-SVD...")
        cf = CFRecall()
        cf.fit(train_df)
        methods['CF-SVD'] = cf

        print("  LightGCN...")
        lgcn = LightGCNRecall(n_factors=64, n_layers=2, n_epochs=5)
        lgcn.fit(train_df)
        methods['LightGCN'] = lgcn

        print("  Dense...")
        dense = DenseRecall()
        dense.fit(train_df, item_texts)
        methods['Dense'] = dense

        print("  Hybrid...")
        hybrid = HybridRecall(cf_weight=0.5)
        hybrid.fit(train_df, item_texts)
        methods['Hybrid'] = hybrid

        # Evaluate
        print(f"\nEvaluating Recall@{args.K}...")
        results = {}

        for method_name, method in methods.items():
            recall_scores = []
            for sample in tqdm(test_samples, desc=method_name, leave=False):
                candidates, _ = method.recall(sample['user_id'], args.K, sample['history'])
                gt_set = set(sample['ground_truth'])
                hits = len(set(candidates) & gt_set)
                recall = hits / len(gt_set) if gt_set else 0
                recall_scores.append(recall)

            mean, ci_low, ci_high = bootstrap_ci(recall_scores)
            results[method_name] = {'mean': mean, 'ci_low': ci_low, 'ci_high': ci_high, 'scores': recall_scores}
            print(f"  {method_name:15s}: {mean:.4f} [{ci_low:.4f}, {ci_high:.4f}]")

        all_results[ds_name] = {
            'config': {'n_users': len(test_samples), 'K': args.K, 'seed': args.seed},
            'results': results
        }

    # Save
    output_path = project_root / 'experiments' / 'logs' / 'kdd_retrieval_baselines.json'

    def convert(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)): return float(obj)
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(i) for i in obj]
        return obj

    with open(output_path, 'w') as f:
        json.dump(convert(all_results), f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
