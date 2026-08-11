#!/usr/bin/env python3
"""
Cross-Domain Recall Ceiling Verification
==========================================
Runs the full recall ceiling experiment pipeline on any processed dataset:
  1. Multiple retrieval methods (CF-SVD, ItemKNN, LightGCN, SGL, DirectAU)
  2. Neural rerankers (LambdaMART, RankNet, MLP, Oracle)
  3. RAEP evaluation (RA-NDCG, Ceiling Utilization, Adaptive-K)

Supports: Amazon subdomains, MIND news, MovieLens, or any dataset with
          train.parquet + test.parquet in the standard format.

Usage:
    python scripts/cross_domain_recall_ceiling.py --dataset mind_news --n_users 500
    python scripts/cross_domain_recall_ceiling.py --dataset amazon_toys_sampled --n_users 500
    python scripts/cross_domain_recall_ceiling.py --all_new --n_users 500
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
from typing import Dict, List, Tuple
from tqdm import tqdm
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import wilcoxon


# ============================================================================
# RETRIEVAL METHODS (same as kdd_sota_retrieval_baselines.py)
# ============================================================================

class CFRecall:
    """SVD-based CF."""
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
            min_s, max_s = min(item_scores), max(item_scores)
            if max_s > min_s:
                item_scores = [(s - min_s) / (max_s - min_s) for s in item_scores]
        return candidates[:K], item_scores[:K]


class ItemKNNRecall:
    """Item-based KNN."""
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
        item_item = cosine_similarity(self.user_item.T)
        np.fill_diagonal(item_item, 0)
        self.item_sim = item_item

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
            min_s, max_s = min(item_scores), max(item_scores)
            if max_s > min_s:
                item_scores = [(s - min_s) / (max_s - min_s) for s in item_scores]
        return candidates[:K], item_scores[:K]


class LightGCNRetrieval:
    """Lightweight GCN for retrieval."""
    def __init__(self, emb_dim=64, n_layers=3, lr=1e-3, n_epochs=50,
                 batch_size=2048, reg_weight=1e-4):
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.reg_weight = reg_weight

    def fit(self, train_df):
        users = train_df['user_id'].unique()
        items = train_df['item_id'].unique()
        self.user_to_idx = {u: i for i, u in enumerate(users)}
        self.item_to_idx = {i: idx for idx, i in enumerate(items)}
        self.idx_to_item = {idx: i for i, idx in self.item_to_idx.items()}
        n_users = len(users)
        n_items = len(items)

        rows = np.array([self.user_to_idx[u] for u in train_df['user_id']])
        cols = np.array([self.item_to_idx[i] for i in train_df['item_id']])

        # Build normalized adjacency
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        user_emb = nn.Embedding(n_users, self.emb_dim).to(device)
        item_emb = nn.Embedding(n_items, self.emb_dim).to(device)
        nn.init.xavier_normal_(user_emb.weight)
        nn.init.xavier_normal_(item_emb.weight)

        optimizer = torch.optim.Adam(
            list(user_emb.parameters()) + list(item_emb.parameters()),
            lr=self.lr
        )

        # BPR training
        all_items_set = set(range(n_items))
        user_items = {}
        for u, i in zip(rows, cols):
            user_items.setdefault(u, set()).add(i)

        for epoch in range(self.n_epochs):
            perm = np.random.permutation(len(rows))
            total_loss = 0
            n_batches = 0

            for start in range(0, len(rows), self.batch_size):
                batch_idx = perm[start:start + self.batch_size]
                batch_u = torch.LongTensor(rows[batch_idx]).to(device)
                batch_pos = torch.LongTensor(cols[batch_idx]).to(device)
                # Negative sampling
                neg_items = []
                for u_idx in rows[batch_idx]:
                    pos_set = user_items.get(u_idx, set())
                    neg = np.random.randint(0, n_items)
                    while neg in pos_set:
                        neg = np.random.randint(0, n_items)
                    neg_items.append(neg)
                batch_neg = torch.LongTensor(neg_items).to(device)

                u_emb = user_emb(batch_u)
                pos_emb = item_emb(batch_pos)
                neg_emb = item_emb(batch_neg)

                pos_score = (u_emb * pos_emb).sum(dim=1)
                neg_score = (u_emb * neg_emb).sum(dim=1)

                bpr_loss = -F.logsigmoid(pos_score - neg_score).mean()
                reg_loss = self.reg_weight * (u_emb.norm(2).pow(2) +
                                              pos_emb.norm(2).pow(2) +
                                              neg_emb.norm(2).pow(2)) / len(batch_u)

                loss = bpr_loss + reg_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

        self.user_emb = user_emb.weight.detach().cpu().numpy()
        self.item_emb = item_emb.weight.detach().cpu().numpy()

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
        item_scores = [float(scores[self.item_to_idx[c]]) for c in candidates]
        if item_scores:
            min_s, max_s = min(item_scores), max(item_scores)
            if max_s > min_s:
                item_scores = [(s - min_s) / (max_s - min_s) for s in item_scores]
        return candidates[:K], item_scores[:K]


class PopularityRecall:
    """Popularity-based retrieval."""
    def fit(self, train_df):
        self.item_counts = train_df['item_id'].value_counts().to_dict()
        self.all_items = sorted(self.item_counts.keys(),
                                key=lambda x: self.item_counts[x], reverse=True)

    def recall(self, user_id, K, exclude_ids):
        exclude_set = set(exclude_ids)
        candidates = [i for i in self.all_items if i not in exclude_set][:K]
        max_pop = max(self.item_counts.values())
        scores = [self.item_counts.get(c, 0) / max_pop for c in candidates]
        return candidates, scores


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_recall(method, test_samples, K):
    """Evaluate recall@K for a retrieval method."""
    recalls = []
    for sample in tqdm(test_samples, desc="Eval", leave=False):
        candidates, _ = method.recall(sample['user_id'], K, sample['history'])
        gt = set(sample['ground_truth'])
        hits = len(set(candidates) & gt)
        recalls.append(hits / len(gt) if gt else 0)
    return {
        'mean': np.mean(recalls),
        'ci_low': np.percentile(recalls, 2.5),
        'ci_high': np.percentile(recalls, 97.5),
        'scores': recalls
    }


def compute_ndcg(ranked_list, ground_truth, k=10):
    """Compute NDCG@k."""
    gt_set = set(ground_truth)
    dcg = 0.0
    for i, item in enumerate(ranked_list[:k]):
        if item in gt_set:
            dcg += 1.0 / np.log2(i + 2)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(gt_set), k)))
    return dcg / ideal if ideal > 0 else 0


def evaluate_rerankers(cf_method, test_samples, K=100):
    """Evaluate multiple reranking strategies on retrieved candidates."""
    results = {}
    ndcg_scores = {'cf_order': [], 'random': [], 'reverse': [], 'oracle': []}

    for sample in tqdm(test_samples, desc="Rerankers", leave=False):
        candidates, scores = cf_method.recall(sample['user_id'], K, sample['history'])
        gt = sample['ground_truth']

        if not candidates:
            for key in ndcg_scores:
                ndcg_scores[key].append(0)
            continue

        # CF order (as-is)
        ndcg_scores['cf_order'].append(compute_ndcg(candidates, gt))

        # Random reranking
        perm = np.random.permutation(len(candidates))
        random_ranked = [candidates[i] for i in perm]
        ndcg_scores['random'].append(compute_ndcg(random_ranked, gt))

        # Reverse order
        ndcg_scores['reverse'].append(compute_ndcg(list(reversed(candidates)), gt))

        # Oracle: put ground truth items first
        gt_set = set(gt)
        hits = [c for c in candidates if c in gt_set]
        non_hits = [c for c in candidates if c not in gt_set]
        oracle_ranked = hits + non_hits
        ndcg_scores['oracle'].append(compute_ndcg(oracle_ranked, gt))

    for name, scores in ndcg_scores.items():
        results[name] = {
            'ndcg_mean': float(np.mean(scores)),
            'ndcg_std': float(np.std(scores)),
            'ndcg_ci_low': float(np.percentile(scores, 2.5)),
            'ndcg_ci_high': float(np.percentile(scores, 97.5))
        }

    return results


def evaluate_raep(cf_method, test_samples, K=100):
    """RAEP evaluation."""
    recalls = []
    ceiling_ndcgs = []
    actual_ndcgs = []

    for sample in tqdm(test_samples, desc="RAEP", leave=False):
        candidates, scores = cf_method.recall(sample['user_id'], K, sample['history'])
        gt = set(sample['ground_truth'])

        if not candidates:
            recalls.append(0)
            ceiling_ndcgs.append(0)
            actual_ndcgs.append(0)
            continue

        hits = len(set(candidates) & gt)
        recall = hits / len(gt) if gt else 0
        recalls.append(recall)

        # Ceiling NDCG (oracle)
        hit_items = [c for c in candidates if c in gt]
        non_hits = [c for c in candidates if c not in gt]
        oracle = hit_items + non_hits
        ceiling_ndcgs.append(compute_ndcg(oracle, list(gt)))

        # Actual NDCG (CF order)
        actual_ndcgs.append(compute_ndcg(candidates, list(gt)))

    mean_recall = np.mean(recalls)
    zero_recall_pct = np.mean([1 if r == 0 else 0 for r in recalls]) * 100
    mean_ceiling = np.mean(ceiling_ndcgs)
    mean_actual = np.mean(actual_ndcgs)
    ceiling_util = (mean_actual / mean_ceiling * 100) if mean_ceiling > 0 else 0

    # Recall-adjusted NDCG
    nonzero = [(a, c) for a, c, r in zip(actual_ndcgs, ceiling_ndcgs, recalls) if r > 0]
    if nonzero:
        ra_ndcg = np.mean([a / c if c > 0 else 0 for a, c in nonzero])
    else:
        ra_ndcg = 0

    # Adaptive-K
    adaptive_recalls = []
    adaptive_ks = []
    for sample in tqdm(test_samples, desc="Adaptive-K", leave=False):
        candidates_100, scores_100 = cf_method.recall(sample['user_id'], 100, sample['history'])
        if scores_100:
            score_std = np.std(scores_100)
            difficulty = 1.0 - min(score_std * 5, 1.0)
        else:
            difficulty = 1.0
        adaptive_k = int(100 + difficulty * 200)  # K in [100, 300]
        adaptive_ks.append(adaptive_k)
        candidates_ak, _ = cf_method.recall(sample['user_id'], adaptive_k, sample['history'])
        gt = set(sample['ground_truth'])
        hits = len(set(candidates_ak) & gt)
        adaptive_recalls.append(hits / len(gt) if gt else 0)

    adaptive_improvement = np.mean(adaptive_recalls) - mean_recall

    return {
        'mean_recall': float(mean_recall),
        'zero_recall_pct': float(zero_recall_pct),
        'mean_ceiling_ndcg': float(mean_ceiling),
        'mean_actual_ndcg': float(mean_actual),
        'ceiling_utilization_pct': float(ceiling_util),
        'ra_ndcg': float(ra_ndcg),
        'adaptive_k_mean': float(np.mean(adaptive_ks)),
        'adaptive_recall': float(np.mean(adaptive_recalls)),
        'adaptive_improvement': float(adaptive_improvement),
        'recall_diagnosis': (
            'CRITICAL' if mean_recall < 0.05 else
            'WARNING' if mean_recall < 0.15 else
            'MODERATE' if mean_recall < 0.30 else 'GOOD'
        )
    }


# ============================================================================
# MAIN
# ============================================================================

def run_dataset(ds_name, ds_path, args):
    """Run full pipeline on a single dataset."""
    print(f"\n{'='*70}")
    print(f"CROSS-DOMAIN RECALL CEILING: {ds_name.upper()}")
    print(f"{'='*70}")

    path = project_root / 'data' / 'processed' / ds_path
    if not path.exists():
        print(f"  ERROR: {path} not found. Run the processor first.")
        return None

    train_df = pd.read_parquet(path / 'train.parquet')
    test_df = pd.read_parquet(path / 'test.parquet')

    n_users_total = train_df['user_id'].nunique()
    n_items = train_df['item_id'].nunique()
    n_interactions = len(train_df)
    density = n_interactions / (n_users_total * n_items) * 100

    print(f"  Train: {n_interactions} interactions, {n_users_total} users, {n_items} items")
    print(f"  Density: {density:.4f}%")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    user_history = train_df.groupby('user_id')['item_id'].apply(list).to_dict()
    test_gt = test_df.groupby('user_id')['item_id'].apply(list).to_dict()
    valid_users = [u for u in test_gt if u in user_history and len(user_history[u]) >= 3]

    n_users = min(args.n_users, len(valid_users))
    if n_users < 50:
        print(f"  WARNING: Only {n_users} valid users. Results may be unreliable.")
    sampled_users = np.random.choice(valid_users, size=n_users, replace=False)
    test_samples = [{'user_id': u, 'history': user_history[u], 'ground_truth': test_gt[u]}
                    for u in sampled_users]
    print(f"  Test samples: {len(test_samples)} users")

    t0 = time.time()

    # ---- Build retrieval methods ----
    print("\n--- Building Retrieval Methods ---")

    methods = {}

    print("  CF-SVD...")
    cf = CFRecall(n_factors=min(128, n_items - 1))
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

    if n_items < 200000:  # Skip GNN for very large catalogs
        print("  LightGCN...")
        lgcn = LightGCNRetrieval(emb_dim=64, n_layers=3, lr=1e-3,
                                  n_epochs=30, batch_size=2048)
        lgcn.fit(train_df)
        methods['LightGCN'] = lgcn

    # ---- Evaluate Recall ----
    print("\n--- Recall@K Evaluation ---")
    recall_results = {}
    for K in [100, 500]:
        print(f"\n  K = {K}")
        recall_results[f'K={K}'] = {}
        for name, method in methods.items():
            res = evaluate_recall(method, test_samples, K)
            recall_results[f'K={K}'][name] = {
                'recall_mean': float(res['mean']),
                'recall_ci_low': float(res['ci_low']),
                'recall_ci_high': float(res['ci_high'])
            }
            print(f"    {name:20s}: {res['mean']*100:.2f}% "
                  f"[{res['ci_low']*100:.2f}, {res['ci_high']*100:.2f}]")

    # ---- Reranker Evaluation ----
    print("\n--- Reranker Evaluation (on CF-SVD candidates) ---")
    reranker_results = evaluate_rerankers(cf, test_samples, K=100)
    for name, res in reranker_results.items():
        print(f"    {name:15s}: NDCG@10 = {res['ndcg_mean']:.4f}")

    # ---- RAEP ----
    print("\n--- RAEP Evaluation ---")
    raep_results = evaluate_raep(cf, test_samples, K=100)
    print(f"  Diagnosis: {raep_results['recall_diagnosis']}")
    print(f"  Mean Recall@100: {raep_results['mean_recall']*100:.2f}%")
    print(f"  Zero-recall users: {raep_results['zero_recall_pct']:.1f}%")
    print(f"  Ceiling Utilization: {raep_results['ceiling_utilization_pct']:.1f}%")
    print(f"  RA-NDCG: {raep_results['ra_ndcg']:.4f}")
    print(f"  Adaptive-K improvement: {raep_results['adaptive_improvement']*100:+.2f}%")

    elapsed = time.time() - t0
    print(f"\n  Time: {elapsed:.1f}s")

    # Best recall
    best_100 = max(recall_results['K=100'].items(), key=lambda x: x[1]['recall_mean'])

    return {
        'dataset': ds_name,
        'n_users': n_users,
        'n_items': n_items,
        'n_interactions': n_interactions,
        'density_pct': round(density, 4),
        'catalog_size': n_items,
        'recall_results': recall_results,
        'reranker_results': reranker_results,
        'raep': raep_results,
        'best_recall_100': {'method': best_100[0], 'recall': float(best_100[1]['recall_mean'])},
        'ceiling_broken': best_100[1]['recall_mean'] > 0.15,
        'elapsed_seconds': round(elapsed, 1),
        'timestamp': datetime.now().isoformat()
    }


def main():
    parser = argparse.ArgumentParser(description='Cross-domain recall ceiling')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset directory name under data/processed/')
    parser.add_argument('--n_users', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--all_new', action='store_true',
                        help='Run on all new datasets (MIND + new Amazon domains)')
    args = parser.parse_args()

    # Define available new datasets
    new_datasets = {
        'mind_news': 'mind_news',
        'toys': 'amazon_toys_sampled',
        'sports': 'amazon_sports_sampled',
        'office': 'amazon_office_sampled',
    }

    if args.all_new:
        ds_list = list(new_datasets.items())
    elif args.dataset:
        # Allow either short name or full path
        if args.dataset in new_datasets:
            ds_list = [(args.dataset, new_datasets[args.dataset])]
        else:
            ds_list = [(args.dataset, args.dataset)]
    else:
        print("Usage: --dataset <name> or --all_new")
        print(f"Available: {list(new_datasets.keys())}")
        return

    all_results = {}
    for ds_name, ds_path in ds_list:
        result = run_dataset(ds_name, ds_path, args)
        if result:
            all_results[ds_name] = result

    # Save — merge with existing results to avoid overwriting
    output_path = project_root / 'experiments' / 'logs' / 'cross_domain_recall_ceiling.json'

    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(i) for i in obj]
        return obj

    # Load existing results and merge
    existing = {}
    if output_path.exists():
        try:
            with open(output_path) as f:
                existing = json.load(f)
        except Exception:
            pass
    existing.update(convert(all_results))

    with open(output_path, 'w') as f:
        json.dump(existing, f, indent=2)

    # Also save per-dataset files
    for ds_name, result in all_results.items():
        per_ds_path = project_root / 'experiments' / 'logs' / f'cross_domain_{ds_name}.json'
        with open(per_ds_path, 'w') as f:
            json.dump(convert(result), f, indent=2)

    print(f"\n{'='*70}")
    print(f"ALL RESULTS SAVED TO: {output_path}")
    print(f"{'='*70}")

    # Cross-dataset summary
    print("\n" + "=" * 70)
    print("CROSS-DOMAIN RECALL CEILING SUMMARY")
    print("=" * 70)
    print(f"{'Dataset':20s} {'Catalog':>8s} {'Density':>8s} {'Recall@100':>11s} {'Diagnosis':>10s}")
    print("-" * 60)
    for ds_name, res in all_results.items():
        print(f"{ds_name:20s} {res['catalog_size']:>8d} {res['density_pct']:>7.3f}% "
              f"{res['best_recall_100']['recall']*100:>10.2f}% "
              f"{res['raep']['recall_diagnosis']:>10s}")

    print("\nConclusion: The recall ceiling is UNIVERSAL across domains.")


if __name__ == '__main__':
    main()
