#!/usr/bin/env python3
"""
Catalog Size vs Recall Ceiling Analysis
=========================================
Systematic analysis across all datasets to quantify how catalog size
and data density independently affect the recall ceiling.

Produces:
1. Catalog size vs Recall@K scatter + fitted curve
2. Density vs Recall@K scatter + fitted curve
3. Multivariate analysis: recall = f(catalog_size, density, K)

Usage:
    python scripts/catalog_size_analysis.py
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression
from tqdm import tqdm


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


def evaluate_dataset(ds_name, ds_path, n_users=500, seed=42):
    """Evaluate recall@K for a dataset at multiple K values."""
    path = project_root / 'data' / 'processed' / ds_path
    if not path.exists():
        print(f"  SKIP: {ds_path} not found")
        return None

    train_df = pd.read_parquet(path / 'train.parquet')
    test_df = pd.read_parquet(path / 'test.parquet')

    n_users_total = train_df['user_id'].nunique()
    n_items = train_df['item_id'].nunique()
    n_interactions = len(train_df)
    density = n_interactions / (n_users_total * n_items) if n_users_total * n_items > 0 else 0

    print(f"\n  {ds_name}: {n_users_total} users, {n_items} items, density={density*100:.4f}%")

    np.random.seed(seed)
    user_history = train_df.groupby('user_id')['item_id'].apply(list).to_dict()
    test_gt = test_df.groupby('user_id')['item_id'].apply(list).to_dict()
    valid_users = [u for u in test_gt if u in user_history and len(user_history[u]) >= 3]

    n_eval = min(n_users, len(valid_users))
    if n_eval < 50:
        print(f"    WARNING: Only {n_eval} valid users")
        return None

    sampled_users = np.random.choice(valid_users, size=n_eval, replace=False)
    test_samples = [{'user_id': u, 'history': user_history[u], 'ground_truth': test_gt[u]}
                    for u in sampled_users]

    # Build CF model
    cf = CFRecall(n_factors=min(128, n_items - 1))
    cf.fit(train_df)

    # Evaluate at multiple K
    results = {}
    for K in [20, 50, 100, 200, 500]:
        recalls = []
        for sample in test_samples:
            candidates, _ = cf.recall(sample['user_id'], K, sample['history'])
            gt = set(sample['ground_truth'])
            hits = len(set(candidates) & gt)
            recalls.append(hits / len(gt) if gt else 0)
        results[f'K={K}'] = float(np.mean(recalls))
        print(f"    Recall@{K}: {np.mean(recalls)*100:.2f}%")

    return {
        'dataset': ds_name,
        'n_users': n_users_total,
        'n_items': n_items,
        'catalog_size': n_items,
        'n_interactions': n_interactions,
        'density': float(density),
        'avg_interactions_per_user': float(n_interactions / n_users_total),
        'recall_at_K': results
    }


def main():
    print("=" * 70)
    print("CATALOG SIZE vs RECALL CEILING ANALYSIS")
    print("=" * 70)

    # All available datasets
    datasets = {
        'beauty': 'amazon_beauty_sampled',
        'movies': 'amazon_movies_sampled',
        'electronics': 'amazon_electronics_sampled',
        'toys': 'amazon_toys_sampled',
        'sports': 'amazon_sports_sampled',
        'office': 'amazon_office_sampled',
        'mind_news': 'mind_news',
        'movielens': 'movielens_25m',
    }

    all_results = {}
    for ds_name, ds_path in datasets.items():
        result = evaluate_dataset(ds_name, ds_path, n_users=500)
        if result:
            all_results[ds_name] = result

    if len(all_results) < 3:
        print("\nNot enough datasets for analysis. Need at least 3.")
        return

    # ---- Analysis ----
    print("\n" + "=" * 70)
    print("ANALYSIS RESULTS")
    print("=" * 70)

    # Extract features
    names = list(all_results.keys())
    catalogs = np.array([all_results[n]['catalog_size'] for n in names])
    densities = np.array([all_results[n]['density'] for n in names])
    avg_interactions = np.array([all_results[n]['avg_interactions_per_user'] for n in names])
    recall_100 = np.array([all_results[n]['recall_at_K']['K=100'] for n in names])
    recall_500 = np.array([all_results[n]['recall_at_K'].get('K=500', 0) for n in names])

    # 1. Catalog Size vs Recall
    print("\n--- Catalog Size vs Recall@100 ---")
    log_catalog = np.log10(catalogs)
    correlation = np.corrcoef(log_catalog, recall_100)[0, 1]
    print(f"  Pearson correlation (log catalog, recall): {correlation:.3f}")

    # Fit: recall ∝ a * log(K/catalog) + b
    X_log = log_catalog.reshape(-1, 1)
    reg = LinearRegression().fit(X_log, recall_100)
    print(f"  Linear fit: recall = {reg.coef_[0]:.4f} * log10(catalog) + {reg.intercept_:.4f}")
    print(f"  R² = {reg.score(X_log, recall_100):.3f}")

    for i, n in enumerate(names):
        pred = reg.predict(log_catalog[i:i+1].reshape(-1, 1))[0]
        print(f"    {n:15s}: catalog={catalogs[i]:>6d}, recall={recall_100[i]*100:>6.2f}%, predicted={pred*100:>6.2f}%")

    # 2. Density vs Recall
    print("\n--- Density vs Recall@100 ---")
    log_density = np.log10(densities + 1e-10)
    correlation_d = np.corrcoef(log_density, recall_100)[0, 1]
    print(f"  Pearson correlation (log density, recall): {correlation_d:.3f}")

    # 3. Multivariate: recall = f(log_catalog, log_density)
    print("\n--- Multivariate Analysis ---")
    X_multi = np.column_stack([log_catalog, log_density])
    reg_multi = LinearRegression().fit(X_multi, recall_100)
    print(f"  recall = {reg_multi.coef_[0]:.4f}*log(catalog) + {reg_multi.coef_[1]:.4f}*log(density) + {reg_multi.intercept_:.4f}")
    print(f"  R² = {reg_multi.score(X_multi, recall_100):.3f}")

    # 4. K/Catalog ratio analysis
    print("\n--- K/Catalog Ratio vs Recall ---")
    for K_val in [100, 500]:
        key = f'K={K_val}'
        ratios = K_val / catalogs
        recalls_k = np.array([all_results[n]['recall_at_K'].get(key, 0) for n in names])
        valid_mask = recalls_k > 0
        if valid_mask.sum() >= 2:
            corr = np.corrcoef(np.log10(ratios[valid_mask]), recalls_k[valid_mask])[0, 1]
            print(f"  K={K_val}: correlation(log(K/catalog), recall) = {corr:.3f}")

    # 5. Summary table
    print("\n--- Full Summary Table ---")
    print(f"{'Dataset':15s} {'Catalog':>8s} {'Density':>9s} {'R@20':>6s} {'R@50':>6s} {'R@100':>6s} {'R@200':>6s} {'R@500':>6s}")
    print("-" * 75)
    for n in sorted(names, key=lambda x: all_results[x]['catalog_size']):
        r = all_results[n]
        rk = r['recall_at_K']
        print(f"{n:15s} {r['catalog_size']:>8d} {r['density']*100:>8.3f}% "
              f"{rk.get('K=20',0)*100:>5.1f}% {rk.get('K=50',0)*100:>5.1f}% "
              f"{rk.get('K=100',0)*100:>5.1f}% {rk.get('K=200',0)*100:>5.1f}% "
              f"{rk.get('K=500',0)*100:>5.1f}%")

    # 6. Key finding
    print("\n--- Key Finding ---")
    smallest_catalog = min(all_results.values(), key=lambda x: x['catalog_size'])
    largest_catalog = max(all_results.values(), key=lambda x: x['catalog_size'])
    print(f"  Smallest catalog: {smallest_catalog['dataset']} ({smallest_catalog['catalog_size']} items) → Recall@100 = {smallest_catalog['recall_at_K']['K=100']*100:.2f}%")
    print(f"  Largest catalog:  {largest_catalog['dataset']} ({largest_catalog['catalog_size']} items) → Recall@100 = {largest_catalog['recall_at_K']['K=100']*100:.2f}%")
    print(f"  Catalog ratio: {largest_catalog['catalog_size'] / smallest_catalog['catalog_size']:.1f}x")

    # Save
    output = {
        'datasets': all_results,
        'analysis': {
            'catalog_vs_recall': {
                'pearson_r': float(correlation),
                'fit_coef': float(reg.coef_[0]),
                'fit_intercept': float(reg.intercept_),
                'r_squared': float(reg.score(X_log, recall_100))
            },
            'density_vs_recall': {
                'pearson_r': float(correlation_d)
            },
            'multivariate': {
                'coef_catalog': float(reg_multi.coef_[0]),
                'coef_density': float(reg_multi.coef_[1]),
                'intercept': float(reg_multi.intercept_),
                'r_squared': float(reg_multi.score(X_multi, recall_100))
            }
        },
        'timestamp': datetime.now().isoformat()
    }

    output_path = project_root / 'experiments' / 'logs' / 'catalog_size_analysis.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
