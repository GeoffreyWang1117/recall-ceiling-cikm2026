#!/usr/bin/env python3
"""
KDD 2026: Industrial-Scale LLM Reranking Validation
====================================================
Adds LLM reranking results to industrial-scale datasets (Table 14).
Tests whether the bottleneck claim holds at scale.

Usage: python scripts/kdd_industrial_llm_reranking.py
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import numpy as np
import pandas as pd
import requests
import re
import time
from datetime import datetime
from tqdm import tqdm
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from scipy.stats import wilcoxon

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma3:4b"
N_USERS = 200
K = 100
SEED = 42
TOP_K = 10

PROMPT_TEMPLATE = """Based on the user's purchase history, rank these candidate items by relevance.

User's recent purchases:
{history}

Candidate items to rank:
{candidates}

Output the top {top_k} most relevant item numbers in order (most relevant first).
Format: Item <number> on each line."""

DATASETS = {
    'sports': 'Sports_and_Outdoors',
    'home_kitchen': 'Home_and_Kitchen',
    'toys': 'Toys_and_Games',
}


def load_and_process_dataset(dataset_key, min_user_interactions=10, min_item_interactions=5, sample_users=2000):
    raw_name = DATASETS[dataset_key]
    raw_path = project_root / 'data' / 'raw' / 'amazon'
    reviews_file = raw_path / f'{raw_name}_reviews.parquet'
    meta_file = raw_path / f'{raw_name}_meta.parquet'

    print(f"Loading {raw_name}...")
    df = pd.read_parquet(reviews_file)
    print(f"  Raw: {len(df):,} reviews")
    df = df.rename(columns={'parent_asin': 'item_id'})

    # Load item titles from meta if available
    item_titles = {}
    if meta_file.exists():
        try:
            meta_df = pd.read_parquet(meta_file, columns=['parent_asin', 'title'])
            item_titles = dict(zip(meta_df['parent_asin'], meta_df['title'].fillna('')))
            print(f"  Loaded {len(item_titles):,} item titles from metadata")
        except Exception as e:
            print(f"  Could not load meta: {e}")

    # Filter
    print(f"  Filtering (min_user={min_user_interactions}, min_item={min_item_interactions})...")
    for _ in range(3):
        user_counts = df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_user_interactions].index
        df = df[df['user_id'].isin(valid_users)]
        item_counts = df['item_id'].value_counts()
        valid_items = item_counts[item_counts >= min_item_interactions].index
        df = df[df['item_id'].isin(valid_items)]

    print(f"  After filtering: {len(df):,} reviews, {df['user_id'].nunique():,} users, {df['item_id'].nunique():,} items")

    unique_users = df['user_id'].unique()
    if len(unique_users) > sample_users:
        np.random.seed(SEED)
        sampled_users = np.random.choice(unique_users, sample_users, replace=False)
        df = df[df['user_id'].isin(sampled_users)]

    if 'timestamp' in df.columns:
        df = df.sort_values(['user_id', 'timestamp'])

    test_indices = df.groupby('user_id').tail(1).index
    train_df = df[~df.index.isin(test_indices)].copy()
    test_df = df[df.index.isin(test_indices)].copy()
    print(f"  Train: {len(train_df):,} | Test: {len(test_df):,}")
    return train_df, test_df, item_titles


class CFRetrieval:
    def __init__(self, n_factors=128):
        self.n_factors = n_factors

    def fit(self, train_df):
        items = train_df['item_id'].unique()
        users = train_df['user_id'].unique()
        self.item_to_idx = {i: idx for idx, i in enumerate(items)}
        self.idx_to_item = {idx: i for i, idx in self.item_to_idx.items()}
        self.user_to_idx = {u: idx for idx, u in enumerate(users)}
        rows = [self.user_to_idx[u] for u in train_df['user_id']]
        cols = [self.item_to_idx[i] for i in train_df['item_id']]
        data = np.ones(len(rows))
        self.user_item = csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))
        n_comp = min(self.n_factors, min(self.user_item.shape) - 1)
        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self.user_factors = self.svd.fit_transform(self.user_item)
        self.item_factors = self.svd.components_.T
        self.user_history = train_df.groupby('user_id')['item_id'].apply(list).to_dict()

    def get_candidates(self, user_id, K=100):
        if user_id not in self.user_to_idx:
            return [], []
        user_idx = self.user_to_idx[user_id]
        scores = np.dot(self.user_factors[user_idx], self.item_factors.T)
        history = self.user_history.get(user_id, [])
        for item in history:
            if item in self.item_to_idx:
                scores[self.item_to_idx[item]] = -np.inf
        top_indices = np.argsort(scores)[::-1][:K]
        candidates = [self.idx_to_item[idx] for idx in top_indices]
        cand_scores = [float(scores[idx]) for idx in top_indices]
        return candidates, cand_scores


def ndcg_at_k(ranked_list, ground_truth, k=10):
    dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(ranked_list[:k]) if item in ground_truth)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(ground_truth))))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(candidates, ground_truth, k=100):
    hits = len(set(candidates[:k]) & set(ground_truth))
    return hits / len(ground_truth) if ground_truth else 0.0


def bootstrap_ci(scores, n_bootstrap=1000):
    scores = np.array(scores)
    boot_means = [np.mean(np.random.choice(scores, len(scores), replace=True)) for _ in range(n_bootstrap)]
    return float(np.mean(scores)), float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def call_llm(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = requests.post(OLLAMA_URL, json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 300}
            }, timeout=60)
            return resp.json()["message"]["content"]
        except Exception as e:
            if attempt == max_retries - 1:
                return ""
            time.sleep(2)
    return ""


def parse_ranking(response, n_candidates):
    numbers = re.findall(r'\d+', response)
    indices, seen = [], set()
    for num in numbers:
        idx = int(num) - 1
        if 0 <= idx < n_candidates and idx not in seen:
            indices.append(idx)
            seen.add(idx)
    for i in range(n_candidates):
        if i not in seen:
            indices.append(i)
    return indices[:n_candidates]


def run_dataset(dataset_key):
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_key} ({DATASETS[dataset_key]})")
    print(f"{'='*60}")

    train_df, test_df, item_titles = load_and_process_dataset(dataset_key)

    # Build item texts
    def get_title(item_id):
        t = item_titles.get(item_id, '')
        if t and len(t) > 3:
            return str(t)[:80]
        return f"Product {item_id}"

    # CF retrieval
    cf = CFRetrieval(n_factors=128)
    cf.fit(train_df)

    # Select test users
    np.random.seed(SEED)
    test_users = test_df.groupby('user_id')['item_id'].apply(list).to_dict()
    valid_users = [u for u in test_users if u in cf.user_to_idx and len(test_users[u]) > 0]
    if len(valid_users) > N_USERS:
        valid_users = np.random.choice(valid_users, N_USERS, replace=False).tolist()
    print(f"Selected {len(valid_users)} users")
    print(f"Catalog: {train_df['item_id'].nunique():,} items")

    # Run experiment
    cf_ndcg_scores, llm_ndcg_scores, recall_scores = [], [], []

    for user_id in tqdm(valid_users, desc=f"{dataset_key} LLM reranking"):
        ground_truth = test_users[user_id]
        history = cf.user_history.get(user_id, [])
        history_texts = [get_title(i) for i in history]

        candidates, cand_scores = cf.get_candidates(user_id, K=K)
        cf_recall = recall_at_k(candidates, ground_truth, k=K)
        recall_scores.append(cf_recall)

        # CF baseline: original order
        cf_ndcg = ndcg_at_k(candidates, ground_truth, k=TOP_K)
        cf_ndcg_scores.append(cf_ndcg)

        # LLM reranking
        cands_with_text = [(item, get_title(item)) for item in candidates[:K]]
        hist_text = "\n".join(f"{i+1}. {t}" for i, t in enumerate(history_texts[-10:]))
        cands_text = "\n".join(f"{i+1}. Item {i+1}: {t}" for i, (_, t) in enumerate(cands_with_text[:30]))
        prompt = PROMPT_TEMPLATE.format(history=hist_text, candidates=cands_text, top_k=TOP_K)

        try:
            response = call_llm(prompt)
            ranking = parse_ranking(response, len(cands_with_text))
            ranked_items = [cands_with_text[i][0] for i in ranking]
            llm_ndcg = ndcg_at_k(ranked_items, ground_truth, k=TOP_K)
        except Exception:
            llm_ndcg = 0.0
        llm_ndcg_scores.append(llm_ndcg)

    # Compute statistics
    cf_mean, cf_ci_low, cf_ci_high = bootstrap_ci(cf_ndcg_scores)
    llm_mean, llm_ci_low, llm_ci_high = bootstrap_ci(llm_ndcg_scores)
    recall_mean, recall_ci_low, recall_ci_high = bootstrap_ci(recall_scores)

    improvement = (llm_mean - cf_mean) / cf_mean * 100 if cf_mean > 0 else 0

    # Wilcoxon test
    try:
        diffs = np.array(llm_ndcg_scores) - np.array(cf_ndcg_scores)
        non_zero = diffs[diffs != 0]
        if len(non_zero) > 10:
            stat, p_val = wilcoxon(non_zero)
        else:
            p_val = 1.0
    except:
        p_val = 1.0

    # Cohen's d
    diffs_all = np.array(llm_ndcg_scores) - np.array(cf_ndcg_scores)
    cohens_d = float(np.mean(diffs_all) / (np.std(diffs_all) + 1e-10))

    print(f"\n  Recall@{K}: {recall_mean*100:.2f}%")
    print(f"  CF NDCG@10: {cf_mean:.4f} [{cf_ci_low:.4f}, {cf_ci_high:.4f}]")
    print(f"  LLM NDCG@10: {llm_mean:.4f} [{llm_ci_low:.4f}, {llm_ci_high:.4f}]")
    print(f"  Improvement: {improvement:+.1f}%")
    print(f"  Wilcoxon p: {p_val:.4f}")
    print(f"  Cohen's d: {cohens_d:.4f}")

    return {
        'n_users': len(valid_users),
        'n_items': int(train_df['item_id'].nunique()),
        'n_interactions': int(len(train_df)),
        'recall_at_100': {'mean': recall_mean, 'ci_low': recall_ci_low, 'ci_high': recall_ci_high},
        'cf_ndcg': {'mean': cf_mean, 'ci_low': cf_ci_low, 'ci_high': cf_ci_high},
        'llm_ndcg': {'mean': llm_mean, 'ci_low': llm_ci_low, 'ci_high': llm_ci_high},
        'improvement_percent': improvement,
        'wilcoxon_p': p_val,
        'cohens_d': cohens_d,
    }


if __name__ == '__main__':
    print("=" * 70)
    print("KDD 2026: Industrial-Scale LLM Reranking Validation")
    print(f"Model: {MODEL}, n_users={N_USERS}, K={K}")
    print("=" * 70)

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'config': {'model': MODEL, 'n_users': N_USERS, 'K': K, 'seed': SEED},
        'datasets': {}
    }

    for ds_key in ['sports', 'home_kitchen', 'toys']:
        result = run_dataset(ds_key)
        all_results['datasets'][ds_key] = result

    output_path = project_root / 'experiments' / 'logs' / 'kdd_industrial_llm_reranking.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for ds_key, result in all_results['datasets'].items():
        print(f"  {ds_key:20s}: CF={result['cf_ndcg']['mean']:.4f}  LLM={result['llm_ndcg']['mean']:.4f}  "
              f"Δ={result['improvement_percent']:+.1f}%  p={result['wilcoxon_p']:.3f}  Recall={result['recall_at_100']['mean']*100:.1f}%")
    print(f"\nSaved to: {output_path}")
