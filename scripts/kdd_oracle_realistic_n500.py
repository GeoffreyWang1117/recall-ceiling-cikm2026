#!/usr/bin/env python3
"""
KDD 2026: Oracle vs Realistic Gap at n=500 across 3 datasets.
Replaces Table 2 (previously n=50) with proper sample size.

Usage: python scripts/kdd_oracle_realistic_n500.py
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
N_USERS = 500
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
    'amazon_beauty': 'data/processed/amazon_beauty_sampled',
    'amazon_movies': 'data/processed/amazon_movies_sampled',
    'amazon_electronics': 'data/processed/amazon_electronics_sampled',
}


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
        n_comp = min(self.n_factors, len(items) - 1)
        self.svd = TruncatedSVD(n_components=n_comp)
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
    indices = []
    seen = set()
    for num in numbers:
        idx = int(num) - 1
        if 0 <= idx < n_candidates and idx not in seen:
            indices.append(idx)
            seen.add(idx)
    for i in range(n_candidates):
        if i not in seen:
            indices.append(i)
    return indices[:n_candidates]


def rerank_with_llm(history_texts, candidates_with_text, top_k=10):
    history = "\n".join(f"{i+1}. {t}" for i, t in enumerate(history_texts[-10:]))
    cands = "\n".join(f"{i+1}. Item {i+1}: {t}" for i, (_, t) in enumerate(candidates_with_text[:30]))
    prompt = PROMPT_TEMPLATE.format(history=history, candidates=cands, top_k=top_k)
    response = call_llm(prompt)
    ranking = parse_ranking(response, len(candidates_with_text))
    return [candidates_with_text[i][0] for i in ranking]


def run_dataset(dataset_name, data_path):
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")

    train_df = pd.read_parquet(Path(project_root / data_path / 'train.parquet'))
    test_df = pd.read_parquet(Path(project_root / data_path / 'test.parquet'))
    print(f"Train: {len(train_df)} interactions, {train_df['user_id'].nunique()} users, {train_df['item_id'].nunique()} items")

    # Build item texts from metadata
    item_texts = {}
    meta_path = Path(project_root / data_path / 'item_metadata.json')
    if meta_path.exists():
        import json as _json
        with open(meta_path) as f:
            meta = _json.load(f)
        for item_id_str, info in meta.items():
            text = info.get('text') or info.get('title') or f"Product {item_id_str}"
            item_texts[int(item_id_str)] = str(text)[:80]
        print(f"Loaded {len(item_texts)} item texts from metadata")
    elif 'title' in train_df.columns:
        for _, row in train_df.drop_duplicates('item_id').iterrows():
            item_texts[row['item_id']] = str(row.get('title', f"Product {row['item_id']}"))[:80]

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

    # Storage
    oracle_ndcg_scores = []
    realistic_ndcg_scores = []
    cf_baseline_ndcg_scores = []
    recall_scores = []

    for user_id in tqdm(valid_users, desc=dataset_name):
        ground_truth = test_users[user_id]
        history = cf.user_history.get(user_id, [])
        history_texts = [item_texts.get(i, f"Product {i}") for i in history]

        # Get CF candidates
        cf_candidates, cf_scores_list = cf.get_candidates(user_id, K=K)
        cf_recall = recall_at_k(cf_candidates, ground_truth, k=K)
        recall_scores.append(cf_recall)

        # CF baseline: just use CF score order
        cf_baseline_ndcg = ndcg_at_k(cf_candidates, ground_truth, k=TOP_K)
        cf_baseline_ndcg_scores.append(cf_baseline_ndcg)

        # === REALISTIC SETTING ===
        realistic_cands = [(item, item_texts.get(item, f"Product {item}")) for item in cf_candidates[:K]]
        try:
            realistic_ranked = rerank_with_llm(history_texts, realistic_cands, top_k=TOP_K)
            realistic_ndcg = ndcg_at_k(realistic_ranked, ground_truth, k=TOP_K)
        except Exception:
            realistic_ndcg = 0.0
        realistic_ndcg_scores.append(realistic_ndcg)

        # === ORACLE SETTING ===
        oracle_candidates = list(cf_candidates[:90])
        for gt_item in ground_truth:
            if gt_item not in oracle_candidates:
                oracle_candidates.append(gt_item)
        oracle_candidates = oracle_candidates[:K]
        np.random.shuffle(oracle_candidates)

        oracle_cands = [(item, item_texts.get(item, f"Product {item}")) for item in oracle_candidates]
        try:
            oracle_ranked = rerank_with_llm(history_texts, oracle_cands, top_k=TOP_K)
            oracle_ndcg = ndcg_at_k(oracle_ranked, ground_truth, k=TOP_K)
        except Exception:
            oracle_ndcg = 0.0
        oracle_ndcg_scores.append(oracle_ndcg)

    # Compute statistics
    oracle_mean, oracle_ci_low, oracle_ci_high = bootstrap_ci(oracle_ndcg_scores)
    realistic_mean, real_ci_low, real_ci_high = bootstrap_ci(realistic_ndcg_scores)
    cf_mean, cf_ci_low, cf_ci_high = bootstrap_ci(cf_baseline_ndcg_scores)
    recall_mean, recall_ci_low, recall_ci_high = bootstrap_ci(recall_scores)

    gap = (oracle_mean - realistic_mean) / oracle_mean * 100 if oracle_mean > 0 else 0

    # Wilcoxon test: LLM realistic vs CF baseline
    try:
        diffs = np.array(realistic_ndcg_scores) - np.array(cf_baseline_ndcg_scores)
        non_zero = diffs[diffs != 0]
        if len(non_zero) > 10:
            stat, p_val = wilcoxon(non_zero)
        else:
            p_val = 1.0
    except:
        p_val = 1.0

    print(f"\n  Oracle NDCG@10:    {oracle_mean:.4f} [{oracle_ci_low:.4f}, {oracle_ci_high:.4f}]")
    print(f"  Realistic NDCG@10: {realistic_mean:.4f} [{real_ci_low:.4f}, {real_ci_high:.4f}]")
    print(f"  CF Baseline:       {cf_mean:.4f} [{cf_ci_low:.4f}, {cf_ci_high:.4f}]")
    print(f"  Recall@{K}:         {recall_mean*100:.2f}%")
    print(f"  Oracle-Realistic Gap: {gap:.1f}%")
    print(f"  LLM vs CF p-value: {p_val:.4f}")

    return {
        'n_users': len(valid_users),
        'n_items': train_df['item_id'].nunique(),
        'oracle': {'mean': oracle_mean, 'ci_low': oracle_ci_low, 'ci_high': oracle_ci_high},
        'realistic': {'mean': realistic_mean, 'ci_low': real_ci_low, 'ci_high': real_ci_high},
        'cf_baseline': {'mean': cf_mean, 'ci_low': cf_ci_low, 'ci_high': cf_ci_high},
        'recall_at_k': {'mean': recall_mean, 'ci_low': recall_ci_low, 'ci_high': recall_ci_high},
        'gap_percent': gap,
        'wilcoxon_p': p_val,
        'oracle_ndcg_scores': oracle_ndcg_scores,
        'realistic_ndcg_scores': realistic_ndcg_scores,
        'cf_baseline_ndcg_scores': cf_baseline_ndcg_scores,
    }


if __name__ == '__main__':
    print("=" * 70)
    print("KDD 2026: Oracle vs Realistic Gap (n=500, 3 datasets)")
    print(f"Model: {MODEL}, K={K}, seed={SEED}")
    print("=" * 70)

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'config': {'model': MODEL, 'n_users': N_USERS, 'K': K, 'seed': SEED, 'top_k': TOP_K},
        'datasets': {}
    }

    for ds_name, ds_path in DATASETS.items():
        result = run_dataset(ds_name, ds_path)
        # Remove raw scores for JSON (too large)
        result_clean = {k: v for k, v in result.items()
                        if k not in ('oracle_ndcg_scores', 'realistic_ndcg_scores', 'cf_baseline_ndcg_scores')}
        all_results['datasets'][ds_name] = result_clean

    # Save
    output_path = project_root / 'experiments' / 'logs' / 'kdd_oracle_realistic_n500.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for ds_name, result in all_results['datasets'].items():
        print(f"  {ds_name:25s}: Oracle={result['oracle']['mean']:.4f}  Realistic={result['realistic']['mean']:.4f}  Gap={result['gap_percent']:.1f}%  Recall={result['recall_at_k']['mean']*100:.1f}%")
    print(f"\nSaved to: {output_path}")
