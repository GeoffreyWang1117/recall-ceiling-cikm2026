#!/usr/bin/env python3
"""
KDD 2026: Prompt Comparison (P1/P3) on Movies Dataset (n=200)
=============================================================
Extends prompt engineering null result beyond Beauty.
Uses cloud model (DeepSeek-V3.2) under realistic conditions.

Usage:
    python scripts/kdd_prompt_movies.py --n_users 200
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import time
import re
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from typing import List, Tuple
from tqdm import tqdm
from scipy import stats
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from dotenv import load_dotenv
import os

load_dotenv()

# ============================================================================
# Prompt Templates
# ============================================================================

PROMPT_P1 = """Based on the user's purchase history, rank these candidate items by relevance.

User's recent purchases:
{history}

Candidate items to rank:
{candidates}

Output the top {top_k} most relevant item numbers in order (most relevant first).
Format: Item <number> on each line."""

PROMPT_P3 = """You are a personalized recommendation expert. Analyze the user's preferences step by step, then rank candidates.

**Step 1: Extract User Preferences**
Review the user's purchase history and identify their key preferences (categories, themes, price range, brands).

User's recent purchases:
{history}

**Step 2: Evaluate Candidates**
For each candidate, assess how well it matches the extracted preferences.

Candidate items to rank:
{candidates}

**Step 3: Final Ranking**
Based on your analysis, output the top {top_k} most relevant item numbers in order (most relevant first).
Format: Item <number> on each line."""


# ============================================================================
# CF Retrieval
# ============================================================================

class CFRetrieval:
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

    def get_candidates(self, user_id, K: int = 100):
        if user_id not in self.user_to_idx:
            return [], []
        user_idx = self.user_to_idx[user_id]
        scores = np.dot(self.user_factors[user_idx], self.item_factors.T)
        history = self.user_history.get(user_id, [])
        for item in history:
            if item in self.item_to_idx:
                scores[self.item_to_idx[item]] = -np.inf
        top_indices = np.argsort(scores)[::-1][:K]
        return [self.idx_to_item[idx] for idx in top_indices], [float(scores[idx]) for idx in top_indices]


# ============================================================================
# Reranker
# ============================================================================

def parse_item_ids(text: str, candidates: List[str], top_k: int) -> List[str]:
    item_ids = []
    patterns = [
        r'[Ii]tem\s+(\d+)', r'#(\d+)',
        r'^(\d+)[.\):\s]', r'^\s*(\d+)\s*$',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        for m in matches:
            idx = int(m) - 1
            if 0 <= idx < len(candidates):
                cand = candidates[idx]
                if cand not in item_ids:
                    item_ids.append(cand)
            if len(item_ids) >= top_k:
                break
        if len(item_ids) >= top_k:
            break

    if len(item_ids) < top_k:
        all_nums = re.findall(r'\b(\d+)\b', text)
        for m in all_nums:
            idx = int(m) - 1
            if 0 <= idx < len(candidates):
                cand = candidates[idx]
                if cand not in item_ids:
                    item_ids.append(cand)
            if len(item_ids) >= top_k:
                break

    for c in candidates:
        if c not in item_ids:
            item_ids.append(c)
        if len(item_ids) >= top_k:
            break
    return item_ids[:top_k]


class OllamaCloudReranker:
    def __init__(self, model: str):
        self.model = model
        self.api_key = os.getenv("OLLAMA_API_KEY")
        self.base_url = "https://ollama.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def rerank(self, prompt: str, candidates: List[str], top_k: int = 10):
        t0 = time.time()
        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 800,
                "temperature": 0.1,
            }
            response = requests.post(self.base_url, headers=self.headers,
                                     json=payload, timeout=180)
            elapsed = time.time() - t0
            if response.status_code == 200:
                result = response.json()
                msg = result["choices"][0]["message"]
                content = msg.get("content", "") or ""
                reasoning = msg.get("reasoning", "") or ""
                text = content if content.strip() else reasoning
                return parse_item_ids(text, candidates, top_k), elapsed
            else:
                print(f"\n  Cloud API error {response.status_code}: {response.text[:200]}")
                return candidates[:top_k], time.time() - t0
        except Exception as e:
            print(f"\n  Cloud error ({self.model}): {e}")
            return candidates[:top_k], time.time() - t0


# ============================================================================
# Metrics
# ============================================================================

def ndcg_at_k(ranked, gt, k: int = 10) -> float:
    dcg = sum(1/np.log2(i+2) for i, item in enumerate(ranked[:k]) if item in gt)
    idcg = sum(1/np.log2(i+2) for i in range(min(k, len(gt))))
    return dcg / idcg if idcg > 0 else 0.0


def bootstrap_ci(scores, n: int = 1000):
    scores = np.array(scores)
    if len(scores) == 0:
        return 0.0, 0.0, 0.0
    boots = [np.mean(np.random.choice(scores, len(scores), replace=True)) for _ in range(n)]
    return float(np.mean(scores)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ============================================================================
# Main
# ============================================================================

def run_experiment(args):
    print("=" * 70)
    print("KDD 2026: PROMPT COMPARISON (P1 vs P3) ON MOVIES")
    print(f"n_users={args.n_users}, K=100, seed={args.seed}")
    print(f"Model: {args.model}")
    print("=" * 70)

    # Load Movies dataset
    data_path = project_root / 'data' / 'processed' / 'amazon_movies_sampled'
    if not data_path.exists():
        data_path = project_root / 'data' / 'processed' / 'amazon_movies'

    train_df = pd.read_parquet(data_path / 'train.parquet')
    test_df = pd.read_parquet(data_path / 'test.parquet')
    print(f"Train: {len(train_df)} interactions, Test: {len(test_df)} interactions")

    # Item texts from metadata
    item_texts = {}
    meta_path = data_path / 'item_metadata.json'
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        for item_id_str, info in meta.items():
            text = info.get('text') or info.get('title') or f"Product {item_id_str}"
            item_texts[int(item_id_str)] = str(text)[:80]
        print(f"Loaded {len(item_texts)} item texts from metadata")

    # Build CF
    cf = CFRetrieval(n_factors=128)
    cf.fit(train_df)

    # Select test users
    np.random.seed(args.seed)
    test_users = test_df.groupby('user_id')['item_id'].apply(list).to_dict()
    valid_users = [u for u in test_users if u in cf.user_to_idx and len(test_users[u]) > 0]
    if len(valid_users) > args.n_users:
        valid_users = list(np.random.choice(valid_users, args.n_users, replace=False))
    print(f"Test users: {len(valid_users)}")

    # Prepare test data
    test_data = []
    for user_id in valid_users:
        gt = set(test_users[user_id])
        history = cf.user_history.get(user_id, [])
        candidates, scores = cf.get_candidates(user_id, K=100)
        if len(candidates) == 0:
            continue

        history_text = "\n".join(
            f"  {i+1}. {item_texts.get(item, f'Product {item}')}"
            for i, item in enumerate(history[-10:])
        )
        cand_text = "\n".join(
            f"  {i+1}. Item {i+1}: {item_texts.get(c, f'Product {c}')}"
            for i, c in enumerate(candidates[:30])
        )

        prompt_p1 = PROMPT_P1.format(history=history_text, candidates=cand_text, top_k=10)
        prompt_p3 = PROMPT_P3.format(history=history_text, candidates=cand_text, top_k=10)

        ndcg_cf = ndcg_at_k(candidates, gt, k=10)
        recall = len(set(candidates) & gt) / len(gt) if len(gt) > 0 else 0

        test_data.append({
            'user_id': user_id,
            'gt': gt,
            'candidates': candidates,
            'prompt_p1': prompt_p1,
            'prompt_p3': prompt_p3,
            'ndcg_cf': ndcg_cf,
            'recall': recall,
        })

    print(f"Valid test cases: {len(test_data)}")
    avg_recall = np.mean([d['recall'] for d in test_data])
    print(f"Average Recall@100: {avg_recall:.4f} ({avg_recall*100:.1f}%)")

    # Initialize reranker
    reranker = OllamaCloudReranker(args.model)
    print(f"Using model: {args.model}")

    # Run P1
    print("\n--- Running P1 (Basic Prompt) ---")
    p1_scores = []
    for d in tqdm(test_data, desc="P1"):
        try:
            ranked, _ = reranker.rerank(d['prompt_p1'], d['candidates'], top_k=10)
            ndcg = ndcg_at_k(ranked, d['gt'], 10)
        except:
            ndcg = d['ndcg_cf']
        p1_scores.append(ndcg)
        time.sleep(0.5)

    # Run P3
    print("\n--- Running P3 (Enhanced CoT Prompt) ---")
    p3_scores = []
    for d in tqdm(test_data, desc="P3"):
        try:
            ranked, _ = reranker.rerank(d['prompt_p3'], d['candidates'], top_k=10)
            ndcg = ndcg_at_k(ranked, d['gt'], 10)
        except:
            ndcg = d['ndcg_cf']
        p3_scores.append(ndcg)
        time.sleep(0.5)

    # CF and Random baselines
    cf_scores_list = [d['ndcg_cf'] for d in test_data]
    random_scores = []
    for d in test_data:
        shuffled = list(d['candidates'])
        np.random.shuffle(shuffled)
        random_scores.append(ndcg_at_k(shuffled, d['gt'], 10))

    # Results
    print("\n" + "=" * 70)
    print("RESULTS: PROMPT COMPARISON ON MOVIES (REALISTIC)")
    print("=" * 70)

    cf_mean = np.mean(cf_scores_list)
    results = {}
    for name, scores in [("Random", random_scores), ("CF_Score", cf_scores_list),
                          ("P1_Basic", p1_scores), ("P3_Enhanced", p3_scores)]:
        mean, ci_lo, ci_hi = bootstrap_ci(scores)
        diff_pct = (mean - cf_mean) / cf_mean * 100 if cf_mean > 0 else 0
        print(f"  {name:<16} NDCG@10={mean:.4f} [{ci_lo:.4f}, {ci_hi:.4f}] vs CF: {diff_pct:+.1f}%")
        results[name] = {
            "ndcg_mean": mean, "ndcg_ci_low": ci_lo, "ndcg_ci_high": ci_hi,
            "vs_cf_pct": diff_pct, "per_user_ndcg": [float(s) for s in scores],
        }

    # Statistical tests
    print("\nStatistical Tests:")
    cf_arr = np.array(cf_scores_list)
    for name, scores in [("P1_Basic", p1_scores), ("P3_Enhanced", p3_scores)]:
        arr = np.array(scores)
        diff = arr - cf_arr
        non_zero = diff[diff != 0]
        if len(non_zero) >= 10:
            try:
                _, w_p = stats.wilcoxon(non_zero)
            except:
                w_p = 1.0
        else:
            w_p = 1.0

        cohens_d = float(np.mean(diff) / np.std(diff)) if np.std(diff) > 0 else 0.0
        print(f"  {name} vs CF: Wilcoxon p={w_p:.4f}, Cohen's d={cohens_d:.3f}")
        results[name]["wilcoxon_p"] = float(w_p)
        results[name]["cohens_d"] = cohens_d

    # P1 vs P3 direct comparison
    p1_arr = np.array(p1_scores)
    p3_arr = np.array(p3_scores)
    diff_p1_p3 = p1_arr - p3_arr
    non_zero = diff_p1_p3[diff_p1_p3 != 0]
    if len(non_zero) >= 10:
        try:
            _, w_p_direct = stats.wilcoxon(non_zero)
        except:
            w_p_direct = 1.0
    else:
        w_p_direct = 1.0
    print(f"  P1 vs P3 direct: Wilcoxon p={w_p_direct:.4f}")

    # Save
    summary = {
        "config": {
            "dataset": "Movies",
            "model": args.model,
            "n_users": len(test_data),
            "K": 100,
            "seed": args.seed,
            "avg_recall_at_100": float(avg_recall),
            "timestamp": datetime.now().isoformat(),
        },
        "results": results,
        "p1_vs_p3_wilcoxon_p": float(w_p_direct),
    }

    output_path = project_root / 'experiments' / 'logs' / 'kdd_prompt_movies.json'
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to: {output_path}")

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_users', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', type=str, default='deepseek-v3.2')
    args = parser.parse_args()
    run_experiment(args)
