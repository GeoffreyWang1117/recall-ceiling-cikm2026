#!/usr/bin/env python3
"""
KDD 2026: Oracle Multi-Model Comparison on Beauty (n=200)
=========================================================
Shows model differences emerge under oracle conditions but disappear
under realistic conditions. Directly supports the bottleneck thesis.

Tests cloud models under ORACLE conditions (ground truth injected).

Usage:
    python scripts/kdd_oracle_multi_model.py --n_users 200
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
# Prompt Template
# ============================================================================

PROMPT_P1 = """Based on the user's purchase history, rank these candidate items by relevance.

User's recent purchases:
{history}

Candidate items to rank:
{candidates}

Output the top {top_k} most relevant item numbers in order (most relevant first).
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


class OpenAIReranker:
    def __init__(self, model: str = "gpt-4o-mini"):
        import openai
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def rerank(self, prompt: str, candidates: List[str], top_k: int = 10):
        t0 = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300
            )
            elapsed = time.time() - t0
            content = response.choices[0].message.content
            return parse_item_ids(content, candidates, top_k), elapsed
        except Exception as e:
            print(f"\n  OpenAI error: {e}")
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
    print("KDD 2026: ORACLE MULTI-MODEL COMPARISON ON BEAUTY")
    print(f"n_users={args.n_users}, K=100, seed={args.seed}")
    print("=" * 70)

    # Load Beauty dataset
    data_path = project_root / 'data' / 'processed' / 'amazon_beauty_sampled'
    if not data_path.exists():
        data_path = project_root / 'data' / 'processed' / 'amazon_beauty'

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
    elif 'title' in train_df.columns:
        for _, row in train_df.drop_duplicates('item_id').iterrows():
            item_texts[row['item_id']] = str(row.get('title', ''))[:80]

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

    # Prepare test data with ORACLE candidate sets
    test_data = []
    for user_id in valid_users:
        gt = set(test_users[user_id])
        history = cf.user_history.get(user_id, [])
        candidates, scores = cf.get_candidates(user_id, K=100)
        if len(candidates) == 0:
            continue

        # ORACLE: Inject ground truth items into candidate set
        oracle_candidates = list(candidates[:90])
        for gt_item in gt:
            if gt_item not in oracle_candidates:
                oracle_candidates.append(gt_item)
        oracle_candidates = oracle_candidates[:100]
        np.random.shuffle(oracle_candidates)

        # Build prompt with oracle candidates
        history_text = "\n".join(
            f"  {i+1}. {item_texts.get(item, f'Product {item}')}"
            for i, item in enumerate(history[-10:])
        )
        cand_text = "\n".join(
            f"  {i+1}. Item {i+1}: {item_texts.get(c, f'Product {c}')}"
            for i, c in enumerate(oracle_candidates[:30])
        )
        prompt = PROMPT_P1.format(history=history_text, candidates=cand_text, top_k=10)

        # Random baseline NDCG under oracle
        shuffled = list(oracle_candidates)
        np.random.shuffle(shuffled)
        ndcg_random = ndcg_at_k(shuffled, gt, k=10)

        test_data.append({
            'user_id': user_id,
            'gt': gt,
            'oracle_candidates': oracle_candidates,
            'prompt': prompt,
            'ndcg_random': ndcg_random,
        })

    print(f"Valid test cases: {len(test_data)}")

    # ===== Initialize cloud models =====
    models = {}

    cloud_models = [
        ("DeepSeek-V3.2", "deepseek-v3.2", "671B"),
        ("Qwen3-Next-80B", "qwen3-next:80b", "80B"),
        ("Mistral-Large-675B", "mistral-large-3:675b", "675B"),
    ]
    if os.getenv("OLLAMA_API_KEY"):
        for name, model_id, params in cloud_models:
            models[name] = {"reranker": OllamaCloudReranker(model_id), "params": params, "type": "cloud"}
            print(f"  [OK] {name} ({params}, cloud)")

    if os.getenv("OPENAI_API_KEY"):
        try:
            models["GPT-4o-mini"] = {"reranker": OpenAIReranker("gpt-4o-mini"), "params": "~8B*", "type": "cloud"}
            print(f"  [OK] GPT-4o-mini (cloud)")
        except Exception as e:
            print(f"  [SKIP] GPT-4o-mini: {e}")

    print(f"\nTotal models: {len(models)}")

    # ===== Run evaluation =====
    all_scores = {"Random": {"ndcg": []}}
    for name in models:
        all_scores[name] = {"ndcg": [], "times": []}

    for d in tqdm(test_data, desc="Oracle evaluation"):
        candidates = d['oracle_candidates']
        gt = d['gt']
        prompt = d['prompt']

        # Random baseline
        all_scores["Random"]["ndcg"].append(d['ndcg_random'])

        # LLM models
        for name, info in models.items():
            try:
                ranked, elapsed = info["reranker"].rerank(prompt, candidates, top_k=10)
                ndcg = ndcg_at_k(ranked, gt, 10)
                all_scores[name]["ndcg"].append(ndcg)
                all_scores[name]["times"].append(elapsed)
            except Exception as e:
                all_scores[name]["ndcg"].append(0.0)
                all_scores[name]["times"].append(0.0)

            if info.get("type") == "cloud":
                time.sleep(0.5)

    # ===== Results =====
    print("\n" + "=" * 70)
    print("RESULTS: ORACLE MULTI-MODEL COMPARISON ON BEAUTY")
    print("=" * 70)

    results_summary = {
        "config": {
            "dataset": "Beauty",
            "condition": "oracle",
            "n_users": len(test_data),
            "K": 100,
            "seed": args.seed,
            "timestamp": datetime.now().isoformat(),
        },
        "results": {},
    }

    print(f"\n{'Method':<22} {'Params':<8} {'NDCG@10':<10} {'95% CI':<24}")
    print("-" * 70)

    method_order = ["Random"] + list(models.keys())
    for method in method_order:
        scores = all_scores[method]["ndcg"]
        mean, ci_lo, ci_hi = bootstrap_ci(scores)
        params = models[method]["params"] if method in models else "---"

        print(f"{method:<22} {params:<8} {mean:.4f}     [{ci_lo:.4f}, {ci_hi:.4f}]")

        results_summary["results"][method] = {
            "params": params,
            "ndcg_mean": mean,
            "ndcg_ci_low": ci_lo,
            "ndcg_ci_high": ci_hi,
            "n_samples": len(scores),
            "per_user_ndcg": [float(s) for s in scores],
        }

    # Pairwise tests between models (oracle should show differences)
    print("\n" + "=" * 70)
    print("PAIRWISE COMPARISONS (Oracle - expect model differences)")
    print("=" * 70)

    llm_names = list(models.keys())
    pairwise = {}
    for i in range(len(llm_names)):
        for j in range(i+1, len(llm_names)):
            a, b = llm_names[i], llm_names[j]
            scores_a = np.array(all_scores[a]["ndcg"])
            scores_b = np.array(all_scores[b]["ndcg"])
            diff = scores_a - scores_b
            non_zero = diff[diff != 0]

            if len(non_zero) >= 10:
                try:
                    _, w_p = stats.wilcoxon(non_zero)
                except:
                    w_p = 1.0
            else:
                w_p = 1.0

            mean_a, mean_b = np.mean(scores_a), np.mean(scores_b)
            sig = "*" if w_p < 0.05 else ""
            print(f"  {a} vs {b}: {mean_a:.4f} vs {mean_b:.4f}, Wilcoxon p={w_p:.4f} {sig}")
            pairwise[f"{a}_vs_{b}"] = {"mean_a": float(mean_a), "mean_b": float(mean_b), "p": float(w_p)}

    results_summary["pairwise"] = pairwise

    # Save
    output_path = project_root / 'experiments' / 'logs' / 'kdd_oracle_multi_model.json'
    with open(output_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSaved to: {output_path}")

    return results_summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_users', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    run_experiment(args)
