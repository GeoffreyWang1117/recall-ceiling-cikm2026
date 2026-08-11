#!/usr/bin/env python3
"""
KDD 2026: Unified Multi-Model Comparison (n=200)
=================================================
Tests ALL models under identical conditions for Table 9 and Table 12.

Models tested:
  Local (Ollama localhost):
    - llama3.1:8b       (Meta, 8B)
    - gemma3:4b         (Google, 4B)
  Cloud (Ollama Cloud):
    - deepseek-v3.2     (DeepSeek, 671B)
    - qwen3-next:80b    (Alibaba, 80B)
    - mistral-large-3:675b (Mistral, 675B)
  OpenAI:
    - gpt-4o-mini       (OpenAI)
  Baselines:
    - Random
    - CF Score (retrieval score ranking)

Usage:
    python scripts/kdd_multi_model_unified.py --n_users 200
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
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from scipy import stats
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from dotenv import load_dotenv
import os

load_dotenv()

# ============================================================================
# Prompt Template (P1 Basic - same as all other experiments)
# ============================================================================

PROMPT_P1 = """Based on the user's purchase history, rank these candidate items by relevance.

User's recent purchases:
{history}

Candidate items to rank:
{candidates}

Output the top {top_k} most relevant item numbers in order (most relevant first).
Format: Item <number> on each line."""


# ============================================================================
# CF Retrieval (consistent with other experiments)
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

    def get_candidates(self, user_id: str, K: int = 100) -> Tuple[List[str], List[float]]:
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
# Rerankers
# ============================================================================

def parse_item_ids(text: str, candidates: List[str], top_k: int) -> List[str]:
    """Parse item IDs from LLM output. Shared by all rerankers."""
    item_ids = []
    candidates_set = set(candidates)
    # Extract numbers that could be item indices (1-based)
    patterns = [
        r'[Ii]tem\s+(\d+)',
        r'#(\d+)',
        r'^(\d+)[.\):\s]',          # "1. ..." or "1) ..."
        r'^\s*(\d+)\s*$',           # standalone number
    ]
    # Try structured patterns first
    for pattern in patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        for m in matches:
            idx = int(m) - 1  # 1-based to 0-based
            if 0 <= idx < len(candidates):
                cand = candidates[idx]
                if cand not in item_ids:
                    item_ids.append(cand)
            if len(item_ids) >= top_k:
                break
        if len(item_ids) >= top_k:
            break

    # Fallback: any number that matches a candidate index
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

    # Fill remaining with original CF order
    for c in candidates:
        if c not in item_ids:
            item_ids.append(c)
        if len(item_ids) >= top_k:
            break

    return item_ids[:top_k]


class OllamaLocalReranker:
    """Local Ollama models via localhost:11434"""

    def __init__(self, model: str):
        self.model = model
        self.base_url = "http://localhost:11434/api/chat"

    def rerank(self, prompt: str, candidates: List[str], top_k: int = 10) -> Tuple[List[str], float]:
        t0 = time.time()
        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 300},
                "think": False,  # Disable thinking for qwen3
            }
            response = requests.post(self.base_url, json=payload, timeout=120)
            elapsed = time.time() - t0
            if response.status_code == 200:
                result = response.json()
                content = result.get("message", {}).get("content", "")
                return parse_item_ids(content, candidates, top_k), elapsed
            else:
                return candidates[:top_k], elapsed
        except Exception as e:
            elapsed = time.time() - t0
            return candidates[:top_k], elapsed


class OllamaCloudReranker:
    """Ollama Cloud models via ollama.com/v1"""

    def __init__(self, model: str):
        self.model = model
        self.api_key = os.getenv("OLLAMA_API_KEY")
        self.base_url = "https://ollama.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def rerank(self, prompt: str, candidates: List[str], top_k: int = 10) -> Tuple[List[str], float]:
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
                # Handle both standard and reasoning models
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
    """OpenAI API reranker"""

    def __init__(self, model: str = "gpt-4o-mini"):
        import openai
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def rerank(self, prompt: str, candidates: List[str], top_k: int = 10) -> Tuple[List[str], float]:
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

def ndcg_at_k(ranked: List[str], gt: set, k: int = 10) -> float:
    dcg = sum(1/np.log2(i+2) for i, item in enumerate(ranked[:k]) if item in gt)
    idcg = sum(1/np.log2(i+2) for i in range(min(k, len(gt))))
    return dcg / idcg if idcg > 0 else 0.0


def bootstrap_ci(scores, n: int = 1000) -> Tuple[float, float, float]:
    scores = np.array(scores)
    if len(scores) == 0:
        return 0.0, 0.0, 0.0
    boots = [np.mean(np.random.choice(scores, len(scores), replace=True)) for _ in range(n)]
    return float(np.mean(scores)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment(args):
    print("=" * 70)
    print("KDD 2026: UNIFIED MULTI-MODEL COMPARISON")
    print(f"n_users={args.n_users}, K=100, seed={args.seed}")
    print("=" * 70)

    # Load Beauty dataset
    data_path = project_root / 'data' / 'processed' / 'amazon_beauty_sampled'
    if not data_path.exists():
        data_path = project_root / 'data' / 'processed' / 'amazon_beauty'

    train_df = pd.read_parquet(data_path / 'train.parquet')
    test_df = pd.read_parquet(data_path / 'test.parquet')
    print(f"Train: {len(train_df)} interactions, Test: {len(test_df)} interactions")

    # Item texts
    item_texts = {}
    if 'title' in train_df.columns:
        for _, row in train_df.drop_duplicates('item_id').iterrows():
            item_texts[row['item_id']] = str(row.get('title', ''))[:80]

    # Build CF retrieval
    cf = CFRetrieval(n_factors=128)
    cf.fit(train_df)

    # Select test users
    np.random.seed(args.seed)
    test_users = test_df.groupby('user_id')['item_id'].apply(list).to_dict()
    valid_users = [u for u in test_users if u in cf.user_to_idx and len(test_users[u]) > 0]
    if len(valid_users) > args.n_users:
        valid_users = list(np.random.choice(valid_users, args.n_users, replace=False))
    print(f"Test users: {len(valid_users)}")

    # Prepare test data (same candidates for all models)
    test_data = []
    for user_id in valid_users:
        gt = set(test_users[user_id])
        history = cf.user_history.get(user_id, [])
        candidates, scores = cf.get_candidates(user_id, K=100)
        if len(candidates) == 0:
            continue

        # Build prompt
        history_text = "\n".join(
            f"  {i+1}. {item_texts.get(item, f'Product {item}')}"
            for i, item in enumerate(history[-10:])
        )
        cand_text = "\n".join(
            f"  {i+1}. Item {i+1}: {item_texts.get(c, f'Product {c}')}"
            for i, c in enumerate(candidates[:30])
        )
        prompt = PROMPT_P1.format(history=history_text, candidates=cand_text, top_k=10)

        # CF baseline ranking
        ndcg_cf = ndcg_at_k(candidates, gt, k=10)

        # Recall quality
        recall = len(set(candidates) & gt) / len(gt) if len(gt) > 0 else 0

        test_data.append({
            'user_id': user_id,
            'gt': gt,
            'candidates': candidates,
            'scores': scores,
            'prompt': prompt,
            'ndcg_cf': ndcg_cf,
            'recall': recall,
        })

    print(f"Valid test cases: {len(test_data)}")
    avg_recall = np.mean([d['recall'] for d in test_data])
    print(f"Average Recall@100: {avg_recall:.4f} ({avg_recall*100:.1f}%)")

    # ===== Initialize all models =====
    models = {}

    # Local Ollama models
    local_models = [
        ("Llama3.1-8B", "llama3.1:8b", "8B"),
        ("Gemma3-4B", "gemma3:4b", "4B"),
    ]
    for name, model_id, params in local_models:
        try:
            r = requests.post("http://localhost:11434/api/chat",
                              json={"model": model_id, "messages": [{"role":"user","content":"test"}],
                                    "stream": False, "options": {"num_predict": 1}, "think": False},
                              timeout=30)
            if r.status_code == 200:
                models[name] = {"reranker": OllamaLocalReranker(model_id), "params": params, "type": "local"}
                print(f"  [OK] {name} ({params})")
            else:
                print(f"  [SKIP] {name}: status {r.status_code}")
        except Exception as e:
            print(f"  [SKIP] {name}: {e}")

    # Ollama Cloud models
    cloud_models = [
        ("DeepSeek-V3.2", "deepseek-v3.2", "671B"),
        ("Qwen3-Next-80B", "qwen3-next:80b", "80B"),
        ("Mistral-Large-675B", "mistral-large-3:675b", "675B"),
    ]
    if os.getenv("OLLAMA_API_KEY"):
        for name, model_id, params in cloud_models:
            models[name] = {"reranker": OllamaCloudReranker(model_id), "params": params, "type": "cloud"}
            print(f"  [OK] {name} ({params}, cloud)")

    # OpenAI
    if os.getenv("OPENAI_API_KEY"):
        try:
            models["GPT-4o-mini"] = {"reranker": OpenAIReranker("gpt-4o-mini"), "params": "~8B*", "type": "cloud"}
            print(f"  [OK] GPT-4o-mini (cloud)")
        except Exception as e:
            print(f"  [SKIP] GPT-4o-mini: {e}")

    print(f"\nTotal models to test: {len(models)}")

    # ===== Run evaluation =====
    all_scores = {
        "Random": {"ndcg": [], "times": []},
        "CF_Score": {"ndcg": [], "times": []},
    }
    for name in models:
        all_scores[name] = {"ndcg": [], "times": []}

    for d in tqdm(test_data, desc="Evaluating users"):
        candidates = d['candidates']
        gt = d['gt']
        prompt = d['prompt']

        # Random baseline
        shuffled = list(candidates)
        np.random.shuffle(shuffled)
        all_scores["Random"]["ndcg"].append(ndcg_at_k(shuffled, gt, 10))
        all_scores["Random"]["times"].append(0.0)

        # CF Score baseline
        all_scores["CF_Score"]["ndcg"].append(d['ndcg_cf'])
        all_scores["CF_Score"]["times"].append(0.01)

        # LLM models
        for name, info in models.items():
            try:
                ranked, elapsed = info["reranker"].rerank(prompt, candidates, top_k=10)
                ndcg = ndcg_at_k(ranked, gt, 10)
                all_scores[name]["ndcg"].append(ndcg)
                all_scores[name]["times"].append(elapsed)
            except Exception as e:
                all_scores[name]["ndcg"].append(d['ndcg_cf'])
                all_scores[name]["times"].append(0.0)

            # Rate limiting for cloud APIs
            if info.get("type") == "cloud":
                time.sleep(0.5)

    # ===== Compute results =====
    print("\n" + "=" * 70)
    print("RESULTS: UNIFIED MULTI-MODEL COMPARISON")
    print("=" * 70)

    cf_scores = np.array(all_scores["CF_Score"]["ndcg"])
    cf_mean = np.mean(cf_scores)

    results_summary = {
        "config": {
            "dataset": "Beauty",
            "n_users": len(test_data),
            "K": 100,
            "top_k": 10,
            "seed": args.seed,
            "avg_recall_at_100": float(avg_recall),
            "timestamp": datetime.now().isoformat(),
        },
        "results": {},
    }

    print(f"\n{'Method':<22} {'Params':<8} {'NDCG@10':<10} {'95% CI':<24} {'vs CF':>8} {'Latency':>8}")
    print("-" * 85)

    method_order = ["Random", "CF_Score"] + list(models.keys())
    for method in method_order:
        scores = all_scores[method]["ndcg"]
        times = all_scores[method]["times"]
        mean, ci_lo, ci_hi = bootstrap_ci(scores)
        avg_time = np.mean(times)
        diff_pct = (mean - cf_mean) / cf_mean * 100 if cf_mean > 0 else 0

        params = "---"
        if method in models:
            params = models[method]["params"]

        print(f"{method:<22} {params:<8} {mean:.4f}     [{ci_lo:.4f}, {ci_hi:.4f}]   {diff_pct:>+7.1f}% {avg_time:>7.1f}s")

        results_summary["results"][method] = {
            "params": params,
            "ndcg_mean": mean,
            "ndcg_ci_low": ci_lo,
            "ndcg_ci_high": ci_hi,
            "vs_cf_pct": diff_pct,
            "avg_latency_s": float(avg_time),
            "n_samples": len(scores),
            "per_user_ndcg": [float(s) for s in scores],
        }

    # ===== Statistical tests =====
    print("\n" + "=" * 70)
    print("STATISTICAL TESTS (vs CF_Score baseline)")
    print("=" * 70)

    p_values = {}
    for method in models:
        method_scores = np.array(all_scores[method]["ndcg"])
        diff = method_scores - cf_scores

        # Paired t-test
        if np.std(diff) > 0:
            t_stat, t_p = stats.ttest_rel(method_scores, cf_scores)
        else:
            t_stat, t_p = 0.0, 1.0

        # Wilcoxon
        non_zero = diff[diff != 0]
        if len(non_zero) >= 10:
            try:
                w_stat, w_p = stats.wilcoxon(non_zero)
            except:
                w_stat, w_p = 0.0, 1.0
        else:
            w_stat, w_p = 0.0, 1.0

        # Effect size
        cohens_d = float(np.mean(diff) / np.std(diff)) if np.std(diff) > 0 else 0.0
        effect = "negligible" if abs(cohens_d) < 0.2 else \
                 "small" if abs(cohens_d) < 0.5 else \
                 "medium" if abs(cohens_d) < 0.8 else "large"

        p_values[method] = w_p

        print(f"  {method:<22} t-test p={t_p:.4f}, Wilcoxon p={w_p:.4f}, Cohen's d={cohens_d:.3f} ({effect})")

        results_summary["results"][method]["t_test_p"] = float(t_p)
        results_summary["results"][method]["wilcoxon_p"] = float(w_p)
        results_summary["results"][method]["cohens_d"] = cohens_d
        results_summary["results"][method]["effect_size"] = effect

    # Holm-Bonferroni correction
    if p_values:
        comparisons = sorted(p_values.items(), key=lambda x: x[1])
        n_comp = len(comparisons)
        print(f"\nHolm-Bonferroni correction ({n_comp} comparisons):")
        holm_results = {}
        for i, (name, p) in enumerate(comparisons):
            adj_alpha = 0.05 / (n_comp - i)
            sig = "YES" if p <= adj_alpha else "NO"
            print(f"  {name:<22} p={p:.4f}  adj_α={adj_alpha:.4f}  significant={sig}")
            holm_results[name] = {"p": float(p), "adj_alpha": float(adj_alpha), "significant": p <= adj_alpha}
        results_summary["holm_bonferroni"] = holm_results

    # ===== Pairwise LLM comparisons =====
    print("\n" + "=" * 70)
    print("PAIRWISE LLM COMPARISONS")
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
            print(f"  {a} vs {b}: {mean_a:.4f} vs {mean_b:.4f}, Wilcoxon p={w_p:.4f}")
            pairwise[f"{a}_vs_{b}"] = {"mean_a": float(mean_a), "mean_b": float(mean_b), "p": float(w_p)}

    results_summary["pairwise_llm"] = pairwise

    # Save
    output_path = project_root / 'experiments' / 'logs' / 'kdd_multi_model_unified.json'
    with open(output_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results_summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_users', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    run_experiment(args)
