"""
P9-11: Large Model Reranking Experiments via Ollama Cloud API

Test super-large models for recommendation reranking:
- kimi-k2:1t (1 Trillion parameters!)
- deepseek-v3.1:671b (671B parameters)
- qwen3-coder:480b (480B parameters)
- gpt-oss:120b (120B parameters)

Using Ollama native API (/api/chat endpoint)
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import os
import json
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.decomposition import TruncatedSVD
import requests
from dotenv import load_dotenv
import re

load_dotenv()

CONFIG = {
    'n_test_users': 60,  # Smaller sample for expensive large model calls
    'seed': 42,
    'K': 500,
    'top_k_eval': 10,
    'top_k_rerank': 30,
    'cf_factors': 128,
    'cold_threshold': 20,
    'active_threshold': 100,
    'models_to_test': [
        'gpt-oss:20b',         # Small baseline
        'gpt-oss:120b',        # 120B
        'qwen3-coder:480b',    # 480B
        'deepseek-v3.1:671b',  # 671B
        # 'kimi-k2:1t',        # 1T - try if others work
    ],
    'request_timeout': 120,
    'max_retries': 2,
}

OLLAMA_API_KEY = os.getenv('OLLAMA_API_KEY')
OLLAMA_CHAT_URL = 'https://api.ollama.com/api/chat'


def ndcg_at_k(ranked_items, ground_truth, k=10):
    if not ground_truth:
        return 0.0
    dcg = 0.0
    for i, item in enumerate(ranked_items[:k]):
        if item in ground_truth:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))
    return dcg / idcg if idcg > 0 else 0.0


class OllamaLargeModelReranker:
    """LLM reranker using Ollama native API for large models"""

    def __init__(self, model_name, item_metadata):
        self.model_name = model_name
        self.item_metadata = item_metadata
        self.headers = {
            'Authorization': f'Bearer {OLLAMA_API_KEY}',
            'Content-Type': 'application/json'
        }
        self.call_count = 0
        self.total_time = 0

    def _build_prompt(self, user_history, candidates):
        """Build reranking prompt"""
        history_items = user_history[-10:] if len(user_history) > 10 else user_history
        history_str = "\n".join([
            f"- {self.item_metadata.get(str(item_id), {}).get('title', f'Movie {item_id}')} "
            f"({self.item_metadata.get(str(item_id), {}).get('genre', 'Unknown')})"
            for item_id in history_items
        ])

        candidates_to_rank = candidates[:CONFIG['top_k_rerank']]
        candidates_str = "\n".join([
            f"{j+1}. {self.item_metadata.get(str(item_id), {}).get('title', f'Movie {item_id}')} "
            f"({self.item_metadata.get(str(item_id), {}).get('genre', 'Unknown')})"
            for j, item_id in enumerate(candidates_to_rank)
        ])

        prompt = f"""Based on this user's viewing history, rank these candidate movies.

VIEWING HISTORY:
{history_str}

CANDIDATES (rank from most to least relevant):
{candidates_str}

Return ONLY comma-separated numbers (1-{len(candidates_to_rank)}) representing your ranking.
Example: 5,2,8,1,3,7,4,6,9,10,..."""

        return prompt, candidates_to_rank

    def _parse_response(self, response_text, candidates):
        """Parse LLM response into ranked items"""
        try:
            # Extract first line with numbers
            first_line = response_text.split('\n')[0] if '\n' in response_text else response_text
            numbers = re.findall(r'\d+', first_line)

            ranked_indices = []
            seen = set()
            for num in numbers:
                idx = int(num) - 1
                if 0 <= idx < len(candidates) and idx not in seen:
                    ranked_indices.append(idx)
                    seen.add(idx)

            ranked_items = [candidates[i] for i in ranked_indices]

            # Add missing items
            for item in candidates:
                if item not in ranked_items:
                    ranked_items.append(item)

            return ranked_items
        except:
            return candidates

    def rerank(self, user_id, user_history, candidates):
        """Rerank using Ollama native API"""
        if not candidates or not user_history:
            return candidates

        prompt, candidates_to_rank = self._build_prompt(user_history, candidates)

        for attempt in range(CONFIG['max_retries']):
            try:
                start_time = time.time()

                response = requests.post(
                    OLLAMA_CHAT_URL,
                    headers=self.headers,
                    json={
                        'model': self.model_name,
                        'messages': [
                            {'role': 'system', 'content': 'You are a movie recommendation expert. Respond only with ranking numbers.'},
                            {'role': 'user', 'content': prompt}
                        ],
                        'stream': False,
                    },
                    timeout=CONFIG['request_timeout']
                )

                elapsed = time.time() - start_time
                self.call_count += 1
                self.total_time += elapsed

                if response.status_code == 200:
                    result = response.json()
                    llm_response = result.get('message', {}).get('content', '')

                    reranked = self._parse_response(llm_response, candidates_to_rank)
                    remaining = [c for c in candidates[CONFIG['top_k_rerank']:] if c not in reranked]
                    return reranked + remaining
                else:
                    print(f"API error {response.status_code}: {response.text[:100]}")

            except requests.exceptions.Timeout:
                print(f"Timeout for {self.model_name} (attempt {attempt+1})")
            except Exception as e:
                print(f"Error: {e}")

            time.sleep(1)

        return candidates

    def get_stats(self):
        return {
            'calls': self.call_count,
            'total_time': self.total_time,
            'avg_latency': self.total_time / self.call_count if self.call_count > 0 else 0
        }


class CFModel:
    """Simple CF model for baseline"""
    def __init__(self, n_factors=128):
        self.n_factors = n_factors

    def fit(self, train_df):
        self.user_ids = sorted(train_df['user_id'].unique())
        self.item_ids = sorted(train_df['item_id'].unique())
        self.user2idx = {uid: idx for idx, uid in enumerate(self.user_ids)}
        self.item2idx = {iid: idx for idx, iid in enumerate(self.item_ids)}
        self.idx2item = {idx: iid for iid, idx in self.item2idx.items()}

        n_users, n_items = len(self.user_ids), len(self.item_ids)
        rows = train_df['user_id'].map(self.user2idx).values
        cols = train_df['item_id'].map(self.item2idx).values
        data = np.ones(len(train_df))
        self.matrix = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))

        n_factors = min(self.n_factors, min(n_users, n_items) - 1)
        svd = TruncatedSVD(n_components=n_factors, random_state=42)
        self.user_factors = svd.fit_transform(self.matrix)
        self.item_factors = svd.components_.T

    def recall(self, user_id, K, exclude_ids):
        if user_id not in self.user2idx:
            return []
        scores = self.item_factors @ self.user_factors[self.user2idx[user_id]]
        for item_id in exclude_ids:
            if item_id in self.item2idx:
                scores[self.item2idx[item_id]] = -np.inf
        top_indices = np.argsort(scores)[::-1][:K]
        return [self.idx2item[idx] for idx in top_indices if scores[idx] > -np.inf]

    def score(self, user_id, item_ids):
        if user_id not in self.user2idx:
            return {item: 0.0 for item in item_ids}
        user_vec = self.user_factors[self.user2idx[user_id]]
        return {item: float(user_vec @ self.item_factors[self.item2idx[item]])
                if item in self.item2idx else 0.0 for item in item_ids}


class CooccurrenceModel:
    """Co-occurrence model"""
    def __init__(self):
        pass

    def fit(self, train_df):
        self.item_ids = sorted(train_df['item_id'].unique())
        self.item2idx = {iid: idx for idx, iid in enumerate(self.item_ids)}
        self.idx2item = {idx: iid for iid, idx in self.item2idx.items()}
        n_items = len(self.item_ids)

        cooc = lil_matrix((n_items, n_items), dtype=np.float32)
        for _, group in train_df.groupby('user_id'):
            items = group['item_id'].tolist()
            for i, item1 in enumerate(items):
                if item1 not in self.item2idx:
                    continue
                idx1 = self.item2idx[item1]
                for j in range(max(0, i-10), min(len(items), i+11)):
                    if i != j and items[j] in self.item2idx:
                        cooc[idx1, self.item2idx[items[j]]] += 1

        self.cooc = cooc.tocsr()
        row_sums = np.array(self.cooc.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1
        self.cooc_norm = self.cooc.multiply(1 / row_sums.reshape(-1, 1)).tocsr()
        self.user_history = train_df.groupby('user_id')['item_id'].apply(list).to_dict()

    def recall(self, user_history, K, exclude_ids):
        if not user_history:
            return []
        scores = np.zeros(len(self.item_ids))
        for item_id in user_history[-20:]:
            if item_id in self.item2idx:
                scores += np.array(self.cooc_norm[self.item2idx[item_id]].todense()).flatten()
        for item_id in exclude_ids:
            if item_id in self.item2idx:
                scores[self.item2idx[item_id]] = -np.inf
        top_indices = np.argsort(scores)[::-1][:K]
        return [self.idx2item[idx] for idx in top_indices if scores[idx] > -np.inf]

    def score(self, user_id, item_ids):
        user_hist = self.user_history.get(user_id, [])
        if not user_hist:
            return {item: 0.0 for item in item_ids}
        scores = {}
        for item in item_ids:
            if item not in self.item2idx:
                scores[item] = 0.0
                continue
            item_idx = self.item2idx[item]
            score = sum(self.cooc_norm[self.item2idx[h], item_idx]
                       for h in user_hist[-20:] if h in self.item2idx)
            scores[item] = float(score)
        return scores


def hybrid_recall(cf_model, cooc_model, user_id, user_history, K, exclude_ids):
    cf_items = set(cf_model.recall(user_id, int(K*0.6)*2, exclude_ids))
    cooc_items = set(cooc_model.recall(user_history, int(K*0.4)*2, exclude_ids))

    item_scores = defaultdict(float)
    for item in cf_items:
        item_scores[item] += 0.6
    for item in cooc_items:
        item_scores[item] += 0.4

    return sorted(item_scores.keys(), key=lambda x: -item_scores[x])[:K]


def main():
    print("=" * 80)
    print("P9-11: Large Model Reranking (Ollama Cloud API)")
    print("=" * 80)
    print(f"Models to test: {CONFIG['models_to_test']}")

    np.random.seed(CONFIG['seed'])

    # Load data
    data_path = Path('data/processed/amazon_movies_sampled')
    train_df = pd.read_parquet(data_path / 'train.parquet')
    test_df = pd.read_parquet(data_path / 'test.parquet')

    with open(data_path / 'item_metadata.json') as f:
        item_metadata = json.load(f)

    # User segmentation
    user_counts = train_df.groupby('user_id').size().to_dict()
    test_users = test_df['user_id'].unique()

    cold = [u for u in test_users if user_counts.get(u, 0) <= CONFIG['cold_threshold']]
    medium = [u for u in test_users if CONFIG['cold_threshold'] < user_counts.get(u, 0) <= CONFIG['active_threshold']]
    active = [u for u in test_users if user_counts.get(u, 0) > CONFIG['active_threshold']]

    n_per = CONFIG['n_test_users'] // 3
    sampled = (
        np.random.choice(cold, min(n_per, len(cold)), replace=False).tolist() +
        np.random.choice(medium, min(n_per, len(medium)), replace=False).tolist() +
        np.random.choice(active, min(n_per, len(active)), replace=False).tolist()
    )

    print(f"\nTest users: {len(sampled)} (Cold: {min(n_per, len(cold))}, Medium: {min(n_per, len(medium))}, Active: {min(n_per, len(active))})")

    # Prepare data
    test_gt = test_df.groupby('user_id')['item_id'].apply(list).to_dict()
    train_items = train_df.groupby('user_id')['item_id'].apply(set).to_dict()
    user_history = train_df.groupby('user_id')['item_id'].apply(list).to_dict()

    # Build models
    print("\nBuilding baseline models...")
    cf_model = CFModel(CONFIG['cf_factors'])
    cf_model.fit(train_df)

    cooc_model = CooccurrenceModel()
    cooc_model.fit(train_df)

    K = CONFIG['K']
    all_results = {}

    # Test each large model
    for model_name in CONFIG['models_to_test']:
        print(f"\n{'='*60}")
        print(f"Testing: {model_name}")
        print("=" * 60)

        reranker = OllamaLargeModelReranker(model_name, item_metadata)

        # Quick connection test
        try:
            test_resp = requests.post(
                OLLAMA_CHAT_URL,
                headers=reranker.headers,
                json={'model': model_name, 'messages': [{'role': 'user', 'content': 'hi'}], 'stream': False},
                timeout=30
            )
            if test_resp.status_code != 200:
                print(f"Skipping {model_name}: API error {test_resp.status_code}")
                continue
            print(f"✓ Model {model_name} ready")
        except Exception as e:
            print(f"Skipping {model_name}: {e}")
            continue

        metrics = {'cold': [], 'medium': [], 'active': []}

        for user_id in tqdm(sampled, desc=model_name):
            if user_id not in test_gt:
                continue

            gt = set(test_gt[user_id])
            exclude = train_items.get(user_id, set())
            hist = user_history.get(user_id, [])
            n_int = user_counts.get(user_id, 0)

            segment = 'cold' if n_int <= CONFIG['cold_threshold'] else (
                'medium' if n_int <= CONFIG['active_threshold'] else 'active')

            candidates = hybrid_recall(cf_model, cooc_model, user_id, hist, K, exclude)
            ranked = reranker.rerank(user_id, hist, candidates)
            ndcg = ndcg_at_k(ranked, gt, CONFIG['top_k_eval'])
            metrics[segment].append(ndcg)

        # Aggregate results
        stats = reranker.get_stats()
        result = {
            seg: np.mean(vals) if vals else 0 for seg, vals in metrics.items()
        }
        result['overall'] = np.mean([v for vals in metrics.values() for v in vals])
        result['avg_latency'] = stats['avg_latency']
        result['total_calls'] = stats['calls']

        all_results[model_name] = result
        print(f"\nResults for {model_name}:")
        print(f"  Cold: {result['cold']:.4f}, Medium: {result['medium']:.4f}, Active: {result['active']:.4f}")
        print(f"  Overall: {result['overall']:.4f}, Avg Latency: {result['avg_latency']:.2f}s")

    # Add baselines
    print("\n" + "=" * 60)
    print("Running CF baseline...")
    print("=" * 60)

    cf_metrics = {'cold': [], 'medium': [], 'active': []}
    for user_id in tqdm(sampled, desc="CF-only"):
        if user_id not in test_gt:
            continue
        gt = set(test_gt[user_id])
        exclude = train_items.get(user_id, set())
        hist = user_history.get(user_id, [])
        n_int = user_counts.get(user_id, 0)
        segment = 'cold' if n_int <= CONFIG['cold_threshold'] else ('medium' if n_int <= CONFIG['active_threshold'] else 'active')

        candidates = hybrid_recall(cf_model, cooc_model, user_id, hist, K, exclude)
        scores = cf_model.score(user_id, candidates)
        ranked = sorted(candidates, key=lambda x: -scores.get(x, 0))
        ndcg = ndcg_at_k(ranked, gt, CONFIG['top_k_eval'])
        cf_metrics[segment].append(ndcg)

    all_results['CF-only'] = {
        seg: np.mean(vals) if vals else 0 for seg, vals in cf_metrics.items()
    }
    all_results['CF-only']['overall'] = np.mean([v for vals in cf_metrics.values() for v in vals])
    all_results['CF-only']['avg_latency'] = 0

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Large Model Comparison (NDCG@10)")
    print("=" * 80)
    print(f"{'Model':<25} {'Cold':<10} {'Medium':<10} {'Active':<10} {'Overall':<10} {'Latency':<10}")
    print("-" * 75)

    baseline = all_results.get('CF-only', {}).get('overall', 0.01)
    for model_name in ['CF-only'] + CONFIG['models_to_test']:
        if model_name not in all_results:
            continue
        r = all_results[model_name]
        improvement = (r['overall'] - baseline) / baseline * 100 if baseline > 0 else 0
        latency = f"{r.get('avg_latency', 0):.1f}s" if r.get('avg_latency', 0) > 0 else "-"
        print(f"{model_name:<25} {r['cold']:<10.4f} {r['medium']:<10.4f} {r['active']:<10.4f} {r['overall']:<10.4f} {latency:<10}")

    # Save
    output = {
        'config': CONFIG,
        'results': all_results,
    }

    output_path = Path('experiments/logs/p9_11_large_model_reranking.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
