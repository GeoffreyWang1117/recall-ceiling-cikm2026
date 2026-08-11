#!/usr/bin/env python3
"""
KDD 2026 Supplementary Experiment 1: Oracle vs Realistic Prompt Comparison
===========================================================================
This experiment directly tests whether prompt improvements appear under oracle
but disappear under realistic conditions, providing direct evidence for the
recall bottleneck hypothesis.

Usage:
    python scripts/kdd_oracle_vs_realistic_prompts.py --n_users 300
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from typing import Dict, List, Tuple
from tqdm import tqdm
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

PROMPT_P1_BASIC = """You are a recommendation assistant. Based on the user's purchase history, rank the candidate items.

User's purchase history:
{history}

Candidate items to rank:
{candidates}

Return ONLY a comma-separated list of item numbers in order of relevance (most relevant first).
"""

PROMPT_P2_DIRECT = """Rank these items for a user who bought: {history_short}

Items:
{candidates}

Output only the item numbers, comma-separated, most relevant first:"""

PROMPT_P3_ENHANCED = """You are analyzing a user's preferences to recommend products.

STEP 1: Analyze the user's purchase history and identify their preferences.
User's purchases:
{history}

Based on these purchases, the user likely prefers:
- Categories: [identify main categories]
- Features: [identify preferred features]
- Price range: [estimate from history]

STEP 2: Evaluate each candidate item against the user's preferences.
Candidate items:
{candidates}

STEP 3: Rank the candidates by how well they match the user's preferences.
Return ONLY a comma-separated list of item numbers (most relevant first):"""

PROMPTS = {
    'P1_basic': PROMPT_P1_BASIC,
    'P2_direct': PROMPT_P2_DIRECT,
    'P3_enhanced': PROMPT_P3_ENHANCED
}

# ============================================================================
# RETRIEVAL & EVALUATION
# ============================================================================

class CFRetrieval:
    """Collaborative filtering retrieval using SVD."""
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

        self.user_item = csr_matrix((data, (rows, cols)),
                                     shape=(len(users), len(items)))

        self.svd = TruncatedSVD(n_components=min(self.n_factors, len(items)-1))
        self.user_factors = self.svd.fit_transform(self.user_item)
        self.item_factors = self.svd.components_.T

        self.user_history = train_df.groupby('user_id')['item_id'].apply(list).to_dict()

    def get_candidates(self, user_id, K=100, exclude_history=True):
        if user_id not in self.user_to_idx:
            return [], []

        user_idx = self.user_to_idx[user_id]
        scores = np.dot(self.user_factors[user_idx], self.item_factors.T)

        if exclude_history:
            history = self.user_history.get(user_id, [])
            for item in history:
                if item in self.item_to_idx:
                    scores[self.item_to_idx[item]] = -np.inf

        top_indices = np.argsort(scores)[::-1][:K]
        candidates = [self.idx_to_item[idx] for idx in top_indices]
        cand_scores = [scores[idx] for idx in top_indices]

        return candidates, cand_scores


def ndcg_at_k(ranked_list, ground_truth, k=10):
    """Calculate NDCG@k."""
    dcg = 0.0
    for i, item in enumerate(ranked_list[:k]):
        if item in ground_truth:
            dcg += 1.0 / np.log2(i + 2)

    idcg = sum([1.0 / np.log2(i + 2) for i in range(min(k, len(ground_truth)))])
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(candidates, ground_truth, k=100):
    """Calculate Recall@k."""
    hits = len(set(candidates[:k]) & set(ground_truth))
    return hits / len(ground_truth) if ground_truth else 0.0


def bootstrap_ci(scores, n_bootstrap=1000, ci=0.95):
    """Calculate bootstrap confidence interval."""
    scores = np.array(scores)
    boot_means = [np.mean(np.random.choice(scores, len(scores), replace=True))
                  for _ in range(n_bootstrap)]
    alpha = (1 - ci) / 2
    return np.mean(scores), np.percentile(boot_means, alpha * 100), np.percentile(boot_means, (1 - alpha) * 100)


# ============================================================================
# LLM RERANKING
# ============================================================================

class LLMReranker:
    """LLM-based reranker."""
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct"):
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=True
        )
        self.model.eval()
        print("Model loaded.")

    def rerank(self, prompt_template: str, history: List[str], candidates: List[Tuple[int, str]],
               history_short: str = None) -> List[int]:
        """Rerank candidates using LLM."""
        # Format history
        history_text = "\n".join([f"- {h}" for h in history[-10:]])  # Last 10 items

        # Format candidates
        candidates_text = "\n".join([f"{i+1}. {title}" for i, (_, title) in enumerate(candidates)])

        # Build prompt
        if history_short is None:
            history_short = ", ".join(history[-5:])

        prompt = prompt_template.format(
            history=history_text,
            history_short=history_short,
            candidates=candidates_text
        )

        # Generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:],
                                         skip_special_tokens=True)

        # Parse ranking
        ranked_indices = self._parse_ranking(response, len(candidates))
        ranked_items = [candidates[i][0] for i in ranked_indices if i < len(candidates)]

        return ranked_items

    def _parse_ranking(self, response: str, n_candidates: int) -> List[int]:
        """Parse LLM response to get ranking."""
        import re
        numbers = re.findall(r'\d+', response)
        indices = []
        seen = set()
        for num in numbers:
            idx = int(num) - 1
            if 0 <= idx < n_candidates and idx not in seen:
                indices.append(idx)
                seen.add(idx)

        # Fill missing with remaining
        for i in range(n_candidates):
            if i not in seen:
                indices.append(i)

        return indices[:n_candidates]


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment(args):
    """Run Oracle vs Realistic prompt comparison experiment."""

    print("=" * 70)
    print("EXPERIMENT: Oracle vs Realistic Prompt Comparison")
    print("=" * 70)

    # Load data
    data_path = project_root / 'data' / 'processed' / 'amazon_beauty_sampled'
    train_df = pd.read_parquet(data_path / 'train.parquet')
    test_df = pd.read_parquet(data_path / 'test.parquet')

    print(f"Train: {len(train_df)} interactions, {train_df['user_id'].nunique()} users")
    print(f"Test: {len(test_df)} interactions, {test_df['user_id'].nunique()} users")

    # Build item texts
    item_texts = {}
    if 'title' in train_df.columns:
        for _, row in train_df.drop_duplicates('item_id').iterrows():
            item_texts[row['item_id']] = str(row.get('title', f"Product {row['item_id']}"))[:100]

    # Build CF retrieval
    print("\nBuilding CF retrieval...")
    cf = CFRetrieval(n_factors=128)
    cf.fit(train_df)

    # Prepare test users
    np.random.seed(args.seed)
    test_users = test_df.groupby('user_id')['item_id'].apply(list).to_dict()
    valid_users = [u for u in test_users.keys() if u in cf.user_to_idx and len(test_users[u]) > 0]

    if len(valid_users) > args.n_users:
        valid_users = np.random.choice(valid_users, args.n_users, replace=False).tolist()

    print(f"Test users: {len(valid_users)}")

    # Load LLM
    reranker = LLMReranker()

    # Results storage
    results = {
        'oracle': {p: {'ndcg_scores': [], 'recall_scores': []} for p in PROMPTS},
        'realistic': {p: {'ndcg_scores': [], 'recall_scores': []} for p in PROMPTS}
    }

    # Run experiment for each user
    for user_id in tqdm(valid_users, desc="Users"):
        ground_truth = test_users[user_id]
        history = cf.user_history.get(user_id, [])
        history_texts = [item_texts.get(i, f"Product {i}") for i in history]

        # Get CF candidates
        cf_candidates, cf_scores = cf.get_candidates(user_id, K=100)
        cf_recall = recall_at_k(cf_candidates, ground_truth, k=100)

        # === REALISTIC SETTING ===
        # Candidates from CF only
        realistic_candidates = [(item, item_texts.get(item, f"Product {item}"))
                               for item in cf_candidates[:100]]

        for prompt_name, prompt_template in PROMPTS.items():
            try:
                ranked = reranker.rerank(prompt_template, history_texts, realistic_candidates)
                ndcg = ndcg_at_k(ranked, ground_truth, k=10)
                recall = recall_at_k(ranked, ground_truth, k=10)
            except Exception as e:
                ndcg, recall = 0.0, 0.0

            results['realistic'][prompt_name]['ndcg_scores'].append(ndcg)
            results['realistic'][prompt_name]['recall_scores'].append(recall)

        # === ORACLE SETTING ===
        # Inject ground truth items into candidates
        oracle_candidates = cf_candidates[:90]  # Keep 90 CF candidates
        for gt_item in ground_truth:
            if gt_item not in oracle_candidates:
                oracle_candidates.append(gt_item)
        oracle_candidates = oracle_candidates[:100]
        np.random.shuffle(oracle_candidates)  # Shuffle to avoid position bias

        oracle_candidates_with_text = [(item, item_texts.get(item, f"Product {item}"))
                                       for item in oracle_candidates]

        for prompt_name, prompt_template in PROMPTS.items():
            try:
                ranked = reranker.rerank(prompt_template, history_texts, oracle_candidates_with_text)
                ndcg = ndcg_at_k(ranked, ground_truth, k=10)
                recall = recall_at_k(ranked, ground_truth, k=10)
            except Exception as e:
                ndcg, recall = 0.0, 0.0

            results['oracle'][prompt_name]['ndcg_scores'].append(ndcg)
            results['oracle'][prompt_name]['recall_scores'].append(recall)

    # Calculate statistics
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    summary = {'config': {'n_users': len(valid_users), 'seed': args.seed}, 'results': {}}

    for setting in ['oracle', 'realistic']:
        print(f"\n=== {setting.upper()} SETTING ===")
        summary['results'][setting] = {}

        for prompt_name in PROMPTS:
            ndcg_scores = results[setting][prompt_name]['ndcg_scores']
            mean, ci_low, ci_high = bootstrap_ci(ndcg_scores)

            summary['results'][setting][prompt_name] = {
                'ndcg@10': {'mean': mean, 'ci_low': ci_low, 'ci_high': ci_high},
                'n_samples': len(ndcg_scores)
            }

            print(f"  {prompt_name:15s}: NDCG@10 = {mean:.4f} [{ci_low:.4f}, {ci_high:.4f}]")

    # Calculate improvements
    print("\n=== PROMPT IMPROVEMENT (P3 vs P1) ===")
    for setting in ['oracle', 'realistic']:
        p1_mean = summary['results'][setting]['P1_basic']['ndcg@10']['mean']
        p3_mean = summary['results'][setting]['P3_enhanced']['ndcg@10']['mean']
        improvement = (p3_mean - p1_mean) / p1_mean * 100 if p1_mean > 0 else 0
        print(f"  {setting:10s}: P1={p1_mean:.4f}, P3={p3_mean:.4f}, Improvement={improvement:+.1f}%")
        summary['results'][setting]['p3_vs_p1_improvement'] = improvement

    # Save results
    output_path = project_root / 'experiments' / 'logs' / 'kdd_oracle_vs_realistic_prompts.json'
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_users', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    run_experiment(args)
