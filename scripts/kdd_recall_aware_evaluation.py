#!/usr/bin/env python3
"""
KDD 2026 Recall-Aware Evaluation Protocol (RAEP)
==================================================
POSITIVE CONTRIBUTION: Instead of just showing what fails, propose a solution.

RAEP provides:
1. Recall-Adjusted NDCG (RA-NDCG): normalizes NDCG by the recall ceiling
2. Recall Diagnostic Dashboard: reports recall before any reranking metric
3. Ceiling Utilization Ratio: how close a reranker gets to the theoretical max
4. Adaptive-K Strategy: selects K based on per-user recall estimation
5. Recall-Stratified Evaluation: break results by recall bin

This gives practitioners a principled framework to decide whether reranking
is worth investing in, and how to evaluate it fairly.

Usage:
    python scripts/kdd_recall_aware_evaluation.py --n_users 500 --dataset beauty
    python scripts/kdd_recall_aware_evaluation.py --all
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
from scipy.stats import wilcoxon


# ============================================================================
# CF RETRIEVAL (shared)
# ============================================================================

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
        item_scores = [scores[self.item_to_idx[c]] for c in candidates]
        if item_scores:
            min_s, max_s = min(item_scores), max(item_scores)
            if max_s > min_s:
                item_scores = [(s - min_s) / (max_s - min_s) for s in item_scores]
        return candidates[:K], item_scores[:K]

    def recall_with_all_scores(self, user_id, exclude_ids):
        """Return scores for ALL items (for adaptive K)."""
        if user_id not in self.user_to_idx:
            return {}, {}
        u_idx = self.user_to_idx[user_id]
        scores = np.dot(self.item_factors, self.user_factors[u_idx])
        exclude_set = set(exclude_ids)
        item_scores = {}
        for idx, score in enumerate(scores):
            item_id = self.idx_to_item[idx]
            if item_id not in exclude_set:
                item_scores[item_id] = float(score)
        return item_scores


# ============================================================================
# RAEP: RECALL-AWARE EVALUATION PROTOCOL
# ============================================================================

class RecallAwareEvaluation:
    """Recall-Aware Evaluation Protocol (RAEP).

    A principled framework for evaluating rerankers that accounts for
    the recall ceiling. Provides:
    - Recall diagnostics before any reranking metric
    - Recall-adjusted metrics
    - Ceiling utilization ratios
    - Recall-stratified breakdowns
    """

    def __init__(self, retrieval_method, K=100):
        self.retrieval = retrieval_method
        self.K = K

    def diagnose_recall(self, test_samples):
        """Step 1: Diagnose retrieval recall before evaluating any reranker.

        Returns per-user recall and aggregate statistics.
        """
        per_user_recall = {}
        recall_scores = []

        for sample in test_samples:
            uid = sample['user_id']
            history = sample['history']
            gt = set(sample['ground_truth'])

            cands, _ = self.retrieval.recall(uid, self.K, history)
            hits = len(set(cands) & gt)
            r = hits / len(gt) if gt else 0

            per_user_recall[uid] = {
                'recall': r,
                'n_gt': len(gt),
                'n_hits': hits,
                'gt_in_candidates': hits > 0
            }
            recall_scores.append(r)

        recall_scores = np.array(recall_scores)

        diagnosis = {
            'mean_recall': float(np.mean(recall_scores)),
            'median_recall': float(np.median(recall_scores)),
            'pct_zero_recall': float(np.mean(recall_scores == 0) * 100),
            'pct_full_recall': float(np.mean(recall_scores == 1.0) * 100),
            'recall_distribution': {
                '0%': float(np.mean(recall_scores == 0) * 100),
                '0-10%': float(np.mean((recall_scores > 0) & (recall_scores <= 0.1)) * 100),
                '10-50%': float(np.mean((recall_scores > 0.1) & (recall_scores <= 0.5)) * 100),
                '50-100%': float(np.mean((recall_scores > 0.5) & (recall_scores < 1.0)) * 100),
                '100%': float(np.mean(recall_scores == 1.0) * 100),
            },
            'recommendation': self._recall_recommendation(np.mean(recall_scores)),
            'per_user': per_user_recall
        }

        return diagnosis

    def _recall_recommendation(self, mean_recall):
        """Generate actionable recommendation based on recall level."""
        if mean_recall < 0.05:
            return ("CRITICAL: Recall < 5%. Reranking is futile. "
                    "Invest ALL resources in improving retrieval. "
                    "Do not evaluate or compare rerankers.")
        elif mean_recall < 0.15:
            return ("WARNING: Recall < 15%. Reranking gains will be marginal. "
                    "Consider improving retrieval before optimizing rerankers. "
                    "Use RA-NDCG for fair comparison if reranking is required.")
        elif mean_recall < 0.30:
            return ("MODERATE: Recall 15-30%. Reranking can provide some gains. "
                    "Report both standard and recall-adjusted metrics. "
                    "Focus on retrieval-reranking co-optimization.")
        else:
            return ("GOOD: Recall > 30%. Reranking can meaningfully differentiate. "
                    "Standard metrics are interpretable. "
                    "Proceed with reranker evaluation.")

    def compute_ceiling(self, test_samples, per_user_recall):
        """Step 2: Compute theoretical ceiling for each user."""
        per_user_ceiling = {}

        for sample in test_samples:
            uid = sample['user_id']
            gt = sample['ground_truth']
            r_info = per_user_recall.get(uid, {})
            n_hits = r_info.get('n_hits', 0)

            # Perfect reranker: place all n_hits items at top
            if n_hits == 0:
                ceiling_ndcg = 0.0
            else:
                # DCG with n_hits items at positions 1..n_hits
                dcg = sum(1.0 / np.log2(i + 2) for i in range(min(n_hits, 10)))
                idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gt), 10)))
                ceiling_ndcg = dcg / idcg if idcg > 0 else 0

            per_user_ceiling[uid] = ceiling_ndcg

        return per_user_ceiling

    def recall_adjusted_ndcg(self, ranked_list, ground_truth, ceiling_ndcg, k=10):
        """RA-NDCG: Recall-Adjusted NDCG.

        Normalizes NDCG by the ceiling achievable given current recall.
        RA-NDCG = NDCG / ceiling_NDCG  (only for users with ceiling > 0)
        """
        gt_set = set(ground_truth) if isinstance(ground_truth, list) else {ground_truth}
        dcg = 0.0
        for i, item in enumerate(ranked_list[:k]):
            if item in gt_set:
                dcg += 1.0 / np.log2(i + 2)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gt_set), k)))
        ndcg = dcg / idcg if idcg > 0 else 0

        if ceiling_ndcg > 0:
            ra_ndcg = ndcg / ceiling_ndcg
        else:
            ra_ndcg = None  # Undefined for zero-recall users

        return ndcg, ra_ndcg

    def stratified_evaluation(self, test_samples, per_user_recall,
                               reranker_fn, k=10):
        """Step 3: Evaluate reranker stratified by recall bins.

        Args:
            reranker_fn: function(candidates, user_id) -> reranked_list
        """
        bins = {
            'zero': (0, 0),        # recall = 0
            'low': (0.001, 0.1),   # 0 < recall <= 10%
            'medium': (0.1, 0.5),  # 10% < recall <= 50%
            'high': (0.5, 1.01),   # > 50%
        }

        results = {name: {'ndcg': [], 'ra_ndcg': [], 'n_users': 0}
                   for name in bins}

        per_user_ceiling = self.compute_ceiling(test_samples, per_user_recall)

        for sample in test_samples:
            uid = sample['user_id']
            gt = sample['ground_truth']
            r_info = per_user_recall.get(uid, {})
            recall = r_info.get('recall', 0)
            ceiling = per_user_ceiling.get(uid, 0)

            # Get candidates and rerank
            cands, _ = self.retrieval.recall(uid, self.K, sample['history'])
            reranked = reranker_fn(cands, uid)

            ndcg, ra_ndcg = self.recall_adjusted_ndcg(reranked, gt, ceiling, k)

            # Assign to bin
            for bin_name, (lo, hi) in bins.items():
                if lo <= recall < hi or (bin_name == 'zero' and recall == 0):
                    results[bin_name]['ndcg'].append(ndcg)
                    if ra_ndcg is not None:
                        results[bin_name]['ra_ndcg'].append(ra_ndcg)
                    results[bin_name]['n_users'] += 1
                    break

        # Aggregate
        summary = {}
        for bin_name, data in results.items():
            summary[bin_name] = {
                'n_users': data['n_users'],
                'mean_ndcg': float(np.mean(data['ndcg'])) if data['ndcg'] else 0,
                'mean_ra_ndcg': float(np.mean(data['ra_ndcg'])) if data['ra_ndcg'] else None,
            }

        return summary

    def full_evaluation(self, test_samples, rerankers: dict, k=10):
        """Complete RAEP evaluation pipeline.

        Args:
            rerankers: dict of {name: reranker_fn}
        """
        # Step 1: Diagnose
        print("Step 1: Recall Diagnosis")
        diagnosis = self.diagnose_recall(test_samples)
        print(f"  Mean Recall@{self.K}: {diagnosis['mean_recall']*100:.2f}%")
        print(f"  Zero-recall users: {diagnosis['pct_zero_recall']:.1f}%")
        print(f"  Recommendation: {diagnosis['recommendation']}")

        per_user_recall = diagnosis['per_user']
        per_user_ceiling = self.compute_ceiling(test_samples, per_user_recall)

        # Step 2: Evaluate each reranker
        print(f"\nStep 2: Reranker Evaluation")
        all_results = {}

        for name, reranker_fn in rerankers.items():
            print(f"\n  {name}:")

            ndcg_scores = []
            ra_ndcg_scores = []
            ceiling_util_scores = []

            for sample in test_samples:
                uid = sample['user_id']
                gt = sample['ground_truth']
                ceiling = per_user_ceiling.get(uid, 0)

                cands, _ = self.retrieval.recall(uid, self.K, sample['history'])
                reranked = reranker_fn(cands, uid)

                ndcg, ra_ndcg = self.recall_adjusted_ndcg(reranked, gt, ceiling, k)
                ndcg_scores.append(ndcg)
                if ra_ndcg is not None:
                    ra_ndcg_scores.append(ra_ndcg)
                    ceiling_util_scores.append(ra_ndcg)

            mean_ndcg = float(np.mean(ndcg_scores))
            mean_ra_ndcg = float(np.mean(ra_ndcg_scores)) if ra_ndcg_scores else None
            mean_ceil_util = float(np.mean(ceiling_util_scores)) if ceiling_util_scores else None

            print(f"    NDCG@{k}: {mean_ndcg:.4f}")
            print(f"    RA-NDCG@{k}: {mean_ra_ndcg:.4f}" if mean_ra_ndcg else "    RA-NDCG@{k}: N/A")
            print(f"    Ceiling Utilization: {mean_ceil_util*100:.1f}%" if mean_ceil_util else "    Ceiling Utilization: N/A")

            # Stratified
            stratified = self.stratified_evaluation(
                test_samples, per_user_recall, reranker_fn, k
            )

            all_results[name] = {
                'ndcg': mean_ndcg,
                'ra_ndcg': mean_ra_ndcg,
                'ceiling_utilization': mean_ceil_util,
                'stratified': stratified,
                'ndcg_scores': [float(s) for s in ndcg_scores]
            }

        # Step 3: Compare (using RA-NDCG where available)
        print(f"\nStep 3: Comparison Table")
        print(f"  {'Method':25s} {'NDCG@10':>10s} {'RA-NDCG@10':>12s} {'Ceil. Util.':>12s}")
        print(f"  {'-'*60}")
        for name, res in all_results.items():
            ra_str = f"{res['ra_ndcg']:.4f}" if res['ra_ndcg'] is not None else "N/A"
            cu_str = f"{res['ceiling_utilization']*100:.1f}%" if res['ceiling_utilization'] is not None else "N/A"
            print(f"  {name:25s} {res['ndcg']:10.4f} {ra_str:>12s} {cu_str:>12s}")

        return {
            'diagnosis': {k: v for k, v in diagnosis.items() if k != 'per_user'},
            'results': {
                name: {k: v for k, v in res.items() if k != 'ndcg_scores'}
                for name, res in all_results.items()
            },
            'all_results': all_results
        }


# ============================================================================
# ADAPTIVE-K STRATEGY
# ============================================================================

class AdaptiveKStrategy:
    """Adaptive-K: select candidate set size per user based on estimated recall.

    For users where recall is estimated to be high with small K, use small K
    (better signal-to-noise). For users with low estimated recall, use larger K
    (more coverage).
    """

    def __init__(self, retrieval_method, K_min=50, K_max=500, n_bins=5):
        self.retrieval = retrieval_method
        self.K_min = K_min
        self.K_max = K_max
        self.n_bins = n_bins

    def estimate_user_difficulty(self, user_id, history):
        """Estimate how hard a user is based on CF score distribution."""
        all_scores = self.retrieval.recall_with_all_scores(user_id, history)
        if not all_scores:
            return 1.0  # Max difficulty

        scores = sorted(all_scores.values(), reverse=True)
        if len(scores) < 100:
            return 0.8

        # Score concentration: if top-100 scores are much higher than rest,
        # the user is "easy" (clear preferences)
        top_100_mean = np.mean(scores[:100])
        rest_mean = np.mean(scores[100:min(500, len(scores))])

        if rest_mean == 0:
            concentration = 1.0
        else:
            concentration = top_100_mean / (rest_mean + 1e-10)

        # Normalize: high concentration = easy = small K needed
        difficulty = 1.0 / (1.0 + np.log1p(concentration))
        return difficulty

    def select_K(self, difficulty):
        """Map difficulty to K value."""
        # Linear interpolation: easy users get K_min, hard users get K_max
        K = int(self.K_min + difficulty * (self.K_max - self.K_min))
        return max(self.K_min, min(self.K_max, K))

    def recall_adaptive(self, user_id, history):
        """Retrieve with adaptive K."""
        difficulty = self.estimate_user_difficulty(user_id, history)
        K = self.select_K(difficulty)
        cands, scores = self.retrieval.recall(user_id, K, history)
        return cands, scores, K, difficulty


# ============================================================================
# MAIN
# ============================================================================

def ndcg_at_k(ranked_list, ground_truth, k=10):
    gt_set = set(ground_truth) if isinstance(ground_truth, list) else {ground_truth}
    dcg = 0.0
    for i, item in enumerate(ranked_list[:k]):
        if item in gt_set:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gt_set), k)))
    return dcg / idcg if idcg > 0 else 0.0


def bootstrap_ci(scores, n_bootstrap=1000, ci=0.95):
    scores = np.array(scores)
    if len(scores) == 0:
        return 0.0, 0.0, 0.0
    boot_means = [np.mean(np.random.choice(scores, len(scores), replace=True))
                  for _ in range(n_bootstrap)]
    alpha = (1 - ci) / 2
    return np.mean(scores), np.percentile(boot_means, alpha * 100), \
           np.percentile(boot_means, (1 - alpha) * 100)


def run_dataset(ds_name, ds_path, args):
    print(f"\n{'='*70}")
    print(f"RAEP EVALUATION: {ds_name.upper()}")
    print(f"{'='*70}")

    path = project_root / 'data' / 'processed' / ds_path
    train_df = pd.read_parquet(path / 'train.parquet')
    test_df = pd.read_parquet(path / 'test.parquet')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    user_history = train_df.groupby('user_id')['item_id'].apply(list).to_dict()
    test_gt = test_df.groupby('user_id')['item_id'].apply(list).to_dict()
    valid_users = [u for u in test_gt if u in user_history and len(user_history[u]) >= 3]

    n_users = min(args.n_users, len(valid_users))
    sampled_users = np.random.choice(valid_users, size=n_users, replace=False)
    test_samples = [{'user_id': u, 'history': user_history[u], 'ground_truth': test_gt[u]}
                    for u in sampled_users]

    # Build retrieval
    cf = CFRecall(n_factors=128)
    cf.fit(train_df)

    # Define rerankers
    def cf_score_reranker(candidates, user_id):
        """No reranking - use CF retrieval order."""
        return candidates

    def random_reranker(candidates, user_id):
        """Random reranking baseline."""
        perm = np.random.permutation(len(candidates))
        return [candidates[i] for i in perm]

    def reverse_reranker(candidates, user_id):
        """Reverse order (worst possible from CF perspective)."""
        return candidates[::-1]

    def popularity_reranker(candidates, user_id):
        """Rerank by popularity."""
        pop = train_df['item_id'].value_counts().to_dict()
        return sorted(candidates, key=lambda x: pop.get(x, 0), reverse=True)

    rerankers = {
        'CF-Score (no rerank)': cf_score_reranker,
        'Random': random_reranker,
        'Reverse-CF': reverse_reranker,
        'Popularity-Rerank': popularity_reranker,
    }

    # Run RAEP
    raep = RecallAwareEvaluation(cf, K=args.K)
    raep_results = raep.full_evaluation(test_samples, rerankers, k=10)

    # ========================================================================
    # ADAPTIVE-K EXPERIMENT
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"ADAPTIVE-K STRATEGY")
    print(f"{'='*70}")

    adaptive = AdaptiveKStrategy(cf, K_min=50, K_max=500)

    fixed_K_recalls = []
    adaptive_K_recalls = []
    adaptive_Ks = []

    for sample in tqdm(test_samples, desc="  Adaptive-K"):
        uid = sample['user_id']
        gt = set(sample['ground_truth'])

        # Fixed K=100
        cands_fixed, _ = cf.recall(uid, 100, sample['history'])
        fixed_recall = len(set(cands_fixed) & gt) / len(gt) if gt else 0
        fixed_K_recalls.append(fixed_recall)

        # Adaptive K
        cands_adaptive, _, K_used, difficulty = adaptive.recall_adaptive(uid, sample['history'])
        adaptive_recall = len(set(cands_adaptive) & gt) / len(gt) if gt else 0
        adaptive_K_recalls.append(adaptive_recall)
        adaptive_Ks.append(K_used)

    fixed_mean = np.mean(fixed_K_recalls)
    adaptive_mean = np.mean(adaptive_K_recalls)
    avg_K = np.mean(adaptive_Ks)

    print(f"  Fixed K=100: Recall = {fixed_mean*100:.2f}%")
    print(f"  Adaptive K (avg={avg_K:.0f}): Recall = {adaptive_mean*100:.2f}%")
    print(f"  Improvement: {(adaptive_mean - fixed_mean)*100:+.2f}%")
    print(f"  K distribution: min={min(adaptive_Ks)}, max={max(adaptive_Ks)}, "
          f"mean={avg_K:.0f}, median={np.median(adaptive_Ks):.0f}")

    adaptive_results = {
        'fixed_K100_recall': float(fixed_mean),
        'adaptive_recall': float(adaptive_mean),
        'avg_K': float(avg_K),
        'improvement': float(adaptive_mean - fixed_mean),
        'K_distribution': {
            'min': int(min(adaptive_Ks)),
            'max': int(max(adaptive_Ks)),
            'mean': float(avg_K),
            'median': float(np.median(adaptive_Ks))
        }
    }

    return {
        'dataset': ds_name,
        'n_users': len(test_samples),
        'raep': {k: v for k, v in raep_results.items() if k != 'all_results'},
        'adaptive_K': adaptive_results
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_users', type=int, default=500)
    parser.add_argument('--K', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--new', action='store_true',
                        help='Run on new datasets (toys/sports/office/mind/movielens)')
    args = parser.parse_args()

    datasets = {
        'beauty': 'amazon_beauty_sampled',
        'movies': 'amazon_movies_sampled',
        'electronics': 'amazon_electronics_sampled'
    }
    new_datasets = {
        'toys': 'amazon_toys_sampled',
        'sports': 'amazon_sports_sampled',
        'office': 'amazon_office_sampled',
        'mind': 'mind_news',
        'movielens': 'movielens_25m',
    }

    if args.new:
        ds_list = list(new_datasets.items())
    elif args.all:
        combined = {**datasets, **new_datasets}
        ds_list = list(combined.items())
    elif args.dataset:
        all_ds = {**datasets, **new_datasets}
        ds_list = [(args.dataset, all_ds[args.dataset])]
    else:
        ds_list = list(datasets.items())

    all_results = {}
    for ds_name, ds_path in ds_list:
        result = run_dataset(ds_name, ds_path, args)
        all_results[ds_name] = result

    output_path = project_root / 'experiments' / 'logs' / 'kdd_recall_aware_evaluation.json'
    existing = {}
    if output_path.exists():
        try:
            with open(output_path) as f:
                existing = json.load(f)
        except Exception:
            pass
    existing.update(all_results)
    with open(output_path, 'w') as f:
        json.dump(existing, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"RESULTS SAVED TO: {output_path}")
    print(f"{'='*70}")

    # Summary
    print("\nRAEP FRAMEWORK SUMMARY")
    print("=" * 70)
    print("The Recall-Aware Evaluation Protocol provides:")
    print("  1. RECALL DIAGNOSIS: Check retrieval before evaluating reranking")
    print("  2. RA-NDCG: Recall-adjusted metric for fair reranker comparison")
    print("  3. CEILING UTILIZATION: How close to theoretical max")
    print("  4. STRATIFIED EVALUATION: Per-recall-bin breakdown")
    print("  5. ADAPTIVE-K: Per-user candidate set size optimization")
    print("\nThis is our POSITIVE CONTRIBUTION: a principled evaluation framework")
    print("that prevents misleading oracle evaluation and guides practitioners.")


if __name__ == '__main__':
    main()
